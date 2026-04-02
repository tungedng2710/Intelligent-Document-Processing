"""FastAPI applications for the doc2md2json pipeline.

Two separate apps, each running on its own port:
    ocr_app  (port 7871)  POST /extract-markdown  — Image → Markdown
    llm_app  (port 7872)  POST /convert-to-json   — Markdown → JSON
"""

from __future__ import annotations

import logging
import time

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import config

ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/tiff", "image/bmp",
    "application/pdf",
}


def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Convert each page of a PDF to a PNG image (bytes)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        images.append(pix.tobytes("png"))
    doc.close()
    return images
from .models import MarkerOCR, OllamaLLM, OllamaOCR
from .utils import parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Two separate FastAPI apps ----
ocr_app = FastAPI(title="doc2md2json — OCR", version="1.0.0")
llm_app = FastAPI(title="doc2md2json — LLM", version="1.0.0")

# --- Model singletons (created once at startup) ---
_ocr_model: MarkerOCR | OllamaOCR | None = None
_llm_model: OllamaLLM | None = None


def _get_ocr_model() -> MarkerOCR | OllamaOCR:
    global _ocr_model
    if _ocr_model is None:
        if config.OCR_ENGINE == "marker":
            _ocr_model = MarkerOCR(force_ocr=config.OCR_FORCE_OCR)
        else:
            _ocr_model = OllamaOCR(
                model=config.OCR_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                timeout=config.OCR_TIMEOUT,
                temperature=config.OCR_TEMPERATURE,
            )
        logger.info(f"OCR model initialised: {_ocr_model.model_name}")
    return _ocr_model


def _get_llm_model() -> OllamaLLM:
    global _llm_model
    if _llm_model is None:
        _llm_model = OllamaLLM(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            temperature=config.LLM_TEMPERATURE,
            think=config.LLM_THINK,
        )
        logger.info(f"LLM model initialised: {_llm_model.model_name}")
    return _llm_model


# ---- Request / Response schemas ----

class MarkdownResponse(BaseModel):
    markdown: str
    model: str
    elapsed_seconds: float


class JsonRequest(BaseModel):
    markdown: str
    prompt_template: str | None = None


class JsonResponse(BaseModel):
    extracted_json: dict | list | None
    raw_response: str | None = None
    model: str
    elapsed_seconds: float


class WrapperResponse(BaseModel):
    extracted_json: dict | list | None
    raw_response: str | None = None
    markdown: str
    elapsed_seconds: float


class ExtractJsonResponse(BaseModel):
    extracted_json: dict | list | None
    raw_response: str | None = None
    markdown: str
    ocr_model: str
    llm_model: str
    elapsed_seconds: float


# ===================== OCR App (port 7871) =====================

@ocr_app.get("/")
def ocr_health():
    ocr = _get_ocr_model()
    return {"status": "ok", "model": ocr.model_name}


@ocr_app.post("/extract-markdown", response_model=MarkdownResponse)
async def extract_markdown(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
):
    """Step 1: Upload an image or PDF → receive markdown text."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image or PDF file, got {content_type}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    ocr = _get_ocr_model()
    t0 = time.time()
    try:
        if isinstance(ocr, MarkerOCR):
            # Marker handles PDFs and images natively (no page splitting)
            markdown = ocr.convert_file(file_bytes, content_type=content_type)
        else:
            # Legacy Ollama OCR: split PDF into images, process page-by-page
            if content_type == "application/pdf":
                try:
                    image_list = pdf_to_images(file_bytes)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}")
                if not image_list:
                    raise HTTPException(status_code=400, detail="PDF has no pages")
            else:
                image_list = [file_bytes]
            pages_md: list[str] = []
            for i, img_bytes in enumerate(image_list):
                md = ocr.extract_markdown(img_bytes, prompt=prompt)
                pages_md.append(md)
                if len(image_list) > 1:
                    logger.info(f"  page {i + 1}/{len(image_list)} done")
            markdown = "\n\n---\n\n".join(pages_md) if len(pages_md) > 1 else pages_md[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"OCR failed: {exc}")
        raise HTTPException(status_code=502, detail=f"OCR model error: {exc}")

    elapsed = round(time.time() - t0, 2)
    logger.info(f"OCR done in {elapsed}s — {len(markdown)} chars")

    return MarkdownResponse(markdown=markdown, model=ocr.model_name, elapsed_seconds=elapsed)


@ocr_app.post("/extract-json", response_model=ExtractJsonResponse)
async def extract_json(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
    prompt_template: str | None = Form(None),
):
    """Combined: Upload an image or PDF → OCR → LLM → structured JSON."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image or PDF file, got {content_type}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # --- Step 1: OCR ---
    ocr = _get_ocr_model()
    t0 = time.time()
    try:
        if isinstance(ocr, MarkerOCR):
            markdown = ocr.convert_file(file_bytes, content_type=content_type)
        else:
            if content_type == "application/pdf":
                try:
                    image_list = pdf_to_images(file_bytes)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}")
                if not image_list:
                    raise HTTPException(status_code=400, detail="PDF has no pages")
            else:
                image_list = [file_bytes]
            pages_md: list[str] = []
            for i, img_bytes in enumerate(image_list):
                md = ocr.extract_markdown(img_bytes, prompt=prompt)
                pages_md.append(md)
                if len(image_list) > 1:
                    logger.info(f"  page {i + 1}/{len(image_list)} done")
            markdown = "\n\n---\n\n".join(pages_md) if len(pages_md) > 1 else pages_md[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"OCR failed: {exc}")
        raise HTTPException(status_code=502, detail=f"OCR model error: {exc}")

    # --- Step 2: LLM extraction ---
    llm = _get_llm_model()
    try:
        raw = llm.extract_json(markdown, prompt_template=prompt_template)
    except Exception as exc:
        logger.error(f"LLM extraction failed: {exc}")
        raise HTTPException(status_code=502, detail=f"LLM model error: {exc}")

    elapsed = round(time.time() - t0, 2)
    parsed = parse_json_response(raw)

    if parsed is None:
        logger.warning(f"JSON parse failed, returning raw response ({len(raw)} chars)")

    logger.info(f"extract-json done in {elapsed}s")

    return ExtractJsonResponse(
        extracted_json=parsed,
        raw_response=raw if parsed is None else None,
        markdown=markdown,
        ocr_model=ocr.model_name,
        llm_model=llm.model_name,
        elapsed_seconds=elapsed,
    )


# ===================== LLM App (port 7872) =====================

@llm_app.get("/")
def llm_health():
    llm = _get_llm_model()
    return {"status": "ok", "model": llm.model_name}


@llm_app.post("/convert-to-json", response_model=JsonResponse)
async def convert_to_json(body: JsonRequest):
    """Step 2: Send markdown text → receive structured JSON."""
    if not body.markdown.strip():
        raise HTTPException(status_code=400, detail="Markdown content is empty")

    llm = _get_llm_model()
    t0 = time.time()
    try:
        raw = llm.extract_json(body.markdown, prompt_template=body.prompt_template)
    except Exception as exc:
        logger.error(f"LLM extraction failed: {exc}")
        raise HTTPException(status_code=502, detail=f"LLM model error: {exc}")

    elapsed = round(time.time() - t0, 2)
    parsed = parse_json_response(raw)

    if parsed is None:
        logger.warning(f"JSON parse failed, returning raw response ({len(raw)} chars)")

    return JsonResponse(
        extracted_json=parsed,
        raw_response=raw if parsed is None else None,
        model=llm.model_name,
        elapsed_seconds=elapsed,
    )


# ===================== Wrapper App (port 7873) =====================

import httpx  # noqa: E402

wrapper_app = FastAPI(title="doc2md2json — Wrapper", version="1.0.0")


@wrapper_app.get("/")
def wrapper_health():
    return {"status": "ok", "upstream": {"ocr": config.OCR_API_URL, "llm": config.LLM_API_URL}}


@wrapper_app.post("/extract", response_model=WrapperResponse)
async def wrapper_extract(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
    prompt_template: str | None = Form(None),
):
    """Wrapper: Upload an image or PDF → call OCR API → call LLM API → return JSON."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image or PDF file, got {content_type}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    t0 = time.time()

    # --- Step 1: call OCR API ---
    async with httpx.AsyncClient(timeout=600) as client:
        ocr_data: dict[str, str] = {}
        if prompt:
            ocr_data["prompt"] = prompt
        try:
            ocr_resp = await client.post(
                f"{config.OCR_API_URL}/extract-markdown",
                files={"file": (file.filename or "upload", file_bytes, content_type)},
                data=ocr_data,
            )
            ocr_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"OCR API error: {exc.response.text}")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"OCR API unreachable: {exc}")

        markdown = ocr_resp.json()["markdown"]

        # --- Step 2: call LLM API ---
        llm_body: dict = {"markdown": markdown}
        if prompt_template:
            llm_body["prompt_template"] = prompt_template
        try:
            llm_resp = await client.post(
                f"{config.LLM_API_URL}/convert-to-json",
                json=llm_body,
            )
            llm_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"LLM API error: {exc.response.text}")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"LLM API unreachable: {exc}")

    llm_json = llm_resp.json()
    elapsed = round(time.time() - t0, 2)
    logger.info(f"wrapper extract done in {elapsed}s")

    return WrapperResponse(
        extracted_json=llm_json.get("extracted_json"),
        raw_response=llm_json.get("raw_response"),
        markdown=markdown,
        elapsed_seconds=elapsed,
    )
