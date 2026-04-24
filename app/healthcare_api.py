from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from scripts.classify_doc import MODEL_NAME as DEFAULT_CLASSIFIER_MODEL
from scripts.classify_doc import OLLAMA_BASE_URL as DEFAULT_OLLAMA_URL
from scripts.classify_doc import classify_document
from scripts.healthcare_ollama_pipeline import DOCUMENT_TYPES
from scripts.healthcare_ollama_pipeline import EXTRACTOR_MODEL as DEFAULT_EXTRACTOR_MODEL
from scripts.healthcare_ollama_pipeline import (
    HealthcareOllamaPipeline,
    extract_information,
    load_templates,
    process_document,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT_DIR / "data" / "prompts"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "output")))
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", str(PROMPTS_DIR)))
DEFAULT_PROMPT_TEMPLATE = os.getenv(
    "DEFAULT_PROMPT_TEMPLATE",
    "healthcare_types/prompt_with_template.txt",
)
DEFAULT_JSON_TEMPLATE = os.getenv(
    "DEFAULT_JSON_TEMPLATE",
    "healthcare_types/page-04-template.json",
)
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "application/pdf",
}

classify_app = FastAPI(title="Healthcare Classify API", version="1.0.0")
extraction_app = FastAPI(title="Healthcare Extraction API", version="1.0.0")
healthcare_pipeline_app = FastAPI(title="Healthcare Ollama Pipeline API", version="1.0.0")


def _ensure_supported_upload(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Expected an image or PDF upload, "
                f"got {content_type or 'unknown content type'}"
            ),
        )
    return content_type


def _resolve_template_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = TEMPLATES_DIR / candidate

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"Template file not found: {resolved}")
    return resolved


def _write_temp_upload(file_bytes: bytes, filename: str, suffix: str) -> Path:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix, prefix="idp-") as temp_file:
        temp_file.write(file_bytes)
        return Path(temp_file.name)


def _prepare_upload_image(file: UploadFile, file_bytes: bytes, content_type: str) -> Path:
    if content_type == "application/pdf":
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if not doc:
                    raise HTTPException(status_code=400, detail="PDF has no pages")
                pix = doc[0].get_pixmap(dpi=PDF_RENDER_DPI)
                image_bytes = pix.tobytes("png")
            return _write_temp_upload(image_bytes, file.filename or "upload", ".png")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    return _write_temp_upload(file_bytes, file.filename or "upload", suffix)


@classify_app.get("/")
def classify_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "classify",
        "ollama_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "classifier_model": os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL),
    }


@classify_app.post("/classify")
async def classify_endpoint(
    file: UploadFile = File(...),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
) -> dict[str, str | None]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    temp_path = _prepare_upload_image(file, file_bytes, content_type)

    try:
        document_type, template_filename = classify_document(
            str(temp_path),
            base_url=ollama_url,
            model=model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if not document_type:
        raise HTTPException(status_code=502, detail="Document classification failed")

    return {
        "status": "success",
        "document_type": document_type,
        "template_filename": template_filename,
    }


@extraction_app.get("/")
def extraction_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "extraction",
        "ollama_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "extractor_model": os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL),
        "templates_dir": str(TEMPLATES_DIR),
    }


@extraction_app.post("/extract")
async def extraction_endpoint(
    file: UploadFile = File(...),
    prompt_template_path: str = Form(default=DEFAULT_PROMPT_TEMPLATE),
    json_template_path: str = Form(default=DEFAULT_JSON_TEMPLATE),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
) -> dict[str, object]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    prompt_template = _resolve_template_path(prompt_template_path).read_text(encoding="utf-8")
    with _resolve_template_path(json_template_path).open(encoding="utf-8") as template_file:
        json_template = json.load(template_file)

    temp_path = _prepare_upload_image(file, file_bytes, content_type)

    try:
        extracted_data = extract_information(
            image_path=str(temp_path),
            prompt_template=prompt_template,
            json_template=json_template,
            base_url=ollama_url,
            model=model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if extracted_data is None:
        raise HTTPException(status_code=502, detail="Information extraction failed")

    return {
        "status": "success",
        "prompt_template_path": str(_resolve_template_path(prompt_template_path)),
        "json_template_path": str(_resolve_template_path(json_template_path)),
        "extracted_data": extracted_data,
    }


@healthcare_pipeline_app.get("/")
def healthcare_pipeline_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "healthcare_ollama_pipeline",
        "ollama_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "classifier_model": os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL),
        "extractor_model": os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL),
        "templates_dir": str(TEMPLATES_DIR),
        "output_dir": str(OUTPUT_DIR),
    }


@healthcare_pipeline_app.post("/process")
async def healthcare_pipeline_endpoint(
    file: UploadFile = File(...),
    output_subdir: str = Form(default="api"),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    classifier_model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
) -> dict[str, object]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    temp_path = _prepare_upload_image(file, file_bytes, content_type)
    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = process_document(
            image_path=str(temp_path),
            output_dir=str(output_dir),
            templates_base_dir=templates_dir,
            ollama_base_url=ollama_url,
            classifier_model=classifier_model,
            extractor_model=extractor_model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if result.get("status") != "success":
        raise HTTPException(status_code=502, detail=result)

    return result


@healthcare_pipeline_app.post("/classify")
async def healthcare_pipeline_classify_endpoint(
    file: UploadFile = File(...),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    classifier_model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
) -> dict[str, object]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    temp_path = _prepare_upload_image(file, file_bytes, content_type)

    pipeline = HealthcareOllamaPipeline(
        templates_base_dir=templates_dir,
        ollama_base_url=ollama_url,
        classifier_model=classifier_model,
        extractor_model=extractor_model,
    )

    try:
        result = pipeline.classify(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    if result.get("status") != "success":
        raise HTTPException(status_code=502, detail=result)

    return result


@healthcare_pipeline_app.post("/extraction")
async def healthcare_pipeline_extraction_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form(default="PHIẾU KHÁM BỆNH VÀO VIỆN"),
    output_subdir: str = Form(default="api"),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
) -> dict[str, object]:
    if document_type not in DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown document_type '{document_type}'. Valid values: {list(DOCUMENT_TYPES.keys())}",
        )

    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    template_info = DOCUMENT_TYPES[document_type]
    templates_base = Path(templates_dir)
    prompt_template, json_template = load_templates(template_info, templates_base)
    if not prompt_template or not json_template:
        raise HTTPException(status_code=502, detail="Failed to load templates for the given document type")

    temp_path = _prepare_upload_image(file, file_bytes, content_type)
    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        extracted_data = extract_information(
            image_path=str(temp_path),
            prompt_template=prompt_template,
            json_template=json_template,
            base_url=ollama_url,
            model=extractor_model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if extracted_data is None:
        raise HTTPException(status_code=502, detail="Information extraction failed")

    output_path = None
    output_path_obj = output_dir / Path(file.filename or "upload").with_suffix(".json").name
    output_path_obj.write_text(json.dumps(extracted_data, ensure_ascii=False, indent=2))
    output_path = str(output_path_obj)

    return {
        "status": "success",
        "document_type": document_type,
        "template_filename": template_info.get("template"),
        "extracted_data": extracted_data,
        "output_path": output_path,
    }
