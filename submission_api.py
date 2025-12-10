#!/usr/bin/env python3
"""FastAPI application exposing a /parse endpoint backed by MonkeyOCR."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import re
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- MonkeyOCR imports -----------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MONKEY_OCR_DIR = ROOT_DIR / "services" / "MonkeyOCR"
if str(MONKEY_OCR_DIR) not in sys.path:
    sys.path.insert(0, str(MONKEY_OCR_DIR))

from magic_pdf.data.dataset import ImageDataset, PymuDocDataset
from magic_pdf.data.data_reader_writer.filebase import FileBasedDataWriter
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.model_manager import model_manager

# --- App configuration -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("submission_api")

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
CONTENT_TYPE_TO_EXT = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
IMAGE_FOLDER_NAME = "images"


def _init_model() -> None:
    config_path = os.getenv(
        "MONKEYOCR_CONFIG", str(MONKEY_OCR_DIR / "model_configs.yaml")
    )
    if not model_manager.is_model_loaded():
        LOGGER.info("Initializing MonkeyOCR with config %s", config_path)
        model_manager.initialize_model(config_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _init_model)
    except Exception as exc:  # pragma: no cover - startup exceptions
        LOGGER.exception("Failed to initialize MonkeyOCR: %s", exc)
        raise
    yield


app = FastAPI(
    title="Submission Parser API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = "; ".join(err.get("msg", "Invalid request") for err in exc.errors())
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": errors or "Invalid request"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )


def _infer_extension(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = (Path(filename).suffix or "").lower()
    if suffix in ALLOWED_EXTENSIONS:
        return suffix
    guessed = CONTENT_TYPE_TO_EXT.get((upload.content_type or "").lower())
    if guessed:
        return guessed
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Provide a PDF or image (png, jpg, jpeg).",
    )


def _embed_local_images(markdown: str, temp_root: Path) -> str:
    """Inline local figure references as base64 data URIs to keep Markdown portable."""

    image_root = temp_root / IMAGE_FOLDER_NAME
    if not image_root.exists():
        return markdown

    cache: dict[str, str] = {}
    pattern = re.compile(r"(!\[[^\]]*\]\()([^()\s]+)(\))")

    def replace(match: re.Match[str]) -> str:
        target = match.group(2).strip().replace("\\", "/")
        prefixes = (
            f"{IMAGE_FOLDER_NAME}/",
            f"./{IMAGE_FOLDER_NAME}/",
            f".\\{IMAGE_FOLDER_NAME}/",
        )
        rel_path = None
        for prefix in prefixes:
            if target.startswith(prefix):
                rel_path = target[len(prefix) :].lstrip("/\\")
                break
        if not rel_path:
            return match.group(0)

        file_path = image_root / rel_path
        if not file_path.is_file():
            return match.group(0)

        if rel_path not in cache:
            mime, _ = mimetypes.guess_type(file_path.name)
            mime = mime or "application/octet-stream"
            encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
            cache[rel_path] = f"data:{mime};base64,{encoded}"
        return f"{match.group(1)}{cache[rel_path]}{match.group(3)}"

    return pattern.sub(replace, markdown)


def _build_dataset(file_bytes: bytes, extension: str):
    if extension == ".pdf":
        return PymuDocDataset(file_bytes)
    return ImageDataset(file_bytes)


async def _run_monkeyocr(file_bytes: bytes, extension: str) -> str:
    """Run MonkeyOCR in a worker thread and return Markdown with inline tables/images."""

    model = model_manager.get_model()
    if model is None:
        raise RuntimeError("MonkeyOCR model is not initialized")

    loop = asyncio.get_running_loop()
    supports_async = model_manager.get_async_support()
    model_lock = model_manager.get_model_lock()

    def pipeline() -> tuple[str, Path]:
        dataset = _build_dataset(file_bytes, extension)
        temp_root = Path(tempfile.mkdtemp(prefix="submission_api_"))
        images_dir = temp_root / IMAGE_FOLDER_NAME
        images_dir.mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(str(images_dir))
        infer_result = dataset.apply(
            doc_analyze_llm,
            MonkeyOCR_model=model,
            split_pages=False,
            pred_abandon=False,
        )
        pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=model)
        markdown = pipe_result.get_markdown(IMAGE_FOLDER_NAME)
        return markdown, temp_root

    if supports_async:
        markdown, temp_root = await loop.run_in_executor(None, pipeline)
    else:
        async with model_lock:
            markdown, temp_root = await loop.run_in_executor(None, pipeline)

    try:
        return _embed_local_images(markdown, temp_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)) -> JSONResponse:
    """Parse a PDF or image and return Markdown that preserves reading order."""

    extension = _infer_extension(file)
    file_bytes = await file.read()
    await file.close()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        markdown = await _run_monkeyocr(file_bytes, extension)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to parse document: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to parse document.")

    return JSONResponse(
        status_code=200,
        content={"status": "success", "markdown": markdown},
    )


@app.get("/health")
async def healthcheck() -> dict[str, object]:
    """Lightweight health endpoint."""

    return {
        "status": "ok",
        "model_loaded": model_manager.is_model_loaded(),
        "async_model": model_manager.get_async_support(),
    }


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("submission_api:app", host="0.0.0.0", port=8080, reload=False)
