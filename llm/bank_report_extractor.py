"""
Bank report extraction pipeline using Qwen3-VL via Ollama.

Sends PDF pages or images to Qwen3-VL with the bank_report_ver_1.1 prompt
to extract structured JSON from banking documents.
"""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "bank_report_ver_1.1.txt"

DEFAULT_OLLAMA_URL = "http://0.0.0.0:7860"
DEFAULT_MODEL = "qwen3-vl:8b-instruct-bf16"
DPI = 200  # Higher DPI for extraction accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt() -> str:
    """Load the bank report extraction prompt from file."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _encode_image_bytes(data: bytes) -> str:
    """Base64-encode raw image bytes."""
    return base64.b64encode(data).decode("utf-8")


def _pdf_page_to_png(doc: fitz.Document, page_idx: int, dpi: int = DPI) -> bytes:
    """Render a single PDF page to PNG bytes."""
    page = doc.load_page(page_idx)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _image_file_to_png(path: str | Path) -> bytes:
    """Convert any supported image format to PNG bytes."""
    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_vlm_response(raw: str) -> dict[str, Any] | list | str:
    """
    Parse VLM output into a JSON object/array.
    Robustly handles markdown fences, <think> blocks, etc.
    Returns the parsed JSON, or a dict with raw_response on failure.
    """
    # Strip <think>...</think> blocks (Qwen3-VL thinking mode)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 1) Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Try extracting from markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Try extracting array from markdown fences
    m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 4) Try finding any JSON object (greedy)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text
    return {"raw_response": raw, "parse_error": "Could not extract JSON from VLM response"}


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------


def extract_from_image_ollama(
    image_b64: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    prompt: str | None = None,
) -> dict[str, Any] | list:
    """
    Send a single page image to Qwen3-VL via Ollama for bank report extraction.

    Returns the parsed JSON extraction result.
    """
    if prompt is None:
        prompt = _load_prompt()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 8192,
            "num_ctx": 8000,
        },
    }

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()

    raw_response = resp.json().get("response", "")
    return _parse_vlm_response(raw_response)


# ---------------------------------------------------------------------------
# PDF / Image extraction pipeline
# ---------------------------------------------------------------------------


def extract_pdf(
    pdf_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    document_type: str = "",
) -> dict[str, Any]:
    """
    Extract structured data from each page of a PDF using the bank report prompt.

    Returns a dict with status and data.
    """
    prompt = _load_prompt()
    if document_type:
        prompt = f"Document type: {document_type}\n\n{prompt}"

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    pages: list[dict[str, Any]] = []

    for i in range(page_count):
        print(f"  Extracting page {i} / {page_count - 1} ...")
        png_bytes = _pdf_page_to_png(doc, i)
        b64 = _encode_image_bytes(png_bytes)
        result = extract_from_image_ollama(b64, model=model, ollama_url=ollama_url, prompt=prompt)
        pages.append({"page": i, "data": result})

    doc.close()

    # If single page, flatten
    if page_count == 1:
        return {
            "status": "success",
            "data": pages[0]["data"],
        }

    return {
        "status": "success",
        "data": pages,
    }


def extract_image(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    document_type: str = "",
) -> dict[str, Any]:
    """
    Extract structured data from a single image file.
    """
    prompt = _load_prompt()
    if document_type:
        prompt = f"Document type: {document_type}\n\n{prompt}"

    png_bytes = _image_file_to_png(image_path)
    b64 = _encode_image_bytes(png_bytes)
    result = extract_from_image_ollama(b64, model=model, ollama_url=ollama_url, prompt=prompt)

    return {
        "status": "success",
        "data": result,
    }


def extract_file(
    file_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    document_type: str = "",
) -> dict[str, Any]:
    """
    Extract structured data from a PDF or image file. Auto-detects by extension.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf(path, model=model, ollama_url=ollama_url, document_type=document_type)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
        return extract_image(path, model=model, ollama_url=ollama_url, document_type=document_type)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bank_report_extractor.py <file_path> [ollama_url]")
        print("  file_path: PDF or image (png, jpg, jpeg)")
        print("  ollama_url: Ollama server URL (default: http://0.0.0.0:7860)")
        sys.exit(1)

    file_path = sys.argv[1]
    ollama_url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OLLAMA_URL

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Extracting: {file_path}")
    print(f"Using model: {DEFAULT_MODEL} at {ollama_url}")
    print(f"Prompt: {PROMPT_PATH.name}")
    print("---")

    result = extract_file(file_path, ollama_url=ollama_url)
    print("---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
