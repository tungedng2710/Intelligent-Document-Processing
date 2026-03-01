"""
Document classification pipeline using Qwen3-VL via Ollama.

Splits a PDF into page images, classifies each page using the VLM,
and groups consecutive same-type pages into logical documents.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DOCUMENT_TYPES = [
    "LIET_KE_GIAO_DICH",
    "SO_QUY",
    "GIAY_DE_NGHI_TIEP_QUY",
    "GIAY_RUT_TIEN",
    "GIAY_GUI_TIEN_TIET_KIEM",
    "PHIEU_HACH_TOAN",
    "GIAY_DE_NGHI_SU_DUNG_DICH_VU_INTERNET_BANKING",
    "GIAY_DE_NGHI_THAY_DOI_THONG_TIN_DICH_VU_INTERNET_BANKING",
    "TO_TRINH_THAM_DINH_TIN_DUNG",
    "CCCD",
    "LENH_CHUYEN_TIEN",
    "GIAY_PHONG_TOA_TAM_KHOA_TAI_KHOAN",
    "OTHER",
]

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "document_classification.txt"

DEFAULT_OLLAMA_URL = "http://0.0.0.0:7860"
DEFAULT_MODEL = "qwen3-vl:8b-instruct-bf16"
DPI = 150  # Resolution for rendering PDF pages to images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt() -> str:
    """Load the classification prompt from file."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _encode_image_bytes(data: bytes) -> str:
    """Base64-encode raw image bytes."""
    return base64.b64encode(data).decode("utf-8")


def _encode_image_file(path: str | Path) -> str:
    """Base64-encode an image file."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


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


def _parse_vlm_response(raw: str) -> dict[str, Any]:
    """
    Parse VLM output into {"document_type": ..., "confidence": ...}.
    Robustly handles markdown fences, <think> blocks, etc.
    """
    # Strip <think>...</think> blocks (Qwen3-VL thinking mode)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try to find JSON in the response
    # 1) Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Try extracting from markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Try finding any JSON object
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback
    return {"document_type": "OTHER", "confidence": 0.0, "raw_response": raw}


def _validate_document_type(doc_type: str) -> str:
    """Ensure the document type is one of the valid codes."""
    normalized = doc_type.strip().upper()
    if normalized in VALID_DOCUMENT_TYPES:
        return normalized
    return "OTHER"


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------


def classify_image_ollama(
    image_b64: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """
    Send a single page image to Qwen3-VL via Ollama and get classification.

    Returns: {"document_type": str, "confidence": float}
    """
    prompt = _load_prompt()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
            "num_ctx": 8000,
        },
    }

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()

    raw_response = resp.json().get("response", "")
    parsed = _parse_vlm_response(raw_response)
    parsed["document_type"] = _validate_document_type(parsed.get("document_type", "OTHER"))
    if "confidence" not in parsed:
        parsed["confidence"] = 0.0
    parsed["confidence"] = round(float(parsed["confidence"]), 2)

    return parsed


# ---------------------------------------------------------------------------
# PDF / Image classification pipeline
# ---------------------------------------------------------------------------


def classify_pdf(
    pdf_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """
    Classify each page of a PDF and group consecutive same-type pages
    into logical documents.

    Returns the full classification result dict.
    """
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    page_results: list[dict[str, Any]] = []

    for i in range(page_count):
        png_bytes = _pdf_page_to_png(doc, i)
        b64 = _encode_image_bytes(png_bytes)
        result = classify_image_ollama(b64, model=model, ollama_url=ollama_url)
        result["page"] = i
        page_results.append(result)
        print(f"  Page {i}: {result['document_type']} (confidence: {result['confidence']})")

    doc.close()
    classification = _group_pages(page_results)

    return {
        "status": "success",
        "page_count": page_count,
        "classification": classification,
    }


def classify_image(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """
    Classify a single image file.

    Returns the full classification result dict.
    """
    png_bytes = _image_file_to_png(image_path)
    b64 = _encode_image_bytes(png_bytes)
    result = classify_image_ollama(b64, model=model, ollama_url=ollama_url)
    result["page"] = 0

    return {
        "status": "success",
        "page_count": 1,
        "classification": [
            {
                "index": 0,
                "document_type": result["document_type"],
                "pages": [0],
            }
        ],
    }


def classify_file(
    file_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """
    Classify a PDF or image file. Auto-detects file type by extension.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return classify_pdf(path, model=model, ollama_url=ollama_url)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
        return classify_image(path, model=model, ollama_url=ollama_url)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Grouping logic
# ---------------------------------------------------------------------------


def _group_pages(page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group consecutive pages with the same document_type into logical documents.

    For example, if pages [0] = CCCD, [1] = CCCD, [2] = OTHER, [3] = CCCD,
    that becomes 3 groups:
        - index 0: CCCD, pages [0, 1]
        - index 1: OTHER, pages [2]
        - index 2: CCCD, pages [3]
    """
    if not page_results:
        return []

    groups: list[dict[str, Any]] = []
    current_type = page_results[0]["document_type"]
    current_pages = [page_results[0]["page"]]

    for pr in page_results[1:]:
        if pr["document_type"] == current_type:
            current_pages.append(pr["page"])
        else:
            groups.append({
                "index": len(groups),
                "document_type": current_type,
                "pages": current_pages,
            })
            current_type = pr["document_type"]
            current_pages = [pr["page"]]

    # Don't forget the last group
    groups.append({
        "index": len(groups),
        "document_type": current_type,
        "pages": current_pages,
    })

    return groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_classifier.py <file_path> [ollama_url]")
        print("  file_path: PDF or image (png, jpg, jpeg)")
        print("  ollama_url: Ollama server URL (default: http://localhost:11434)")
        sys.exit(1)

    file_path = sys.argv[1]
    ollama_url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OLLAMA_URL

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Classifying: {file_path}")
    print(f"Using model: {DEFAULT_MODEL} at {ollama_url}")
    print("---")

    result = classify_file(file_path, ollama_url=ollama_url)
    print("---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
