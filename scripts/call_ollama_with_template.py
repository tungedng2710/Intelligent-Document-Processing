"""
Call Ollama with a prompt template and a JSON schema template.

Usage (single image):
    python scripts/call_ollama_with_template.py \
        --prompt-template data/prompts/healthcare_types/prompt_with_template.txt \
        --json-template data/prompts/healthcare_types/page-38-template.json \
        --image data/healthcare/hoso1/images/page-38.png \
        --output-dir data/healthcare/hoso1/pred_qwen3.5 \
        --model qwen3.5:9b-q4_K_M \
        --host http://localhost:11434

Usage (multi-page PDF with batch_size):
    python scripts/call_ollama_with_template.py \
        --prompt-template data/prompts/healthcare_types/prompt_with_template.txt \
        --json-template data/prompts/healthcare_types/page-38-template.json \
        --pdf document.pdf \
        --output-dir data/healthcare/hoso1/pred_qwen3.5 \
        --model qwen3.5:9b-q4_K_M \
        --host http://localhost:11434 \
        --batch_size 2
"""

import argparse
import base64
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pymupdf  # PyMuPDF
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call_ollama(image_paths: list[Path], prompt: str, model: str, host: str) -> str:
    """Send one or more page images + filled prompt to Ollama and return raw content."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encode_image_b64(p) for p in image_paths],
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "think": False,
        },
    }
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=500)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_content(content: str):
    """Try to parse content as JSON, including stripping markdown code fences."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    return {"raw": content}


def build_prompt(prompt_template: str, json_template: dict) -> str:
    return prompt_template.replace(
        "{json_template}",
        json.dumps(json_template, ensure_ascii=False, indent=2),
    )


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> tuple[list[Path], Path]:
    """Rasterize each PDF page to a PNG in a temp directory.

    Returns (image_paths, tmp_dir). Caller must delete tmp_dir when done.
    """
    doc = pymupdf.open(str(pdf_path))
    tmp_dir = Path(tempfile.mkdtemp(prefix="ollama_tpl_pdf_"))
    image_paths: list[Path] = []
    for i in range(len(doc)):
        page = doc[i]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = tmp_dir / f"page-{i + 1:03d}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)
    doc.close()
    logger.info(f"Converted {len(image_paths)} page(s) from '{pdf_path.name}' to {tmp_dir}")
    return image_paths, tmp_dir


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def process_single_image(
    image_path: Path,
    prompt: str,
    model: str,
    host: str,
    output_dir: Path,
) -> None:
    """Process a single image and save result to output_dir/<stem>.json."""
    logger.info(f"Processing single image: {image_path.name}")
    t0 = time.time()
    content = call_ollama([image_path], prompt, model, host)
    elapsed = time.time() - t0
    result = parse_content(content)
    out_path = output_dir / image_path.with_suffix(".json").name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info(f"Saved to {out_path} ({elapsed:.1f}s)")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def process_pdf(
    pdf_path: Path,
    prompt: str,
    model: str,
    host: str,
    output_dir: Path,
    batch_size: int = 1,
    dpi: int = 150,
) -> None:
    """Convert PDF to page images, process in batches, save per-batch JSONs."""
    image_paths, tmp_dir = pdf_to_images(pdf_path, dpi=dpi)
    total = len(image_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = [image_paths[i:i + batch_size] for i in range(0, total, batch_size)]
    all_results = []
    try:
        for idx, batch in enumerate(batches):
            pages = list(range(idx * batch_size + 1, idx * batch_size + len(batch) + 1))
            logger.info(f"[batch {idx + 1}/{len(batches)}] pages {pages}")
            t0 = time.time()
            content = call_ollama(batch, prompt, model, host)
            elapsed = time.time() - t0
            result = parse_content(content)
            batch_record = {
                "batch_index": idx,
                "pages": pages,
                "elapsed_seconds": round(elapsed, 2),
                "result": result,
            }
            all_results.append(batch_record)
            # Save individual batch file
            batch_out = output_dir / f"batch-{idx + 1:03d}_pages-{pages[0]}-{pages[-1]}.json"
            batch_out.write_text(json.dumps(batch_record, ensure_ascii=False, indent=2))
            logger.info(f"  -> saved {batch_out.name} ({elapsed:.1f}s)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Save combined summary
    summary = {
        "source": str(pdf_path),
        "model": model,
        "batch_size": batch_size,
        "total_pages": total,
        "batches": all_results,
    }
    summary_out = output_dir / "result.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"Summary saved to {summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Call Ollama with a prompt template and JSON schema template."
    )
    parser.add_argument("--prompt-template", required=True, help="Path to the prompt template .txt file")
    parser.add_argument("--json-template", required=True, help="Path to the JSON template file")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single input image file")
    src.add_argument("--pdf", help="Path to a (multi-page) PDF file")

    parser.add_argument("--output-dir", required=True, help="Directory to save the output JSON(s)")
    parser.add_argument("--model", default="gemma4:e4b-it-bf16", help="Ollama model name")
    parser.add_argument("--host", default="http://0.0.0.0:11434", help="Ollama host URL")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of pages to feed into the VLM per call (PDF mode only, default: 1)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF rasterization (default: 150)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        logger.error("--batch_size must be >= 1")
        sys.exit(1)

    prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")
    with open(args.json_template, encoding="utf-8") as f:
        json_template = json.load(f)
    prompt = build_prompt(prompt_template, json_template)
    output_dir = Path(args.output_dir)

    try:
        if args.image:
            image_path = Path(args.image)
            if not image_path.is_file():
                logger.error(f"Image not found: {image_path}")
                sys.exit(1)
            process_single_image(image_path, prompt, args.model, args.host, output_dir)
        else:
            pdf_path = Path(args.pdf)
            if not pdf_path.is_file():
                logger.error(f"PDF not found: {pdf_path}")
                sys.exit(1)
            process_pdf(pdf_path, prompt, args.model, args.host, output_dir, args.batch_size, args.dpi)
    except requests.exceptions.Timeout:
        logger.error("TIMEOUT while calling Ollama")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        logger.error(f"CONNECTION ERROR — is Ollama running at {args.host}?")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
