"""
Call Ollama vision model on document images and save predictions.

Usage (folder mode):
    python scripts/call_ollama.py \
        --data_path data/r3_bank_reports/train_data_v1.2 \
        --prompt data/prompts/bank_report_ver_1.0.txt \
        --model qwen3.5:latest \
        --port localhost:11434

Usage (image(s) mode):
    python scripts/call_ollama.py \
        --images page1.png page2.png \
        --prompt data/prompts/bank_report_ver_1.0.txt \
        --model qwen3.5:latest \
        --port localhost:11434 \
        --batch_size 2 \
        --output result.json   # optional

Usage (PDF mode):
    python scripts/call_ollama.py \
        --pdf document.pdf \
        --prompt data/prompts/bank_report_ver_1.0.txt \
        --model qwen3.5:latest \
        --port localhost:11434 \
        --batch_size 2 \
        --output result.json   # optional
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def call_ollama_chat(
    image_paths: list[Path],
    prompt_text: str,
    model: str,
    base_url: str,
    timeout: int = 300,
) -> dict:
    """Send one or more images + prompt to Ollama /api/chat and return the parsed response."""
    images_b64 = [encode_image_base64(p) for p in image_paths]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": images_b64,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
        },
    }

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def process_folder(
    folder: Path,
    prompt_text: str,
    model: str,
    base_url: str,
) -> dict:
    """Process all images in folder/images/ and save predictions."""
    images_dir = folder / "images"
    if not images_dir.is_dir():
        logger.warning(f"No images/ directory in {folder.name}, skipping")
        return {"folder": folder.name, "processed": 0, "errors": 0}

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        logger.warning(f"No images found in {images_dir}")
        return {"folder": folder.name, "processed": 0, "errors": 0}

    pred_dir = folder / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0

    for img_path in image_files:
        out_file = pred_dir / f"{img_path.stem}.json"
        if out_file.exists():
            logger.info(f"[skip] {img_path.name} — prediction already exists")
            processed += 1
            continue

        logger.info(f"[{folder.name}] Processing {img_path.name} ...")
        t0 = time.time()
        try:
            response = call_ollama_chat([img_path], prompt_text, model, base_url)
            elapsed = time.time() - t0

            content = response.get("message", {}).get("content", "")

            prediction = {
                "image": str(img_path.name),
                "model": model,
                "category": folder.name,
                "content": content,
                "elapsed_seconds": round(elapsed, 2),
            }

            parsed = _try_parse_json(content)
            if parsed is not None:
                prediction["parsed_json"] = parsed

            out_file.write_text(json.dumps(prediction, ensure_ascii=False, indent=2))
            logger.info(f"  -> saved {out_file.name} ({elapsed:.1f}s)")
            processed += 1

        except requests.exceptions.Timeout:
            logger.error(f"  -> TIMEOUT for {img_path.name}")
            errors += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"  -> CONNECTION ERROR — is Ollama running at {base_url}?")
            errors += 1
        except Exception as exc:
            logger.error(f"  -> ERROR for {img_path.name}: {exc}")
            errors += 1

    return {"folder": folder.name, "processed": processed, "errors": errors}


def _try_parse_json(content: str) -> dict | None:
    """Attempt to parse content as JSON, including from markdown code blocks."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> tuple[list[Path], Path]:
    """Convert each page of a PDF to a PNG in a temp directory.

    Returns (list_of_image_paths, tmp_dir). The caller is responsible for
    deleting tmp_dir when done.
    """
    doc = pymupdf.open(str(pdf_path))
    tmp_dir = Path(tempfile.mkdtemp(prefix="ollama_pdf_"))
    image_paths: list[Path] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = tmp_dir / f"page-{page_num + 1:03d}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)
    doc.close()
    logger.info(f"Converted {len(image_paths)} page(s) from {pdf_path.name} to {tmp_dir}")
    return image_paths, tmp_dir


def _make_batch_prediction(
    batch_paths: list[Path],
    batch_index: int,
    page_numbers: list[int],
    prompt_text: str,
    model: str,
    base_url: str,
) -> dict:
    """Call the VLM for a single batch of pages and return a prediction dict."""
    logger.info(
        f"[batch {batch_index + 1}] pages {page_numbers} — "
        f"{[p.name for p in batch_paths]}"
    )
    t0 = time.time()
    response = call_ollama_chat(batch_paths, prompt_text, model, base_url)
    elapsed = time.time() - t0
    content = response.get("message", {}).get("content", "")
    prediction = {
        "batch_index": batch_index,
        "pages": page_numbers,
        "images": [p.name for p in batch_paths],
        "content": content,
        "elapsed_seconds": round(elapsed, 2),
    }
    parsed = _try_parse_json(content)
    if parsed is not None:
        prediction["parsed_json"] = parsed
    logger.info(f"  -> done in {elapsed:.1f}s")
    return prediction


def process_images(
    image_paths: list[Path],
    prompt_text: str,
    model: str,
    base_url: str,
    output_path: Path | None = None,
    batch_size: int = 1,
) -> None:
    """Process one or more images in batches and print (and optionally save) the result."""
    total = len(image_paths)
    logger.info(f"Processing {total} image(s) with batch_size={batch_size}")

    batches = [image_paths[i:i + batch_size] for i in range(0, total, batch_size)]
    page_numbers_per_batch = [
        list(range(i + 1, min(i + batch_size, total) + 1))
        for i in range(0, total, batch_size)
    ]

    results = []
    for idx, (batch, pages) in enumerate(zip(batches, page_numbers_per_batch)):
        pred = _make_batch_prediction(batch, idx, pages, prompt_text, model, base_url)
        results.append(pred)

    output = {
        "source": "images",
        "model": model,
        "batch_size": batch_size,
        "total_pages": total,
        "results": results,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    total_elapsed = sum(r["elapsed_seconds"] for r in results)
    logger.info(f"Completed {len(results)} batch(es) in {total_elapsed:.1f}s total")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        logger.info(f"Saved to {output_path}")


def process_pdf(
    pdf_path: Path,
    prompt_text: str,
    model: str,
    base_url: str,
    output_path: Path | None = None,
    batch_size: int = 1,
    dpi: int = 150,
) -> None:
    """Convert a PDF to page images, process in batches, save results."""
    image_paths, tmp_dir = pdf_to_images(pdf_path, dpi=dpi)
    total = len(image_paths)
    batches = [image_paths[i:i + batch_size] for i in range(0, total, batch_size)]
    page_numbers_per_batch = [
        list(range(i + 1, min(i + batch_size, total) + 1))
        for i in range(0, total, batch_size)
    ]

    results = []
    try:
        for idx, (batch, pages) in enumerate(zip(batches, page_numbers_per_batch)):
            pred = _make_batch_prediction(batch, idx, pages, prompt_text, model, base_url)
            results.append(pred)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    output = {
        "source": str(pdf_path),
        "model": model,
        "batch_size": batch_size,
        "total_pages": total,
        "results": results,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    total_elapsed = sum(r["elapsed_seconds"] for r in results)
    logger.info(f"Completed {len(results)} batch(es) in {total_elapsed:.1f}s total")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Call Ollama on bank report images")
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to image files to process (mutually exclusive with --pdf and --data_path)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a (multi-page) PDF file to process (mutually exclusive with --images and --data_path)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of pages to feed into the VLM per call (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the JSON result",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/tungn197/idp/data/r3_bank_reports/train_data_v1.2",
        help="Path to data directory containing category folders",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to prompt .txt file (if omitted, sends image without a text prompt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl:8b-instruct-bf16",
        help="Ollama model name",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="localhost:11434",
        help="Ollama host:port (e.g. localhost:11434)",
    )
    args = parser.parse_args()

    if args.prompt:
        prompt_path = Path(args.prompt)
        if not prompt_path.is_file():
            logger.error(f"Prompt file not found: {prompt_path}")
            sys.exit(1)
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    else:
        prompt_text = "Extract all text and structured data from this document image."
        logger.info("No prompt file specified, using default prompt")

    base_url = f"http://{args.port}"

    # Verify Ollama is reachable
    try:
        r = requests.get(f"{base_url}/api/version", timeout=5)
        r.raise_for_status()
        logger.info(f"Ollama reachable at {base_url} — version {r.json().get('version')}")
    except Exception as exc:
        logger.error(f"Cannot reach Ollama at {base_url}: {exc}")
        sys.exit(1)

    # Validate mutually exclusive source arguments
    sources = sum([
        bool(args.images),
        bool(args.pdf),
        bool(args.data_path and not (args.images or args.pdf)),
    ])
    if bool(args.images) and bool(args.pdf):
        logger.error("--images and --pdf are mutually exclusive")
        sys.exit(1)

    if args.batch_size < 1:
        logger.error("--batch_size must be >= 1")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    # ── PDF mode ──────────────────────────────────────────────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        if pdf_path.suffix.lower() != ".pdf":
            logger.warning(f"File '{pdf_path.name}' does not have a .pdf extension; proceeding anyway")
        try:
            process_pdf(pdf_path, prompt_text, args.model, base_url, output_path, args.batch_size)
        except requests.exceptions.Timeout:
            logger.error("TIMEOUT while calling Ollama")
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error(f"CONNECTION ERROR — is Ollama running at {base_url}?")
            sys.exit(1)
        except Exception as exc:
            logger.error(f"ERROR: {exc}")
            sys.exit(1)
        return

    # ── Image(s) mode ─────────────────────────────────────────────────────────
    if args.images:
        image_paths = []
        for img_str in args.images:
            p = Path(img_str)
            if not p.is_file():
                logger.error(f"Image file not found: {p}")
                sys.exit(1)
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                logger.warning(
                    f"Unrecognised extension '{p.suffix}'; proceeding anyway"
                )
            image_paths.append(p)
        try:
            process_images(image_paths, prompt_text, args.model, base_url, output_path, args.batch_size)
        except requests.exceptions.Timeout:
            logger.error("TIMEOUT while calling Ollama")
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error(f"CONNECTION ERROR — is Ollama running at {base_url}?")
            sys.exit(1)
        except Exception as exc:
            logger.error(f"ERROR: {exc}")
            sys.exit(1)
        return

    # ── Folder mode ──────────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not data_path.is_dir():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    # Discover category folders (those with images/ subdirectories)
    folders = sorted(
        d for d in data_path.iterdir() if d.is_dir() and (d / "images").is_dir()
    )
    if not folders:
        logger.error(f"No category folders with images/ found in {data_path}")
        sys.exit(1)

    logger.info(f"Found {len(folders)} category folders: {[f.name for f in folders]}")
    logger.info(f"Model: {args.model} | Prompt: {args.prompt or 'default'}")

    # Process each folder
    summary = []
    for folder in folders:
        result = process_folder(folder, prompt_text, args.model, base_url)
        summary.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_processed = 0
    total_errors = 0
    for s in summary:
        print(f"  {s['folder']:40s}  processed={s['processed']}  errors={s['errors']}")
        total_processed += s["processed"]
        total_errors += s["errors"]
    print("-" * 60)
    print(f"  {'TOTAL':40s}  processed={total_processed}  errors={total_errors}")
    print(f"  Output: <data_path>/<category>/predictions/")
    print("=" * 60)


if __name__ == "__main__":
    main()
