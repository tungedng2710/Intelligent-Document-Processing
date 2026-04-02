"""
Call vLLM vision model (OpenAI-compatible API) on document images and save predictions.

Usage (single or multi-page image mode):
    python scripts/call_vllm.py \
        --images page1.png page2.png \
        --prompt data/prompts/bank_report_ver_1.0.txt \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --port localhost:9888 \
        --output data/result.json

Usage (folder mode):
    python scripts/call_vllm.py \
        --data_path data/r3_bank_reports/train_data_v1.2 \
        --prompt data/prompts/bank_report_ver_1.0.txt \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --port localhost:9888
"""

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def get_mime_type(image_path: Path) -> str:
    return MIME_MAP.get(image_path.suffix.lower(), "image/png")


def call_vllm_chat(
    image_paths: list[Path],
    prompt_text: str,
    model: str,
    base_url: str,
    timeout: int = 300,
) -> dict:
    """Send one or more images + prompt to vLLM /v1/chat/completions and return the parsed response."""
    content_parts = []
    for img_path in image_paths:
        b64 = encode_image_base64(img_path)
        mime = get_mime_type(img_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
            },
        })
    content_parts.append({
        "type": "text",
        "text": prompt_text,
    })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_parts,
            }
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": 12000,
    }

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_content(response: dict) -> str:
    """Extract assistant message content from OpenAI-style response."""
    choices = response.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


def try_parse_json(content: str):
    """Try to parse content as JSON, including from markdown code blocks."""
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
            response = call_vllm_chat([img_path], prompt_text, model, base_url)
            elapsed = time.time() - t0

            content = extract_content(response)

            prediction = {
                "image": str(img_path.name),
                "model": model,
                "category": folder.name,
                "content": content,
                "elapsed_seconds": round(elapsed, 2),
            }

            parsed = try_parse_json(content)
            if parsed is not None:
                prediction["parsed_json"] = parsed

            out_file.write_text(json.dumps(prediction, ensure_ascii=False, indent=2))
            logger.info(f"  -> saved {out_file.name} ({elapsed:.1f}s)")
            processed += 1

        except requests.exceptions.Timeout:
            logger.error(f"  -> TIMEOUT for {img_path.name}")
            errors += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"  -> CONNECTION ERROR — is vLLM running at {base_url}?")
            errors += 1
        except Exception as exc:
            logger.error(f"  -> ERROR for {img_path.name}: {exc}")
            errors += 1

    return {"folder": folder.name, "processed": processed, "errors": errors}


def process_images(
    image_paths: list[Path],
    prompt_text: str,
    model: str,
    base_url: str,
    output_path: Path | None = None,
) -> None:
    """Process one or more images (pages) and print (and optionally save) the result."""
    logger.info(f"Processing {len(image_paths)} image(s): {[str(p) for p in image_paths]}")
    t0 = time.time()
    response = call_vllm_chat(image_paths, prompt_text, model, base_url)
    elapsed = time.time() - t0

    content = extract_content(response)

    prediction = {
        "images": [str(p) for p in image_paths],
        "model": model,
        "content": content,
        "elapsed_seconds": round(elapsed, 2),
    }

    parsed = try_parse_json(content)
    if parsed is not None:
        prediction["parsed_json"] = parsed

    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    logger.info(f"Completed in {elapsed:.1f}s")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2))
        logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Call vLLM on document images")
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to one or more image files (pages) to process",
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
        default=None,
        help="Path to data directory containing category folders (folder mode)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to prompt .txt file (if omitted, uses a default prompt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name served by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="localhost:9888",
        help="vLLM host:port (e.g. localhost:9888)",
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

    base_url = args.port if args.port.startswith("http") else f"http://{args.port}"

    # Verify vLLM is reachable and auto-detect model
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        logger.info(f"vLLM reachable at {base_url} — models: {[m.get('id') for m in models]}")
        if args.model == "default" and models:
            args.model = models[0].get("id", "default")
            logger.info(f"Auto-selected model: {args.model}")
    except Exception as exc:
        logger.error(f"Cannot reach vLLM at {base_url}: {exc}")
        sys.exit(1)

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
        output_path = Path(args.output) if args.output else None
        try:
            process_images(image_paths, prompt_text, args.model, base_url, output_path)
        except requests.exceptions.Timeout:
            logger.error("TIMEOUT while calling vLLM")
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error(f"CONNECTION ERROR — is vLLM running at {base_url}?")
            sys.exit(1)
        except Exception as exc:
            logger.error(f"ERROR: {exc}")
            sys.exit(1)
        return

    # ── Folder mode ──────────────────────────────────────────────────────────
    if not args.data_path:
        logger.error("Must specify either --images or --data_path")
        sys.exit(1)

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    folders = sorted(
        d for d in data_path.iterdir() if d.is_dir() and (d / "images").is_dir()
    )
    if not folders:
        logger.error(f"No category folders with images/ found in {data_path}")
        sys.exit(1)

    logger.info(f"Found {len(folders)} category folders: {[f.name for f in folders]}")
    logger.info(f"Model: {args.model} | Prompt: {args.prompt or 'default'}")

    summary = []
    for folder in folders:
        result = process_folder(folder, prompt_text, args.model, base_url)
        summary.append(result)

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
