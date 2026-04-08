"""
Call Ollama vision model on a flat folder of images with optional chunking.

Chunking groups N consecutive sorted images into a single inference call.
Default chunk=1 → single-page inference (one image per call).

Usage:
    python scripts/call_ollama_folder.py \
        --folder data/healthcare/hoso1_pages \
        --prompt data/prompts/document_extraction_v2_semantic.txt \
        --model qwen3.5_ocr \
        --port localhost:11434

    # Multi-page chunks (e.g. 2 pages per call):
    python scripts/call_ollama_folder.py \
        --folder data/healthcare/hoso1_pages \
        --prompt data/prompts/document_extraction_v2_semantic.txt \
        --model qwen3.5_ocr \
        --port localhost:11434 \
        --chunk 2 \
        --output data/result_hoso1.json
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


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def call_ollama_chat(
    image_paths: list[Path],
    prompt_text: str,
    model: str,
    base_url: str,
    timeout: int = 300,
) -> dict:
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
    }

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def try_parse_json(content: str):
    """Try to parse raw JSON or JSON inside a markdown code block."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Ollama vision inference on a flat folder of images"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing image files",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Number of consecutive images per inference call (default: 1 = single-page)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to prompt .txt file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5_ocr",
        help="Ollama model name",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="localhost:11434",
        help="Ollama host:port (e.g. localhost:11434)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save combined JSON results (default: <folder_parent>/<folder_name>.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing predictions",
    )
    args = parser.parse_args()

    if args.chunk < 1:
        logger.error("--chunk must be >= 1")
        sys.exit(1)

    folder = Path(args.folder)
    if not folder.is_dir():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    # Collect and sort images
    image_files = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        logger.error(f"No image files found in {folder}")
        sys.exit(1)
    logger.info(f"Found {len(image_files)} image(s) in {folder}")

    # Load prompt
    if args.prompt:
        prompt_path = Path(args.prompt)
        if not prompt_path.is_file():
            logger.error(f"Prompt file not found: {prompt_path}")
            sys.exit(1)
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    else:
        prompt_text = "Extract all text and structured data from this document image."
        logger.info("No prompt file specified — using default prompt")

    base_url = f"http://{args.port}"

    # Verify Ollama is reachable
    try:
        r = requests.get(f"{base_url}/api/version", timeout=5)
        r.raise_for_status()
        logger.info(f"Ollama reachable at {base_url} — version {r.json().get('version')}")
    except Exception as exc:
        logger.error(f"Cannot reach Ollama at {base_url}: {exc}")
        sys.exit(1)

    # Output path — default: <folder_parent>/<folder_name>.json
    output_path = Path(args.output) if args.output else folder.parent / f"{folder.name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if not overwriting
    results = []
    processed_keys: set[str] = set()
    if output_path.exists() and not args.overwrite:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            results = existing if isinstance(existing, list) else []
            processed_keys = {
                r["chunk_key"] for r in results if "chunk_key" in r
            }
            logger.info(f"Resuming — {len(processed_keys)} chunk(s) already processed")
        except Exception:
            logger.warning("Could not parse existing output; starting fresh")

    # Build chunks of image paths
    chunks = [
        image_files[i : i + args.chunk]
        for i in range(0, len(image_files), args.chunk)
    ]
    logger.info(
        f"Processing {len(chunks)} chunk(s) "
        f"(chunk_size={args.chunk}, model={args.model})"
    )

    total_errors = 0

    for idx, chunk_paths in enumerate(chunks, start=1):
        chunk_key = "|".join(p.name for p in chunk_paths)

        if chunk_key in processed_keys:
            logger.info(f"[chunk {idx}/{len(chunks)}] skip — already done: {chunk_key}")
            continue

        logger.info(
            f"[chunk {idx}/{len(chunks)}] "
            f"{[p.name for p in chunk_paths]} ..."
        )
        t0 = time.time()
        try:
            response = call_ollama_chat(
                chunk_paths, prompt_text, args.model, base_url, timeout=args.timeout
            )
            elapsed = round(time.time() - t0, 2)
            content = response.get("message", {}).get("content", "")

            record = {
                "chunk_index": idx,
                "chunk_key": chunk_key,
                "images": [str(p) for p in chunk_paths],
                "model": args.model,
                "content": content,
                "elapsed_seconds": elapsed,
            }

            parsed = try_parse_json(content)
            if parsed is not None:
                record["parsed_json"] = parsed

            results.append(record)
            logger.info(f"  -> done ({elapsed}s)")

        except requests.exceptions.Timeout:
            logger.error(f"  -> TIMEOUT for chunk {idx}")
            total_errors += 1
            continue
        except requests.exceptions.ConnectionError:
            logger.error(f"  -> CONNECTION ERROR — is Ollama running at {base_url}?")
            total_errors += 1
            continue
        except Exception as exc:
            logger.error(f"  -> ERROR for chunk {idx}: {exc}")
            total_errors += 1
            continue

        # Save after every chunk so progress is not lost on failure
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    logger.info(
        f"Done. {len(results)} chunk(s) saved to {output_path} "
        f"({total_errors} error(s))"
    )


if __name__ == "__main__":
    main()
