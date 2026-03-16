"""
Convert markdown content from prediction files to structured JSON key-value pairs
using an LLM (Ollama).

Reads prediction .json files (which contain an OCR markdown "content" field),
sends the markdown to an LLM to extract structured key-value pairs, and saves
the result alongside the original prediction.

Usage:
    # Process all categories in train_data_v1.2
    python scripts/md_to_json.py \
        --data_path data/r3_bank_reports/train_data_v1.2

    # Process a single category
    python scripts/md_to_json.py \
        --data_path data/r3_bank_reports/train_data_v1.2 \
        --category giay_rut_tien

    # Use a different model / port
    python scripts/md_to_json.py \
        --data_path data/r3_bank_reports/train_data_v1.2 \
        --model qwen3.5:2b-bf16 \
        --port 0.0.0.0:11434

    # Overwrite existing extractions
    python scripts/md_to_json.py \
        --data_path data/r3_bank_reports/train_data_v1.2 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt for markdown -> JSON extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a document information extraction engine. Given the OCR markdown text of a Vietnamese bank document below, extract all key-value information into a single flat or nested JSON object.

Rules:
1. Use the exact field labels from the document as JSON keys (keep Vietnamese text as-is).
2. Values must be extracted exactly as they appear — do not correct, translate, or interpret.
3. If the document contains a table, represent it as a JSON array of objects using column headers as keys.
4. Ignore decorative elements, signatures, page numbers, and stamps.
5. If a field is present but empty, use null.
6. Return ONLY valid JSON — no explanation, no markdown fences, no extra text.

--- DOCUMENT START ---
{content}
--- DOCUMENT END ---

JSON:"""


def call_ollama(
    content: str,
    model: str,
    base_url: str,
    timeout: int = 300,
) -> str:
    """Send markdown content to Ollama and return the raw response text."""
    prompt = EXTRACTION_PROMPT.format(content=content)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
        },
    }

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def parse_json_response(text: str) -> dict | list | None:
    """Try to parse JSON from LLM response, handling markdown fences."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding first { ... } or [ ... ]
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

    return None


def process_predictions(
    pred_dir: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    overwrite: bool = False,
) -> dict:
    """Process all prediction files in a directory."""
    pred_files = sorted(pred_dir.glob("*.json"))
    if not pred_files:
        logger.warning(f"No prediction files in {pred_dir}")
        return {"folder": pred_dir.parent.name, "processed": 0, "errors": 0, "parse_failures": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0
    parse_failures = 0

    for pred_file in pred_files:
        out_file = output_dir / pred_file.name
        if out_file.exists() and not overwrite:
            logger.info(f"[skip] {pred_file.name} — extraction already exists")
            processed += 1
            continue

        try:
            pred = json.loads(pred_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"[error] Cannot read {pred_file.name}: {exc}")
            errors += 1
            continue

        md_content = pred.get("content", "")
        if not md_content.strip():
            logger.warning(f"[skip] {pred_file.name} — empty content")
            continue

        logger.info(f"[{pred_dir.parent.name}] Extracting {pred_file.name} ...")
        t0 = time.time()

        try:
            raw_response = call_ollama(md_content, model, base_url)
            elapsed = time.time() - t0

            parsed = parse_json_response(raw_response)

            result = {
                "image": pred.get("image", ""),
                "model": pred.get("model", ""),
                "extraction_model": model,
                "category": pred.get("category", ""),
                "content": md_content,
                "extracted_json": parsed,
                "raw_llm_response": raw_response if parsed is None else None,
                "extraction_elapsed_seconds": round(elapsed, 2),
            }

            if parsed is None:
                parse_failures += 1
                logger.warning(f"  -> JSON parse failed for {pred_file.name}, saving raw response")

            out_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"  -> saved {out_file.name} ({elapsed:.1f}s)")
            processed += 1

        except requests.exceptions.Timeout:
            logger.error(f"  -> TIMEOUT for {pred_file.name}")
            errors += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"  -> CONNECTION ERROR — is Ollama running at {base_url}?")
            errors += 1
            break
        except Exception as exc:
            logger.error(f"  -> ERROR for {pred_file.name}: {exc}")
            errors += 1

    return {
        "folder": pred_dir.parent.name,
        "processed": processed,
        "errors": errors,
        "parse_failures": parse_failures,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert markdown predictions to structured JSON via LLM"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/r3_bank_reports/train_data_v1.2",
        help="Path to data directory containing category folders",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process only this category folder (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:2b-bf16",
        help="Ollama model name for extraction",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="0.0.0.0:7860",
        help="Ollama host:port",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="extractions",
        help="Name of output subfolder inside each category (default: extractions)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extraction files",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    base_url = f"http://{args.port}"

    # Verify Ollama is reachable
    try:
        r = requests.get(f"{base_url}/api/version", timeout=5)
        r.raise_for_status()
        logger.info(f"Ollama OK at {base_url} (version: {r.json()})")
    except Exception as exc:
        logger.error(f"Cannot reach Ollama at {base_url}: {exc}")
        sys.exit(1)

    # Discover category folders
    if args.category:
        folders = [data_path / args.category]
        if not folders[0].is_dir():
            logger.error(f"Category folder not found: {folders[0]}")
            sys.exit(1)
    else:
        folders = sorted(
            d for d in data_path.iterdir() if d.is_dir() and (d / "predictions").is_dir()
        )

    if not folders:
        logger.error("No category folders with predictions/ found")
        sys.exit(1)

    logger.info(f"Categories to process: {[f.name for f in folders]}")
    logger.info(f"Model: {args.model}")

    all_stats = []
    for folder in folders:
        pred_dir = folder / "predictions"
        output_dir = folder / args.output_suffix
        stats = process_predictions(pred_dir, output_dir, args.model, base_url, args.overwrite)
        all_stats.append(stats)
        logger.info(f"  [{folder.name}] processed={stats['processed']} errors={stats['errors']} parse_failures={stats['parse_failures']}")

    # Summary
    total_processed = sum(s["processed"] for s in all_stats)
    total_errors = sum(s["errors"] for s in all_stats)
    total_parse_failures = sum(s["parse_failures"] for s in all_stats)
    logger.info(
        f"\nDone. Total: processed={total_processed}, errors={total_errors}, "
        f"parse_failures={total_parse_failures}"
    )


if __name__ == "__main__":
    main()
