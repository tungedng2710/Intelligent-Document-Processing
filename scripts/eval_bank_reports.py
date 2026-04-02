"""
Evaluate bank report extraction predictions against ground truth annotations.

Compares the predicted content (from predictions/ folder) to the ground truth
"json" field in annotations/ for each category folder.

Metrics:
  - Field-level accuracy: exact match of each key-value pair
  - Field-level similarity: normalized edit distance (Levenshtein)
  - JSON parse rate: % of predictions that produced valid JSON
  - Per-category and overall summary

Usage:
    python scripts/eval_bank_reports.py \
        --data_path data/r3_bank_reports/train_data_v1.2

    python scripts/eval_bank_reports.py \
        --data_path data/r3_bank_reports/train_data_v1.2 \
        --output data/outputs/metrics/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization & similarity
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison: strip, collapse whitespace, NFC unicode."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def normalized_similarity(s1: str, s2: str) -> float:
    """Return normalized similarity [0, 1] between two strings (1 = identical)."""
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


# ---------------------------------------------------------------------------
# JSON flattening
# ---------------------------------------------------------------------------

def flatten_json(obj, prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict/list into dot-separated key -> string value pairs."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v) if v is not None else ""
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v) if v is not None else ""
    else:
        items[prefix] = str(obj) if obj is not None else ""
    return items


# ---------------------------------------------------------------------------
# Parse prediction content to JSON
# ---------------------------------------------------------------------------

def try_parse_json(content: str) -> dict | None:
    """Try to extract JSON from prediction content (raw or markdown-wrapped)."""
    content = content.strip()

    # Direct JSON parse
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code block ```json ... ```
    pattern = r"```(?:json)?\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    # Try finding the first { ... } block
    brace_start = content.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(content[brace_start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    return None


# ---------------------------------------------------------------------------
# Evaluation of a single sample
# ---------------------------------------------------------------------------

def evaluate_sample(gt_json: dict, pred_json: dict | None) -> dict:
    """Compare a single predicted JSON against ground truth JSON."""
    gt_flat = flatten_json(gt_json)

    if pred_json is None:
        # No valid JSON was extracted from prediction
        return {
            "json_parsed": False,
            "total_fields": len(gt_flat),
            "matched_fields": 0,
            "missing_fields": len(gt_flat),
            "extra_fields": 0,
            "exact_match": 0,
            "field_accuracy": 0.0,
            "field_similarity": 0.0,
            "field_details": {},
        }

    pred_flat = flatten_json(pred_json)

    gt_keys = set(gt_flat.keys())
    pred_keys = set(pred_flat.keys())

    matched_keys = gt_keys & pred_keys
    missing_keys = gt_keys - pred_keys
    extra_keys = pred_keys - gt_keys

    exact_matches = 0
    total_similarity = 0.0
    field_details = {}

    for key in sorted(gt_keys):
        gt_val = normalize_text(gt_flat[key])
        if key in pred_flat:
            pred_val = normalize_text(pred_flat[key])
            sim = normalized_similarity(gt_val, pred_val)
            exact = gt_val == pred_val
            if exact:
                exact_matches += 1
            total_similarity += sim
            field_details[key] = {
                "gt": gt_val,
                "pred": pred_val,
                "similarity": round(sim, 4),
                "exact_match": exact,
            }
        else:
            total_similarity += 0.0
            field_details[key] = {
                "gt": gt_val,
                "pred": None,
                "similarity": 0.0,
                "exact_match": False,
            }

    n_gt = len(gt_keys)
    return {
        "json_parsed": True,
        "total_fields": n_gt,
        "matched_fields": len(matched_keys),
        "missing_fields": len(missing_keys),
        "extra_fields": len(extra_keys),
        "exact_match": exact_matches,
        "field_accuracy": exact_matches / n_gt if n_gt > 0 else 0.0,
        "field_similarity": total_similarity / n_gt if n_gt > 0 else 0.0,
        "field_details": field_details,
    }


# ---------------------------------------------------------------------------
# Process a category folder
# ---------------------------------------------------------------------------

def evaluate_folder(folder: Path) -> dict:
    """Evaluate all predictions in folder/predictions/ against folder/annotations/."""
    annotations_dir = folder / "annotations"
    predictions_dir = folder / "predictions"

    if not predictions_dir.is_dir():
        logger.warning(f"No predictions/ in {folder.name}, skipping")
        return {"folder": folder.name, "samples": [], "error": "no predictions dir"}

    if not annotations_dir.is_dir():
        logger.warning(f"No annotations/ in {folder.name}, skipping")
        return {"folder": folder.name, "samples": [], "error": "no annotations dir"}

    pred_files = sorted(predictions_dir.glob("*.json"))
    if not pred_files:
        logger.warning(f"No prediction files in {predictions_dir}")
        return {"folder": folder.name, "samples": [], "error": "no prediction files"}

    samples = []
    for pred_file in pred_files:
        ann_file = annotations_dir / pred_file.name
        if not ann_file.exists():
            logger.warning(f"No matching annotation for {pred_file.name}, skipping")
            continue

        # Load ground truth
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)
        gt_json = ann.get("json")
        if gt_json is None:
            logger.warning(f"No 'json' field in {ann_file.name}, skipping")
            continue
        # If gt_json is a string, parse it
        if isinstance(gt_json, str):
            try:
                gt_json = json.loads(gt_json)
            except json.JSONDecodeError:
                logger.warning(f"Cannot parse 'json' field in {ann_file.name}, skipping")
                continue

        # Load prediction
        with open(pred_file, "r", encoding="utf-8") as f:
            pred = json.load(f)

        # Try parsed_json first, then try parsing content
        pred_json = pred.get("parsed_json")
        if pred_json is None:
            pred_json = try_parse_json(pred.get("content", ""))

        result = evaluate_sample(gt_json, pred_json)
        result["file"] = pred_file.stem
        samples.append(result)

    return {"folder": folder.name, "samples": samples}


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(samples: list[dict]) -> dict:
    """Compute aggregate metrics from a list of sample results."""
    if not samples:
        return {
            "n_samples": 0,
            "json_parse_rate": 0.0,
            "avg_field_accuracy": 0.0,
            "avg_field_similarity": 0.0,
            "avg_exact_match_ratio": 0.0,
            "total_fields": 0,
            "total_exact_matches": 0,
        }

    n = len(samples)
    json_parsed = sum(1 for s in samples if s["json_parsed"])
    total_fields = sum(s["total_fields"] for s in samples)
    total_exact = sum(s["exact_match"] for s in samples)

    return {
        "n_samples": n,
        "json_parse_rate": json_parsed / n,
        "avg_field_accuracy": sum(s["field_accuracy"] for s in samples) / n,
        "avg_field_similarity": sum(s["field_similarity"] for s in samples) / n,
        "total_fields": total_fields,
        "total_exact_matches": total_exact,
        "overall_exact_match_rate": total_exact / total_fields if total_fields > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate bank report extraction predictions")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/tungn197/idp/data/r3_bank_reports/train_data_v1.2",
        help="Path to data directory containing category folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed JSON results (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-field details for each sample",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    # Discover category folders
    folders = sorted(
        d for d in data_path.iterdir()
        if d.is_dir() and (d / "predictions").is_dir()
    )
    if not folders:
        logger.error(f"No category folders with predictions/ found in {data_path}")
        sys.exit(1)

    logger.info(f"Found {len(folders)} category folders: {[f.name for f in folders]}")

    # Evaluate each folder
    all_results = {}
    all_samples = []

    for folder in folders:
        logger.info(f"Evaluating {folder.name} ...")
        folder_result = evaluate_folder(folder)
        samples = folder_result["samples"]
        metrics = aggregate_metrics(samples)
        all_results[folder.name] = {
            "metrics": metrics,
            "samples": samples,
        }
        all_samples.extend(samples)

        # Print category summary
        m = metrics
        print(f"\n{'=' * 60}")
        print(f"  {folder.name}")
        print(f"{'=' * 60}")
        print(f"  Samples evaluated:      {m['n_samples']}")
        print(f"  JSON parse rate:        {m['json_parse_rate']:.1%}")
        print(f"  Avg field accuracy:     {m['avg_field_accuracy']:.1%}")
        print(f"  Avg field similarity:   {m['avg_field_similarity']:.1%}")
        print(f"  Overall exact match:    {m['overall_exact_match_rate']:.1%} ({m['total_exact_matches']}/{m['total_fields']})")

        if args.verbose and samples:
            for s in samples:
                print(f"\n  --- {s['file']} ---")
                print(f"      JSON parsed: {s['json_parsed']}, Accuracy: {s['field_accuracy']:.1%}, Similarity: {s['field_similarity']:.1%}")
                for key, detail in s.get("field_details", {}).items():
                    marker = "✓" if detail["exact_match"] else "✗"
                    print(f"      {marker} [{detail['similarity']:.2f}] {key}")
                    if not detail["exact_match"]:
                        print(f"          GT:   {detail['gt']}")
                        print(f"          PRED: {detail['pred']}")

    # Overall summary
    overall = aggregate_metrics(all_samples)
    print(f"\n{'=' * 60}")
    print(f"  OVERALL")
    print(f"{'=' * 60}")
    print(f"  Total samples:          {overall['n_samples']}")
    print(f"  JSON parse rate:        {overall['json_parse_rate']:.1%}")
    print(f"  Avg field accuracy:     {overall['avg_field_accuracy']:.1%}")
    print(f"  Avg field similarity:   {overall['avg_field_similarity']:.1%}")
    print(f"  Overall exact match:    {overall['overall_exact_match_rate']:.1%} ({overall['total_exact_matches']}/{overall['total_fields']})")
    print(f"{'=' * 60}\n")

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip field_details from output to keep file size reasonable unless verbose
        save_data = {
            "overall": overall,
            "categories": {},
        }
        for cat_name, cat_data in all_results.items():
            cat_save = {"metrics": cat_data["metrics"]}
            cat_samples = []
            for s in cat_data["samples"]:
                sample_copy = {k: v for k, v in s.items() if k != "field_details"}
                if args.verbose:
                    sample_copy["field_details"] = s.get("field_details", {})
                cat_samples.append(sample_copy)
            cat_save["samples"] = cat_samples
            save_data["categories"][cat_name] = cat_save

        output_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2))
        logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
