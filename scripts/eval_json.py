"""
Compare a healthcare form prediction JSON against a ground-truth JSON.

Metrics per field (leaf node):
  - exact_match : 1 if values are identical after normalization, else 0
  - similarity  : normalized Levenshtein similarity [0, 1]
  - status      : "exact" | "partial" | "wrong" | "missing" | "extra"

Summary metrics:
  - field_accuracy   : fraction of GT fields with exact match
  - field_similarity : mean similarity across all GT fields
  - null_accuracy    : accuracy when GT value is null/empty
  - filled_accuracy  : accuracy when GT value is non-empty

Usage:
    # Single file mode:
    python scripts/eval_json.py \
        --gt   data/healthcare/hoso1_annotations/page-03.json \
        --pred data/healthcare/hoso1_annotations/prediction_page-03.json

    # Folder mode (evaluate all matching JSON files and compute averages):
    python scripts/eval_json.py \
        --gt-dir   data/healthcare/hoso1_annotations/ \
        --pred-dir data/healthcare/hoso1_predictions/

    # Save detailed results:
    python scripts/eval_json.py \
        --gt-dir   data/healthcare/hoso1_annotations/ \
        --pred-dir data/healthcare/hoso1_predictions/ \
        --output data/outputs/metrics/eval_folder.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalization & similarity
# ---------------------------------------------------------------------------

def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def similarity(a, b) -> float:
    s1, s2 = normalize_text(a), normalize_text(b)
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


# ---------------------------------------------------------------------------
# JSON flattening
# ---------------------------------------------------------------------------

def flatten(obj, prefix: str = "") -> dict[str, object]:
    """Recursively flatten nested dict/list to dot-path keys."""
    items: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                items.update(flatten(v, path))
            else:
                items[path] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                items.update(flatten(v, path))
            else:
                items[path] = v
    else:
        items[prefix] = obj
    return items


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(gt: dict, pred: dict) -> dict:
    gt_flat = flatten(gt)
    pred_flat = flatten(pred)

    gt_keys = set(gt_flat.keys())
    pred_keys = set(pred_flat.keys())

    field_details: dict[str, dict] = {}
    total_sim = 0.0
    exact_count = 0
    null_total = null_correct = 0
    filled_total = filled_exact = 0

    for key in sorted(gt_keys):
        gt_val = gt_flat[key]
        is_null_gt = gt_val is None or normalize_text(gt_val) == ""

        if key not in pred_flat:
            status = "missing"
            sim = 0.0
            pred_val = None
        else:
            pred_val = pred_flat[key]
            sim = similarity(gt_val, pred_val)
            exact = normalize_text(gt_val) == normalize_text(pred_val)
            if exact:
                status = "exact"
                exact_count += 1
            elif sim >= 0.8:
                status = "partial"
            else:
                status = "wrong"

        field_details[key] = {
            "gt": gt_val,
            "pred": pred_val if key in pred_flat else "<missing>",
            "similarity": round(sim, 4),
            "status": status,
        }

        total_sim += sim

        if is_null_gt:
            null_total += 1
            if status == "exact":
                null_correct += 1
        else:
            filled_total += 1
            if status == "exact":
                filled_exact += 1

    extra_keys = pred_keys - gt_keys
    extra_fields = {k: pred_flat[k] for k in sorted(extra_keys)}

    n = len(gt_keys)
    summary = {
        "total_gt_fields": n,
        "exact_matches": exact_count,
        "partial_matches": sum(1 for d in field_details.values() if d["status"] == "partial"),
        "wrong_fields": sum(1 for d in field_details.values() if d["status"] == "wrong"),
        "missing_fields": sum(1 for d in field_details.values() if d["status"] == "missing"),
        "extra_fields_in_pred": len(extra_keys),
        "field_accuracy": round(exact_count / n, 4) if n else 0.0,
        "mean_similarity": round(total_sim / n, 4) if n else 0.0,
        "null_field_accuracy": round(null_correct / null_total, 4) if null_total else None,
        "filled_field_accuracy": round(filled_exact / filled_total, 4) if filled_total else None,
    }

    return {
        "summary": summary,
        "field_details": field_details,
        "extra_fields": extra_fields,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

STATUS_ICON = {"exact": "✓", "partial": "~", "wrong": "✗", "missing": "?"}
STATUS_LABEL = {"exact": "EXACT", "partial": "PARTL", "wrong": "WRONG", "missing": "MISS "}

def print_report(result: dict, gt_path: str, pred_path: str) -> None:
    s = result["summary"]
    print("\n" + "=" * 70)
    print(f"  GT  : {gt_path}")
    print(f"  PRED: {pred_path}")
    print("=" * 70)
    print(f"  Total GT fields     : {s['total_gt_fields']}")
    print(f"  Exact matches  ✓   : {s['exact_matches']}")
    print(f"  Partial matches ~  : {s['partial_matches']}")
    print(f"  Wrong fields   ✗   : {s['wrong_fields']}")
    print(f"  Missing fields ?   : {s['missing_fields']}")
    print(f"  Extra fields (pred): {s['extra_fields_in_pred']}")
    print(f"  Field accuracy     : {s['field_accuracy']:.1%}")
    print(f"  Mean similarity    : {s['mean_similarity']:.1%}")
    if s["null_field_accuracy"] is not None:
        print(f"  Null-field acc.    : {s['null_field_accuracy']:.1%}  ({sum(1 for d in result['field_details'].values() if d['gt'] is None or d['gt'] == '')} null fields)")
    if s["filled_field_accuracy"] is not None:
        print(f"  Filled-field acc.  : {s['filled_field_accuracy']:.1%}  ({sum(1 for d in result['field_details'].values() if d['gt'] is not None and d['gt'] != '')} filled fields)")
    print("=" * 70)

    # Show non-exact fields
    non_exact = {k: v for k, v in result["field_details"].items() if v["status"] != "exact"}
    if non_exact:
        print("\n  Non-exact fields:")
        for key, info in non_exact.items():
            icon = STATUS_ICON.get(info["status"], "?")
            label = STATUS_LABEL.get(info["status"], "?")
            print(f"\n  [{label}] {icon} {key}")
            print(f"    GT  : {repr(info['gt'])}")
            print(f"    PRED: {repr(info['pred'])}")
            print(f"    SIM : {info['similarity']:.1%}")
    else:
        print("\n  All fields matched exactly!")

    if result["extra_fields"]:
        print("\n  Extra fields in prediction (not in GT):")
        for k, v in result["extra_fields"].items():
            print(f"    + {k}: {repr(v)}")

    print()


# ---------------------------------------------------------------------------
# Folder evaluation
# ---------------------------------------------------------------------------

def evaluate_folders(gt_dir: Path, pred_dir: Path) -> dict:
    """Evaluate all JSON files in gt_dir that have a matching file in pred_dir."""
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No JSON files found in {gt_dir}")

    per_file_results: dict[str, dict] = {}
    matched = 0
    skipped: list[str] = []

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        if not pred_file.exists():
            skipped.append(gt_file.name)
            continue

        gt   = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        result = evaluate(gt, pred)
        per_file_results[gt_file.name] = result
        matched += 1

    if matched == 0:
        raise FileNotFoundError(
            f"No matching prediction files found in {pred_dir} "
            f"for GT files in {gt_dir}"
        )

    # Compute averages across all file summaries
    sum_keys = [
        "total_gt_fields", "exact_matches", "partial_matches",
        "wrong_fields", "missing_fields", "extra_fields_in_pred",
    ]
    avg_keys = [
        "field_accuracy", "mean_similarity",
        "null_field_accuracy", "filled_field_accuracy",
    ]

    totals = {k: 0 for k in sum_keys}
    avg_accum = {k: [] for k in avg_keys}

    for r in per_file_results.values():
        s = r["summary"]
        for k in sum_keys:
            totals[k] += s[k]
        for k in avg_keys:
            if s[k] is not None:
                avg_accum[k].append(s[k])

    avg_summary = {}
    for k in sum_keys:
        avg_summary[k] = totals[k]
    for k in avg_keys:
        vals = avg_accum[k]
        avg_summary[k] = round(sum(vals) / len(vals), 4) if vals else None

    avg_summary["num_files_evaluated"] = matched
    avg_summary["num_files_skipped"] = len(skipped)

    return {
        "average_summary": avg_summary,
        "per_file": per_file_results,
        "skipped_files": skipped,
    }


def print_folder_report(folder_result: dict, gt_dir: str, pred_dir: str) -> None:
    s = folder_result["average_summary"]
    per_file = folder_result["per_file"]
    skipped = folder_result["skipped_files"]

    print("\n" + "=" * 70)
    print(f"  GT DIR  : {gt_dir}")
    print(f"  PRED DIR: {pred_dir}")
    print("=" * 70)
    print(f"  Files evaluated     : {s['num_files_evaluated']}")
    print(f"  Files skipped       : {s['num_files_skipped']}")
    print("-" * 70)

    # Per-file summary table
    print(f"  {'File':<35} {'Accuracy':>10} {'Similarity':>12} {'Exact':>7} {'Total':>7}")
    print("  " + "-" * 73)
    for fname, r in per_file.items():
        fs = r["summary"]
        print(
            f"  {fname:<35} {fs['field_accuracy']:>9.1%} "
            f"{fs['mean_similarity']:>11.1%} "
            f"{fs['exact_matches']:>7} {fs['total_gt_fields']:>7}"
        )

    print("  " + "-" * 73)
    print(f"  {'AVERAGE':<35} {s['field_accuracy']:>9.1%} {s['mean_similarity']:>11.1%} "
          f"{s['exact_matches']:>7} {s['total_gt_fields']:>7}")
    print("=" * 70)

    print(f"\n  Overall field accuracy     : {s['field_accuracy']:.1%}")
    print(f"  Overall mean similarity    : {s['mean_similarity']:.1%}")
    if s["null_field_accuracy"] is not None:
        print(f"  Overall null-field acc.    : {s['null_field_accuracy']:.1%}")
    if s["filled_field_accuracy"] is not None:
        print(f"  Overall filled-field acc.  : {s['filled_field_accuracy']:.1%}")

    if skipped:
        print(f"\n  Skipped (no matching pred): {', '.join(skipped)}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate healthcare JSON prediction vs ground truth.")
    # Single file mode
    parser.add_argument("--gt",   default=None, help="Path to ground truth JSON file")
    parser.add_argument("--pred", default=None, help="Path to prediction JSON file")
    # Folder mode
    parser.add_argument("--gt-dir",   default=None, help="Path to ground truth folder of JSON files")
    parser.add_argument("--pred-dir", default=None, help="Path to prediction folder of JSON files")
    parser.add_argument("--output", default=None, help="Optional path to save detailed results JSON")
    parser.add_argument("--verbose", type=int, default=2, choices=[1, 2],
                        help="Folder mode verbosity: 1=summary only, 2=full per-file details (default: 2)")
    args = parser.parse_args()

    folder_mode = args.gt_dir is not None or args.pred_dir is not None
    file_mode   = args.gt is not None or args.pred is not None

    if folder_mode and file_mode:
        parser.error("Use either --gt/--pred (single file) or --gt-dir/--pred-dir (folder), not both.")
    if folder_mode:
        if not args.gt_dir or not args.pred_dir:
            parser.error("Both --gt-dir and --pred-dir are required for folder mode.")
        gt_dir   = Path(args.gt_dir)
        pred_dir = Path(args.pred_dir)
        folder_result = evaluate_folders(gt_dir, pred_dir)
        print_folder_report(folder_result, str(gt_dir), str(pred_dir))

        # Print per-file details only when verbose=2
        if args.verbose >= 2:
            for fname, result in folder_result["per_file"].items():
                print_report(result, str(gt_dir / fname), str(pred_dir / fname))

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(folder_result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Detailed results saved to: {out}")
    else:
        if not args.gt or not args.pred:
            parser.error("Both --gt and --pred are required for single file mode.")
        gt_path   = Path(args.gt)
        pred_path = Path(args.pred)

        gt   = json.loads(gt_path.read_text(encoding="utf-8"))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))

        result = evaluate(gt, pred)
        print_report(result, str(gt_path), str(pred_path))

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Detailed results saved to: {out}")


if __name__ == "__main__":
    main()
