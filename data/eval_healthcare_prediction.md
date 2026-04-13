# Evaluation Method: Healthcare Form Extraction

## Overview

The evaluation compares a VLM-predicted JSON against a human-annotated ground truth (GT) JSON for structured healthcare document extraction. Both files follow the same schema defined by the form template (e.g., `phieu_kham_benh_vao_vien_template.json`).

The script: `scripts/eval_healthcare_prediction.py`

---

## Pipeline

```
Document image
      │
      ▼
 VLM (Ollama)  ◄── prompt + JSON template
      │
      ▼
Predicted JSON
      │
      ▼
  Evaluator  ◄── Ground Truth JSON
      │
      ▼
 Metrics Report
```

---

## Step 1 — JSON Flattening

Both the GT and prediction JSONs are deeply nested (sections → subsections → leaf fields). Before comparison, both are **flattened** into a dictionary of `dot-path key → value` pairs.

**Example:**
```json
{
  "I. HÀNH CHÍNH": {
    "8. Địa chỉ": {
      "Xã ( phường)": "Xã Trung Thành"
    }
  }
}
```
Becomes:
```
"I. HÀNH CHÍNH.8. Địa chỉ.Xã ( phường)" → "Xã Trung Thành"
```

List elements use bracket notation: `key[0]`, `key[1]`, etc.

This ensures every leaf value — regardless of nesting depth — is evaluated independently and unambiguously.

---

## Step 2 — Text Normalization

Before any comparison, both GT and predicted values are normalized:

1. **Unicode NFC normalization** — unify equivalent Vietnamese character representations
2. **Strip** leading/trailing whitespace
3. **Collapse** multiple spaces into one
4. **Lowercase** the entire string

`null` values are treated as empty strings `""` for comparison purposes.

This ensures minor formatting differences (extra spaces, case) do not penalize the prediction unfairly.

---

## Step 3 — Per-Field Comparison

Each GT leaf field is compared to the corresponding prediction field. Four statuses are assigned:

| Status | Condition | Icon |
|--------|-----------|------|
| `exact` | Normalized strings are identical | ✓ |
| `partial` | Not identical but similarity ≥ 0.80 | ~ |
| `wrong` | Similarity < 0.80 | ✗ |
| `missing` | Key exists in GT but not in prediction | ? |

Fields present in the prediction but absent from the GT are flagged as `extra` (schema leak).

### Similarity Score

Similarity is computed using **normalized Levenshtein distance**:

$$\text{similarity}(s_1, s_2) = 1 - \frac{\text{edit\_distance}(s_1, s_2)}{\max(|s_1|, |s_2|)}$$

This yields a score in $[0, 1]$ where $1.0$ = identical and $0.0$ = completely different.

---

## Step 4 — Summary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `field_accuracy` | exact\_matches / total\_gt\_fields | Fraction of fields extracted perfectly |
| `mean_similarity` | Σ sim(field) / total\_gt\_fields | Average character-level similarity across all GT fields |
| `null_field_accuracy` | exact on null fields / total null fields | How well the model identifies empty/absent values |
| `filled_field_accuracy` | exact on non-null fields / total non-null fields | How well the model reads actual content |

**`field_accuracy` is the primary metric.** `mean_similarity` is the secondary metric for partial credit.

Splitting accuracy into `null_field_accuracy` and `filled_field_accuracy` helps diagnose whether errors come from hallucinating values on blank fields or from misreading filled content.

---

## Output

### Console report (always printed)

```
======================================================================
  GT  : data/healthcare/hoso1_annotations/page-03.json
  PRED: data/healthcare/hoso1_annotations/prediction_page-03.json
======================================================================
  Total GT fields     : 55
  Exact matches  ✓   : 49
  Partial matches ~  : 1
  Wrong fields   ✗   : 5
  Missing fields ?   : 0
  Extra fields (pred): 0
  Field accuracy     : 89.1%
  Mean similarity    : 96.0%
  Null-field acc.    : 100.0%  (10 null fields)
  Filled-field acc.  : 86.7%  (45 filled fields)
======================================================================

  Non-exact fields:

  [WRONG] ✗ I. HÀNH CHÍNH.11. BHYT có giá trị đến
    GT  : 'Ngày 31 tháng 12 năm 2023'
    PRED: '31/12/2023'
    SIM : 32.0%
  ...
```

### JSON file (optional, via `--output`)

```json
{
  "summary": { ... },
  "field_details": {
    "I. HÀNH CHÍNH.1. Họ và tên": {
      "gt": "NGUYỄN THỊ ĐENG",
      "pred": "NGUYỄN THỊ ĐENG",
      "similarity": 1.0,
      "status": "exact"
    },
    ...
  },
  "extra_fields": {}
}
```

---

## Usage

```bash
# Basic comparison (console output only)
python scripts/eval_healthcare_prediction.py \
  --gt   data/healthcare/hoso1_annotations/page-03.json \
  --pred data/healthcare/hoso1_annotations/prediction_page-03.json

# Save detailed results to file
python scripts/eval_healthcare_prediction.py \
  --gt   data/healthcare/hoso1_annotations/page-03.json \
  --pred data/healthcare/hoso1_annotations/prediction_page-03.json \
  --output data/outputs/metrics/eval_page-03.json
```

---

## Limitations

- **Schema dependency:** Evaluation is keyed to the GT schema. If the prediction adds or renames keys, those differences are caught as `extra` or `missing` but not penalized in `field_accuracy`.
- **Semantic equivalence not captured:** `"31/12/2023"` and `"Ngày 31 tháng 12 năm 2023"` represent the same date but score low similarity because they differ at the character level. This is a known limitation when GT and template use different date formats.
- **No ordering-aware list comparison:** List fields are compared by index. Reordered lists with the same content will score as wrong.
- **Single-document scope:** The script evaluates one document at a time. Batch evaluation across a dataset requires looping over multiple file pairs.
