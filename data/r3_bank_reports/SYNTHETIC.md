# Synthetic Bank Document Dataset — Data Guideline

Source: `/home/public/ocr/idp/data-synthesizer/bank_docs_synthesis/dataset_degraded`

---

## 1. Overview

This dataset contains synthetically generated and degraded Vietnamese bank form images, paired with structured JSON annotations. It is intended for training and evaluating document understanding models (detection, layout analysis, information extraction).

| Category | Vietnamese Name | Description |
|---|---|---|
| `giay_phong_toa_tam_khoa_tai_khoan` | Giấy phong tỏa tài khoản | Account freeze form |
| `giay_rut_tien` | Giấy rút tiền | Cash withdrawal form |
| `phieu_hach_toan` | Phiếu hạch toán | Accounting / journal slip |

Each category contains **10,000 image–annotation pairs**.

---

## 2. Directory Structure

```
dataset_degraded/
├── {category}/
│   ├── annotations/     # JSON annotation files
│   └── images/          # PNG image files
```

### File Naming Convention

```
{category}_{NNNN}_deg{D}.{ext}
```

- `{NNNN}`: zero-padded 4-digit sample index (e.g. `0001`, `0002`, …)
- `{D}`: degradation variant — `1` or `2` (each base sample has two independently degraded renders)
- `{ext}`: `json` for annotations, `png` for images

---

## 3. Annotation Format

Each annotation is a JSON file with the following top-level keys:

```
{
  "category":         string,
  "bboxes":           list[BboxEntry],
  "cell_bboxes":      list[BboxEntry],
  "layout":           list[BboxEntry],
  "json":             dict,
  "augmentation":     dict,
  "augmented_assets": list[AssetEntry]
}
```

### 3.1 `BboxEntry` Schema

Used by `bboxes`, `cell_bboxes`, and `layout`:

```json
{
  "label": "<string>",
  "bbox": {
    "x":      <float>,   // left edge, pixels
    "y":      <float>,   // top edge, pixels
    "width":  <float>,   // box width, pixels
    "height": <float>    // box height, pixels
  }
}
```

Coordinate system: **top-left origin**, absolute pixel floats, axis-aligned.

---

### 3.2 `bboxes` — Fine-Grained Element Bounding Boxes

Individual visual elements on the document. Labels are **per document category** (see Section 4).

Typical element types:
- `*_label_cell` — printed field label text
- `*_value_cell` — field value text area
- `*_colon` — the colon separator between label and value
- `divider_line_*` — horizontal rule separating sections
- `title_cell` — document title
- `bank_name_cell`, `branch_cell` — header identifiers
- `teller`, `supervisor`, `approver`, `inputter` — signer role zones
- `*_signatures` — signature asset region
- `*_title_stamp` — title stamp asset region
- `logo`, `qr`, `denom_table` — graphical elements

---

### 3.3 `cell_bboxes` — Row-Level Bounding Boxes

Coarser bounding boxes that span an entire form row (label + separator + value as one box). Same label set as `bboxes` but without colon entries. Useful for row-level detection tasks.

---

### 3.4 `layout` — Semantic Layout Regions

High-level document zones. Common labels across all categories:

| Label | Description |
|---|---|
| `header_left` | Top-left block (bank name, branch) |
| `header_right` | Top-right block (transaction ref, date) |
| `title` | Document title area |
| `subtitle` | Secondary title (phieu_hach_toan only) |
| `body` / `body_left` / `body_right` | Main form fields |
| `footer` | Signature and stamp area |

---

### 3.5 `json` — Extraction Ground Truth

Key–value pairs representing the structured information extracted from the document. Keys are Vietnamese or bilingual field names; values are the text strings as rendered on the form (including newlines for multi-line cells).

Example (`giay_phong_toa_tam_khoa_tai_khoan`):
```json
{
  "Title": "GIẤY PHONG TỎA TÀI KHOẢN",
  "Chi nhánh": "NH TMCP QD\nVN0000482 CN THANH HOA",
  "Số GD": "GBXR762633705951",
  "Transaction date": "24/09/2020 16:33:10",
  "Tên tài khoản": "ĐINH DUY CƯỜNG",
  "Số TK": "654450915733",
  ...
}
```

This field is the **primary ground truth** for information extraction tasks.

---

### 3.6 `augmentation` — Applied Augmentation Flags

Records which synthetic visual elements were added to the image:

```json
{
  "signatures": {
    "teller":     { "exists": bool },
    "supervisor": { "exists": bool }
  },
  "title_stamps": {
    "teller":     { "exists": bool, "title": str, "name": str },
    "supervisor": { "exists": bool, "title": str, "name": str }
  },
  "rect_stamps": {
    "<field_name>": { "exists": bool }
  },
  "circle_stamps": {
    "teller":     { "exists": bool },
    "supervisor": { "exists": bool }
  }
}
```

Use this field to filter or stratify samples by augmentation type.

---

### 3.7 `augmented_assets` — Pasted Asset Details

List of asset images composited onto the document, with their exact placement bounding boxes:

```json
{
  "label":      "<element label matching bboxes>",
  "category":   "signatures" | "title_stamps" | "rect_stamps" | "circle_stamps",
  "location":   "teller" | "supervisor" | ...,
  "asset_file": "<filename of the source asset image>",
  "bbox":       { "x": float, "y": float, "width": float, "height": float }
}
```

---

## 4. Category-Specific Label Vocabularies

### `giay_phong_toa_tam_khoa_tai_khoan`

**`bboxes` labels:**
`acc_name_colon`, `acc_name_label_cell`, `acc_name_value_cell`,
`acc_num_colon`, `acc_num_label_cell`, `acc_num_value_cell`,
`amount_words_colon`, `amount_words_label_cell`, `amount_words_value_cell`,
`bank_name_cell`, `branch_cell`,
`content_colon`, `content_label_cell`, `content_value_cell`,
`customer_code_colon`, `customer_code_label_cell`, `customer_code_value_cell`,
`divider_line_1`, `divider_line_2`,
`end_date_colon`, `end_date_label_cell`, `end_date_value_cell`,
`freeze_amount_colon`, `freeze_amount_label_cell`, `freeze_amount_value_cell`,
`freeze_date_colon`, `freeze_date_label_cell`, `freeze_date_value_cell`,
`supervisor`, `supervisor_signatures`, `supervisor_title_stamp`,
`teller`, `teller_signatures`, `teller_title_stamp`,
`title_cell`, `tx_date_cell`, `tx_ref_cell`

**`layout` regions:** `header_left`, `header_right`, `title`, `body`, `footer`

---

### `giay_rut_tien`

**`bboxes` labels:**
`acc_name`, `acc_num`, `addr`, `amount_hdr`, `amount_val`,
`approver`, `approver_signatures`,
`bank`, `bank_only`, `charges_hdr`, `currency_cbs`, `currency_label`,
`date_val`, `debit_credit`, `debit_label`, `denom_table`, `details`, `disclaimer`,
`divider_line`, `fee_amt`, `fee_cbs`,
`id_date`, `id_num`, `id_place`,
`logo`, `phone`, `qr`,
`recv_label`, `recv_name`,
`sig_chief`, `sig_chief_signatures`, `sig_holder`, `sig_receiver`, `sig_receiver_signatures`, `sig_receiver_title_stamp`,
`teller`, `teller_signatures`, `teller_title_stamp`,
`title`, `tx_code`, `tx_code_rect_stamps`

**`layout` regions:** `header_left`, `header_right`, `title`, `body_left`, `body_right`, `footer`

---

### `phieu_hach_toan`

**`bboxes` labels:**
`amount_cell`, `amount_words_cell`,
`approver`, `approver_signatures`, `approver_title_stamp`,
`cash_header_cell`, `company_cell`, `content_cell`, `date_cell`, `date_empty_cell`,
`denom_table`, `detail_header_cell`,
`inputter`, `inputter_signatures`,
`journal_cell`, `logo`, `qr`,
`receiver_code_cell`, `receiver_name_cell`,
`sender_code_cell`, `sender_name_cell`,
`title_cell`, `tx_code_cell`, `tx_type_cell`

**`layout` regions:** `header_left`, `header_right`, `title`, `subtitle`, `body_left`, `body_right`, `footer`

---

## 5. Usage Notes

- **Detection / localization:** Use `bboxes` for fine-grained element detection; use `layout` for region-level detection.
- **Information extraction:** Use the `json` field as ground truth. The `bboxes` entries can be used to align extracted text to spatial locations.
- **Signature / stamp tasks:** Use `augmentation` flags to filter samples with or without signatures/stamps; use `augmented_assets` for their precise bounding boxes.
- **Degradation variants:** The `deg1` and `deg2` variants of the same sample share the same `json` ground truth but differ in visual degradation (blur, noise, rotation, etc.).
- **Image dimensions:** Images are PNG; verify dimensions per sample as they may vary by template and augmentation.
