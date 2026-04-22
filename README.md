# Intelligent Document Processing (IDP)

AI-powered healthcare document processing pipeline built on **Ollama** (Qwen3.5 + Gemma4), providing document classification and structured information extraction from medical document images.

## Services

| Port | Container | Description |
|------|-----------|-------------|
| **7865** | `healthcare-ollama-pipeline-api` | Pipeline wrapper — classify + extraction in one service |
| **7871** | `classify-api` | Document classification only |
| **7872** | `extraction-api` | Information extraction only (requires template paths) |

All services are backed by an Ollama instance (default: `http://localhost:7860`).

## Quick Start

```bash
docker compose up -d
```

Rebuild after code changes:

```bash
docker compose up -d --build
```

Stop and remove containers:

```bash
docker compose down
```

## API Reference

### Port 7865 — Pipeline Wrapper

The recommended entry point. Internally calls classify, then extraction.

#### `POST /classify`

Classify a single document image or PDF (uses page 1 for PDF).

```bash
curl -X POST http://localhost:7865/classify \
  -F "file=@/path/to/document.pdf"
```

Response:

```json
{
  "status": "success",
  "message": "Document classified successfully",
  "document_type": "PHIẾU KHÁM BỆNH VÀO VIỆN",
  "template_info": { "json": "page-03.json", "template": "healthcare_types/page-03-template.json", "prompt_template": "healthcare_types/prompt_with_template.txt" }
}
```

#### `POST /extraction`

Classify then extract structured JSON from a single document image or PDF.

```bash
curl -X POST http://localhost:7865/extraction \
  -F "file=@/path/to/document.pdf"
```

Response:

```json
{
  "status": "success",
  "document_type": "PHIẾU KHÁM BỆNH VÀO VIỆN",
  "template_filename": "healthcare_types/page-03-template.json",
  "extracted_data": { ... },
  "output_path": "/app/output/api/document.json"
}
```

#### `POST /process`

Legacy full-pipeline endpoint (equivalent to `/extraction`).

```bash
curl -X POST http://localhost:7865/process \
  -F "file=@/path/to/document.pdf"
```

### Port 7871 — Classify API

```bash
curl -X POST http://localhost:7871/classify \
  -F "file=@/path/to/document.pdf"
```

### Port 7872 — Extraction API

Requires explicit template paths:

```bash
curl -X POST http://localhost:7872/extract \
  -F "file=@/path/to/document.pdf" \
  -F "prompt_template_path=healthcare_types/prompt_with_template.txt" \
  -F "json_template_path=healthcare_types/page-03-template.json"
```

### Optional form parameters (all endpoints)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ollama_url` | `$OLLAMA_BASE_URL` | Ollama server URL |
| `classifier_model` | `$CLASSIFIER_MODEL` (`qwen3.5:9b-bf16`) | Model used for classification |
| `extractor_model` | `$EXTRACTOR_MODEL` (`qwen3.5:9b-bf16`) | Model used for extraction |
| `output_subdir` | `api` | Sub-directory under `/app/output` for saved JSON |

## Supported Document Types

| Document | Template |
|----------|----------|
| PHIẾU KHÁM BỆNH VÀO VIỆN | page-03-template.json |
| PHIẾU CHỈ ĐỊNH CẬN LÂM SÀNG | page-04-template.json |
| PHIẾU TRẢ KẾT QUẢ HUYẾT HỌC | page-23-template.json |
| PHIẾU CHĂM SÓC | page-24-template.json |
| PHIẾU THEO DÕI CHỨC NĂNG SỐNG | page-36-template.json |
| TỜ ĐIỀU TRỊ | page-38-template.json |
| GIẤY RA VIỆN | page-60-template.json |
| BẢNG KÊ CHI PHÍ ĐIỀU TRỊ NGOẠI TRÚ | page-61-template.json |
| PHIẾU CÔNG KHAI DỊCH VỤ KCB NỘI TRÚ | page-62-template.json |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_OLLAMA_BASE_URL` | `http://host.docker.internal:7860` | Ollama URL inside container |
| `CLASSIFIER_MODEL` | `qwen3.5:9b-bf16` | Classification model |
| `EXTRACTOR_MODEL` | `qwen3.5:9b-bf16` | Extraction model |
| `TEMPLATES_DIR` | `/app/data/prompts` | Base directory for prompt/JSON templates |
| `OUTPUT_DIR` | `/app/output` | Output directory for extracted JSON files |
| `PDF_RENDER_DPI` | `200` | DPI used to render page 1 when uploading PDF |

Override at startup:

```bash
CLASSIFIER_MODEL=qwen3.5:9b-bf16 EXTRACTOR_MODEL=gemma4:e4b-it-bf16 docker compose up -d
```

## Requirements

- Docker + Docker Compose
- Ollama running on port `7860` with the required models pulled

## Troubleshooting 502 (Linux)

If API endpoints return `502 Bad Gateway`, the container usually cannot reach Ollama even though Ollama works on the host.

Common cause on Linux: Ollama is bound to `127.0.0.1:11434` only.

Check host binding:

```bash
ss -ltnp | grep 11434
```

If you see `127.0.0.1:11434`, restart Ollama bound to all interfaces:

```bash
pkill ollama || true
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then restart the API stack:

```bash
docker compose up -d --build
```
