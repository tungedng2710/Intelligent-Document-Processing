# Intelligent Document Processing (IDP)

AI-powered document processing services built on **Qwen3-VL**, providing:

- **Document Classification** — classifies uploaded documents by type
- **Bank Report Extraction** — extracts structured data from bank report PDFs

## Quick Start

```bash
bash launch_apis.sh
```

This single command will:
1. Install [ollama](https://ollama.com) if not already installed
2. Start the ollama server and pull the required model (`qwen3-vl:8b-instruct-bf16`)
3. Launch both API services in the background

| Service | URL |
|---|---|
| Document Classification | http://0.0.0.0:7871 |
| Bank Report Extraction | http://0.0.0.0:7872 |

Logs are written to `logs/classify.log` and `logs/extract.log`.

## Other Commands

```bash
bash launch_apis.sh stop      # stop both services
bash launch_apis.sh restart   # restart both services
```

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`
