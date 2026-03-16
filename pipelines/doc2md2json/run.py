#!/usr/bin/env python
"""Launch the doc2md2json API servers.

Two services run on separate ports:
    OCR  (extract-markdown)  → port 7871
    LLM  (convert-to-json)   → port 7872

Usage:
    # Start both servers with defaults
    python -m pipelines.doc2md2json.run

    # Start only one
    python -m pipelines.doc2md2json.run --service ocr
    python -m pipelines.doc2md2json.run --service llm

    # Override ports
    python -m pipelines.doc2md2json.run --ocr-port 7871 --llm-port 7872
"""

from __future__ import annotations

import argparse
import os
import threading

import uvicorn


def _run_uvicorn(app: str, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="doc2md2json API servers")
    parser.add_argument("--ocr-model", default=None, help="OCR model name (Ollama)")
    parser.add_argument("--llm-model", default=None, help="LLM model name (Ollama)")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL")
    parser.add_argument("--host", default=None, help="API bind host")
    parser.add_argument("--ocr-port", type=int, default=None, help="OCR API port (default 7871)")
    parser.add_argument("--llm-port", type=int, default=None, help="LLM API port (default 7872)")
    parser.add_argument(
        "--service",
        choices=["ocr", "llm", "both"],
        default="both",
        help="Which service to start (default: both)",
    )
    args = parser.parse_args()

    # CLI args override env vars (which override config defaults)
    if args.ocr_model:
        os.environ["OCR_MODEL"] = args.ocr_model
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.ollama_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url

    # Import config after env vars are set
    from . import config

    host = args.host or config.API_HOST
    ocr_port = args.ocr_port or config.OCR_API_PORT
    llm_port = args.llm_port or config.LLM_API_PORT

    if args.service == "ocr":
        _run_uvicorn("pipelines.doc2md2json.api:ocr_app", host, ocr_port)
    elif args.service == "llm":
        _run_uvicorn("pipelines.doc2md2json.api:llm_app", host, llm_port)
    else:
        # Start both in parallel — OCR in a background thread, LLM in main thread
        ocr_thread = threading.Thread(
            target=_run_uvicorn,
            args=("pipelines.doc2md2json.api:ocr_app", host, ocr_port),
            daemon=True,
        )
        ocr_thread.start()
        _run_uvicorn("pipelines.doc2md2json.api:llm_app", host, llm_port)


if __name__ == "__main__":
    main()
