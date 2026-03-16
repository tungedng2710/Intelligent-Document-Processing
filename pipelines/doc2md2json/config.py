"""Centralized configuration for the doc2md2json pipeline."""

from __future__ import annotations

import os


# --- Ollama server ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:7860")

# --- OCR model (Step 1: image → markdown) ---
OCR_MODEL = os.getenv("OCR_MODEL", "deepseek-ocr:3b-bf16")
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "300"))
OCR_TEMPERATURE = float(os.getenv("OCR_TEMPERATURE", "0"))

# --- LLM model (Step 2: markdown → JSON) ---
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b-bf16")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_THINK = os.getenv("LLM_THINK", "false").lower() == "true"

# --- API servers ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
OCR_API_PORT = int(os.getenv("OCR_API_PORT", "7871"))
LLM_API_PORT = int(os.getenv("LLM_API_PORT", "7872"))
