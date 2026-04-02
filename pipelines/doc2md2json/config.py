"""Centralized configuration for the doc2md2json pipeline."""

from __future__ import annotations

import os


# --- Ollama server ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:7860")

# --- OCR engine (Step 1: document → markdown) ---
# Set OCR_ENGINE to "marker" (default) or "ollama"
OCR_ENGINE = os.getenv("OCR_ENGINE", "marker")
OCR_FORCE_OCR = os.getenv("OCR_FORCE_OCR", "false").lower() == "true"

# Ollama OCR fallback settings (only used when OCR_ENGINE="ollama")
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
WRAPPER_API_PORT = int(os.getenv("WRAPPER_API_PORT", "7873"))

# URLs used by the wrapper to call the two upstream services
OCR_API_URL = os.getenv("OCR_API_URL", f"http://0.0.0.0:{OCR_API_PORT}")
LLM_API_URL = os.getenv("LLM_API_URL", f"http://0.0.0.0:{LLM_API_PORT}")
