"""DeepSeek OCR model via Ollama for image-to-markdown extraction."""

from __future__ import annotations

import base64
import logging

import requests

from .base import OCRModel

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = "Convert the image to markdown text. Extract all text and tables."


class OllamaOCR(OCRModel):
    """Image → Markdown using a vision model served by Ollama."""

    def __init__(
        self,
        model: str = "deepseek-ocr:3b-bf16",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        temperature: float = 0,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def extract_markdown(self, image_bytes: bytes, prompt: str | None = None) -> str:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = prompt or DEFAULT_OCR_PROMPT

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
