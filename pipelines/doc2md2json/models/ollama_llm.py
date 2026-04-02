"""Qwen3.5 LLM via Ollama for markdown-to-JSON extraction."""

from __future__ import annotations

import logging

import requests

from .base import LLMModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a JSON extraction engine. You ONLY output valid JSON. No markdown, no explanation, no extra text.
Extract all key-value pairs from the given document text. Rules:
- Keys: use exact field labels from the document (keep Vietnamese as-is).
- Values: extract exactly as they appear, do not correct or translate.
- Tables: represent as a JSON array of objects with column headers as keys.
- Empty fields: use null.
- Ignore signatures, stamps, page numbers, decorative elements.
Output ONLY a JSON object. Nothing else."""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract all information from this document into a flat JSON object with key-value pairs.

Document:
{content}"""


class OllamaLLM(LLMModel):
    """Text → Text using an LLM served by Ollama."""

    def __init__(
        self,
        model: str = "qwen3.5:2b-bf16",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        temperature: float = 0,
        think: bool = False,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._think = think

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": self._think,
            "options": {"temperature": self._temperature},
            "format": "json",
        }

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def extract_json(self, markdown_content: str, prompt_template: str | None = None) -> str:
        """Convenience method: fill the extraction prompt and call generate."""
        template = prompt_template or EXTRACTION_PROMPT_TEMPLATE
        prompt = template.format(content=markdown_content)
        return self.generate(prompt, system=SYSTEM_PROMPT)
