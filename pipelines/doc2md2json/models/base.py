"""Abstract base classes for OCR and LLM models.

Subclass these to swap in different backends (Ollama, vLLM, API, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OCRModel(ABC):
    """Takes an image (raw bytes) and returns markdown text."""

    @abstractmethod
    def extract_markdown(self, image_bytes: bytes, prompt: str | None = None) -> str:
        """Return markdown text extracted from the image."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a human-readable model identifier."""


class LLMModel(ABC):
    """Takes text input and returns text output."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the raw response text."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a human-readable model identifier."""
