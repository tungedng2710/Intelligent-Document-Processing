"""Marker-based OCR model for document-to-markdown extraction.

Uses the marker-pdf library to convert PDFs and images to markdown.
Models are loaded once and reused across requests.
"""

from __future__ import annotations

import logging
import os
import tempfile

from .base import OCRModel

logger = logging.getLogger(__name__)

# Content-type → file suffix mapping
_SUFFIX_MAP = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
}


class MarkerOCR(OCRModel):
    """Document → Markdown using marker-pdf (surya OCR + layout models)."""

    def __init__(self, force_ocr: bool = False):
        self._force_ocr = force_ocr
        self._converter = None
        self._artifact_dict = None

    def _ensure_converter(self):
        """Lazy-load marker models and converter on first use."""
        if self._converter is not None:
            return

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        logger.info("Loading marker models (this may take a moment)...")
        self._artifact_dict = create_model_dict()

        config = {}
        if self._force_ocr:
            config["force_ocr"] = True

        self._converter = PdfConverter(
            config=config,
            artifact_dict=self._artifact_dict,
        )
        logger.info("Marker models loaded successfully.")

    @property
    def model_name(self) -> str:
        return "marker-pdf"

    def extract_markdown(self, image_bytes: bytes, prompt: str | None = None) -> str:
        """Convert a single image to markdown via marker."""
        return self.convert_file(image_bytes, content_type="image/png")

    def convert_file(self, file_bytes: bytes, content_type: str = "application/pdf") -> str:
        """Convert file bytes (PDF or image) to markdown using marker.

        Marker handles multi-page PDFs natively, so the entire document
        is processed in one call.
        """
        self._ensure_converter()

        suffix = _SUFFIX_MAP.get(content_type, ".pdf")

        # Marker requires a file path, so write to a temp file
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, file_bytes)
            os.close(fd)

            rendered = self._converter(tmp_path)
            return rendered.markdown
        finally:
            os.unlink(tmp_path)
