"""Port through which the text extraction reads a PDF: the text layer of each page, and OCR.

The use case depends on :class:`PdfReader` only, so the PDF library stays in its adapter
(:mod:`thematic_redundancy.extraction.pymupdf_reader`) and the tests can read synthetic
documents through a fake.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class PdfReadError(Exception):
    """A PDF cannot be opened, or a page of it cannot be read; other PDFs are unaffected."""


class OcrError(Exception):
    """OCR of one page failed; the other pages of the PDF are unaffected."""


@dataclass(frozen=True)
class TextLayer:
    """The text layer of one page."""

    text: str
    """Text in reading order: its blocks from top to bottom, separated by a blank line."""
    chars: int
    """Characters of the text in content-stream order, stripped of surrounding whitespace:
    the measure that tells a low-text page."""


class PdfPages(Protocol):
    """An open PDF, page by page; pages are counted from 0."""

    @property
    def page_count(self) -> int: ...

    def text_layer(self, index: int) -> TextLayer:
        """Return the text layer of page ``index``.

        Raises:
            PdfReadError: if the page cannot be read.
        """
        ...

    def ocr_text(self, index: int) -> str:
        """Render page ``index``, read it with OCR, and return the text in reading order.

        Raises:
            OcrError: if OCR fails on this page.
        """
        ...

    def warning_count(self) -> int:
        """Return how many warnings the PDF library reported since the PDF was opened."""
        ...


class PdfReader(Protocol):
    """Opens PDFs for reading."""

    def open(self, path: Path) -> AbstractContextManager[PdfPages]:
        """Open the PDF at ``path`` for the duration of a ``with`` block.

        Raises:
            PdfReadError: if the file is missing, damaged beyond repair, or encrypted.
        """
        ...
