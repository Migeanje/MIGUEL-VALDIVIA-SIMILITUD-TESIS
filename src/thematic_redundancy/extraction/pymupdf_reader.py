"""PyMuPDF adapter of the PDF-reader port, with OCR by PyMuPDF's built-in Tesseract (D07).

- **Text layer.** Each page is read once into a MuPDF text page. Its text in content-stream
  order gives the low-text measure, the same as ``page.get_text()``; its blocks, sorted top
  to bottom and then left to right, give the text in reading order, one blank line between
  blocks. Sorting puts a running header first and a running footer last even when the
  content stream draws them elsewhere.
- **OCR.** The whole page is rendered at ``ocr.dpi`` and read with ``ocr.languages``
  (``full=True``), using the language files in ``paths.tessdata_dir``.
- **Failures.** Anything that PyMuPDF raises becomes :class:`PdfReadError` or
  :class:`OcrError`, so one damaged PDF or page never stops a run. MuPDF's own messages
  are kept off the console and counted instead (:meth:`warning_count`).
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pymupdf

from thematic_redundancy.extraction.ocr_assets import FETCH_COMMAND, language_codes
from thematic_redundancy.extraction.pdf_reader import OcrError, PdfReadError, TextLayer
from thematic_redundancy.shared.config import AppConfig

OcrEngine = Callable[[pymupdf.Page], str]
"""Reads one rendered page with OCR and returns its text in reading order."""

_TEXT_BLOCK = 0
"""Type of a text block in ``page.get_text("blocks")``; image blocks have type 1."""


class PyMuPdfReader:
    """:class:`~thematic_redundancy.extraction.pdf_reader.PdfReader` over PyMuPDF."""

    def __init__(
        self,
        tessdata_dir: Path,
        languages: str,
        dpi: int,
        *,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        """Read PDFs, with OCR in ``languages`` at ``dpi`` from the files in ``tessdata_dir``.

        ``ocr_engine`` replaces Tesseract, for tests.
        """
        self._ocr = ocr_engine or _tesseract_engine(tessdata_dir, languages, dpi)
        # MuPDF prints repairs and syntax problems of damaged PDFs on stderr; they are
        # counted per document instead, so that the console shows the run's progress only.
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)

    @classmethod
    def from_config(cls, config: AppConfig, project_root: Path) -> "PyMuPdfReader":
        """Return a reader with the configured OCR settings and language files.

        Raises:
            FileNotFoundError: if OCR is on and a configured language file is missing.
            ValueError: if the tessdata directory resolves outside the project root, or if
                ``ocr.languages`` repeats a language.
        """
        tessdata_dir = config.paths.resolve_against(project_root).tessdata_dir
        if config.extraction.ocr_window_pages > 0:
            names = [f"{code}.traineddata" for code in language_codes(config.ocr.languages)]
            missing = [name for name in names if not (tessdata_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Tesseract language files {missing} are missing from {tessdata_dir}; "
                    f"fetch them once with: {FETCH_COMMAND}"
                )
        return cls(tessdata_dir, config.ocr.languages, config.ocr.dpi)

    @contextmanager
    def open(self, path: Path) -> Iterator["_PyMuPdfPages"]:
        """Open the PDF at ``path`` for the duration of a ``with`` block.

        Raises:
            PdfReadError: if the file is missing, cannot be parsed, or needs a password.
        """
        pymupdf.TOOLS.reset_mupdf_warnings()
        try:
            document = pymupdf.open(path, filetype="pdf")
        except Exception as error:
            raise PdfReadError(f"cannot open the PDF: {_one_line(error)}") from error
        try:
            if document.needs_pass:
                raise PdfReadError("the PDF is encrypted and needs a password")
            yield _PyMuPdfPages(document, self._ocr)
        finally:
            document.close()


class _PyMuPdfPages:
    """:class:`~thematic_redundancy.extraction.pdf_reader.PdfPages` of one open document."""

    def __init__(self, document: pymupdf.Document, ocr: OcrEngine) -> None:
        self._document = document
        self._ocr = ocr

    @property
    def page_count(self) -> int:
        return self._document.page_count

    def text_layer(self, index: int) -> TextLayer:
        try:
            page = self._document[index]
            textpage = page.get_textpage(flags=pymupdf.TEXTFLAGS_TEXT)
            content_order = page.get_text("text", textpage=textpage)
            blocks = page.get_text("blocks", textpage=textpage, sort=True)
        except Exception as error:
            raise PdfReadError(f"cannot read page {index}: {_one_line(error)}") from error
        return TextLayer(text=_reading_order(blocks), chars=len(content_order.strip()))

    def ocr_text(self, index: int) -> str:
        try:
            return self._ocr(self._document[index])
        except Exception as error:
            raise OcrError(f"OCR failed on page {index}: {_one_line(error)}") from error

    def warning_count(self) -> int:
        return len(pymupdf.TOOLS.mupdf_warnings(reset=False).splitlines())


def _tesseract_engine(tessdata_dir: Path, languages: str, dpi: int) -> OcrEngine:
    """Return the engine that reads a whole rendered page with PyMuPDF's Tesseract."""

    def read(page: pymupdf.Page) -> str:
        textpage = page.get_textpage_ocr(
            language=languages, dpi=dpi, full=True, tessdata=str(tessdata_dir)
        )
        return _reading_order(page.get_text("blocks", textpage=textpage, sort=True))

    return read


def _reading_order(blocks: Sequence[Sequence[Any]]) -> str:
    """Join the text of already sorted blocks, one blank line between two blocks."""
    return "\n".join(block[4] for block in blocks if block[6] == _TEXT_BLOCK)


def _one_line(error: BaseException) -> str:
    return " ".join(str(error).split()) or type(error).__name__
