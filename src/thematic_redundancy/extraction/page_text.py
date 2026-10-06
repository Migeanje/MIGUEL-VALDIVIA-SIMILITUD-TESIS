"""Rules that turn what was read from the pages of one PDF into page records.

Nothing here reads a PDF: the extraction use case reads the pages through the PDF-reader
port and hands the results to these functions, so every rule is plain text in, text out.

- **Low-text page.** A page whose text layer has fewer than ``extraction.min_text_chars``
  characters, stripped of surrounding whitespace. This is the data card's measure, taken on
  the text in content-stream order.
- **OCR window (D25).** Only the low-text pages among the first ``extraction.ocr_window_pages``
  pages are OCR'd, because the objectives (T10) lie there and the thesis body is not used.
- **Running headers and footers.** A line among the first or last
  ``header_footer.edge_lines`` non-empty lines of a page is a running line when its
  :func:`line_key` occurs at the edges of at least ``header_footer.min_share`` of the
  document's pages with text, and the document has at least ``header_footer.min_pages`` such
  pages. A standalone page number at the edge of a page (:func:`is_page_number`) goes too.
  An edge line of digits only goes only when it is a page number: bare page numbers make
  ``#`` a running key, which a year such as ``2024`` shares. A page whose only line is a
  running line keeps it.

Any change to these rules that alters the page records must raise
:data:`~thematic_redundancy.extraction.text_manifest.EXTRACTION_VERSION`, so that a rerun
extracts every PDF again.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from thematic_redundancy.shared.config import ExtractionConfig, HeaderFooterConfig

TextSource = Literal["text_layer", "ocr", "empty"]
"""Where the text of a page record came from:

- ``text_layer``: the PDF's own text layer;
- ``ocr``: OCR of the rendered page, because its text layer is low-text;
- ``empty``: nothing, once running lines are removed. The page may hold images only, or be
  a low-text page outside the OCR window whose few characters were all running lines.
"""

Count = Annotated[StrictInt, Field(ge=0)]

_DIGITS = re.compile(r"\d+")

_DIGITS_ONLY = re.compile(r"\d+(?:\s+\d+)*")
"""A line of digits only, such as ``12``, ``2024`` or ``1500``, once stripped; its
:func:`line_key` holds only ``#``."""

_PAGE_NUMBER = re.compile(
    r"[-–—]?\s*(?:p[aá]g(?:ina|\.)?\s*)?(?:(?P<arabic>\d{1,3})|(?P<roman>[ivxlcdm]{1,7}))"
    r"(?:\s*(?:de|/)\s*\d{1,3})?\s*[-–—]?",
    re.IGNORECASE,
)
"""A line that is only a page number: ``12``, ``xii``, ``Página 12``, ``pág. 3``,
``12 de 120``, ``12/120`` or ``- 12 -``. Arabic numbers have at most 3 digits, so a year
such as ``2024`` is not one."""

_ROMAN = re.compile(r"m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})")
"""A valid roman numeral in lower case. Unlike :data:`_PAGE_NUMBER`, it heeds letter case, so
upper-case numerals, which number chapters, never count as page numbers."""


class PageText(BaseModel):
    """The text of one PDF page, as the pipeline keeps it, and how it was obtained."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    page_index: Count
    """Position of the page in its PDF, counted from 0."""
    text: str
    """Text of the page without its running lines, stripped of surrounding whitespace."""
    source: TextSource
    char_count: Count
    """Characters of ``text``."""
    low_text: bool
    """The page's text layer is low-text."""
    ocr_attempted: bool
    ocr_failed: bool
    """OCR ran on the page but raised an error, so its text layer was kept."""
    lines_removed: Count
    """Running header, footer and page-number lines removed from the page."""

    @model_validator(mode="after")
    def _require_consistent_fields(self) -> Self:
        if self.text != self.text.strip():
            raise ValueError("text must be stripped of surrounding whitespace")
        if self.char_count != len(self.text):
            raise ValueError("char_count must be the length of text")
        if (self.source == "empty") != (self.char_count == 0):
            raise ValueError("a page is 'empty' exactly when it has no text")
        if self.ocr_attempted and not self.low_text:
            raise ValueError("only low-text pages are OCR'd")
        if self.ocr_failed and not self.ocr_attempted:
            raise ValueError("OCR can fail only on a page where it was attempted")
        if self.source == "ocr" and (not self.ocr_attempted or self.ocr_failed):
            raise ValueError("an 'ocr' page needs an OCR attempt that succeeded")
        return self


@dataclass(frozen=True)
class PageReading:
    """What was read from one page: its text layer and, for a low-text page, its OCR."""

    layer_text: str
    """Text layer in reading order."""
    layer_chars: int
    """The low-text measure of the text layer (see the module documentation)."""
    ocr_attempted: bool
    ocr_text: str | None
    """OCR text, or ``None`` when OCR was not attempted or failed."""
    ocr_failed: bool


@dataclass(frozen=True)
class CleanedPages:
    """Page texts without their running lines, and how many lines each page lost."""

    texts: tuple[str, ...]
    removed: tuple[int, ...]


def is_low_text(chars: int, min_text_chars: int) -> bool:
    """Tell whether a text layer of ``chars`` characters, stripped, is low-text."""
    return chars < min_text_chars


def needs_ocr(page_index: int, chars: int, settings: ExtractionConfig) -> bool:
    """Tell whether the page at ``page_index``, whose text layer has ``chars`` characters
    stripped, is OCR'd: it must be low-text and inside the OCR window."""
    return is_low_text(chars, settings.min_text_chars) and page_index < settings.ocr_window_pages


def line_key(line: str) -> str:
    """Return the form under which a line is compared with the lines of other pages.

    Letter case, runs of whitespace and the value of every number are ignored, so
    ``Página 12`` and ``página 13`` share the key ``página #``.
    """
    folded = unicodedata.normalize("NFC", line).casefold()
    return " ".join(_DIGITS.sub("#", folded).split())


def is_page_number(line: str) -> bool:
    """Tell whether ``line`` holds only a page number, such as ``12``, ``xii``,
    ``Página 12`` or ``12 de 120``.

    Roman numerals count only in lower case, the way front matter is numbered; upper-case
    ones number chapters.
    """
    match = _PAGE_NUMBER.fullmatch(line.strip())
    if match is None:
        return False
    roman = match["roman"]
    return roman is None or _ROMAN.fullmatch(roman) is not None


def remove_running_lines(texts: Sequence[str], settings: HeaderFooterConfig) -> CleanedPages:
    """Remove the running headers, footers and page numbers from the pages of one document.

    ``texts`` holds the pages in order. Only whole lines among the first and last
    ``settings.edge_lines`` non-empty lines of a page are removed, and a page whose only
    line is a running line keeps it. Each resulting text is stripped of surrounding
    whitespace.
    """
    pages = [text.splitlines() for text in texts]
    edges = [_edge_indices(lines, settings.edge_lines) for lines in pages]
    running = _running_keys(pages, edges, settings)
    cleaned: list[str] = []
    removed: list[int] = []
    for lines, indices in zip(pages, edges, strict=True):
        if sum(1 for line in lines if line.strip()) <= 1:
            dropped: set[int] = set()
        else:
            dropped = {index for index in indices if _is_dropped(lines[index], running)}
        kept = [line for index, line in enumerate(lines) if index not in dropped]
        cleaned.append("\n".join(kept).strip())
        removed.append(len(dropped))
    return CleanedPages(texts=tuple(cleaned), removed=tuple(removed))


def _is_dropped(line: str, running: frozenset[str]) -> bool:
    """Tell whether an edge line goes: a page number always does, and a running line does
    unless it holds digits only, since a year shares the key ``#`` of bare page numbers."""
    if is_page_number(line):
        return True
    return line_key(line) in running and _DIGITS_ONLY.fullmatch(line.strip()) is None


def _edge_indices(lines: Sequence[str], edge_lines: int) -> frozenset[int]:
    """Return the indices of the first and last ``edge_lines`` non-empty lines."""
    content = [index for index, line in enumerate(lines) if line.strip()]
    return frozenset(content[:edge_lines]) | frozenset(content[-edge_lines:])


def _running_keys(
    pages: Sequence[Sequence[str]],
    edges: Sequence[frozenset[int]],
    settings: HeaderFooterConfig,
) -> frozenset[str]:
    """Return the keys of the edge lines that repeat on enough pages to be running lines."""
    with_text = sum(1 for lines in pages if any(line.strip() for line in lines))
    if with_text < settings.min_pages:
        return frozenset()
    counts: Counter[str] = Counter()
    for lines, indices in zip(pages, edges, strict=True):
        counts.update({line_key(lines[index]) for index in indices})
    # The share is taken as the decimal written in the configuration: in floating point,
    # 0.14 * 50 is 7.000000000000001, which would ask for 8 pages instead of 7.
    needed = max(2, math.ceil(Fraction(repr(settings.min_share)) * with_text))
    return frozenset(key for key, count in counts.items() if count >= needed)


def assemble_pages(
    readings: Sequence[PageReading], settings: ExtractionConfig
) -> tuple[PageText, ...]:
    """Build the page records of one document from what was read from its pages, in order.

    A page takes its OCR text when OCR read something, and its text layer otherwise. The
    running lines are then removed across the whole document, OCR pages included, and a
    page left without text is ``empty``.
    """
    chosen: list[tuple[str, TextSource]] = [
        (reading.ocr_text, "ocr")
        if reading.ocr_text is not None and reading.ocr_text.strip()
        else (reading.layer_text, "text_layer")
        for reading in readings
    ]
    cleaned = remove_running_lines([text for text, _ in chosen], settings.header_footer)
    pages: list[PageText] = []
    for index, reading in enumerate(readings):
        text = cleaned.texts[index]
        pages.append(
            PageText(
                page_index=index,
                text=text,
                source=chosen[index][1] if text else "empty",
                char_count=len(text),
                low_text=is_low_text(reading.layer_chars, settings.min_text_chars),
                ocr_attempted=reading.ocr_attempted,
                ocr_failed=reading.ocr_failed,
                lines_removed=cleaned.removed[index],
            )
        )
    return tuple(pages)
