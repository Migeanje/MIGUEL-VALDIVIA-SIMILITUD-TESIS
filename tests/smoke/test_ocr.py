"""Smoke check that PyMuPDF's built-in Tesseract reads Spanish text from a page without text.

PyMuPDF bundles the OCR engine, so no Tesseract installation is involved (D07); the check
only needs the configured language files, which the OCR assets command fetches once per
machine (see the README). The test is marked ``smoke``, so the default run excludes it.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from thematic_redundancy.extraction.ocr_assets import FETCH_COMMAND, language_codes
from thematic_redundancy.shared.config import AppConfig, load_config

if TYPE_CHECKING:
    import pymupdf

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

SOURCE_LINES = (
    "Optimización del diseño de la compañía minera",
    "Análisis económico de la producción en el año 2024",
)

KEY_WORDS = frozenset(
    {"Optimización", "diseño", "compañía", "minera", "Análisis", "económico", "producción", "año"}
)
"""Words of :data:`SOURCE_LINES` that must survive OCR with their accents and ñ."""

SCAN_DPI = 200
"""Resolution of the image that stands in for a scanned page."""


def words_of(text: str) -> set[str]:
    """Return the words of ``text`` in NFC form, ignoring whitespace and punctuation."""
    return set(re.findall(r"\w+", unicodedata.normalize("NFC", text)))


def configured_tessdata_dir(config: AppConfig) -> Path:
    """Return the configured tessdata directory, or fail with the fix if a file is missing."""
    tessdata_dir = config.paths.resolve_against(PROJECT_ROOT).tessdata_dir
    file_names = [f"{code}.traineddata" for code in language_codes(config.ocr.languages)]
    missing = [name for name in file_names if not (tessdata_dir / name).is_file()]
    if missing:
        pytest.fail(
            f"Tesseract language files {missing} are missing from '{tessdata_dir}'. "
            f"Fetch them once with: {FETCH_COMMAND}",
            pytrace=False,
        )
    return tessdata_dir


def image_only_document() -> pymupdf.Document:
    """Return a one-page PDF whose only content is a raster image of :data:`SOURCE_LINES`."""
    import pymupdf

    source = pymupdf.open()
    source_page = source.new_page(width=420, height=130)
    for index, line in enumerate(SOURCE_LINES):
        source_page.insert_text((36, 56 + 28 * index), line, fontname="helv", fontsize=14)
    # The text layer shows that the font drew every accented glyph that OCR must read back.
    assert words_of(source_page.get_text()) >= KEY_WORDS

    scan = pymupdf.open()
    scan_page = scan.new_page(width=source_page.rect.width, height=source_page.rect.height)
    scan_page.insert_image(scan_page.rect, pixmap=source_page.get_pixmap(dpi=SCAN_DPI))
    return scan


def test_builtin_tesseract_reads_spanish_accents_from_a_page_without_text() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    tessdata_dir = configured_tessdata_dir(config)
    scan = image_only_document()  # Keep the document alive: its pages refer back to it.
    page = scan[0]
    assert page.get_text() == ""  # Without a text layer, only OCR can read the page.

    textpage = page.get_textpage_ocr(
        language=config.ocr.languages, dpi=config.ocr.dpi, full=True, tessdata=str(tessdata_dir)
    )
    text = page.get_text(textpage=textpage)

    missing = KEY_WORDS - words_of(text)
    assert not missing, f"OCR lost {sorted(missing)}; it read {text!r}"
