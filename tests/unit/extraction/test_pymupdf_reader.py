"""Behavior of the PyMuPDF adapter of the PDF-reader port, on synthetic PDFs.

OCR goes through an injected engine here, so these tests need no language files; the smoke
suite runs the real Tesseract OCR through the same adapter.
"""

from pathlib import Path

import pymupdf
import pytest
from synthetic_pdfs import (
    FOOTER,
    HEADER,
    ImagePage,
    body_lines,
    thesis_page,
    write_broken_xref_pdf,
    write_corrupt_file,
    write_encrypted_pdf,
    write_pdf,
)

from thematic_redundancy.extraction.ocr_assets import FETCH_COMMAND
from thematic_redundancy.extraction.pdf_reader import OcrError, PdfReadError
from thematic_redundancy.extraction.pymupdf_reader import PyMuPdfReader
from thematic_redundancy.shared.config import AppConfig, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def non_empty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


class FakeOcr:
    """OCR engine that answers with fixed text, or fails, and records the pages it read."""

    def __init__(self, text: str = "Texto leído por OCR", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.pages: list[int] = []

    def __call__(self, page: pymupdf.Page) -> str:
        self.pages.append(page.number)
        if self.fail:
            raise RuntimeError("synthetic OCR failure")
        return self.text


def reader(ocr: FakeOcr | None = None) -> PyMuPdfReader:
    return PyMuPdfReader(Path("tessdata"), "spa+eng", 300, ocr_engine=ocr or FakeOcr())


def test_the_text_layer_comes_in_reading_order_with_a_blank_line_between_blocks(
    tmp_path: Path,
) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [thesis_page(1), thesis_page(2)])

    with reader().open(path) as pages:
        assert pages.page_count == 2
        layer = pages.text_layer(1)

    # The content stream holds the footer first; reading order puts the header first.
    assert non_empty_lines(layer.text) == thesis_page(2)
    assert "\n\n" in layer.text
    stream_order = "\n".join(reversed(thesis_page(2)))
    assert layer.chars == len(stream_order)


def test_the_low_text_measure_counts_the_stripped_text_in_content_stream_order(
    tmp_path: Path,
) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [[HEADER, *body_lines(1)]])

    with reader().open(path) as pages:
        layer = pages.text_layer(0)
    with pymupdf.open(path) as document:
        expected = len(document[0].get_text().strip())

    assert layer.chars == expected


def test_an_image_only_page_has_an_empty_text_layer(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [ImagePage])

    with reader().open(path) as pages:
        layer = pages.text_layer(0)

    assert (layer.text, layer.chars) == ("", 0)


def test_ocr_goes_through_the_engine_for_the_requested_page_only(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [thesis_page(1), ImagePage, thesis_page(3)])
    ocr = FakeOcr()

    with reader(ocr).open(path) as pages:
        text = pages.ocr_text(1)

    assert text == "Texto leído por OCR"
    assert ocr.pages == [1]


def test_an_ocr_failure_is_reported_as_an_ocr_error_of_that_page(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [ImagePage, ImagePage])

    with reader(FakeOcr(fail=True)).open(path) as pages:
        with pytest.raises(OcrError, match="page 1"):
            pages.ocr_text(1)
        assert pages.text_layer(0).chars == 0  # The document stays readable.


@pytest.mark.parametrize(
    "write",
    [write_corrupt_file, write_encrypted_pdf, lambda path: path],
    ids=["not-a-pdf", "encrypted", "missing"],
)
def test_a_pdf_that_cannot_be_read_is_reported_as_a_read_error(tmp_path: Path, write) -> None:
    path = write(tmp_path / "doc.pdf")

    with pytest.raises(PdfReadError), reader().open(path):
        pass


def test_a_page_outside_the_document_is_a_read_error(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", [thesis_page(1)])

    with reader().open(path) as pages, pytest.raises(PdfReadError, match="page 3"):
        pages.text_layer(3)


def test_library_warnings_are_counted_per_document(tmp_path: Path) -> None:
    damaged = write_broken_xref_pdf(tmp_path / "damaged.pdf")
    clean = write_pdf(tmp_path / "clean.pdf", [[HEADER, FOOTER]])

    with reader().open(damaged) as pages:
        assert pages.page_count == 1
        assert pages.warning_count() > 0
    with reader().open(clean) as pages:
        pages.text_layer(0)
        assert pages.warning_count() == 0


@pytest.fixture
def config() -> AppConfig:
    return load_config(DEFAULT_CONFIG_PATH)


def test_the_reader_from_config_refuses_missing_language_files(
    config: AppConfig, tmp_path: Path
) -> None:
    (tmp_path / "tessdata").mkdir()
    (tmp_path / "tessdata" / "spa.traineddata").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="eng.traineddata") as caught:
        PyMuPdfReader.from_config(config, tmp_path)

    assert FETCH_COMMAND in str(caught.value)


def test_the_reader_from_config_needs_no_language_files_when_ocr_is_off(
    config: AppConfig, tmp_path: Path
) -> None:
    extraction = config.extraction.model_copy(update={"ocr_window_pages": 0})

    reader_ = PyMuPdfReader.from_config(
        config.model_copy(update={"extraction": extraction}), tmp_path
    )

    assert isinstance(reader_, PyMuPdfReader)
