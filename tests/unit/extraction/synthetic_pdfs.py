"""Synthetic PDFs for the extraction tests, built with PyMuPDF in a temporary directory.

No test reads a thesis: every page holds made-up text with the marker ``SYNTHETICTEXT``,
which a test can look for wherever text must not appear, such as the console. The test
modules beside this one import it directly, since pytest puts their directory on the path.
"""

from collections.abc import Sequence
from pathlib import Path

import pymupdf

MARKER = "SYNTHETICTEXT"

HEADER = "Universidad de Ensayo"
FOOTER = "Escuela de Pruebas Sinteticas"

PAGE_WIDTH, PAGE_HEIGHT = 420, 600

ImagePage = None
"""A page spec that stands for a scanned page: one raster image, no text layer."""

PageSpec = Sequence[str] | None
"""The lines of a text page from top to bottom, or :data:`ImagePage`."""

WORDS = ("alfa", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa")


def body_lines(number: int) -> list[str]:
    """Return the body of synthetic page ``number`` (1 to 100), unique to that page beyond
    its digits."""
    word = f"{WORDS[(number - 1) % 10]}-{WORDS[(number - 1) // 10]}"
    return [
        f"{MARKER} parrafo {word} sobre un tablero de control.",
        f"Segunda linea del parrafo {word} con mas detalle.",
        f"Tercera linea que cierra la pagina {word}.",
    ]


def thesis_page(number: int) -> list[str]:
    """Return a page with a running header, its body, a running footer and its number."""
    return [HEADER, *body_lines(number), FOOTER, f"Pagina {number}"]


def write_pdf(path: Path, pages: Sequence[PageSpec]) -> Path:
    """Write a PDF with one page per spec and return its path.

    Each text page draws its lines bottom to top, so the content stream holds the footer
    first and the header last, as some word processors write them; reading order must
    still put the header first. Lines sit 40 points apart, so each one is its own block.
    """
    document = pymupdf.open()
    for spec in pages:
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        if spec is ImagePage:
            pixmap = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 60, 80), False)
            pixmap.clear_with(180)
            page.insert_image(page.rect, pixmap=pixmap)
            continue
        for index, line in reversed(list(enumerate(spec))):
            page.insert_text((36, 40 + 40 * index), line, fontname="helv", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return path


def write_encrypted_pdf(path: Path) -> Path:
    """Write a one-page PDF that needs a password to open, and return its path."""
    document = pymupdf.open()
    document.new_page().insert_text((36, 60), f"{MARKER} pagina cifrada", fontname="helv")
    document.save(
        path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="propietario", user_pw="usuario"
    )
    document.close()
    return path


def write_broken_xref_pdf(path: Path) -> Path:
    """Write a one-page PDF whose cross-reference table is damaged, which MuPDF repairs
    with warnings."""
    path.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]>>endobj\n"
        b"xref\n0 1\nbroken\ntrailer<</Root 1 0 R>>\nstartxref\n999\n%%EOF\n"
    )
    return path


def write_corrupt_file(path: Path) -> Path:
    """Write a file named like a PDF that is not one, and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this file is not a PDF at all\n" * 4)
    return path
