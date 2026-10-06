"""Behavior of the page-text rules: low-text pages, the OCR window, and header/footer removal.

Every page here is synthetic text, so no thesis text is involved.
"""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from thematic_redundancy.extraction.page_text import (
    PageReading,
    PageText,
    assemble_pages,
    is_low_text,
    is_page_number,
    line_key,
    needs_ocr,
    remove_running_lines,
)
from thematic_redundancy.shared.config import ExtractionConfig, HeaderFooterConfig

HEADER = "Universidad Sintética de Ejemplo"
FOOTER = "Facultad de Pruebas — Escuela de Ensayos"

FIVE_PAGES = HeaderFooterConfig(edge_lines=3, min_share=0.5, min_pages=5)
SETTINGS = ExtractionConfig(min_text_chars=50, ocr_window_pages=100, header_footer=FIVE_PAGES)


WORDS = ("alfa", "beta", "gamma", "delta", "épsilon", "zeta", "eta", "theta", "iota", "kappa")
"""Words that name the synthetic pages, so that the body lines of a page differ from every
other page's in more than their digits."""


def body(number: int) -> list[str]:
    """Return the body lines of synthetic page ``number`` (1 to 100), unique to that page."""
    word = f"{WORDS[(number - 1) % 10]}-{WORDS[(number - 1) // 10]}"
    return [
        f"Párrafo {word} sobre el diseño de un tablero de control sintético.",
        f"La segunda línea del párrafo {word} continúa la idea con más detalle.",
        f"Una tercera línea cierra la página {word} del documento de prueba.",
        f"Cuarta línea de la página {word}, que sigue sin repetirse en ninguna otra.",
    ]


def page(*lines: str) -> str:
    return "\n".join(lines)


def document(pages: int) -> list[str]:
    """Return ``pages`` pages that share a header and a footer and number themselves."""
    return [
        page(HEADER, *body(number), FOOTER, f"Página {number}") for number in range(1, pages + 1)
    ]


# --- Low-text pages and the OCR window ---------------------------------------------------


@pytest.mark.parametrize(("chars", "expected"), [(0, True), (49, True), (50, False), (900, False)])
def test_a_page_with_fewer_characters_than_the_minimum_is_low_text(
    chars: int, expected: bool
) -> None:
    assert is_low_text(chars, 50) is expected


@pytest.mark.parametrize(
    ("page_index", "chars", "expected"),
    [
        (0, 0, True),
        (99, 49, True),
        (100, 0, False),  # Outside the window of 100 pages, which counts from page 0.
        (250, 0, False),
        (5, 50, False),  # A page with enough text is never OCR'd.
    ],
)
def test_only_low_text_pages_inside_the_window_need_ocr(
    page_index: int, chars: int, expected: bool
) -> None:
    assert needs_ocr(page_index, chars, SETTINGS) is expected


def test_a_window_of_zero_pages_turns_ocr_off() -> None:
    settings = SETTINGS.model_copy(update={"ocr_window_pages": 0})

    assert not needs_ocr(0, 0, settings)


# --- Line keys and page numbers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Página 12", "Página 13"),
        ("PÁGINA 7", "página 120"),
        ("Informe  final   2023", "informe final 2024"),
        ("\tUniversidad Sintética ", "universidad sintética"),
    ],
)
def test_lines_that_differ_only_in_digits_case_or_spacing_share_a_key(
    first: str, second: str
) -> None:
    assert line_key(first) == line_key(second)


def test_lines_with_different_words_have_different_keys() -> None:
    assert line_key("Capítulo de resultados") != line_key("Capítulo de discusión")


@pytest.mark.parametrize(
    "line",
    ["7", " 12 ", "123", "xii", "iv", "Página 12", "pág. 3", "Pagina xii", "12 de 120", "4/80"]
    + ["- 15 -", "— 9 —", "Página 3 de 80"],
)
def test_standalone_page_numbers_are_recognized(line: str) -> None:
    assert is_page_number(line)


@pytest.mark.parametrize(
    "line",
    [
        "2024",  # A year, not a page: theses have fewer than 1,000 pages.
        "XII",  # Upper-case roman numerals number chapters, not pages.
        "CAPÍTULO I",
        "Tabla 3",
        "iiii",  # Not a valid roman numeral.
        "12 sobre el tema",
        "",
    ],
)
def test_other_short_lines_are_not_page_numbers(line: str) -> None:
    assert not is_page_number(line)


# --- Running header and footer removal ---------------------------------------------------


def test_a_repeated_header_and_footer_and_the_page_numbers_are_removed() -> None:
    cleaned = remove_running_lines(document(6), FIVE_PAGES)

    assert cleaned.texts[0] == page(*body(1))
    assert cleaned.texts[5] == page(*body(6))
    assert cleaned.removed == (3, 3, 3, 3, 3, 3)


def test_running_lines_are_found_even_when_their_numbers_change() -> None:
    pages = [page(f"Informe de avance {number}", *body(number)) for number in range(1, 7)]

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.texts == tuple(page(*body(number)) for number in range(1, 7))


def test_a_line_must_repeat_on_the_configured_share_of_pages() -> None:
    # The header sits on 3 of 6 pages: exactly the 50% share, so it is a running header.
    pages = [
        page(HEADER, *body(number)) if number <= 3 else page(*body(number))
        for number in range(1, 7)
    ]

    assert remove_running_lines(pages, FIVE_PAGES).removed == (1, 1, 1, 0, 0, 0)
    # On 2 of 6 pages it falls below the share, and stays.
    pages[2] = page(*body(3))
    assert remove_running_lines(pages, FIVE_PAGES).removed == (0, 0, 0, 0, 0, 0)


def test_a_share_that_floating_point_rounds_up_still_counts_exactly() -> None:
    # 0.14 * 50 is 7.000000000000001 in floating point; 7 pages of 50 are still 14%.
    pages = [
        page(HEADER, *body(number)) if number <= 7 else page(*body(number))
        for number in range(1, 51)
    ]
    settings = FIVE_PAGES.model_copy(update={"min_share": 0.14})

    assert remove_running_lines(pages, settings).removed == (1,) * 7 + (0,) * 43


def test_short_documents_keep_their_repeated_lines_but_lose_page_numbers() -> None:
    cleaned = remove_running_lines(document(4), FIVE_PAGES)

    assert cleaned.texts[0] == page(HEADER, *body(1), FOOTER)
    assert cleaned.removed == (1, 1, 1, 1)


def test_pages_without_text_do_not_count_toward_the_share() -> None:
    pages = [*document(5), "", "   \n ", ""]

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.removed == (3, 3, 3, 3, 3, 0, 0, 0)
    assert cleaned.texts[5:] == ("", "", "")


def test_a_repeated_line_inside_the_body_is_kept() -> None:
    # The repeated line is the fourth of seven content lines, so it is at neither edge.
    pages = [
        page(*body(number)[:3], "Línea repetida en medio del texto", *body(number)[1:])
        for number in range(1, 7)
    ]

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.removed == (0, 0, 0, 0, 0, 0)
    assert cleaned.texts == tuple(pages)


def test_only_the_configured_number_of_edge_lines_is_inspected() -> None:
    pages = [page("Sección fija", HEADER, *body(number)) for number in range(1, 7)]
    one_line = FIVE_PAGES.model_copy(update={"edge_lines": 1})

    cleaned = remove_running_lines(pages, one_line)

    assert cleaned.texts[0] == page(HEADER, *body(1))
    assert remove_running_lines(pages, FIVE_PAGES).texts[0] == page(*body(1))


def test_a_page_whose_only_line_is_a_running_line_keeps_it() -> None:
    pages = [*document(6), HEADER, "15"]

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.texts[6:] == (HEADER, "15")
    assert cleaned.removed[6:] == (0, 0)


def test_a_page_of_running_lines_only_ends_up_empty() -> None:
    pages = [*document(6), page(HEADER, "", FOOTER, "Página 7")]

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.texts[6] == ""
    assert cleaned.removed[6] == 3


def test_page_numbers_are_removed_only_at_the_edges_of_a_page() -> None:
    lines = [*body(1), "12", *body(2)]
    pages = [page("3", *lines, "Página 4 de 80")]

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.texts == (page(*lines),)
    assert cleaned.removed == (2,)


def test_an_edge_line_of_digits_only_goes_only_when_it_is_a_page_number() -> None:
    # Bare page numbers make "#" a running key, which a year or any other number shares.
    pages = [page(*body(number), str(number + 10)) for number in range(1, 7)]
    pages[0] = page(*body(1), "2024")  # A cover page with its year at the foot.
    pages[1] = page("1500", *body(2), "12")

    cleaned = remove_running_lines(pages, FIVE_PAGES)

    assert cleaned.texts[0] == page(*body(1), "2024")
    assert cleaned.texts[1] == page("1500", *body(2))
    assert cleaned.texts[2:] == tuple(page(*body(number)) for number in range(3, 7))
    assert cleaned.removed == (0, 1, 1, 1, 1, 1)


def test_blank_lines_left_at_the_ends_of_a_page_are_trimmed() -> None:
    pages = [page("", "  ", "Página 1", "", *body(1), "", "  ")]

    assert remove_running_lines(pages, FIVE_PAGES).texts == (page(*body(1)),)


# --- Page records ------------------------------------------------------------------------


def reading(
    text: str,
    chars: int | None = None,
    *,
    ocr_text: str | None = None,
    ocr_attempted: bool = False,
    ocr_failed: bool = False,
) -> PageReading:
    return PageReading(
        layer_text=text,
        layer_chars=len(text.strip()) if chars is None else chars,
        ocr_attempted=ocr_attempted or ocr_text is not None or ocr_failed,
        ocr_text=ocr_text,
        ocr_failed=ocr_failed,
    )


def sources(pages: Sequence[PageText]) -> list[str]:
    return [record.source for record in pages]


def test_each_page_record_tells_where_its_text_came_from() -> None:
    texts = document(6)
    readings = [
        reading(texts[0]),
        reading("", ocr_text=page(HEADER, *body(2), FOOTER, "Página 2")),
        reading("Figura 3", ocr_failed=True),
        reading(""),
        reading(texts[4]),
        reading(texts[5]),
    ]

    pages = assemble_pages(readings, SETTINGS)

    assert [record.page_index for record in pages] == [0, 1, 2, 3, 4, 5]
    assert sources(pages) == ["text_layer", "ocr", "text_layer", "empty"] + ["text_layer"] * 2
    assert pages[0].text == page(*body(1))
    # Running lines come off OCR text too.
    assert pages[1].text == page(*body(2))
    assert pages[2].text == "Figura 3"
    assert pages[3].text == ""
    assert [record.char_count for record in pages] == [len(record.text) for record in pages]
    assert [record.low_text for record in pages] == [False, True, True, True, False, False]
    assert [record.ocr_attempted for record in pages] == [False, True, True, False, False, False]
    assert [record.ocr_failed for record in pages] == [False, False, True, False, False, False]
    assert [record.lines_removed for record in pages] == [3, 3, 0, 0, 3, 3]


def test_ocr_that_reads_nothing_falls_back_to_the_text_layer() -> None:
    pages = assemble_pages([reading("Gráfico 4", ocr_text="  \n ")], SETTINGS)

    assert (pages[0].source, pages[0].text, pages[0].ocr_failed) == (
        "text_layer",
        "Gráfico 4",
        False,
    )


def test_a_low_text_flag_comes_from_the_text_layer_measure() -> None:
    long_text = "x" * 60
    pages = assemble_pages([reading(long_text, chars=49), reading("corto", chars=50)], SETTINGS)

    assert [record.low_text for record in pages] == [True, False]


@pytest.mark.parametrize(
    "fields",
    [
        {"source": "empty", "text": "algo", "char_count": 4},
        {"source": "text_layer", "text": "", "char_count": 0},
        {"text": "algo", "char_count": 3},
        {"text": " algo", "char_count": 5},
        {"ocr_failed": True},
        {"source": "ocr", "low_text": True},
        {"ocr_attempted": True, "low_text": False},
        {"source": "ocr", "low_text": True, "ocr_attempted": True, "ocr_failed": True},
    ],
)
def test_page_records_refuse_contradictory_fields(fields: dict[str, object]) -> None:
    valid: dict[str, object] = {
        "page_index": 0,
        "text": "algo",
        "source": "text_layer",
        "char_count": 4,
        "low_text": False,
        "ocr_attempted": False,
        "ocr_failed": False,
        "lines_removed": 0,
    }
    PageText.model_validate(valid)

    with pytest.raises(ValidationError):
        PageText.model_validate(valid | fields)


def test_page_record_errors_never_echo_the_page_text() -> None:
    with pytest.raises(ValidationError) as caught:
        PageText.model_validate(
            {
                "page_index": 0,
                "text": "TEXTO PRIVADO",
                "source": "text_layer",
                "char_count": 99,
                "low_text": False,
                "ocr_attempted": False,
                "ocr_failed": False,
                "lines_removed": 0,
            }
        )

    assert "TEXTO PRIVADO" not in str(caught.value)
