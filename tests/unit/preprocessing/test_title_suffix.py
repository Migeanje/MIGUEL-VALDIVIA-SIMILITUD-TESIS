"""Title suffixes: the trailing place and year of a thesis title, and how they are cut.

Every title here is synthetic.
"""

import pytest

from thematic_redundancy.corpus import profile
from thematic_redundancy.corpus.ficha import QUALITY_NOTE_MAX_LENGTH
from thematic_redundancy.preprocessing import light_cleaner, title_suffix
from thematic_redundancy.preprocessing.title_suffix import (
    MIN_TITLE_WORDS,
    NO_SUFFIX,
    NOTE_MAX_LENGTH,
    classify_title_suffix,
    strip_title_suffix,
)

# Stripping


@pytest.mark.parametrize(
    ("title", "expected", "suffix", "pattern", "places"),
    [
        (
            "Mejora de procesos en una planta textil, Arequipa 2025",
            "Mejora de procesos en una planta textil",
            ", Arequipa 2025",
            "place_and_year",
            ("Arequipa",),
        ),
        (
            "Mejora de procesos en una planta textil Arequipa - 2025",
            "Mejora de procesos en una planta textil",
            "Arequipa - 2025",
            "place_and_year",
            ("Arequipa",),
        ),
        (
            "Mejora de procesos en una planta textil, 2026.",
            "Mejora de procesos en una planta textil",
            ", 2026.",
            "year_after_separator",
            (),
        ),
        (
            "Optimización de la voladura en una mina de Pasco 2026",
            "Optimización de la voladura en una mina",
            "de Pasco 2026",
            "place_and_year",
            ("Pasco",),
        ),
        (
            "Mejora de procesos en una planta textil en Arequipa, 2023",
            "Mejora de procesos en una planta textil",
            "en Arequipa, 2023",
            "place_and_year",
            ("Arequipa",),
        ),
        (
            "Mejora de procesos en una planta textil - Arequipa, Perú 2021",
            "Mejora de procesos en una planta textil",
            "- Arequipa, Perú 2021",
            "place_and_year",
            ("Arequipa", "Perú"),
        ),
        (
            "MEJORA DE PROCESOS EN UNA PLANTA, AREQUIPA - PERU 2020-2021",
            "MEJORA DE PROCESOS EN UNA PLANTA",
            ", AREQUIPA - PERU 2020-2021",
            "place_and_year",
            ("Arequipa", "Perú"),
        ),
        (
            "Mejora de procesos en una planta textil (Arequipa, 2022).",
            "Mejora de procesos en una planta textil",
            "(Arequipa, 2022).",
            "place_and_year",
            ("Arequipa",),
        ),
        (
            "Mejora de procesos en una planta textil (2021)",
            "Mejora de procesos en una planta textil",
            "(2021)",
            "year_after_separator",
            (),
        ),
        (
            "Mejora de procesos en una planta textil en el año 2024",
            "Mejora de procesos en una planta textil",
            "en el año 2024",
            "year_after_temporal_word",
            (),
        ),
        (
            "Mejora de procesos en una planta textil periodo 2022-2023",
            "Mejora de procesos en una planta textil",
            "periodo 2022-2023",
            "year_after_temporal_word",
            (),
        ),
        (
            "Mejora de los procesos productivos de una planta 2024",
            "Mejora de los procesos productivos de una planta",
            "2024",
            "year_after_other_word",
            (),
        ),
        (
            "Mejora de procesos en una planta textil, Moquegua",
            "Mejora de procesos en una planta textil",
            ", Moquegua",
            "place_after_separator",
            ("Moquegua",),
        ),
        (
            "Mejora de procesos en una planta de la ciudad de Arequipa",
            "Mejora de procesos en una planta",
            "de la ciudad de Arequipa",
            "place_after_word",
            ("Arequipa",),
        ),
        (
            "Mejora de procesos productivos para la región Arequipa",
            "Mejora de procesos productivos",
            "para la región Arequipa",
            "place_after_word",
            ("Arequipa",),
        ),
        (
            "Gestión de riesgos en una empresa del sur del Perú",
            "Gestión de riesgos en una empresa",
            "del sur del Perú",
            "place_after_word",
            ("Perú",),
        ),
    ],
    ids=[
        "comma-place-year",
        "place-dash-year",
        "comma-year-period",
        "place-year-after-a-word",
        "en-place-comma-year",
        "dash-places-year",
        "capitals-and-range",
        "parenthesized-place-and-year",
        "parenthesized-year",
        "temporal-words",
        "temporal-word-range",
        "year-after-a-word",
        "comma-place",
        "city-of-place",
        "region-place",
        "south-of-the-country",
    ],
)
def test_strip_title_suffix_cuts_a_trailing_place_and_year_with_their_connectors(
    title: str, expected: str, suffix: str, pattern: str, places: tuple[str, ...]
) -> None:
    stripped = strip_title_suffix(title)

    assert stripped.title == expected
    assert stripped.suffix == suffix
    assert (stripped.pattern, stripped.places) == (pattern, places)
    assert stripped.stripped
    assert not stripped.refused


@pytest.mark.parametrize(
    ("title", "expected", "suffix", "places"),
    [
        (
            "Mejora de procesos en una planta en Arequipa en el año 2023",
            "Mejora de procesos en una planta",
            "en Arequipa en el año 2023",
            ("Arequipa",),
        ),
        (
            "Mejora de procesos en una planta textil entre 2019 y 2020",
            "Mejora de procesos en una planta textil",
            "entre 2019 y 2020",
            (),
        ),
    ],
    ids=["place-then-year", "two-years"],
)
def test_strip_title_suffix_cuts_again_until_no_suffix_is_left(
    title: str, expected: str, suffix: str, places: tuple[str, ...]
) -> None:
    stripped = strip_title_suffix(title)

    assert (stripped.title, stripped.suffix, stripped.places) == (expected, suffix, places)
    assert stripped.pattern == classify_title_suffix(title).pattern
    assert classify_title_suffix(stripped.title).pattern == NO_SUFFIX


@pytest.mark.parametrize(
    "title",
    [
        "Implementación de la norma ISO 9001:2015",
        "Implementación de la norma ISO 9001-2015",
        "Análisis del COVID 19",
        "Mejora de procesos para la unión",
        "Mejora de la calidad del servicio",
    ],
    ids=["norm-after-colon", "norm-after-dash", "short-number", "lower-case-place", "plain"],
)
def test_strip_title_suffix_leaves_a_title_without_suffix_unchanged(title: str) -> None:
    stripped = strip_title_suffix(title)

    assert stripped.title == title
    assert (stripped.suffix, stripped.pattern, stripped.places) == ("", NO_SUFFIX, ())
    assert not stripped.stripped
    assert not stripped.refused
    assert stripped.quality_note is None


@pytest.mark.parametrize(
    ("title", "suffix", "pattern"),
    [
        ("Arequipa 2023", "Arequipa 2023", "place_and_year"),
        ("Puno, 2025", "Puno, 2025", "place_and_year"),
        ("Minería en Arequipa", "en Arequipa", "place_after_word"),
        ("Plan maestro de Arequipa 2024", "de Arequipa 2024", "place_and_year"),
    ],
    ids=["only-a-suffix", "only-place-and-year", "one-word-left", "two-words-left"],
)
def test_strip_title_suffix_keeps_the_whole_title_when_too_little_would_be_left(
    title: str, suffix: str, pattern: str
) -> None:
    stripped = strip_title_suffix(title)

    assert stripped.title == title
    assert (stripped.suffix, stripped.pattern) == (suffix, pattern)
    assert not stripped.stripped
    assert stripped.refused
    assert stripped.quality_note is not None
    assert "kept" in stripped.quality_note
    assert suffix in stripped.quality_note


def test_strip_title_suffix_never_leaves_fewer_words_than_the_minimum() -> None:
    titles = [
        "Arequipa",
        "Perú 2024",
        "Análisis, 2023",
        "Diseño de planta en el año 2024",
        "Gestión en la ciudad de Arequipa, Perú 2021",
        "Mejora de procesos en Arequipa 2023",
    ]

    for title in titles:
        result = strip_title_suffix(title).title
        assert result
        assert result == title or len(result.split()) >= MIN_TITLE_WORDS


def test_strip_title_suffix_stops_before_a_second_cut_that_would_leave_too_little() -> None:
    stripped = strip_title_suffix("Gestión vial en Arequipa en el año 2023")

    assert stripped.title == "Gestión vial en Arequipa"
    assert (stripped.suffix, stripped.places) == ("en el año 2023", ())
    assert stripped.stripped


def test_strip_title_suffix_collapses_whitespace_first() -> None:
    stripped = strip_title_suffix("  Mejora de\tprocesos  en una\nplanta ,  Arequipa  2023 ")

    assert stripped.title == "Mejora de procesos en una planta"
    assert stripped.suffix == ", Arequipa 2023"


def test_strip_title_suffix_is_stable_on_its_own_result() -> None:
    once = strip_title_suffix("Mejora de procesos en una planta en Arequipa en el año 2023")

    again = strip_title_suffix(once.title)

    assert again.title == once.title
    assert (again.suffix, again.stripped) == ("", False)


# Quality notes


def test_quality_note_names_the_suffix_that_was_removed() -> None:
    stripped = strip_title_suffix("Mejora de procesos en una planta textil, Arequipa 2025")

    assert stripped.quality_note == 'title suffix removed (place_and_year): ", Arequipa 2025"'


def test_quality_note_fits_a_ficha_quality_note_even_for_a_very_long_suffix() -> None:
    title = "Mejora de procesos en una planta textil en " + ", ".join(["Arequipa"] * 40) + " 2023"

    note = strip_title_suffix(title).quality_note

    assert note is not None
    assert len(note) <= QUALITY_NOTE_MAX_LENGTH
    assert note.startswith("title suffix removed (place_and_year):")


def test_the_note_length_limit_equals_the_ficha_quality_note_limit() -> None:
    # preprocessing may not import corpus, so the limit is written twice; this keeps it equal.
    assert NOTE_MAX_LENGTH == QUALITY_NOTE_MAX_LENGTH


# Classification, at its new home


@pytest.mark.parametrize(
    ("title", "pattern", "places"),
    [
        ("Mejora de procesos, Arequipa 2023", "place_and_year", ("Arequipa",)),
        ("Mejora de procesos en el año 2024", "year_after_temporal_word", ()),
        ("Mejora de procesos en la ciudad de Arequipa", "place_after_word", ("Arequipa",)),
        ("Mejora de la calidad", NO_SUFFIX, ()),
    ],
)
def test_classify_title_suffix_tells_the_pattern_and_the_places(
    title: str, pattern: str, places: tuple[str, ...]
) -> None:
    assert classify_title_suffix(title) == (pattern, places)


def test_the_metadata_profile_uses_the_same_patterns_place_list_and_normalization() -> None:
    assert profile.PLACE_NAMES is title_suffix.PLACE_NAMES
    assert profile.TITLE_SUFFIX_PATTERNS is title_suffix.TITLE_SUFFIX_PATTERNS
    assert profile.NO_SUFFIX is title_suffix.NO_SUFFIX
    assert profile.TRAILING_YEAR is title_suffix.TRAILING_YEAR
    assert profile.TRAILING_PLACE is title_suffix.TRAILING_PLACE
    assert profile.TEMPORAL_WORDS is title_suffix.TEMPORAL_WORDS
    assert profile.classify_title_suffix is title_suffix.classify_title_suffix
    assert profile.normalize_text is light_cleaner.normalize_text
