"""Aggregate profile of the snapshot metadata, used by the exploratory analysis.

Every record here is synthetic. Token counts come from a stand-in tokenizer that gives one
token per word plus the two special tokens of a sequence, so every expected value can be
worked out by hand.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

import pytest

from thematic_redundancy.corpus.profile import (
    KeywordCount,
    LineBreakProfile,
    abstract_profile,
    access_class,
    advisor_profile,
    calendar_profile,
    classify_title_suffix,
    describe,
    duplicate_profile,
    histogram,
    issue_day,
    keyword_profile,
    language_profile,
    leaked_strings,
    normalize_keyword,
    ocde_code,
    ocde_profile,
    other_items_profile,
    pack_sentences,
    profile_metadata,
    rights_profile,
    sensitive_strings,
    split_sentences,
    title_profile,
)
from thematic_redundancy.corpus.snapshot import SnapshotRecord

RENATI_TYPE = "https://purl.org/pe-repo/renati/type#"
TESIS = f"{RENATI_TYPE}tesis"
SUFICIENCIA = f"{RENATI_TYPE}trabajoDeSuficienciaProfesional"
OPEN = "https://purl.org/coar/access_right/c_abf2"
EMBARGOED = "https://purl.org/coar/access_right/c_f1cf"
RESTRICTED = "https://purl.org/coar/access_right/c_16ec"
METADATA_ONLY = "https://purl.org/coar/access_right/c_14cb"
OCDE = "https://purl.org/pe-repo/ocde/ford#"

SNAPSHOT_DATE = date(2026, 10, 2)
HARVEST_TIME = datetime(2026, 10, 2, 22, 44, 12, tzinfo=UTC)
PROGRAMS = ("sistemas", "industrial", "minas")

FieldValue = str | Sequence[str] | None


def count_tokens(text: str) -> int:
    """Stand-in tokenizer: one token per word, plus the two special tokens of a sequence."""
    return len(text.split()) + 2


def entry(value: str) -> dict[str, Any]:
    """Return one metadata value in the shape that DSpace serves."""
    return {"value": value, "language": None, "authority": None, "confidence": -1, "place": 0}


def record(
    number: int, fields: Mapping[str, FieldValue] | None = None, *, program: str = "sistemas"
) -> SnapshotRecord:
    """Return a synthetic open-access thesis issued on 2023-05-10.

    ``fields`` replaces or adds metadata: a string gives one value, a sequence several, and
    ``None`` drops the field.
    """
    values: dict[str, FieldValue] = {
        "renati.type": TESIS,
        "dc.date.issued": "2023-05-10",
        "dc.title": f"Título sintético {number:03d}",
        "dc.rights": OPEN,
        **(fields or {}),
    }
    metadata = {
        key: [entry(value)] if isinstance(value, str) else [entry(item) for item in value]
        for key, value in values.items()
        if value is not None
    }
    return SnapshotRecord(
        uuid=UUID(f"00000000-0000-4000-8000-{number:012d}"),
        handle=f"123456789/{1000 + number}",
        program_key=program,
        collection_uuid=UUID("00000000-0000-4000-8000-0000000000c1"),
        harvested_at=HARVEST_TIME,
        metadata=metadata,
    )


# Dates


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023", date(2023, 1, 1)),
        ("2023-07", date(2023, 7, 1)),
        ("2023-07-15", date(2023, 7, 15)),
        ("2023-07-15T10:20:30Z", date(2023, 7, 15)),
        (" 2023-07-15 ", date(2023, 7, 15)),
        ("2023-13", None),
        ("julio de 2023", None),
        ("", None),
    ],
    ids=["year", "month", "day", "timestamp", "padded", "bad-month", "words", "empty"],
)
def test_issue_day_reads_a_partial_date_as_its_first_day(value: str, expected: date | None) -> None:
    assert issue_day(value) == expected


# Statistics


def test_describe_gives_the_mean_and_linear_percentiles() -> None:
    stats = describe([5, 1, 4, 2, 3])

    assert stats is not None
    assert (stats.count, stats.mean, stats.min, stats.max) == (5, 3, 1, 5)
    assert (stats.p25, stats.median, stats.p75) == (2, 3, 4)
    assert stats.p95 == pytest.approx(4.8)


def test_describe_of_nothing_is_none() -> None:
    assert describe([]) is None


def test_histogram_counts_values_per_bin_and_keeps_the_empty_bins_in_between() -> None:
    result = histogram([1, 15, 16, 50], bin_width=16)

    assert result.bin_width == 16
    assert result.counts == {0: 2, 16: 1, 32: 0, 48: 1}


def test_histogram_of_nothing_has_no_bins() -> None:
    assert histogram([], bin_width=8).counts == {}


def test_histogram_refuses_a_bin_narrower_than_one() -> None:
    with pytest.raises(ValueError, match="bin_width"):
        histogram([1], bin_width=0)


# Sentences and chunks


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Primera oración. Segunda oración? ¿Tercera? ¡Cuarta! Él cerró.",
            ["Primera oración.", "Segunda oración?", "¿Tercera?", "¡Cuarta!", "Él cerró."],
        ),
        (
            "Una oración que\nsigue en otra línea. Otra\r\nmás.",
            ["Una oración que sigue en otra línea.", "Otra más."],
        ),
        ("Se midió 3.5 kg. luego se pesó 2.1 kg.", ["Se midió 3.5 kg. luego se pesó 2.1 kg."]),
        ('Dijo "basta." Luego siguió.', ['Dijo "basta."', "Luego siguió."]),
        ("El costo bajó. 25 % menos.", ["El costo bajó.", "25 % menos."]),
        ("   ", []),
    ],
    ids=[
        "punctuation",
        "hard-wrapped",
        "decimals-and-lower-case",
        "closing-quote",
        "digit",
        "blank",
    ],
)
def test_split_sentences_joins_wrapped_lines_and_splits_after_sentence_punctuation(
    text: str, expected: list[str]
) -> None:
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    ("sentence_tokens", "expected"),
    [
        ([], 0),
        ([0, 0], 0),
        ([50, 50, 50], 2),
        ([126], 1),
        ([100, 26], 1),
        ([100, 27], 2),
        ([130], 2),
        ([130, 10], 2),
        ([10, 130], 3),
        ([252], 2),
    ],
    ids=[
        "none",
        "empty-sentences",
        "greedy",
        "exact-fit",
        "fills-a-chunk",
        "one-token-over",
        "long-sentence",
        "remainder-shared",
        "long-sentence-starts-a-chunk",
        "two-full-pieces",
    ],
)
def test_pack_sentences_counts_greedy_sentence_aligned_chunks(
    sentence_tokens: list[int], expected: int
) -> None:
    assert pack_sentences(sentence_tokens, budget=126) == expected


def test_pack_sentences_refuses_a_budget_below_one() -> None:
    with pytest.raises(ValueError, match="budget"):
        pack_sentences([1], budget=0)


# Abstracts


def test_abstract_profile_measures_the_present_abstracts_and_their_chunks() -> None:
    theses = [
        record(1, {"dc.description.abstract": "Uno dos tres. Cuatro cinco."}),
        record(
            2, {"dc.description.abstract": "Uno dos tres cuatro cinco seis siete ocho nueve diez."}
        ),
        record(
            3, {"dc.description.abstract": "Uno dos\ntres cuatro. Cinco seis siete ocho. Nueve."}
        ),
        record(
            4,
            {
                "dc.description.abstract": (
                    "Uno dos tres cuatro cinco. Seis siete ocho nueve diez. "
                    "Once doce trece catorce quince."
                )
            },
        ),
        record(5, {"dc.description.abstract": "   "}),
        record(6),
    ]

    profile = abstract_profile(theses, count_tokens, token_limit=10)

    assert (profile.present, profile.missing) == (4, 2)
    lengths = profile.lengths
    assert lengths is not None
    assert (lengths.tokens.min, lengths.tokens.median, lengths.tokens.max) == (7, 11.5, 17)
    assert (lengths.words.min, lengths.words.median, lengths.words.max) == (5, 9.5, 15)
    assert lengths.characters.min == len("Uno dos tres. Cuatro cinco.")
    assert lengths.characters.max == len(
        "Uno dos tres cuatro cinco. Seis siete ocho nueve diez. Once doce trece catorce quince."
    )
    assert (lengths.token_limit, lengths.above_limit) == (10, 3)
    assert lengths.share_above_limit == pytest.approx(0.75)
    chunks = profile.chunks
    assert (chunks.token_limit, chunks.special_tokens) == (10, 2)
    assert chunks.sentence_aligned == {1: 1, 2: 2, 3: 1}
    assert chunks.lower_bound == {1: 1, 2: 3}
    assert chunks.sentence_aligned_stats is not None
    assert chunks.sentence_aligned_stats.median == 2
    assert chunks.sentence_sums_differing == 0


def test_abstract_profile_counts_hard_line_breaks_and_text_defects() -> None:
    theses = [
        record(
            1,
            {
                "dc.description.abstract": (
                    "Una oración que\nsigue. Otra oración\r\nque sigue y se divi-\ndió. Fin."
                )
            },
        ),
        record(2, {"dc.description.abstract": "Una oración.\nOtra oración."}),
        record(3, {"dc.description.abstract": "Sin saltos. Palabras clave: gestión, calidad"}),
        record(
            4, {"dc.description.abstract": "Texto con un car\N{REPLACEMENT CHARACTER}cter dañado."}
        ),
    ]

    profile = abstract_profile(theses, count_tokens, token_limit=128)

    assert profile.line_breaks == LineBreakProfile(
        texts_with_breaks=2, breaks=4, inside_sentence=3, after_hyphen=1
    )
    assert profile.with_replacement_character == 1
    assert profile.with_keywords_section == 1


def test_abstract_profile_refuses_a_limit_that_leaves_no_room_for_content() -> None:
    with pytest.raises(ValueError, match="token_limit"):
        abstract_profile([record(1)], count_tokens, token_limit=2)


# Titles


@pytest.mark.parametrize(
    ("title", "pattern", "places"),
    [
        ("Mejora de procesos, Arequipa 2023", "place_and_year", ("Arequipa",)),
        ("Mejora de procesos - Arequipa, Perú 2021", "place_and_year", ("Arequipa", "Perú")),
        ("Mejora de procesos en Arequipa – 2024.", "place_and_year", ("Arequipa",)),
        ("MEJORA DE PROCESOS, AREQUIPA - PERU 2020-2021", "place_and_year", ("Arequipa", "Perú")),
        ("Mejora de procesos en Camana (2022)", "place_and_year", ("Camaná",)),
        ("Mejora de procesos, 2025", "year_after_separator", ()),
        ("Mejora de procesos - 2019", "year_after_separator", ()),
        ("Mejora de procesos (2021)", "year_after_separator", ()),
        ("Mejora de procesos en el año 2024", "year_after_temporal_word", ()),
        ("Mejora de procesos periodo 2022-2023", "year_after_temporal_word", ()),
        ("Mejora de procesos productivos 2024", "year_after_other_word", ()),
        ("Mejora de procesos en TecnoArequipa 2023", "year_after_other_word", ()),
        ("Mejora de procesos, Moquegua", "place_after_separator", ("Moquegua",)),
        ("Mejora de procesos en la ciudad de Arequipa", "place_after_word", ("Arequipa",)),
        ("Implementación de la norma ISO 9001:2015", "none", ()),
        ("Implementación de la norma ISO 9001-2015", "none", ()),
        ("Análisis del COVID 19", "none", ()),
        ("Mejora de procesos para la unión", "none", ()),
        ("Mejora de la calidad", "none", ()),
    ],
    ids=[
        "comma-place-year",
        "dash-places-year",
        "en-dash-and-period",
        "capitals-without-accent-and-range",
        "unaccented-place-and-parenthesis",
        "comma-year",
        "dash-year",
        "parenthesized-year",
        "temporal-word",
        "temporal-word-range",
        "other-word",
        "place-inside-a-word",
        "comma-place",
        "place-after-a-word",
        "norm-version-after-colon",
        "norm-version-after-dash",
        "short-number",
        "lower-case-place-name",
        "plain",
    ],
)
def test_classify_title_suffix_tells_the_trailing_place_and_year_patterns(
    title: str, pattern: str, places: tuple[str, ...]
) -> None:
    suffix = classify_title_suffix(title)

    assert (suffix.pattern, suffix.places) == (pattern, places)


def test_title_profile_counts_suffix_patterns_places_and_tokens() -> None:
    theses = [
        record(1, {"dc.title": "Mejora de procesos, Arequipa 2023"}),
        record(2, {"dc.title": "Mejora de procesos - Arequipa, Perú 2021"}),
        record(3, {"dc.title": "Mejora de procesos, 2025"}),
        record(4, {"dc.title": "Mejora de procesos en la ciudad de Arequipa"}),
        record(5, {"dc.title": "Mejora de la calidad"}),
        record(6, {"dc.title": None}),
    ]

    profile = title_profile(theses, count_tokens, token_limit=6)

    assert profile.present == 5
    assert profile.suffix_patterns == {
        "place_and_year": 2,
        "year_after_separator": 1,
        "year_after_temporal_word": 0,
        "year_after_other_word": 0,
        "place_after_separator": 0,
        "place_after_word": 1,
        "none": 1,
    }
    assert (profile.with_suffix, profile.with_year_suffix, profile.with_place_suffix) == (4, 3, 3)
    assert profile.share_with_suffix == pytest.approx(0.8)
    assert profile.places == {"Arequipa": 3, "Perú": 1}
    lengths = profile.lengths
    assert lengths is not None
    assert (lengths.tokens.min, lengths.tokens.max) == (6, 10)
    assert (lengths.above_limit, lengths.share_above_limit) == (3, pytest.approx(0.6))


# Keywords


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Gestión de Calidad", "gestión de calidad"),
        ("GESTIÓN DE CALIDAD", "gestión de calidad"),
        ("  Lean   Manufacturing. ", "lean manufacturing"),
        ("Producción", "producción"),
        ("Produccion", "produccion"),
        ("Gestio\N{COMBINING ACUTE ACCENT}n", "gestión"),
        ("AÑO", "año"),
    ],
    ids=["title-case", "capitals", "spacing-and-period", "accent", "no-accent", "decomposed", "ñ"],
)
def test_normalize_keyword_folds_case_and_spacing_but_keeps_accents(
    value: str, expected: str
) -> None:
    assert normalize_keyword(value) == expected


def test_keyword_profile_counts_keywords_per_thesis_and_ranks_them_by_theses() -> None:
    theses = [
        record(1, {"dc.subject": ["Calidad", "Lean", "calidad."]}),
        record(2, {"dc.subject": ["CALIDAD", "Seguridad"]}),
        record(3, {"dc.subject": ["Seguridad", "Minería", "Lean"]}, program="minas"),
        record(4, {"dc.subject": "Costos, presupuestos"}, program="minas"),
        record(5, program="minas"),
    ]

    profile = keyword_profile(theses, PROGRAMS, top=2, top_per_program=1)

    assert profile.values == 9
    assert profile.per_thesis_counts == {0: 1, 1: 1, 2: 2, 3: 1}
    assert profile.per_thesis is not None
    assert profile.per_thesis.median == 2
    assert profile.theses_without_keywords == 1
    assert (profile.distinct, profile.used_by_one_thesis) == (5, 2)
    assert profile.top == [
        KeywordCount(keyword="calidad", theses=2),
        KeywordCount(keyword="lean", theses=2),
    ]
    assert profile.top_by_program == {
        "sistemas": [KeywordCount(keyword="calidad", theses=2)],
        "industrial": [],
        "minas": [KeywordCount(keyword="costos, presupuestos", theses=1)],
    }
    assert (profile.values_with_comma, profile.values_with_trailing_period) == (1, 1)


# OCDE and language


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (f"{OCDE}2.11.04", "2.11.04"),
        ("2.02.01", "2.02.01"),
        (f"  {OCDE}1.02.00 ", "1.02.00"),
    ],
    ids=["uri", "bare-code", "padded"],
)
def test_ocde_code_is_the_fragment_of_the_ford_uri(value: str, expected: str) -> None:
    assert ocde_code(value) == expected


def test_ocde_profile_counts_codes_fields_and_codes_per_program() -> None:
    theses = [
        record(1, {"dc.subject.ocde": f"{OCDE}2.11.04"}, program="industrial"),
        record(2, {"dc.subject.ocde": f"{OCDE}2.11.04"}, program="industrial"),
        record(3, {"dc.subject.ocde": f"{OCDE}2.02.04"}),
        record(4, {"dc.subject.ocde": f"{OCDE}5.02.04"}),
        record(5, program="minas"),
    ]

    profile = ocde_profile(theses, PROGRAMS)

    assert (profile.theses_with_code, profile.theses_without_code, profile.distinct) == (4, 1, 3)
    assert profile.by_code == {"2.11.04": 2, "2.02.04": 1, "5.02.04": 1}
    assert list(profile.by_code) == ["2.11.04", "2.02.04", "5.02.04"]
    assert profile.by_field == {"2 Engineering and technology": 3, "5 Social sciences": 1}
    assert profile.by_program == {
        "sistemas": {"2.02.04": 1, "5.02.04": 1},
        "industrial": {"2.11.04": 2},
        "minas": {},
    }


def test_language_profile_counts_each_language_value() -> None:
    theses = [
        record(1, {"dc.language.iso": "spa"}),
        record(2, {"dc.language.iso": "spa"}),
        record(3, {"dc.language.iso": ["spa", "eng"]}),
        record(4),
    ]

    assert language_profile(theses) == {"spa": 3, "eng": 1, "(none)": 1}


# Advisors


def test_advisor_profile_counts_advisors_and_theses_per_advisor_without_naming_anyone() -> None:
    orcid = "0000-0000-0000-0001"
    theses = [
        record(
            1,
            {
                "dc.contributor.advisor": "Sintético Uno, Asesor",
                "renati.advisor.orcid": f"https://orcid.org/{orcid}",
            },
        ),
        record(
            2,
            {"dc.contributor.advisor": "SINTÉTICO  UNO ,ASESOR", "renati.advisor.orcid": orcid},
        ),
        # The same advisor, spelled without the accent: another name for the same ORCID.
        record(
            3, {"dc.contributor.advisor": "Sintetico Uno, Asesor", "renati.advisor.orcid": orcid}
        ),
        record(4, {"dc.contributor.advisor": "Sintético Dos, Asesora"}),
        record(
            5,
            {"dc.contributor.advisor": ["Sintético Dos, Asesora", "Sintético Tres, Asesor"]},
        ),
        record(6),
    ]

    profile = advisor_profile(theses)

    assert (
        profile.theses_with_advisor,
        profile.theses_without_advisor,
        profile.theses_with_several_advisors,
    ) == (5, 1, 1)
    assert profile.distinct_advisors == 4
    assert profile.theses_per_advisor is not None
    assert (profile.theses_per_advisor.median, profile.theses_per_advisor.max) == (1.5, 2)
    assert profile.advisors_by_theses == {"1": 2, "2-5": 2, "6-10": 0, "11-20": 0, "21+": 0}
    assert profile.share_of_theses_with_top_advisors == 1
    assert (profile.theses_with_advisor_orcid, profile.distinct_advisor_orcids) == (3, 1)
    assert profile.orcids_with_several_names == 1
    published = profile.model_dump_json().casefold()
    for fragment in ("sintético", "sintetico", "asesor", orcid):
        assert fragment not in published


# Rights


@pytest.mark.parametrize(
    ("rights", "expected"),
    [
        (OPEN, "open"),
        (EMBARGOED, "embargoed"),
        (RESTRICTED, "restricted"),
        (METADATA_ONLY, "metadata_only"),
        ([OPEN, EMBARGOED], "embargoed"),
        ("https://example.org/rights/other", "other"),
        (None, "(none)"),
    ],
    ids=["open", "embargoed", "restricted", "metadata-only", "mixed", "other", "missing"],
)
def test_access_class_follows_the_coar_access_right(rights: FieldValue, expected: str) -> None:
    assert access_class(record(1, {"dc.rights": rights}).metadata) == expected


def test_rights_profile_tells_lapsed_embargoes_from_pending_ones() -> None:
    theses = [
        record(1),
        record(2, {"dc.rights": EMBARGOED, "dc.date.embargoEnd": "2025-05-03"}),
        record(3, {"dc.rights": EMBARGOED, "dc.date.embargoEnd": "2027-03-18"}),
        # An embargo that ends on the snapshot day has not lapsed yet.
        record(4, {"dc.rights": EMBARGOED, "dc.date.embargoEnd": "2026-10-02"}),
        record(5, {"dc.rights": EMBARGOED}),
        # Open again after its embargo.
        record(6, {"dc.date.embargoEnd": "2023-10-12"}),
        record(7, {"dc.rights": RESTRICTED}),
    ]

    profile = rights_profile(theses, SNAPSHOT_DATE)

    assert profile.by_access == {
        "open": 2,
        "embargoed": 4,
        "restricted": 1,
        "metadata_only": 0,
        "other": 0,
        "(none)": 0,
    }
    assert (
        profile.embargoed_end_passed,
        profile.embargoed_end_pending,
        profile.embargoed_end_missing,
        profile.not_embargoed_with_end,
    ) == (1, 2, 1, 1)


# Program and year


def test_calendar_profile_counts_theses_per_program_and_year_and_flags_future_dates() -> None:
    theses = [
        record(1, {"dc.date.issued": "2021-03-01"}),
        record(2, {"dc.date.issued": "2021"}),
        record(3, {"dc.date.issued": "2023-07"}, program="minas"),
        # A bare year counts from its first day, so 2026 is not after the snapshot date.
        record(4, {"dc.date.issued": "2026"}, program="minas"),
        record(5, {"dc.date.issued": "2026-12"}, program="minas"),
        record(6, {"dc.date.issued": "2026-10-03"}, program="industrial"),
        record(7, {"dc.date.issued": "2026-10-02"}, program="industrial"),
        record(8, {"dc.date.issued": "s.f."}, program="industrial"),
        record(9, {"dc.date.issued": "2024-05-05"}, program="otro"),
    ]

    profile = calendar_profile(theses, PROGRAMS, SNAPSHOT_DATE)

    assert profile.by_program == {"sistemas": 2, "industrial": 3, "minas": 3, "otro": 1}
    assert list(profile.by_program) == ["sistemas", "industrial", "minas", "otro"]
    assert profile.by_year == {2021: 2, 2022: 0, 2023: 1, 2024: 1, 2025: 0, 2026: 4}
    assert profile.by_program_and_year == {
        "sistemas": {2021: 2, 2022: 0, 2023: 0, 2024: 0, 2025: 0, 2026: 0},
        "industrial": {2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0, 2026: 2},
        "minas": {2021: 0, 2022: 0, 2023: 1, 2024: 0, 2025: 0, 2026: 2},
        "otro": {2021: 0, 2022: 0, 2023: 0, 2024: 1, 2025: 0, 2026: 0},
    }
    assert (profile.future_dated, profile.unparsed_dates) == (2, 1)
    assert (profile.earliest, profile.latest) == (date(2021, 1, 1), date(2026, 12, 1))


def test_other_items_profile_counts_the_items_that_are_not_theses_by_type() -> None:
    others = [
        record(1, {"renati.type": SUFICIENCIA}),
        record(2, {"renati.type": SUFICIENCIA, "dc.date.issued": "2026-12-04"}, program="minas"),
        # The same type, spelled with spaces, as some items of the repository spell it.
        record(
            3, {"renati.type": f"{RENATI_TYPE}Trabajo de Suficiencia Profesional"}, program="minas"
        ),
        record(4, {"renati.type": None}, program="minas"),
    ]

    profile = other_items_profile(others, PROGRAMS, SNAPSHOT_DATE)

    assert profile.items == 4
    assert profile.by_type == {
        "trabajoDeSuficienciaProfesional": 2,
        "(none)": 1,
        "Trabajo de Suficiencia Profesional": 1,
    }
    assert profile.by_program == {"sistemas": 1, "industrial": 0, "minas": 3}
    assert profile.by_year == {2023: 3, 2024: 0, 2025: 0, 2026: 1}
    assert profile.future_dated == 1


# Duplicates


def test_duplicate_profile_finds_theses_that_repeat_both_a_title_and_an_abstract() -> None:
    theses = [
        record(1, {"dc.title": "Título repetido", "dc.description.abstract": "Resumen repetido."}),
        # The same thesis, with other letter case and spacing.
        record(
            2, {"dc.title": "TÍTULO  REPETIDO", "dc.description.abstract": "Resumen\nrepetido."}
        ),
        # Only the title, or only the abstract, repeats.
        record(3, {"dc.title": "Título repetido", "dc.description.abstract": "Otro resumen."}),
        record(4, {"dc.title": "Otro título", "dc.description.abstract": "Resumen repetido."}),
        # Accents keep two texts apart.
        record(5, {"dc.title": "Titulo repetido", "dc.description.abstract": "Resumen repetido."}),
        *(
            record(number, {"dc.title": "Título triple", "dc.description.abstract": "Triple."})
            for number in (6, 7, 8)
        ),
        # Without an abstract, a thesis is never counted as a duplicate.
        record(9, {"dc.title": "Título repetido"}),
        record(10, {"dc.title": "Título repetido"}),
    ]

    profile = duplicate_profile(theses)

    assert (profile.groups, profile.records, profile.extra_records) == (2, 5, 3)


# The whole profile, and what it may never publish


def corpus() -> list[SnapshotRecord]:
    """Two theses and one other item, carrying every field that the profile reads."""
    return [
        record(
            1,
            {
                "dc.title": "Diseño sintético de una línea de envasado, Arequipa 2023",
                "dc.description.abstract": (
                    "Este resumen sintético describe una línea de envasado ficticia.\n"
                    "Su segunda oración también es inventada para la prueba."
                ),
                "dc.subject": ["Envasado", "Calidad"],
                "dc.subject.ocde": f"{OCDE}2.11.04",
                "dc.language.iso": "spa",
                "dc.contributor.author": "Sintética Una, Autora",
                "dc.contributor.advisor": "Sintético Uno, Asesor",
                "renati.juror": ["Sintético Jurado, Primero", "Sintética Jurada, Segunda"],
                "renati.advisor.orcid": "0000-0000-0000-0001",
            },
        ),
        record(
            2,
            {
                "dc.title": "Modelo sintético de ventilación para una mina ficticia",
                "dc.description.abstract": "Otro resumen sintético, de una sola oración larga.",
                "dc.subject": ["Ventilación", "calidad"],
                "dc.subject.ocde": f"{OCDE}2.07.05",
                "dc.language.iso": "spa",
                "dc.contributor.author": ["Sintético Dos, Autor", "Sintética Tres, Autora"],
                "dc.contributor.advisor": "Sintética Dos, Asesora",
                "dc.date.issued": "2026-12-01",
            },
            program="minas",
        ),
        record(
            3,
            {
                "renati.type": SUFICIENCIA,
                "dc.title": "Informe sintético de suficiencia profesional",
                "dc.contributor.author": "Sintético Cuatro, Autor",
            },
            program="minas",
        ),
    ]


def test_profile_metadata_profiles_the_theses_and_reports_the_other_items_apart() -> None:
    profile = profile_metadata(
        corpus(),
        snapshot_id="20261002T224412Z",
        snapshot_date=SNAPSHOT_DATE,
        thesis_type="tesis",
        program_keys=PROGRAMS,
        count_tokens=count_tokens,
        token_limit=128,
    )

    assert (profile.snapshot_id, profile.snapshot_date) == ("20261002T224412Z", SNAPSHOT_DATE)
    assert (profile.items, profile.theses, profile.other_items.items) == (3, 2, 1)
    assert profile.calendar.by_program == {"sistemas": 1, "industrial": 0, "minas": 1}
    assert profile.calendar.future_dated == 1
    assert profile.other_items.by_type == {"trabajoDeSuficienciaProfesional": 1}
    assert profile.abstracts.present == 2
    assert profile.titles.suffix_patterns["place_and_year"] == 1
    assert profile.keywords.top[0] == KeywordCount(keyword="calidad", theses=2)
    assert profile.ocde.by_code == {"2.11.04": 1, "2.07.05": 1}
    assert profile.languages == {"spa": 2}
    assert profile.advisors.distinct_advisors == 2
    assert profile.rights.by_access["open"] == 2
    assert profile.duplicates.groups == 0


def test_the_published_profile_holds_no_title_abstract_name_or_orcid() -> None:
    records = corpus()
    profile = profile_metadata(
        records,
        snapshot_id="20261002T224412Z",
        snapshot_date=SNAPSHOT_DATE,
        thesis_type="tesis",
        program_keys=PROGRAMS,
        count_tokens=count_tokens,
        token_limit=128,
    )

    published = json.dumps(profile.model_dump(mode="json"), ensure_ascii=False)

    assert leaked_strings(published, sensitive_strings(records)) == []


def test_leaked_strings_finds_titles_sentences_names_and_orcids_in_any_case_and_spacing() -> None:
    sensitive = sensitive_strings(corpus())
    text = (
        "DISEÑO SINTÉTICO DE UNA LÍNEA DE ENVASADO, AREQUIPA 2023 | "
        "su segunda oración también es  inventada para la prueba. | "
        "Asesor Sintético Uno | Primero Sintético Jurado | 0000-0000-0000-0001"
    )

    leaked = leaked_strings(text, sensitive)

    assert "diseño sintético de una línea de envasado, arequipa 2023" in leaked
    assert "su segunda oración también es inventada para la prueba." in leaked
    assert "asesor sintético uno" in leaked
    assert "sintético uno" in leaked
    assert "primero sintético jurado" in leaked
    assert "0000-0000-0000-0001" in leaked


def test_leaked_strings_ignores_keywords_places_and_short_sentences() -> None:
    sensitive = sensitive_strings(corpus())

    assert leaked_strings("calidad · ventilación · Arequipa 2023 · 2.11.04 · spa", sensitive) == []
