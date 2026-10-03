"""Light cleaner: the embedding input, which keeps meaning, casing, accents and punctuation.

Every text here is synthetic Spanish written for the test.
"""

import subprocess
import sys

import pytest

from thematic_redundancy.preprocessing.light_cleaner import clean_light, normalize_text

# Unicode


def test_clean_light_composes_decomposed_accents_into_nfc() -> None:
    decomposed = "La investigacio\u0301n del an\u0303o en el Peru\u0301."

    assert clean_light(decomposed) == "La investigación del año en el Perú."


def test_clean_light_keeps_casing_accents_eñe_and_punctuation() -> None:
    text = "¿Cómo MEJORAR la Producción de Añil? ¡Así! (según la NTP 1.5; 25 %): «sí»."

    assert clean_light(text) == text


# Hard-wrapped lines


def test_clean_light_joins_a_line_break_inside_a_sentence_with_one_space() -> None:
    text = "El sistema propuesto mejora\nla productividad de la planta\nde envasado."

    assert (
        clean_light(text)
        == "El sistema propuesto mejora la productividad de la planta de envasado."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Se diseñó un sistema.\nLuego se validó.", "Se diseñó un sistema.\nLuego se validó."),
        ("¿Funciona el modelo?\nSí, funciona.", "¿Funciona el modelo?\nSí, funciona."),
        ("Se logró la meta!\nÉse fue el mejor año.", "Se logró la meta!\nÉse fue el mejor año."),
        ("Quedan dudas…\nEl estudio sigue.", "Quedan dudas…\nEl estudio sigue."),
        ('Se llamó "plan".\n¿Por qué?', 'Se llamó "plan".\n¿Por qué?'),
        (
            "Ver la norma (anexo).\n«Calidad» es clave.",
            "Ver la norma (anexo).\n«Calidad» es clave.",
        ),
        ("Se resume así.\n- Primer punto.", "Se resume así.\n- Primer punto."),
        ("Se resume así.\n• Primer punto.", "Se resume así.\n• Primer punto."),
    ],
    ids=["period", "question", "exclamation", "ellipsis", "quote", "bracket", "dash", "bullet"],
)
def test_clean_light_keeps_a_line_break_after_sentence_final_punctuation(
    text: str, expected: str
) -> None:
    assert clean_light(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "La empresa Minera Andina S.A.C.\nque opera en la sierra.",
            "La empresa Minera Andina S.A.C. que opera en la sierra.",
        ),
        ("El costo bajó a S/.\n1,500 por mes.", "El costo bajó a S/. 1,500 por mes."),
        ("Se mejoró el flujo.\nluego se midió.", "Se mejoró el flujo. luego se midió."),
    ],
    ids=["abbreviation", "currency-before-amount", "lowercase-start"],
)
def test_clean_light_joins_a_break_after_a_period_when_no_sentence_starts_after_it(
    text: str, expected: str
) -> None:
    assert clean_light(text) == expected


def test_clean_light_keeps_blank_line_paragraph_breaks_as_one_blank_line() -> None:
    text = "Primer párrafo con una idea\nque sigue aquí.\n\n\n  \nSegundo párrafo\n\nTercero."

    assert clean_light(text) == (
        "Primer párrafo con una idea que sigue aquí.\n\nSegundo párrafo\n\nTercero."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("línea uno\r\nlínea dos\rlínea tres", "línea uno línea dos línea tres"),
        ("línea uno\u2028línea dos\x85línea tres", "línea uno línea dos línea tres"),
        ("línea uno\x0blínea dos\x0clínea tres", "línea uno línea dos línea tres"),
        ("Primer párrafo.\u2029Segundo párrafo.", "Primer párrafo.\n\nSegundo párrafo."),
    ],
    ids=["crlf-and-cr", "unicode-line-separators", "vertical-tab-and-form-feed", "paragraph"],
)
def test_clean_light_reads_every_line_separator_as_a_line_break(text: str, expected: str) -> None:
    assert clean_light(text) == expected


# Hyphenation at line ends


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("La investi-\ngación aplicada.", "La investigación aplicada."),
        ("La investi-  \n   gación aplicada.", "La investigación aplicada."),
        ("La investi\u2010\ngación aplicada.", "La investigación aplicada."),
        ("La investi\u00ad\ngación aplicada.", "La investigación aplicada."),
        ("Una compa-\nñía minera.", "Una compañía minera."),
    ],
    ids=["hyphen", "spaces-around-break", "unicode-hyphen", "soft-hyphen", "eñe"],
)
def test_clean_light_rejoins_a_word_split_by_a_hyphen_at_a_line_end(
    text: str, expected: str
) -> None:
    assert clean_light(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ingeniería Mecánica-\nEléctrica del sur.", "Ingeniería Mecánica-Eléctrica del sur."),
        ("El efecto del COVID-\n19 en la planta.", "El efecto del COVID-19 en la planta."),
        ("El periodo 2020-\n2021 fue atípico.", "El periodo 2020-2021 fue atípico."),
    ],
    ids=["capital-after", "digit-after", "digits-around"],
)
def test_clean_light_keeps_the_hyphen_of_a_compound_split_at_a_line_end(
    text: str, expected: str
) -> None:
    assert clean_light(text) == expected


def test_clean_light_treats_a_dash_after_a_space_as_punctuation_not_hyphenation() -> None:
    assert clean_light("Un dato -\nque sigue.") == "Un dato - que sigue."


def test_clean_light_keeps_hyphenated_compounds_inside_a_line() -> None:
    text = "La relación costo-beneficio de la línea Mecánica-Eléctrica."

    assert clean_light(text) == text


def test_clean_light_removes_soft_hyphens_inside_a_line() -> None:
    assert clean_light("La investi\u00adgación y la pro\u00adducción.") == (
        "La investigación y la producción."
    )


# Whitespace and invisible characters


def test_clean_light_collapses_spaces_tabs_and_no_break_spaces_into_one_space() -> None:
    text = "  El\tsistema \u00a0 mejora\u2009la\u202fplanta   piloto.  "

    assert clean_light(text) == "El sistema mejora la planta piloto."


def test_clean_light_removes_invisible_format_characters() -> None:
    text = "\ufeffLa pro\u200bducción\u2060 y la calidad."

    assert clean_light(text) == "La producción y la calidad."


def test_clean_light_turns_control_characters_into_spaces() -> None:
    assert clean_light("La planta\x00piloto\x07 opera.") == "La planta piloto opera."


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n", "\u00ad", "\u200b\n\ufeff"])
def test_clean_light_returns_an_empty_string_for_text_without_content(text: str) -> None:
    assert clean_light(text) == ""


# Quotes and dashes


def test_clean_light_straightens_curly_quotes() -> None:
    text = (
        "La \u201ccalidad\u201d y la \u2018gestión\u2019 de O\u2019Brien, "
        "\u201enorma\u201f y \u201aregla\u201b."
    )

    assert clean_light(text) == "La \"calidad\" y la 'gestión' de O'Brien, \"norma\" y 'regla'."


def test_clean_light_turns_hyphen_look_alikes_into_the_ascii_hyphen() -> None:
    text = "Un análisis físico\u2010químico, no\u2011lineal, 555\u20121234 y \u22125 °C."

    assert clean_light(text) == "Un análisis físico-químico, no-lineal, 555-1234 y -5 °C."


def test_clean_light_keeps_guillemets_primes_and_en_and_em_dashes() -> None:
    text = "La «calidad» \u2013 sin cambios \u2014 en 2019\u20132020, a 5\u2032 y 3\u2033."

    assert clean_light(text) == text


# Idempotency

IDEMPOTENCY_CASES = [
    "El sistema propuesto mejora\nla productividad.\nLuego se validó.\n\nOtro párrafo.",
    "La investi-\ngación y la Mecánica-\nEléctrica del COVID-\n19.",
    "Termina con guion-\n\nY sigue otro párrafo con una investi\u00ad\n\ngación.",
    "Un dato \u2013\nque sigue \u2014 y otro.\n¿Pregunta?\n«Cita».\n- Punto uno.\n• Punto dos.",
    (
        "  Espacios\t\u00a0raros \u200b y \u201ccomillas\u201d "
        "\u2018simples\u2019 \u2010 guiones \u2212 signos.  "
    ),
    "S.A.C.\nque opera.\r\nOtra línea\rmás\u2028y\u2029fin.",
    "La investigacio\u0301n del an\u0303o.\n\u0301Marca suelta.",
]


@pytest.mark.parametrize("text", IDEMPOTENCY_CASES)
def test_clean_light_is_idempotent(text: str) -> None:
    once = clean_light(text)

    assert clean_light(once) == once


# Dependencies


def test_the_light_cleaner_and_the_title_suffix_module_do_not_import_spacy() -> None:
    code = (
        "import sys\n"
        "import thematic_redundancy.preprocessing.light_cleaner\n"
        "import thematic_redundancy.preprocessing.title_suffix\n"
        "print('spacy' in sys.modules)\n"
    )

    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert run.stdout.strip() == "False"


# Flat normalization


def test_normalize_text_flattens_every_whitespace_run_into_one_space_in_nfc() -> None:
    text = "  La investigacio\u0301n\n\ny la\tcalidad\u00a0total  "

    assert normalize_text(text) == "La investigación y la calidad total"
