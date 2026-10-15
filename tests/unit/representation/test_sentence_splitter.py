"""The rule sentence splitter of the chunker (D27, C1).

Every text here is synthetic. Each protection of a period has a case that splits wrongly
without it, and the two documented limits are pinned as they are.
"""

from itertools import pairwise

import pytest

from thematic_redundancy.representation.sentences import ABBREVIATIONS, sentence_spans


def sentences(text: str) -> list[str]:
    return [text[start:end] for start, end in sentence_spans(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "El estudio mide el tiempo. Se concluye que mejora.",
            ["El estudio mide el tiempo.", "Se concluye que mejora."],
        ),
        ("Texto sin punto final", ["Texto sin punto final"]),
        ("¿Es viable? Sí, lo es. ¡Funciona!", ["¿Es viable?", "Sí, lo es.", "¡Funciona!"]),
        (
            'Se dijo "basta." Luego siguió. Fin. "Otra frase" cierra.',
            ['Se dijo "basta."', "Luego siguió.", "Fin.", '"Otra frase" cierra.'],
        ),
        (
            "Se cerró la etapa (fase final.) Después se midió.",
            ["Se cerró la etapa (fase final.)", "Después se midió."],
        ),
        (
            "El proceso dura 3.5 horas. Se mide 12.75 kg en total.",
            ["El proceso dura 3.5 horas.", "Se mide 12.75 kg en total."],
        ),
        (
            "Se tomó una muestra de 30. Los resultados fueron buenos.",
            ["Se tomó una muestra de 30.", "Los resultados fueron buenos."],
        ),
        ("Sube el costo... Baja la venta.", ["Sube el costo...", "Baja la venta."]),
    ],
    ids=["two", "no-final-period", "openers", "quotes", "bracket", "decimals", "number", "dots"],
)
def test_a_sentence_ends_at_its_punctuation_before_a_capital(
    text: str, expected: list[str]
) -> None:
    assert sentences(text) == expected


def test_every_line_break_is_a_boundary_even_without_punctuation() -> None:
    text = "Primera línea sin punto\nsegunda línea en minúscula\n\nTercera, tras un párrafo."

    assert sentences(text) == [
        "Primera línea sin punto",
        "segunda línea en minúscula",
        "Tercera, tras un párrafo.",
    ]


def test_spans_are_trimmed_ordered_and_lose_no_text() -> None:
    text = "  Uno dos.   Tres cuatro!\n\n  Cinco seis?  \n"

    spans = sentence_spans(text)

    assert sentences(text) == ["Uno dos.", "Tres cuatro!", "Cinco seis?"]
    assert all(end <= next_start for (_, end), (next_start, _) in pairwise(spans))
    assert all(text[start:end] == text[start:end].strip() != "" for start, end in spans)
    assert "".join("".join(sentences(text)).split()) == "".join(text.split())


@pytest.mark.parametrize("text", ["", "   ", "\n\n", " \t \n "])
def test_a_blank_text_has_no_sentence(text: str) -> None:
    assert sentence_spans(text) == []


# Protections of a period


@pytest.mark.parametrize(
    "text",
    [
        "Trabaja en Minera Andina S.A.C. La empresa crece cada año.",
        "Según el Dr. Pérez, el método funciona.",
        "Se revisan costos, plazos, etc. Luego se ordenan.",
        "Se aplica la Ing. Vargas como guía.",
    ],
    ids=["s.a.c", "dr", "etc", "ing"],
)
def test_a_known_abbreviation_does_not_end_a_sentence(text: str) -> None:
    assert sentences(text) == [text]


@pytest.mark.parametrize("abbreviation", ["S.A.C.", "Dr.", "etc.", "Sr.", "Fig.", "E.I.R.L."])
def test_the_abbreviations_list_holds_the_usual_ones(abbreviation: str) -> None:
    assert abbreviation.lower().rstrip(".") in ABBREVIATIONS


@pytest.mark.parametrize(
    "text",
    [
        "Lo propuso J. Ramírez en su estudio.",
        "Se usan métodos ágiles, e.g. Scrum en el equipo.",
    ],
    ids=["initial", "dotted-abbreviation"],
)
def test_an_initial_or_a_dotted_abbreviation_does_not_end_a_sentence(text: str) -> None:
    assert sentences(text) == [text]


def test_a_numbered_item_keeps_its_marker_after_a_colon() -> None:
    text = "Objetivos: 1. Determinar el costo. 2. Evaluar la mejora."

    assert sentences(text) == ["Objetivos: 1. Determinar el costo.", "2. Evaluar la mejora."]


def test_a_lettered_item_keeps_its_marker() -> None:
    text = "Se busca: a) Medir el costo. b) Evaluar la mejora."

    assert sentences(text) == ["Se busca: a) Medir el costo.", "b) Evaluar la mejora."]


def test_an_item_that_opens_a_line_keeps_its_marker() -> None:
    text = "1. Determinar el costo.\n2. Evaluar la mejora.\nIV. Proponer un plan."

    assert sentences(text) == [
        "1. Determinar el costo.",
        "2. Evaluar la mejora.",
        "IV. Proponer un plan.",
    ]


def test_an_item_after_a_semicolon_keeps_its_marker() -> None:
    text = "Se plantea; 2. Diseñar el plan. Luego se valida."

    assert sentences(text) == ["Se plantea; 2. Diseñar el plan.", "Luego se valida."]


# Documented limits


def test_a_lone_letter_before_a_period_reads_as_an_initial() -> None:
    # Known limit: "fase a." looks like an initial, so the sentence goes on.
    text = "Se evaluó la fase a. Luego se midió el resultado."

    assert sentences(text) == [text]


def test_a_period_before_a_lower_case_word_never_ends_a_sentence() -> None:
    # Known limit: such a period may end a sentence, but it is never split.
    text = "El valor fue alto. luego bajó con el tiempo."

    assert sentences(text) == [text]
