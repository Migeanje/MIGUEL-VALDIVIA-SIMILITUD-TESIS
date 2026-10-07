"""Behavior of the objectives locator on synthetic page texts.

Every text here is made up. Pages are laid out the way the text extraction stores them:
one block per line of the PDF, with a blank line between blocks, so a paragraph that wraps
over several lines arrives as several blocks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import get_args
from uuid import UUID

import pytest
from pydantic import ValidationError

from thematic_redundancy.corpus.ficha import ObjectivesStatus
from thematic_redundancy.extraction.objectives import (
    OBJECTIVES_FLAGS,
    LocatedObjectives,
    LocatorStatus,
    ObjectivesRecord,
    locate_objectives,
)
from thematic_redundancy.shared.config import ObjectivesConfig, VerificationSampleConfig

SETTINGS = ObjectivesConfig(
    general_max_chars=1200,
    specific_max_chars=4000,
    min_chars=40,
    verification_sample=VerificationSampleConfig(
        per_program=12, seed=1, workbook_dir=Path("data/labels")
    ),
)

GENERAL = "Diseñar una propuesta que permita mejorar la gestión de distribución de la empresa."
GENERAL_LINES = (
    "Diseñar una propuesta que permita mejorar la gestión de",
    "distribución de la empresa.",
)
SPECIFIC_LINES = (
    "• Diagnosticar la situación actual de los procesos de la",
    "empresa de transporte.",
    "• Proponer mejoras en la gestión de la flota.",
)
SPECIFIC = (
    "• Diagnosticar la situación actual de los procesos de la empresa de transporte.\n"
    "• Proponer mejoras en la gestión de la flota."
)
FILLER = (
    "El presente capítulo describe el contexto de la investigación realizada.",
    "Se presentan los conceptos que sustentan el desarrollo del trabajo.",
)


@dataclass(frozen=True)
class Page:
    page_index: int
    text: str
    source: str = "text_layer"


def blocks(*lines: str) -> str:
    """Join ``lines`` the way the extraction stores them: one block per line."""
    return "\n\n".join(lines)


def pages(texts: dict[int, str], *, ocr: frozenset[int] = frozenset()) -> list[Page]:
    """Return one page per entry, keyed by page index, in index order."""
    return [
        Page(index, text, "ocr" if index in ocr else "text_layer")
        for index, text in sorted(texts.items())
    ]


def chapter_one(*objective_lines: str) -> str:
    return blocks("CAPÍTULO I", "PLANTEAMIENTO DEL PROBLEMA", *FILLER, *objective_lines)


def standard_objectives(heading: str = "1.3.1. Objetivo general") -> tuple[str, ...]:
    return (
        "1.3. Objetivos",
        heading,
        *GENERAL_LINES,
        "1.3.2. Objetivos específicos",
        *SPECIFIC_LINES,
        "1.4. Justificación",
        "La investigación se justifica por su aporte práctico.",
    )


def locate(texts: dict[int, str], **options: frozenset[int]) -> LocatedObjectives:
    return locate_objectives(pages(texts, **options), SETTINGS)


# Statuses


def test_the_locator_statuses_are_objectives_statuses_of_the_ficha() -> None:
    assert set(get_args(LocatorStatus)) == {"extracted", "not_found"}
    assert set(get_args(LocatorStatus)) <= set(get_args(ObjectivesStatus))


def test_a_document_without_objectives_is_not_found() -> None:
    found = locate({0: blocks(*FILLER), 1: blocks("Conclusiones", *FILLER)})

    assert found == LocatedObjectives(status="not_found")


def test_a_document_without_pages_is_not_found() -> None:
    assert locate_objectives([], SETTINGS) == LocatedObjectives(status="not_found")


# The standard layout


def test_a_numbered_general_heading_gives_both_objectives_in_their_original_text() -> None:
    found = locate({20: chapter_one(*standard_objectives())})

    assert found == LocatedObjectives(
        status="extracted",
        general=GENERAL,
        specific=SPECIFIC,
        page_start=20,
        page_end=20,
        pattern="objetivo_general",
        flags=(),
    )


def test_lines_within_one_block_are_joined_like_separate_blocks() -> None:
    text = "\n".join(standard_objectives())

    found = locate({20: text})

    assert (found.general, found.specific) == (GENERAL, SPECIFIC)


@pytest.mark.parametrize(
    ("heading", "pattern"),
    [
        ("1.3.1. Objetivo general", "objetivo_general"),
        ("1.3.1 OBJETIVO GENERAL", "objetivo_general"),
        ("1.3.1. Objetivo General.", "objetivo_general"),
        ("Objetivo general:", "objetivo_general"),
        ("Objetivos generales", "objetivo_general"),
        ("1.3.1. Objetivo general de la investigación", "objetivo_general"),
        ("a) Objetivo general", "objetivo_general"),
        ("II. Objetivo general", "objetivo_general"),
        ("• Objetivo general", "objetivo_general"),
        ("1.3.1. Objetivo principal", "objetivo_principal"),
        ("OBJETIVO PRINCIPAL:", "objetivo_principal"),
    ],
)
def test_general_heading_variants_are_recognized(heading: str, pattern: str) -> None:
    found = locate({20: chapter_one(*standard_objectives(heading))})

    assert (found.general, found.pattern) == (GENERAL, pattern)


def test_numbering_on_its_own_line_is_recognized_and_stops_the_general_objective() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1",
                "Objetivo general",
                *GENERAL_LINES,
                "1.3.2",
                "Objetivo específico",
                *SPECIFIC_LINES,
                "1.4",
                "Justificación",
            )
        }
    )

    assert (found.general, found.specific) == (GENERAL, SPECIFIC)


def test_an_inline_general_objective_follows_the_heading_on_the_same_line() -> None:
    found = locate(
        {
            20: blocks(
                "Objetivo general: Evaluar la eficiencia energética del sistema de",
                "bombeo de la planta.",
                "Objetivos específicos: Medir el consumo de energía.",
                "Hipótesis",
            )
        }
    )

    assert found.general == "Evaluar la eficiencia energética del sistema de bombeo de la planta."
    assert found.specific == "Medir el consumo de energía."


def test_a_heading_that_ends_in_a_page_number_is_a_contents_line() -> None:
    found = locate({3: blocks("1.3.1 Objetivo general 21", "Determinar algo de prueba.")})

    assert found.status == "not_found"


def test_an_introduction_ending_in_a_colon_is_left_out_of_the_general_objective() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1. Objetivo general",
                "El objetivo general de la presente investigación es el siguiente:",
                *GENERAL_LINES,
                "1.3.2. Objetivos específicos",
                *SPECIFIC_LINES,
            )
        }
    )

    assert found.general == GENERAL


# Front matter and contents pages


def test_a_contents_page_with_dotted_leaders_is_skipped() -> None:
    contents = blocks(
        "ÍNDICE",
        "CAPÍTULO I ................................................ 18",
        "1.3.1. Objetivo general ................................. 21",
        "1.3.2. Objetivos específicos ............................ 21",
        "1.4. Justificación ....................................... 22",
    )

    found = locate({2: contents, 20: chapter_one(*standard_objectives())})

    assert (found.page_start, found.general) == (20, GENERAL)


def test_a_contents_page_with_page_numbers_on_their_own_lines_is_skipped() -> None:
    contents = blocks(
        "Planteamiento del problema",
        "18",
        "Descripción de la realidad problemática",
        "18",
        "Objetivo general",
        "Determinar el número de páginas del documento de prueba",
        "21",
        "Objetivos específicos",
        "21",
        "Justificación",
        "22",
    )

    found = locate({2: contents, 20: chapter_one(*standard_objectives())})

    assert (found.page_start, found.general) == (20, GENERAL)


def test_a_document_whose_only_heading_is_in_the_contents_is_not_found() -> None:
    contents = blocks("CONTENIDO", "1.3.1. Objetivo general ......... 21", *FILLER)

    assert locate({2: contents, 20: blocks(*FILLER)}).status == "not_found"


def test_a_listed_heading_without_text_gives_way_to_the_next_one() -> None:
    listing = blocks(
        "1.3.1. Objetivo general", "1.3.2. Objetivos específicos", "1.4. Justificación"
    )

    found = locate({3: listing, 20: chapter_one(*standard_objectives())})

    assert (found.page_start, found.general) == (20, GENERAL)


@pytest.mark.parametrize("title", ["RESUMEN", "Abstract", "Resumen ejecutivo"])
def test_an_abstract_that_names_the_general_objective_gives_way_to_the_body(title: str) -> None:
    abstract = blocks(
        title,
        "Objetivo general: Analizar el tema de prueba descrito en la tesis de ejemplo.",
        "Palabras clave: prueba, ejemplo.",
    )

    found = locate({4: abstract, 20: chapter_one(*standard_objectives())})

    assert (found.page_start, found.general, found.flags) == (20, GENERAL, ())


def test_an_objective_found_only_in_the_abstract_is_flagged() -> None:
    abstract = blocks(
        "RESUMEN",
        "Objetivo general: Determinar el efecto de la temperatura en el rendimiento",
        "del proceso de flotación.",
        "Palabras clave: flotación, temperatura.",
    )

    found = locate({4: abstract, 20: blocks(*FILLER)})

    assert found.status == "extracted"
    assert found.general == (
        "Determinar el efecto de la temperatura en el rendimiento del proceso de flotación."
    )
    assert found.page_start == 4
    assert "abstract_only" in found.flags


@pytest.mark.parametrize("title", ["DEDICATORIA", "Agradecimientos"])
def test_dedication_and_acknowledgement_pages_are_never_used(title: str) -> None:
    page = blocks(
        title, "Objetivo principal:", "Agradecer a quienes apoyaron este trabajo de prueba."
    )

    assert locate({3: page, 20: blocks(*FILLER)}).status == "not_found"


def test_the_first_occurrence_in_the_body_wins_over_a_later_one() -> None:
    annex = blocks(
        "MATRIZ DE CONSISTENCIA",
        "Objetivo general",
        "Otra redacción del objetivo que aparece en los anexos del documento.",
    )

    found = locate({20: chapter_one(*standard_objectives()), 90: annex})

    assert (found.page_start, found.general) == (20, GENERAL)


# Page breaks


def test_a_general_objective_continues_across_a_page_break() -> None:
    found = locate(
        {
            20: blocks(
                *FILLER,
                "1.4.1 Objetivo general",
                "Determinar la influencia de la granulometría del mineral en la",
            ),
            21: blocks(
                "recuperación de cobre en la planta concentradora.",
                "1.4.2 Objetivos específicos",
                *SPECIFIC_LINES,
                "1.5 Hipótesis",
            ),
        }
    )

    assert found.general == (
        "Determinar la influencia de la granulometría del mineral en la recuperación de cobre "
        "en la planta concentradora."
    )
    assert (found.page_start, found.page_end, found.specific) == (20, 21, SPECIFIC)


def test_the_page_range_ends_where_the_specific_objectives_end() -> None:
    found = locate(
        {
            20: blocks("1.3.1. Objetivo general", *GENERAL_LINES, "1.3.2. Objetivos específicos"),
            21: blocks(*SPECIFIC_LINES),
            22: blocks("1.4. Justificación", *FILLER),
        }
    )

    assert (found.page_start, found.page_end, found.specific) == (20, 21, SPECIFIC)


# Objectives blocks without a general heading


def test_general_and_specific_subheadings_under_an_objectives_block_are_recognized() -> None:
    found = locate(
        {
            18: blocks(
                "1.2. Objetivos de la Investigación",
                "1.2.1.  General",
                *GENERAL_LINES,
                "1.2.2.  Específicos",
                *SPECIFIC_LINES,
                "1.3. Hipótesis",
            )
        }
    )

    assert (found.general, found.specific) == (GENERAL, SPECIFIC)
    assert (found.pattern, found.flags) == ("general_subheading", ())


def test_an_introduction_naming_the_general_objective_works_as_a_heading() -> None:
    found = locate(
        {
            17: blocks(
                "1.3. Objetivos",
                "El objetivo general es el siguiente:",
                "• Reducir los costos de perforación mediante el cambio del modelo",
                "de broca en la mina.",
                "Los objetivos específicos corresponden a:",
                "• Comparar el rendimiento de ambos modelos de broca.",
                "• Evaluar el costo total de la operación.",
                "1.4. Justificación",
            )
        }
    )

    assert found.general == (
        "Reducir los costos de perforación mediante el cambio del modelo de broca en la mina."
    )
    assert found.specific == (
        "• Comparar el rendimiento de ambos modelos de broca.\n"
        "• Evaluar el costo total de la operación."
    )
    assert found.pattern == "objetivo_general_intro"


@pytest.mark.parametrize(
    "heading",
    ["1.3. Objetivos", "OBJETIVOS", "1.3. Objetivos de la investigación", "Objetivos del estudio"],
)
def test_an_objectives_block_without_a_general_heading_falls_back_to_its_first_sentence(
    heading: str,
) -> None:
    found = locate(
        {
            18: blocks(
                heading,
                "Diseñar un sistema de control automático para el horno de secado",
                "de la planta.",
                "Además se busca reducir el consumo de energía.",
                "1.4. Justificación",
            )
        }
    )

    assert found.general == (
        "Diseñar un sistema de control automático para el horno de secado de la planta."
    )
    assert found.pattern == "objetivos_block"
    assert found.flags == ("specific_missing", "block_fallback")


def test_a_general_heading_anywhere_wins_over_an_earlier_objectives_block() -> None:
    found = locate(
        {
            18: blocks("1.3. Objetivos", "Los objetivos se presentan en la siguiente sección."),
            19: blocks(*standard_objectives()[1:]),
        }
    )

    assert (found.page_start, found.pattern, found.general) == (19, "objetivo_general", GENERAL)


# Specific objectives


def test_numbered_and_lettered_items_are_kept_one_per_line() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1 Objetivo general",
                *GENERAL_LINES,
                "1.3.2 Objetivos específicos",
                "1. Obtener un modelo matemático que describa el comportamiento del",
                "proceso.",
                "2. Mantener una relación constante entre los insumos",
                "a) Validar el modelo con datos de planta",
                "OE3: Comparar los resultados obtenidos.",
                "1.4 Justificación",
            )
        }
    )

    assert found.specific == (
        "1. Obtener un modelo matemático que describa el comportamiento del proceso.\n"
        "2. Mantener una relación constante entre los insumos\n"
        "a) Validar el modelo con datos de planta\n"
        "OE3: Comparar los resultados obtenidos."
    )


def test_lines_that_wrap_with_a_number_or_a_lower_case_word_do_not_stop_the_items() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1 Objetivo general",
                *GENERAL_LINES,
                "1.3.2 Objetivos específicos",
                "• Reducir el consumo de agua a",
                "2.5 metros cúbicos por hora en la planta.",
                "• Resolver el",
                "problema de abastecimiento.",
                "1.4 Justificación",
            )
        }
    )

    assert found.specific == (
        "• Reducir el consumo de agua a 2.5 metros cúbicos por hora en la planta.\n"
        "• Resolver el problema de abastecimiento."
    )


@pytest.mark.parametrize(
    "stop",
    [
        "Justificación",
        "1.4. Justificación e importancia",
        "1.5 Hipótesis",
        "Alcance de la investigación",
        "Delimitación",
        "Limitaciones del estudio",
        "Variables",
        "Metodología",
        "MARCO TEÓRICO",
        "Antecedentes",
        "CAPÍTULO II",
        "II. MARCO TEÓRICO",
        "1.4",
        "1.4. Descripción de la empresa",
    ],
)
def test_the_specific_objectives_end_at_the_next_section(stop: str) -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1. Objetivo general",
                *GENERAL_LINES,
                "1.3.2. Objetivos específicos",
                *SPECIFIC_LINES,
                stop,
                "Texto de la sección siguiente que no pertenece a los objetivos.",
            )
        }
    )

    assert found.specific == SPECIFIC


@pytest.mark.parametrize("stop", ["Justificación", "1.3.2. Alcance", "CAPÍTULO II"])
def test_a_general_objective_without_a_final_period_ends_at_the_next_section(stop: str) -> None:
    found = locate(
        {
            20: blocks(
                "Objetivo general",
                "Diseñar un sistema de riego automatizado para el fundo agrícola",
                stop,
                "Texto de la sección siguiente que no pertenece al objetivo.",
            )
        }
    )

    assert found.general == "Diseñar un sistema de riego automatizado para el fundo agrícola"
    assert "specific_missing" in found.flags


def test_specific_objectives_after_an_introductory_paragraph_are_found() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1. Objetivo general",
                *GENERAL_LINES,
                "Para alcanzar el objetivo general se plantean los siguientes objetivos",
                "específicos:",
                *SPECIFIC_LINES,
                "1.4. Justificación",
            )
        }
    )

    assert found.specific == SPECIFIC


def test_missing_specific_objectives_are_flagged() -> None:
    found = locate({20: blocks("1.3.1. Objetivo general", *GENERAL_LINES, "1.4. Justificación")})

    assert (found.general, found.specific) == (GENERAL, None)
    assert found.flags == ("specific_missing",)


# Noise and bounds


def test_ocr_noise_is_tolerated_and_the_page_is_flagged() -> None:
    found = locate(
        {
            20: "1.3.1  0BJETIVO   GENERAL :\n Implementar un sistema\x02  de  monitoreo\x0c\n"
            "remoto para la planta de tratamiento.\n1.3.2 0bjetivos especificos\n"
            "- Medir la calidad del agua en línea.\n1.4 Justificación"
        },
        ocr=frozenset({20}),
    )

    assert (
        found.general == "Implementar un sistema de monitoreo remoto para la planta de tratamiento."
    )
    assert found.specific == "- Medir la calidad del agua en línea."
    assert found.flags == ("ocr_page",)


def test_private_use_bullets_become_visible_bullets() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1. Objetivo general",
                " " + GENERAL,
                "1.3.2. Objetivos específicos",
                " Diagnosticar la situación actual de los procesos.",
            )
        }
    )

    assert found.general == GENERAL
    assert found.specific == "• Diagnosticar la situación actual de los procesos."


def test_a_short_general_objective_is_flagged() -> None:
    found = locate({20: blocks("Objetivo general", "Mejorar la planta.", "Justificación")})

    assert found.general == "Mejorar la planta."
    assert found.flags == ("general_too_short", "specific_missing")


def test_a_heading_followed_by_numbers_only_is_no_objective() -> None:
    found = locate({20: blocks("Objetivo general", "21", "Objetivos específicos", "22")})

    assert found.status == "not_found"


def test_long_objectives_are_cut_at_their_caps_and_flagged() -> None:
    settings = SETTINGS.model_copy(update={"general_max_chars": 80, "specific_max_chars": 60})
    words = " ".join(["optimizar el proceso productivo de la planta"] * 6)
    texts = {
        20: blocks(
            "1.3.1. Objetivo general",
            f"Determinar cómo {words}",
            "1.3.2. Objetivos específicos",
            f"• Evaluar cómo {words}",
            "1.4. Justificación",
        )
    }

    found = locate_objectives(pages(texts), settings)

    assert found.general is not None and found.specific is not None
    assert len(found.general) <= 80 and found.general.startswith("Determinar cómo optimizar")
    assert len(found.specific) <= 60 and found.specific.startswith("• Evaluar cómo optimizar")
    assert found.flags == ("general_too_long", "specific_too_long")


def test_flags_come_from_the_declared_vocabulary_in_a_fixed_order() -> None:
    assert OBJECTIVES_FLAGS == (
        "general_too_short",
        "general_too_long",
        "specific_too_long",
        "specific_missing",
        "abstract_only",
        "block_fallback",
        "ocr_page",
    )


# Rules added while tuning on the corpus (round 2)


def test_an_introduction_below_the_specific_heading_is_left_out_of_the_items() -> None:
    found = locate(
        {
            20: blocks(
                "1.3.1. Objetivo general",
                *GENERAL_LINES,
                "2.2.Objetivos específicos",
                "Los objetivos específicos a cumplir serían:",
                *SPECIFIC_LINES,
                "1.4. Justificación",
            )
        }
    )

    assert found.specific == SPECIFIC


@pytest.mark.parametrize(
    "heading_lines",
    [("1.7.2. Objetivos Específicos",), ("1.7.2.", "Objetivos Específicos")],
    ids=["numbered-heading", "numbering-on-its-own-line"],
)
def test_items_numbered_below_the_specific_heading_are_items(
    heading_lines: tuple[str, ...],
) -> None:
    found = locate(
        {
            20: blocks(
                "1.7.1. Objetivo general",
                *GENERAL_LINES,
                *heading_lines,
                "1.7.2.1. Modelar el flujo de potencia de la red de",
                "distribución.",
                "1.7.2.2. Evaluar las pérdidas técnicas.",
                "1.8. Justificación",
            )
        }
    )

    assert found.specific == (
        "1.7.2.1. Modelar el flujo de potencia de la red de distribución.\n"
        "1.7.2.2. Evaluar las pérdidas técnicas."
    )


@pytest.mark.parametrize(
    "heading",
    [
        "1.4.2 Objetivos secundarios",
        "1.4.2. OBJETIVO SECUNDARIO",
        "Y como objetivos secundarios se establece:",
        "1.1.2. Objeticos Específicos",
        "Objetivos Espec´ıficos",
    ],
    ids=["secundarios", "secundario", "introduction", "typo", "latex-accent"],
)
def test_specific_heading_synonyms_typos_and_split_accents_are_recognized(heading: str) -> None:
    found = locate(
        {20: blocks("1.4.1 Objetivo principal", *GENERAL_LINES, heading, *SPECIFIC_LINES)}
    )

    assert found.specific == SPECIFIC


def test_accents_split_from_their_letter_are_joined_back() -> None:
    found = locate(
        {
            20: blocks(
                "1.2.1. Objetivo general",
                "Analizar la situaci´on actual de las personas con discapacidad en la",
                "regi´on y dise˜nar una aplicaci´on m´ovil de apoyo.",
            )
        }
    )

    assert found.general == (
        "Analizar la situación actual de las personas con discapacidad en la región y diseñar "
        "una aplicación móvil de apoyo."
    )


@pytest.mark.parametrize(
    "block",
    [
        "Objetivo del Proyecto",
        "Objetivos de la Investigación (Solo va el título)",
        "1.2. Objetivos específicos",
    ],
    ids=["singular-with-qualifier", "template-remark", "mislabeled-specific-block"],
)
def test_a_general_subheading_below_other_objectives_headings_is_recognized(block: str) -> None:
    found = locate(
        {
            16: blocks(
                block,
                "1.2.1. General.",
                "− Diseñar una propuesta que permita mejorar la gestión de",
                "distribución de la empresa.",
                "1.2.2. Específicos",
                "− Diagnosticar la situación actual de los procesos.",
                "1.3. Hipótesis",
            )
        }
    )

    assert (found.general, found.pattern) == (GENERAL, "general_subheading")
    assert found.specific == "− Diagnosticar la situación actual de los procesos."


# The record of the objectives table


EXTRACTED_RECORD: dict[str, object] = {
    "item_uuid": UUID("00000000-0000-4000-8000-000000000001"),
    "handle": "20.500.00000/1",
    "program": "sistemas",
    "status": "extracted",
    "objective_general": GENERAL,
    "objectives_specific": SPECIFIC,
    "page_start": 20,
    "page_end": 21,
    "pattern": "objetivo_general",
    "flags": ("ocr_page",),
    "general_chars": len(GENERAL),
    "specific_chars": len(SPECIFIC),
    "pages": 80,
    "pages_sha256": "0" * 64,
}
NOT_FOUND_RECORD: dict[str, object] = EXTRACTED_RECORD | {
    "status": "not_found",
    "objective_general": None,
    "objectives_specific": None,
    "page_start": None,
    "page_end": None,
    "pattern": None,
    "flags": (),
    "general_chars": 0,
    "specific_chars": 0,
}


@pytest.mark.parametrize(
    "fields",
    [
        EXTRACTED_RECORD,
        EXTRACTED_RECORD
        | {"objectives_specific": None, "specific_chars": 0, "flags": ("specific_missing",)},
        EXTRACTED_RECORD | {"page_start": 21, "page_end": 21, "flags": ()},
        EXTRACTED_RECORD | {"flags": ("general_too_long", "block_fallback", "ocr_page")},
        NOT_FOUND_RECORD,
    ],
    ids=["extracted", "without-specific", "one-page", "several-flags", "not-found"],
)
def test_consistent_objectives_records_are_accepted(fields: dict[str, object]) -> None:
    record = ObjectivesRecord.model_validate(fields)

    assert record.model_dump() == fields


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            EXTRACTED_RECORD | {"objective_general": None, "general_chars": 0},
            "an 'extracted' record needs",
        ),
        (EXTRACTED_RECORD | {"page_start": None}, "an 'extracted' record needs"),
        (EXTRACTED_RECORD | {"page_end": None}, "an 'extracted' record needs"),
        (EXTRACTED_RECORD | {"pattern": None}, "an 'extracted' record needs"),
        (
            NOT_FOUND_RECORD | {"objective_general": GENERAL, "general_chars": len(GENERAL)},
            "a 'not_found' record holds no objectives",
        ),
        (
            NOT_FOUND_RECORD | {"objectives_specific": SPECIFIC, "specific_chars": len(SPECIFIC)},
            "a 'not_found' record holds no objectives",
        ),
        (NOT_FOUND_RECORD | {"page_start": 20}, "a 'not_found' record holds no objectives"),
        (NOT_FOUND_RECORD | {"page_end": 21}, "a 'not_found' record holds no objectives"),
        (
            NOT_FOUND_RECORD | {"pattern": "objetivo_general"},
            "a 'not_found' record holds no objectives",
        ),
        (NOT_FOUND_RECORD | {"flags": ("ocr_page",)}, "a 'not_found' record has no flags"),
        (EXTRACTED_RECORD | {"general_chars": len(GENERAL) + 1}, "general_chars must be"),
        (NOT_FOUND_RECORD | {"general_chars": 1}, "general_chars must be"),
        (EXTRACTED_RECORD | {"specific_chars": len(SPECIFIC) - 1}, "specific_chars must be"),
        (EXTRACTED_RECORD | {"objectives_specific": None}, "specific_chars must be"),
        (EXTRACTED_RECORD | {"page_start": 22}, "page_end must not come before page_start"),
        (EXTRACTED_RECORD | {"flags": ("ocr_page", "ocr_page")}, "a flag may appear only once"),
    ],
    ids=[
        "extracted-without-general",
        "extracted-without-page-start",
        "extracted-without-page-end",
        "extracted-without-pattern",
        "not-found-with-general",
        "not-found-with-specific",
        "not-found-with-page-start",
        "not-found-with-page-end",
        "not-found-with-pattern",
        "not-found-with-flags",
        "general-chars-off",
        "general-chars-without-text",
        "specific-chars-off",
        "specific-chars-without-text",
        "page-end-before-page-start",
        "duplicate-flags",
    ],
)
def test_objectives_records_refuse_contradictory_fields(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message) as caught:
        ObjectivesRecord.model_validate(fields)

    assert caught.value.error_count() == 1


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            NOT_FOUND_RECORD
            | {
                "objective_general": GENERAL,
                "general_chars": len(GENERAL),
                "flags": ("ocr_page", "ocr_page"),
            },
            "a 'not_found' record holds no objectives",
        ),
        (
            NOT_FOUND_RECORD | {"flags": ("ocr_page", "ocr_page")},
            "a 'not_found' record has no flags",
        ),
    ],
    ids=["text-and-duplicate-flags", "duplicate-flags-only"],
)
def test_a_not_found_record_reports_its_first_broken_rule(
    fields: dict[str, object], message: str
) -> None:
    # The rules run in order and the first broken one is reported: its content, then its
    # flags. Any flag already breaks a 'not_found' record, so the duplicate-flag rule, which
    # comes last, is never the one reported for it.
    with pytest.raises(ValidationError, match=message) as caught:
        ObjectivesRecord.model_validate(fields)

    assert caught.value.error_count() == 1
    assert "a flag may appear only once" not in str(caught.value)


def test_objectives_record_errors_never_echo_the_objectives() -> None:
    with pytest.raises(ValidationError) as caught:
        ObjectivesRecord.model_validate(EXTRACTED_RECORD | {"general_chars": 1})

    # A long input is printed cut in the middle, so check that no input is printed at all.
    assert "input_value" not in str(caught.value)
    assert GENERAL not in str(caught.value)
