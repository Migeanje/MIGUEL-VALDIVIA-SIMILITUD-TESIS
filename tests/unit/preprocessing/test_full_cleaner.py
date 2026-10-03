"""Full cleaner: the lemma tokens that TF-IDF, c-TF-IDF and NPMI read.

These tests run the installed spaCy model ``es_core_news_md``, loaded once per test session.
Every text here is synthetic Spanish written for the test.
"""

import subprocess
import sys

import pytest

from thematic_redundancy.preprocessing.full_cleaner import (
    EXCLUDED_COMPONENTS,
    LEGAL_ENTITY_SUFFIXES,
    clean_full,
    clean_full_batch,
    join_tokens,
    spanish_pipeline,
)

NO_DOMAIN_STOPWORDS: frozenset[str] = frozenset()


def test_clean_full_lowercases_lemmatizes_and_drops_stopwords_and_punctuation() -> None:
    tokens = clean_full(
        "Las investigaciones mejoraron los procesos productivos.",
        domain_stopwords=NO_DOMAIN_STOPWORDS,
    )

    assert tokens == ["investigación", "mejorar", "proceso", "productivo"]


def test_clean_full_keeps_accents_and_eñe() -> None:
    tokens = clean_full(
        "La señalización de las áreas y el diseño ergonómico.",
        domain_stopwords=NO_DOMAIN_STOPWORDS,
    )

    assert tokens == ["señalización", "área", "diseño", "ergonómico"]


def test_clean_full_lemmatizes_text_written_in_capitals() -> None:
    tokens = clean_full("MEJORA DE LOS PROCESOS PRODUCTIVOS", domain_stopwords=NO_DOMAIN_STOPWORDS)

    assert tokens == ["mejora", "proceso", "productivo"]


def test_clean_full_keeps_lean_instead_of_reading_it_as_a_form_of_leer() -> None:
    tokens = clean_full(
        "Se aplicaron herramientas Lean para reducir desperdicios. "
        "La filosofía Lean Manufacturing mejora la productividad.",
        domain_stopwords=NO_DOMAIN_STOPWORDS,
    )

    assert tokens.count("lean") == 2
    assert "leer" not in tokens


def test_clean_full_drops_numbers_symbols_and_one_character_tokens() -> None:
    tokens = clean_full(
        "Se redujo 25 % el costo en 2023, de 3,5 a 2.8 kg por S/. 1,500, con la técnica 5S "
        "y el factor x a 20 °C.",
        domain_stopwords=NO_DOMAIN_STOPWORDS,
    )

    assert {"costo", "kg", "técnica", "5s", "factor"} <= set(tokens)
    assert not {"25", "%", "2023", "3,5", "2.8", "1,500", "s/", "s", "x", "20", "°c"} & set(tokens)
    assert all(len(token) >= 2 and any(char.isalpha() for char in token) for token in tokens)


def test_clean_full_drops_legal_entity_suffixes_in_any_spelling() -> None:
    tokens = clean_full(
        "Minera Andina S.A.C., Servicios Unidos E.I.R.L., Textil Sur S.R.L., Agro Valle SAC, "
        "Pesca Norte S. A., Metal Andes s.a., Ruta Uno S.A.A. y Grupo Alfa SCRL",
        domain_stopwords=NO_DOMAIN_STOPWORDS,
    )

    assert {"textil", "agro", "metal", "ruta", "grupo"} <= set(tokens)
    assert not any(token.replace(".", "") in LEGAL_ENTITY_SUFFIXES for token in tokens)
    assert {"sa", "sac", "saa", "srl", "scrl", "eirl"} <= LEGAL_ENTITY_SUFFIXES


def test_clean_full_unifies_the_spellings_of_an_apostrophe_inside_a_word() -> None:
    tokens = clean_full(
        "La técnica 5S, las 5'S, las 5\u2019S y las 5\u00b4S.", domain_stopwords=NO_DOMAIN_STOPWORDS
    )

    assert tokens == ["técnica", "5s", "5s", "5s", "5s"]


def test_clean_full_drops_domain_stopwords_by_lemma_and_by_form() -> None:
    tokens = clean_full(
        "Las tesis analizan la productividad de las empresas de Arequipa.",
        domain_stopwords=frozenset({"tesis", "empresa", "arequipa"}),
    )

    assert tokens == ["analizar", "productividad"]


def test_clean_full_compares_domain_stopwords_in_nfc_lower_case() -> None:
    tokens = clean_full(
        "La minería del Perú.", domain_stopwords=frozenset({"PERU\u0301", "Minería"})
    )

    assert tokens == []


def test_clean_full_applies_the_light_cleaner_first() -> None:
    tokens = clean_full(
        "La investi-\ngación de \u201cla planta\u201d.", domain_stopwords=NO_DOMAIN_STOPWORDS
    )

    assert tokens == ["investigación", "planta"]


def test_clean_full_splits_a_lemma_that_expands_a_clitic_pronoun() -> None:
    tokens = clean_full(
        "Se decidió implementarlo en la planta.", domain_stopwords=NO_DOMAIN_STOPWORDS
    )

    assert "implementar" in tokens
    assert all(" " not in token for token in tokens)
    assert "él" not in tokens


def test_clean_full_drops_web_addresses_and_email_addresses() -> None:
    tokens = clean_full(
        "Más datos sobre la planta en www.ejemplo.com.pe o en informes@ejemplo.pe.",
        domain_stopwords=NO_DOMAIN_STOPWORDS,
    )

    assert "planta" in tokens
    assert not any("ejemplo" in token for token in tokens)


@pytest.mark.parametrize("text", ["", "   \n ", "2023 - 25 % ; 3,5", "de la y en el"])
def test_clean_full_returns_no_tokens_for_a_text_without_content_words(text: str) -> None:
    assert clean_full(text, domain_stopwords=NO_DOMAIN_STOPWORDS) == []


def test_clean_full_batch_matches_cleaning_one_text_at_a_time() -> None:
    texts = [
        "Las investigaciones mejoraron los procesos productivos.",
        "",
        "La señalización de las áreas de la empresa minera.",
    ]
    domain = frozenset({"empresa"})

    batched = clean_full_batch((text for text in texts), domain_stopwords=domain, batch_size=2)

    assert batched == [clean_full(text, domain_stopwords=domain) for text in texts]


def test_clean_full_batch_refuses_a_batch_size_below_one() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        clean_full_batch(["Texto."], domain_stopwords=NO_DOMAIN_STOPWORDS, batch_size=0)


def test_join_tokens_joins_with_single_spaces() -> None:
    assert join_tokens(["investigación", "planta", "piloto"]) == "investigación planta piloto"
    assert join_tokens([]) == ""


def test_spanish_pipeline_is_loaded_once_without_the_unneeded_components() -> None:
    nlp = spanish_pipeline()

    assert spanish_pipeline() is nlp
    assert set(EXCLUDED_COMPONENTS).isdisjoint(nlp.component_names)
    assert {"morphologizer", "lemmatizer"} <= set(nlp.pipe_names)


def test_importing_the_cleaners_does_not_import_spacy() -> None:
    code = (
        "import sys\n"
        "import thematic_redundancy.preprocessing.full_cleaner\n"
        "import thematic_redundancy.preprocessing.stopwords\n"
        "print('spacy' in sys.modules)\n"
    )

    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert run.stdout.strip() == "False"
