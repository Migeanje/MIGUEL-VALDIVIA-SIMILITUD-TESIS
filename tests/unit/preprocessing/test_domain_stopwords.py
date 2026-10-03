"""Domain stopwords: the curated list of genre and boilerplate lemmas, and how it is read.

The list files written here are synthetic; one test reads the curated list of the project.
"""

import unicodedata
from pathlib import Path

import pytest

from thematic_redundancy.preprocessing.stopwords import (
    DOMAIN_STOPWORDS_FILE,
    StopwordCandidate,
    load_domain_stopwords,
    spacy_stopwords,
    stopword_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURATED_LIST = PROJECT_ROOT / DOMAIN_STOPWORDS_FILE


def write_list(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "stopwords.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_domain_stopwords_reads_one_lemma_per_line_and_skips_comments_and_blanks(
    tmp_path: Path,
) -> None:
    path = write_list(tmp_path, "# Géneros de la tesis\n\ntesis\n   investigación  \n#otro\nperú\n")

    assert load_domain_stopwords(path) == {"tesis", "investigación", "perú"}


def test_load_domain_stopwords_puts_each_lemma_in_nfc_and_lower_case(tmp_path: Path) -> None:
    path = write_list(tmp_path, "Peru\u0301\nAREQUIPA\nInvestigacio\u0301n\n")

    assert load_domain_stopwords(path) == {"perú", "arequipa", "investigación"}


def test_load_domain_stopwords_refuses_a_line_with_more_than_one_word(tmp_path: Path) -> None:
    path = write_list(tmp_path, "tesis\nmejora continua\n")

    with pytest.raises(ValueError, match="line 2"):
        load_domain_stopwords(path)


def test_load_domain_stopwords_reads_an_empty_list(tmp_path: Path) -> None:
    path = write_list(tmp_path, "# Sin palabras todavía.\n\n")

    assert load_domain_stopwords(path) == frozenset()


def test_stopword_candidates_rank_lemmas_by_the_documents_that_hold_them() -> None:
    documents = [
        ["tesis", "planta", "tesis", "voladura"],
        ["tesis", "planta"],
        ["tesis", "costo"],
        ["proceso"],
    ]

    candidates = stopword_candidates(documents, min_ratio=0.5)

    assert candidates == [
        StopwordCandidate(lemma="tesis", documents=3, ratio=0.75),
        StopwordCandidate(lemma="planta", documents=2, ratio=0.5),
    ]


def test_stopword_candidates_break_ties_in_lemma_order_and_round_ratios() -> None:
    documents = [["voladura", "calidad"], ["calidad", "voladura"], ["costo"]]

    candidates = stopword_candidates(documents, min_ratio=0.5)

    assert [candidate.lemma for candidate in candidates] == ["calidad", "voladura"]
    assert candidates[0].ratio == 0.6667


def test_stopword_candidates_of_no_documents_are_none() -> None:
    assert stopword_candidates([], min_ratio=0.1) == []


@pytest.mark.parametrize("min_ratio", [0.0, -0.1, 1.5])
def test_stopword_candidates_refuse_a_ratio_outside_zero_to_one(min_ratio: float) -> None:
    with pytest.raises(ValueError, match="min_ratio"):
        stopword_candidates([["tesis"]], min_ratio=min_ratio)


def test_spacy_stopwords_are_the_lower_case_spanish_list() -> None:
    stopwords = spacy_stopwords()

    assert {"de", "la", "los", "que", "para"} <= stopwords
    assert all(word == word.lower() for word in stopwords)


# The curated list

GENRE_AND_BOILERPLATE = {
    "tesis",
    "investigación",
    "objetivo",
    "presente",
    "propuesta",
    "empresa",
    "mejora",
    "desarrollo",
    "universidad",
    "arequipa",
    "perú",
}
TOPIC_WORDS = {
    "productividad",
    "mantenimiento",
    "calidad",
    "costo",
    "seguridad",
    "voladura",
    "energía",
    "sistema",
    "proceso",
    "lean",
}


def test_the_curated_list_holds_genre_and_boilerplate_words_and_no_topic_words() -> None:
    stopwords = load_domain_stopwords(CURATED_LIST)

    assert stopwords >= GENRE_AND_BOILERPLATE
    assert stopwords.isdisjoint(TOPIC_WORDS)


def test_the_curated_list_writes_each_lemma_once_in_nfc_lower_case() -> None:
    lines = [
        line.strip()
        for line in CURATED_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(lines) == len(set(lines))
    assert all(line == unicodedata.normalize("NFC", line).lower() for line in lines)


def test_the_curated_list_repeats_no_spacy_stopword() -> None:
    assert load_domain_stopwords(CURATED_LIST).isdisjoint(spacy_stopwords())


def test_the_curated_list_starts_with_its_curation_criteria() -> None:
    first_line = CURATED_LIST.read_text(encoding="utf-8").splitlines()[0]

    assert first_line.startswith("#")
