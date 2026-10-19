"""The TF-IDF baseline (T15, D28): its pure core, and the build that cleans, vectorizes, checks
and saves the sections of a snapshot.

The core's documents are synthetic token lists, already cleaned. Their expected vectors come
from :func:`oracle`, an independent computation of the formulas that D28 fixes: a term's
document frequency counts theses, its idf is ``ln((1 + n) / (1 + df)) + 1`` (smooth idf), its
weight is ``(1 + ln(count)) * idf`` (sublinear tf), and every vector is L2-normalized.

For the build, the fichas and the chunk table are synthetic and saved on disk as their builds
save them, so the build reads them through ``load_fichas`` and ``load_chunks``. The cleaner is a
fake behind the ``SectionCleaner`` port, so spaCy never loads: it puts a text in lower case,
splits it on whitespace, strips punctuation around each word, and drops words of one
character, the fake stopwords and the domain stopwords. Every text is synthetic; the invented
word :data:`MARKER` stands for thesis content, which only the data files may hold, never the
manifest, the output of the command or an error message.

The core's tests share this module with the build's, since ``tests/smoke/test_tfidf.py`` takes
the basename ``test_tfidf``, and test modules need unique basenames here.
"""

import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID

import numpy as np
import pytest
import sklearn
import yaml
from scipy import sparse

import thematic_redundancy.representation.build_tfidf as build_module
from thematic_redundancy.corpus.build_fichas import (
    BUILD_COMMAND as FICHAS_BUILD_COMMAND,
)
from thematic_redundancy.corpus.build_fichas import (
    EXCLUSIONS_FILE_NAME,
    FICHAS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    Exclusion,
    FichasManifest,
)
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.preprocessing.full_cleaner import (
    EXCLUDED_COMPONENTS,
    FULL_CLEANER_VERSION,
    KEPT_FORMS,
    LEGAL_ENTITY_SUFFIXES,
    MIN_TOKEN_LENGTH,
)
from thematic_redundancy.preprocessing.stopwords import (
    DOMAIN_STOPWORDS_FILE,
    load_domain_stopwords,
)
from thematic_redundancy.representation.build_chunks import BUILD_COMMAND as CHUNKS_BUILD_COMMAND
from thematic_redundancy.representation.build_chunks import (
    CHUNKER_VERSION,
    CHUNKS_FILE_NAME,
    CHUNKS_MANIFEST_FILE_NAME,
    ChunkRecord,
    ChunksManifest,
    chunk_fichas,
    section_texts,
)
from thematic_redundancy.representation.build_tfidf import (
    BUILD_COMMAND,
    MATRIX_FILE_NAMES,
    SECTIONS_FILE_NAME,
    TERMS_FILE_NAME,
    TFIDF_DIR_NAME,
    TFIDF_MANIFEST_FILE_NAME,
    CleanedSection,
    TfidfManifest,
    TfidfTerm,
    build_tfidf,
    load_full_clean_sections,
    load_tfidf,
    main,
)
from thematic_redundancy.representation.tfidf import (
    TFIDF_VERSION,
    CleanerIdentity,
    TfidfCheckError,
    TfidfSettings,
    fit_tfidf,
    lemma_list_sha256,
    section_terms,
    thesis_terms,
    variant_agreement,
    zero_rows,
)
from thematic_redundancy.shared.config import (
    AppConfig,
    TfidfConfig,
    TfidfCountsConfig,
    load_config,
)
from thematic_redundancy.shared.storage import ParquetTableStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


# The pure core (tfidf.py)

D26_SHA256 = "8a889062a2f117e2e64bd3e97ad0cd1384b8bce1721dbb0622cb24a53ce1b50f"
"""The D26 pin of ``test_domain_stopwords.py``: a deliberate change of the list updates both."""

SETTINGS = TfidfSettings(max_ngram=2, min_df=2, max_df=1.0, sublinear_tf=True)

Document = Sequence[Sequence[str]]

# Four theses: three sections, two sections, three sections and one section. The sections differ
# in length, so the two document vectors differ.
DOCUMENTS: tuple[Document, ...] = (
    (
        ("planta", "textil", "proceso"),
        ("proceso", "costo", "planta", "textil", "proceso", "costo", "proceso"),
        ("reducir", "costo"),
    ),
    (("red", "datos"), ("red", "datos", "planta", "servicio", "calidad")),
    (
        ("planta", "textil"),
        ("costo", "proceso", "calidad", "servicio"),
        ("medir", "calidad", "servicio"),
    ),
    (("red", "datos", "servicio"),),
)


def oracle_terms(section: Sequence[str], max_ngram: int) -> list[str]:
    bigrams = [" ".join(section[i : i + 2]) for i in range(len(section) - 1)]
    return [*section, *(bigrams if max_ngram == 2 else [])]


def oracle(
    documents: Sequence[Document],
    *,
    min_df: int = 2,
    max_df: float = 1.0,
    max_ngram: int = 2,
    sublinear: bool = True,
) -> dict[str, object]:
    """Return the vocabulary, document frequencies, idf and both matrices, computed by hand."""
    n = len(documents)
    by_thesis = [[t for s in doc for t in oracle_terms(s, max_ngram)] for doc in documents]
    df = Counter(term for terms in by_thesis for term in set(terms))
    vocabulary = sorted(term for term, count in df.items() if min_df <= count <= max_df * n)
    column = {term: index for index, term in enumerate(vocabulary)}
    idf = np.array([math.log((1 + n) / (1 + df[term])) + 1 for term in vocabulary])

    def unit(terms: Sequence[str]) -> np.ndarray:
        vector = np.zeros(len(vocabulary))
        for term, count in Counter(t for t in terms if t in column).items():
            weight = 1 + math.log(count) if sublinear else count
            vector[column[term]] = weight * idf[column[term]]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    concatenated = np.array([unit(terms) for terms in by_thesis])
    means = []
    for doc in documents:
        mean = np.mean([unit(oracle_terms(s, max_ngram)) for s in doc], axis=0)
        norm = np.linalg.norm(mean)
        means.append(mean / norm if norm else mean)
    return {
        "terms": tuple(vocabulary),
        "df": np.array([df[term] for term in vocabulary]),
        "idf": idf,
        "concatenated": concatenated,
        "section_mean": np.array(means),
    }


def is_canonical(matrix: sparse.csr_matrix) -> bool:
    """Tell whether each row's column indices strictly increase, from the arrays themselves."""
    return all(
        np.all(np.diff(matrix.indices[start:end]) > 0)
        for start, end in zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True)
    )


# Terms


def test_section_terms_are_the_unigrams_then_the_bigrams_of_adjacent_tokens() -> None:
    assert section_terms(["costo", "proceso", "planta"], 2) == [
        "costo",
        "proceso",
        "planta",
        "costo proceso",
        "proceso planta",
    ]
    assert section_terms(["costo", "proceso", "planta"], 1) == ["costo", "proceso", "planta"]


def test_a_section_of_one_token_or_none_gives_no_bigram() -> None:
    assert section_terms(["costo"], 2) == ["costo"]
    assert section_terms([], 2) == []


def test_no_bigram_spans_the_end_of_a_section() -> None:
    terms = thesis_terms([["red", "datos"], ["planta", "servicio"]], 2)

    assert terms == ["red", "datos", "red datos", "planta", "servicio", "planta servicio"]
    assert "datos planta" not in terms


# Vectorizer


def test_the_vocabulary_is_sorted_and_holds_the_terms_of_at_least_min_df_theses() -> None:
    fitted = fit_tfidf(DOCUMENTS, SETTINGS)
    expected = oracle(DOCUMENTS)

    assert fitted.terms == expected["terms"]
    assert list(fitted.terms) == sorted(fitted.terms)
    assert "planta textil" in fitted.terms and "reducir" not in fitted.terms
    assert all(isinstance(term, str) for term in fitted.terms)


def test_min_df_counts_theses_not_sections() -> None:
    # "turno" is in two sections of one thesis; "lote" is in one section of each of two.
    documents = (
        (("turno", "lote", "planta"), ("turno", "planta")),
        (("lote", "planta"),),
        (("planta", "red"),),
    )

    fitted = fit_tfidf(documents, SETTINGS)

    assert "turno" not in fitted.terms
    assert "lote" in fitted.terms
    frequency = dict(zip(fitted.terms, fitted.document_frequency.tolist(), strict=True))
    assert frequency["lote"] == 2 and frequency["planta"] == 3


def test_max_df_one_keeps_a_term_of_every_thesis_and_a_lower_cap_drops_it() -> None:
    # "planta" is in each of the first three theses, and in three of all four.
    assert "planta" in fit_tfidf(DOCUMENTS[:3], SETTINGS).terms

    capped = fit_tfidf(DOCUMENTS, TfidfSettings(2, 2, 0.5, True))

    assert "planta" not in capped.terms
    assert capped.terms == oracle(DOCUMENTS, max_df=0.5)["terms"]


@pytest.mark.parametrize("sublinear", [True, False], ids=["sublinear-tf", "raw-tf"])
def test_the_concatenated_vectors_follow_tf_smooth_idf_and_l2(sublinear: bool) -> None:
    fitted = fit_tfidf(DOCUMENTS, TfidfSettings(2, 2, 1.0, sublinear))
    expected = oracle(DOCUMENTS, sublinear=sublinear)

    np.testing.assert_allclose(fitted.concatenated.toarray(), expected["concatenated"], atol=1e-12)
    np.testing.assert_allclose(fitted.idf, expected["idf"], atol=1e-12)
    np.testing.assert_array_equal(fitted.document_frequency, expected["df"])


def test_unigrams_only_when_max_ngram_is_one() -> None:
    fitted = fit_tfidf(DOCUMENTS, TfidfSettings(1, 2, 1.0, True))

    assert all(" " not in term for term in fitted.terms)
    np.testing.assert_allclose(
        fitted.section_mean.toarray(), oracle(DOCUMENTS, max_ngram=1)["section_mean"], atol=1e-12
    )


def test_the_section_mean_is_the_normalized_mean_of_the_normalized_section_vectors() -> None:
    fitted = fit_tfidf(DOCUMENTS, SETTINGS)

    np.testing.assert_allclose(
        fitted.section_mean.toarray(), oracle(DOCUMENTS)["section_mean"], atol=1e-12
    )


def test_a_two_section_thesis_weighs_each_section_equally() -> None:
    # The first thesis's two sections share no term, and the second is three times longer.
    # Under equal weights each section's unit vector holds half of the squared norm of the
    # document vector; under concatenation the longer section holds more.
    documents = (
        (("red", "datos"), ("planta", "servicio", "planta", "calidad", "servicio", "turno")),
        (("red", "datos", "planta", "servicio", "calidad", "turno"),),
    )
    fitted = fit_tfidf(documents, SETTINGS)
    first = np.array([term in {"red", "datos", "red datos"} for term in fitted.terms])

    section_mean = fitted.section_mean.toarray()[0]
    concatenated = fitted.concatenated.toarray()[0]

    assert first.sum() == 3 and (~first).sum() == 5
    assert np.linalg.norm(section_mean[first]) == pytest.approx(math.sqrt(0.5), abs=1e-12)
    assert np.linalg.norm(section_mean[~first]) == pytest.approx(math.sqrt(0.5), abs=1e-12)
    assert np.linalg.norm(concatenated[~first]) > np.linalg.norm(concatenated[first]) + 0.1


def test_a_one_section_thesis_has_its_section_vector_under_both_variants() -> None:
    fitted = fit_tfidf(DOCUMENTS, SETTINGS)

    np.testing.assert_allclose(
        fitted.section_mean.toarray()[3], fitted.concatenated.toarray()[3], atol=1e-12
    )


def test_the_two_variants_differ_when_the_section_lengths_differ() -> None:
    fitted = fit_tfidf(DOCUMENTS, SETTINGS)
    cosine = fitted.section_mean.toarray()[0] @ fitted.concatenated.toarray()[0]

    assert cosine < 0.99


def test_both_matrices_are_canonical_float64_csr_with_unit_rows_and_one_support() -> None:
    fitted = fit_tfidf(DOCUMENTS, SETTINGS)

    for matrix in (fitted.section_mean, fitted.concatenated):
        assert sparse.issparse(matrix) and matrix.format == "csr"
        assert matrix.dtype == np.float64
        assert matrix.shape == (len(DOCUMENTS), len(fitted.terms))
        assert is_canonical(matrix)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    assert fitted.section_mean.nnz == fitted.concatenated.nnz
    assert (fitted.section_mean != 0).toarray().tolist() == (
        fitted.concatenated != 0
    ).toarray().tolist()


def test_a_thesis_whose_terms_all_fall_below_min_df_has_a_zero_row() -> None:
    documents = (*DOCUMENTS, (("único", "raro"),))

    fitted = fit_tfidf(documents, SETTINGS)

    assert zero_rows(fitted.concatenated) == 1
    assert zero_rows(fitted.section_mean) == 1
    assert fitted.concatenated[4].nnz == 0 and fitted.section_mean[4].nnz == 0
    assert zero_rows(fit_tfidf(DOCUMENTS, SETTINGS).section_mean) == 0


def test_min_df_above_max_df_times_the_documents_is_refused() -> None:
    with pytest.raises(TfidfCheckError, match=r"min_df \(3\) is above max_df \(0.5\) × 4 theses"):
        fit_tfidf(DOCUMENTS, TfidfSettings(2, 3, 0.5, True))


def test_no_term_left_is_refused() -> None:
    documents = ((("alfa",),), (("beta",),), (("gamma",),))

    with pytest.raises(TfidfCheckError, match="no term occurs in at least 2 theses"):
        fit_tfidf(documents, SETTINGS)


# Agreement


def rows(*vectors: Sequence[float]) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.array(vectors, dtype=np.float64))


def test_variant_agreement_counts_the_rows_strictly_below_the_threshold() -> None:
    primary = rows([1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8])
    other = rows([1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.8, 0.6])

    agreement = variant_agreement(primary, other, 1.0)

    # Cosines: 1, 0, 1 and 0.96; a cosine equal to the threshold is not below it.
    assert agreement.threshold == 1.0
    assert agreement.below == 2
    assert agreement.minimum == 0.0
    assert agreement.median == pytest.approx(0.98, abs=1e-12)
    assert variant_agreement(primary, other, 0.0).below == 0


# The domain stopword hash


def test_the_lemma_hash_is_the_sha256_of_the_sorted_lemmas_joined_by_line_feeds() -> None:
    lemmas = {"tesis", "área", "arequipa"}

    digest = lemma_list_sha256(lemmas)

    assert digest == hashlib.sha256("arequipa\ntesis\nárea".encode()).hexdigest()
    assert lemma_list_sha256(["área", "tesis", "arequipa"]) == digest


def test_the_lemma_hash_of_the_curated_list_is_the_d26_pin() -> None:
    lemmas = load_domain_stopwords(PROJECT_ROOT / DOMAIN_STOPWORDS_FILE)

    assert len(lemmas) == 89
    assert lemma_list_sha256(lemmas) == D26_SHA256


# The build (build_tfidf.py)

SNAPSHOT = "20261002T150405Z"
RUN_TIME = datetime(2026, 10, 19, 9, 0, 0, tzinfo=UTC)
MARKER = "zqxmarcador"
"""An invented word, planted in two theses, so it is a vocabulary term as well as a token."""

FAKE_STOPWORDS = frozenset(
    {"cada", "con", "de", "del", "el", "en", "la", "las", "los", "para", "se", "un", "una", "y"}
)
DOMAIN_LIST = "# Domain stopwords of the tests.\ntesis\narequipa\nproponer\n"
DOMAIN = frozenset({"tesis", "arequipa", "proponer"})
FAKE_IDENTITY = CleanerIdentity(
    spacy_version="3.8.0",
    model_name="xx_fake_md",
    model_version="0.0.1",
    spacy_stopwords=len(FAKE_STOPWORDS),
    spacy_stopwords_sha256=hashlib.sha256("\n".join(sorted(FAKE_STOPWORDS)).encode()).hexdigest(),
)


def fake_clean(text: str, domain: frozenset[str] = DOMAIN) -> list[str]:
    words = (word.strip(".,;:()¿?¡!") for word in text.lower().split())
    return [w for w in words if len(w) > 1 and w not in FAKE_STOPWORDS and w not in domain]


@dataclass
class FakeCleaner:
    domain: frozenset[str]
    identity: CleanerIdentity = FAKE_IDENTITY
    calls: list[list[str]] = field(default_factory=list)

    def clean(self, texts: Sequence[str]) -> list[list[str]]:
        self.calls.append(list(texts))
        return [fake_clean(text, self.domain) for text in texts]


@dataclass
class FakeLoader:
    """Loads a fake cleaner with the domain stopwords that it is given, and keeps it."""

    cleaners: list[FakeCleaner] = field(default_factory=list)

    def __call__(self, domain: frozenset[str]) -> FakeCleaner:
        cleaner = FakeCleaner(domain)
        self.cleaners.append(cleaner)
        return cleaner


def item(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def ficha(number: int, **overrides: Any) -> Ficha:
    values: dict[str, Any] = {
        "item_uuid": item(number),
        "doc_code": doc_code_for(item(number)),
        "handle_url": f"https://hdl.handle.net/123456789/{1000 + number}",
        "snapshot_id": SNAPSHOT,
        "document_type": "tesis",
        "renati_type_raw": "https://purl.org/pe-repo/renati/type#tesis",
        "title": "Gestión del costo en una planta textil, Arequipa 2023",
        "abstract": (
            "Se mide el costo de cada proceso de la planta textil. Se propone reducir el costo "
            f"del proceso {MARKER}."
        ),
        "objectives": "Reducir el costo del proceso en la planta textil.",
        "objectives_status": "extracted",
        "program_key": "sistemas",
        "program_name": "Ingeniería de Sistemas",
        "issue_date_raw": "2023-05-10",
        "issue_year": 2023,
        "issued_after_snapshot": False,
        "rights": "open",
        "pdf_status": "downloaded",
        "pdf_sha256": f"{number:064x}",
        "source_format": "digital",
        "section_source": "title_abstract_objectives",
        "include": True,
    }
    return Ficha(**{**values, **overrides})


METADATA_ONLY: dict[str, Any] = {
    "objectives": None,
    "objectives_status": "no_pdf",
    "pdf_status": "restricted",
    "pdf_sha256": None,
    "source_format": "none",
    "section_source": "title_abstract",
}

FICHAS = (
    ficha(1),
    ficha(
        2,
        title="Red de datos para medir la calidad del servicio",
        abstract="Se analiza la red de datos de la planta. Se mide la calidad del servicio.",
        **METADATA_ONLY,
    ),
    ficha(
        3,
        title="Calidad del servicio en la red de datos de una planta",
        abstract=f"Se evalúa la calidad del servicio de la red de datos {MARKER}.",
        objectives="Medir la calidad del servicio y reducir el costo.",
    ),
    ficha(
        4,
        title="Mantenimiento de una planta textil",
        abstract="Se estudia el mantenimiento del proceso de la planta textil.",
        objectives="Reducir las paradas del proceso de mantenimiento.",
    ),
    ficha(5, include=False, exclusion_reason="duplicate", duplicate_of=item(1)),
)
INCLUDED = tuple(f for f in FICHAS if f.include)
DOC_CODES = tuple(f.doc_code for f in INCLUDED)
WORDS = {w for f in FICHAS for t in (f.title, f.abstract, f.objectives) if t for w in fake_clean(t)}
"""Every word of the synthetic texts that the cleaner keeps."""


def oracle_counts(fichas: Sequence[Ficha] = FICHAS, domain: frozenset[str] = DOMAIN) -> dict:
    """Count by hand what the build must count, from the fake cleaner's tokens."""
    theses = [section_texts(f) for f in fichas if f.include]
    tokens = [[fake_clean(text, domain) for text in texts.values()] for texts in theses]
    nonempty = [[section for section in doc if section] for doc in tokens]
    terms = [
        {t for s in doc for t in s} | {f"{a} {b}" for s in doc for a, b in pairwise(s)}
        for doc in nonempty
    ]
    frequency = Counter(term for thesis in terms for term in thesis)
    vocabulary = {term for term, count in frequency.items() if count >= 2}
    bigrams = sum(" " in term for term in vocabulary)
    return {
        "documents": len(theses),
        "sections": sum(len(doc) for doc in nonempty),
        "empty_sections": sum(len(doc) for doc in tokens) - sum(len(doc) for doc in nonempty),
        "tokens": sum(len(s) for doc in nonempty for s in doc),
        "terms": len(vocabulary),
        "unigrams": len(vocabulary) - bigrams,
        "bigrams": bigrams,
        "nonzeros": sum(len(thesis & vocabulary) for thesis in terms),
    }


COUNTS = oracle_counts()


def expected_counts(counts: dict | None = None, **changes: int) -> TfidfCountsConfig:
    counts = counts or COUNTS
    values = {name: counts[name] for name in ("documents", "sections", "terms", "nonzeros")}
    return TfidfCountsConfig(snapshot_id=SNAPSHOT, **{**values, **changes})


def tfidf_config(**changes: Any) -> TfidfConfig:
    values: dict[str, Any] = {
        "max_ngram": 2,
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
        "document_vector": "section_mean",
        "sensitivity_cosine_threshold": 0.95,
        "expected": expected_counts(),
    }
    return TfidfConfig(**{**values, **changes})


def app_config(**changes: Any) -> AppConfig:
    return load_config(DEFAULT_CONFIG_PATH).model_copy(update={"tfidf": tfidf_config(**changes)})


def interim(project_root: Path) -> Path:
    return project_root / "data" / "interim" / SNAPSHOT


def outputs(project_root: Path) -> Path:
    return interim(project_root) / TFIDF_DIR_NAME


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fichas_outputs(interim_dir: Path, fichas: Sequence[Ficha] = FICHAS) -> str:
    """Save ``fichas`` as the fichas build does, over earlier outputs, and return the table's
    SHA-256."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    store = ParquetTableStore()
    store.write(list(fichas), interim_dir / FICHAS_FILE_NAME, model_cls=Ficha)
    store.write([], interim_dir / EXCLUSIONS_FILE_NAME, model_cls=Exclusion)
    manifest = FichasManifest(
        snapshot_id=SNAPSHOT,
        created_at=RUN_TIME,
        metadata_sha256="a" * 64,
        objectives_table_sha256="b" * 64,
        fichas_file=FICHAS_FILE_NAME,
        fichas_sha256=sha256(interim_dir / FICHAS_FILE_NAME),
        exclusions_file=EXCLUSIONS_FILE_NAME,
        exclusions_sha256=sha256(interim_dir / EXCLUSIONS_FILE_NAME),
        counts={},
        seconds=0.1,
    )
    (interim_dir / FICHAS_MANIFEST_FILE_NAME).write_bytes(manifest.model_dump_json().encode())
    return manifest.fichas_sha256


@dataclass(frozen=True)
class WordCounter:
    """Counts two special tokens plus one per word."""

    def count(self, text: str) -> int:
        return len(text.split()) + 2


def write_chunk_outputs(interim_dir: Path, fichas: Sequence[Ficha] = FICHAS) -> str:
    """Chunk ``fichas`` and save the table as the chunk build does; return its SHA-256."""
    records = chunk_fichas(fichas, WordCounter(), 128)
    ParquetTableStore().write(list(records), interim_dir / CHUNKS_FILE_NAME, model_cls=ChunkRecord)
    manifest = ChunksManifest(
        snapshot_id=SNAPSHOT,
        created_at=RUN_TIME,
        chunker_version=CHUNKER_VERSION,
        fichas_sha256=sha256(interim_dir / FICHAS_FILE_NAME),
        max_tokens=128,
        tokenizers=(),
        chunks_file=CHUNKS_FILE_NAME,
        chunks_sha256=sha256(interim_dir / CHUNKS_FILE_NAME),
        counts={},
        cut_kinds={},
        sections_used={},
        sections_recovered=0,
        seconds=0.1,
    )
    (interim_dir / CHUNKS_MANIFEST_FILE_NAME).write_bytes(manifest.model_dump_json().encode())
    return manifest.chunks_sha256


def write_inputs(
    project_root: Path,
    fichas: Sequence[Ficha] = FICHAS,
    *,
    chunked: Sequence[Ficha] | None = None,
    domain_list: str = DOMAIN_LIST,
) -> None:
    """Save the fichas, the chunk table of ``chunked`` (``fichas`` by default) and the domain
    stopword list under ``project_root``."""
    write_fichas_outputs(interim(project_root), fichas)
    write_chunk_outputs(interim(project_root), fichas if chunked is None else chunked)
    write_domain_list(project_root, domain_list)


def write_domain_list(project_root: Path, content: str) -> None:
    (project_root / DOMAIN_STOPWORDS_FILE).parent.mkdir(parents=True, exist_ok=True)
    (project_root / DOMAIN_STOPWORDS_FILE).write_bytes(content.encode())


def build(
    project_root: Path,
    config: AppConfig | None = None,
    loader: FakeLoader | None = None,
    **seams: Any,
) -> Any:
    return build_tfidf(
        config or app_config(),
        project_root,
        snapshot_id=SNAPSHOT,
        load_cleaner=loader or FakeLoader(),
        clock=lambda: RUN_TIME,
        timer=lambda: 0.0,
        **seams,
    )


def output_files(project_root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(outputs(project_root).iterdir())}


def documents_of(sections: Sequence[CleanedSection], theses: int) -> list[list[list[str]]]:
    documents: list[list[list[str]]] = [[] for _ in range(theses)]
    for section in sections:
        documents[section.document_row].append(list(section.tokens))
    return documents


DATA_FILES = (SECTIONS_FILE_NAME, TERMS_FILE_NAME, *MATRIX_FILE_NAMES.values())


# What the build writes


def test_the_build_cleans_each_section_on_its_own_and_saves_the_outputs(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    loader = FakeLoader()

    built = build(tmp_path, loader=loader)

    # One cleaner, given the domain lemmas, cleans every section as a text of its own.
    assert [cleaner.domain for cleaner in loader.cleaners] == [DOMAIN]
    texts = [text for f in INCLUDED for text in section_texts(f).values()]
    assert loader.cleaners[0].calls == [texts]
    sections = load_full_clean_sections(interim(tmp_path), tmp_path)
    expected_rows = [
        CleanedSection(
            doc_code=f.doc_code,
            document_row=row,
            section=section,
            n_tokens=len(fake_clean(text)),
            tokens=tuple(fake_clean(text)),
        )
        for row, f in enumerate(INCLUDED)
        for section, text in section_texts(f).items()
    ]
    assert sections == tuple(expected_rows)
    manifest = built.manifest
    assert manifest.counts == COUNTS
    assert manifest.sections_used == {"title+abstract+objectives": 3, "title+abstract": 1}
    assert manifest.files == {name: sha256(outputs(tmp_path) / name) for name in DATA_FILES}
    saved = (outputs(tmp_path) / TFIDF_MANIFEST_FILE_NAME).read_text("utf-8")
    assert json.loads(saved) == json.loads(manifest.model_dump_json())
    assert built.tfidf_dir == outputs(tmp_path)


def test_the_terms_table_holds_each_column_with_its_document_frequency_and_idf(
    tmp_path: Path,
) -> None:
    write_inputs(tmp_path)
    build(tmp_path)

    terms = ParquetTableStore().read(TfidfTerm, outputs(tmp_path) / TERMS_FILE_NAME)

    assert [term.column for term in terms] == list(range(COUNTS["terms"]))
    assert [term.term for term in terms] == sorted(term.term for term in terms)
    assert all(term.n_words == term.term.count(" ") + 1 for term in terms)
    sections = load_full_clean_sections(interim(tmp_path), tmp_path)
    by_thesis: dict[str, set[str]] = {}
    for section in sections:
        bigrams = {f"{a} {b}" for a, b in pairwise(section.tokens)}
        by_thesis.setdefault(section.doc_code, set()).update(section.tokens, bigrams)
    for term in terms:
        frequency = sum(term.term in thesis for thesis in by_thesis.values())
        assert term.document_frequency == frequency >= 2
        assert term.idf == pytest.approx(math.log(5 / (1 + frequency)) + 1, abs=1e-12)


def test_the_manifest_records_the_provenance_without_text(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    fichas_sha256 = sha256(interim(tmp_path) / FICHAS_FILE_NAME)
    chunks_sha256 = sha256(interim(tmp_path) / CHUNKS_FILE_NAME)

    manifest = build(tmp_path).manifest

    assert (manifest.snapshot_id, manifest.created_at) == (SNAPSHOT, RUN_TIME)
    assert manifest.tfidf_version == TFIDF_VERSION
    assert manifest.fichas_sha256 == fichas_sha256
    assert (manifest.chunker_version, manifest.chunks_sha256) == (CHUNKER_VERSION, chunks_sha256)
    assert manifest.cleaner.model_dump() == {
        "full_cleaner_version": FULL_CLEANER_VERSION,
        "spacy_version": "3.8.0",
        "model_name": "xx_fake_md",
        "model_version": "0.0.1",
        "excluded_components": EXCLUDED_COMPONENTS,
        "min_token_length": MIN_TOKEN_LENGTH,
        "kept_forms": tuple(sorted(KEPT_FORMS)),
        "legal_entity_suffixes": tuple(sorted(LEGAL_ENTITY_SUFFIXES)),
        "spacy_stopwords": len(FAKE_STOPWORDS),
        "spacy_stopwords_sha256": FAKE_IDENTITY.spacy_stopwords_sha256,
    }
    assert manifest.domain_stopwords.model_dump() == {
        "path": "config/stopwords_domain_es.txt",
        "lemmas": 3,
        "lemmas_sha256": hashlib.sha256(b"arequipa\nproponer\ntesis").hexdigest(),
        "file_sha256": hashlib.sha256(DOMAIN_LIST.encode()).hexdigest(),
    }
    assert manifest.vectorizer.model_dump() == {
        "max_ngram": 2,
        "ngrams_within_sections": True,
        "min_df": 2,
        "max_df": 1.0,
        "document_frequency_unit": "thesis",
        "sublinear_tf": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "dtype": "float64",
    }
    assert set(manifest.versions) == {"python", "numpy", "scipy", "scikit-learn", "pyarrow"}
    assert manifest.versions["scikit-learn"] == sklearn.__version__
    assert manifest.document_variant == "section_mean"
    saved = (outputs(tmp_path) / TFIDF_MANIFEST_FILE_NAME).read_text("utf-8")
    assert MARKER not in saved
    assert not any(f'"{word}"' in saved for word in WORDS)


def test_the_manifest_summarizes_the_cosine_between_the_two_variants(tmp_path: Path) -> None:
    write_inputs(tmp_path)

    manifest = build(tmp_path).manifest

    primary = load_tfidf(interim(tmp_path), tmp_path).matrix
    other = load_tfidf(interim(tmp_path), tmp_path, variant="concatenated").matrix
    cosines = np.asarray(primary.multiply(other).sum(axis=1)).ravel()
    agreement = manifest.variant_agreement
    assert agreement.threshold == 0.95
    assert agreement.minimum == pytest.approx(cosines.min(), abs=1e-6)
    assert agreement.median == pytest.approx(np.median(cosines), abs=1e-6)
    assert agreement.below == int((cosines < 0.95).sum())


def test_the_reader_gives_the_configured_variant_or_the_one_asked_for(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    build(tmp_path, app_config(document_vector="concatenated"))
    sections = load_full_clean_sections(interim(tmp_path), tmp_path)
    fitted = fit_tfidf(documents_of(sections, 4), TfidfSettings(2, 2, 1.0, True))

    configured = load_tfidf(interim(tmp_path), tmp_path)
    asked = load_tfidf(interim(tmp_path), tmp_path, variant="section_mean")

    assert configured.variant == "concatenated" and asked.variant == "section_mean"
    for loaded, matrix in ((configured, fitted.concatenated), (asked, fitted.section_mean)):
        assert loaded.doc_codes == DOC_CODES
        assert loaded.terms == fitted.terms
        np.testing.assert_array_equal(loaded.idf, fitted.idf)
        assert loaded.matrix.format == "csr" and loaded.matrix.dtype == np.float64
        assert (loaded.matrix != matrix).nnz == 0


def test_an_empty_section_is_left_out_of_the_vectors_and_counted(tmp_path: Path) -> None:
    # Only stopwords: the section has text, so it has chunks, but no token once cleaned.
    fichas = (*FICHAS[:3], FICHAS[3].model_copy(update={"objectives": "De la y en el."}), FICHAS[4])
    counts = oracle_counts(fichas)
    write_inputs(tmp_path, fichas)

    manifest = build(tmp_path, app_config(expected=expected_counts(counts))).manifest

    assert counts["empty_sections"] == 1 and manifest.counts == counts
    assert manifest.sections_used == {"title+abstract+objectives": 3, "title+abstract": 1}
    sections = load_full_clean_sections(interim(tmp_path), tmp_path)
    assert [s.section for s in sections if s.doc_code == DOC_CODES[3]] == ["title", "abstract"]
    fitted = fit_tfidf(documents_of(sections, 4), TfidfSettings(2, 2, 1.0, True))
    assert (load_tfidf(interim(tmp_path), tmp_path).matrix != fitted.section_mean).nnz == 0


# Checks that stop the build before anything is written


@pytest.mark.parametrize(
    ("chunked", "message"),
    [
        ((FICHAS[2], FICHAS[0], FICHAS[1], FICHAS[3]), "in another order than in the chunk table"),
        (FICHAS[:3], "1 included thesis is missing from the chunk table"),
        (
            (*FICHAS[:4], ficha(6, title="Otra tesis")),
            "1 thesis of the chunk table is not included",
        ),
        (
            (FICHAS[0].model_copy(update=METADATA_ONLY), *FICHAS[1:]),
            "1 thesis uses other sections than the chunk table records",
        ),
    ],
    ids=["order", "missing-thesis", "extra-thesis", "sections-used"],
)
def test_sections_that_do_not_match_the_chunk_table_stop_the_build_before_cleaning(
    tmp_path: Path, chunked: Sequence[Ficha], message: str
) -> None:
    write_inputs(tmp_path, chunked=chunked)
    loader = FakeLoader()

    with pytest.raises(TfidfCheckError) as caught:
        build(tmp_path, loader=loader)

    assert message in str(caught.value)
    assert str(caught.value).endswith(f"; run {CHUNKS_BUILD_COMMAND} again")
    assert MARKER not in str(caught.value)
    assert loader.cleaners == []
    assert not outputs(tmp_path).exists()


def test_a_thesis_without_any_term_stops_the_build(tmp_path: Path) -> None:
    # Every word of the sixth thesis occurs in no other thesis, so min_df 2 drops them all.
    lone = ficha(
        6, title="Xilófono ornamental", abstract="Zarzuela barroca andina.", **METADATA_ONLY
    )
    fichas = (*FICHAS, lone)
    counts = oracle_counts(fichas)
    write_inputs(tmp_path, fichas)

    with pytest.raises(TfidfCheckError) as caught:
        build(tmp_path, app_config(expected=expected_counts(counts)))

    assert str(caught.value) == (
        f"the TF-IDF of snapshot {SNAPSHOT} fails 1 check: 1 thesis has no term, so it has no "
        "cosine"
    )
    assert not outputs(tmp_path).exists()


@pytest.mark.parametrize("name", ["documents", "sections", "terms", "nonzeros"])
def test_a_count_off_the_expectation_stops_the_build(tmp_path: Path, name: str) -> None:
    write_inputs(tmp_path)
    expected = expected_counts(**{name: COUNTS[name] + 1})

    with pytest.raises(TfidfCheckError) as caught:
        build(tmp_path, app_config(expected=expected))

    assert str(caught.value) == (
        f"the TF-IDF of snapshot {SNAPSHOT} fails 1 check: {name} is {COUNTS[name]}, but "
        f"tfidf.expected declares {COUNTS[name] + 1}"
    )
    assert not outputs(tmp_path).exists()


def test_every_count_off_the_expectation_is_named(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    names = ("documents", "sections", "terms", "nonzeros")
    expected = expected_counts(**{name: COUNTS[name] + 2 for name in names})

    with pytest.raises(TfidfCheckError, match="fails 4 checks: documents is ") as caught:
        build(tmp_path, app_config(expected=expected))

    for name in names:
        assert f"{name} is {COUNTS[name]}, but tfidf.expected declares {COUNTS[name] + 2}" in str(
            caught.value
        )


def test_an_expectation_of_another_snapshot_is_refused_before_any_cleaning(
    tmp_path: Path,
) -> None:
    write_inputs(tmp_path)
    expected = expected_counts().model_copy(update={"snapshot_id": "20261002T224412Z"})
    loader = FakeLoader()

    with pytest.raises(ValueError, match="declares the counts of snapshot '20261002T224412Z'"):
        build(tmp_path, app_config(expected=expected), loader=loader)

    assert loader.cleaners == []


def test_the_build_refuses_min_df_above_max_df_times_the_documents(tmp_path: Path) -> None:
    write_inputs(tmp_path)

    with pytest.raises(TfidfCheckError, match=r"min_df \(3\) is above max_df \(0.5\) × 4"):
        build(tmp_path, app_config(min_df=3, max_df=0.5))

    assert not outputs(tmp_path).exists()


# Bytes


def test_two_builds_write_byte_identical_files(tmp_path: Path) -> None:
    for root in (tmp_path / "first", tmp_path / "second"):
        write_inputs(root)
        build(root)

    first, second = output_files(tmp_path / "first"), output_files(tmp_path / "second")

    assert first.keys() == {*DATA_FILES, TFIDF_MANIFEST_FILE_NAME}
    assert first == second
    for name in MATRIX_FILE_NAMES.values():
        matrix = sparse.load_npz(io.BytesIO(first[name]))
        assert matrix.format == "csr" and matrix.dtype == np.float64
        assert matrix.has_canonical_format


# Reading the outputs


class UnreadStore:
    """A table store that fails the test if the reader reads any table through it."""

    def read_verified(self, *_: object, **__: object) -> NoReturn:
        pytest.fail("the reader read a table before it checked the outputs")


def no_matrix_read(*_: object, **__: object) -> NoReturn:
    pytest.fail("the reader parsed a matrix before it checked it")


@pytest.mark.parametrize(
    "damage",
    [
        "changed-matrix",
        "changed-sections",
        "changed-terms",
        "truncated-manifest",
        "not-json-manifest",
        "refused-manifest-value",
        "missing-matrix",
        "missing-terms",
    ],
)
def test_damaged_or_missing_outputs_are_refused_without_quoting_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    write_inputs(tmp_path)
    build(tmp_path)
    directory = outputs(tmp_path)
    manifest = directory / TFIDF_MANIFEST_FILE_NAME
    content = manifest.read_bytes()
    matrix = directory / MATRIX_FILE_NAMES["section_mean"]
    named = {
        "changed-matrix": matrix,
        "changed-sections": directory / SECTIONS_FILE_NAME,
        "changed-terms": directory / TERMS_FILE_NAME,
        "missing-matrix": matrix,
        "missing-terms": directory / TERMS_FILE_NAME,
    }.get(damage)
    if damage.startswith("changed"):
        assert named is not None
        named.write_bytes(named.read_bytes() + b"\0")
        # A changed matrix must be refused before it is parsed.
        monkeypatch.setattr(build_module, "load_npz", no_matrix_read)
    elif damage == "truncated-manifest":
        manifest.write_bytes(content[: len(content) // 2])
    elif damage == "not-json-manifest":
        manifest.write_bytes(f"{MARKER} ".encode() + content)
    elif damage == "refused-manifest-value":
        # A snapshot id may not hold a space, so the manifest is refused.
        refused = f'"{MARKER} {MARKER}"'.encode()
        manifest.write_bytes(content.replace(f'"{SNAPSHOT}"'.encode(), refused, 1))
    else:
        assert named is not None
        named.unlink()

    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        load_tfidf(interim(tmp_path), tmp_path)

    message = str(caught.value)
    if damage.startswith("missing"):
        assert caught.type is FileNotFoundError
        assert message == f"{named} is missing; run {BUILD_COMMAND}"
    elif damage.startswith("changed"):
        assert caught.type is ValueError
        assert message == f"{named} differs from {manifest.name}; run {BUILD_COMMAND} again"
    else:
        assert caught.type is ValueError
        assert message == f"{manifest} is damaged; run {BUILD_COMMAND} again"
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert MARKER not in message


def rewrite_matrix(project_root: Path, matrix: Any) -> Path:
    """Save ``matrix`` as the section-mean matrix, with its SHA-256 in the manifest."""
    buffer = io.BytesIO()
    sparse.save_npz(buffer, matrix, compressed=False)
    path = outputs(project_root) / MATRIX_FILE_NAMES["section_mean"]
    path.write_bytes(buffer.getvalue())
    manifest_path = outputs(project_root) / TFIDF_MANIFEST_FILE_NAME
    manifest = TfidfManifest.model_validate_json(manifest_path.read_bytes())
    files = {**manifest.files, path.name: sha256(path)}
    manifest_path.write_bytes(
        manifest.model_copy(update={"files": files}).model_dump_json().encode()
    )
    return path


def unsorted(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return ``matrix`` with the column indices of its first row in reverse order."""
    indices, data = matrix.indices.copy(), matrix.data.copy()
    end = matrix.indptr[1]
    indices[:end], data[:end] = indices[:end][::-1].copy(), data[:end][::-1].copy()
    return sparse.csr_matrix((data, indices, matrix.indptr.copy()), shape=matrix.shape)


@pytest.mark.parametrize(
    "form",
    ["one-row-short", "one-term-short", "float32", "unsorted-indices", "not-unit-rows", "coo"],
)
def test_a_matrix_of_another_shape_or_form_is_refused(tmp_path: Path, form: str) -> None:
    write_inputs(tmp_path)
    build(tmp_path)
    good = sparse.load_npz(outputs(tmp_path) / MATRIX_FILE_NAMES["section_mean"])
    bad = {
        "one-row-short": lambda: good[:-1],
        "one-term-short": lambda: good[:, :-1],
        "float32": lambda: good.astype(np.float32),
        "unsorted-indices": lambda: unsorted(good),
        "not-unit-rows": lambda: good * 2.0,
        "coo": lambda: good.tocoo(),
    }[form]()
    path = rewrite_matrix(tmp_path, bad)

    with pytest.raises(ValueError) as caught:
        load_tfidf(interim(tmp_path), tmp_path)

    assert str(caught.value) == (
        f"{path} does not hold 4 canonical, L2-normalized float64 CSR rows of "
        f"{COUNTS['terms']} terms; run {BUILD_COMMAND} again"
    )


# Outputs of the current inputs and code only

STALE_FIELDS = (
    "snapshot_id",
    "fichas_sha256",
    "tfidf_version",
    "cleaner.full_cleaner_version",
    "domain_stopwords.lemmas_sha256",
)
"""The fields of the TF-IDF manifest that must still match the inputs and the code."""


def rewrite_fichas_manifest(interim_dir: Path, **changes: Any) -> None:
    path = interim_dir / FICHAS_MANIFEST_FILE_NAME
    manifest = FichasManifest.model_validate_json(path.read_bytes())
    path.write_bytes(manifest.model_copy(update=changes).model_dump_json().encode())


@pytest.mark.parametrize(
    ("drift", "fields"),
    [
        ("rebuilt-fichas", ("fichas_sha256",)),
        ("other-snapshot", ("snapshot_id",)),
        ("new-tfidf-version", ("tfidf_version",)),
        ("new-full-cleaner-version", ("cleaner.full_cleaner_version",)),
        ("edited-domain-list", ("domain_stopwords.lemmas_sha256",)),
        (
            "new-tfidf-version-and-edited-domain-list",
            ("tfidf_version", "domain_stopwords.lemmas_sha256"),
        ),
    ],
)
def test_outputs_that_the_current_inputs_or_code_would_not_give_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str, fields: tuple[str, ...]
) -> None:
    write_inputs(tmp_path)
    saved = build(tmp_path).manifest
    current: dict[str, Any] = {}
    if drift == "rebuilt-fichas":
        changed = FICHAS[3].model_copy(update={"abstract": "Se estudia otra planta."})
        current["fichas_sha256"] = write_fichas_outputs(
            interim(tmp_path), (*FICHAS[:3], changed, FICHAS[4])
        )
    if drift == "other-snapshot":
        rewrite_fichas_manifest(interim(tmp_path), snapshot_id="20261003T120000Z")
        current["snapshot_id"] = "20261003T120000Z"
    if "new-tfidf-version" in drift:
        monkeypatch.setattr(build_module, "TFIDF_VERSION", TFIDF_VERSION + 1)
        current["tfidf_version"] = TFIDF_VERSION + 1
    if drift == "new-full-cleaner-version":
        monkeypatch.setattr(build_module, "FULL_CLEANER_VERSION", FULL_CLEANER_VERSION + 1)
        current["cleaner.full_cleaner_version"] = FULL_CLEANER_VERSION + 1
    if drift.endswith("edited-domain-list"):
        write_domain_list(tmp_path, DOMAIN_LIST + "planta\n")
        lemmas = "arequipa\nplanta\nproponer\ntesis"
        current["domain_stopwords.lemmas_sha256"] = hashlib.sha256(lemmas.encode()).hexdigest()
    monkeypatch.setattr(build_module, "load_npz", no_matrix_read)
    loaded = None

    with pytest.raises(ValueError) as caught:
        loaded = load_tfidf(interim(tmp_path), tmp_path, store=UnreadStore())

    assert caught.type is ValueError and loaded is None
    message = str(caught.value)
    assert message.startswith(f"the TF-IDF outputs in {outputs(tmp_path)} are stale: ")
    assert message.endswith(f"; run {BUILD_COMMAND} again")
    sources = {
        "snapshot_id": FICHAS_MANIFEST_FILE_NAME,
        "fichas_sha256": FICHAS_MANIFEST_FILE_NAME,
        "tfidf_version": "the code",
        "cleaner.full_cleaner_version": "the code",
        "domain_stopwords.lemmas_sha256": "config/stopwords_domain_es.txt",
    }
    for name in STALE_FIELDS:
        if name in fields:
            old = saved.model_dump()
            for part in name.split("."):
                old = old[part]
            assert old != current[name]
            assert (
                f"{name} is {old} in {TFIDF_MANIFEST_FILE_NAME} but {current[name]} in "
                f"{sources[name]}" in message
            )
        else:
            assert f"{name} is " not in message
    assert MARKER not in message
    with pytest.raises(ValueError, match="are stale: "):
        load_full_clean_sections(interim(tmp_path), tmp_path, store=UnreadStore())


def test_outputs_stay_valid_after_edits_that_change_no_input(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    build(tmp_path)
    # A new comment in the list, and a fichas manifest written again with only new run fields.
    write_domain_list(tmp_path, "# Reviewed again.\n" + DOMAIN_LIST)
    rewrite_fichas_manifest(
        interim(tmp_path), created_at=datetime(2026, 10, 20, 8, 0, 0, tzinfo=UTC), seconds=2.5
    )

    loaded = load_tfidf(interim(tmp_path), tmp_path)

    assert loaded.doc_codes == DOC_CODES


def test_outputs_without_the_fichas_manifest_are_refused(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    build(tmp_path)
    (interim(tmp_path) / FICHAS_MANIFEST_FILE_NAME).unlink()

    with pytest.raises(FileNotFoundError) as caught:
        load_tfidf(interim(tmp_path), tmp_path, store=UnreadStore())

    assert str(caught.value) == (
        f"no {FICHAS_MANIFEST_FILE_NAME} in {interim(tmp_path)}; run {FICHAS_BUILD_COMMAND} first"
    )


# Command line


def write_project(project_root: Path, **changes: Any) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["tfidf"] = {**tfidf_config().model_dump(mode="json"), **changes}
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "config" / "default.yaml").write_bytes(
        yaml.safe_dump(raw, allow_unicode=True).encode()
    )
    write_domain_list(project_root, DOMAIN_LIST)


def run_main(project_root: Path, **seams: Any) -> int:
    values: dict[str, Any] = {
        "clock": lambda: RUN_TIME,
        "timer": lambda: 0.0,
        "load_cleaner": FakeLoader(),
        **seams,
    }
    return main(["--project-root", str(project_root), "--snapshot-id", SNAPSHOT], **values)


def visible_words(text: str) -> set[str]:
    return {word.strip(".,;:()").lower() for word in text.split()}


def test_the_command_reports_counts_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    write_inputs(tmp_path)

    assert run_main(tmp_path) == 0

    captured = capsys.readouterr()
    out = captured.out
    assert (
        f"Snapshot {SNAPSHOT}: {COUNTS['documents']} documents, {COUNTS['sections']} sections "
        f"(0 empty), {COUNTS['tokens']} tokens" in out
    )
    assert (
        f"Terms: {COUNTS['terms']} ({COUNTS['unigrams']} unigrams, {COUNTS['bigrams']} bigrams "
        "within sections); min_df 2 theses, max_df 1.0, sublinear tf" in out
    )
    assert f"Nonzeros: {COUNTS['nonzeros']} per matrix; 0 zero rows" in out
    assert "section_mean vs concatenated, cosine per document: min " in out
    assert "Document vector read downstream: section_mean" in out
    assert f"cleaning {COUNTS['sections']} sections of 4 theses" in captured.err
    assert not (WORDS | {MARKER}) & visible_words(captured.out + captured.err)


def test_the_command_reports_a_failed_check_without_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path, expected={**expected_counts().model_dump(), "terms": 1})
    write_inputs(tmp_path)

    assert run_main(tmp_path) == 1

    captured = capsys.readouterr()
    error = captured.err.splitlines()[-1]
    assert error.startswith("error: the TF-IDF of snapshot ") and captured.out == ""
    assert error.endswith(f"terms is {COUNTS['terms']}, but tfidf.expected declares 1")
    assert not (WORDS | {MARKER}) & visible_words(captured.err)
    assert not outputs(tmp_path).exists()


def test_the_command_reports_a_missing_chunk_table_on_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    write_fichas_outputs(interim(tmp_path))

    assert run_main(tmp_path) == 1

    err = capsys.readouterr().err
    assert err.startswith(f"error: no {CHUNKS_MANIFEST_FILE_NAME}") and err.count("\n") == 1
    assert "build_chunks" in err


@pytest.mark.parametrize("content", ["tfidf: [\n", None], ids=["not-yaml", "refused-value"])
def test_the_command_reports_a_bad_configuration_on_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
) -> None:
    write_project(tmp_path, min_df=0)
    if content is not None:
        (tmp_path / "config" / "default.yaml").write_bytes(content.encode())

    assert run_main(tmp_path) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1


def interrupt(*_: object) -> NoReturn:
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    "seam",
    ["load_cleaner", "clock"],
    ids=["while-loading-the-cleaner", "between-the-outputs-and-their-manifest"],
)
def test_an_interrupted_command_exits_130_and_asks_for_a_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seam: str
) -> None:
    write_project(tmp_path)
    write_inputs(tmp_path)

    code = run_main(tmp_path, **{seam: interrupt})

    assert code == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith(
        "interrupted: TF-IDF outputs that do not match their manifest are refused by readers; "
        "run the command again\n"
    )
    # The clock stamps the manifest, so the data files were already written then, and no reader
    # accepts them without their manifest.
    written = {path.name for path in outputs(tmp_path).iterdir()} if seam == "clock" else set()
    assert written == (set(DATA_FILES) if seam == "clock" else set())
    with pytest.raises(FileNotFoundError, match=f"no {TFIDF_MANIFEST_FILE_NAME}"):
        load_tfidf(interim(tmp_path), tmp_path)


def test_importing_the_build_does_not_import_spacy() -> None:
    # The reader serves later steps, which must not pay for spaCy; only the cleaner loads it.
    code = (
        "import sys\n"
        "import thematic_redundancy.representation.build_tfidf\n"
        "print('spacy' in sys.modules)\n"
    )

    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert run.stdout.strip() == "False"
