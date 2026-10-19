"""Smoke checks of the TF-IDF baseline with the real cleaner and corpus, offline (T15, D28).

They copy the saved fichas and chunk outputs of the configured snapshot, with the configuration
and the domain stopword list, into a temporary project, and build the baseline there twice with
spaCy's real pipeline; the project's own outputs are never touched. They are skipped when those
inputs are missing. They assert counts, hashes and shapes only, and print no text: tokens are
compared by counting mismatches. Run them with
``uv run pytest -m smoke tests/smoke/test_tfidf.py``.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from thematic_redundancy.corpus.build_fichas import (
    EXCLUSIONS_FILE_NAME,
    FICHAS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    load_fichas,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.preprocessing.stopwords import (
    DOMAIN_STOPWORDS_FILE,
    load_domain_stopwords,
    spacy_stopwords,
)
from thematic_redundancy.representation.build_chunks import (
    CHUNKS_FILE_NAME,
    CHUNKS_MANIFEST_FILE_NAME,
    load_chunks,
    section_texts,
)
from thematic_redundancy.representation.build_tfidf import (
    MATRIX_FILE_NAMES,
    SECTIONS_FILE_NAME,
    TERMS_FILE_NAME,
    TFIDF_DIR_NAME,
    TfidfBuild,
    build_tfidf,
    load_full_clean_sections,
    load_full_cleaner,
    load_tfidf,
)
from thematic_redundancy.representation.tfidf import zero_rows
from thematic_redundancy.shared.config import AppConfig, load_config

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path("config") / "default.yaml"
INPUT_FILES = (
    FICHAS_FILE_NAME,
    EXCLUSIONS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    CHUNKS_FILE_NAME,
    CHUNKS_MANIFEST_FILE_NAME,
)
DATA_FILES = (SECTIONS_FILE_NAME, TERMS_FILE_NAME, *MATRIX_FILE_NAMES.values())
RUN_FIELDS = {"created_at", "clean_seconds", "vectorize_seconds", "seconds"}
"""Manifest fields that describe a run, not its outputs."""

D26_SHA256 = "8a889062a2f117e2e64bd3e97ad0cd1384b8bce1721dbb0622cb24a53ce1b50f"
"""The D26 pin of ``test_domain_stopwords.py``: a deliberate change of the list updates both."""

SAMPLE_SEED = 20261019
SAMPLE_SECTIONS = 150


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(PROJECT_ROOT / CONFIG_FILE)


@pytest.fixture(scope="module")
def snapshot_id(config: AppConfig) -> str:
    return config.tfidf.expected.snapshot_id


def interim(root: Path, snapshot_id: str) -> Path:
    return root / "data" / INTERIM_DIR_NAME / snapshot_id


def copy_project(root: Path, config: AppConfig, snapshot_id: str) -> Path:
    """Copy the inputs of the build into a new project at ``root``, or skip."""
    source = config.paths.resolve_against(PROJECT_ROOT).data_dir / INTERIM_DIR_NAME / snapshot_id
    if not all((source / name).is_file() for name in INPUT_FILES):
        pytest.skip("the fichas or the chunk table of the configured snapshot have not been built")
    target = interim(root, snapshot_id)
    target.mkdir(parents=True)
    for name in INPUT_FILES:
        shutil.copyfile(source / name, target / name)
    for relative in (CONFIG_FILE, DOMAIN_STOPWORDS_FILE):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, root / relative)
    return root


@pytest.fixture(scope="module")
def first(
    config: AppConfig, snapshot_id: str, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, TfidfBuild]:
    root = copy_project(tmp_path_factory.mktemp("first"), config, snapshot_id)
    return root, build_tfidf(config, root, snapshot_id=snapshot_id)


def digests(directory: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in DATA_FILES
    }


def test_the_real_build_reaches_the_declared_counts_with_the_frozen_stopwords(
    first: tuple[Path, TfidfBuild], config: AppConfig
) -> None:
    manifest = first[1].manifest
    expected = config.tfidf.expected

    assert {
        name: manifest.counts[name] for name in ("documents", "sections", "terms", "nonzeros")
    } == {
        "documents": expected.documents,
        "sections": expected.sections,
        "terms": expected.terms,
        "nonzeros": expected.nonzeros,
    }
    assert manifest.counts["empty_sections"] == 0
    domain = manifest.domain_stopwords
    assert (domain.lemmas, domain.lemmas_sha256) == (89, D26_SHA256)
    cleaner = manifest.cleaner
    assert cleaner.model_name == "es_core_news_md"
    assert cleaner.spacy_stopwords == len(spacy_stopwords())
    assert manifest.document_variant == config.tfidf.document_vector


@pytest.mark.parametrize("variant", ["section_mean", "concatenated"])
def test_the_vectors_follow_the_chunk_table_and_have_unit_rows(
    first: tuple[Path, TfidfBuild], snapshot_id: str, variant: str
) -> None:
    root, built = first
    interim_dir = interim(root, snapshot_id)
    records = load_chunks(interim_dir)
    chunked = {record.doc_code: record.sections_used for record in records}

    loaded = load_tfidf(interim_dir, root, variant=variant)  # type: ignore[arg-type]

    matrix = loaded.matrix
    assert loaded.doc_codes == tuple(chunked)
    assert matrix.shape == (built.manifest.counts["documents"], len(loaded.terms))
    assert matrix.dtype == np.float64 and matrix.format == "csr" and matrix.has_canonical_format
    assert matrix.nnz == built.manifest.counts["nonzeros"] and zero_rows(matrix) == 0
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    assert np.abs(norms - 1.0).max() < 1e-12
    used: dict[str, tuple[str, ...]] = {}
    for section in load_full_clean_sections(interim_dir, root):
        used[section.doc_code] = (*used.get(section.doc_code, ()), section.section)
    assert sum(used[code] != sections for code, sections in chunked.items()) == 0


def test_a_second_build_writes_byte_identical_files(
    first: tuple[Path, TfidfBuild],
    config: AppConfig,
    snapshot_id: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root, built = first
    second_root = copy_project(tmp_path_factory.mktemp("second"), config, snapshot_id)

    again = build_tfidf(config, second_root, snapshot_id=snapshot_id)

    first_dir = interim(root, snapshot_id) / TFIDF_DIR_NAME
    second_dir = interim(second_root, snapshot_id) / TFIDF_DIR_NAME
    assert digests(first_dir) == digests(second_dir) == again.manifest.files
    assert built.manifest.model_dump(exclude=RUN_FIELDS) == again.manifest.model_dump(
        exclude=RUN_FIELDS
    )


def test_cleaning_a_section_does_not_depend_on_its_batch(
    first: tuple[Path, TfidfBuild], config: AppConfig, snapshot_id: str
) -> None:
    root, _ = first
    interim_dir = interim(root, snapshot_id)
    texts = {
        (ficha.doc_code, section): text
        for ficha in load_fichas(interim_dir)
        if ficha.include
        for section, text in section_texts(ficha).items()
    }
    saved = {
        (section.doc_code, section.section): section.tokens
        for section in load_full_clean_sections(interim_dir, root)
    }
    sample = random.Random(SAMPLE_SEED).sample(sorted(saved), SAMPLE_SECTIONS)
    cleaner = load_full_cleaner(load_domain_stopwords(root / DOMAIN_STOPWORDS_FILE))

    alone = cleaner.clean([texts[key] for key in sample])

    assert len(alone) == SAMPLE_SECTIONS
    assert sum(tuple(tokens) != saved[key] for key, tokens in zip(sample, alone, strict=True)) == 0
