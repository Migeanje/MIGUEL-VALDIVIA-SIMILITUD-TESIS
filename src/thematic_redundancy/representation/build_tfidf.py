"""Build the TF-IDF baseline of a snapshot: the lexical document vectors (T15, D28).

:func:`build_tfidf` reads the fichas through
:func:`~thematic_redundancy.corpus.build_fichas.load_fichas` and holds them against the chunk
table (:func:`~thematic_redundancy.representation.build_chunks.load_chunks`), so the baseline
and the embeddings read the same theses and the same sections:

- **Sections** (:func:`thesis_sections`): the texts of
  :func:`~thematic_redundancy.representation.build_chunks.section_texts`, the title without its
  place and year suffix (D10). The doc codes, their order and each thesis's sections must equal
  the chunk table's (:func:`check_parity`).
- **Tokens**: each section is full-cleaned on its own, through the
  :class:`~thematic_redundancy.representation.tfidf.SectionCleaner` port, with the domain
  stopwords of D26 (:data:`~thematic_redundancy.preprocessing.stopwords.DOMAIN_STOPWORDS_FILE`).
  A section left without a token is left out of the vectors and counted.
- **Vectors** (:func:`~thematic_redundancy.representation.tfidf.fit_tfidf`), with the parameters
  of ``tfidf``: ``section_mean``, the one that later steps read by default, and
  ``concatenated``, with the cosine between the two.
- **Checks**: no thesis may lack a term, and the counts must equal ``tfidf.expected``. When a
  check fails, the build stops, names every failed check, and writes nothing.

The outputs go to the gitignored ``data/interim/<snapshot_id>/tfidf/``, never to ``results/``,
since they hold the tokens of the theses:

- ``full_clean_sections.parquet``: one row per section with tokens, in the row order of the
  matrices: the thesis, its row, the section and its tokens;
- ``tfidf_terms.parquet``: one row per column of the matrices: the term, its words, its
  document frequency (theses) and its idf;
- ``documents_section_mean.npz`` and ``documents_concatenated.npz``: one L2-normalized float64
  row per thesis, as canonical CSR, saved uncompressed so that their bytes repeat;
- ``tfidf_manifest.json``: the inputs, the cleaner and both stopword lists, the vectorizer, the
  counts, the cosine between the two variants, each file's SHA-256 and the seconds, without
  text or terms.

The manifest is written last, as the commit point: :func:`load_tfidf` accepts the outputs only
while they match it, and while it matches the current fichas, the current domain stopword list,
:data:`~thematic_redundancy.representation.tfidf.TFIDF_VERSION` and
:data:`~thematic_redundancy.preprocessing.full_cleaner.FULL_CLEANER_VERSION`::

    uv run python -m thematic_redundancy.representation.build_tfidf [--project-root PATH]
        [--snapshot-id ID]
"""

import argparse
import hashlib
import io
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow
import scipy
import sklearn
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
)
from scipy.sparse import csr_matrix, load_npz, save_npz

from thematic_redundancy.corpus.build_fichas import (
    FICHAS_MANIFEST_FILE_NAME,
    load_fichas,
    verify_fichas_outputs,
)
from thematic_redundancy.corpus.ficha import Ficha
from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    Count,
    SnapshotId,
    UtcDatetime,
    check_snapshot_id,
    latest_snapshot_id,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME, Sha256
from thematic_redundancy.preprocessing.full_cleaner import (
    EXCLUDED_COMPONENTS,
    FULL_CLEANER_VERSION,
    KEPT_FORMS,
    LEGAL_ENTITY_SUFFIXES,
    MIN_TOKEN_LENGTH,
    clean_full_batch,
    spanish_pipeline,
)
from thematic_redundancy.preprocessing.stopwords import (
    DOMAIN_STOPWORDS_FILE,
    load_domain_stopwords,
    spacy_stopwords,
)
from thematic_redundancy.representation.build_chunks import BUILD_COMMAND as CHUNKS_BUILD_COMMAND
from thematic_redundancy.representation.build_chunks import (
    ChunkRecord,
    Section,
    chunks_manifest,
    load_chunks,
    section_texts,
)
from thematic_redundancy.representation.tfidf import (
    DTYPE,
    NORM,
    SMOOTH_IDF,
    TFIDF_VERSION,
    USE_IDF,
    CleanerIdentity,
    SectionCleaner,
    TfidfCheckError,
    TfidfMatrices,
    TfidfSettings,
    fit_tfidf,
    lemma_list_sha256,
    variant_agreement,
    zero_rows,
)
from thematic_redundancy.shared.config import (
    AppConfig,
    TfidfCountsConfig,
    TfidfVariant,
    load_config,
)
from thematic_redundancy.shared.filesystem import write_files_atomically, write_text_atomically
from thematic_redundancy.shared.storage import (
    ChecksumMismatchError,
    ParquetTableStore,
    TableStore,
)

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

BUILD_COMMAND = "uv run python -m thematic_redundancy.representation.build_tfidf"
"""Command that writes the TF-IDF outputs; the reader's error messages point to it."""

TFIDF_DIR_NAME = "tfidf"
TFIDF_MANIFEST_FILE_NAME = "tfidf_manifest.json"
SECTIONS_FILE_NAME = "full_clean_sections.parquet"
TERMS_FILE_NAME = "tfidf_terms.parquet"
MATRIX_FILE_NAMES: dict[TfidfVariant, str] = {
    "section_mean": "documents_section_mean.npz",
    "concatenated": "documents_concatenated.npz",
}

UNIT_NORM_TOLERANCE = 1e-9
"""Largest distance from 1 that the reader accepts in the L2 norm of a row."""

LoadCleaner = Callable[[frozenset[str]], SectionCleaner]
"""Loads the cleaner, given the domain stopwords."""

ThesisSections = tuple[str, dict[Section, str]]
"""A thesis's doc code and the texts of its sections, in section order."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class CleanedSection(_Model):
    """One row of ``full_clean_sections.parquet``: the tokens of a section that has some."""

    doc_code: str
    document_row: Count
    """Row of the section's thesis in the matrices."""
    section: Section
    n_tokens: Annotated[StrictInt, Field(ge=1)]
    tokens: tuple[str, ...]


class TfidfTerm(_Model):
    """One row of ``tfidf_terms.parquet``: a column of the matrices."""

    column: Count
    term: str
    n_words: Annotated[StrictInt, Field(ge=1, le=2)]
    document_frequency: Annotated[StrictInt, Field(ge=1)]
    """Theses that hold the term."""
    idf: StrictFloat


class CleanerRecord(_Model):
    """The full cleaner that gave the tokens: its rules, its spaCy model and its stopwords."""

    full_cleaner_version: Annotated[StrictInt, Field(ge=1)]
    spacy_version: str
    model_name: str
    model_version: str
    excluded_components: tuple[str, ...]
    min_token_length: Count
    kept_forms: tuple[str, ...]
    legal_entity_suffixes: tuple[str, ...]
    spacy_stopwords: Count
    spacy_stopwords_sha256: Sha256
    """SHA-256 of spaCy's stopwords, sorted and joined by line feeds."""


class DomainStopwordsRecord(_Model):
    """The domain stopword list (D26) that the cleaner dropped."""

    path: str
    """Relative to the project root."""
    lemmas: Count
    lemmas_sha256: Sha256
    """SHA-256 of the lemmas, sorted and joined by line feeds: the D26 pin, which the reader
    compares with the current list."""
    file_sha256: Sha256


class VectorizerRecord(_Model):
    """The parameters of the vectorizer (D28 C3)."""

    max_ngram: Annotated[StrictInt, Field(ge=1, le=2)]
    ngrams_within_sections: StrictBool
    min_df: Annotated[StrictInt, Field(ge=1)]
    max_df: StrictFloat
    document_frequency_unit: str
    sublinear_tf: StrictBool
    norm: str
    use_idf: StrictBool
    smooth_idf: StrictBool
    dtype: str


class VariantAgreementRecord(_Model):
    """Cosine between the ``section_mean`` and ``concatenated`` vectors of each thesis."""

    threshold: StrictFloat
    minimum: StrictFloat
    median: StrictFloat
    below: Count
    """Theses whose cosine is below ``threshold``."""


class TfidfManifest(_Model):
    """Content of ``tfidf_manifest.json``: provenance, counts and time, without text or terms."""

    snapshot_id: SnapshotId
    created_at: UtcDatetime
    tfidf_version: Annotated[StrictInt, Field(ge=1)]
    fichas_sha256: Sha256
    chunker_version: Annotated[StrictInt, Field(ge=1)]
    chunks_sha256: Sha256
    """The chunk table that the sections were checked against; provenance only."""
    cleaner: CleanerRecord
    domain_stopwords: DomainStopwordsRecord
    versions: dict[str, str]
    """Python and the libraries that the bytes depend on."""
    vectorizer: VectorizerRecord
    document_variant: TfidfVariant
    """The matrix that :func:`load_tfidf` gives unless asked for the other."""
    counts: dict[str, Count]
    sections_used: dict[str, Count]
    """Theses per combination of sections, such as ``title+abstract``."""
    variant_agreement: VariantAgreementRecord
    files: dict[str, Sha256]
    """SHA-256 of each output file."""
    clean_seconds: StrictFloat
    vectorize_seconds: StrictFloat
    seconds: StrictFloat


@dataclass(frozen=True)
class TfidfBuild:
    """Outcome of :func:`build_tfidf`."""

    snapshot_id: str
    manifest: TfidfManifest
    tfidf_dir: Path


@dataclass(frozen=True)
class DocumentTfidf:
    """The saved TF-IDF vectors of a snapshot, as :func:`load_tfidf` reads them."""

    variant: TfidfVariant
    doc_codes: tuple[str, ...]
    matrix: csr_matrix
    """One L2-normalized float64 row per thesis of :attr:`doc_codes`, of :attr:`variant`."""
    terms: tuple[str, ...]
    """The term of each column."""
    idf: np.ndarray


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _silent(_: str) -> None:
    pass


# The cleaner


class SpacyFullCleaner:
    """The :class:`~thematic_redundancy.representation.tfidf.SectionCleaner` of the project:
    :func:`~thematic_redundancy.preprocessing.full_cleaner.clean_full_batch` with the domain
    stopwords, over spaCy's Spanish pipeline, loaded once per process."""

    def __init__(self, domain_stopwords: frozenset[str]) -> None:
        import spacy

        pipeline = spanish_pipeline()
        stopwords = spacy_stopwords()
        self._domain_stopwords = domain_stopwords
        self.identity = CleanerIdentity(
            spacy_version=spacy.__version__,
            model_name=f"{pipeline.meta['lang']}_{pipeline.meta['name']}",
            model_version=str(pipeline.meta["version"]),
            spacy_stopwords=len(stopwords),
            spacy_stopwords_sha256=lemma_list_sha256(stopwords),
        )

    def clean(self, texts: Sequence[str]) -> list[list[str]]:
        return clean_full_batch(texts, domain_stopwords=self._domain_stopwords)


def load_full_cleaner(domain_stopwords: frozenset[str]) -> SectionCleaner:
    """Return the project's cleaner, with ``domain_stopwords``.

    Raises:
        OSError: if the spaCy model is not installed.
    """
    return SpacyFullCleaner(domain_stopwords)


# Sections


def thesis_sections(fichas: Iterable[Ficha]) -> tuple[ThesisSections, ...]:
    """Return the doc code and the section texts of each included thesis, in order."""
    return tuple((ficha.doc_code, section_texts(ficha)) for ficha in fichas if ficha.include)


def check_parity(theses: Sequence[ThesisSections], records: Sequence[ChunkRecord]) -> None:
    """Check that ``theses`` are those of the chunk table, in its order, each with the sections
    that the chunk table records for it.

    Raises:
        TfidfCheckError: naming every difference, in counts.
    """
    used = {doc_code: tuple(sections) for doc_code, sections in theses}
    chunked = {record.doc_code: record.sections_used for record in records}
    problems = []
    missing = len(used.keys() - chunked.keys())
    if missing:
        problems.append(
            f"{missing} included thes{'es are' if missing > 1 else 'is is'} missing from the "
            "chunk table"
        )
    extra = len(chunked.keys() - used.keys())
    if extra:
        problems.append(
            f"{extra} thes{'es' if extra > 1 else 'is'} of the chunk table "
            f"{'are' if extra > 1 else 'is'} not included"
        )
    if not missing and not extra and list(used) != list(chunked):
        problems.append("the theses are in another order than in the chunk table")
    differ = sum(sections != chunked[code] for code, sections in used.items() if code in chunked)
    if differ:
        problems.append(
            f"{differ} thes{'es use' if differ > 1 else 'is uses'} other sections than the "
            "chunk table records"
        )
    if problems:
        raise TfidfCheckError(
            f"the sections to vectorize do not match the chunk table: {'; '.join(problems)}; "
            f"run {CHUNKS_BUILD_COMMAND} again"
        )


def _cleaned_sections(
    theses: Sequence[ThesisSections], tokens: Sequence[Sequence[str]]
) -> tuple[list[CleanedSection], list[list[list[str]]]]:
    """Return the rows of the sections with tokens, and each thesis's token lists, from the
    tokens of every section of ``theses`` in order."""
    sections_of = [
        (row, doc_code, section)
        for row, (doc_code, texts) in enumerate(theses)
        for section in texts
    ]
    rows = []
    documents: list[list[list[str]]] = [[] for _ in theses]
    for (row, doc_code, section), section_tokens in zip(sections_of, tokens, strict=True):
        if section_tokens:
            rows.append(
                CleanedSection(
                    doc_code=doc_code,
                    document_row=row,
                    section=section,
                    n_tokens=len(section_tokens),
                    tokens=tuple(section_tokens),
                )
            )
            documents[row].append(list(section_tokens))
    return rows, documents


# Checks


def count_tfidf(
    theses: Sequence[ThesisSections], rows: Sequence[CleanedSection], matrices: TfidfMatrices
) -> dict[str, int]:
    """Return the counts that the manifest records, those of ``tfidf.expected`` among them."""
    sections = sum(len(texts) for _, texts in theses)
    bigrams = sum(" " in term for term in matrices.terms)
    return {
        "documents": len(theses),
        "sections": len(rows),
        "empty_sections": sections - len(rows),
        "tokens": sum(row.n_tokens for row in rows),
        "terms": len(matrices.terms),
        "unigrams": len(matrices.terms) - bigrams,
        "bigrams": bigrams,
        "nonzeros": matrices.concatenated.nnz,
    }


def check_tfidf(counts: dict[str, int], zero: int, expected: TfidfCountsConfig) -> None:
    """Check that no thesis lacks a term and that the counts equal ``expected``.

    Raises:
        TfidfCheckError: naming every check that fails.
    """
    problems = []
    if zero:
        problems.append(
            f"{zero} thes{'es have' if zero > 1 else 'is has'} no term, so "
            f"{'they have' if zero > 1 else 'it has'} no cosine"
        )
    for name in ("documents", "sections", "terms", "nonzeros"):
        declared = getattr(expected, name)
        if counts[name] != declared:
            problems.append(f"{name} is {counts[name]}, but tfidf.expected declares {declared}")
    if problems:
        raise TfidfCheckError(
            f"the TF-IDF of snapshot {expected.snapshot_id} fails {len(problems)} "
            f"check{'s' if len(problems) > 1 else ''}: {'; '.join(problems)}"
        )


# Use case


def build_tfidf(
    config: AppConfig,
    project_root: Path,
    *,
    snapshot_id: str | None = None,
    load_cleaner: LoadCleaner = load_full_cleaner,
    store: TableStore | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    report: Callable[[str], None] = _silent,
) -> TfidfBuild:
    """Clean, vectorize, check and save the included theses of a snapshot, the one harvested
    last by default.

    ``load_cleaner`` loads the cleaner with the domain stopwords of the project's list;
    ``report`` receives one line of progress at a time.

    Raises:
        FileNotFoundError: if the snapshot, the fichas outputs, the chunk table or the domain
            stopword list is missing.
        TfidfCheckError: if the sections differ from the chunk table's, a thesis has no term,
            or a count differs from ``tfidf.expected``; nothing is written then.
        ValueError: if ``tfidf.expected`` belongs to another snapshot, or the fichas outputs or
            the chunk table are damaged.
        OSError: if reading or writing the disk fails, or the cleaner cannot be loaded.
    """
    started = timer()
    store = store or ParquetTableStore()
    data_dir = config.paths.resolve_against(project_root).data_dir
    snapshot_id = check_snapshot_id(snapshot_id or latest_snapshot_id(data_dir / RAW_DIR_NAME))
    settings = config.tfidf
    expected = settings.expected
    if snapshot_id != expected.snapshot_id:
        raise ValueError(
            f"tfidf.expected declares the counts of snapshot '{expected.snapshot_id}', not of "
            f"'{snapshot_id}'; vectorize that snapshot and declare its counts first"
        )
    interim_dir = data_dir / INTERIM_DIR_NAME / snapshot_id
    fichas_sha256 = verify_fichas_outputs(interim_dir).fichas_sha256
    theses = thesis_sections(load_fichas(interim_dir, store=store))
    chunks = chunks_manifest(interim_dir)
    check_parity(theses, load_chunks(interim_dir, store=store))
    domain, domain_record = _domain_stopwords(project_root)
    report("loading the full cleaner")
    cleaner = load_cleaner(domain)
    texts = [text for _, sections in theses for text in sections.values()]
    report(f"cleaning {len(texts)} sections of {len(theses)} theses")
    clean_started = timer()
    tokens = cleaner.clean(texts)
    clean_seconds = timer() - clean_started
    rows, documents = _cleaned_sections(theses, tokens)
    report("vectorizing")
    vectorize_started = timer()
    matrices = fit_tfidf(
        documents,
        TfidfSettings(
            max_ngram=settings.max_ngram,
            min_df=settings.min_df,
            max_df=settings.max_df,
            sublinear_tf=settings.sublinear_tf,
        ),
    )
    counts = count_tfidf(theses, rows, matrices)
    check_tfidf(counts, zero_rows(matrices.concatenated), expected)
    agreement = variant_agreement(
        matrices.section_mean, matrices.concatenated, settings.sensitivity_cosine_threshold
    )
    vectorize_seconds = timer() - vectorize_started
    tfidf_dir = interim_dir / TFIDF_DIR_NAME
    files = _write_outputs(tfidf_dir, rows, matrices, store)
    manifest = TfidfManifest(
        snapshot_id=snapshot_id,
        created_at=clock().astimezone(UTC),
        tfidf_version=TFIDF_VERSION,
        fichas_sha256=fichas_sha256,
        chunker_version=chunks.chunker_version,
        chunks_sha256=chunks.chunks_sha256,
        cleaner=_cleaner_record(cleaner.identity),
        domain_stopwords=domain_record,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
        },
        vectorizer=VectorizerRecord(
            max_ngram=settings.max_ngram,
            ngrams_within_sections=True,
            min_df=settings.min_df,
            max_df=settings.max_df,
            document_frequency_unit="thesis",
            sublinear_tf=settings.sublinear_tf,
            norm=NORM,
            use_idf=USE_IDF,
            smooth_idf=SMOOTH_IDF,
            dtype=np.dtype(DTYPE).name,
        ),
        document_variant=settings.document_vector,
        counts=counts,
        sections_used=dict(Counter("+".join(sections) for _, sections in theses)),
        variant_agreement=VariantAgreementRecord(
            threshold=agreement.threshold,
            minimum=round(agreement.minimum, 6),
            median=round(agreement.median, 6),
            below=agreement.below,
        ),
        files=files,
        clean_seconds=round(max(clean_seconds, 0.0), 3),
        vectorize_seconds=round(max(vectorize_seconds, 0.0), 3),
        seconds=round(max(timer() - started, 0.0), 3),
    )
    # The commit point: until it is written, the manifest describes the previous outputs.
    write_text_atomically(tfidf_dir / TFIDF_MANIFEST_FILE_NAME, manifest.model_dump_json(indent=2))
    return TfidfBuild(snapshot_id=snapshot_id, manifest=manifest, tfidf_dir=tfidf_dir)


def _domain_stopwords(project_root: Path) -> tuple[frozenset[str], DomainStopwordsRecord]:
    path = project_root / DOMAIN_STOPWORDS_FILE
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    lemmas = load_domain_stopwords(path)
    return lemmas, DomainStopwordsRecord(
        path=DOMAIN_STOPWORDS_FILE.as_posix(),
        lemmas=len(lemmas),
        lemmas_sha256=lemma_list_sha256(lemmas),
        file_sha256=file_sha256,
    )


def _cleaner_record(identity: CleanerIdentity) -> CleanerRecord:
    return CleanerRecord(
        full_cleaner_version=FULL_CLEANER_VERSION,
        spacy_version=identity.spacy_version,
        model_name=identity.model_name,
        model_version=identity.model_version,
        excluded_components=EXCLUDED_COMPONENTS,
        min_token_length=MIN_TOKEN_LENGTH,
        kept_forms=tuple(sorted(KEPT_FORMS)),
        legal_entity_suffixes=tuple(sorted(LEGAL_ENTITY_SUFFIXES)),
        spacy_stopwords=identity.spacy_stopwords,
        spacy_stopwords_sha256=identity.spacy_stopwords_sha256,
    )


def _write_outputs(
    tfidf_dir: Path, rows: Sequence[CleanedSection], matrices: TfidfMatrices, store: TableStore
) -> dict[str, str]:
    """Write the tables, then the matrices, and return the SHA-256 of each file as written."""
    tfidf_dir.mkdir(parents=True, exist_ok=True)
    store.write(rows, tfidf_dir / SECTIONS_FILE_NAME, model_cls=CleanedSection)
    terms = [
        TfidfTerm(
            column=column,
            term=term,
            n_words=term.count(" ") + 1,
            document_frequency=int(frequency),
            idf=float(idf),
        )
        for column, (term, frequency, idf) in enumerate(
            zip(matrices.terms, matrices.document_frequency, matrices.idf, strict=True)
        )
    ]
    store.write(terms, tfidf_dir / TERMS_FILE_NAME, model_cls=TfidfTerm)
    write_files_atomically(
        tfidf_dir,
        {
            MATRIX_FILE_NAMES["section_mean"]: _npz(matrices.section_mean),
            MATRIX_FILE_NAMES["concatenated"]: _npz(matrices.concatenated),
        },
    )
    names = (SECTIONS_FILE_NAME, TERMS_FILE_NAME, *MATRIX_FILE_NAMES.values())
    return {name: hashlib.sha256((tfidf_dir / name).read_bytes()).hexdigest() for name in names}


def _npz(matrix: csr_matrix) -> bytes:
    buffer = io.BytesIO()
    save_npz(buffer, matrix, compressed=False)
    return buffer.getvalue()


# Reading the outputs


def tfidf_manifest(interim_dir: Path) -> TfidfManifest:
    """Return the TF-IDF manifest of the snapshot whose interim directory is ``interim_dir``.

    Raises:
        FileNotFoundError: if there is none.
        ValueError: if it is damaged; the message names no content of the file.
    """
    path = interim_dir / TFIDF_DIR_NAME / TFIDF_MANIFEST_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name} in {path.parent}; run {BUILD_COMMAND} first")
    try:
        return TfidfManifest.model_validate_json(path.read_bytes())
    except ValidationError:
        # Not chained: the validation error may quote the content of the file.
        raise ValueError(f"{path} is damaged; run {BUILD_COMMAND} again") from None


def load_tfidf(
    interim_dir: Path,
    project_root: Path,
    *,
    variant: TfidfVariant | None = None,
    store: TableStore | None = None,
) -> DocumentTfidf:
    """Return the saved TF-IDF vectors, once they match their manifest, and the manifest
    matches the current inputs and code.

    ``variant`` picks the matrix; by default, the one that the manifest records from
    ``tfidf.document_vector``. ``project_root`` holds the current domain stopword list. Each
    file is read once, then checked, then parsed.

    Raises:
        FileNotFoundError: if the manifest, the fichas outputs, the domain stopword list or an
            output file is missing.
        ValueError: if the manifest is damaged; if the outputs are stale, built from other
            fichas or another snapshot, with another domain stopword list, or by another
            :data:`~thematic_redundancy.representation.tfidf.TFIDF_VERSION` or
            :data:`~thematic_redundancy.preprocessing.full_cleaner.FULL_CLEANER_VERSION`; or if
            a file differs from the manifest, or a matrix is not one canonical, L2-normalized
            float64 CSR row per thesis and one column per term.
    """
    manifest = tfidf_manifest(interim_dir)
    _check_current(manifest, interim_dir, project_root)
    store = store or ParquetTableStore()
    sections = _read_table(store, CleanedSection, interim_dir, SECTIONS_FILE_NAME, manifest)
    terms = _read_table(store, TfidfTerm, interim_dir, TERMS_FILE_NAME, manifest)
    doc_codes = tuple(dict.fromkeys(section.doc_code for section in sections))
    variant = variant or manifest.document_variant
    path = interim_dir / TFIDF_DIR_NAME / MATRIX_FILE_NAMES[variant]
    return DocumentTfidf(
        variant=variant,
        doc_codes=doc_codes,
        matrix=_read_matrix(path, manifest, len(doc_codes), len(terms)),
        terms=tuple(term.term for term in terms),
        idf=np.array([term.idf for term in terms], dtype=DTYPE),
    )


def load_full_clean_sections(
    interim_dir: Path, project_root: Path, *, store: TableStore | None = None
) -> tuple[CleanedSection, ...]:
    """Return the saved tokens of each section, in the row order of the matrices, as
    :func:`load_tfidf` checks them.

    Raises:
        FileNotFoundError: as :func:`load_tfidf` does.
        ValueError: as :func:`load_tfidf` does.
    """
    manifest = tfidf_manifest(interim_dir)
    _check_current(manifest, interim_dir, project_root)
    return _read_table(
        store or ParquetTableStore(), CleanedSection, interim_dir, SECTIONS_FILE_NAME, manifest
    )


def _check_current(manifest: TfidfManifest, interim_dir: Path, project_root: Path) -> None:
    """Refuse outputs that the current fichas, domain stopwords or code would not give.

    ``manifest`` must record the snapshot and the fichas table's SHA-256 of the fichas manifest
    in ``interim_dir``, the lemma SHA-256 of the domain stopword list under ``project_root``,
    and the code's :data:`TFIDF_VERSION` and :data:`FULL_CLEANER_VERSION`. Not compared: the
    chunk table, whose new versions may give the same sections, and the spaCy model and its
    stopwords, which would mean importing spaCy; a change of either moves the counts that the
    build checks against ``tfidf.expected``.

    Raises:
        FileNotFoundError: if the fichas outputs or the domain stopword list are missing.
        ValueError: if the fichas outputs are damaged, or naming each field that differs with
            both its values.
    """
    fichas = verify_fichas_outputs(interim_dir)
    lemmas = load_domain_stopwords(project_root / DOMAIN_STOPWORDS_FILE)
    current = (
        ("snapshot_id", manifest.snapshot_id, fichas.snapshot_id, FICHAS_MANIFEST_FILE_NAME),
        ("fichas_sha256", manifest.fichas_sha256, fichas.fichas_sha256, FICHAS_MANIFEST_FILE_NAME),
        ("tfidf_version", manifest.tfidf_version, TFIDF_VERSION, "the code"),
        (
            "cleaner.full_cleaner_version",
            manifest.cleaner.full_cleaner_version,
            FULL_CLEANER_VERSION,
            "the code",
        ),
        (
            "domain_stopwords.lemmas_sha256",
            manifest.domain_stopwords.lemmas_sha256,
            lemma_list_sha256(lemmas),
            DOMAIN_STOPWORDS_FILE.as_posix(),
        ),
    )
    stale = [
        f"{name} is {saved} in {TFIDF_MANIFEST_FILE_NAME} but {value} in {source}"
        for name, saved, value, source in current
        if saved != value
    ]
    if stale:
        raise ValueError(
            f"the TF-IDF outputs in {interim_dir / TFIDF_DIR_NAME} are stale: "
            f"{'; '.join(stale)}; run {BUILD_COMMAND} again"
        )


def _read_table[M: BaseModel](
    store: TableStore, model_cls: type[M], interim_dir: Path, name: str, manifest: TfidfManifest
) -> tuple[M, ...]:
    path = interim_dir / TFIDF_DIR_NAME / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {BUILD_COMMAND}")
    try:
        return store.read_verified(model_cls, path, sha256=manifest.files.get(name, ""))
    except ChecksumMismatchError:
        raise ValueError(
            f"{path} differs from {TFIDF_MANIFEST_FILE_NAME}; run {BUILD_COMMAND} again"
        ) from None


def _read_matrix(path: Path, manifest: TfidfManifest, rows: int, columns: int) -> csr_matrix:
    """Return the matrix at ``path`` once it has the SHA-256 that ``manifest`` lists for it and
    holds ``rows`` canonical, L2-normalized float64 CSR rows of ``columns`` terms."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {BUILD_COMMAND}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.files.get(path.name):
        raise ValueError(
            f"{path} differs from {TFIDF_MANIFEST_FILE_NAME}; run {BUILD_COMMAND} again"
        )
    matrix = load_npz(io.BytesIO(content))
    if not _well_formed(matrix, rows, columns):
        raise ValueError(
            f"{path} does not hold {rows} canonical, L2-normalized float64 CSR rows of "
            f"{columns} terms; run {BUILD_COMMAND} again"
        )
    return matrix


def _well_formed(matrix: csr_matrix, rows: int, columns: int) -> bool:
    if matrix.format != "csr" or matrix.dtype != DTYPE or matrix.shape != (rows, columns):
        return False
    if not matrix.has_canonical_format:
        return False
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return bool(np.all(np.abs(norms - 1.0) <= UNIT_NORM_TOLERANCE))


# Command line


def describe(built: TfidfBuild) -> str:
    """Describe the build in counts only."""
    manifest = built.manifest
    counts = manifest.counts
    vectorizer = manifest.vectorizer
    agreement = manifest.variant_agreement
    cleaner = manifest.cleaner
    domain = manifest.domain_stopwords
    return "\n".join(
        [
            f"Snapshot {built.snapshot_id}: {counts['documents']} documents, "
            f"{counts['sections']} sections ({counts['empty_sections']} empty), "
            f"{counts['tokens']} tokens",
            "Sections used: "
            + "; ".join(f"{used} {number}" for used, number in manifest.sections_used.items()),
            f"Terms: {counts['terms']} ({counts['unigrams']} unigrams, {counts['bigrams']} "
            f"bigrams within sections); min_df {vectorizer.min_df} theses, max_df "
            f"{vectorizer.max_df}, {'sublinear' if vectorizer.sublinear_tf else 'raw'} tf",
            f"Nonzeros: {counts['nonzeros']} per matrix; 0 zero rows",
            f"section_mean vs concatenated, cosine per document: min {agreement.minimum:.4f}, "
            f"median {agreement.median:.4f}, {agreement.below} below {agreement.threshold}",
            f"Document vector read downstream: {manifest.document_variant}",
            f"Full cleaner version {cleaner.full_cleaner_version}: {cleaner.model_name} "
            f"{cleaner.model_version} (spaCy {cleaner.spacy_version}), "
            f"{cleaner.spacy_stopwords} spaCy stopwords; {domain.lemmas} domain stopwords, "
            f"lemma SHA-256 {domain.lemmas_sha256[:12]}",
            f"TF-IDF {built.tfidf_dir}",
            f"Seconds: {manifest.seconds:.1f} (cleaning {manifest.clean_seconds:.1f}, "
            f"vectorizing {manifest.vectorize_seconds:.1f})",
        ]
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.representation.build_tfidf",
        description=(
            "Full-clean the sections of every included thesis, vectorize them as TF-IDF, check "
            f"the declared counts, and save them under data/interim/<snapshot>/{TFIDF_DIR_NAME}/ "
            "with their manifest."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--snapshot-id", help="snapshot to vectorize (default: the snapshot harvested last)"
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    load_cleaner: LoadCleaner = load_full_cleaner,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and
    130 after an interruption."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    try:
        config = load_config(project_root / CONFIG_PATH)
        built = build_tfidf(
            config,
            project_root,
            snapshot_id=arguments.snapshot_id,
            load_cleaner=load_cleaner,
            clock=clock,
            timer=timer,
            report=_progress,
        )
    except KeyboardInterrupt:
        print(
            "interrupted: TF-IDF outputs that do not match their manifest are refused by "
            "readers; run the command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(describe(built))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
