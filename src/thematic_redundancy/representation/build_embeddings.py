"""Embed the chunk table of a snapshot with every configured model (T14).

:func:`build_embeddings` reads the chunks through
:func:`~thematic_redundancy.representation.build_chunks.load_chunks` and, for each model of
``embedding.models``:

- **Chunk vectors** (:func:`embed_texts`). Each distinct chunk text is encoded once, through
  the sentence-encoder port, unless the model's cache already holds its vector under the D27
  C6 key (:mod:`~.vector_cache`); a rerun therefore encodes nothing. The cache keeps the raw
  vectors, and it is saved after every :data:`CHECKPOINT_CHUNKS` new vectors, so an
  interrupted run keeps what it embedded.
- **Section and document vectors** (:func:`~.vectors.pool_chunk_vectors`). The section vector
  is the mean of its chunk vectors; the ``section_mean`` document vector, the equal-weight mean
  of the sections in ``sections_used``, L2-normalized (D09, C4); and the ``plain_mean`` one,
  the mean of all the thesis's chunk vectors, L2-normalized, the sensitivity variant (C7).
- **Checks**. The counts must equal ``embedding.expected`` before any model loads; each model
  must read 128 tokens (D09) and no fewer than the longest chunk has.

The outputs go to the gitignored ``data/interim/<snapshot_id>/embeddings/``, never to
``results/``:

- ``embedded_sections.parquet``: one row per section, the row order of every section matrix,
  with its thesis's row in the document matrices and its number of chunks;
- ``<model id, with "/" as "--">/``: ``documents_section_mean.npy`` and
  ``documents_plain_mean.npy``, one L2-normalized row per thesis, and ``sections.npy``, one raw
  row per section, all float32;
- ``embeddings_manifest.json``: the chunk table's SHA-256 and chunker version, the cache-key
  scheme, the encoder settings, each model's revision, dimension, cache use and output
  SHA-256 values, the configured document vector, the cosine between the two variants, the
  counts and the seconds, without text.

The manifest is written last, as the commit point: :func:`load_embeddings` accepts the outputs
only while they match it::

    uv run python -m thematic_redundancy.representation.build_embeddings [--project-root PATH]
        [--snapshot-id ID]
"""

import argparse
import hashlib
import io
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError

from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    Count,
    SnapshotId,
    UtcDatetime,
    check_snapshot_id,
    latest_snapshot_id,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME, Sha256
from thematic_redundancy.representation.build_chunks import (
    ChunkRecord,
    Section,
    chunks_manifest,
    load_chunks,
)
from thematic_redundancy.representation.st_encoder import load_sentence_encoder
from thematic_redundancy.representation.vector_cache import (
    CACHE_DIR,
    CACHE_KEY_SCHEME,
    ChunkVectorCache,
    cache_key,
    model_slug,
)
from thematic_redundancy.representation.vectors import (
    EncoderIdentity,
    EncoderSettings,
    PooledVectors,
    SentenceEncoder,
    VariantAgreement,
    pool_chunk_vectors,
    variant_agreement,
)
from thematic_redundancy.shared.config import (
    MODEL_MAX_SEQ_LENGTH,
    AppConfig,
    DocumentVariant,
    EmbeddingCountsConfig,
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

BUILD_COMMAND = "uv run python -m thematic_redundancy.representation.build_embeddings"
"""Command that writes the embeddings; the reader's error messages point to it."""

EMBEDDINGS_DIR_NAME = "embeddings"
EMBEDDINGS_MANIFEST_FILE_NAME = "embeddings_manifest.json"
SECTIONS_FILE_NAME = "embedded_sections.parquet"
SECTION_VECTORS_FILE_NAME = "sections.npy"
DOCUMENT_FILE_NAMES: dict[DocumentVariant, str] = {
    "section_mean": "documents_section_mean.npy",
    "plain_mean": "documents_plain_mean.npy",
}

EMBEDDER_VERSION = 1
"""Version of the pooling rules; it changes whenever the same chunk vectors would give other
section or document vectors."""

CHECKPOINT_CHUNKS = 1024
"""New chunk vectors embedded between two saves of a model's cache."""

LoadEncoder = Callable[[str, EncoderSettings], SentenceEncoder]


class EmbeddingsCheckError(ValueError):
    """The chunks or a model break a rule that the build enforces, so nothing was written."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class EmbeddedSection(_Model):
    """One row of ``embedded_sections.parquet``: a section, in the row order of every section
    matrix."""

    doc_code: str
    document_row: Count
    """Row of the section's thesis in the document matrices."""
    section: Section
    n_chunks: Annotated[StrictInt, Field(ge=1)]


class VariantAgreementRecord(_Model):
    """Cosine between the ``section_mean`` and ``plain_mean`` vectors of each thesis."""

    threshold: StrictFloat
    minimum: StrictFloat
    median: StrictFloat
    below: Count
    """Theses whose cosine is below ``threshold``."""


class ModelEmbeddingsRecord(_Model):
    """The embeddings of one model: its identity, its cache use and its output files."""

    model_id: str
    revision: str
    max_seq_length: Count
    dimension: Annotated[StrictInt, Field(ge=1)]
    normalize: bool
    """Whether the chunk vectors were normalized when encoded; part of the cache key."""
    directory: str
    files: dict[str, Sha256]
    """SHA-256 of each output file in ``directory``."""
    cache_file: str
    """The model's cache file, relative to ``paths.data_dir``."""
    distinct_texts: Count
    from_cache: Count
    embedded: Count
    """Distinct texts encoded by this run; ``from_cache`` + ``embedded`` = ``distinct_texts``."""
    embed_seconds: StrictFloat
    variant_agreement: VariantAgreementRecord


class EncoderSettingsRecord(_Model):
    batch_size: Annotated[StrictInt, Field(ge=1)]
    device: str
    threads: Annotated[StrictInt, Field(ge=1)]


class EmbeddingsManifest(_Model):
    """Content of ``embeddings_manifest.json``: provenance, counts and time, without text."""

    snapshot_id: SnapshotId
    created_at: UtcDatetime
    embedder_version: Annotated[StrictInt, Field(ge=1)]
    chunker_version: Annotated[StrictInt, Field(ge=1)]
    chunks_sha256: Sha256
    cache_key_scheme: str
    cache_dir: str
    """Directory of the cache files, relative to ``paths.data_dir``."""
    encoder: EncoderSettingsRecord
    document_variant: DocumentVariant
    """The document vector that :func:`load_embeddings` gives unless asked for the other."""
    sections_file: str
    sections_sha256: Sha256
    counts: dict[str, Count]
    models: tuple[ModelEmbeddingsRecord, ...]
    seconds: StrictFloat


@dataclass(frozen=True)
class EmbeddedTexts:
    """Outcome of :func:`embed_texts`."""

    vectors: np.ndarray
    """One float32 row per text, in order."""
    distinct_texts: int
    from_cache: int
    embedded: int


@dataclass(frozen=True)
class EmbeddingsBuild:
    """Outcome of :func:`build_embeddings`."""

    snapshot_id: str
    manifest: EmbeddingsManifest
    embeddings_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class DocumentEmbeddings:
    """The saved embeddings of one model, as :func:`load_embeddings` reads them."""

    model_id: str
    revision: str
    variant: DocumentVariant
    doc_codes: tuple[str, ...]
    vectors: np.ndarray
    """One L2-normalized float32 row per thesis of :attr:`doc_codes`, of :attr:`variant`."""
    sections: tuple[tuple[str, str], ...]
    """``(doc_code, section)`` of each row of :attr:`section_vectors`."""
    section_vectors: np.ndarray
    """The raw mean of the chunk vectors of each section, float32."""


@dataclass(frozen=True)
class _ModelOutcome:
    identity: EncoderIdentity
    embedded: EmbeddedTexts
    seconds: float
    pooled: PooledVectors
    agreement: VariantAgreement


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _silent(_: str) -> None:
    pass


# Embedding


def embed_texts(
    texts: Sequence[str],
    encoder: SentenceEncoder,
    cache: ChunkVectorCache,
    *,
    checkpoint: int = CHECKPOINT_CHUNKS,
    report: Callable[[str], None] = _silent,
) -> EmbeddedTexts:
    """Return the raw vector of each of ``texts``, encoding only the distinct texts whose
    vector ``cache`` lacks, ``checkpoint`` at a time, and saving the cache after each group.

    Raises:
        ValueError: if the encoder gives other than one finite vector of its dimension per text.
    """
    identity = encoder.identity
    keys = [
        cache_key(
            model_id=identity.model_id,
            model_revision=identity.revision,
            max_seq_length=identity.max_seq_length,
            normalize=False,
            text=text,
        )
        for text in texts
    ]
    text_of = dict(zip(keys, texts, strict=True))
    missing = cache.missing(keys)
    for start in range(0, len(missing), checkpoint):
        group = missing[start : start + checkpoint]
        try:
            cache.add(group, encoder.encode([text_of[key] for key in group]))
        except ValueError as error:
            raise ValueError(f"{identity.model_id} gave {error}") from None
        cache.save()
        report(f"{identity.model_id}: embedded {start + len(group)} of {len(missing)} new texts")
    if not missing:
        report(f"{identity.model_id}: all {len(text_of)} distinct texts were in the cache")
    return EmbeddedTexts(
        vectors=cache.vectors(keys),
        distinct_texts=len(text_of),
        from_cache=len(text_of) - len(missing),
        embedded=len(missing),
    )


def check_counts(records: Sequence[ChunkRecord], expected: EmbeddingCountsConfig) -> dict[str, int]:
    """Return the documents and the chunks of ``records``, once they equal ``expected``.

    Raises:
        EmbeddingsCheckError: naming every count that differs.
    """
    counts = {"documents": len({record.doc_code for record in records}), "chunks": len(records)}
    problems = [
        f"{name} is {value}, but embedding.expected declares {getattr(expected, name)}"
        for name, value in counts.items()
        if value != getattr(expected, name)
    ]
    if problems:
        raise EmbeddingsCheckError(
            f"the chunks of snapshot {expected.snapshot_id} fail {len(problems)} "
            f"check{'s' if len(problems) > 1 else ''}: {'; '.join(problems)}"
        )
    return counts


def _check_model(identity: EncoderIdentity, longest_chunk: int) -> None:
    """Refuse a model that does not read the 128 tokens of D09, or that would truncate a chunk."""
    if identity.max_seq_length != MODEL_MAX_SEQ_LENGTH:
        raise EmbeddingsCheckError(
            f"{identity.model_id} reads up to {identity.max_seq_length} tokens, not "
            f"{MODEL_MAX_SEQ_LENGTH} as D09 states"
        )
    if longest_chunk > identity.max_seq_length:
        raise EmbeddingsCheckError(
            f"the chunk table has a chunk of {longest_chunk} tokens, more than "
            f"{identity.model_id} reads ({identity.max_seq_length}), so it would be truncated"
        )


def _embed_model(
    encoder: SentenceEncoder,
    records: Sequence[ChunkRecord],
    cache_path: Path,
    threshold: float,
    *,
    checkpoint: int,
    timer: Callable[[], float],
    report: Callable[[str], None],
) -> _ModelOutcome:
    """Embed and pool ``records`` with one model, after checking that it reads them whole."""
    identity = encoder.identity
    _check_model(identity, max((record.n_tokens for record in records), default=0))
    cache = ChunkVectorCache.open(cache_path, identity.dimension)
    started = timer()
    texts = [record.text for record in records]
    embedded = embed_texts(texts, encoder, cache, checkpoint=checkpoint, report=report)
    seconds = timer() - started
    pooled = pool_chunk_vectors(records, embedded.vectors)
    agreement = variant_agreement(pooled.section_mean, pooled.plain_mean, threshold)
    return _ModelOutcome(identity, embedded, seconds, pooled, agreement)


# Use case


def build_embeddings(
    config: AppConfig,
    project_root: Path,
    *,
    snapshot_id: str | None = None,
    load_encoder: LoadEncoder = load_sentence_encoder,
    store: TableStore | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    report: Callable[[str], None] = _silent,
    checkpoint: int = CHECKPOINT_CHUNKS,
) -> EmbeddingsBuild:
    """Embed, pool and save the chunks of a snapshot, the one harvested last by default.

    ``load_encoder`` loads the encoder of each model of ``embedding.models``; ``report``
    receives one line of progress at a time.

    Raises:
        FileNotFoundError: if the snapshot, the chunk table or a model is missing.
        EmbeddingsCheckError: if the counts differ from ``embedding.expected``, or a model
            does not read the chunks whole; no output is written then.
        ValueError: if ``embedding.expected`` belongs to another snapshot, the chunk table or
            a cache file is damaged, or an encoder gives unusable vectors.
        OSError: if reading or writing the disk fails, or a model cannot be loaded.
    """
    started = timer()
    store = store or ParquetTableStore()
    data_dir = config.paths.resolve_against(project_root).data_dir
    snapshot_id = check_snapshot_id(snapshot_id or latest_snapshot_id(data_dir / RAW_DIR_NAME))
    settings = config.embedding
    expected = settings.expected
    if snapshot_id != expected.snapshot_id:
        raise ValueError(
            f"embedding.expected declares the counts of snapshot '{expected.snapshot_id}', not "
            f"of '{snapshot_id}'; embed that snapshot and declare its counts first"
        )
    interim_dir = data_dir / INTERIM_DIR_NAME / snapshot_id
    chunks = chunks_manifest(interim_dir)
    records = load_chunks(interim_dir, store=store)
    counts = check_counts(records, expected)
    slugs = {model_id: model_slug(model_id) for model_id in settings.models}
    encoder_settings = EncoderSettings(
        batch_size=settings.batch_size, device=settings.device, threads=settings.threads
    )
    outcomes = []
    for model_id in settings.models:
        report(f"{model_id}: loading")
        # Each encoder is dropped once embedded, so one model at a time is in memory.
        outcomes.append(
            _embed_model(
                load_encoder(model_id, encoder_settings),
                records,
                data_dir / CACHE_DIR / f"{slugs[model_id]}.npy",
                settings.sensitivity_cosine_threshold,
                checkpoint=checkpoint,
                timer=timer,
                report=report,
            )
        )
    embeddings_dir = interim_dir / EMBEDDINGS_DIR_NAME
    sections = _section_rows(outcomes[0].pooled)
    store.write(sections, embeddings_dir / SECTIONS_FILE_NAME, model_cls=EmbeddedSection)
    model_records = tuple(
        _write_model(embeddings_dir, slugs[outcome.identity.model_id], outcome)
        for outcome in outcomes
    )
    manifest = EmbeddingsManifest(
        snapshot_id=snapshot_id,
        created_at=clock().astimezone(UTC),
        embedder_version=EMBEDDER_VERSION,
        chunker_version=chunks.chunker_version,
        chunks_sha256=chunks.chunks_sha256,
        cache_key_scheme=CACHE_KEY_SCHEME,
        cache_dir=CACHE_DIR.as_posix(),
        encoder=EncoderSettingsRecord(**asdict(encoder_settings)),
        document_variant=settings.document_vector,
        sections_file=SECTIONS_FILE_NAME,
        sections_sha256=hashlib.sha256(
            (embeddings_dir / SECTIONS_FILE_NAME).read_bytes()
        ).hexdigest(),
        counts={**counts, "sections": len(sections)},
        models=model_records,
        seconds=round(max(timer() - started, 0.0), 3),
    )
    # The commit point: until it is written, the manifest describes the previous outputs.
    write_text_atomically(
        embeddings_dir / EMBEDDINGS_MANIFEST_FILE_NAME, manifest.model_dump_json(indent=2)
    )
    return EmbeddingsBuild(
        snapshot_id=snapshot_id,
        manifest=manifest,
        embeddings_dir=embeddings_dir,
        cache_dir=data_dir / CACHE_DIR,
    )


def _section_rows(pooled: PooledVectors) -> list[EmbeddedSection]:
    document_rows = {doc_code: row for row, doc_code in enumerate(pooled.doc_codes)}
    return [
        EmbeddedSection(
            doc_code=doc_code,
            document_row=document_rows[doc_code],
            section=section,  # type: ignore[arg-type]
            n_chunks=n_chunks,
        )
        for (doc_code, section), n_chunks in zip(
            pooled.sections, pooled.section_chunks, strict=True
        )
    ]


def _write_model(embeddings_dir: Path, slug: str, outcome: _ModelOutcome) -> ModelEmbeddingsRecord:
    """Write the vectors of one model to its directory and return its manifest record."""
    pooled = outcome.pooled
    contents = {
        DOCUMENT_FILE_NAMES["section_mean"]: _npy(pooled.section_mean),
        DOCUMENT_FILE_NAMES["plain_mean"]: _npy(pooled.plain_mean),
        SECTION_VECTORS_FILE_NAME: _npy(pooled.section_vectors),
    }
    directory = embeddings_dir / slug
    directory.mkdir(parents=True, exist_ok=True)
    write_files_atomically(directory, contents)
    identity, embedded, agreement = outcome.identity, outcome.embedded, outcome.agreement
    return ModelEmbeddingsRecord(
        model_id=identity.model_id,
        revision=identity.revision,
        max_seq_length=identity.max_seq_length,
        dimension=identity.dimension,
        normalize=False,
        directory=slug,
        files={name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
        cache_file=(CACHE_DIR / f"{slug}.npy").as_posix(),
        distinct_texts=embedded.distinct_texts,
        from_cache=embedded.from_cache,
        embedded=embedded.embedded,
        embed_seconds=round(max(outcome.seconds, 0.0), 3),
        variant_agreement=VariantAgreementRecord(
            threshold=agreement.threshold,
            minimum=round(agreement.minimum, 6),
            median=round(agreement.median, 6),
            below=agreement.below,
        ),
    )


def _npy(vectors: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


# Reading the outputs


def embeddings_manifest(interim_dir: Path) -> EmbeddingsManifest:
    """Return the embeddings manifest of the snapshot whose interim directory is ``interim_dir``.

    Raises:
        FileNotFoundError: if there is none.
        ValueError: if it is damaged; the message names no content of the file.
    """
    path = interim_dir / EMBEDDINGS_DIR_NAME / EMBEDDINGS_MANIFEST_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name} in {path.parent}; run {BUILD_COMMAND} first")
    try:
        return EmbeddingsManifest.model_validate_json(path.read_bytes())
    except ValidationError:
        # Not chained: the validation error may quote the content of the file.
        raise ValueError(f"{path} is damaged; run {BUILD_COMMAND} again") from None


def load_embeddings(
    interim_dir: Path,
    model_id: str,
    *,
    variant: DocumentVariant | None = None,
    store: TableStore | None = None,
) -> DocumentEmbeddings:
    """Return the saved embeddings of ``model_id``, once they match their manifest.

    ``variant`` picks the document vectors; by default, the one that the manifest records from
    ``embedding.document_vector``. Each file is read once, then checked, then parsed.

    Raises:
        FileNotFoundError: if the manifest or an output file is missing.
        ValueError: if the manifest is damaged or has no embeddings of ``model_id``, or a file
            differs from it.
    """
    manifest = embeddings_manifest(interim_dir)
    directory = interim_dir / EMBEDDINGS_DIR_NAME
    manifest_name = EMBEDDINGS_MANIFEST_FILE_NAME
    record = next((model for model in manifest.models if model.model_id == model_id), None)
    if record is None:
        raise ValueError(
            f"{directory / manifest_name} has no embeddings of {model_id}; run {BUILD_COMMAND}"
        )
    sections_path = directory / SECTIONS_FILE_NAME
    if not sections_path.is_file():
        raise FileNotFoundError(f"{sections_path} is missing; run {BUILD_COMMAND}")
    try:
        sections = (store or ParquetTableStore()).read_verified(
            EmbeddedSection, sections_path, sha256=manifest.sections_sha256
        )
    except ChecksumMismatchError:
        raise ValueError(
            f"{sections_path} differs from {manifest_name}; run {BUILD_COMMAND} again"
        ) from None
    doc_codes = tuple(dict.fromkeys(section.doc_code for section in sections))
    variant = variant or manifest.document_variant
    model_dir = directory / model_slug(model_id)
    return DocumentEmbeddings(
        model_id=model_id,
        revision=record.revision,
        variant=variant,
        doc_codes=doc_codes,
        vectors=_read_vectors(model_dir / DOCUMENT_FILE_NAMES[variant], record, len(doc_codes)),
        sections=tuple((section.doc_code, section.section) for section in sections),
        section_vectors=_read_vectors(model_dir / SECTION_VECTORS_FILE_NAME, record, len(sections)),
    )


def _read_vectors(path: Path, record: ModelEmbeddingsRecord, rows: int) -> np.ndarray:
    """Return the matrix at ``path`` once it has the SHA-256 that ``record`` lists for it and
    ``rows`` float32 rows of the model's dimension."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run {BUILD_COMMAND}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record.files.get(path.name):
        raise ValueError(
            f"{path} differs from {EMBEDDINGS_MANIFEST_FILE_NAME}; run {BUILD_COMMAND} again"
        )
    vectors = np.load(io.BytesIO(content), allow_pickle=False)
    if vectors.dtype != np.float32 or vectors.shape != (rows, record.dimension):
        raise ValueError(f"{path} does not hold {rows} vectors of {record.dimension} dimensions")
    return vectors


# Command line


def describe(built: EmbeddingsBuild) -> str:
    """Describe the build in counts only."""
    manifest = built.manifest
    counts = manifest.counts
    lines = [
        f"Snapshot {built.snapshot_id}: {counts['documents']} documents, {counts['sections']} "
        f"sections, {counts['chunks']} chunks, embedded by {len(manifest.models)} models"
    ]
    for model in manifest.models:
        agreement = model.variant_agreement
        lines += [
            f"{model.model_id} at {model.revision[:12]}: {model.dimension} dimensions; "
            f"{model.distinct_texts} distinct texts, {model.from_cache} from the cache, "
            f"{model.embedded} embedded in {model.embed_seconds:.1f} s",
            f"  section_mean vs plain_mean, cosine per document: min {agreement.minimum:.4f}, "
            f"median {agreement.median:.4f}, {agreement.below} below {agreement.threshold}",
        ]
    encoder = manifest.encoder
    return "\n".join(
        [
            *lines,
            f"Document vector read downstream: {manifest.document_variant}",
            f"Encoder: batch size {encoder.batch_size}, device {encoder.device}, "
            f"{encoder.threads} threads",
            f"Chunk vector cache {built.cache_dir}",
            f"Embeddings {built.embeddings_dir}",
            f"Seconds: {manifest.seconds:.1f}",
        ]
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.representation.build_embeddings",
        description=(
            "Embed the chunks of every included thesis with every configured model, from the "
            "local Hugging Face cache only, pool them into section and document vectors, and "
            f"save them under data/interim/<snapshot>/{EMBEDDINGS_DIR_NAME}/ with their "
            "manifest."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--snapshot-id", help="snapshot to embed (default: the snapshot harvested last)"
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    load_encoder: LoadEncoder = load_sentence_encoder,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and
    130 after an interruption."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    try:
        config = load_config(project_root / CONFIG_PATH)
        built = build_embeddings(
            config,
            project_root,
            snapshot_id=arguments.snapshot_id,
            load_encoder=load_encoder,
            clock=clock,
            timer=timer,
            report=_progress,
        )
    except KeyboardInterrupt:
        print(
            "interrupted: the cache keeps the chunk vectors saved at each checkpoint, and "
            "outputs that do not match their manifest are refused by readers; run the command "
            "again",
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
