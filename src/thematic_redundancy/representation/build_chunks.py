"""Build the chunk table of a snapshot: the sentence-aligned chunks of every included thesis (T13).

:func:`build_chunk_table` reads the fichas through
:func:`~thematic_redundancy.corpus.build_fichas.load_fichas` and chunks the sections that the
embeddings read (D09), as D27 fixes them:

- **Section texts** (:func:`section_texts`). The title is light-cleaned and loses its place and
  year suffix (D10); the abstract and the objectives are light-cleaned
  (:func:`~thematic_redundancy.preprocessing.light_cleaner.clean_light`). A missing section has
  no chunk, so a metadata-only thesis has its title and abstract only (C4).
- **Chunks** (:func:`chunk_fichas`). The title is one chunk when it fits (C3); the abstract and
  the objectives are cut into sentence-aligned chunks (:mod:`~.chunker`). A chunk fits
  ``chunking.max_tokens`` under the tokenizer of every configured embedding model, loaded from
  the local Hugging Face cache only (C5).
- **Checks** (:func:`check_chunks`). Each chunk is counted again by each tokenizer on its own,
  each section is recovered from its chunks, and the counts must equal ``chunking.expected``.
  When a check fails, the build stops, names every failed check, and writes nothing.

The outputs go to the gitignored ``data/interim/<snapshot_id>/``, never to ``results/``, since
they hold thesis text (C6):

- ``chunks.parquet``: one row per chunk, keyed by (``doc_code``, ``section``, ``chunk_index``),
  with its span of the section text, its token count, how it ends, the SHA-256 of its text, the
  sections that its thesis uses (C4) and the text;
- ``chunks_manifest.json``: the inputs and the tokenizers (each ``tokenizer.json`` SHA-256),
  the table's SHA-256, the counts and the seconds, without text.

The table is written first and the manifest last, as the commit point: :func:`load_chunks`
accepts the table only while it matches the manifest, so a build stopped between the two is
refused until the command runs again::

    uv run python -m thematic_redundancy.representation.build_chunks [--project-root PATH]
        [--snapshot-id ID]
"""

import argparse
import hashlib
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError

from thematic_redundancy.corpus.build_fichas import load_fichas, verify_fichas_outputs
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
from thematic_redundancy.preprocessing.light_cleaner import clean_light
from thematic_redundancy.preprocessing.title_suffix import strip_title_suffix
from thematic_redundancy.representation.chunker import (
    CUT_KINDS,
    CutKind,
    MaxTokenCounter,
    TokenCounter,
    chunk_section,
    chunk_title,
)
from thematic_redundancy.representation.hf_tokenizers import (
    TokenizerIdentity,
    load_token_counter,
)
from thematic_redundancy.shared.config import AppConfig, ChunkCountsConfig, load_config
from thematic_redundancy.shared.filesystem import write_text_atomically
from thematic_redundancy.shared.storage import (
    ChecksumMismatchError,
    ParquetTableStore,
    TableStore,
)

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

BUILD_COMMAND = "uv run python -m thematic_redundancy.representation.build_chunks"
"""Command that writes the chunk table; the reader's error messages point to it."""

CHUNKS_FILE_NAME = "chunks.parquet"
CHUNKS_MANIFEST_FILE_NAME = "chunks_manifest.json"

CHUNKER_VERSION = 1
"""Version of the chunking rules; it changes whenever the same fichas would give other chunks."""

Section = Literal["title", "abstract", "objectives"]
SECTIONS: tuple[Section, ...] = ("title", "abstract", "objectives")
"""The sections of a thesis that the embeddings read, in this order (D09)."""


class ChunksCheckError(ValueError):
    """The chunks break a rule that the build enforces, so nothing was written."""


class CountedTokenizer(TokenCounter, Protocol):
    """A token counter that names the tokenizer it counts with."""

    identity: TokenizerIdentity


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ChunkRecord(_Model):
    """One row of the chunk table: a chunk of one section of one included thesis."""

    doc_code: str
    section: Section
    chunk_index: Count
    """Position of the chunk in its section, from 0."""
    char_start: Count
    char_end: Count
    """Span of the chunk in the section text that :func:`section_texts` gives."""
    n_tokens: Count
    """Tokens of the chunk, special tokens included: the largest count of the tokenizers."""
    cut_kind: CutKind
    sections_used: tuple[Section, ...]
    """Sections of the thesis that have chunks; the same in every row of the thesis (C4)."""
    text_sha256: Sha256
    text: str


class TokenizerRecord(_Model):
    """A tokenizer that counted the chunks, and the largest chunk under it alone."""

    model_id: str
    revision: str | None
    tokenizer_sha256: Sha256
    vocabulary_size: Count
    largest_chunk_tokens: Count


class ChunksManifest(_Model):
    """Content of ``chunks_manifest.json``: provenance, counts and time, without text."""

    snapshot_id: SnapshotId
    created_at: UtcDatetime
    chunker_version: Annotated[StrictInt, Field(ge=1)]
    fichas_sha256: Sha256
    max_tokens: Count
    tokenizers: tuple[TokenizerRecord, ...]
    chunks_file: str
    chunks_sha256: Sha256
    counts: dict[str, Count]
    """The counts that :func:`check_chunks` compared with ``chunking.expected``."""
    cut_kinds: dict[str, Count]
    sections_used: dict[str, Count]
    """Theses per combination of sections used, such as ``title+abstract``."""
    sections_recovered: Count
    seconds: StrictFloat


class ChunkChecks(NamedTuple):
    """What :func:`check_chunks` found, once every check passed."""

    counts: dict[str, int]
    largest_tokens: dict[str, int]
    """Largest chunk under each tokenizer counting on its own."""
    sections: int
    """Sections recovered whole from their chunks."""


@dataclass(frozen=True)
class ChunksBuild:
    """Outcome of :func:`build_chunk_table`."""

    snapshot_id: str
    records: tuple[ChunkRecord, ...]
    manifest: ChunksManifest
    chunks_path: Path


# Chunking


def section_texts(ficha: Ficha) -> dict[Section, str]:
    """Return the texts of the sections of ``ficha`` that the embeddings read, in
    :data:`SECTIONS` order, leaving out a missing or blank one."""
    raw: dict[Section, str | None] = {
        "title": ficha.title,
        "abstract": ficha.abstract,
        "objectives": ficha.objectives,
    }
    texts: dict[Section, str] = {}
    for section, value in raw.items():
        if value is None:
            continue
        text = clean_light(value)
        if section == "title":
            text = strip_title_suffix(text).title
        if text.strip():
            texts[section] = text
    return texts


def chunk_fichas(
    fichas: Iterable[Ficha], counter: TokenCounter, max_tokens: int
) -> tuple[ChunkRecord, ...]:
    """Return the chunk rows of the included ``fichas``, in their order, then by section."""
    records = []
    for ficha in fichas:
        if not ficha.include:
            continue
        texts = section_texts(ficha)
        chunks = {
            section: (chunk_title if section == "title" else chunk_section)(
                text, counter, max_tokens
            )
            for section, text in texts.items()
        }
        used = tuple(section for section in SECTIONS if chunks.get(section))
        for section in used:
            for index, chunk in enumerate(chunks[section]):
                text = texts[section][chunk.start : chunk.end]
                records.append(
                    ChunkRecord(
                        doc_code=ficha.doc_code,
                        section=section,
                        chunk_index=index,
                        char_start=chunk.start,
                        char_end=chunk.end,
                        n_tokens=chunk.n_tokens,
                        cut_kind=chunk.cut_kind,
                        sections_used=used,
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        text=text,
                    )
                )
    return tuple(records)


def count_chunks(records: Sequence[ChunkRecord]) -> dict[str, int]:
    """Return the counts that ``chunking.expected`` declares, from ``records``."""
    titles = sum(record.section == "title" for record in records)
    return {
        "documents": len({record.doc_code for record in records}),
        "chunks": len(records),
        "title_chunks": titles,
        "body_chunks": len(records) - titles,
    }


def check_chunks(
    records: Sequence[ChunkRecord],
    fichas: Iterable[Ficha],
    counters: Mapping[str, TokenCounter],
    max_tokens: int,
    expected: ChunkCountsConfig,
) -> ChunkChecks:
    """Check ``records`` against the fichas that they come from, and return what was found.

    - Each chunk is counted again by each of ``counters`` on its own, and must fit
      ``max_tokens`` under every one.
    - Each section of each included thesis must be recovered: its chunks, in order, are spans
      of its text with only whitespace around and between them.
    - The counts must equal ``expected``.

    Raises:
        ChunksCheckError: naming every check that fails.
    """
    problems = []
    largest = {}
    for name, counter in counters.items():
        counts = [counter.count(record.text) for record in records]
        largest[name] = max(counts, default=0)
        above = sum(count > max_tokens for count in counts)
        if above:
            problems.append(
                f"{above} chunk{'s' if above > 1 else ''} above {max_tokens} tokens under {name}"
            )
    by_section: dict[tuple[str, str], list[ChunkRecord]] = defaultdict(list)
    for record in records:
        by_section[(record.doc_code, record.section)].append(record)
    texts = {
        (ficha.doc_code, section): text
        for ficha in fichas
        if ficha.include
        for section, text in section_texts(ficha).items()
    }
    lost = sum(not _recovers(text, by_section.get(key, [])) for key, text in texts.items())
    lost += len(by_section.keys() - texts.keys())  # Chunks of a section that has no text.
    if lost:
        problems.append(f"{lost} section{'s are' if lost > 1 else ' is'} not recovered")
    counts = count_chunks(records)
    for name, value in counts.items():
        declared = getattr(expected, name)
        if value != declared:
            problems.append(f"{name} is {value}, but chunking.expected declares {declared}")
    if problems:
        raise ChunksCheckError(
            f"the chunks of snapshot {expected.snapshot_id} fail {len(problems)} "
            f"check{'s' if len(problems) > 1 else ''}: {'; '.join(problems)}"
        )
    return ChunkChecks(counts=counts, largest_tokens=largest, sections=len(texts))


def _recovers(text: str, chunks: Sequence[ChunkRecord]) -> bool:
    """Tell whether ``chunks``, in order, hold all of ``text`` but whitespace."""
    cursor = 0
    for index, chunk in enumerate(chunks):
        if (
            chunk.chunk_index != index
            or chunk.char_start < cursor
            or text[cursor : chunk.char_start].strip()
            or text[chunk.char_start : chunk.char_end] != chunk.text
            or not chunk.text
        ):
            return False
        cursor = chunk.char_end
    return bool(chunks) and not text[cursor:].strip()


# Use case


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def build_chunk_table(
    config: AppConfig,
    project_root: Path,
    *,
    snapshot_id: str | None = None,
    load_counter: Callable[[str], CountedTokenizer] = load_token_counter,
    store: TableStore | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> ChunksBuild:
    """Chunk, check and save the included theses of a snapshot, the one harvested last by
    default.

    ``load_counter`` loads the tokenizer of each model of ``embedding.models``.

    Raises:
        FileNotFoundError: if the snapshot, the fichas outputs or a tokenizer is missing.
        ChunksCheckError: if the chunks fail a check; nothing is written then.
        ValueError: if ``chunking.expected`` belongs to another snapshot, or if the fichas
            outputs are damaged.
        OSError: if reading or writing the disk fails, or a tokenizer cannot be loaded.
    """
    started = timer()
    store = store or ParquetTableStore()
    data_dir = config.paths.resolve_against(project_root).data_dir
    snapshot_id = check_snapshot_id(snapshot_id or latest_snapshot_id(data_dir / RAW_DIR_NAME))
    expected = config.chunking.expected
    if snapshot_id != expected.snapshot_id:
        raise ValueError(
            f"chunking.expected declares the counts of snapshot '{expected.snapshot_id}', not "
            f"of '{snapshot_id}'; chunk that snapshot and declare its counts first"
        )
    interim_dir = data_dir / INTERIM_DIR_NAME / snapshot_id
    fichas_sha256 = verify_fichas_outputs(interim_dir).fichas_sha256
    fichas = load_fichas(interim_dir, store=store)
    counters = {model: load_counter(model) for model in config.embedding.models}
    max_tokens = config.chunking.max_tokens
    records = chunk_fichas(fichas, MaxTokenCounter(tuple(counters.values())), max_tokens)
    checks = check_chunks(records, fichas, counters, max_tokens, expected)
    chunks_path = interim_dir / CHUNKS_FILE_NAME
    store.write(records, chunks_path, model_cls=ChunkRecord)
    manifest = ChunksManifest(
        snapshot_id=snapshot_id,
        created_at=clock().astimezone(UTC),
        chunker_version=CHUNKER_VERSION,
        fichas_sha256=fichas_sha256,
        max_tokens=max_tokens,
        tokenizers=tuple(
            TokenizerRecord(
                model_id=counter.identity.model_id,
                revision=counter.identity.revision,
                tokenizer_sha256=counter.identity.tokenizer_sha256,
                vocabulary_size=counter.identity.vocabulary_size,
                largest_chunk_tokens=checks.largest_tokens[model],
            )
            for model, counter in counters.items()
        ),
        chunks_file=CHUNKS_FILE_NAME,
        chunks_sha256=_sha256(chunks_path),
        counts=checks.counts,
        cut_kinds={kind: sum(r.cut_kind == kind for r in records) for kind in CUT_KINDS},
        sections_used=dict(
            Counter(
                "+".join(record.sections_used) for record in records if record.section == "title"
            )
        ),
        sections_recovered=checks.sections,
        seconds=round(max(timer() - started, 0.0), 3),
    )
    # The commit point: until it is written, the manifest describes the previous table.
    write_text_atomically(
        interim_dir / CHUNKS_MANIFEST_FILE_NAME, manifest.model_dump_json(indent=2)
    )
    return ChunksBuild(
        snapshot_id=snapshot_id, records=records, manifest=manifest, chunks_path=chunks_path
    )


# Reading the outputs


def chunks_manifest(interim_dir: Path) -> ChunksManifest:
    """Return the chunk manifest in ``interim_dir``.

    Raises:
        FileNotFoundError: if there is none.
        ValueError: if it is damaged; the message names no content of the file.
    """
    path = interim_dir / CHUNKS_MANIFEST_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name} in {interim_dir}; run {BUILD_COMMAND} first")
    try:
        return ChunksManifest.model_validate_json(path.read_bytes())
    except ValidationError:
        # Not chained: the validation error may quote the content of the file.
        raise ValueError(
            f"{path.name} in {interim_dir} is damaged; run {BUILD_COMMAND} again"
        ) from None


def load_chunks(interim_dir: Path, *, store: TableStore | None = None) -> tuple[ChunkRecord, ...]:
    """Return the chunk table saved in ``interim_dir``, once it matches its manifest.

    Raises:
        FileNotFoundError: if the manifest or the table is missing.
        ValueError: if the manifest is damaged, or the table differs from it.
    """
    manifest = chunks_manifest(interim_dir)
    path = interim_dir / CHUNKS_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing from {interim_dir}; run {BUILD_COMMAND}")
    try:
        return (store or ParquetTableStore()).read_verified(
            ChunkRecord, path, sha256=manifest.chunks_sha256
        )
    except ChecksumMismatchError as error:
        raise ValueError(
            f"{path.name} differs from {CHUNKS_MANIFEST_FILE_NAME}, as after a stopped build or "
            f"a later edit; run {BUILD_COMMAND} again"
        ) from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Command line


def describe(built: ChunksBuild) -> str:
    """Describe the build in counts only."""
    manifest = built.manifest
    counts = manifest.counts
    cuts = manifest.cut_kinds
    tokenizers = manifest.tokenizers
    return "\n".join(
        [
            f"Snapshot {built.snapshot_id}: {counts['documents']} documents, {counts['chunks']} "
            f"chunks ({counts['title_chunks']} title, {counts['body_chunks']} body)",
            "Sections used: "
            + "; ".join(f"{used} {number}" for used, number in manifest.sections_used.items()),
            f"Chunk ends: sentence {cuts['sentence']}, clause {cuts['clause']}, comma "
            f"{cuts['comma']}, window {cuts['window']}",
            "Largest chunk, recounted by each tokenizer alone: "
            + ", ".join(f"{t.largest_chunk_tokens} ({t.model_id})" for t in tokenizers)
            + f"; 0 chunks above {manifest.max_tokens} tokens under any tokenizer",
            f"Checks passed: text recovered for all {manifest.sections_recovered} sections; "
            "every count as declared in chunking.expected",
            "Tokenizers: "
            + "; ".join(
                f"{t.model_id} at {(t.revision or 'no revision')[:12]}, tokenizer.json SHA-256 "
                f"{t.tokenizer_sha256[:12]}"
                for t in tokenizers
            ),
            f"Chunks {built.chunks_path}",
            f"Seconds: {manifest.seconds:.1f}",
        ]
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.representation.build_chunks",
        description=(
            "Chunk the sections of every included thesis under the tokenizer of every "
            "configured model, check the declared counts, and save the chunks as "
            f"data/interim/<snapshot>/{CHUNKS_FILE_NAME} with their manifest."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--snapshot-id", help="snapshot to chunk (default: the snapshot harvested last)"
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    load_counter: Callable[[str], CountedTokenizer] = load_token_counter,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and
    130 after an interruption."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    try:
        config = load_config(project_root / CONFIG_PATH)
        built = build_chunk_table(
            config,
            project_root,
            snapshot_id=arguments.snapshot_id,
            load_counter=load_counter,
            clock=clock,
            timer=timer,
        )
    except KeyboardInterrupt:
        print(
            "interrupted: a chunk table that does not match its manifest is refused by "
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
