"""Build the fichas dataset of a snapshot: one :class:`Ficha` per harvested record (T12).

:func:`build_fichas` applies these rules to every record, in snapshot order:

1. **Inclusion (D01).** A record is a thesis when one ``renati.type`` value has exactly the
   fragment ``snapshot.thesis_type`` (``#tesis``). Any other record is excluded as
   ``non_thesis_type``, and its ficha keeps the raw ``renati.type``.
2. **Duplicates (D24).** Theses with the same title, abstract, authors, advisor and issue
   date (whitespace collapsed and case folded, accents kept) are one thesis. The record
   accessioned first (``dc.date.accessioned``) stays, and a tie goes to the smaller item
   uuid. Every other record of the group is excluded as ``duplicate``, with ``duplicate_of``
   naming the record that stays.
3. **Objectives (O05).** A record whose PDF is on disk takes the locator's row of
   ``objectives.parquet``: ``extracted``, with the general objective, a blank line and the
   specific objectives, or ``not_found``. A record without a PDF on disk is ``no_pdf``, and
   so is one whose PDF is declared in ``fichas.wrong_pdfs``: its SHA-256 stays recorded,
   with a quality note. A ficha with objectives is represented by its title, abstract and
   objectives; any other by its title and abstract. The source format tells whether the
   PDF text used came from the text layer, from OCR or from both (``none`` without one).
4. **People.** The advisor code comes from the advisor's ORCID iD when it is valid, and
   otherwise from the normalized name (:mod:`~thematic_redundancy.corpus.anonymize`); an
   ORCID iD that is malformed or fails its check digit leaves a quality note. Each author
   gets one code, in repository order. No name is stored.
5. **Other fields** come from the metadata: the handle URL, the first issue date with its
   year and the future-dated flag (after the UTC date of the harvest), the keywords as
   harvested, the OCDE codes, the language, the COAR access right and the embargo end.
   Each flag of the objectives locator becomes a quality note.

The same rules fill the fichas of excluded records, so every ficha describes its record
fully. :func:`check_fichas` then stops the build unless every document code is unique, no
PDF serves two included theses, and the counts equal those declared in ``fichas.expected``
for the snapshot.

:func:`build_dataset` reads the inputs, builds and checks the fichas, and only then writes,
atomically, into the gitignored ``data/interim/<snapshot_id>/``:

- ``fichas.parquet``: every ficha, through the table-store port; it holds thesis texts;
- ``exclusions.parquet``: the excluded records with their reason, and no text;
- ``fichas_manifest.json``: the inputs' and outputs' SHA-256, the counts and the seconds.

``--report-out DIR`` also writes the quality report, numbers, handles and lemmas only, as
``DIR/quality_report.json`` and ``DIR/quality_report.md``
(:mod:`~thematic_redundancy.corpus.fichas_quality`)::

    uv run python -m thematic_redundancy.corpus.build_fichas [--project-root PATH]
        [--snapshot-id ID] [--report-out DIR]
"""

import argparse
import hashlib
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, JsonValue, StrictFloat, ValidationError

from thematic_redundancy.corpus.anonymize import (
    default_key_path,
    load_or_create_key,
    normalize_orcid,
    person_identity,
    pseudonym,
)
from thematic_redundancy.corpus.ficha import (
    ADVISOR_CODE_PREFIX,
    AUTHOR_CODE_PREFIX,
    THESIS,
    AccessRights,
    DocumentType,
    ExclusionReason,
    Ficha,
    ObjectivesStatus,
    SourceFormat,
    doc_code_for,
)
from thematic_redundancy.corpus.fichas_quality import (
    Lemmatizer,
    duplicate_pairs,
    quality_report,
    spacy_lemmatizer,
    write_quality_report,
)
from thematic_redundancy.corpus.pdf_manifest import (
    FILE_STATUSES,
    PDF_DIR_NAME,
    PDF_MANIFEST_FILE_NAME,
    PdfEntry,
    PdfManifest,
    read_pdf_manifest,
)
from thematic_redundancy.corpus.profile import (
    ABSTRACT_KEY,
    ADVISOR_KEY,
    ADVISOR_ORCID_KEY,
    EMBARGO_END_KEY,
    ISSUED_KEY,
    KEYWORD_KEY,
    LANGUAGE_KEY,
    OCDE_KEY,
    TITLE_KEY,
    TYPE_KEY,
    access_class,
    issue_day,
    ocde_code,
)
from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    Count,
    SnapshotId,
    SnapshotRecord,
    UtcDatetime,
    check_snapshot_id,
    latest_snapshot_id,
    read_snapshot,
)
from thematic_redundancy.corpus.thesis_files import is_thesis
from thematic_redundancy.extraction.locate_objectives import (
    OBJECTIVES_FILE_NAME,
    OBJECTIVES_MANIFEST_FILE_NAME,
    ObjectivesManifest,
)
from thematic_redundancy.extraction.objectives import ObjectivesRecord
from thematic_redundancy.extraction.text_manifest import (
    INTERIM_DIR_NAME,
    PAGES_DIR_NAME,
    TEXT_MANIFEST_FILE_NAME,
    Sha256,
    TextEntry,
    TextManifest,
    read_text_manifest,
    write_json_atomically,
)
from thematic_redundancy.preprocessing.light_cleaner import normalize_text
from thematic_redundancy.preprocessing.stopwords import (
    DOMAIN_STOPWORDS_FILE,
    load_domain_stopwords,
)
from thematic_redundancy.shared.config import (
    AppConfig,
    FichasCountsConfig,
    WrongPdfConfig,
    load_config,
)
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

LOCATE_COMMAND = "uv run python -m thematic_redundancy.extraction.locate_objectives"
"""Command that writes the objectives table; error messages point to it."""

FICHAS_FILE_NAME = "fichas.parquet"
EXCLUSIONS_FILE_NAME = "exclusions.parquet"
FICHAS_MANIFEST_FILE_NAME = "fichas_manifest.json"

AUTHOR_KEY = "dc.contributor.author"
ACCESSIONED_KEY = "dc.date.accessioned"
HANDLE_RESOLVER = "https://hdl.handle.net/"
"""Prefix of a thesis's public handle URL, the link that every displayed thesis carries."""

OBJECTIVES_SEPARATOR = "\n\n"
"""Text between the general and the specific objectives of a ficha."""

NOTE_INVALID_ADVISOR_ORCID = (
    "advisor ORCID iD is malformed or fails its check digit; the advisor code comes from the name"
)
NOTE_WRONG_PDF = "PDF on disk belongs to another item ({handle})"
NOTE_LOCATOR_FLAG = "objectives locator flag: {flag}"

_OTHER_TYPES: dict[str, DocumentType] = {
    "trabajodesuficienciaprofesional": "trabajo_suficiencia",
    "trabajoacademico": "trabajo_academico",
}
"""Document types other than the thesis, by ``renati.type`` fragment with its spaces dropped
and its case folded, which covers the five spellings of the first snapshot."""

_ACCESS_RIGHTS: dict[str, AccessRights] = {
    "open": "open",
    "embargoed": "embargoed",
    "restricted": "restricted",
    "metadata_only": "restricted",
}
"""Access right of a ficha per access class of the metadata profile; others are unknown."""

Metadata = Mapping[str, Sequence[Mapping[str, JsonValue]]]


class FichasCheckError(ValueError):
    """The fichas break a rule that the build enforces, so nothing was written."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class Exclusion(_Model):
    """One row of the exclusion log: an excluded record and why, without any text."""

    item_uuid: UUID
    doc_code: str
    handle: str | None
    program_key: str
    document_type: DocumentType
    exclusion_reason: ExclusionReason
    duplicate_of: UUID | None


class FichasManifest(_Model):
    """Content of ``fichas_manifest.json``: provenance, counts and time, without text."""

    snapshot_id: SnapshotId
    created_at: UtcDatetime
    metadata_sha256: Sha256
    objectives_table_sha256: Sha256
    fichas_file: str
    fichas_sha256: Sha256
    exclusions_file: str
    exclusions_sha256: Sha256
    counts: dict[str, Count]
    """The counts that :func:`check_fichas` compared with ``fichas.expected``."""
    seconds: StrictFloat


@dataclass(frozen=True)
class FichaSources:
    """Everything that the fichas of one snapshot are made from."""

    snapshot_id: str
    snapshot_date: date
    """UTC date of the harvest; an issue date after it is future-dated."""
    records: tuple[SnapshotRecord, ...]
    pdf_entries: Mapping[UUID, PdfEntry]
    text_entries: Mapping[UUID, TextEntry]
    objectives: tuple[ObjectivesRecord, ...]


@dataclass(frozen=True)
class FichasBuild:
    """Outcome of :func:`build_dataset`."""

    snapshot_id: str
    fichas: tuple[Ficha, ...]
    records: tuple[SnapshotRecord, ...]
    manifest: FichasManifest
    fichas_path: Path


# Rules


def document_type(metadata: Metadata, thesis_type: str) -> DocumentType:
    """Return the kind of work of a record, from its ``renati.type``."""
    if is_thesis(metadata, thesis_type):
        return THESIS
    for value in _values(metadata, TYPE_KEY):
        fragment = value.rpartition("#")[2]
        kind = _OTHER_TYPES.get("".join(fragment.split()).casefold())
        if kind is not None:
            return kind
    return "other"


def duplicate_key(record: SnapshotRecord) -> tuple[object, ...] | None:
    """Return what two records of one thesis share (D24), or ``None`` without a title or an
    abstract to compare."""
    metadata = record.metadata
    title, abstract = _first(metadata, TITLE_KEY), _first(metadata, ABSTRACT_KEY)
    if title is None or abstract is None:
        return None
    return (
        _folded(title),
        _folded(abstract),
        tuple(_folded(name) for name in _values(metadata, AUTHOR_KEY)),
        tuple(_folded(name) for name in _values(metadata, ADVISOR_KEY)),
        _first(metadata, ISSUED_KEY),
    )


def duplicates_of(theses: Sequence[SnapshotRecord]) -> dict[UUID, UUID]:
    """Return, for each D24 duplicate among ``theses``, the item that stays in its place.

    Raises:
        ValueError: if a record of a group lacks a readable ``dc.date.accessioned``.
    """
    groups: dict[tuple[object, ...], list[SnapshotRecord]] = {}
    for record in theses:
        key = duplicate_key(record)
        if key is not None:
            groups.setdefault(key, []).append(record)
    replaced = {}
    for group in groups.values():
        if len(group) > 1:
            kept, *others = sorted(group, key=_canonical_order)
            replaced.update({other.uuid: kept.uuid for other in others})
    return replaced


def objectives_text(row: ObjectivesRecord) -> str | None:
    """Return the general objective, then the specific objectives, of a locator row."""
    parts = [part for part in (row.objective_general, row.objectives_specific) if part]
    return OBJECTIVES_SEPARATOR.join(parts) or None


def source_format(entry: TextEntry) -> SourceFormat:
    """Return how the page texts of a PDF were obtained, from its text-manifest entry."""
    if entry.ocr_pages and entry.text_layer_pages:
        return "mixed"
    return "ocr" if entry.ocr_pages else "digital"


def advisor_code(metadata: Metadata, key: bytes) -> tuple[str | None, tuple[str, ...]]:
    """Return the code of the first advisor, and the quality notes it leaves."""
    names = _values(metadata, ADVISOR_KEY)
    orcid = _first(metadata, ADVISOR_ORCID_KEY)
    identity = person_identity(orcid, names[0] if names else None)
    notes = () if orcid is None or normalize_orcid(orcid) else (NOTE_INVALID_ADVISOR_ORCID,)
    return (pseudonym(identity, key, ADVISOR_CODE_PREFIX) if identity else None), notes


def author_codes(metadata: Metadata, key: bytes) -> tuple[str, ...]:
    """Return one code per distinct author, in repository order."""
    codes: list[str] = []
    for name in _values(metadata, AUTHOR_KEY):
        identity = person_identity(None, name)
        if identity and (code := pseudonym(identity, key, AUTHOR_CODE_PREFIX)) not in codes:
            codes.append(code)
    return tuple(codes)


def build_fichas(sources: FichaSources, config: AppConfig, key: bytes) -> tuple[Ficha, ...]:
    """Return one ficha per record of ``sources``, in snapshot order; see the module rules.

    Raises:
        ValueError: if the inputs disagree (a PDF on disk without its objectives row or page
            texts, a row without a PDF, a declared wrong PDF that is no copy), or if a record
            makes no valid ficha; the message names the item.
    """
    thesis_type = config.snapshot.thesis_type
    kinds = {record.uuid: document_type(record.metadata, thesis_type) for record in sources.records}
    builder = _Builder(
        sources=sources,
        config=config,
        key=key,
        kinds=kinds,
        replaced=duplicates_of([r for r in sources.records if kinds[r.uuid] == THESIS]),
        wrong=_wrong_pdfs(sources, config.fichas.wrong_pdfs),
        rows=_objectives_rows(sources),
    )
    return tuple(builder.ficha(record) for record in sources.records)


@dataclass(frozen=True)
class _Builder:
    sources: FichaSources
    config: AppConfig
    key: bytes
    kinds: Mapping[UUID, DocumentType]
    replaced: Mapping[UUID, UUID]
    """Item that stays in place of each duplicate."""
    wrong: Mapping[UUID, str]
    """Handle of the item whose PDF each wrong-PDF item carries."""
    rows: Mapping[UUID, ObjectivesRecord]

    def ficha(self, record: SnapshotRecord) -> Ficha:
        metadata = record.metadata
        advisor, notes = advisor_code(metadata, self.key)
        objectives, status, fmt, objective_notes = self._objectives(record)
        exclusion: ExclusionReason | None = None
        if self.kinds[record.uuid] != THESIS:
            exclusion = "non_thesis_type"
        elif record.uuid in self.replaced:
            exclusion = "duplicate"
        entry = self.sources.pdf_entries.get(record.uuid)
        issued = _first(metadata, ISSUED_KEY)
        day = issue_day(issued) if issued is not None else None
        embargo = _first(metadata, EMBARGO_END_KEY)
        try:
            return Ficha(
                item_uuid=record.uuid,
                doc_code=doc_code_for(record.uuid),
                handle_url=self._handle_url(record),
                snapshot_id=self.sources.snapshot_id,
                document_type=self.kinds[record.uuid],
                renati_type_raw=_first(metadata, TYPE_KEY),
                title=_first(metadata, TITLE_KEY),
                abstract=_first(metadata, ABSTRACT_KEY),
                objectives=objectives,
                objectives_status=status,
                program_key=record.program_key,
                program_name=self._program_name(record.program_key),
                issue_date_raw=issued,
                issue_year=day.year if day else None,
                issued_after_snapshot=day is not None and day > self.sources.snapshot_date,
                keywords=tuple(_values(metadata, KEYWORD_KEY)),
                ocde_codes=tuple(ocde_code(value) for value in _values(metadata, OCDE_KEY)),
                language=_first(metadata, LANGUAGE_KEY),
                rights=_ACCESS_RIGHTS.get(access_class(metadata), "unknown"),
                embargo_end=issue_day(embargo) if embargo else None,
                advisor_code=advisor,
                author_codes=author_codes(metadata, self.key),
                pdf_status=entry.status if entry is not None else "not_attempted",
                pdf_sha256=entry.sha256 if entry is not None and _on_disk(entry) else None,
                source_format=fmt,
                section_source="title_abstract_objectives" if objectives else "title_abstract",
                include=exclusion is None,
                exclusion_reason=exclusion,
                duplicate_of=self.replaced.get(record.uuid),
                quality_notes=(*notes, *objective_notes),
            )
        except ValidationError as error:
            raise ValueError(
                f"record {record.uuid} ({record.handle}) makes no valid ficha: "
                f"{error.error_count()} problem(s): {_problems(error)}"
            ) from error

    def _objectives(
        self, record: SnapshotRecord
    ) -> tuple[str | None, ObjectivesStatus, SourceFormat, tuple[str, ...]]:
        """Return the objectives text, status, source format and notes of ``record``."""
        owner = self.wrong.get(record.uuid)
        if owner is not None:
            return None, "no_pdf", "none", (NOTE_WRONG_PDF.format(handle=owner),)
        entry = self.sources.pdf_entries.get(record.uuid)
        if entry is None or not _on_disk(entry):
            return None, "no_pdf", "none", ()
        row = self.rows.get(record.uuid)
        if row is None:
            raise ValueError(
                f"item {record.uuid} has its PDF on disk but no row in {OBJECTIVES_FILE_NAME}; "
                f"run {LOCATE_COMMAND} first"
            )
        text = self.sources.text_entries.get(record.uuid)
        if text is None or text.status != "ok":
            raise ValueError(f"item {record.uuid} has no extracted page texts in the text manifest")
        notes = tuple(NOTE_LOCATOR_FLAG.format(flag=flag) for flag in row.flags)
        return objectives_text(row), row.status, source_format(text), notes

    def _handle_url(self, record: SnapshotRecord) -> str:
        if record.handle:
            return f"{HANDLE_RESOLVER}{record.handle}"
        return f"{str(self.config.repository.base_url).rstrip('/')}/items/{record.uuid}"

    def _program_name(self, key: str) -> str:
        for program in self.config.snapshot.programs:
            if program.key == key:
                return program.name
        raise ValueError(f"program '{key}' is not declared in snapshot.programs")


def _wrong_pdfs(sources: FichaSources, declared: Sequence[WrongPdfConfig]) -> dict[UUID, str]:
    """Return the owner's handle per declared wrong-PDF item, once its copy is confirmed."""
    by_handle = {record.handle: record.uuid for record in sources.records if record.handle}
    wrong = {}
    for declaration in declared:
        handles = (declaration.item_handle, declaration.pdf_of_handle)
        missing = [handle for handle in handles if handle not in by_handle]
        if missing:
            raise ValueError(
                f"fichas.wrong_pdfs names a handle that the snapshot lacks: {', '.join(missing)}"
            )
        copy, owner = (sources.pdf_entries.get(by_handle[handle]) for handle in handles)
        if copy is None or owner is None or not _on_disk(copy) or copy.sha256 != owner.sha256:
            raise ValueError(
                f"the PDF of {declaration.item_handle} is not a copy of the PDF of "
                f"{declaration.pdf_of_handle}; correct fichas.wrong_pdfs"
            )
        wrong[by_handle[declaration.item_handle]] = declaration.pdf_of_handle
    return wrong


def _objectives_rows(sources: FichaSources) -> dict[UUID, ObjectivesRecord]:
    """Return the objectives rows by item, once each belongs to one PDF on disk."""
    rows: dict[UUID, ObjectivesRecord] = {}
    for row in sources.objectives:
        entry = sources.pdf_entries.get(row.item_uuid)
        if entry is None or not _on_disk(entry):
            raise ValueError(
                f"{OBJECTIVES_FILE_NAME} holds item {row.item_uuid}, whose PDF is not on disk"
            )
        if row.item_uuid in rows:
            raise ValueError(f"{OBJECTIVES_FILE_NAME} repeats item {row.item_uuid}")
        rows[row.item_uuid] = row
    return rows


# Checks


def count_fichas(fichas: Sequence[Ficha]) -> dict[str, int]:
    """Return the counts that ``fichas.expected`` declares, in its order."""
    included = [ficha for ficha in fichas if ficha.include]
    return {
        "records": len(fichas),
        "included": len(included),
        "duplicates": sum(ficha.exclusion_reason == "duplicate" for ficha in fichas),
        "non_thesis": sum(ficha.exclusion_reason == "non_thesis_type" for ficha in fichas),
        "objectives_extracted": sum(ficha.objectives_status == "extracted" for ficha in included),
        "metadata_only": sum(ficha.section_source == "title_abstract" for ficha in included),
    }


def check_fichas(fichas: Sequence[Ficha], expected: FichasCountsConfig) -> dict[str, int]:
    """Return the counts of ``fichas`` once they pass every enforced check.

    Each ficha validated when it was made. Across them, every ``doc_code`` must be unique, no
    PDF whose text is used may serve two included theses, and each count of
    :func:`count_fichas` must equal ``expected``.

    Raises:
        FichasCheckError: listing every failed check at once.
    """
    problems = []
    codes = Counter(ficha.doc_code for ficha in fichas)
    repeated = sorted(code for code, count in codes.items() if count > 1)
    if repeated:
        problems.append(f"doc_code must be unique; repeated: {', '.join(repeated)}")
    used = Counter(
        ficha.pdf_sha256
        for ficha in fichas
        if ficha.include and ficha.objectives_status in ("extracted", "not_found")
    )
    shared = sum(count > 1 for count in used.values())
    if shared:
        problems.append(
            f"{shared} PDF serves several included theses; declare the copies in fichas.wrong_pdfs"
        )
    counts = count_fichas(fichas)
    for name, value in counts.items():
        declared = getattr(expected, name)
        if value != declared:
            problems.append(f"{name} is {value}, but fichas.expected declares {declared}")
    if problems:
        raise FichasCheckError(
            f"the fichas of snapshot {expected.snapshot_id} fail {len(problems)} "
            f"check{'s' if len(problems) > 1 else ''}: {'; '.join(problems)}"
        )
    return counts


# Use case


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def build_dataset(
    config: AppConfig,
    project_root: Path,
    *,
    snapshot_id: str | None = None,
    store: TableStore | None = None,
    key: bytes | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> FichasBuild:
    """Build, check and save the fichas of a snapshot, the one harvested last by default.

    ``key`` is the pseudonym key; by default it is loaded, or created once, at
    :func:`~thematic_redundancy.corpus.anonymize.default_key_path`.

    Raises:
        FileNotFoundError: if the snapshot or an input of an earlier step is missing.
        FichasCheckError: if the fichas fail an enforced check; nothing is written then.
        ValueError: if ``fichas.expected`` belongs to another snapshot, if an input is
            damaged, changed or belongs to another snapshot, or if the inputs disagree.
        OSError: if reading or writing the disk fails.
    """
    started = timer()
    store = store or ParquetTableStore()
    data_dir = config.paths.resolve_against(project_root).data_dir
    raw_dir = data_dir / RAW_DIR_NAME
    snapshot_id = check_snapshot_id(snapshot_id or latest_snapshot_id(raw_dir))
    expected = config.fichas.expected
    if snapshot_id != expected.snapshot_id:
        raise ValueError(
            f"fichas.expected declares the counts of snapshot '{expected.snapshot_id}', not of "
            f"'{snapshot_id}'; inspect that snapshot and declare its counts first"
        )
    snapshot_dir = raw_dir / snapshot_id
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"snapshot '{snapshot_id}' not found under {raw_dir}")
    snapshot_manifest, records = read_snapshot(snapshot_dir)
    interim_dir = data_dir / INTERIM_DIR_NAME / snapshot_id
    pdf_manifest = _of_snapshot(
        read_pdf_manifest(snapshot_dir / PDF_DIR_NAME / PDF_MANIFEST_FILE_NAME),
        "PDF",
        snapshot_id,
    )
    text_manifest = _of_snapshot(
        read_text_manifest(interim_dir / PAGES_DIR_NAME / TEXT_MANIFEST_FILE_NAME),
        "text",
        snapshot_id,
    )
    objectives_sha256 = _verified_objectives_table(interim_dir, snapshot_id)
    sources = FichaSources(
        snapshot_id=snapshot_id,
        snapshot_date=snapshot_manifest.harvested_at.date(),
        records=records,
        pdf_entries=pdf_manifest.items,
        text_entries=text_manifest.items,
        objectives=store.read(ObjectivesRecord, interim_dir / OBJECTIVES_FILE_NAME),
    )
    if key is None:
        key = load_or_create_key(default_key_path(config.paths, project_root))
    fichas = build_fichas(sources, config, key)
    counts = check_fichas(fichas, expected)
    handles = {record.uuid: record.handle for record in records}
    fichas_path = interim_dir / FICHAS_FILE_NAME
    exclusions_path = interim_dir / EXCLUSIONS_FILE_NAME
    store.write(fichas, fichas_path, model_cls=Ficha)
    store.write(
        [_exclusion(ficha, handles[ficha.item_uuid]) for ficha in fichas if not ficha.include],
        exclusions_path,
        model_cls=Exclusion,
    )
    manifest = FichasManifest(
        snapshot_id=snapshot_id,
        created_at=clock().astimezone(UTC),
        metadata_sha256=snapshot_manifest.metadata_sha256,
        objectives_table_sha256=objectives_sha256,
        fichas_file=fichas_path.name,
        fichas_sha256=_sha256(fichas_path),
        exclusions_file=exclusions_path.name,
        exclusions_sha256=_sha256(exclusions_path),
        counts=counts,
        seconds=round(max(timer() - started, 0.0), 3),
    )
    write_json_atomically(
        interim_dir / FICHAS_MANIFEST_FILE_NAME, manifest.model_dump_json(indent=2)
    )
    return FichasBuild(
        snapshot_id=snapshot_id,
        fichas=fichas,
        records=records,
        manifest=manifest,
        fichas_path=fichas_path,
    )


def _of_snapshot[M: PdfManifest | TextManifest](
    manifest: M | None, name: str, snapshot_id: str
) -> M:
    """Return ``manifest`` once it exists and belongs to ``snapshot_id``."""
    if manifest is None:
        raise FileNotFoundError(f"snapshot '{snapshot_id}' has no {name} manifest")
    if manifest.snapshot_id != snapshot_id:
        raise ValueError(f"the {name} manifest belongs to snapshot '{manifest.snapshot_id}'")
    return manifest


def _verified_objectives_table(interim_dir: Path, snapshot_id: str) -> str:
    """Return the SHA-256 of the objectives table once it matches its manifest."""
    path = interim_dir / OBJECTIVES_MANIFEST_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no objectives table in {interim_dir}; run {LOCATE_COMMAND} first")
    manifest = ObjectivesManifest.model_validate_json(path.read_bytes())
    digest = _sha256(interim_dir / OBJECTIVES_FILE_NAME)
    if manifest.snapshot_id != snapshot_id or digest != manifest.table_sha256:
        raise ValueError(
            f"{OBJECTIVES_FILE_NAME} changed since {path.name} was written; run "
            f"{LOCATE_COMMAND} again"
        )
    return digest


def _exclusion(ficha: Ficha, handle: str | None) -> Exclusion:
    assert ficha.exclusion_reason is not None
    return Exclusion(
        item_uuid=ficha.item_uuid,
        doc_code=ficha.doc_code,
        handle=handle,
        program_key=ficha.program_key,
        document_type=ficha.document_type,
        exclusion_reason=ficha.exclusion_reason,
        duplicate_of=ficha.duplicate_of,
    )


# Helpers


def _values(metadata: Metadata, key: str) -> list[str]:
    """Return the non-blank text values of the field ``key``, stripped, in order."""
    return [
        text.strip()
        for entry in metadata.get(key, ())
        if isinstance(text := entry.get("value"), str) and text.strip()
    ]


def _first(metadata: Metadata, key: str) -> str | None:
    values = _values(metadata, key)
    return values[0] if values else None


def _folded(text: str) -> str:
    return normalize_text(text).casefold()


def _canonical_order(record: SnapshotRecord) -> tuple[datetime, str]:
    """Order of the records of one D24 group: accessioned first, then the smaller uuid."""
    value = _first(record.metadata, ACCESSIONED_KEY)
    try:
        moment = datetime.fromisoformat(value) if value else None
    except ValueError:
        moment = None
    if moment is None:
        raise ValueError(
            f"record {record.uuid} has no {ACCESSIONED_KEY} that is a date, which the D24 "
            "rule needs to choose the record that stays"
        )
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC), str(record.uuid))


def _on_disk(entry: PdfEntry) -> bool:
    return entry.status in FILE_STATUSES


def _problems(error: ValidationError) -> str:
    """Return the fields and messages of ``error``, without any input value."""
    return "; ".join(
        f"{'.'.join(str(part) for part in issue['loc']) or 'ficha'}: {issue['msg']}"
        for issue in error.errors(include_input=False, include_url=False)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Command line


def describe(built: FichasBuild) -> str:
    """Describe the build in counts only."""
    fichas = built.fichas
    counts = built.manifest.counts
    included = [ficha for ficha in fichas if ficha.include]
    reasons = Counter(ficha.exclusion_reason for ficha in fichas if not ficha.include)
    statuses = Counter(ficha.objectives_status for ficha in included)
    notes = Counter(note for ficha in included for note in ficha.quality_notes)
    pairs = duplicate_pairs(fichas, built.records)
    lines = [
        f"Snapshot {built.snapshot_id}: {counts['records']} records, {counts['included']} "
        f"included theses, {len(fichas) - len(included)} excluded ("
        + ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items()))
        + ")",
        f"Objectives: extracted {counts['objectives_extracted']}, metadata only "
        f"{counts['metadata_only']} ("
        + ", ".join(f"{status} {count}" for status, count in sorted(statuses.items()))
        + ")",
        f"D24: {len(pairs)} duplicate records; the record that stays is also the smaller uuid "
        f"in {sum(pair['kept_is_smaller_uuid'] for pair in pairs)}, has the same PDF in "
        f"{sum(pair['same_pdf'] for pair in pairs)}, and equal objectives text in "
        f"{sum(pair['objectives_equal'] is True for pair in pairs)}",
        f"Advisor ORCID iD not usable (code from the name): {notes[NOTE_INVALID_ADVISOR_ORCID]}",
        f"Checks passed: {len({ficha.doc_code for ficha in fichas})} unique doc codes; every "
        "count as declared in fichas.expected",
        f"Fichas {built.fichas_path}",
        f"Seconds: {built.manifest.seconds:.1f}",
    ]
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.corpus.build_fichas",
        description=(
            "Build one ficha per record of a snapshot, check the declared counts, and save "
            f"them as data/interim/<snapshot>/{FICHAS_FILE_NAME} with the exclusion log."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--snapshot-id", help="snapshot to build (default: the snapshot harvested last)"
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        metavar="DIR",
        help=(
            "also write the quality report, numbers, handles and lemmas only, as "
            "quality_report.json and quality_report.md into DIR"
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    lemmatize: Lemmatizer | None = None,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and
    130 after an interruption."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    try:
        config = load_config(project_root / CONFIG_PATH)
        built = build_dataset(
            config, project_root, snapshot_id=arguments.snapshot_id, clock=clock, timer=timer
        )
        print(describe(built))
        if arguments.report_out is not None:
            started = timer()
            report = quality_report(
                built.fichas,
                built.records,
                program_keys=[program.key for program in config.snapshot.programs],
                domain_stopwords=load_domain_stopwords(project_root / DOMAIN_STOPWORDS_FILE),
                lemmatize=lemmatize or spacy_lemmatizer,
                provenance={
                    "snapshot_id": built.snapshot_id,
                    "created_at": built.manifest.created_at.isoformat().replace("+00:00", "Z"),
                    "metadata_sha256": built.manifest.metadata_sha256,
                    "objectives_table_sha256": built.manifest.objectives_table_sha256,
                    "fichas_sha256": built.manifest.fichas_sha256,
                },
            )
            paths = write_quality_report(arguments.report_out, report, built.records)
            seconds = max(timer() - started, 0.0)
            print(f"Quality report {paths[0].parent} ({seconds:.1f} s, stopword re-check included)")
    except KeyboardInterrupt:
        print("interrupted: nothing half-written was left; run the command again", file=sys.stderr)
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
