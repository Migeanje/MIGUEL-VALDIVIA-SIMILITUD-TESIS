"""Raw corpus snapshot on disk: the record and manifest formats, and their atomic writing.

A snapshot is the directory ``<data_dir>/raw/<snapshot_id>/``, holding two files:

- ``metadata.jsonl``: one :class:`SnapshotRecord` per line, as UTF-8 JSON with LF endings;
- ``manifest.json``: the :class:`SnapshotManifest`, which describes the harvest and holds
  the SHA-256 of ``metadata.jsonl``.

Snapshots are immutable. A directory is written once, as a whole, and is never overwritten.
Readers check ``metadata.jsonl`` against the SHA-256 in the manifest before using it. Later
steps may add their own subdirectories, such as ``pdfs/``, but never touch those two files.
"""

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    ValidationError,
    field_validator,
)

RAW_DIR_NAME = "raw"
"""Directory under ``paths.data_dir`` that holds one directory per snapshot."""

METADATA_FILE_NAME = "metadata.jsonl"
MANIFEST_FILE_NAME = "manifest.json"

IDENTITY_NUMBER_SUFFIX = ".dni"
"""Metadata keys with this ending, in any letter case, hold national identity numbers."""

_SNAPSHOT_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?")
_DEFAULT_ID_FORMAT = "%Y%m%dT%H%M%SZ"


class SnapshotExistsError(FileExistsError):
    """Something already exists where a new snapshot would go; snapshots are never overwritten."""


def default_snapshot_id(moment: datetime) -> str:
    """Return the id of a snapshot harvested at the UTC time ``moment``.

    For example, 2026-10-02 15:04:05 UTC gives ``20261002T150405Z``.
    """
    return moment.strftime(_DEFAULT_ID_FORMAT)


def check_snapshot_id(snapshot_id: str) -> str:
    """Return ``snapshot_id`` when it is one plain directory name.

    An id has 1 to 64 letters, digits, dots, hyphens or underscores, and starts and ends
    with a letter or a digit, so it can neither leave ``data/raw`` nor name a hidden entry.

    Raises:
        ValueError: if ``snapshot_id`` breaks these rules.
    """
    if not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError(
            f"snapshot id {snapshot_id!r} must have 1 to 64 letters, digits, '.', '-' or '_', "
            "and start and end with a letter or a digit"
        )
    return snapshot_id


def require_new_snapshot(snapshot_dir: Path) -> None:
    """Raise :class:`SnapshotExistsError` when anything already exists at ``snapshot_dir``."""
    if snapshot_dir.exists() or snapshot_dir.is_symlink():
        raise SnapshotExistsError(_exists_message(snapshot_dir))


def is_identity_number_key(key: str) -> bool:
    """Tell whether a metadata key holds a national identity number (DNI)."""
    return key.casefold().endswith(IDENTITY_NUMBER_SUFFIX)


def without_identity_numbers[V](metadata: Mapping[str, V]) -> tuple[dict[str, V], tuple[str, ...]]:
    """Return the metadata without its DNI fields, and the sorted names of those fields."""
    kept = {key: values for key, values in metadata.items() if not is_identity_number_key(key)}
    dropped = tuple(sorted(key for key in metadata if is_identity_number_key(key)))
    return kept, dropped


def _require_utc(value: datetime) -> datetime:
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"must be a UTC time, got '{value.isoformat()}'")
    return value


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_require_utc)]
SnapshotId = Annotated[str, AfterValidator(check_snapshot_id)]
Count = Annotated[StrictInt, Field(ge=0)]


class _SnapshotModel(BaseModel):
    """Base of the snapshot formats: immutable, closed to unknown keys, input kept out of errors."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class SnapshotRecord(_SnapshotModel):
    """One harvested item: its identity, its program, and its public metadata."""

    uuid: UUID
    handle: str | None
    program_key: str
    collection_uuid: UUID
    """Collection that listed the item; the item's program comes from it."""
    harvested_at: UtcDatetime
    metadata: dict[str, list[dict[str, JsonValue]]]
    """Metadata as the repository serves it, minus the DNI fields."""

    @field_validator("metadata")
    @classmethod
    def _refuse_identity_numbers(
        cls, value: dict[str, list[dict[str, JsonValue]]]
    ) -> dict[str, list[dict[str, JsonValue]]]:
        leaked = sorted(key for key in value if is_identity_number_key(key))
        if leaked:
            raise ValueError(f"DNI fields must be dropped before a record is built: {leaked}")
        return value


class ProgramSummary(_SnapshotModel):
    """Items harvested for one program, and the collection that they came from."""

    collection_uuid: UUID
    items: Count


class SnapshotManifest(_SnapshotModel):
    """Provenance and summary of one snapshot, saved as ``manifest.json``."""

    snapshot_id: SnapshotId
    harvested_at: UtcDatetime
    base_url: str
    search_url: str
    query_parameters: dict[str, str]
    """Parameters that every search sent; each search added its scope and its page."""
    programs: dict[str, ProgramSummary]
    """Programs in configuration order, keyed by program key."""
    total_items: Count
    faculty_community_uuid: UUID
    faculty_total: Count
    """Items of the faculty community under the same filter, as the repository reported them.
    It equals ``total_items`` when the program collections cover the whole faculty."""
    items_by_type: dict[str, Count]
    """Items per ``renati.type`` value. An item with several values counts under each one,
    and an item without one counts under ``(none)``."""
    future_dated_items: Count
    """Items whose first ``dc.date.issued`` lies after the UTC date of ``harvested_at``. A
    year or a month counts from its first day, so it only counts when it starts later."""
    unparsed_issue_dates: Count
    """Items whose first ``dc.date.issued`` is missing or is not a date."""
    dropped_metadata_keys: tuple[str, ...]
    """Names of the DNI fields that were dropped before writing, sorted."""
    metadata_file: str = METADATA_FILE_NAME
    metadata_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def encode_records(records: Iterable[SnapshotRecord]) -> bytes:
    """Serialize ``records`` as JSON lines: UTF-8, one object per line, each ended by LF."""
    return b"".join(f"{record.model_dump_json()}\n".encode() for record in records)


def write_snapshot(snapshot_dir: Path, metadata: bytes, manifest: SnapshotManifest) -> None:
    """Write a snapshot directory as a whole, so it appears complete or not at all.

    Both files go into a hidden staging directory beside ``snapshot_dir``, and one rename
    then turns it into ``snapshot_dir``. The rename refuses a target that already exists,
    except an empty directory on POSIX, so an existing snapshot is never overwritten. Any
    failure removes the staging directory.

    Raises:
        SnapshotExistsError: if ``snapshot_dir`` appeared before the rename.
    """
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=snapshot_dir.parent, prefix=f".{snapshot_dir.name}.", suffix=".partial"
        )
    )
    try:
        _write_durably(staging / METADATA_FILE_NAME, metadata)
        _write_durably(
            staging / MANIFEST_FILE_NAME, f"{manifest.model_dump_json(indent=2)}\n".encode()
        )
        try:
            os.rename(staging, snapshot_dir)
        except OSError as error:
            if snapshot_dir.exists():
                raise SnapshotExistsError(_exists_message(snapshot_dir)) from error
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_snapshot_manifest(snapshot_dir: Path) -> SnapshotManifest:
    """Return the manifest of the snapshot at ``snapshot_dir``.

    Raises:
        FileNotFoundError: if the snapshot has no manifest.
        ValueError: if the manifest is not a valid snapshot manifest.
    """
    path = snapshot_dir / MANIFEST_FILE_NAME
    content = path.read_bytes()
    try:
        return SnapshotManifest.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            f"{path} is not a valid snapshot manifest ({error.error_count()} problems)"
        ) from error


def read_snapshot(snapshot_dir: Path) -> tuple[SnapshotManifest, tuple[SnapshotRecord, ...]]:
    """Return the manifest and the records of the snapshot at ``snapshot_dir``.

    The records are read only once ``metadata.jsonl`` matches the SHA-256 in the manifest,
    so a snapshot changed after its harvest is never used.

    Raises:
        FileNotFoundError: if a file of the snapshot is missing.
        ValueError: if the manifest is invalid, if ``metadata.jsonl`` does not match its
            SHA-256, or if a record is invalid.
    """
    manifest = read_snapshot_manifest(snapshot_dir)
    path = snapshot_dir / METADATA_FILE_NAME
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != manifest.metadata_sha256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, but its manifest records {manifest.metadata_sha256}; "
            "the snapshot changed after the harvest, so it is not used"
        )
    records = []
    for number, line in enumerate(content.splitlines(), start=1):
        try:
            records.append(SnapshotRecord.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"line {number} of {path} is not a valid record") from error
    return manifest, tuple(records)


def latest_snapshot_id(raw_dir: Path) -> str:
    """Return the id of the snapshot under ``raw_dir`` that was harvested last.

    Only directories named like a snapshot id that hold a manifest count, so the hidden
    staging directory of an unfinished harvest and stray files are skipped. Snapshots
    harvested in the same second are told apart by the greater id.

    Raises:
        FileNotFoundError: if ``raw_dir`` holds no snapshot.
        ValueError: if a snapshot manifest is invalid.
    """
    harvests = []
    if raw_dir.is_dir():
        for path in raw_dir.iterdir():
            if (
                _SNAPSHOT_ID.fullmatch(path.name)
                and path.is_dir()
                and (path / MANIFEST_FILE_NAME).is_file()
            ):
                harvests.append((read_snapshot_manifest(path).harvested_at, path.name))
    if not harvests:
        raise FileNotFoundError(f"no snapshot under {raw_dir}; run the harvest first")
    return max(harvests)[1]


def _write_durably(path: Path, content: bytes) -> None:
    """Create ``path`` with ``content`` and flush it to the disk before returning."""
    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def _exists_message(snapshot_dir: Path) -> str:
    return (
        f"snapshot '{snapshot_dir.name}' already exists at {snapshot_dir}; snapshots are never "
        "overwritten, so choose another --snapshot-id"
    )
