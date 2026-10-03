"""Thesis PDFs of a snapshot on disk: the manifest format, and its atomic writing.

The PDFs of snapshot ``<snapshot_id>`` live in ``data/raw/<snapshot_id>/pdfs/``, each named
after its item's uuid, such as ``pdfs/0e2b8a51-....pdf``. The original file names are never
used or recorded, because they may hold author names. ``pdfs/manifest.json`` holds a
:class:`PdfManifest`: one :class:`PdfEntry` per thesis looked at, in snapshot order, and a
summary of them.
"""

import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from thematic_redundancy.corpus.snapshot import Count, SnapshotId, UtcDatetime

PDF_DIR_NAME = "pdfs"
"""Directory of a snapshot that holds its thesis PDFs and their manifest."""

PDF_MANIFEST_FILE_NAME = "manifest.json"

PdfStatus = Literal[
    "downloaded",
    "already_present",
    "restricted",
    "not_found",
    "no_thesis_file",
    "ambiguous",
    "integrity_error",
    "error",
]
"""Outcome of the latest look at a thesis:

- ``downloaded``: its PDF was fetched and verified;
- ``already_present``: its verified PDF was already on disk, so nothing was fetched;
- ``restricted``: anonymous users may not read it, by its ``dc.rights`` (no request made)
  or by an HTTP 401 or 403 answer;
- ``not_found``: the repository answered HTTP 404;
- ``no_thesis_file`` / ``ambiguous``: its ``ORIGINAL`` bundle holds no candidate PDF, or
  several, so nothing was fetched;
- ``integrity_error``: the PDF failed a check and was deleted;
- ``error``: the repository failed, even after the retries.
"""

FILE_STATUSES: frozenset[PdfStatus] = frozenset({"downloaded", "already_present"})
"""Statuses of an entry whose verified PDF is on disk."""

_Md5 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_FILE_FIELDS = (
    "file",
    "bitstream_uuid",
    "size_bytes",
    "checksum_algorithm",
    "checksum",
    "md5",
    "sha256",
)
"""Fields that an entry with a PDF on disk must set."""


def pdf_file_name(item: UUID) -> str:
    """Return the name under which the thesis PDF of ``item`` is saved."""
    return f"{item}.pdf"


class _PdfModel(BaseModel):
    """Base of the PDF manifest formats: immutable, closed to unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class PdfEntry(_PdfModel):
    """What the latest look at one thesis found."""

    program_key: str
    status: PdfStatus
    reason: str | None = None
    """Short code behind a status without a file, such as ``rights_restricted``,
    ``http_403``, ``multiple_candidates`` or ``checksum_mismatch``."""
    detail: str | None = None
    """Readable explanation of the reason; it never holds a file name."""
    file: str | None = None
    """Name of the PDF in the PDF directory, set when the PDF is on disk."""
    bitstream_uuid: UUID | None = None
    """Repository file chosen as the thesis."""
    size_bytes: Count | None = None
    """Size that the repository lists for that file; a PDF on disk has exactly this size."""
    checksum_algorithm: str | None = None
    """Algorithm of ``checksum``, as the repository spells it, such as ``MD5``."""
    checksum: str | None = None
    """Digest that the repository lists for that file; a PDF on disk matches it."""
    md5: _Md5 | None = None
    sha256: _Sha256 | None = None
    """Digests of the PDF on disk."""
    candidate_count: Count | None = None
    """PDFs of the ``ORIGINAL`` bundle that could be the thesis; unset when not listed."""
    downloaded_at: UtcDatetime | None = None
    """When the PDF on disk was fetched, if a run of this manifest fetched it."""
    checked_at: UtcDatetime
    """When this entry was last established."""

    @model_validator(mode="after")
    def _require_file_fields_exactly_with_a_file(self) -> Self:
        if self.status in FILE_STATUSES:
            missing = [name for name in _FILE_FIELDS if getattr(self, name) is None]
            if missing:
                raise ValueError(f"a '{self.status}' entry needs {', '.join(missing)}")
        elif self.file is not None:
            raise ValueError(f"a '{self.status}' entry has no file on disk")
        return self


class PdfSummary(_PdfModel):
    """Counts over every entry of a manifest."""

    items: Count
    files: Count
    """Entries whose verified PDF is on disk."""
    total_bytes: Count
    """Size of those PDFs together."""
    by_status: dict[str, Count]
    by_program: dict[str, dict[str, Count]]
    """Entries per status within each program, in snapshot order."""


class PdfManifest(_PdfModel):
    """Content of ``pdfs/manifest.json``."""

    snapshot_id: SnapshotId
    base_url: str
    """Repository that the PDFs came from."""
    thesis_type: str
    """``renati.type`` fragment that made an item a thesis."""
    selection_rule: str
    """How the thesis file of an item was told from its other files."""
    updated_at: UtcDatetime
    summary: PdfSummary
    items: dict[UUID, PdfEntry]
    """Entries keyed by item uuid, in snapshot order."""

    @model_validator(mode="after")
    def _require_files_named_after_their_item(self) -> Self:
        for item, entry in self.items.items():
            if entry.file is not None and entry.file != pdf_file_name(item):
                raise ValueError(f"the file of item {item} must be named {pdf_file_name(item)}")
        return self


def summarize(entries: Mapping[UUID, PdfEntry]) -> PdfSummary:
    """Count ``entries`` by status and by program, and add up the size of their files."""
    by_program: dict[str, Counter[str]] = {}
    for entry in entries.values():
        by_program.setdefault(entry.program_key, Counter())[entry.status] += 1
    files = [entry for entry in entries.values() if entry.status in FILE_STATUSES]
    return PdfSummary(
        items=len(entries),
        files=len(files),
        total_bytes=sum(entry.size_bytes or 0 for entry in files),
        by_status=dict(Counter(entry.status for entry in entries.values())),
        by_program={program: dict(counts) for program, counts in by_program.items()},
    )


def read_pdf_manifest(path: Path) -> PdfManifest | None:
    """Return the manifest saved at ``path``, or ``None`` when there is none yet.

    Raises:
        ValueError: if the file is not a valid manifest. It is not replaced silently,
            because it may hold the only record of refusals that are never asked again.
    """
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        return PdfManifest.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            f"the PDF manifest {path} is damaged ({error.error_count()} problems); move it "
            "away to rebuild it, which checks every thesis against the repository again"
        ) from error


def write_pdf_manifest(path: Path, manifest: PdfManifest) -> None:
    """Replace the manifest at ``path`` atomically and durably.

    The content goes to ``<path>.part`` first and is flushed to the disk, and one rename
    then puts it in place, so readers see the previous manifest or the new one, never a
    partial one.
    """
    staged = path.with_name(f"{path.name}.part")
    try:
        with staged.open("wb") as file:
            file.write(f"{manifest.model_dump_json(indent=2)}\n".encode())
            file.flush()
            os.fsync(file.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
