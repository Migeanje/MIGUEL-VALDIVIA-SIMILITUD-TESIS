"""Page texts of a snapshot on disk: the extraction manifest format, and its atomic writing.

The page texts of snapshot ``<snapshot_id>`` live in ``data/interim/<snapshot_id>/pages/``,
one Parquet table of :class:`~thematic_redundancy.extraction.page_text.PageText` records per
thesis PDF, named after its item: ``pages/<item uuid>.parquet``. ``pages/manifest.json``
holds a :class:`TextManifest`: the settings of the latest run, one :class:`TextEntry` per PDF
looked at, in snapshot order, and a summary of them. An entry whose PDF the PDF manifest no
longer lists as on disk is kept, with its page file, but counted only as ``orphaned``. The
manifest holds counts only, never text.
"""

import hashlib
import os
from collections import Counter
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    model_validator,
)

from thematic_redundancy.corpus.snapshot import Count, SnapshotId, UtcDatetime
from thematic_redundancy.shared.config import AppConfig

INTERIM_DIR_NAME = "interim"
"""Directory under ``paths.data_dir`` that holds one directory per snapshot."""

PAGES_DIR_NAME = "pages"
"""Directory of a snapshot's interim data that holds its page texts and their manifest."""

TEXT_MANIFEST_FILE_NAME = "manifest.json"

EXTRACTION_VERSION = 2
"""Version of the extraction rules, part of the settings fingerprint that decides whether a
PDF is skipped. Raise it whenever a change of the code alters the page records, such as a
change of the page-text rules, so that every PDF is extracted again:

- 1: the first rules (T09);
- 2: an edge line of digits only, such as a year, goes only when it is a page number (T09a).
"""

TextStatus = Literal["ok", "error"]
"""Outcome of the latest extraction of a PDF:

- ``ok``: its page records are on disk;
- ``error``: the PDF could not be read, so it has no page records; a later run tries again.
"""

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Seconds = Annotated[StrictFloat, Field(ge=0.0)]


def pages_file_name(item: UUID) -> str:
    """Return the name under which the page records of ``item`` are saved."""
    return f"{item}.parquet"


class _TextModel(BaseModel):
    """Base of the extraction manifest formats: immutable, closed to unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ExtractionSettings(_TextModel):
    """Everything that shapes the page records of a PDF, besides the PDF itself."""

    extraction_version: Annotated[StrictInt, Field(ge=1)]
    ocr_languages: str
    ocr_dpi: Count
    min_text_chars: Count
    ocr_window_pages: Count
    header_footer_edge_lines: Count
    header_footer_min_share: StrictFloat
    header_footer_min_pages: Count

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        header_footer = config.extraction.header_footer
        return cls(
            extraction_version=EXTRACTION_VERSION,
            ocr_languages=config.ocr.languages,
            ocr_dpi=config.ocr.dpi,
            min_text_chars=config.extraction.min_text_chars,
            ocr_window_pages=config.extraction.ocr_window_pages,
            header_footer_edge_lines=header_footer.edge_lines,
            header_footer_min_share=header_footer.min_share,
            header_footer_min_pages=header_footer.min_pages,
        )

    def fingerprint(self) -> str:
        """Return the SHA-256 of these settings, which changes whenever one of them does."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class TextEntry(_TextModel):
    """What the latest extraction of one thesis PDF found."""

    program_key: str
    status: TextStatus
    reason: str | None = None
    """Short code of an ``error``: ``unreadable_pdf``, ``pdf_missing`` or ``pdf_changed``."""
    detail: str | None = None
    """Readable explanation of the reason; it never holds text of the PDF."""
    pdf_sha256: Sha256
    """SHA-256 of the PDF, as the PDF manifest records it."""
    config_fingerprint: Sha256
    """Fingerprint of the :class:`ExtractionSettings` used."""
    file: str | None = None
    """Name of the page records file in the pages directory, set when the status is ``ok``."""
    output_sha256: Sha256 | None = None
    """SHA-256 of that file, which tells whether it is still the one written."""
    pages: Count = 0
    text_layer_pages: Count = 0
    ocr_pages: Count = 0
    empty_pages: Count = 0
    """Pages by the source of their text; together they are ``pages``."""
    low_text_pages: Count = 0
    ocr_attempted: Count = 0
    ocr_failed: Count = 0
    unreadable_pages: Count = 0
    """Pages whose text layer could not be read; they count as having no text layer."""
    lines_removed: Count = 0
    """Running header, footer and page-number lines removed from the pages."""
    pdf_warnings: Count = 0
    """Warnings that the PDF library reported while reading the PDF."""
    seconds: Seconds = 0.0
    """Time that the extraction took, OCR included."""
    ocr_seconds: Seconds = 0.0
    """Time spent in OCR alone."""
    extracted_at: UtcDatetime

    @model_validator(mode="after")
    def _require_consistent_fields(self) -> Self:
        if self.status == "ok":
            if self.file is None or self.output_sha256 is None or self.reason is not None:
                raise ValueError("an 'ok' entry needs file and output_sha256, and no reason")
        elif self.reason is None or self.file is not None or self.output_sha256 is not None:
            raise ValueError("an 'error' entry needs a reason, and no file")
        if self.text_layer_pages + self.ocr_pages + self.empty_pages != self.pages:
            raise ValueError("the pages by source must add up to pages")
        if self.ocr_failed > self.ocr_attempted or self.ocr_pages > self.ocr_attempted:
            raise ValueError("OCR pages and failures cannot exceed the OCR attempts")
        return self


class TextSummary(_TextModel):
    """Counts over the entries of a manifest whose PDF the PDF manifest still lists as on
    disk; the other entries are only counted as ``orphaned``."""

    items: Count
    ok: Count
    error: Count
    stale: Count
    """``ok`` entries extracted with other settings than the manifest's."""
    orphaned: Count = 0
    """Entries kept, with their page files, although the PDF manifest no longer lists their
    PDF as on disk, such as an item restricted since; they count nowhere else."""
    pages: Count
    text_layer_pages: Count
    ocr_pages: Count
    empty_pages: Count
    low_text_pages: Count
    ocr_attempted: Count
    ocr_failed: Count
    unreadable_pages: Count
    lines_removed: Count
    pdf_warnings: Count
    seconds: Seconds
    ocr_seconds: Seconds
    by_program: dict[str, dict[str, Count]]
    """Entries per status within each program, in snapshot order."""


class TextManifest(_TextModel):
    """Content of ``pages/manifest.json``."""

    snapshot_id: SnapshotId
    settings: ExtractionSettings
    """Settings of the latest run."""
    config_fingerprint: Sha256
    """Fingerprint of ``settings``."""
    updated_at: UtcDatetime
    summary: TextSummary
    items: dict[UUID, TextEntry]
    """Entries keyed by item uuid, in snapshot order."""

    @model_validator(mode="after")
    def _require_files_named_after_their_item(self) -> Self:
        if self.config_fingerprint != self.settings.fingerprint():
            raise ValueError("config_fingerprint must be the fingerprint of settings")
        for item, entry in self.items.items():
            if entry.file is not None and entry.file != pages_file_name(item):
                raise ValueError(
                    f"the page records of item {item} must be named {pages_file_name(item)}"
                )
        return self


_COUNTED = (
    "pages",
    "text_layer_pages",
    "ocr_pages",
    "empty_pages",
    "low_text_pages",
    "ocr_attempted",
    "ocr_failed",
    "unreadable_pages",
    "lines_removed",
    "pdf_warnings",
)
"""Entry fields that a summary adds up."""


def summarize(
    entries: Mapping[UUID, TextEntry], fingerprint: str, available: Collection[UUID]
) -> TextSummary:
    """Count ``entries`` by status and by program, and add up their page counts and times;
    ``fingerprint`` is the one of the current settings.

    Only the entries of the ``available`` items, whose PDF the PDF manifest lists as on
    disk, are counted; the others are only counted as ``orphaned``.
    """
    current = [entry for item, entry in entries.items() if item in available]
    by_program: dict[str, Counter[str]] = {}
    for entry in current:
        by_program.setdefault(entry.program_key, Counter())[entry.status] += 1
    statuses = Counter(entry.status for entry in current)
    return TextSummary(
        items=len(current),
        ok=statuses["ok"],
        error=statuses["error"],
        stale=sum(
            entry.status == "ok" and entry.config_fingerprint != fingerprint for entry in current
        ),
        orphaned=len(entries) - len(current),
        **{name: sum(getattr(entry, name) for entry in current) for name in _COUNTED},
        seconds=round(sum(entry.seconds for entry in current), 3),
        ocr_seconds=round(sum(entry.ocr_seconds for entry in current), 3),
        by_program={program: dict(counts) for program, counts in by_program.items()},
    )


def read_text_manifest(path: Path) -> TextManifest | None:
    """Return the manifest saved at ``path``, or ``None`` when there is none yet.

    Raises:
        ValueError: if the file is not a valid manifest.
    """
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        return TextManifest.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            f"the text manifest {path} is damaged ({error.error_count()} problems); move it "
            "away to rebuild it, which extracts every PDF again"
        ) from error


def write_json_atomically(path: Path, content: str) -> None:
    """Replace the file at ``path`` with ``content`` atomically and durably.

    The content goes to ``<path>.part`` first and is flushed to the disk, and one rename
    then puts it in place, so readers see the previous file or the new one, never a
    partial one.
    """
    staged = path.with_name(f"{path.name}.part")
    try:
        with staged.open("wb") as file:
            file.write(f"{content}\n".encode())
            file.flush()
            os.fsync(file.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def write_text_manifest(path: Path, manifest: TextManifest) -> None:
    """Replace the manifest at ``path`` atomically and durably."""
    write_json_atomically(path, manifest.model_dump_json(indent=2))
