"""Extract the text of every thesis PDF of a snapshot, page by page, resumably.

:func:`extract_texts` reads the PDF manifest of a snapshot, the latest one by default, and
takes every thesis whose PDF is on disk (``downloaded`` or ``already_present``), in snapshot
order. Business rules such as the D24 duplicates or the wrong file of item 11777 belong to
the fichas (T12), so every PDF is extracted. For each PDF it:

1. skips it when its page records are on disk, unchanged, and were made from the PDF whose
   SHA-256 the PDF manifest records, with the same settings and extraction rules
   (:class:`ExtractionSettings` fingerprint). The skip does not hash the PDF on disk: the
   snapshot is frozen, and the data card's combined PDF digest verifies its files;
2. checks the PDF against the SHA-256 that the PDF manifest records;
3. reads the text layer of every page; a low-text page among the first
   ``extraction.ocr_window_pages`` pages is OCR'd as well (D25);
4. removes the running headers, footers and page numbers across the document, and saves
   one :class:`~thematic_redundancy.extraction.page_text.PageText` record per page into
   ``data/interim/<snapshot_id>/pages/<item uuid>.parquet``, atomically.

A PDF that cannot be read is recorded as ``error`` and the run goes on; a page whose OCR
fails keeps its text layer. The manifest ``pages/manifest.json`` is replaced atomically after
every PDF, so an interrupted run loses nothing, and running the command again resumes it.
An entry whose PDF the PDF manifest no longer lists as on disk is ``orphaned``: it is kept,
with its page records, but left out of the totals. One run at a time may use a pages
directory.

Run it after the PDF download; it takes hours, so run it in your own terminal::

    uv run python -m thematic_redundancy.extraction.extract_text [--project-root PATH]
        [--snapshot-id ID] [--limit N] [--only UUID ...] [--summary-out PATH]
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID

import yaml

from thematic_redundancy.corpus.pdf_manifest import (
    FILE_STATUSES,
    PDF_DIR_NAME,
    PDF_MANIFEST_FILE_NAME,
    PdfEntry,
    pdf_file_name,
    read_pdf_manifest,
)
from thematic_redundancy.corpus.snapshot import RAW_DIR_NAME, check_snapshot_id, latest_snapshot_id
from thematic_redundancy.extraction.page_text import (
    PageReading,
    PageText,
    assemble_pages,
    needs_ocr,
)
from thematic_redundancy.extraction.pdf_reader import (
    OcrError,
    PdfPages,
    PdfReader,
    PdfReadError,
    TextLayer,
)
from thematic_redundancy.extraction.pymupdf_reader import PyMuPdfReader
from thematic_redundancy.extraction.text_manifest import (
    INTERIM_DIR_NAME,
    PAGES_DIR_NAME,
    TEXT_MANIFEST_FILE_NAME,
    ExtractionSettings,
    TextEntry,
    TextManifest,
    TextSummary,
    pages_file_name,
    read_text_manifest,
    summarize,
    write_json_atomically,
    write_text_manifest,
)
from thematic_redundancy.shared.config import AppConfig, ExtractionConfig, load_config
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

DOWNLOAD_COMMAND = "uv run python -m thematic_redundancy.corpus.download_pdfs"
"""Command that fetches the PDFs; error messages point to it."""

PART_SUFFIX = ".part"
"""Ending of a file still being written; a run deletes any that an earlier run left."""

LOCK_FILE_NAME = ".lock"
"""File in the pages directory that a run locks, so that only one run uses the directory."""

_READ_CHUNK_BYTES = 1024 * 1024


class ExtractionInProgressError(RuntimeError):
    """Another run is using the same pages directory."""


@dataclass(frozen=True)
class SelectedPdf:
    """A thesis PDF that a run will look at."""

    item_uuid: UUID
    program_key: str


@dataclass(frozen=True)
class ExtractionPlan:
    """What a run is about to do, announced once it holds the pages directory."""

    snapshot_id: str
    pages_dir: Path
    pdfs: int
    """Thesis PDFs on disk in the snapshot, selected or not."""
    selected: tuple[SelectedPdf, ...]
    """PDFs that the run will look at, in order."""
    orphaned: tuple[UUID, ...]
    """Items of the text manifest whose PDF the PDF manifest no longer lists as on disk."""


@dataclass(frozen=True)
class ItemOutcome:
    """The manifest entry of one PDF after a run looked at it."""

    item_uuid: UUID
    program_key: str
    entry: TextEntry
    skipped: bool
    """The page records were current, so the PDF was not read."""


@dataclass(frozen=True)
class ExtractionReport:
    """Outcome of :func:`extract_texts`."""

    snapshot_id: str
    pages_dir: Path
    pdfs: int
    """Thesis PDFs on disk in the snapshot, selected or not."""
    outcomes: tuple[ItemOutcome, ...]
    """Outcome of each PDF that this run looked at, in order."""
    manifest: TextManifest
    """The manifest as saved, which also covers PDFs of earlier runs."""

    @property
    def manifest_path(self) -> Path:
        return self.pages_dir / TEXT_MANIFEST_FILE_NAME


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def extract_texts(
    config: AppConfig,
    project_root: Path,
    reader: PdfReader,
    *,
    store: TableStore | None = None,
    snapshot_id: str | None = None,
    limit: int | None = None,
    only: Sequence[UUID] | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
    on_start: Callable[[ExtractionPlan], None] | None = None,
    on_item: Callable[[int, ItemOutcome], None] | None = None,
) -> ExtractionReport:
    """Extract the page texts of a snapshot's thesis PDFs into its interim ``pages/``.

    The snapshot is ``data/raw/<snapshot_id>`` under ``paths.data_dir``, and defaults to the
    one harvested last. ``only`` keeps the PDFs of those items, and ``limit`` the first PDFs
    after that, both in snapshot order. ``store`` saves the page records, as Parquet by
    default. ``timer`` measures the time of each PDF and of its OCR. ``on_start`` hears the
    plan once the run holds the pages directory, and ``on_item`` each outcome with its
    position, counted from 1, once the manifest holds it.

    Raises:
        ExtractionInProgressError: if another run is using the pages directory.
        FileNotFoundError: if the snapshot or its PDF manifest does not exist.
        ValueError: if ``limit`` is below 1, if ``only`` names an item without a PDF, if the
            snapshot id is not one plain name, or if a manifest is damaged or belongs to
            another snapshot.
        OSError: if reading or writing the disk fails.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    data_dir = config.paths.resolve_against(project_root).data_dir
    raw_dir = data_dir / RAW_DIR_NAME
    if snapshot_id is None:
        snapshot_id = latest_snapshot_id(raw_dir)
    snapshot_dir = raw_dir / check_snapshot_id(snapshot_id)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"snapshot '{snapshot_id}' not found under {raw_dir}")
    pdf_dir = snapshot_dir / PDF_DIR_NAME
    pdf_manifest = read_pdf_manifest(pdf_dir / PDF_MANIFEST_FILE_NAME)
    if pdf_manifest is None:
        raise FileNotFoundError(f"no PDF manifest in {pdf_dir}; run {DOWNLOAD_COMMAND} first")
    if pdf_manifest.snapshot_id != snapshot_id:
        raise ValueError(
            f"the PDF manifest in {pdf_dir} belongs to snapshot '{pdf_manifest.snapshot_id}'"
        )
    available = {
        item: entry for item, entry in pdf_manifest.items.items() if entry.status in FILE_STATUSES
    }
    selected = _select(available, snapshot_id, only=only, limit=limit)
    settings = ExtractionSettings.from_config(config)
    pages_dir = data_dir / INTERIM_DIR_NAME / snapshot_id / PAGES_DIR_NAME
    pages_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_run(pages_dir):
        _remove_partial_files(pages_dir)
        manifests = _ManifestStore(
            pages_dir / TEXT_MANIFEST_FILE_NAME,
            snapshot_id,
            settings,
            order=tuple(pdf_manifest.items),
            available=frozenset(available),
        )
        extractor = _PdfExtractor(
            reader,
            store or ParquetTableStore(),
            config.extraction,
            settings.fingerprint(),
            pdf_dir=pdf_dir,
            pages_dir=pages_dir,
            timer=timer,
        )
        if on_start is not None:
            on_start(
                ExtractionPlan(
                    snapshot_id=snapshot_id,
                    pages_dir=pages_dir,
                    pdfs=len(available),
                    selected=tuple(SelectedPdf(item, pdf.program_key) for item, pdf in selected),
                    orphaned=manifests.orphaned(),
                )
            )
        outcomes: list[ItemOutcome] = []
        for position, (item, pdf) in enumerate(selected, start=1):
            previous = manifests.entry(item)
            if previous is not None and extractor.is_current(item, pdf, previous):
                outcome = ItemOutcome(item, pdf.program_key, previous, skipped=True)
            else:
                now = _as_utc(clock())
                entry = extractor.extract(item, pdf, now)
                manifests.save(item, entry, now)
                outcome = ItemOutcome(item, pdf.program_key, entry, skipped=False)
            outcomes.append(outcome)
            if on_item is not None:
                on_item(position, outcome)
        manifest = manifests.manifest(_as_utc(clock()))
    return ExtractionReport(
        snapshot_id=snapshot_id,
        pages_dir=pages_dir,
        pdfs=len(available),
        outcomes=tuple(outcomes),
        manifest=manifest,
    )


def _as_utc(moment: datetime) -> datetime:
    if moment.utcoffset() is None:
        raise ValueError(
            f"the extraction clock must give a time with a time zone, got '{moment.isoformat()}'"
        )
    return moment.astimezone(UTC)


def _select(
    available: Mapping[UUID, PdfEntry],
    snapshot_id: str,
    *,
    only: Sequence[UUID] | None,
    limit: int | None,
) -> list[tuple[UUID, PdfEntry]]:
    """Keep the PDFs of ``only``, when given, then the first ``limit``, in snapshot order."""
    selected = list(available.items())
    if only is not None:
        unknown = sorted(str(item) for item in set(only) - available.keys())
        if unknown:
            raise ValueError(
                f"no thesis PDF on disk in snapshot '{snapshot_id}' for item(s) "
                f"{', '.join(unknown)}"
            )
        wanted = set(only)
        selected = [(item, pdf) for item, pdf in selected if item in wanted]
    return selected if limit is None else selected[:limit]


class _ManifestStore:
    """The entries of the text manifest, saved to disk after every change.

    An entry whose item is not ``available``, because the PDF manifest no longer lists its
    PDF as on disk, is orphaned: it is kept, and so is its page file, but the summary leaves
    it out of the totals. Nothing is deleted automatically.
    """

    def __init__(
        self,
        path: Path,
        snapshot_id: str,
        settings: ExtractionSettings,
        *,
        order: Sequence[UUID],
        available: Collection[UUID],
    ) -> None:
        saved = read_text_manifest(path)
        if saved is not None and saved.snapshot_id != snapshot_id:
            raise ValueError(
                f"the text manifest {path} belongs to snapshot '{saved.snapshot_id}', not to "
                f"'{snapshot_id}'"
            )
        self._path = path
        self._saved = saved
        self._entries: dict[UUID, TextEntry] = dict(saved.items) if saved is not None else {}
        self._order = {item: index for index, item in enumerate(order)}
        self._available = available
        self._snapshot_id = snapshot_id
        self._settings = settings
        self._fingerprint = settings.fingerprint()

    def entry(self, item: UUID) -> TextEntry | None:
        return self._entries.get(item)

    def orphaned(self) -> tuple[UUID, ...]:
        """Return the orphaned items, in snapshot order."""
        return tuple(item for item in self._sorted() if item not in self._available)

    def save(self, item: UUID, entry: TextEntry, now: datetime) -> None:
        self._entries[item] = entry
        self._saved = self._build(now)
        write_text_manifest(self._path, self._saved)

    def manifest(self, now: datetime) -> TextManifest:
        """Return the manifest as saved; it is written when none exists yet, when it holds
        the settings of another run, or when its summary is out of date, such as after an
        item became orphaned."""
        if (
            self._saved is None
            or self._saved.config_fingerprint != self._fingerprint
            or self._saved.summary != self._summary(self._sorted())
        ):
            self._saved = self._build(now)
            write_text_manifest(self._path, self._saved)
        return self._saved

    def _sorted(self) -> dict[UUID, TextEntry]:
        last = len(self._order)
        return dict(sorted(self._entries.items(), key=lambda item: self._order.get(item[0], last)))

    def _summary(self, entries: Mapping[UUID, TextEntry]) -> TextSummary:
        return summarize(entries, self._fingerprint, self._available)

    def _build(self, now: datetime) -> TextManifest:
        entries = self._sorted()
        return TextManifest(
            snapshot_id=self._snapshot_id,
            settings=self._settings,
            config_fingerprint=self._fingerprint,
            updated_at=now,
            summary=self._summary(entries),
            items=entries,
        )


class _ExtractionFailure(Exception):
    """A PDF yields no page records."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class _PdfExtractor:
    """Establishes the manifest entry and the page records of one PDF."""

    def __init__(
        self,
        reader: PdfReader,
        store: TableStore,
        settings: ExtractionConfig,
        fingerprint: str,
        *,
        pdf_dir: Path,
        pages_dir: Path,
        timer: Callable[[], float],
    ) -> None:
        self._reader = reader
        self._store = store
        self._settings = settings
        self._fingerprint = fingerprint
        self._pdf_dir = pdf_dir
        self._pages_dir = pages_dir
        self._timer = timer
        self._ocr_seconds = 0.0

    def is_current(self, item: UUID, pdf: PdfEntry, previous: TextEntry) -> bool:
        """Tell whether the saved page records of ``item`` still stand: made from the PDF
        whose SHA-256 the PDF manifest records, with these settings, and unchanged on disk.

        The PDF on disk is not hashed here, since the snapshot is frozen and the data card's
        combined PDF digest verifies its files; a PDF that is extracted is checked first.
        """
        if (
            previous.status != "ok"
            or previous.pdf_sha256 != pdf.sha256
            or previous.config_fingerprint != self._fingerprint
        ):
            return False
        try:
            return _sha256_of(self._pages_dir / pages_file_name(item)) == previous.output_sha256
        except FileNotFoundError:
            return False

    def extract(self, item: UUID, pdf: PdfEntry, now: datetime) -> TextEntry:
        started = self._timer()
        self._ocr_seconds = 0.0
        output = self._pages_dir / pages_file_name(item)
        found: dict[str, Any] = {
            "program_key": pdf.program_key,
            "pdf_sha256": pdf.sha256,
            "config_fingerprint": self._fingerprint,
            "extracted_at": now,
        }
        try:
            pages, unreadable, warnings = self._read(self._pdf_dir / pdf_file_name(item), pdf)
        except _ExtractionFailure as failure:
            output.unlink(missing_ok=True)  # Page records of an older PDF must not survive.
            return TextEntry(
                **found,
                status="error",
                reason=failure.reason,
                detail=failure.detail,
                seconds=self._seconds_since(started),
                ocr_seconds=round(self._ocr_seconds, 3),
            )
        self._store.write(pages, output, model_cls=PageText)
        sources = Counter(page.source for page in pages)
        return TextEntry(
            **found,
            status="ok",
            file=output.name,
            output_sha256=_sha256_of(output),
            pages=len(pages),
            text_layer_pages=sources["text_layer"],
            ocr_pages=sources["ocr"],
            empty_pages=sources["empty"],
            low_text_pages=sum(page.low_text for page in pages),
            ocr_attempted=sum(page.ocr_attempted for page in pages),
            ocr_failed=sum(page.ocr_failed for page in pages),
            unreadable_pages=unreadable,
            lines_removed=sum(page.lines_removed for page in pages),
            pdf_warnings=warnings,
            seconds=self._seconds_since(started),
            ocr_seconds=round(self._ocr_seconds, 3),
        )

    def _read(self, path: Path, pdf: PdfEntry) -> tuple[tuple[PageText, ...], int, int]:
        """Return the page records of the PDF at ``path``, its unreadable pages, and the
        warnings that the PDF library reported."""
        try:
            digest = _sha256_of(path)
        except FileNotFoundError as error:
            raise _ExtractionFailure("pdf_missing", f"no PDF at {path}") from error
        if digest != pdf.sha256:
            raise _ExtractionFailure(
                "pdf_changed",
                "the PDF on disk no longer matches the SHA-256 that the PDF manifest records; "
                "run the PDF download again",
            )
        readings: list[PageReading] = []
        unreadable = 0
        try:
            with self._reader.open(path) as document:
                for index in range(document.page_count):
                    reading, readable = self._read_page(document, index)
                    readings.append(reading)
                    unreadable += not readable
                warnings = document.warning_count()
        except PdfReadError as error:
            raise _ExtractionFailure("unreadable_pdf", str(error)) from error
        return assemble_pages(readings, self._settings), unreadable, warnings

    def _read_page(self, document: PdfPages, index: int) -> tuple[PageReading, bool]:
        """Read one page, OCR it when it needs OCR, and tell whether its text layer was
        readable; an unreadable text layer counts as an empty one."""
        try:
            layer, readable = document.text_layer(index), True
        except PdfReadError:
            layer, readable = TextLayer(text="", chars=0), False
        if not needs_ocr(index, layer.chars, self._settings):
            return PageReading(
                layer_text=layer.text,
                layer_chars=layer.chars,
                ocr_attempted=False,
                ocr_text=None,
                ocr_failed=False,
            ), readable
        started = self._timer()
        try:
            text: str | None = document.ocr_text(index)
        except OcrError:
            text = None
        finally:
            self._ocr_seconds += self._timer() - started
        return PageReading(
            layer_text=layer.text,
            layer_chars=layer.chars,
            ocr_attempted=True,
            ocr_text=text,
            ocr_failed=text is None,
        ), readable

    def _seconds_since(self, started: float) -> float:
        return round(max(self._timer() - started, 0.0), 3)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_partial_files(pages_dir: Path) -> None:
    """Delete the partial files that a killed run left behind, hidden ones included."""
    for partial in pages_dir.glob(f"*{PART_SUFFIX}"):
        partial.unlink(missing_ok=True)


@contextmanager
def _exclusive_run(pages_dir: Path) -> Iterator[None]:
    """Hold the lock of ``pages_dir`` while the block runs.

    The lock is the operating system's lock on :data:`LOCK_FILE_NAME`, which it releases
    when the process ends, so a killed run leaves no stale lock behind.

    Raises:
        ExtractionInProgressError: if another run holds the lock.
    """
    with (pages_dir / LOCK_FILE_NAME).open("a+b") as lock_file:
        try:
            _lock(lock_file)
        except OSError as error:
            raise ExtractionInProgressError(
                f"another text extraction into {pages_dir} is running; wait until it ends"
            ) from error
        try:
            yield
        finally:
            _unlock(lock_file)


if sys.platform == "win32":

    def _lock(file: BinaryIO) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(file: BinaryIO) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)

else:

    def _lock(file: BinaryIO) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(file: BinaryIO) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def write_summary(path: Path, manifest: TextManifest) -> None:
    """Write the numbers of ``manifest`` to ``path`` as JSON: settings and counts, never an
    item or any text. The directory is created when it is missing."""
    summary = manifest.summary
    content = manifest.model_dump(mode="json", exclude={"items"}) | {
        "ocr_seconds_per_page": round(summary.ocr_seconds / summary.ocr_attempted, 3)
        if summary.ocr_attempted
        else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(path, json.dumps(content, indent=2))


class _Console:
    """Prints the plan, then one line of counts per PDF as soon as its outcome is saved."""

    def __init__(self) -> None:
        self._total = 0
        self._program_width = 0
        self._extracted = 0
        self._seconds = 0.0

    def start(self, plan: ExtractionPlan) -> None:
        self._total = len(plan.selected)
        self._program_width = max((len(item.program_key) for item in plan.selected), default=0)
        print(
            f"Snapshot {plan.snapshot_id}: {self._total} of {plan.pdfs} thesis PDFs selected; "
            f"page texts in {plan.pages_dir}",
            flush=True,
        )
        if plan.orphaned:
            print(
                f"Orphaned: {len(plan.orphaned)} entries whose PDF the PDF manifest no longer "
                "lists as on disk, kept with their page files and left out of the totals: "
                f"{', '.join(str(item) for item in plan.orphaned)}",
                flush=True,
            )

    def item(self, position: int, outcome: ItemOutcome) -> None:
        width = len(str(self._total))
        prefix = (
            f"[{position:>{width}}/{self._total}] {outcome.program_key:<{self._program_width}}  "
            f"{outcome.item_uuid}  "
        )
        if outcome.skipped:
            print(f"{prefix}unchanged, skipped", flush=True)
            return
        self._extracted += 1
        self._seconds += outcome.entry.seconds
        remaining = self._seconds / self._extracted * (self._total - position)
        eta = timedelta(seconds=round(remaining))
        print(f"{prefix}{_describe(outcome.entry)}, ETA {eta}", flush=True)


def _describe(entry: TextEntry) -> str:
    if entry.status == "error":
        return f"error ({entry.reason}), {entry.seconds:.1f} s"
    return (
        f"ok  {entry.pages} pages ({entry.text_layer_pages} text layer, {entry.ocr_pages} OCR, "
        f"{entry.empty_pages} empty), OCR {entry.ocr_attempted} tried / {entry.ocr_failed} "
        f"failed, {entry.lines_removed} lines removed, {entry.seconds:.1f} s"
    )


def _summary(report: ExtractionReport) -> str:
    """Describe this run's outcomes, then the manifest that also holds earlier runs."""
    extracted = [outcome.entry for outcome in report.outcomes if not outcome.skipped]
    failed = sum(entry.status == "error" for entry in extracted)
    sources = Counter[str]()
    for entry in extracted:
        sources.update(
            text_layer=entry.text_layer_pages, ocr=entry.ocr_pages, empty=entry.empty_pages
        )
    attempted = sum(entry.ocr_attempted for entry in extracted)
    ocr_seconds = sum(entry.ocr_seconds for entry in extracted)
    per_page = f"{ocr_seconds / attempted:.2f} s per page" if attempted else "no OCR"
    totals = report.manifest.summary
    return "\n".join(
        [
            f"This run: {len(report.outcomes)} PDFs, {len(extracted)} extracted, "
            f"{len(report.outcomes) - len(extracted)} skipped, {failed} failed",
            f"Pages extracted: {sum(sources.values()):,} ({sources['text_layer']:,} text layer, "
            f"{sources['ocr']:,} OCR, {sources['empty']:,} empty), "
            f"{sum(entry.low_text_pages for entry in extracted):,} low-text",
            f"OCR: {attempted:,} pages tried, "
            f"{sum(entry.ocr_failed for entry in extracted):,} failed, {ocr_seconds:.1f} s, "
            f"{per_page}",
            f"Header/footer lines removed: {sum(entry.lines_removed for entry in extracted):,}",
            f"Manifest {report.manifest_path}: {totals.items} PDFs, {totals.ok} ok, "
            f"{totals.error} errors, {totals.stale} with older settings, "
            f"{totals.orphaned} orphaned",
        ]
    )


def _positive_count(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got '{text}'") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _item_uuid(text: str) -> UUID:
    try:
        return UUID(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an item uuid, got '{text}'") from None


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.extraction.extract_text",
        description=(
            "Extract the text of every thesis PDF of a snapshot, page by page, into "
            "data/interim/<snapshot>/pages/. Low-text pages near the start of each PDF are "
            "OCR'd. PDFs already extracted with the same settings are skipped, so running the "
            "command again resumes an interrupted run."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--snapshot-id",
        help="snapshot whose PDFs to extract (default: the snapshot harvested last)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_count,
        metavar="N",
        help="look at the first N PDFs only, in snapshot order (default: all)",
    )
    parser.add_argument(
        "--only",
        type=_item_uuid,
        nargs="+",
        metavar="UUID",
        help="look at the PDFs of these items only (default: all)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        metavar="PATH",
        help="also write the manifest's numbers, without items or text, as JSON to PATH",
    )
    return parser.parse_args(argv)


def _one_line(error: BaseException) -> str:
    return " ".join(str(error).split())


def main(
    argv: Sequence[str] | None = None,
    *,
    reader: PdfReader | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the command line and return its exit code; ``reader`` replaces PyMuPDF in tests.

    The exit code is 0 when every PDF looked at was extracted or skipped, 1 after a fatal
    error or when some PDF failed, and 130 after an interruption.
    """
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    started = timer()
    console = _Console()

    def print_time() -> None:
        print(f"Wall time: {timer() - started:.1f} s")

    try:
        config = load_config(project_root / CONFIG_PATH)
        if reader is None:
            reader = PyMuPdfReader.from_config(config, project_root)
        report = extract_texts(
            config,
            project_root,
            reader,
            snapshot_id=arguments.snapshot_id,
            limit=arguments.limit,
            only=arguments.only,
            clock=clock,
            timer=timer,
            on_start=console.start,
            on_item=console.item,
        )
        if arguments.summary_out is not None:
            write_summary(arguments.summary_out, report.manifest)
    except KeyboardInterrupt:
        print(
            "interrupted: the manifest keeps every PDF finished so far; run the command "
            "again to resume",
            file=sys.stderr,
        )
        print_time()
        return 130
    except (ExtractionInProgressError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {_one_line(error)}", file=sys.stderr)
        return 1
    print(_summary(report))
    print_time()
    failed = sum(
        outcome.entry.status == "error" for outcome in report.outcomes if not outcome.skipped
    )
    if failed:
        print(f"{failed} PDFs failed; the manifest records why, and the next run tries them again")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
