"""Download the PDF of every thesis in a snapshot, politely, verifiably and resumably.

:func:`download_theses` reads a snapshot written by the harvest, the latest one by default,
and takes its theses: the items whose ``renati.type`` fragment equals
``snapshot.thesis_type``. For each thesis, in snapshot order, it:

1. records an item whose ``dc.rights`` closes its files (embargoed, restricted or metadata
   only access) as ``restricted``, with reason ``rights_restricted``, without any request;
2. keeps a PDF already on disk whose size and SHA-256 match its manifest entry, as
   ``already_present``, without any request, and never asks again for what the repository
   refused before with HTTP 401, 403 or 404;
3. lists the item's bundles in one request and looks for the thesis among them with
   :func:`~thematic_redundancy.corpus.thesis_files.thesis_candidates`. No candidate, or
   several, is recorded as ``no_thesis_file`` or ``ambiguous``, and nothing is fetched;
4. streams the thesis into ``<item uuid>.pdf.part``, checks that it starts with ``%PDF``
   and that its size and checksum match what the repository lists, and only then renames
   it to ``<item uuid>.pdf`` and records its MD5 and SHA-256. A PDF that fails a check is
   deleted and recorded as ``integrity_error``; it is not fetched again within the run.

The manifest ``pdfs/manifest.json`` is replaced atomically after every thesis, so an
interrupted run loses nothing, and running the command again resumes it. A PDF found on
disk without a manifest entry is kept when it matches the repository's listing. One run at
a time may use a PDF directory, and a run stops after :data:`MAX_CONSECUTIVE_FAILURES`
failed theses in a row, which point at the repository or the network rather than at the
theses.

Run it after the harvest::

    uv run python -m thematic_redundancy.corpus.download_pdfs [--project-root PATH]
        [--snapshot-id ID] [--limit N] [--per-program N]
"""

import argparse
import hashlib
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID

import httpx
import yaml

from thematic_redundancy.corpus.dspace import DSpaceRestRepository
from thematic_redundancy.corpus.pdf_manifest import (
    FILE_STATUSES,
    PDF_DIR_NAME,
    PDF_MANIFEST_FILE_NAME,
    PdfEntry,
    PdfManifest,
    PdfStatus,
    pdf_file_name,
    read_pdf_manifest,
    summarize,
    write_pdf_manifest,
)
from thematic_redundancy.corpus.repository import (
    Bitstream,
    BitstreamRepository,
    Bundle,
    Checksum,
    RepositoryError,
    ResourceUnavailableError,
)
from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    SnapshotRecord,
    check_snapshot_id,
    latest_snapshot_id,
    read_snapshot,
)
from thematic_redundancy.corpus.thesis_files import (
    ORIGINAL_BUNDLE,
    RESTRICTED_ACCESS_RIGHTS,
    SELECTION_RULE,
    access_restriction,
    is_thesis,
    thesis_candidates,
)
from thematic_redundancy.shared.config import AppConfig, load_config

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

MAX_CONSECUTIVE_FAILURES = 3
"""Failed theses in a row (``error`` or ``integrity_error``) that stop a run."""

FAILURE_STATUSES: frozenset[PdfStatus] = frozenset({"error", "integrity_error"})
"""Statuses that a later run tries again; every other status stands."""

PDF_MAGIC = b"%PDF"
"""First bytes of every PDF file."""

PART_SUFFIX = ".part"
"""Ending of a file still being written; a run deletes any that an earlier run left."""

LOCK_FILE_NAME = ".lock"
"""File in the PDF directory that a run locks, so that only one run uses the directory."""

_HASH_NAMES = {"md5": "md5", "sha1": "sha1", "sha256": "sha256", "sha512": "sha512"}
"""Checksum algorithms that can be verified, by their name without case, ``-`` or ``_``."""

_READ_CHUNK_BYTES = 1024 * 1024


class DownloadStoppedError(RuntimeError):
    """A run stopped after too many failed theses in a row; the manifest keeps its outcomes."""


class DownloadInProgressError(RuntimeError):
    """Another run is using the same PDF directory."""


@dataclass(frozen=True)
class SelectedThesis:
    """A thesis that a run will look at."""

    uuid: UUID
    program_key: str


@dataclass(frozen=True)
class DownloadPlan:
    """What a run is about to do, announced once it holds the PDF directory."""

    snapshot_id: str
    pdf_dir: Path
    theses: int
    """Theses in the snapshot, selected or not."""
    selected: tuple[SelectedThesis, ...]
    """Theses that the run will look at, in order."""


@dataclass(frozen=True)
class ItemOutcome:
    """The manifest entry that a run established for one thesis."""

    item_uuid: UUID
    program_key: str
    entry: PdfEntry


@dataclass(frozen=True)
class DownloadReport:
    """Outcome of :func:`download_theses`."""

    snapshot_id: str
    pdf_dir: Path
    theses: int
    """Theses in the snapshot, selected or not."""
    outcomes: tuple[ItemOutcome, ...]
    """Outcome of each thesis that this run looked at, in order."""
    manifest: PdfManifest
    """The manifest as saved, which also covers theses of earlier runs."""

    @property
    def manifest_path(self) -> Path:
        return self.pdf_dir / PDF_MANIFEST_FILE_NAME


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def download_theses(
    config: AppConfig,
    project_root: Path,
    repository: BitstreamRepository,
    *,
    snapshot_id: str | None = None,
    limit: int | None = None,
    per_program: int | None = None,
    clock: Callable[[], datetime] = _utc_now,
    on_start: Callable[[DownloadPlan], None] | None = None,
    on_item: Callable[[int, ItemOutcome], None] | None = None,
) -> DownloadReport:
    """Fetch the thesis PDFs of a snapshot into its ``pdfs/`` directory.

    The snapshot is ``data/raw/<snapshot_id>`` under ``paths.data_dir``, and defaults to
    the one harvested last. ``per_program`` keeps the first theses of each program, and
    ``limit`` the first theses overall, both in snapshot order. ``on_start`` hears the plan
    once the run holds the PDF directory, and ``on_item`` each outcome with its position,
    counted from 1, once the manifest holds it.

    Raises:
        DownloadInProgressError: if another run is using the PDF directory.
        DownloadStoppedError: after :data:`MAX_CONSECUTIVE_FAILURES` failed theses in a row.
        FileNotFoundError: if the snapshot does not exist.
        ValueError: if a limit is below 1, if the snapshot id is not one plain name, if the
            snapshot fails its digest check, or if the PDF manifest is damaged.
        OSError: if reading or writing the disk fails.
    """
    for name, value in (("limit", limit), ("per_program", per_program)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    raw_dir = config.paths.resolve_against(project_root).data_dir / RAW_DIR_NAME
    if snapshot_id is None:
        snapshot_id = latest_snapshot_id(raw_dir)
    snapshot_dir = raw_dir / check_snapshot_id(snapshot_id)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"snapshot '{snapshot_id}' not found under {raw_dir}")
    _, records = read_snapshot(snapshot_dir)
    theses = [
        record for record in records if is_thesis(record.metadata, config.snapshot.thesis_type)
    ]
    selected = _select(theses, limit=limit, per_program=per_program)
    pdf_dir = snapshot_dir / PDF_DIR_NAME
    pdf_dir.mkdir(exist_ok=True)
    with _exclusive_run(pdf_dir):
        _remove_partial_files(pdf_dir)
        store = _ManifestStore(pdf_dir / PDF_MANIFEST_FILE_NAME, config, snapshot_id, records)
        fetcher = _ThesisFetcher(repository, pdf_dir)
        if on_start is not None:
            plan = DownloadPlan(
                snapshot_id=snapshot_id,
                pdf_dir=pdf_dir,
                theses=len(theses),
                selected=tuple(SelectedThesis(item.uuid, item.program_key) for item in selected),
            )
            on_start(plan)
        outcomes: list[ItemOutcome] = []
        failures_in_a_row = 0
        for position, record in enumerate(selected, start=1):
            now = _as_utc(clock())
            entry = fetcher.fetch(record, store.entry(record.uuid), now)
            store.save(record.uuid, entry, now)
            outcome = ItemOutcome(
                item_uuid=record.uuid, program_key=record.program_key, entry=entry
            )
            outcomes.append(outcome)
            if on_item is not None:
                on_item(position, outcome)
            failures_in_a_row = failures_in_a_row + 1 if entry.status in FAILURE_STATUSES else 0
            if failures_in_a_row == MAX_CONSECUTIVE_FAILURES:
                raise DownloadStoppedError(
                    f"{MAX_CONSECUTIVE_FAILURES} theses failed in a row, which points at the "
                    "repository or the network; the run stopped, and the manifest keeps every "
                    "outcome so far, so run the command again later to resume"
                )
        manifest = store.manifest(_as_utc(clock()))
    return DownloadReport(
        snapshot_id=snapshot_id,
        pdf_dir=pdf_dir,
        theses=len(theses),
        outcomes=tuple(outcomes),
        manifest=manifest,
    )


def _as_utc(moment: datetime) -> datetime:
    if moment.utcoffset() is None:
        raise ValueError(
            f"the download clock must give a time with a time zone, got '{moment.isoformat()}'"
        )
    return moment.astimezone(UTC)


def _select(
    theses: Sequence[SnapshotRecord], *, limit: int | None, per_program: int | None
) -> list[SnapshotRecord]:
    """Keep the first ``per_program`` theses of each program, then the first ``limit``."""
    taken: Counter[str] = Counter()
    selected: list[SnapshotRecord] = []
    for record in theses:
        if limit is not None and len(selected) == limit:
            break
        if per_program is not None and taken[record.program_key] == per_program:
            continue
        taken[record.program_key] += 1
        selected.append(record)
    return selected


class _ManifestStore:
    """The entries of the PDF manifest, saved to disk after every change."""

    def __init__(
        self,
        path: Path,
        config: AppConfig,
        snapshot_id: str,
        records: Sequence[SnapshotRecord],
    ) -> None:
        saved = read_pdf_manifest(path)
        if saved is not None and saved.snapshot_id != snapshot_id:
            raise ValueError(
                f"the PDF manifest {path} belongs to snapshot '{saved.snapshot_id}', not to "
                f"'{snapshot_id}'"
            )
        self._path = path
        self._saved = saved
        self._entries: dict[UUID, PdfEntry] = dict(saved.items) if saved is not None else {}
        self._order = {record.uuid: index for index, record in enumerate(records)}
        self._fields: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "base_url": str(config.repository.base_url).rstrip("/"),
            "thesis_type": config.snapshot.thesis_type,
            "selection_rule": SELECTION_RULE,
        }

    def entry(self, item: UUID) -> PdfEntry | None:
        return self._entries.get(item)

    def save(self, item: UUID, entry: PdfEntry, now: datetime) -> None:
        self._entries[item] = entry
        self._saved = self._build(now)
        write_pdf_manifest(self._path, self._saved)

    def manifest(self, now: datetime) -> PdfManifest:
        """Return the manifest as saved; one is written if none exists yet."""
        if self._saved is None:
            self._saved = self._build(now)
            write_pdf_manifest(self._path, self._saved)
        return self._saved

    def _build(self, now: datetime) -> PdfManifest:
        last = len(self._order)
        entries = dict(
            sorted(self._entries.items(), key=lambda item: self._order.get(item[0], last))
        )
        return PdfManifest(
            **self._fields, updated_at=now, summary=summarize(entries), items=entries
        )


class _IntegrityFailure(Exception):
    """A download is not the file that the repository lists."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class _ThesisFetcher:
    """Establishes the manifest entry of one thesis, asking the repository only when needed."""

    def __init__(self, repository: BitstreamRepository, pdf_dir: Path) -> None:
        self._repository = repository
        self._pdf_dir = pdf_dir

    def fetch(self, record: SnapshotRecord, previous: PdfEntry | None, now: datetime) -> PdfEntry:
        found: dict[str, Any] = {"program_key": record.program_key, "checked_at": now}
        restriction = access_restriction(record.metadata)
        if restriction is not None:
            return PdfEntry(
                **found,
                status="restricted",
                reason="rights_restricted",
                detail=f"dc.rights is {RESTRICTED_ACCESS_RIGHTS[restriction]} ({restriction})",
            )
        target = self._pdf_dir / pdf_file_name(record.uuid)
        if previous is not None:
            if _was_refused(previous):
                return previous
            if previous.status in FILE_STATUSES and _matches_entry(target, previous):
                return previous.model_copy(update={"status": "already_present", "checked_at": now})
        try:
            bundles = self._repository.list_bundles(record.uuid)
        except RepositoryError as error:
            return _failure(found, error)
        candidates = thesis_candidates(bundles)
        if len(candidates) != 1:
            return _no_single_thesis(found, bundles, len(candidates))
        thesis = candidates[0]
        found |= {
            "bitstream_uuid": thesis.uuid,
            "size_bytes": thesis.size_bytes,
            "checksum_algorithm": thesis.checksum.algorithm,
            "checksum": thesis.checksum.value,
            "candidate_count": 1,
        }
        algorithm = _hash_name(thesis.checksum.algorithm)
        if algorithm is None:
            return PdfEntry(
                **found,
                status="integrity_error",
                reason="unsupported_checksum_algorithm",
                detail=f"the repository lists a {thesis.checksum.algorithm} checksum, "
                "which this downloader cannot verify",
            )
        on_disk = _matching_file(target, thesis, algorithm)
        if on_disk is not None:
            same_file = previous is not None and previous.bitstream_uuid == thesis.uuid
            return PdfEntry(
                **found,
                status="already_present",
                file=target.name,
                md5=on_disk["md5"],
                sha256=on_disk["sha256"],
                downloaded_at=previous.downloaded_at
                if previous is not None and same_file
                else None,
            )
        try:
            digests = self._download(thesis, algorithm, target)
        except RepositoryError as error:
            return _failure(found, error)
        except _IntegrityFailure as failure:
            return PdfEntry(
                **found, status="integrity_error", reason=failure.reason, detail=failure.detail
            )
        return PdfEntry(
            **found,
            status="downloaded",
            file=target.name,
            md5=digests["md5"],
            sha256=digests["sha256"],
            downloaded_at=now,
        )

    def _download(self, thesis: Bitstream, algorithm: str, target: Path) -> dict[str, str]:
        """Fetch ``thesis`` into ``target`` through a partial file, and return its digests.

        The partial file replaces ``target`` only once every check passed; on any failure,
        an interruption included, it is deleted and ``target`` stays as it was.
        """
        partial = target.with_name(f"{target.name}{PART_SUFFIX}")
        try:
            with partial.open("wb") as file:
                sink = _CheckingSink(file, thesis.size_bytes, {"md5", "sha256", algorithm})
                self._repository.download(thesis.uuid, sink)
                digests = sink.finish(algorithm, thesis.checksum)
                file.flush()
                os.fsync(file.fileno())
            os.replace(partial, target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return digests


def _was_refused(entry: PdfEntry) -> bool:
    """Tell whether the repository refused the thesis for good: HTTP 401, 403 or 404."""
    return entry.status == "not_found" or (
        entry.status == "restricted" and entry.reason in {"http_401", "http_403"}
    )


def _failure(found: Mapping[str, Any], error: RepositoryError) -> PdfEntry:
    if isinstance(error, ResourceUnavailableError):
        status: PdfStatus = "not_found" if error.status_code == 404 else "restricted"
        return PdfEntry(
            **found, status=status, reason=f"http_{error.status_code}", detail=str(error)
        )
    return PdfEntry(**found, status="error", reason="repository_error", detail=str(error))


def _no_single_thesis(found: Mapping[str, Any], bundles: Sequence[Bundle], count: int) -> PdfEntry:
    if count > 1:
        return PdfEntry(
            **found,
            status="ambiguous",
            reason="multiple_candidates",
            detail=f"the {ORIGINAL_BUNDLE} bundle holds {count} PDFs that may be the thesis",
            candidate_count=count,
        )
    if any(bundle.name == ORIGINAL_BUNDLE for bundle in bundles):
        reason = "no_candidate_pdf"
        detail = (
            f"the {ORIGINAL_BUNDLE} bundle holds no PDF but a similarity report or an "
            "authorization form"
        )
    else:
        reason, detail = "no_original_bundle", f"the item has no {ORIGINAL_BUNDLE} bundle"
    return PdfEntry(
        **found, status="no_thesis_file", reason=reason, detail=detail, candidate_count=0
    )


def _hash_name(algorithm: str) -> str | None:
    """Return the :mod:`hashlib` name of a repository checksum algorithm, if one fits."""
    return _HASH_NAMES.get(algorithm.strip().lower().replace("-", "").replace("_", ""))


class _Hashes:
    """Digests of the same bytes in several algorithms at once."""

    def __init__(self, names: Iterable[str]) -> None:
        self._hashes = {name: hashlib.new(name, usedforsecurity=False) for name in names}

    def update(self, data: bytes) -> None:
        for digest in self._hashes.values():
            digest.update(data)

    def hexdigests(self) -> dict[str, str]:
        return {name: digest.hexdigest() for name, digest in self._hashes.items()}


class _CheckingSink:
    """Byte sink that writes a download into a file, hashing and checking it on the way.

    It ends a download as soon as the body cannot be the listed PDF: when its first bytes
    are not ``%PDF``, or when it grows past the listed size.
    """

    def __init__(self, file: BinaryIO, size_bytes: int, algorithms: Iterable[str]) -> None:
        self._file = file
        self._listed_size = size_bytes
        self._algorithms = tuple(algorithms)
        self.restart()

    def restart(self) -> None:
        self._file.seek(0)
        self._file.truncate()
        self._hashes = _Hashes(self._algorithms)
        self._size = 0
        self._head = b""

    def write(self, data: bytes, /) -> int:
        self._size += len(data)
        if self._size > self._listed_size:
            raise _IntegrityFailure(
                "size_mismatch",
                f"the repository lists {self._listed_size:,} bytes, but sent more",
            )
        if len(self._head) < len(PDF_MAGIC):
            self._head += data[: len(PDF_MAGIC) - len(self._head)]
            if len(self._head) == len(PDF_MAGIC) and self._head != PDF_MAGIC:
                raise _not_a_pdf()
        self._file.write(data)
        self._hashes.update(data)
        return len(data)

    def finish(self, algorithm: str, checksum: Checksum) -> dict[str, str]:
        """Check the whole download against the listing and return its digests."""
        if self._head != PDF_MAGIC:
            raise _not_a_pdf()
        if self._size != self._listed_size:
            raise _IntegrityFailure(
                "size_mismatch",
                f"received {self._size:,} bytes, but the repository lists {self._listed_size:,}",
            )
        digests = self._hashes.hexdigests()
        if digests[algorithm] != checksum.value.strip().lower():
            raise _IntegrityFailure(
                "checksum_mismatch",
                f"the {checksum.algorithm} digest differs from the one the repository lists",
            )
        return digests


def _not_a_pdf() -> _IntegrityFailure:
    return _IntegrityFailure("not_a_pdf", f"the file does not start with {PDF_MAGIC.decode()}")


def _digest_file(path: Path, algorithms: Iterable[str]) -> tuple[bytes, dict[str, str]]:
    """Return the first bytes of the file at ``path`` and its digests."""
    hashes = _Hashes(algorithms)
    with path.open("rb") as file:
        head = file.read(len(PDF_MAGIC))
        hashes.update(head)
        while chunk := file.read(_READ_CHUNK_BYTES):
            hashes.update(chunk)
    return head, hashes.hexdigests()


def _matches_entry(path: Path, entry: PdfEntry) -> bool:
    """Tell whether ``path`` still holds the PDF that ``entry`` describes."""
    try:
        if path.stat().st_size != entry.size_bytes:
            return False
        head, digests = _digest_file(path, {"sha256"})
    except FileNotFoundError:
        return False
    return head == PDF_MAGIC and digests["sha256"] == entry.sha256


def _matching_file(path: Path, thesis: Bitstream, algorithm: str) -> dict[str, str] | None:
    """Return the digests of the file at ``path`` when it is ``thesis`` as listed."""
    try:
        if path.stat().st_size != thesis.size_bytes:
            return None
        head, digests = _digest_file(path, {"md5", "sha256", algorithm})
    except FileNotFoundError:
        return None
    if head != PDF_MAGIC or digests[algorithm] != thesis.checksum.value.strip().lower():
        return None
    return digests


def _remove_partial_files(pdf_dir: Path) -> None:
    """Delete the partial files that a killed run left behind."""
    for partial in pdf_dir.glob(f"*{PART_SUFFIX}"):
        partial.unlink(missing_ok=True)


@contextmanager
def _exclusive_run(pdf_dir: Path) -> Iterator[None]:
    """Hold the lock of ``pdf_dir`` while the block runs.

    The lock is the operating system's lock on :data:`LOCK_FILE_NAME`, which it releases
    when the process ends, so a killed run leaves no stale lock behind.

    Raises:
        DownloadInProgressError: if another run holds the lock.
    """
    with (pdf_dir / LOCK_FILE_NAME).open("a+b") as lock_file:
        try:
            _lock(lock_file)
        except OSError as error:
            raise DownloadInProgressError(
                f"another download into {pdf_dir} is running; wait until it ends"
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


class _Console:
    """Prints the plan, then one line per thesis as soon as its outcome is saved."""

    def __init__(self) -> None:
        self._total = 0
        self._program_width = 0

    def start(self, plan: DownloadPlan) -> None:
        self._total = len(plan.selected)
        self._program_width = max((len(item.program_key) for item in plan.selected), default=0)
        print(
            f"Snapshot {plan.snapshot_id}: {self._total} of {plan.theses} theses selected; "
            f"PDFs in {plan.pdf_dir}",
            flush=True,
        )

    def item(self, position: int, outcome: ItemOutcome) -> None:
        width = len(str(self._total))
        print(
            f"[{position:>{width}}/{self._total}] {outcome.program_key:<{self._program_width}}  "
            f"{outcome.item_uuid}  {_describe(outcome.entry)}",
            flush=True,
        )


def _describe(entry: PdfEntry) -> str:
    text = entry.status if entry.reason is None else f"{entry.status} ({entry.reason})"
    if entry.status in FILE_STATUSES and entry.size_bytes is not None:
        return f"{text}  {entry.size_bytes:,} bytes"
    return text


def _summary(report: DownloadReport) -> str:
    """Describe this run's outcomes, then the manifest that also holds earlier runs."""
    by_status = Counter(outcome.entry.status for outcome in report.outcomes)
    by_program: dict[str, Counter[str]] = {}
    for outcome in report.outcomes:
        by_program.setdefault(outcome.program_key, Counter())[outcome.entry.status] += 1
    fetched = sum(
        outcome.entry.size_bytes or 0
        for outcome in report.outcomes
        if outcome.entry.status == "downloaded"
    )
    status_width = max((len(status) for status in by_status), default=0)
    program_width = max((len(program) for program in by_program), default=0)
    totals = report.manifest.summary
    return "\n".join(
        [
            f"Outcomes of this run ({len(report.outcomes)} theses):",
            *(f"  {status:<{status_width}}  {count}" for status, count in by_status.items()),
            "Outcomes by program:",
            *(
                f"  {program:<{program_width}}  "
                + ", ".join(f"{status} {count}" for status, count in counts.items())
                for program, counts in by_program.items()
            ),
            f"Downloaded in this run: {fetched:,} bytes",
            f"Manifest {report.manifest_path}: {totals.items} theses, {totals.files} files, "
            f"{totals.total_bytes:,} bytes",
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


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.corpus.download_pdfs",
        description=(
            "Download the thesis PDF of every thesis in a snapshot from the DSpace REST API "
            "into data/raw/<snapshot>/pdfs/, one request at a time. Files already verified "
            "are kept, so running the command again resumes an interrupted run."
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
        help="snapshot whose theses to fetch (default: the snapshot harvested last)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_count,
        metavar="N",
        help="look at the first N theses only, in snapshot order (default: all)",
    )
    parser.add_argument(
        "--per-program",
        type=_positive_count,
        metavar="N",
        help="look at the first N theses of each program only (default: all)",
    )
    return parser.parse_args(argv)


def _one_line(error: BaseException) -> str:
    return " ".join(str(error).split())


def main(
    argv: Sequence[str] | None = None,
    *,
    repository: BitstreamRepository | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> int:
    """Run the command line and return its exit code; ``repository`` replaces HTTP in tests.

    The exit code is 0 when every thesis looked at has a standing outcome, 1 after a fatal
    error or when some thesis failed and needs another run, and 130 after an interruption.
    """
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    started = time.perf_counter()
    adapter = repository if isinstance(repository, DSpaceRestRepository) else None
    console = _Console()

    def print_cost() -> None:
        if adapter is not None:
            print(f"HTTP requests: {adapter.request_count}")
        print(f"Wall time: {time.perf_counter() - started:.1f} s")

    try:
        config = load_config(project_root / CONFIG_PATH)
        with ExitStack() as stack:
            if repository is None:
                client = stack.enter_context(httpx.Client())
                repository = adapter = DSpaceRestRepository.from_config(client, config.repository)
            report = download_theses(
                config,
                project_root,
                repository,
                snapshot_id=arguments.snapshot_id,
                limit=arguments.limit,
                per_program=arguments.per_program,
                clock=clock,
                on_start=console.start,
                on_item=console.item,
            )
    except KeyboardInterrupt:
        print(
            "interrupted: the manifest keeps every thesis finished so far; run the command "
            "again to resume",
            file=sys.stderr,
        )
        print_cost()
        return 130
    except DownloadStoppedError as error:
        print(f"error: {_one_line(error)}", file=sys.stderr)
        print_cost()
        return 1
    except (
        DownloadInProgressError,
        RepositoryError,
        OSError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"error: {_one_line(error)}", file=sys.stderr)
        return 1
    print(_summary(report))
    print_cost()
    failed = sum(outcome.entry.status in FAILURE_STATUSES for outcome in report.outcomes)
    if failed:
        print(f"{failed} theses failed; run the command again to retry them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
