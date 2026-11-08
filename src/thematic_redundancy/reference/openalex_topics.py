"""Download the OpenAlex Topics list once (T25, D17), the reference of "never touched" topics::

    uv run python -m thematic_redundancy.reference.openalex_topics [--project-root PATH]
        [--download-id ID]

This is a remote operation: it reads the API of the ``openalex`` block of the configuration,
``https://api.openalex.org``, and needs the user's authorization (U8). It pages through
``/topics`` with :class:`~thematic_redundancy.reference.openalex.OpenAlexTopicsClient`, which
sends no email, ``mailto`` parameter, API key or cookie, and keeps every topic record as
OpenAlex serves it. The download mirrors the split of a corpus snapshot:

- ``<paths.data_dir>/raw/openalex/topics/<download_id>/topics.jsonl``, gitignored: one topic
  record per line, as compact JSON in UTF-8 with LF endings, in the order of the listing;
- ``<paths.results_dir>/reference/openalex/topics/<download_id>/manifest.json``, tracked by git:
  the :class:`TopicsManifest`, with the source and the query, OpenAlex's ``meta.count``, the
  records written and their distinct ids, the size and SHA-256 of ``topics.jsonl``, the commit
  and the changed paths of the code, and the library versions.

How a download behaves:

- **Id.** It defaults to the UTC retrieval time, such as ``20261105T140309Z``, and follows the
  rules of a snapshot id. An id whose data folder or manifest exists is refused before any
  request: a download is never overwritten.
- **One at a time.** A run holds the lock of ``<paths.data_dir>/raw/openalex/topics/``.
- **Complete or nothing.** The records written must equal the ``meta.count`` of the first page,
  and every topic ``id`` must be unique; otherwise nothing is written. ``topics.jsonl`` is
  written first and ``manifest.json`` last, each atomically; a failure or an interruption
  before the manifest is in place takes ``topics.jsonl`` back.
- **Exit codes.** 0 on success, 1 after an error, and 130 after an interruption.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    Count,
    SnapshotId,
    UtcDatetime,
    check_snapshot_id,
    default_snapshot_id,
)
from thematic_redundancy.reference.openalex import (
    USER_AGENT,
    OpenAlexError,
    OpenAlexTopicsClient,
    TopicsListing,
)
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.filesystem import (
    exclusive_run,
    write_bytes_atomically,
    write_text_atomically,
)
from thematic_redundancy.topic_map.run_experiment import GitState, read_git_state

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

OPENALEX_DIR_NAME = "openalex"
TOPICS_DIR_NAME = "topics"
REFERENCE_DIR_NAME = "reference"
"""Directory under ``paths.results_dir`` that holds the records of the external references."""
TOPICS_FILE_NAME = "topics.jsonl"
MANIFEST_FILE_NAME = "manifest.json"

LIBRARIES = ("httpx", "pydantic")
"""The libraries that the download depends on: the requests, and the checks of each page."""

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DownloadError(RuntimeError):
    """The download cannot be written: the listing is incomplete, or another run holds the lock."""


class DownloadExistsError(FileExistsError):
    """A download with this id already exists; downloads are never overwritten."""


class _ManifestModel(BaseModel):
    """Base of the manifest: immutable, closed to unknown keys, input kept out of errors."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class DataFile(_ManifestModel):
    """One gitignored data file of a download."""

    path: str
    """Location relative to the project root, with ``/``."""
    size_bytes: Count
    sha256: Sha256


class TopicsManifest(_ManifestModel):
    """Provenance and summary of one download, saved as ``manifest.json``."""

    download_id: SnapshotId
    retrieved_at: UtcDatetime
    """UTC time just before the first request."""
    base_url: str
    url: str
    query_parameters: dict[str, str]
    """Query of the first request; each later one asked for the next ``page``."""
    user_agent: str
    """The only identification sent: no email, ``mailto`` parameter, API key or cookie."""
    settings: dict[str, JsonValue]
    """The values of the ``openalex`` block that the download used."""
    meta_count: Count
    """How many topics OpenAlex reported, ``meta.count`` of the first page."""
    records_written: Count
    distinct_ids: Count
    pages: Count
    """Pages read: the ones that ``meta.count`` needs at ``per-page``, and at least one."""
    requests: Count
    """HTTP requests sent, retries included."""
    data_files: dict[str, DataFile]
    config_sha256: Sha256
    """SHA-256 of ``config/default.yaml`` as the download read it."""
    git_commit: str
    git_dirty: tuple[str, ...]
    """Changed or untracked paths of the code when the download ran."""
    versions: dict[str, str]
    seconds: float
    """Time from the first request to the last, the paced waits and retries included."""


@dataclass(frozen=True)
class DownloadPlaces:
    """Where one download goes; every path is absolute."""

    lock_dir: Path
    """``<data_dir>/raw/openalex/topics``: the folders of every download, and the run lock."""
    data_dir: Path
    data_file: Path
    results_dir: Path
    manifest_file: Path
    results_relative: str
    """``results_dir`` relative to the project root, with ``/``, as git names it."""


@dataclass(frozen=True)
class DownloadPlan:
    """A download about to start: its id, its retrieval time and its places."""

    download_id: str
    retrieved_at: datetime
    places: DownloadPlaces


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def check_download_id(download_id: str) -> str:
    """Return ``download_id`` when it follows the rules of a snapshot id.

    Raises:
        ValueError: if it is not one plain directory name of 1 to 64 characters.
    """
    try:
        return check_snapshot_id(download_id)
    except ValueError:
        raise ValueError(
            f"download id {download_id!r} must have 1 to 64 letters, digits, '.', '-' or '_', "
            "and start and end with a letter or a digit, as a snapshot id"
        ) from None


def download_places(config: AppConfig, project_root: Path, download_id: str) -> DownloadPlaces:
    """Return the places of download ``download_id`` under the configured directories.

    Raises:
        ValueError: if the id is not one plain name, or a directory resolves outside the
            project root.
    """
    root = Path(project_root).resolve()
    paths = config.paths.resolve_against(root)
    lock_dir = paths.data_dir / RAW_DIR_NAME / OPENALEX_DIR_NAME / TOPICS_DIR_NAME
    data_dir = lock_dir / check_download_id(download_id)
    results_dir = paths.results_dir / REFERENCE_DIR_NAME / OPENALEX_DIR_NAME / TOPICS_DIR_NAME
    results_dir = results_dir / download_id
    return DownloadPlaces(
        lock_dir=lock_dir,
        data_dir=data_dir,
        data_file=data_dir / TOPICS_FILE_NAME,
        results_dir=results_dir,
        manifest_file=results_dir / MANIFEST_FILE_NAME,
        results_relative=results_dir.relative_to(root).as_posix(),
    )


def plan_download(
    config: AppConfig,
    project_root: Path,
    *,
    download_id: str | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> DownloadPlan:
    """Fix the retrieval time and the id of a new download, and refuse an existing one.

    Raises:
        DownloadExistsError: if the download's data folder or manifest exists.
        ValueError: if ``clock`` gives a time without a time zone, or the id or a directory
            is refused by :func:`download_places`.
    """
    moment = clock()
    if moment.utcoffset() is None:
        raise ValueError(
            f"the download clock must give a time with a time zone, got '{moment.isoformat()}'"
        )
    retrieved_at = moment.astimezone(UTC)
    chosen = default_snapshot_id(retrieved_at) if download_id is None else download_id
    places = download_places(config, project_root, chosen)
    require_new_download(places, chosen)
    return DownloadPlan(download_id=chosen, retrieved_at=retrieved_at, places=places)


def require_new_download(places: DownloadPlaces, download_id: str) -> None:
    """Raise :class:`DownloadExistsError` when the data folder or the manifest exists."""
    for path in (places.data_dir, places.manifest_file):
        if path.exists() or path.is_symlink():
            raise DownloadExistsError(
                f"OpenAlex Topics download '{download_id}' already exists at {path}; downloads "
                "are never overwritten, so choose another --download-id"
            )


def check_completeness(listing: TopicsListing) -> int:
    """Return the number of distinct topic ids, once the listing is complete.

    Raises:
        DownloadError: if a topic id appears twice, or the records do not number
            ``meta.count``.
    """
    counts = Counter(record["id"] for record in listing.records)
    repeated = [(identifier, times) for identifier, times in counts.items() if times > 1]
    if repeated:
        identifier, times = repeated[0]
        others = f", and {len(repeated) - 1} other ids repeat" if len(repeated) > 1 else ""
        raise DownloadError(
            f"the download holds {len(listing.records)} topics but {len(counts)} distinct ids: "
            f"{identifier} appears {times} times{others}; nothing was written"
        )
    if len(listing.records) != listing.meta_count:
        raise DownloadError(
            f"the download holds {len(listing.records)} topics, but OpenAlex reported "
            f"{listing.meta_count} (meta.count of the first page); nothing was written"
        )
    return len(counts)


def encode_topics(records: Iterable[dict[str, Any]]) -> bytes:
    """Serialize ``records`` as JSON lines: compact JSON in UTF-8, one per line, each ended
    by LF, keys and values in the order the API gave them.

    Raises:
        ValueError: if a record holds a number that JSON cannot write, such as NaN.
    """
    return b"".join(f"{_compact_json(record)}\n".encode() for record in records)


def _compact_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def run_download(
    plan: DownloadPlan,
    config: AppConfig,
    project_root: Path,
    source: OpenAlexTopicsClient,
    *,
    git: Callable[[Path, str], GitState] = read_git_state,
    monotonic: Callable[[], float] = time.monotonic,
) -> TopicsManifest:
    """Hold the run lock, read the Topics list through ``source``, check it and write it.

    Raises:
        DownloadExistsError: if the download appeared since ``plan`` was made.
        DownloadError: if another run holds the lock, or the listing is incomplete.
        OpenAlexError: if the API refuses, fails or answers unexpectedly.
        OSError: if git cannot tell the code state, or writing fails.
    """
    root = Path(project_root).resolve()
    places = plan.places
    places.lock_dir.mkdir(parents=True, exist_ok=True)
    busy = DownloadError(
        f"another download holds the lock of {places.lock_dir}; let it finish first"
    )
    with exclusive_run(places.lock_dir, busy):
        require_new_download(places, plan.download_id)
        state = git(root, places.results_relative)
        started = monotonic()
        listing = source.list_topics()
        seconds = monotonic() - started
        distinct = check_completeness(listing)
        content = encode_topics(listing.records)
        manifest = TopicsManifest(
            download_id=plan.download_id,
            retrieved_at=plan.retrieved_at,
            base_url=str(config.openalex.base_url).rstrip("/"),
            url=source.url,
            query_parameters=source.first_parameters(),
            user_agent=USER_AGENT,
            settings=config.openalex.model_dump(mode="json"),
            meta_count=listing.meta_count,
            records_written=len(listing.records),
            distinct_ids=distinct,
            pages=listing.pages,
            requests=source.request_count,
            data_files={
                TOPICS_FILE_NAME: DataFile(
                    path=places.data_file.relative_to(root).as_posix(),
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            },
            config_sha256=hashlib.sha256((root / CONFIG_PATH).read_bytes()).hexdigest(),
            git_commit=state.commit,
            git_dirty=state.dirty,
            versions=library_versions(),
            seconds=round(max(seconds, 0.0), 3),
        )
        write_download(places, content, manifest)
    return manifest


def library_versions() -> dict[str, str]:
    """Return the versions of Python and of the libraries the download depends on."""
    return {"python": platform.python_version(), **{name: version(name) for name in LIBRARIES}}


def write_download(places: DownloadPlaces, content: bytes, manifest: TopicsManifest) -> None:
    """Write ``topics.jsonl``, then ``manifest.json`` last, each atomically.

    The manifest is the commit point. A failure or an interruption before it is in place
    removes ``topics.jsonl`` and the folders this call made, where the system allows it.

    Raises:
        DownloadError: if writing failed and nothing was left behind.
        OSError: if writing failed and a file could not be removed; the message names it.
    """
    places.data_dir.mkdir()
    made_results_dir = not places.results_dir.exists()
    try:
        write_bytes_atomically(places.data_file, content)
        places.results_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomically(places.manifest_file, manifest.model_dump_json(indent=2))
    except BaseException as error:
        if places.manifest_file.exists():  # The commit point was reached; the download stands.
            raise
        left = _take_back(places, made_results_dir=made_results_dir)
        if not isinstance(error, Exception):
            raise
        if left:
            raise OSError(
                f"{error}; the download left {', '.join(str(path) for path in left)} without "
                f"its {MANIFEST_FILE_NAME}: move them away and run the command again"
            ) from error
        raise DownloadError(f"{error}; nothing was written") from error


def _take_back(places: DownloadPlaces, *, made_results_dir: bool) -> list[Path]:
    """Remove what an unfinished :func:`write_download` made; return the files left behind."""
    with suppress(OSError):
        places.data_file.unlink(missing_ok=True)
    with suppress(OSError):
        places.data_dir.rmdir()
    if made_results_dir:
        with suppress(OSError):
            places.results_dir.rmdir()
    return leftovers(places)


def leftovers(places: DownloadPlaces) -> list[Path]:
    """Return the files of an unfinished download: any file under its data folder, and under
    its results folder, while that folder holds no manifest."""
    if places.manifest_file.exists():
        return []
    found = []
    for folder in (places.data_dir, places.results_dir):
        if folder.is_dir():
            found.extend(sorted(path for path in folder.rglob("*") if path.is_file()))
    return found


def _summary(plan: DownloadPlan, manifest: TopicsManifest) -> str:
    """Describe the download: where it is, its counts, and the state of the code."""
    data = manifest.data_files[TOPICS_FILE_NAME]
    changed = len(manifest.git_dirty)
    if changed == 0:
        tree = "clean"
    else:
        tree = f"{changed} changed path{'s' if changed > 1 else ''} (recorded in the manifest)"
    return "\n".join(
        [
            f"OpenAlex Topics download {manifest.download_id}",
            f"Retrieved at {manifest.retrieved_at:%Y-%m-%dT%H:%M:%SZ} from {manifest.url}",
            f"Topics: {manifest.records_written} written, {manifest.distinct_ids} distinct ids, "
            f"meta.count {manifest.meta_count}",
            f"Pages: {manifest.pages}; HTTP requests: {manifest.requests}",
            f"Data: {plan.places.data_file} ({data.size_bytes:,} bytes, sha256 {data.sha256})",
            f"Manifest: {plan.places.manifest_file}",
            f"Git: commit {manifest.git_commit}, {tree}",
            f"Download time: {manifest.seconds:.1f} s",
        ]
    )


def _interruption_note(plan: DownloadPlan | None) -> str:
    if plan is not None and plan.places.manifest_file.exists():
        return (
            f"the download was complete before the interruption: {plan.places.manifest_file} "
            "was written last"
        )
    left = [] if plan is None else leftovers(plan.places)
    if not left:
        return "nothing was written; run the command again"
    return (
        f"the download left {', '.join(str(path) for path in left)} without its "
        f"{MANIFEST_FILE_NAME}; move them away and run the command again"
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.reference.openalex_topics",
        description=(
            "Download the OpenAlex Topics list once (T25), page by page, from the API in "
            "the openalex block of the configuration: the topics go to "
            "data/raw/openalex/topics/<id>/topics.jsonl (gitignored) and their manifest to "
            "results/reference/openalex/topics/<id>/manifest.json (tracked by git). This is a "
            "remote operation that needs the user's authorization (U8). It sends no personal "
            "data: no email, mailto parameter, API key or cookie. Nothing is written unless "
            "every topic arrived, and an existing download is never overwritten."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--download-id",
        help=(
            "name of the new download's folders, under the rules of a snapshot id (default: "
            "the UTC retrieval time, such as 20261105T140309Z)"
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
    git: Callable[[Path, str], GitState] = read_git_state,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and 130
    after an interruption. ``transport`` replaces the network, and the other arguments the
    clocks, git and the waits, in tests."""
    arguments = _parse_arguments(argv)
    root = arguments.project_root.resolve()
    plan: DownloadPlan | None = None
    try:
        config = load_config(root / CONFIG_PATH)
        plan = plan_download(config, root, download_id=arguments.download_id, clock=clock)
        with httpx.Client(transport=transport) as client:
            source = OpenAlexTopicsClient.from_config(
                client, config.openalex, clock=monotonic, sleeper=sleeper
            )
            manifest = run_download(plan, config, root, source, git=git, monotonic=monotonic)
    except KeyboardInterrupt:
        print(f"interrupted: {_interruption_note(plan)}", file=sys.stderr)
        return 130
    except (DownloadError, OpenAlexError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(_summary(plan, manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
