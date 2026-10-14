"""Harvest the metadata of the program collections into a new raw snapshot.

:func:`harvest_snapshot` uses an :class:`ItemRepository` to list every item of each
configured program collection whose ``dc.date.issued`` year falls within the snapshot years.
It keeps every item type, because the theses are chosen later, with an exclusion log. It
refuses any listing that does not match the total that the repository reports, or that
lists an item twice. It also counts the items of the whole faculty community under the same
filter, and refuses a count that the program listings do not add up to, since a program
collection may be missing from the configuration; a difference that is known and out of
scope is accepted by giving that difference. Before anything reaches the disk, it drops the
metadata fields that hold national identity numbers (keys ending in ``.dni``) and records
their names. It then writes ``data/raw/<snapshot_id>/`` atomically, with ``metadata.jsonl``
and ``manifest.json``, and never overwrites an existing snapshot.

Run it once per snapshot::

    uv run python -m thematic_redundancy.corpus.harvest [--project-root PATH] [--snapshot-id ID]
        [--accept-faculty-difference D]

The default snapshot id is the UTC harvest time, such as ``20261002T150405Z``.
"""

import argparse
import hashlib
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import httpx
import yaml
from pydantic import JsonValue

from thematic_redundancy.corpus.dspace import DSpaceRestRepository
from thematic_redundancy.corpus.repository import (
    IssuedYears,
    ItemRepository,
    RepositoryError,
    ScopeListing,
)
from thematic_redundancy.corpus.snapshot import (
    METADATA_FILE_NAME,
    RAW_DIR_NAME,
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    check_snapshot_id,
    default_snapshot_id,
    encode_records,
    require_new_snapshot,
    without_identity_numbers,
    write_snapshot,
)
from thematic_redundancy.shared.config import AppConfig, ProgramConfig, load_config

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

TYPE_KEY = "renati.type"
ISSUED_KEY = "dc.date.issued"

NO_TYPE = "(none)"
"""Key under which ``items_by_type`` counts the items without a ``renati.type`` value."""

_ISSUED_DATE = re.compile(r"([0-9]{4})(?:-([0-9]{2})(?:-([0-9]{2}))?)?(?=$|[T ])")
"""Start of a ``dc.date.issued`` value: a year, a month, or a day, maybe followed by a time."""


class HarvestError(RuntimeError):
    """The listings do not reconcile: an item is missing or listed twice, or the faculty
    community counts items that the program listings do not hold."""


@dataclass(frozen=True)
class HarvestReport:
    """Outcome of :func:`harvest_snapshot`: where the snapshot is, and its manifest."""

    snapshot_dir: Path
    manifest: SnapshotManifest


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def harvest_snapshot(
    config: AppConfig,
    project_root: Path,
    repository: ItemRepository,
    *,
    snapshot_id: str | None = None,
    accepted_faculty_difference: int | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> HarvestReport:
    """List the configured program collections and write them as a new snapshot.

    The snapshot goes to ``data/raw/<snapshot_id>`` under ``paths.data_dir``, and
    ``snapshot_id`` defaults to the UTC harvest time. The id and the target are checked
    before the first request. Nothing is written unless every listing reconciles, and
    unless the faculty community holds as many items as the program listings together, or
    exactly ``accepted_faculty_difference`` more (fewer, when it is negative): a difference
    found in an earlier attempt, checked, and accepted as out of scope.

    Raises:
        SnapshotExistsError: if the snapshot already exists.
        HarvestError: if a listing misses an item that the repository counted, if an item
            appears twice, or if the faculty total differs from the programs' sum by
            neither 0 nor the accepted difference.
        RepositoryError: if the repository fails or answers unexpectedly.
        ValueError: if ``snapshot_id`` is not one plain name, if ``clock`` gives a time
            without a time zone, or if the data directory resolves outside the project root.
    """
    harvested_at = _as_utc(clock())
    if snapshot_id is None:
        snapshot_id = default_snapshot_id(harvested_at)
    data_dir = config.paths.resolve_against(project_root).data_dir
    snapshot_dir = data_dir / RAW_DIR_NAME / check_snapshot_id(snapshot_id)
    require_new_snapshot(snapshot_dir)
    years = IssuedYears(first=config.snapshot.year_start, last=config.snapshot.year_end)
    records: list[SnapshotRecord] = []
    programs: dict[str, ProgramSummary] = {}
    dropped_keys: set[str] = set()
    listed_under: dict[UUID, str] = {}
    for program in config.snapshot.programs:
        listing = repository.list_items(program.collection_uuid, years)
        _reconcile(program, listing, listed_under)
        for item in listing.items:
            metadata, dropped = without_identity_numbers(item.metadata)
            dropped_keys.update(dropped)
            records.append(
                SnapshotRecord(
                    uuid=item.uuid,
                    handle=item.handle,
                    program_key=program.key,
                    collection_uuid=program.collection_uuid,
                    harvested_at=harvested_at,
                    metadata=dict(sorted(metadata.items())),
                )
            )
        programs[program.key] = ProgramSummary(
            collection_uuid=program.collection_uuid, items=len(listing.items)
        )
    faculty_total = repository.count_items(config.repository.faculty_community_uuid, years)
    _reconcile_faculty(
        config.repository.faculty_community_uuid,
        faculty_total,
        len(records),
        accepted_faculty_difference,
    )
    search = repository.describe_search(years)
    content = encode_records(records)
    future_dated, unparsed = _count_issue_dates(records, harvested_at.date())
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        harvested_at=harvested_at,
        base_url=str(config.repository.base_url).rstrip("/"),
        search_url=search.search_url,
        query_parameters=dict(search.query_parameters),
        programs=programs,
        total_items=len(records),
        faculty_community_uuid=config.repository.faculty_community_uuid,
        faculty_total=faculty_total,
        items_by_type=_count_types(records),
        future_dated_items=future_dated,
        unparsed_issue_dates=unparsed,
        dropped_metadata_keys=tuple(sorted(dropped_keys)),
        metadata_sha256=hashlib.sha256(content).hexdigest(),
    )
    write_snapshot(snapshot_dir, content, manifest)
    return HarvestReport(snapshot_dir=snapshot_dir, manifest=manifest)


def _as_utc(moment: datetime) -> datetime:
    if moment.utcoffset() is None:
        raise ValueError(
            f"the harvest clock must give a time with a time zone, got '{moment.isoformat()}'"
        )
    return moment.astimezone(UTC)


def _reconcile(
    program: ProgramConfig, listing: ScopeListing, listed_under: dict[UUID, str]
) -> None:
    """Refuse a listing that misses or repeats items, and remember the items it holds."""
    if len(listing.items) != listing.reported_total:
        raise HarvestError(
            f"the listing of program '{program.key}' (collection {program.collection_uuid}) "
            f"holds {len(listing.items)} items, but the repository reported "
            f"{listing.reported_total}; nothing was written"
        )
    for item in listing.items:
        if item.uuid in listed_under:
            if listed_under[item.uuid] == program.key:
                where = f"twice under program '{program.key}'"
            else:
                where = (
                    f"under program '{listed_under[item.uuid]}' and again under program "
                    f"'{program.key}'"
                )
            raise HarvestError(f"item {item.uuid} is listed {where}; nothing was written")
        listed_under[item.uuid] = program.key


def _reconcile_faculty(
    faculty: UUID, faculty_total: int, total_items: int, accepted: int | None
) -> None:
    """Refuse a faculty total that the program listings do not account for, unless their
    difference is the accepted one.

    A larger faculty total may mean that a program collection is missing from the
    configuration, and a smaller one that a configured collection lies outside the faculty.
    The difference, not the faculty total, is what a person checked, so a change on either
    side is refused again.
    """
    difference = faculty_total - total_items
    if difference in (0, accepted):
        return
    given = "" if accepted is None else f" (--accept-faculty-difference {accepted} was given)"
    raise HarvestError(
        f"the faculty community {faculty} holds {faculty_total} items in the snapshot years, "
        f"but the program listings add up to {total_items}, a difference of {difference}"
        f"{given}; a program collection may be missing from snapshot.programs, so nothing "
        "was written. If the difference is known and out of scope, run the harvest again "
        f"with --accept-faculty-difference {difference}"
    )


def _text_values(metadata: Mapping[str, Sequence[Mapping[str, JsonValue]]], key: str) -> list[str]:
    """Return the non-blank text values of the metadata field ``key``, in order."""
    return [
        text.strip()
        for entry in metadata.get(key, ())
        if isinstance(text := entry.get("value"), str) and text.strip()
    ]


def _count_types(records: Sequence[SnapshotRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(set(_text_values(record.metadata, TYPE_KEY)) or {NO_TYPE})
    return dict(sorted(counts.items()))


def _earliest_issue_day(value: str) -> date | None:
    """Return the first day that a ``dc.date.issued`` value can mean, or ``None``.

    The value may be a year, a month, a day, or a timestamp. A year or a month starts on
    its first day, so only a date that surely lies ahead counts as future.
    """
    match = _ISSUED_DATE.match(value)
    if match is None:
        return None
    year, month, day = (int(part) if part else 1 for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:  # Such as month 13.
        return None


def _count_issue_dates(records: Sequence[SnapshotRecord], harvest_day: date) -> tuple[int, int]:
    """Count the records issued after ``harvest_day`` and those without a readable date."""
    future = unparsed = 0
    for record in records:
        values = _text_values(record.metadata, ISSUED_KEY)
        issued = _earliest_issue_day(values[0]) if values else None
        if issued is None:
            unparsed += 1
        elif issued > harvest_day:
            future += 1
    return future, unparsed


def _summary(report: HarvestReport) -> str:
    """Describe the snapshot: where it is, its counts, and what was dropped."""
    manifest = report.manifest
    key_width = max(len(key) for key in (*manifest.programs, "total"))
    type_width = max((len(value) for value in manifest.items_by_type), default=0)
    difference = manifest.faculty_total - manifest.total_items
    if difference == 0:
        faculty_note = "equal to the sum of the programs"
    else:
        faculty_note = (
            f"accepted with --accept-faculty-difference {difference}; the programs add up to "
            f"{manifest.total_items}"
        )
    return "\n".join(
        [
            f"Snapshot {manifest.snapshot_id} written to {report.snapshot_dir}",
            f"Harvested at {manifest.harvested_at:%Y-%m-%dT%H:%M:%SZ}",
            "Items per program:",
            *(
                f"  {key:<{key_width}}  {program.items:>5}"
                for key, program in manifest.programs.items()
            ),
            f"  {'total':<{key_width}}  {manifest.total_items:>5}",
            f"Faculty community total: {manifest.faculty_total} ({faculty_note})",
            f"Items per {TYPE_KEY}:",
            *(
                f"  {value:<{type_width}}  {count:>5}"
                for value, count in manifest.items_by_type.items()
            ),
            f"Items issued after the harvest date: {manifest.future_dated_items}",
            f"Items without a readable {ISSUED_KEY}: {manifest.unparsed_issue_dates}",
            f"Dropped metadata keys: {', '.join(manifest.dropped_metadata_keys) or 'none'}",
            f"{METADATA_FILE_NAME} sha256 {manifest.metadata_sha256}",
        ]
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.corpus.harvest",
        description=(
            "Harvest the metadata of the configured program collections from the DSpace REST "
            "API into a new snapshot under data/raw/. DNI fields are dropped before writing."
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
        help="name of the new snapshot directory (default: the UTC harvest time)",
    )
    parser.add_argument(
        "--accept-faculty-difference",
        type=_whole_number,
        metavar="D",
        help=(
            "write the snapshot even though the faculty community holds D items more than "
            "the program collections together (fewer, when D is negative); give the D that a "
            "refused harvest reported, once that difference is known to be out of scope. Any "
            "other difference is still refused (default: the totals must agree)"
        ),
    )
    return parser.parse_args(argv)


def _whole_number(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got '{text}'") from None


def main(
    argv: Sequence[str] | None = None,
    *,
    repository: ItemRepository | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> int:
    """Run the command line and return its exit code; ``repository`` replaces HTTP in tests."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    started = time.perf_counter()
    adapter = repository if isinstance(repository, DSpaceRestRepository) else None
    try:
        config = load_config(project_root / CONFIG_PATH)
        with ExitStack() as stack:
            if repository is None:
                client = stack.enter_context(httpx.Client())
                repository = adapter = DSpaceRestRepository.from_config(client, config.repository)
            report = harvest_snapshot(
                config,
                project_root,
                repository,
                snapshot_id=arguments.snapshot_id,
                accepted_faculty_difference=arguments.accept_faculty_difference,
                clock=clock,
            )
    except (HarvestError, RepositoryError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(_summary(report))
    if adapter is not None:
        print(f"HTTP requests: {adapter.request_count}")
    print(f"Wall time: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
