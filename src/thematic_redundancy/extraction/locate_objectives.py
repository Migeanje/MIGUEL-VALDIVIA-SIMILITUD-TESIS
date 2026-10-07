"""Locate the objectives of every thesis PDF of a snapshot, from its extracted page texts.

:func:`locate_all` reads the text manifest ``data/interim/<snapshot_id>/pages/manifest.json``
of a snapshot, the latest one by default, and takes every PDF whose page texts are ``ok``,
in snapshot order. For each one it checks that the page file is still the one extracted
(its SHA-256), runs the objectives locator
(:func:`~thematic_redundancy.extraction.objectives.locate_objectives`) on its pages, and
adds one :class:`~thematic_redundancy.extraction.objectives.ObjectivesRecord` to the table.
Business rules such as the D24 duplicates or the wrong file of item 11777 belong to the
fichas (T12), so every page file is looked at.

The run takes about a minute, as no OCR is involved, so it always rebuilds the whole table.
It writes, atomically and into the gitignored ``data/interim/<snapshot_id>/``:

- ``objectives.parquet``: one row per PDF, with the objectives' text;
- ``objectives_manifest.json``: the settings, the locator version, the table's SHA-256, the
  time taken and the counts, with no text.

``--summary-out`` also writes that manifest, which holds numbers only, to a results file,
and ``--make-sample`` then draws the manual verification sample and writes its workbook
(:mod:`thematic_redundancy.labeling.objectives_check`), never over an existing one::

    uv run python -m thematic_redundancy.extraction.locate_objectives [--project-root PATH]
        [--snapshot-id ID] [--summary-out PATH] [--make-sample]
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

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
from thematic_redundancy.extraction.objectives import (
    LOCATOR_VERSION,
    OBJECTIVES_FLAGS,
    HeadingPattern,
    ObjectivesRecord,
    locate_objectives,
)
from thematic_redundancy.extraction.page_text import PageText
from thematic_redundancy.extraction.text_manifest import (
    INTERIM_DIR_NAME,
    PAGES_DIR_NAME,
    TEXT_MANIFEST_FILE_NAME,
    Sha256,
    read_text_manifest,
    write_json_atomically,
)
from thematic_redundancy.labeling.objectives_check import (
    WORKBOOK_FILE_NAME,
    make_verification_workbook,
    workbook_directory,
)
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

EXTRACT_COMMAND = "uv run python -m thematic_redundancy.extraction.extract_text"
"""Command that extracts the page texts; error messages point to it."""

OBJECTIVES_FILE_NAME = "objectives.parquet"
OBJECTIVES_MANIFEST_FILE_NAME = "objectives_manifest.json"

_PATTERNS: tuple[HeadingPattern, ...] = (
    "objetivo_general",
    "objetivo_principal",
    "objetivo_general_intro",
    "general_subheading",
    "objetivos_block",
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ObjectivesSettings(_Model):
    """Everything that shapes the objectives table, besides the page texts."""

    locator_version: StrictInt
    general_max_chars: Count
    specific_max_chars: Count
    min_chars: Count

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        objectives = config.objectives
        return cls(
            locator_version=LOCATOR_VERSION,
            general_max_chars=objectives.general_max_chars,
            specific_max_chars=objectives.specific_max_chars,
            min_chars=objectives.min_chars,
        )

    def fingerprint(self) -> str:
        """Return the SHA-256 of these settings, which changes whenever one of them does."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class Spread(_Model):
    """Median, 90th percentile (nearest rank) and maximum of some counts."""

    median: StrictFloat
    p90: Count
    max: Count

    @classmethod
    def of(cls, values: Sequence[int]) -> Self | None:
        if not values:
            return None
        ordered = sorted(values)
        return cls(
            median=float(statistics.median(ordered)),
            p90=ordered[math.ceil(0.9 * len(ordered)) - 1],
            max=ordered[-1],
        )


class ObjectivesSummary(_Model):
    """Counts over the objectives table: numbers only, never text or an item."""

    documents: Count
    """PDFs whose page texts were looked at."""
    text_errors: Count
    """PDFs left out because their text extraction failed."""
    by_status: dict[str, Count]
    by_program: dict[str, dict[str, Count]]
    """Documents per status within each program, in snapshot order."""
    by_pattern: dict[str, Count]
    """Extracted documents per heading pattern."""
    flags: dict[str, Count]
    """Extracted documents per quality flag; a document may carry several."""
    general_chars: Spread | None
    specific_chars: Spread | None
    page_start: Spread | None
    """Page index, counted from 0, of the heading that the objectives were found under."""


class ObjectivesManifest(_Model):
    """Content of ``objectives_manifest.json``, and of the summary file."""

    snapshot_id: SnapshotId
    settings: ObjectivesSettings
    settings_fingerprint: Sha256
    text_settings_fingerprint: Sha256
    """Settings fingerprint of the text manifest whose page texts were read."""
    created_at: UtcDatetime
    table_file: str
    table_sha256: Sha256
    seconds: StrictFloat
    summary: ObjectivesSummary


@dataclass(frozen=True)
class ObjectivesReport:
    """Outcome of :func:`locate_all`."""

    snapshot_id: str
    table_path: Path
    manifest_path: Path
    manifest: ObjectivesManifest
    records: tuple[ObjectivesRecord, ...]
    snapshot_records: tuple[SnapshotRecord, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def locate_all(
    config: AppConfig,
    project_root: Path,
    *,
    store: TableStore | None = None,
    snapshot_id: str | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> ObjectivesReport:
    """Locate the objectives in every extracted PDF of a snapshot and save the table.

    The snapshot is ``data/raw/<snapshot_id>`` under ``paths.data_dir``, and defaults to the
    one harvested last. ``store`` reads the page records and saves the table, as Parquet by
    default; ``timer`` measures the run.

    Raises:
        FileNotFoundError: if the snapshot, its text manifest or a page file is missing.
        ValueError: if a page file changed since its extraction, if a manifest is damaged or
            belongs to another snapshot, or if the text manifest names an unknown item.
        OSError: if reading or writing the disk fails.
    """
    started = timer()
    store = store or ParquetTableStore()
    data_dir = config.paths.resolve_against(project_root).data_dir
    raw_dir = data_dir / RAW_DIR_NAME
    if snapshot_id is None:
        snapshot_id = latest_snapshot_id(raw_dir)
    snapshot_dir = raw_dir / check_snapshot_id(snapshot_id)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"snapshot '{snapshot_id}' not found under {raw_dir}")
    _, snapshot_records = read_snapshot(snapshot_dir)
    by_uuid = {record.uuid: record for record in snapshot_records}
    interim_dir = data_dir / INTERIM_DIR_NAME / snapshot_id
    pages_dir = interim_dir / PAGES_DIR_NAME
    text_manifest = read_text_manifest(pages_dir / TEXT_MANIFEST_FILE_NAME)
    if text_manifest is None:
        raise FileNotFoundError(f"no page texts in {pages_dir}; run {EXTRACT_COMMAND} first")
    if text_manifest.snapshot_id != snapshot_id:
        raise ValueError(
            f"the text manifest in {pages_dir} belongs to snapshot '{text_manifest.snapshot_id}'"
        )
    records: list[ObjectivesRecord] = []
    for item, entry in text_manifest.items.items():
        if entry.status != "ok" or entry.file is None:
            continue
        if item not in by_uuid:
            raise ValueError(f"the text manifest names item {item}, which the snapshot lacks")
        path = pages_dir / entry.file
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.output_sha256:
            raise ValueError(
                f"the page file {path} changed since its extraction; run {EXTRACT_COMMAND} again"
            )
        pages = store.read(PageText, path)
        found = locate_objectives(pages, config.objectives)
        records.append(
            ObjectivesRecord(
                item_uuid=item,
                handle=by_uuid[item].handle,
                program=entry.program_key,
                status=found.status,
                objective_general=found.general,
                objectives_specific=found.specific,
                page_start=found.page_start,
                page_end=found.page_end,
                pattern=found.pattern,
                flags=found.flags,
                general_chars=len(found.general or ""),
                specific_chars=len(found.specific or ""),
                pages=len(pages),
                pages_sha256=digest,
            )
        )
    table_path = interim_dir / OBJECTIVES_FILE_NAME
    store.write(records, table_path, model_cls=ObjectivesRecord)
    settings = ObjectivesSettings.from_config(config)
    manifest = ObjectivesManifest(
        snapshot_id=snapshot_id,
        settings=settings,
        settings_fingerprint=settings.fingerprint(),
        text_settings_fingerprint=text_manifest.config_fingerprint,
        created_at=_as_utc(clock()),
        table_file=table_path.name,
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        seconds=round(max(timer() - started, 0.0), 3),
        summary=summarize(
            records,
            text_errors=sum(entry.status == "error" for entry in text_manifest.items.values()),
        ),
    )
    manifest_path = interim_dir / OBJECTIVES_MANIFEST_FILE_NAME
    write_json_atomically(manifest_path, manifest.model_dump_json(indent=2))
    return ObjectivesReport(
        snapshot_id=snapshot_id,
        table_path=table_path,
        manifest_path=manifest_path,
        manifest=manifest,
        records=tuple(records),
        snapshot_records=snapshot_records,
    )


def _as_utc(moment: datetime) -> datetime:
    if moment.utcoffset() is None:
        raise ValueError(f"the clock must give a time with a time zone, got '{moment}'")
    return moment.astimezone(UTC)


def summarize(records: Sequence[ObjectivesRecord], *, text_errors: int = 0) -> ObjectivesSummary:
    """Count ``records`` by status, program, pattern and flag, with the spread of their
    lengths and heading pages."""
    extracted = [record for record in records if record.status == "extracted"]
    by_program: dict[str, dict[str, int]] = {}
    for record in records:
        counts = by_program.setdefault(record.program, {"extracted": 0, "not_found": 0})
        counts[record.status] += 1
    patterns = Counter(record.pattern for record in extracted)
    flags = Counter(flag for record in extracted for flag in record.flags)
    return ObjectivesSummary(
        documents=len(records),
        text_errors=text_errors,
        by_status={"extracted": len(extracted), "not_found": len(records) - len(extracted)},
        by_program=by_program,
        by_pattern={pattern: patterns[pattern] for pattern in _PATTERNS},
        flags={flag: flags[flag] for flag in OBJECTIVES_FLAGS},
        general_chars=Spread.of([record.general_chars for record in extracted]),
        specific_chars=Spread.of(
            [record.specific_chars for record in extracted if record.specific_chars]
        ),
        page_start=Spread.of(
            [record.page_start for record in extracted if record.page_start is not None]
        ),
    )


def write_summary(path: Path, manifest: ObjectivesManifest) -> None:
    """Write ``manifest`` to ``path`` as JSON: settings and counts, never an item or any
    text. The directory is created when it is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(path, json.dumps(manifest.model_dump(mode="json"), indent=2))


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _describe(report: ObjectivesReport) -> str:
    """Describe the run in numbers only."""
    summary = report.manifest.summary
    extracted = summary.by_status["extracted"]
    not_found = summary.by_status["not_found"]
    lines = [
        f"Snapshot {report.snapshot_id}: {summary.documents} page files, "
        f"{summary.text_errors} left out after a failed text extraction",
        f"Objectives: extracted {extracted}, not found {not_found} "
        f"({_share(not_found, summary.documents)})",
        "By program: "
        + ", ".join(
            f"{program} {counts['extracted']}/{sum(counts.values())}"
            for program, counts in summary.by_program.items()
        ),
        "By pattern: "
        + ", ".join(f"{pattern} {count}" for pattern, count in summary.by_pattern.items()),
        "Flags: " + ", ".join(f"{flag} {count}" for flag, count in summary.flags.items()),
        f"Table {report.table_path}",
        f"Seconds: {report.manifest.seconds:.1f}",
    ]
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.extraction.locate_objectives",
        description=(
            "Locate the general and specific objectives in the extracted page texts of every "
            "thesis PDF of a snapshot, and save them as data/interim/<snapshot>/"
            f"{OBJECTIVES_FILE_NAME}."
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
        help="snapshot whose page texts to read (default: the snapshot harvested last)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        metavar="PATH",
        help="also write the settings and counts, without items or text, as JSON to PATH",
    )
    parser.add_argument(
        "--make-sample",
        action="store_true",
        help=(
            "then draw the manual verification sample and write its workbook; an existing "
            "workbook is never overwritten"
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and
    130 after an interruption."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    try:
        config = load_config(project_root / CONFIG_PATH)
        snapshot_id = arguments.snapshot_id
        if snapshot_id is None:
            data_dir = config.paths.resolve_against(project_root).data_dir
            snapshot_id = latest_snapshot_id(data_dir / RAW_DIR_NAME)
        if arguments.make_sample:
            workbook = workbook_directory(config, project_root, snapshot_id) / WORKBOOK_FILE_NAME
            if workbook.exists():
                raise FileExistsError(
                    f"the verification workbook {workbook} already exists and may hold "
                    "verdicts; move it away first to draw a new one"
                )
        report = locate_all(config, project_root, snapshot_id=snapshot_id, clock=clock, timer=timer)
        if arguments.summary_out is not None:
            write_summary(arguments.summary_out, report.manifest)
        print(_describe(report))
        if arguments.make_sample:
            manifest = report.manifest
            made = make_verification_workbook(
                config,
                project_root,
                report.snapshot_id,
                report.records,
                report.snapshot_records,
                provenance={
                    "locator_version": manifest.settings.locator_version,
                    "settings_fingerprint": manifest.settings_fingerprint,
                    "table_sha256": manifest.table_sha256,
                },
                clock=clock,
            )
            items = made.sample.items
            programs = Counter(item.program for item in items)
            statuses = Counter(item.status for item in items)
            per_program = ", ".join(
                f"{program.key} {programs[program.key]}" for program in config.snapshot.programs
            )
            print(
                f"Verification sample: {len(items)} theses ({per_program}); "
                f"extracted {statuses['extracted']}, not found {statuses['not_found']}; "
                f"{made.redacted} cells with a person's name redacted"
            )
            print(f"Workbook {made.workbook_path}")
    except KeyboardInterrupt:
        print("interrupted: nothing half-written was left; run the command again", file=sys.stderr)
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
