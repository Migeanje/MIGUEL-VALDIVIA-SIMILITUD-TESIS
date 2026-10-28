"""The app's data layer (T28): the saved artifacts of one snapshot, read and checked, never
written, with no Streamlit import.

The app shows what the commands saved and computes nothing new:

- T19's records (:mod:`thematic_redundancy.topic_map.results_store`): ``manifest.json`` first,
  then ``selection.json`` (the provisional winner and the run of its 2-D map), ``topics.parquet``
  (each topic's size and terms) and the gitignored ``map.parquet`` (the 2-D UMAP coordinates of
  every thesis in the representative run, through T16's ``project``), each checked against the
  SHA-256 that T19's manifest records;
- T24a's records (:mod:`thematic_redundancy.gaps.results_store`): ``manifest.json``, which must
  name the T19 manifest read above, then ``gaps.parquet`` and the gitignored ``members.parquet``,
  each checked against it; only the reference run's stable gaps are kept;
- the fichas, through :func:`~thematic_redundancy.corpus.build_fichas.load_fichas`, whose manifest
  must be the one T19 ran on: from each included thesis only its display fields
  (:data:`THESIS_COLUMNS`: doc code, title, program, year and repository link). No author or
  advisor, not even as a code, and no abstract, objectives or keywords reach a display table.

Every reader opens files for reading only. A file that is missing, that differs from its
manifest, or that disagrees with the other records raises :class:`ArtifactError`, which names
the file, the problem and the command that writes it again, so the app can say so in Spanish
(:func:`thematic_redundancy.app.texts.error_message`) instead of showing a traceback.

The filters (:func:`filter_points`, :func:`filter_gaps`) are pure functions of the loaded
tables: an empty selection keeps everything.
"""

import argparse
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError

from thematic_redundancy.corpus.build_fichas import BUILD_COMMAND as FICHAS_COMMAND
from thematic_redundancy.corpus.build_fichas import (
    FICHAS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    load_fichas,
)
from thematic_redundancy.corpus.snapshot import RAW_DIR_NAME, check_snapshot_id, latest_snapshot_id
from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.gaps import results_store as gap_store
from thematic_redundancy.gaps.results_store import GapManifest, GapRow, MemberRow
from thematic_redundancy.gaps.run_gaps import RUN_COMMAND as GAPS_COMMAND
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore
from thematic_redundancy.topic_map import results_store as t19_store
from thematic_redundancy.topic_map.experiment import MapRow, TopicRow
from thematic_redundancy.topic_map.results_store import ExperimentManifest
from thematic_redundancy.topic_map.run_experiment import RUN_COMMAND as T19_COMMAND

CONFIG_PATH = Path("config") / "default.yaml"
NOISE_TOPIC = NOISE_LABEL
"""The topic of the theses that HDBSCAN leaves as noise (-1): they have no topic."""
HIGH_AFFINITY = 2
"""The U4 score of rule (c)'s high-confidence group (D32 (5))."""
THESIS_COLUMNS = ("doc_code", "title", "program", "program_name", "year", "url")
"""The only fields of a thesis that the app loads for display."""
POINT_COLUMNS = ("doc_code", "topic", "x", "y", *THESIS_COLUMNS[1:])
MEMBER_COLUMNS = ("gap_id", "part", *THESIS_COLUMNS)

ArtifactProblem = Literal["missing", "invalid", "inconsistent", "stale"]
"""Why an artifact cannot be used:

- ``missing``: the file is not there;
- ``invalid``: it is damaged, or it differs from the SHA-256 that its manifest records;
- ``inconsistent``: it disagrees with another record of the snapshot;
- ``stale``: it was computed from other inputs than the records read with it.
"""


class ArtifactError(Exception):
    """A saved artifact that the app needs cannot be used, so nothing is shown from it."""

    def __init__(
        self,
        problem: ArtifactProblem,
        artifact: str,
        directory: Path,
        *,
        command: str | None,
        detail: str,
    ) -> None:
        self.problem = problem
        self.artifact = artifact
        """The file's name."""
        self.directory = directory
        self.command = command
        """The command that writes the file again, if any."""
        self.detail = detail
        """What is wrong, in English, for the server's log; never shown to the user."""
        super().__init__(f"{artifact} in {directory} is {problem}: {detail}")


class SettingsError(ValueError):
    """The app's arguments are not valid."""


class _ArgumentParser(argparse.ArgumentParser):
    """An argument parser that raises :class:`SettingsError` instead of exiting, since the app
    runs inside Streamlit's server."""

    def error(self, message: str) -> NoReturn:
        raise SettingsError(message)


# --- Settings and sources ---------------------------------------------------------------------


@dataclass(frozen=True)
class AppSettings:
    """What the app reads: the project root and the snapshot (``None``: the harvested last)."""

    project_root: Path
    snapshot_id: str | None


def parse_arguments(argv: Sequence[str]) -> AppSettings:
    """Return the settings in ``argv``, the arguments after ``--`` of ``streamlit run``:
    ``--project-root PATH`` (default: the current directory) and ``--snapshot-id ID`` (default:
    the snapshot harvested last).

    Raises:
        SettingsError: if an argument is unknown, lacks its value or names no valid snapshot.
    """
    parser = _ArgumentParser(
        prog="streamlit run src/thematic_redundancy/app/main.py --", add_help=False
    )
    parser.add_argument("--project-root", type=Path, default=Path())
    parser.add_argument("--snapshot-id")
    arguments = parser.parse_args(list(argv))
    if arguments.snapshot_id is not None:
        try:
            check_snapshot_id(arguments.snapshot_id)
        except ValueError as error:
            raise SettingsError(str(error)) from None
    return AppSettings(arguments.project_root, arguments.snapshot_id)


@dataclass(frozen=True)
class AppSources:
    """Where the saved artifacts of one snapshot are, and the configuration that places them."""

    project_root: Path
    snapshot_id: str
    config: AppConfig
    t19_results: Path
    t19_processed: Path
    gaps_results: Path
    gaps_processed: Path
    fichas_dir: Path

    @property
    def program_names(self) -> dict[str, str]:
        """Each program's name, by key, as the configuration declares them."""
        return {program.key: program.name for program in self.config.snapshot.programs}


def locate_sources(settings: AppSettings) -> AppSources:
    """Return the folders of the snapshot's artifacts under the project root, from its
    ``config/default.yaml``; nothing in them is read yet.

    Raises:
        ArtifactError: if the configuration is missing or not valid, or no snapshot is named
            and none was harvested.
    """
    root = Path(settings.project_root).resolve()
    config_path = root / CONFIG_PATH
    if not config_path.is_file():
        _fail("missing", config_path, command=None, detail="the configuration is missing")
    try:
        config = load_config(config_path)
        paths = config.paths.resolve_against(root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        _fail("invalid", config_path, command=None, detail=_kind(error))
    snapshot_id = settings.snapshot_id
    if snapshot_id is None:
        raw = paths.data_dir / RAW_DIR_NAME
        try:
            snapshot_id = latest_snapshot_id(raw)
        except FileNotFoundError:
            _fail("missing", raw, command=None, detail="no snapshot was harvested")
        except ValueError as error:
            _fail("invalid", raw, command=None, detail=_kind(error))
    return AppSources(
        project_root=root,
        snapshot_id=snapshot_id,
        config=config,
        t19_results=t19_store.results_dir(paths.results_dir, snapshot_id),
        t19_processed=t19_store.processed_dir(paths.data_dir, snapshot_id),
        gaps_results=gap_store.results_dir(paths.results_dir, snapshot_id),
        gaps_processed=gap_store.processed_dir(paths.data_dir, snapshot_id),
        fichas_dir=paths.data_dir / INTERIM_DIR_NAME / snapshot_id,
    )


# --- T19's experiment -------------------------------------------------------------------------


@dataclass(frozen=True)
class Winner:
    """T19's winning configuration (D30): its name, representation, algorithm and parameter."""

    configuration: str
    representation: str
    algorithm: str
    parameter: int
    """K for K-means, ``min_cluster_size`` for HDBSCAN."""


@dataclass(frozen=True)
class Experiment:
    """What the app reads from T19's ``manifest.json`` and ``selection.json``."""

    snapshot_id: str
    winner: Winner
    provisional: bool
    """Whether the winner waits for T22's labeled-set F1 (stage 2 of D30 E3)."""
    map_run: str
    """The run of the 2-D map: T19's representative (medoid) run of the winner."""
    map_seed: int
    manifest_sha256: str
    """The SHA-256 of T19's ``manifest.json``, which the gap run records."""
    fichas_manifest_sha256: str
    """The SHA-256 of the fichas manifest that T19 ran on."""
    inputs: Mapping[str, str]
    """The SHA-256 of each input manifest that T19 ran on, by name (the embeddings' among them)."""
    files: Mapping[str, str]
    """The SHA-256 of T19's tracked records, by name."""
    data_files: Mapping[str, str]
    """The SHA-256 of T19's gitignored records, by name."""


def load_experiment(sources: AppSources) -> Experiment:
    """Read T19's manifest and its selection, checked against it.

    Raises:
        ArtifactError: if a record is missing, damaged, differs from the manifest, or is of
            another snapshot, or if the two records disagree.
    """
    results = sources.t19_results
    manifest_path = results / t19_store.MANIFEST_FILE_NAME
    content = _read(manifest_path, T19_COMMAND)
    manifest = _parsed(ExperimentManifest, content, manifest_path, T19_COMMAND)
    if manifest.snapshot_id != sources.snapshot_id:
        _fail(
            "inconsistent",
            manifest_path,
            command=T19_COMMAND,
            detail=f"it is of snapshot {manifest.snapshot_id}, not {sources.snapshot_id}",
        )
    selection_path = results / t19_store.SELECTION_FILE_NAME
    selection_bytes = _verified(selection_path, manifest.files, T19_COMMAND)
    try:
        selection = json.loads(selection_bytes)
        chosen = selection["winner"]
        winner = Winner(
            configuration=str(chosen["configuration"]),
            representation=str(chosen["representation"]),
            algorithm=str(chosen["algorithm"]),
            parameter=int(chosen["parameter"]),
        )
        provisional = bool(selection["provisional"])
        map_run, map_seed = str(selection["map"]["run_id"]), int(selection["map"]["seed"])
    except (ValueError, KeyError, TypeError) as error:
        _fail("invalid", selection_path, command=T19_COMMAND, detail=_kind(error))
    if winner.configuration != manifest.winner or provisional != manifest.provisional:
        _fail(
            "inconsistent",
            selection_path,
            command=T19_COMMAND,
            detail="it names another winner or provisional flag than T19's manifest",
        )
    fichas_sha256 = manifest.inputs.get(FICHAS_MANIFEST_FILE_NAME)
    if fichas_sha256 is None:
        _fail("invalid", manifest_path, command=T19_COMMAND, detail="it names no fichas manifest")
    return Experiment(
        snapshot_id=manifest.snapshot_id,
        winner=winner,
        provisional=provisional,
        map_run=map_run,
        map_seed=map_seed,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        fichas_manifest_sha256=fichas_sha256,
        inputs=dict(manifest.inputs),
        files=dict(manifest.files),
        data_files=dict(manifest.data_files),
    )


# --- The theses -------------------------------------------------------------------------------


def load_theses(sources: AppSources, experiment: Experiment) -> pd.DataFrame:
    """Return the display fields of each included thesis (:data:`THESIS_COLUMNS`), in the
    fichas' order: its doc code, title, program key and name, issue year and repository link.

    Raises:
        ArtifactError: if the fichas are missing or differ from their manifest, or if their
            manifest is not the one T19 ran on.
    """
    interim = sources.fichas_dir
    manifest_path = interim / FICHAS_MANIFEST_FILE_NAME
    content = _read(manifest_path, FICHAS_COMMAND)
    if hashlib.sha256(content).hexdigest() != experiment.fichas_manifest_sha256:
        _fail(
            "stale",
            manifest_path,
            command=T19_COMMAND,
            detail="the fichas changed after T19 ran on them",
        )
    try:
        fichas = load_fichas(interim)
    except FileNotFoundError:
        _fail("missing", interim / FICHAS_FILE_NAME, command=FICHAS_COMMAND, detail="not found")
    except ValueError as error:
        _fail("invalid", interim / FICHAS_FILE_NAME, command=FICHAS_COMMAND, detail=_kind(error))
    rows = [
        (
            ficha.doc_code,
            ficha.title,
            ficha.program_key,
            ficha.program_name,
            ficha.issue_year,
            ficha.handle_url,
        )
        for ficha in fichas
        if ficha.include
    ]
    theses = pd.DataFrame(rows, columns=list(THESIS_COLUMNS))
    theses["year"] = theses["year"].astype("Int64")
    return theses


# --- The 2-D map ------------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicMapView:
    """The winner's map run: each thesis's topic and 2-D coordinates, and the run's topics."""

    run_id: str
    points: pd.DataFrame
    """One row per thesis (:data:`POINT_COLUMNS`); ``topic`` is -1 for the noise."""
    topics: tuple[TopicRow, ...]
    """The run's topics in ascending order, with their sizes and top terms (T19)."""


def load_topic_map(
    sources: AppSources, experiment: Experiment, theses: pd.DataFrame
) -> TopicMapView:
    """Return the 2-D map of T19's map run (``map.parquet``) with its topics and the display
    fields of ``theses``, each record checked against T19's manifest.

    Raises:
        ArtifactError: if a record is missing or differs from the manifest, or the map is of
            another run, its labels disagree with the run's topics, or a thesis has no ficha.
    """
    map_path = sources.t19_processed / t19_store.MAP_FILE_NAME
    rows = _verified_table(MapRow, map_path, experiment.data_files, T19_COMMAND)
    if any(row.run_id != experiment.map_run for row in rows):
        _fail("inconsistent", map_path, command=T19_COMMAND, detail="it holds another run")
    topics = _run_topics(sources, experiment, experiment.map_run)
    labels = [row.label for row in rows]
    known = {topic.topic for topic in topics}
    sizes_agree = all(labels.count(topic.topic) == topic.size for topic in topics)
    if not sizes_agree or any(label not in known and label != NOISE_TOPIC for label in labels):
        _fail(
            "inconsistent",
            map_path,
            command=T19_COMMAND,
            detail=f"its labels disagree with the topics of {t19_store.TOPICS_FILE_NAME}",
        )
    points = pd.DataFrame(
        {
            "doc_code": [row.doc_code for row in rows],
            "topic": labels,
            "x": [row.map_x for row in rows],
            "y": [row.map_y for row in rows],
        }
    )
    joined = _with_display_fields(points, theses, map_path, T19_COMMAND, unique=True)
    return TopicMapView(
        run_id=experiment.map_run, points=joined[list(POINT_COLUMNS)], topics=topics
    )


# --- The gaps ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StableGap:
    """One stable gap of the reference run (T24a), with what the app shows of its topics."""

    number: int
    """Its place among the reference run's stable gaps, from 1, as ``summary.md`` lists them."""
    row: GapRow
    topic_terms: tuple[tuple[str, ...], ...]
    """The top terms of each of its topics (T19), in the order of ``row.topics``."""
    topic_sizes: tuple[int, ...]


@dataclass(frozen=True)
class GapRecords:
    """The gap run's stable gaps and their theses, with the values the run used."""

    reference_run: str
    runs: int
    """The winner's seed runs that the stability filter compared."""
    min_runs: int
    """The runs a gap must be present in to be stable (``gaps.stability_min_runs``)."""
    windows: tuple[tuple[int, int], tuple[int, int]]
    """Rule (d)'s earlier and later windows, first and last year of each."""
    program_names: Mapping[str, str]
    gaps: tuple[StableGap, ...]
    members: pd.DataFrame
    """The theses of the stable gaps (:data:`MEMBER_COLUMNS`)."""


@dataclass(frozen=True)
class SavedGapRun:
    """The gap run's records as saved: its manifest, every run's gap rows and their members."""

    manifest: GapManifest
    rows: tuple[GapRow, ...]
    members: tuple[MemberRow, ...]
    gaps_path: Path
    """Where ``gaps.parquet`` is, which an error about its rows names."""


def read_gap_run(sources: AppSources, experiment: Experiment) -> SavedGapRun:
    """Return the gap run's manifest, rows and members, each checked against the manifest, which
    must name the snapshot and T19's manifest read in ``experiment``.

    Raises:
        ArtifactError: if a record is missing or differs from its manifest, or the gap run read
            another T19 record or snapshot.
    """
    results = sources.gaps_results
    manifest_path = results / gap_store.MANIFEST_FILE_NAME
    content = _read(manifest_path, GAPS_COMMAND)
    manifest = _parsed(GapManifest, content, manifest_path, GAPS_COMMAND)
    if manifest.snapshot_id != sources.snapshot_id:
        _fail(
            "inconsistent",
            manifest_path,
            command=GAPS_COMMAND,
            detail=f"it is of snapshot {manifest.snapshot_id}, not {sources.snapshot_id}",
        )
    if manifest.experiment.get(t19_store.MANIFEST_FILE_NAME) != experiment.manifest_sha256:
        _fail(
            "stale",
            manifest_path,
            command=GAPS_COMMAND,
            detail="the gaps were computed on another T19 record",
        )
    gaps_path = results / gap_store.GAPS_FILE_NAME
    rows = _verified_table(GapRow, gaps_path, manifest.files, GAPS_COMMAND)
    members_path = sources.gaps_processed / gap_store.MEMBERS_FILE_NAME
    member_rows = _verified_table(MemberRow, members_path, manifest.data_files, GAPS_COMMAND)
    return SavedGapRun(manifest=manifest, rows=rows, members=member_rows, gaps_path=gaps_path)


def load_gaps(sources: AppSources, experiment: Experiment, theses: pd.DataFrame) -> GapRecords:
    """Return the stable gaps of the gap run's reference run, in its order, with their topics'
    terms and the display fields of their theses, each record checked against its manifest.

    Raises:
        ArtifactError: if a record is missing or differs from its manifest, the gap run read
            another T19 record or snapshot, or a gap names a topic or a thesis that is unknown.
    """
    saved = read_gap_run(sources, experiment)
    manifest, rows, member_rows, gaps_path = (
        saved.manifest,
        saved.rows,
        saved.members,
        saved.gaps_path,
    )
    manifest_path = gaps_path.parent / gap_store.MANIFEST_FILE_NAME
    members_path = sources.gaps_processed / gap_store.MEMBERS_FILE_NAME
    try:
        settings = manifest.settings
        min_runs = int(settings["stability_min_runs"])
        first, split, last = (
            int(settings[name])
            for name in ("temporal_first_year", "temporal_split_year", "temporal_last_year")
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail("invalid", manifest_path, command=GAPS_COMMAND, detail=_kind(error))
    reference = manifest.reference_run
    topics = {topic.topic: topic for topic in _run_topics(sources, experiment, reference)}
    stable = [row for row in rows if row.run_id == reference and row.reference and row.stable]
    gaps = []
    for number, row in enumerate(stable, start=1):
        if any(topic not in topics for topic in row.topics):
            _fail("inconsistent", gaps_path, command=GAPS_COMMAND, detail="unknown topic")
        gaps.append(
            StableGap(
                number=number,
                row=row,
                topic_terms=tuple(topics[topic].terms for topic in row.topics),
                topic_sizes=tuple(topics[topic].size for topic in row.topics),
            )
        )
    wanted = {row.gap_id for row in stable}
    members = pd.DataFrame(
        [(row.gap_id, row.part, row.doc_code) for row in member_rows if row.gap_id in wanted],
        columns=["gap_id", "part", "doc_code"],
    )
    joined = _with_display_fields(members, theses, members_path, GAPS_COMMAND, unique=False)
    return GapRecords(
        reference_run=reference,
        runs=len(manifest.runs),
        min_runs=min_runs,
        windows=((first, split - 1), (split, last)),
        program_names=sources.program_names,
        gaps=tuple(gaps),
        members=joined[list(MEMBER_COLUMNS)],
    )


def members_of(records: GapRecords, gap_id: str) -> pd.DataFrame:
    """Return the theses of the gap ``gap_id``, in stored order (:data:`MEMBER_COLUMNS`)."""
    members = records.members
    return members[members["gap_id"] == gap_id].reset_index(drop=True)


# --- Filters ----------------------------------------------------------------------------------


def filter_points(
    points: pd.DataFrame,
    *,
    topics: Collection[int] = (),
    programs: Collection[str] = (),
    include_noise: bool = True,
) -> pd.DataFrame:
    """Return the map's points of ``topics`` (the noise is topic -1) and ``programs``; an empty
    selection keeps every topic or program. ``include_noise`` false drops the noise."""
    keep = pd.Series(True, index=points.index)
    if topics:
        keep &= points["topic"].isin(list(topics))
    if programs:
        keep &= points["program"].isin(list(programs))
    if not include_noise:
        keep &= points["topic"] != NOISE_TOPIC
    return points[keep]


def filter_gaps(
    gaps: Sequence[StableGap],
    *,
    rules: Collection[str] = (),
    programs: Collection[str] = (),
    high_affinity_only: bool = False,
) -> tuple[StableGap, ...]:
    """Return the gaps of ``rules`` that concern ``programs``, in order; an empty selection
    keeps every rule or program.

    A gap concerns a program when its topics hold theses of that program, or, under rule (c),
    when the program is its source or its target. ``high_affinity_only`` keeps rule (c)'s
    high-confidence group alone: its gaps of affinity :data:`HIGH_AFFINITY`.
    """
    chosen = []
    for gap in gaps:
        row = gap.row
        concerned = {*row.programs, *(p for p in (row.source, row.target) if p is not None)}
        if rules and row.rule not in rules:
            continue
        if programs and not concerned & set(programs):
            continue
        if high_affinity_only and row.affinity != HIGH_AFFINITY:
            continue
        chosen.append(gap)
    return tuple(chosen)


# --- Helpers ----------------------------------------------------------------------------------


def read_topic_rows(sources: AppSources, experiment: Experiment) -> tuple[TopicRow, ...]:
    """Return every topic of T19's final runs (``topics.parquet``), checked against T19's
    manifest.

    Raises:
        ArtifactError: if the table is missing or differs from the manifest.
    """
    path = sources.t19_results / t19_store.TOPICS_FILE_NAME
    return _verified_table(TopicRow, path, experiment.files, T19_COMMAND)


def _run_topics(sources: AppSources, experiment: Experiment, run: str) -> tuple[TopicRow, ...]:
    """Return the topics of the run ``run`` in ascending order, checked against T19's manifest."""
    rows = read_topic_rows(sources, experiment)
    return tuple(sorted((row for row in rows if row.run_id == run), key=lambda row: row.topic))


def _with_display_fields(
    frame: pd.DataFrame, theses: pd.DataFrame, path: Path, command: str, *, unique: bool
) -> pd.DataFrame:
    """Return ``frame`` with the display fields of the theses of its ``doc_code`` column, each
    of which must have a ficha, and must be listed once when ``unique``."""
    known = set(theses["doc_code"])
    unknown = sum(code not in known for code in frame["doc_code"])
    if unknown:
        _fail("inconsistent", path, command=command, detail=f"{unknown} theses have no ficha")
    if unique and frame["doc_code"].duplicated().any():
        _fail("inconsistent", path, command=command, detail="a thesis is listed twice")
    return frame.merge(theses, on="doc_code", how="left", validate="many_to_one")


def _read(path: Path, command: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        _fail("missing", path, command=command, detail="not found")
    except OSError as error:
        _fail("invalid", path, command=command, detail=_kind(error))


def _parsed[M: BaseModel](model_cls: type[M], content: bytes, path: Path, command: str) -> M:
    try:
        return model_cls.model_validate_json(content)
    except ValidationError as error:
        _fail("invalid", path, command=command, detail=f"{error.error_count()} validation errors")


def _verified(path: Path, recorded: Mapping[str, str], command: str) -> bytes:
    """Return the bytes of ``path`` once their SHA-256 is the one ``recorded`` for its name."""
    content = _read(path, command)
    if hashlib.sha256(content).hexdigest() != recorded.get(path.name):
        _fail("invalid", path, command=command, detail="it differs from its manifest")
    return content


def _verified_table[M: BaseModel](
    model_cls: type[M],
    path: Path,
    recorded: Mapping[str, str],
    command: str,
    store: TableStore | None = None,
) -> tuple[M, ...]:
    """Return the rows of the table ``path``, parsed from the bytes whose SHA-256 is checked."""
    sha256 = recorded.get(path.name)
    if sha256 is None:
        _fail("invalid", path, command=command, detail="its manifest does not list it")
    if not path.is_file():
        _fail("missing", path, command=command, detail="not found")
    try:
        return (store or ParquetTableStore()).read_verified(model_cls, path, sha256=sha256)
    except (OSError, ValueError) as error:
        _fail("invalid", path, command=command, detail=_kind(error))


def _fail(problem: ArtifactProblem, path: Path, *, command: str | None, detail: str) -> NoReturn:
    raise ArtifactError(problem, path.name, path.parent, command=command, detail=detail)


def _kind(error: BaseException) -> str:
    """Return the kind of ``error``, never its message, which may quote a file's content."""
    return type(error).__name__
