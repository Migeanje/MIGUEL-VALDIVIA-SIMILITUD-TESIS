"""The records of the temporal split (T27, O04), and where they go.

Under ``results/temporal/<snapshot_id>/``, which git tracks, aggregates only (D33's rule):

- ``runs.parquet``: one row per seed run of the winner: its earlier map (theses, topics, noise,
  terms) and the later period's theses, atypical ones and full-map noise among them;
- ``consistency.parquet``: one row per seed run, period and noise handling: the ARI and the NMI
  between the earlier map's labels and the full map's, and for the later period the counterpart
  agreement (:mod:`~thematic_redundancy.evaluation.temporal`);
- ``topics.parquet``: one row per topic of each earlier map: its size, its top c-TF-IDF terms
  (vocabulary lemmas, as T19's records hold them), its counterparts in the full map, and the
  later theses assigned to it and flagged atypical;
- ``composition.parquet``: the expert sample's candidates and quota per pool and program;
- ``summary.md``: the periods, the metrics per seed and over the seeds, and the sample;
- ``manifest.json``, written last, as the commit point: the provenance, the values used and the
  SHA-256 of every other file.

Under the gitignored ``data/processed/<snapshot_id>/temporal/``, as T19 keeps its per-document
labels there: ``assignments.parquet``, each thesis's label on the earlier map and on the full
map for every seed run, with its similarity and atypical flag when it is a later thesis; and
``sample.parquet``, the expert sample, with each thesis's doc code and title.

Every row says whether the winner was still provisional (T19's flag). Every tracked output, the
tables as the JSON of their rows, passes the gap run's privacy guard
(:func:`~thematic_redundancy.gaps.results_store.check_privacy`) with the theses' doc codes
added, before anything is written; a hit refuses every write.
"""

import hashlib
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from thematic_redundancy.gaps.results_store import check_privacy
from thematic_redundancy.shared.filesystem import write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore

TEMPORAL_DIR_NAME = "temporal"
PROCESSED_DIR_NAME = "processed"
RUNS_FILE_NAME = "runs.parquet"
CONSISTENCY_FILE_NAME = "consistency.parquet"
TOPICS_FILE_NAME = "topics.parquet"
COMPOSITION_FILE_NAME = "composition.parquet"
SUMMARY_FILE_NAME = "summary.md"
MANIFEST_FILE_NAME = "manifest.json"
ASSIGNMENTS_FILE_NAME = "assignments.parquet"
SAMPLE_FILE_NAME = "sample.parquet"

Period = Literal["fitted", "assigned"]
NoiseHandlingName = Literal["own_label", "exclude_either"]


def results_dir(results_root: Path, snapshot_id: str) -> Path:
    """Return ``<results_root>/temporal/<snapshot_id>``."""
    return results_root / TEMPORAL_DIR_NAME / snapshot_id


def processed_dir(data_root: Path, snapshot_id: str) -> Path:
    """Return ``<data_root>/processed/<snapshot_id>/temporal``."""
    return data_root / PROCESSED_DIR_NAME / snapshot_id / TEMPORAL_DIR_NAME


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class TemporalRunRow(_Row):
    """One row of ``runs.parquet``: one seed run's earlier map and later theses."""

    run_id: str
    """T19's run of the full map with this seed."""
    temporal_run_id: str
    """This run's earlier map: T19's run id and the fitted years."""
    seed: int
    reference: bool
    """Whether this is T19's medoid run, whose map gives the expert sample."""
    provisional: bool
    fitted_theses: int
    assigned_theses: int
    topics: int
    noise_theses: int
    noise_share: float
    vocabulary_terms: int
    """The terms of the earlier map's c-TF-IDF: T15's terms that the earlier theses form."""
    full_map_topics: int
    full_map_noise_assigned: int
    """The later theses that are noise in the full map."""
    atypical_threshold: float | None
    """O06's threshold with a global scope; ``None`` with a per-topic one (see the topics)."""
    atypical_theses: int
    atypical_share: float
    umap_seconds: float
    cluster_seconds: float
    ctfidf_seconds: float
    threshold_seconds: float
    assign_seconds: float


class ConsistencyRow(_Row):
    """One row of ``consistency.parquet``: one seed run, period and noise handling."""

    run_id: str
    temporal_run_id: str
    seed: int
    reference: bool
    provisional: bool
    period: Period
    """``assigned``: the later theses' assignment against their full-map labels (O04);
    ``fitted``: the earlier theses' labels on both maps, how far the map itself moves."""
    noise_handling: NoiseHandlingName
    points: int
    ari: float | None
    nmi: float | None
    agreement: float | None
    """The counterpart agreement, for the later period only."""
    agreeing: int | None
    undefined_reason: str | None


class TemporalTopicRow(_Row):
    """One row of ``topics.parquet``: one topic of one earlier map, its terms vocabulary
    lemmas."""

    temporal_run_id: str
    seed: int
    reference: bool
    provisional: bool
    topic: int
    size: int
    """The topic's earlier theses."""
    terms: tuple[str, ...]
    scores: tuple[float, ...]
    atypical_threshold: float
    counterpart_own_label: int | None
    """The full-map label of most of the topic's members, the full map's noise counted."""
    counterpart_exclude_either: int | None
    """The same without the members that are full-map noise; ``None`` when none is left."""
    assigned: int
    """The later theses assigned to the topic."""
    atypical: int


class CompositionRow(_Row):
    """One row of ``composition.parquet``: one pool and program of the expert sample."""

    pool: str
    program: str
    candidates: int
    quota: int
    provisional: bool


class AssignmentRow(_Row):
    """One row of the gitignored ``assignments.parquet``: one thesis on one seed run's maps."""

    temporal_run_id: str
    run_id: str
    seed: int
    doc_code: str
    period: Period
    label: int
    """On the earlier map: the clustering's label of an earlier thesis, -1 for noise, or the
    assigned topic of a later one."""
    full_label: int
    """T19's label on the full map, -1 for noise."""
    similarity: float | None
    runner_up: int | None
    runner_up_similarity: float | None
    margin: float | None
    threshold: float | None
    atypical: bool | None
    """For a later thesis, O06's flag; ``None`` for an earlier one."""
    provisional: bool


class SampleRow(_Row):
    """One row of the gitignored ``sample.parquet``: one thesis of the expert sample, in the
    order the experts see it."""

    position: int
    doc_code: str
    title: str
    program: str
    year: int
    issued_after_snapshot: bool
    pool: str
    temporal_run_id: str
    seed: int
    topic: int
    topic_size: int
    topic_terms: tuple[str, ...]
    similarity: float
    threshold: float
    atypical: bool
    runner_up: int | None
    runner_up_similarity: float | None
    margin: float | None
    full_label: int
    counterpart_agrees: bool
    """Whether the full-map label is the topic's counterpart under ``own_label``."""
    provisional: bool


class TemporalManifest(BaseModel):
    """Content of ``manifest.json``: provenance, values, counts and times, without text."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    snapshot_id: str
    git_commit: str
    experiment: dict[str, str]
    """SHA-256 of each T19 record read, ``manifest.json`` included, by file name."""
    inputs: dict[str, str]
    """SHA-256 of every other input, by file name."""
    winner: str
    representation: str
    algorithm: str
    parameter: int
    provisional: bool
    runs: tuple[str, ...]
    temporal_runs: tuple[str, ...]
    reference_run: str
    settings: dict[str, int | float | str]
    counts: dict[str, int]
    versions: dict[str, str]
    cpu: dict[str, Any]
    files: dict[str, str]
    """SHA-256 of each file written to the results directory, by name."""
    data_files: dict[str, str]
    """SHA-256 of each file written to the data directory, by name."""
    started_at: datetime
    finished_at: datetime
    seconds: float
    timings: dict[str, float]


def tracked_texts(
    *,
    runs: Sequence[TemporalRunRow],
    consistency: Sequence[ConsistencyRow],
    topics: Sequence[TemporalTopicRow],
    composition: Sequence[CompositionRow],
    summary: str,
) -> dict[str, str]:
    """Return the text of each tracked output but the manifest, by file name: a table as the
    JSON of its rows, one per line."""
    return {
        RUNS_FILE_NAME: _json_lines(runs),
        CONSISTENCY_FILE_NAME: _json_lines(consistency),
        TOPICS_FILE_NAME: _json_lines(topics),
        COMPOSITION_FILE_NAME: _json_lines(composition),
        SUMMARY_FILE_NAME: summary,
    }


def write_outputs(
    *,
    results: Path,
    processed: Path,
    runs: Sequence[TemporalRunRow],
    consistency: Sequence[ConsistencyRow],
    topics: Sequence[TemporalTopicRow],
    composition: Sequence[CompositionRow],
    assignments: Sequence[AssignmentRow],
    sample: Sequence[SampleRow],
    summary: str,
    manifest: TemporalManifest,
    guarded: frozenset[str],
    store: TableStore | None = None,
    write_text: Callable[[Path, str], None] = write_text_atomically,
) -> TemporalManifest:
    """Write the records once every tracked output passes the privacy guard: the gitignored
    assignments and sample first, then the tracked tables and ``summary.md``, then
    ``manifest.json`` last, and return the manifest with the SHA-256 of every file written.

    ``manifest`` is complete but for :attr:`~TemporalManifest.files` and
    :attr:`~TemporalManifest.data_files`, which this fills.

    Raises:
        PrivacyError: if a tracked output, the manifest included, holds a string of
            ``guarded`` (the snapshot's and the theses' doc codes); nothing is written then.
        OSError: if writing fails.
    """
    store = store or ParquetTableStore()
    texts = tracked_texts(
        runs=runs, consistency=consistency, topics=topics, composition=composition, summary=summary
    )
    check_privacy({**texts, MANIFEST_FILE_NAME: manifest.model_dump_json(indent=2)}, guarded)
    results.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    store.write(assignments, processed / ASSIGNMENTS_FILE_NAME, model_cls=AssignmentRow)
    store.write(sample, processed / SAMPLE_FILE_NAME, model_cls=SampleRow)
    store.write(runs, results / RUNS_FILE_NAME, model_cls=TemporalRunRow)
    store.write(consistency, results / CONSISTENCY_FILE_NAME, model_cls=ConsistencyRow)
    store.write(topics, results / TOPICS_FILE_NAME, model_cls=TemporalTopicRow)
    store.write(composition, results / COMPOSITION_FILE_NAME, model_cls=CompositionRow)
    write_text(results / SUMMARY_FILE_NAME, summary.rstrip("\n"))
    tracked = (
        RUNS_FILE_NAME,
        CONSISTENCY_FILE_NAME,
        TOPICS_FILE_NAME,
        COMPOSITION_FILE_NAME,
        SUMMARY_FILE_NAME,
    )
    final = manifest.model_copy(
        update={
            "files": {name: _sha256(results / name) for name in tracked},
            "data_files": {
                name: _sha256(processed / name)
                for name in (ASSIGNMENTS_FILE_NAME, SAMPLE_FILE_NAME)
            },
        }
    )
    text = final.model_dump_json(indent=2)
    check_privacy({MANIFEST_FILE_NAME: text}, guarded)
    # The commit point: without it, the temporal split is not complete.
    write_text(results / MANIFEST_FILE_NAME, text)
    return final


def existing_records(results: Path, processed: Path) -> Path | None:
    """Return a record of an earlier temporal split, the sample, the assignments or the
    manifest, if any."""
    for path in (
        processed / SAMPLE_FILE_NAME,
        processed / ASSIGNMENTS_FILE_NAME,
        results / MANIFEST_FILE_NAME,
    ):
        if path.exists():
            return path
    return None


def _json_lines(rows: Sequence[BaseModel]) -> str:
    return "\n".join(row.model_dump_json() for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
