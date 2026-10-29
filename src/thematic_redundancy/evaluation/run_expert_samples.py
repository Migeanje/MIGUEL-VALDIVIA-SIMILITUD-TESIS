"""Generate the anonymized result samples of the expert session (T31), or forecast them with
``--dry-run``::

    uv run python -m thematic_redundancy.evaluation.run_expert_samples [--project-root PATH]
        [--snapshot-id ID] [--dry-run]

The samples are those that ``docs/expert_session/muestras_anonimizadas.md`` specifies, drawn
and rendered by :mod:`.expert_samples`. The command reads, and never refits:

- T19's records of the snapshot with the gap run's reader
  (:func:`~thematic_redundancy.gaps.run_gaps.read_experiment`: ``manifest.json``,
  ``selection.json``, ``topics.parquet`` and the gitignored ``labels.parquet``, each checked
  against the SHA-256 that T19's manifest records), for sample C and the full map;
- T27's temporal split (:func:`read_temporal`): its ``manifest.json``, ``topics.parquet`` and
  the gitignored ``sample.parquet`` and ``assignments.parquet``, each checked against T27's
  manifest, which must name the same T19 records. Samples A and B are drawn from its expert
  sample, so the command refuses to go on without it;
- T24a's gap run (:func:`read_gap_records`): its ``manifest.json``, ``gaps.parquet`` and the
  gitignored ``members.parquet``, checked the same way, for sample D. Sample B's nearby gaps
  need T24a's gaps as objects, so they are computed again with the gap run's own functions on
  T19's records, and the reference run's rows and members must equal the recorded ones;
- T19's inputs through T19's loader (the winner's document vectors, which play the simulated
  proposals: no encoder runs), the fichas (texts, program and year) and the snapshot (the
  person names and ORCID iDs of the name guard, and the strings of the privacy guard).

A real run refuses a dirty git tree, since the manifest records the commit, and earlier samples
of the snapshot, since a draw is a one-time event; then it holds the run lock of
``data/expert_session/<snapshot_id>/muestras/``, inside the gitignored data tree, and writes
there, ``manifest.json`` last:

- ``muestra_a.md`` to ``muestra_d.md``: the printable samples, in Spanish;
- ``clave.csv``: the key that links each case code to its thesis, group or gap;
- ``composicion.json``: the counts per sample, program and rule, which pass the privacy guard
  (:func:`~thematic_redundancy.gaps.results_store.check_privacy`) as a tracked output would;
- ``manifest.json``: the snapshot, the git commit, the SHA-256 of every record and input read
  and of every output, the values used, the guard's counts, the library versions and the times.

Every sample passes the name guard (:func:`~.expert_samples.check_texts`) before anything is
written, in the dry run too; a hit stops the command. Nothing is written outside the samples'
directory. ``--dry-run`` computes and guards everything and writes nothing. Progress goes to
stderr, a summary without thesis text to stdout; the exit code is 0 on success, 1 after an
error, and 130 after an interruption.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from thematic_redundancy.corpus.build_fichas import load_fichas
from thematic_redundancy.corpus.ficha import Ficha
from thematic_redundancy.corpus.profile import sensitive_strings
from thematic_redundancy.corpus.snapshot import (
    METADATA_FILE_NAME,
    RAW_DIR_NAME,
    check_snapshot_id,
    latest_snapshot_id,
    read_snapshot,
)
from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL
from thematic_redundancy.evaluation.expert_samples import (
    GAP_RULES,
    SAMPLE_FILE_NAMES,
    AssignmentCase,
    Candidate,
    GapCase,
    QueryCase,
    QuerySpace,
    Replacement,
    SampleSettings,
    Thesis,
    TopicCase,
    check_texts,
    composition,
    draw_assignments,
    draw_gaps,
    draw_topics,
    key_csv,
    key_rows,
    person_strings,
    render_samples,
    simulate_queries,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.gaps.affinity import load_career_affinity
from thematic_redundancy.gaps.results_store import (
    GAPS_FILE_NAME,
    MEMBERS_FILE_NAME,
    GapManifest,
    GapRow,
    MemberRow,
    check_privacy,
    gap_rows,
    member_rows,
)
from thematic_redundancy.gaps.results_store import MANIFEST_FILE_NAME as GAPS_MANIFEST
from thematic_redundancy.gaps.results_store import processed_dir as gaps_processed_dir
from thematic_redundancy.gaps.results_store import results_dir as gaps_results_dir
from thematic_redundancy.gaps.rules import GapSettings, ThesisMetadata
from thematic_redundancy.gaps.run_gaps import (
    ExperimentRecords,
    build_maps,
    read_experiment,
    run_gap_rules,
    thesis_metadata,
)
from thematic_redundancy.recommendation.alternatives import NeighborReference, PairAnchor
from thematic_redundancy.redundancy.saturation import saturation_quartiles
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.filesystem import exclusive_run, write_text_atomically
from thematic_redundancy.shared.storage import ChecksumMismatchError, ParquetTableStore, TableStore
from thematic_redundancy.topic_map.build import TopicMap, topic_centroids
from thematic_redundancy.topic_map.ctfidf import Topic
from thematic_redundancy.topic_map.experiment import ExperimentInputs
from thematic_redundancy.topic_map.results_store import cpu_record, library_versions
from thematic_redundancy.topic_map.run_experiment import GitState, read_git_state
from thematic_redundancy.topic_map.run_experiment import load_inputs as load_experiment_inputs
from thematic_redundancy.topic_map.temporal_store import (
    ASSIGNMENTS_FILE_NAME,
    SAMPLE_FILE_NAME,
    AssignmentRow,
    SampleRow,
    TemporalManifest,
    TemporalTopicRow,
)
from thematic_redundancy.topic_map.temporal_store import MANIFEST_FILE_NAME as TEMPORAL_MANIFEST
from thematic_redundancy.topic_map.temporal_store import TOPICS_FILE_NAME as TEMPORAL_TOPICS
from thematic_redundancy.topic_map.temporal_store import processed_dir as temporal_processed_dir
from thematic_redundancy.topic_map.temporal_store import results_dir as temporal_results_dir

CONFIG_PATH = Path("config") / "default.yaml"
AFFINITY_PATH = Path("config") / "career_affinity.yaml"
RUN_COMMAND = "uv run python -m thematic_redundancy.evaluation.run_expert_samples"
TEMPORAL_COMMAND = "uv run python -m thematic_redundancy.topic_map.run_temporal"
GAPS_COMMAND = "uv run python -m thematic_redundancy.gaps.run_gaps"
EXPERT_SESSION_DIR_NAME = "expert_session"
SAMPLES_DIR_NAME = "muestras"
KEY_FILE_NAME = "clave.csv"
COMPOSITION_FILE_NAME = "composicion.json"
MANIFEST_FILE_NAME = "manifest.json"
OUTPUT_FILE_NAMES = (
    *SAMPLE_FILE_NAMES.values(),
    KEY_FILE_NAME,
    COMPOSITION_FILE_NAME,
    MANIFEST_FILE_NAME,
)
"""Every file a run writes, ``manifest.json`` last."""
NEIGHBOR_REFERENCE: NeighborReference = "proposal"
"""T26's recommended reference of the neighbor groups (T26 note 5)."""
PAIR_ANCHOR: PairAnchor = "midpoint"
"""T26's recommended anchor of a rule (b) gap (T26 note 6)."""
MAX_GAP_DISTANCE: float | None = None
"""T26's recommended limit on a nearby gap's distance: none (T26 note 7)."""
ANTECEDENT_POOL: int | None = None
"""T26's recommended candidate pool of the antecedents: every candidate (T26 note 1)."""


class SamplesRunError(ValueError):
    """The samples cannot be generated: an input is missing or disagrees with another."""


def samples_dir(data_root: Path, snapshot_id: str) -> Path:
    """Return ``<data_root>/expert_session/<snapshot_id>/muestras``."""
    return data_root / EXPERT_SESSION_DIR_NAME / snapshot_id / SAMPLES_DIR_NAME


# --- Inputs -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleInputs:
    """What the samples read besides the records of T19, T27 and T24a."""

    experiment: ExperimentInputs
    """T19's inputs through T19's loader: every model's document vectors, in one order."""
    theses: Mapping[str, Thesis]
    """Each thesis's texts, program and year, by doc code."""
    metadata: ThesisMetadata
    """Each thesis's program and year in the vectors' order, as the gap rules read them."""
    persons: frozenset[str]
    """The snapshot's person-name forms and ORCID iDs: the name guard."""
    sensitive: frozenset[str]
    """The snapshot's strings that no tracked output may hold (``corpus/profile.py``)."""
    identifiers: frozenset[str]
    """Every thesis's doc code, uuid and handle, which no sample may show."""
    input_sha256: dict[str, str]
    """The SHA-256 of the inputs read beside T19's input manifests, by file name."""


def thesis_texts(fichas: Sequence[Ficha], doc_codes: Sequence[str]) -> dict[str, Thesis]:
    """Return what the samples may show of each of ``doc_codes``, from ``fichas``.

    Raises:
        ValueError: if a doc code has no ficha or a ficha has no issue year, as the gap run's
            :func:`~thematic_redundancy.gaps.run_gaps.thesis_metadata` refuses them.
    """
    metadata = thesis_metadata(fichas, doc_codes)
    by_code = {ficha.doc_code: ficha for ficha in fichas}
    return {
        code: Thesis(
            doc_code=code,
            program=program,
            year=int(year),
            title=str(by_code[code].title),
            abstract=by_code[code].abstract,
            objectives=by_code[code].objectives,
            keywords=tuple(by_code[code].keywords),
        )
        for code, program, year in zip(doc_codes, metadata.programs, metadata.years, strict=True)
    }


def load_inputs(config: AppConfig, project_root: Path, snapshot_id: str) -> SampleInputs:
    """Load T19's inputs with T19's loader, the fichas and the snapshot's guards.

    Raises:
        FileNotFoundError: if an input is missing.
        ValueError: if an input is stale or damaged, or the inputs disagree on the theses.
    """
    data_dir = config.paths.resolve_against(project_root).data_dir
    experiment = load_experiment_inputs(config, project_root, snapshot_id)
    fichas = load_fichas(data_dir / INTERIM_DIR_NAME / snapshot_id)
    snapshot, records = read_snapshot(data_dir / RAW_DIR_NAME / snapshot_id)
    identifiers = set()
    for ficha in fichas:
        handle = str(ficha.handle_url)
        identifiers |= {ficha.doc_code, str(ficha.item_uuid), handle}
        if "/handle/" in handle:
            identifiers.add(handle.split("/handle/", 1)[1])
    return SampleInputs(
        experiment=experiment,
        theses=thesis_texts(fichas, experiment.doc_codes),
        metadata=thesis_metadata(fichas, experiment.doc_codes),
        persons=person_strings(records),
        sensitive=sensitive_strings(records),
        identifiers=frozenset(identifiers),
        input_sha256={METADATA_FILE_NAME: snapshot.metadata_sha256},
    )


@dataclass(frozen=True)
class TemporalRecords:
    """What the samples read from T27's records: its reference run's map and expert sample."""

    provisional: bool
    run_id: str
    """T19's medoid run, whose earlier map T27 drew the sample from."""
    temporal_run_id: str
    first_year: int
    split_year: int
    last_year: int
    sample: tuple[SampleRow, ...]
    """T27's expert sample, in presentation order."""
    labels: Mapping[str, int]
    """Each earlier thesis's label on the earlier map, -1 for noise, by doc code."""
    later: frozenset[str]
    """The later theses, assigned to the earlier map."""
    topics: tuple[TemporalTopicRow, ...]
    """The earlier map's groups, by ascending label."""
    sha256: Mapping[str, str]
    """The SHA-256 of each T27 record read, by file name."""

    @classmethod
    def from_rows(
        cls,
        *,
        provisional: bool,
        run_id: str,
        temporal_run_id: str,
        first_year: int,
        split_year: int,
        last_year: int,
        sample: Sequence[SampleRow],
        assignments: Sequence[AssignmentRow],
        topics: Sequence[TemporalTopicRow],
        sha256: Mapping[str, str],
    ) -> "TemporalRecords":
        """Keep the rows of ``temporal_run_id``, T27's reference run, whether read from T27's
        records or computed in memory by T27's own functions.

        Raises:
            SamplesRunError: if the sample is of another run or names a thesis that is not a
                later one, or the groups' sizes disagree with the earlier theses' labels.
        """
        if any(row.temporal_run_id != temporal_run_id for row in sample):
            raise SamplesRunError(f"T27's expert sample is not of its reference map {run_id}")
        labels: dict[str, int] = {}
        later: set[str] = set()
        for row in assignments:
            if row.temporal_run_id != temporal_run_id:
                continue
            if row.period == "fitted":
                labels[row.doc_code] = row.label
            else:
                later.add(row.doc_code)
        if any(row.doc_code not in later for row in sample):
            raise SamplesRunError("T27's expert sample holds a thesis that is not a later one")
        kept = tuple(
            sorted((row for row in topics if row.temporal_run_id == temporal_run_id), key=_topic)
        )
        sizes = Counter(label for label in labels.values() if label != NOISE_LABEL)
        if {row.topic: row.size for row in kept} != dict(sizes):
            raise SamplesRunError(
                f"T27's {TEMPORAL_TOPICS} and {ASSIGNMENTS_FILE_NAME} disagree on the groups "
                "of the earlier map or their sizes"
            )
        return cls(
            provisional=provisional,
            run_id=run_id,
            temporal_run_id=temporal_run_id,
            first_year=first_year,
            split_year=split_year,
            last_year=last_year,
            sample=tuple(sorted(sample, key=_position)),
            labels=labels,
            later=frozenset(later),
            topics=kept,
            sha256=dict(sha256),
        )


def read_temporal(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    records: ExperimentRecords,
    *,
    store: TableStore | None = None,
) -> TemporalRecords:
    """Read T27's records of ``snapshot_id``, each checked against T27's manifest.

    Raises:
        FileNotFoundError: if T27's temporal split, or a record of it, is missing.
        SamplesRunError: if T27 ran on other T19 records, or a record differs from the SHA-256
            in T27's manifest.
        ValueError: if a record is damaged.
    """
    store = store or ParquetTableStore()
    paths = config.paths.resolve_against(project_root)
    results = temporal_results_dir(paths.results_dir, snapshot_id)
    processed = temporal_processed_dir(paths.data_dir, snapshot_id)
    files = {
        TEMPORAL_MANIFEST: results / TEMPORAL_MANIFEST,
        TEMPORAL_TOPICS: results / TEMPORAL_TOPICS,
        SAMPLE_FILE_NAME: processed / SAMPLE_FILE_NAME,
        ASSIGNMENTS_FILE_NAME: processed / ASSIGNMENTS_FILE_NAME,
    }
    _require("T27's temporal split", snapshot_id, files, TEMPORAL_COMMAND, project_root)
    content = files[TEMPORAL_MANIFEST].read_bytes()
    manifest = TemporalManifest.model_validate_json(content)
    if manifest.snapshot_id != snapshot_id:
        raise SamplesRunError(
            f"T27's manifest is of snapshot {manifest.snapshot_id}, not {snapshot_id}"
        )
    if manifest.experiment != dict(records.sha256):
        raise SamplesRunError(
            "T27's temporal split ran on other T19 records than the current ones; run it again "
            "on these"
        )
    reference = records.run_ids[records.reference]
    if manifest.reference_run != reference or reference not in manifest.runs:
        raise SamplesRunError(f"T27's reference run is not T19's medoid run {reference}")
    temporal_run = manifest.temporal_runs[manifest.runs.index(reference)]
    sample = _verified(store, SampleRow, files[SAMPLE_FILE_NAME], manifest.data_files, "T27")
    assignments = _verified(
        store, AssignmentRow, files[ASSIGNMENTS_FILE_NAME], manifest.data_files, "T27"
    )
    topics = _verified(store, TemporalTopicRow, files[TEMPORAL_TOPICS], manifest.files, "T27")
    settings = manifest.settings
    return TemporalRecords.from_rows(
        provisional=manifest.provisional,
        run_id=reference,
        temporal_run_id=temporal_run,
        first_year=int(settings["first_year"]),
        split_year=int(settings["split_year"]),
        last_year=int(settings["last_year"]),
        sample=sample,
        assignments=assignments,
        topics=topics,
        sha256={
            TEMPORAL_MANIFEST: hashlib.sha256(content).hexdigest(),
            TEMPORAL_TOPICS: manifest.files[TEMPORAL_TOPICS],
            SAMPLE_FILE_NAME: manifest.data_files[SAMPLE_FILE_NAME],
            ASSIGNMENTS_FILE_NAME: manifest.data_files[ASSIGNMENTS_FILE_NAME],
        },
    )


@dataclass(frozen=True)
class GapRecords:
    """What the samples read from T24a's records: its reference run's gaps and members."""

    run_id: str
    rows: tuple[GapRow, ...]
    """The reference run's gaps, in the run's order."""
    members: tuple[MemberRow, ...]
    """The reference run's members, in the order written."""
    sha256: Mapping[str, str]
    """The SHA-256 of each T24a record read, by file name."""

    def theses(self) -> dict[str, tuple[str, ...]]:
        """Return each gap's theses, its topics' members without the bridges, by gap id."""
        found: dict[str, list[str]] = {}
        for row in self.members:
            if row.part != "bridge":
                found.setdefault(row.gap_id, []).append(row.doc_code)
        return {gap: tuple(codes) for gap, codes in found.items()}


def read_gap_records(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    records: ExperimentRecords,
    *,
    store: TableStore | None = None,
) -> GapRecords:
    """Read T24a's records of ``snapshot_id``, each checked against T24a's manifest.

    Raises:
        FileNotFoundError: if T24a's gap run, or a record of it, is missing.
        SamplesRunError: if T24a ran on other T19 records, or a record differs from the
            SHA-256 in T24a's manifest.
        ValueError: if a record is damaged.
    """
    store = store or ParquetTableStore()
    paths = config.paths.resolve_against(project_root)
    results = gaps_results_dir(paths.results_dir, snapshot_id)
    processed = gaps_processed_dir(paths.data_dir, snapshot_id)
    files = {
        GAPS_MANIFEST: results / GAPS_MANIFEST,
        GAPS_FILE_NAME: results / GAPS_FILE_NAME,
        MEMBERS_FILE_NAME: processed / MEMBERS_FILE_NAME,
    }
    _require("T24a's gap run", snapshot_id, files, GAPS_COMMAND, project_root)
    content = files[GAPS_MANIFEST].read_bytes()
    manifest = GapManifest.model_validate_json(content)
    if manifest.snapshot_id != snapshot_id:
        raise SamplesRunError(
            f"T24a's manifest is of snapshot {manifest.snapshot_id}, not {snapshot_id}"
        )
    if manifest.experiment != dict(records.sha256):
        raise SamplesRunError(
            "T24a's gap run ran on other T19 records than the current ones; run it again on these"
        )
    reference = records.run_ids[records.reference]
    if manifest.reference_run != reference:
        raise SamplesRunError(f"T24a's reference run is not T19's medoid run {reference}")
    rows = _verified(store, GapRow, files[GAPS_FILE_NAME], manifest.files, "T24a")
    members = _verified(store, MemberRow, files[MEMBERS_FILE_NAME], manifest.data_files, "T24a")
    return GapRecords(
        run_id=reference,
        rows=tuple(row for row in rows if row.run_id == reference),
        members=tuple(row for row in members if row.run_id == reference),
        sha256={
            GAPS_MANIFEST: hashlib.sha256(content).hexdigest(),
            GAPS_FILE_NAME: manifest.files[GAPS_FILE_NAME],
            MEMBERS_FILE_NAME: manifest.data_files[MEMBERS_FILE_NAME],
        },
    )


# --- The run ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleRun:
    """Everything a run computed, ready to describe and to write."""

    snapshot_id: str
    records: ExperimentRecords
    temporal: TemporalRecords
    gaps: GapRecords
    inputs: SampleInputs
    settings: SampleSettings
    assignments: tuple[AssignmentCase, ...]
    queries: tuple[QueryCase, ...]
    topics: tuple[TopicCase, ...]
    gap_cases: tuple[GapCase, ...]
    replacements: tuple[Replacement, ...]
    texts: dict[str, str]
    """The printable samples, by file name, past the name guard."""
    key: str
    """``clave.csv``."""
    composition: dict[str, Any]
    reference_topics: int
    """The groups of T19's reference map, noise left out."""
    stable_gaps: int
    input_sha256: dict[str, str]
    timings: dict[str, float]

    @property
    def provisional(self) -> bool:
        return self.records.manifest.provisional or self.temporal.provisional

    @property
    def guarded(self) -> frozenset[str]:
        """The strings no output that could be tracked may hold: the snapshot's and the
        theses' doc codes."""
        return self.inputs.sensitive | frozenset(self.inputs.experiment.doc_codes)

    def composition_text(self) -> str:
        return json.dumps(self.composition, indent=2, ensure_ascii=False)

    def guard_counts(self) -> dict[str, int]:
        return {
            "texts": len(self.texts),
            "persons": len(self.inputs.persons),
            "identifiers": len(self.inputs.identifiers | set(self.inputs.experiment.doc_codes)),
        }


def shown_map(
    vectors: np.ndarray, labels: np.ndarray, topics: Sequence[Topic], seed: int
) -> TopicMap:
    """Return a map of known ``labels`` and ``topics`` for T26's recommender, which reads its
    groups, terms and centroids (the mean of each group's ``vectors``, D30 E5). The samples
    draw no map, so its coordinates are zeros.

    Raises:
        SamplesRunError: if a group's size differs from its count in ``labels``.
    """
    for topic in topics:
        size = int(np.count_nonzero(labels == topic.topic))
        if size != topic.size:
            raise SamplesRunError(
                f"group {topic.topic} has {topic.size} theses in its record and {size} labels"
            )
    centroids = topic_centroids(vectors, labels, [topic.topic for topic in topics])
    return TopicMap(
        seed=seed,
        clustering=None,
        labels=labels,
        topics=tuple(topics),
        noise=None,
        centroids=centroids,
        coordinates=np.zeros((labels.size, 2)),
        reduction=None,
    )


def compute(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    *,
    load: Callable[[AppConfig, Path, str], SampleInputs],
    read_temporal: Callable[[AppConfig, Path, str, ExperimentRecords], TemporalRecords],
    report: Callable[[str], None],
    timer: Callable[[], float],
) -> SampleRun:
    """Read the records and inputs, check them against each other and draw the samples.

    Raises:
        FileNotFoundError: if a record or an input is missing, T27's split among them.
        OSError: if an input cannot be read.
        ValueError: if an input is damaged or disagrees with another, or a sample holds a
            person's name or an identifier (a :class:`SamplesRunError`, a
            :class:`~.expert_samples.ExpertSampleError` or a
            :class:`~thematic_redundancy.gaps.results_store.PrivacyError`).
        yaml.YAMLError: if the career-affinity file is not valid YAML.
    """
    timings: dict[str, float] = {}
    started = timer()
    report("reading the records of T19, T27 and T24a")
    records = read_experiment(config, project_root, snapshot_id)
    temporal = read_temporal(config, project_root, snapshot_id, records)
    gap_records = read_gap_records(config, project_root, snapshot_id, records)
    timings["records"] = timer() - started
    started = timer()
    report("loading the document vectors, the fichas and the snapshot")
    inputs = load(config, project_root, snapshot_id)
    timings["load"] = timer() - started
    experiment = inputs.experiment
    if experiment.input_sha256 != records.manifest.inputs:
        raise SamplesRunError("T19's inputs differ from the ones T19 ran on")
    codes = experiment.doc_codes
    if inputs.metadata.doc_codes != codes or set(inputs.theses) != set(codes):
        raise SamplesRunError("the fichas' theses do not follow the order of the document vectors")
    models = {model.model_id: model for model in experiment.models}
    representation = str(records.winner["representation"])
    if representation not in models:
        raise SamplesRunError(
            f"T19's inputs hold no document vectors of the winner {representation}"
        )
    vectors = np.asarray(models[representation].vectors, dtype=np.float64)
    settings = SampleSettings.from_config(config)
    reference, seed = records.reference, records.seeds[records.reference]
    reference_run = records.run_ids[reference]

    started = timer()
    report("computing T24a's gaps again on T19's records, for the nearby gaps")
    maps = build_maps(records, codes, vectors)
    affinity = load_career_affinity(project_root / AFFINITY_PATH, programs=list(settings.programs))
    _, gaps, supports, _ = run_gap_rules(
        maps,
        vectors,
        inputs.metadata,
        affinity,
        GapSettings.from_config(config),
        run_ids=records.run_ids,
        seeds=records.seeds,
        reference=reference,
    )
    recomputed = gap_rows(reference_run, seed, gaps[reference], inputs.metadata, supports=supports)
    if recomputed != gap_records.rows or member_rows(reference_run, gaps[reference]) != (
        gap_records.members
    ):
        raise SamplesRunError(
            "T24a's records of the reference run differ from what its rules give now on T19's "
            f"records and the configuration; run `{GAPS_COMMAND} --snapshot-id {snapshot_id}` "
            "again"
        )
    timings["gaps"] = timer() - started

    started = timer()
    report("drawing samples A to D")
    full_terms = {row.topic: row.terms for row in records.topics[reference_run]}
    full_map = TopicMap(
        seed=seed,
        clustering=None,
        labels=maps[reference].labels,
        topics=tuple(
            Topic(row.topic, row.size, row.terms, row.scores)
            for row in records.topics[reference_run]
        ),
        noise=None,
        centroids=maps[reference].centroids,
        coordinates=np.zeros((len(codes), 2)),
        reduction=None,
    )
    earlier = np.array([row for row, code in enumerate(codes) if code in temporal.labels])
    if set(temporal.labels) | temporal.later != set(codes) or len(earlier) != len(temporal.labels):
        raise SamplesRunError("T27's periods do not cover the theses of the document vectors")
    earlier_labels = np.array([temporal.labels[codes[row]] for row in earlier], dtype=np.int64)
    earlier_map = shown_map(
        vectors[earlier],
        earlier_labels,
        [Topic(row.topic, row.size, row.terms, row.scores) for row in temporal.topics],
        seed,
    )
    earlier_terms = {row.topic: row.terms for row in temporal.topics}
    members: dict[int, list[str]] = {}
    for row, label in zip(earlier, earlier_labels, strict=True):
        if label != NOISE_LABEL:
            members.setdefault(int(label), []).append(codes[row])
    theses = inputs.theses
    assignments = draw_assignments(
        [
            Candidate(
                row.doc_code,
                theses[row.doc_code].program,
                theses[row.doc_code].year,
                row.topic,
                row.atypical,
            )
            for row in temporal.sample
        ],
        members=members,
        topic_terms=earlier_terms,
        saturation=saturation_quartiles(
            earlier_labels,
            method=settings.saturation_method,
            boundary=settings.saturation_boundary,
        ),
        settings=settings,
    )
    stable = tuple(support for support in supports if support.stable)
    queries, replacements = simulate_queries(
        assignments,
        QuerySpace(
            doc_codes=tuple(codes),
            vectors=vectors,
            earlier=earlier,
            earlier_map=earlier_map,
            full_map=full_map,
            gaps=stable,
            full_terms=full_terms,
        ),
        theses=theses,
        settings=settings,
        neighbor_reference=NEIGHBOR_REFERENCE,
        pair_anchor=PAIR_ANCHOR,
        max_distance=MAX_GAP_DISTANCE,
        pool_size=ANTECEDENT_POOL,
    )
    topics = draw_topics(
        records.labels[reference_run], full_terms, theses=theses, settings=settings
    )
    gap_cases = draw_gaps(
        [row for row in gap_records.rows if row.stable],
        gap_records.theses(),
        topic_terms=full_terms,
        settings=settings,
    )
    texts = render_samples(
        assignments,
        queries,
        topics,
        gap_cases,
        theses=theses,
        settings=settings,
        earlier_years=(temporal.first_year, temporal.split_year - 1),
        later_years=(temporal.split_year, temporal.last_year),
        provisional=records.manifest.provisional or temporal.provisional,
    )
    timings["samples"] = timer() - started
    started = timer()
    report("applying the name guard")
    check_texts(
        texts,
        persons=inputs.persons,
        identifiers=inputs.identifiers | frozenset(codes),
    )
    timings["guard"] = timer() - started
    return SampleRun(
        snapshot_id=snapshot_id,
        records=records,
        temporal=temporal,
        gaps=gap_records,
        inputs=inputs,
        settings=settings,
        assignments=assignments,
        queries=queries,
        topics=topics,
        gap_cases=gap_cases,
        replacements=replacements,
        texts=texts,
        key=key_csv(key_rows(assignments, queries, topics, gap_cases, replacements)),
        composition=composition(
            assignments, queries, topics, gap_cases, replacements, programs=settings.programs
        ),
        reference_topics=len(records.topics[reference_run]),
        stable_gaps=len(stable),
        input_sha256={
            **experiment.input_sha256,
            **inputs.input_sha256,
            CONFIG_PATH.name: _sha256(project_root / CONFIG_PATH),
            AFFINITY_PATH.name: _sha256(project_root / AFFINITY_PATH),
        },
        timings={name: _seconds(value) for name, value in timings.items()},
    )


# --- Outputs ----------------------------------------------------------------------------------


class SamplesManifest(BaseModel):
    """Content of ``manifest.json``: provenance, values, counts and times, without text."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    snapshot_id: str
    git_commit: str
    provisional: bool
    """Whether T19's winner, and so T27's map, is still provisional: every sample says so."""
    winner: str
    representation: str
    reference_run: str
    """T19's medoid run, whose map gives samples C and D."""
    temporal_run: str
    """T27's earlier map of that run, which gives samples A and B."""
    experiment: dict[str, str]
    """SHA-256 of each T19 record read, by file name."""
    temporal: dict[str, str]
    """SHA-256 of each T27 record read, by file name."""
    gaps: dict[str, str]
    """SHA-256 of each T24a record read, by file name."""
    inputs: dict[str, str]
    """SHA-256 of every other input, by file name."""
    settings: dict[str, Any]
    composition: dict[str, Any]
    guard: dict[str, int]
    """The texts the name guard checked and the strings it searched."""
    versions: dict[str, str]
    cpu: dict[str, Any]
    files: dict[str, str]
    """SHA-256 of every other file written, by name."""
    started_at: datetime
    finished_at: datetime
    seconds: float
    timings: dict[str, float]


def write_outputs(directory: Path, run: SampleRun, manifest: SamplesManifest) -> SamplesManifest:
    """Write the key, the samples and the composition, then ``manifest.json`` last, once the
    composition and the manifest pass the privacy guard; return the manifest with the SHA-256
    of every other file written.

    Raises:
        PrivacyError: if the composition or the manifest holds a string of the snapshot or a
            doc code; nothing is written then.
        OSError: if writing fails.
    """
    composition_text = run.composition_text()
    check_privacy(
        {
            COMPOSITION_FILE_NAME: composition_text,
            MANIFEST_FILE_NAME: manifest.model_dump_json(indent=2),
        },
        run.guarded,
    )
    contents = {KEY_FILE_NAME: run.key, **run.texts, COMPOSITION_FILE_NAME: composition_text}
    for name, text in contents.items():
        write_text_atomically(directory / name, text)
    final = manifest.model_copy(
        update={"files": {name: _sha256(directory / name) for name in contents}}
    )
    text = final.model_dump_json(indent=2)
    check_privacy({MANIFEST_FILE_NAME: text}, run.guarded)
    # The commit point: without it, the samples are not complete.
    write_text_atomically(directory / MANIFEST_FILE_NAME, text)
    return final


def existing_samples(directory: Path) -> Path | None:
    """Return a file of earlier samples in ``directory``, if any."""
    for name in OUTPUT_FILE_NAMES:
        if (directory / name).exists():
            return directory / name
    return None


# --- Text -------------------------------------------------------------------------------------


def describe(run: SampleRun, *, seconds: float) -> str:
    """Describe a run in counts, without thesis text."""
    records, temporal, settings = run.records, run.temporal, run.settings
    reference = records.run_ids[records.reference]
    vectors = next(
        model.vectors
        for model in run.inputs.experiment.models
        if model.model_id == records.winner["representation"]
    )
    queries = Counter(case.program for case in run.queries)
    assignments = Counter(case.program for case in run.assignments)
    rules = Counter(case.rule for case in run.gap_cases)
    return "\n".join(
        [
            f"Winner {records.winner['configuration']}"
            + (" (provisional)" if run.provisional else "")
            + f"; reference run {reference} (seed {records.seeds[records.reference]}); T27's "
            f"map {temporal.temporal_run_id}",
            f"Inputs: {len(run.inputs.experiment.doc_codes)} theses; "
            f"{records.winner['representation']} ({vectors.shape[1]} dimensions); "
            f"{len(temporal.labels)} theses of {temporal.first_year} to "
            f"{temporal.split_year - 1} and {len(temporal.later)} of {temporal.split_year} to "
            f"{temporal.last_year}",
            *(f"  T19 {name} SHA-256 {digest}" for name, digest in records.sha256.items()),
            *(f"  T27 {name} SHA-256 {digest}" for name, digest in temporal.sha256.items()),
            *(f"  T24a {name} SHA-256 {digest}" for name, digest in run.gaps.sha256.items()),
            *(f"  {name} SHA-256 {digest}" for name, digest in run.input_sha256.items()),
            f"A: {len(run.assignments)} cases ({_by_program(assignments, settings)}) from "
            f"T27's expert sample of {len(temporal.sample)}",
            f"B: {len(run.queries)} cases, {len(run.replacements)} replaced for lack of an "
            f"abstract ({_by_program(queries, settings)}); rankings of {settings.ranking_size} "
            "theses, no redundancy verdict",
            f"C: {len(run.topics)} cases of the {run.reference_topics} groups of the reference map",
            f"D: {len(run.gap_cases)} cases ("
            + ", ".join(f"({rule}) {rules[rule]}" for rule in GAP_RULES)
            + f") of the {run.stable_gaps} stable gaps of the reference run",
            f"Name guard: {len(run.texts)} samples checked against "
            f"{len(run.inputs.persons):,} person-name forms and ORCID iDs and "
            f"{run.guard_counts()['identifiers']:,} identifiers, no hit",
            f"Seconds: {seconds:.1f}",
        ]
    )


def _by_program(counts: Counter[str], settings: SampleSettings) -> str:
    return ", ".join(f"{program} {counts[program]}" for program in settings.programs)


# --- The command ------------------------------------------------------------------------------


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.evaluation.run_expert_samples",
        description=(
            "Generate the anonymized result samples of the expert session (T31): A, T27's "
            "assignments of the later theses to the earlier map; B, simulated queries with "
            "their similarity rankings and recommendations, without a redundancy verdict; C, "
            "groups of the topic map; D, stable gaps. Writes the printable Spanish samples, "
            "their key, their composition and a manifest under "
            "data/expert_session/<snapshot>/muestras/ (gitignored), and nothing elsewhere. "
            "The samples are a one-time draw, so they are never overwritten."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument("--snapshot-id", help="snapshot to use (default: the harvested last)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print the samples' counts, apply the guards, and write nothing",
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[AppConfig, Path, str], SampleInputs] = load_inputs,
    read_temporal: Callable[
        [AppConfig, Path, str, ExperimentRecords], TemporalRecords
    ] = read_temporal,
    git: Callable[[Path, str], GitState] = read_git_state,
    clock: Callable[[], datetime] = _now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and 130
    after an interruption."""
    arguments = _parse_arguments(argv)
    root = arguments.project_root.resolve()
    try:
        config = load_config(root / CONFIG_PATH)
        paths = config.paths.resolve_against(root)
        snapshot_id = check_snapshot_id(
            arguments.snapshot_id or latest_snapshot_id(paths.data_dir / RAW_DIR_NAME)
        )
        directory = samples_dir(paths.data_dir, snapshot_id)
        relative = directory.relative_to(root).as_posix()
        if arguments.dry_run:
            return _dry_run(
                config,
                root,
                snapshot_id,
                directory,
                load=load,
                read_temporal=read_temporal,
                git=git,
                timer=timer,
            )
        state = git(root, relative)
        if state.dirty:
            raise SamplesRunError(
                f"the git tree has {len(state.dirty)} changed or untracked paths; commit them "
                "first, since the manifest records the commit"
            )
        _refuse_earlier_samples(snapshot_id, directory)
        directory.mkdir(parents=True, exist_ok=True)
        busy = SamplesRunError(f"another run holds the lock of {relative}; let it finish first")
        with exclusive_run(directory, busy):
            _refuse_earlier_samples(snapshot_id, directory)
            started_at, started = clock(), timer()
            run = compute(
                config,
                root,
                snapshot_id,
                load=load,
                read_temporal=read_temporal,
                report=_progress,
                timer=timer,
            )
            _progress(f"writing the samples, {MANIFEST_FILE_NAME} last")
            finished_at, seconds = clock(), timer() - started
            manifest = write_outputs(
                directory,
                run,
                _manifest(
                    run,
                    git_commit=state.commit,
                    started_at=started_at,
                    finished_at=finished_at,
                    seconds=seconds,
                ),
            )
    except KeyboardInterrupt:
        print(
            f"interrupted: the samples are complete only once {MANIFEST_FILE_NAME} is written; "
            "move the partial files away, if any, and run the command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(f"Snapshot {snapshot_id}: commit {manifest.git_commit}, seed {run.settings.seed}")
    print(describe(run, seconds=manifest.seconds))
    print(
        f"Wrote {relative}/: the samples, {KEY_FILE_NAME}, {COMPOSITION_FILE_NAME}, "
        f"{MANIFEST_FILE_NAME} last"
    )
    return 0


def _refuse_earlier_samples(snapshot_id: str, directory: Path) -> None:
    earlier = existing_samples(directory)
    if earlier is not None:
        raise FileExistsError(
            f"samples of snapshot {snapshot_id} already exist ({earlier.name}); they are a "
            "one-time draw, so they are never overwritten: move them away first to draw them "
            "again"
        )


def _dry_run(
    config: AppConfig,
    root: Path,
    snapshot_id: str,
    directory: Path,
    *,
    load: Callable[[AppConfig, Path, str], SampleInputs],
    read_temporal: Callable[[AppConfig, Path, str, ExperimentRecords], TemporalRecords],
    git: Callable[[Path, str], GitState],
    timer: Callable[[], float],
) -> int:
    relative = directory.relative_to(root).as_posix()
    try:
        state: GitState | None = git(root, relative)
    except OSError:
        state = None
    started = timer()
    run = compute(
        config,
        root,
        snapshot_id,
        load=load,
        read_temporal=read_temporal,
        report=_progress,
        timer=timer,
    )
    check_privacy({COMPOSITION_FILE_NAME: run.composition_text()}, run.guarded)
    seconds = timer() - started
    if state is None:
        tree = "not readable here; a real run needs a repository with a commit"
    elif not state.dirty:
        tree = f"commit {state.commit}, clean"
    else:
        count = len(state.dirty)
        tree = (
            f"commit {state.commit}, {count} changed path{'s' if count > 1 else ''}; a real run "
            "refuses them until they are committed"
        )
    earlier = existing_samples(directory)
    print(
        "\n".join(
            [
                f"Dry run of the expert samples on snapshot {snapshot_id}: everything is "
                "computed and guarded, and nothing is written",
                f"Git: {tree}",
                *(
                    []
                    if earlier is None
                    else [
                        f"Samples of this snapshot exist ({earlier.name}); a real run refuses "
                        "to overwrite them"
                    ]
                ),
                describe(run, seconds=seconds),
                f"Privacy guard: {COMPOSITION_FILE_NAME} checked against {len(run.guarded):,} "
                "strings (the snapshot's, and the theses' doc codes), no hit (the manifest is "
                "checked when it is written)",
                f"Command of the real run: {RUN_COMMAND} --snapshot-id {snapshot_id}",
            ]
        )
    )
    return 0


def _manifest(
    run: SampleRun,
    *,
    git_commit: str,
    started_at: datetime,
    finished_at: datetime,
    seconds: float,
) -> SamplesManifest:
    records = run.records
    return SamplesManifest(
        snapshot_id=run.snapshot_id,
        git_commit=git_commit,
        provisional=run.provisional,
        winner=str(records.winner["configuration"]),
        representation=str(records.winner["representation"]),
        reference_run=records.run_ids[records.reference],
        temporal_run=run.temporal.temporal_run_id,
        experiment=dict(records.sha256),
        temporal=dict(run.temporal.sha256),
        gaps=dict(run.gaps.sha256),
        inputs=run.input_sha256,
        settings={
            **run.settings.record(),
            "neighbor_reference": NEIGHBOR_REFERENCE,
            "pair_anchor": PAIR_ANCHOR,
            "max_distance": MAX_GAP_DISTANCE,
            "antecedent_pool_size": ANTECEDENT_POOL,
        },
        composition=run.composition,
        guard=run.guard_counts(),
        versions=library_versions(),
        cpu=cpu_record(),
        files={},
        started_at=started_at,
        finished_at=finished_at,
        seconds=_seconds(seconds),
        timings=run.timings,
    )


# --- Helpers ----------------------------------------------------------------------------------


def _require(
    what: str, snapshot_id: str, files: Mapping[str, Path], command: str, root: Path
) -> None:
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        listed = ", ".join(_shown(path, root) for path in missing)
        raise FileNotFoundError(
            f"{what} of snapshot {snapshot_id} is missing ({listed}), and the samples are drawn "
            f"from it: run `{command} --snapshot-id {snapshot_id}` first"
        )


def _shown(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _verified[M: BaseModel](
    store: TableStore, model_cls: type[M], path: Path, recorded: Mapping[str, str], owner: str
) -> tuple[M, ...]:
    expected = recorded.get(path.name)
    if expected is None:
        raise SamplesRunError(f"{owner}'s manifest records no SHA-256 for {path.name}")
    try:
        return store.read_verified(model_cls, path, sha256=expected)
    except ChecksumMismatchError as error:
        raise SamplesRunError(
            f"{path.name} differs from the SHA-256 that {owner}'s manifest records, so it is not "
            "used"
        ) from error


def _topic(row: TemporalTopicRow) -> int:
    return row.topic


def _position(row: SampleRow) -> int:
    return row.position


def _seconds(value: float) -> float:
    return round(max(value, 0.0), 3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
