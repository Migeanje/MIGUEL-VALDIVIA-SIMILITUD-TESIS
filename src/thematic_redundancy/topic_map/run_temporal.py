"""Evaluate the topic map on a temporal split (T27, O04), or compute it with ``--dry-run``::

    uv run python -m thematic_redundancy.topic_map.run_temporal [--project-root PATH]
        [--snapshot-id ID] [--dry-run]

The command reads T19's records of the snapshot with the gap run's reader
(:func:`~thematic_redundancy.gaps.run_gaps.read_experiment`: ``manifest.json``,
``selection.json``, ``topics.parquet`` and the gitignored ``labels.parquet``, each checked
against the SHA-256 that T19's manifest records, under the configured experiment declaration);
T19's inputs through T19's loader, whose input manifests must be those T19 ran on; the fichas,
for each thesis's program, issue year, title and future-date flag; and the snapshot, for the
privacy guard.

For each of the winner's seed runs it fits the winner's configuration, with no new sweep, on
the theses issued from ``snapshot.year_start`` to ``temporal.split_year - 1``
(:func:`~.temporal.fit_temporal_map`), assigns the theses issued from ``temporal.split_year``
to ``snapshot.year_end`` to its topics with O06's flag as ``proposal`` declares it
(:func:`~.temporal.assign_period`), and compares both periods with T19's full-map labels of the
same seed (:mod:`~thematic_redundancy.evaluation.temporal`), under D30 E4's two ARI noise
handlings. The expert sample is drawn from the map of T19's medoid run with
``temporal.sample`` (:func:`~.temporal.draw_expert_sample`).

``--dry-run`` computes everything, prints it and passes the tracked outputs through the privacy
guard, writing nothing. The expert sample is a one-time draw, so a real run refuses an earlier
temporal split of the snapshot (its manifest, its assignments or its sample) and a dirty git
tree (changes outside ``results/temporal/<snapshot>/``), since the manifest records the commit;
then it holds the run lock of that directory and writes the records of :mod:`.temporal_store`,
``manifest.json`` last. While T19's winner is provisional, every output says so. Progress goes
to stderr, a summary without thesis text to stdout; the exit code is 0 on success, 1 after an
error, and 130 after an interruption.
"""

import argparse
import functools
import hashlib
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
from thematic_redundancy.evaluation.stability import NOISE_HANDLINGS, NoiseHandling
from thematic_redundancy.evaluation.temporal import (
    CounterpartAgreement,
    PartitionAgreement,
    counterpart_agreement,
    partition_agreement,
    topic_counterparts,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.gaps.results_store import check_privacy
from thematic_redundancy.gaps.run_gaps import ExperimentRecords, read_experiment, thesis_metadata
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.topic_map.build import TopicMapPorts, TopicMapSettings
from thematic_redundancy.topic_map.clustering import ClusteringRun, HdbscanRun, KMeansRun
from thematic_redundancy.topic_map.experiment import ExperimentInputs
from thematic_redundancy.topic_map.reduction import dense_vectors
from thematic_redundancy.topic_map.results_store import cpu_record, library_versions
from thematic_redundancy.topic_map.run_experiment import GitState, read_git_state
from thematic_redundancy.topic_map.run_experiment import load_inputs as load_experiment_inputs
from thematic_redundancy.topic_map.temporal import (
    ATYPICAL_POOL,
    ExpertSample,
    PeriodAssignment,
    Periods,
    TemporalMap,
    TemporalSettings,
    assign_period,
    draw_expert_sample,
    fit_temporal_map,
    period_vocabulary,
    split_periods,
)
from thematic_redundancy.topic_map.temporal_store import (
    MANIFEST_FILE_NAME,
    AssignmentRow,
    CompositionRow,
    ConsistencyRow,
    SampleRow,
    TemporalManifest,
    TemporalRunRow,
    TemporalTopicRow,
    existing_records,
    processed_dir,
    results_dir,
    tracked_texts,
    write_outputs,
)

CONFIG_PATH = Path("config") / "default.yaml"
RUN_COMMAND = "uv run python -m thematic_redundancy.topic_map.run_temporal"


class TemporalRunError(ValueError):
    """The temporal split cannot go on: an input disagrees with T19's records or another input."""


# --- Inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Theses:
    """Each thesis's program, issue year, title and future-date flag, in the vectors' order."""

    doc_codes: tuple[str, ...]
    programs: tuple[str, ...]
    years: tuple[int, ...]
    titles: tuple[str, ...]
    after_snapshot: tuple[bool, ...]
    """Whether the thesis's issue date lies after the snapshot date (T06)."""


@dataclass(frozen=True)
class TemporalInputs:
    """What the temporal split reads besides T19's records and the configuration."""

    experiment: ExperimentInputs
    """T19's inputs through T19's loader: every model's document vectors, the full-clean
    section tokens and T15's vocabulary, in one order."""
    theses: Theses
    sensitive: frozenset[str]
    """The snapshot's strings that no tracked output may hold (``corpus/profile.py``)."""
    input_sha256: dict[str, str]
    """The SHA-256 of the inputs read beside T19's input manifests, by file name."""


def thesis_records(fichas: Sequence[Ficha], doc_codes: Sequence[str]) -> Theses:
    """Return the program, year, title and future-date flag of each of ``doc_codes``, in that
    order, from ``fichas``.

    Raises:
        ValueError: if a doc code has no ficha or a ficha has no issue year, as the gap run's
            :func:`~thematic_redundancy.gaps.run_gaps.thesis_metadata` refuses them.
    """
    metadata = thesis_metadata(fichas, doc_codes)
    by_code = {ficha.doc_code: ficha for ficha in fichas}
    chosen = [by_code[code] for code in doc_codes]
    return Theses(
        doc_codes=tuple(doc_codes),
        programs=metadata.programs,
        years=tuple(int(year) for year in metadata.years),
        titles=tuple(str(ficha.title) for ficha in chosen),
        after_snapshot=tuple(bool(ficha.issued_after_snapshot) for ficha in chosen),
    )


def load_inputs(config: AppConfig, project_root: Path, snapshot_id: str) -> TemporalInputs:
    """Load T19's inputs with T19's loader, the fichas and the snapshot's sensitive strings.

    Raises:
        FileNotFoundError: if an input is missing.
        ValueError: if an input is stale or damaged, or the inputs disagree on the theses.
    """
    data_dir = config.paths.resolve_against(project_root).data_dir
    experiment = load_experiment_inputs(config, project_root, snapshot_id)
    fichas = load_fichas(data_dir / INTERIM_DIR_NAME / snapshot_id)
    snapshot, records = read_snapshot(data_dir / RAW_DIR_NAME / snapshot_id)
    return TemporalInputs(
        experiment=experiment,
        theses=thesis_records(fichas, experiment.doc_codes),
        sensitive=sensitive_strings(records),
        input_sha256={METADATA_FILE_NAME: snapshot.metadata_sha256},
    )


def winner_clustering(winner: Mapping[str, Any]) -> ClusteringRun:
    """Return the clustering of ``selection.json``'s winner: its algorithm and parameter.

    Raises:
        TemporalRunError: if the algorithm is neither K-means nor HDBSCAN.
    """
    algorithm, parameter = str(winner.get("algorithm")), int(winner["parameter"])
    if algorithm == "hdbscan":
        return HdbscanRun(min_cluster_size=parameter)
    if algorithm == "kmeans":
        return KMeansRun(n_clusters=parameter)
    raise TemporalRunError(f"the winner's algorithm {algorithm} is neither kmeans nor hdbscan")


# --- The run --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    """One seed run: its earlier map, the later theses' assignment and the comparisons."""

    run_id: str
    temporal_run_id: str
    seed: int
    reference: bool
    temporal_map: TemporalMap
    assignment: PeriodAssignment
    full_labels: np.ndarray
    """T19's full-map label of every thesis, in the vectors' order."""
    counterparts: dict[NoiseHandling, dict[int, int | None]]
    assigned: dict[NoiseHandling, PartitionAgreement]
    agreement: dict[NoiseHandling, CounterpartAgreement]
    fitted: dict[NoiseHandling, PartitionAgreement]
    assign_seconds: float


@dataclass(frozen=True)
class TemporalRun:
    """Everything a temporal split computed, ready to describe and to write."""

    snapshot_id: str
    records: ExperimentRecords
    inputs: TemporalInputs
    settings: TemporalSettings
    periods: Periods
    vocabulary: tuple[str, ...]
    results: tuple[SeedResult, ...]
    input_sha256: dict[str, str]
    timings: dict[str, float]

    @property
    def provisional(self) -> bool:
        return self.records.manifest.provisional

    @property
    def reference(self) -> SeedResult:
        """T19's medoid run (D30 E3), whose map is shown and gives the expert sample."""
        return self.results[self.records.reference]

    @functools.cached_property
    def sample(self) -> ExpertSample:
        """The expert sample, drawn from the later theses on the reference run's map."""
        settings = self.settings
        return draw_expert_sample(
            [self.inputs.theses.programs[row] for row in self.periods.assigned.tolist()],
            self.reference.assignment.atypical,
            program_order=settings.programs,
            size=settings.sample_size,
            atypical_quota=settings.sample_atypical,
            program_floor=settings.sample_program_floor,
            seed=settings.sample_seed,
        )

    @property
    def guarded(self) -> frozenset[str]:
        """The strings no tracked output may hold: the snapshot's and the theses' doc codes."""
        return self.inputs.sensitive | frozenset(self.inputs.theses.doc_codes)

    def counts(self) -> dict[str, int]:
        theses, periods = self.inputs.theses, self.periods
        assigned = periods.assigned.tolist()
        counts = {
            "theses": len(theses.doc_codes),
            "fitted_theses": int(periods.fitted.size),
            "assigned_theses": int(periods.assigned.size),
            "assigned_issued_after_snapshot": sum(theses.after_snapshot[row] for row in assigned),
            "t15_vocabulary_terms": len(self.inputs.experiment.vocabulary),
            "vocabulary_terms": len(self.vocabulary),
            "sample": len(self.sample.rows),
            "sample_atypical": sum(pool == ATYPICAL_POOL for pool in self.sample.pools),
        }
        for name, rows in (("fitted", periods.fitted), ("assigned", periods.assigned)):
            by_program = Counter(theses.programs[row] for row in rows.tolist())
            counts |= {f"{name}_{key}": by_program[key] for key in self.settings.programs}
        by_year = Counter(theses.years)
        counts |= {f"year_{year}": by_year[year] for year in sorted(by_year)}
        return counts

    def run_rows(self) -> tuple[TemporalRunRow, ...]:
        fitted, assigned = self.periods.fitted.size, self.periods.assigned.size
        rows = []
        for result in self.results:
            temporal_map, assignment = result.temporal_map, result.assignment
            full = result.full_labels
            seconds = temporal_map.seconds
            rows.append(
                TemporalRunRow(
                    run_id=result.run_id,
                    temporal_run_id=result.temporal_run_id,
                    seed=result.seed,
                    reference=result.reference,
                    provisional=self.provisional,
                    fitted_theses=int(fitted),
                    assigned_theses=int(assigned),
                    topics=len(temporal_map.topics),
                    noise_theses=temporal_map.noise_points,
                    noise_share=temporal_map.noise_points / fitted,
                    vocabulary_terms=temporal_map.vocabulary,
                    full_map_topics=len(set(full.tolist()) - {NOISE_LABEL}),
                    full_map_noise_assigned=int(
                        np.count_nonzero(full[self.periods.assigned] == NOISE_LABEL)
                    ),
                    atypical_threshold=(
                        temporal_map.threshold.values[0]
                        if temporal_map.threshold.scope == "global"
                        else None
                    ),
                    atypical_theses=assignment.atypical_count,
                    atypical_share=assignment.atypical_count / assigned,
                    umap_seconds=_seconds(seconds["umap"]),
                    cluster_seconds=_seconds(seconds["cluster"]),
                    ctfidf_seconds=_seconds(seconds["ctfidf"]),
                    threshold_seconds=_seconds(seconds["threshold"]),
                    assign_seconds=_seconds(result.assign_seconds),
                )
            )
        return tuple(rows)

    def consistency_rows(self) -> tuple[ConsistencyRow, ...]:
        rows = []
        for result in self.results:
            common = {
                "run_id": result.run_id,
                "temporal_run_id": result.temporal_run_id,
                "seed": result.seed,
                "reference": result.reference,
                "provisional": self.provisional,
            }
            for handling in self.settings.noise_handlings:
                partition, agreement = result.assigned[handling], result.agreement[handling]
                rows.append(
                    ConsistencyRow(
                        **common,
                        period="assigned",
                        noise_handling=handling,
                        points=partition.points,
                        ari=partition.ari,
                        nmi=partition.nmi,
                        agreement=agreement.share,
                        agreeing=agreement.agreeing,
                        undefined_reason=partition.undefined_reason or agreement.undefined_reason,
                    )
                )
            for handling in self.settings.noise_handlings:
                partition = result.fitted[handling]
                rows.append(
                    ConsistencyRow(
                        **common,
                        period="fitted",
                        noise_handling=handling,
                        points=partition.points,
                        ari=partition.ari,
                        nmi=partition.nmi,
                        agreement=None,
                        agreeing=None,
                        undefined_reason=partition.undefined_reason,
                    )
                )
        return tuple(rows)

    def topic_rows(self) -> tuple[TemporalTopicRow, ...]:
        rows = []
        for result in self.results:
            labels, flags = result.assignment.labels, result.assignment.atypical
            for topic in result.temporal_map.topics:
                members = labels == topic.topic
                rows.append(
                    TemporalTopicRow(
                        temporal_run_id=result.temporal_run_id,
                        seed=result.seed,
                        reference=result.reference,
                        provisional=self.provisional,
                        topic=topic.topic,
                        size=topic.size,
                        terms=topic.terms,
                        scores=topic.scores,
                        atypical_threshold=result.temporal_map.threshold.value_for(topic.topic),
                        counterpart_own_label=result.counterparts["own_label"][topic.topic],
                        counterpart_exclude_either=result.counterparts["exclude_either"][
                            topic.topic
                        ],
                        assigned=int(np.count_nonzero(members)),
                        atypical=int(np.count_nonzero(members & flags)),
                    )
                )
        return tuple(rows)

    def composition_rows(self) -> tuple[CompositionRow, ...]:
        return tuple(
            CompositionRow(
                pool=stratum.pool,
                program=stratum.program,
                candidates=stratum.candidates,
                quota=stratum.quota,
                provisional=self.provisional,
            )
            for stratum in self.sample.strata
        )

    def assignment_rows(self) -> tuple[AssignmentRow, ...]:
        codes, periods = self.inputs.theses.doc_codes, self.periods
        fitted = {row: index for index, row in enumerate(periods.fitted.tolist())}
        assigned = {row: index for index, row in enumerate(periods.assigned.tolist())}
        rows = []
        for result in self.results:
            common = {
                "temporal_run_id": result.temporal_run_id,
                "run_id": result.run_id,
                "seed": result.seed,
                "provisional": self.provisional,
            }
            for row, code in enumerate(codes):
                full = int(result.full_labels[row])
                if row in fitted:
                    rows.append(
                        AssignmentRow(
                            **common,
                            doc_code=code,
                            period="fitted",
                            label=int(result.temporal_map.labels[fitted[row]]),
                            full_label=full,
                            similarity=None,
                            runner_up=None,
                            runner_up_similarity=None,
                            margin=None,
                            threshold=None,
                            atypical=None,
                        )
                    )
                    continue
                item = result.assignment.assignments[assigned[row]]
                check = result.assignment.checks[assigned[row]]
                rows.append(
                    AssignmentRow(
                        **common,
                        doc_code=code,
                        period="assigned",
                        label=item.topic,
                        full_label=full,
                        similarity=item.similarity,
                        runner_up=item.runner_up,
                        runner_up_similarity=item.runner_up_similarity,
                        margin=item.margin,
                        threshold=check.threshold,
                        atypical=check.atypical,
                    )
                )
        return tuple(rows)

    def sample_rows(self) -> tuple[SampleRow, ...]:
        theses, result = self.inputs.theses, self.reference
        topics = {topic.topic: topic for topic in result.temporal_map.topics}
        counterparts = result.counterparts[self.settings.noise_handling]
        rows = []
        for index, pool, position in zip(
            self.sample.rows, self.sample.pools, self.sample.positions, strict=True
        ):
            row = int(self.periods.assigned[index])
            item = result.assignment.assignments[index]
            check = result.assignment.checks[index]
            full = int(result.full_labels[row])
            rows.append(
                SampleRow(
                    position=position,
                    doc_code=theses.doc_codes[row],
                    title=theses.titles[row],
                    program=theses.programs[row],
                    year=theses.years[row],
                    issued_after_snapshot=theses.after_snapshot[row],
                    pool=pool,
                    temporal_run_id=result.temporal_run_id,
                    seed=result.seed,
                    topic=item.topic,
                    topic_size=topics[item.topic].size,
                    topic_terms=topics[item.topic].terms,
                    similarity=item.similarity,
                    threshold=check.threshold,
                    atypical=check.atypical,
                    runner_up=item.runner_up,
                    runner_up_similarity=item.runner_up_similarity,
                    margin=item.margin,
                    full_label=full,
                    counterpart_agrees=counterparts.get(item.topic) == full,
                    provisional=self.provisional,
                )
            )
        return tuple(sorted(rows, key=_position))


def full_labels(records: ExperimentRecords, run: str, doc_codes: Sequence[str]) -> np.ndarray:
    """Return T19's labels of ``run`` in the order of ``doc_codes``.

    Raises:
        TemporalRunError: if the labels do not cover exactly those theses.
    """
    by_code = records.labels[run]
    if len(by_code) != len(doc_codes) or any(code not in by_code for code in doc_codes):
        raise TemporalRunError(
            f"the labels of run {run} do not cover the {len(doc_codes)} theses of the document "
            "vectors"
        )
    return np.array([by_code[code] for code in doc_codes], dtype=np.int64)


def compute(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    *,
    load: Callable[[AppConfig, Path, str], TemporalInputs],
    ports: TopicMapPorts | None,
    report: Callable[[str], None],
    timer: Callable[[], float],
) -> TemporalRun:
    """Read the inputs, check them against T19's records and compute the temporal split.

    Raises:
        OSError: if an input cannot be read.
        ValueError: if an input is damaged or disagrees with T19's records, or a step refuses
            its input (a :class:`TemporalRunError` or a
            :class:`~thematic_redundancy.topic_map.TopicMapError`).
    """
    started = timer()
    report("reading T19's records")
    records = read_experiment(config, project_root, snapshot_id)
    report("loading the document vectors, the fichas and the snapshot")
    inputs = load(config, project_root, snapshot_id)
    timings = {"load": timer() - started}
    experiment = inputs.experiment
    if experiment.input_sha256 != records.manifest.inputs:
        changed = sorted(
            name
            for name in {*experiment.input_sha256, *records.manifest.inputs}
            if experiment.input_sha256.get(name) != records.manifest.inputs.get(name)
        )
        raise TemporalRunError(f"the inputs {changed} differ from the ones T19 ran on")
    if inputs.theses.doc_codes != experiment.doc_codes:
        raise TemporalRunError("the fichas' theses do not follow the order of the document vectors")
    models = {model.model_id: model for model in experiment.models}
    representation = str(records.winner["representation"])
    if representation not in models:
        raise TemporalRunError(
            f"T19's inputs hold no document vectors of the winner {representation}"
        )
    vectors = dense_vectors(models[representation].vectors)
    clustering = winner_clustering(records.winner)
    settings = TemporalSettings.from_config(config)
    topic_settings = TopicMapSettings.from_config(config)
    periods = split_periods(
        inputs.theses.years,
        first_year=settings.first_year,
        split_year=settings.split_year,
        last_year=settings.last_year,
    )
    documents = [experiment.documents[row] for row in periods.fitted.tolist()]
    vocabulary = period_vocabulary(
        experiment.vocabulary, documents, max_ngram=topic_settings.max_ngram
    )
    ports = ports or TopicMapPorts()
    first, last = periods.fitted_years
    results = []
    for index, (run, seed) in enumerate(zip(records.run_ids, records.seeds, strict=True)):
        report(
            f"seed {seed}: the map of the {periods.fitted.size} theses of {first} to {last}, "
            f"then the {periods.assigned.size} later theses"
        )
        full = full_labels(records, run, experiment.doc_codes)
        temporal_map = fit_temporal_map(
            vectors[periods.fitted],
            documents,
            vocabulary=vocabulary,
            clustering=clustering,
            seed=seed,
            settings=topic_settings,
            ports=ports,
            percentile=settings.percentile,
            method=settings.method,
            scope=settings.scope,
            timer=timer,
        )
        started = timer()
        assignment = assign_period(vectors[periods.assigned], temporal_map)
        assign_seconds = timer() - started
        results.append(
            _seed_result(
                run,
                seed,
                index == records.reference,
                temporal_map,
                assignment,
                full,
                periods,
                f"{run}__fit{first}-{last}",
                assign_seconds,
            )
        )
        for step, value in temporal_map.seconds.items():
            timings[f"seed{seed}_{step}"] = value
        timings[f"seed{seed}_assign"] = assign_seconds
    return TemporalRun(
        snapshot_id=snapshot_id,
        records=records,
        inputs=inputs,
        settings=settings,
        periods=periods,
        vocabulary=vocabulary,
        results=tuple(results),
        input_sha256={
            **experiment.input_sha256,
            **inputs.input_sha256,
            CONFIG_PATH.name: _sha256(project_root / CONFIG_PATH),
        },
        timings={name: _seconds(value) for name, value in timings.items()},
    )


def _seed_result(
    run: str,
    seed: int,
    reference: bool,
    temporal_map: TemporalMap,
    assignment: PeriodAssignment,
    full: np.ndarray,
    periods: Periods,
    temporal_run_id: str,
    assign_seconds: float,
) -> SeedResult:
    fitted, assigned = full[periods.fitted], full[periods.assigned]
    counterparts = {
        handling: topic_counterparts(temporal_map.labels, fitted, noise_handling=handling)
        for handling in NOISE_HANDLINGS
    }
    return SeedResult(
        run_id=run,
        temporal_run_id=temporal_run_id,
        seed=seed,
        reference=reference,
        temporal_map=temporal_map,
        assignment=assignment,
        full_labels=full,
        counterparts=counterparts,
        assigned={
            handling: partition_agreement(assignment.labels, assigned, noise_handling=handling)
            for handling in NOISE_HANDLINGS
        },
        agreement={
            handling: counterpart_agreement(
                assignment.labels, assigned, counterparts[handling], noise_handling=handling
            )
            for handling in NOISE_HANDLINGS
        },
        fitted={
            handling: partition_agreement(temporal_map.labels, fitted, noise_handling=handling)
            for handling in NOISE_HANDLINGS
        },
        assign_seconds=assign_seconds,
    )


# --- Text -----------------------------------------------------------------------------------


def render_summary(run: TemporalRun, *, git_commit: str | None) -> str:
    """Return ``summary.md``: the periods, the earlier maps, the consistency of the later
    theses per seed and over the seeds, the earlier theses on both maps, and the expert
    sample's composition. It holds no doc code and no thesis text."""
    records, settings, periods = run.records, run.settings, run.periods
    winner = records.winner
    first, last = periods.fitted_years
    split, end = periods.assigned_years
    reference = run.reference
    lines = [
        f"# Temporal split of snapshot {run.snapshot_id}",
        "",
        f"T19's winner {winner['configuration']} ({winner['representation']}, "
        f"{winner['algorithm']} {winner['parameter']}) fitted with no new sweep (D30 E3) on the "
        f"{periods.fitted.size} theses of {first} to {last}, once per seed run of T19, at commit "
        f"`{git_commit or 'unknown'}`. The {periods.assigned.size} theses of {split} to {end} are "
        "assigned to its topics by nearest centroid (D14) and flagged atypical below O06's "
        f"P{settings.percentile:g} of the member-to-own-centroid similarities ({settings.scope}, "
        f"{settings.method}). Each seed's assignment is compared with T19's full-map labels of "
        f"the same seed. The reference run is `{reference.run_id}` (seed {reference.seed}), "
        "T19's medoid run, whose map gives the expert sample.",
    ]
    if run.provisional:
        lines += [
            "",
            "The winner is provisional until T22 closes stage 2 of the selection (D30 E3); if it "
            "changes, this split is computed again by the same rules.",
        ]
    lines += [
        "",
        "Which theses were assigned where, and the expert sample, are in the gitignored "
        "`data/processed/` only. O04 declares no pass threshold, so none is applied here.",
        "",
        "## Periods",
        "",
        *_period_lines(run),
        "",
        "## Earlier maps",
        "",
        *_map_table(run),
        "",
        f"## Consistency of the later theses (O04), {split} to {end}",
        "",
        *_consistency_table(run, "assigned"),
        "",
        "ARI and NMI compare the later theses' topics with their full-map labels, whatever the "
        "label ids. Agreement is the share of the later theses whose full-map label is their "
        "topic's counterpart: the full-map label of most of the topic's own earlier members, "
        "learned from the earlier theses alone. `own_label` counts the full map's noise as one "
        "more label; `exclude_either` leaves out the theses that are noise in either map "
        f"(D30 E4: `{settings.noise_handling}` primary).",
        "",
        f"## Earlier theses on both maps, {first} to {last}",
        "",
        *_consistency_table(run, "fitted"),
        "",
        "How far the map itself moves when the later theses join it: the earlier theses' labels "
        "on the earlier map against their full-map labels.",
        "",
        "## Expert sample (O04)",
        "",
        *_sample_lines(run),
    ]
    return "\n".join(lines)


def describe(run: TemporalRun, *, seconds: float) -> str:
    """Describe a run in counts and metrics, without thesis text."""
    records, periods = run.records, run.periods
    first, last = periods.fitted_years
    split, end = periods.assigned_years
    model = next(
        m for m in run.inputs.experiment.models if m.model_id == records.winner["representation"]
    )
    return "\n".join(
        [
            f"Winner {records.winner['configuration']}"
            + (" (provisional)" if run.provisional else "")
            + f", {len(records.run_ids)} seed runs; reference run {run.reference.run_id} (seed "
            f"{run.reference.seed})",
            f"Inputs: {len(run.inputs.theses.doc_codes)} theses; {model.model_id} "
            f"({model.vectors.shape[1]} dimensions); {periods.fitted.size} theses of {first} to "
            f"{last} fitted and {periods.assigned.size} theses of {split} to {end} assigned",
            *(f"  T19 {name} SHA-256 {digest}" for name, digest in records.sha256.items()),
            *(f"  {name} SHA-256 {digest}" for name, digest in run.input_sha256.items()),
            *_period_lines(run),
            "Earlier maps:",
            *_map_table(run),
            f"Consistency of the later theses ({split} to {end}):",
            *_consistency_table(run, "assigned"),
            f"Earlier theses on both maps ({first} to {last}):",
            *_consistency_table(run, "fitted"),
            *_sample_lines(run),
            f"Seconds: {seconds:.1f}",
        ]
    )


def _period_lines(run: TemporalRun) -> list[str]:
    theses, periods, settings = run.inputs.theses, run.periods, run.settings
    first, last = periods.fitted_years
    split, end = periods.assigned_years
    fitted = Counter(theses.programs[row] for row in periods.fitted.tolist())
    assigned = Counter(theses.programs[row] for row in periods.assigned.tolist())
    years = Counter(theses.years)
    later = sum(theses.after_snapshot[row] for row in periods.assigned.tolist())
    lines = [
        "| Year | " + " | ".join(str(year) for year in sorted(years)) + " |",
        "|---|" + "---|" * len(years),
        "| Theses | " + " | ".join(str(years[year]) for year in sorted(years)) + " |",
        "",
        f"| Program | {first} to {last} | {split} to {end} |",
        "|---|---|---|",
        *(f"| {key} | {fitted[key]} | {assigned[key]} |" for key in settings.programs),
        f"| All | {periods.fitted.size} | {periods.assigned.size} |",
        "",
        f"{later} of the later theses carry an issue date after the snapshot date (T06); they are "
        "assigned by their issue year, as every other thesis.",
    ]
    return lines


def _map_table(run: TemporalRun) -> list[str]:
    lines = [
        "| Seed | Topics | Noise | Terms | Atypical threshold | Atypical | Full-map noise |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in run.run_rows():
        mark = " (reference)" if row.reference else ""
        threshold = (
            "per topic" if row.atypical_threshold is None else f"{row.atypical_threshold:.4f}"
        )
        lines.append(
            f"| {row.seed}{mark} | {row.topics} | {row.noise_theses} ({_percent(row.noise_share)}) "
            f"| {row.vocabulary_terms} | {threshold} | {row.atypical_theses} of "
            f"{row.assigned_theses} ({_percent(row.atypical_share)}) | "
            f"{row.full_map_noise_assigned} |"
        )
    return lines


def _consistency_table(run: TemporalRun, period: str) -> list[str]:
    handlings = run.settings.noise_handlings
    metrics = ("ari", "nmi", "agreement") if period == "assigned" else ("ari", "nmi")
    columns = [(metric, handling) for metric in metrics for handling in handlings]
    header = " | ".join(
        f"{metric.upper() if metric != 'agreement' else 'Agreement'} {handling}"
        for metric, handling in columns
    )
    lines = [f"| Seed | {header} |", "|---|" + "---|" * len(columns)]
    rows = [row for row in run.consistency_rows() if row.period == period]
    by_seed: dict[int, dict[tuple[str, str], float | None]] = {}
    for row in rows:
        values = by_seed.setdefault(row.seed, {})
        for metric in metrics:
            values[(metric, row.noise_handling)] = getattr(row, metric)
    for result in run.results:
        mark = " (reference)" if result.reference else ""
        cells = [_value(by_seed[result.seed][column]) for column in columns]
        lines.append(f"| {result.seed}{mark} | {' | '.join(cells)} |")
    summary = []
    for column in columns:
        values = [
            by_seed[r.seed][column] for r in run.results if by_seed[r.seed][column] is not None
        ]
        summary.append(_mean_sd(values))
    lines.append(f"| Mean (SD) | {' | '.join(summary)} |")
    return lines


def _sample_lines(run: TemporalRun) -> list[str]:
    sample, settings = run.sample, run.settings
    drawn = len(sample.rows)
    atypical = sum(pool == ATYPICAL_POOL for pool in sample.pools)
    lines = [
        f"Expert sample: {drawn} of the {run.periods.assigned.size} later theses on the reference "
        f"run's map (seed {run.reference.seed}), {atypical} of them atypical, drawn with seed "
        f"{settings.sample_seed}: up to {settings.sample_atypical} atypical, the rest typical, "
        f"each pool spread over the programs with at least {settings.sample_program_floor} per "
        "program that has any, and shown in a seeded order.",
        "",
        "| Pool | Program | Candidates | Drawn |",
        "|---|---|---|---|",
        *(
            f"| {stratum.pool} | {stratum.program} | {stratum.candidates} | {stratum.quota} |"
            for stratum in sample.strata
        ),
    ]
    return lines


def _value(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def _mean_sd(values: Sequence[float]) -> str:
    if not values:
        return "undefined"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{float(np.mean(values)):.3f} ({float(np.std(values, ddof=1)):.3f})"


def _percent(share: float) -> str:
    return f"{100 * share:.1f} %"


# --- The command ----------------------------------------------------------------------------


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.topic_map.run_temporal",
        description=(
            "Evaluate the topic map on a temporal split (T27, O04): fit T19's winner on the "
            "earlier theses (2021-2024), assign the later ones (2025-2026) by nearest centroid "
            "with O06's atypical flag, compare them with T19's full map, and draw the expert "
            "sample. Writes the aggregates under results/temporal/<snapshot>/ (tracked by git) "
            "and each thesis's assignment and the sample under "
            "data/processed/<snapshot>/temporal/ (gitignored). The sample is a one-time draw, "
            "so an earlier temporal split is never overwritten."
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
        help="compute and print everything, apply the privacy guard, and write nothing",
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[AppConfig, Path, str], TemporalInputs] = load_inputs,
    git: Callable[[Path, str], GitState] = read_git_state,
    clock: Callable[[], datetime] = _now,
    timer: Callable[[], float] = time.perf_counter,
    ports: TopicMapPorts | None = None,
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
        results = results_dir(paths.results_dir, snapshot_id)
        processed = processed_dir(paths.data_dir, snapshot_id)
        relative = results.relative_to(root).as_posix()
        if arguments.dry_run:
            return _dry_run(
                config,
                root,
                snapshot_id,
                results,
                processed,
                load=load,
                git=git,
                timer=timer,
                ports=ports,
            )
        state = git(root, relative)
        if state.dirty:
            raise TemporalRunError(
                f"the git tree has {len(state.dirty)} changed or untracked paths outside "
                f"{relative}; commit them first, since the manifest records the commit"
            )
        _refuse_an_earlier_split(snapshot_id, results, processed)
        results.mkdir(parents=True, exist_ok=True)
        busy = TemporalRunError(
            f"another temporal split holds the lock of {relative}; let it finish first"
        )
        with exclusive_run(results, busy):
            _refuse_an_earlier_split(snapshot_id, results, processed)
            started_at, started = clock(), timer()
            run = compute(
                config, root, snapshot_id, load=load, ports=ports, report=_progress, timer=timer
            )
            summary = render_summary(run, git_commit=state.commit)
            _progress(f"writing the records, {MANIFEST_FILE_NAME} last")
            finished_at, seconds = clock(), timer() - started
            manifest = write_outputs(
                results=results,
                processed=processed,
                runs=run.run_rows(),
                consistency=run.consistency_rows(),
                topics=run.topic_rows(),
                composition=run.composition_rows(),
                assignments=run.assignment_rows(),
                sample=run.sample_rows(),
                summary=summary,
                manifest=_manifest(
                    run,
                    git_commit=state.commit,
                    started_at=started_at,
                    finished_at=finished_at,
                    seconds=seconds,
                ),
                guarded=run.guarded,
            )
    except KeyboardInterrupt:
        print(
            f"interrupted: a temporal split is complete only once {MANIFEST_FILE_NAME} is "
            "written; move the partial records away, if any, and run the command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(f"Snapshot {snapshot_id}: commit {manifest.git_commit}")
    print(describe(run, seconds=manifest.seconds))
    print(f"Wrote {relative}/ and the gitignored assignments and sample, {MANIFEST_FILE_NAME} last")
    return 0


def _refuse_an_earlier_split(snapshot_id: str, results: Path, processed: Path) -> None:
    earlier = existing_records(results, processed)
    if earlier is not None:
        raise FileExistsError(
            f"a temporal split of snapshot {snapshot_id} already exists ({earlier}); its expert "
            "sample is a one-time draw, so it is never overwritten: move its records away first "
            "to run it again"
        )


def _dry_run(
    config: AppConfig,
    root: Path,
    snapshot_id: str,
    results: Path,
    processed: Path,
    *,
    load: Callable[[AppConfig, Path, str], TemporalInputs],
    git: Callable[[Path, str], GitState],
    timer: Callable[[], float],
    ports: TopicMapPorts | None,
) -> int:
    relative = results.relative_to(root).as_posix()
    try:
        state: GitState | None = git(root, relative)
    except OSError:
        state = None
    started = timer()
    run = compute(config, root, snapshot_id, load=load, ports=ports, report=_progress, timer=timer)
    texts = tracked_texts(
        runs=run.run_rows(),
        consistency=run.consistency_rows(),
        topics=run.topic_rows(),
        composition=run.composition_rows(),
        summary=render_summary(run, git_commit=None if state is None else state.commit),
    )
    check_privacy(texts, run.guarded)
    seconds = timer() - started
    if state is None:
        tree = "not readable here; a real run needs a repository with a commit"
    elif not state.dirty:
        tree = f"commit {state.commit}, clean"
    else:
        count = len(state.dirty)
        tree = (
            f"commit {state.commit}, {count} changed path{'s' if count > 1 else ''} outside "
            f"{relative}; a real run refuses them until they are committed"
        )
    earlier = existing_records(results, processed)
    print(
        "\n".join(
            [
                f"Dry run of the temporal split on snapshot {snapshot_id}: everything is "
                "computed and nothing is written",
                f"Git: {tree}",
                *(
                    []
                    if earlier is None
                    else [
                        f"A temporal split of this snapshot exists ({earlier.name}); a real run "
                        "refuses to overwrite it"
                    ]
                ),
                describe(run, seconds=seconds),
                f"Privacy guard: {len(texts)} tracked outputs checked against "
                f"{len(run.guarded):,} strings (the snapshot's, and the theses' doc codes), no "
                "hit (the manifest is checked when it is written)",
                f"Command of the real run: {RUN_COMMAND} --snapshot-id {snapshot_id}",
            ]
        )
    )
    return 0


def _manifest(
    run: TemporalRun,
    *,
    git_commit: str,
    started_at: datetime,
    finished_at: datetime,
    seconds: float,
) -> TemporalManifest:
    records, winner = run.records, run.records.winner
    return TemporalManifest(
        snapshot_id=run.snapshot_id,
        git_commit=git_commit,
        experiment=dict(records.sha256),
        inputs=run.input_sha256,
        winner=str(winner["configuration"]),
        representation=str(winner["representation"]),
        algorithm=str(winner["algorithm"]),
        parameter=int(winner["parameter"]),
        provisional=run.provisional,
        runs=records.run_ids,
        temporal_runs=tuple(result.temporal_run_id for result in run.results),
        reference_run=run.reference.run_id,
        settings=run.settings.record(),
        counts=run.counts(),
        versions=library_versions(),
        cpu=cpu_record(),
        files={},
        data_files={},
        started_at=started_at,
        finished_at=finished_at,
        seconds=_seconds(seconds),
        timings=run.timings,
    )


# --- Helpers --------------------------------------------------------------------------------


def _seconds(value: float) -> float:
    return round(max(value, 0.0), 3)


def _position(row: SampleRow) -> int:
    return row.position


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
