"""Run the gap rules of D32 on the records of the T19 experiment (T24a), or forecast the run
with ``--dry-run``::

    uv run python -m thematic_redundancy.gaps.run_gaps [--project-root PATH]
        [--snapshot-id ID] [--dry-run]

The command reads T19's records of the snapshot and refits nothing. ``manifest.json`` comes
first; ``selection.json`` (the winner and its medoid run), ``topics.parquet`` (each topic's
size, terms and NPMI at D30 E4's primary setting) and the gitignored ``labels.parquet`` are each
checked against the SHA-256 that it records, and the configuration must declare the experiment
T19 ran under. The document vectors come through T19's own loader
(:func:`~thematic_redundancy.topic_map.run_experiment.load_inputs`), whose input manifests must be
those T19 ran on, and the centroids are recomputed on the winner's vectors with
:func:`~thematic_redundancy.topic_map.build.topic_centroids` (D30 E5). Program and year come from
the fichas, the career affinity from ``config/career_affinity.yaml`` (U4), the values from
``gaps`` in the configuration (D32), and the strings no output may hold from the snapshot.

It computes rule (a)'s cuts over the winner's seed runs, rules (a) to (d) on every run, the
stability filter with the medoid run as reference, and the sensitivity table of
:mod:`.sensitivity`. ``--dry-run`` stops there: it prints the results and passes the outputs
through the privacy guard, writing nothing. A real run first refuses a dirty git tree (changes
outside ``results/gaps/<snapshot>/``), since the manifest records the commit; then it holds the
run lock of that directory and writes the records of :mod:`.results_store`, ``manifest.json``
last. Progress goes to stderr, a summary without thesis text to stdout; the exit code is 0 on
success, 1 after an error, and 130 after an interruption.
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
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
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.gaps.affinity import CareerAffinity, load_career_affinity
from thematic_redundancy.gaps.results_store import (
    MANIFEST_FILE_NAME,
    GapManifest,
    GapRow,
    MemberRow,
    check_privacy,
    gap_id,
    gap_rows,
    gap_topics,
    member_rows,
    processed_dir,
    results_dir,
    tracked_texts,
    write_outputs,
)
from thematic_redundancy.gaps.rules import (
    SIZE_QUANTILE,
    Gap,
    GapMap,
    GapSettings,
    SmallTopicCuts,
    ThesisMetadata,
    TransferableTopic,
    interstitial_pairs,
    small_coherent_topics,
    small_topic_cuts,
    temporal_declines,
    transferable_topics,
)
from thematic_redundancy.gaps.sensitivity import (
    LITERAL_DISTANCE_QUANTILE,
    LITERAL_SMALL_PERCENT,
    MIN_OVERLAP,
    RARE_MAX_SHARE,
    RULES,
    SensitivityRow,
    sensitivity_rows,
)
from thematic_redundancy.gaps.stability import GapSupport, gap_support
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.shared.storage import (
    ChecksumMismatchError,
    ParquetTableStore,
    TableStore,
)
from thematic_redundancy.topic_map.build import topic_centroids
from thematic_redundancy.topic_map.experiment import ExperimentInputs, TopicRow, run_id
from thematic_redundancy.topic_map.results_store import (
    LABELS_FILE_NAME,
    SELECTION_FILE_NAME,
    TOPICS_FILE_NAME,
    ExperimentManifest,
    LabelRow,
    cpu_record,
    library_versions,
)
from thematic_redundancy.topic_map.results_store import MANIFEST_FILE_NAME as T19_MANIFEST
from thematic_redundancy.topic_map.results_store import processed_dir as t19_processed_dir
from thematic_redundancy.topic_map.results_store import results_dir as t19_results_dir
from thematic_redundancy.topic_map.run_experiment import GitState, load_inputs, read_git_state

CONFIG_PATH = Path("config") / "default.yaml"
AFFINITY_PATH = Path("config") / "career_affinity.yaml"
RUN_COMMAND = "uv run python -m thematic_redundancy.gaps.run_gaps"


class GapRunError(ValueError):
    """The gap run cannot go on: an input disagrees with T19's records or with another input."""


# --- Inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentRecords:
    """What the gap run reads from T19's records, each file checked against T19's manifest."""

    manifest: ExperimentManifest
    winner: Mapping[str, Any]
    """``selection.json``'s winner: configuration, representation, algorithm and parameter."""
    run_ids: tuple[str, ...]
    """The winner's seed runs, in the order of the configured seeds."""
    seeds: tuple[int, ...]
    reference: int
    """The index of the medoid run, ``selection.json``'s representative run."""
    topics: Mapping[str, tuple[TopicRow, ...]]
    """Each run's topics, by topic id."""
    labels: Mapping[str, Mapping[str, int]]
    """Each run's label of each thesis, by doc code."""
    sha256: Mapping[str, str]
    """The SHA-256 of each T19 file read, by name."""


def read_experiment(
    config: AppConfig, project_root: Path, snapshot_id: str, *, store: TableStore | None = None
) -> ExperimentRecords:
    """Read T19's records of ``snapshot_id``, each checked against T19's manifest.

    Raises:
        FileNotFoundError: if a record is missing.
        GapRunError: if a record differs from the SHA-256 in T19's manifest, the manifest is of
            another snapshot or declaration, or the records disagree with each other.
        ValueError: if a record is damaged.
    """
    store = store or ParquetTableStore()
    paths = config.paths.resolve_against(project_root)
    results = t19_results_dir(paths.results_dir, snapshot_id)
    processed = t19_processed_dir(paths.data_dir, snapshot_id)
    content = (results / T19_MANIFEST).read_bytes()
    manifest = ExperimentManifest.model_validate_json(content)
    if manifest.snapshot_id != snapshot_id:
        raise GapRunError(
            f"T19's manifest is of snapshot {manifest.snapshot_id}, not {snapshot_id}"
        )
    declared = config.experiment_declaration_sha256()
    if manifest.declaration_sha256 != declared:
        raise GapRunError(
            f"T19 ran under the experiment declaration {manifest.declaration_sha256}, and the "
            f"configuration declares {declared}; its records are not this declaration's"
        )
    selection = json.loads(_verified(results / SELECTION_FILE_NAME, manifest.files))
    try:
        winner = dict(selection["winner"])
        runs = tuple(
            run_id(
                str(winner["representation"]),
                str(winner["algorithm"]),
                int(winner["parameter"]),
                seed,
            )
            for seed in config.seeds
        )
        representative = str(selection["representative"]["run_id"])
    except (KeyError, TypeError) as error:
        raise GapRunError(f"{SELECTION_FILE_NAME} lacks the winner or its medoid run") from error
    if winner.get("configuration") != manifest.winner:
        raise GapRunError(f"{SELECTION_FILE_NAME} and T19's manifest name different winners")
    if representative not in runs:
        raise GapRunError(f"the medoid run {representative} is not one of the winner's seed runs")
    topic_rows = _verified_table(store, TopicRow, results / TOPICS_FILE_NAME, manifest.files)
    label_rows = _verified_table(store, LabelRow, processed / LABELS_FILE_NAME, manifest.data_files)
    labels: dict[str, dict[str, int]] = {run: {} for run in runs}
    rows_per_run = dict.fromkeys(runs, 0)
    for row in label_rows:
        if row.run_id in labels:
            labels[row.run_id][row.doc_code] = row.label
            rows_per_run[row.run_id] += 1
    for run in runs:
        if not labels[run] or rows_per_run[run] != len(labels[run]):
            raise GapRunError(f"{LABELS_FILE_NAME} must hold one label per thesis for run {run}")
    topics = {
        run: tuple(sorted((row for row in topic_rows if row.run_id == run), key=_topic))
        for run in runs
    }
    return ExperimentRecords(
        manifest=manifest,
        winner=winner,
        run_ids=runs,
        seeds=tuple(config.seeds),
        reference=runs.index(representative),
        topics=topics,
        labels=labels,
        sha256={
            T19_MANIFEST: hashlib.sha256(content).hexdigest(),
            SELECTION_FILE_NAME: manifest.files[SELECTION_FILE_NAME],
            TOPICS_FILE_NAME: manifest.files[TOPICS_FILE_NAME],
            LABELS_FILE_NAME: manifest.data_files[LABELS_FILE_NAME],
        },
    )


@dataclass(frozen=True)
class CorpusInputs:
    """What the gap run reads besides T19's records and the configuration."""

    experiment: ExperimentInputs
    """T19's inputs through T19's loader: every model's document vectors, in one order."""
    metadata: ThesisMetadata
    """Each thesis's program and year, from the fichas, in that order."""
    sensitive: frozenset[str]
    """The snapshot's strings that no tracked output may hold (``corpus/profile.py``)."""
    input_sha256: dict[str, str]
    """The SHA-256 of the inputs read beside T19's input manifests, by file name."""


def thesis_metadata(fichas: Sequence[Ficha], doc_codes: Sequence[str]) -> ThesisMetadata:
    """Return the program and year of each of ``doc_codes``, in that order, from ``fichas``.

    Raises:
        GapRunError: if a doc code has no ficha or a ficha has no issue year.
    """
    by_code = {ficha.doc_code: ficha for ficha in fichas}
    missing = sum(code not in by_code for code in doc_codes)
    if missing:
        noun = "thesis" if missing == 1 else "theses"
        raise GapRunError(f"the fichas lack {missing} {noun} of the document vectors")
    chosen = [by_code[code] for code in doc_codes]
    undated = sum(ficha.issue_year is None for ficha in chosen)
    if undated:
        raise GapRunError(f"{undated} theses have no issue year, which rule (d) needs")
    return ThesisMetadata(
        doc_codes=tuple(doc_codes),
        programs=tuple(str(ficha.program_key) for ficha in chosen),
        years=tuple(ficha.issue_year for ficha in chosen),
    )


def load_corpus(config: AppConfig, project_root: Path, snapshot_id: str) -> CorpusInputs:
    """Load T19's inputs with T19's loader, the fichas and the snapshot's sensitive strings.

    Raises:
        FileNotFoundError: if an input is missing.
        ValueError: if an input is stale or damaged, or the inputs disagree on the theses.
    """
    data_dir = config.paths.resolve_against(project_root).data_dir
    experiment = load_inputs(config, project_root, snapshot_id)
    fichas = load_fichas(data_dir / INTERIM_DIR_NAME / snapshot_id)
    snapshot, records = read_snapshot(data_dir / RAW_DIR_NAME / snapshot_id)
    return CorpusInputs(
        experiment=experiment,
        metadata=thesis_metadata(fichas, experiment.doc_codes),
        sensitive=sensitive_strings(records),
        input_sha256={METADATA_FILE_NAME: snapshot.metadata_sha256},
    )


def build_maps(
    records: ExperimentRecords, doc_codes: Sequence[str], vectors: np.ndarray
) -> tuple[GapMap, ...]:
    """Return each seed run's map: T19's labels in the order of ``doc_codes``, T19's topics and
    their NPMI at the primary setting, and the centroids of the winner's ``vectors``.

    Raises:
        GapRunError: if T19's labels and topics disagree with each other or with the theses.
    """
    maps = []
    for run in records.run_ids:
        by_code = records.labels[run]
        if len(by_code) != len(doc_codes) or any(code not in by_code for code in doc_codes):
            raise GapRunError(
                f"the labels of run {run} do not cover the {len(doc_codes)} theses of the "
                "document vectors"
            )
        labels = np.array([by_code[code] for code in doc_codes], dtype=np.int64)
        rows = records.topics[run]
        topics = tuple(row.topic for row in rows)
        for row in rows:
            size = int(np.count_nonzero(labels == row.topic))
            if size != row.size:
                raise GapRunError(
                    f"topic {row.topic} of run {run} has {row.size} theses in "
                    f"{TOPICS_FILE_NAME} and {size} in {LABELS_FILE_NAME}"
                )
        maps.append(
            GapMap(
                labels=labels,
                topics=topics,
                centroids=topic_centroids(vectors, labels, topics),
                npmi=tuple(row.npmi for row in rows),
            )
        )
    return tuple(maps)


# --- The run --------------------------------------------------------------------------------


@dataclass(frozen=True)
class GapRun:
    """Everything a gap run computed, ready to describe and to write."""

    snapshot_id: str
    records: ExperimentRecords
    corpus: CorpusInputs
    dimensions: int
    settings: GapSettings
    inputs: dict[str, str]
    cuts: SmallTopicCuts
    gaps: tuple[tuple[Gap, ...], ...]
    supports: tuple[GapSupport, ...]
    sensitivity: tuple[SensitivityRow, ...]
    rows: tuple[GapRow, ...]
    members: tuple[MemberRow, ...]

    @property
    def reference_run(self) -> str:
        return self.records.run_ids[self.records.reference]

    @property
    def stable(self) -> tuple[GapSupport, ...]:
        return tuple(support for support in self.supports if support.stable)

    def cuts_record(self) -> dict[str, Any]:
        """Return the content of ``cuts.json``."""
        return {
            "size_cut": self.cuts.size_cut,
            "size_quantile": SIZE_QUANTILE,
            "npmi_floor": self.cuts.npmi_floor,
            "npmi_floor_quantile": self.cuts.npmi_floor_quantile,
            "runs": list(self.records.run_ids),
            "topics": self.cuts.topics,
            "npmi_topics": self.cuts.npmi_topics,
        }

    def counts(self) -> dict[str, int]:
        reference = self.gaps[self.records.reference]
        return {
            "documents": len(self.corpus.metadata.doc_codes),
            "runs": len(self.gaps),
            "gaps": sum(len(run) for run in self.gaps),
            "reference_gaps": len(reference),
            "stable": len(self.stable),
            **{f"stable_{rule}": _count(self.stable, rule) for rule in RULES},
        }


def run_gap_rules(
    maps: Sequence[GapMap],
    vectors: np.ndarray,
    metadata: ThesisMetadata,
    affinity: CareerAffinity,
    settings: GapSettings,
    *,
    run_ids: Sequence[str],
    seeds: Sequence[int],
    reference: int,
) -> tuple[
    SmallTopicCuts,
    tuple[tuple[Gap, ...], ...],
    tuple[GapSupport, ...],
    tuple[SensitivityRow, ...],
]:
    """Return rule (a)'s cuts over every run, each run's gaps of rules (a) to (d), the support of
    the reference run's gaps (D32) and the sensitivity table."""
    cuts = small_topic_cuts(maps, npmi_floor_quantile=settings.npmi_floor_quantile)
    gaps = tuple(
        (
            *small_coherent_topics(
                gap_map, metadata, size_cut=cuts.size_cut, npmi_floor=cuts.npmi_floor
            ),
            *interstitial_pairs(
                gap_map,
                vectors,
                metadata,
                nearest_topics=settings.nearest_topics,
                bridge_max_docs=settings.bridge_max_docs,
            ),
            *transferable_topics(
                gap_map,
                metadata,
                affinity,
                min_affinity=settings.min_affinity,
                present_min_count=settings.present_min_count,
                present_min_share=settings.present_min_share,
                rare_max_share=settings.rare_max_share,
            ),
            *temporal_declines(
                gap_map,
                metadata,
                first_year=settings.temporal_first_year,
                split_year=settings.temporal_split_year,
                last_year=settings.temporal_last_year,
                min_count=settings.temporal_min_count,
                min_decline=settings.min_decline,
            ),
        )
        for gap_map in maps
    )
    supports = gap_support(
        gaps,
        reference=reference,
        min_runs=settings.stability_min_runs,
        min_overlap=settings.stability_min_overlap,
    )
    table = sensitivity_rows(
        maps,
        vectors,
        metadata,
        affinity,
        settings,
        run_ids=run_ids,
        seeds=seeds,
        reference=reference,
        primary=gaps,
    )
    return cuts, gaps, supports, table


def compute(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    *,
    load: Callable[[AppConfig, Path, str], CorpusInputs],
    report: Callable[[str], None],
) -> GapRun:
    """Read the inputs, check them against T19's records and compute the gap run.

    Raises:
        OSError: if an input cannot be read.
        ValueError: if an input is damaged or disagrees with T19's records (a
            :class:`GapRunError` or a :class:`~.rules.GapInputError`).
        yaml.YAMLError: if the career-affinity file is not valid YAML.
    """
    report("reading T19's records")
    records = read_experiment(config, project_root, snapshot_id)
    report("loading the document vectors, the fichas and the snapshot")
    corpus = load(config, project_root, snapshot_id)
    experiment = corpus.experiment
    if experiment.input_sha256 != records.manifest.inputs:
        changed = sorted(
            name
            for name in {*experiment.input_sha256, *records.manifest.inputs}
            if experiment.input_sha256.get(name) != records.manifest.inputs.get(name)
        )
        raise GapRunError(f"the inputs {changed} differ from the ones T19 ran on")
    if corpus.metadata.doc_codes != experiment.doc_codes:
        raise GapRunError("the fichas' theses do not follow the order of the document vectors")
    models = {model.model_id: model for model in experiment.models}
    representation = str(records.winner["representation"])
    if representation not in models:
        raise GapRunError(f"T19's inputs hold no document vectors of the winner {representation}")
    vectors = np.asarray(models[representation].vectors, dtype=np.float64)
    affinity_path = project_root / AFFINITY_PATH
    affinity = load_career_affinity(
        affinity_path, programs=[program.key for program in config.snapshot.programs]
    )
    settings = GapSettings.from_config(config)
    maps = build_maps(records, experiment.doc_codes, vectors)
    report(f"rules (a) to (d) and the stability filter on {len(maps)} runs")
    cuts, gaps, supports, table = run_gap_rules(
        maps,
        vectors,
        corpus.metadata,
        affinity,
        settings,
        run_ids=records.run_ids,
        seeds=records.seeds,
        reference=records.reference,
    )
    rows: list[GapRow] = []
    members: list[MemberRow] = []
    for index, (run, seed, run_gaps) in enumerate(
        zip(records.run_ids, records.seeds, gaps, strict=True)
    ):
        rows += gap_rows(
            run,
            seed,
            run_gaps,
            corpus.metadata,
            supports=supports if index == records.reference else None,
        )
        members += member_rows(run, run_gaps)
    return GapRun(
        snapshot_id=snapshot_id,
        records=records,
        corpus=corpus,
        dimensions=int(vectors.shape[1]),
        settings=settings,
        inputs={
            **experiment.input_sha256,
            **corpus.input_sha256,
            CONFIG_PATH.name: _sha256(project_root / CONFIG_PATH),
            AFFINITY_PATH.name: _sha256(affinity_path),
        },
        cuts=cuts,
        gaps=gaps,
        supports=supports,
        sensitivity=table,
        rows=tuple(rows),
        members=tuple(members),
    )


# --- Text -----------------------------------------------------------------------------------


def render_summary(run: GapRun, *, git_commit: str | None) -> str:
    """Return ``summary.md``: the run, the cuts, the gaps per run, the stable gaps with the T19
    terms of their topics, and the sensitivity table. It holds no doc code."""
    records, settings = run.records, run.settings
    winner = records.winner
    seed = records.seeds[records.reference]
    lines = [
        f"# Gaps of snapshot {run.snapshot_id}",
        "",
        f"Rules (a) to (d) of D32 and their stability filter on the {len(records.run_ids)} seed "
        f"runs of T19's winner {winner['configuration']} ({winner['representation']}, "
        f"{winner['algorithm']} {winner['parameter']}), at commit "
        f"`{git_commit or 'unknown'}`. The reference run is `{run.reference_run}` (seed {seed}), "
        "T19's medoid run, whose map is shown.",
    ]
    if records.manifest.provisional:
        lines += [
            "",
            "The winner is provisional until T22 closes stage 2 of the selection (D30 E3); if "
            "it changes, the cuts and the gaps are computed again by the same rules.",
        ]
    lines += [
        "",
        "Topics are named by run and topic id; their terms are T19's (`topics.parquet`). Which "
        "theses form each gap is in the gitignored `data/processed/` only.",
        "",
        "## Rule (a) cuts",
        "",
        *_cut_lines(run),
        "",
        "## Gaps per run",
        "",
        *_gaps_table(run),
        "",
        "## Stable gaps of the reference run",
        "",
        _stable_line(run),
    ]
    terms = {row.topic: row for row in records.topics[run.reference_run]}
    for rule in RULES:
        stable = [support for support in run.stable if support.gap.rule == rule]
        if not stable:
            continue
        lines += ["", f"### Rule ({rule})", ""]
        if rule == "c":
            high = sum(
                1
                for support in stable
                if isinstance(support.gap, TransferableTopic) and support.gap.affinity == 2
            )
            lines += [
                f"Affinity 2, the high-confidence group of D32 (5): {high}; affinity 1: "
                f"{len(stable) - high}.",
                "",
            ]
        for support in stable:
            gap = support.gap
            lines.append(
                f"- `{gap_id(run.reference_run, gap)}`, in {support.support} of "
                f"{len(support.overlaps)} runs: {gap.reason}."
            )
            for topic in gap_topics(gap):
                row = terms[topic]
                lines.append(f"  - Topic {topic} ({row.size} theses): {', '.join(row.terms)}")
    lines += [
        "",
        "## Sensitivity",
        "",
        "Reported beside the declared rules and never used to select a gap (D32 (1)). A cell "
        "counts the run's gaps of the rule; for `literal_b`, the theses it flags, those that "
        "are noise, and the Jaccard index with the reference run's.",
        "",
        *_sensitivity_table(run),
        "",
        f"Declared values: {_settings_text(settings)}.",
    ]
    return "\n".join(lines)


def describe(run: GapRun, *, seconds: float) -> str:
    """Describe a run in counts, cuts and outcomes, without thesis text."""
    records = run.records
    return "\n".join(
        [
            f"Winner {records.winner['configuration']}"
            + (" (provisional)" if records.manifest.provisional else "")
            + f", {len(records.run_ids)} seed runs; reference run {run.reference_run} (seed "
            f"{records.seeds[records.reference]})",
            f"Inputs: {len(run.corpus.metadata.doc_codes)} theses; "
            f"{records.winner['representation']} ({run.dimensions} dimensions)",
            *(f"  T19 {name} SHA-256 {digest}" for name, digest in records.sha256.items()),
            *(f"  {name} SHA-256 {digest}" for name, digest in run.inputs.items()),
            *_cut_lines(run),
            "Gaps per run:",
            *_gaps_table(run),
            _stable_line(run),
            "Sensitivity (never used for selection):",
            *_sensitivity_table(run),
            f"Seconds: {seconds:.1f}",
        ]
    )


def _cut_lines(run: GapRun) -> list[str]:
    cuts = run.cuts
    return [
        f"Rule (a) size cut {cuts.size_cut}: the first quartile of the sizes of the "
        f"{cuts.topics} topics of the {cuts.runs} runs.",
        f"Rule (a) NPMI floor {cuts.npmi_floor:.4f}: the {cuts.npmi_floor_quantile:g} quantile "
        f"of their {cuts.npmi_topics} defined per-topic NPMI values (primary setting).",
    ]


def _gaps_table(run: GapRun) -> list[str]:
    lines = ["| Seed | (a) | (b) | (c) | (d) | All | Stable |", "|---|---|---|---|---|---|---|"]
    for index, (seed, gaps) in enumerate(zip(run.records.seeds, run.gaps, strict=True)):
        stable = f"{len(run.stable)} of {len(gaps)}" if index == run.records.reference else ""
        cells = [str(_count(gaps, rule)) for rule in RULES]
        mark = " (reference)" if index == run.records.reference else ""
        lines.append(f"| {seed}{mark} | {' | '.join(cells)} | {len(gaps)} | {stable} |")
    return lines


def _stable_line(run: GapRun) -> str:
    settings, reference = run.settings, run.gaps[run.records.reference]
    by_rule = ", ".join(f"({rule}) {_count(run.stable, rule)}" for rule in RULES)
    return (
        f"Stable gaps of the reference run: {len(run.stable)} of its {len(reference)}, present "
        f"in at least {settings.stability_min_runs} of the {len(run.gaps)} runs at a Jaccard "
        f"overlap of at least {settings.stability_min_overlap:g}: {by_rule}."
    )


def _sensitivity_table(run: GapRun) -> list[str]:
    seeds = run.records.seeds
    header = " | ".join(f"Seed {seed}" for seed in seeds)
    lines = [
        f"| Variant | Rule | Setting | {header} | Stable |",
        f"|---|---|---|{'---|' * len(seeds)}---|",
    ]
    groups: dict[tuple[str, str], list[SensitivityRow]] = {}
    for row in run.sensitivity:
        groups.setdefault((row.variant, row.rule), []).append(row)
    for (variant, rule), rows in groups.items():
        cells = [_sensitivity_cell(row) for row in rows]
        stable = next((row.stable for row in rows if row.reference), None)
        lines.append(
            f"| {variant} | ({rule}) | {rows[0].setting} | {' | '.join(cells)} | "
            f"{'' if stable is None else stable} |"
        )
    return lines


def _sensitivity_cell(row: SensitivityRow) -> str:
    if row.gaps is not None:
        return str(row.gaps)
    jaccard = "" if row.jaccard_to_reference is None else f", J {row.jaccard_to_reference:.3f}"
    return f"{row.theses} ({row.noise_theses} noise{jaccard})"


def _settings_text(settings: GapSettings) -> str:
    return ", ".join(f"{name} {value:g}" for name, value in asdict(settings).items())


# --- The command ----------------------------------------------------------------------------


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.gaps.run_gaps",
        description=(
            "Run the gap rules (a) to (d) of D32 and their stability filter on the T19 records "
            "of a snapshot, and write the records under results/gaps/<snapshot>/ (aggregates, "
            "tracked by git) and data/processed/<snapshot>/gaps/ (the theses of each gap, "
            "gitignored)."
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
    load: Callable[[AppConfig, Path, str], CorpusInputs] = load_corpus,
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
        results = results_dir(paths.results_dir, snapshot_id)
        processed = processed_dir(paths.data_dir, snapshot_id)
        relative = results.relative_to(root).as_posix()
        if arguments.dry_run:
            return _dry_run(config, root, snapshot_id, relative, load=load, git=git, timer=timer)
        state = git(root, relative)
        if state.dirty:
            raise GapRunError(
                f"the git tree has {len(state.dirty)} changed or untracked paths outside "
                f"{relative}; commit them first, since the manifest records the commit"
            )
        results.mkdir(parents=True, exist_ok=True)
        busy = GapRunError(f"another gap run holds the lock of {relative}; let it finish first")
        with exclusive_run(results, busy):
            started_at, started = clock(), timer()
            run = compute(config, root, snapshot_id, load=load, report=_progress)
            summary = render_summary(run, git_commit=state.commit)
            _progress(f"writing the records, {MANIFEST_FILE_NAME} last")
            finished_at, seconds = clock(), timer() - started
            manifest = write_outputs(
                results=results,
                processed=processed,
                gaps=run.rows,
                members=run.members,
                sensitivity=run.sensitivity,
                cuts=run.cuts_record(),
                summary=summary,
                manifest=_manifest(
                    run,
                    git_commit=state.commit,
                    started_at=started_at,
                    finished_at=finished_at,
                    seconds=seconds,
                ),
                sensitive=run.corpus.sensitive,
            )
    except KeyboardInterrupt:
        print(
            f"interrupted: records that differ from {MANIFEST_FILE_NAME} are stale; run the "
            "command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(f"Snapshot {snapshot_id}: commit {manifest.git_commit}")
    print(describe(run, seconds=manifest.seconds))
    print(f"Wrote {relative}/ and the gitignored members, {MANIFEST_FILE_NAME} last")
    return 0


def _dry_run(
    config: AppConfig,
    root: Path,
    snapshot_id: str,
    relative: str,
    *,
    load: Callable[[AppConfig, Path, str], CorpusInputs],
    git: Callable[[Path, str], GitState],
    timer: Callable[[], float],
) -> int:
    try:
        state: GitState | None = git(root, relative)
    except OSError:
        state = None
    started = timer()
    run = compute(config, root, snapshot_id, load=load, report=_progress)
    texts = tracked_texts(
        gaps=run.rows,
        sensitivity=run.sensitivity,
        cuts=run.cuts_record(),
        summary=render_summary(run, git_commit=None if state is None else state.commit),
    )
    check_privacy(texts, run.corpus.sensitive)
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
    print(
        "\n".join(
            [
                f"Dry run of the gap run on snapshot {snapshot_id}: everything is computed and "
                "nothing is written",
                f"Git: {tree}",
                describe(run, seconds=seconds),
                f"Privacy guard: {len(texts)} tracked outputs checked against "
                f"{len(run.corpus.sensitive)} strings of the snapshot, no hit (the manifest is "
                "checked when it is written)",
                f"Command of the real run: {RUN_COMMAND} --snapshot-id {snapshot_id}",
            ]
        )
    )
    return 0


def _manifest(
    run: GapRun, *, git_commit: str, started_at: datetime, finished_at: datetime, seconds: float
) -> GapManifest:
    records = run.records
    return GapManifest(
        snapshot_id=run.snapshot_id,
        git_commit=git_commit,
        experiment=dict(records.sha256),
        inputs=run.inputs,
        winner=str(records.winner["configuration"]),
        representation=str(records.winner["representation"]),
        provisional=records.manifest.provisional,
        runs=records.run_ids,
        reference_run=run.reference_run,
        settings=asdict(run.settings),
        sensitivity={
            "literal_small_percent": LITERAL_SMALL_PERCENT,
            "literal_distance_quantile": LITERAL_DISTANCE_QUANTILE,
            "rare_max_share": RARE_MAX_SHARE,
            "min_overlap": MIN_OVERLAP,
        },
        counts=run.counts(),
        versions=library_versions(),
        cpu=cpu_record(),
        files={},
        data_files={},
        started_at=started_at,
        finished_at=finished_at,
        seconds=round(max(seconds, 0.0), 3),
    )


# --- Helpers --------------------------------------------------------------------------------


def _verified(path: Path, recorded: Mapping[str, str]) -> bytes:
    """Return the bytes of ``path`` once their SHA-256 is the one T19's manifest records."""
    expected = recorded.get(path.name)
    if expected is None:
        raise GapRunError(f"T19's manifest records no SHA-256 for {path.name}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise GapRunError(_differs(path.name))
    return content


def _verified_table[M: (TopicRow, LabelRow)](
    store: TableStore, model_cls: type[M], path: Path, recorded: Mapping[str, str]
) -> tuple[M, ...]:
    expected = recorded.get(path.name)
    if expected is None:
        raise GapRunError(f"T19's manifest records no SHA-256 for {path.name}")
    try:
        return store.read_verified(model_cls, path, sha256=expected)
    except ChecksumMismatchError as error:
        raise GapRunError(_differs(path.name)) from error


def _differs(name: str) -> str:
    return f"{name} differs from the SHA-256 that T19's manifest records, so it is not used"


def _topic(row: TopicRow) -> int:
    return row.topic


def _count(items: Sequence[Gap] | Sequence[GapSupport], rule: str) -> int:
    return sum(
        1 for item in items if (item.gap if isinstance(item, GapSupport) else item).rule == rule
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
