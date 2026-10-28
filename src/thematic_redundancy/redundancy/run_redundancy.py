"""Run T22 on the labeled set once its consensus labels exist, or forecast the run with
``--dry-run``::

    uv run python -m thematic_redundancy.redundancy.run_redundancy [--project-root PATH]
        [--snapshot-id ID] [--dry-run] [--labels PATH]

The command reads, and checks before any use:

- T20a's draw: its ``manifest.json`` under ``results/labeled_set/<snapshot>/draw/`` and the
  gitignored ``pairs.parquet``, which must have the SHA-256 that the manifest records. The
  labeled set's pairs (phase ``labeled``), in draw order, are the pairs of the run.
- The consensus labels of :mod:`.labels`, at the contract's path, one per labeled pair and no
  pilot pair. ``--labels`` reads another file, and only with ``--dry-run``.
- T19's ``manifest.json`` and ``selection.json``, which must have the SHA-256 that T19's
  manifest records and the configuration's experiment declaration. The two models' stage-1
  configurations are rebuilt from ``selection.json``, and T19's provisional stage 2 must be
  what they give.
- The document vectors of both embedding models and the TF-IDF matrix (``document_vector``),
  through T20a's own loader, whose input manifests must be those that T20a drew from and T19 ran
  on. Each labeled pair's cosine under each representation
  (:func:`~thematic_redundancy.redundancy.search.pair_cosines`) must equal the score that T20a
  stored, exactly.

With T22's functions and the values declared in ``redundancy`` and ``experiment.selection.f1``,
it computes, for each representation, τ by cross-validation, the F1 of all the labeled pairs at
it with its bootstrap interval, and the precision-recall curve; the paired comparison of the two
embedding models and of each against TF-IDF; stage 2 of the winner closed by D30 E3's full rule
(:mod:`.stage_two`), which confirms or changes T19's provisional winner, or is undefined; and
acceptance criteria 1 (its F1 part, and T19's NPMI part while the winner stands) and 2 (the
winner's cross-validated F1 against :data:`CRITERION_TWO_MIN_F1`).

``--dry-run`` stops there: it prints the results and passes the outputs through the privacy
guard, writing nothing. A run happens once: a real run refuses an earlier run of the snapshot
(its ``manifest.json`` or its gitignored ``pair_scores.parquet``) and a dirty git tree (changes
outside ``results/experiments/<snapshot>/t22/``), since the manifest records the commit; then it
holds the run lock of that directory and writes the records of :mod:`.results_store`,
``manifest.json`` last. Progress goes to stderr, a summary without thesis text to stdout; the
exit code is 0 on success, 1 after an error, and 130 after an interruption.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from thematic_redundancy.corpus.snapshot import RAW_DIR_NAME, check_snapshot_id, latest_snapshot_id
from thematic_redundancy.gaps.results_store import check_privacy
from thematic_redundancy.labeling import draw_pairs
from thematic_redundancy.labeling.draw_pairs import (
    PAIRS_FILE_NAME,
    DrawInputs,
    DrawManifest,
    PairRow,
    load_inputs,
)
from thematic_redundancy.labeling.pairs import LABELED, PILOT, TFIDF
from thematic_redundancy.redundancy.comparison import (
    BootstrapSettings,
    F1Interval,
    RepresentationComparison,
    bootstrap_f1,
    compare_representations,
)
from thematic_redundancy.redundancy.labels import (
    ADJUDICATION,
    AGREEMENT,
    REDUNDANT,
    RELATED,
    UNRELATED,
    ConsensusLabel,
    consensus_labels_path,
    labels_for_pairs,
    read_consensus_labels,
)
from thematic_redundancy.redundancy.results_store import (
    MANIFEST_FILE_NAME,
    PAIR_SCORES_FILE_NAME,
    ComparisonRow,
    CurveRow,
    FoldRow,
    PairScoreRow,
    RedundancyManifest,
    RepresentationRow,
    existing_run,
    processed_dir,
    results_dir,
    tracked_texts,
    write_outputs,
)
from thematic_redundancy.redundancy.search import pair_cosines
from thematic_redundancy.redundancy.stage_two import (
    StageTwo,
    candidate_from_record,
    close_stage_two,
)
from thematic_redundancy.redundancy.threshold import (
    CvSettings,
    MaxF1Threshold,
    ThresholdCalibration,
    calibrate_threshold,
    max_f1_threshold,
)
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.shared.storage import ChecksumMismatchError, ParquetTableStore, TableStore
from thematic_redundancy.topic_map.experiment import ExperimentSettings
from thematic_redundancy.topic_map.results_store import MANIFEST_FILE_NAME as T19_MANIFEST
from thematic_redundancy.topic_map.results_store import SELECTION_FILE_NAME as T19_SELECTION
from thematic_redundancy.topic_map.results_store import (
    ExperimentManifest,
    cpu_record,
    library_versions,
)
from thematic_redundancy.topic_map.results_store import results_dir as t19_results_dir
from thematic_redundancy.topic_map.run_experiment import GitState, read_git_state
from thematic_redundancy.topic_map.selection import (
    Candidate,
    evaluate_gates,
    stage_two_provisional,
)

CONFIG_PATH = Path("config") / "default.yaml"
RUN_COMMAND = "uv run python -m thematic_redundancy.redundancy.run_redundancy"
CRITERION_TWO_MIN_F1 = 0.70
"""Acceptance criterion 2 of the plan: "Direct-redundancy F1 is ≥ 0.70 at the selected τ,
cross-validated and reported with a bootstrap CI"."""
CRITERION_TWO_F1 = "cv_f1"
"""The F1 that criterion 2 compares with :data:`CRITERION_TWO_MIN_F1`: the cross-validated F1,
as T22's note 7 proposes; the F1 of all the labeled pairs at τ, with its interval, is beside it."""
STAGE_TWO = "stage_two"
CRITERION_ONE = "criterion_one"

Compare = Callable[..., RepresentationComparison]
"""The comparison of two representations, with the signature of
:func:`~thematic_redundancy.redundancy.comparison.compare_representations`."""


class RedundancyRunError(ValueError):
    """The run cannot go on: an input disagrees with T20a's draw, T19's records or another input."""


# --- Inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawRecords:
    """What the run reads from T20a's draw, ``pairs.parquet`` checked against its manifest."""

    manifest: DrawManifest
    pairs: tuple[PairRow, ...]
    """The labeled set's pairs, in draw order."""
    pilot: int
    """The pilot's pairs, which the labeled set never holds."""
    pair_ids: frozenset[str]
    """Every drawn pair's id, the pilot's included, which no tracked output may hold."""
    sha256: dict[str, str]


def read_draw(
    config: AppConfig, project_root: Path, snapshot_id: str, *, store: TableStore | None = None
) -> DrawRecords:
    """Read T20a's draw of ``snapshot_id``, ``pairs.parquet`` checked against its manifest.

    Raises:
        FileNotFoundError: if a record is missing.
        RedundancyRunError: if ``pairs.parquet`` differs from the SHA-256 in the manifest, the
            manifest is of another snapshot, or the pairs are not scored under the configured
            representations.
        ValueError: if a record is damaged.
    """
    store = store or ParquetTableStore()
    paths = config.paths.resolve_against(project_root)
    content = (
        draw_pairs.results_dir(paths.results_dir, snapshot_id) / MANIFEST_FILE_NAME
    ).read_bytes()
    manifest = DrawManifest.model_validate_json(content)
    if manifest.snapshot_id != snapshot_id:
        raise RedundancyRunError(
            f"T20a's manifest is of snapshot {manifest.snapshot_id}, not {snapshot_id}"
        )
    expected = manifest.data_files.get(PAIRS_FILE_NAME)
    if expected is None:
        raise RedundancyRunError(f"T20a's manifest records no SHA-256 for {PAIRS_FILE_NAME}")
    path = draw_pairs.records_dir(config, project_root, snapshot_id) / PAIRS_FILE_NAME
    try:
        rows = store.read_verified(PairRow, path, sha256=expected)
    except ChecksumMismatchError as error:
        raise RedundancyRunError(
            f"{PAIRS_FILE_NAME} differs from the SHA-256 that T20a's manifest records, so it is "
            "not used"
        ) from error
    representations = (*config.embedding.models, TFIDF)
    if any(row.representations != representations for row in rows):
        raise RedundancyRunError(
            f"T20a's pairs are not scored under {list(representations)}, the configured models "
            "and the TF-IDF baseline"
        )
    labeled = tuple(sorted((row for row in rows if row.phase == LABELED), key=_draw_order))
    if not labeled:
        raise RedundancyRunError("T20a's draw holds no pair of the labeled set")
    return DrawRecords(
        manifest=manifest,
        pairs=labeled,
        pilot=sum(row.phase == PILOT for row in rows),
        pair_ids=frozenset(row.pair_id for row in rows),
        sha256={MANIFEST_FILE_NAME: _digest(content), PAIRS_FILE_NAME: expected},
    )


@dataclass(frozen=True)
class ExperimentRecords:
    """What the run reads from T19's records, ``selection.json`` checked against T19's
    manifest."""

    manifest: ExperimentManifest
    selection: dict[str, Any]
    winners: tuple[Candidate, Candidate]
    """Each model's stage-1 configuration, in the order of ``embedding.models``."""
    sha256: dict[str, str]

    @property
    def provisional_winner(self) -> dict[str, Any]:
        """``selection.json``'s winner: configuration, representation, algorithm, parameter."""
        return dict(self.selection["winner"])


def read_experiment(
    config: AppConfig, project_root: Path, snapshot_id: str, settings: ExperimentSettings
) -> ExperimentRecords:
    """Read T19's records of ``snapshot_id``, ``selection.json`` checked against T19's manifest,
    and rebuild each model's stage-1 configuration from it.

    Raises:
        FileNotFoundError: if a record is missing.
        RedundancyRunError: if ``selection.json`` differs from the SHA-256 in T19's manifest,
            the manifest is of another snapshot or declaration, or ``selection.json`` is not
            what T19's rules give on its own candidates.
        ValueError: if a record is damaged.
    """
    results = t19_results_dir(config.paths.resolve_against(project_root).results_dir, snapshot_id)
    content = (results / T19_MANIFEST).read_bytes()
    manifest = ExperimentManifest.model_validate_json(content)
    if manifest.snapshot_id != snapshot_id:
        raise RedundancyRunError(
            f"T19's manifest is of snapshot {manifest.snapshot_id}, not {snapshot_id}"
        )
    declared = config.experiment_declaration_sha256()
    if manifest.declaration_sha256 != declared:
        raise RedundancyRunError(
            f"T19 ran under the experiment declaration {manifest.declaration_sha256}, and the "
            f"configuration declares {declared}; its records are not this declaration's"
        )
    expected = manifest.files.get(T19_SELECTION)
    selection_bytes = (results / T19_SELECTION).read_bytes()
    if expected is None or _digest(selection_bytes) != expected:
        raise RedundancyRunError(
            f"{T19_SELECTION} differs from the SHA-256 that T19's manifest records, so it is "
            "not used"
        )
    selection = json.loads(selection_bytes)
    try:
        winners = tuple(_stage_one_winner(selection, model, settings) for model in settings.models)
        recorded = selection["stage_two"]["decision"]
        winner = selection["winner"]
    except (KeyError, TypeError, ValueError) as error:
        raise RedundancyRunError(
            f"{T19_SELECTION} lacks the stage-1 configurations or stage 2 that T19 writes: {error}"
        ) from error
    if len(winners) != 2:
        raise RedundancyRunError("stage 2 compares two embedding models")
    first, second = winners
    provisional = stage_two_provisional(
        first, second, gates=settings.gates, fallback_model=settings.fallback_model
    )
    if _plain(asdict(provisional)) != recorded:
        raise RedundancyRunError(
            f"T19's provisional stage 2 is not what its candidates give; {T19_SELECTION} is not "
            "the record of T19's rules"
        )
    by_model = {candidate.representation: candidate for candidate in winners}
    if winner.get("configuration") != manifest.winner or (
        by_model[provisional.winner].configuration != manifest.winner
    ):
        raise RedundancyRunError(f"{T19_SELECTION} and T19's manifest name different winners")
    return ExperimentRecords(
        manifest=manifest,
        selection=selection,
        winners=(first, second),
        sha256={T19_MANIFEST: _digest(content), T19_SELECTION: expected},
    )


def _stage_one_winner(
    selection: Mapping[str, Any], model: str, settings: ExperimentSettings
) -> Candidate:
    name = selection["stage_one"][model]["decision"]["winner"]
    record = selection["configurations"][name]
    candidate = candidate_from_record(record)
    if candidate.representation != model:
        raise ValueError(f"{name} is not a configuration of {model}")
    if list(record["gates_failed"]) != list(evaluate_gates(candidate, settings.gates).failed):
        raise ValueError(f"the gates recorded for {name} are not the declared gates' result")
    return candidate


def labeled_scores(inputs: DrawInputs, pairs: Sequence[PairRow]) -> tuple[np.ndarray, ...]:
    """Return the cosine of each of ``pairs`` under each representation, each embedding model
    then TF-IDF, once it equals the score that T20a stored.

    Raises:
        RedundancyRunError: if a representation lacks a thesis of the pairs or scores a pair
            otherwise than T20a did.
    """
    scores = []
    for index, vectors in enumerate((*inputs.semantic, inputs.tfidf)):
        position = {code: row for row, code in enumerate(vectors.doc_codes)}
        codes = [(pair.doc_code_a, pair.doc_code_b) for pair in pairs]
        missing = {code for pair in codes for code in pair if code not in position}
        if missing:
            raise RedundancyRunError(
                f"the document vectors of {vectors.name} lack {len(missing)} theses of the "
                "labeled pairs"
            )
        rows = np.array([(position[a], position[b]) for a, b in codes], dtype=np.int64)
        computed = pair_cosines(vectors.vectors, rows)
        stored = np.array([pair.scores[index] for pair in pairs], dtype=np.float64)
        if not np.array_equal(computed, stored):
            differ = int(np.count_nonzero(computed != stored))
            raise RedundancyRunError(
                f"the scores of {vectors.name} differ from those T20a stored for {differ} of the "
                f"{len(pairs)} labeled pairs (by up to {np.max(np.abs(computed - stored)):.3g}); "
                "these are not the document vectors T20a drew from"
            )
        scores.append(computed)
    return tuple(scores)


# --- The run --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Representation:
    """One representation over the labeled pairs: τ by cross-validation, the F1 interval at
    it, and the precision-recall curve of all the pairs."""

    name: str
    scores: np.ndarray
    calibration: ThresholdCalibration
    interval: F1Interval
    curve: MaxF1Threshold


@dataclass(frozen=True)
class RedundancyRun:
    """Everything a run computed, ready to describe and to write."""

    snapshot_id: str
    config: AppConfig
    labels_path: Path
    draw: DrawRecords
    experiment: ExperimentRecords
    labels: tuple[ConsensusLabel, ...]
    """One per labeled pair, in draw order."""
    cv: CvSettings
    bootstrap: BootstrapSettings
    representations: tuple[Representation, ...]
    comparisons: tuple[tuple[str, RepresentationComparison], ...]
    """``(purpose, comparison)``: the two models (stage 2), then each against TF-IDF."""
    stage_two: StageTwo
    input_sha256: dict[str, str]
    sensitive: frozenset[str]
    timings: dict[str, float]

    @property
    def guarded(self) -> frozenset[str]:
        """The strings no tracked output may hold: the snapshot's, the theses' doc codes and
        every drawn pair's id."""
        return self.sensitive | self.draw.pair_ids

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(representation.name for representation in self.representations)

    @property
    def provisional_configuration(self) -> str:
        return str(self.experiment.provisional_winner["configuration"])

    @property
    def winner(self) -> Candidate | None:
        """The winner by the full rule, a stage-1 configuration; ``None`` when undefined."""
        model = self.stage_two.winner
        return next((c for c in self.experiment.winners if c.representation == model), None)

    def counts(self) -> dict[str, int]:
        levels = Counter(label.label_3 for label in self.labels)
        sources = Counter(label.source for label in self.labels)
        return {
            "labeled_pairs": len(self.labels),
            "pilot_pairs": self.draw.pilot,
            "redundant": levels[REDUNDANT],
            "related": levels[RELATED],
            "unrelated": levels[UNRELATED],
            "agreement": sources[AGREEMENT],
            "adjudication": sources[ADJUDICATION],
            "known_duplicates": sum(pair.known_duplicate for pair in self.draw.pairs),
        }


def compute(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    *,
    labels_path: Path,
    load: Callable[[AppConfig, Path, str], DrawInputs],
    compare: Compare,
    report: Callable[[str], None],
    timer: Callable[[], float],
) -> RedundancyRun:
    """Read and check every input, then calibrate, compare and close stage 2.

    Raises:
        OSError: if an input cannot be read.
        ValueError: if an input is damaged or disagrees with another (a
            :class:`RedundancyRunError` or a :class:`~.labels.LabelsError`), or τ cannot be
            calibrated on the labels.
    """
    started = timer()
    settings = ExperimentSettings.from_config(config)
    report("reading T20a's draw, the consensus labels and T19's records")
    draw = read_draw(config, project_root, snapshot_id)
    labels = labels_for_pairs(
        read_consensus_labels(labels_path), [pair.pair_id for pair in draw.pairs]
    )
    experiment = read_experiment(config, project_root, snapshot_id, settings)
    report("loading the document vectors through T20a's loader")
    inputs = load(config, project_root, snapshot_id)
    _check_inputs(config, inputs, draw, experiment)
    loaded = timer()
    report(f"scoring the {len(draw.pairs)} labeled pairs under {len(inputs.semantic) + 1} vectors")
    scores = labeled_scores(inputs, draw.pairs)
    checked = timer()
    redundancy, f1 = config.redundancy, config.experiment.selection.f1
    cv = CvSettings(
        folds=redundancy.cv_folds,
        seed=redundancy.cv_seed,
        stratified=redundancy.cv_stratified,
        f1_tie=redundancy.f1_tie,
        final_tau=redundancy.final_tau,
    )
    bootstrap = BootstrapSettings(
        resamples=f1.bootstrap_resamples, seed=f1.bootstrap_seed, interval=f1.interval
    )
    binary = np.array([label.label for label in labels], dtype=np.int64)
    names = (*settings.models, TFIDF)
    report(f"calibrating tau by {cv.folds}-fold cross-validation and bootstrapping the F1")
    representations = tuple(
        _representation(name, values, binary, cv, bootstrap)
        for name, values in zip(names, scores, strict=True)
    )
    by_name = {representation.name: representation.scores for representation in representations}
    plan = ((STAGE_TWO, names[0], names[1]), (CRITERION_ONE, names[0], TFIDF))
    plan += ((CRITERION_ONE, names[1], TFIDF),)
    comparisons = tuple(
        (purpose, compare(by_name[a], by_name[b], binary, names=(a, b), cv=cv, bootstrap=bootstrap))
        for purpose, a, b in plan
    )
    stage_two = close_stage_two(
        *experiment.winners,
        comparisons[0][1].difference,
        gates=settings.gates,
        fallback_model=settings.fallback_model,
    )
    computed = timer()
    return RedundancyRun(
        snapshot_id=snapshot_id,
        config=config,
        labels_path=labels_path,
        draw=draw,
        experiment=experiment,
        labels=labels,
        cv=cv,
        bootstrap=bootstrap,
        representations=representations,
        comparisons=comparisons,
        stage_two=stage_two,
        input_sha256={**inputs.input_sha256, CONFIG_PATH.name: _sha256(project_root / CONFIG_PATH)},
        sensitive=inputs.sensitive | frozenset(code for code, _ in inputs.theses),
        timings={
            "load": round(loaded - started, 3),
            "check": round(checked - loaded, 3),
            "compute": round(computed - checked, 3),
        },
    )


def _check_inputs(
    config: AppConfig, inputs: DrawInputs, draw: DrawRecords, experiment: ExperimentRecords
) -> None:
    names = tuple(vectors.name for vectors in inputs.semantic)
    if names != config.embedding.models or inputs.tfidf.name != TFIDF:
        raise RedundancyRunError(
            f"the document vectors are of {[*names, inputs.tfidf.name]}, not of the configured "
            f"models {list(config.embedding.models)} and {TFIDF}"
        )
    for name, digest in draw.manifest.inputs.items():
        # The configuration may have gained keys since the draw; the pairs fix the draw.
        if name != CONFIG_PATH.name and inputs.input_sha256.get(name) != digest:
            raise RedundancyRunError(
                f"{name} differs from the one T20a drew from, so the scores are not the draw's"
            )
    for name, digest in experiment.manifest.inputs.items():
        if inputs.input_sha256.get(name) != digest:
            raise RedundancyRunError(
                f"{name} differs from the one T19 ran on, so its stage 1 is not of these vectors"
            )


def _representation(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    cv: CvSettings,
    bootstrap: BootstrapSettings,
) -> Representation:
    try:
        calibration = calibrate_threshold(scores, labels, cv)
    except ValueError as error:
        positives = int(labels.sum())
        raise RedundancyRunError(
            f"tau cannot be calibrated by {cv.folds}-fold cross-validation on {positives} "
            f"redundant pair{'' if positives == 1 else 's'} of {labels.size}: {error}"
        ) from error
    return Representation(
        name=name,
        scores=scores,
        calibration=calibration,
        interval=bootstrap_f1(scores, labels, tau=calibration.tau, settings=bootstrap),
        curve=max_f1_threshold(scores, labels, f1_tie=cv.f1_tie),
    )


# --- Records --------------------------------------------------------------------------------


def representation_rows(run: RedundancyRun) -> tuple[RepresentationRow, ...]:
    """Return one row per representation, each embedding model then TF-IDF."""
    rows = []
    for item in run.representations:
        calibration, interval = item.calibration, item.interval
        at_tau, pooled = calibration.at_tau, calibration.pooled
        rows.append(
            RepresentationRow(
                representation=item.name,
                pairs=calibration.pairs,
                positives=calibration.positives,
                tau=calibration.tau,
                true_positives=at_tau.true_positives,
                false_positives=at_tau.false_positives,
                false_negatives=at_tau.false_negatives,
                true_negatives=at_tau.true_negatives,
                precision=at_tau.precision,
                recall=at_tau.recall,
                f1=at_tau.f1,
                f1_lower=interval.lower,
                f1_upper=interval.upper,
                undefined_resamples=interval.undefined_resamples,
                cv_true_positives=pooled.true_positives,
                cv_false_positives=pooled.false_positives,
                cv_false_negatives=pooled.false_negatives,
                cv_true_negatives=pooled.true_negatives,
                cv_precision=pooled.precision,
                cv_recall=pooled.recall,
                cv_f1=calibration.cv_f1,
                mean_fold_f1=calibration.mean_fold_f1,
            )
        )
    return tuple(rows)


def fold_rows(run: RedundancyRun) -> tuple[FoldRow, ...]:
    """Return one row per representation and fold."""
    return tuple(
        FoldRow(
            representation=item.name,
            fold=outcome.fold,
            train_pairs=outcome.train_pairs,
            train_positives=outcome.train_positives,
            tau=outcome.tau,
            train_f1=outcome.train_f1,
            test_pairs=sum(asdict(outcome.test).values()),
            test_positives=outcome.test.true_positives + outcome.test.false_negatives,
            true_positives=outcome.test.true_positives,
            false_positives=outcome.test.false_positives,
            false_negatives=outcome.test.false_negatives,
            true_negatives=outcome.test.true_negatives,
            test_f1=outcome.test_f1,
        )
        for item in run.representations
        for outcome in item.calibration.folds
    )


def curve_rows(run: RedundancyRun) -> tuple[CurveRow, ...]:
    """Return every point of each representation's precision-recall curve, highest τ first."""
    rows = []
    for item in run.representations:
        for index, point in enumerate(item.curve.curve):
            counts = point.counts
            rows.append(
                CurveRow(
                    representation=item.name,
                    point=index,
                    tau=point.tau,
                    true_positives=counts.true_positives,
                    false_positives=counts.false_positives,
                    false_negatives=counts.false_negatives,
                    true_negatives=counts.true_negatives,
                    precision=counts.precision,
                    recall=counts.recall,
                    f1=point.f1,
                    selected=point.tau == item.calibration.tau,
                )
            )
    return tuple(rows)


def comparison_rows(run: RedundancyRun) -> tuple[ComparisonRow, ...]:
    """Return one row per paired comparison, in the order they were computed."""
    return tuple(
        ComparisonRow(
            purpose=purpose,
            first=comparison.first_name,
            second=comparison.second_name,
            first_tau=comparison.difference.first_tau,
            second_tau=comparison.difference.second_tau,
            first_f1=comparison.difference.first_f1,
            second_f1=comparison.difference.second_f1,
            difference=comparison.difference.difference,
            lower=comparison.difference.lower,
            upper=comparison.difference.upper,
            verdict=comparison.difference.verdict,
            undefined_resamples=comparison.difference.undefined_resamples,
        )
        for purpose, comparison in run.comparisons
    )


def pair_score_rows(run: RedundancyRun) -> tuple[PairScoreRow, ...]:
    """Return one row per labeled pair, in draw order, for the gitignored records."""
    folds = run.representations[0].calibration.fold_of_pair
    taus = [item.calibration.tau for item in run.representations]
    rows = []
    for index, (pair, label) in enumerate(zip(run.draw.pairs, run.labels, strict=True)):
        scores = tuple(float(item.scores[index]) for item in run.representations)
        rows.append(
            PairScoreRow(
                pair_id=pair.pair_id,
                label_3=label.label_3,
                label=label.label,
                source=label.source,
                fold=folds[index],
                representations=run.names,
                scores=scores,
                predicted=tuple(score >= tau for score, tau in zip(scores, taus, strict=True)),
                known_duplicate=pair.known_duplicate,
            )
        )
    return tuple(rows)


def selection_record(run: RedundancyRun) -> dict[str, Any]:
    """Return ``selection.json``: stage 2 closed by the full rule against T19's provisional
    winner, criteria 1 and 2, the findings and what follows from the outcome."""
    stage, winner = run.stage_two, run.winner
    one, two = _criterion_one(run), _criterion_two(run)
    record = {
        "snapshot_id": run.snapshot_id,
        "provisional_winner": run.experiment.provisional_winner,
        "stage_one_winners": {c.representation: c.configuration for c in run.experiment.winners},
        "stage_two": {
            "status": stage.status,
            "outcome": stage.outcome(),
            "decision": None if stage.decision is None else asdict(stage.decision),
            "provisional": asdict(stage.provisional),
            "f1": asdict(stage.f1),
            "undefined_reason": stage.undefined_reason,
            "sensitivities": [asdict(item) for item in stage.sensitivities],
        },
        "winner": None
        if winner is None
        else {
            "configuration": winner.configuration,
            "representation": winner.representation,
            "algorithm": winner.algorithm,
            "parameter": winner.parameter,
            "provisional": False,
        },
        "criterion_one": one,
        "criterion_two": two,
        "known_duplicates": {"labeled": run.counts()["known_duplicates"]},
        "findings": _findings(run, one, two),
        "consequences": _consequences(run),
    }
    return _plain(record)


def _criterion_one(run: RedundancyRun) -> dict[str, Any]:
    """Acceptance criterion 1 (UD1): the winner beats TF-IDF on F1, with a paired interval of
    ΔF1 that excludes 0, and on NPMI, T19's part, which holds while T19's winner stands."""
    per_model = {
        comparison.first_name: {
            "difference": comparison.difference.difference,
            "lower": comparison.difference.lower,
            "upper": comparison.difference.upper,
            "verdict": comparison.difference.verdict,
            "beats_tfidf": comparison.first_beats_second,
        }
        for purpose, comparison in run.comparisons
        if purpose == CRITERION_ONE
    }
    t19 = run.experiment.selection.get("criterion_one", {})
    winner = run.stage_two.winner
    if winner is None:
        return {
            "representation": None,
            "f1_met": None,
            "npmi_met": None,
            "met": None,
            "per_model": per_model,
            "t19_npmi": t19,
            "reason": "stage 2 is undefined, so criterion 1 has no winner to judge; the F1 part "
            "of each model is in per_model",
        }
    f1_met = bool(per_model[winner]["beats_tfidf"])
    confirmed = run.stage_two.outcome() == "confirmed"
    npmi_met = t19.get("npmi_met") if confirmed else None
    met = False if False in (f1_met, npmi_met) else (True if npmi_met is True else None)
    if confirmed:
        npmi = "T19's NPMI part, for the same winner, is " + ("met" if npmi_met else "not met")
    else:
        npmi = (
            "the NPMI part is pending: T19's records judge the provisional winner, and hold no "
            "NPMI of the baseline at the new winner's matched K"
        )
    f1 = "beats" if f1_met else "does not beat"
    return {
        "representation": winner,
        "f1_met": f1_met,
        "npmi_met": npmi_met,
        "met": met,
        "per_model": per_model,
        "t19_npmi": t19,
        "reason": f"the winner {f1} TF-IDF on F1 with a paired {run.bootstrap.interval:.0%} "
        f"interval of ΔF1 that excludes 0; {npmi}",
    }


def _criterion_two(run: RedundancyRun) -> dict[str, Any]:
    """Acceptance criterion 2: the winner's cross-validated F1 is at least 0.70."""
    per_representation = {
        item.name: {
            "cv_f1": item.calibration.cv_f1,
            "f1": item.calibration.f1_at_tau,
            "lower": item.interval.lower,
            "upper": item.interval.upper,
            "met": item.calibration.cv_f1 is not None
            and item.calibration.cv_f1 >= CRITERION_TWO_MIN_F1,
        }
        for item in run.representations
    }
    winner = run.stage_two.winner
    chosen = per_representation.get(winner, {}) if winner is not None else {}
    return {
        "min_f1": CRITERION_TWO_MIN_F1,
        "judged_on": CRITERION_TWO_F1,
        "representation": winner,
        "cv_f1": chosen.get("cv_f1"),
        "f1": chosen.get("f1"),
        "lower": chosen.get("lower"),
        "upper": chosen.get("upper"),
        "met": chosen.get("met"),
        "per_representation": per_representation,
    }


def headline(run: RedundancyRun) -> str:
    """The sentence that says whether T19's provisional winner is confirmed."""
    stage, provisional = run.stage_two, run.provisional_configuration
    if stage.decision is None:
        return (
            f"Stage 2 (D30 E3) is not closed: stage 2 is undefined, so the provisional winner "
            f"{provisional} stays provisional (a decision for the user)"
        )
    at = f"closed at step {stage.decision.deciding_step} ({stage.decision.deciding_criterion})"
    winner = run.winner
    assert winner is not None
    if stage.outcome() == "confirmed":
        return f"Stage 2 (D30 E3) is {at}: the provisional winner {provisional} is confirmed"
    return (
        f"Stage 2 (D30 E3) is {at}: the provisional winner {provisional} changes to "
        f"{winner.configuration}"
    )


def _findings(run: RedundancyRun, one: Mapping[str, Any], two: Mapping[str, Any]) -> list[str]:
    stage = run.stage_two
    findings = [headline(run)]
    if stage.undefined_reason is not None:
        findings.append(f"stage 2: {stage.undefined_reason}")
    findings += [f"stage 2: {item.reason}" for item in stage.sensitivities if item.flipped]
    if one["met"] is False:
        findings.append(f"criterion 1 is not met: {one['reason']}")
    if two["met"] is False:
        findings.append(
            f"criterion 2 is not met: the winner's cross-validated F1, {two['cv_f1']:.4f}, is "
            f"below {CRITERION_TWO_MIN_F1:.2f}"
        )
    duplicates = run.counts()["known_duplicates"]
    if duplicates:
        findings.append(
            f"O09: {duplicates} known true duplicate{'s are' if duplicates > 1 else ' is'} among "
            f"the labeled pairs, flagged in {PAIR_SCORES_FILE_NAME}"
        )
    return findings


def _consequences(run: RedundancyRun) -> list[str]:
    provisional, winner = run.provisional_configuration, run.winner
    outcome = run.stage_two.outcome()
    if winner is None:
        return [
            f"stage 2 is undefined ({run.stage_two.undefined_reason}): how to close it is a "
            f"decision for the user; until then {provisional} stays provisional and T23 to T26 "
            "stand on it"
        ]
    if outcome == "confirmed":
        return [
            f"{provisional} is the winner by D30's full rule: T23 to T26, built on it, stand; "
            "T19's records keep provisional true, and these records close it"
        ]
    return [
        f"the winner changes from {provisional} to {winner.configuration}: T23 to T26 and the "
        "gap run's cuts (D32 (3) and (4)) are rebuilt on it before they are used, since T19's "
        "records name the provisional winner",
        "criterion 1's NPMI part needs the baseline's NPMI at the new winner's matched K, which "
        "T19's records do not hold",
    ]


# --- Text -----------------------------------------------------------------------------------


def render_summary(run: RedundancyRun, *, git_commit: str | None) -> str:
    """Return ``summary.md``: stage 2 against T19's provisional winner, τ and F1 per
    representation, the paired comparisons, criteria 1 and 2, the labels and the folds. It holds
    no doc code and no pair id."""
    counts, stage = run.counts(), run.stage_two
    one, two = _criterion_one(run), _criterion_two(run)
    names = [_short(name) for name in run.names]
    lines = [
        f"# T22 on snapshot {run.snapshot_id}: the threshold and stage 2 of the winner",
        "",
        "The direct-redundancy threshold τ of each representation and stage 2 of the winner "
        f"(D30 E3, UD3), run at commit `{git_commit or 'unknown'}` on the "
        f"{counts['labeled_pairs']} pairs of T20a's labeled set (the {counts['pilot_pairs']} "
        "pilot pairs left out) with their consensus labels; a pair is redundant (1) when its "
        f"consensus level is {REDUNDANT} "
        "(D31 (b), D34 (3)). Which theses form each pair is in the gitignored records only.",
        "",
        "## Stage 2 of the winner",
        "",
        f"**{headline(run)}.**",
        "",
        _winners_line(run),
        "",
        _row(("Step", "Criterion", names[0], names[1], "Outcome", "Reason")),
        _rule(6),
        *_step_lines(run),
        "",
        *(
            [f"Stage 2 is undefined: {stage.undefined_reason}.", ""]
            if stage.undefined_reason
            else []
        ),
        *(f"Sensitivity: {item.reason}." for item in stage.sensitivities),
        "",
        *(f"- {item}" for item in _consequences(run)),
        "",
        "## Threshold and F1 per representation",
        "",
        f"τ by {run.cv.folds}-fold cross-validation, each fold's τ the max-F1 point of its "
        "training pairs; the CV F1 pools the held-out counts. The F1 at τ is that of all the "
        f"labeled pairs at the final τ, with its {run.bootstrap.interval:.0%} bootstrap interval "
        f"({run.bootstrap.resamples:,} resamples, seed {run.bootstrap.seed}).",
        "",
        _row(("Representation", "τ", "CV F1", "Mean fold F1", "F1 at τ", "Interval", "P", "R")),
        _rule(8),
        *(
            _row(
                (
                    _short(item.name),
                    f"{item.calibration.tau:.4f}",
                    _number(item.calibration.cv_f1),
                    _number(item.calibration.mean_fold_f1),
                    _number(item.calibration.f1_at_tau),
                    _bounds(item.interval.lower, item.interval.upper),
                    _number(item.calibration.at_tau.precision),
                    _number(item.calibration.at_tau.recall),
                )
            )
            for item in run.representations
        ),
        "",
        "## Paired comparisons",
        "",
        "ΔF1 = F1(first) − F1(second) over the same pairs, each at its own final τ, with its "
        "paired percentile interval; an interval that holds or touches 0 is a tie.",
        "",
        _row(("Purpose", "First", "Second", "F1", "F1", "ΔF1", "Interval", "Verdict")),
        _rule(8),
        *(
            _row(
                (
                    purpose,
                    _short(comparison.first_name),
                    _short(comparison.second_name),
                    _number(comparison.difference.first_f1),
                    _number(comparison.difference.second_f1),
                    _number(comparison.difference.difference),
                    _bounds(comparison.difference.lower, comparison.difference.upper),
                    comparison.difference.verdict or "undefined",
                )
            )
            for purpose, comparison in run.comparisons
        ),
        "",
        "## Acceptance criteria",
        "",
        f"- Criterion 1 (UD1): {_met(one['met'])}; {one['reason']}.",
        f"- Criterion 2: {_met(two['met'])}; the winner's cross-validated F1 is "
        f"{_number(two['cv_f1'])} against {CRITERION_TWO_MIN_F1:.2f} (the F1 at τ is "
        f"{_number(two['f1'])}, interval {_bounds(two['lower'], two['upper'])}).",
        "",
        "## Labels",
        "",
        f"{REDUNDANT} {counts['redundant']}, {RELATED} {counts['related']}, {UNRELATED} "
        f"{counts['unrelated']}; by agreement {counts['agreement']}, by adjudication "
        f"{counts['adjudication']}. Known true duplicates among the labeled pairs (O09): "
        f"{counts['known_duplicates']}.",
        "",
        "## Folds",
        "",
        _row(("Representation", "Fold", "Train pairs", "Train positives", "τ", "Test F1")),
        _rule(6),
        *(
            _row(
                (
                    _short(item.name),
                    str(outcome.fold),
                    str(outcome.train_pairs),
                    str(outcome.train_positives),
                    f"{outcome.tau:.4f}",
                    _number(outcome.test_f1),
                )
            )
            for item in run.representations
            for outcome in item.calibration.folds
        ),
        "",
        f"Declared values: {_settings_text(run)}.",
    ]
    return "\n".join(lines)


def describe(run: RedundancyRun, *, seconds: float) -> str:
    """Describe a run in counts, thresholds and outcomes, in ASCII and without thesis text."""
    counts, stage = run.counts(), run.stage_two
    lines = [
        f"Labels: {run.labels_path}",
        f"  {counts['labeled_pairs']} labeled pairs ({counts['pilot_pairs']} pilot pairs left "
        f"out): {REDUNDANT} {counts['redundant']}, {RELATED} {counts['related']}, {UNRELATED} "
        f"{counts['unrelated']}; agreement {counts['agreement']}, adjudication "
        f"{counts['adjudication']}; known duplicates (O09) {counts['known_duplicates']}",
        *(f"  {name} SHA-256 {digest}" for name, digest in run.input_sha256.items()),
        f"Threshold per representation ({run.cv.folds}-fold cross-validation, "
        f"{run.bootstrap.interval:.0%} bootstrap interval of the F1 at tau):",
        *(
            f"  {_short(item.name)}: tau {item.calibration.tau:.4f}, CV F1 "
            f"{_number(item.calibration.cv_f1)} (mean of the folds "
            f"{_number(item.calibration.mean_fold_f1)}), F1 at tau "
            f"{_number(item.calibration.f1_at_tau)} "
            f"{_bounds(item.interval.lower, item.interval.upper)}, precision "
            f"{_number(item.calibration.at_tau.precision)}, recall "
            f"{_number(item.calibration.at_tau.recall)}"
            for item in run.representations
        ),
        "Paired comparisons (dF1 = F1(first) - F1(second), paired interval):",
        *(
            f"  {purpose}: {_short(comparison.first_name)} - {_short(comparison.second_name)} "
            f"{_number(comparison.difference.difference)} "
            f"{_bounds(comparison.difference.lower, comparison.difference.upper)}, "
            f"{comparison.difference.verdict or 'undefined'}"
            for purpose, comparison in run.comparisons
        ),
        _console(headline(run)),
        *(
            f"  step {step.step} {step.criterion}: {step.outcome} ({_console(step.reason)})"
            for step in (stage.decision.steps if stage.decision else ())
        ),
        *([f"  {_console(stage.undefined_reason)}"] if stage.undefined_reason else []),
        *(f"  {_console(item.reason)}" for item in stage.sensitivities),
        f"Criterion 1: {_met(_criterion_one(run)['met'])}; criterion 2: "
        f"{_met(_criterion_two(run)['met'])}",
        f"Seconds: {seconds:.1f} ("
        + ", ".join(f"{name} {value:.1f}" for name, value in run.timings.items())
        + ")",
    ]
    return "\n".join(lines)


def _winners_line(run: RedundancyRun) -> str:
    provisional = run.experiment.provisional_winner
    text = (
        f"T19's provisional winner: {provisional['configuration']}, "
        f"{_short(str(provisional['representation']))} with {provisional['algorithm']} at "
        f"{provisional['parameter']}; "
    )
    winner = run.winner
    if winner is None:
        return text + "the full rule names no winner."
    return text + (
        f"the full rule's winner: {winner.configuration}, {_short(winner.representation)} with "
        f"{winner.algorithm} at {winner.parameter}."
    )


def _step_lines(run: RedundancyRun) -> list[str]:
    stage = run.stage_two
    steps = stage.decision.steps if stage.decision is not None else stage.provisional.steps[:1]
    lines = [
        _row((str(step.step), step.criterion, _cell(step.first), _cell(step.second), step.outcome,
              step.reason))
        for step in steps
    ]  # fmt: skip
    if stage.decision is None:
        f1 = stage.f1
        lines.append(
            _row(("2", "f1", _number(f1.first_f1), _number(f1.second_f1), f1.outcome, f1.reason))
        )
    return lines


def _cell(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{key} {_number(item)}" for key, item in value.items())
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value) or "none failed"
    if isinstance(value, float):
        return _number(value)
    return "" if value is None else _short(str(value))


def _settings_text(run: RedundancyRun) -> str:
    cv, bootstrap = run.cv, run.bootstrap
    return (
        f"cv_folds {cv.folds}, cv_seed {cv.seed}, cv_stratified {str(cv.stratified).lower()}, "
        f"f1_tie {cv.f1_tie}, final_tau {cv.final_tau}; bootstrap_resamples "
        f"{bootstrap.resamples}, bootstrap_seed {bootstrap.seed}, interval {bootstrap.interval:g}; "
        f"criterion 2 at {CRITERION_TWO_MIN_F1:.2f} on the {CRITERION_TWO_F1}"
    )


def _number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _bounds(lower: float | None, upper: float | None) -> str:
    return "undefined" if lower is None or upper is None else f"[{lower:.4f}, {upper:.4f}]"


def _met(value: bool | None) -> str:
    return {True: "met", False: "not met", None: "pending"}[value]


def _short(name: str) -> str:
    """Return a model id without its organization, such as ``paraphrase-multilingual-...``."""
    return name.rsplit("/", 1)[-1]


def _console(text: str) -> str:
    """Return ``text`` with its few non-ASCII symbols spelled out, for the console."""
    for symbol, spelled in (("ΔF1", "dF1"), ("τ", "tau"), ("≥", ">="), ("−", "-")):
        text = text.replace(symbol, spelled)
    return text


def _row(cells: Sequence[str]) -> str:
    return f"| {' | '.join(cell.replace('|', '/') for cell in cells)} |"


def _rule(columns: int) -> str:
    return "|" + "---|" * columns


# --- The command ----------------------------------------------------------------------------


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.redundancy.run_redundancy",
        description=(
            "Run T22 on the labeled set of a snapshot once: read T20a's drawn pairs, their "
            "consensus labels (labeling.records_dir/<snapshot>/labels/consensus.parquet, one per "
            "labeled pair) and T19's records; calibrate tau by cross-validation and bootstrap "
            "the F1 of each representation; compare SBERT with TF-IDF; and close stage 2 of the "
            "winner by D30's full rule, confirming or changing T19's provisional winner. It "
            "writes the aggregates under results/experiments/<snapshot>/t22/ (tracked by git) "
            "and the pair scores under data/processed/<snapshot>/experiments/t22/ (gitignored). "
            "An existing run is never overwritten."
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
    parser.add_argument(
        "--labels",
        type=Path,
        metavar="PATH",
        help=(
            "consensus labels to read instead of the contract's file, "
            "labeling.records_dir/<snapshot>/labels/consensus.parquet; for dry runs only, such "
            "as a check with synthetic labels (default: the contract's file)"
        ),
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[AppConfig, Path, str], DrawInputs] = load_inputs,
    compare: Compare = compare_representations,
    git: Callable[[Path, str], GitState] = read_git_state,
    clock: Callable[[], datetime] = _now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and 130
    after an interruption."""
    arguments = _parse_arguments(argv)
    root = arguments.project_root.resolve()
    try:
        if arguments.labels is not None and not arguments.dry_run:
            raise RedundancyRunError(
                "--labels is for dry runs only: a real run reads the contract's file, "
                "labeling.records_dir/<snapshot>/labels/consensus.parquet"
            )
        config = load_config(root / CONFIG_PATH)
        paths = config.paths.resolve_against(root)
        snapshot_id = check_snapshot_id(
            arguments.snapshot_id or latest_snapshot_id(paths.data_dir / RAW_DIR_NAME)
        )
        results = results_dir(paths.results_dir, snapshot_id)
        processed = processed_dir(paths.data_dir, snapshot_id)
        labels = (
            arguments.labels.resolve()
            if arguments.labels is not None
            else consensus_labels_path(draw_pairs.records_dir(config, root, snapshot_id))
        )
        relative = results.relative_to(root).as_posix()
        if arguments.dry_run:
            return _dry_run(
                config, root, snapshot_id, labels, results, processed,
                load=load, compare=compare, git=git, timer=timer,
            )  # fmt: skip
        state = git(root, relative)
        if state.dirty:
            raise RedundancyRunError(
                f"the git tree has {len(state.dirty)} changed or untracked paths outside "
                f"{relative}; commit them first, since the manifest records the commit"
            )
        _refuse_an_earlier_run(snapshot_id, results, processed)
        results.mkdir(parents=True, exist_ok=True)
        busy = RedundancyRunError(f"another run of T22 holds the lock of {relative}; let it finish")
        with exclusive_run(results, busy):
            _refuse_an_earlier_run(snapshot_id, results, processed)
            started_at, started = clock(), timer()
            run = compute(
                config, root, snapshot_id, labels_path=labels, load=load, compare=compare,
                report=_progress, timer=timer,
            )  # fmt: skip
            summary = render_summary(run, git_commit=state.commit)
            _progress(f"writing the records, {MANIFEST_FILE_NAME} last")
            finished_at, seconds = clock(), timer() - started
            manifest = write_outputs(
                results=results,
                processed=processed,
                representations=representation_rows(run),
                folds=fold_rows(run),
                curves=curve_rows(run),
                comparisons=comparison_rows(run),
                pair_scores=pair_score_rows(run),
                selection=selection_record(run),
                summary=summary,
                manifest=_manifest(
                    run,
                    git_commit=state.commit,
                    started_at=started_at,
                    finished_at=finished_at,
                    seconds=seconds,
                ),
                sensitive=run.guarded,
            )
    except KeyboardInterrupt:
        print(
            f"interrupted: a run is complete only once {MANIFEST_FILE_NAME} is written; move the "
            "partial records away, if any, and run the command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(f"Snapshot {snapshot_id}: commit {manifest.git_commit}")
    print(describe(run, seconds=manifest.seconds))
    print(
        f"Wrote {relative}/ and the gitignored {PAIR_SCORES_FILE_NAME}, {MANIFEST_FILE_NAME} last"
    )
    return 0


def _refuse_an_earlier_run(snapshot_id: str, results: Path, processed: Path) -> None:
    earlier = existing_run(results, processed)
    if earlier is not None:
        raise FileExistsError(
            f"a run of T22 on snapshot {snapshot_id} already exists ({earlier}); its records are "
            "never overwritten: move them away first to run again"
        )


def _dry_run(
    config: AppConfig,
    root: Path,
    snapshot_id: str,
    labels: Path,
    results: Path,
    processed: Path,
    *,
    load: Callable[[AppConfig, Path, str], DrawInputs],
    compare: Compare,
    git: Callable[[Path, str], GitState],
    timer: Callable[[], float],
) -> int:
    relative = results.relative_to(root).as_posix()
    try:
        state: GitState | None = git(root, relative)
    except OSError:
        state = None
    started = timer()
    run = compute(
        config, root, snapshot_id, labels_path=labels, load=load, compare=compare,
        report=_progress, timer=timer,
    )  # fmt: skip
    texts = tracked_texts(
        representations=representation_rows(run),
        folds=fold_rows(run),
        curves=curve_rows(run),
        comparisons=comparison_rows(run),
        selection=selection_record(run),
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
    earlier = existing_run(results, processed)
    print(
        "\n".join(
            [
                f"Dry run of T22 on snapshot {snapshot_id}: everything is computed and nothing is "
                "written",
                f"Git: {tree}",
                *(
                    []
                    if earlier is None
                    else [
                        f"A run of this snapshot exists ({earlier.name}); a real run refuses to "
                        "overwrite it"
                    ]
                ),
                describe(run, seconds=seconds),
                f"Privacy guard: {len(texts)} tracked outputs checked against "
                f"{len(run.guarded):,} strings (the snapshot's, the theses' doc codes and the "
                "drawn pairs' ids), no hit (the manifest is checked when it is written)",
                f"Command of the real run: {RUN_COMMAND} --snapshot-id {snapshot_id}",
            ]
        )
    )
    return 0


def _manifest(
    run: RedundancyRun,
    *,
    git_commit: str,
    started_at: datetime,
    finished_at: datetime,
    seconds: float,
) -> RedundancyManifest:
    cv, bootstrap, winner = run.cv, run.bootstrap, run.winner
    return RedundancyManifest(
        snapshot_id=run.snapshot_id,
        git_commit=git_commit,
        settings={
            "cv": {
                "folds": cv.folds,
                "seed": cv.seed,
                "stratified": cv.stratified,
                "f1_tie": cv.f1_tie,
                "final_tau": cv.final_tau,
            },
            "bootstrap": {
                "resamples": bootstrap.resamples,
                "seed": bootstrap.seed,
                "interval": bootstrap.interval,
            },
            "criterion_two": {"min_f1": CRITERION_TWO_MIN_F1, "judged_on": CRITERION_TWO_F1},
            "redundant_level": REDUNDANT,
            "embedding_document_vector": run.config.embedding.document_vector,
            "tfidf_document_vector": run.config.tfidf.document_vector,
        },
        representations=run.names,
        labeled_set={
            **run.draw.sha256,
            run.labels_path.name: _sha256(run.labels_path),
        },
        experiment=dict(run.experiment.sha256),
        inputs=run.input_sha256,
        provisional_winner=run.provisional_configuration,
        stage_two=run.stage_two.status,
        outcome=run.stage_two.outcome(),
        winner=None if winner is None else winner.configuration,
        counts=run.counts(),
        versions=library_versions(),
        cpu=cpu_record(),
        files={},
        data_files={},
        started_at=started_at,
        finished_at=finished_at,
        seconds=round(max(seconds, 0.0), 3),
        timings=run.timings,
    )


# --- Helpers --------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """``value`` as plain JSON data: tuples become lists."""
    return json.loads(json.dumps(value))


def _draw_order(row: PairRow) -> int:
    return row.draw_order


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _digest(path.read_bytes())


if __name__ == "__main__":
    raise SystemExit(main())
