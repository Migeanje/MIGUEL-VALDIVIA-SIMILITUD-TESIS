"""The records of T22's run on the labeled set, and the privacy guard they pass before any is
written.

Under ``results/experiments/<snapshot_id>/t22/``, which git tracks, beside T19's ``t19/``,
aggregates only, with no doc code, pair id or thesis text:

- ``representations.parquet``: one row per representation (each embedding model, then
  ``tfidf``): the final τ, the counts, precision, recall and F1 of all the labeled pairs at it
  with the bootstrap interval of that F1, and the cross-validated counts and F1;
- ``folds.parquet``: one row per representation and fold: the fold's τ, chosen on its training
  pairs, and its held-out pairs counted at it;
- ``curves.parquet``: the precision-recall curve of all the labeled pairs, one row per point
  (a distinct score) of each representation, the selected τ flagged;
- ``comparisons.parquet``: each paired comparison, ΔF1 with its bootstrap interval and verdict:
  the two embedding models (stage 2 of the winner) and each model against TF-IDF (acceptance
  criterion 1);
- ``selection.json``: stage 2 of the winner closed by the full rule, against T19's provisional
  winner, with every step and its reason, the sensitivity, criteria 1 and 2 and the findings;
- ``summary.md``: the same, readable;
- ``manifest.json``, written last, as the commit point: the provenance, the values used and the
  SHA-256 of every other file.

Under the gitignored ``data/processed/<snapshot_id>/experiments/t22/``, as T19 keeps its
per-document records there: ``pair_scores.parquet``, one row per labeled pair with its pair id,
its labels, its fold, its scores and its prediction under each representation.

Every tracked output, the tables as the JSON of their rows, passes the gap run's
:func:`~thematic_redundancy.gaps.results_store.check_privacy` before anything is written; a hit
refuses every write.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from thematic_redundancy.gaps.results_store import check_privacy
from thematic_redundancy.shared.filesystem import write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore
from thematic_redundancy.topic_map.results_store import EXPERIMENTS_DIR_NAME, PROCESSED_DIR_NAME

T22_DIR_NAME = "t22"
REPRESENTATIONS_FILE_NAME = "representations.parquet"
FOLDS_FILE_NAME = "folds.parquet"
CURVES_FILE_NAME = "curves.parquet"
COMPARISONS_FILE_NAME = "comparisons.parquet"
SELECTION_FILE_NAME = "selection.json"
SUMMARY_FILE_NAME = "summary.md"
MANIFEST_FILE_NAME = "manifest.json"
PAIR_SCORES_FILE_NAME = "pair_scores.parquet"
TRACKED_FILE_NAMES = (
    REPRESENTATIONS_FILE_NAME,
    FOLDS_FILE_NAME,
    CURVES_FILE_NAME,
    COMPARISONS_FILE_NAME,
    SELECTION_FILE_NAME,
    SUMMARY_FILE_NAME,
)
"""The tracked outputs but the manifest, in the order they are written."""


def results_dir(results_root: Path, snapshot_id: str) -> Path:
    """Return ``<results_root>/experiments/<snapshot_id>/t22``."""
    return results_root / EXPERIMENTS_DIR_NAME / snapshot_id / T22_DIR_NAME


def processed_dir(data_root: Path, snapshot_id: str) -> Path:
    """Return ``<data_root>/processed/<snapshot_id>/experiments/t22``."""
    return data_root / PROCESSED_DIR_NAME / snapshot_id / EXPERIMENTS_DIR_NAME / T22_DIR_NAME


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class RepresentationRow(_Row):
    """One row of ``representations.parquet``: one representation over the labeled pairs."""

    representation: str
    pairs: int
    positives: int
    """The pairs labeled redundant (1)."""
    tau: float
    """The final τ: a pair is predicted redundant when its score is at least τ."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    """The F1 of all the labeled pairs at :attr:`tau`, the one the bootstrap resamples."""
    f1_lower: float | None
    f1_upper: float | None
    undefined_resamples: int
    cv_true_positives: int
    """The held-out counts, summed over the folds."""
    cv_false_positives: int
    cv_false_negatives: int
    cv_true_negatives: int
    cv_precision: float | None
    cv_recall: float | None
    cv_f1: float | None
    """The cross-validated F1: the F1 of the summed held-out counts."""
    mean_fold_f1: float | None


class FoldRow(_Row):
    """One row of ``folds.parquet``: one fold of one representation's cross-validation."""

    representation: str
    fold: int
    train_pairs: int
    train_positives: int
    tau: float
    """The max-F1 point of the fold's training pairs."""
    train_f1: float
    test_pairs: int
    test_positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    test_f1: float | None


class CurveRow(_Row):
    """One row of ``curves.parquet``: one point of a representation's precision-recall curve
    over all the labeled pairs, the pairs scored at least :attr:`tau`."""

    representation: str
    point: int
    """0 for the highest threshold, then down."""
    tau: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float | None
    """Always defined: every point predicts at least one pair."""
    recall: float | None
    f1: float
    selected: bool
    """Whether :attr:`tau` is the final τ."""


class ComparisonRow(_Row):
    """One row of ``comparisons.parquet``: ΔF1 = F1(first) − F1(second) over the labeled pairs,
    each at its own final τ, with its paired bootstrap interval and verdict."""

    purpose: str
    """``stage_two`` (the two embedding models) or ``criterion_one`` (a model and TF-IDF)."""
    first: str
    second: str
    first_tau: float
    second_tau: float
    first_f1: float | None
    second_f1: float | None
    difference: float | None
    lower: float | None
    upper: float | None
    verdict: str | None
    undefined_resamples: int


class PairScoreRow(_Row):
    """One row of the gitignored ``pair_scores.parquet``: one labeled pair, joined to T20a's
    ``pairs.parquet`` by :attr:`pair_id`."""

    pair_id: str
    label_3: str
    label: int
    source: str
    fold: int
    """The fold that holds the pair out; the folds are the same for every representation."""
    representations: tuple[str, ...]
    scores: tuple[float, ...]
    """The pair's cosine under each representation, in the order of :attr:`representations`."""
    predicted: tuple[bool, ...]
    """Whether each representation predicts the pair redundant at its final τ."""
    known_duplicate: bool


class RedundancyManifest(BaseModel):
    """Content of ``manifest.json``: provenance, values, outcome, counts and time, without a doc
    code or a pair id."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    snapshot_id: str
    git_commit: str
    settings: dict[str, Any]
    """The cross-validation and bootstrap values used, and the reading of criterion 2."""
    representations: tuple[str, ...]
    labeled_set: dict[str, str]
    """SHA-256 of T20a's draw manifest, of its ``pairs.parquet`` and of the consensus labels."""
    experiment: dict[str, str]
    """SHA-256 of T19's ``manifest.json`` and ``selection.json``."""
    inputs: dict[str, str]
    """SHA-256 of the input manifests read, the snapshot's metadata and the configuration."""
    provisional_winner: str
    """T19's provisional winner, a configuration such as ``C4``."""
    stage_two: str
    """``closed`` or ``undefined``."""
    outcome: str
    """``confirmed``, ``changed`` or ``undefined``, against :attr:`provisional_winner`."""
    winner: str | None
    """The winning configuration by the full rule; ``None`` when stage 2 is undefined."""
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
    representations: Sequence[RepresentationRow],
    folds: Sequence[FoldRow],
    curves: Sequence[CurveRow],
    comparisons: Sequence[ComparisonRow],
    selection: Mapping[str, Any],
    summary: str,
) -> dict[str, str]:
    """Return the text of each tracked output but the manifest, by file name: a table as the
    JSON of its rows, one per line."""
    return {
        REPRESENTATIONS_FILE_NAME: _lines(representations),
        FOLDS_FILE_NAME: _lines(folds),
        CURVES_FILE_NAME: _lines(curves),
        COMPARISONS_FILE_NAME: _lines(comparisons),
        SELECTION_FILE_NAME: selection_text(selection),
        SUMMARY_FILE_NAME: summary.rstrip("\n"),
    }


def selection_text(selection: Mapping[str, Any]) -> str:
    """Return ``selection.json`` as it is written: indented, keys sorted, text as it is."""
    return json.dumps(selection, indent=2, sort_keys=True, ensure_ascii=False)


def write_outputs(
    *,
    results: Path,
    processed: Path,
    representations: Sequence[RepresentationRow],
    folds: Sequence[FoldRow],
    curves: Sequence[CurveRow],
    comparisons: Sequence[ComparisonRow],
    pair_scores: Sequence[PairScoreRow],
    selection: Mapping[str, Any],
    summary: str,
    manifest: RedundancyManifest,
    sensitive: frozenset[str],
    store: TableStore | None = None,
    write_text: Callable[[Path, str], None] = write_text_atomically,
) -> RedundancyManifest:
    """Write the records once every tracked output passes the privacy guard: the gitignored
    pair scores first, then the tracked tables, ``selection.json`` and ``summary.md``, then
    ``manifest.json`` last, and return the manifest with the SHA-256 of every file written.

    ``manifest`` is complete but for :attr:`~RedundancyManifest.files` and
    :attr:`~RedundancyManifest.data_files`, which this fills with the SHA-256 of the files
    written.

    Raises:
        PrivacyError: if a tracked output, the manifest included, holds a string of
            ``sensitive``; nothing is written when one of the outputs but the manifest does.
        OSError: if writing fails.
    """
    store = store or ParquetTableStore()
    texts = tracked_texts(
        representations=representations,
        folds=folds,
        curves=curves,
        comparisons=comparisons,
        selection=selection,
        summary=summary,
    )
    check_privacy({**texts, MANIFEST_FILE_NAME: manifest.model_dump_json(indent=2)}, sensitive)
    results.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    store.write(pair_scores, processed / PAIR_SCORES_FILE_NAME, model_cls=PairScoreRow)
    store.write(representations, results / REPRESENTATIONS_FILE_NAME, model_cls=RepresentationRow)
    store.write(folds, results / FOLDS_FILE_NAME, model_cls=FoldRow)
    store.write(curves, results / CURVES_FILE_NAME, model_cls=CurveRow)
    store.write(comparisons, results / COMPARISONS_FILE_NAME, model_cls=ComparisonRow)
    write_text(results / SELECTION_FILE_NAME, texts[SELECTION_FILE_NAME])
    write_text(results / SUMMARY_FILE_NAME, texts[SUMMARY_FILE_NAME])
    final = manifest.model_copy(
        update={
            "files": {name: _sha256(results / name) for name in TRACKED_FILE_NAMES},
            "data_files": {PAIR_SCORES_FILE_NAME: _sha256(processed / PAIR_SCORES_FILE_NAME)},
        }
    )
    text = final.model_dump_json(indent=2)
    check_privacy({MANIFEST_FILE_NAME: text}, sensitive)
    # The commit point: without it, the run is not complete.
    write_text(results / MANIFEST_FILE_NAME, text)
    return final


def existing_run(results: Path, processed: Path) -> Path | None:
    """Return a record of an earlier run, ``pair_scores.parquet`` or ``manifest.json``, if any."""
    for path in (processed / PAIR_SCORES_FILE_NAME, results / MANIFEST_FILE_NAME):
        if path.exists():
            return path
    return None


def _lines(rows: Sequence[BaseModel]) -> str:
    return "\n".join(row.model_dump_json() for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
