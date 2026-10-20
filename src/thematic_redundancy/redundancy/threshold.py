"""The direct-redundancy threshold τ (T22, D15, D30 E3): F1 at a threshold, the max-F1 point
of the precision-recall curve, and τ chosen by k-fold cross-validation.

A labeled pair has a similarity score and a binary label, 1 for redundant. It is **predicted
redundant when its score is at least τ** (``score >= tau``), the rule of every point of a
precision-recall curve as scikit-learn's ``precision_recall_curve`` draws it, so pairs with
equal scores always fall on the same side of τ.

**F1** is ``2·TP / (2·TP + FP + FN)``, the harmonic mean of precision and recall wherever both
are defined, and 0 when positives exist and none is found. It is undefined (``None``) only
when there is no positive label and no positive prediction, as the package's metrics leave a
value undefined for a valid input.

**The max-F1 point** (:func:`max_f1_threshold`): the candidate thresholds are the distinct
scores, from the highest down; the point with the highest F1 gives τ, a score of the pairs.
When several points share it, ``f1_tie`` picks the highest or the lowest of their thresholds;
the plan does not fix it, so the caller declares it. With no positive pair, τ is undefined.

**Cross-validation** (:func:`calibrate_threshold`, D15: "τ is chosen by 5-fold
cross-validation"):

1. The pairs are split into ``folds`` folds by scikit-learn's ``StratifiedKFold`` (each fold
   keeps the share of positive pairs) or ``KFold``, both with ``shuffle=True`` and
   ``random_state=seed``. The split depends on the labels and the seed only, never on the
   scores, so two representations scored over the same pairs share their folds.
2. For each fold, τ is the max-F1 point of the other folds' pairs (the training pairs), and
   the fold's own pairs (the held-out pairs) are counted at that τ. A training split with no
   positive pair has no τ, and the calibration stops.
3. The **CV F1** is the F1 of the held-out counts summed over the folds; each pair is held out
   once, so they add up to the labeled set. The mean of the folds' F1 is reported beside it,
   and is undefined when a fold's F1 is.
4. The **final τ** is, by ``final_tau``, the max-F1 point of all the pairs (``"all_pairs"``)
   or the mean of the folds' τ (``"fold_mean"``); the plan does not fix it either. The counts
   and F1 of all the pairs at that τ are the in-sample values that the bootstrap of
   :mod:`~thematic_redundancy.redundancy.comparison` resamples.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

F1Tie = Literal["highest_tau", "lowest_tau"]
FinalTau = Literal["all_pairs", "fold_mean"]
F1_TIES: tuple[F1Tie, ...] = ("highest_tau", "lowest_tau")
FINAL_TAUS: tuple[FinalTau, ...] = ("all_pairs", "fold_mean")

SEED_LIMIT = 2**32
"""Seeds lie in ``[0, 2**32)``, as the configuration's seeds do."""

Scores = Sequence[float] | np.ndarray
Labels = Sequence[int] | Sequence[bool] | np.ndarray


@dataclass(frozen=True)
class ConfusionCounts:
    """Pairs counted by their label and their prediction at a threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float | None:
        """``TP / (TP + FP)``; ``None`` when no pair is predicted redundant."""
        predicted = self.true_positives + self.false_positives
        return None if predicted == 0 else self.true_positives / predicted

    @property
    def recall(self) -> float | None:
        """``TP / (TP + FN)``; ``None`` when no pair is labeled redundant."""
        positives = self.true_positives + self.false_negatives
        return None if positives == 0 else self.true_positives / positives

    @property
    def f1(self) -> float | None:
        """``2·TP / (2·TP + FP + FN)``; ``None`` with no positive label and no prediction."""
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return None if denominator == 0 else 2 * self.true_positives / denominator


@dataclass(frozen=True)
class CurvePoint:
    """One point of the precision-recall curve: the pairs scored at least ``tau``."""

    tau: float
    counts: ConfusionCounts
    f1: float
    """Always defined: every point predicts at least one pair."""


@dataclass(frozen=True)
class MaxF1Threshold:
    """The max-F1 point of the precision-recall curve of some labeled pairs."""

    tau: float | None
    f1: float | None
    counts: ConfusionCounts | None
    curve: tuple[CurvePoint, ...]
    """Every point, one per distinct score, from the highest threshold down."""
    undefined_reason: Literal["no_positive_pairs"] | None


@dataclass(frozen=True)
class CvSettings:
    """How :func:`calibrate_threshold` chooses τ; every value is the caller's declaration."""

    folds: int
    seed: int
    """The ``random_state`` of the fold split."""
    stratified: bool
    """``StratifiedKFold`` when true, else ``KFold``."""
    f1_tie: F1Tie
    final_tau: FinalTau

    def __post_init__(self) -> None:
        if not _is_int(self.folds) or self.folds < 2:
            raise ValueError(f"folds must be an integer of at least 2, not {self.folds!r}")
        if not _is_int(self.seed) or not 0 <= self.seed < SEED_LIMIT:
            raise ValueError(f"seed must be an integer in [0, 2**32), not {self.seed!r}")
        if not isinstance(self.stratified, bool):
            raise ValueError(f"stratified must be a boolean, not {self.stratified!r}")
        _check_f1_tie(self.f1_tie)
        if self.final_tau not in FINAL_TAUS:
            raise ValueError(f"final_tau must be one of {FINAL_TAUS}, not {self.final_tau!r}")


@dataclass(frozen=True)
class FoldOutcome:
    """One fold: τ chosen on the training pairs, and the held-out pairs counted at it."""

    fold: int
    train_pairs: int
    train_positives: int
    tau: float
    train_f1: float
    """The F1 of the training pairs at ``tau``, their maximum."""
    test: ConfusionCounts
    test_f1: float | None
    """``None`` when the held-out pairs hold no positive and none is predicted."""


@dataclass(frozen=True)
class ThresholdCalibration:
    """τ chosen by cross-validation, with the held-out and the in-sample F1."""

    settings: CvSettings
    pairs: int
    positives: int
    fold_of_pair: tuple[int, ...]
    """The fold that holds each pair out, in the pairs' order."""
    folds: tuple[FoldOutcome, ...]
    pooled: ConfusionCounts
    """The held-out counts summed over the folds."""
    cv_f1: float | None
    """The F1 of :attr:`pooled`: the cross-validated F1."""
    mean_fold_f1: float | None
    """The mean of the folds' F1; ``None`` when one of them is undefined."""
    tau: float
    """The final τ, by ``settings.final_tau``."""
    at_tau: ConfusionCounts
    """All the pairs counted at :attr:`tau`."""
    f1_at_tau: float | None


def confusion_at(scores: Scores, labels: Labels, *, tau: float) -> ConfusionCounts:
    """Return the counts of the pairs at threshold ``tau``: a pair is predicted redundant when
    its score is at least ``tau``.

    Raises:
        ValueError: if the pairs are invalid (see :func:`labeled_pairs`) or ``tau`` is not
            finite.
    """
    score_array, label_array = labeled_pairs(scores, labels)
    return _counts(score_array, label_array, _finite_tau(tau))


def max_f1_threshold(scores: Scores, labels: Labels, *, f1_tie: F1Tie) -> MaxF1Threshold:
    """Return the max-F1 point of the precision-recall curve of the pairs.

    Raises:
        ValueError: if the pairs are invalid (see :func:`labeled_pairs`) or ``f1_tie`` is
            unknown.
    """
    _check_f1_tie(f1_tie)
    score_array, label_array = labeled_pairs(scores, labels)
    return _max_f1(score_array, label_array, f1_tie)


def calibrate_threshold(
    scores: Scores, labels: Labels, settings: CvSettings
) -> ThresholdCalibration:
    """Return τ chosen by cross-validation over the labeled pairs, as the module describes.

    Raises:
        ValueError: if the pairs are invalid (see :func:`labeled_pairs`), there are fewer pairs
            than folds or no positive pair, or a fold's training pairs hold no positive pair.
    """
    score_array, label_array = labeled_pairs(scores, labels)
    pairs, positives = int(score_array.size), int(label_array.sum())
    if pairs < settings.folds:
        raise ValueError(f"{pairs} pairs cannot be split into {settings.folds} folds")
    if positives == 0:
        raise ValueError("no positive pair: τ cannot be calibrated")

    splitter = StratifiedKFold if settings.stratified else KFold
    split = splitter(n_splits=settings.folds, shuffle=True, random_state=settings.seed)
    fold_of_pair = np.full(pairs, -1, dtype=np.int64)
    outcomes = []
    for fold, (train, test) in enumerate(split.split(np.zeros(pairs), label_array)):
        fold_of_pair[test] = fold
        if not label_array[train].any():
            raise ValueError(f"the training pairs of fold {fold} hold no positive pair")
        chosen = _best_point(_curve(score_array[train], label_array[train]), settings.f1_tie)
        counts = _counts(score_array[test], label_array[test], chosen.tau)
        outcomes.append(
            FoldOutcome(
                fold=fold,
                train_pairs=int(train.size),
                train_positives=int(label_array[train].sum()),
                tau=chosen.tau,
                train_f1=chosen.f1,
                test=counts,
                test_f1=counts.f1,
            )
        )

    pooled = ConfusionCounts(
        *(
            sum(getattr(outcome.test, name) for outcome in outcomes)
            for name in ("true_positives", "false_positives", "false_negatives", "true_negatives")
        )
    )
    fold_f1 = [outcome.test_f1 for outcome in outcomes]
    if settings.final_tau == "all_pairs":
        tau = _best_point(_curve(score_array, label_array), settings.f1_tie).tau
    else:
        tau = float(np.mean([outcome.tau for outcome in outcomes]))
    at_tau = _counts(score_array, label_array, tau)
    return ThresholdCalibration(
        settings=settings,
        pairs=pairs,
        positives=positives,
        fold_of_pair=tuple(int(fold) for fold in fold_of_pair),
        folds=tuple(outcomes),
        pooled=pooled,
        cv_f1=pooled.f1,
        mean_fold_f1=None if None in fold_f1 else float(np.mean(fold_f1)),
        tau=tau,
        at_tau=at_tau,
        f1_at_tau=at_tau.f1,
    )


def labeled_pairs(scores: Scores, labels: Labels) -> tuple[np.ndarray, np.ndarray]:
    """Return the scores as float64 and the labels as int64, checked.

    Raises:
        ValueError: if there is no pair, either is not 1-D, they differ in length, a score is
            not finite, or a label is not 0 or 1 (an integer or a boolean).
    """
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("the scores and the labels must be 1-D")
    if score_array.size != label_array.size:
        raise ValueError(
            f"the scores and the labels must cover the same pairs: {score_array.size} scores, "
            f"{label_array.size} labels"
        )
    if score_array.size == 0:
        raise ValueError("there is no pair")
    if not np.isfinite(score_array).all():
        raise ValueError("every score must be finite")
    if label_array.dtype.kind not in "biu" or not np.isin(label_array, (0, 1)).all():
        raise ValueError("every label must be 0 or 1, as an integer or a boolean")
    return score_array, label_array.astype(np.int64)


def _max_f1(scores: np.ndarray, labels: np.ndarray, f1_tie: F1Tie) -> MaxF1Threshold:
    curve = _curve(scores, labels)
    if not labels.any():
        return MaxF1Threshold(None, None, None, curve, "no_positive_pairs")
    chosen = _best_point(curve, f1_tie)
    return MaxF1Threshold(chosen.tau, chosen.f1, chosen.counts, curve, None)


def _curve(scores: np.ndarray, labels: np.ndarray) -> tuple[CurvePoint, ...]:
    """Return one point per distinct score, from the highest down."""
    order = np.argsort(-scores, kind="stable")
    ordered, ordered_labels = scores[order], labels[order]
    # The last pair of each run of equal scores closes one point of the curve.
    ends = np.flatnonzero(np.append(ordered[1:] != ordered[:-1], True))
    positives, pairs = int(labels.sum()), int(labels.size)
    curve = []
    for tau, tp, predicted in zip(
        ordered[ends], np.cumsum(ordered_labels)[ends], ends + 1, strict=True
    ):
        tp, predicted = int(tp), int(predicted)
        counts = ConfusionCounts(
            tp, predicted - tp, positives - tp, pairs - predicted - positives + tp
        )
        # 2·TP + FP + FN is predicted + positives, at least 1: F1 is always defined here.
        curve.append(CurvePoint(float(tau), counts, 2 * tp / (predicted + positives)))
    return tuple(curve)


def _best_point(curve: tuple[CurvePoint, ...], f1_tie: F1Tie) -> CurvePoint:
    best = max(point.f1 for point in curve)
    tied = [point for point in curve if point.f1 == best]
    return tied[0] if f1_tie == "highest_tau" else tied[-1]


def _counts(scores: np.ndarray, labels: np.ndarray, tau: float) -> ConfusionCounts:
    predicted = scores >= tau
    positive = labels == 1
    return ConfusionCounts(
        int(np.sum(predicted & positive)),
        int(np.sum(predicted & ~positive)),
        int(np.sum(~predicted & positive)),
        int(np.sum(~predicted & ~positive)),
    )


def _finite_tau(tau: float) -> float:
    value = float(tau)
    if not np.isfinite(value):
        raise ValueError(f"tau must be finite, not {tau!r}")
    return value


def _check_f1_tie(f1_tie: str) -> None:
    if f1_tie not in F1_TIES:
        raise ValueError(f"f1_tie must be one of {F1_TIES}, not {f1_tie!r}")


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)
