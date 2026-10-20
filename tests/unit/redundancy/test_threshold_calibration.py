"""The direct-redundancy threshold τ (T22, D15, D30 E3): F1 at a threshold, the max-F1 point
of the precision-recall curve, and τ chosen by k-fold cross-validation.

Hand-computed cases pin F1 and τ; scikit-learn's ``precision_recall_curve`` is the oracle of
the curve, and its ``KFold`` and ``StratifiedKFold`` of the fold assignment, which the test
recomputes on its own to check that each τ comes from the training folds only.
"""

import dataclasses

import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold

from thematic_redundancy.redundancy.threshold import (
    ConfusionCounts,
    CvSettings,
    calibrate_threshold,
    confusion_at,
    max_f1_threshold,
)

SEED = 20261019

# Six pairs, three positive; τ = 0.6 gives TP 3, FP 1, FN 0, so F1 = 6/7, the maximum.
SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
LABELS = [1, 1, 0, 1, 0, 0]
CURVE_F1 = {0.9: 2 / 4, 0.8: 4 / 5, 0.7: 4 / 6, 0.6: 6 / 7, 0.5: 6 / 8, 0.4: 6 / 9}

# Two thresholds share the highest F1, 2/3: τ = 0.9 (TP 1, FN 1) and τ = 0.6 (TP 2, FP 2).
TIED_SCORES = [0.9, 0.8, 0.7, 0.6]
TIED_LABELS = [1, 0, 0, 1]


def settings(**changes) -> CvSettings:
    base = CvSettings(folds=5, seed=7, stratified=True, f1_tie="highest_tau", final_tau="all_pairs")
    return dataclasses.replace(base, **changes)


def synthetic_pairs(pairs: int = 120, positives: int = 30, overlap: float = 1.0):
    """Scores of positive pairs drawn higher than those of negative pairs, with overlap."""
    rng = np.random.default_rng(SEED)
    labels = np.zeros(pairs, dtype=np.int64)
    labels[rng.choice(pairs, positives, replace=False)] = 1
    scores = rng.normal(loc=labels * 1.0, scale=overlap)
    return np.round(scores, 3), labels


def oracle_max_f1(scores, labels, *, highest: bool) -> tuple[float, float]:
    """τ and F1 at the max-F1 point, from scikit-learn's precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]  # drop the appended (1, 0) end point
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    chosen = best[-1] if highest else best[0]  # thresholds ascend
    return float(thresholds[chosen]), float(f1[chosen])


# --- F1 at a threshold ---------------------------------------------------------------------


def test_counts_at_a_threshold_include_pairs_equal_to_it() -> None:
    counts = confusion_at(SCORES, LABELS, tau=0.6)

    assert counts == ConfusionCounts(
        true_positives=3, false_positives=1, false_negatives=0, true_negatives=2
    )
    assert counts.f1 == pytest.approx(6 / 7)
    assert counts.precision == pytest.approx(3 / 4)
    assert counts.recall == 1.0


def test_pairs_with_equal_scores_fall_on_the_same_side_of_the_threshold() -> None:
    scores, labels = [0.9, 0.7, 0.7, 0.5], [1, 1, 0, 0]

    at = confusion_at(scores, labels, tau=0.7)
    above = confusion_at(scores, labels, tau=0.7000001)

    assert (at.true_positives, at.false_positives) == (2, 1)
    assert (above.true_positives, above.false_positives) == (1, 0)


def test_f1_is_zero_when_positives_exist_but_none_is_predicted() -> None:
    counts = confusion_at(SCORES, LABELS, tau=0.95)

    assert counts.f1 == 0.0
    assert counts.precision is None
    assert counts.recall == 0.0


def test_f1_is_undefined_with_no_positive_label_and_no_prediction() -> None:
    counts = confusion_at([0.1, 0.2], [0, 0], tau=0.5)

    assert counts == ConfusionCounts(0, 0, 0, 2)
    assert counts.f1 is None
    assert counts.recall is None


def test_f1_is_zero_with_no_positive_label_but_some_prediction() -> None:
    assert confusion_at([0.1, 0.9], [0, 0], tau=0.5).f1 == 0.0


def test_boolean_labels_are_accepted() -> None:
    assert confusion_at(SCORES, [bool(label) for label in LABELS], tau=0.6).f1 == pytest.approx(
        6 / 7
    )


# --- The max-F1 point ----------------------------------------------------------------------


def test_the_max_f1_point_is_the_hand_computed_one() -> None:
    result = max_f1_threshold(SCORES, LABELS, f1_tie="highest_tau")

    assert result.tau == 0.6
    assert result.f1 == pytest.approx(6 / 7)
    assert result.counts == ConfusionCounts(3, 1, 0, 2)
    assert result.undefined_reason is None
    assert [point.tau for point in result.curve] == [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert [point.f1 for point in result.curve] == pytest.approx(list(CURVE_F1.values()))


def test_the_order_of_the_pairs_does_not_matter() -> None:
    order = [3, 0, 5, 1, 4, 2]

    result = max_f1_threshold(
        [SCORES[i] for i in order], [LABELS[i] for i in order], f1_tie="highest_tau"
    )

    assert (result.tau, result.f1) == (0.6, pytest.approx(6 / 7))


@pytest.mark.parametrize(("f1_tie", "tau"), [("highest_tau", 0.9), ("lowest_tau", 0.6)])
def test_a_tie_in_f1_is_broken_by_the_declared_rule(f1_tie: str, tau: float) -> None:
    result = max_f1_threshold(TIED_SCORES, TIED_LABELS, f1_tie=f1_tie)

    assert result.tau == tau
    assert result.f1 == pytest.approx(2 / 3)


def test_tied_scores_form_one_point_of_the_curve() -> None:
    result = max_f1_threshold([0.9, 0.7, 0.7, 0.5], [1, 1, 0, 0], f1_tie="highest_tau")

    assert [point.tau for point in result.curve] == [0.9, 0.7, 0.5]
    assert result.tau == 0.7
    assert result.f1 == pytest.approx(4 / 5)


@pytest.mark.parametrize("trial", range(6))
def test_the_max_f1_point_matches_scikit_learns_curve(trial: int) -> None:
    rng = np.random.default_rng(SEED + trial)
    labels = rng.integers(0, 2, 60)
    labels[0] = 1
    scores = np.round(rng.normal(loc=labels * 0.8), 1)  # coarse: many tied scores

    for highest, rule in ((True, "highest_tau"), (False, "lowest_tau")):
        tau, f1 = oracle_max_f1(scores, labels, highest=highest)
        result = max_f1_threshold(scores, labels, f1_tie=rule)
        assert result.tau == pytest.approx(tau)
        assert result.f1 == pytest.approx(f1, abs=1e-12)


def test_all_positive_pairs_put_tau_at_the_lowest_score() -> None:
    result = max_f1_threshold([0.3, 0.9, 0.5], [1, 1, 1], f1_tie="highest_tau")

    assert result.tau == 0.3
    assert result.f1 == 1.0


def test_no_positive_pair_leaves_tau_undefined() -> None:
    result = max_f1_threshold([0.3, 0.9, 0.5], [0, 0, 0], f1_tie="highest_tau")

    assert result.tau is None
    assert result.f1 is None
    assert result.counts is None
    assert result.undefined_reason == "no_positive_pairs"
    assert [point.f1 for point in result.curve] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        ([], [], "no pair"),
        ([0.1, 0.2], [1], "pairs"),
        ([0.1, np.nan], [1, 0], "finite"),
        ([0.1, np.inf], [1, 0], "finite"),
        ([0.1, 0.2], [1, 2], "0 or 1"),
        ([0.1, 0.2], [1.0, 0.0], "0 or 1"),
        ([[0.1, 0.2]], [[1, 0]], "1-D"),
    ],
)
def test_invalid_pairs_are_refused(scores, labels, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        max_f1_threshold(scores, labels, f1_tie="highest_tau")
    with pytest.raises(ValueError, match=message):
        confusion_at(scores, labels, tau=0.5)


def test_an_unknown_tie_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="f1_tie"):
        max_f1_threshold(SCORES, LABELS, f1_tie="middle")


# --- Cross-validation ----------------------------------------------------------------------


def oracle_folds(labels: np.ndarray, cv: CvSettings) -> list[tuple[np.ndarray, np.ndarray]]:
    kind = StratifiedKFold if cv.stratified else KFold
    splitter = kind(n_splits=cv.folds, shuffle=True, random_state=cv.seed)
    return list(splitter.split(np.zeros(labels.size), labels))


@pytest.mark.parametrize("stratified", [True, False])
def test_each_fold_chooses_tau_on_its_training_pairs_and_scores_the_held_out_ones(
    stratified: bool,
) -> None:
    scores, labels = synthetic_pairs()
    cv = settings(stratified=stratified)

    result = calibrate_threshold(scores, labels, cv)

    folds = oracle_folds(labels, cv)
    assert len(result.folds) == 5
    for fold, (outcome, (train, test)) in enumerate(zip(result.folds, folds, strict=True)):
        tau, train_f1 = oracle_max_f1(scores[train], labels[train], highest=True)
        assert outcome.fold == fold
        assert outcome.tau == tau
        assert outcome.train_f1 == pytest.approx(train_f1, abs=1e-12)
        assert (outcome.train_pairs, outcome.train_positives) == (train.size, labels[train].sum())
        assert outcome.test == confusion_at(scores[test], labels[test], tau=tau)
        assert outcome.test_f1 == outcome.test.f1
        assert [result.fold_of_pair[i] for i in test] == [fold] * test.size


def test_the_cv_f1_pools_the_held_out_counts() -> None:
    scores, labels = synthetic_pairs()

    result = calibrate_threshold(scores, labels, settings())

    pooled = [
        sum(getattr(outcome.test, field.name) for outcome in result.folds)
        for field in dataclasses.fields(ConfusionCounts)
    ]
    assert result.pooled == ConfusionCounts(*pooled)
    assert sum(pooled) == 120
    assert result.cv_f1 == result.pooled.f1
    assert result.mean_fold_f1 == pytest.approx(
        np.mean([outcome.test_f1 for outcome in result.folds])
    )
    assert (result.pairs, result.positives) == (120, 30)


def test_the_held_out_f1_differs_from_the_optimistic_in_sample_f1() -> None:
    scores, labels = synthetic_pairs()

    result = calibrate_threshold(scores, labels, settings())

    assert result.cv_f1 < max_f1_threshold(scores, labels, f1_tie="highest_tau").f1


def test_stratified_folds_spread_the_positives_evenly() -> None:
    scores, labels = synthetic_pairs(positives=30)

    result = calibrate_threshold(scores, labels, settings(stratified=True))

    held_out_positives = [
        outcome.test.true_positives + outcome.test.false_negatives for outcome in result.folds
    ]
    assert held_out_positives == [6, 6, 6, 6, 6]


def test_the_final_tau_on_all_pairs_is_their_max_f1_point() -> None:
    scores, labels = synthetic_pairs()

    result = calibrate_threshold(scores, labels, settings(final_tau="all_pairs"))

    whole = max_f1_threshold(scores, labels, f1_tie="highest_tau")
    assert result.tau == whole.tau
    assert result.at_tau == whole.counts
    assert result.f1_at_tau == whole.f1


def test_the_final_tau_by_fold_mean_is_the_mean_of_the_fold_taus() -> None:
    scores, labels = synthetic_pairs()

    result = calibrate_threshold(scores, labels, settings(final_tau="fold_mean"))

    tau = float(np.mean([outcome.tau for outcome in result.folds]))
    assert result.tau == pytest.approx(tau)
    assert result.at_tau == confusion_at(scores, labels, tau=result.tau)
    assert result.f1_at_tau == result.at_tau.f1


def test_the_tie_rule_reaches_every_fold() -> None:
    scores, labels = synthetic_pairs(overlap=1.5)
    scores = np.round(scores, 1)

    for rule, highest in (("highest_tau", True), ("lowest_tau", False)):
        result = calibrate_threshold(scores, labels, settings(f1_tie=rule))
        for outcome, (train, _) in zip(result.folds, oracle_folds(labels, settings()), strict=True):
            assert outcome.tau == oracle_max_f1(scores[train], labels[train], highest=highest)[0]


def test_the_folds_repeat_with_the_seed_and_change_with_another_one() -> None:
    scores, labels = synthetic_pairs()

    first = calibrate_threshold(scores, labels, settings(seed=7))
    again = calibrate_threshold(scores.copy(), labels.copy(), settings(seed=7))
    other = calibrate_threshold(scores, labels, settings(seed=8))

    assert first == again
    assert first.fold_of_pair != other.fold_of_pair


def test_the_folds_depend_on_the_labels_and_seed_only_not_on_the_scores() -> None:
    scores, labels = synthetic_pairs()
    other_scores = np.random.default_rng(SEED + 1).random(labels.size)

    first = calibrate_threshold(scores, labels, settings())
    second = calibrate_threshold(other_scores, labels, settings())

    assert first.fold_of_pair == second.fold_of_pair


def test_all_positive_pairs_calibrate_to_the_lowest_training_score() -> None:
    scores = np.linspace(0.1, 0.9, 10)

    result = calibrate_threshold(scores, np.ones(10, dtype=int), settings(stratified=False))

    assert result.cv_f1 is not None
    assert result.tau == pytest.approx(0.1)
    assert result.f1_at_tau == 1.0


def test_a_held_out_fold_with_no_positive_and_no_prediction_has_no_f1() -> None:
    # The two positives hold the top scores, 8/9 and 1, in two different folds. The other 3
    # folds train τ = 8/9 and hold out negatives below it, so they have no F1. Holding out 1
    # trains τ = 8/9 and finds it (F1 1); holding out 8/9 trains τ = 1 and misses it (F1 0).
    scores = np.linspace(0.0, 1.0, 10)
    labels = np.array([0] * 8 + [1, 1])

    with pytest.warns(UserWarning):  # scikit-learn: fewer positives than folds
        result = calibrate_threshold(scores, labels, settings(stratified=True))

    assert sorted(outcome.test_f1 for outcome in result.folds if outcome.test_f1 is not None) == [
        0.0,
        1.0,
    ]
    assert sum(outcome.test_f1 is None for outcome in result.folds) == 3
    assert result.mean_fold_f1 is None
    assert result.pooled == ConfusionCounts(1, 0, 1, 8)
    assert result.cv_f1 == pytest.approx(2 / 3)


def test_no_positive_pair_cannot_be_calibrated() -> None:
    with pytest.raises(ValueError, match="positive"):
        calibrate_threshold([0.1, 0.2, 0.3, 0.4, 0.5], [0] * 5, settings(stratified=False))


def test_a_training_split_without_a_positive_is_refused() -> None:
    labels = np.array([0] * 9 + [1])

    with pytest.warns(UserWarning), pytest.raises(ValueError, match="fold"):
        calibrate_threshold(np.linspace(0, 1, 10), labels, settings(stratified=True))


def test_fewer_pairs_than_folds_are_refused() -> None:
    with pytest.raises(ValueError, match="folds"):
        calibrate_threshold([0.1, 0.9, 0.5], [0, 1, 1], settings(stratified=False))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"folds": 1}, "folds"),
        ({"folds": True}, "folds"),
        ({"folds": 5.0}, "folds"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"stratified": 1}, "stratified"),
        ({"f1_tie": "middle"}, "f1_tie"),
        ({"final_tau": "median"}, "final_tau"),
    ],
)
def test_invalid_settings_are_refused(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        settings(**changes)
