"""Bootstrap intervals of F1 at a fixed τ, and the paired comparison of two representations
over the same labeled pairs (T22, D30 E3, acceptance criteria 1 and 2).

The oracle redraws the resamples with numpy in a plain loop: each resample is ``pairs``
draws with replacement from ``numpy.random.default_rng(seed)``, shared by both
representations, and the interval is the percentile interval of the defined values.
"""

import dataclasses

import numpy as np
import pytest

from thematic_redundancy.redundancy.comparison import (
    BootstrapSettings,
    bootstrap_f1,
    compare_representations,
    paired_f1_difference,
    resample_indices,
)
from thematic_redundancy.redundancy.threshold import CvSettings, calibrate_threshold, confusion_at

SEED = 20261019
DECLARED = BootstrapSettings(resamples=2000, seed=7, interval=0.95)  # D30 E3
CV = CvSettings(folds=5, seed=7, stratified=True, f1_tie="highest_tau", final_tau="all_pairs")


def synthetic_pairs(pairs: int = 120, positives: int = 30):
    rng = np.random.default_rng(SEED)
    labels = np.zeros(pairs, dtype=np.int64)
    labels[rng.choice(pairs, positives, replace=False)] = 1
    good = np.round(rng.normal(loc=labels * 2.0, scale=0.8), 3)
    weak = np.round(rng.normal(loc=labels * 0.6, scale=0.8), 3)
    return good, weak, labels


def oracle_f1(scores, labels, tau) -> float | None:
    predicted = scores >= tau
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    return None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def oracle_draws(pairs: int, settings: BootstrapSettings) -> np.ndarray:
    rng = np.random.default_rng(settings.seed)
    return rng.integers(0, pairs, size=(settings.resamples, pairs))


def percentile_interval(values: list[float], interval: float) -> tuple[float, float]:
    tail = (1 - interval) / 2
    lower, upper = np.quantile(values, [tail, 1 - tail])
    return float(lower), float(upper)


# --- Resamples -----------------------------------------------------------------------------


def test_resamples_are_seeded_draws_with_replacement() -> None:
    draws = resample_indices(50, DECLARED)

    assert draws.shape == (2000, 50)
    assert np.array_equal(draws, oracle_draws(50, DECLARED))
    assert np.array_equal(draws, resample_indices(50, DECLARED))
    assert not np.array_equal(draws, resample_indices(50, dataclasses.replace(DECLARED, seed=8)))


# --- One F1 --------------------------------------------------------------------------------


def test_the_f1_interval_is_the_percentile_interval_of_the_resampled_f1() -> None:
    scores, _, labels = synthetic_pairs()

    result = bootstrap_f1(scores, labels, tau=0.9, settings=DECLARED)

    values = [oracle_f1(scores[row], labels[row], 0.9) for row in oracle_draws(120, DECLARED)]
    assert None not in values
    assert result.f1 == confusion_at(scores, labels, tau=0.9).f1
    assert (result.lower, result.upper) == pytest.approx(percentile_interval(values, 0.95))
    assert result.lower < result.f1 < result.upper
    assert result.undefined_resamples == 0
    assert (result.tau, result.settings) == (0.9, DECLARED)


def test_a_narrower_interval_lies_inside_a_wider_one() -> None:
    scores, _, labels = synthetic_pairs()

    wide = bootstrap_f1(scores, labels, tau=0.9, settings=DECLARED)
    narrow = bootstrap_f1(
        scores, labels, tau=0.9, settings=dataclasses.replace(DECLARED, interval=0.5)
    )

    assert wide.lower < narrow.lower < narrow.upper < wide.upper


def test_the_f1_interval_repeats_with_the_seed() -> None:
    scores, _, labels = synthetic_pairs()

    first = bootstrap_f1(scores, labels, tau=0.9, settings=DECLARED)
    again = bootstrap_f1(scores.copy(), labels.copy(), tau=0.9, settings=DECLARED)
    other = bootstrap_f1(scores, labels, tau=0.9, settings=dataclasses.replace(DECLARED, seed=8))

    assert first == again
    assert (first.lower, first.upper) != (other.lower, other.upper)


def test_resamples_with_no_f1_are_left_out_and_counted() -> None:
    # One positive, the only pair above τ: a resample that misses it predicts nothing.
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35])
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    settings = dataclasses.replace(DECLARED, resamples=500)

    result = bootstrap_f1(scores, labels, tau=0.5, settings=settings)

    values = [oracle_f1(scores[row], labels[row], 0.5) for row in oracle_draws(8, settings)]
    defined = [value for value in values if value is not None]
    assert result.undefined_resamples == values.count(None) > 0
    assert (result.lower, result.upper) == pytest.approx(percentile_interval(defined, 0.95))
    assert result.f1 == 1.0


def test_no_defined_resample_leaves_the_interval_undefined() -> None:
    result = bootstrap_f1([0.1, 0.2], [0, 0], tau=0.5, settings=DECLARED)

    assert result.f1 is None
    assert (result.lower, result.upper) == (None, None)
    assert result.undefined_resamples == 2000


# --- Paired difference ---------------------------------------------------------------------


def test_the_difference_resamples_the_same_pairs_for_both_representations() -> None:
    good, weak, labels = synthetic_pairs()

    result = paired_f1_difference(
        good, weak, labels, first_tau=1.0, second_tau=0.3, settings=DECLARED
    )

    differences = [
        oracle_f1(good[row], labels[row], 1.0) - oracle_f1(weak[row], labels[row], 0.3)
        for row in oracle_draws(120, DECLARED)
    ]
    assert result.first_f1 == confusion_at(good, labels, tau=1.0).f1
    assert result.second_f1 == confusion_at(weak, labels, tau=0.3).f1
    assert result.difference == pytest.approx(result.first_f1 - result.second_f1)
    assert (result.lower, result.upper) == pytest.approx(percentile_interval(differences, 0.95))
    assert result.undefined_resamples == 0


def test_a_representation_compared_with_itself_differs_by_exactly_zero() -> None:
    good, _, labels = synthetic_pairs()

    result = paired_f1_difference(
        good, good.copy(), labels, first_tau=1.0, second_tau=1.0, settings=DECLARED
    )

    assert (result.difference, result.lower, result.upper) == (0.0, 0.0, 0.0)
    assert result.verdict == "tie"


def test_a_clearly_better_first_representation_wins_and_the_swap_loses() -> None:
    good, weak, labels = synthetic_pairs()

    forward = paired_f1_difference(
        good, weak, labels, first_tau=1.0, second_tau=0.3, settings=DECLARED
    )
    backward = paired_f1_difference(
        weak, good, labels, first_tau=0.3, second_tau=1.0, settings=DECLARED
    )

    assert forward.lower > 0
    assert forward.verdict == "first_higher"
    assert backward.upper < 0
    assert backward.verdict == "second_higher"
    assert (backward.lower, backward.upper) == pytest.approx((-forward.upper, -forward.lower))


@pytest.mark.parametrize("swap", [False, True])
def test_an_interval_that_touches_zero_is_a_tie(swap: bool) -> None:
    # The two differ on one positive pair only, which the first finds and the second misses,
    # so a resample's difference is positive when it draws that pair and exactly 0 otherwise:
    # about 37% of the resamples miss it, so the lower bound is exactly 0.
    _, _, labels = synthetic_pairs()
    first = np.where(labels == 1, 0.9, 0.1)
    second = first.copy()
    second[np.flatnonzero(labels)[0]] = 0.2
    if swap:
        first, second = second, first

    result = paired_f1_difference(
        first, second, labels, first_tau=0.5, second_tau=0.5, settings=DECLARED
    )

    assert (result.upper if swap else result.lower) == 0.0
    assert result.difference != 0.0
    assert result.verdict == "tie"


def test_resamples_where_either_f1_is_undefined_are_left_out_of_the_difference() -> None:
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35])
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    other = scores.copy()
    other[3] = 0.6  # a negative above τ: the second always predicts when it is drawn
    settings = dataclasses.replace(DECLARED, resamples=500)

    result = paired_f1_difference(
        scores, other, labels, first_tau=0.5, second_tau=0.5, settings=settings
    )

    pairs = [
        (oracle_f1(scores[row], labels[row], 0.5), oracle_f1(other[row], labels[row], 0.5))
        for row in oracle_draws(8, settings)
    ]
    defined = [a - b for a, b in pairs if a is not None and b is not None]
    assert result.undefined_resamples == len(pairs) - len(defined) > 0
    assert (result.lower, result.upper) == pytest.approx(percentile_interval(defined, 0.95))


def test_no_defined_resample_leaves_the_difference_without_a_verdict() -> None:
    result = paired_f1_difference(
        [0.1, 0.2], [0.2, 0.1], [0, 0], first_tau=0.5, second_tau=0.5, settings=DECLARED
    )

    assert (result.difference, result.lower, result.upper, result.verdict) == (
        None,
        None,
        None,
        None,
    )
    assert result.undefined_resamples == 2000


# --- The comparison of two representations -------------------------------------------------


def test_the_comparison_calibrates_each_tau_by_cv_then_bootstraps_at_those_taus() -> None:
    good, weak, labels = synthetic_pairs()

    result = compare_representations(
        good, weak, labels, names=("mpnet", "tfidf"), cv=CV, bootstrap=DECLARED
    )

    first = calibrate_threshold(good, labels, CV)
    second = calibrate_threshold(weak, labels, CV)
    assert (result.first_name, result.second_name) == ("mpnet", "tfidf")
    assert (result.first, result.second) == (first, second)
    assert result.first.fold_of_pair == result.second.fold_of_pair
    assert result.first_interval == bootstrap_f1(good, labels, tau=first.tau, settings=DECLARED)
    assert result.second_interval == bootstrap_f1(weak, labels, tau=second.tau, settings=DECLARED)
    assert result.difference == paired_f1_difference(
        good, weak, labels, first_tau=first.tau, second_tau=second.tau, settings=DECLARED
    )
    assert result.difference.verdict == "first_higher"
    assert result.first_beats_second


def test_the_first_does_not_beat_the_second_on_a_tie_or_a_loss() -> None:
    good, weak, labels = synthetic_pairs()

    tie = compare_representations(
        good, good.copy(), labels, names=("a", "b"), cv=CV, bootstrap=DECLARED
    )
    loss = compare_representations(
        weak, good, labels, names=("tfidf", "mpnet"), cv=CV, bootstrap=DECLARED
    )

    assert tie.difference.verdict == "tie"
    assert not tie.first_beats_second
    assert loss.difference.verdict == "second_higher"
    assert not loss.first_beats_second


def test_the_comparison_repeats_exactly() -> None:
    good, weak, labels = synthetic_pairs()

    first = compare_representations(good, weak, labels, names=("a", "b"), cv=CV, bootstrap=DECLARED)
    again = compare_representations(good, weak, labels, names=("a", "b"), cv=CV, bootstrap=DECLARED)

    assert first == again


# --- Invalid input -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"resamples": 0}, "resamples"),
        ({"resamples": True}, "resamples"),
        ({"resamples": 20.0}, "resamples"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"interval": 0.0}, "interval"),
        ({"interval": 1.0}, "interval"),
        ({"interval": 95}, "interval"),
    ],
)
def test_invalid_bootstrap_settings_are_refused(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(DECLARED, **changes)


def test_scores_of_another_length_are_refused() -> None:
    good, weak, labels = synthetic_pairs()

    with pytest.raises(ValueError, match="pairs"):
        paired_f1_difference(
            good, weak[:-1], labels, first_tau=0.5, second_tau=0.5, settings=DECLARED
        )


@pytest.mark.parametrize("tau", [np.nan, np.inf])
def test_a_threshold_that_is_not_finite_is_refused(tau: float) -> None:
    good, _, labels = synthetic_pairs()

    with pytest.raises(ValueError, match="tau"):
        bootstrap_f1(good, labels, tau=tau, settings=DECLARED)


@pytest.mark.parametrize("names", [("a", "a"), ("", "b"), ("a",)])
def test_the_two_representations_need_two_different_names(names) -> None:
    good, weak, labels = synthetic_pairs()

    with pytest.raises(ValueError, match="names"):
        compare_representations(good, weak, labels, names=names, cv=CV, bootstrap=DECLARED)
