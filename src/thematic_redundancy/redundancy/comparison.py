"""Bootstrap intervals of F1 at a fixed τ, and the paired comparison of two representations
over the same labeled pairs (T22, D30 E3; acceptance criteria 1 and 2).

**Resamples** (:func:`resample_indices`): ``resamples`` rows of ``pairs`` draws with
replacement, ``numpy.random.default_rng(seed).integers(0, pairs, size=(resamples, pairs))``.
The draws depend on the number of pairs and the settings only, so every interval computed with
the same settings over the same pairs reads the same resamples. D30 declares 2,000 resamples,
seed 7 and a 95% interval (``experiment.selection.f1``).

**F1 interval** (:func:`bootstrap_f1`): the F1 of the pairs at a fixed τ (see
:mod:`~thematic_redundancy.redundancy.threshold`), recomputed in each resample at that same τ,
which is never chosen again; the interval is the percentile interval of those values, the
``(1 - interval) / 2`` and ``1 - (1 - interval) / 2`` quantiles, by numpy's default linear
interpolation.

**Paired difference** (:func:`paired_f1_difference`): ΔF1 = F1(first) − F1(second), each at its
own fixed τ, recomputed in every resample from the same drawn pairs for both, so the
resampling noise common to both cancels. The **verdict** follows D30: when the percentile
interval of ΔF1 excludes 0, the side it lies on wins (``"first_higher"`` when its lower bound
is above 0, ``"second_higher"`` when its upper bound is below 0); an interval that holds or
touches 0 is a ``"tie"``.

**Undefined resamples.** A resample whose F1 is undefined, with no positive pair drawn and none
predicted, has no value; for the difference, a resample where either F1 is undefined has none.
Such resamples are left out of the quantiles and counted in ``undefined_resamples``; when no
resample has a value, the interval and the verdict are ``None``.

**Comparison** (:func:`compare_representations`): each representation's τ is calibrated by
cross-validation with the same settings, so both share the folds, then each F1 interval and
the paired difference are bootstrapped at those fixed τ with the same resamples. For acceptance
criterion 1, the semantic winner is the first and TF-IDF the second: the winner's F1 must
exceed TF-IDF's with an interval that excludes 0, which is
:attr:`RepresentationComparison.first_beats_second`.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from thematic_redundancy.redundancy.threshold import (
    SEED_LIMIT,
    CvSettings,
    Labels,
    Scores,
    ThresholdCalibration,
    calibrate_threshold,
    labeled_pairs,
)

Verdict = Literal["first_higher", "second_higher", "tie"]


@dataclass(frozen=True)
class BootstrapSettings:
    """The resampling of the labeled pairs; D30 declares 2,000 resamples, seed 7 and 0.95."""

    resamples: int
    seed: int
    interval: float
    """Coverage of the percentile interval, strictly between 0 and 1."""

    def __post_init__(self) -> None:
        if not _is_int(self.resamples) or self.resamples < 1:
            raise ValueError(f"resamples must be an integer of at least 1, not {self.resamples!r}")
        if not _is_int(self.seed) or not 0 <= self.seed < SEED_LIMIT:
            raise ValueError(f"seed must be an integer in [0, 2**32), not {self.seed!r}")
        if isinstance(self.interval, bool) or not 0.0 < self.interval < 1.0:
            raise ValueError(f"interval must lie strictly between 0 and 1, not {self.interval!r}")


@dataclass(frozen=True)
class F1Interval:
    """The F1 of the pairs at a fixed τ, and its bootstrap percentile interval."""

    tau: float
    f1: float | None
    lower: float | None
    upper: float | None
    undefined_resamples: int
    settings: BootstrapSettings


@dataclass(frozen=True)
class F1Difference:
    """ΔF1 = F1(first) − F1(second) over the same pairs, its paired interval and the verdict."""

    first_tau: float
    second_tau: float
    first_f1: float | None
    second_f1: float | None
    difference: float | None
    lower: float | None
    upper: float | None
    verdict: Verdict | None
    undefined_resamples: int
    settings: BootstrapSettings


@dataclass(frozen=True)
class RepresentationComparison:
    """Two representations over the same labeled pairs: each τ by cross-validation, each F1
    with its interval, and their paired difference."""

    first_name: str
    second_name: str
    first: ThresholdCalibration
    second: ThresholdCalibration
    first_interval: F1Interval
    second_interval: F1Interval
    difference: F1Difference

    @property
    def first_beats_second(self) -> bool:
        """The first's F1 is higher and the interval of the difference excludes 0."""
        return self.difference.verdict == "first_higher"


def resample_indices(pairs: int, settings: BootstrapSettings) -> np.ndarray:
    """Return the ``(resamples, pairs)`` drawn pair indices that every bootstrap here reads.

    Raises:
        ValueError: if ``pairs`` is below 1.
    """
    if not _is_int(pairs) or pairs < 1:
        raise ValueError(f"pairs must be an integer of at least 1, not {pairs!r}")
    rng = np.random.default_rng(settings.seed)
    return rng.integers(0, pairs, size=(settings.resamples, pairs))


def bootstrap_f1(
    scores: Scores, labels: Labels, *, tau: float, settings: BootstrapSettings
) -> F1Interval:
    """Return the F1 of the pairs at the fixed ``tau`` and its percentile interval.

    Raises:
        ValueError: if the pairs are invalid (see
            :func:`~thematic_redundancy.redundancy.threshold.labeled_pairs`) or ``tau`` is not
            finite.
    """
    score_array, label_array = labeled_pairs(scores, labels)
    threshold = _finite_tau(tau, "tau")
    draws = resample_indices(score_array.size, settings)
    observed = _f1(score_array >= threshold, label_array)
    values = _f1(score_array[draws] >= threshold, label_array[draws])
    lower, upper, undefined = _percentile_interval(values, settings.interval)
    return F1Interval(
        tau=threshold,
        f1=_value(observed),
        lower=lower,
        upper=upper,
        undefined_resamples=undefined,
        settings=settings,
    )


def paired_f1_difference(
    first_scores: Scores,
    second_scores: Scores,
    labels: Labels,
    *,
    first_tau: float,
    second_tau: float,
    settings: BootstrapSettings,
) -> F1Difference:
    """Return ΔF1 = F1(first) − F1(second) over the same pairs, each at its fixed τ, with its
    paired percentile interval and D30's verdict.

    Raises:
        ValueError: if either set of pairs is invalid (see
            :func:`~thematic_redundancy.redundancy.threshold.labeled_pairs`), the two hold
            different numbers of pairs, or a τ is not finite.
    """
    first_array, label_array = labeled_pairs(first_scores, labels)
    second_array, _ = labeled_pairs(second_scores, labels)
    first_threshold = _finite_tau(first_tau, "first_tau")
    second_threshold = _finite_tau(second_tau, "second_tau")
    draws = resample_indices(label_array.size, settings)
    first_f1 = _value(_f1(first_array >= first_threshold, label_array))
    second_f1 = _value(_f1(second_array >= second_threshold, label_array))
    differences = _f1(first_array[draws] >= first_threshold, label_array[draws]) - _f1(
        second_array[draws] >= second_threshold, label_array[draws]
    )
    lower, upper, undefined = _percentile_interval(differences, settings.interval)
    return F1Difference(
        first_tau=first_threshold,
        second_tau=second_threshold,
        first_f1=first_f1,
        second_f1=second_f1,
        difference=None if first_f1 is None or second_f1 is None else first_f1 - second_f1,
        lower=lower,
        upper=upper,
        verdict=_verdict(lower, upper),
        undefined_resamples=undefined,
        settings=settings,
    )


def compare_representations(
    first_scores: Scores,
    second_scores: Scores,
    labels: Labels,
    *,
    names: tuple[str, str],
    cv: CvSettings,
    bootstrap: BootstrapSettings,
) -> RepresentationComparison:
    """Return the comparison of two representations' scores over the same labeled pairs.

    Raises:
        ValueError: if ``names`` are not two different non-empty strings, or as
            :func:`~thematic_redundancy.redundancy.threshold.calibrate_threshold` and
            :func:`paired_f1_difference` raise.
    """
    if (
        len(names) != 2
        or not all(isinstance(name, str) and name for name in names)
        or names[0] == names[1]
    ):
        raise ValueError(f"names must be two different non-empty strings, not {names!r}")
    first = calibrate_threshold(first_scores, labels, cv)
    second = calibrate_threshold(second_scores, labels, cv)
    return RepresentationComparison(
        first_name=names[0],
        second_name=names[1],
        first=first,
        second=second,
        first_interval=bootstrap_f1(first_scores, labels, tau=first.tau, settings=bootstrap),
        second_interval=bootstrap_f1(second_scores, labels, tau=second.tau, settings=bootstrap),
        difference=paired_f1_difference(
            first_scores,
            second_scores,
            labels,
            first_tau=first.tau,
            second_tau=second.tau,
            settings=bootstrap,
        ),
    )


def _f1(predicted: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """F1 along the last axis, NaN where it is undefined."""
    positive = labels == 1
    true_positives = np.sum(predicted & positive, axis=-1)
    denominator = np.sum(predicted, axis=-1) + np.sum(positive, axis=-1)  # 2·TP + FP + FN
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * true_positives / denominator, np.nan)


def _percentile_interval(
    values: np.ndarray, interval: float
) -> tuple[float | None, float | None, int]:
    defined = values[~np.isnan(values)]
    undefined = int(values.size - defined.size)
    if defined.size == 0:
        return None, None, undefined
    tail = (1.0 - interval) / 2.0
    lower, upper = np.quantile(defined, [tail, 1.0 - tail])
    return float(lower), float(upper), undefined


def _verdict(lower: float | None, upper: float | None) -> Verdict | None:
    if lower is None or upper is None:
        return None
    if lower > 0:
        return "first_higher"
    if upper < 0:
        return "second_higher"
    return "tie"


def _value(f1: np.ndarray) -> float | None:
    value = float(f1)
    return None if np.isnan(value) else value


def _finite_tau(tau: float, label: str) -> float:
    value = float(tau)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite, not {tau!r}")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)
