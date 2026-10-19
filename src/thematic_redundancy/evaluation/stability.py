"""Stability of a clustering across seeds: the pairwise adjusted Rand index (T17, D12).

Every pair of runs is compared with scikit-learn's adjusted Rand index (ARI), which is 1 for
two identical partitions whatever their label ids, about 0 for independent ones, and 1 also
when both runs put all points in one cluster, or each point in its own (scikit-learn's
convention). The summary is the mean and the minimum over the pairs.

**Noise handling**, chosen by the caller with no default (T18 declares it):

- ``"own_label"``: :data:`~.clustering_metrics.NOISE_LABEL` is one more label, so every point
  is compared;
- ``"exclude_either"``: a point that is noise in either run is left out of that pair, so each
  pair compares the points that both runs cluster.

Undefined values follow the package's convention, ``None`` with a reason: a pair with no point
to compare is ``"no_points"``; the summary is ``"fewer_than_two_runs"`` with fewer than two
runs, and ``"undefined_pairs"`` when a pair is undefined.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from sklearn.metrics import adjusted_rand_score

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array

NoiseHandling = Literal["own_label", "exclude_either"]
NOISE_HANDLINGS: tuple[NoiseHandling, ...] = ("own_label", "exclude_either")


@dataclass(frozen=True)
class PairAgreement:
    """The ARI between two runs."""

    first: int
    second: int
    """Indices of the two runs, ``first < second``."""
    points: int
    """The points compared."""
    ari: float | None
    undefined_reason: Literal["no_points"] | None


@dataclass(frozen=True)
class SeedStability:
    """The ARI of every pair of runs, and their mean and minimum."""

    runs: int
    noise_handling: NoiseHandling
    pairs: tuple[PairAgreement, ...]
    """Every pair, in the order of ``itertools.combinations(range(runs), 2)``."""
    matrix: tuple[tuple[float | None, ...], ...]
    """``runs × runs``, symmetric: the ARI of each pair, and of each run with itself on the
    diagonal; ``None`` where undefined."""
    mean: float | None
    minimum: float | None
    undefined_reason: Literal["fewer_than_two_runs", "undefined_pairs"] | None


def seed_stability(
    labelings: Sequence[Sequence[int] | np.ndarray], *, noise_handling: NoiseHandling
) -> SeedStability:
    """Return the ARI of every pair of ``labelings``, one labeling per run over the same points
    in the same order.

    Raises:
        ValueError: if ``noise_handling`` is unknown, a labeling is invalid (see
            :func:`~.clustering_metrics.label_array`), or the labelings differ in length.
    """
    if noise_handling not in NOISE_HANDLINGS:
        raise ValueError(f"noise_handling must be one of {NOISE_HANDLINGS}, not {noise_handling!r}")
    runs = [label_array(labels) for labels in labelings]
    for index, labels in enumerate(runs[1:], start=1):
        if labels.size != runs[0].size:
            raise ValueError(f"run {index} has {labels.size} labels but run 0 has {runs[0].size}")
    pairs = tuple(
        _agreement(first, second, runs[first], runs[second], noise_handling)
        for first, second in combinations(range(len(runs)), 2)
    )
    matrix = [[None] * len(runs) for _ in runs]
    for run, labels in enumerate(runs):
        matrix[run][run] = _agreement(run, run, labels, labels, noise_handling).ari
    for pair in pairs:
        matrix[pair.first][pair.second] = matrix[pair.second][pair.first] = pair.ari

    values = [pair.ari for pair in pairs]
    reason: Literal["fewer_than_two_runs", "undefined_pairs"] | None = None
    if len(runs) < 2:
        reason = "fewer_than_two_runs"
    elif None in values:
        reason = "undefined_pairs"
    return SeedStability(
        runs=len(runs),
        noise_handling=noise_handling,
        pairs=pairs,
        matrix=tuple(tuple(row) for row in matrix),
        mean=None if reason else float(np.mean(values)),
        minimum=None if reason else float(min(values)),
        undefined_reason=reason,
    )


def _agreement(
    first: int,
    second: int,
    first_labels: np.ndarray,
    second_labels: np.ndarray,
    noise_handling: NoiseHandling,
) -> PairAgreement:
    if noise_handling == "exclude_either":
        kept = (first_labels != NOISE_LABEL) & (second_labels != NOISE_LABEL)
        first_labels, second_labels = first_labels[kept], second_labels[kept]
    points = int(first_labels.size)
    if points == 0:
        return PairAgreement(first, second, 0, None, "no_points")
    return PairAgreement(
        first, second, points, float(adjusted_rand_score(first_labels, second_labels)), None
    )
