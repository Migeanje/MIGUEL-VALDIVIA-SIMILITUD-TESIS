"""Consistency of a topic assignment with another map: the temporal split's metrics (T27, O04).

O04 measures success on the temporal split by "assignment consistency between the 2021–2024
map and the full map". The theses of the later period are assigned to the topics of a map fitted
on the earlier period, and their assignment is compared with their labels in the full map, the
map of every thesis (T19). The topic ids of two maps are unrelated, so every metric here ignores
how either map numbers its labels:

- **ARI**, the adjusted Rand index of the two labelings, through
  :func:`~.stability.seed_stability` (T17): 1 for the same partition, about 0 for independent
  ones;
- **NMI**, the normalized mutual information, scikit-learn's ``normalized_mutual_info_score``
  with its default arithmetic-mean normalization: 1 for the same partition, 0 for independent
  ones;
- **counterpart agreement**, the share of the assigned theses whose full-map label is the
  counterpart of the topic they were assigned to. A topic's counterpart is the full-map label
  of most of the topic's own members, the earlier theses that the earlier map clustered into it
  (:func:`topic_counterparts`; a tie goes to the lower label). The counterparts are learned from
  the earlier period alone, so no assigned thesis chooses its own counterpart. Unlike ARI and
  NMI, it reads as a plain percentage.

**Noise handling**, as in :mod:`.stability`, chosen by the caller with no default:

- ``"own_label"``: :data:`~.clustering_metrics.NOISE_LABEL` is one more label, so every point is
  compared, and a thesis that is noise in the full map agrees only with a topic whose
  counterpart is the noise;
- ``"exclude_either"``: a point that is noise in either labeling is left out, of the compared
  points and of the members that a counterpart is learned from; a topic whose members are all
  full-map noise has no counterpart (``None``), so its assigned theses can only disagree.

Undefined values follow the package's convention, ``None`` with a reason: ``"no_points"`` when
no point is left to compare. An invalid input raises ``ValueError``. No pass threshold is
declared here: O04 declares none.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array
from thematic_redundancy.evaluation.stability import (
    NOISE_HANDLINGS,
    NoiseHandling,
    seed_stability,
)


@dataclass(frozen=True)
class PartitionAgreement:
    """The ARI and the NMI of two labelings of the same points."""

    noise_handling: NoiseHandling
    points: int
    """The points compared."""
    ari: float | None
    nmi: float | None
    undefined_reason: Literal["no_points"] | None


@dataclass(frozen=True)
class CounterpartAgreement:
    """How many assigned theses carry, in the reference map, their topic's counterpart."""

    noise_handling: NoiseHandling
    points: int
    """The assigned theses compared."""
    agreeing: int
    share: float | None
    """``agreeing / points``; ``None`` when no thesis is compared."""
    undefined_reason: Literal["no_points"] | None


def partition_agreement(
    first: Sequence[int] | np.ndarray,
    second: Sequence[int] | np.ndarray,
    *,
    noise_handling: NoiseHandling,
) -> PartitionAgreement:
    """Return the ARI and the NMI of two labelings of the same points, in the same order.

    Raises:
        ValueError: if ``noise_handling`` is unknown, a labeling is invalid (see
            :func:`~.clustering_metrics.label_array`), or the labelings differ in length.
    """
    pair = seed_stability([first, second], noise_handling=noise_handling).pairs[0]
    if pair.ari is None:
        return PartitionAgreement(noise_handling, 0, None, None, "no_points")
    first_kept, second_kept = _kept(label_array(first), label_array(second), noise_handling)
    nmi = float(normalized_mutual_info_score(first_kept, second_kept))
    return PartitionAgreement(noise_handling, pair.points, pair.ari, nmi, None)


def topic_counterparts(
    labels: Sequence[int] | np.ndarray,
    reference: Sequence[int] | np.ndarray,
    *,
    noise_handling: NoiseHandling,
) -> dict[int, int | None]:
    """Return each topic of ``labels`` with its counterpart in ``reference``: the reference
    label of most of the topic's members (a tie goes to the lower label).

    Args:
        labels: the map's labels of its own theses, -1 for its noise, which is no topic.
        reference: the reference map's labels of the same theses, in the same order.
        noise_handling: under ``"exclude_either"``, members that are reference noise are left
            out, and a topic with none left has the counterpart ``None``.

    Raises:
        ValueError: if ``noise_handling`` is unknown, a labeling is invalid, or the labelings
            differ in length.
    """
    map_labels, reference_labels = _pair(labels, reference, noise_handling)
    counterparts: dict[int, int | None] = {}
    for topic in sorted(set(map_labels.tolist()) - {NOISE_LABEL}):
        members = reference_labels[map_labels == topic]
        if noise_handling == "exclude_either":
            members = members[members != NOISE_LABEL]
        counts = Counter(members.tolist())
        counterparts[topic] = (
            min(counts, key=lambda label: (-counts[label], label)) if counts else None
        )
    return counterparts


def counterpart_agreement(
    assigned: Sequence[int] | np.ndarray,
    reference: Sequence[int] | np.ndarray,
    counterparts: Mapping[int, int | None],
    *,
    noise_handling: NoiseHandling,
) -> CounterpartAgreement:
    """Return how many ``assigned`` theses carry in ``reference`` the counterpart of their topic.

    Args:
        assigned: each thesis's assigned topic, never noise.
        reference: the reference map's label of each thesis, in the same order.
        counterparts: :func:`topic_counterparts` of the map the theses were assigned to, with
            the same ``noise_handling``.
        noise_handling: under ``"exclude_either"``, theses that are reference noise are left
            out.

    Raises:
        ValueError: if ``noise_handling`` is unknown, a labeling is invalid, the labelings
            differ in length, an assigned label is noise, or a topic has no counterpart entry.
    """
    topics, reference_labels = _pair(assigned, reference, noise_handling)
    if np.any(topics == NOISE_LABEL):
        raise ValueError("an assigned thesis cannot be noise: assignment always gives a topic")
    missing = sorted(set(topics.tolist()) - set(counterparts))
    if missing:
        raise ValueError(f"topic {missing[0]} has no counterpart entry")
    if noise_handling == "exclude_either":
        kept = reference_labels != NOISE_LABEL
        topics, reference_labels = topics[kept], reference_labels[kept]
    points = int(topics.size)
    if points == 0:
        return CounterpartAgreement(noise_handling, 0, 0, None, "no_points")
    agreeing = sum(
        counterparts[topic] == label
        for topic, label in zip(topics.tolist(), reference_labels.tolist(), strict=True)
    )
    return CounterpartAgreement(noise_handling, points, agreeing, agreeing / points, None)


def _pair(
    first: Sequence[int] | np.ndarray,
    second: Sequence[int] | np.ndarray,
    noise_handling: NoiseHandling,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_handling not in NOISE_HANDLINGS:
        raise ValueError(f"noise_handling must be one of {NOISE_HANDLINGS}, not {noise_handling!r}")
    first_labels, second_labels = label_array(first), label_array(second)
    if first_labels.size != second_labels.size:
        raise ValueError(
            f"the reference has {second_labels.size} labels but the map has {first_labels.size}"
        )
    return first_labels, second_labels


def _kept(
    first: np.ndarray, second: np.ndarray, noise_handling: NoiseHandling
) -> tuple[np.ndarray, np.ndarray]:
    """Return the points that ``noise_handling`` compares, as :mod:`.stability` keeps them."""
    if noise_handling == "exclude_either":
        kept = (first != NOISE_LABEL) & (second != NOISE_LABEL)
        return first[kept], second[kept]
    return first, second
