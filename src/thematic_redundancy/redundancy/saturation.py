"""Topic saturation: how crowded the topic of a proposal is (T23; D13, the ``saturation`` row of
the initial configuration).

An informative signal only (D13): it is reported beside direct redundancy, never flags a
proposal, and never enters the redundancy threshold.

- **Sizes.** A topic's size is its number of member theses, counted from a map's labels; noise
  (:data:`~thematic_redundancy.evaluation.clustering_metrics.NOISE_LABEL`) is no topic.
- **Cuts.** "Quartiles of topic size": the 25th, 50th and 75th percentiles of the sizes, one
  value per topic, by :func:`numpy.percentile` with the method passed.
- **Quartiles by value.** Each topic gets the quartile its size falls in, 1 for the smallest
  topics (the least saturated) to 4 for the largest, so equal sizes always share a quartile. A
  size equal to a cut goes to the quartile below the cut (``boundary="lower"``: quartile k holds
  the sizes above cut k-1 and up to cut k) or above it (``"upper"``: from cut k-1 and below
  cut k). The boundary is an argument with no default. Under ``"lower"`` the smallest size is
  always in quartile 1, and the largest sizes stay in quartile 3 when they tie on the top cut;
  under ``"upper"`` the largest size is always in quartile 4.
- **Undefined.** With no topic, or with every topic of one size, the quartiles mean nothing:
  each topic's quartile is ``None`` and ``undefined_reason`` says why (``"no_topics"``,
  ``"equal_sizes"``), as in :mod:`thematic_redundancy.evaluation`.

The result depends only on how many theses each label has, not on their order. Nothing here
reads or writes a file or holds a thesis text.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array
from thematic_redundancy.topic_map.assignment import PERCENTILE_METHODS, PercentileMethod

SaturationUndefined = Literal["no_topics", "equal_sizes"]
QuartileBoundary = Literal["lower", "upper"]
"""Where a size equal to a cut goes: to the quartile below the cut, or above it."""
QUARTILE_BOUNDARIES: tuple[str, ...] = get_args(QuartileBoundary)
QUARTILE_CUTS = (25.0, 50.0, 75.0)
"""The percentiles of the topic sizes that split them into quartiles."""


@dataclass(frozen=True)
class TopicSaturation:
    """One topic's size and its quartile among the map's topic sizes."""

    topic: int
    size: int
    """The topic's member theses."""
    quartile: int | None
    """1 (the smallest topics) to 4 (the largest); ``None`` when the quartiles are undefined."""

    @property
    def level(self) -> str | None:
        """``"Q1"`` to ``"Q4"``; ``None`` when the quartiles are undefined."""
        return None if self.quartile is None else f"Q{self.quartile}"


@dataclass(frozen=True)
class SaturationQuartiles:
    """The quartiles of the topic sizes of one map."""

    method: PercentileMethod
    boundary: QuartileBoundary
    cuts: tuple[float, float, float] | None
    """The 25th, 50th and 75th percentiles of the sizes; ``None`` when there is no topic."""
    topics: tuple[TopicSaturation, ...]
    """One entry per topic, by ascending label."""
    undefined_reason: SaturationUndefined | None
    """Why every quartile is ``None``; ``None`` when they are defined."""

    def of_topic(self, topic: int) -> TopicSaturation:
        """Return the entry of ``topic``, such as a proposal's assigned topic.

        Raises:
            ValueError: if ``topic`` is not a topic of the labels, such as the noise label.
        """
        for entry in self.topics:
            if entry.topic == topic:
                return entry
        raise ValueError(f"topic {topic} is not a topic of the labels")


def saturation_quartiles(
    labels: Sequence[int] | np.ndarray, *, method: PercentileMethod, boundary: QuartileBoundary
) -> SaturationQuartiles:
    """Return each topic's size and its quartile among the topic sizes.

    Args:
        labels: one label per thesis, as a topic map gives them, -1 for noise.
        method: one of :data:`~thematic_redundancy.topic_map.assignment.PERCENTILE_METHODS`,
            passed to :func:`numpy.percentile` for the cuts.
        boundary: ``"lower"`` or ``"upper"``: where a size equal to a cut goes.

    Raises:
        ValueError: if the method or the boundary is unknown, or the labels are not 1-D
            integers of at least -1.
    """
    if method not in PERCENTILE_METHODS:
        raise ValueError(f"the method must be one of {PERCENTILE_METHODS}, not {method!r}")
    if boundary not in QUARTILE_BOUNDARIES:
        raise ValueError(f"the boundary must be one of {QUARTILE_BOUNDARIES}, not {boundary!r}")
    label_values = label_array(labels)
    topics, sizes = np.unique(label_values[label_values != NOISE_LABEL], return_counts=True)
    if topics.size == 0:
        return SaturationQuartiles(
            method=method, boundary=boundary, cuts=None, topics=(), undefined_reason="no_topics"
        )
    first, second, third = (
        float(cut) for cut in np.percentile(sizes, QUARTILE_CUTS, method=method)
    )
    cuts = (first, second, third)
    if sizes.min() == sizes.max():
        reason: SaturationUndefined | None = "equal_sizes"
        quartiles: list[int | None] = [None] * topics.size
    else:
        reason = None
        side = "left" if boundary == "lower" else "right"
        quartiles = [int(index) + 1 for index in np.searchsorted(cuts, sizes, side=side)]
    return SaturationQuartiles(
        method=method,
        boundary=boundary,
        cuts=cuts,
        topics=tuple(
            TopicSaturation(topic=int(topic), size=int(size), quartile=quartile)
            for topic, size, quartile in zip(topics, sizes, quartiles, strict=True)
        ),
        undefined_reason=reason,
    )
