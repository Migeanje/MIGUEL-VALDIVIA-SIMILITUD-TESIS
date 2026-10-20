"""A new proposal's place on a topic map: nearest-centroid assignment and the atypical flag
(T23; D14, O06).

**Assignment (D14).** A proposal's document vector, computed elsewhere with the corpus's model
and document-vector rule (D09), goes to the topic whose centroid is nearest by cosine in the
original space; nothing is re-clustered, and the rule is the same for K-means and HDBSCAN maps.
The centroids are the map's (:class:`~.build.TopicMap`), each the plain mean of its members'
original vectors (D30 E5); noise has none, so it is never a target. The result names the
nearest topic and its similarity, the runner-up and its similarity, and the margin between
them. An exact tie in similarity goes to the lower topic label, so the runner-up of a tie is the
other tied topic, with a margin of 0.

**Atypical flag (O06).** "Flag it when its similarity to the nearest centroid is below the P5
of member-to-own-centroid similarities." A member-to-own-centroid similarity is the cosine
between a clustered thesis's original vector and its own topic's centroid; noise theses are
left out. The threshold is :func:`numpy.percentile` of those similarities, with the percentile
(:data:`ATYPICAL_PERCENTILE` for O06), numpy's method and the scope passed with no default: one
threshold for the map (``"global"``), over every clustered thesis, or one per topic
(``"per_topic"``), over that topic's members. A proposal is atypical when its similarity is
strictly below the threshold of its topic.

Every cosine, a member's or a proposal's, is computed one vector at a time by the same steps,
so a thesis submitted as a proposal gets exactly its member similarity to that centroid, and a
batch gives what each proposal gives alone. Cosines are clipped to [-1, 1] against rounding.
Nothing here reads or writes a file or holds a thesis text. An invalid input raises
:class:`~thematic_redundancy.topic_map.TopicMapError`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Self, get_args

import numpy as np

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.build import TopicMap, topic_centroids
from thematic_redundancy.topic_map.reduction import dense_vectors

ATYPICAL_PERCENTILE = 5.0
"""O06's percentile: below the P5 of the member-to-own-centroid similarities is atypical."""

PercentileMethod = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]
"""The methods of :func:`numpy.percentile`; numpy's own default is ``"linear"``."""
PERCENTILE_METHODS: tuple[str, ...] = get_args(PercentileMethod)

ThresholdScope = Literal["global", "per_topic"]
"""One threshold for the whole map, or one per topic."""
THRESHOLD_SCOPES: tuple[str, ...] = get_args(ThresholdScope)

CENTROID_DIRECTION_TOLERANCE = 1e-6
"""How far each component of a centroid's unit vector may lie from that of the mean of its
members' vectors, which absorbs rounding and float32 storage, not another labeling."""


@dataclass(frozen=True)
class TopicCentroids:
    """The targets of the assignment: one map's topic labels and their centroids."""

    topics: tuple[int, ...]
    """The topic labels, strictly ascending and at least 0: noise is never a target."""
    centroids: np.ndarray
    """``topics × dimensions``, row ``i`` for ``topics[i]``, in the original space; a
    read-only float64 copy of what was given."""

    def __post_init__(self) -> None:
        topics = _topic_labels(self.topics)
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "centroids", _centroid_matrix(self.centroids, topics))

    @classmethod
    def from_topic_map(cls, topic_map: TopicMap) -> Self:
        """Return the map's topics and its centroids in the original space (D14)."""
        return cls(
            topics=tuple(topic.topic for topic in topic_map.topics), centroids=topic_map.centroids
        )

    @property
    def dimensions(self) -> int:
        return int(self.centroids.shape[1])


@dataclass(frozen=True)
class TopicAssignment:
    """A proposal's nearest topic by cosine to the centroids (D14)."""

    topic: int
    similarity: float
    """The cosine between the proposal and the topic's centroid."""
    runner_up: int | None
    """The second-nearest topic; ``None`` when the map has one topic."""
    runner_up_similarity: float | None
    margin: float | None
    """``similarity - runner_up_similarity``, never negative; ``None`` without a runner-up."""


@dataclass(frozen=True)
class AtypicalThreshold:
    """O06's threshold of one topic map."""

    percentile: float
    method: PercentileMethod
    scope: ThresholdScope
    topics: tuple[int, ...]
    members: tuple[int, ...]
    """The clustered theses of each topic, in topic order."""
    values: tuple[float, ...]
    """The threshold that applies to each topic, in topic order; one value repeated when the
    scope is global."""

    def value_for(self, topic: int) -> float:
        """Return the threshold that applies to ``topic``.

        Raises:
            TopicMapError: if ``topic`` is not one of the map's topics.
        """
        if topic not in self.topics:
            raise TopicMapError(f"topic {topic} has no atypical threshold")
        return self.values[self.topics.index(topic)]


@dataclass(frozen=True)
class AtypicalCheck:
    """O06's flag for one assigned proposal."""

    topic: int
    similarity: float
    threshold: float
    atypical: bool
    """``similarity < threshold``: strictly below."""


def assign_proposal(vector: Any, centroids: TopicCentroids) -> TopicAssignment:
    """Return the topic whose centroid is nearest to ``vector`` by cosine (D14).

    Args:
        vector: the proposal's document vector: 1-D, finite, not zero, and of the centroids'
            dimensions.
        centroids: the map's topics and centroids.

    Raises:
        TopicMapError: if the vector is invalid.
    """
    row = np.asarray(vector, dtype=np.float64)
    if row.ndim != 1:
        raise TopicMapError(f"the proposal vector must be 1-D, not of shape {row.shape}")
    return assign_proposals(row[np.newaxis, :], centroids)[0]


def assign_proposals(vectors: Any, centroids: TopicCentroids) -> tuple[TopicAssignment, ...]:
    """Return the assignment of each row of ``vectors``, as :func:`assign_proposal` gives it.

    Raises:
        TopicMapError: if ``vectors`` is not a 2-D matrix of finite, non-zero rows of the
            centroids' dimensions.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise TopicMapError(f"the proposal vectors must be 2-D, not of shape {matrix.shape}")
    if matrix.shape[1] != centroids.dimensions:
        raise TopicMapError(
            f"the proposal vectors have {matrix.shape[1]} dimensions, not {centroids.dimensions}"
        )
    if not np.isfinite(matrix).all():
        raise TopicMapError("the proposal vectors must hold finite values only")
    unit_centroids = _unit_rows(centroids.centroids)
    topics = np.asarray(centroids.topics)
    assignments = []
    for index, row in enumerate(matrix):
        unit = _unit_vector(row, f"proposal {index}")
        assignments.append(_nearest(_cosines(unit, unit_centroids), topics))
    return tuple(assignments)


def member_similarities(
    vectors: Any, labels: Sequence[int] | np.ndarray, centroids: TopicCentroids
) -> np.ndarray:
    """Return the cosine between each clustered thesis's vector and its own topic's centroid,
    in the theses' order, the noise left out (O06).

    Args:
        vectors: the theses' original document vectors, one row per thesis, the ones the map
            was built from.
        labels: the map's labels, -1 for noise.
        centroids: the map's topics and centroids; every topic must have members, and each
            centroid must point where the mean of its members' vectors points (L2-normalized
            centroids are accepted).

    Raises:
        TopicMapError: if the vectors, the labels and the centroids do not belong together, or
            a clustered thesis's vector is zero.
        ValueError: if the labels are not 1-D integers of at least -1.
    """
    matrix = dense_vectors(vectors)
    label_values = label_array(labels)
    if label_values.size != matrix.shape[0]:
        raise TopicMapError(f"{label_values.size} labels were given for {matrix.shape[0]} vectors")
    if matrix.shape[1] != centroids.dimensions:
        raise TopicMapError(
            f"the vectors have {matrix.shape[1]} dimensions, not {centroids.dimensions}"
        )
    _check_topics(label_values, centroids.topics)
    clustered = np.flatnonzero(label_values != NOISE_LABEL)
    units = [_unit_vector(matrix[index], f"thesis {index}") for index in clustered]
    _check_member_means(matrix, label_values, centroids)
    unit_centroids = _unit_rows(centroids.centroids)
    rows = np.searchsorted(np.asarray(centroids.topics), label_values[clustered])
    return np.array(
        [_cosines(unit, unit_centroids)[row] for unit, row in zip(units, rows, strict=True)],
        dtype=np.float64,
    )


def atypical_threshold(
    vectors: Any,
    labels: Sequence[int] | np.ndarray,
    centroids: TopicCentroids,
    *,
    percentile: float,
    method: PercentileMethod,
    scope: ThresholdScope,
) -> AtypicalThreshold:
    """Return O06's threshold: the ``percentile`` of the member-to-own-centroid similarities,
    over the whole map or within each topic.

    Args:
        vectors: as :func:`member_similarities` takes them.
        labels: as :func:`member_similarities` takes them.
        centroids: as :func:`member_similarities` takes them.
        percentile: 0 to 100; O06 declares :data:`ATYPICAL_PERCENTILE`.
        method: one of :data:`PERCENTILE_METHODS`, passed to :func:`numpy.percentile`.
        scope: ``"global"`` or ``"per_topic"``.

    Raises:
        TopicMapError: if an argument is invalid, or as :func:`member_similarities` raises.
        ValueError: if the labels are not 1-D integers of at least -1.
    """
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, Real)
        or not 0 <= percentile <= 100
    ):
        raise TopicMapError(f"the percentile must be a number from 0 to 100, not {percentile!r}")
    if method not in PERCENTILE_METHODS:
        raise TopicMapError(f"the method must be one of {PERCENTILE_METHODS}, not {method!r}")
    if scope not in THRESHOLD_SCOPES:
        raise TopicMapError(f"the scope must be one of {THRESHOLD_SCOPES}, not {scope!r}")
    similarities = member_similarities(vectors, labels, centroids)
    label_values = label_array(labels)
    member_labels = label_values[label_values != NOISE_LABEL]
    members = tuple(int(np.count_nonzero(member_labels == topic)) for topic in centroids.topics)
    if scope == "global":
        value = float(np.percentile(similarities, percentile, method=method))
        values = (value,) * len(centroids.topics)
    else:
        values = tuple(
            float(np.percentile(similarities[member_labels == topic], percentile, method=method))
            for topic in centroids.topics
        )
    return AtypicalThreshold(
        percentile=float(percentile),
        method=method,
        scope=scope,
        topics=centroids.topics,
        members=members,
        values=values,
    )


def check_atypical(assignment: TopicAssignment, threshold: AtypicalThreshold) -> AtypicalCheck:
    """Return O06's flag: whether the proposal's similarity to its nearest centroid is strictly
    below the threshold of that topic.

    Raises:
        TopicMapError: if the threshold has no value for the assigned topic.
    """
    limit = threshold.value_for(assignment.topic)
    return AtypicalCheck(
        topic=assignment.topic,
        similarity=assignment.similarity,
        threshold=limit,
        atypical=assignment.similarity < limit,
    )


def _nearest(similarities: np.ndarray, topics: np.ndarray) -> TopicAssignment:
    """Rank the topics by similarity, descending, an exact tie by label, ascending."""
    order = np.lexsort((topics, -similarities))
    best = order[0]
    topic, similarity = int(topics[best]), float(similarities[best])
    if order.size == 1:
        return TopicAssignment(
            topic=topic,
            similarity=similarity,
            runner_up=None,
            runner_up_similarity=None,
            margin=None,
        )
    second = order[1]
    runner_up_similarity = float(similarities[second])
    return TopicAssignment(
        topic=topic,
        similarity=similarity,
        runner_up=int(topics[second]),
        runner_up_similarity=runner_up_similarity,
        margin=similarity - runner_up_similarity,
    )


def _cosines(unit: np.ndarray, unit_centroids: np.ndarray) -> np.ndarray:
    return np.clip(unit_centroids @ unit, -1.0, 1.0)


def _unit_vector(row: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(row)
    if norm == 0:
        raise TopicMapError(f"{name} is a zero vector, which has no cosine")
    return row / norm


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _topic_labels(topics: Any) -> tuple[int, ...]:
    values = np.asarray(topics)
    if values.ndim != 1 or values.size == 0:
        raise TopicMapError("the centroids need at least one topic, in a 1-D sequence")
    if values.dtype.kind not in "iu":
        raise TopicMapError(f"the topic labels must be integers, not {values.dtype}")
    if values.min() < 0:
        raise TopicMapError(
            f"the topic labels must be 0 or above, not {values.min()}: noise "
            f"({NOISE_LABEL}) has no centroid"
        )
    if np.any(np.diff(values) <= 0):
        raise TopicMapError("the topic labels must be strictly ascending")
    return tuple(int(value) for value in values)


def _centroid_matrix(centroids: Any, topics: tuple[int, ...]) -> np.ndarray:
    matrix = np.array(centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise TopicMapError(f"the centroids must be 2-D, not of shape {matrix.shape}")
    if matrix.shape[0] != len(topics):
        raise TopicMapError(f"{matrix.shape[0]} centroids were given for {len(topics)} topics")
    if matrix.shape[1] == 0:
        raise TopicMapError("the centroids need at least one dimension")
    if not np.isfinite(matrix).all():
        raise TopicMapError("the centroids must hold finite values only")
    zero = np.flatnonzero(np.linalg.norm(matrix, axis=1) == 0)
    if zero.size:
        raise TopicMapError(
            f"the centroid of topic {topics[zero[0]]} is a zero vector, which has no cosine"
        )
    matrix.setflags(write=False)
    return matrix


def _check_topics(labels: np.ndarray, topics: tuple[int, ...]) -> None:
    found = set(np.unique(labels[labels != NOISE_LABEL]).tolist())
    if found != set(topics):
        raise TopicMapError(
            "the labels and the centroids name different topics: without members "
            f"{sorted(set(topics) - found)}, without a centroid {sorted(found - set(topics))}"
        )


def _check_member_means(matrix: np.ndarray, labels: np.ndarray, centroids: TopicCentroids) -> None:
    """Check that each centroid points where the mean of its members' vectors points."""
    means = topic_centroids(matrix, labels, centroids.topics)
    norms = np.linalg.norm(means, axis=1)
    given = _unit_rows(centroids.centroids)
    for row, topic in enumerate(centroids.topics):
        if norms[row] == 0 or not np.allclose(
            means[row] / norms[row], given[row], rtol=0, atol=CENTROID_DIRECTION_TOLERANCE
        ):
            raise TopicMapError(
                f"the centroid of topic {topic} is not the mean of its members' vectors: the "
                "vectors or the labels belong to another map"
            )
