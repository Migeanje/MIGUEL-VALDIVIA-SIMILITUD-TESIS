"""Alternative research lines for a proposal: its neighbor topics and the gaps near it (T26;
thesis plan section 3.6.7, the ``recommender`` row of the initial configuration).

**Assigned topic.** The proposal goes to its nearest topic centroid by T23's
:func:`~thematic_redundancy.topic_map.assignment.assign_proposal` (D14).

**Neighbor topics.** The ``neighbor_topics`` topics (3 in the configuration) with the highest
cosine between their centroid and a reference, the assigned topic left out. Noise (-1) has no
centroid on a :class:`~thematic_redundancy.topic_map.build.TopicMap`, so it is never a neighbor.
The reference is passed with no default:

- ``"proposal"``: the proposal's own vector, so the first neighbor is T23's runner-up;
- ``"assigned_topic"``: the assigned topic's centroid, the nearness that gap rule (b) uses
  between topics.

An exact tie in cosine goes to the lower topic label, as in T23 and in rule (b).

**Nearby gaps.** The gaps are T24's stable gaps of the shown map
(:func:`~thematic_redundancy.gaps.stability.stable_gaps`, D16); an unstable one is refused. A
gap's distance to the proposal is the cosine distance, ``1 − cosine``, between the proposal's
vector and the gap's anchor:

- a gap of one topic, rules (a), (c) and (d): that topic's centroid;
- a pair of topics, rule (b), by ``pair_anchor``, passed with no default: ``"midpoint"``, the
  normalized sum of the two unit centroids (the midpoint direction rule (b) measures its bridges
  against), or ``"nearer_topic"``, the nearer of the two centroids.

The nearby gaps are the ``nearby_gaps`` gaps (3 in the configuration) of smallest distance,
within ``max_distance`` when it is a number (``None``: no limit), the distance included. An exact
tie keeps the gaps' given order. Several gaps can share a topic, as rule (c) gives one per pair
of programs; each counts as one gap.

Every result carries its similarity and a ``reason``; nothing holds a thesis text, only topic
labels, the topics' c-TF-IDF terms, doc codes inside T24's gaps, and numbers. The proposal
vector is dense, in the map's original space. An invalid argument raises
:class:`~.antecedents.RecommendationError`; an invalid proposal vector raises T23's
:class:`~thematic_redundancy.topic_map.TopicMapError`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, get_args

import numpy as np

from thematic_redundancy.gaps.rules import Gap, InterstitialPair
from thematic_redundancy.gaps.stability import GapSupport
from thematic_redundancy.recommendation.antecedents import RecommendationError
from thematic_redundancy.topic_map.assignment import (
    TopicAssignment,
    TopicCentroids,
    assign_proposal,
)
from thematic_redundancy.topic_map.build import TopicMap

NeighborReference = Literal["proposal", "assigned_topic"]
"""What the neighbor topics are near: the proposal, or its assigned topic's centroid."""
NEIGHBOR_REFERENCES: tuple[str, ...] = get_args(NeighborReference)

PairAnchor = Literal["midpoint", "nearer_topic"]
"""Where a rule (b) pair of topics sits for the distance: their midpoint, or the nearer one."""
PAIR_ANCHORS: tuple[str, ...] = get_args(PairAnchor)


@dataclass(frozen=True)
class NeighborTopic:
    """A topic near the proposal, or near its assigned topic, as an alternative line."""

    rank: int
    """1 for the nearest."""
    topic: int
    similarity: float
    """The cosine between this topic's centroid and the reference."""
    size: int
    """The topic's theses."""
    terms: tuple[str, ...]
    """The topic's top c-TF-IDF terms."""
    assigned_topic: int
    reference: NeighborReference

    @property
    def reason(self) -> str:
        seen_from = (
            "proposal" if self.reference == "proposal" else f"assigned topic {self.assigned_topic}"
        )
        return (
            f"topic {self.topic} ({self.size} theses) is neighbor {self.rank} of the {seen_from} "
            f"by centroid cosine ({self.similarity:.4f}); the assigned topic "
            f"{self.assigned_topic} is left out"
        )


@dataclass(frozen=True)
class NearbyGap:
    """A stable gap near the proposal, as an alternative line."""

    rank: int
    """1 for the nearest."""
    rule: str
    topics: tuple[int, ...]
    """The gap's topic, or its two topics for rule (b)."""
    anchor_topic: int | None
    """The topic whose centroid gave the distance; ``None`` for a pair's midpoint."""
    similarity: float
    """The cosine between the proposal and the gap's anchor."""
    distance: float
    """``1 − similarity``."""
    support: GapSupport
    """The gap and the runs in which it is present."""

    @property
    def reason(self) -> str:
        if len(self.topics) == 1:
            anchor = f"the centroid of topic {self.topics[0]}"
        elif self.anchor_topic is None:
            anchor = f"the midpoint of topics {self.topics[0]} and {self.topics[1]}"
        else:
            anchor = (
                f"the centroid of topic {self.anchor_topic}, the nearer of topics "
                f"{self.topics[0]} and {self.topics[1]}"
            )
        return (
            f"nearby gap {self.rank}: {anchor} lies at cosine distance {self.distance:.4f} from "
            f"the proposal; {self.support.reason}"
        )


@dataclass(frozen=True)
class AlternativeLines:
    """The proposal's assigned topic, its neighbor topics and the gaps near it."""

    assignment: TopicAssignment
    assigned_size: int
    assigned_terms: tuple[str, ...]
    neighbors: tuple[NeighborTopic, ...]
    gaps: tuple[NearbyGap, ...]
    neighbor_reference: NeighborReference
    pair_anchor: PairAnchor
    max_distance: float | None


def alternative_lines(
    proposal: Any,
    topic_map: TopicMap,
    gaps: Sequence[GapSupport],
    *,
    neighbor_topics: int,
    nearby_gaps: int,
    neighbor_reference: NeighborReference,
    pair_anchor: PairAnchor,
    max_distance: float | None,
) -> AlternativeLines:
    """Return the proposal's assigned topic, neighbor topics and nearby gaps on ``topic_map``.

    Args:
        proposal: the proposal's dense document vector, in the map's original space.
        topic_map: the shown map, whose centroids and topic labels the gaps refer to.
        gaps: the stable gaps of that map, in a fixed order.
        neighbor_topics: how many neighbor topics (``recommender.neighbor_topics``), at least 0.
        nearby_gaps: how many nearby gaps (``recommender.nearby_gaps``), at least 0.
        neighbor_reference: ``"proposal"`` or ``"assigned_topic"`` (see the module).
        pair_anchor: ``"midpoint"`` or ``"nearer_topic"`` (see the module).
        max_distance: the largest cosine distance of a nearby gap, from 0 to 2; ``None`` for no
            limit.

    Raises:
        RecommendationError: if an argument is invalid, a gap is not stable, or a gap names a
            topic the map does not have.
        TopicMapError: if the proposal vector is invalid.
    """
    _check_count(neighbor_topics, "neighbor_topics")
    _check_count(nearby_gaps, "nearby_gaps")
    if neighbor_reference not in NEIGHBOR_REFERENCES:
        raise RecommendationError(
            f"neighbor_reference must be one of {NEIGHBOR_REFERENCES}, not {neighbor_reference!r}"
        )
    if pair_anchor not in PAIR_ANCHORS:
        raise RecommendationError(f"pair_anchor must be one of {PAIR_ANCHORS}, not {pair_anchor!r}")
    if max_distance is not None and (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, Real)
        or not (math.isfinite(max_distance) and 0.0 <= max_distance <= 2.0)
    ):
        raise RecommendationError(
            f"max_distance must be None or a number from 0 to 2, not {max_distance!r}"
        )
    centroids = TopicCentroids.from_topic_map(topic_map)
    assignment = assign_proposal(proposal, centroids)
    space = _Space.of(proposal, centroids)
    for support in gaps:
        _check_gap(support, space.rows)
    assigned_row = space.rows[assignment.topic]
    reference = (
        space.from_proposal
        if neighbor_reference == "proposal"
        else space.from_centroid(assigned_row)
    )
    return AlternativeLines(
        assignment=assignment,
        assigned_size=topic_map.topics[assigned_row].size,
        assigned_terms=topic_map.topics[assigned_row].terms,
        neighbors=_neighbors(
            reference, topic_map, assignment.topic, neighbor_topics, neighbor_reference
        ),
        gaps=_nearby(gaps, space, pair_anchor, nearby_gaps, max_distance),
        neighbor_reference=neighbor_reference,
        pair_anchor=pair_anchor,
        max_distance=None if max_distance is None else float(max_distance),
    )


@dataclass(frozen=True, eq=False)
class _Space:
    """The map's unit centroids and the proposal's cosine with each, as T23 computes it."""

    rows: dict[int, int]
    """Each topic label's row in the centroids."""
    unit_centroids: np.ndarray
    unit_proposal: np.ndarray
    from_proposal: np.ndarray

    @classmethod
    def of(cls, proposal: Any, centroids: TopicCentroids) -> "_Space":
        row = np.asarray(proposal, dtype=np.float64)
        unit_proposal = row / np.linalg.norm(row)
        unit_centroids = centroids.centroids / np.linalg.norm(
            centroids.centroids, axis=1, keepdims=True
        )
        return cls(
            rows={topic: index for index, topic in enumerate(centroids.topics)},
            unit_centroids=unit_centroids,
            unit_proposal=unit_proposal,
            from_proposal=np.clip(unit_centroids @ unit_proposal, -1.0, 1.0),
        )

    def from_centroid(self, row: int) -> np.ndarray:
        return np.clip(self.unit_centroids @ self.unit_centroids[row], -1.0, 1.0)


def _neighbors(
    reference: np.ndarray,
    topic_map: TopicMap,
    assigned: int,
    count: int,
    seen_from: NeighborReference,
) -> tuple[NeighborTopic, ...]:
    """Rank the topics by cosine with the reference, descending, an exact tie by label."""
    labels = np.asarray([topic.topic for topic in topic_map.topics])
    order = [int(row) for row in np.lexsort((labels, -reference)) if labels[row] != assigned]
    return tuple(
        NeighborTopic(
            rank=rank,
            topic=int(labels[row]),
            similarity=float(reference[row]),
            size=topic_map.topics[row].size,
            terms=topic_map.topics[row].terms,
            assigned_topic=assigned,
            reference=seen_from,
        )
        for rank, row in enumerate(order[:count], start=1)
    )


def _nearby(
    gaps: Sequence[GapSupport],
    space: _Space,
    pair_anchor: PairAnchor,
    count: int,
    max_distance: float | None,
) -> tuple[NearbyGap, ...]:
    """Rank the gaps by distance to the proposal, ascending, an exact tie by given order."""
    anchored = [_anchor(support.gap, space, pair_anchor) for support in gaps]
    distances = [1.0 - similarity for _, similarity in anchored]
    order = sorted(range(len(gaps)), key=lambda position: (distances[position], position))
    kept = [p for p in order if max_distance is None or distances[p] <= max_distance][:count]
    return tuple(
        NearbyGap(
            rank=rank,
            rule=gaps[position].gap.rule,
            topics=_gap_topics(gaps[position].gap),
            anchor_topic=anchored[position][0],
            similarity=anchored[position][1],
            distance=distances[position],
            support=gaps[position],
        )
        for rank, position in enumerate(kept, start=1)
    )


def _anchor(gap: Gap, space: _Space, pair_anchor: PairAnchor) -> tuple[int | None, float]:
    """Return the topic that anchors the gap (``None`` for a midpoint) and its cosine with the
    proposal."""
    topics = _gap_topics(gap)
    rows = [space.rows[topic] for topic in topics]
    if len(rows) == 1:
        return topics[0], float(space.from_proposal[rows[0]])
    if pair_anchor == "nearer_topic":
        # An exact tie goes to the pair's first topic, the lower label.
        nearer = 0 if space.from_proposal[rows[0]] >= space.from_proposal[rows[1]] else 1
        return topics[nearer], float(space.from_proposal[rows[nearer]])
    midpoint = space.unit_centroids[rows[0]] + space.unit_centroids[rows[1]]
    norm = np.linalg.norm(midpoint)
    if norm == 0:
        raise RecommendationError(
            f"topics {topics[0]} and {topics[1]} have opposite centroids, so no midpoint"
        )
    return None, float(np.clip(midpoint / norm @ space.unit_proposal, -1.0, 1.0))


def _check_gap(support: Any, rows: dict[int, int]) -> None:
    if not isinstance(support, GapSupport):
        raise RecommendationError(
            f"each gap must be a GapSupport of T24's stability filter, not {type(support).__name__}"
        )
    if not support.stable:
        raise RecommendationError(
            f"a rule ({support.gap.rule}) gap is not stable (D16): {support.reason}"
        )
    for topic in _gap_topics(support.gap):
        if topic not in rows:
            raise RecommendationError(
                f"a rule ({support.gap.rule}) gap names topic {topic}, which the map does not "
                "have: the gaps belong to another map"
            )


def _gap_topics(gap: Gap) -> tuple[int, ...]:
    if isinstance(gap, InterstitialPair):
        return (gap.first_topic, gap.second_topic)
    return (gap.topic,)


def _check_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
        raise RecommendationError(f"{name} must be an integer of at least 0, not {value!r}")
