"""Alternative research lines (T26): neighbor topics and nearby gaps, on a hand-built map.

The map lies in the plane ``z = 0``: five topics whose members sit symmetrically around the
angles below, so each centroid points exactly at its topic's angle, and two noise theses beside
the proposal, at 30 degrees. The proposal goes to topic 1 (40 degrees). Seen from the proposal,
topic 0 (30 degrees away) is nearer than topic 2 (45); seen from topic 1's centroid, topic 2
(35) is nearer than topic 0 (40).
"""

import math

import numpy as np
import pytest

from thematic_redundancy.gaps.rules import (
    GapMap,
    InterstitialPair,
    SmallCoherentTopic,
    TemporalDecline,
    ThesisMetadata,
    TransferableTopic,
    interstitial_pairs,
    small_coherent_topics,
    small_topic_cuts,
)
from thematic_redundancy.gaps.stability import GapSupport, stable_gaps
from thematic_redundancy.recommendation.alternatives import (
    AlternativeLines,
    alternative_lines,
)
from thematic_redundancy.recommendation.antecedents import RecommendationError
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.assignment import TopicCentroids, assign_proposal
from thematic_redundancy.topic_map.build import TopicMap, topic_centroids
from thematic_redundancy.topic_map.ctfidf import Topic

ANGLES = {0: 0.0, 1: 40.0, 2: 75.0, 3: 150.0, 4: 250.0}
NOISE_ANGLES = (30.0, 31.0)
PROPOSAL_ANGLE = 30.0


def direction(degrees: float, z: float = 0.0) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array([math.cos(radians), math.sin(radians), z])


PROPOSAL = direction(PROPOSAL_ANGLE)


def cos_degrees(degrees: float) -> float:
    return math.cos(math.radians(degrees))


def build_map(angles=ANGLES, noise_angles=NOISE_ANGLES) -> tuple[TopicMap, np.ndarray]:
    """Return the map and its theses' vectors; odd topics have three members, even ones two."""
    vectors, labels = [], []
    for topic, angle in angles.items():
        offsets = (-5.0, 0.0, 5.0) if topic % 2 else (-5.0, 5.0)
        for offset in offsets:
            vectors.append(direction(angle + offset))
            labels.append(topic)
    for angle in noise_angles:
        vectors.append(direction(angle, z=0.05))
        labels.append(-1)
    return map_of(np.array(vectors), np.array(labels, dtype=np.int64)), np.array(vectors)


def map_of(matrix: np.ndarray, label_array: np.ndarray) -> TopicMap:
    """Return the topic map of these theses, with two made-up terms per topic."""
    ordered = sorted({int(label) for label in label_array if label != -1})
    topics = tuple(
        Topic(
            topic=topic,
            size=int(np.count_nonzero(label_array == topic)),
            terms=(f"termino{topic}", f"tema{topic}"),
            scores=(0.5, 0.25),
        )
        for topic in ordered
    )
    noise_size = int(np.count_nonzero(label_array == -1))
    return TopicMap(
        seed=7,
        clustering=None,
        labels=label_array,
        topics=topics,
        noise=Topic(topic=-1, size=noise_size, terms=("ruido",), scores=(1.0,))
        if noise_size
        else None,
        centroids=topic_centroids(matrix, label_array, ordered),
        coordinates=np.zeros((label_array.size, 2)),
        reduction=None,
    )


TOPIC_MAP, VECTORS = build_map()


def lines(
    gaps=(),
    *,
    topic_map: TopicMap = TOPIC_MAP,
    proposal=PROPOSAL,
    neighbor_topics: int = 3,
    nearby_gaps: int = 3,
    neighbor_reference: str = "proposal",
    pair_anchor: str = "midpoint",
    max_distance: float | None = None,
) -> AlternativeLines:
    return alternative_lines(
        proposal,
        topic_map,
        gaps,
        neighbor_topics=neighbor_topics,
        nearby_gaps=nearby_gaps,
        neighbor_reference=neighbor_reference,
        pair_anchor=pair_anchor,
        max_distance=max_distance,
    )


# --- Gap factories --------------------------------------------------------------------------


def members(topic: int) -> frozenset[str]:
    return frozenset({f"DOC-{topic:04d}a", f"DOC-{topic:04d}b"})


def small(topic: int) -> SmallCoherentTopic:
    return SmallCoherentTopic(
        topic=topic,
        size=2,
        size_threshold=2.0,
        topics_in_map=5,
        npmi=0.3,
        npmi_floor=0.1,
        members=members(topic),
        program_counts=(("sistemas", 2),),
    )


def pair(first: int, second: int) -> InterstitialPair:
    return InterstitialPair(
        first_topic=first,
        second_topic=second,
        centroid_cosine=0.26,
        neighbor_rank=1,
        nearest_topics=3,
        bridge_documents=0,
        bridge_max_docs=1,
        bridges=frozenset(),
        first_members=members(first),
        second_members=members(second),
    )


def transferable(topic: int) -> TransferableTopic:
    return TransferableTopic(
        topic=topic,
        source="industrial",
        target="mecanica",
        affinity=2,
        min_affinity=2,
        source_count=3,
        source_theses=10,
        source_share=0.3,
        target_count=0,
        target_theses=8,
        target_share=0.0,
        present_min_count=3,
        present_min_share=0.2,
        rare_max_share=0.05,
        members=members(topic),
    )


def decline(topic: int) -> TemporalDecline:
    return TemporalDecline(
        topic=topic,
        early_years=(2021, 2023),
        late_years=(2024, 2026),
        early_count=3,
        early_theses=10,
        early_share=0.3,
        late_count=0,
        late_theses=10,
        late_share=0.0,
        decline=0.3,
        min_count=3,
        min_decline=0.1,
        members=members(topic),
        program_counts=(("minas", 3),),
    )


def supported(gap, *, stable: bool = True) -> GapSupport:
    return GapSupport(
        gap=gap,
        overlaps=(1.0, 1.0, 1.0, 1.0, 0.0),
        support=4 if stable else 3,
        min_runs=4,
        min_overlap=0.5,
        stable=stable,
    )


GAPS = (
    supported(small(3)),
    supported(pair(0, 2)),
    supported(decline(1)),
    supported(transferable(0)),
    supported(small(4)),
)
"""Their distances to the proposal: 120, 7.5 (midpoint) or 30 (nearer topic), 10, 30 and
140 degrees."""


def described(found) -> list[tuple[str, tuple[int, ...]]]:
    return [(gap.rule, gap.topics) for gap in found.gaps]


# --- Assigned topic and neighbors -----------------------------------------------------------


def test_the_assigned_topic_is_t23s_nearest_centroid():
    found = lines()

    assert found.assignment == assign_proposal(PROPOSAL, TopicCentroids.from_topic_map(TOPIC_MAP))
    assert found.assignment.topic == 1
    assert (found.assigned_size, found.assigned_terms) == (3, ("termino1", "tema1"))


def test_neighbors_seen_from_the_proposal():
    found = lines(neighbor_reference="proposal")

    assert [n.topic for n in found.neighbors] == [0, 2, 3]
    assert [n.rank for n in found.neighbors] == [1, 2, 3]
    assert [n.similarity for n in found.neighbors] == pytest.approx(
        [cos_degrees(30), cos_degrees(45), cos_degrees(120)], abs=1e-12
    )
    assert [(n.size, n.terms[0]) for n in found.neighbors] == [
        (2, "termino0"),
        (2, "termino2"),
        (3, "termino3"),
    ]
    first = found.neighbors[0]
    assert (first.topic, first.similarity) == (
        found.assignment.runner_up,
        pytest.approx(found.assignment.runner_up_similarity, abs=1e-12),
    )


def test_neighbors_seen_from_the_assigned_topic():
    found = lines(neighbor_reference="assigned_topic")

    assert [n.topic for n in found.neighbors] == [2, 0, 3]
    assert [n.similarity for n in found.neighbors] == pytest.approx(
        [cos_degrees(35), cos_degrees(40), cos_degrees(110)], abs=1e-12
    )
    assert {(n.assigned_topic, n.reference) for n in found.neighbors} == {(1, "assigned_topic")}


@pytest.mark.parametrize("reference", ["proposal", "assigned_topic"])
def test_neighbors_leave_out_the_assigned_topic_and_the_noise(reference):
    found = lines(neighbor_topics=10, neighbor_reference=reference)

    topics = [n.topic for n in found.neighbors]
    assert sorted(topics) == [0, 2, 3, 4]  # the noise theses lie nearest the proposal


def test_a_tie_between_neighbors_goes_to_the_lower_label():
    # Topics 1 and 2 are both orthogonal to the proposal, exactly; topic 3 points away.
    axes = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    topic_map = map_of(axes, np.array([0, 0, 2, 2, 1, 3]))

    found = lines(topic_map=topic_map, proposal=np.array([1.0, 0, 0]), neighbor_topics=2)

    assert [n.topic for n in found.neighbors] == [1, 2]
    assert found.neighbors[0].similarity == found.neighbors[1].similarity


@pytest.mark.parametrize(("count", "expected"), [(0, []), (1, [0]), (2, [0, 2])])
def test_the_number_of_neighbors_is_the_one_asked_for(count, expected):
    assert [n.topic for n in lines(neighbor_topics=count).neighbors] == expected


# --- Nearby gaps ----------------------------------------------------------------------------


def test_gaps_rank_by_distance_with_a_pairs_midpoint():
    found = lines(GAPS, pair_anchor="midpoint")

    assert described(found) == [("b", (0, 2)), ("d", (1,)), ("c", (0,))]
    assert [gap.anchor_topic for gap in found.gaps] == [None, 1, 0]
    assert [gap.distance for gap in found.gaps] == pytest.approx(
        [1 - cos_degrees(7.5), 1 - cos_degrees(10), 1 - cos_degrees(30)], abs=1e-12
    )
    assert [gap.similarity + gap.distance for gap in found.gaps] == pytest.approx([1.0] * 3)
    assert [gap.support for gap in found.gaps] == [GAPS[1], GAPS[2], GAPS[3]]


def test_gaps_rank_by_distance_with_a_pairs_nearer_topic_and_keep_the_order_of_ties():
    found = lines(GAPS, pair_anchor="nearer_topic")

    assert described(found) == [("d", (1,)), ("b", (0, 2)), ("c", (0,))]
    assert [gap.anchor_topic for gap in found.gaps] == [1, 0, 0]
    assert found.gaps[1].distance == found.gaps[2].distance


def test_every_gap_ranks_when_more_are_asked_for():
    found = lines(GAPS, nearby_gaps=10)

    assert described(found) == [("b", (0, 2)), ("d", (1,)), ("c", (0,)), ("a", (3,)), ("a", (4,))]
    assert [gap.rank for gap in found.gaps] == [1, 2, 3, 4, 5]
    assert lines(GAPS, nearby_gaps=0).gaps == ()


def test_max_distance_keeps_the_gaps_within_it_inclusive():
    limit = lines(GAPS).gaps[2].distance  # the transferable gap on topic 0, 30 degrees away

    within = lines(GAPS, nearby_gaps=10, max_distance=0.1)
    at_limit = lines(GAPS, nearby_gaps=10, max_distance=limit)

    assert described(within) == [("b", (0, 2)), ("d", (1,))]
    assert described(at_limit) == [("b", (0, 2)), ("d", (1,)), ("c", (0,))]


def test_the_t24_rules_and_filter_feed_the_recommender():
    gap_map = GapMap(
        labels=TOPIC_MAP.labels,
        topics=tuple(topic.topic for topic in TOPIC_MAP.topics),
        centroids=TOPIC_MAP.centroids,
        npmi=(0.2, 0.2, 0.2, 0.2, 0.2),
    )
    count = TOPIC_MAP.labels.size
    metadata = ThesisMetadata(
        doc_codes=tuple(f"DOC-{row:04d}" for row in range(count)),
        programs=("sistemas",) * count,
        years=(2022,) * count,
    )
    size_cut = small_topic_cuts([gap_map], npmi_floor_quantile=0.0).size_cut
    run = small_coherent_topics(
        gap_map, metadata, size_cut=size_cut, npmi_floor=0.1
    ) + interstitial_pairs(gap_map, VECTORS, metadata, nearest_topics=3, bridge_max_docs=1)
    stable = stable_gaps([run] * 5, reference=0, min_runs=4, min_overlap=1.0)

    found = lines(stable, nearby_gaps=len(stable))

    assert len(found.gaps) == len(stable) > 0
    distances = [gap.distance for gap in found.gaps]
    assert distances == sorted(distances)


# --- Evidence, determinism and invalid inputs -----------------------------------------------


def test_reasons_state_the_evidence():
    found = lines(GAPS)

    neighbor = found.neighbors[0].reason
    assert "topic 0" in neighbor and "neighbor 1" in neighbor and "0.8660" in neighbor
    assert "assigned topic 1" in neighbor
    gap = found.gaps[0].reason
    assert "midpoint of topics 0 and 2" in gap and f"{1 - cos_degrees(7.5):.4f}" in gap
    assert GAPS[1].reason in gap
    assert "DOC-" not in neighbor + gap


def test_the_result_repeats_exactly_and_records_its_arguments():
    found = lines(GAPS, neighbor_reference="assigned_topic", pair_anchor="nearer_topic")

    assert found == lines(GAPS, neighbor_reference="assigned_topic", pair_anchor="nearer_topic")
    assert (found.neighbor_reference, found.pair_anchor, found.max_distance) == (
        "assigned_topic",
        "nearer_topic",
        None,
    )


def test_an_unstable_gap_or_a_gap_of_another_map_is_refused():
    with pytest.raises(RecommendationError, match="not stable"):
        lines((supported(small(2), stable=False),))
    with pytest.raises(RecommendationError, match="topic 9"):
        lines((supported(pair(1, 9)),))


@pytest.mark.parametrize(
    "arguments",
    [
        {"neighbor_topics": -1},
        {"neighbor_topics": True},
        {"nearby_gaps": -1},
        {"nearby_gaps": 1.0},
        {"neighbor_reference": "centroid"},
        {"pair_anchor": "mean"},
        {"max_distance": -0.1},
        {"max_distance": 2.5},
        {"max_distance": math.nan},
        {"max_distance": True},
    ],
)
def test_an_invalid_argument_is_refused(arguments):
    with pytest.raises(RecommendationError):
        lines(GAPS, **arguments)


def test_an_invalid_proposal_is_refused_by_t23():
    with pytest.raises(TopicMapError):
        lines(proposal=np.zeros(3))
    with pytest.raises(TopicMapError):
        lines(proposal=np.ones(4))
