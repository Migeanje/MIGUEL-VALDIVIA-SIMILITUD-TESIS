"""A new proposal's place on a topic map (T23): nearest-centroid assignment (D14) and the
atypical flag (O06), on hand-built synthetic maps.

The member-to-own-centroid similarities are checked against an independent oracle written
with plain loops, and the threshold against :func:`numpy.percentile` on that oracle.
"""

import inspect
import math
from dataclasses import fields

import numpy as np
import pytest

from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.assignment import (
    ATYPICAL_PERCENTILE,
    AtypicalCheck,
    AtypicalThreshold,
    TopicAssignment,
    TopicCentroids,
    assign_proposal,
    assign_proposals,
    atypical_threshold,
    check_atypical,
    member_similarities,
)
from thematic_redundancy.topic_map.build import TopicMap, topic_centroids
from thematic_redundancy.topic_map.ctfidf import Topic
from thematic_redundancy.topic_map.reduction import Reduction, UmapSettings

SEED = 20261019
UMAP_SETTINGS = UmapSettings(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")


def hand_map(vectors: np.ndarray, labels: np.ndarray) -> TopicMap:
    """A topic map built as T16 builds one, with a UMAP space and 2-D coordinates of their own
    that differ from the original vectors."""
    label_values = np.asarray(labels, dtype=np.int64)
    topics = sorted(set(label_values.tolist()) - {-1})
    rng = np.random.default_rng(SEED)
    points = label_values.size
    return TopicMap(
        seed=SEED,
        clustering=None,
        labels=label_values,
        topics=tuple(
            Topic(topic=topic, size=int(np.sum(label_values == topic)), terms=(), scores=())
            for topic in topics
        ),
        noise=None,
        centroids=topic_centroids(vectors, label_values, topics),
        coordinates=rng.normal(size=(points, 2)),
        reduction=Reduction(
            seed=SEED, settings=UMAP_SETTINGS, embedding=rng.normal(size=(points, 5))
        ),
    )


def blobs(
    sizes: tuple[int, ...], *, noise: int = 0, dimensions: int = 8, spread: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """Theses around one axis per topic (labels 0, 2, 4, ...), then ``noise`` theses around
    the opposite of the first axis, far from every topic, labelled -1."""
    rng = np.random.default_rng(SEED)
    vectors, labels = [], []
    for index, size in enumerate(sizes):
        center = np.eye(dimensions)[index]
        vectors.append(center + rng.normal(0, spread, (size, dimensions)))
        labels += [2 * index] * size
    if noise:
        vectors.append(-np.eye(dimensions)[0] + rng.normal(0, spread, (noise, dimensions)))
        labels += [-1] * noise
    return np.vstack(vectors), np.asarray(labels, dtype=np.int64)


def oracle_member_similarities(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Each clustered thesis's cosine to the plain mean of its topic's members, in row order."""
    similarities = []
    for row, label in zip(vectors, labels, strict=True):
        if label == -1:
            continue
        members = [other for other, own in zip(vectors, labels, strict=True) if own == label]
        centroid = sum(members) / len(members)
        dot = sum(a * b for a, b in zip(row, centroid, strict=True))
        norms = math.sqrt(sum(a * a for a in row)) * math.sqrt(sum(b * b for b in centroid))
        similarities.append(dot / norms)
    return np.asarray(similarities)


def assignment(topic: int, similarity: float) -> TopicAssignment:
    return TopicAssignment(
        topic=topic, similarity=similarity, runner_up=None, runner_up_similarity=None, margin=None
    )


# --- Nearest centroid (D14) --------------------------------------------------------------


def test_a_proposal_goes_to_the_centroid_nearest_by_cosine() -> None:
    centroids = TopicCentroids(topics=(0, 1, 7), centroids=np.eye(3))

    result = assign_proposal([3.0, 1.0, 0.0], centroids)

    assert result.topic == 0
    assert result.similarity == pytest.approx(3 / math.sqrt(10), abs=1e-15)
    assert result.runner_up == 1
    assert result.runner_up_similarity == pytest.approx(1 / math.sqrt(10), abs=1e-15)
    assert result.margin == pytest.approx(2 / math.sqrt(10), abs=1e-15)


def test_the_cosine_decides_not_the_euclidean_distance() -> None:
    centroids = TopicCentroids(topics=(0, 1), centroids=np.array([[10.0, 0.0], [0.5, 0.5]]))
    proposal = np.array([1.0, 0.2])

    result = assign_proposal(proposal, centroids)

    assert result.topic == 0
    assert np.linalg.norm(proposal - [0.5, 0.5]) < np.linalg.norm(proposal - [10.0, 0.0])
    assert result.similarity == pytest.approx(1 / math.sqrt(1.04), abs=1e-15)
    assert result.runner_up_similarity == pytest.approx(0.6 / math.sqrt(0.52), abs=1e-15)


@pytest.mark.parametrize("scale", [1e-4, 0.37, 3.7, 1e5])
def test_the_assignment_ignores_the_scale_of_the_proposal_and_of_each_centroid(
    scale: float,
) -> None:
    vectors, labels = blobs((6, 9, 5))
    plain = hand_map(vectors, labels)
    centroids = TopicCentroids.from_topic_map(plain)
    scaled = TopicCentroids(
        topics=centroids.topics,
        centroids=centroids.centroids * np.array([[scale], [1 / scale], [7.0]]),
    )
    proposals = np.random.default_rng(SEED).normal(size=(40, 8))

    for proposal in proposals:
        reference = assign_proposal(proposal, centroids)
        for result in (
            assign_proposal(proposal * scale, centroids),
            assign_proposal(proposal, scaled),
        ):
            assert (result.topic, result.runner_up) == (reference.topic, reference.runner_up)
            assert result.similarity == pytest.approx(reference.similarity, abs=1e-12)
            assert result.margin == pytest.approx(reference.margin, abs=1e-12)


def test_an_exact_tie_goes_to_the_lower_topic_label() -> None:
    centroids = TopicCentroids(topics=(2, 5), centroids=np.array([[0.0, 1.0], [1.0, 0.0]]))

    result = assign_proposal([1.0, 1.0], centroids)

    assert (result.topic, result.runner_up) == (2, 5)
    assert result.similarity == result.runner_up_similarity
    assert result.margin == 0.0


def test_the_runner_up_of_a_tie_is_the_other_tied_topic() -> None:
    centroids = TopicCentroids(
        topics=(0, 1, 4), centroids=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    )

    result = assign_proposal([0.0, 3.0], centroids)

    assert (result.topic, result.runner_up, result.margin) == (1, 4, 0.0)
    assert result.similarity == pytest.approx(1.0)


def test_the_runner_up_is_never_the_winner() -> None:
    vectors, labels = blobs((5, 7, 6, 8))
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))

    results = assign_proposals(np.random.default_rng(SEED).normal(size=(200, 8)), centroids)

    for result in results:
        assert result.runner_up != result.topic
        assert result.margin is not None and result.margin >= 0
        assert result.margin == result.similarity - result.runner_up_similarity


def test_a_map_with_one_topic_has_no_runner_up() -> None:
    centroids = TopicCentroids(topics=(3,), centroids=np.array([[1.0, 1.0]]))

    result = assign_proposal([1.0, 0.0], centroids)

    assert result == TopicAssignment(
        topic=3,
        similarity=pytest.approx(1 / math.sqrt(2)),
        runner_up=None,
        runner_up_similarity=None,
        margin=None,
    )


def test_the_centroids_are_the_maps_own_in_the_original_space() -> None:
    vectors, labels = blobs((6, 9, 5), noise=4)
    topic_map = hand_map(vectors, labels)

    centroids = TopicCentroids.from_topic_map(topic_map)

    assert centroids.topics == (0, 2, 4)
    assert centroids.dimensions == 8
    np.testing.assert_array_equal(centroids.centroids, topic_map.centroids)
    assert not centroids.centroids.flags.writeable
    assert assign_proposal(vectors[0], centroids).topic == 0


def test_noise_has_no_centroid_so_it_is_never_a_target() -> None:
    members = np.array([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.2, 1.0, 0.0]])
    noise = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.0, 1.0]])
    topic_map = hand_map(np.vstack([members, noise]), np.array([0, 0, 1, 1, -1, -1, -1]))
    centroids = TopicCentroids.from_topic_map(topic_map)

    results = assign_proposals(noise, centroids)

    assert centroids.topics == (0, 1)
    assert {result.topic for result in results} <= {0, 1}
    with pytest.raises(TopicMapError, match="0 or above"):
        TopicCentroids(topics=(-1, 0), centroids=np.eye(2))


def test_many_proposals_are_assigned_as_each_one_alone() -> None:
    vectors, labels = blobs((5, 7, 6))
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    proposals = np.random.default_rng(SEED).normal(size=(25, 8))

    together = assign_proposals(proposals, centroids)

    assert together == tuple(assign_proposal(proposal, centroids) for proposal in proposals)
    assert assign_proposals(np.zeros((0, 8)), centroids) == ()
    assert assign_proposals(proposals, centroids) == together


@pytest.mark.parametrize(
    ("topics", "centroids", "message"),
    [
        ((), np.zeros((0, 2)), "at least one topic"),
        ((0, 0), np.eye(2), "strictly ascending"),
        ((1, 0), np.eye(2), "strictly ascending"),
        ((0.0, 1.0), np.eye(2), "integers"),
        ((0, 1), np.ones(2), "2-D"),
        ((0, 1), np.eye(3), "3 centroids were given for 2 topics"),
        ((0, 1), np.array([[1.0, np.nan], [0.0, 1.0]]), "finite"),
        ((0, 1), np.array([[1.0, 0.0], [0.0, 0.0]]), "topic 1 is a zero vector"),
    ],
)
def test_invalid_centroids_are_refused(topics: tuple, centroids: np.ndarray, message: str) -> None:
    with pytest.raises(TopicMapError, match=message):
        TopicCentroids(topics=topics, centroids=centroids)


@pytest.mark.parametrize(
    ("proposal", "message"),
    [
        ([0.0, 0.0, 0.0], "proposal 0 is a zero vector"),
        ([1.0, 0.0], "2 dimensions, not 3"),
        ([1.0, np.inf, 0.0], "finite"),
        ([[1.0, 0.0, 0.0]], "1-D"),
    ],
)
def test_invalid_proposals_are_refused(proposal: list, message: str) -> None:
    centroids = TopicCentroids(topics=(0, 1, 2), centroids=np.eye(3))

    with pytest.raises(TopicMapError, match=message):
        assign_proposal(proposal, centroids)


def test_a_batch_of_proposals_must_be_a_matrix() -> None:
    centroids = TopicCentroids(topics=(0, 1, 2), centroids=np.eye(3))

    with pytest.raises(TopicMapError, match="2-D"):
        assign_proposals([1.0, 0.0, 0.0], centroids)
    with pytest.raises(TopicMapError, match="proposal 1 is a zero vector"):
        assign_proposals([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], centroids)


# --- Atypical threshold (O06) ------------------------------------------------------------


def test_member_similarities_are_each_members_cosine_to_its_own_centroid() -> None:
    vectors, labels = blobs((6, 9, 5), noise=7)
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))

    similarities = member_similarities(vectors, labels, centroids)

    np.testing.assert_allclose(
        similarities, oracle_member_similarities(vectors, labels), rtol=0, atol=1e-12
    )
    assert similarities.shape == (20,)


def test_a_member_nearer_another_centroid_keeps_its_own_centroids_similarity() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.1], [0.3, 1.0], [0.0, 1.0], [0.1, 1.0]])
    labels = np.array([0, 0, 0, 1, 1])
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))

    similarities = member_similarities(vectors, labels, centroids)

    own = oracle_member_similarities(vectors, labels)[2]
    assert similarities[2] == pytest.approx(own, abs=1e-12)
    assert assign_proposal(vectors[2], centroids).topic == 1
    assert similarities[2] < assign_proposal(vectors[2], centroids).similarity


@pytest.mark.parametrize(
    "method", ["linear", "lower", "higher", "nearest", "midpoint", "hazen", "inverted_cdf"]
)
def test_the_global_threshold_is_numpys_percentile_of_every_members_similarity(
    method: str,
) -> None:
    vectors, labels = blobs((23, 31, 17, 29), noise=12)
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    expected = np.percentile(oracle_member_similarities(vectors, labels), 5, method=method)

    threshold = atypical_threshold(
        vectors, labels, centroids, percentile=ATYPICAL_PERCENTILE, method=method, scope="global"
    )

    assert threshold.values == pytest.approx((expected,) * 4, abs=1e-12)
    assert (threshold.percentile, threshold.method, threshold.scope) == (5.0, method, "global")
    assert threshold.topics == (0, 2, 4, 6)
    assert threshold.members == (23, 31, 17, 29)


@pytest.mark.parametrize("method", ["linear", "lower", "hazen"])
def test_the_per_topic_threshold_is_numpys_percentile_within_each_topic(method: str) -> None:
    vectors, labels = blobs((23, 31, 17, 29), noise=12, spread=0.5)
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    oracle = oracle_member_similarities(vectors, labels)
    clustered = labels[labels != -1]

    threshold = atypical_threshold(
        vectors, labels, centroids, percentile=5, method=method, scope="per_topic"
    )

    expected = [np.percentile(oracle[clustered == t], 5, method=method) for t in (0, 2, 4, 6)]
    assert threshold.values == pytest.approx(tuple(expected), abs=1e-12)
    assert len(set(threshold.values)) == 4
    assert threshold.value_for(4) == threshold.values[2]


def test_the_percentile_is_the_one_passed() -> None:
    vectors, labels = blobs((23, 31, 17, 29))
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    oracle = oracle_member_similarities(vectors, labels)

    for percentile in (0, 5, 50, 95, 100):
        threshold = atypical_threshold(
            vectors, labels, centroids, percentile=percentile, method="linear", scope="global"
        )
        assert threshold.values[0] == pytest.approx(np.percentile(oracle, percentile), abs=1e-12)
    assert ATYPICAL_PERCENTILE == 5.0


def test_noise_theses_never_enter_the_threshold() -> None:
    vectors, labels = blobs((23, 31, 17, 29), noise=40)
    clustered = labels != -1
    with_noise = atypical_threshold(
        vectors,
        labels,
        TopicCentroids.from_topic_map(hand_map(vectors, labels)),
        percentile=5,
        method="linear",
        scope="global",
    )
    without_noise = atypical_threshold(
        vectors[clustered],
        labels[clustered],
        TopicCentroids.from_topic_map(hand_map(vectors[clustered], labels[clustered])),
        percentile=5,
        method="linear",
        scope="global",
    )

    assert with_noise == without_noise
    assert sum(with_noise.members) == 100


def test_the_centroids_may_be_given_l2_normalized() -> None:
    vectors, labels = blobs((6, 9, 5), noise=3)
    plain = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    unit = TopicCentroids(
        topics=plain.topics,
        centroids=plain.centroids / np.linalg.norm(plain.centroids, axis=1, keepdims=True),
    )

    for scope in ("global", "per_topic"):
        first, second = (
            atypical_threshold(vectors, labels, c, percentile=5, method="linear", scope=scope)
            for c in (plain, unit)
        )
        assert first.values == pytest.approx(second.values, abs=1e-12)


def test_the_threshold_arguments_have_no_default() -> None:
    parameters = inspect.signature(atypical_threshold).parameters

    for name in ("percentile", "method", "scope"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


# --- Atypical flag (O06) -----------------------------------------------------------------

THRESHOLD = AtypicalThreshold(
    percentile=5.0,
    method="linear",
    scope="per_topic",
    topics=(0, 3),
    members=(10, 12),
    values=(0.5, 0.8),
)


def test_a_proposal_strictly_below_its_topics_threshold_is_atypical() -> None:
    below = check_atypical(assignment(3, float(np.nextafter(0.8, 0.0))), THRESHOLD)
    equal = check_atypical(assignment(3, 0.8), THRESHOLD)
    above = check_atypical(assignment(3, 0.81), THRESHOLD)

    assert below == AtypicalCheck(
        topic=3, similarity=float(np.nextafter(0.8, 0.0)), threshold=0.8, atypical=True
    )
    assert (equal.atypical, above.atypical) == (False, False)
    assert check_atypical(assignment(0, 0.6), THRESHOLD).atypical is False


def test_a_member_resubmitted_at_the_threshold_is_not_atypical() -> None:
    vectors, labels = blobs((23, 31, 17, 29))
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    threshold = atypical_threshold(
        vectors, labels, centroids, percentile=5, method="lower", scope="global"
    )
    similarities = member_similarities(vectors, labels, centroids)
    index = int(np.flatnonzero(similarities == threshold.values[0])[0])

    check = check_atypical(assign_proposal(vectors[index], centroids), threshold)

    assert check.topic == labels[index]
    assert check.similarity == check.threshold
    assert check.atypical is False


def test_a_topic_without_a_threshold_is_refused() -> None:
    with pytest.raises(TopicMapError, match="topic 1 has no atypical threshold"):
        check_atypical(assignment(1, 0.9), THRESHOLD)


def test_a_proposal_far_from_every_topic_is_flagged_and_a_typical_one_is_not() -> None:
    vectors, labels = blobs((23, 31, 17, 29), noise=12)
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    threshold = atypical_threshold(
        vectors, labels, centroids, percentile=5, method="linear", scope="global"
    )
    far = np.eye(8)[7] + np.eye(8)[6]
    typical = centroids.centroids[1]

    far_check = check_atypical(assign_proposal(far, centroids), threshold)
    typical_check = check_atypical(assign_proposal(typical, centroids), threshold)

    assert far_check.atypical is True
    assert far_check.similarity < far_check.threshold
    assert typical_check == AtypicalCheck(
        topic=2, similarity=pytest.approx(1.0), threshold=threshold.values[1], atypical=False
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"percentile": -0.5}, "percentile"),
        ({"percentile": 100.5}, "percentile"),
        ({"percentile": float("nan")}, "percentile"),
        ({"percentile": True}, "percentile"),
        ({"percentile": "5"}, "percentile"),
        ({"method": "type7"}, "method"),
        ({"scope": "pooled"}, "scope"),
    ],
)
def test_invalid_threshold_arguments_are_refused(arguments: dict, message: str) -> None:
    vectors, labels = blobs((6, 9, 5))
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    chosen = {"percentile": 5, "method": "linear", "scope": "global"} | arguments

    with pytest.raises(TopicMapError, match=message):
        atypical_threshold(vectors, labels, centroids, **chosen)


def test_vectors_or_labels_of_another_map_are_refused() -> None:
    vectors, labels = blobs((6, 9, 5), noise=3)
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    shuffled = np.random.default_rng(SEED).permutation(labels)
    moved = vectors.copy()
    moved[0] = moved[1]

    for wrong_vectors, wrong_labels, message in [
        (vectors, labels[:-1], "22 labels were given for 23 vectors"),
        (vectors[:, :7], labels, "7 dimensions, not 8"),
        (vectors, shuffled, "topic 0 is not the mean of its members"),
        (moved, labels, "topic 0 is not the mean of its members"),
        (vectors, np.where(labels == 4, -1, labels), r"without members \[4\]"),
        (vectors, np.where(labels == 4, 9, labels), r"without a centroid \[9\]"),
    ]:
        with pytest.raises(TopicMapError, match=message):
            atypical_threshold(
                wrong_vectors,
                wrong_labels,
                centroids,
                percentile=5,
                method="linear",
                scope="global",
            )


def test_a_zero_member_vector_is_refused() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    labels = np.array([0, 0, 0, 1, 1])
    centroids = TopicCentroids(topics=(0, 1), centroids=[[1.0, 0.0], [0.0, 1.5]])

    with pytest.raises(TopicMapError, match="thesis 1 is a zero vector"):
        member_similarities(vectors, labels, centroids)


def test_the_results_hold_numbers_only() -> None:
    vectors, labels = blobs((6, 9, 5))
    centroids = TopicCentroids.from_topic_map(hand_map(vectors, labels))
    threshold = atypical_threshold(
        vectors, labels, centroids, percentile=5, method="linear", scope="global"
    )
    result = assign_proposal(vectors[3], centroids)
    check = check_atypical(result, threshold)

    for value in [getattr(result, f.name) for f in fields(result)] + [
        getattr(check, f.name) for f in fields(check)
    ]:
        assert value is None or type(value) in (int, float, bool)
