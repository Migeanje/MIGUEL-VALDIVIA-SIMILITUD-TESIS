"""Topic saturation (T23, D13): the quartiles of topic size, on synthetic labels.

The cuts are checked against :func:`numpy.percentile` of the sizes, and each quartile against
an oracle that counts the cuts below the size (or not above it, by the boundary).
"""

import inspect

import numpy as np
import pytest

from thematic_redundancy.redundancy.saturation import (
    QUARTILE_CUTS,
    SaturationQuartiles,
    TopicSaturation,
    saturation_quartiles,
)
from thematic_redundancy.topic_map.assignment import TopicCentroids, assign_proposal

SEED = 20261019


def labels_of(sizes: dict[int, int], *, noise: int = 0) -> np.ndarray:
    """One label per thesis, ``sizes[topic]`` theses per topic plus ``noise`` noise theses,
    in a shuffled order."""
    labels = [topic for topic, size in sizes.items() for _ in range(size)] + [-1] * noise
    return np.random.default_rng(SEED).permutation(np.asarray(labels, dtype=np.int64))


def quartiles(result: SaturationQuartiles) -> list[int | None]:
    return [entry.quartile for entry in result.topics]


# --- Sizes and cuts ----------------------------------------------------------------------


def test_each_topic_is_sized_by_its_members_and_noise_is_no_topic() -> None:
    labels = labels_of({3: 4, 0: 7, 11: 2, 5: 9}, noise=30)

    result = saturation_quartiles(labels, method="linear", boundary="lower")

    assert [(entry.topic, entry.size) for entry in result.topics] == [
        (0, 7),
        (3, 4),
        (5, 9),
        (11, 2),
    ]


def test_eight_evenly_spaced_sizes_fall_two_per_quartile() -> None:
    labels = labels_of({topic: 2 * (topic + 1) for topic in range(8)}, noise=5)

    for boundary in ("lower", "upper"):
        result = saturation_quartiles(labels, method="linear", boundary=boundary)

        assert result.cuts == (5.5, 9.0, 12.5)
        assert quartiles(result) == [1, 1, 2, 2, 3, 3, 4, 4]
        assert [entry.level for entry in result.topics] == ["Q1", "Q1", "Q2", "Q2"] + [
            "Q3",
            "Q3",
            "Q4",
            "Q4",
        ]
        assert result.undefined_reason is None


@pytest.mark.parametrize(
    "method", ["linear", "lower", "higher", "nearest", "midpoint", "hazen", "inverted_cdf"]
)
@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_the_cuts_are_numpys_quartiles_of_the_sizes_and_each_size_its_quartile(
    method: str, boundary: str
) -> None:
    rng = np.random.default_rng(SEED)
    sizes = {topic: int(size) for topic, size in enumerate(rng.integers(5, 60, 37))}

    result = saturation_quartiles(labels_of(sizes, noise=50), method=method, boundary=boundary)

    values = np.array(list(sizes.values()))
    cuts = np.percentile(values, [25, 50, 75], method=method)
    assert result.cuts == tuple(cuts)
    for entry in result.topics:
        below = cuts < entry.size if boundary == "lower" else cuts <= entry.size
        assert entry.quartile == 1 + int(np.sum(below))
    assert (result.method, result.boundary) == (method, boundary)
    assert QUARTILE_CUTS == (25.0, 50.0, 75.0)


def test_equal_sizes_share_a_quartile_and_a_size_on_a_cut_follows_the_boundary() -> None:
    labels = labels_of(dict(enumerate([5, 5, 5, 5, 6, 7, 8, 20])))

    lower = saturation_quartiles(labels, method="linear", boundary="lower")
    upper = saturation_quartiles(labels, method="linear", boundary="upper")

    assert lower.cuts == upper.cuts == (5.0, 5.5, 7.25)
    assert quartiles(lower) == [1, 1, 1, 1, 3, 3, 4, 4]
    assert quartiles(upper) == [2, 2, 2, 2, 3, 3, 4, 4]


def test_the_largest_sizes_tied_on_the_top_cut_stay_in_the_third_quartile_when_lower() -> None:
    labels = labels_of(dict(enumerate([5, 6, 7, 9, 9, 9, 9, 9])))

    lower = saturation_quartiles(labels, method="linear", boundary="lower")
    upper = saturation_quartiles(labels, method="linear", boundary="upper")

    assert lower.cuts == (6.75, 9.0, 9.0)
    assert quartiles(lower) == [1, 1, 2, 2, 2, 2, 2, 2]
    assert quartiles(upper) == [1, 1, 2, 4, 4, 4, 4, 4]


def test_the_order_of_the_theses_does_not_matter() -> None:
    labels = labels_of({0: 3, 1: 8, 2: 13, 3: 21, 4: 34}, noise=9)
    reference = saturation_quartiles(labels, method="linear", boundary="lower")

    for seed in range(5):
        shuffled = np.random.default_rng(seed).permutation(labels)
        assert saturation_quartiles(shuffled, method="linear", boundary="lower") == reference


# --- Undefined quartiles -----------------------------------------------------------------


@pytest.mark.parametrize(
    "labels", [labels_of({0: 4, 1: 4, 2: 4}, noise=3), labels_of({7: 9}), [4, 4, -1]]
)
def test_topics_all_of_one_size_have_no_quartile(labels: np.ndarray) -> None:
    result = saturation_quartiles(labels, method="linear", boundary="lower")

    assert result.undefined_reason == "equal_sizes"
    assert quartiles(result) == [None] * len(result.topics)
    assert {entry.level for entry in result.topics} == {None}
    assert len(set(result.cuts)) == 1


@pytest.mark.parametrize("labels", [[], [-1, -1, -1]])
def test_a_map_without_topics_has_no_quartiles(labels: list[int]) -> None:
    result = saturation_quartiles(labels, method="linear", boundary="upper")

    assert result == SaturationQuartiles(
        method="linear", boundary="upper", cuts=None, topics=(), undefined_reason="no_topics"
    )


# --- The proposal's topic ----------------------------------------------------------------


def test_the_saturation_of_a_proposals_assigned_topic() -> None:
    result = saturation_quartiles(
        labels_of({0: 3, 2: 10, 5: 30, 8: 60}, noise=12), method="linear", boundary="lower"
    )
    centroids = TopicCentroids(topics=(0, 2, 5, 8), centroids=np.eye(4))

    assigned = assign_proposal([0.1, 0.2, 0.9, 0.3], centroids)

    assert result.of_topic(assigned.topic) == TopicSaturation(topic=5, size=30, quartile=3)
    assert result.of_topic(assigned.topic).level == "Q3"
    assert result.of_topic(8).level == "Q4"


@pytest.mark.parametrize("topic", [-1, 1, 99])
def test_a_label_that_is_no_topic_has_no_saturation(topic: int) -> None:
    result = saturation_quartiles([0, 0, 2, 2, 2, -1], method="linear", boundary="lower")

    with pytest.raises(ValueError, match=f"topic {topic} is not a topic of the labels"):
        result.of_topic(topic)


# --- Arguments ---------------------------------------------------------------------------


def test_the_method_and_the_boundary_have_no_default() -> None:
    parameters = inspect.signature(saturation_quartiles).parameters

    for name in ("method", "boundary"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("labels", "arguments", "message"),
    [
        ([0, 1, 1], {"method": "type7"}, "method"),
        ([0, 1, 1], {"boundary": "middle"}, "boundary"),
        ([[0, 1], [1, 1]], {}, "1-D"),
        ([0, -2, 1], {}, "-1 or above"),
        ([0.0, 1.0], {}, "integers"),
    ],
)
def test_invalid_arguments_are_refused(labels: list, arguments: dict, message: str) -> None:
    chosen = {"method": "linear", "boundary": "lower"} | arguments

    with pytest.raises(ValueError, match=message):
        saturation_quartiles(labels, **chosen)


def test_the_result_holds_numbers_and_declared_names_only() -> None:
    result = saturation_quartiles(labels_of({0: 3, 1: 8, 2: 13}), method="hazen", boundary="upper")

    assert all(isinstance(cut, float) for cut in result.cuts)
    for entry in result.topics:
        assert (type(entry.topic), type(entry.size), type(entry.quartile)) == (int, int, int)
