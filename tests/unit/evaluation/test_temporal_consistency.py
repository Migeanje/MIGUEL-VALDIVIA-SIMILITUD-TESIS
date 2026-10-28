"""Behavior of the temporal split's consistency metrics (T27, O04), on hand-built labelings."""

import math

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from thematic_redundancy.evaluation.stability import NOISE_HANDLINGS
from thematic_redundancy.evaluation.temporal import (
    CounterpartAgreement,
    PartitionAgreement,
    counterpart_agreement,
    partition_agreement,
    topic_counterparts,
)

# Partition agreement: ARI and NMI


@pytest.mark.parametrize("noise_handling", NOISE_HANDLINGS)
def test_two_partitions_that_differ_only_in_their_label_ids_agree_fully(
    noise_handling: str,
) -> None:
    result = partition_agreement([0, 0, 1, 1, 2], [7, 7, 3, 3, 9], noise_handling=noise_handling)

    assert result == PartitionAgreement(
        noise_handling=noise_handling, points=5, ari=1.0, nmi=1.0, undefined_reason=None
    )


def test_ari_and_nmi_take_their_hand_computed_values() -> None:
    # Contingency [[2, 1, 0], [0, 1, 2]] over 6 points: the pair counts give
    # (2 - 6 * 3 / 15) / ((6 + 3) / 2 - 6 * 3 / 15) = 8 / 33; the mutual information is
    # (2 / 3) ln 2, and the arithmetic mean of the entropies (ln 2 + ln 3) / 2.
    result = partition_agreement([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], noise_handling="own_label")

    assert result.points == 6
    assert result.ari == pytest.approx(8 / 33, abs=1e-12)
    assert result.nmi == pytest.approx((2 / 3) * math.log(2) / ((math.log(2) + math.log(3)) / 2))


def test_independent_partitions_share_no_information() -> None:
    result = partition_agreement([0, 0, 1, 1], [0, 1, 0, 1], noise_handling="own_label")

    assert result.nmi == 0.0
    assert result.ari == pytest.approx(-0.5)


def test_own_label_compares_every_point_with_noise_as_one_more_label() -> None:
    first, second = [0, 0, 1, 1, 1, -1], [3, 3, 4, 4, -1, -1]

    result = partition_agreement(first, second, noise_handling="own_label")

    assert result.points == 6
    assert result.ari == pytest.approx(adjusted_rand_score(first, second))
    assert result.nmi == pytest.approx(normalized_mutual_info_score(first, second))
    assert result.ari < 1.0


def test_exclude_either_leaves_out_the_points_that_are_noise_in_either_labeling() -> None:
    # The last two points are noise in one labeling each; the rest is one partition.
    result = partition_agreement(
        [0, 0, 1, 1, 1, -1], [3, 3, 4, 4, -1, 4], noise_handling="exclude_either"
    )

    assert result == PartitionAgreement(
        noise_handling="exclude_either", points=4, ari=1.0, nmi=1.0, undefined_reason=None
    )


def test_nothing_left_to_compare_is_undefined() -> None:
    result = partition_agreement([-1, 0], [1, -1], noise_handling="exclude_either")

    assert result == PartitionAgreement(
        noise_handling="exclude_either",
        points=0,
        ari=None,
        nmi=None,
        undefined_reason="no_points",
    )


@pytest.mark.parametrize(
    ("first", "second", "noise_handling", "message"),
    [
        ([0, 1], [0, 1, 1], "own_label", "3 labels .* 2"),
        ([0, 1], [0, 1], "drop_noise", "noise_handling must be one of"),
        ([0, -2], [0, 1], "own_label", "-1"),
        ([[0, 1]], [0, 1], "own_label", "1-D"),
    ],
)
def test_invalid_labelings_are_refused(
    first: list, second: list, noise_handling: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        partition_agreement(first, second, noise_handling=noise_handling)


# Topic counterparts


def test_a_topics_counterpart_is_the_reference_label_of_most_of_its_members() -> None:
    labels = [0, 0, 0, 1, 1, -1, 2]
    reference = [5, 5, 7, -1, -1, 5, 8]

    assert topic_counterparts(labels, reference, noise_handling="own_label") == {
        0: 5,
        1: -1,
        2: 8,
    }


def test_exclude_either_learns_the_counterparts_from_the_members_clustered_in_both() -> None:
    labels = [0, 0, 0, 1, 1, -1, 2, 2, 2]
    reference = [5, 5, 7, -1, -1, 5, -1, -1, 3]

    assert topic_counterparts(labels, reference, noise_handling="exclude_either") == {
        0: 5,
        1: None,
        2: 3,
    }


def test_a_tie_between_reference_labels_goes_to_the_lower_label() -> None:
    assert topic_counterparts([0, 0, 1, 1], [9, 4, -1, 6], noise_handling="own_label") == {
        0: 4,
        1: -1,
    }


def test_a_map_of_noise_alone_has_no_counterpart() -> None:
    assert topic_counterparts([-1, -1], [0, 1], noise_handling="own_label") == {}


def test_counterparts_need_labelings_of_one_length() -> None:
    with pytest.raises(ValueError, match="3 labels .* 2"):
        topic_counterparts([0, 1], [0, 1, 1], noise_handling="own_label")


# Counterpart agreement


def test_an_assigned_thesis_agrees_when_its_reference_label_is_its_topics_counterpart() -> None:
    result = counterpart_agreement(
        [0, 0, 1, 1, 0], [5, 7, -1, 5, -1], {0: 5, 1: -1}, noise_handling="own_label"
    )

    assert result == CounterpartAgreement(
        noise_handling="own_label", points=5, agreeing=2, share=0.4, undefined_reason=None
    )


def test_exclude_either_leaves_out_the_assigned_theses_that_are_noise_in_the_reference() -> None:
    # Topic 1 has no clustered counterpart, so its theses can only disagree.
    result = counterpart_agreement(
        [0, 0, 1, 1, 0], [5, 7, -1, 5, -1], {0: 5, 1: None}, noise_handling="exclude_either"
    )

    assert result == CounterpartAgreement(
        noise_handling="exclude_either",
        points=3,
        agreeing=1,
        share=pytest.approx(1 / 3),
        undefined_reason=None,
    )


def test_no_assigned_thesis_left_is_undefined() -> None:
    result = counterpart_agreement([0, 0], [-1, -1], {0: 1}, noise_handling="exclude_either")

    assert result == CounterpartAgreement(
        noise_handling="exclude_either",
        points=0,
        agreeing=0,
        share=None,
        undefined_reason="no_points",
    )


def test_the_agreement_does_not_depend_on_how_the_reference_numbers_its_labels() -> None:
    # Each topic's members mostly share one reference label (10 of 15), so no counterpart is a
    # tie, which the lower label would settle differently under another numbering.
    rng = np.random.default_rng(27)
    fitted = np.repeat(np.arange(4), 15)
    majority = np.array([2, 4, 0, 1])
    fitted_reference = np.where(
        np.tile(np.arange(15) < 10, 4), majority[fitted], rng.integers(-1, 5, 60)
    )
    assigned, assigned_reference = rng.integers(0, 4, 40), rng.integers(-1, 5, 40)
    renumber = {-1: -1, 0: 12, 1: 3, 2: 40, 3: 8, 4: 0}

    def renumbered(labels: np.ndarray) -> list[int]:
        return [renumber[int(label)] for label in labels]

    for noise_handling in NOISE_HANDLINGS:
        plain = counterpart_agreement(
            assigned,
            assigned_reference,
            topic_counterparts(fitted, fitted_reference, noise_handling=noise_handling),
            noise_handling=noise_handling,
        )
        other = counterpart_agreement(
            assigned,
            renumbered(assigned_reference),
            topic_counterparts(fitted, renumbered(fitted_reference), noise_handling=noise_handling),
            noise_handling=noise_handling,
        )
        assert (other.points, other.agreeing) == (plain.points, plain.agreeing)


@pytest.mark.parametrize(
    ("assigned", "reference", "counterparts", "message"),
    [
        ([0, -1], [0, 1], {0: 0}, "assigned .* noise"),
        ([0, 2], [0, 1], {0: 0}, "topic 2 has no counterpart"),
        ([0, 0], [0, 1, 1], {0: 0}, "3 labels .* 2"),
    ],
)
def test_invalid_assignments_are_refused(
    assigned: list, reference: list, counterparts: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        counterpart_agreement(assigned, reference, counterparts, noise_handling="own_label")


def test_an_unknown_noise_handling_is_refused() -> None:
    with pytest.raises(ValueError, match="noise_handling must be one of"):
        counterpart_agreement([0], [0], {0: 0}, noise_handling="drop_noise")
    with pytest.raises(ValueError, match="noise_handling must be one of"):
        topic_counterparts([0], [0], noise_handling="drop_noise")
