"""Stability across seeds: the pairwise adjusted Rand index of several labelings (T17, D12).

Each pair is checked against scikit-learn and against an independent oracle, the contingency
formula of Hubert and Arabie written in plain numpy.
"""

import inspect
from itertools import combinations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from thematic_redundancy.evaluation.stability import SeedStability, seed_stability

SEED = 20261019


def pairs_of(n: np.ndarray) -> np.ndarray:
    return n * (n - 1) / 2


def oracle_ari(first: np.ndarray, second: np.ndarray) -> float:
    """Hubert and Arabie's adjusted Rand index from the contingency table."""
    _, first_ids = np.unique(first, return_inverse=True)
    _, second_ids = np.unique(second, return_inverse=True)
    table = np.zeros((first_ids.max() + 1, second_ids.max() + 1))
    np.add.at(table, (first_ids, second_ids), 1)
    index = pairs_of(table).sum()
    rows, columns = pairs_of(table.sum(axis=1)).sum(), pairs_of(table.sum(axis=0)).sum()
    expected = rows * columns / pairs_of(np.array(first.size))
    return float((index - expected) / ((rows + columns) / 2 - expected))


def by_label(labelings, handling: str = "own_label") -> SeedStability:
    return seed_stability(labelings, noise_handling=handling)


# --- Known structure ---------------------------------------------------------------------


def test_relabelled_copies_of_one_clustering_agree_exactly() -> None:
    labels = np.random.default_rng(SEED).integers(-1, 5, 200)
    renamed = np.where(labels == -1, -1, (labels * 3 + 7) % 5 + 10)

    for handling in ("own_label", "exclude_either"):
        result = by_label([labels, renamed, labels.copy()], handling)

        assert result.mean == 1.0
        assert result.minimum == 1.0
        assert all(pair.ari == 1.0 for pair in result.pairs)


def test_independent_random_labelings_agree_by_chance_only() -> None:
    rng = np.random.default_rng(SEED)
    labelings = [rng.integers(0, 5, 3000) for _ in range(4)]

    result = by_label(labelings)

    assert abs(result.mean) < 0.01
    assert all(abs(pair.ari) < 0.01 for pair in result.pairs)


def test_each_pair_equals_scikit_learn_and_the_oracle() -> None:
    rng = np.random.default_rng(SEED + 1)
    base = rng.integers(0, 4, 300)
    labelings = [np.where(rng.random(300) < 0.3, rng.integers(0, 4, 300), base) for _ in range(5)]

    result = by_label(labelings)

    assert [(pair.first, pair.second) for pair in result.pairs] == list(combinations(range(5), 2))
    for pair in result.pairs:
        first, second = labelings[pair.first], labelings[pair.second]
        assert pair.points == 300
        assert pair.ari == pytest.approx(adjusted_rand_score(first, second), rel=1e-12)
        assert pair.ari == pytest.approx(oracle_ari(first, second), rel=1e-12)


def test_the_mean_and_the_minimum_summarize_the_pairs() -> None:
    rng = np.random.default_rng(SEED + 2)
    base = rng.integers(0, 4, 200)
    labelings = [
        np.where(rng.random(200) < share, rng.integers(0, 4, 200), base)
        for share in (0.0, 0.1, 0.5, 0.9)
    ]

    result = by_label(labelings)

    values = [pair.ari for pair in result.pairs]
    assert len(set(values)) == 6
    assert result.mean == pytest.approx(np.mean(values), rel=1e-15)
    assert result.minimum == min(values)
    assert result.minimum < result.mean
    assert result.undefined_reason is None


def test_the_matrix_holds_each_pair_on_both_sides_and_ones_on_its_diagonal() -> None:
    rng = np.random.default_rng(SEED + 3)
    labelings = [rng.integers(0, 3, 50) for _ in range(3)]

    result = by_label(labelings)

    assert result.runs == 3
    assert len(result.matrix) == 3
    for pair in result.pairs:
        assert result.matrix[pair.first][pair.second] == pair.ari
        assert result.matrix[pair.second][pair.first] == pair.ari
    assert [result.matrix[run][run] for run in range(3)] == [1.0, 1.0, 1.0]


# --- Noise handling ----------------------------------------------------------------------


FIRST = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1, 2])
SECOND = np.array([0, 0, -1, 1, 1, 2, -1, 0, 2, 2])


def test_noise_as_its_own_label_compares_every_point() -> None:
    result = by_label([FIRST, SECOND], "own_label")

    (pair,) = result.pairs
    assert pair.points == 10
    assert pair.ari == pytest.approx(adjusted_rand_score(FIRST, SECOND), rel=1e-12)
    assert pair.ari == pytest.approx(oracle_ari(FIRST, SECOND), rel=1e-12)


def test_excluding_noise_drops_the_points_that_are_noise_in_either_run() -> None:
    result = by_label([FIRST, SECOND], "exclude_either")

    (pair,) = result.pairs
    kept = (FIRST != -1) & (SECOND != -1)
    assert pair.points == int(kept.sum()) == 6
    assert pair.ari == pytest.approx(adjusted_rand_score(FIRST[kept], SECOND[kept]), rel=1e-12)
    assert pair.ari == pytest.approx(oracle_ari(FIRST[kept], SECOND[kept]), rel=1e-12)
    # The two options differ on this pair.
    assert pair.ari != pytest.approx(by_label([FIRST, SECOND], "own_label").mean)


def test_the_result_records_the_noise_handling() -> None:
    assert by_label([FIRST, SECOND], "exclude_either").noise_handling == "exclude_either"
    assert by_label([FIRST, SECOND], "own_label").noise_handling == "own_label"


# --- Undefined: None with a reason -------------------------------------------------------


@pytest.mark.parametrize("labelings", [[], [[0, 1, 1]]])
def test_fewer_than_two_runs_have_no_stability(labelings) -> None:
    result = by_label(labelings)

    assert result.runs == len(labelings)
    assert result.pairs == ()
    assert (result.mean, result.minimum) == (None, None)
    assert result.undefined_reason == "fewer_than_two_runs"


def test_a_pair_left_without_points_is_undefined_and_so_is_the_summary() -> None:
    labelings = [[-1, -1, 0], [0, 1, -1], [0, 0, 1]]

    result = by_label(labelings, "exclude_either")

    first, second, third = result.pairs
    assert (first.points, first.ari, first.undefined_reason) == (0, None, "no_points")
    assert second.ari is not None
    assert third.ari is not None
    assert result.matrix[0][1] is None
    assert result.matrix[1][0] is None
    assert (result.mean, result.minimum) == (None, None)
    assert result.undefined_reason == "undefined_pairs"


def test_an_all_noise_run_has_no_diagonal_entry_when_noise_is_excluded() -> None:
    result = by_label([[-1, -1], [0, 1]], "exclude_either")

    assert result.matrix[0][0] is None
    assert result.matrix[1][1] == 1.0


def test_empty_labelings_have_no_points() -> None:
    result = by_label([[], []], "own_label")

    assert result.pairs[0].undefined_reason == "no_points"
    assert result.undefined_reason == "undefined_pairs"


# --- Arguments and determinism -----------------------------------------------------------


def test_the_noise_handling_has_no_default() -> None:
    parameter = inspect.signature(seed_stability).parameters["noise_handling"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("labelings", "handling", "message"),
    [
        ([[0, 1], [0, 1, 1]], "own_label", "run 1 has 3 labels but run 0 has 2"),
        ([[0, 1], [0, -3]], "own_label", "-1 or above"),
        ([[0, 1], [0.5, 1.0]], "own_label", "integers"),
        ([[0, 1], [0, 1]], "drop", "noise_handling"),
    ],
)
def test_invalid_arguments_raise(labelings, handling: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        seed_stability(labelings, noise_handling=handling)


def test_the_same_input_gives_the_same_result() -> None:
    rng = np.random.default_rng(SEED + 4)
    labelings = [rng.integers(-1, 4, 80) for _ in range(4)]

    first = by_label(labelings, "exclude_either")
    second = by_label([list(labels) for labels in labelings], "exclude_either")

    assert first == second
