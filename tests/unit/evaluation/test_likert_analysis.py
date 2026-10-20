"""Analysis of the expert questionnaire (T32; plan §3.8.2, PDF p. 82, printed p. 73): medians,
quartiles and frequencies per item and per dimension, and Cronbach's alpha per dimension.

Oracles: hand computations in exact fractions; Cronbach's formula with the sample variance and
its covariance-matrix form, both in plain numpy; pingouin 0.7.0's ``cronbach_alpha`` with
listwise deletion; and the two worked examples of its docstring, which pingouin says were
tested against the R package psych: 0.8016949152542373 for the ``cronbach_wide_missing``
dataset after listwise deletion, and 0.5917188485995826 for the long-format ``cronbach_alpha``
dataset.
"""

import inspect
from fractions import Fraction

import numpy as np
import pandas as pd
import pingouin
import pytest

from thematic_redundancy.evaluation.likert import (
    CronbachAlpha,
    Quartiles,
    cronbach_alpha,
    likert_analysis,
)

SEED = 20261019
RESPONSES = {
    "i1": [5, 4, 4, 3, None, 5],
    "i2": [4, 4, 5, 3, 4, 5],
    "i3": [2, 1, 2, 3, 2, 1],
    "i4": [5, 5, 4, 4, 4, None],
}
DIMENSIONS = {"D1": ["i1", "i2"], "D2": ["i3", "i4"]}


def analyse(responses=RESPONSES, dimensions=DIMENSIONS, *, reverse=(), method="linear"):
    return likert_analysis(
        responses, dimensions, low=1, high=5, reverse_coded=reverse, quartile_method=method
    )


def plain_alpha(matrix: np.ndarray) -> float:
    """Cronbach's alpha from the sample variances of the items and of the total."""
    items = matrix.shape[1]
    ratio = matrix.var(axis=0, ddof=1).sum() / matrix.sum(axis=1).var(ddof=1)
    return items / (items - 1) * (1 - ratio)


def covariance_alpha(matrix: np.ndarray) -> float:
    """Cronbach's alpha from the item covariance matrix: k / (k - 1) * (1 - trace / sum)."""
    covariance = np.cov(matrix, rowvar=False)
    items = matrix.shape[1]
    return items / (items - 1) * (1 - np.trace(covariance) / covariance.sum())


def fraction_alpha(rows: list[list[int]]) -> Fraction:
    def variance(values: list[int]) -> Fraction:
        mean = Fraction(sum(values), len(values))
        return sum((value - mean) ** 2 for value in values) / (len(values) - 1)

    columns = [list(column) for column in zip(*rows, strict=True)]
    totals = [sum(row) for row in rows]
    items = len(columns)
    return Fraction(items, items - 1) * (1 - sum(map(variance, columns)) / variance(totals))


# --- Items -------------------------------------------------------------------------------


def test_an_item_reports_frequencies_missing_answers_median_and_quartiles() -> None:
    item = analyse().items[0]

    assert (item.item, item.dimension, item.reverse_coded) == ("i1", "D1", False)
    assert item.frequencies == ((1, 0), (2, 0), (3, 1), (4, 2), (5, 2))
    assert item.missing == 1
    # Sorted answers 3, 4, 4, 5, 5; numpy's linear quartiles are at positions 1 and 3.
    assert item.answers == Quartiles(5, 4.0, 4.0, 5.0, 1.0, None)


def test_a_missing_answer_is_left_out_and_never_counted_as_a_low_answer() -> None:
    result = analyse({"a": [4, None, None, 5], "b": [4, 4, 5, 5]}, {"D": ["a", "b"]})

    item = result.items[0]
    assert (item.answers.count, item.missing, item.answers.median) == (2, 2, 4.5)
    assert sum(count for _, count in item.frequencies) == 2
    assert result.dimensions[0].alpha.excluded_respondents == 2


def test_an_item_with_no_answer_has_no_median() -> None:
    item = analyse({"a": [None, None], "b": [4, 5]}, {"D": ["a", "b"]}).items[0]

    assert item.answers == Quartiles(0, None, None, None, None, "no_values")
    assert item.frequencies == ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0))


@pytest.mark.parametrize("method", ["linear", "inverted_cdf", "hazen", "median_unbiased"])
def test_the_quartiles_follow_the_given_numpy_method(method: str) -> None:
    item = analyse(method=method).items[1]
    answers = np.array(RESPONSES["i2"], dtype=float)

    assert item.answers.first_quartile == np.quantile(answers, 0.25, method=method)
    assert item.answers.third_quartile == np.quantile(answers, 0.75, method=method)
    assert item.answers.median == np.median(answers)


def test_a_reverse_coded_item_is_recoded_before_every_statistic() -> None:
    plain, recoded = analyse().items[2], analyse(reverse={"i3"}).items[2]

    assert plain.reverse_coded is False and recoded.reverse_coded is True
    assert plain.frequencies == ((1, 2), (2, 3), (3, 1), (4, 0), (5, 0))
    assert recoded.frequencies == ((1, 0), (2, 0), (3, 1), (4, 3), (5, 2))
    assert (plain.answers.median, recoded.answers.median) == (2.0, 4.0)


# --- Dimensions --------------------------------------------------------------------------


def test_a_dimension_pools_its_answers_and_scores_its_complete_respondents() -> None:
    dimension = analyse().dimensions[0]

    assert (dimension.dimension, dimension.items, dimension.missing) == ("D1", ("i1", "i2"), 1)
    assert dimension.frequencies == ((1, 0), (2, 0), (3, 2), (4, 5), (5, 4))
    assert dimension.answers.count == 11
    assert dimension.answers.median == 4.0
    # Respondent 4 missed i1; the others' means are 4.5, 4, 4.5, 3 and 5.
    assert dimension.respondent_scores.count == 5
    assert dimension.respondent_scores.median == 4.5


def test_a_dimension_alpha_matches_the_hand_value() -> None:
    alpha = analyse().dimensions[0].alpha

    # Item variances 0.7 and 0.7, total variance 2.3: alpha = 2 * (1 - 1.4 / 2.3) = 18 / 23.
    assert alpha == CronbachAlpha(2, 5, 1, pytest.approx(18 / 23, abs=1e-12), None)


def test_reverse_coding_changes_the_dimension_alpha() -> None:
    plain, recoded = analyse().dimensions[1].alpha, analyse(reverse={"i3"}).dimensions[1].alpha
    complete = np.array([[2, 5], [1, 5], [2, 4], [3, 4], [2, 4]], dtype=float)

    assert plain.alpha == pytest.approx(plain_alpha(complete), abs=1e-12)
    complete[:, 0] = 6 - complete[:, 0]
    assert recoded.alpha == pytest.approx(plain_alpha(complete), abs=1e-12)
    assert recoded.alpha != pytest.approx(plain.alpha)


def test_alpha_if_item_deleted_uses_the_remaining_items_and_the_same_respondents() -> None:
    rng = np.random.default_rng(SEED)
    base = rng.integers(2, 5, 12)
    answers = {
        f"q{item}": [int(np.clip(value + rng.integers(-1, 2), 1, 5)) for value in base]
        for item in range(4)
    }
    answers["q2"][3] = None

    dimension = analyse(answers, {"D": list(answers)}).dimensions[0]

    matrix = np.array([answers[f"q{item}"] for item in range(4)], dtype=float).T
    complete = matrix[~np.isnan(matrix).any(axis=1)]
    assert (dimension.alpha.respondents, dimension.alpha.excluded_respondents) == (11, 1)
    assert [row.item for row in dimension.alpha_if_item_deleted] == ["q0", "q1", "q2", "q3"]
    for column, row in enumerate(dimension.alpha_if_item_deleted):
        kept = np.delete(complete, column, axis=1)
        assert row.alpha == pytest.approx(plain_alpha(kept), abs=1e-12)
        assert row.undefined_reason is None


def test_deleting_one_of_two_items_leaves_no_alpha() -> None:
    rows = analyse().dimensions[0].alpha_if_item_deleted

    assert [(row.item, row.alpha, row.undefined_reason) for row in rows] == [
        ("i1", None, "fewer_than_two_items"),
        ("i2", None, "fewer_than_two_items"),
    ]


def test_items_and_dimensions_follow_the_order_of_the_dimensions() -> None:
    result = analyse(dimensions={"D2": ["i4", "i3"], "D1": ["i2", "i1"]})

    assert [item.item for item in result.items] == ["i4", "i3", "i2", "i1"]
    assert [dimension.dimension for dimension in result.dimensions] == ["D2", "D1"]
    assert (result.respondents, result.reverse_coded) == (6, ())


def test_the_analysis_is_deterministic() -> None:
    assert analyse(reverse={"i3"}) == analyse(reverse={"i3"})


# --- Cronbach's alpha --------------------------------------------------------------------


def test_alpha_matches_a_hand_computation_in_fractions() -> None:
    rows = [[1, 2, 2], [2, 3, 3], [3, 3, 4], [4, 5, 5]]

    result = cronbach_alpha(rows)

    assert fraction_alpha(rows) == Fraction(56, 57)
    assert result == CronbachAlpha(3, 4, 0, pytest.approx(56 / 57, abs=1e-12), None)


def test_alpha_matches_numpy_formulas_and_pingouin_on_random_answers() -> None:
    rng = np.random.default_rng(SEED + 1)
    for _ in range(20):
        respondents, items = rng.integers(3, 12), rng.integers(2, 7)
        trait = rng.integers(1, 6, respondents)[:, None]
        matrix = np.clip(trait + rng.integers(-2, 3, (respondents, items)), 1, 5).astype(float)
        if matrix.sum(axis=1).var() == 0:
            continue

        result = cronbach_alpha(matrix)

        assert result.alpha == pytest.approx(plain_alpha(matrix), abs=1e-12)
        assert result.alpha == pytest.approx(covariance_alpha(matrix), abs=1e-12)
        oracle, _ = pingouin.cronbach_alpha(data=pd.DataFrame(matrix), nan_policy="listwise")
        assert result.alpha == pytest.approx(oracle, abs=1e-12)


def test_alpha_matches_the_pingouin_docstring_examples() -> None:
    wide = pingouin.read_dataset("cronbach_wide_missing")
    long = pingouin.read_dataset("cronbach_alpha")

    listwise = cronbach_alpha(wide.to_numpy(dtype=float))
    likert = cronbach_alpha(
        long.pivot(index="Subj", columns="Items", values="Scores").to_numpy(dtype=float)
    )

    assert listwise.alpha == pytest.approx(0.8016949152542373, abs=1e-12)
    assert listwise.excluded_respondents == int(wide.isna().any(axis=1).sum()) > 0
    assert listwise.respondents + listwise.excluded_respondents == len(wide)
    assert likert.alpha == pytest.approx(0.5917188485995826, abs=1e-12)


@pytest.mark.parametrize(
    ("scores", "items", "respondents", "excluded", "reason"),
    [
        ([[1], [2], [3]], 1, 3, 0, "fewer_than_two_items"),
        ([[1, 2], [np.nan, 3]], 2, 1, 1, "fewer_than_two_respondents"),
        (np.empty((0, 3)), 3, 0, 0, "fewer_than_two_respondents"),
        ([[3, 3], [3, 3], [3, 3]], 2, 3, 0, "zero_total_variance"),
        ([[1, 2], [2, 1], [3, 0]], 2, 3, 0, "zero_total_variance"),
    ],
)
def test_undefined_alphas_give_their_reason(scores, items, respondents, excluded, reason) -> None:
    assert cronbach_alpha(scores) == CronbachAlpha(items, respondents, excluded, None, reason)


def test_alpha_can_be_negative() -> None:
    result = cronbach_alpha([[1, 5], [5, 2], [2, 4], [4, 1]])

    assert result.alpha is not None and result.alpha < 0
    assert result.alpha == pytest.approx(plain_alpha(np.array([[1, 5], [5, 2], [2, 4], [4, 1]])))


# --- Arguments ---------------------------------------------------------------------------


def test_the_methodological_parameters_are_required_keywords() -> None:
    parameters = inspect.signature(likert_analysis).parameters

    for name in ("low", "high", "reverse_coded", "quartile_method"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("responses", "dimensions", "changes", "message"),
    [
        ({"a": [6], "b": [5]}, None, {}, "outside the scale 1 to 5"),
        ({"a": [4.0], "b": [5]}, None, {}, "not an integer or None"),
        ({"a": [False], "b": [5]}, None, {}, "not an integer or None"),
        ({"a": [4], "b": [5, 5]}, None, {}, "'b' has 2 answers but item 'a' has 1"),
        ({"": [4]}, {"D": [""]}, {}, "non-empty string"),
        ({"a": [4], "b": [5]}, {"D": ["a"]}, {}, "'b' is in no dimension"),
        ({"a": [4], "b": [5]}, {"D": ["a", "b", "a"]}, {}, "'a' is in dimensions 'D' and 'D'"),
        ({"a": [4], "b": [5]}, {"D": ["a", "b"], "E": []}, {}, "'E' has no item"),
        ({"a": [4], "b": [5]}, None, {"reverse_coded": {"z"}}, "'z' has no answers"),
        ({"a": [4], "b": [5]}, None, {"reverse_coded": "a"}, "not a string"),
        ({"a": [4], "b": [5]}, None, {"quartile_method": "midpoint"}, "quartile_method"),
        ({"a": [4], "b": [5]}, None, {"low": 5}, "low < high"),
        ({"a": [4], "b": [5]}, None, {"low": 1.0}, "low < high"),
    ],
)
def test_invalid_analysis_arguments_raise(responses, dimensions, changes, message) -> None:
    arguments = {"low": 1, "high": 5, "reverse_coded": (), "quartile_method": "linear"}
    with pytest.raises(ValueError, match=message):
        likert_analysis(responses, dimensions or {"D": list(responses)}, **(arguments | changes))


@pytest.mark.parametrize(
    ("scores", "message"),
    [([1, 2, 3], "2-D"), ([[1, 2], [np.inf, 3]], "finite")],
)
def test_invalid_alpha_scores_raise(scores, message) -> None:
    with pytest.raises(ValueError, match=message):
        cronbach_alpha(scores)
