"""Content validity of the questionnaire items: Aiken's V and its score interval (T30; plan
§3.8.2, PDF p. 82, printed p. 73).

Oracles: hand computations of ``V = S / (n (c - 1))`` (Aiken, 1985) in exact fractions. The
score interval of Penfield and Giacobbi (2004) is Wilson's (1927) score interval for ``S``
successes in ``n (c - 1)`` trials, so it is checked against scipy's
``BinomTestResult.proportion_ci(method="wilson")`` and against hand values for one item
(``S = 14``, ``N = 15``: 0.7018 to 0.9881 at 95%).
"""

import inspect
from fractions import Fraction

import pytest
from scipy.stats import binomtest

from thematic_redundancy.evaluation.aiken import (
    AikenDimension,
    aiken_score_interval,
    aiken_v,
    aiken_validity,
)


def validity(
    ratings, dimensions=None, *, low=1, high=4, confidence=0.95, criterion="v", cutoff=0.7
):
    return aiken_validity(
        ratings,
        dimensions or {"D1": list(ratings)},
        low=low,
        high=high,
        confidence=confidence,
        criterion=criterion,
        cutoff=cutoff,
    )


# --- V -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratings", "low", "high", "expected"),
    [
        ([4, 4, 3, 4, 4], 1, 4, Fraction(14, 15)),  # S = 3 + 3 + 2 + 3 + 3; c - 1 = 3
        ([3, 2, 3], 0, 3, Fraction(8, 9)),
        ([5, 4, None, 3], 1, 5, Fraction(9, 12)),  # the missing rating is left out: n = 3
        ([2, 2, 1, 2], 1, 2, Fraction(3, 4)),
        ([4, 4, 4], 1, 4, Fraction(1)),
        ([1, 1, 1], 1, 4, Fraction(0)),
    ],
)
def test_v_is_the_distance_sum_over_n_times_c_minus_one(ratings, low, high, expected) -> None:
    assert aiken_v(ratings, low=low, high=high) == pytest.approx(float(expected), abs=1e-15)


def test_v_of_an_item_with_no_rating_is_undefined() -> None:
    assert aiken_v([None, None], low=1, high=4) is None
    assert aiken_v([], low=1, high=4) is None


# --- Score interval ----------------------------------------------------------------------


def test_the_score_interval_matches_the_hand_values() -> None:
    lower, upper = aiken_score_interval(14 / 15, judges=5, categories=4, confidence=0.95)

    assert round(lower, 4) == 0.7018
    assert round(upper, 4) == 0.9881


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
@pytest.mark.parametrize(("judges", "categories"), [(3, 4), (5, 5), (8, 4), (7, 2)])
def test_the_score_interval_is_wilsons_over_n_times_c_minus_one_trials(
    judges: int, categories: int, confidence: float
) -> None:
    trials = judges * (categories - 1)
    for distance_sum in range(trials + 1):
        lower, upper = aiken_score_interval(
            distance_sum / trials, judges=judges, categories=categories, confidence=confidence
        )

        wilson = binomtest(distance_sum, trials).proportion_ci(
            confidence_level=confidence, method="wilson"
        )
        assert lower == pytest.approx(wilson.low, abs=1e-12)
        assert upper == pytest.approx(wilson.high, abs=1e-12)
        assert 0.0 <= lower <= distance_sum / trials <= upper <= 1.0


def test_a_higher_confidence_widens_the_interval() -> None:
    narrow = aiken_score_interval(0.8, judges=6, categories=5, confidence=0.90)
    wide = aiken_score_interval(0.8, judges=6, categories=5, confidence=0.95)

    assert wide[0] < narrow[0] < 0.8 < narrow[1] < wide[1]


def test_unanimous_top_ratings_reach_one_and_their_lower_bound_grows_with_the_judges() -> None:
    # With V = 1 the lower bound is N / (N + z^2): N = 6 for 2 judges on 1-4, N = 9 for 3.
    two = aiken_score_interval(1.0, judges=2, categories=4, confidence=0.95)
    three = aiken_score_interval(1.0, judges=3, categories=4, confidence=0.95)

    assert two[1] == three[1] == 1.0
    assert round(two[0], 4) == 0.6097
    assert round(three[0], 4) == 0.7009


# --- Items and dimensions ----------------------------------------------------------------


def test_each_item_reports_its_ratings_v_interval_and_acceptance() -> None:
    result = validity(
        {"q1": [4, 4, 3, 4, 4], "q2": [3, None, 2, 2, 3], "q3": [2, 1, 2, 1, 1]},
        {"D1": ["q1", "q2"], "D2": ["q3"]},
    )

    assert (result.low, result.high, result.judges) == (1, 4, 5)
    assert result.z == pytest.approx(1.959963984540054, abs=1e-12)
    first, second, third = result.items
    assert (first.item, first.dimension, first.judges, first.missing) == ("q1", "D1", 5, 0)
    assert first.distance_sum == 14
    assert first.v == pytest.approx(14 / 15, abs=1e-15)
    assert (round(first.ci_lower, 4), round(first.ci_upper, 4)) == (0.7018, 0.9881)
    assert first.accepted is True
    assert (second.judges, second.missing, second.distance_sum) == (4, 1, 6)
    assert second.v == pytest.approx(0.5, abs=1e-15)
    assert second.accepted is False
    assert (third.dimension, third.distance_sum, third.accepted) == ("D2", 2, False)
    assert third.undefined_reason is None


def test_v_equal_to_the_cutoff_is_accepted() -> None:
    # 5 judges on 1-3: S = 2 + 2 + 1 + 1 + 1 = 7, V = 7 / 10.
    result = validity({"q1": [3, 3, 2, 2, 2]}, low=1, high=3, cutoff=0.7)

    assert result.items[0].v == 0.7
    assert result.items[0].accepted is True


def test_the_lower_bound_criterion_needs_enough_judges() -> None:
    ratings = {"two": [4, 4, None], "three": [4, 4, 4]}

    by_v = validity(ratings, criterion="v", cutoff=0.7)
    by_lower = validity(ratings, criterion="ci_lower", cutoff=0.7)

    assert [item.accepted for item in by_v.items] == [True, True]
    assert [item.accepted for item in by_lower.items] == [False, True]
    assert by_lower.dimensions[0].rejected_items == ("two",)


def test_a_dimension_summarizes_its_items() -> None:
    result = validity(
        {"q1": [4, 4, 4], "q2": [4, 3, 3], "q3": [2, 2, 1], "q4": [4, 4, 3]},
        {"D1": ["q1", "q2", "q3"], "D2": ["q4"]},
    )

    first, second = result.dimensions
    assert first == AikenDimension(
        dimension="D1",
        items=("q1", "q2", "q3"),
        mean_v=pytest.approx((1 + 7 / 9 + 2 / 9) / 3, abs=1e-15),
        minimum_v=pytest.approx(2 / 9, abs=1e-15),
        accepted_items=("q1", "q2"),
        rejected_items=("q3",),
        undefined_items=(),
        undefined_reason=None,
    )
    assert second.items == ("q4",)
    assert second.mean_v == pytest.approx(8 / 9, abs=1e-15)


def test_an_item_with_no_rating_is_undefined_and_so_is_its_dimension_mean() -> None:
    result = validity({"q1": [4, 4], "q2": [None, None]})

    item = result.items[1]
    assert (item.judges, item.missing, item.distance_sum) == (0, 2, 0)
    assert item.v is item.ci_lower is item.ci_upper is item.accepted is None
    assert item.undefined_reason == "no_ratings"
    dimension = result.dimensions[0]
    assert dimension.mean_v is dimension.minimum_v is None
    assert dimension.undefined_items == ("q2",)
    assert dimension.accepted_items == ("q1",)
    assert dimension.undefined_reason == "undefined_items"


def test_items_follow_the_order_of_the_dimensions() -> None:
    result = validity({"a": [4], "b": [4], "c": [4]}, {"D2": ["c"], "D1": ["b", "a"]})

    assert [item.item for item in result.items] == ["c", "b", "a"]
    assert [dimension.dimension for dimension in result.dimensions] == ["D2", "D1"]


# --- Arguments ---------------------------------------------------------------------------


def test_the_methodological_parameters_are_required_keywords() -> None:
    parameters = inspect.signature(aiken_validity).parameters

    for name in ("low", "high", "confidence", "criterion", "cutoff"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("ratings", "dimensions", "changes", "message"),
    [
        ({"q1": [4, 5]}, None, {}, "outside the scale 1 to 4"),
        ({"q1": [4, 0]}, None, {}, "outside the scale 1 to 4"),
        ({"q1": [4, 2.5]}, None, {}, "not an integer or None"),
        ({"q1": [4, True]}, None, {}, "not an integer or None"),
        ({"q1": [4], "q2": [4, 3]}, None, {}, "'q2' has 2 answers but item 'q1' has 1"),
        ({}, {"D1": []}, {}, "at least one item"),
        ({"q1": [4], "q2": [4]}, {"D1": ["q1"]}, {}, "'q2' is in no dimension"),
        ({"q1": [4]}, {"D1": ["q1"], "D2": ["q1"]}, {}, "'q1' is in dimensions 'D1' and 'D2'"),
        ({"q1": [4]}, {"D1": ["q1", "q9"]}, {}, "'q9', which has no answers"),
        ({"q1": [4]}, {"D1": ["q1"], "D2": []}, {}, "'D2' has no item"),
        ({"q1": [4]}, {"D1": "q1"}, {}, "a sequence of item ids"),
        ({"q1": [4]}, None, {"low": 4}, "low < high"),
        ({"q1": [4]}, None, {"confidence": 1.0}, "confidence"),
        ({"q1": [4]}, None, {"confidence": 0.0}, "confidence"),
        ({"q1": [4]}, None, {"criterion": "lower"}, "criterion"),
        ({"q1": [4]}, None, {"cutoff": 1.2}, "cutoff"),
        ({"q1": [4]}, None, {"cutoff": -0.1}, "cutoff"),
    ],
)
def test_invalid_validity_arguments_raise(ratings, dimensions, changes, message) -> None:
    arguments = {"low": 1, "high": 4, "confidence": 0.95, "criterion": "v", "cutoff": 0.7}
    with pytest.raises(ValueError, match=message):
        aiken_validity(ratings, dimensions or {"D1": list(ratings)}, **(arguments | changes))


@pytest.mark.parametrize(
    ("v", "judges", "categories", "confidence", "message"),
    [
        (1.1, 3, 4, 0.95, "v must be"),
        (0.5, 0, 4, 0.95, "judges"),
        (0.5, 3, 1, 0.95, "categories"),
        (0.5, 3, 4, 1.5, "confidence"),
    ],
)
def test_invalid_interval_arguments_raise(v, judges, categories, confidence, message) -> None:
    with pytest.raises(ValueError, match=message):
        aiken_score_interval(v, judges=judges, categories=categories, confidence=confidence)
