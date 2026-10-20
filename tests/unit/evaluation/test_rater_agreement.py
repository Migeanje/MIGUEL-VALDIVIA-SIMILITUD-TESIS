"""Agreement between annotators: Cohen's kappa, weighted kappa, Fleiss' kappa and the consensus
(T21, D15, D31).

Oracles: hand computations in exact fractions; a published worked example, the SAS PROC FREQ
printout for a 3 × 3 table that statsmodels 0.15.0 reproduces in
``statsmodels/stats/tests/test_inter_rater.py`` (``table10``: simple kappa 0.4842, weighted
kappa 0.4701); the example of scikit-learn 1.9.1's ``cohen_kappa_score`` docstring (0.6875); and
``cohen_kappa_score`` itself, with ``weights`` None, linear and quadratic. For Fleiss' kappa: the
two examples of the same statsmodels test file, the Wikipedia worked example (``table1``, kappa
0.210) and the diagnoses of Fleiss (1971) with the value of R's ``irr::kappam.fleiss``
(0.4302445); statsmodels' ``fleiss_kappa`` itself; and Scott's pi by hand for two ratings per item.
"""

import inspect
from fractions import Fraction

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa as statsmodels_fleiss_kappa

from thematic_redundancy.evaluation.agreement import (
    AgreedLabel,
    Disagreement,
    cohen_kappa,
    consensus_labels,
    fleiss_kappa,
)

SEED = 20261019
LEVELS = ("low", "mid", "high")
SAS_TABLE = ((0, 4, 1), (0, 8, 0), (0, 1, 5))
"""Margins: rows (5, 8, 6), columns (0, 13, 6); 13 of the 19 items on the diagonal."""
WIKIPEDIA_COUNTS = (
    (0, 0, 0, 0, 14),
    (0, 2, 6, 4, 2),
    (0, 0, 3, 5, 6),
    (0, 3, 9, 2, 0),
    (2, 2, 8, 1, 1),
    (7, 7, 0, 0, 0),
    (3, 2, 6, 3, 0),
    (2, 5, 3, 2, 2),
    (6, 5, 2, 1, 0),
    (0, 2, 2, 3, 7),
)
"""Ten items, 14 ratings each, over five categories: the Wikipedia worked example of Fleiss'
kappa, as statsmodels 0.15.0's ``test_inter_rater.py`` reproduces it (``table1``, 0.210)."""
FLEISS_1971 = (
    (4, 4, 4, 4, 4, 4),
    (2, 2, 2, 5, 5, 5),
    (2, 3, 3, 3, 3, 5),
    (5, 5, 5, 5, 5, 5),
    (2, 2, 2, 4, 4, 4),
    (1, 1, 3, 3, 3, 3),
    (3, 3, 3, 3, 5, 5),
    (1, 1, 3, 3, 3, 4),
    (1, 1, 4, 4, 4, 4),
    (5, 5, 5, 5, 5, 5),
    (1, 4, 4, 4, 4, 4),
    (1, 2, 4, 4, 4, 4),
    (2, 2, 2, 3, 3, 3),
    (1, 4, 4, 4, 4, 4),
    (2, 2, 4, 4, 4, 5),
    (3, 3, 3, 3, 3, 5),
    (1, 1, 1, 4, 5, 5),
    (1, 1, 1, 1, 1, 2),
    (2, 2, 4, 4, 4, 4),
    (1, 3, 3, 5, 5, 5),
    (5, 5, 5, 5, 5, 5),
    (2, 4, 4, 4, 4, 4),
    (2, 2, 4, 5, 5, 5),
    (1, 1, 4, 4, 4, 4),
    (1, 4, 4, 4, 4, 5),
    (2, 2, 2, 2, 2, 4),
    (1, 1, 1, 1, 5, 5),
    (2, 2, 4, 4, 4, 4),
    (1, 3, 3, 3, 3, 3),
    (5, 5, 5, 5, 5, 5),
)
"""Thirty patients, six psychiatrists' diagnoses each, over five categories (Fleiss, 1971), as
statsmodels 0.15.0's ``test_inter_rater.py`` holds them; R's ``irr::kappam.fleiss`` gives
0.4302445 (``test_fleiss_kappa_irr``)."""


def from_table(table, categories=LEVELS) -> tuple[list[str], list[str]]:
    """The two annotators' labels behind a contingency table, row = first annotator."""
    first, second = [], []
    for row, counts in enumerate(table):
        for column, count in enumerate(counts):
            first += [categories[row]] * count
            second += [categories[column]] * count
    return first, second


def kappa(first, second, weights, categories=LEVELS):
    return cohen_kappa(first, second, categories=categories, weights=weights)


def exactly(fraction: Fraction):
    return pytest.approx(float(fraction), abs=1e-15)


# --- Worked examples ---------------------------------------------------------------------


def test_the_sas_table_gives_the_published_simple_kappa() -> None:
    result = kappa(*from_table(SAS_TABLE), "none")

    assert result.items == 19
    assert result.confusion == SAS_TABLE
    assert result.observed_agreement == exactly(Fraction(13, 19))
    # (5 * 0 + 8 * 13 + 6 * 6) / 19^2: both annotators' margins, not one annotator's twice.
    assert result.expected_agreement == exactly(Fraction(140, 361))
    assert result.kappa == exactly(Fraction(107, 221))
    assert round(result.kappa, 4) == 0.4842
    assert result.undefined_reason is None


def test_the_sas_table_gives_the_published_linear_weighted_kappa() -> None:
    result = kappa(*from_table(SAS_TABLE), "linear")

    # Disagreement weights |i - j| / 2: observed 3.5 / 19, expected 125.5 / 361.
    assert result.observed_agreement == exactly(Fraction(31, 38))
    assert result.expected_agreement == exactly(Fraction(471, 722))
    assert result.kappa == exactly(Fraction(118, 251))
    assert round(result.kappa, 4) == 0.4701


def test_the_sas_table_gives_the_hand_computed_quadratic_weighted_kappa() -> None:
    result = kappa(*from_table(SAS_TABLE), "quadratic")

    # Disagreement weights ((i - j) / 2)^2: observed 2.25 / 19, expected 77.75 / 361.
    assert result.observed_agreement == exactly(Fraction(67, 76))
    assert result.expected_agreement == exactly(Fraction(1133, 1444))
    assert result.kappa == exactly(Fraction(140, 311))


def test_the_scikit_learn_docstring_example_gives_eleven_sixteenths() -> None:
    first = ["negative", "positive", "negative", "neutral", "positive"]
    second = ["negative", "positive", "negative", "neutral", "negative"]

    result = kappa(first, second, "none", ("negative", "neutral", "positive"))

    assert result.observed_agreement == exactly(Fraction(4, 5))
    assert result.expected_agreement == exactly(Fraction(9, 25))
    assert result.kappa == 0.6875


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_kappa_equals_scikit_learn_on_correlated_random_labels(weights: str) -> None:
    rng = np.random.default_rng(SEED)
    first = rng.choice(LEVELS, 300)
    second = np.where(rng.random(300) < 0.6, first, rng.choice(LEVELS, 300))

    result = kappa(list(first), list(second), weights)

    expected = cohen_kappa_score(
        first, second, labels=list(LEVELS), weights=None if weights == "none" else weights
    )
    assert result.kappa == pytest.approx(expected, rel=1e-12)
    assert sum(map(sum, result.confusion)) == 300


# --- Categories and symmetry -------------------------------------------------------------


def test_only_the_weighted_kappas_depend_on_the_order_of_the_categories() -> None:
    first, second = from_table(SAS_TABLE)
    reordered = ("mid", "low", "high")

    assert kappa(first, second, "none", reordered).kappa == exactly(Fraction(107, 221))
    for weights in ("linear", "quadratic"):
        result = kappa(first, second, weights, reordered).kappa

        assert result != pytest.approx(kappa(first, second, weights).kappa)
        assert result == pytest.approx(
            cohen_kappa_score(first, second, labels=list(reordered), weights=weights), rel=1e-12
        )


def test_an_unused_category_is_an_empty_row_and_column() -> None:
    result = kappa(*from_table(SAS_TABLE), "none", (*LEVELS, "other"))

    assert result.confusion == (*(row + (0,) for row in SAS_TABLE), (0, 0, 0, 0))
    assert result.kappa == exactly(Fraction(107, 221))


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_swapping_the_annotators_transposes_the_table_and_keeps_kappa(weights: str) -> None:
    first, second = from_table(SAS_TABLE)

    forward, backward = kappa(first, second, weights), kappa(second, first, weights)

    assert backward.confusion == tuple(zip(*SAS_TABLE, strict=True))
    assert backward.kappa == pytest.approx(forward.kappa, abs=1e-15)
    assert backward.expected_agreement == pytest.approx(forward.expected_agreement, abs=1e-15)


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_full_agreement_is_one_and_opposite_ends_are_minus_one(weights: str) -> None:
    first = ["low", "high"] * 5

    assert kappa(first, first, weights).kappa == 1.0
    assert kappa(first, ["high", "low"] * 5, weights).kappa == pytest.approx(-1.0, abs=1e-15)


def test_integer_categories_work_like_strings() -> None:
    first, second = from_table(SAS_TABLE, (0, 1, 2))

    assert kappa(first, second, "linear", (0, 1, 2)).kappa == exactly(Fraction(118, 251))


# --- Undefined cases ---------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_kappa_is_undefined_when_both_annotators_use_one_same_category(weights: str) -> None:
    result = kappa(["mid"] * 4, ["mid"] * 4, weights)

    assert result.kappa is None
    assert result.undefined_reason == "expected_agreement_is_one"
    assert result.observed_agreement == 1.0
    assert result.expected_agreement == 1.0


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_kappa_is_zero_when_each_annotator_uses_a_different_single_category(weights) -> None:
    result = kappa(["low"] * 4, ["high"] * 4, weights)

    assert result.kappa == 0.0
    assert result.observed_agreement == 0.0
    assert result.expected_agreement == 0.0
    assert result.undefined_reason is None


def test_no_items_give_no_kappa() -> None:
    result = kappa([], [], "none")

    assert result.items == 0
    assert result.confusion == ((0, 0, 0),) * 3
    assert result.observed_agreement is None
    assert result.expected_agreement is None
    assert result.kappa is None
    assert result.undefined_reason == "no_items"


# --- Consensus ---------------------------------------------------------------------------


def test_the_consensus_keeps_agreed_labels_and_lists_every_other_pair() -> None:
    result = consensus_labels(
        ["p1", "p2", "p3", "p4", "p5"],
        ["low", "mid", "high", "mid", "mid"],
        ["low", "high", "high", "low", "mid"],
        categories=LEVELS,
    )

    assert result.agreed == (
        AgreedLabel("p1", "low"),
        AgreedLabel("p3", "high"),
        AgreedLabel("p5", "mid"),
    )
    # Adjacent levels are not averaged or broken towards either annotator.
    assert result.needs_adjudication == (
        Disagreement("p2", "mid", "high"),
        Disagreement("p4", "mid", "low"),
    )


def test_every_pair_is_either_agreed_or_left_for_adjudication() -> None:
    rng = np.random.default_rng(SEED + 1)
    first, second = list(rng.choice(LEVELS, 120)), list(rng.choice(LEVELS, 120))
    pair_ids = [f"pair-{index:03d}" for index in range(120)]

    result = consensus_labels(pair_ids, first, second, categories=LEVELS)

    agreed = [label.pair_id for label in result.agreed]
    disputed = [pair.pair_id for pair in result.needs_adjudication]
    assert sorted(agreed + disputed) == pair_ids
    assert len(agreed) == sum(a == b for a, b in zip(first, second, strict=True))
    by_id = dict(zip(pair_ids, zip(first, second, strict=True), strict=True))
    assert all(
        by_id[pair.pair_id] == (pair.first, pair.second) for pair in result.needs_adjudication
    )


# --- Fleiss' kappa -----------------------------------------------------------------------


def from_counts(counts, categories) -> list[list]:
    """The ratings behind an ``items × categories`` table of counts."""
    return [
        [category for category, count in zip(categories, row, strict=True) for _ in range(count)]
        for row in counts
    ]


def fleiss_by_hand(counts) -> tuple[Fraction, Fraction, Fraction]:
    """Fleiss' (1971) observed agreement, chance agreement and kappa, in exact fractions."""
    raters, items = sum(counts[0]), len(counts)
    observed = sum(
        Fraction(sum(count * (count - 1) for count in row), raters * (raters - 1)) for row in counts
    ) / len(counts)
    shares = [Fraction(sum(column), items * raters) for column in zip(*counts, strict=True)]
    expected = sum(share * share for share in shares)
    return observed, expected, (observed - expected) / (1 - expected)


FIVE = ("c1", "c2", "c3", "c4", "c5")


def test_the_wikipedia_example_gives_the_published_fleiss_kappa() -> None:
    result = fleiss_kappa(from_counts(WIKIPEDIA_COUNTS, FIVE), categories=FIVE)

    observed, expected, value = fleiss_by_hand(WIKIPEDIA_COUNTS)
    assert result.items == 10
    assert result.raters_per_item == 14
    assert result.counts == WIKIPEDIA_COUNTS
    assert result.category_shares == tuple(
        exactly(Fraction(total, 140)) for total in (20, 28, 39, 21, 32)
    )
    assert result.observed_agreement == exactly(observed)
    assert result.expected_agreement == exactly(expected)
    assert result.kappa == exactly(value)
    assert round(result.kappa, 3) == 0.210
    assert result.undefined_reason is None


def test_the_fleiss_1971_diagnoses_give_the_value_of_r_irr() -> None:
    result = fleiss_kappa(FLEISS_1971, categories=(1, 2, 3, 4, 5))

    assert result.items == 30
    assert result.raters_per_item == 6
    assert result.kappa == pytest.approx(0.4302445, abs=5e-8)


@pytest.mark.parametrize("raters", [2, 3, 5])
def test_fleiss_kappa_equals_statsmodels_on_correlated_random_ratings(raters: int) -> None:
    rng = np.random.default_rng(SEED + raters)
    truth = rng.choice(LEVELS, 120)
    ratings = [
        [label if rng.random() < 0.6 else rng.choice(LEVELS) for _ in range(raters)]
        for label in truth
    ]

    result = fleiss_kappa(ratings, categories=LEVELS)

    table = np.array([[row.count(level) for level in LEVELS] for row in ratings])
    assert result.counts == tuple(map(tuple, table.tolist()))
    assert result.kappa == pytest.approx(
        statsmodels_fleiss_kappa(table, method="fleiss"), rel=1e-12
    )


def test_with_two_ratings_per_item_fleiss_kappa_is_scotts_pi() -> None:
    first, second = from_table(SAS_TABLE)

    result = fleiss_kappa(list(zip(first, second, strict=True)), categories=LEVELS)

    # Scott's pi pools both annotators' margins, (5 + 0, 8 + 13, 6 + 6) of 38 ratings, so it
    # differs from Cohen's kappa (107/221), which keeps each annotator's own margins.
    expected = Fraction(5**2 + 21**2 + 12**2, 38**2)
    pi = (Fraction(13, 19) - expected) / (1 - expected)
    assert result.observed_agreement == exactly(Fraction(13, 19))
    assert result.expected_agreement == exactly(expected)
    assert result.kappa == exactly(pi)
    assert result.kappa != pytest.approx(float(Fraction(107, 221)))


def test_fleiss_kappa_ignores_the_order_of_the_ratings_and_of_the_categories() -> None:
    ratings = from_counts(WIKIPEDIA_COUNTS, FIVE)
    rng = np.random.default_rng(SEED)
    shuffled = [list(rng.permutation(item)) for item in ratings]

    result = fleiss_kappa(shuffled, categories=tuple(reversed(FIVE)))

    assert result.counts == tuple(tuple(reversed(row)) for row in WIKIPEDIA_COUNTS)
    assert result.kappa == exactly(fleiss_by_hand(WIKIPEDIA_COUNTS)[2])


def test_an_unused_category_is_an_empty_column_for_fleiss_kappa() -> None:
    first, second = from_table(SAS_TABLE)
    ratings = list(zip(first, second, strict=True))

    result = fleiss_kappa(ratings, categories=(*LEVELS, "other"))

    assert all(row[-1] == 0 for row in result.counts)
    assert result.category_shares[-1] == 0.0
    assert result.kappa == pytest.approx(fleiss_kappa(ratings, categories=LEVELS).kappa)


def test_fleiss_kappa_is_one_under_full_agreement() -> None:
    result = fleiss_kappa([["low"] * 3, ["high"] * 3, ["mid"] * 3], categories=LEVELS)

    assert result.observed_agreement == 1.0
    assert result.kappa == 1.0


def test_fleiss_kappa_is_undefined_when_every_rating_is_one_same_category() -> None:
    result = fleiss_kappa([["mid", "mid"]] * 4, categories=LEVELS)

    assert result.kappa is None
    assert result.undefined_reason == "expected_agreement_is_one"
    assert result.observed_agreement == 1.0
    assert result.expected_agreement == 1.0
    assert result.category_shares == (0.0, 1.0, 0.0)


def test_no_items_give_no_fleiss_kappa() -> None:
    result = fleiss_kappa([], categories=LEVELS)

    assert result.items == 0
    assert result.raters_per_item is None
    assert result.counts == ()
    assert result.category_shares is None
    assert result.observed_agreement is None
    assert result.expected_agreement is None
    assert result.kappa is None
    assert result.undefined_reason == "no_items"


# --- Arguments ---------------------------------------------------------------------------


def test_the_categories_and_the_weights_are_required_keywords() -> None:
    for function, names in (
        (cohen_kappa, ("categories", "weights")),
        (consensus_labels, ("categories",)),
        (fleiss_kappa, ("categories",)),
    ):
        parameters = inspect.signature(function).parameters
        for name in names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("first", "second", "categories", "weights", "message"),
    [
        (["low"], ["low", "mid"], LEVELS, "none", "1 labels but the second annotator has 2"),
        (["low"], ["top"], LEVELS, "none", "'top' is not one of the categories"),
        (["low"], [None], LEVELS, "none", "None is not one of the categories"),
        (["low"], ["low"], ("low", "low", "mid"), "none", "repeat"),
        (["low"], ["low"], ("low",), "none", "at least two categories"),
        ([True], [True], (True, False), "none", "boolean"),
        ([0.5], [0.5], (0.5, 1), "none", "string or a non-boolean integer"),
        (["low"], [True], ("low", 1), "none", "True is not one of the categories"),
        (["low"], ["low"], LEVELS, "cubic", "weights"),
    ],
)
def test_invalid_kappa_arguments_raise(first, second, categories, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        cohen_kappa(first, second, categories=categories, weights=weights)


@pytest.mark.parametrize(
    ("pair_ids", "first", "second", "message"),
    [
        (["p1", "p1"], ["low", "mid"], ["low", "mid"], "'p1' is repeated"),
        (["p1", ""], ["low", "mid"], ["low", "mid"], "non-empty string"),
        (["p1", 2], ["low", "mid"], ["low", "mid"], "non-empty string"),
        (["p1"], ["low", "mid"], ["low", "mid"], "1 pair ids but 2 labels"),
        (["p1"], ["low"], ["top"], "'top' is not one of the categories"),
    ],
)
def test_invalid_consensus_arguments_raise(pair_ids, first, second, message) -> None:
    with pytest.raises(ValueError, match=message):
        consensus_labels(pair_ids, first, second, categories=LEVELS)


@pytest.mark.parametrize(
    ("ratings", "categories", "message"),
    [
        ([["low", "mid"], ["low", "mid", "high"]], LEVELS, "item 1 has 3 ratings but item 0 has 2"),
        ([["low"], ["mid"]], LEVELS, "item 0 has 1 ratings; each item needs at least two"),
        ([["low", "top"]], LEVELS, "'top' is not one of the categories"),
        ([["low", True]], ("low", 1), "True is not one of the categories"),
        (["lowmid"], LEVELS, "item 0 must be a sequence of labels, not the string 'lowmid'"),
        ([["low", "low"]], ("low", "low"), "repeat"),
        ([["low", "low"]], ("low",), "at least two categories"),
    ],
)
def test_invalid_fleiss_arguments_raise(ratings, categories, message) -> None:
    with pytest.raises(ValueError, match=message):
        fleiss_kappa(ratings, categories=categories)
