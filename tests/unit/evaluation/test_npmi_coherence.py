"""NPMI topic coherence (T17, D12) on tiny corpora whose expected values are worked out by hand,
and on seeded random corpora against an independent brute-force oracle.

A reference document is a sequence of segments (token lists), as a thesis is a sequence of
sections: in both co-occurrence units, a multi-word term occurs only where its words appear
consecutively inside one segment, and a window never spans two segments.
"""

import inspect
import math
from itertools import combinations

import numpy as np
import pytest

from thematic_redundancy.evaluation.coherence import (
    CoherenceResult,
    npmi,
    npmi_coherence,
)


def docs(*texts: str) -> list[list[list[str]]]:
    """One single-segment document per text; ``|`` separates the segments of a document."""
    return [[segment.split() for segment in text.split("|")] for text in texts]


def by_document(topics, reference, top_n: int = 2) -> CoherenceResult:
    return npmi_coherence(topics, reference, top_n=top_n, unit="document", window_size=None)


def by_window(topics, reference, window_size: int, top_n: int = 2) -> CoherenceResult:
    return npmi_coherence(topics, reference, top_n=top_n, unit="window", window_size=window_size)


def exact(value: float) -> object:
    return pytest.approx(value, rel=0, abs=1e-15)


# --- The formula and its limits ----------------------------------------------------------


def test_pairs_that_never_co_occur_get_minus_one_exactly() -> None:
    assert npmi(0, 2, 3, 10) == -1.0


def test_pairs_in_every_unit_get_one_the_limit_of_complete_co_occurrence() -> None:
    assert npmi(10, 10, 10, 10) == 1.0


def test_pairs_that_only_occur_together_get_one() -> None:
    assert npmi(3, 3, 3, 10) == 1.0


def test_independent_terms_get_zero() -> None:
    # p(a, b) = 2/10 = p(a) p(b) = 4/10 * 5/10.
    assert npmi(2, 4, 5, 10) == exact(0.0)


def test_the_formula_matches_a_hand_computed_value() -> None:
    # log(p(a, b) / (p(a) p(b))) / -log p(a, b), with p(a, b) = 2/4, p(a) = 3/4, p(b) = 2/4.
    assert npmi(2, 3, 2, 4) == exact(math.log(4 / 3) / math.log(2))


@pytest.mark.parametrize(
    "counts",
    [
        (1, 0, 2, 4),  # an absent term has no NPMI
        (3, 2, 4, 4),  # more joint than single occurrences
        (1, 5, 2, 4),  # more occurrences than units
        (1, 3, 3, 4),  # 3 + 3 - 1 = 5 units needed
        (-1, 2, 2, 4),
        (0, 1, 1, 0),
    ],
)
def test_impossible_counts_raise(counts: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="counts"):
        npmi(*counts)


# --- Document co-occurrence --------------------------------------------------------------


def test_document_coherence_matches_a_hand_computed_corpus() -> None:
    reference = docs("a b c", "a b", "a c", "d")

    result = by_document([["a", "b", "c"], ["b", "d"]], reference, top_n=2)
    three = by_document([["a", "b", "c"]], reference, top_n=3)

    # 4 documents; a in 3, b in 2, c in 2, d in 1; ab 2, ac 2, bc 1, bd 0.
    ab = math.log((2 / 4) / ((3 / 4) * (2 / 4))) / -math.log(2 / 4)
    bc = math.log((1 / 4) / ((2 / 4) * (2 / 4))) / -math.log(1 / 4)
    assert three.units == 4
    assert three.topics[0].terms == ("a", "b", "c")
    assert three.topics[0].pair_npmi == (exact(ab), exact(ab), exact(bc))
    assert bc == 0.0
    assert three.topics[0].npmi == exact((ab + ab + bc) / 3)
    assert three.mean == exact((ab + ab + bc) / 3)
    assert result.topics[0].pair_npmi == (exact(ab),)
    assert result.topics[1].pair_npmi == (-1.0,)
    assert result.mean == exact((ab - 1.0) / 2)
    assert result.undefined_reason is None


def test_document_units_are_documents_not_segments() -> None:
    reference = docs("a|b", "c")

    result = by_document([["a", "b"]], reference)

    # 2 documents, both terms in the first: p(a, b) = p(a) = p(b) = 1/2.
    assert result.units == 2
    assert result.topics[0].npmi == 1.0


def test_an_empty_document_still_counts_as_a_unit() -> None:
    reference = [[["a", "b"]], [[]], [["a"]]]

    result = by_document([["a", "b"]], reference)

    assert result.units == 3
    assert result.topics[0].npmi == exact(math.log((1 / 3) / ((2 / 3) * (1 / 3))) / math.log(3))


# --- Window co-occurrence ----------------------------------------------------------------


def test_window_and_document_units_differ_where_terms_are_far_apart() -> None:
    reference = docs("a x x x b", "a b", "x")

    document = by_document([["a", "b"]], reference)
    window = by_window([["a", "b"]], reference, window_size=3)

    # Documents: a and b both in 2 of 3, always together.
    assert document.units == 3
    assert document.topics[0].npmi == 1.0
    # Windows of 3: [a x x] [x x x] [x x b], then [a b] (shorter than 3, one window), then [x].
    assert window.units == 5
    assert window.topics[0].npmi == exact(math.log((1 / 5) / ((2 / 5) * (2 / 5))) / math.log(5))


@pytest.mark.parametrize(
    ("window_size", "units", "expected"),
    [
        # [a x b] [x b y] [b y y] [y y y]: a 1, b 3, ab 1.
        (3, 4, math.log((1 / 4) / ((1 / 4) * (3 / 4))) / math.log(4)),
        # [a x] [x b] [b y] [y y] [y y]: a 1, b 2, ab 0.
        (2, 5, -1.0),
        # [a x b y] [x b y y] [b y y y]: a 1, b 3, ab 1.
        (4, 3, math.log((1 / 3) / ((1 / 3) * (3 / 3))) / math.log(3)),
    ],
)
def test_a_segment_of_l_tokens_gives_l_minus_w_plus_one_windows(
    window_size: int, units: int, expected: float
) -> None:
    result = by_window([["a", "b"]], docs("a x b y y y"), window_size=window_size)

    assert result.units == units
    assert result.topics[0].npmi == exact(expected)


def test_windows_count_short_segments_once_and_empty_segments_never() -> None:
    reference = [[["a", "b"], []], [["a", "x", "b", "y"]]]

    result = by_window([["a", "b"]], reference, window_size=3)

    # [a b], nothing for the empty segment, then [a x b] [x b y]: a 2, b 3, ab 2 of 3.
    assert result.units == 3
    assert result.topics[0].npmi == exact(
        math.log((2 / 3) / ((2 / 3) * (3 / 3))) / -math.log(2 / 3)
    )


def test_a_window_never_spans_two_segments() -> None:
    reference = docs("a x|y b")

    window = by_window([["a", "b"]], reference, window_size=4)
    document = by_document([["a", "b"]], reference)

    assert window.units == 2
    assert window.topics[0].npmi == -1.0
    assert document.topics[0].npmi == 1.0


# --- Multi-word terms --------------------------------------------------------------------


def test_a_bigram_occurs_only_where_its_words_are_consecutive_within_a_segment() -> None:
    # "a b" is consecutive only in the first document: apart in the second, reversed in the
    # third, split across two segments in the fourth.
    reference = docs("a b c", "a x b", "b a", "a|b", "c")

    with_c = by_document([["a b", "c"]], reference)
    with_a = by_document([["a b", "a"]], reference)

    assert with_c.topics[0].npmi == exact(math.log((1 / 5) / ((1 / 5) * (2 / 5))) / math.log(5))
    assert with_a.topics[0].npmi == exact(math.log((1 / 5) / ((1 / 5) * (4 / 5))) / math.log(5))


def test_a_bigram_is_in_a_window_only_when_both_its_words_are() -> None:
    # Windows of 2: [x a] [a b] [b c]; "a b" is only in the second, x only in the first.
    result = by_window([["a b", "x"]], docs("x a b c"), window_size=2)

    assert result.units == 3
    assert result.topics[0].npmi == -1.0


# --- Topics and terms --------------------------------------------------------------------


def test_only_the_first_top_n_terms_are_scored() -> None:
    reference = docs("a b", "a c")

    result = by_document([["a", "b", "never"]], reference, top_n=2)

    assert result.topics[0].terms == ("a", "b")
    assert result.topics[0].absent_terms == ()
    assert result.topics[0].npmi is not None


def test_pairs_are_scored_in_the_order_of_the_terms() -> None:
    reference = docs("a b", "a c", "b c d", "d")

    result = by_document([["d", "a", "c"]], reference, top_n=3)
    reordered = by_document([["a", "c", "d"]], reference, top_n=3)

    # (d, a), (d, c), (a, c) against (a, c), (a, d), (c, d).
    first = result.topics[0].pair_npmi
    second = reordered.topics[0].pair_npmi
    assert first == pytest.approx((second[1], second[2], second[0]), rel=1e-15)
    assert result.topics[0].npmi == exact(reordered.topics[0].npmi)


def test_an_absent_term_is_reported_and_leaves_its_topic_and_the_mean_undefined() -> None:
    reference = docs("a b", "a c", "b c")

    result = by_document([["a", "zeta", "b"], ["a", "b", "c"], ["b a", "c", "a"]], reference, 3)

    absent, scored, absent_bigram = result.topics
    assert absent.absent_terms == ("zeta",)
    assert absent.undefined_reason == "absent_terms"
    assert absent.npmi is None
    # Only the pairs with the absent term are undefined.
    assert absent.pair_npmi[0] is None
    assert absent.pair_npmi[1] == exact(scored.pair_npmi[0])
    assert absent.pair_npmi[2] is None
    assert scored.undefined_reason is None
    assert scored.absent_terms == ()
    assert scored.npmi is not None
    # Its words occur, but never consecutively: the bigram is absent.
    assert absent_bigram.absent_terms == ("b a",)
    assert result.mean is None
    assert result.undefined_reason == "undefined_topics"


def test_no_topic_gives_no_mean() -> None:
    result = by_document([], docs("a b"))

    assert result.topics == ()
    assert result.mean is None
    assert result.undefined_reason == "no_topics"


def test_an_empty_reference_leaves_every_term_absent() -> None:
    result = by_window([["a", "b"]], [], window_size=3)

    assert result.units == 0
    assert result.topics[0].absent_terms == ("a", "b")
    assert result.undefined_reason == "undefined_topics"


def test_the_result_records_its_parameters() -> None:
    result = by_window([["a", "b"]], docs("a b"), window_size=5, top_n=2)

    assert (result.unit, result.window_size, result.top_n) == ("window", 5, 2)


def test_the_same_input_gives_the_same_result() -> None:
    reference = docs("a b c", "a x b", "c a b", "b")
    topics = [["a", "b", "c"], ["a b", "c", "x"]]

    first = by_window(topics, reference, window_size=2, top_n=3)
    second = by_window([list(topic) for topic in topics], list(reference), 2, top_n=3)

    assert first == second


# --- An independent oracle ---------------------------------------------------------------


def oracle(topics, reference, top_n: int, window_size: int | None):
    """Enumerate every unit as a list of token lists, test each term in each unit by slicing,
    and count with numpy."""
    units: list[list[list[str]]] = []
    for document in reference:
        if window_size is None:
            units.append([list(segment) for segment in document])
            continue
        for segment in document:
            tokens = list(segment)
            for start in range(max(len(tokens) - window_size + 1, 1) if tokens else 0):
                units.append([tokens[start : start + window_size]])

    def occurs(term: str, unit: list[list[str]]) -> bool:
        words = term.split(" ")
        return any(
            tokens[at : at + len(words)] == words
            for tokens in unit
            for at in range(len(tokens) - len(words) + 1)
        )

    values = []
    for topic in topics:
        terms = topic[:top_n]
        presence = np.array([[occurs(term, unit) for unit in units] for term in terms], bool)
        pairs = []
        for i, j in combinations(range(len(terms)), 2):
            p_i, p_j = presence[i].mean(), presence[j].mean()
            p_ij = (presence[i] & presence[j]).mean()
            if p_i == 0 or p_j == 0:
                pairs.append(None)
            elif p_ij == 0:
                pairs.append(-1.0)
            elif p_ij == 1:
                pairs.append(1.0)
            else:
                pairs.append(float(np.log(p_ij / (p_i * p_j)) / -np.log(p_ij)))
        values.append(pairs)
    return len(units), values


def random_case(seed: int):
    rng = np.random.default_rng(seed)
    lemmas = ["a", "b", "c", "d", "e", "f"]
    reference = [
        [
            [str(token) for token in rng.choice(lemmas, size=rng.integers(0, 13))]
            for _ in range(rng.integers(1, 4))
        ]
        for _ in range(30)
    ]
    terms = lemmas + [f"{first} {second}" for first, second in combinations(lemmas, 2)]
    topics = [[str(term) for term in rng.choice(terms, size=5, replace=False)] for _ in range(6)]
    return topics, reference


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("window_size", [None, 2, 3, 5, 20])
def test_coherence_matches_an_independent_oracle(seed: int, window_size: int | None) -> None:
    topics, reference = random_case(seed)
    unit = "document" if window_size is None else "window"

    result = npmi_coherence(topics, reference, top_n=4, unit=unit, window_size=window_size)
    units, expected = oracle(topics, reference, 4, window_size)

    assert result.units == units
    for topic, pairs in zip(result.topics, expected, strict=True):
        assert len(topic.pair_npmi) == 6
        for got, want in zip(topic.pair_npmi, pairs, strict=True):
            assert (got is None) == (want is None)
            if want is not None:
                assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
        if None not in pairs:
            assert topic.npmi == pytest.approx(np.mean(pairs), rel=1e-12, abs=1e-12)


# --- Arguments ---------------------------------------------------------------------------


def test_the_methodological_arguments_have_no_default() -> None:
    parameters = inspect.signature(npmi_coherence).parameters

    for name in ("top_n", "unit", "window_size"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("topics", "top_n", "unit", "window_size", "message"),
    [
        ([["a", "b"]], 1, "document", None, "top_n"),
        ([["a", "b"]], 3, "document", None, "fewer than 3 terms"),
        ([["a", "b", "a"]], 3, "document", None, "repeats"),
        ([["a", ""]], 2, "document", None, "term"),
        ([["a", "a  b"]], 2, "document", None, "term"),
        ([["a", " b"]], 2, "document", None, "term"),
        ([["a", "b"]], 2, "sentence", None, "unit"),
        ([["a", "b"]], 2, "window", None, "window_size"),
        ([["a", "b"]], 2, "window", 1, "window_size"),
        ([["a", "b c d"]], 2, "window", 2, "window_size"),
        ([["a", "b"]], 2, "document", 10, "window_size"),
    ],
)
def test_invalid_arguments_raise(topics, top_n, unit, window_size, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        npmi_coherence(topics, docs("a b"), top_n=top_n, unit=unit, window_size=window_size)


@pytest.mark.parametrize("reference", [[["a", "b"], ["c"]], ["a b"]])
def test_token_lists_passed_as_documents_raise(reference) -> None:
    """Each document is a sequence of segments, so a flat list of token lists would turn each
    token into a segment of characters."""
    with pytest.raises(TypeError, match="segment"):
        by_document([["a", "b"]], reference)


@pytest.mark.parametrize(
    "topics",
    [["ab"], [["a", "b"], "ab"], "ab"],
    ids=["one-string-topic", "a-string-among-lists", "topics-as-one-string"],
)
def test_a_topic_passed_as_a_string_raises(topics) -> None:
    """Each topic is a sequence of terms, so a string would turn each character into a term."""
    with pytest.raises(TypeError, match="topic"):
        by_document(topics, docs("a b"))
