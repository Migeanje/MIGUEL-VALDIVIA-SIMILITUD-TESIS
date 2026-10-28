"""The assignment of the labeled set to the pairs of annotators (T20b, D34 (8), D31 (b))."""

import random
from collections import Counter
from dataclasses import dataclass

import pytest

from thematic_redundancy.labeling.assignment import (
    Assignment,
    annotator_pairs,
    assign_pairs,
    pairs_of,
)

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass(frozen=True)
class Drawn:
    pair_id: str
    pool: str
    source: str | None = None


def labeled_set() -> list[Drawn]:
    """The labeled set's pools as D34 (5) declares them: 20 + 20, 20, 30 and 30 pairs."""
    sizes = (
        ("sbert_high", MINILM, 20),
        ("sbert_high", MPNET, 20),
        ("tfidf_high", None, 20),
        ("mid_range", None, 30),
        ("random", None, 30),
    )
    pairs = []
    for pool, source, size in sizes:
        pairs += [Drawn(f"L-{len(pairs) + i:08x}", pool, source) for i in range(size)]
    return pairs


def test_the_pairs_of_annotators_are_every_two_of_them() -> None:
    assert annotator_pairs(3) == ((1, 2), (1, 3), (2, 3))
    assert annotator_pairs(4) == ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    with pytest.raises(ValueError, match="at least 2 annotators"):
        annotator_pairs(1)


@pytest.mark.parametrize(
    ("annotators", "groups", "per_group", "per_annotator"),
    [(3, 3, 40, 80), (4, 6, 20, 60)],
)
def test_the_120_pairs_are_spread_evenly_over_the_pairs_of_annotators(
    annotators: int, groups: int, per_group: int, per_annotator: int
) -> None:
    pairs = labeled_set()

    assigned = assign_pairs(pairs, annotators, seed=20261021)

    assert sorted(a.pair_id for a in assigned) == sorted(p.pair_id for p in pairs)
    sizes = Counter(a.group for a in assigned)
    assert sizes == {group: per_group for group in range(1, groups + 1)}
    for a in assigned:
        assert a.annotators == annotator_pairs(annotators)[a.group - 1]
    for annotator in range(1, annotators + 1):
        assert len(pairs_of(assigned, annotator)) == per_annotator


@pytest.mark.parametrize("annotators", [3, 4])
def test_every_pool_and_model_is_spread_evenly_too(annotators: int) -> None:
    pairs = labeled_set()
    kind = {p.pair_id: (p.pool, p.source) for p in pairs}

    assigned = assign_pairs(pairs, annotators, seed=20261021)

    groups = len(annotator_pairs(annotators))
    for wanted in set(kind.values()):
        counts = Counter(a.group for a in assigned if kind[a.pair_id] == wanted)
        shares = [counts[group] for group in range(1, groups + 1)]
        assert max(shares) - min(shares) <= 1, wanted
    for pool in ("sbert_high", "tfidf_high", "mid_range", "random"):
        counts = Counter(a.group for a in assigned if kind[a.pair_id][0] == pool)
        shares = [counts[group] for group in range(1, groups + 1)]
        assert max(shares) - min(shares) <= 1, pool


def test_the_assignment_follows_the_documented_seed_scheme() -> None:
    pairs = labeled_set()
    dealt: list[str] = []
    for pool, source in (
        ("sbert_high", MINILM),
        ("sbert_high", MPNET),
        ("tfidf_high", None),
        ("mid_range", None),
        ("random", None),
    ):
        members = sorted(p.pair_id for p in pairs if (p.pool, p.source) == (pool, source))
        random.Random(f"7:assignment:{pool}" + (f":{source}" if source else "")).shuffle(members)
        dealt += members

    assigned = assign_pairs(list(reversed(pairs)), 3, seed=7)

    groups = ((1, 2), (1, 3), (2, 3))
    assert assigned == tuple(
        Assignment(identifier, index % 3 + 1, groups[index % 3])
        for index, identifier in enumerate(dealt)
    )


def test_the_same_seed_gives_the_same_assignment_and_another_seed_another_one() -> None:
    pairs = labeled_set()
    shuffled = list(pairs)
    random.Random(3).shuffle(shuffled)

    first = assign_pairs(pairs, 4, seed=1)

    assert assign_pairs(shuffled, 4, seed=1) == first
    assert assign_pairs(pairs, 4, seed=2) != first


def test_groups_that_do_not_divide_the_pairs_differ_by_one_pair_at_most() -> None:
    pairs = labeled_set()[:-1]

    sizes = Counter(a.group for a in assign_pairs(pairs, 4, seed=1))

    assert sum(sizes.values()) == 119
    assert max(sizes.values()) - min(sizes.values()) == 1


def test_a_repeated_pair_or_an_unknown_pool_is_refused() -> None:
    pairs = labeled_set()

    with pytest.raises(ValueError, match="repeats"):
        assign_pairs([*pairs, pairs[0]], 3, seed=1)
    with pytest.raises(ValueError, match="unknown pools"):
        assign_pairs([*pairs, Drawn("L-ffffffff", "top_20")], 3, seed=1)
