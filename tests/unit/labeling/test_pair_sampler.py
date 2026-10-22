"""Behavior of the labeled set's pair sampler (T20a, D34): the universe, the ranks, the pools,
the strata, the pilot and the draw, on hand-made cases and on the synthetic corpus of
``conftest.py``."""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from thematic_redundancy.labeling.objectives_check import allocate
from thematic_redundancy.labeling.pairs import (
    BETWEEN,
    LABELED,
    MID_RANGE,
    PILOT,
    RANDOM,
    SBERT_HIGH,
    TFIDF,
    TFIDF_HIGH,
    DocumentVectors,
    LabeledSetDraw,
    PairUniverse,
    SamplerError,
    allocate_with_floor,
    build_universe,
    draw_labeled_set,
    mid_range_mask,
    pair_id,
    percentile_ranks,
    pilot_allocation,
    top_count,
    top_pairs,
)
from thematic_redundancy.redundancy.search import pair_cosines

# Ranks, the top and the mid-range


def test_a_pairs_percentile_rank_is_its_average_rank_over_the_pairs() -> None:
    ranks = percentile_ranks(np.array([0.5, 0.1, 0.9, 0.5]))

    assert ranks.tolist() == [0.625, 0.25, 1.0, 0.625]


@pytest.mark.parametrize(
    ("pairs", "share", "expected"),
    [(276_396, 0.01, 2_764), (100, 0.07, 7), (300, 0.01, 3), (301, 0.01, 4), (1, 0.01, 1)],
)
def test_the_top_count_rounds_the_share_of_the_pairs_up_exactly(
    pairs: int, share: float, expected: int
) -> None:
    # 0.07 × 100 is 7.000000000000001 in binary floating point, which would round up to 8.
    assert top_count(pairs, share) == expected


def test_the_top_is_the_pairs_with_the_highest_scores_and_a_tie_goes_to_the_earlier() -> None:
    scores = np.array([0.3, 0.9, 0.5, 0.9, 0.5, 0.1])

    assert top_pairs(scores, 3).tolist() == [1, 3, 2]
    assert top_pairs(scores, 1).tolist() == [1]


def mid_scores() -> tuple[list[np.ndarray], dict[int, int]]:
    """Return three representations' scores of 100 pairs, each a permutation of 1 to 100 so a
    pair's percentile rank is its score over 100, and the sum of the six edge pairs' ranks.

    Pairs 0 and 1 sit at a mean of exactly 0.90 and pairs 2 and 3 at exactly 0.99; pair 4 is
    just below the band and pair 5 inside it."""
    fixed = [
        {0: 90, 1: 89, 2: 99, 3: 98, 4: 88, 5: 97},
        {0: 91, 1: 93, 2: 100, 3: 99, 4: 90, 5: 96},
        {0: 89, 1: 88, 2: 98, 3: 100, 4: 91, 5: 97},
    ]
    scores = []
    for index, chosen in enumerate(fixed):
        rest = [value for value in range(1, 101) if value not in chosen.values()]
        random.Random(index).shuffle(rest)
        scores.append(np.array([*chosen.values(), *rest], dtype=np.float64))
    return scores, {pair: sum(row[pair] for row in fixed) for pair in range(6)}


def test_the_mid_range_holds_its_lower_edge_and_leaves_out_its_upper_edge() -> None:
    scores, sums = mid_scores()

    inside = mid_range_mask(scores, low=0.90, high=0.99)

    assert inside[:6].tolist() == [True, True, False, False, False, True]
    assert sums == {0: 270, 1: 270, 2: 297, 3: 297, 4: 269, 5: 290}
    totals = np.sum(scores, axis=0)
    assert inside.tolist() == ((totals >= 270) & (totals < 297)).tolist()


# Allocation


@pytest.mark.parametrize(
    ("sizes", "total", "floor", "expected"),
    [
        # MiniLM's top 1 % of the real corpus (D34 (6)): every stratum gets 2, then 8 by size.
        (
            {
                "sistemas": 13,
                "industrial": 1909,
                "electronica": 8,
                "mecanica": 198,
                "minas": 272,
                "between": 364,
            },
            20,
            2,
            {
                "sistemas": 2,
                "industrial": 7,
                "electronica": 2,
                "mecanica": 3,
                "minas": 3,
                "between": 3,
            },
        ),
        # A stratum with one candidate gives it, and the rest goes to the others.
        ({"a": 1, "b": 100, "c": 50}, 10, 2, {"a": 1, "b": 5, "c": 4}),
        # A stratum without candidates gets nothing, not even the floor.
        ({"a": 0, "b": 10, "c": 10}, 6, 2, {"a": 0, "b": 3, "c": 3}),
        # Sizes that hold no more than the total give all they hold.
        ({"a": 2, "b": 3}, 10, 2, {"a": 2, "b": 3}),
        # A tie of remainders goes to the earlier key.
        ({"a": 10, "b": 10}, 5, 2, {"a": 3, "b": 2}),
        # Without a floor, it is the largest remainder of the sizes.
        ({"a": 6, "b": 3, "c": 1}, 5, 0, {"a": 3, "b": 2, "c": 0}),
    ],
)
def test_a_quota_gives_each_stratum_its_floor_and_the_rest_by_largest_remainder(
    sizes: dict[str, int], total: int, floor: int, expected: dict[str, int]
) -> None:
    quotas = allocate_with_floor(sizes, total, floor)

    assert quotas == expected
    assert list(quotas) == list(sizes)
    assert sum(quotas.values()) == min(total, sum(sizes.values()))


def test_without_a_floor_the_allocation_is_the_objectives_samples() -> None:
    sizes = {"extracted": 11, "not_found": 2}

    assert (
        allocate_with_floor(sizes, 12, 0)
        == allocate(sizes, 12)
        == {"extracted": 10, "not_found": 2}
    )


def test_floors_that_exceed_the_quota_are_refused() -> None:
    with pytest.raises(ValueError, match="floor"):
        allocate_with_floor({"a": 5, "b": 5, "c": 5}, 5, 2)


@pytest.mark.parametrize(
    ("pilot_size", "expected"),
    [
        (10, (2, 1, 2, 3, 2)),
        (8, (1, 1, 2, 2, 2)),
        (13, (2, 2, 3, 3, 3)),
    ],
)
def test_the_pilot_gives_each_pool_its_floor_and_the_rest_by_largest_remainder(
    kit: SimpleNamespace, pilot_size: int, expected: tuple[int, ...]
) -> None:
    # The pools hold 40, 20, 30 and 30 pairs: with 10, the 2 left after the floors go to
    # SBERT-high (remainder 0.68) and to the mid-range, earlier than random at the same 0.5.
    settings = kit.config.labeling.model_copy(update={"pilot_size": pilot_size})

    allocation = pilot_allocation(settings, kit.models)

    assert [(pool, model) for pool, model, _ in allocation] == [
        (SBERT_HIGH, kit.models[0]),
        (SBERT_HIGH, kit.models[1]),
        (TFIDF_HIGH, None),
        (MID_RANGE, None),
        (RANDOM, None),
    ]
    assert tuple(quota for *_, quota in allocation) == expected


# The universe


def tiny_universe(kit: SimpleNamespace) -> PairUniverse:
    vectors = {
        "DOC-000000a0": [1.0, 0.0],
        "DOC-000000b0": [1.0, 1.0],
        "DOC-000000c0": [0.0, 1.0],
        "DOC-000000d0": [-1.0, 2.0],
    }
    rows = ("DOC-000000d0", "DOC-000000b0", "DOC-000000a0", "DOC-000000c0")
    dense = np.array([vectors[code] for code in rows])
    return build_universe(
        [
            ("DOC-000000c0", "minas"),
            ("DOC-000000a0", "sistemas"),
            ("DOC-000000d0", "industrial"),
            ("DOC-000000b0", "sistemas"),
        ],
        [
            DocumentVectors(model, rows, dense * (index + 1))
            for index, model in enumerate(kit.models)
        ],
        DocumentVectors(TFIDF, rows, sparse.csr_matrix(np.abs(dense))),
        programs=kit.programs,
    )


def test_a_pairs_stratum_is_the_program_both_share_or_between(kit: SimpleNamespace) -> None:
    universe = tiny_universe(kit)

    assert universe.doc_codes == ("DOC-000000a0", "DOC-000000b0", "DOC-000000c0", "DOC-000000d0")
    assert universe.strata == (*kit.programs, BETWEEN)
    assert [universe.codes(pair) for pair in range(universe.size)] == [
        ("DOC-000000a0", "DOC-000000b0"),
        ("DOC-000000a0", "DOC-000000c0"),
        ("DOC-000000a0", "DOC-000000d0"),
        ("DOC-000000b0", "DOC-000000c0"),
        ("DOC-000000b0", "DOC-000000d0"),
        ("DOC-000000c0", "DOC-000000d0"),
    ]
    assert [universe.stratum_of(pair) for pair in range(universe.size)] == [
        "sistemas",
        BETWEEN,
        BETWEEN,
        BETWEEN,
        BETWEEN,
        BETWEEN,
    ]
    assert universe.counts(range(universe.size)) == {
        "sistemas": 1,
        "industrial": 0,
        "electronica": 0,
        "mecanica": 0,
        "minas": 0,
        BETWEEN: 5,
    }


def test_the_universe_scores_each_pair_with_the_vectors_of_its_own_theses(
    kit: SimpleNamespace,
) -> None:
    universe = tiny_universe(kit)
    root = math.sqrt(2)

    assert universe.representations == (*kit.models, TFIDF)
    expected = [1 / root, 0.0, -1 / math.sqrt(5), 1 / root, 1 / math.sqrt(10), 2 / math.sqrt(5)]
    for scores in universe.scores[:2]:
        assert scores == pytest.approx(expected)
    assert universe.scores[2][2] == pytest.approx(1 / math.sqrt(5))
    # Pairs 0 and 3 tie at 1/√2, so they share the ranks 4 and 5.
    assert universe.ranks[0].tolist() == [4.5 / 6, 2 / 6, 1 / 6, 4.5 / 6, 3 / 6, 1.0]


def test_the_synthetic_universe_matches_pair_cosines_on_aligned_rows(
    universe: PairUniverse, corpus: object
) -> None:
    by_code = {code: row for row, (code, _) in enumerate(corpus.theses)}  # type: ignore[attr-defined]
    rows = [by_code[code] for code in universe.doc_codes]
    pairs = np.column_stack((universe.first, universe.second))
    for name, scores in zip(universe.representations, universe.scores, strict=True):
        aligned = corpus.vectors[name][rows]  # type: ignore[attr-defined]
        assert np.array_equal(scores, pair_cosines(aligned, pairs))
    assert universe.size == 130 * 129 // 2
    assert list(universe.doc_codes) == sorted(universe.doc_codes)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "theses of .* differ"),
        ("repeated", "theses of .* differ"),
        ("program", "program 'quimica' .* not one of"),
        ("duplicate_thesis", "repeat"),
    ],
)
def test_a_universe_whose_inputs_disagree_is_refused(
    kit: SimpleNamespace, inputs: object, change: str, message: str
) -> None:
    theses = list(inputs.theses)  # type: ignore[attr-defined]
    semantic = list(inputs.semantic)  # type: ignore[attr-defined]
    first = semantic[0]
    if change == "missing":
        semantic[0] = DocumentVectors(first.name, first.doc_codes[1:], first.vectors[1:])
    elif change == "repeated":
        codes = (first.doc_codes[1], *first.doc_codes[1:])
        semantic[0] = DocumentVectors(first.name, codes, first.vectors)
    elif change == "program":
        theses[3] = (theses[3][0], "quimica")
    else:
        theses.append(theses[0])

    with pytest.raises(SamplerError, match=message):
        build_universe(theses, semantic, inputs.tfidf, programs=kit.programs)  # type: ignore[attr-defined]


# The draw


STEPS = (
    (PILOT, SBERT_HIGH, 0, 2),
    (PILOT, SBERT_HIGH, 1, 1),
    (PILOT, TFIDF_HIGH, None, 2),
    (PILOT, MID_RANGE, None, 3),
    (PILOT, RANDOM, None, 2),
    (LABELED, SBERT_HIGH, 0, 20),
    (LABELED, SBERT_HIGH, 1, 20),
    (LABELED, TFIDF_HIGH, None, 20),
    (LABELED, MID_RANGE, None, 30),
    (LABELED, RANDOM, None, 30),
)
"""The declared steps: phase, pool, model index and quota."""


def pool_candidates(draw: LabeledSetDraw, pool: str, source: str | None) -> np.ndarray:
    universe = draw.universe
    if pool == SBERT_HIGH:
        return np.sort(draw.tops[universe.representations.index(source)])
    if pool == TFIDF_HIGH:
        return np.sort(draw.tops[universe.representations.index(TFIDF)])
    if pool == MID_RANGE:
        return draw.mid_range
    return np.arange(universe.size)


def test_the_tops_and_the_mid_range_are_those_of_the_declared_values(
    draw: LabeledSetDraw, kit: SimpleNamespace
) -> None:
    universe, settings = draw.universe, kit.config.labeling

    assert draw.top_count == 84
    for top, scores in zip(draw.tops, universe.scores, strict=True):
        assert top.tolist() == top_pairs(scores, 84).tolist()
    expected = mid_range_mask(universe.scores, low=settings.mid_low, high=settings.mid_high)
    assert draw.mid_range.tolist() == np.flatnonzero(expected).tolist()
    assert draw.seed == settings.seed == 20261021


def test_the_draw_follows_the_declared_order_and_never_repeats_a_pair(
    draw: LabeledSetDraw, kit: SimpleNamespace
) -> None:
    observed = [
        (step.phase, step.pool, step.source, step.quota, len(step.pairs)) for step in draw.steps
    ]

    assert observed == [
        (phase, pool, None if model is None else kit.models[model], quota, quota)
        for phase, pool, model, quota in STEPS
    ]
    assert len(draw.pairs) == len(set(draw.pairs)) == 130
    assert draw.pairs[:10] == tuple(p for step in draw.steps[:5] for p in step.pairs)


def test_every_pair_comes_from_its_pool_and_is_left_by_the_earlier_steps(
    draw: LabeledSetDraw,
) -> None:
    taken: set[int] = set()
    for step in draw.steps:
        candidates = [p for p in pool_candidates(draw, step.pool, step.source) if p not in taken]
        assert set(step.pairs) <= set(candidates)
        assert step.candidates == draw.universe.counts(candidates)
        taken.update(step.pairs)


def test_the_labeled_high_and_mid_range_pools_spread_their_quota_over_the_strata(
    draw: LabeledSetDraw, kit: SimpleNamespace
) -> None:
    floor = kit.config.labeling.stratum_floor
    stratified = [step for step in draw.steps if step.stratified]

    assert [(step.phase, step.pool) for step in stratified] == [
        (LABELED, SBERT_HIGH),
        (LABELED, SBERT_HIGH),
        (LABELED, TFIDF_HIGH),
        (LABELED, MID_RANGE),
    ]
    for step in draw.steps:
        drawn = draw.universe.counts(step.pairs)
        if step.stratified:
            assert step.quotas == allocate_with_floor(step.candidates, step.quota, floor)
            assert drawn == step.quotas
            assert all(
                drawn[stratum] >= min(floor, count) for stratum, count in step.candidates.items()
            )
        else:
            assert step.quotas is None
    # The floor bites: some stratum's proportional share is below it, and it still gets it.
    raised = [
        (step.pool, stratum)
        for step in stratified
        for stratum, count in step.candidates.items()
        if 0 < step.quota * count / sum(step.candidates.values()) < floor
        and step.quotas[stratum] == min(floor, count)  # type: ignore[index]
    ]
    assert raised


def test_every_sample_is_the_documented_draw_from_the_configured_seed(
    draw: LabeledSetDraw, kit: SimpleNamespace
) -> None:
    # The oracle: random.Random("<seed>:<phase>:<pool>[:<model>][:<stratum>]").sample over the
    # candidates left, in universe order. The random pool is so a uniform sample of every pair
    # that no earlier step drew, whatever its stratum.
    universe, seed = draw.universe, kit.config.labeling.seed
    taken: set[int] = set()
    for step in draw.steps:
        candidates = [
            int(p) for p in pool_candidates(draw, step.pool, step.source) if p not in taken
        ]
        key = f"{seed}:{step.phase}:{step.pool}" + (f":{step.source}" if step.source else "")
        if step.stratified:
            expected: list[int] = []
            for stratum, quota in step.quotas.items():  # type: ignore[union-attr]
                own = [p for p in candidates if universe.stratum_of(p) == stratum]
                expected += random.Random(f"{key}:{stratum}").sample(own, quota)
        else:
            expected = random.Random(key).sample(candidates, step.quota)
        assert list(step.pairs) == expected
        taken.update(step.pairs)


def test_the_random_pool_reaches_pairs_outside_every_other_pool(draw: LabeledSetDraw) -> None:
    random_pairs = [p for step in draw.steps if step.pool == RANDOM for p in step.pairs]
    elsewhere = {int(p) for top in draw.tops for p in top} | set(draw.mid_range.tolist())

    assert len(random_pairs) == 32
    assert any(p not in elsewhere for p in random_pairs)
    assert min(draw.universe.mean_ranks()[random_pairs]) < 0.5


def test_the_same_seed_draws_the_same_pairs_and_another_seed_others(
    universe: PairUniverse, draw: LabeledSetDraw, kit: SimpleNamespace
) -> None:
    settings = kit.config.labeling

    again = draw_labeled_set(universe, settings)
    other = draw_labeled_set(universe, settings.model_copy(update={"seed": settings.seed + 1}))

    assert again.steps == draw.steps
    assert other.pairs != draw.pairs
    assert [len(step.pairs) for step in other.steps] == [len(step.pairs) for step in draw.steps]


def test_a_pool_without_enough_pairs_left_stops_the_draw(
    universe: PairUniverse, kit: SimpleNamespace
) -> None:
    settings = kit.config.labeling.model_copy(update={"random": universe.size})

    with pytest.raises(SamplerError, match="random .* fewer than its quota"):
        draw_labeled_set(universe, settings)


def test_a_pair_id_names_its_phase_and_both_theses_in_either_order() -> None:
    pilot = pair_id(PILOT, "DOC-000000a0", "DOC-000000b0")

    assert pilot == pair_id(PILOT, "DOC-000000b0", "DOC-000000a0")
    assert pilot.startswith("P-") and len(pilot) == 10
    labeled = pair_id(LABELED, "DOC-000000a0", "DOC-000000b0")
    assert labeled == "L-" + pilot[2:]
    assert pair_id(LABELED, "DOC-000000a0", "DOC-000000c0") != labeled
