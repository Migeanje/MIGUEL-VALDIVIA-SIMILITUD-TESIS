"""The cross-seed stability filter of the gap rules (T24, D16: present in at least 4 of 5 seeds).

Topic ids differ from seed to seed, so a gap of the reference run is matched in another run by
the theses involved: a gap of the same rule (and, for rule (c), the same source and target)
whose topics' members overlap by a Jaccard index of at least the given value.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from thematic_redundancy.gaps.rules import (
    Gap,
    GapInputError,
    GapMap,
    InterstitialPair,
    SmallCoherentTopic,
    TemporalDecline,
    ThesisMetadata,
    TransferableTopic,
    small_coherent_topics,
)
from thematic_redundancy.gaps.stability import (
    GapSupport,
    gap_overlap,
    gap_support,
    jaccard,
    stable_gaps,
)


def members(*names: str) -> frozenset[str]:
    return frozenset(names)


def small(topic: int, theses: frozenset[str]) -> SmallCoherentTopic:
    return SmallCoherentTopic(
        topic=topic,
        size=len(theses),
        size_threshold=float(len(theses)),
        topics_in_map=8,
        npmi=0.3,
        npmi_floor=0.1,
        members=theses,
        program_counts=(("minas", len(theses)),),
    )


def declining(topic: int, theses: frozenset[str]) -> TemporalDecline:
    return TemporalDecline(
        topic=topic,
        early_years=(2021, 2023),
        late_years=(2024, 2026),
        early_count=len(theses),
        early_theses=50,
        early_share=len(theses) / 50,
        late_count=0,
        late_theses=50,
        late_share=0.0,
        decline=len(theses) / 50,
        min_count=3,
        min_decline=0.01,
        members=theses,
        program_counts=(("minas", len(theses)),),
    )


def transferable(topic: int, theses: frozenset[str], source: str, target: str) -> TransferableTopic:
    return TransferableTopic(
        topic=topic,
        source=source,
        target=target,
        affinity=1,
        min_affinity=1,
        source_count=len(theses),
        source_theses=10,
        source_share=len(theses) / 10,
        target_count=0,
        target_theses=10,
        target_share=0.0,
        present_min_count=3,
        present_min_share=0.2,
        rare_max_share=0.05,
        members=theses,
    )


def pair(
    first: frozenset[str], second: frozenset[str], topics: tuple[int, int] = (0, 1)
) -> InterstitialPair:
    return InterstitialPair(
        first_topic=topics[0],
        second_topic=topics[1],
        centroid_cosine=0.8,
        neighbor_rank=1,
        nearest_topics=3,
        bridge_documents=0,
        bridge_max_docs=1,
        bridges=frozenset(),
        first_members=first,
        second_members=second,
    )


GAP = members("a", "b", "c", "d")
OTHER = members("w", "x", "y", "z")


def support(
    runs: Sequence[Sequence[Gap]], *, min_runs: int = 4, min_overlap: float = 0.5
) -> tuple[GapSupport, ...]:
    return gap_support(runs, reference=0, min_runs=min_runs, min_overlap=min_overlap)


# --- The overlap -------------------------------------------------------------------------------


def test_jaccard_is_the_shared_share_of_the_union() -> None:
    assert jaccard(members("a", "b", "c"), members("a", "b", "d")) == 0.5
    assert jaccard(GAP, GAP) == 1.0
    assert jaccard(GAP, OTHER) == 0.0
    with pytest.raises(GapInputError, match="empty"):
        jaccard(frozenset(), frozenset())


def test_gaps_of_different_rules_or_programs_never_overlap() -> None:
    assert gap_overlap(small(0, GAP), small(5, GAP)) == 1.0
    assert gap_overlap(small(0, GAP), declining(0, GAP)) is None
    assert (
        gap_overlap(
            transferable(0, GAP, "minas", "industrial"), transferable(4, GAP, "minas", "industrial")
        )
        == 1.0
    )
    assert (
        gap_overlap(
            transferable(0, GAP, "minas", "industrial"), transferable(0, GAP, "minas", "sistemas")
        )
        is None
    )
    assert (
        gap_overlap(
            transferable(0, GAP, "minas", "industrial"), transferable(0, GAP, "industrial", "minas")
        )
        is None
    )


def test_a_pair_overlaps_by_its_weaker_topic_in_the_better_order() -> None:
    reference = pair(GAP, OTHER)
    swapped = pair(OTHER, GAP, topics=(2, 6))
    half = pair(members("a", "b", "c", "e"), members("w", "x", "q", "r"))

    assert gap_overlap(reference, swapped) == 1.0
    assert gap_overlap(reference, half) == pytest.approx(2 / 6)


# --- The filter --------------------------------------------------------------------------------


def test_a_gap_found_in_four_of_five_seeds_is_kept_and_in_three_is_dropped() -> None:
    four = [[small(0, GAP)], [small(3, GAP)], [small(1, GAP)], [small(7, GAP)], []]
    three = [[small(0, GAP)], [small(3, GAP)], [small(1, GAP)], [], [small(2, OTHER)]]

    (kept,) = support(four)
    (dropped,) = support(three)

    assert (kept.support, kept.stable) == (4, True)
    assert kept.overlaps == (1.0, 1.0, 1.0, 1.0, None)
    assert (dropped.support, dropped.stable) == (3, False)
    assert dropped.overlaps == (1.0, 1.0, 1.0, None, 0.0)
    assert stable_gaps(four, reference=0, min_runs=4, min_overlap=0.5) == (kept,)
    assert stable_gaps(three, reference=0, min_runs=4, min_overlap=0.5) == ()


def test_a_match_needs_an_overlap_at_or_above_the_threshold() -> None:
    at = members("a", "b", "e")  # Against a, b, c: 2 shared of 4, exactly 0.5.
    below = members("a", "f", "g")  # 1 shared of 5.
    reference = small(0, members("a", "b", "c"))
    runs = [[reference], [small(1, at)], [small(2, at)], [small(3, at)], [small(4, below)]]

    (result,) = support(runs, min_overlap=0.5)

    assert result.overlaps == (1.0, 0.5, 0.5, 0.5, 0.2)
    assert (result.support, result.stable) == (4, True)
    (stricter,) = support(runs, min_overlap=0.6)
    assert (stricter.support, stricter.stable) == (1, False)


def test_each_run_counts_once_with_its_best_match() -> None:
    reference = small(0, GAP)
    run = [small(1, members("a", "b", "x")), small(2, members("a", "b", "c", "d", "e"))]

    (result,) = support([[reference], run], min_runs=2)

    assert result.overlaps == (1.0, 0.8)
    assert result.support == 2


def test_pairs_and_programs_are_matched_with_their_own_keys() -> None:
    gaps = [pair(GAP, OTHER), transferable(3, GAP, "minas", "industrial")]
    runs = [
        gaps,
        [pair(OTHER, GAP, topics=(4, 9)), transferable(1, GAP, "minas", "industrial")],
        [pair(GAP, members("q")), transferable(1, GAP, "minas", "sistemas")],
    ]

    first, second = support(runs, min_runs=2)

    assert first.gap == gaps[0] and second.gap == gaps[1]
    assert first.overlaps == (1.0, 1.0, 0.0)
    assert second.overlaps == (1.0, 1.0, None)
    assert first.support == second.support == 2


def test_the_reference_runs_gaps_keep_their_order_and_evidence() -> None:
    reference = [small(0, GAP), declining(4, OTHER)]
    runs = [reference, [declining(1, OTHER), small(2, GAP)]]

    results = support(runs, min_runs=2)

    assert [result.gap for result in results] == reference
    assert all(result.min_runs == 2 and result.min_overlap == 0.5 for result in results)
    assert "rule (a)" in results[0].reason and "2 of 2 runs" in results[0].reason
    assert results[0].gap.reason in results[0].reason


def test_the_reference_can_be_any_run() -> None:
    runs = [[], [small(5, GAP)], [small(1, GAP)]]

    (result,) = gap_support(runs, reference=1, min_runs=2, min_overlap=0.5)

    assert result.gap == small(5, GAP)
    assert result.overlaps == (None, 1.0, 1.0)


def test_the_filter_refuses_invalid_arguments() -> None:
    runs = [[small(0, GAP)]] * 5

    with pytest.raises(GapInputError, match="run"):
        gap_support([], reference=0, min_runs=1, min_overlap=0.5)
    with pytest.raises(GapInputError, match="reference"):
        gap_support(runs, reference=5, min_runs=4, min_overlap=0.5)
    with pytest.raises(GapInputError, match="reference"):
        gap_support(runs, reference=-1, min_runs=4, min_overlap=0.5)
    with pytest.raises(GapInputError, match="min_runs"):
        gap_support(runs, reference=0, min_runs=6, min_overlap=0.5)
    with pytest.raises(GapInputError, match="min_runs"):
        gap_support(runs, reference=0, min_runs=0, min_overlap=0.5)
    with pytest.raises(GapInputError, match="min_overlap"):
        gap_support(runs, reference=0, min_runs=4, min_overlap=0.0)
    with pytest.raises(GapInputError, match="min_overlap"):
        gap_support(runs, reference=0, min_runs=4, min_overlap=1.5)


# --- With the rules, across seeds whose topic ids differ ---------------------------------------


def seed_map(
    partition: Sequence[Sequence[int]], order: Sequence[int], coherence: Sequence[float]
) -> GapMap:
    """A map of 40 theses where group ``partition[i]`` takes topic id ``order[i]`` and NPMI
    ``coherence[i]``; the theses of no group are noise."""
    labels = np.full(40, -1, dtype=np.int64)
    for group, topic in zip(partition, order, strict=True):
        labels[list(group)] = topic
    npmi_by_topic = dict(zip(order, coherence, strict=True))
    topics = tuple(sorted(order))
    return GapMap(
        labels=labels,
        topics=topics,
        centroids=np.eye(len(topics)),
        npmi=tuple(npmi_by_topic[topic] for topic in topics),
    )


def test_a_small_topic_renumbered_in_every_seed_is_kept_and_a_merged_one_is_dropped() -> None:
    # Sizes 3, 10, 10, 10 and 7: the bottom quartile holds the groups of 3 and 7, and only the
    # group of 3 passes the NPMI floor.
    small_group = range(0, 3)
    large = [range(3, 13), range(13, 23), range(23, 33), range(33, 40)]
    coherence = (0.4, 0.1, 0.1, 0.1, 0.1)
    metadata = ThesisMetadata(
        doc_codes=tuple(f"doc-{index:02d}" for index in range(40)),
        programs=("minas",) * 40,
        years=(2022,) * 40,
    )
    orders = [(0, 1, 2, 3, 4), (4, 0, 1, 2, 3), (2, 4, 0, 3, 1), (3, 2, 4, 1, 0), (1, 3, 2, 0, 4)]
    stable_maps = [seed_map([small_group, *large], order, coherence) for order in orders]
    # In the last two seeds the small group is absorbed by the first large group.
    merged = [range(0, 13), *large[1:]]
    unstable_maps = stable_maps[:3] + [
        seed_map(merged, order[:4], coherence[1:]) for order in orders[3:]
    ]

    def gaps_of(maps: Sequence[GapMap]) -> list[tuple[SmallCoherentTopic, ...]]:
        return [small_coherent_topics(each, metadata, npmi_floor=0.2) for each in maps]

    (kept,) = stable_gaps(gaps_of(stable_maps), reference=0, min_runs=4, min_overlap=0.5)
    (dropped,) = gap_support(gaps_of(unstable_maps), reference=0, min_runs=4, min_overlap=0.5)

    assert kept.support == 5
    assert [gaps[0].topic for gaps in gaps_of(stable_maps)] == [0, 4, 2, 3, 1]
    assert kept.gap.members == frozenset({"doc-00", "doc-01", "doc-02"})
    assert (dropped.support, dropped.stable) == (3, False)
    assert dropped.overlaps == (1.0, 1.0, 1.0, None, None)
