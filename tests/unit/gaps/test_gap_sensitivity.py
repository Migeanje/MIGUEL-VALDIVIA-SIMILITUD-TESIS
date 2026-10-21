"""The sensitivity table of the gap run (T24a, D32 (1)), reported and never used for selection.

The variants are the thesis plan's literal rule (a) (topics with fewer than 5 % of the corpus's
theses) and literal rule (b) (theses whose mean cosine distance to the run's topic centroids is
above the run's P75), rule (c) with ``rare_max_share`` 0.01 and the stability filter with
``stability_min_overlap`` 0.75, beside the declared rules as their baseline. The table is checked
on the synthetic experiment of ``conftest.py``, whose numbers are worked there by hand.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from thematic_redundancy.gaps.rules import GapInputError, GapMap, ThesisMetadata
from thematic_redundancy.gaps.sensitivity import (
    LITERAL_DISTANCE_QUANTILE,
    LITERAL_SMALL_PERCENT,
    MIN_OVERLAP,
    RARE_MAX_SHARE,
    VARIANTS,
    literal_distant_theses,
    literal_small_cut,
    literal_small_topics,
    sensitivity_rows,
)
from thematic_redundancy.gaps.stability import gap_support


def metadata(count: int) -> ThesisMetadata:
    return ThesisMetadata(
        doc_codes=tuple(f"D{index:03d}" for index in range(count)),
        programs=("sistemas",) * count,
        years=(2022,) * count,
    )


def test_the_values_are_those_of_d32() -> None:
    assert (LITERAL_SMALL_PERCENT, LITERAL_DISTANCE_QUANTILE) == (5, 0.75)
    assert (RARE_MAX_SHARE, MIN_OVERLAP) == (0.01, 0.75)
    assert VARIANTS == ("primary", "literal_a", "literal_b", "rare_max_share", "min_overlap")


@pytest.mark.parametrize(
    ("theses", "cut"), [(744, 37), (740, 36), (741, 37), (20, 0), (21, 1), (100, 4)]
)
def test_the_literal_size_cut_is_the_largest_size_below_5_percent(theses: int, cut: int) -> None:
    # 744 theses: 5 % is 37.2, so 37 is below it; 740: 5 % is exactly 37, which is not below.
    assert literal_small_cut(theses) == cut


def test_literal_a_counts_every_topic_below_5_percent_whatever_its_npmi() -> None:
    # 100 theses: topic 0 has 4 (below 5), topic 1 has 5 (exactly 5 %), topic 2 the other 91.
    labels = np.array([0] * 4 + [1] * 5 + [2] * 91)
    gap_map = GapMap(labels=labels, topics=(0, 1, 2), centroids=np.eye(3), npmi=(-0.9, 0.9, 0.5))

    found = literal_small_topics(gap_map, metadata(100))

    assert [(gap.topic, gap.size, gap.size_threshold) for gap in found] == [(0, 4, 4.0)]
    assert found[0].npmi == -0.9


def test_literal_b_flags_the_theses_above_the_p75_of_the_mean_distance() -> None:
    # Centroids on x and y. Seven theses lie on an axis, mean distance exactly 0.5; one lies
    # between the axes (0.293); one, of topic 0, points away from x (1.5); and one noise thesis
    # points away from both (1.707). The P75 of the ten means is 0.5 itself, and only the means
    # strictly above it count, so the seven ties stay out.
    half = np.sqrt(0.5)
    vectors = np.array(
        [[1, 0]] * 4 + [[0, 1]] * 3 + [[half, half], [-1, 0], [-half, -half]], dtype=float
    )
    labels = np.array([0] * 4 + [1] * 3 + [0, 0, -1])
    gap_map = GapMap(labels=labels, topics=(0, 1), centroids=np.eye(2), npmi=(0.1, 0.1))
    data = metadata(10)

    found = literal_distant_theses(gap_map, vectors, data)

    assert found.flagged == frozenset({"D008", "D009"})
    assert found.noise == 1
    assert found.threshold == pytest.approx(0.5)


def test_literal_b_needs_a_topic_and_matching_vectors() -> None:
    gap_map = GapMap(labels=np.array([-1, -1]), topics=(), centroids=np.zeros((0, 2)), npmi=())
    with pytest.raises(GapInputError):
        literal_distant_theses(gap_map, np.eye(2), metadata(2))
    one = GapMap(labels=np.array([0, 0]), topics=(0,), centroids=np.ones((1, 2)), npmi=(0.1,))
    with pytest.raises(GapInputError):
        literal_distant_theses(one, np.eye(3)[:2], metadata(2))


# The table on the synthetic experiment


@pytest.fixture(scope="module")
def table(scenario: SimpleNamespace, kit: SimpleNamespace) -> dict[tuple[str, str], list]:
    rows = sensitivity_rows(
        kit.maps,
        scenario.vectors,
        scenario.metadata,
        kit.affinity,
        kit.settings,
        run_ids=kit.runs,
        seeds=kit.seeds,
        reference=3,
        primary=kit.gaps,
    )
    grouped: dict[tuple[str, str], list] = {}
    for row in rows:
        grouped.setdefault((row.variant, row.rule), []).append(row)
    return grouped


def test_the_table_has_one_row_per_variant_rule_and_run_in_order(
    table: dict[tuple[str, str], list], kit: SimpleNamespace
) -> None:
    assert list(table) == [
        *(("primary", rule) for rule in "abcd"),
        ("literal_a", "a"),
        ("literal_b", "b"),
        ("rare_max_share", "c"),
        *(("min_overlap", rule) for rule in "abcd"),
    ]
    for rows in table.values():
        assert [row.run_id for row in rows] == list(kit.runs)
        assert [row.seed for row in rows] == list(kit.seeds)
        assert [row.reference for row in rows] == [False, False, False, True, False]


def test_the_primary_rows_count_the_declared_gaps_and_the_stable_ones(
    table: dict[tuple[str, str], list], kit: SimpleNamespace
) -> None:
    for rule in "abcd":
        counts = [sum(gap.rule == rule for gap in run) for run in kit.gaps]
        assert [row.gaps for row in table["primary", rule]] == counts
    assert [row.gaps for row in table["primary", "a"]] == [2, 2, 2, 2, 2]
    assert [row.gaps for row in table["primary", "c"]] == [3, 3, 3, 2, 3]
    assert [row.gaps for row in table["primary", "d"]] == [1, 1, 1, 1, 0]
    supports = gap_support(kit.gaps, reference=3, min_runs=4, min_overlap=0.5)
    for rule in "abcd":
        rows = table["primary", rule]
        expected = sum(s.stable for s in supports if s.gap.rule == rule)
        assert [row.stable for row in rows] == [None, None, None, expected, None]
    assert [table["primary", rule][3].stable for rule in "acd"] == [1, 1, 1]
    # C's 9 theses, then E's 6, then D's 7 with C's.
    assert [row.theses for row in table["primary", "a"]] == [15, 15, 15, 16, 13]
    others = [row for rows in table.values() for row in rows if row.variant != "literal_b"]
    assert {row.noise_theses for row in others} == {None}


def test_literal_a_flags_the_topics_below_5_percent_of_the_148_theses(
    table: dict[tuple[str, str], list],
) -> None:
    rows = table["literal_a", "a"]
    # At most 7 theses: E (6) in seeds 7, 13 and 21, D (7) in seed 42, both in seed 73. C (9)
    # is a primary gap and not a literal one; D is in 2 runs only, so none is stable.
    assert [row.gaps for row in rows] == [1, 1, 1, 1, 2]
    assert [row.theses for row in rows] == [6, 6, 6, 7, 13]
    assert rows[3].stable == 0
    assert "at most 7" in rows[0].setting


def test_literal_b_counts_the_flagged_theses_and_their_overlap_with_the_reference(
    table: dict[tuple[str, str], list], scenario: SimpleNamespace, kit: SimpleNamespace
) -> None:
    flagged = []
    for gap_map in kit.maps:
        units = scenario.vectors / np.linalg.norm(scenario.vectors, axis=1, keepdims=True)
        cents = gap_map.centroids / np.linalg.norm(gap_map.centroids, axis=1, keepdims=True)
        mean = (1 - units @ cents.T).mean(axis=1)
        flagged.append(mean > np.quantile(mean, 0.75))
    rows = table["literal_b", "b"]

    assert [row.theses for row in rows] == [int(mask.sum()) for mask in flagged]
    assert [row.noise_theses for row in rows] == [
        int((mask & (gap_map.labels == -1)).sum())
        for mask, gap_map in zip(flagged, kit.maps, strict=True)
    ]
    reference = flagged[3]
    assert [row.jaccard_to_reference for row in rows] == [
        pytest.approx((mask & reference).sum() / (mask | reference).sum()) for mask in flagged
    ]
    assert rows[3].jaccard_to_reference == 1.0
    assert {row.gaps for row in rows} == {None}
    assert {row.stable for row in rows} == {None}


def test_a_rare_share_of_001_adds_cs_one_industrial_thesis_as_absent(
    table: dict[tuple[str, str], list],
) -> None:
    rows = table["rare_max_share", "c"]
    # C's one Industrial thesis is 1 of 103 (0.0097), rare at 0.01 and not absent at 0.
    assert [row.gaps for row in rows] == [4, 4, 4, 3, 3]
    assert rows[3].stable == 2
    assert table["primary", "c"][3].stable == 1
    assert "0.01" in rows[0].setting


def test_an_overlap_of_075_keeps_the_gaps_and_filters_them_more_strictly(
    table: dict[tuple[str, str], list], kit: SimpleNamespace
) -> None:
    supports = gap_support(kit.gaps, reference=3, min_runs=4, min_overlap=MIN_OVERLAP)
    for rule in "abcd":
        rows = table["min_overlap", rule]
        assert [row.gaps for row in rows] == [row.gaps for row in table["primary", rule]]
        assert rows[3].stable == sum(s.stable for s in supports if s.gap.rule == rule)
        assert rows[3].stable <= table["primary", rule][3].stable
    assert "0.75" in table["min_overlap", "a"][0].setting
