"""Gap rules (a) to (d) over one topic map (T24, D16), on hand-built synthetic maps.

Each rule is shown firing and not firing at the boundaries that the plan declares: the bottom
quartile of topic size and the NPMI floor (a), both cut once over every seed run (D32), each
topic's 3 nearest topics and at most one
bridge thesis (b), the affinity matrix of U4 with D29 and the per-program shares of R5 (c), and
the share change between 2021–2023 and 2024–2026 with a minimum count of 3 (d). No thesis is
read: the theses are doc codes, program keys, years and, for rule (b), points on a circle.
"""

import inspect
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from thematic_redundancy.evaluation.coherence import CoherenceResult, TopicCoherence
from thematic_redundancy.gaps.affinity import CareerAffinity, load_career_affinity
from thematic_redundancy.gaps.rules import (
    GapInputError,
    GapMap,
    GapSettings,
    InterstitialPair,
    SmallCoherentTopic,
    SmallTopicCuts,
    TemporalDecline,
    ThesisMetadata,
    TransferableTopic,
    interstitial_pairs,
    small_coherent_topics,
    small_topic_cuts,
    temporal_declines,
    transferable_topics,
)
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map.build import TopicMap
from thematic_redundancy.topic_map.ctfidf import Topic

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AFFINITY_PATH = PROJECT_ROOT / "config" / "career_affinity.yaml"
PROGRAMS = ("sistemas", "industrial", "electronica", "mecanica", "minas")


@pytest.fixture(scope="module")
def affinity() -> CareerAffinity:
    return load_career_affinity(AFFINITY_PATH, programs=PROGRAMS)


def make_map(
    labels: Sequence[int],
    *,
    npmi: Sequence[float | None] | None = None,
    centroids: np.ndarray | None = None,
) -> GapMap:
    """A map whose topics are the labels present, with one axis per topic as its centroid."""
    label_array = np.asarray(labels, dtype=np.int64)
    topics = tuple(sorted({int(label) for label in label_array if label != -1}))
    if centroids is None:
        centroids = np.eye(max(len(topics), 1))[: len(topics)]
    if npmi is None:
        npmi = [0.5] * len(topics)
    return GapMap(labels=label_array, topics=topics, centroids=centroids, npmi=tuple(npmi))


def make_metadata(programs: Sequence[str], years: Sequence[int] | None = None) -> ThesisMetadata:
    count = len(programs)
    return ThesisMetadata(
        doc_codes=tuple(f"doc-{index:04d}" for index in range(count)),
        programs=tuple(programs),
        years=tuple(years) if years is not None else (2022,) * count,
    )


def labels_by_size(sizes: Sequence[int]) -> list[int]:
    """Labels where topic ``i`` has ``sizes[i]`` theses, in topic order."""
    return [topic for topic, size in enumerate(sizes) for _ in range(size)]


def codes(indices: Sequence[int] | range) -> frozenset[str]:
    return frozenset(f"doc-{index:04d}" for index in indices)


def on_circle(degrees: Sequence[float]) -> np.ndarray:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return np.column_stack([np.cos(radians), np.sin(radians)])


# --- Inputs ---------------------------------------------------------------------------------


def test_the_map_freezes_its_arrays_and_checks_its_topics() -> None:
    gap_map = make_map([0, 0, 1, -1])

    assert gap_map.topics == (0, 1)
    assert not gap_map.labels.flags.writeable
    assert not gap_map.centroids.flags.writeable
    with pytest.raises(GapInputError, match="topics"):
        GapMap(labels=np.array([0, 1, 2]), topics=(0, 1), centroids=np.eye(2), npmi=(0.1, 0.1))
    with pytest.raises(GapInputError, match="centroid"):
        GapMap(labels=np.array([0, 1]), topics=(0, 1), centroids=np.eye(3), npmi=(0.1, 0.1))
    with pytest.raises(GapInputError, match="NPMI"):
        GapMap(labels=np.array([0, 1]), topics=(0, 1), centroids=np.eye(2), npmi=(0.1,))
    with pytest.raises(GapInputError, match="NPMI"):
        GapMap(labels=np.array([0, 1]), topics=(0, 1), centroids=np.eye(2), npmi=(0.1, 1.5))
    with pytest.raises(GapInputError, match="centroid"):
        GapMap(
            labels=np.array([0, 1]),
            topics=(0, 1),
            centroids=np.array([[1.0, 0.0], [0.0, 0.0]]),
            npmi=(0.1, 0.1),
        )
    with pytest.raises(ValueError, match="labels"):
        GapMap(labels=np.array([0.5, 1.0]), topics=(0, 1), centroids=np.eye(2), npmi=(0.1, 0.1))


def test_the_map_is_read_from_a_topic_map_and_its_coherence_in_topic_order() -> None:
    labels = np.array([3, 3, 7, 7, 7, -1], dtype=np.int64)
    topics = (
        Topic(topic=3, size=2, terms=("mina", "suelo"), scores=(0.4, 0.3)),
        Topic(topic=7, size=3, terms=("red", "datos"), scores=(0.5, 0.2)),
    )
    topic_map = TopicMap(
        seed=7,
        clustering=None,
        labels=labels,
        topics=topics,
        noise=None,
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
        coordinates=np.zeros((6, 2)),
        reduction=None,
    )
    coherence = CoherenceResult(
        unit="document",
        window_size=None,
        top_n=2,
        units=6,
        topics=(
            TopicCoherence(("mina", "suelo"), (0.25,), 0.25, (), None),
            TopicCoherence(("red", "datos"), (None,), None, ("datos",), "absent_terms"),
        ),
        mean=None,
        undefined_reason="undefined_topics",
    )

    gap_map = GapMap.from_topic_map(topic_map, coherence)

    assert gap_map.topics == (3, 7)
    assert gap_map.npmi == (0.25, None)
    np.testing.assert_array_equal(gap_map.labels, labels)
    swapped = replace(coherence, topics=tuple(reversed(coherence.topics)))
    with pytest.raises(GapInputError, match="terms"):
        GapMap.from_topic_map(topic_map, swapped)


def test_the_metadata_must_be_consistent() -> None:
    with pytest.raises(GapInputError, match="lengths"):
        ThesisMetadata(doc_codes=("a", "b"), programs=("minas",), years=(2021, 2022))
    with pytest.raises(GapInputError, match="unique"):
        ThesisMetadata(doc_codes=("a", "a"), programs=("minas",) * 2, years=(2021, 2022))
    with pytest.raises(GapInputError, match="year"):
        ThesisMetadata(doc_codes=("a", "b"), programs=("minas",) * 2, years=(2021, True))
    with pytest.raises(GapInputError, match="doc code"):
        ThesisMetadata(doc_codes=("a", ""), programs=("minas",) * 2, years=(2021, 2022))


def test_every_rule_refuses_metadata_of_another_length() -> None:
    gap_map = make_map([0, 0, 1, 1])
    metadata = make_metadata(["minas"] * 3)

    with pytest.raises(GapInputError, match="theses"):
        small_coherent_topics(gap_map, metadata, size_cut=2.0, npmi_floor=0.0)
    with pytest.raises(GapInputError, match="theses"):
        temporal_declines(
            gap_map,
            metadata,
            first_year=2021,
            split_year=2024,
            last_year=2026,
            min_count=3,
            min_decline=0.1,
        )


@pytest.mark.parametrize(
    ("rule", "undeclared"),
    [
        (small_topic_cuts, {"npmi_floor_quantile"}),
        (small_coherent_topics, {"size_cut", "npmi_floor"}),
        (interstitial_pairs, {"nearest_topics", "bridge_max_docs"}),
        (
            transferable_topics,
            {"min_affinity", "present_min_count", "present_min_share", "rare_max_share"},
        ),
        (
            temporal_declines,
            {"first_year", "split_year", "last_year", "min_count", "min_decline"},
        ),
    ],
)
def test_no_threshold_has_a_hidden_default(rule, undeclared: set[str]) -> None:
    parameters = inspect.signature(rule).parameters

    for name in undeclared:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


def test_the_declared_values_come_from_the_configuration() -> None:
    settings = GapSettings.from_config(load_config(PROJECT_ROOT / "config" / "default.yaml"))

    assert settings == GapSettings(
        nearest_topics=3,
        bridge_max_docs=1,
        temporal_first_year=2021,
        temporal_split_year=2024,
        temporal_last_year=2026,
        temporal_min_count=3,
        stability_min_runs=4,
        npmi_floor_quantile=0.25,
        min_affinity=1,
        present_min_count=5,
        present_min_share=0.05,
        rare_max_share=0.0,
        min_decline=0.025,
        stability_min_overlap=0.5,
    )


# --- Rule (a): the cuts, once over every seed run (D32) ---------------------------------------


def test_the_size_cut_is_the_first_quartile_of_every_runs_topic_sizes() -> None:
    first = make_map(labels_by_size([10, 2, 8]))
    second = make_map(labels_by_size([4, 6]))  # Q1 of 2, 4, 6, 8, 10 is exactly 4.

    cuts = small_topic_cuts([first, second], npmi_floor_quantile=0.25)

    assert cuts.size_cut == 4.0
    assert (cuts.runs, cuts.topics) == (2, 5)


def test_the_size_cut_interpolates_between_sizes() -> None:
    labels = labels_by_size([16, 2, 14, 4, 12, 6, 10, 8])  # Q1 is 4 + 0.75 * (6 - 4) = 5.5.

    assert small_topic_cuts([make_map(labels)], npmi_floor_quantile=0.25).size_cut == 5.5


def test_the_size_cut_leaves_the_noise_out_of_the_topic_sizes() -> None:
    labels = [-1, *labels_by_size([2, 4, 6, 8, 10])]  # With the noise as a size, Q1 is 2.5.

    assert small_topic_cuts([make_map(labels)], npmi_floor_quantile=0.25).size_cut == 4.0


def test_the_npmi_floor_is_a_quantile_of_every_runs_defined_npmi() -> None:
    first = make_map(labels_by_size([3, 3, 3]), npmi=[0.4, None, 0.1])
    second = make_map(labels_by_size([3, 3]), npmi=[0.3, 0.2])

    cuts = small_topic_cuts([first, second], npmi_floor_quantile=0.25)

    # The defined values are 0.1, 0.2, 0.3 and 0.4: Q1 is 0.1 + 0.75 * 0.1 = 0.175.
    assert cuts.npmi_floor == pytest.approx(0.175, abs=1e-15)
    assert (cuts.npmi_floor_quantile, cuts.npmi_topics) == (0.25, 4)
    assert small_topic_cuts([first, second], npmi_floor_quantile=0.0).npmi_floor == 0.1
    assert small_topic_cuts([first, second], npmi_floor_quantile=1.0).npmi_floor == 0.4


def test_the_cuts_report_their_evidence() -> None:
    cuts = small_topic_cuts(
        [make_map(labels_by_size([2, 4]), npmi=[0.1, 0.3])], npmi_floor_quantile=0.5
    )

    assert cuts == SmallTopicCuts(
        size_cut=2.5,
        npmi_floor=pytest.approx(0.2, abs=1e-15),
        npmi_floor_quantile=0.5,
        runs=1,
        topics=2,
        npmi_topics=2,
    )


@pytest.mark.parametrize(
    ("maps", "quantile", "message"),
    [
        ([], 0.25, "no run"),
        ([make_map([-1, -1])], 0.25, "no topic"),
        ([make_map([0, 1], npmi=[None, None])], 0.25, "no defined NPMI"),
        ([make_map([0, 1])], -0.1, "quantile"),
        ([make_map([0, 1])], 1.1, "quantile"),
        ([make_map([0, 1])], float("nan"), "quantile"),
    ],
)
def test_the_cuts_refuse_inputs_without_a_value(maps, quantile: float, message: str) -> None:
    with pytest.raises(GapInputError, match=message):
        small_topic_cuts(maps, npmi_floor_quantile=quantile)


def test_one_cut_keeps_a_topic_small_in_every_run_whatever_each_maps_own_quartile() -> None:
    # Topic 0 has 16 theses in two runs whose own first quartiles are 15.0 (sizes 12, 16, 17,
    # 30) and 17.5 (16, 18, 20, 40): a quartile per map calls it small in the second run only.
    # The one cut over both runs, Q1 of all eight sizes, is 16.0 and calls it small in both.
    first = make_map(labels_by_size([16, 12, 17, 30]))
    second = make_map(labels_by_size([16, 18, 20, 40]))
    cuts = small_topic_cuts([first, second], npmi_floor_quantile=0.0)

    found = [
        [
            gap.topic
            for gap in small_coherent_topics(
                each,
                make_metadata(["minas"] * len(each.labels)),
                size_cut=cuts.size_cut,
                npmi_floor=0.0,
            )
        ]
        for each in (first, second)
    ]

    assert cuts.size_cut == 16.0
    assert found == [[0, 1], [0]]


# --- Rule (a): small coherent topics ----------------------------------------------------------


def test_rule_a_keeps_the_topics_at_or_below_the_size_cut() -> None:
    labels = labels_by_size([10, 2, 8, 4, 6])

    gaps = small_coherent_topics(
        make_map(labels), make_metadata(["minas"] * len(labels)), size_cut=4.0, npmi_floor=0.1
    )

    assert [gap.topic for gap in gaps] == [1, 3]
    assert [gap.size for gap in gaps] == [2, 4]
    assert all(gap.size_threshold == 4.0 and gap.topics_in_map == 5 for gap in gaps)


def test_rule_a_needs_the_npmi_at_or_above_the_floor() -> None:
    labels = labels_by_size([2, 2, 2, 9, 9, 9, 9, 9])  # Topics 0, 1 and 2 are small.
    npmi = [0.05, 0.1, None, 0.9, 0.9, 0.9, 0.9, 0.9]

    gaps = small_coherent_topics(
        make_map(labels, npmi=npmi),
        make_metadata(["minas"] * len(labels)),
        size_cut=2.0,
        npmi_floor=0.1,
    )

    assert [gap.topic for gap in gaps] == [1]
    assert (gaps[0].npmi, gaps[0].npmi_floor) == (0.1, 0.1)


def test_rule_a_reports_its_evidence() -> None:
    labels = labels_by_size([3, 9, 9, 9])
    programs = ["minas", "sistemas", "minas", *["industrial"] * 27]

    (gap,) = small_coherent_topics(
        make_map(labels, npmi=[0.21, 0.3, 0.3, 0.3]),
        make_metadata(programs),
        size_cut=7.5,
        npmi_floor=0.2,
    )

    assert gap == SmallCoherentTopic(
        topic=0,
        size=3,
        size_threshold=7.5,
        topics_in_map=4,
        npmi=0.21,
        npmi_floor=0.2,
        members=codes(range(3)),
        program_counts=(("minas", 2), ("sistemas", 1)),
    )
    assert gap.rule == "a"
    assert gap.topic_members == (codes(range(3)),)
    assert gap.match_key == ()
    assert "topic 0" in gap.reason and "7.5" in gap.reason and "0.2100" in gap.reason


def test_rule_a_finds_nothing_on_a_map_without_topics() -> None:
    gaps = small_coherent_topics(
        make_map([-1, -1]), make_metadata(["minas"] * 2), size_cut=1.0, npmi_floor=0
    )

    assert gaps == ()


@pytest.mark.parametrize("floor", [float("nan"), -1.5, 1.5])
def test_rule_a_refuses_a_floor_outside_the_npmi_range(floor: float) -> None:
    with pytest.raises(GapInputError, match="floor"):
        small_coherent_topics(
            make_map([0, 1]), make_metadata(["minas"] * 2), size_cut=1.0, npmi_floor=floor
        )


@pytest.mark.parametrize("cut", [float("nan"), float("inf"), -1.0])
def test_rule_a_refuses_a_size_cut_that_is_not_a_finite_size(cut: float) -> None:
    with pytest.raises(GapInputError, match="size cut"):
        small_coherent_topics(
            make_map([0, 1]), make_metadata(["minas"] * 2), size_cut=cut, npmi_floor=0.0
        )


# --- Rule (b): interstitial pairs ------------------------------------------------------------

BRIDGE_CENTROIDS = [0.0, 30.0, 90.0, 120.0]
"""Four topics on a circle, so each topic's 3 nearest topics are the other three."""


def bridge_layout() -> tuple[GapMap, np.ndarray, ThesisMetadata]:
    """Three members at each centroid, two noise theses midway between topics 0 and 1, one
    noise thesis midway between topics 2 and 3, and a member of topic 2 at 95 degrees, whose two
    nearest topics are 2 and 3 but which lies outside the middle half of their arc."""
    angles = [*[0.0] * 3, *[30.0] * 3, *[90.0] * 3, *[120.0] * 3, 15.0, 15.0, 105.0, 95.0]
    labels = [*[0] * 3, *[1] * 3, *[2] * 3, *[3] * 3, -1, -1, -1, 2]
    gap_map = make_map(labels, centroids=on_circle(BRIDGE_CENTROIDS))
    return gap_map, on_circle(angles), make_metadata(["minas"] * len(labels))


def test_rule_b_keeps_the_nearby_pairs_with_at_most_one_bridge_thesis() -> None:
    gap_map, vectors, metadata = bridge_layout()

    gaps = interstitial_pairs(gap_map, vectors, metadata, nearest_topics=3, bridge_max_docs=1)

    assert [(gap.first_topic, gap.second_topic) for gap in gaps] == [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    by_pair = {(gap.first_topic, gap.second_topic): gap for gap in gaps}
    assert by_pair[2, 3].bridge_documents == 1
    assert by_pair[2, 3].bridges == codes([14])
    assert by_pair[1, 2].bridge_documents == 0


def test_rule_b_counts_the_bridges_of_a_pair_that_has_too_many() -> None:
    gap_map, vectors, metadata = bridge_layout()

    gaps = interstitial_pairs(gap_map, vectors, metadata, nearest_topics=3, bridge_max_docs=2)

    first = gaps[0]
    assert (first.first_topic, first.second_topic) == (0, 1)
    assert first.bridge_documents == 2
    assert first.bridges == codes([12, 13])


def test_rule_b_reports_its_evidence() -> None:
    gap_map, vectors, metadata = bridge_layout()

    gaps = interstitial_pairs(gap_map, vectors, metadata, nearest_topics=3, bridge_max_docs=1)

    gap = next(gap for gap in gaps if (gap.first_topic, gap.second_topic) == (2, 3))
    assert isinstance(gap, InterstitialPair)
    assert gap.centroid_cosine == pytest.approx(np.cos(np.deg2rad(30.0)))
    assert gap.neighbor_rank == 1
    assert (gap.nearest_topics, gap.bridge_max_docs) == (3, 1)
    assert gap.first_members == codes([6, 7, 8, 15])
    assert gap.second_members == codes([9, 10, 11])
    assert gap.topic_members == (gap.first_members, gap.second_members)
    assert gap.rule == "b" and gap.match_key == ()
    assert "topics 2 and 3" in gap.reason and "1 bridge" in gap.reason


def test_rule_b_reads_only_each_topics_nearest_topics() -> None:
    # Topic 4, at 200 degrees, is among the 3 nearest topics of 3, 2 and 0, never of 1, and
    # 1 is not among its own 3 nearest; every member sits on its centroid, so no thesis bridges.
    centroid_angles = [0.0, 30.0, 60.0, 90.0, 200.0]
    labels = labels_by_size([2] * 5)
    gap_map = make_map(labels, centroids=on_circle(centroid_angles))
    vectors = on_circle([angle for angle in centroid_angles for _ in range(2)])

    gaps = interstitial_pairs(
        gap_map, vectors, make_metadata(["minas"] * 10), nearest_topics=3, bridge_max_docs=1
    )

    pairs = [(gap.first_topic, gap.second_topic) for gap in gaps]
    assert pairs == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    assert (
        next(gap for gap in gaps if gap.second_topic == 4 and gap.first_topic == 0).neighbor_rank
        == 3
    )


def test_rule_b_finds_nothing_with_fewer_than_two_topics() -> None:
    gap_map = make_map([0, 0, -1], centroids=on_circle([10.0]))

    gaps = interstitial_pairs(
        gap_map,
        on_circle([10.0, 10.0, 50.0]),
        make_metadata(["minas"] * 3),
        nearest_topics=3,
        bridge_max_docs=1,
    )

    assert gaps == ()


def test_rule_b_refuses_invalid_inputs() -> None:
    gap_map, vectors, metadata = bridge_layout()

    def run(vectors: np.ndarray, nearest: int = 3, bridges: int = 1) -> object:
        return interstitial_pairs(
            gap_map, vectors, metadata, nearest_topics=nearest, bridge_max_docs=bridges
        )

    with pytest.raises(GapInputError, match="nearest_topics"):
        run(vectors, nearest=0)
    with pytest.raises(GapInputError, match="bridge_max_docs"):
        run(vectors, bridges=-1)
    with pytest.raises(GapInputError, match="vectors"):
        run(vectors[:-1])
    with pytest.raises(GapInputError, match="dimensions"):
        run(np.hstack([vectors, vectors]))
    zero = vectors.copy()
    zero[0] = 0.0
    with pytest.raises(GapInputError, match="zero"):
        run(zero)


# --- Rule (c): transferable topics -----------------------------------------------------------


def transfer(
    gap_map: GapMap,
    metadata: ThesisMetadata,
    affinity: CareerAffinity,
    *,
    min_affinity: int = 1,
    present_min_count: int = 3,
    present_min_share: float = 0.2,
    rare_max_share: float = 0.05,
) -> list[tuple[int, str, str]]:
    gaps = transferable_topics(
        gap_map,
        metadata,
        affinity,
        min_affinity=min_affinity,
        present_min_count=present_min_count,
        present_min_share=present_min_share,
        rare_max_share=rare_max_share,
    )
    return [(gap.topic, gap.source, gap.target) for gap in gaps]


def test_rule_c_compares_shares_of_each_programs_theses_not_counts(
    affinity: CareerAffinity,
) -> None:
    # Industrial has 100 theses and Mecánica 20 (affinity 2). Topic 0 holds 6 Mecánica theses
    # (30%) and 4 Industrial ones (4%): a transferable gap from Mecánica to Industrial, although
    # Industrial has more theses on it. 5 Mecánica theses are noise and still count in its 20.
    programs = ["mecanica"] * 20 + ["industrial"] * 100
    labels = [*[0] * 6, *[1] * 9, *[-1] * 5, *[0] * 4, *[1] * 96]

    gaps = transferable_topics(
        make_map(labels),
        make_metadata(programs),
        affinity,
        min_affinity=1,
        present_min_count=3,
        present_min_share=0.2,
        rare_max_share=0.05,
    )

    assert gaps == (
        TransferableTopic(
            topic=0,
            source="mecanica",
            target="industrial",
            affinity=2,
            min_affinity=1,
            source_count=6,
            source_theses=20,
            source_share=0.3,
            target_count=4,
            target_theses=100,
            target_share=0.04,
            present_min_count=3,
            present_min_share=0.2,
            rare_max_share=0.05,
            members=codes([*range(6), *range(20, 24)]),
        ),
    )
    assert gaps[0].rule == "c"
    assert gaps[0].match_key == ("mecanica", "industrial")
    assert "mecanica" in gaps[0].reason and "industrial" in gaps[0].reason


def test_rule_c_reads_a_small_programs_few_theses_as_a_large_share(
    affinity: CareerAffinity,
) -> None:
    # Minas has 10 theses: its 1 thesis on topic 0 is 10% of them, so the topic is not rare
    # there under a 5% cap, and it is rare under a 10% cap, the boundary included.
    programs = ["industrial"] * 100 + ["minas"] * 10
    labels = [*[0] * 30, *[1] * 70, 0, *[1] * 9]
    gap_map, metadata = make_map(labels), make_metadata(programs)

    assert transfer(gap_map, metadata, affinity, rare_max_share=0.05) == []
    assert transfer(gap_map, metadata, affinity, rare_max_share=0.1) == [(0, "industrial", "minas")]


def test_rule_c_never_targets_electronica_as_decided_in_d29(affinity: CareerAffinity) -> None:
    # Electrónica and Mecánica have affinity 2. Topic 0 is Electrónica's alone and topic 1 is
    # Mecánica's alone: Electrónica may be the source, never the target.
    programs = ["electronica"] * 20 + ["mecanica"] * 20
    labels = [*[0] * 10, *[2] * 10, *[1] * 10, *[2] * 10]

    found = transfer(make_map(labels), make_metadata(programs), affinity, rare_max_share=0.0)

    assert found == [(0, "electronica", "mecanica")]


def test_rule_c_needs_an_affinity_at_or_above_the_threshold(affinity: CareerAffinity) -> None:
    # Topic 0 is Sistemas' alone; Sistemas has affinity 1 with Industrial and 0 with Minas.
    programs = ["sistemas"] * 20 + ["industrial"] * 20 + ["minas"] * 20
    labels = [*[0] * 10, *[1] * 50]
    gap_map, metadata = make_map(labels), make_metadata(programs)

    assert transfer(gap_map, metadata, affinity, min_affinity=1) == [(0, "sistemas", "industrial")]
    assert transfer(gap_map, metadata, affinity, min_affinity=2) == []


def test_rule_c_needs_the_minimum_count_in_the_source(affinity: CareerAffinity) -> None:
    # Minas has 6 theses: 2 on topic 0 (33%) is under the minimum count of 3, and 3 meet it.
    programs = ["minas"] * 6 + ["industrial"] * 20
    two = [0, 0, *[1] * 24]
    three = [0, 0, 0, *[1] * 23]
    metadata = make_metadata(programs)

    assert transfer(make_map(two), metadata, affinity) == []
    assert transfer(make_map(three), metadata, affinity) == [(0, "minas", "industrial")]


def test_rule_c_needs_the_minimum_share_in_the_source(affinity: CareerAffinity) -> None:
    # Industrial has 20 theses: 4 on topic 0 is exactly the 20% floor, 3 is under it.
    programs = ["industrial"] * 20 + ["minas"] * 20
    metadata = make_metadata(programs)

    at_floor = make_map([*[0] * 4, *[1] * 36])
    under = make_map([*[0] * 3, *[1] * 37])

    assert transfer(at_floor, metadata, affinity) == [(0, "industrial", "minas")]
    assert transfer(under, metadata, affinity) == []


def test_rule_c_lists_every_source_and_target_in_program_order(affinity: CareerAffinity) -> None:
    programs = ["sistemas"] * 10 + ["industrial"] * 10 + ["mecanica"] * 10 + ["minas"] * 10
    # Topic 0: Sistemas and Mecánica; topic 1: Industrial; topic 2: the rest.
    labels = [*[0] * 5, *[2] * 5, *[1] * 5, *[2] * 5, *[0] * 5, *[2] * 5, *[2] * 10]

    found = transfer(make_map(labels), make_metadata(programs), affinity)

    assert found == [
        (0, "sistemas", "industrial"),
        (0, "mecanica", "industrial"),
        (0, "mecanica", "minas"),
        (1, "industrial", "sistemas"),
        (1, "industrial", "mecanica"),
        (1, "industrial", "minas"),
    ]


def test_rule_c_refuses_invalid_thresholds_and_unknown_programs(affinity: CareerAffinity) -> None:
    gap_map = make_map([0, 1])
    metadata = make_metadata(["minas", "industrial"])

    with pytest.raises(GapInputError, match="min_affinity"):
        transfer(gap_map, metadata, affinity, min_affinity=0)
    with pytest.raises(GapInputError, match="min_affinity"):
        transfer(gap_map, metadata, affinity, min_affinity=3)
    with pytest.raises(GapInputError, match="present_min_count"):
        transfer(gap_map, metadata, affinity, present_min_count=0)
    with pytest.raises(GapInputError, match="present_min_share"):
        transfer(gap_map, metadata, affinity, present_min_share=0.0)
    with pytest.raises(GapInputError, match="rare_max_share"):
        transfer(gap_map, metadata, affinity, rare_max_share=-0.1)
    with pytest.raises(GapInputError, match="rare_max_share"):
        transfer(gap_map, metadata, affinity, present_min_share=0.2, rare_max_share=0.2)
    with pytest.raises(GapInputError, match="quimica"):
        transfer(gap_map, make_metadata(["minas", "quimica"]), affinity)


# --- Rule (d): temporal declines -------------------------------------------------------------


def decline(
    gap_map: GapMap, metadata: ThesisMetadata, *, min_count: int = 3, min_decline: float = 0.02
) -> tuple[TemporalDecline, ...]:
    return temporal_declines(
        gap_map,
        metadata,
        first_year=2021,
        split_year=2024,
        last_year=2026,
        min_count=min_count,
        min_decline=min_decline,
    )


def windows(early: Sequence[int], late: Sequence[int]) -> tuple[GapMap, ThesisMetadata]:
    """A map of ``early`` theses from 2021–2023 and ``late`` ones from 2024–2026."""
    years = [2021 + index % 3 for index in range(len(early))]
    years += [2024 + index % 3 for index in range(len(late))]
    labels = [*early, *late]
    return make_map(labels), make_metadata(["minas"] * len(labels), years)


def test_rule_d_keeps_the_topics_whose_share_fell_by_the_threshold() -> None:
    # 100 theses in each window, 5 of the early ones noise. Topic 0 falls from 10% to 2%;
    # topic 1 falls by 2 points but has only 2 early theses; topic 2 grows from 2% to 10%;
    # topic 3 falls from 3% with exactly 3 early theses.
    early = [*[0] * 10, *[1] * 2, *[2] * 2, *[3] * 3, *[-1] * 5, *[4] * 78]
    late = [*[0] * 2, *[2] * 10, *[4] * 88]

    gaps = decline(*windows(early, late))

    assert [gap.topic for gap in gaps] == [0, 3]
    first = gaps[0]
    assert (first.early_count, first.early_theses, first.early_share) == (10, 100, 0.1)
    assert (first.late_count, first.late_theses, first.late_share) == (2, 100, 0.02)
    assert first.decline == pytest.approx(0.08)
    assert (first.early_years, first.late_years) == ((2021, 2023), (2024, 2026))
    assert (first.min_count, first.min_decline) == (3, 0.02)
    assert first.members == codes([*range(10), 100, 101])
    assert first.program_counts == (("minas", 12),)
    assert first.rule == "d" and first.match_key == ()
    assert "2021-2023" in first.reason and "2024-2026" in first.reason


def test_rule_d_compares_shares_of_each_window_not_counts() -> None:
    # Topic 0 has 10 of 100 early theses and 5 of 50 late ones: the same 10% share.
    early = [*[0] * 10, *[1] * 90]
    late = [*[0] * 5, *[1] * 45]

    assert decline(*windows(early, late), min_decline=0.01) == ()


def test_rule_d_includes_the_decline_threshold_itself() -> None:
    early = [*[0] * 3, *[1] * 97]
    late = [1] * 100

    assert [gap.topic for gap in decline(*windows(early, late), min_decline=0.03)] == [0]
    assert decline(*windows(early, late), min_decline=0.031) == ()


def test_rule_d_refuses_invalid_windows_and_thresholds() -> None:
    gap_map, metadata = windows([0, 0, 0], [0, 1])

    def run(**overrides: float) -> object:
        arguments = {
            "first_year": 2021,
            "split_year": 2024,
            "last_year": 2026,
            "min_count": 3,
            "min_decline": 0.1,
        } | overrides
        return temporal_declines(gap_map, metadata, **arguments)

    with pytest.raises(GapInputError, match="split_year"):
        run(split_year=2021)
    with pytest.raises(GapInputError, match="split_year"):
        run(split_year=2027)
    with pytest.raises(GapInputError, match="min_count"):
        run(min_count=0)
    with pytest.raises(GapInputError, match="min_decline"):
        run(min_decline=0.0)
    with pytest.raises(GapInputError, match="min_decline"):
        run(min_decline=1.5)
    with pytest.raises(GapInputError, match="2025"):
        run(last_year=2024, split_year=2023)
    with pytest.raises(GapInputError, match="no thesis"):
        run(first_year=2020, split_year=2021, last_year=2026)


# --- Determinism --------------------------------------------------------------------------------


def test_every_rule_returns_the_same_gaps_on_a_second_call(affinity: CareerAffinity) -> None:
    gap_map, vectors, metadata = bridge_layout()
    programs = ["industrial"] * 8 + ["minas"] * 8
    years = [2021, 2022, 2023, 2024] * 4
    transfer_metadata = make_metadata(programs, years)

    def everything() -> tuple[object, ...]:
        return (
            small_coherent_topics(
                gap_map,
                metadata,
                size_cut=small_topic_cuts([gap_map], npmi_floor_quantile=0.0).size_cut,
                npmi_floor=0.0,
            ),
            interstitial_pairs(gap_map, vectors, metadata, nearest_topics=3, bridge_max_docs=1),
            transfer(gap_map, transfer_metadata, affinity, rare_max_share=0.1),
            temporal_declines(
                gap_map,
                transfer_metadata,
                first_year=2021,
                split_year=2024,
                last_year=2026,
                min_count=1,
                min_decline=0.01,
            ),
        )

    assert everything() == everything()
