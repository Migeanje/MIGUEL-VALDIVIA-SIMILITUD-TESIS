"""The temporal split of the topic map (T27, O04): the periods, the earlier period's vocabulary,
the map fitted on the earlier theses, the later theses' assignment and the expert sample.

UMAP and BERTopic are the recording stand-ins of ``conftest.py``; HDBSCAN is a stand-in that
gives planted labels, so every expected value is known. The c-TF-IDF is checked against the
independent oracle of ``conftest.py``, and the assignment against T23's own functions.
"""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thematic_redundancy.labeling.pairs import allocate_with_floor
from thematic_redundancy.representation.tfidf import section_terms
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.assignment import (
    ATYPICAL_PERCENTILE,
    TopicCentroids,
    assign_proposals,
    atypical_threshold,
    check_atypical,
)
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicCtfidf
from thematic_redundancy.topic_map.build import TopicMapPorts, TopicMapSettings, topic_centroids
from thematic_redundancy.topic_map.clustering import HdbscanRun, HdbscanSettings, KMeansSettings
from thematic_redundancy.topic_map.ctfidf import ctfidf_topics
from thematic_redundancy.topic_map.reduction import ProjectionSettings, UmapSettings
from thematic_redundancy.topic_map.temporal import (
    SAMPLE_POOLS,
    ExpertSample,
    SampleStratum,
    TemporalMap,
    TemporalSettings,
    assign_period,
    draw_expert_sample,
    fit_temporal_map,
    period_vocabulary,
    split_periods,
)

SETTINGS = TopicMapSettings(
    umap=UmapSettings(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"),
    projection=ProjectionSettings(method="umap", n_neighbors=15, min_dist=0.0, metric="cosine"),
    kmeans=KMeansSettings(n_init=10, init="k-means++"),
    hdbscan=HdbscanSettings(min_samples=None, metric="euclidean", cluster_selection_method="eom"),
    top_n=5,
    max_ngram=2,
    include_noise_class=True,
)
PROGRAMS = ("sistemas", "industrial", "electronica", "mecanica", "minas")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


class PlantedHdbscan:
    """An HDBSCAN factory whose models give the planted labels and record what they got."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = np.asarray(labels)
        self.calls: list[tuple[int, Any, tuple[int, ...]]] = []

    def __call__(self, size: int, settings: HdbscanSettings) -> Any:
        return _PlantedModel(self, size, settings)


class _PlantedModel:
    def __init__(self, factory: PlantedHdbscan, size: int, settings: HdbscanSettings) -> None:
        self.factory, self.size, self.settings = factory, size, settings

    def fit(self, X: Any) -> "_PlantedModel":
        self.factory.calls.append((self.size, self.settings, np.shape(X)))
        assert np.shape(X)[0] == self.factory.labels.size
        self.labels_ = self.factory.labels
        return self


@pytest.fixture(scope="module")
def scene(corpus_maker: Any) -> Any:
    """48 theses of three themes, the first 12 of each fitted and the last 4 assigned, plus two
    far outliers in the assigned period; the earlier map's planted labels make theses 0 and 1
    noise."""
    corpus = corpus_maker(per_theme=16, outliers=2)
    themes = corpus.themes
    position = np.concatenate([np.arange(16)] * 3 + [np.array([20, 21])])
    years = np.where(position < 12, 2021 + position % 4, 2025 + position % 2)
    periods = split_periods(years, first_year=2021, split_year=2025, last_year=2026)
    planted = themes[periods.fitted].copy()
    planted[:2] = -1
    return corpus, years, periods, planted


def fit(scene: Any, recording_umap: Any, fakes: Any, **overrides: Any) -> TemporalMap:
    corpus, _, periods, planted = scene
    documents = [corpus.documents[row] for row in periods.fitted]
    arguments: dict[str, Any] = {
        "vocabulary": period_vocabulary(corpus.vocabulary, documents, max_ngram=2),
        "clustering": HdbscanRun(min_cluster_size=10),
        "seed": 42,
        "settings": SETTINGS,
        "ports": TopicMapPorts(
            umap=recording_umap,
            hdbscan=overrides.pop("hdbscan", PlantedHdbscan(planted)),
            ctfidf=BertopicCtfidf(fakes.components()),
        ),
        "percentile": ATYPICAL_PERCENTILE,
        "method": "linear",
        "scope": "global",
    }
    arguments.update(overrides)
    return fit_temporal_map(corpus.vectors[periods.fitted], documents, **arguments)


# The settings


def test_the_settings_follow_the_configuration() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    settings = TemporalSettings.from_config(config)

    assert settings == TemporalSettings(
        first_year=2021,
        split_year=2025,
        last_year=2026,
        percentile=ATYPICAL_PERCENTILE,
        method="linear",
        scope="global",
        noise_handling="own_label",
        noise_handling_sensitivity="exclude_either",
        programs=PROGRAMS,
        sample_seed=20261027,
        sample_size=40,
        sample_atypical=10,
        sample_program_floor=2,
    )
    assert settings.noise_handlings == ("own_label", "exclude_either")
    assert settings.record() == {
        "first_year": 2021,
        "split_year": 2025,
        "last_year": 2026,
        "atypical_percentile": 5.0,
        "atypical_scope": "global",
        "atypical_method": "linear",
        "noise_handling": "own_label",
        "noise_handling_sensitivity": "exclude_either",
        "sample_seed": 20261027,
        "sample_size": 40,
        "sample_atypical": 10,
        "sample_program_floor": 2,
    }


# The periods


def test_the_periods_split_the_theses_by_issue_year_at_the_split_year() -> None:
    years = [2024, 2025, 2021, 2026, 2023, 2025]

    periods = split_periods(years, first_year=2021, split_year=2025, last_year=2026)

    assert (periods.first_year, periods.split_year, periods.last_year) == (2021, 2025, 2026)
    assert periods.fitted.tolist() == [0, 2, 4]
    assert periods.assigned.tolist() == [1, 3, 5]
    assert periods.fitted_years == (2021, 2024)
    assert periods.assigned_years == (2025, 2026)
    assert not periods.fitted.flags.writeable and not periods.assigned.flags.writeable


@pytest.mark.parametrize(
    ("years", "bounds", "message"),
    [
        ([2021, 2027], (2021, 2025, 2026), "1 thesis was issued outside 2021 to 2026"),
        ([2020, 2019, 2025], (2021, 2025, 2026), "2 theses were issued outside 2021 to 2026"),
        ([2025, 2026], (2021, 2025, 2026), "no thesis of 2021 to 2024"),
        ([2021, 2024], (2021, 2025, 2026), "no thesis of 2025 to 2026"),
        ([2021, 2025], (2021, 2021, 2026), "split year 2021 must be after 2021"),
        ([2021, 2025], (2021, 2027, 2026), "split year 2027 must be after 2021 and not after"),
        ([2021.0, 2025.0], (2021, 2025, 2026), "integer"),
    ],
)
def test_a_split_that_cannot_hold_is_refused(
    years: list, bounds: tuple[int, int, int], message: str
) -> None:
    first, split, last = bounds
    with pytest.raises(TopicMapError, match=message):
        split_periods(years, first_year=first, split_year=split, last_year=last)


# The earlier period's vocabulary (T16 note 4)


def test_the_vocabulary_keeps_t15s_terms_that_the_documents_form_in_t15s_order() -> None:
    vocabulary = ("gamma", "alfa", "beta gamma", "alfa beta", "delta", "beta")
    documents = [(("alfa", "beta"), ("gamma",)), (("beta",),)]

    # "beta gamma" would span two sections and "delta" is in no document.
    assert period_vocabulary(vocabulary, documents, max_ngram=2) == (
        "gamma",
        "alfa",
        "alfa beta",
        "beta",
    )


def test_the_full_vocabulary_does_not_belong_to_the_earlier_theses_but_theirs_does(
    scene: Any, fakes: Any
) -> None:
    corpus, _, periods, planted = scene
    documents = [corpus.documents[row] for row in periods.fitted]
    vocabulary = period_vocabulary(corpus.vocabulary, documents, max_ngram=2)
    later_only = set(corpus.vocabulary) - set(vocabulary)
    formed = {
        term for document in documents for section in document for term in section_terms(section, 2)
    }

    assert later_only and not later_only & formed
    assert set(vocabulary) <= formed
    arguments = {
        "max_ngram": 2,
        "top_n": 5,
        "include_noise_class": True,
        "backend": BertopicCtfidf(fakes.components()),
    }
    with pytest.raises(TopicMapError, match="occur in none of the documents"):
        ctfidf_topics(planted, documents, vocabulary=corpus.vocabulary, **arguments)
    assert ctfidf_topics(planted, documents, vocabulary=vocabulary, **arguments).topics


# The earlier period's map


def test_the_map_is_one_seeded_unsupervised_umap_fit_on_the_earlier_theses(
    scene: Any, recording_umap: Any, fakes: Any
) -> None:
    corpus, _, periods, planted = scene
    hdbscan = PlantedHdbscan(planted)

    temporal_map = fit(scene, recording_umap, fakes, hdbscan=hdbscan)

    assert [(call.seed, call.settings) for call in recording_umap.calls] == [(42, SETTINGS.umap)]
    (call,) = recording_umap.calls
    np.testing.assert_array_equal(call.args[0], corpus.vectors[periods.fitted])
    assert call.kwargs == {}
    assert [(size, settings, shape) for size, settings, shape in hdbscan.calls] == [
        (10, SETTINGS.hdbscan, (periods.fitted.size, 5))
    ]
    assert temporal_map.seed == 42
    assert temporal_map.clustering == HdbscanRun(min_cluster_size=10)


def test_the_map_keeps_the_clusterings_labels_noise_included(
    scene: Any, recording_umap: Any, fakes: Any
) -> None:
    _, _, _, planted = scene

    temporal_map = fit(scene, recording_umap, fakes)

    np.testing.assert_array_equal(temporal_map.labels, planted)
    assert temporal_map.noise_points == 2
    assert not temporal_map.labels.flags.writeable


def test_the_topics_terms_are_the_c_tf_idf_over_the_earlier_vocabulary(
    scene: Any, recording_umap: Any, fakes: Any, oracle: Any
) -> None:
    corpus, _, periods, planted = scene
    documents = tuple(corpus.documents[row] for row in periods.fitted)
    vocabulary = period_vocabulary(corpus.vocabulary, documents, max_ngram=2)

    temporal_map = fit(scene, recording_umap, fakes)

    classes, counts = oracle.counts(planted, documents, vocabulary)
    weights = oracle.formula(counts).toarray()
    expected = [
        (label, int(np.sum(planted == label)), *oracle.top_terms(weights[row], vocabulary, 5))
        for row, label in enumerate(classes)
        if label != -1
    ]
    assert [(t.topic, t.size, t.terms, t.scores) for t in temporal_map.topics] == [
        (label, size, terms, pytest.approx(scores)) for label, size, terms, scores in expected
    ]
    assert temporal_map.vocabulary == len(vocabulary)


def test_the_centroids_and_the_atypical_threshold_are_t23s_on_the_original_vectors(
    scene: Any, recording_umap: Any, fakes: Any
) -> None:
    corpus, _, periods, planted = scene
    vectors = corpus.vectors[periods.fitted]

    temporal_map = fit(scene, recording_umap, fakes, method="hazen", scope="per_topic")

    assert temporal_map.centroids.topics == (0, 1, 2)
    np.testing.assert_allclose(
        temporal_map.centroids.centroids, topic_centroids(vectors, planted, (0, 1, 2))
    )
    assert temporal_map.threshold == atypical_threshold(
        vectors,
        planted,
        TopicCentroids(topics=(0, 1, 2), centroids=topic_centroids(vectors, planted, (0, 1, 2))),
        percentile=ATYPICAL_PERCENTILE,
        method="hazen",
        scope="per_topic",
    )


def test_a_map_of_noise_alone_is_refused(scene: Any, recording_umap: Any, fakes: Any) -> None:
    _, _, periods, _ = scene
    noise = PlantedHdbscan(np.full(periods.fitted.size, -1))

    with pytest.raises(TopicMapError, match="no topic"):
        fit(scene, recording_umap, fakes, hdbscan=noise)


def test_the_steps_are_timed(scene: Any, recording_umap: Any, fakes: Any) -> None:
    ticks = iter(range(100))

    temporal_map = fit(scene, recording_umap, fakes, timer=lambda: float(next(ticks)))

    assert temporal_map.seconds == {"umap": 1.0, "cluster": 1.0, "ctfidf": 1.0, "threshold": 1.0}


# The later period's assignment


def test_each_later_thesis_goes_to_its_nearest_centroid_with_its_atypical_flag(
    scene: Any, recording_umap: Any, fakes: Any
) -> None:
    corpus, _, periods, _ = scene
    temporal_map = fit(scene, recording_umap, fakes)
    later = corpus.vectors[periods.assigned]

    assigned = assign_period(later, temporal_map)

    expected = assign_proposals(later, temporal_map.centroids)
    assert assigned.assignments == expected
    assert assigned.checks == tuple(check_atypical(a, temporal_map.threshold) for a in expected)
    np.testing.assert_array_equal(assigned.labels, [a.topic for a in expected])
    themes = corpus.themes[periods.assigned]
    assert assigned.labels[themes >= 0].tolist() == themes[themes >= 0].tolist()
    # The two far outliers are atypical, and so is none of the later theses of a theme here.
    assert assigned.atypical.tolist() == (themes == -1).tolist()
    assert assigned.atypical_count == 2


def test_a_later_vector_of_other_dimensions_is_refused(
    scene: Any, recording_umap: Any, fakes: Any
) -> None:
    temporal_map = fit(scene, recording_umap, fakes)

    with pytest.raises(TopicMapError, match="dimensions"):
        assign_period(np.ones((2, 3)), temporal_map)


# The expert sample


def sample_scene() -> tuple[tuple[str, ...], np.ndarray]:
    """40 typical theses (30 industrial, 6 minas, 4 sistemas) and 6 atypical ones (3 industrial,
    2 minas, 1 sistemas), interleaved."""
    programs = ["industrial"] * 30 + ["minas"] * 6 + ["sistemas"] * 4
    programs += ["industrial"] * 3 + ["minas"] * 2 + ["sistemas"]
    atypical = np.array([False] * 40 + [True] * 6)
    order = np.random.default_rng(5).permutation(46)
    return tuple(programs[i] for i in order), atypical[order]


def draw(seed: int = 20261027, **overrides: Any) -> ExpertSample:
    programs, atypical = overrides.pop("scene", sample_scene())
    arguments: dict[str, Any] = {
        "program_order": PROGRAMS,
        "size": 12,
        "atypical_quota": 4,
        "program_floor": 1,
        "seed": seed,
    }
    arguments.update(overrides)
    return draw_expert_sample(programs, atypical, **arguments)


def test_the_quotas_follow_largest_remainder_with_a_floor_per_program() -> None:
    # Atypical pool, 4 of 6: a floor of 1 each, then industrial's 2 left beat minas's 1. Typical
    # pool, 8 of 40: floors 3, then 5 by largest remainder of 3, 29 and 5: industrial 3 + 1,
    # minas 0 + 1.
    sample = draw()

    assert sample.strata == (
        SampleStratum("atypical", "sistemas", 1, 1),
        SampleStratum("atypical", "industrial", 3, 2),
        SampleStratum("atypical", "electronica", 0, 0),
        SampleStratum("atypical", "mecanica", 0, 0),
        SampleStratum("atypical", "minas", 2, 1),
        SampleStratum("typical", "sistemas", 4, 1),
        SampleStratum("typical", "industrial", 30, 5),
        SampleStratum("typical", "electronica", 0, 0),
        SampleStratum("typical", "mecanica", 0, 0),
        SampleStratum("typical", "minas", 6, 2),
    )
    programs, atypical = sample_scene()
    for pool, flag in zip(SAMPLE_POOLS, (True, False), strict=True):
        sizes = Counter(p for p, a in zip(programs, atypical, strict=True) if a == flag)
        quota = 4 if flag else 8
        allocation = allocate_with_floor({p: sizes[p] for p in PROGRAMS}, quota, 1)
        assert {s.program: s.quota for s in sample.strata if s.pool == pool} == allocation


def test_each_drawn_thesis_belongs_to_its_stratum_and_none_twice() -> None:
    programs, atypical = sample_scene()

    sample = draw()

    assert len(sample.rows) == len(set(sample.rows)) == 12
    drawn = Counter(
        (pool, programs[row]) for row, pool in zip(sample.rows, sample.pools, strict=True)
    )
    assert drawn == Counter({(s.pool, s.program): s.quota for s in sample.strata if s.quota})
    for row, pool in zip(sample.rows, sample.pools, strict=True):
        assert atypical[row] == (pool == "atypical")


def test_the_draw_is_reproducible_from_its_seed() -> None:
    assert draw() == draw()
    assert draw(seed=1).rows != draw().rows


def test_the_positions_present_the_sample_in_a_seeded_order() -> None:
    sample = draw()

    assert sorted(sample.positions) == list(range(1, 13))
    assert list(sample.positions) != list(range(1, 13))
    assert sample.positions == draw().positions


def test_fewer_atypical_theses_than_the_quota_give_the_rest_to_the_typical_pool() -> None:
    sample = draw(atypical_quota=10)

    assert sum(s.quota for s in sample.strata if s.pool == "atypical") == 6
    assert sum(s.quota for s in sample.strata if s.pool == "typical") == 6
    assert len(sample.rows) == 12


def test_a_size_beyond_the_theses_takes_them_all() -> None:
    sample = draw(size=100, atypical_quota=10)

    assert sorted(sample.rows) == list(range(46))
    assert sorted(sample.positions) == list(range(1, 47))


def test_no_atypical_quota_draws_from_the_typical_pool_alone() -> None:
    sample = draw(atypical_quota=0)

    assert set(sample.pools) == {"typical"}
    assert len(sample.rows) == 12


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scene": (("industrial", "derecho"), np.array([False, False]))}, "derecho"),
        ({"scene": (("industrial",), np.array([False, True]))}, "2 atypical flags .* 1"),
        ({"atypical_quota": 13}, "atypical quota 13 exceeds the size 12"),
        ({"size": 0}, "size must be at least 1"),
    ],
)
def test_an_invalid_sample_request_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(TopicMapError, match=message):
        draw(**overrides)
