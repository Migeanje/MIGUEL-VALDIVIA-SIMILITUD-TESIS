"""The topic map of one configuration and one seed (T16, D30), on synthetic theses.

UMAP and BERTopic are stand-ins (``conftest.py``); K-means, HDBSCAN and PCA are real. The
outputs also go through T17's metrics, as the experiment (T19) will pass them.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from thematic_redundancy.evaluation.clustering_metrics import internal_metrics
from thematic_redundancy.evaluation.coherence import npmi_coherence
from thematic_redundancy.evaluation.stability import seed_stability
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicCtfidf
from thematic_redundancy.topic_map.build import (
    TopicMap,
    TopicMapPorts,
    TopicMapSettings,
    build_topic_map,
    build_topic_maps,
    cluster,
    topic_centroids,
)
from thematic_redundancy.topic_map.clustering import (
    HdbscanRun,
    HdbscanSettings,
    KMeansRun,
    KMeansSettings,
    hdbscan_labels,
    kmeans_labels,
    kmeans_model,
)
from thematic_redundancy.topic_map.ctfidf import ctfidf_topics
from thematic_redundancy.topic_map.reduction import (
    ProjectionSettings,
    UmapSettings,
    fit_reduction,
    project,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

SETTINGS = TopicMapSettings(
    umap=UmapSettings(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"),
    projection=ProjectionSettings(method="umap", n_neighbors=15, min_dist=0.0, metric="cosine"),
    kmeans=KMeansSettings(n_init=10, init="k-means++"),
    hdbscan=HdbscanSettings(min_samples=None, metric="euclidean", cluster_selection_method="eom"),
    top_n=5,
    max_ngram=2,
    include_noise_class=True,
)
RUNS = (KMeansRun(n_clusters=3), HdbscanRun(min_cluster_size=4))


@pytest.fixture
def ports(recording_umap: Any, bertopic_components: Any) -> TopicMapPorts:
    return TopicMapPorts(umap=recording_umap, ctfidf=BertopicCtfidf(bertopic_components))


def maps_for(corpus: Any, ports: TopicMapPorts, seed: int = 7, **overrides: Any) -> tuple:
    return build_topic_maps(
        corpus.vectors,
        corpus.documents,
        overrides.pop("runs", RUNS),
        vocabulary=corpus.vocabulary,
        seed=seed,
        settings=overrides.pop("settings", SETTINGS),
        ports=ports,
    )


def single_map(corpus: Any, labels: Any, ports: TopicMapPorts, **overrides: Any) -> TopicMap:
    arguments: dict[str, Any] = {
        "vocabulary": corpus.vocabulary,
        "seed": 7,
        "settings": SETTINGS,
        "ports": ports,
    }
    arguments.update(overrides)
    return build_topic_map(corpus.vectors, labels, corpus.documents, **arguments)


def test_the_settings_follow_the_configuration() -> None:
    settings = TopicMapSettings.from_config(load_config(DEFAULT_CONFIG_PATH))

    assert settings == replace(SETTINGS, top_n=10)


# One UMAP fit per (model, seed), shared by every clustering (D30 E1)


def test_one_umap_fit_and_one_projection_serve_every_clustering(
    corpus_with_outliers: Any, ports: TopicMapPorts, recording_umap: Any
) -> None:
    kmeans_map, hdbscan_map = maps_for(corpus_with_outliers, ports)

    assert len(recording_umap.fits(5)) == 1
    assert len(recording_umap.fits(2)) == 1
    assert kmeans_map.reduction is hdbscan_map.reduction
    assert kmeans_map.coordinates is hdbscan_map.coordinates
    assert kmeans_map.reduction.seed == 7


def test_each_run_clusters_the_shared_reduction(
    corpus_with_outliers: Any, ports: TopicMapPorts
) -> None:
    kmeans_map, hdbscan_map = maps_for(corpus_with_outliers, ports)

    embedding = kmeans_map.reduction.embedding
    np.testing.assert_array_equal(
        kmeans_map.labels, kmeans_labels(embedding, 3, 7, SETTINGS.kmeans)
    )
    np.testing.assert_array_equal(
        hdbscan_map.labels, hdbscan_labels(embedding, 4, SETTINGS.hdbscan)
    )
    assert (kmeans_map.clustering, hdbscan_map.clustering) == RUNS


def test_the_reduction_and_the_projection_are_unsupervised_and_seeded(
    corpus: Any, ports: TopicMapPorts, recording_umap: Any
) -> None:
    maps_for(corpus, ports, seed=42)

    for call in recording_umap.calls:
        assert call.seed == 42
        assert len(call.args) == 1 and call.kwargs == {}
    (reduction,) = recording_umap.fits(5)
    assert reduction.settings == SETTINGS.umap
    (projection,) = recording_umap.fits(2)
    assert (projection.settings.n_neighbors, projection.settings.min_dist) == (15, 0.0)
    # Both read the original vectors: the map is not a projection of the reduction.
    np.testing.assert_array_equal(reduction.args[0], corpus.vectors)
    np.testing.assert_array_equal(projection.args[0], corpus.vectors)


def test_kmeans_gets_the_run_s_seed_and_its_declared_arguments(
    corpus: Any, ports: TopicMapPorts
) -> None:
    received: list[dict[str, Any]] = []

    def recording(n_clusters: int, seed: int, settings: KMeansSettings) -> Any:
        model = kmeans_model(n_clusters, seed, settings)
        received.append(model.get_params())
        return model

    maps_for(corpus, replace(ports, kmeans=recording), seed=21, runs=(KMeansRun(3),))

    (params,) = received
    assert (params["random_state"], params["n_init"], params["init"]) == (21, 10, "k-means++")


def test_hdbscan_gets_its_declared_arguments(corpus: Any, ports: TopicMapPorts) -> None:
    received: list[tuple[int, HdbscanSettings]] = []

    def recording(min_cluster_size: int, settings: HdbscanSettings) -> Any:
        received.append((min_cluster_size, settings))
        return ports.hdbscan(min_cluster_size, settings)

    maps_for(corpus, replace(ports, hdbscan=recording), runs=(HdbscanRun(6),))

    assert received == [(6, SETTINGS.hdbscan)]


def test_cluster_dispatches_each_kind_of_run(corpus: Any, ports: TopicMapPorts) -> None:
    points = corpus.vectors[:, :5]

    np.testing.assert_array_equal(
        cluster(points, KMeansRun(3), seed=7, settings=SETTINGS, ports=ports),
        kmeans_labels(points, 3, 7, SETTINGS.kmeans),
    )
    np.testing.assert_array_equal(
        cluster(points, HdbscanRun(5), seed=7, settings=SETTINGS, ports=ports),
        hdbscan_labels(points, 5, SETTINGS.hdbscan),
    )


def test_cluster_refuses_an_unknown_run(corpus: Any, ports: TopicMapPorts) -> None:
    with pytest.raises(TopicMapError, match="unknown clustering run"):
        cluster(corpus.vectors, "kmeans", seed=7, settings=SETTINGS, ports=ports)  # type: ignore[arg-type]


# What a map holds


def test_a_kmeans_map_has_one_topic_per_cluster_with_its_centroid(
    corpus: Any, ports: TopicMapPorts
) -> None:
    (kmeans_map,) = maps_for(corpus, ports, runs=(KMeansRun(3),))

    labels = kmeans_map.labels
    assert [topic.topic for topic in kmeans_map.topics] == [0, 1, 2]
    assert [topic.size for topic in kmeans_map.topics] == [
        int(np.count_nonzero(labels == t)) for t in (0, 1, 2)
    ]
    assert kmeans_map.noise is None and kmeans_map.noise_points == 0
    # Centroids in the original space (D14, D30 E5): the mean of the members' vectors.
    assert kmeans_map.centroids.shape == (3, corpus.vectors.shape[1])
    for row, topic in enumerate(kmeans_map.topics):
        np.testing.assert_allclose(
            kmeans_map.centroids[row], corpus.vectors[labels == topic.topic].mean(axis=0)
        )
    assert kmeans_map.coordinates.shape == (len(corpus.documents), 2)
    assert kmeans_map.terms() == tuple(topic.terms for topic in kmeans_map.topics)


def test_the_topics_are_the_c_tf_idf_topics_of_the_map_s_labels(
    corpus_with_outliers: Any, ports: TopicMapPorts
) -> None:
    for topic_map in maps_for(corpus_with_outliers, ports):
        expected = ctfidf_topics(
            topic_map.labels,
            corpus_with_outliers.documents,
            vocabulary=corpus_with_outliers.vocabulary,
            max_ngram=2,
            top_n=SETTINGS.top_n,
            include_noise_class=True,
            backend=ports.ctfidf,
        )
        assert (topic_map.topics, topic_map.noise) == (expected.topics, expected.noise)


def test_hdbscan_noise_stays_noise_and_has_no_topic_or_centroid(
    corpus_with_outliers: Any, ports: TopicMapPorts
) -> None:
    _, hdbscan_map = maps_for(corpus_with_outliers, ports)

    labels = hdbscan_map.labels
    np.testing.assert_array_equal(labels[-3:], [-1, -1, -1])
    assert hdbscan_map.noise_points == int(np.count_nonzero(labels == -1)) >= 3
    assert -1 not in [topic.topic for topic in hdbscan_map.topics]
    assert hdbscan_map.noise is not None and hdbscan_map.noise.topic == -1
    assert hdbscan_map.noise.size == hdbscan_map.noise_points
    assert hdbscan_map.centroids.shape[0] == len(hdbscan_map.topics)
    assert hdbscan_map.coordinates.shape[0] == labels.size


def test_without_the_noise_class_the_map_has_no_noise_terms(
    corpus_with_outliers: Any, ports: TopicMapPorts
) -> None:
    settings = replace(SETTINGS, include_noise_class=False)

    _, hdbscan_map = maps_for(corpus_with_outliers, ports, settings=settings)

    assert hdbscan_map.noise is None
    assert hdbscan_map.noise_points >= 3


def test_precomputed_labels_are_kept_and_no_reduction_is_fitted(
    corpus: Any, ports: TopicMapPorts, recording_umap: Any
) -> None:
    labels = np.where(corpus.themes == 0, 30, np.where(corpus.themes == 1, 7, 12))
    labels[:2] = -1

    topic_map = single_map(corpus, labels, ports)

    np.testing.assert_array_equal(topic_map.labels, labels)
    assert [topic.topic for topic in topic_map.topics] == [7, 12, 30]
    assert [topic.size for topic in topic_map.topics] == [12, 12, 10]
    assert topic_map.noise_points == 2
    assert (topic_map.clustering, topic_map.reduction) == (None, None)
    assert recording_umap.fits(5) == []
    assert len(recording_umap.fits(2)) == 1
    np.testing.assert_allclose(topic_map.centroids[2], corpus.vectors[labels == 30].mean(axis=0))


def test_given_coordinates_are_used_without_a_new_projection(
    corpus: Any, ports: TopicMapPorts, recording_umap: Any
) -> None:
    coordinates = np.arange(len(corpus.documents) * 2, dtype=float).reshape(-1, 2)

    topic_map = single_map(corpus, corpus.themes, ports, coordinates=coordinates)

    np.testing.assert_array_equal(topic_map.coordinates, coordinates)
    assert recording_umap.calls == []


def test_the_map_s_arrays_are_read_only_copies(corpus: Any, ports: TopicMapPorts) -> None:
    labels = corpus.themes.copy()

    topic_map = single_map(corpus, labels, ports)
    labels[0] = 2

    assert topic_map.labels[0] == 0
    for array in (topic_map.labels, topic_map.centroids, topic_map.coordinates):
        assert not array.flags.writeable
    assert topic_map.labels.dtype == np.int64


def test_a_map_repeats_for_a_seed(corpus_with_outliers: Any, ports: TopicMapPorts) -> None:
    first = maps_for(corpus_with_outliers, ports, seed=13)
    again = maps_for(corpus_with_outliers, ports, seed=13)

    for one, other in zip(first, again, strict=True):
        np.testing.assert_array_equal(one.labels, other.labels)
        np.testing.assert_array_equal(one.centroids, other.centroids)
        np.testing.assert_array_equal(one.coordinates, other.coordinates)
        np.testing.assert_array_equal(one.reduction.embedding, other.reduction.embedding)
        assert (one.topics, one.noise) == (other.topics, other.noise)


def test_topic_centroids_are_the_mean_of_each_topic_s_members() -> None:
    vectors = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [9.0, 9.0]])
    labels = np.array([5, 5, 1, -1])

    centroids = topic_centroids(vectors, labels, (1, 5))

    np.testing.assert_array_equal(centroids, [[0.0, 2.0], [2.0, 0.0]])


# Invalid inputs


def test_inputs_of_different_lengths_are_refused(corpus: Any, ports: TopicMapPorts) -> None:
    with pytest.raises(TopicMapError, match="labels"):
        single_map(corpus, corpus.themes[:-1], ports)
    with pytest.raises(TopicMapError, match="documents"):
        build_topic_map(
            corpus.vectors,
            corpus.themes,
            corpus.documents[:-1],
            vocabulary=corpus.vocabulary,
            seed=7,
            settings=SETTINGS,
            ports=ports,
        )


def test_sparse_vectors_are_refused(corpus: Any, ports: TopicMapPorts) -> None:
    with pytest.raises(TopicMapError, match="dense"):
        build_topic_map(
            sparse.csr_matrix(corpus.vectors),
            corpus.themes,
            corpus.documents,
            vocabulary=corpus.vocabulary,
            seed=7,
            settings=SETTINGS,
            ports=ports,
        )


def test_a_reduction_of_another_seed_or_size_is_refused(
    corpus: Any, ports: TopicMapPorts, recording_umap: Any
) -> None:
    reduction = fit_reduction(corpus.vectors, SETTINGS.umap, 13, umap_factory=recording_umap)
    smaller = fit_reduction(corpus.vectors[:-1], SETTINGS.umap, 7, umap_factory=recording_umap)

    with pytest.raises(TopicMapError, match="seed"):
        single_map(corpus, corpus.themes, ports, reduction=reduction)
    with pytest.raises(TopicMapError, match="rows"):
        single_map(corpus, corpus.themes, ports, reduction=smaller)


def test_coordinates_of_another_shape_are_refused(corpus: Any, ports: TopicMapPorts) -> None:
    with pytest.raises(TopicMapError, match="coordinates"):
        single_map(corpus, corpus.themes, ports, coordinates=np.zeros((len(corpus.themes), 3)))


# The outputs feed T17's metrics


def test_the_outputs_feed_the_internal_metrics_npmi_and_seed_stability(
    corpus_with_outliers: Any, ports: TopicMapPorts
) -> None:
    runs_by_seed = [maps_for(corpus_with_outliers, ports, seed=seed) for seed in (7, 13, 21)]

    for kmeans_map, hdbscan_map in runs_by_seed:
        for topic_map in (kmeans_map, hdbscan_map):
            metrics = internal_metrics(corpus_with_outliers.vectors, topic_map.labels)
            assert metrics.undefined_reason is None
            assert metrics.noise_points == topic_map.noise_points
            coherence = npmi_coherence(
                topic_map.terms(),
                corpus_with_outliers.documents,
                top_n=SETTINGS.top_n,
                unit="document",
                window_size=None,
            )
            assert coherence.undefined_reason is None
            assert len(coherence.topics) == len(topic_map.topics)
            assert all(topic.absent_terms == () for topic in coherence.topics)
    for algorithm in (0, 1):
        stability = seed_stability(
            [maps[algorithm].labels for maps in runs_by_seed], noise_handling="own_label"
        )
        assert stability.undefined_reason is None
        assert stability.mean is not None


def test_the_pca_projection_can_replace_the_umap_one(
    corpus: Any, ports: TopicMapPorts, recording_umap: Any
) -> None:
    settings = replace(SETTINGS, projection=replace(SETTINGS.projection, method="pca"))

    topic_map = single_map(corpus, corpus.themes, ports, settings=settings)

    np.testing.assert_allclose(
        topic_map.coordinates, project(corpus.vectors, settings.projection, 7), atol=1e-12
    )
    assert recording_umap.calls == []
