"""K-means and HDBSCAN labels for the topic map (T16, D30 E1), with the real libraries.

Blobs on orthogonal axes have a known partition; three points far from every blob are what
HDBSCAN must leave as noise.
"""

from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.clustering import (
    HdbscanSettings,
    KMeansSettings,
    hdbscan_labels,
    hdbscan_model,
    kmeans_labels,
    kmeans_model,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

KMEANS = KMeansSettings(n_init=10, init="k-means++")
HDBSCAN = HdbscanSettings(min_samples=None, metric="euclidean", cluster_selection_method="eom")


def blobs(outliers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    centers = np.eye(6)[:3] * 10
    points = np.vstack([c + rng.normal(0, 0.2, (15, 6)) for c in centers])
    truth = np.repeat(np.arange(3), 15)
    if outliers:
        far = np.eye(6)[3 : 3 + outliers] * 60
        points = np.vstack([points, far])
        truth = np.concatenate([truth, np.full(outliers, -1)])
    return points, truth


def test_the_settings_follow_the_configuration() -> None:
    experiment = load_config(DEFAULT_CONFIG_PATH).experiment

    assert KMeansSettings.from_config(experiment.kmeans) == KMEANS
    assert HdbscanSettings.from_config(experiment.hdbscan) == HDBSCAN


def test_the_kmeans_model_takes_the_declared_arguments_and_the_seed() -> None:
    params = kmeans_model(12, 42, KMeansSettings(n_init=7, init="random")).get_params()

    assert params["n_clusters"] == 12
    assert params["n_init"] == 7
    assert params["init"] == "random"
    assert params["random_state"] == 42
    # Everything else stays at scikit-learn 1.9.1's defaults (D30 E1).
    assert (params["algorithm"], params["max_iter"], params["tol"]) == ("lloyd", 300, 1e-4)


@pytest.mark.parametrize("min_samples", [None, 3])
def test_the_hdbscan_model_takes_the_declared_arguments(min_samples: int | None) -> None:
    settings = HdbscanSettings(
        min_samples=min_samples, metric="euclidean", cluster_selection_method="leaf"
    )

    params = hdbscan_model(10, settings).get_params()

    assert isinstance(hdbscan_model(10, settings), hdbscan.HDBSCAN)
    assert params["min_cluster_size"] == 10
    assert params["min_samples"] == min_samples
    assert params["metric"] == "euclidean"
    assert params["cluster_selection_method"] == "leaf"


def test_kmeans_recovers_the_blobs_and_repeats_for_a_seed() -> None:
    points, truth = blobs()

    labels = kmeans_labels(points, 3, 7, KMEANS)
    again = kmeans_labels(points, 3, 7, KMEANS)

    assert adjusted_rand_score(truth, labels) == 1.0
    np.testing.assert_array_equal(labels, again)
    assert labels.dtype == np.int64 and labels.ndim == 1
    assert not labels.flags.writeable


def test_kmeans_gets_the_seed_and_the_settings() -> None:
    received: list[tuple[int, int, KMeansSettings]] = []

    def recording(n_clusters: int, seed: int, settings: KMeansSettings) -> Any:
        received.append((n_clusters, seed, settings))
        return kmeans_model(n_clusters, seed, settings)

    kmeans_labels(blobs()[0], 3, 73, KMEANS, model_factory=recording)

    assert received == [(3, 73, KMEANS)]


def test_kmeans_runs_on_a_sparse_matrix() -> None:
    points, truth = blobs()

    labels = kmeans_labels(sparse.csr_matrix(points), 3, 7, KMEANS)

    assert adjusted_rand_score(truth, labels) == 1.0
    np.testing.assert_array_equal(labels, kmeans_labels(points, 3, 7, KMEANS))


def test_hdbscan_keeps_its_noise_label() -> None:
    points, truth = blobs(outliers=3)

    labels = hdbscan_labels(points, 5, HDBSCAN)

    direct = hdbscan.HDBSCAN(min_cluster_size=5).fit(points).labels_
    np.testing.assert_array_equal(labels, direct)
    np.testing.assert_array_equal(labels[-3:], [-1, -1, -1])
    assert adjusted_rand_score(truth[:-3], labels[:-3]) == 1.0
    assert labels.dtype == np.int64
    assert not labels.flags.writeable


def test_hdbscan_gets_the_size_and_the_settings() -> None:
    received: list[tuple[int, HdbscanSettings]] = []

    def recording(min_cluster_size: int, settings: HdbscanSettings) -> Any:
        received.append((min_cluster_size, settings))
        return hdbscan_model(min_cluster_size, settings)

    hdbscan_labels(blobs()[0], 4, HDBSCAN, model_factory=recording)

    assert received == [(4, HDBSCAN)]


@pytest.mark.parametrize("n_clusters", [1, 46])
def test_kmeans_refuses_a_k_outside_two_to_the_points(n_clusters: int) -> None:
    with pytest.raises(TopicMapError, match="n_clusters"):
        kmeans_labels(blobs()[0], n_clusters, 7, KMEANS)


def test_hdbscan_refuses_a_min_cluster_size_below_two() -> None:
    with pytest.raises(TopicMapError, match="min_cluster_size"):
        hdbscan_labels(blobs()[0], 1, HDBSCAN)


@pytest.mark.parametrize("points", [np.zeros(4), np.array([[np.nan, 1.0], [0.0, 1.0]])])
def test_invalid_points_are_refused(points: np.ndarray) -> None:
    with pytest.raises(TopicMapError):
        kmeans_labels(points, 2, 7, KMEANS)
    with pytest.raises(TopicMapError):
        hdbscan_labels(points, 2, HDBSCAN)
