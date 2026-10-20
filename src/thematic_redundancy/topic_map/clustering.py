"""K-means and HDBSCAN labels for the topic map (T16, D30 E1).

- **K-means**, on the UMAP reduction for the semantic runs and on the sparse TF-IDF matrix for
  the baseline (D28 C4): scikit-learn's ``KMeans`` with the declared ``n_init`` and ``init``
  (``experiment.kmeans``) and the run's seed as ``random_state``; every other argument keeps
  scikit-learn 1.9.1's default (Lloyd, ``max_iter`` 300, ``tol`` 1e-4).
- **HDBSCAN**, on the UMAP reduction: hdbscan's ``HDBSCAN`` with the run's
  ``min_cluster_size`` and the declared ``min_samples`` (``None`` keeps its default, the
  ``min_cluster_size``), ``metric`` and ``cluster_selection_method`` (``experiment.hdbscan``).
  It is deterministic, so it takes no seed. Its noise label, -1, is returned as it is (D12,
  D16): never relabelled, and left to the metrics to exclude and report.

hdbscan is imported on first use. Labels are returned as read-only 1-D int64 arrays.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from scipy import sparse

from thematic_redundancy.evaluation.clustering_metrics import label_array
from thematic_redundancy.shared.config import ExperimentHdbscanConfig, ExperimentKMeansConfig
from thematic_redundancy.topic_map import TopicMapError


@dataclass(frozen=True)
class KMeansSettings:
    """The declared K-means arguments besides K and the seed."""

    n_init: int
    init: str

    @classmethod
    def from_config(cls, config: ExperimentKMeansConfig) -> Self:
        return cls(n_init=config.n_init, init=config.init)


@dataclass(frozen=True)
class HdbscanSettings:
    """The declared HDBSCAN arguments besides ``min_cluster_size``."""

    min_samples: int | None
    metric: str
    cluster_selection_method: str

    @classmethod
    def from_config(cls, config: ExperimentHdbscanConfig) -> Self:
        return cls(
            min_samples=config.min_samples,
            metric=config.metric,
            cluster_selection_method=config.cluster_selection_method,
        )


@dataclass(frozen=True)
class KMeansRun:
    """A K-means clustering with ``n_clusters`` clusters."""

    n_clusters: int


@dataclass(frozen=True)
class HdbscanRun:
    """An HDBSCAN clustering with ``min_cluster_size``."""

    min_cluster_size: int


ClusteringRun = KMeansRun | HdbscanRun

KMeansFactory = Callable[[int, int, KMeansSettings], Any]
"""Port: return an unfitted K-means model for K, the seed and the settings."""

HdbscanFactory = Callable[[int, HdbscanSettings], Any]
"""Port: return an unfitted HDBSCAN model for ``min_cluster_size`` and the settings."""


def kmeans_model(n_clusters: int, seed: int, settings: KMeansSettings) -> Any:
    """Return scikit-learn's ``KMeans`` with the declared arguments and ``random_state=seed``."""
    from sklearn.cluster import KMeans

    return KMeans(
        n_clusters=n_clusters, n_init=settings.n_init, init=settings.init, random_state=seed
    )


def hdbscan_model(min_cluster_size: int, settings: HdbscanSettings) -> Any:
    """Return hdbscan's ``HDBSCAN`` with the declared arguments."""
    from hdbscan import HDBSCAN

    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=settings.min_samples,
        metric=settings.metric,
        cluster_selection_method=settings.cluster_selection_method,
    )


def kmeans_labels(
    points: Any,
    n_clusters: int,
    seed: int,
    settings: KMeansSettings,
    *,
    model_factory: KMeansFactory = kmeans_model,
) -> np.ndarray:
    """Return the K-means labels of ``points``, dense or sparse.

    Raises:
        TopicMapError: if the points are invalid, or ``n_clusters`` is not from 2 to the
            number of points.
    """
    matrix = _check_points(points)
    if not 2 <= n_clusters <= matrix.shape[0]:
        raise TopicMapError(
            f"n_clusters must be from 2 to the {matrix.shape[0]} points, not {n_clusters}"
        )
    return _labels(model_factory(n_clusters, seed, settings).fit(matrix).labels_)


def hdbscan_labels(
    points: Any,
    min_cluster_size: int,
    settings: HdbscanSettings,
    *,
    model_factory: HdbscanFactory = hdbscan_model,
) -> np.ndarray:
    """Return the HDBSCAN labels of ``points``, noise as -1.

    Raises:
        TopicMapError: if the points are invalid, or ``min_cluster_size`` is below 2.
    """
    matrix = _check_points(points)
    if min_cluster_size < 2:
        raise TopicMapError(f"min_cluster_size must be at least 2, not {min_cluster_size}")
    return _labels(model_factory(min_cluster_size, settings).fit(matrix).labels_)


def _check_points(points: Any) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(points):
        matrix = sparse.csr_matrix(points, dtype=np.float64)
        values = matrix.data
    else:
        matrix = np.asarray(points, dtype=np.float64)
        if matrix.ndim != 2:
            raise TopicMapError(f"the points must be 2-D, not of shape {matrix.shape}")
        values = matrix
    if not np.isfinite(values).all():
        raise TopicMapError("the points must hold finite values only")
    return matrix


def _labels(labels: Any) -> np.ndarray:
    array = label_array(np.asarray(labels))  # a fresh int64 copy
    array.setflags(write=False)
    return array
