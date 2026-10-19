"""Internal metrics of one clustering (T17, D12; D28 C4), on synthetic data of known structure.

Well-separated blobs lie on orthogonal axes, so they are apart in angle as well as in distance;
random labels on random directions carry no structure. Every value is also checked against
scikit-learn called directly on the clustered rows (and their L2-normalized copy for DB and CH).
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from thematic_redundancy.evaluation.clustering_metrics import (
    NOISE_LABEL,
    InternalMetrics,
    internal_metrics,
    label_array,
)

SEED = 20261019


def blobs(
    clusters: int = 4, per_cluster: int = 50, dimensions: int = 12, spread: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Return points around ``10 * e_k`` for each cluster ``k``, and their labels."""
    rng = np.random.default_rng(SEED)
    centers = np.eye(dimensions)[:clusters] * 10
    vectors = np.vstack([c + rng.normal(0, spread, (per_cluster, dimensions)) for c in centers])
    return vectors, np.repeat(np.arange(clusters), per_cluster)


def random_points(points: int = 600, dimensions: int = 20, clusters: int = 4, seed: int = SEED):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(points, dimensions)), rng.integers(0, clusters, points)


def with_noise(vectors: np.ndarray, labels: np.ndarray, noise: int) -> tuple[np.ndarray, ...]:
    """Insert ``noise`` random points labelled as noise among the clustered ones."""
    rng = np.random.default_rng(SEED + 1)
    extra = rng.normal(size=(noise, vectors.shape[1])) * 5
    order = rng.permutation(vectors.shape[0] + noise)
    all_vectors = np.vstack([vectors, extra])[order]
    all_labels = np.concatenate([labels, np.full(noise, NOISE_LABEL)])[order]
    return all_vectors, all_labels


def metric_values(metrics: InternalMetrics) -> tuple[float | None, ...]:
    return (
        metrics.silhouette_cosine,
        metrics.davies_bouldin_l2,
        metrics.calinski_harabasz_l2,
    )


def assert_same_metrics(actual: InternalMetrics, expected: InternalMetrics) -> None:
    for got, want in zip(metric_values(actual), metric_values(expected), strict=True):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


# --- Known structure ---------------------------------------------------------------------


def test_well_separated_blobs_score_as_well_separated() -> None:
    vectors, labels = blobs()

    metrics = internal_metrics(vectors, labels)

    assert metrics.undefined_reason is None
    assert metrics.points == 200
    assert metrics.clusters == 4
    assert metrics.noise_points == 0
    assert metrics.noise_fraction == 0.0
    assert metrics.silhouette_cosine > 0.99
    assert metrics.davies_bouldin_l2 < 0.15
    assert metrics.calinski_harabasz_l2 > 5000


def test_random_labels_score_as_no_structure() -> None:
    vectors, labels = random_points()

    metrics = internal_metrics(vectors, labels)

    assert abs(metrics.silhouette_cosine) < 0.05
    assert metrics.davies_bouldin_l2 > 5
    assert metrics.calinski_harabasz_l2 < 5


def test_each_metric_equals_scikit_learn_on_the_clustered_rows() -> None:
    vectors, labels = with_noise(*random_points(points=300, clusters=3), noise=40)
    kept = labels != NOISE_LABEL
    unit_rows = normalize(vectors[kept])

    metrics = internal_metrics(vectors, labels)

    assert metrics.silhouette_cosine == pytest.approx(
        silhouette_score(vectors[kept], labels[kept], metric="cosine"), rel=1e-12
    )
    assert metrics.davies_bouldin_l2 == pytest.approx(
        davies_bouldin_score(unit_rows, labels[kept]), rel=1e-12
    )
    assert metrics.calinski_harabasz_l2 == pytest.approx(
        calinski_harabasz_score(unit_rows, labels[kept]), rel=1e-12
    )


# --- L2 normalization and the cosine distance --------------------------------------------


def test_rescaling_each_point_changes_no_metric() -> None:
    """Silhouette uses the cosine and DB/CH the L2-normalized rows, so a positive factor per
    point changes none of them, while it does change DB and CH on the raw rows."""
    vectors, labels = random_points(points=200, clusters=3)
    factors = np.random.default_rng(SEED + 2).uniform(0.1, 10.0, size=(200, 1))
    scaled = vectors * factors

    original = internal_metrics(vectors, labels)
    rescaled = internal_metrics(scaled, labels)

    assert_same_metrics(rescaled, original)
    # The data discriminate: on the raw rows, the factors move both metrics and the Euclidean
    # silhouette.
    assert davies_bouldin_score(scaled, labels) != pytest.approx(original.davies_bouldin_l2)
    assert calinski_harabasz_score(scaled, labels) != pytest.approx(original.calinski_harabasz_l2)
    assert silhouette_score(scaled, labels) != pytest.approx(original.silhouette_cosine)


def test_the_silhouette_uses_the_cosine_distance() -> None:
    """Two clusters that differ in direction but overlap in Euclidean distance."""
    rng = np.random.default_rng(SEED + 3)
    angles = np.concatenate([rng.normal(0.3, 0.05, 30), rng.normal(1.2, 0.05, 30)])
    radii = rng.uniform(0.5, 20.0, 60)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
    labels = np.repeat([0, 1], 30)

    metrics = internal_metrics(vectors, labels)

    assert metrics.silhouette_cosine == pytest.approx(
        silhouette_score(vectors, labels, metric="cosine"), rel=1e-12
    )
    assert metrics.silhouette_cosine > 0.9
    assert silhouette_score(vectors, labels, metric="euclidean") < 0.5


# --- Sparse input ------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix])
def test_sparse_and_dense_inputs_give_the_same_metrics(matrix_type) -> None:
    rng = np.random.default_rng(SEED + 4)
    dense = rng.random((120, 40)) * (rng.random((120, 40)) < 0.2)
    dense[:, 0] += 0.01  # no zero row
    labels = rng.integers(-1, 3, 120)

    from_dense = internal_metrics(dense, labels)
    from_sparse = internal_metrics(matrix_type(dense), labels)

    assert from_sparse.noise_points == from_dense.noise_points
    assert from_sparse.clusters == from_dense.clusters
    assert_same_metrics(from_sparse, from_dense)


# --- Noise -------------------------------------------------------------------------------


def test_noise_points_change_only_the_noise_share() -> None:
    vectors, labels = blobs(clusters=3, per_cluster=20)
    noisy_vectors, noisy_labels = with_noise(vectors, labels, noise=15)

    clean = internal_metrics(vectors, labels)
    noisy = internal_metrics(noisy_vectors, noisy_labels)

    assert_same_metrics(noisy, clean)
    assert noisy.clusters == clean.clusters == 3
    assert (clean.points, clean.noise_points, clean.noise_fraction) == (60, 0, 0.0)
    assert (noisy.points, noisy.noise_points) == (75, 15)
    assert noisy.noise_fraction == pytest.approx(15 / 75, rel=0, abs=1e-15)
    assert noisy.clustered_points == 60


def test_the_noise_share_is_over_all_points() -> None:
    vectors = np.eye(5)[[0, 0, 1, 1, 2, 3, 4, 2]] + 0.01
    labels = [0, 0, 1, 1, -1, -1, -1, -1]

    metrics = internal_metrics(vectors, labels)

    assert metrics.noise_points == 4
    assert metrics.noise_fraction == 0.5


# --- Invariance and determinism ----------------------------------------------------------


def test_renaming_the_clusters_changes_nothing() -> None:
    vectors, labels = with_noise(*random_points(points=150, clusters=3), noise=10)
    renamed = np.select([labels == 0, labels == 1, labels == 2], [7, 3, 12], default=NOISE_LABEL)

    assert internal_metrics(vectors, renamed) == internal_metrics(vectors, labels)


def test_the_same_input_gives_the_same_result() -> None:
    vectors, labels = with_noise(*random_points(points=150, clusters=3), noise=10)

    first = internal_metrics(vectors, labels)
    second = internal_metrics(vectors.copy(), list(labels))

    assert first == second


# --- Undefined metrics: None with a reason -----------------------------------------------


def assert_undefined(metrics: InternalMetrics, reason: str) -> None:
    assert metrics.undefined_reason == reason
    assert metric_values(metrics) == (None, None, None)


def test_an_empty_input_has_no_metric_and_no_noise_share() -> None:
    metrics = internal_metrics(np.empty((0, 3)), [])

    assert_undefined(metrics, "no_points")
    assert (metrics.points, metrics.noise_points, metrics.clusters) == (0, 0, 0)
    assert metrics.noise_fraction is None


def test_an_empty_sparse_input_has_no_metric() -> None:
    assert_undefined(internal_metrics(sparse.csr_matrix((0, 3)), np.array([], int)), "no_points")


def test_all_noise_has_no_metric_but_reports_the_noise() -> None:
    metrics = internal_metrics(np.eye(3) + 0.1, [-1, -1, -1])

    assert_undefined(metrics, "all_noise")
    assert (metrics.points, metrics.noise_points, metrics.clusters) == (3, 3, 0)
    assert metrics.noise_fraction == 1.0


@pytest.mark.parametrize("labels", [[4, 4, 4, 4], [4, -1, 4, -1]])
def test_one_cluster_has_no_metric(labels: list[int]) -> None:
    metrics = internal_metrics(np.eye(4) + 0.1, labels)

    assert_undefined(metrics, "one_cluster")
    assert metrics.clusters == 1


@pytest.mark.parametrize("labels", [[0, 1, 2], [0, -1, 1, 2, -1]])
def test_one_point_per_cluster_has_no_metric(labels: list[int]) -> None:
    metrics = internal_metrics(np.eye(5)[: len(labels)] + 0.1, labels)

    assert_undefined(metrics, "one_point_per_cluster")
    assert metrics.clusters == 3


def test_two_clusters_over_three_points_are_defined() -> None:
    """The smallest defined case: 2 clusters, one more point than clusters."""
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    metrics = internal_metrics(vectors, [0, 0, 1])

    assert metrics.undefined_reason is None
    assert None not in metric_values(metrics)


# --- Invalid input raises ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("vectors", "labels", "message"),
    [
        (np.ones((3, 2)), [0, 1], "3 rows but 2 labels"),
        (np.ones(3), [0, 1, 0], "2-D"),
        (np.array([[1.0, 0.0], [np.nan, 1.0], [0.0, 1.0]]), [0, 1, 1], "finite"),
        (np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]), [0, 1, 1], "zero vector"),
        (sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])), [0, 1, 1], "zero"),
        (sparse.csr_matrix(np.array([[1.0, np.inf], [0.0, 1.0]])), [0, 1], "finite"),
    ],
)
def test_invalid_vectors_raise(vectors, labels, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        internal_metrics(vectors, labels)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([0, -2, 1], "-1 or above"),
        ([0.0, 1.5, 1.0], "integers"),
        ([[0, 1, 1]], "1-D"),
        (["a", "b", "a"], "integers"),
    ],
)
def test_invalid_labels_raise(labels, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        label_array(labels)


def test_label_array_accepts_an_empty_list_and_integer_arrays() -> None:
    assert label_array([]).dtype == np.int64
    assert label_array([]).size == 0
    np.testing.assert_array_equal(label_array(np.array([2, -1], dtype=np.int8)), [2, -1])
