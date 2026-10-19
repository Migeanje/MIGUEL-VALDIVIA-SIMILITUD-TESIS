"""Internal metrics of one clustering (T17, D12; D28 C4).

- **Silhouette** with the cosine distance, on the vectors as given, dense or sparse (D12: on the
  original embeddings).
- **Davies-Bouldin** and **Calinski-Harabasz** on the L2-normalized vectors, since
  scikit-learn computes both with the Euclidean distance only (D12). Both need a dense array,
  so a sparse input is normalized, then densified for them: ``points × dimensions × 8`` bytes,
  about 85 MB for the 744 × 14,308 TF-IDF matrix (D28 C4).
- **Noise.** Points labelled :data:`NOISE_LABEL`, as HDBSCAN labels its noise, are left out of
  all three metrics; their count and their share of all the points are reported.

Undefined metrics follow one convention across :mod:`thematic_redundancy.evaluation`: for a
valid input whose metric has no value, the metric is ``None`` and ``undefined_reason`` says
why; an invalid input raises ``ValueError``. Here the three metrics are undefined together, as
scikit-learn defines them only for 2 to ``n - 1`` clusters over ``n`` points, and the reason is
the first that applies of:

- ``"no_points"``: the input is empty;
- ``"all_noise"``: every point is noise;
- ``"one_cluster"``: the clustered points form a single cluster;
- ``"one_point_per_cluster"``: there are as many clusters as clustered points.

Every value is computed exactly, without sampling, so the same input gives the same result.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

NOISE_LABEL = -1
"""The label of a point that belongs to no cluster, as HDBSCAN gives it."""

ClusteringUndefined = Literal["no_points", "all_noise", "one_cluster", "one_point_per_cluster"]


@dataclass(frozen=True)
class InternalMetrics:
    """The internal metrics of one labeling, over its clustered points."""

    points: int
    """All the points, noise included."""
    noise_points: int
    noise_fraction: float | None
    """``noise_points / points``; ``None`` when there is no point."""
    clusters: int
    """Distinct labels among the clustered points."""
    silhouette_cosine: float | None
    davies_bouldin_l2: float | None
    calinski_harabasz_l2: float | None
    undefined_reason: ClusteringUndefined | None
    """Why the three metrics are ``None``; ``None`` when they are defined."""

    @property
    def clustered_points(self) -> int:
        return self.points - self.noise_points


def label_array(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return ``labels`` as a 1-D int64 array.

    Raises:
        ValueError: if the labels are not a 1-D sequence of integers of at least
            :data:`NOISE_LABEL`.
    """
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"the labels must be 1-D, not of shape {array.shape}")
    if array.size == 0:
        return np.zeros(0, dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError(f"the labels must be integers, not {array.dtype}")
    if array.min() < NOISE_LABEL:
        raise ValueError(f"the labels must be {NOISE_LABEL} or above, not {array.min()}")
    return array.astype(np.int64)


def internal_metrics(
    vectors: np.ndarray | sparse.spmatrix, labels: Sequence[int] | np.ndarray
) -> InternalMetrics:
    """Return the silhouette (cosine), Davies-Bouldin and Calinski-Harabasz (both on the
    L2-normalized vectors) of the clustered points, with the noise share.

    Args:
        vectors: one row per point, dense or sparse, finite, with no zero row (a zero vector
            has no direction, so no cosine).
        labels: one cluster label per row; :data:`NOISE_LABEL` marks noise.

    Raises:
        ValueError: if the vectors or the labels are invalid, or their lengths differ.
    """
    matrix = _check_vectors(vectors)
    label_values = label_array(labels)
    if matrix.shape[0] != label_values.size:
        raise ValueError(
            f"the vectors have {matrix.shape[0]} rows but {label_values.size} labels were given"
        )
    clustered = np.flatnonzero(label_values != NOISE_LABEL)
    points = label_values.size
    noise_points = points - clustered.size
    kept_labels = label_values[clustered]
    clusters = int(np.unique(kept_labels).size)
    reason = _undefined_reason(points, clustered.size, clusters)
    metrics = InternalMetrics(
        points=points,
        noise_points=noise_points,
        noise_fraction=noise_points / points if points else None,
        clusters=clusters,
        silhouette_cosine=None,
        davies_bouldin_l2=None,
        calinski_harabasz_l2=None,
        undefined_reason=reason,
    )
    if reason is not None:
        return metrics
    kept = matrix[clustered]
    unit_rows = normalize(kept, norm="l2")
    dense = unit_rows.toarray() if sparse.issparse(unit_rows) else unit_rows
    return replace(
        metrics,
        silhouette_cosine=float(silhouette_score(kept, kept_labels, metric="cosine")),
        davies_bouldin_l2=float(davies_bouldin_score(dense, kept_labels)),
        calinski_harabasz_l2=float(calinski_harabasz_score(dense, kept_labels)),
    )


def _undefined_reason(points: int, clustered: int, clusters: int) -> ClusteringUndefined | None:
    if points == 0:
        return "no_points"
    if clustered == 0:
        return "all_noise"
    if clusters == 1:
        return "one_cluster"
    if clusters == clustered:
        return "one_point_per_cluster"
    return None


def _check_vectors(vectors: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.csr_matrix:
    """Return ``vectors`` as float64, CSR when sparse, after checking that they are a 2-D
    matrix of finite values with no zero row."""
    if sparse.issparse(vectors):
        matrix = sparse.csr_matrix(vectors, dtype=np.float64)
        values = matrix.data
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    else:
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"the vectors must be a 2-D matrix, not of shape {matrix.shape}")
        values = matrix
        norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.zeros(matrix.shape[0])
    if not np.isfinite(values).all():
        raise ValueError("the vectors must hold finite values only")
    if np.any(norms == 0):
        raise ValueError(
            f"row {int(np.flatnonzero(norms == 0)[0])} is a zero vector, which has no cosine"
        )
    return matrix
