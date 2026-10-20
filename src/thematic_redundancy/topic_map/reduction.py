"""UMAP reduction and the 2-D projection of the visualization map (T16, D30 E1).

- **Reduction** (:func:`fit_reduction`): the space that the semantic runs cluster. One UMAP fit
  per (model, seed), with the ``umap`` block's settings (15 neighbours, 5 components,
  ``min_dist`` 0.0, cosine: BERTopic 0.17.4's defaults) and the seed as ``random_state``; the
  experiment shares it between K-means and HDBSCAN, so their comparison is paired. The fit is
  unsupervised: ``fit_transform`` receives the vectors and nothing else, never labels.
- **Projection** (:func:`project`): the 2-D coordinates of the visualization map, from the
  original vectors, never from the reduction. ``umap`` fits a UMAP of its own with two
  components; ``pca`` (thesis plan section 3.6.3) uses scikit-learn's PCA. Both take the run's
  seed. No metric or selection reads the projection.

With a ``random_state``, umap-learn runs single-threaded; asking for ``n_jobs=1`` says so and
avoids its warning. UMAP takes about 12 s to import, so :func:`umap_model` imports it on first
use and tests pass their own factory.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self

import numpy as np
from scipy import sparse

from thematic_redundancy.shared.config import ProjectionConfig, UmapConfig
from thematic_redundancy.topic_map import TopicMapError

ProjectionMethod = Literal["umap", "pca"]

PROJECTION_COMPONENTS = 2


@dataclass(frozen=True)
class UmapSettings:
    """The UMAP arguments of a fit, besides its seed."""

    n_neighbors: int
    n_components: int
    min_dist: float
    metric: str

    @classmethod
    def from_config(cls, config: UmapConfig) -> Self:
        return cls(
            n_neighbors=config.n_neighbors,
            n_components=config.n_components,
            min_dist=config.min_dist,
            metric=config.metric,
        )


@dataclass(frozen=True)
class ProjectionSettings:
    """How the 2-D coordinates are computed (``topic_map.projection``)."""

    method: ProjectionMethod
    n_neighbors: int
    """For ``umap`` only, as are :attr:`min_dist` and :attr:`metric`."""
    min_dist: float
    metric: str

    @classmethod
    def from_config(cls, config: ProjectionConfig) -> Self:
        return cls(
            method=config.method,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
        )


class SupportsFitTransform(Protocol):
    def fit_transform(self, X: Any) -> Any: ...


UmapFactory = Callable[[UmapSettings, int], SupportsFitTransform]
"""Port: return an unfitted UMAP model for the settings and the seed."""


def umap_model(settings: UmapSettings, seed: int) -> SupportsFitTransform:
    """Return umap-learn's ``UMAP`` with ``settings`` and ``random_state=seed``."""
    from umap import UMAP

    return UMAP(
        n_neighbors=settings.n_neighbors,
        n_components=settings.n_components,
        min_dist=settings.min_dist,
        metric=settings.metric,
        random_state=seed,
        n_jobs=1,
    )


def pca_model(seed: int) -> Any:
    """Return scikit-learn's ``PCA`` with two components and ``random_state=seed``."""
    from sklearn.decomposition import PCA

    return PCA(n_components=PROJECTION_COMPONENTS, random_state=seed)


@dataclass(frozen=True)
class Reduction:
    """One UMAP fit: the space that the clusterings of a (model, seed) share."""

    seed: int
    settings: UmapSettings
    embedding: np.ndarray
    """``points × n_components``, as UMAP returned it, read-only."""


def dense_vectors(vectors: Any) -> np.ndarray:
    """Return ``vectors`` as a float64 matrix after checking that it is dense, 2-D, finite and
    of at least 2 rows.

    Raises:
        TopicMapError: otherwise.
    """
    if sparse.issparse(vectors):
        raise TopicMapError("the document vectors must be a dense matrix, not a sparse one")
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise TopicMapError(f"the document vectors must be 2-D, not of shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise TopicMapError(f"the document vectors need at least 2 rows, not {matrix.shape[0]}")
    if not np.isfinite(matrix).all():
        raise TopicMapError("the document vectors must hold finite values only")
    return matrix


def fit_reduction(
    vectors: Any, settings: UmapSettings, seed: int, *, umap_factory: UmapFactory = umap_model
) -> Reduction:
    """Fit UMAP on ``vectors``, unsupervised, and return the reduction.

    Raises:
        TopicMapError: if the vectors are invalid (see :func:`dense_vectors`), or UMAP returns
            another shape or a value that is not finite.
    """
    matrix = dense_vectors(vectors)
    model = umap_factory(settings, seed)
    embedding = np.array(model.fit_transform(matrix))
    expected = (matrix.shape[0], settings.n_components)
    if embedding.shape != expected:
        raise TopicMapError(f"UMAP returned shape {embedding.shape}, not {expected}")
    if not np.isfinite(embedding).all():
        raise TopicMapError("UMAP returned values that are not finite")
    return Reduction(seed=seed, settings=settings, embedding=_read_only(embedding))


def project(
    vectors: Any,
    settings: ProjectionSettings,
    seed: int,
    *,
    umap_factory: UmapFactory = umap_model,
) -> np.ndarray:
    """Return the 2-D coordinates of ``vectors`` for the visualization map, read-only.

    Raises:
        TopicMapError: if the vectors are invalid (see :func:`dense_vectors`).
    """
    if settings.method == "umap":
        umap_settings = UmapSettings(
            n_neighbors=settings.n_neighbors,
            n_components=PROJECTION_COMPONENTS,
            min_dist=settings.min_dist,
            metric=settings.metric,
        )
        return fit_reduction(vectors, umap_settings, seed, umap_factory=umap_factory).embedding
    coordinates = np.array(pca_model(seed).fit_transform(dense_vectors(vectors)))
    return _read_only(coordinates)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array
