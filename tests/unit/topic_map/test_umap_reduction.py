"""The UMAP reduction and the 2-D projection of the visualization map (T16, D30 E1).

UMAP is replaced by a recording stand-in (``conftest.py``), or by a fake ``umap`` module where
the adapter itself is under test; the smoke module runs the real UMAP.
"""

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA

from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.reduction import (
    ProjectionSettings,
    Reduction,
    UmapSettings,
    dense_vectors,
    fit_reduction,
    pca_model,
    project,
    umap_model,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

SETTINGS = UmapSettings(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
UMAP_PROJECTION = ProjectionSettings(method="umap", n_neighbors=10, min_dist=0.1, metric="cosine")
PCA_PROJECTION = ProjectionSettings(method="pca", n_neighbors=15, min_dist=0.0, metric="cosine")


@pytest.fixture
def vectors() -> np.ndarray:
    return np.random.default_rng(3).normal(size=(40, 12))


@pytest.fixture
def fake_umap_module(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a ``umap`` module whose ``UMAP`` records its keyword arguments."""
    created: list[dict[str, Any]] = []

    class UMAP:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    module = types.ModuleType("umap")
    module.UMAP = UMAP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umap", module)
    return created


def test_the_settings_follow_the_configuration() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    assert UmapSettings.from_config(config.umap) == SETTINGS
    assert ProjectionSettings.from_config(config.topic_map.projection) == ProjectionSettings(
        method="umap", n_neighbors=15, min_dist=0.0, metric="cosine"
    )


def test_the_umap_model_takes_the_settings_and_the_seed(fake_umap_module: list) -> None:
    umap_model(SETTINGS, 42)

    assert fake_umap_module == [
        {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
            "n_jobs": 1,
        }
    ]


def test_the_reduction_is_fitted_on_the_vectors_alone(
    recording_umap: Any, vectors: np.ndarray
) -> None:
    fit_reduction(vectors, SETTINGS, 13, umap_factory=recording_umap)

    (call,) = recording_umap.calls
    assert call.settings == SETTINGS
    assert call.seed == 13
    # Unsupervised: fit_transform gets the vectors and nothing else, never y.
    assert len(call.args) == 1
    assert call.kwargs == {}
    np.testing.assert_array_equal(call.args[0], vectors)


def test_the_reduction_records_its_seed_settings_and_embedding(
    recording_umap: Any, vectors: np.ndarray
) -> None:
    reduction = fit_reduction(vectors, SETTINGS, 21, umap_factory=recording_umap)

    assert isinstance(reduction, Reduction)
    assert (reduction.seed, reduction.settings) == (21, SETTINGS)
    assert reduction.embedding.shape == (40, 5)
    assert not reduction.embedding.flags.writeable


def test_a_reduction_repeats_for_a_seed(recording_umap: Any, vectors: np.ndarray) -> None:
    first = fit_reduction(vectors, SETTINGS, 7, umap_factory=recording_umap)
    again = fit_reduction(vectors, SETTINGS, 7, umap_factory=recording_umap)
    other = fit_reduction(vectors, SETTINGS, 73, umap_factory=recording_umap)

    np.testing.assert_array_equal(first.embedding, again.embedding)
    assert not np.array_equal(first.embedding, other.embedding)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.zeros(5), "2-D"),
        (np.zeros((1, 4)), "at least 2"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), "finite"),
        (sparse.csr_matrix(np.eye(3)), "dense"),
    ],
)
def test_invalid_vectors_are_refused(recording_umap: Any, bad: Any, message: str) -> None:
    with pytest.raises(TopicMapError, match=message):
        fit_reduction(bad, SETTINGS, 7, umap_factory=recording_umap)
    assert recording_umap.calls == []


def test_dense_vectors_returns_a_float_matrix() -> None:
    matrix = dense_vectors([[1, 2], [3, 4]])

    assert matrix.dtype == np.float64
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize(
    "output",
    [np.zeros((40, 4)), np.zeros((39, 5)), np.full((40, 5), np.inf)],
)
def test_a_reduction_of_another_shape_or_with_non_finite_values_is_refused(
    vectors: np.ndarray, output: np.ndarray
) -> None:
    class Broken:
        def fit_transform(self, X: Any) -> np.ndarray:
            return output

    with pytest.raises(TopicMapError, match="UMAP"):
        fit_reduction(vectors, SETTINGS, 7, umap_factory=lambda settings, seed: Broken())


def test_the_umap_projection_is_a_2d_fit_of_its_own(
    recording_umap: Any, vectors: np.ndarray
) -> None:
    coordinates = project(vectors, UMAP_PROJECTION, 42, umap_factory=recording_umap)

    (call,) = recording_umap.calls
    assert call.settings == UmapSettings(
        n_neighbors=10, n_components=2, min_dist=0.1, metric="cosine"
    )
    assert call.seed == 42
    assert call.kwargs == {}
    assert coordinates.shape == (40, 2)
    assert not coordinates.flags.writeable


def test_the_pca_model_takes_two_components_and_the_seed() -> None:
    params = pca_model(73).get_params()

    assert (params["n_components"], params["random_state"]) == (2, 73)


def test_the_pca_projection_is_scikit_learn_s_pca_and_fits_no_umap(
    recording_umap: Any, vectors: np.ndarray
) -> None:
    coordinates = project(vectors, PCA_PROJECTION, 21, umap_factory=recording_umap)

    expected = PCA(n_components=2, random_state=21).fit_transform(vectors)
    np.testing.assert_allclose(coordinates, expected, rtol=1e-12, atol=1e-12)
    assert recording_umap.calls == []
    assert not coordinates.flags.writeable


def test_the_projection_refuses_invalid_vectors(recording_umap: Any) -> None:
    with pytest.raises(TopicMapError, match="finite"):
        project(np.array([[np.inf, 0.0], [1.0, 1.0]]), PCA_PROJECTION, 7)
