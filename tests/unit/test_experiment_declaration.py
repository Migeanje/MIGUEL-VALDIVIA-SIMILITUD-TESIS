"""The experiment declaration of T18 (D30): the ``experiment`` block of ``config/default.yaml``,
together with the values that it reuses from the other blocks.

The declaration was fixed on 2026-10-19, before any run of the experiment: T19 had not run, and
no clustering, topic model, UMAP or metric had touched a thesis vector or text. These tests pin
it, as the D26 and U4 tests pin the stopword list and the career-affinity matrix, and check that
the configured grid gives the run counts that D30 derived.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from thematic_redundancy.shared.config import AppConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

D30_DECLARATION_SHA256 = "b974c54c3378c8fe02f7983323f0d58fa7672e4ca7d1999df15f708d2b76850b"
"""SHA-256 of the declaration decided on 2026-10-19, as the canonical JSON of
:func:`canonical_json`."""


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(DEFAULT_CONFIG_PATH)


def canonical_json(config: AppConfig) -> str:
    """Return the declaration as canonical JSON: keys sorted, no whitespace, UTF-8 text.

    It is built here from the loaded configuration, independently of the configuration code: the
    whole ``experiment`` block, and from the other blocks only the values the experiment reuses.
    """
    declaration: dict[str, Any] = {
        "embedding": {
            "models": list(config.embedding.models),
            "document_vector": config.embedding.document_vector,
        },
        "experiment": config.experiment.model_dump(mode="json"),
        "hdbscan": {"min_cluster_sizes": list(config.hdbscan.min_cluster_sizes)},
        "kmeans": {
            "k_start": config.kmeans.k_start,
            "k_stop": config.kmeans.k_stop,
            "k_step": config.kmeans.k_step,
        },
        "seeds": list(config.seeds),
        "tfidf": {"document_vector": config.tfidf.document_vector},
        "umap": {
            "n_neighbors": config.umap.n_neighbors,
            "n_components": config.umap.n_components,
            "min_dist": config.umap.min_dist,
            "metric": config.umap.metric,
        },
    }
    return json.dumps(declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_the_declaration_is_frozen_as_decided_in_d30(config: AppConfig) -> None:
    """The declaration was fixed on 2026-10-19, before any run, as decision D30 of the plan.

    The pin covers the loaded values, so a comment, a blank line or the order of the keys in the
    YAML file keeps it, and any changed value breaks it. A deliberate change needs a new decision
    in the plan, made before the runs it affects, and updates this value together with it.
    """
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()

    assert digest == D30_DECLARATION_SHA256


def test_the_configuration_hashes_the_same_declaration(config: AppConfig) -> None:
    # T19 writes this hash into its manifest as declaration_sha256, so it must be the pinned one.
    declaration = config.experiment_declaration()

    assert json.dumps(
        declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) == canonical_json(config)
    assert config.experiment_declaration_sha256() == D30_DECLARATION_SHA256


def test_the_grid_gives_the_run_counts_of_d30(config: AppConfig) -> None:
    models = len(config.embedding.models)
    seeds = len(config.seeds)
    k_values = len(config.kmeans.k_values())
    sizes = len(config.hdbscan.min_cluster_sizes)
    # The semantic runs cluster the UMAP output; the baseline is K-means on the sparse matrix.
    assert config.experiment.clustering_space == "umap"
    assert config.experiment.baseline.algorithm == "kmeans"

    umap_fits = models * seeds
    sbert_kmeans = models * k_values * seeds
    hdbscan = models * sizes * seeds
    tfidf_kmeans = k_values * seeds

    assert (models, seeds, k_values, sizes) == (2, 5, 22, 4)
    assert umap_fits == 10
    assert (sbert_kmeans, hdbscan, tfidf_kmeans) == (220, 40, 110)
    assert sbert_kmeans + hdbscan + tfidf_kmeans == 370
    # Five configurations, 2 models × 2 algorithms plus the baseline, each run with every seed;
    # the at most 5 matched-K runs of the baseline come on top of both counts.
    assert (models * 2 + 1) * seeds == 25


def test_the_fallback_model_is_one_of_the_compared_models(config: AppConfig) -> None:
    # The loader does not tie the two blocks, since tests replace the models with fakes; the
    # declared file must, and T19 checks it again before any run.
    assert config.experiment.selection.fallbacks.model in config.embedding.models
