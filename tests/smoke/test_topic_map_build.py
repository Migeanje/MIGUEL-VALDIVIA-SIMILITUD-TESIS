"""Smoke checks of the topic map (T16) with the real UMAP, HDBSCAN, K-means and BERTopic.

The unit tests replace UMAP and BERTopic, which take about 12 s and 50 s to import; these tests
run them on synthetic data only: Spanish-like theses drawn from fixed theme lists, and vectors
drawn around one axis per theme, plus three far outliers. No thesis of the corpus is read.

They pin what the stand-ins assume (D30 E1, E4; D22):

- the reduction is umap-learn's UMAP fitted unsupervised: byte-identical to a direct
  ``fit_transform`` without ``y``, unlike one with ``y``, and repeatable for a seed;
- BERTopic's manual mode, as the adapter builds it, gives the same c-TF-IDF as BERTopic's
  ``ClassTfidfTransformer()`` on classes counted directly, keeps accents, and never holds a real
  UMAP;
- the whole map repeats for a seed and feeds T17's metrics.

Heavy libraries are imported inside the tests, so collecting this module stays fast. The three
outliers are far from every theme in the original space, but UMAP, which links every point to
15 neighbours, may still place them inside a cluster.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from thematic_redundancy.evaluation.clustering_metrics import internal_metrics
from thematic_redundancy.evaluation.coherence import npmi_coherence
from thematic_redundancy.evaluation.stability import seed_stability
from thematic_redundancy.representation.tfidf import TfidfSettings, fit_tfidf, section_terms
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map.bertopic_ctfidf import (
    BertopicCtfidf,
    build_topic_model,
    load_bertopic_components,
)
from thematic_redundancy.topic_map.build import TopicMapSettings, build_topic_maps
from thematic_redundancy.topic_map.clustering import HdbscanRun, KMeansRun
from thematic_redundancy.topic_map.ctfidf import ctfidf_topics, encode_document
from thematic_redundancy.topic_map.reduction import fit_reduction, umap_model

pytestmark = pytest.mark.smoke

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

THEMES = (
    ("voladura", "explosivo", "fragmentación", "macizo", "perforación", "detonador", "diseño"),
    ("mantenimiento", "rodamiento", "disponibilidad", "falla", "motor", "repuesto", "lubricación"),
    ("inventario", "módulo", "datos", "facturación", "usuario", "aplicación", "servidor"),
)
SHARED = ("empresa", "proceso", "análisis", "propuesta")
PER_THEME = 20
OUTLIERS = 3


def make_corpus() -> tuple[tuple, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return the theses' sections, their vectors, their themes (-1 for an outlier) and T15's
    vocabulary over them."""
    rng = np.random.default_rng(20261019)
    dimensions = 32
    documents, vectors, themes = [], [], []
    for index, words in enumerate(THEMES):
        for _ in range(PER_THEME):
            documents.append(thesis(rng, words))
            vectors.append(np.eye(dimensions)[index] * 10 + rng.normal(0, 0.4, dimensions))
            themes.append(index)
    for index in range(OUTLIERS):
        documents.append(thesis(rng, THEMES[index]))
        vectors.append(np.eye(dimensions)[len(THEMES) + index] * 40)
        themes.append(-1)
    settings = TfidfSettings(max_ngram=2, min_df=2, max_df=1.0, sublinear_tf=True)
    vocabulary = fit_tfidf(documents, settings).terms
    return tuple(documents), np.vstack(vectors), np.asarray(themes), vocabulary


def thesis(rng: np.random.Generator, words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            str(rng.choice(words)) if rng.random() < 0.8 else str(rng.choice(SHARED))
            for _ in range(length)
        )
        for length in (3, 9, 5)
    )


@pytest.fixture(scope="module")
def corpus() -> tuple[tuple, np.ndarray, np.ndarray, tuple[str, ...]]:
    return make_corpus()


@pytest.fixture(scope="module")
def settings() -> TopicMapSettings:
    return TopicMapSettings.from_config(load_config(DEFAULT_CONFIG_PATH))


def test_the_reduction_is_umap_fitted_unsupervised_and_repeatable(
    corpus: tuple, settings: TopicMapSettings
) -> None:
    from umap import UMAP

    _, vectors, themes, _ = corpus
    params = umap_model(settings.umap, 7).get_params()
    assert isinstance(umap_model(settings.umap, 7), UMAP)
    assert (params["n_neighbors"], params["n_components"]) == (15, 5)
    assert (params["min_dist"], params["metric"], params["random_state"]) == (0.0, "cosine", 7)

    reduction = fit_reduction(vectors, settings.umap, 7)
    again = fit_reduction(vectors, settings.umap, 7)

    unsupervised = umap_model(settings.umap, 7).fit_transform(vectors)
    supervised = umap_model(settings.umap, 7).fit_transform(vectors, y=themes)
    assert reduction.embedding.tobytes() == unsupervised.tobytes()
    assert reduction.embedding.tobytes() == again.embedding.tobytes()
    assert not np.allclose(reduction.embedding, supervised)


def test_bertopic_s_manual_mode_equals_class_tfidf_on_directly_counted_classes(
    corpus: tuple, settings: TopicMapSettings
) -> None:
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer

    documents, _, themes, vocabulary = corpus
    # Sizes that do not follow the ids, and noise, so that BERTopic renumbers the topics.
    labels = np.where(themes == 0, 2, np.where(themes == 1, 0, 1))
    labels[[0, 1, 2, 3, 40, 41]] = -1

    result = BertopicCtfidf().class_weights(
        [encode_document(document) for document in documents], labels, vocabulary, 2
    )

    classes = sorted(set(labels.tolist()))
    column = {term: index for index, term in enumerate(vocabulary)}
    counts = np.zeros((len(classes), len(vocabulary)))
    for label, document in zip(labels.tolist(), documents, strict=True):
        for section in document:
            for term in section_terms(section, 2):
                if term in column:
                    counts[classes.index(label), column[term]] += 1
    direct = ClassTfidfTransformer().fit(sparse.csr_matrix(counts))
    expected = direct.transform(sparse.csr_matrix(counts)).toarray()
    assert result.classes == tuple(classes) == (-1, 0, 1, 2)
    np.testing.assert_allclose(result.weights.toarray(), expected, rtol=1e-12, atol=0)

    model = build_topic_model(load_bertopic_components(), vocabulary, 2)
    assert model.language == "spanish"
    assert type(model.umap_model) is BaseDimensionalityReduction
    assert type(model.hdbscan_model) is BaseCluster

    topics = ctfidf_topics(
        labels,
        documents,
        vocabulary=vocabulary,
        max_ngram=2,
        top_n=settings.top_n,
        include_noise_class=True,
        backend=BertopicCtfidf(),
    )
    mining = next(topic for topic in topics.topics if topic.topic == 2)
    assert set(mining.terms) & {"fragmentación", "perforación", "diseño"}
    assert not any("fragmentacin" in term for topic in topics.topics for term in topic.terms)


def test_the_topic_maps_repeat_for_a_seed_and_feed_the_metrics(
    corpus: tuple, settings: TopicMapSettings
) -> None:
    from hdbscan import HDBSCAN

    documents, vectors, _, vocabulary = corpus
    runs = (KMeansRun(n_clusters=3), HdbscanRun(min_cluster_size=5))

    def maps(seed: int) -> tuple:
        return build_topic_maps(
            vectors, documents, runs, vocabulary=vocabulary, seed=seed, settings=settings
        )

    first, again, other = maps(7), maps(7), maps(13)

    for one, two in zip(first, again, strict=True):
        assert one.labels.tobytes() == two.labels.tobytes()
        assert one.centroids.tobytes() == two.centroids.tobytes()
        assert one.coordinates.tobytes() == two.coordinates.tobytes()
        assert (one.topics, one.noise) == (two.topics, two.noise)
    kmeans_map, hdbscan_map = first
    assert kmeans_map.reduction is hdbscan_map.reduction
    # HDBSCAN's labels, noise included, are kept exactly as hdbscan gives them on the shared
    # reduction. Through UMAP, these small synthetic sets rarely leave noise; the -1 class
    # through the real BERTopic is pinned by the parity test above.
    direct = HDBSCAN(min_cluster_size=5).fit(hdbscan_map.reduction.embedding).labels_
    np.testing.assert_array_equal(hdbscan_map.labels, direct)
    assert (hdbscan_map.noise is not None) == (hdbscan_map.noise_points > 0)
    assert kmeans_map.coordinates.shape == (len(documents), 2)
    for topic_map in (*first, *other):
        assert internal_metrics(vectors, topic_map.labels).undefined_reason is None
        coherence = npmi_coherence(
            topic_map.terms(), documents, top_n=settings.top_n, unit="document", window_size=None
        )
        assert coherence.undefined_reason is None
    stability = seed_stability([first[0].labels, other[0].labels], noise_handling="own_label")
    assert stability.mean is not None and stability.mean > 0.9
