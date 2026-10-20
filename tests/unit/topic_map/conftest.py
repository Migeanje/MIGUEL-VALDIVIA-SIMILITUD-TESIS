"""Shared synthetic data and fakes for the topic map's unit tests (T16).

The real UMAP and BERTopic take about 12 s and 50 s to import, so the unit tests replace them:

- :class:`RecordingUmap` is a UMAP factory whose models record every call and return a seeded
  random projection, so the seed visibly reaches them;
- :func:`fake_bertopic_components` gives BERTopic stand-ins that behave as BERTopic 0.17.4's
  manual mode does on the path the adapter uses: the dimensionality model receives ``y``, the
  cluster model takes ``y`` as its labels, topic ids are renumbered by descending size with -1
  kept, each topic's documents are joined by one space, and the c-TF-IDF follows
  ``ClassTfidfTransformer()``'s formula, written here independently (D30 E4). The vectorizer is
  scikit-learn's real ``CountVectorizer``.

The smoke module ``tests/smoke/test_topic_map_build.py`` checks the same paths with the real
libraries. K-means, PCA and HDBSCAN are real here: they import quickly.

Every corpus is synthetic: Spanish-like tokens drawn from fixed theme lists, and vectors drawn
around one axis per theme.
"""

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from thematic_redundancy.representation.tfidf import TfidfSettings, fit_tfidf, section_terms
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicComponents

SEED = 20261019

_MINING = "voladura explosivo fragmentación macizo perforación detonador tajo roca"
_MAINTENANCE = (
    "mantenimiento rodamiento disponibilidad falla motor repuesto confiabilidad lubricación"
)
_SOFTWARE = "sistema inventario módulo datos facturación usuario aplicación servidor"
THEME_WORDS: dict[str, tuple[str, ...]] = {
    "mining": tuple(_MINING.split()),
    "maintenance": tuple(_MAINTENANCE.split()),
    "software": tuple(_SOFTWARE.split()),
}
SHARED_WORDS = ("empresa", "proceso", "análisis", "propuesta", "diseño")
SECTION_LENGTHS = (3, 9, 5)
"""Tokens of the title, the abstract and the objectives of each synthetic thesis."""

TFIDF_SETTINGS = TfidfSettings(max_ngram=2, min_df=2, max_df=1.0, sublinear_tf=True)
"""T15's vectorizer settings (D28 C3), which give the vocabulary."""


@dataclass(frozen=True)
class Corpus:
    """Synthetic theses: their sections' tokens, their vectors and their themes."""

    documents: tuple[tuple[tuple[str, ...], ...], ...]
    vectors: np.ndarray
    themes: np.ndarray
    """The theme index of each thesis; -1 for an outlier."""
    vocabulary: tuple[str, ...]


def make_corpus(per_theme: int = 12, outliers: int = 0, seed: int = SEED) -> Corpus:
    """Return ``per_theme`` theses per theme, in theme order, then ``outliers`` theses whose
    vectors lie far from every theme; the vocabulary is T15's over all of them."""
    rng = np.random.default_rng(seed)
    themes = list(THEME_WORDS)
    dimensions = 12
    documents: list[tuple[tuple[str, ...], ...]] = []
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for index, theme in enumerate(themes):
        center = np.eye(dimensions)[index] * 10
        for _ in range(per_theme):
            documents.append(_thesis(rng, THEME_WORDS[theme]))
            vectors.append(center + rng.normal(0, 0.3, dimensions))
            labels.append(index)
    words = tuple(word for theme in themes for word in THEME_WORDS[theme])
    for index in range(outliers):
        documents.append(_thesis(rng, words))
        vectors.append(np.eye(dimensions)[len(themes) + index] * 40)
        labels.append(-1)
    vocabulary = fit_tfidf(documents, TFIDF_SETTINGS).terms
    return Corpus(
        documents=tuple(documents),
        vectors=np.vstack(vectors),
        themes=np.asarray(labels),
        vocabulary=vocabulary,
    )


def _thesis(rng: np.random.Generator, words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    sections = []
    for length in SECTION_LENGTHS:
        pool = np.where(rng.random(length) < 0.8, 0, 1)
        sections.append(
            tuple(
                str(rng.choice(words)) if source == 0 else str(rng.choice(SHARED_WORDS))
                for source in pool
            )
        )
    return tuple(sections)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return make_corpus()


@pytest.fixture(scope="session")
def corpus_with_outliers() -> Corpus:
    return make_corpus(outliers=3)


# The oracle of the c-TF-IDF (D30 E4), independent of the code under test


def class_term_counts(
    labels: np.ndarray,
    documents: tuple[tuple[tuple[str, ...], ...], ...],
    vocabulary: tuple[str, ...],
    max_ngram: int = 2,
) -> tuple[tuple[int, ...], sparse.csr_matrix]:
    """Count each vocabulary term per class, terms formed inside one section only."""
    classes = tuple(sorted(set(labels.tolist())))
    row = {label: index for index, label in enumerate(classes)}
    column = {term: index for index, term in enumerate(vocabulary)}
    counts = np.zeros((len(classes), len(vocabulary)))
    for label, document in zip(labels.tolist(), documents, strict=True):
        for section in document:
            for term in section_terms(section, max_ngram):
                if term in column:
                    counts[row[label], column[term]] += 1
    return classes, sparse.csr_matrix(counts)


def ctfidf_formula(counts: sparse.spmatrix) -> sparse.csr_matrix:
    """BERTopic 0.17.4's ``ClassTfidfTransformer()``: each class row L1-normalized, times
    idf = log(A / f + 1), where A is the integer mean of the words per class and f a term's
    total count."""
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    frequency = np.asarray(counts.sum(axis=0)).ravel()
    average = int(np.asarray(counts.sum(axis=1)).ravel().mean())
    idf = np.log(average / frequency + 1)
    return sparse.csr_matrix(normalize(counts, norm="l1") @ sparse.diags(idf))


def oracle_top_terms(
    row: np.ndarray, vocabulary: tuple[str, ...], top_n: int
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    ranked = sorted(
        (index for index in range(row.size) if row[index] > 0),
        key=lambda index: (-row[index], vocabulary[index]),
    )[:top_n]
    return tuple(vocabulary[i] for i in ranked), tuple(float(row[i]) for i in ranked)


# UMAP stand-in


@dataclass
class UmapCall:
    settings: Any
    seed: int
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class RecordingUmap:
    """A UMAP factory: ``factory(settings, seed)`` returns a model whose ``fit_transform``
    records its arguments and returns a seeded random projection to ``n_components``."""

    def __init__(self) -> None:
        self.calls: list[UmapCall] = []

    def __call__(self, settings: Any, seed: int) -> "_RecordingUmapModel":
        call = UmapCall(settings=settings, seed=seed)
        self.calls.append(call)
        return _RecordingUmapModel(call)

    def fits(self, n_components: int) -> list[UmapCall]:
        return [call for call in self.calls if call.settings.n_components == n_components]


class _RecordingUmapModel:
    def __init__(self, call: UmapCall) -> None:
        self._call = call

    def fit_transform(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self._call.args, self._call.kwargs = args, kwargs
        vectors = np.asarray(args[0], dtype=np.float64)
        rng = np.random.default_rng(self._call.seed)
        projection = rng.normal(size=(vectors.shape[1], self._call.settings.n_components))
        return (vectors @ projection).astype(np.float32)


@pytest.fixture
def recording_umap() -> RecordingUmap:
    return RecordingUmap()


# BERTopic stand-ins


class FakeBaseDimensionalityReduction:
    """As ``bertopic.dimensionality.BaseDimensionalityReduction``: the identity."""

    def __init__(self) -> None:
        self.fitted_with_y: Any = "not fitted"

    def fit(self, X: Any = None, y: Any = None) -> "FakeBaseDimensionalityReduction":
        self.fitted_with_y = y
        return self

    def transform(self, X: Any) -> Any:
        return X


class FakeBaseCluster:
    """As ``bertopic.cluster.BaseCluster``: the labels are ``y``."""

    def fit(self, X: Any, y: Any = None) -> "FakeBaseCluster":
        self.labels_ = y
        return self


class FakeClassTfidfTransformer:
    """``ClassTfidfTransformer()``'s defaults, through :func:`ctfidf_formula`."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, X: Any) -> "FakeClassTfidfTransformer":
        self._counts = X
        return self

    def transform(self, X: Any) -> sparse.csr_matrix:
        return ctfidf_formula(X)


class FakeBERTopic:
    """BERTopic 0.17.4's ``fit_transform`` on the manual-mode path (see the module)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.language = kwargs.get("language", "english")
        self.umap_model = kwargs["umap_model"]
        self.hdbscan_model = kwargs["hdbscan_model"]
        self.vectorizer_model = kwargs["vectorizer_model"]
        self.ctfidf_model = kwargs["ctfidf_model"]
        self.fit_calls: list[dict[str, Any]] = []

    def fit_transform(
        self, documents: list[str], embeddings: Any = None, images: Any = None, y: Any = None
    ) -> tuple[list[int], None]:
        self.fit_calls.append({"documents": documents, "embeddings": embeddings, "y": y})
        reduced = self.umap_model.fit(embeddings, y=y).transform(embeddings)
        labels = [int(label) for label in self.hdbscan_model.fit(reduced, y=y).labels_]
        sizes = Counter(labels)
        ranked = sorted((t for t in sizes if t != -1), key=lambda t: (-sizes[t], -t))
        mapping = {-1: -1, **{old: new for new, old in enumerate(ranked)}}
        topics = [mapping[label] for label in labels]
        classes = sorted(set(topics))
        joined = [
            " ".join(doc for doc, topic in zip(documents, topics, strict=True) if topic == c)
            for c in classes
        ]
        counts = self.vectorizer_model.fit_transform(joined)
        self.c_tf_idf_ = self.ctfidf_model.fit(counts).transform(counts)
        self.topics_ = topics
        return topics, None


def fake_bertopic_components(bertopic: type = FakeBERTopic) -> BertopicComponents:
    return BertopicComponents(
        bertopic=bertopic,
        base_cluster=FakeBaseCluster,
        base_dimensionality_reduction=FakeBaseDimensionalityReduction,
        class_tfidf_transformer=FakeClassTfidfTransformer,
        count_vectorizer=CountVectorizer,
    )


@pytest.fixture
def bertopic_components() -> BertopicComponents:
    return fake_bertopic_components()


# Test modules reach the helpers through these fixtures, since nothing imports a conftest.


@pytest.fixture(scope="session")
def oracle() -> SimpleNamespace:
    """The independent c-TF-IDF oracle: ``counts``, ``formula`` and ``top_terms``."""
    return SimpleNamespace(
        counts=class_term_counts, formula=ctfidf_formula, top_terms=oracle_top_terms
    )


@pytest.fixture(scope="session")
def corpus_maker() -> Any:
    """:func:`make_corpus`, for corpora of other sizes."""
    return make_corpus


@pytest.fixture(scope="session")
def fakes() -> SimpleNamespace:
    """The BERTopic stand-ins, to subclass, and :func:`fake_bertopic_components`."""
    return SimpleNamespace(
        BERTopic=FakeBERTopic,
        BaseCluster=FakeBaseCluster,
        BaseDimensionalityReduction=FakeBaseDimensionalityReduction,
        ClassTfidfTransformer=FakeClassTfidfTransformer,
        components=fake_bertopic_components,
    )
