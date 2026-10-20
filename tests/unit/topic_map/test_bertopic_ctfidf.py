"""The BERTopic adapter of the c-TF-IDF port (T16, D22, D30 E4), with BERTopic stand-ins.

What is checked here: how the adapter builds BERTopic (``language="spanish"``, manual mode, a
fixed vocabulary, the defaults of ``ClassTfidfTransformer``), that ``y`` only ever reaches the
identity reduction, and that BERTopic's renumbered topics are mapped back to the given labels.
The smoke module checks the same against the real BERTopic.
"""

import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from thematic_redundancy.topic_map import TopicMapError, bertopic_ctfidf
from thematic_redundancy.topic_map.bertopic_ctfidf import (
    LANGUAGE,
    BertopicCtfidf,
    build_topic_model,
    load_bertopic_components,
    require_identity_reduction,
)
from thematic_redundancy.topic_map.ctfidf import analyze, encode_document


def encoded(corpus: Any) -> list[str]:
    return [encode_document(document) for document in corpus.documents]


def size_labels(corpus: Any) -> np.ndarray:
    """Labels whose sizes rank in another order than their ids, with noise: BERTopic must
    renumber them."""
    labels = np.where(corpus.themes == 0, 0, np.where(corpus.themes == 1, 1, 2))
    labels[[0, 1, 2, 3, 4]] = 2
    labels[[12, 13]] = -1
    return labels


def test_the_language_is_spanish() -> None:
    assert LANGUAGE == "spanish"


def test_bertopic_is_built_in_manual_mode_with_a_fixed_vocabulary(
    corpus: Any, bertopic_components: Any, fakes: Any
) -> None:
    model = build_topic_model(bertopic_components, corpus.vocabulary, 2)

    kwargs = model.kwargs
    assert kwargs["language"] == "spanish"
    assert kwargs["embedding_model"] is None
    assert type(kwargs["umap_model"]) is fakes.BaseDimensionalityReduction
    assert type(kwargs["hdbscan_model"]) is fakes.BaseCluster
    assert type(kwargs["ctfidf_model"]) is fakes.ClassTfidfTransformer
    assert kwargs["ctfidf_model"].kwargs == {}
    assert kwargs["nr_topics"] is None
    assert kwargs["calculate_probabilities"] is False
    vectorizer = kwargs["vectorizer_model"]
    assert list(vectorizer.vocabulary) == list(corpus.vocabulary)
    text = encode_document(corpus.documents[0])
    assert vectorizer.analyzer(text) == analyze(text, 2)


def test_the_vectorizer_forms_terms_up_to_the_given_max_ngram(
    corpus: Any, bertopic_components: Any
) -> None:
    model = build_topic_model(bertopic_components, corpus.vocabulary, 1)

    text = encode_document(corpus.documents[0])
    assert model.kwargs["vectorizer_model"].analyzer(text) == analyze(text, 1)


def test_labels_reach_bertopic_as_y_with_the_identity_reduction_only(
    corpus: Any, fakes: Any
) -> None:
    models: list[Any] = []

    class Recorded(fakes.BERTopic):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            models.append(self)

    labels = size_labels(corpus)
    documents = encoded(corpus)

    BertopicCtfidf(fakes.components(Recorded)).class_weights(
        documents, labels, corpus.vocabulary, 2
    )

    (model,) = models
    (call,) = model.fit_calls
    assert call["documents"] == documents
    np.testing.assert_array_equal(call["y"], labels)
    assert call["embeddings"].shape == (len(documents), 1)
    # BERTopic forwards y to its dimensionality model, which is the identity here.
    assert type(model.umap_model) is fakes.BaseDimensionalityReduction
    np.testing.assert_array_equal(model.umap_model.fitted_with_y, labels)


def test_a_dimensionality_model_other_than_the_identity_is_refused(
    bertopic_components: Any,
) -> None:
    class RealLookingUmap:
        def fit_transform(self, X: Any, y: Any = None) -> Any:
            return X

    model = types.SimpleNamespace(umap_model=RealLookingUmap())

    with pytest.raises(TopicMapError, match="supervised"):
        require_identity_reduction(model, bertopic_components)


def test_the_adapter_checks_the_reduction_before_fitting(
    corpus: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []

    def with_a_real_looking_umap(components: Any, vocabulary: Any, max_ngram: int) -> Any:
        model = build_topic_model(components, vocabulary, max_ngram)
        model.umap_model = types.SimpleNamespace(fit_transform=lambda X, y=None: X)
        built.append(model)
        return model

    monkeypatch.setattr(bertopic_ctfidf, "build_topic_model", with_a_real_looking_umap)

    with pytest.raises(TopicMapError, match="supervised"):
        BertopicCtfidf(fakes.components()).class_weights(
            encoded(corpus), size_labels(corpus), corpus.vocabulary, 2
        )
    assert built and built[0].fit_calls == []


def test_the_weights_are_mapped_back_to_the_given_labels(
    corpus: Any, bertopic_components: Any, oracle: Any
) -> None:
    labels = size_labels(corpus)

    result = BertopicCtfidf(bertopic_components).class_weights(
        encoded(corpus), labels, corpus.vocabulary, 2
    )

    classes, counts = oracle.counts(labels, corpus.documents, corpus.vocabulary)
    assert result.classes == classes == (-1, 0, 1, 2)
    assert result.weights.shape == (4, len(corpus.vocabulary))
    assert result.weights.dtype == np.float64
    np.testing.assert_allclose(
        result.weights.toarray(), oracle.formula(counts).toarray(), rtol=1e-12, atol=0
    )


def test_a_bertopic_that_breaks_the_label_partition_is_refused(corpus: Any, fakes: Any) -> None:
    class Merging(fakes.BERTopic):
        def fit_transform(self, documents: Any, embeddings: Any = None, y: Any = None) -> Any:
            topics, probabilities = super().fit_transform(documents, embeddings=embeddings, y=y)
            self.topics_ = [min(topic, 0) for topic in topics]
            return self.topics_, probabilities

    with pytest.raises(TopicMapError, match="one to one"):
        BertopicCtfidf(fakes.components(Merging)).class_weights(
            encoded(corpus), size_labels(corpus), corpus.vocabulary, 2
        )


def test_a_vectorizer_that_reorders_the_vocabulary_is_refused(corpus: Any, fakes: Any) -> None:
    class Reordering(fakes.BERTopic):
        def fit_transform(self, documents: Any, embeddings: Any = None, y: Any = None) -> Any:
            result = super().fit_transform(documents, embeddings=embeddings, y=y)
            self.vectorizer_model.get_feature_names_out = lambda: np.array(
                list(reversed(self.vectorizer_model.vocabulary))
            )
            return result

    with pytest.raises(TopicMapError, match="vocabulary"):
        BertopicCtfidf(fakes.components(Reordering)).class_weights(
            encoded(corpus), size_labels(corpus), corpus.vocabulary, 2
        )


def test_the_components_are_imported_from_bertopic_and_scikit_learn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "bertopic": types.ModuleType("bertopic"),
        "bertopic.cluster": types.ModuleType("bertopic.cluster"),
        "bertopic.dimensionality": types.ModuleType("bertopic.dimensionality"),
        "bertopic.vectorizers": types.ModuleType("bertopic.vectorizers"),
    }
    markers = {
        name: type(name, (), {})
        for name in (
            "BERTopic",
            "BaseCluster",
            "BaseDimensionalityReduction",
            "ClassTfidfTransformer",
        )
    }
    modules["bertopic"].BERTopic = markers["BERTopic"]  # type: ignore[attr-defined]
    modules["bertopic.cluster"].BaseCluster = markers["BaseCluster"]  # type: ignore[attr-defined]
    modules["bertopic.dimensionality"].BaseDimensionalityReduction = markers[  # type: ignore[attr-defined]
        "BaseDimensionalityReduction"
    ]
    modules["bertopic.vectorizers"].ClassTfidfTransformer = markers[  # type: ignore[attr-defined]
        "ClassTfidfTransformer"
    ]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    components = load_bertopic_components()

    from sklearn.feature_extraction.text import CountVectorizer

    assert components.bertopic is markers["BERTopic"]
    assert components.base_cluster is markers["BaseCluster"]
    assert components.base_dimensionality_reduction is markers["BaseDimensionalityReduction"]
    assert components.class_tfidf_transformer is markers["ClassTfidfTransformer"]
    assert components.count_vectorizer is CountVectorizer


def test_importing_the_topic_map_imports_no_heavy_library() -> None:
    code = (
        "import sys; import thematic_redundancy.topic_map.build; "
        "print(sorted(m for m in ('bertopic', 'umap', 'hdbscan') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "[]"
