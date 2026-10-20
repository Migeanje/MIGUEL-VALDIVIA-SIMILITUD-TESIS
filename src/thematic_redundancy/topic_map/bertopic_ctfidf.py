"""The c-TF-IDF port's adapter: BERTopic 0.17.4 in its manual mode (T16, D22, D30 E1 and E4).

BERTopic only computes the c-TF-IDF of labels it is given: the dimensionality model is
``BaseDimensionalityReduction`` (the identity), the cluster model is ``BaseCluster``, which takes
``y`` as its labels, and the labels go in as ``y``. The UMAP that the semantic runs cluster is
fitted apart, unsupervised (:mod:`.reduction`), because BERTopic forwards ``y`` to its
dimensionality model's ``fit`` or ``fit_transform``, which would make a real UMAP supervised
(D30 E4, risk X8); :func:`require_identity_reduction` refuses any other dimensionality model.

BERTopic is always built with ``language="spanish"`` (D22): with ``embedding_model=None`` the
default ``"english"`` strips every character outside ``[A-Za-z0-9 ]``, so ``fragmentación``
would become ``fragmentacin``. Its vectorizer is a ``CountVectorizer`` with the fixed
vocabulary and the section-aware analyzer of :mod:`.ctfidf`, which BERTopic does not fit a
vocabulary for; its c-TF-IDF model is ``ClassTfidfTransformer()`` with its defaults (no BM25,
no frequent-word reduction).

BERTopic renumbers the topics by descending size, -1 kept, and orders the rows of ``c_tf_idf_``
by its own topic ids, so the adapter maps each row back to the label it came from, after
checking that the renumbering kept the partition one to one. The embeddings that BERTopic needs
in place of an embedding model are a column of zeros: the identity and ``BaseCluster`` ignore
them.

BERTopic takes about 50 s to import, so nothing is imported until :func:`load_bertopic_components`
runs, and tests pass their own components.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.ctfidf import ClassWeights, section_analyzer

LANGUAGE = "spanish"
"""BERTopic's ``language`` (D22); never its default."""


@dataclass(frozen=True)
class BertopicComponents:
    """The classes the adapter builds BERTopic from."""

    bertopic: Any
    base_cluster: Any
    base_dimensionality_reduction: Any
    class_tfidf_transformer: Any
    count_vectorizer: Any


def load_bertopic_components() -> BertopicComponents:
    """Import BERTopic's classes and scikit-learn's ``CountVectorizer``."""
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    return BertopicComponents(
        bertopic=BERTopic,
        base_cluster=BaseCluster,
        base_dimensionality_reduction=BaseDimensionalityReduction,
        class_tfidf_transformer=ClassTfidfTransformer,
        count_vectorizer=CountVectorizer,
    )


def build_topic_model(
    components: BertopicComponents, vocabulary: Sequence[str], max_ngram: int
) -> Any:
    """Return a BERTopic model in manual mode, in Spanish, over the fixed ``vocabulary``."""
    return components.bertopic(
        language=LANGUAGE,
        embedding_model=None,
        umap_model=components.base_dimensionality_reduction(),
        hdbscan_model=components.base_cluster(),
        vectorizer_model=components.count_vectorizer(
            analyzer=section_analyzer(max_ngram), vocabulary=list(vocabulary)
        ),
        ctfidf_model=components.class_tfidf_transformer(),
        nr_topics=None,
        calculate_probabilities=False,
    )


def require_identity_reduction(model: Any, components: BertopicComponents) -> None:
    """Refuse a model whose dimensionality model is not BERTopic's identity.

    Raises:
        TopicMapError: since the labels go in as ``y``, which BERTopic forwards to that model,
            and a real UMAP would then be supervised.
    """
    if type(model.umap_model) is not components.base_dimensionality_reduction:
        raise TopicMapError(
            "BERTopic's dimensionality model must be its identity, since the labels go in as y "
            "and BERTopic forwards y to it: a real UMAP would be fitted supervised"
        )


class BertopicCtfidf:
    """The :class:`~.ctfidf.CtfidfBackend` adapter over BERTopic's manual mode."""

    def __init__(self, components: BertopicComponents | None = None) -> None:
        self._components = components

    def class_weights(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        vocabulary: tuple[str, ...],
        max_ngram: int,
    ) -> ClassWeights:
        """Return the c-TF-IDF of each class of ``labels``, rows in ascending label order.

        Raises:
            TopicMapError: if the dimensionality model is not the identity, or BERTopic did not
                keep the partition one to one or the vocabulary in order.
        """
        components = self._components or load_bertopic_components()
        model = build_topic_model(components, vocabulary, max_ngram)
        require_identity_reduction(model, components)
        label_values = np.asarray(labels)
        topics, _ = model.fit_transform(
            list(documents), embeddings=np.zeros((len(documents), 1)), y=label_values
        )
        names = tuple(str(name) for name in model.vectorizer_model.get_feature_names_out())
        if names != tuple(vocabulary):
            raise TopicMapError("BERTopic's vectorizer did not keep the vocabulary in order")
        return _weights_by_label(label_values, [int(topic) for topic in topics], model.c_tf_idf_)


def _weights_by_label(labels: np.ndarray, topics: list[int], matrix: Any) -> ClassWeights:
    """Reorder BERTopic's rows, one per BERTopic topic id ascending, to the given labels."""
    pairs = set(zip(labels.tolist(), topics, strict=True))
    classes = sorted({label for label, _ in pairs})
    bertopic_topics = sorted({topic for _, topic in pairs})
    if not len(pairs) == len(classes) == len(bertopic_topics):
        raise TopicMapError(
            f"BERTopic did not keep the label partition one to one: {len(classes)} labels "
            f"became {len(bertopic_topics)} topics through {len(pairs)} pairs"
        )
    weights = sparse.csr_matrix(matrix, dtype=np.float64)
    if weights.shape[0] != len(bertopic_topics):
        raise TopicMapError(
            f"BERTopic gave {weights.shape[0]} c-TF-IDF rows for {len(bertopic_topics)} topics"
        )
    row_of_topic = {topic: row for row, topic in enumerate(bertopic_topics)}
    topic_of_label = dict(pairs)
    order = [row_of_topic[topic_of_label[label]] for label in classes]
    reordered = sparse.csr_matrix(weights[order])
    reordered.sort_indices()
    return ClassWeights(classes=tuple(classes), weights=reordered)
