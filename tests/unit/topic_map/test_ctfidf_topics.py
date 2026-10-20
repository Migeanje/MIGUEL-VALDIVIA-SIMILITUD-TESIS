"""Topic terms by c-TF-IDF over a fixed vocabulary (T16, D30 E4), on synthetic theses.

The production adapter runs here with BERTopic stand-ins (see ``conftest.py``); its terms and
scores are checked against an independent oracle that counts each vocabulary term per class
inside each section and applies ``ClassTfidfTransformer()``'s formula.
"""

from typing import Any

import numpy as np
import pytest

from thematic_redundancy.representation.tfidf import TfidfSettings, fit_tfidf, thesis_terms
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicCtfidf
from thematic_redundancy.topic_map.clustering import KMeansSettings, kmeans_labels
from thematic_redundancy.topic_map.ctfidf import (
    SECTION_BOUNDARY,
    ClassWeights,
    Topic,
    TopicSet,
    analyze,
    ctfidf_topics,
    encode_document,
    section_analyzer,
    top_terms,
)

TOP_N = 5

TFIDF_SETTINGS = TfidfSettings(max_ngram=2, min_df=2, max_df=1.0, sublinear_tf=True)
"""T15's vectorizer settings (D28 C3)."""

_MINING = "voladura explosivo fragmentación macizo perforación detonador tajo roca"
_MAINTENANCE = (
    "mantenimiento rodamiento disponibilidad falla motor repuesto confiabilidad lubricación"
)
_SOFTWARE = "sistema inventario módulo datos facturación usuario aplicación servidor"
THEME_WORDS = {
    "mining": set(_MINING.split()),
    "maintenance": set(_MAINTENANCE.split()),
    "software": set(_SOFTWARE.split()),
}
"""The theme lists of ``conftest.py``, repeated so this module states what it expects."""


def theme_labels(themes: np.ndarray) -> np.ndarray:
    """Labels 4, 9 and 2 for the three themes, so that ids follow neither size nor order."""
    return np.array([{0: 4, 1: 9, 2: 2, -1: -1}[int(t)] for t in themes])


def topics_for(corpus: Any, labels: np.ndarray, components: Any, **overrides: Any) -> TopicSet:
    arguments: dict[str, Any] = {
        "vocabulary": corpus.vocabulary,
        "max_ngram": 2,
        "top_n": TOP_N,
        "include_noise_class": True,
        "backend": BertopicCtfidf(components),
    }
    arguments.update(overrides)
    return ctfidf_topics(labels, corpus.documents, **arguments)


def expected_topics(
    oracle: Any, corpus: Any, labels: np.ndarray, fitted: np.ndarray, top_n: int = TOP_N
) -> dict[int, tuple[tuple[str, ...], tuple[float, ...]]]:
    classes, counts = oracle.counts(
        labels[fitted], tuple(corpus.documents[i] for i in fitted), corpus.vocabulary
    )
    with np.errstate(divide="ignore"):  # a term with no count in the fit has an infinite idf
        weights = oracle.formula(counts).toarray()
    return {
        label: oracle.top_terms(weights[row], corpus.vocabulary, top_n)
        for row, label in enumerate(classes)
    }


# Documents and the analyzer


def test_an_encoded_document_gives_the_terms_of_t15_with_no_bigram_across_sections() -> None:
    sections = (("diseño", "malla"), ("voladura", "roca", "tajo"), ("costo",))

    text = encode_document(sections)

    assert text.count(SECTION_BOUNDARY) == len(sections) + 1
    assert text.startswith(SECTION_BOUNDARY) and text.endswith(SECTION_BOUNDARY)
    assert analyze(text, 2) == thesis_terms(sections, 2)
    assert "malla voladura" not in analyze(text, 2)
    assert "tajo costo" not in analyze(text, 2)


def test_documents_joined_by_a_space_give_no_bigram_across_documents() -> None:
    first, second = (("alfa", "beta"),), (("gamma", "delta"),)

    joined = " ".join([encode_document(first), encode_document(second)])

    assert analyze(joined, 2) == thesis_terms(first, 2) + thesis_terms(second, 2)


def test_unigrams_only_when_max_ngram_is_one() -> None:
    text = encode_document((("a", "b", "c"),))

    assert analyze(text, 1) == ["a", "b", "c"]


def test_the_section_analyzer_is_analyze_with_its_max_ngram() -> None:
    text = encode_document((("motor", "falla"), ("repuesto",)))

    assert section_analyzer(2)(text) == analyze(text, 2)
    assert section_analyzer(1)(text) == analyze(text, 1)


# Top terms


def test_top_terms_rank_by_score_then_by_term_in_code_point_order() -> None:
    vocabulary = ("zeta", "Árbol", "alfa", "árbol", "beta", "Zeta")
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.5])

    terms, values = top_terms(scores, vocabulary, 6)

    assert terms == ("beta", "Zeta", "alfa", "zeta", "Árbol", "árbol")
    assert values == (0.9, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert all(type(value) is float for value in values)


def test_top_terms_keep_positive_scores_only_and_at_most_top_n() -> None:
    vocabulary = ("a", "b", "c", "d", "e")
    scores = np.array([0.0, 0.3, 0.0, 0.7, 0.1])

    assert top_terms(scores, vocabulary, 2) == (("d", "b"), (0.7, 0.3))
    assert top_terms(scores, vocabulary, 10) == (("d", "b", "e"), (0.7, 0.3, 0.1))
    assert top_terms(np.zeros(5), vocabulary, 3) == ((), ())


# The topics


def test_each_topic_has_the_oracle_terms_and_scores(
    corpus: Any, bertopic_components: Any, oracle: Any
) -> None:
    labels = theme_labels(corpus.themes)

    result = topics_for(corpus, labels, bertopic_components)

    expected = expected_topics(oracle, corpus, labels, np.arange(labels.size))
    assert [topic.topic for topic in result.topics] == [2, 4, 9]
    for topic in result.topics:
        terms, scores = expected[topic.topic]
        assert topic.terms == terms
        assert topic.scores == pytest.approx(scores, rel=1e-12)
        assert topic.size == int(np.count_nonzero(labels == topic.topic))
    assert result.noise is None
    assert result.terms() == tuple(topic.terms for topic in result.topics)


def test_each_theme_s_first_terms_are_its_own_words(corpus: Any, bertopic_components: Any) -> None:
    result = topics_for(corpus, theme_labels(corpus.themes), bertopic_components)

    by_label = {topic.topic: topic for topic in result.topics}
    for label, theme in ((4, "mining"), (9, "maintenance"), (2, "software")):
        assert set(by_label[label].terms[:3]) <= set(THEME_WORDS[theme])


def test_the_noise_class_is_fitted_and_kept_apart_from_the_topics(
    corpus: Any, bertopic_components: Any, oracle: Any
) -> None:
    labels = theme_labels(corpus.themes)
    labels[::7] = -1

    result = topics_for(corpus, labels, bertopic_components)

    expected = expected_topics(oracle, corpus, labels, np.arange(labels.size))
    assert [topic.topic for topic in result.topics] == [2, 4, 9]
    assert result.noise is not None
    assert result.noise.topic == -1
    assert result.noise.size == int(np.count_nonzero(labels == -1))
    assert result.noise.terms == expected[-1][0]
    for topic in result.topics:
        assert topic.terms == expected[topic.topic][0]
        assert topic.scores == pytest.approx(expected[topic.topic][1], rel=1e-12)
    assert len(result.terms()) == 3


def test_without_the_noise_class_the_noise_theses_are_left_out_of_the_fit(
    corpus: Any, bertopic_components: Any, oracle: Any
) -> None:
    labels = theme_labels(corpus.themes)
    labels[::7] = -1
    clustered = np.flatnonzero(labels != -1)

    without = topics_for(corpus, labels, bertopic_components, include_noise_class=False)
    with_noise = topics_for(corpus, labels, bertopic_components, include_noise_class=True)

    expected = expected_topics(oracle, corpus, labels, clustered)
    assert without.noise is None
    for topic in without.topics:
        assert topic.terms == expected[topic.topic][0]
        assert topic.scores == pytest.approx(expected[topic.topic][1], rel=1e-12)
    assert [t.scores for t in without.topics] != [t.scores for t in with_noise.topics]
    noise_only = set(corpus.vocabulary) - {
        term for i in clustered for term in thesis_terms(corpus.documents[i], 2)
    }
    assert noise_only, "the corpus must hold terms found only in the noise theses"
    every_term = topics_for(
        corpus, labels, bertopic_components, include_noise_class=False, top_n=10_000
    )
    assert not noise_only & {term for topic in every_term.topics for term in topic.terms}


def test_a_bigram_is_counted_only_inside_a_section(bertopic_components: Any) -> None:
    # "roca tajo" occurs inside a section of the second class only; in the first class its two
    # words meet across a section end, and across two theses, which must not count.
    documents = (
        (("perno", "roca"), ("tajo", "malla")),
        (("perno", "roca"),),
        (("tajo", "malla"),),
        (("roca", "tajo", "cable"),),
        (("roca", "tajo", "cable"),),
    )
    labels = np.array([0, 0, 0, 1, 1])
    vocabulary = tuple(sorted({"perno", "roca", "tajo", "malla", "cable", "roca tajo"}))

    result = ctfidf_topics(
        labels,
        documents,
        vocabulary=vocabulary,
        max_ngram=2,
        top_n=len(vocabulary),
        include_noise_class=True,
        backend=BertopicCtfidf(bertopic_components),
    )

    first, second = result.topics
    assert "roca tajo" not in first.terms
    assert "roca tajo" in second.terms


def test_the_baseline_path_gives_comparable_terms_from_labels_on_the_sparse_matrix(
    corpus: Any, bertopic_components: Any, oracle: Any
) -> None:
    # D28 C4: the TF-IDF baseline is K-means on T15's sparse section-mean matrix, and its
    # labels reach the same c-TF-IDF as the semantic runs' labels.
    matrix = fit_tfidf(corpus.documents, TFIDF_SETTINGS).section_mean
    labels = kmeans_labels(matrix, 3, 7, KMeansSettings(n_init=10, init="k-means++"))

    result = topics_for(corpus, labels, bertopic_components)

    expected = expected_topics(oracle, corpus, labels, np.arange(labels.size))
    assert len(result.topics) == 3
    for topic in result.topics:
        assert topic.terms == expected[topic.topic][0]
        assert topic.scores == pytest.approx(expected[topic.topic][1], rel=1e-12)
        assert set(topic.terms) <= set(corpus.vocabulary)


def test_the_labels_are_never_renumbered(corpus: Any, bertopic_components: Any) -> None:
    labels = np.where(corpus.themes == 0, 30, np.where(corpus.themes == 1, 7, 12))

    result = topics_for(corpus, labels, bertopic_components)

    assert [topic.topic for topic in result.topics] == [7, 12, 30]
    assert [topic.size for topic in result.topics] == [12, 12, 12]


def test_all_noise_without_the_noise_class_gives_no_topic_and_calls_no_backend(
    corpus: Any,
) -> None:
    class Unreachable:
        def class_weights(self, *args: Any) -> ClassWeights:
            raise AssertionError("the backend must not be called")

    labels = np.full(len(corpus.documents), -1)

    result = topics_for(corpus, labels, None, include_noise_class=False, backend=Unreachable())

    assert result == TopicSet(topics=(), noise=None)


def test_topics_are_frozen_records(corpus: Any, bertopic_components: Any) -> None:
    topic = topics_for(corpus, theme_labels(corpus.themes), bertopic_components).topics[0]

    assert isinstance(topic, Topic)
    with pytest.raises(AttributeError):
        topic.size = 0  # type: ignore[misc]


# Invalid inputs


def test_labels_and_documents_of_different_lengths_are_refused(
    corpus: Any, bertopic_components: Any
) -> None:
    with pytest.raises(TopicMapError, match="labels"):
        topics_for(corpus, theme_labels(corpus.themes)[:-1], bertopic_components)


def test_labels_below_the_noise_label_are_refused(corpus: Any, bertopic_components: Any) -> None:
    labels = theme_labels(corpus.themes)
    labels[0] = -2

    with pytest.raises(ValueError, match="-1 or above"):
        topics_for(corpus, labels, bertopic_components)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("titulo", "sequence of sections"),
        (("titulo",), "sequence of tokens"),
        ((("dos palabras",),), "whitespace"),
        ((("marca" + SECTION_BOUNDARY,),), "boundary"),
        ((("",),), "empty"),
        (((3,),), "string"),
    ],
)
def test_malformed_documents_are_refused_without_quoting_them(
    corpus: Any, bertopic_components: Any, document: Any, message: str
) -> None:
    documents = (document, *corpus.documents[1:])

    with pytest.raises(TopicMapError, match=message) as caught:
        ctfidf_topics(
            theme_labels(corpus.themes),
            documents,
            vocabulary=corpus.vocabulary,
            max_ngram=2,
            top_n=TOP_N,
            include_noise_class=True,
            backend=BertopicCtfidf(bertopic_components),
        )
    assert "titulo" not in str(caught.value)
    assert "palabras" not in str(caught.value)
    assert "marca" not in str(caught.value)


@pytest.mark.parametrize(
    ("vocabulary", "message"),
    [
        ((), "empty"),
        (("motor", "motor"), "repeated"),
        (("motor falla repuesto",), "words"),
        (("motor  falla",), "words"),
        (("",), "words"),
    ],
)
def test_malformed_vocabularies_are_refused(
    corpus: Any, bertopic_components: Any, vocabulary: tuple[str, ...], message: str
) -> None:
    with pytest.raises(TopicMapError, match=message):
        topics_for(corpus, theme_labels(corpus.themes), bertopic_components, vocabulary=vocabulary)


def test_a_vocabulary_term_absent_from_the_theses_is_refused(
    corpus: Any, bertopic_components: Any
) -> None:
    vocabulary = tuple(sorted({*corpus.vocabulary, "zzzausente"}))

    with pytest.raises(TopicMapError, match="1 of the") as caught:
        topics_for(corpus, theme_labels(corpus.themes), bertopic_components, vocabulary=vocabulary)
    assert "zzzausente" not in str(caught.value)


@pytest.mark.parametrize("top_n", [0, -1])
def test_top_n_below_one_is_refused(corpus: Any, bertopic_components: Any, top_n: int) -> None:
    with pytest.raises(TopicMapError, match="top_n"):
        topics_for(corpus, theme_labels(corpus.themes), bertopic_components, top_n=top_n)


def test_a_backend_that_returns_other_classes_is_refused(corpus: Any) -> None:
    class WrongClasses:
        def class_weights(self, documents: Any, labels: Any, vocabulary: Any, max_ngram: int):
            from scipy import sparse

            return ClassWeights(classes=(0, 1, 2), weights=sparse.csr_matrix((3, len(vocabulary))))

    with pytest.raises(TopicMapError, match="classes"):
        topics_for(corpus, theme_labels(corpus.themes), None, backend=WrongClasses())


def test_a_backend_that_returns_another_shape_is_refused(corpus: Any) -> None:
    class WrongShape:
        def class_weights(self, documents: Any, labels: Any, vocabulary: Any, max_ngram: int):
            from scipy import sparse

            return ClassWeights(classes=(2, 4, 9), weights=sparse.csr_matrix((3, 4)))

    with pytest.raises(TopicMapError, match="shape"):
        topics_for(corpus, theme_labels(corpus.themes), None, backend=WrongShape())
