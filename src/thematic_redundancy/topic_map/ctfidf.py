"""Topic terms by class-based TF-IDF over a fixed vocabulary (T16, D30 E4).

Every configuration, the TF-IDF baseline included (D28 C4), gets its topic terms from its labels
the same way, as D30 E4 declares:

- **Counts.** The raw count of each vocabulary term in each class, from T15's full-clean section
  tokens: unigrams, and bigrams formed inside one section only (D28 C3), never across two
  sections or two theses.
- **Vocabulary.** Fixed, given by the caller (T15's 14,308 terms on the real corpus), with no
  class-level ``min_df`` or ``max_df``. Every term must occur in the theses, or the vocabulary
  does not belong to them. When the noise theses are left out of the fit, a term found only in
  them has no count in any class, so it gets no score; it is left out of the backend's
  vocabulary, which spares BERTopic a zero frequency in its idf and changes no other weight.
- **Weighting.** BERTopic 0.17.4's ``ClassTfidfTransformer()`` defaults, computed by the
  :class:`CtfidfBackend` port (:mod:`.bertopic_ctfidf` is its adapter).
- **Classes.** Every label is a class. The noise label is a class of the fit when
  ``include_noise_class`` is true, so it shares the weighting, but its terms are kept apart from
  the topics (:attr:`TopicSet.noise`), since NPMI and the topic count leave it out.
- **Top terms** (:func:`top_terms`): positive scores only, by score descending, then by term
  ascending in code-point order, at most ``top_n``; no padding.

BERTopic joins each topic's documents with one space and keeps every character outside ``\\n``
and ``\\t`` under ``language="spanish"``, so a thesis is encoded as its sections between
:data:`SECTION_BOUNDARY` marks (:func:`encode_document`), and the analyzer (:func:`analyze`)
splits on them before forming terms with T15's
:func:`~thematic_redundancy.representation.tfidf.section_terms`.
"""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array
from thematic_redundancy.representation.tfidf import section_terms
from thematic_redundancy.topic_map import TopicMapError

SECTION_BOUNDARY = "¶"
"""The pilcrow that marks where a section starts and ends in an encoded document; no token may
hold it."""


@dataclass(frozen=True)
class Topic:
    """One class of a labeling, with its top c-TF-IDF terms."""

    topic: int
    """The class's label, as the clustering gave it."""
    size: int
    """Theses with this label."""
    terms: tuple[str, ...]
    """The top terms, in rank order."""
    scores: tuple[float, ...]
    """The c-TF-IDF score of each term."""


@dataclass(frozen=True)
class TopicSet:
    """The topics of one labeling and, when it is a class of the fit, the noise's terms."""

    topics: tuple[Topic, ...]
    """One per cluster label, by ascending label; the noise label is never among them."""
    noise: Topic | None

    def terms(self) -> tuple[tuple[str, ...], ...]:
        """Return each topic's terms in topic order, as NPMI reads them."""
        return tuple(topic.terms for topic in self.topics)


@dataclass(frozen=True)
class ClassWeights:
    """What a :class:`CtfidfBackend` returns: one c-TF-IDF row per class."""

    classes: tuple[int, ...]
    """The classes, ascending."""
    weights: sparse.csr_matrix
    """One row per class of :attr:`classes`, one column per vocabulary term, in order."""


class CtfidfBackend(Protocol):
    """Port that computes the c-TF-IDF weights of labelled, encoded documents."""

    def class_weights(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        vocabulary: tuple[str, ...],
        max_ngram: int,
    ) -> ClassWeights:
        """Return the c-TF-IDF of each class of ``labels`` over ``vocabulary``, the documents
        encoded by :func:`encode_document` and analyzed by :func:`analyze`."""
        ...


def encode_document(sections: Sequence[Sequence[str]]) -> str:
    """Return a thesis as one text: each section's tokens joined by spaces, with
    :data:`SECTION_BOUNDARY` before, between and after the sections."""
    return "".join(f"{SECTION_BOUNDARY} {' '.join(tokens)} " for tokens in sections) + (
        SECTION_BOUNDARY
    )


def analyze(text: str, max_ngram: int) -> list[str]:
    """Return the terms of an encoded text: each section's terms in turn, so no bigram spans a
    section end, or two documents joined by a space."""
    return [
        term
        for segment in text.split(SECTION_BOUNDARY)
        for term in section_terms(segment.split(), max_ngram)
    ]


def section_analyzer(max_ngram: int) -> Callable[[str], list[str]]:
    """Return :func:`analyze` with ``max_ngram`` fixed, as a vectorizer's analyzer."""
    return functools.partial(analyze, max_ngram=max_ngram)


def top_terms(
    scores: np.ndarray, vocabulary: Sequence[str], top_n: int
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return the at most ``top_n`` terms of positive score, by score descending, then by term
    ascending (code-point order), with their scores."""
    positive = np.flatnonzero(scores > 0)
    ranked = sorted(positive.tolist(), key=lambda index: (-scores[index], vocabulary[index]))
    chosen = ranked[:top_n]
    return tuple(vocabulary[i] for i in chosen), tuple(float(scores[i]) for i in chosen)


def ctfidf_topics(
    labels: Sequence[int] | np.ndarray,
    documents: Sequence[Sequence[Sequence[str]]],
    *,
    vocabulary: Sequence[str],
    max_ngram: int,
    top_n: int,
    include_noise_class: bool,
    backend: CtfidfBackend,
) -> TopicSet:
    """Return the topics of ``labels`` with their top c-TF-IDF terms (see the module).

    Args:
        labels: one label per thesis; :data:`NOISE_LABEL` marks noise.
        documents: each thesis as the full-clean token lists of its sections.
        vocabulary: the fixed terms, unique, each of 1 to ``max_ngram`` words.
        max_ngram: the longest term, in words (T15's ``tfidf.max_ngram``).
        top_n: the most terms kept per class, at least 1.
        include_noise_class: whether the noise theses form a class of the fit.
        backend: the c-TF-IDF port.

    Raises:
        TopicMapError: if an argument is invalid, a vocabulary term occurs in none of the
            fitted theses, or the backend breaks its contract. A message never quotes a token.
        ValueError: if the labels are not 1-D integers of at least :data:`NOISE_LABEL`.
    """
    label_values = label_array(labels)
    if label_values.size != len(documents):
        raise TopicMapError(f"{label_values.size} labels were given for {len(documents)} documents")
    if top_n < 1:
        raise TopicMapError(f"top_n must be at least 1, not {top_n}")
    terms = _check_vocabulary(vocabulary, max_ngram)
    _check_documents(documents)
    present = _terms_present(documents, max_ngram)
    missing = sum(1 for term in terms if term not in present)
    if missing:
        raise TopicMapError(
            f"{missing} of the {len(terms)} vocabulary terms occur in none of the documents, "
            "so the vocabulary does not belong to them"
        )
    fitted = (
        np.arange(label_values.size)
        if include_noise_class
        else np.flatnonzero(label_values != NOISE_LABEL)
    )
    if fitted.size == 0:
        return TopicSet(topics=(), noise=None)
    fitted_documents = [documents[index] for index in fitted]
    if fitted.size < label_values.size:
        # A term found only in the noise theses would have no count in any class, so a zero
        # frequency in the idf; it gets no score, which leaving it out of the fit gives.
        present = _terms_present(fitted_documents, max_ngram)
        terms = tuple(term for term in terms if term in present)
    fitted_labels = label_values[fitted]
    classes = tuple(sorted(set(fitted_labels.tolist())))
    result = backend.class_weights(
        [encode_document(document) for document in fitted_documents],
        fitted_labels,
        terms,
        max_ngram,
    )
    if result.classes != classes:
        raise TopicMapError(
            f"the c-TF-IDF backend returned the classes {list(result.classes)}, not {list(classes)}"
        )
    if result.weights.shape != (len(classes), len(terms)):
        raise TopicMapError(
            f"the c-TF-IDF backend returned a matrix of shape {result.weights.shape}, "
            f"not {(len(classes), len(terms))}"
        )
    weights = sparse.csr_matrix(result.weights)
    topics = []
    noise = None
    for row, label in enumerate(classes):
        top, scores = top_terms(weights.getrow(row).toarray().ravel(), terms, top_n)
        topic = Topic(
            topic=label,
            size=int(np.count_nonzero(label_values == label)),
            terms=top,
            scores=scores,
        )
        if label == NOISE_LABEL:
            noise = topic
        else:
            topics.append(topic)
    return TopicSet(topics=tuple(topics), noise=noise)


def _check_vocabulary(vocabulary: Sequence[str], max_ngram: int) -> tuple[str, ...]:
    terms = tuple(vocabulary)
    if not terms:
        raise TopicMapError("the vocabulary is empty")
    if len(set(terms)) != len(terms):
        raise TopicMapError(f"the vocabulary has {len(terms) - len(set(terms))} repeated terms")
    for column, term in enumerate(terms):
        words = term.split(" ")
        if not 1 <= len(words) <= max_ngram or not all(_is_token(word) for word in words):
            raise TopicMapError(
                f"vocabulary term {column} must be 1 to {max_ngram} words joined by single spaces"
            )
    return terms


def _is_token(word: str) -> bool:
    """Whether ``word`` is one token that the encoding keeps: no whitespace, no boundary."""
    return bool(word) and SECTION_BOUNDARY not in word and word.split() == [word]


def _check_documents(documents: Sequence[Sequence[Sequence[str]]]) -> None:
    """Check that each document is a sequence of sections, each a sequence of tokens that the
    encoding keeps: non-empty strings with no whitespace and no section boundary."""
    for index, document in enumerate(documents):
        if isinstance(document, str):
            raise TopicMapError(f"document {index} must be a sequence of sections, not a string")
        for position, section in enumerate(document):
            if isinstance(section, str):
                raise TopicMapError(
                    f"section {position} of document {index} must be a sequence of tokens, "
                    "not a string"
                )
            for token in section:
                where = f"a token of section {position} of document {index}"
                if not isinstance(token, str):
                    raise TopicMapError(f"{where} must be a string")
                if not token:
                    raise TopicMapError(f"{where} is empty")
                if SECTION_BOUNDARY in token:
                    raise TopicMapError(f"{where} holds the section boundary mark")
                if token.split() != [token]:
                    raise TopicMapError(f"{where} holds whitespace")


def _terms_present(documents: Sequence[Sequence[Sequence[str]]], max_ngram: int) -> set[str]:
    """Return every term that the documents' sections form."""
    return {
        term
        for document in documents
        for tokens in document
        for term in section_terms(list(tokens), max_ngram)
    }
