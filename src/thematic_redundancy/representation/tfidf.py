"""The TF-IDF baseline's pure core: terms, document vectors and their agreement (T15, D28).

- **Terms** (:func:`section_terms`, :func:`thesis_terms`). A section gives its tokens, then,
  when ``max_ngram`` is 2, its pairs of adjacent tokens; no bigram spans the end of a section.
- **Vectors** (:func:`fit_tfidf`). One vocabulary and one idf are fit over the theses, so a
  term's document frequency counts theses, not sections. Two document vectors share them:
  ``concatenated``, one vector over all the sections of a thesis, and ``section_mean``, the
  equal-weight mean of the thesis's section vectors, each L2-normalized first, then
  L2-normalized, as D09 builds the SBERT document vector. Both are canonical CSR matrices of
  float64 rows, so their bytes repeat.
- **Agreement** (:func:`variant_agreement`): the cosine between the two vectors of each thesis.
- **Stopword hash** (:func:`lemma_list_sha256`): the hash that pins the domain list (D26).

The cleaning that gives the tokens sits behind the :class:`SectionCleaner` port. Nothing here
reads or writes a file, or loads spaCy.
"""

import functools
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TFIDF_VERSION = 1
"""Version of the rules of this module; it changes whenever the same tokens would give other
terms or vectors."""

NORM = "l2"
USE_IDF = True
SMOOTH_IDF = True
DTYPE = np.float64
"""Vectorizer settings that D28 fixes in code: cosine needs L2 rows, and the plan varies none."""


class TfidfCheckError(ValueError):
    """The vectors break a rule that the build enforces, so nothing was written."""


@dataclass(frozen=True)
class CleanerIdentity:
    """What a manifest records of the cleaner that gave the tokens."""

    spacy_version: str
    model_name: str
    model_version: str
    spacy_stopwords: int
    """Size of spaCy's stopword list."""
    spacy_stopwords_sha256: str
    """:func:`lemma_list_sha256` of spaCy's stopword list."""


class SectionCleaner(Protocol):
    """Port that turns texts into the full-clean tokens that TF-IDF reads."""

    identity: CleanerIdentity

    def clean(self, texts: Sequence[str]) -> list[list[str]]:
        """Return the tokens of each text, in order."""
        ...


@dataclass(frozen=True)
class TfidfSettings:
    """The vectorizer parameters that the configuration declares (``tfidf``)."""

    max_ngram: int
    min_df: int
    """Fewest theses that a term must occur in."""
    max_df: float
    """Largest share of the theses that a term may occur in; 1.0 turns the cap off."""
    sublinear_tf: bool


@dataclass(frozen=True)
class TfidfMatrices:
    """Outcome of :func:`fit_tfidf`: one column per term, one row per thesis."""

    terms: tuple[str, ...]
    """The vocabulary, sorted in code-point order."""
    document_frequency: np.ndarray
    """Theses that hold each term."""
    idf: np.ndarray
    section_mean: sparse.csr_matrix
    concatenated: sparse.csr_matrix


@dataclass(frozen=True)
class VariantAgreement:
    """Cosine between the two document vectors of each thesis."""

    threshold: float
    minimum: float
    median: float
    below: int
    """Theses whose cosine is below ``threshold``."""


def section_terms(tokens: Sequence[str], max_ngram: int) -> list[str]:
    """Return the tokens of one section, then, when ``max_ngram`` is 2, each pair of adjacent
    tokens joined by a space."""
    terms = list(tokens)
    if max_ngram >= 2:
        terms += [f"{first} {second}" for first, second in pairwise(tokens)]
    return terms


def thesis_terms(sections: Sequence[Sequence[str]], max_ngram: int) -> list[str]:
    """Return the terms of each section of a thesis in turn, so no bigram spans two sections.

    It is the vectorizer's analyzer: a thesis is the token lists of its sections.
    """
    return [term for tokens in sections for term in section_terms(tokens, max_ngram)]


def fit_tfidf(
    documents: Sequence[Sequence[Sequence[str]]], settings: TfidfSettings
) -> TfidfMatrices:
    """Fit the vocabulary and idf over ``documents`` and return both document vectors.

    Each document is a thesis: the token lists of its sections, in section order. A thesis
    whose terms all fall outside the vocabulary has a zero row in both matrices.

    Raises:
        TfidfCheckError: if ``min_df`` is above ``max_df`` times the theses, or if no term is
            left.
    """
    theses = len(documents)
    if settings.min_df > settings.max_df * theses:
        raise TfidfCheckError(
            f"min_df ({settings.min_df}) is above max_df ({settings.max_df}) × {theses} theses, "
            "so no term could be kept"
        )
    vectorizer = TfidfVectorizer(
        analyzer=functools.partial(thesis_terms, max_ngram=settings.max_ngram),
        min_df=settings.min_df,
        max_df=settings.max_df,
        sublinear_tf=settings.sublinear_tf,
        norm=NORM,
        use_idf=USE_IDF,
        smooth_idf=SMOOTH_IDF,
        dtype=DTYPE,
    )
    try:
        concatenated = vectorizer.fit_transform(documents)
    except ValueError:
        # sklearn's empty-vocabulary errors; they quote no document.
        raise TfidfCheckError(
            f"no term occurs in at least {settings.min_df} theses and in at most a "
            f"{settings.max_df} share of them"
        ) from None
    # Each section alone, with the same vocabulary and idf, as one L2-normalized row.
    sections = vectorizer.transform([[tokens] for document in documents for tokens in document])
    counts = [len(document) for document in documents]
    weights = np.repeat([1.0 / max(count, 1) for count in counts], counts)
    mean = sparse.csr_matrix(
        (weights, (np.repeat(np.arange(theses), counts), np.arange(sections.shape[0]))),
        shape=(theses, sections.shape[0]),
    )
    section_mean = _canonical(normalize(mean @ sections, norm=NORM))
    concatenated = _canonical(concatenated)
    terms = tuple(str(term) for term in vectorizer.get_feature_names_out())
    return TfidfMatrices(
        terms=terms,
        document_frequency=np.bincount(concatenated.indices, minlength=len(terms)),
        idf=np.asarray(vectorizer.idf_, dtype=DTYPE),
        section_mean=section_mean,
        concatenated=concatenated,
    )


def _canonical(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Return ``matrix`` as float64 CSR with each row's column indices in increasing order, so
    equal matrices have equal arrays, and equal bytes once saved."""
    canonical = sparse.csr_matrix(matrix, dtype=DTYPE)
    canonical.sort_indices()
    return canonical


def zero_rows(matrix: sparse.csr_matrix) -> int:
    """Return the rows of ``matrix`` that hold no value: theses without a cosine."""
    return int(np.count_nonzero(np.diff(matrix.indptr) == 0))


def variant_agreement(
    primary: sparse.csr_matrix, other: sparse.csr_matrix, threshold: float
) -> VariantAgreement:
    """Summarize the cosine between the rows of two matrices of L2-normalized rows, one row per
    thesis, counting the theses strictly below ``threshold``."""
    cosines = np.asarray(primary.multiply(other).sum(axis=1)).ravel()
    return VariantAgreement(
        threshold=threshold,
        minimum=float(cosines.min()),
        median=float(np.median(cosines)),
        below=int(np.count_nonzero(cosines < threshold)),
    )


def lemma_list_sha256(lemmas: Iterable[str]) -> str:
    """Return the SHA-256 of ``lemmas`` sorted in code-point order, joined by line feeds and
    encoded in UTF-8: the hash that pins the domain list (D26), whatever its comments, blank
    lines or line order."""
    return hashlib.sha256("\n".join(sorted(lemmas)).encode()).hexdigest()
