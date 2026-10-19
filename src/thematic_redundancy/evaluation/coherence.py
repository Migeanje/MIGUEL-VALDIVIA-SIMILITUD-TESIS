"""NPMI topic coherence against a reference corpus (T17, D12).

A topic's coherence is the mean NPMI over every pair of its first ``top_n`` terms; the
corpus-level value is the mean over the topics. For two terms ``a`` and ``b``::

    NPMI(a, b) = log(p(a, b) / (p(a) p(b))) / -log p(a, b)

where ``p`` is the share of the co-occurrence units that hold the term, or both terms.

**Reference corpus.** A sequence of documents, each a sequence of segments (token lists), as a
thesis is a sequence of sections (D28 C1). A term of several words joined by single spaces,
such as a bigram of the TF-IDF vocabulary (D28 C3), occurs where its words appear consecutively
inside one segment, never across two. To make each section a document of its own, pass it as
a document of one segment.

**Co-occurrence unit**, chosen by the caller with no default (T18 declares it):

- ``"document"``: each document is one unit, and a term is in it when it occurs in any of its
  segments. A document with no token still counts as a unit.
- ``"window"``: a window of ``window_size`` consecutive tokens slides one token at a time over
  each segment, and each position is one unit. A segment of ``L`` tokens gives
  ``L - window_size + 1`` windows, one when it is shorter than the window, none when it is
  empty; no window spans two segments. A term is in a window when all its words are.

**Limits and missing terms**, with no epsilon smoothing:

- a pair that never co-occurs gets -1, the limit of NPMI as ``p(a, b)`` goes to 0;
- a pair present in every unit gets 1, the limit as ``p(a, b) = p(a) = p(b)`` goes to 1;
- a term that occurs in no unit has no NPMI. It is listed in the topic's ``absent_terms``; its
  pairs are ``None``, and so is the topic's coherence (``undefined_reason`` ``"absent_terms"``)
  and the corpus mean (``"undefined_topics"``), so a missing term is never dropped unseen.
  ``None`` with a reason is the convention of every metric in this package.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from scipy import sparse

CooccurrenceUnit = Literal["document", "window"]
UNITS: tuple[CooccurrenceUnit, ...] = ("document", "window")


@dataclass(frozen=True)
class TopicCoherence:
    """The NPMI of one topic."""

    terms: tuple[str, ...]
    """The topic's first ``top_n`` terms, in rank order."""
    pair_npmi: tuple[float | None, ...]
    """The NPMI of each pair of :attr:`terms`, in the order of
    ``itertools.combinations(terms, 2)``; ``None`` for a pair with an absent term."""
    npmi: float | None
    """The mean of :attr:`pair_npmi`; ``None`` when a term is absent."""
    absent_terms: tuple[str, ...]
    """The terms that occur in no unit, in rank order."""
    undefined_reason: Literal["absent_terms"] | None


@dataclass(frozen=True)
class CoherenceResult:
    """The NPMI of each topic and their mean, with the parameters that gave them."""

    unit: CooccurrenceUnit
    window_size: int | None
    top_n: int
    units: int
    """The co-occurrence units counted: documents or windows."""
    topics: tuple[TopicCoherence, ...]
    mean: float | None
    """The mean of the topics' NPMI; ``None`` when there is no topic or one is undefined."""
    undefined_reason: Literal["no_topics", "undefined_topics"] | None


def npmi(joint: int, first: int, second: int, units: int) -> float:
    """Return the NPMI of two terms from the number of units that hold both, each, and all.

    A pair that never co-occurs gets -1, and a pair present in every unit gets 1 (see the
    module's docstring).

    Raises:
        ValueError: if the counts are impossible, including a term in no unit.
    """
    if not (
        0 <= joint <= min(first, second)
        and 1 <= first <= units
        and 1 <= second <= units
        and first + second - joint <= units
    ):
        raise ValueError(
            f"impossible counts: {joint} joint, {first} and {second} single, {units} units"
        )
    if joint == 0:
        return -1.0
    if joint == units:
        return 1.0
    log_units = math.log(units)
    log_joint = math.log(joint) - log_units
    pmi = log_joint - (math.log(first) - log_units) - (math.log(second) - log_units)
    return pmi / -log_joint


def npmi_coherence(
    topics: Sequence[Sequence[str]],
    reference: Sequence[Sequence[Sequence[str]]],
    *,
    top_n: int,
    unit: CooccurrenceUnit,
    window_size: int | None,
) -> CoherenceResult:
    """Return the NPMI coherence of each topic's first ``top_n`` terms against ``reference``.

    Args:
        topics: each topic's terms in rank order, at least ``top_n`` of them.
        reference: the documents, each a sequence of segments, each a sequence of tokens.
        top_n: how many of each topic's first terms are scored, at least 2.
        unit: ``"document"`` or ``"window"`` (see the module's docstring).
        window_size: the tokens per window, at least 2 and at least the words of the longest
            term, for ``"window"``; ``None`` for ``"document"``.

    Raises:
        ValueError: if an argument is invalid, a topic has fewer than ``top_n`` terms, repeats
            one, or holds a malformed term.
        TypeError: if a document or a segment is a string rather than a sequence.
    """
    scored = _check_topics(topics, top_n)
    words = {term: _term_words(term) for topic in scored for term in topic}
    _check_unit(unit, window_size, max((len(w) for w in words.values()), default=1))
    terms = list(words)
    units, presence = _presence(reference, [words[term] for term in terms], window_size)
    counts = np.asarray(presence.sum(axis=0)).ravel()
    joint = (presence.T @ presence).toarray()
    column = {term: index for index, term in enumerate(terms)}

    results = []
    for topic in scored:
        indices = [column[term] for term in topic]
        pairs = tuple(
            npmi(int(joint[i, j]), int(counts[i]), int(counts[j]), units)
            if counts[i] and counts[j]
            else None
            for i, j in combinations(indices, 2)
        )
        absent = tuple(
            term for term, index in zip(topic, indices, strict=True) if not counts[index]
        )
        results.append(
            TopicCoherence(
                terms=topic,
                pair_npmi=pairs,
                npmi=None if absent else float(np.mean(pairs)),
                absent_terms=absent,
                undefined_reason="absent_terms" if absent else None,
            )
        )

    reason: Literal["no_topics", "undefined_topics"] | None = None
    if not results:
        reason = "no_topics"
    elif any(topic.npmi is None for topic in results):
        reason = "undefined_topics"
    return CoherenceResult(
        unit=unit,
        window_size=window_size,
        top_n=top_n,
        units=units,
        topics=tuple(results),
        mean=None if reason else float(np.mean([topic.npmi for topic in results])),
        undefined_reason=reason,
    )


def _check_topics(topics: Sequence[Sequence[str]], top_n: int) -> list[tuple[str, ...]]:
    """Return the first ``top_n`` terms of each topic, after checking them."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 2:
        raise ValueError(f"top_n must be an integer of at least 2, not {top_n!r}")
    scored = []
    for index, topic in enumerate(topics):
        first = tuple(topic[:top_n])
        if len(first) < top_n:
            raise ValueError(f"topic {index} has fewer than {top_n} terms")
        repeated = next((term for term in first if first.count(term) > 1), None)
        if repeated is not None:
            raise ValueError(f"topic {index} repeats the term {repeated!r} in its first {top_n}")
        scored.append(first)
    return scored


def _term_words(term: str) -> tuple[str, ...]:
    """Return the words of ``term``, which are joined by single spaces."""
    words = tuple(term.split(" ")) if isinstance(term, str) else ()
    if not words or "" in words:
        raise ValueError(f"{term!r} is not a term: words joined by single spaces")
    return words


def _check_unit(unit: str, window_size: int | None, longest: int) -> None:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, not {unit!r}")
    if unit == "document":
        if window_size is not None:
            raise ValueError(f"window_size must be None for document units, not {window_size!r}")
        return
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < max(2, longest)
    ):
        raise ValueError(
            f"window_size must be an integer of at least 2 and at least the {longest} words of "
            f"the longest term for window units, not {window_size!r}"
        )


def _presence(
    reference: Sequence[Sequence[Sequence[str]]],
    terms: Sequence[tuple[str, ...]],
    window_size: int | None,
) -> tuple[int, sparse.csr_matrix]:
    """Return the number of units and a 0/1 matrix of which unit (row) holds which term
    (column); windows when ``window_size`` is given, documents otherwise."""
    starts: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    for column, words in enumerate(terms):
        starts.setdefault(words[0], []).append((column, words))
    rows: list[np.ndarray] = [np.zeros(0, dtype=np.int64)]
    columns: list[np.ndarray] = [np.zeros(0, dtype=np.int64)]
    units = 0
    for document in reference:
        for segment in _segments(document):
            windows = 0
            if window_size is not None and segment:
                windows = max(len(segment) - window_size + 1, 1)
            for at, token in enumerate(segment):
                for column, words in starts.get(token, ()):
                    if segment[at : at + len(words)] != words:
                        continue
                    if window_size is None:
                        holders = np.array([units])
                    else:
                        # The windows that start at most at ``at`` and end at least at the
                        # term's last word.
                        first = max(0, at + len(words) - window_size)
                        holders = units + np.arange(first, min(at, windows - 1) + 1)
                    rows.append(holders)
                    columns.append(np.full(holders.size, column))
            units += windows
        if window_size is None:
            units += 1
    row_index = np.concatenate(rows)
    presence = sparse.csr_matrix(
        (np.ones(row_index.size, dtype=np.int64), (row_index, np.concatenate(columns))),
        shape=(units, len(terms)),
    )
    presence.sum_duplicates()
    presence.data[:] = 1
    return units, presence


def _segments(document: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    """Return the segments of ``document`` as tuples of tokens."""
    if isinstance(document, str):
        raise TypeError("each document must be a sequence of segments, not a string")
    segments = []
    for segment in document:
        if isinstance(segment, str):
            raise TypeError("each segment must be a sequence of tokens, not a string")
        segments.append(tuple(segment))
    return segments
