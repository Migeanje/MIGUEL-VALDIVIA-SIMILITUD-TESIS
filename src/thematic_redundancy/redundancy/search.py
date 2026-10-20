"""Cosine search over document vectors, the direct-redundancy signal's similarity (T22, D13).

The vectors are one row per document, dense (the SBERT document vectors, D09) or sparse (the
TF-IDF matrix, D28). Every row and the query are L2-normalized here, in float64, so the score
is the cosine whatever the vectors' lengths or dtype; a row or a query with no direction (a
norm of 0, or one that is not finite) is refused, as
:func:`~thematic_redundancy.representation.vectors.l2_normalize` refuses it.

- :func:`cosine_scores`: the cosine of every row with a query.
- :func:`cosine_top_k`: the ``k`` rows most similar to a query, by score descending; rows with
  the same score keep the order of their index, so the ranking repeats exactly. ``exclude``
  names the query's own row when the query is a document of the corpus, and the search leaves
  it out; an identical copy of that document in another row is still found.
- :func:`nearest_documents`: the same search with a row of the corpus as the query, left out.
- :func:`pair_cosines`: the cosine of each given pair of rows, as the labeled pairs need it.

``k`` larger than the rows that can be returned gives them all. Nothing here reads a file.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from thematic_redundancy.representation.vectors import l2_normalize

Vectors = np.ndarray | sparse.sparray | sparse.spmatrix
"""One row per document: a dense 2-D array, or a scipy sparse matrix or array."""


@dataclass(frozen=True)
class Neighbor:
    """A row of the corpus and its cosine with the query."""

    index: int
    score: float


def cosine_scores(documents: Vectors, query: Vectors) -> np.ndarray:
    """Return the cosine of each row of ``documents`` with ``query``, as float64.

    ``query`` is a 1-D array, or a single row, dense or sparse.

    Raises:
        ValueError: if ``documents`` is not 2-D, ``query`` is not one vector of as many
            dimensions, or a row or the query has no direction.
    """
    unit_rows = _unit_rows(documents)
    return _scores(unit_rows, _unit_query(query, unit_rows.shape[1]))


def cosine_top_k(
    documents: Vectors, query: Vectors, *, k: int, exclude: int | None
) -> tuple[Neighbor, ...]:
    """Return the ``k`` rows most similar to ``query``, best first, leaving out row
    ``exclude`` (``None`` leaves out nothing).

    Raises:
        ValueError: as :func:`cosine_scores`, or if ``k`` is below 1 or ``exclude`` is not a
            row of ``documents``.
    """
    unit_rows = _unit_rows(documents)
    _check_k(k)
    if exclude is not None:
        _check_row(exclude, unit_rows.shape[0], "exclude")
    return _ranked(_scores(unit_rows, _unit_query(query, unit_rows.shape[1])), k, exclude)


def nearest_documents(documents: Vectors, index: int, *, k: int) -> tuple[Neighbor, ...]:
    """Return the ``k`` rows most similar to row ``index``, best first, without the row itself.

    Raises:
        ValueError: as :func:`cosine_scores`, or if ``k`` is below 1 or ``index`` is not a row
            of ``documents``.
    """
    unit_rows = _unit_rows(documents)
    _check_k(k)
    _check_row(index, unit_rows.shape[0], "index")
    query = _dense_row(unit_rows, index)
    return _ranked(_scores(unit_rows, query), k, index)


def pair_cosines(documents: Vectors, pairs: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
    """Return the cosine of each pair of rows ``(a, b)`` of ``documents``, in the pairs' order.

    Raises:
        ValueError: if ``documents`` is not 2-D or a row has no direction, or a pair is not two
            different rows of ``documents``.
    """
    unit_rows = _unit_rows(documents)
    array = np.asarray(pairs, dtype=np.int64)
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("each pair must hold two row indices")
    rows = unit_rows.shape[0]
    outside = (array < 0) | (array >= rows)
    if outside.any():
        raise ValueError(f"a pair names a row outside the {rows} documents: {array[outside][0]}")
    if (array[:, 0] == array[:, 1]).any():
        raise ValueError("each pair must join two different documents")
    left, right = unit_rows[array[:, 0]], unit_rows[array[:, 1]]
    if sparse.issparse(unit_rows):
        return np.asarray(left.multiply(right).sum(axis=1), dtype=np.float64).ravel()
    return np.einsum("ij,ij->i", left, right)


def _unit_rows(documents: Vectors) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(documents):
        if documents.ndim != 2:
            raise ValueError(f"the documents must be 2-D, not of shape {documents.shape}")
        matrix = sparse.csr_matrix(documents, dtype=np.float64)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        lost = int((~np.isfinite(norms) | (norms == 0)).sum())
        if lost:
            raise ValueError(f"{lost} vector{'s have' if lost > 1 else ' has'} no direction")
        return normalize(matrix, norm="l2")
    array = np.asarray(documents, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"the documents must be 2-D, not of shape {array.shape}")
    return l2_normalize(array)


def _unit_query(query: Vectors, dimensions: int) -> np.ndarray:
    vector = query.toarray() if sparse.issparse(query) else np.asarray(query, dtype=np.float64)
    if vector.ndim == 2:
        if vector.shape[0] != 1:
            raise ValueError(f"the query must be one row, not of shape {vector.shape}")
        vector = vector[0]
    if vector.ndim != 1 or vector.shape[0] != dimensions:
        raise ValueError(
            f"the query has shape {vector.shape}, but the documents have {dimensions} dimensions"
        )
    return l2_normalize(vector.astype(np.float64)[np.newaxis, :])[0]


def _dense_row(unit_rows: np.ndarray | sparse.csr_matrix, index: int) -> np.ndarray:
    if sparse.issparse(unit_rows):
        return unit_rows[index].toarray()[0]
    return unit_rows[index]


def _scores(unit_rows: np.ndarray | sparse.csr_matrix, unit_query: np.ndarray) -> np.ndarray:
    return np.asarray(unit_rows @ unit_query, dtype=np.float64).ravel()


def _ranked(scores: np.ndarray, k: int, exclude: int | None) -> tuple[Neighbor, ...]:
    order = np.lexsort((np.arange(scores.size), -scores))
    if exclude is not None:
        order = order[order != exclude]
    return tuple(Neighbor(int(row), float(scores[row])) for row in order[:k])


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
        raise ValueError(f"k must be an integer of at least 1, not {k!r}")


def _check_row(row: int, rows: int, label: str) -> None:
    if isinstance(row, bool) or not isinstance(row, int | np.integer) or not 0 <= row < rows:
        raise ValueError(f"{label} must be a row of the {rows} documents, not {row!r}")
