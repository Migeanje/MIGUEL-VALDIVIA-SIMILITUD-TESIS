"""Antecedents of a proposal: the theses to read first, ranked by Maximal Marginal Relevance
(T26; thesis plan section 3.6.7, the ``recommender`` row of the initial configuration).

**MMR** (Carbonell and Goldstein, 1998). The antecedents are picked one at a time from a
candidate pool; each step picks the candidate ``d`` that maximizes

    ``λ·sim(q, d) − (1 − λ)·max_{s ∈ S} sim(d, s)``

where ``q`` is the proposal, ``S`` the antecedents already picked and ``sim`` the cosine in the
original space of the document vectors (D14), computed by
:func:`~thematic_redundancy.redundancy.search.cosine_scores` and
:func:`~thematic_redundancy.redundancy.search.cosine_top_k` (T22). The redundancy term of the
first step, with ``S`` empty, is 0, so the first antecedent is always the candidate most similar
to the proposal (at ``λ = 0`` every first score is 0, and the tie rule below picks it). Cosines
can be negative, so the term is the true maximum, never floored at 0. ``λ = 1`` gives the plain
top-``k`` by similarity; ``λ = 0`` ignores the proposal after the first pick and takes, each
step, the candidate least similar to those already picked. The configuration declares ``top_k``
5 and ``mmr_lambda`` 0.7.

**Pool.** ``pool_size`` is passed with no default: ``None`` makes every document of the corpus a
candidate; an integer ``N`` makes the ``N`` documents most similar to the proposal the
candidates, in :func:`~thematic_redundancy.redundancy.search.cosine_top_k`'s order. ``exclude``
names the proposal's own row when the proposal is a thesis of the corpus, and that row is never a
candidate; an identical copy of it in another row still is, as in T22's search.

**Ties.** A step's exact tie in MMR score goes to the candidate more similar to the proposal,
then to the lower row, so the ranking repeats exactly.

The documents are dense (the SBERT document vectors, D09) or scipy sparse (the TF-IDF matrix,
D28): both go through T22's cosine. ``k`` larger than the candidates gives them all. Nothing here
reads a file or holds a thesis text. An invalid argument raises :class:`RecommendationError`;
a document or a proposal with no direction raises T22's :class:`ValueError`.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import sparse

from thematic_redundancy.redundancy.search import Vectors, cosine_scores, cosine_top_k


class RecommendationError(ValueError):
    """An argument of the recommender breaks its contract, so nothing was recommended."""


@dataclass(frozen=True)
class Antecedent:
    """One antecedent, in the order MMR picked it."""

    rank: int
    """1 for the first pick."""
    index: int
    """The thesis's row in the documents."""
    similarity: float
    """The cosine between the thesis and the proposal."""
    redundancy: float | None
    """The highest cosine between the thesis and the antecedents picked before it; ``None``
    for the first pick, whose redundancy term is 0."""
    mmr_score: float
    """``λ·similarity − (1 − λ)·redundancy`` at the step that picked it."""


@dataclass(frozen=True)
class AntecedentRanking:
    """The antecedents of one proposal and the arguments that ranked them."""

    antecedents: tuple[Antecedent, ...]
    k: int
    mmr_lambda: float
    pool_size: int | None
    """As requested; ``None`` for the whole corpus."""
    candidates: int
    """The documents MMR chose among: the pool, without the excluded row."""
    exclude: int | None


def rank_antecedents(
    documents: Vectors,
    proposal: Vectors,
    *,
    k: int,
    mmr_lambda: float,
    pool_size: int | None,
    exclude: int | None,
) -> AntecedentRanking:
    """Return the ``k`` antecedents of ``proposal`` among ``documents``, picked by MMR.

    Args:
        documents: one row per thesis, dense or sparse.
        proposal: the proposal's vector, in the documents' space: a 1-D array or one row.
        k: the antecedents to return (``recommender.top_k``), at least 1.
        mmr_lambda: ``λ``, from 0 to 1 (``recommender.mmr_lambda``).
        pool_size: the candidates, the most similar first; ``None`` for every document.
        exclude: the proposal's own row when it is a thesis of the corpus, else ``None``.

    Raises:
        RecommendationError: if ``k``, ``mmr_lambda`` or ``pool_size`` is invalid, or there is
            no document.
        ValueError: as :func:`~thematic_redundancy.redundancy.search.cosine_top_k` raises.
    """
    _check_count(k, "k", minimum=1)
    if (
        isinstance(mmr_lambda, bool)
        or not isinstance(mmr_lambda, Real)
        or not (math.isfinite(mmr_lambda) and 0.0 <= mmr_lambda <= 1.0)
    ):
        raise RecommendationError(f"mmr_lambda must be a number from 0 to 1, not {mmr_lambda!r}")
    if pool_size is not None:
        _check_count(pool_size, "pool_size", minimum=1)
    matrix = _row_matrix(documents)
    if matrix.ndim != 2:
        raise RecommendationError(f"the documents must be 2-D, not of shape {matrix.shape}")
    rows = matrix.shape[0]
    if rows == 0:
        raise RecommendationError("there is no document to rank")
    pool = cosine_top_k(
        matrix, proposal, k=rows if pool_size is None else pool_size, exclude=exclude
    )
    indices = np.array([neighbor.index for neighbor in pool], dtype=np.int64)
    similarities = np.array([neighbor.score for neighbor in pool], dtype=np.float64)
    pool_rows = matrix[indices]
    weight = float(mmr_lambda)
    # The highest cosine with the picked antecedents; a cosine can be negative, so the running
    # maximum starts from the first pick's cosines, and the empty set's term is 0.
    redundancy = np.zeros(indices.size)
    picked = np.zeros(indices.size, dtype=bool)
    antecedents = []
    for rank in range(1, min(k, indices.size) + 1):
        scores = weight * similarities - (1.0 - weight) * redundancy
        # The pool is ordered by similarity, then row, so argmax's first maximum breaks ties.
        best = int(np.argmax(np.where(picked, -np.inf, scores)))
        antecedents.append(
            Antecedent(
                rank=rank,
                index=int(indices[best]),
                similarity=float(similarities[best]),
                redundancy=None if rank == 1 else float(redundancy[best]),
                mmr_score=float(scores[best]),
            )
        )
        cosines = cosine_scores(pool_rows, pool_rows[best])
        redundancy = cosines if rank == 1 else np.maximum(redundancy, cosines)
        picked[best] = True
    return AntecedentRanking(
        antecedents=tuple(antecedents),
        k=k,
        mmr_lambda=weight,
        pool_size=pool_size,
        candidates=int(indices.size),
        exclude=exclude,
    )


def _row_matrix(documents: Vectors) -> np.ndarray | sparse.csr_matrix:
    """Return the documents as rows that can be indexed: CSR when sparse, float64 otherwise."""
    if sparse.issparse(documents):
        return sparse.csr_matrix(documents)
    return np.asarray(documents, dtype=np.float64)


def _check_count(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise RecommendationError(f"{name} must be an integer of at least {minimum}, not {value!r}")
