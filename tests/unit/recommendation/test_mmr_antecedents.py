"""Antecedents ranked by Maximal Marginal Relevance (T26), on hand-built and seeded vectors.

The hand-built corpus has four theses around the query ``(1, 0, 0)``: ``d0`` is the most
similar, ``d1`` a near-duplicate of ``d0``, ``d2`` a little less similar but in another
direction, and ``d3`` off the query's subject. The seeded corpora check MMR against a plain
reimplementation of Carbonell and Goldstein's rule, written here apart from the module.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from thematic_redundancy.recommendation.antecedents import (
    AntecedentRanking,
    RecommendationError,
    rank_antecedents,
)
from thematic_redundancy.redundancy.search import cosine_top_k
from thematic_redundancy.shared.config import load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
SEED = 20261019
LAMBDA = 0.7

QUERY = np.array([1.0, 0.0, 0.0])
DOCUMENTS = np.array(
    [
        [1.0, 0.25, 0.0],
        [1.0, 0.26, 0.0],
        [1.0, -0.30, 0.10],
        [0.2, 1.0, 0.0],
    ]
)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def rank(
    documents=DOCUMENTS,
    proposal=QUERY,
    *,
    k: int = 4,
    mmr_lambda: float = LAMBDA,
    pool_size: int | None = None,
    exclude: int | None = None,
) -> AntecedentRanking:
    return rank_antecedents(
        documents, proposal, k=k, mmr_lambda=mmr_lambda, pool_size=pool_size, exclude=exclude
    )


def indices(ranking: AntecedentRanking) -> list[int]:
    return [antecedent.index for antecedent in ranking.antecedents]


def reference_mmr(
    documents: np.ndarray, query: np.ndarray, k: int, mmr_lambda: float, candidates: list[int]
) -> list[int]:
    """Carbonell and Goldstein's rule, step by step, ties to the more similar, then lower row."""
    similarity = {row: cosine(documents[row], query) for row in candidates}
    picked: list[int] = []
    while len(picked) < min(k, len(candidates)):
        best = None
        for row in candidates:
            if row in picked:
                continue
            redundancy = max((cosine(documents[row], documents[s]) for s in picked), default=0.0)
            key = (mmr_lambda * similarity[row] - (1 - mmr_lambda) * redundancy, similarity[row])
            if best is None or key > best[0]:
                best = (key, row)
        picked.append(best[1])
    return picked


def seeded(rows: int, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(SEED)
    return generator.normal(size=(rows, dimensions)), generator.normal(size=dimensions)


# --- The MMR rule ---------------------------------------------------------------------------


def test_mmr_picks_a_diverse_thesis_over_a_near_duplicate():
    assert indices(rank()) == [0, 2, 1, 3]
    assert indices(rank(mmr_lambda=1.0)) == [0, 1, 2, 3]


def test_each_antecedent_records_its_similarity_redundancy_and_score():
    ranking = rank()

    picked: list[int] = []
    for rank_number, antecedent in enumerate(ranking.antecedents, start=1):
        row = antecedent.index
        similarity = cosine(DOCUMENTS[row], QUERY)
        redundancy = max(cosine(DOCUMENTS[row], DOCUMENTS[s]) for s in picked) if picked else None
        assert antecedent.rank == rank_number
        assert antecedent.similarity == pytest.approx(similarity, abs=1e-12)
        if redundancy is None:
            assert antecedent.redundancy is None
        else:
            assert antecedent.redundancy == pytest.approx(redundancy, abs=1e-12)
        assert antecedent.mmr_score == pytest.approx(
            LAMBDA * similarity - (1 - LAMBDA) * (redundancy or 0.0), abs=1e-12
        )
        picked.append(row)
    # The third pick, d1, is redundant by its highest cosine with d0 and d2, not their mean.
    assert ranking.antecedents[2].redundancy == pytest.approx(
        cosine(DOCUMENTS[1], DOCUMENTS[0]), abs=1e-12
    )


def test_lambda_one_is_the_plain_top_k_by_similarity():
    documents, query = seeded(40, 8)

    ranking = rank(documents, query, k=7, mmr_lambda=1.0, exclude=3)

    plain = cosine_top_k(documents, query, k=7, exclude=3)
    assert indices(ranking) == [neighbor.index for neighbor in plain]
    assert [a.similarity for a in ranking.antecedents] == [n.score for n in plain]
    assert [a.mmr_score for a in ranking.antecedents] == [n.score for n in plain]


def test_lambda_zero_takes_the_least_redundant_thesis_at_each_step():
    assert indices(rank(mmr_lambda=0.0)) == [0, 3, 2, 1]

    documents, query = seeded(30, 6)
    ranking = rank(documents, query, k=6, mmr_lambda=0.0)
    picked = [ranking.antecedents[0].index]
    for antecedent in ranking.antecedents[1:]:
        others = [row for row in range(30) if row not in picked]
        lowest = min(max(cosine(documents[row], documents[s]) for s in picked) for row in others)
        assert antecedent.redundancy == pytest.approx(lowest, abs=1e-12)
        picked.append(antecedent.index)


@pytest.mark.parametrize("pool_size", [None, 20])
def test_mmr_agrees_with_a_plain_reimplementation_at_every_step(pool_size):
    documents, query = seeded(60, 12)
    pool = [n.index for n in cosine_top_k(documents, query, k=pool_size or 60, exclude=None)]

    ranking = rank(documents, query, k=5, pool_size=pool_size)

    assert indices(ranking) == reference_mmr(documents, query, 5, LAMBDA, pool)


# --- Candidates, ties and k -----------------------------------------------------------------


def test_the_proposals_own_row_is_left_out_but_its_copy_is_found():
    documents = np.vstack([DOCUMENTS, QUERY, QUERY])  # rows 4 and 5 are the proposal

    excluded = rank(documents, k=3, exclude=4)
    kept = rank(documents, k=3, exclude=None)

    assert 4 not in indices(excluded)
    assert indices(excluded)[0] == 5
    assert excluded.candidates == 5
    assert indices(kept)[0] == 4


def test_exact_ties_go_to_the_more_similar_thesis_then_to_the_lower_row():
    documents = np.vstack([DOCUMENTS, DOCUMENTS[0]])  # row 4 repeats row 0

    ranking = rank(documents, k=5, mmr_lambda=1.0)

    assert indices(ranking)[:2] == [0, 4]
    # With λ = 0 every first-step score is 0: the most similar thesis goes first.
    assert indices(rank(mmr_lambda=0.0, k=1)) == [0]


@pytest.mark.parametrize(("k", "returned"), [(1, 1), (2, 2), (3, 3), (4, 4), (9, 4)])
def test_k_antecedents_are_returned_or_every_candidate(k, returned):
    ranking = rank(k=k)

    assert len(ranking.antecedents) == returned
    assert ranking.k == k


def test_the_pool_holds_the_most_similar_theses_only():
    two = rank(pool_size=2)
    three = rank(pool_size=3)
    without_best = rank(pool_size=2, exclude=0)

    assert indices(two) == [0, 1]  # d2, the diverse thesis, is outside the pool
    assert (two.pool_size, two.candidates) == (2, 2)
    assert indices(three) == [0, 2, 1]
    assert indices(without_best) == [1, 2]
    assert (without_best.candidates, without_best.exclude) == (2, 0)


def test_the_ranking_records_its_arguments_and_repeats_exactly():
    documents, query = seeded(25, 5)

    first = rank(documents, query, k=4, pool_size=None, exclude=7)

    assert first == rank(documents, query, k=4, pool_size=None, exclude=7)
    assert (first.k, first.mmr_lambda, first.pool_size, first.exclude) == (4, LAMBDA, None, 7)
    assert first.candidates == 24


def test_sparse_documents_rank_as_their_dense_copy():
    generator = np.random.default_rng(SEED)
    dense = generator.random((30, 20)) * (generator.random((30, 20)) < 0.3)
    dense[:, 0] += 0.01  # every row keeps a direction
    query = generator.random(20)

    from_sparse = rank(sparse.csr_matrix(dense), sparse.csr_matrix(query), k=5, exclude=2)
    from_dense = rank(dense, query, k=5, exclude=2)

    assert indices(from_sparse) == indices(from_dense)
    for left, right in zip(from_sparse.antecedents, from_dense.antecedents, strict=True):
        assert left.mmr_score == pytest.approx(right.mmr_score, abs=1e-12)


def test_the_configured_recommender_gives_five_antecedents_at_lambda_0_7():
    recommender = load_config(DEFAULT_CONFIG_PATH).recommender
    documents, query = seeded(30, 6)

    ranking = rank(documents, query, k=recommender.top_k, mmr_lambda=recommender.mmr_lambda)

    assert (recommender.top_k, recommender.mmr_lambda) == (5, 0.7)
    assert (recommender.neighbor_topics, recommender.nearby_gaps) == (3, 3)
    assert len(ranking.antecedents) == 5


# --- Invalid arguments ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"k": 0},
        {"k": True},
        {"k": 2.0},
        {"mmr_lambda": -0.1},
        {"mmr_lambda": 1.1},
        {"mmr_lambda": math.nan},
        {"mmr_lambda": True},
        {"mmr_lambda": "0.7"},
        {"pool_size": 0},
        {"pool_size": True},
    ],
)
def test_an_invalid_argument_is_refused(arguments):
    with pytest.raises(RecommendationError):
        rank(**arguments)


def test_no_documents_and_an_outside_row_are_refused():
    with pytest.raises(RecommendationError, match="no document"):
        rank(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="exclude"):
        rank(exclude=4)
    with pytest.raises(ValueError, match="direction"):
        rank(proposal=np.zeros(3))
