"""Cosine search over document vectors (T22): scores, top-k, the query left out, and the
cosines of given pairs, for dense (SBERT) and sparse (TF-IDF) vectors alike.

Scores are checked against hand-computed cosines and against scikit-learn's
``cosine_similarity``; sparse and dense inputs must give the same ranking.
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from thematic_redundancy.redundancy.search import (
    Neighbor,
    cosine_scores,
    cosine_top_k,
    nearest_documents,
    pair_cosines,
)

SEED = 20261019

# Unit vectors at known angles; rows 1 and 2 have the same cosine with the query [1, 0].
PLANE = np.array(
    [
        [1.0, 0.0],  # 0: cosine 1 with the query
        [0.6, 0.8],  # 1: cosine 0.6
        [0.6, -0.8],  # 2: cosine 0.6, a tie with row 1
        [0.0, 1.0],  # 3: cosine 0
        [-1.0, 0.0],  # 4: cosine -1
    ]
)
QUERY = np.array([1.0, 0.0])


def indices(neighbors: tuple[Neighbor, ...]) -> list[int]:
    return [neighbor.index for neighbor in neighbors]


def scores_of(neighbors: tuple[Neighbor, ...]) -> list[float]:
    return [neighbor.score for neighbor in neighbors]


def random_corpus(rows: int = 40, columns: int = 12) -> np.ndarray:
    return np.random.default_rng(SEED).normal(size=(rows, columns))


def sparse_corpus(rows: int = 40, columns: int = 30) -> sparse.csr_matrix:
    rng = np.random.default_rng(SEED)
    dense = rng.random((rows, columns)) * (rng.random((rows, columns)) < 0.3)
    dense[:, 0] += 0.1  # no empty row
    return sparse.csr_matrix(dense)


# --- Scores --------------------------------------------------------------------------------


def test_scores_are_the_hand_computed_cosines() -> None:
    assert cosine_scores(PLANE, QUERY) == pytest.approx([1.0, 0.6, 0.6, 0.0, -1.0])


def test_scores_ignore_the_length_of_every_vector() -> None:
    scaled = PLANE * np.array([[10.0], [0.5], [3.0], [7.0], [2.0]])

    assert cosine_scores(scaled, QUERY * 4) == pytest.approx([1.0, 0.6, 0.6, 0.0, -1.0])


def test_scores_match_scikit_learn_for_dense_and_sparse_vectors() -> None:
    dense = random_corpus()
    matrix = sparse_corpus()

    assert cosine_scores(dense, dense[3]) == pytest.approx(
        cosine_similarity(dense, dense[3:4]).ravel(), abs=1e-12
    )
    assert cosine_scores(matrix, matrix[5]) == pytest.approx(
        cosine_similarity(matrix, matrix[5]).ravel(), abs=1e-12
    )


def test_float32_vectors_are_scored_in_float64() -> None:
    dense = random_corpus().astype(np.float32)

    scores = cosine_scores(dense, dense[0])

    assert scores.dtype == np.float64
    assert scores == pytest.approx(
        cosine_similarity(dense.astype(np.float64), dense[:1].astype(np.float64)).ravel(),
        abs=1e-6,
    )


@pytest.mark.parametrize("query", [np.array([[1.0, 0.0]]), sparse.csr_matrix([[1.0, 0.0]])])
def test_a_one_row_query_is_accepted(query) -> None:
    assert cosine_scores(PLANE, query) == pytest.approx([1.0, 0.6, 0.6, 0.0, -1.0])


# --- Top-k ---------------------------------------------------------------------------------


def test_top_k_ranks_by_score_and_breaks_ties_by_the_lower_index() -> None:
    result = cosine_top_k(PLANE, QUERY, k=4, exclude=None)

    assert indices(result) == [0, 1, 2, 3]
    assert scores_of(result) == pytest.approx([1.0, 0.6, 0.6, 0.0])
    assert all(isinstance(neighbor.score, float) for neighbor in result)


def test_tied_rows_keep_the_index_order_wherever_they_sit() -> None:
    corpus = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    result = cosine_top_k(corpus, np.array([0.0, 2.0]), k=5, exclude=None)

    assert indices(result) == [0, 2, 4, 1, 3]


def test_the_excluded_row_is_left_out_and_the_next_one_takes_its_place() -> None:
    result = cosine_top_k(PLANE, QUERY, k=2, exclude=0)

    assert indices(result) == [1, 2]


def test_k_larger_than_the_corpus_returns_every_row() -> None:
    assert indices(cosine_top_k(PLANE, QUERY, k=50, exclude=None)) == [0, 1, 2, 3, 4]
    assert indices(cosine_top_k(PLANE, QUERY, k=50, exclude=4)) == [0, 1, 2, 3]


def test_nearest_documents_uses_a_row_as_the_query_and_leaves_it_out() -> None:
    corpus = random_corpus()
    expected = np.argsort(-cosine_similarity(corpus, corpus[7:8]).ravel(), kind="stable")
    expected = [int(row) for row in expected if row != 7][:5]

    result = nearest_documents(corpus, 7, k=5)

    assert indices(result) == expected
    assert 7 not in indices(nearest_documents(corpus, 7, k=100))
    assert len(nearest_documents(corpus, 7, k=100)) == 39


def test_an_identical_copy_of_the_query_row_is_still_found() -> None:
    corpus = np.vstack([PLANE, PLANE[1]])

    result = nearest_documents(corpus, 1, k=1)

    assert indices(result) == [5]
    assert result[0].score == pytest.approx(1.0)


def test_sparse_and_dense_vectors_give_the_same_ranking() -> None:
    matrix = sparse_corpus()
    dense = matrix.toarray()

    for row in (0, 11, 39):
        from_sparse = nearest_documents(matrix, row, k=10)
        from_dense = nearest_documents(dense, row, k=10)
        assert indices(from_sparse) == indices(from_dense)
        assert scores_of(from_sparse) == pytest.approx(scores_of(from_dense), abs=1e-12)


def test_the_search_repeats_exactly() -> None:
    corpus = random_corpus()

    assert nearest_documents(corpus, 3, k=8) == nearest_documents(corpus.copy(), 3, k=8)


# --- Pairs ---------------------------------------------------------------------------------


def test_pair_cosines_are_the_cosines_of_each_pair() -> None:
    dense = random_corpus()
    pairs = [(0, 1), (5, 2), (39, 0), (7, 8)]
    expected = [cosine_similarity(dense[[a]], dense[[b]])[0, 0] for a, b in pairs]

    assert pair_cosines(dense, pairs) == pytest.approx(expected, abs=1e-12)
    assert pair_cosines(sparse_corpus(), pairs) == pytest.approx(
        [cosine_similarity(sparse_corpus()[a], sparse_corpus()[b])[0, 0] for a, b in pairs],
        abs=1e-12,
    )


def test_pair_cosines_are_symmetric_and_scale_free() -> None:
    scaled = PLANE * 5.0

    assert pair_cosines(scaled, [(0, 1), (1, 0), (1, 2)]) == pytest.approx([0.6, 0.6, -0.28])


def test_no_pair_gives_an_empty_array() -> None:
    assert pair_cosines(PLANE, []).shape == (0,)


# --- Invalid input -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("documents", "query", "message"),
    [
        (PLANE, np.array([1.0, 0.0, 0.0]), "dimensions"),
        (PLANE, np.array([0.0, 0.0]), "no direction"),
        (np.vstack([PLANE, [0.0, 0.0]]), QUERY, "no direction"),
        (np.vstack([PLANE, [np.nan, 1.0]]), QUERY, "no direction"),
        (sparse.csr_matrix(np.vstack([PLANE, [0.0, 0.0]])), QUERY, "no direction"),
        (PLANE[0], QUERY, "2-D"),
        (PLANE, np.ones((2, 2)), "one row"),
    ],
    ids=[
        "query-length",
        "zero-query",
        "zero-row",
        "nan-row",
        "sparse-zero-row",
        "1-D-corpus",
        "two-row-query",
    ],
)
def test_invalid_vectors_are_refused(documents, query, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_scores(documents, query)


@pytest.mark.parametrize(
    ("k", "exclude", "message"),
    [(0, None, "k"), (-1, None, "k"), (2, 5, "exclude"), (2, -1, "exclude")],
)
def test_invalid_search_arguments_are_refused(k: int, exclude: int | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_top_k(PLANE, QUERY, k=k, exclude=exclude)


@pytest.mark.parametrize("index", [-1, 5])
def test_a_row_outside_the_corpus_is_refused(index: int) -> None:
    with pytest.raises(ValueError, match="index"):
        nearest_documents(PLANE, index, k=2)


@pytest.mark.parametrize(
    ("pairs", "message"),
    [([(0, 5)], "outside"), ([(-1, 2)], "outside"), ([(2, 2)], "different"), ([(0, 1, 2)], "two")],
)
def test_invalid_pairs_are_refused(pairs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pair_cosines(PLANE, pairs)
