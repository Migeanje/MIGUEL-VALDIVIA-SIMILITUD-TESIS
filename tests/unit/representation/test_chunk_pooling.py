"""Pooling of chunk vectors into section and document vectors (T14, D09; D27, C4 and C7).

The chunks are small stand-ins with hand-made vectors of 3 dimensions, so every expected
vector is worked out by hand; seeded random vectors check the norms.
"""

from typing import NamedTuple

import numpy as np
import pytest

from thematic_redundancy.representation.vectors import (
    l2_normalize,
    pool_chunk_vectors,
    variant_agreement,
)

FULL = ("title", "abstract", "objectives")
METADATA_ONLY = ("title", "abstract")


class Chunk(NamedTuple):
    doc_code: str
    section: str
    sections_used: tuple[str, ...]


# DOC-a uses its three sections, and DOC-b, a metadata-only thesis, its title and abstract.
CHUNKS = (
    Chunk("DOC-a", "title", FULL),
    Chunk("DOC-a", "abstract", FULL),
    Chunk("DOC-a", "abstract", FULL),
    Chunk("DOC-a", "objectives", FULL),
    Chunk("DOC-b", "title", METADATA_ONLY),
    Chunk("DOC-b", "abstract", METADATA_ONLY),
    Chunk("DOC-b", "abstract", METADATA_ONLY),
)
VECTORS = np.array(
    [[3, 0, 0], [0, 3, 0], [0, 1, 0], [0, 0, 6], [4, 0, 0], [0, 4, 0], [0, 0, 8]],
    dtype=np.float32,
)


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float64)
    return vector / np.sqrt(vector @ vector)


def assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_each_section_vector_is_the_mean_of_its_chunks() -> None:
    pooled = pool_chunk_vectors(CHUNKS, VECTORS)

    assert pooled.doc_codes == ("DOC-a", "DOC-b")
    assert pooled.sections == (
        ("DOC-a", "title"),
        ("DOC-a", "abstract"),
        ("DOC-a", "objectives"),
        ("DOC-b", "title"),
        ("DOC-b", "abstract"),
    )
    assert pooled.section_chunks == (1, 2, 1, 1, 2)
    # Raw means, not normalized.
    assert_exact(
        pooled.section_vectors,
        np.array([[3, 0, 0], [0, 2, 0], [0, 0, 6], [4, 0, 0], [0, 2, 4]], dtype=np.float64),
    )


def test_the_document_vector_weighs_each_used_section_equally_then_is_normalized() -> None:
    pooled = pool_chunk_vectors(CHUNKS, VECTORS)

    # DOC-a: ([3,0,0] + [0,2,0] + [0,0,6]) / 3 = [1, 2/3, 2], of norm 7/3, gives [3,2,6] / 7.
    # DOC-b, metadata-only, over 2 sections (C4): ([4,0,0] + [0,2,4]) / 2 = [2,1,2], of norm 3.
    assert_exact(
        pooled.section_mean, np.array([[3, 2, 6], [6, 3, 6]], dtype=np.float64) / [[7], [9]]
    )


def test_the_plain_mean_weighs_each_chunk_equally_then_is_normalized() -> None:
    pooled = pool_chunk_vectors(CHUNKS, VECTORS)

    assert_exact(pooled.plain_mean, np.stack([unit(3, 4, 6), unit(1, 1, 2)]))


@pytest.mark.parametrize("abstract_chunks", [1, 2, 5])
def test_a_section_weighs_the_same_whatever_its_number_of_chunks(abstract_chunks: int) -> None:
    chunks = [Chunk("DOC-c", "title", METADATA_ONLY)] + [
        Chunk("DOC-c", "abstract", METADATA_ONLY)
    ] * abstract_chunks
    vectors = np.array([[1, 0, 0]] + [[0, 1, 0]] * abstract_chunks, dtype=np.float32)

    pooled = pool_chunk_vectors(chunks, vectors)

    assert_exact(pooled.section_mean, unit(1, 1, 0)[None, :])
    assert_exact(pooled.plain_mean, unit(1, abstract_chunks, 0)[None, :])


def test_three_sections_weigh_a_third_each() -> None:
    chunks = [Chunk("DOC-d", section, FULL) for section in FULL for _ in range(len(section))]
    vectors = np.array(
        [[2, 0, 0]] * 5 + [[0, 2, 0]] * 8 + [[0, 0, 2]] * 10, dtype=np.float32
    )  # title, abstract and objectives have 5, 8 and 10 chunks.

    pooled = pool_chunk_vectors(chunks, vectors)

    assert_exact(pooled.section_mean, unit(1, 1, 1)[None, :])


@pytest.mark.parametrize("seed", range(5))
def test_every_document_vector_has_unit_norm_and_the_same_input_gives_the_same_output(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    chunks = [
        Chunk(f"DOC-{number}", section, FULL)
        for number in range(12)
        for section in FULL
        for _ in range(int(rng.integers(1, 5)))
    ]
    vectors = (rng.standard_normal((len(chunks), 16)) * 5).astype(np.float32)

    pooled = pool_chunk_vectors(chunks, vectors)
    again = pool_chunk_vectors(chunks, vectors.copy())

    for matrix in (pooled.section_mean, pooled.plain_mean):
        assert matrix.shape == (12, 16)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0, atol=1e-12)
    for name in ("section_vectors", "section_mean", "plain_mean"):
        assert np.array_equal(getattr(pooled, name), getattr(again, name))


def test_l2_normalize_scales_each_row_and_refuses_a_row_without_direction() -> None:
    assert_exact(l2_normalize(np.array([[0.0, 3.0, 4.0]])), np.array([[0.0, 0.6, 0.8]]))

    with pytest.raises(ValueError, match="1 vector has no direction"):
        l2_normalize(np.array([[1.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        (CHUNKS, VECTORS[:-1], "7 chunks need 7 vectors"),
        (CHUNKS, np.where(VECTORS == 8, np.nan, VECTORS), "not finite"),
        (
            (*CHUNKS[:-1], Chunk("DOC-b", "abstract", FULL)),
            VECTORS,
            "the chunks of DOC-b disagree on the sections used",
        ),
        (CHUNKS[1:], VECTORS[1:], "the sections of DOC-a with chunks are not its sections_used"),
        (
            (Chunk("DOC-e", "title", METADATA_ONLY), Chunk("DOC-e", "abstract", METADATA_ONLY)),
            np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32),
            "1 vector has no direction",
        ),
    ],
    ids=["rows", "not-finite", "disagreeing-rows", "missing-section", "opposite-sections"],
)
def test_pooling_refuses_inconsistent_input(
    chunks: tuple[Chunk, ...], vectors: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pool_chunk_vectors(chunks, vectors)


def test_the_variant_agreement_summarizes_the_cosine_of_each_thesis() -> None:
    pooled = pool_chunk_vectors(CHUNKS, VECTORS)
    # DOC-a: [3,2,6]/7 · [3,4,6]/√61 = 53 / (7√61); DOC-b: [2,1,2]/3 · [1,1,2]/√6 = 7 / (3√6).
    cosines = (53 / (7 * np.sqrt(61)), 7 / (3 * np.sqrt(6)))

    agreement = variant_agreement(pooled.section_mean, pooled.plain_mean, 0.96)

    assert agreement.threshold == 0.96
    assert agreement.minimum == pytest.approx(cosines[1], abs=1e-12)
    assert agreement.median == pytest.approx(sum(cosines) / 2, abs=1e-12)
    assert agreement.below == 1  # DOC-b, at 0.9526.


def test_a_cosine_equal_to_the_threshold_is_not_below_it() -> None:
    first, second = np.array([[1.0, 0.0]]), np.array([[0.5, np.sqrt(0.75)]])

    assert variant_agreement(first, second, 0.5).below == 0
    assert variant_agreement(first, second, 0.5000001).below == 1
