"""Pure core of the embeddings (T14): the sentence-encoder port, and the pooling of chunk
vectors into section and document vectors (D09; D27, C4 and C7).

- :class:`SentenceEncoder` is the port that the build depends on: it turns texts into the raw
  vectors of one model, not normalized. The sentence-transformers adapter
  (:mod:`~.st_encoder`) sits at the edge, and unit tests use fake encoders.
- :func:`pool_chunk_vectors` averages the raw vectors of the chunks of each section into its
  section vector, then gives each thesis two document vectors, both L2-normalized:

  - ``section_mean`` (D09, the default): the equal-weight mean of the vectors of the sections
    in its ``sections_used``, so a metadata-only thesis averages its title and abstract (C4);
  - ``plain_mean``, the sensitivity variant (C7): the mean of all its chunk vectors, which
    weighs each section by its number of chunks.

- :func:`variant_agreement` summarizes the cosine between the two document vectors of each
  thesis.

The sums run in float64, and only the document vectors are normalized.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EncoderSettings:
    """How an encoder runs: the chunks per batch, the torch device, and the CPU threads."""

    batch_size: int
    device: str
    threads: int


@dataclass(frozen=True)
class EncoderIdentity:
    """Which model encodes: its id, its revision, the most tokens it reads, and the dimension
    of its vectors."""

    model_id: str
    revision: str
    """Commit of the model in the Hugging Face cache; part of the cache key."""
    max_seq_length: int
    dimension: int


class SentenceEncoder(Protocol):
    """Port that turns texts into the raw vectors of one model."""

    identity: EncoderIdentity

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one vector per text, in order, as an array of shape
        ``(len(texts), identity.dimension)``: the model's output, not normalized."""
        ...


class PooledChunk(Protocol):
    """What the pooling reads of a chunk, such as a row of the chunk table."""

    @property
    def doc_code(self) -> str: ...

    @property
    def section(self) -> str: ...

    @property
    def sections_used(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class PooledVectors:
    """Section and document vectors, in the order of the first chunk of each."""

    doc_codes: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    """``(doc_code, section)`` of each row of :attr:`section_vectors`."""
    section_chunks: tuple[int, ...]
    """Chunks averaged into each row of :attr:`section_vectors`."""
    section_vectors: np.ndarray
    """Mean of the raw vectors of the chunks of each section, not normalized."""
    section_mean: np.ndarray
    """Per thesis, the equal-weight mean of its section vectors, L2-normalized (D09)."""
    plain_mean: np.ndarray
    """Per thesis, the mean of all its chunk vectors, L2-normalized (the sensitivity variant)."""


@dataclass(frozen=True)
class VariantAgreement:
    """Cosine between the two document vectors of each thesis, summarized."""

    threshold: float
    minimum: float
    median: float
    below: int
    """Theses whose cosine is below :attr:`threshold`."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return ``vectors`` with each row scaled to an L2 norm of 1.

    Raises:
        ValueError: if a row has a norm of 0, or one that is not finite, so it has no direction.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    lost = int((~np.isfinite(norms) | (norms == 0)).sum())
    if lost:
        raise ValueError(f"{lost} vector{'s have' if lost > 1 else ' has'} no direction")
    return vectors / norms


def pool_chunk_vectors(chunks: Sequence[PooledChunk], vectors: np.ndarray) -> PooledVectors:
    """Pool the raw ``vectors`` of ``chunks``, one row per chunk, into section and document
    vectors.

    The chunks of a thesis may come in any order; its sections and the theses keep the order
    of their first chunk.

    Raises:
        ValueError: if ``vectors`` does not have one finite row per chunk; if the chunks of a
            thesis disagree on its ``sections_used``, or their sections are not exactly those;
            or if a document vector is 0, which has no direction.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(chunks):
        raise ValueError(f"{len(chunks)} chunks need {len(chunks)} vectors, not {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("a chunk vector holds a value that is not finite")
    section_rows: dict[tuple[str, str], int] = {}
    used: dict[str, tuple[str, ...]] = {}
    found: dict[str, list[str]] = {}
    for chunk in chunks:
        if (chunk.doc_code, chunk.section) not in section_rows:
            section_rows[(chunk.doc_code, chunk.section)] = len(section_rows)
            found.setdefault(chunk.doc_code, []).append(chunk.section)
        sections_used = tuple(chunk.sections_used)
        if used.setdefault(chunk.doc_code, sections_used) != sections_used:
            raise ValueError(f"the chunks of {chunk.doc_code} disagree on the sections used")
    for doc_code, sections_used in used.items():
        if sorted(found[doc_code]) != sorted(sections_used):
            raise ValueError(f"the sections of {doc_code} with chunks are not its sections_used")
    doc_rows = {doc_code: row for row, doc_code in enumerate(used)}
    section_of_chunk = np.array(
        [section_rows[(chunk.doc_code, chunk.section)] for chunk in chunks], dtype=np.intp
    )
    doc_of_section = np.array([doc_rows[doc] for doc, _ in section_rows], dtype=np.intp)
    section_vectors = _group_mean(matrix, section_of_chunk, len(section_rows))
    return PooledVectors(
        doc_codes=tuple(used),
        sections=tuple(section_rows),
        section_chunks=tuple(np.bincount(section_of_chunk, minlength=len(section_rows)).tolist()),
        section_vectors=section_vectors,
        section_mean=l2_normalize(_group_mean(section_vectors, doc_of_section, len(doc_rows))),
        plain_mean=l2_normalize(
            _group_mean(matrix, doc_of_section[section_of_chunk], len(doc_rows))
        ),
    )


def _group_mean(values: np.ndarray, groups: np.ndarray, size: int) -> np.ndarray:
    """Return the mean of the rows of ``values`` in each of ``size`` groups, in group order;
    ``groups`` gives the group of each row, and each group has a row."""
    sums = np.zeros((size, values.shape[1]), dtype=np.float64)
    np.add.at(sums, groups, values)
    return sums / np.bincount(groups, minlength=size)[:, None]


def variant_agreement(first: np.ndarray, second: np.ndarray, threshold: float) -> VariantAgreement:
    """Summarize the cosine between each row of ``first`` and the same row of ``second``, whose
    rows have an L2 norm of 1, as document vectors do.

    Raises:
        ValueError: if there is no row to compare.
    """
    cosines = np.einsum("ij,ij->i", first, second)
    if not len(cosines):
        raise ValueError("there is no document vector to compare")
    return VariantAgreement(
        threshold=threshold,
        minimum=float(cosines.min()),
        median=float(np.median(cosines)),
        below=int((cosines < threshold).sum()),
    )
