"""Smoke checks of the embeddings with the real models, offline (T14).

They read the saved chunk table of the configured snapshot through ``load_chunks`` and both
models from the local Hugging Face cache only; they are skipped when either is missing. They
embed only the chunks of two theses, the first that uses its three sections and the first
metadata-only one, into caches under a temporary directory, never the project's cache; they are
skipped when the table has no thesis of either kind. They assert shapes, values and counts only
and print no text. Run them with
``HF_HUB_OFFLINE=1 uv run pytest -m smoke tests/smoke/test_embeddings.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.representation.build_chunks import (
    ChunkRecord,
    chunks_manifest,
    load_chunks,
)
from thematic_redundancy.representation.build_embeddings import EmbeddedTexts, embed_texts
from thematic_redundancy.representation.st_encoder import (
    SentenceTransformerEncoder,
    load_sentence_encoder,
)
from thematic_redundancy.representation.vector_cache import ChunkVectorCache, model_slug
from thematic_redundancy.representation.vectors import (
    EncoderSettings,
    l2_normalize,
    pool_chunk_vectors,
)
from thematic_redundancy.shared.config import MODEL_MAX_SEQ_LENGTH, AppConfig, load_config

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}


class CountingEncoder:
    """Passes texts to a real encoder and counts them."""

    def __init__(self, encoder: SentenceTransformerEncoder) -> None:
        self.identity = encoder.identity
        self._encoder = encoder
        self.texts = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.texts += len(texts)
        return self._encoder.encode(texts)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(PROJECT_ROOT / "config" / "default.yaml")


@pytest.fixture(scope="module")
def interim_dir(config: AppConfig) -> Path:
    data_dir = config.paths.resolve_against(PROJECT_ROOT).data_dir
    return data_dir / INTERIM_DIR_NAME / config.embedding.expected.snapshot_id


@pytest.fixture(scope="module")
def sample(interim_dir: Path) -> tuple[ChunkRecord, ...]:
    try:
        records = load_chunks(interim_dir)
    except FileNotFoundError:
        pytest.skip("the chunk table of the configured snapshot has not been built")
    full = next((record.doc_code for record in records if len(record.sections_used) == 3), None)
    if full is None:
        pytest.skip("the chunk table has no thesis that uses its three sections")
    metadata_only = next(
        (record.doc_code for record in records if len(record.sections_used) == 2), None
    )
    if metadata_only is None:
        pytest.skip("the chunk table has no metadata-only thesis")
    return tuple(record for record in records if record.doc_code in (full, metadata_only))


@pytest.fixture(scope="module")
def encoders(config: AppConfig) -> dict[str, SentenceTransformerEncoder]:
    embedding = config.embedding
    settings = EncoderSettings(embedding.batch_size, embedding.device, embedding.threads)
    try:
        return {model: load_sentence_encoder(model, settings) for model in embedding.models}
    except OSError:
        pytest.skip("a configured model is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def caches(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("chunk_vectors")


@pytest.fixture(scope="module")
def embedded(
    sample: tuple[ChunkRecord, ...],
    encoders: dict[str, SentenceTransformerEncoder],
    caches: Path,
) -> dict[str, EmbeddedTexts]:
    texts = [record.text for record in sample]
    return {
        model: embed_texts(
            texts,
            encoder,
            ChunkVectorCache.open(caches / f"{model_slug(model)}.npy", encoder.identity.dimension),
        )
        for model, encoder in encoders.items()
    }


def test_each_model_is_the_cached_revision_whose_tokenizer_counted_the_chunks(
    encoders: dict[str, SentenceTransformerEncoder], interim_dir: Path, config: AppConfig
) -> None:
    revisions = {
        tokenizer.model_id: tokenizer.revision
        for tokenizer in chunks_manifest(interim_dir).tokenizers
    }

    assert list(encoders) == list(config.embedding.models) == list(DIMENSIONS)
    for model, encoder in encoders.items():
        identity = encoder.identity
        assert identity.revision == revisions[model]
        assert identity.max_seq_length == MODEL_MAX_SEQ_LENGTH
        assert identity.dimension == DIMENSIONS[model]


def test_the_sample_embeds_into_finite_raw_vectors(
    sample: tuple[ChunkRecord, ...], embedded: dict[str, EmbeddedTexts]
) -> None:
    distinct = len({record.text for record in sample})

    assert 5 <= len(sample) <= 20
    for model, result in embedded.items():
        assert (result.distinct_texts, result.from_cache, result.embedded) == (
            distinct,
            0,
            distinct,
        )
        assert result.vectors.dtype == np.float32
        assert result.vectors.shape == (len(sample), DIMENSIONS[model])
        assert np.isfinite(result.vectors).all()
        # The raw output, not normalized: the cache key says normalize=false.
        assert np.abs(np.linalg.norm(result.vectors, axis=1) - 1).max() > 1e-3


def test_a_rerun_reads_every_vector_from_the_cache_file(
    sample: tuple[ChunkRecord, ...],
    encoders: dict[str, SentenceTransformerEncoder],
    embedded: dict[str, EmbeddedTexts],
    caches: Path,
) -> None:
    texts = [record.text for record in sample]
    for model, encoder in encoders.items():
        counting = CountingEncoder(encoder)
        cache = ChunkVectorCache.open(
            caches / f"{model_slug(model)}.npy", encoder.identity.dimension
        )

        again = embed_texts(texts, counting, cache)

        assert counting.texts == 0
        assert (again.from_cache, again.embedded) == (again.distinct_texts, 0)
        assert np.array_equal(again.vectors, embedded[model].vectors)
        # The cached vectors are what the model gives for these texts.
        np.testing.assert_allclose(encoder.encode(texts), again.vectors, rtol=0, atol=1e-4)


def test_the_sample_pools_into_unit_document_vectors_with_equal_section_weights(
    sample: tuple[ChunkRecord, ...], embedded: dict[str, EmbeddedTexts]
) -> None:
    for result in embedded.values():
        pooled = pool_chunk_vectors(sample, result.vectors)

        assert len(pooled.doc_codes) == 2
        assert len(pooled.sections) == 5
        for matrix in (pooled.section_mean, pooled.plain_mean):
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0, atol=1e-9)
        # The metadata-only thesis averages its title and abstract only, half and half (C4).
        metadata_only = next(r.doc_code for r in sample if len(r.sections_used) == 2)
        rows = [row for row, (doc, _) in enumerate(pooled.sections) if doc == metadata_only]
        assert len(rows) == 2
        expected = l2_normalize(pooled.section_vectors[rows].mean(axis=0, keepdims=True))
        document = pooled.doc_codes.index(metadata_only)
        np.testing.assert_allclose(
            pooled.section_mean[document : document + 1], expected, rtol=0, atol=1e-12
        )
