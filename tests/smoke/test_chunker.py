"""Smoke checks of the chunker on the real corpus with the real tokenizers (T13, D27).

They read the saved fichas of the configured snapshot through ``load_fichas`` and the two
tokenizers from the local Hugging Face cache only, so they run offline; they are skipped when
either is missing on this machine. They assert counts and properties only and print no text.
Run them with ``uv run pytest -m smoke tests/smoke/test_chunker.py``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pytest

from thematic_redundancy.corpus.build_fichas import load_fichas
from thematic_redundancy.corpus.ficha import Ficha
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.representation.build_chunks import (
    CHUNKS_MANIFEST_FILE_NAME,
    ChunkRecord,
    chunk_fichas,
    chunks_manifest,
    count_chunks,
    load_chunks,
    section_texts,
)
from thematic_redundancy.representation.chunker import MaxTokenCounter
from thematic_redundancy.representation.hf_tokenizers import (
    TOKENIZER_FILE_NAME,
    HuggingFaceTokenCounter,
    load_token_counter,
)
from thematic_redundancy.shared.config import AppConfig, load_config

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(PROJECT_ROOT / "config" / "default.yaml")


@pytest.fixture(scope="module")
def interim_dir(config: AppConfig) -> Path:
    data_dir = config.paths.resolve_against(PROJECT_ROOT).data_dir
    return data_dir / INTERIM_DIR_NAME / config.chunking.expected.snapshot_id


@pytest.fixture(scope="module")
def fichas(interim_dir: Path) -> tuple[Ficha, ...]:
    try:
        return load_fichas(interim_dir)
    except FileNotFoundError:
        pytest.skip("the fichas of the configured snapshot are not on this machine")


@pytest.fixture(scope="module")
def tokenizers(config: AppConfig) -> dict[str, HuggingFaceTokenCounter]:
    try:
        return {model: load_token_counter(model) for model in config.embedding.models}
    except OSError:
        pytest.skip("a configured tokenizer is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def records(
    fichas: tuple[Ficha, ...], tokenizers: dict[str, HuggingFaceTokenCounter], config: AppConfig
) -> tuple[ChunkRecord, ...]:
    counter = MaxTokenCounter(tuple(tokenizers.values()))
    return chunk_fichas(fichas, counter, config.chunking.max_tokens)


def test_every_chunk_fits_each_real_tokenizer_on_its_own(
    records: tuple[ChunkRecord, ...],
    tokenizers: dict[str, HuggingFaceTokenCounter],
    config: AppConfig,
) -> None:
    limit = config.chunking.max_tokens
    for tokenizer in tokenizers.values():
        counts = [tokenizer.count(record.text) for record in records]
        assert sum(count > limit for count in counts) == 0
    assert max(record.n_tokens for record in records) <= limit


def test_the_chunks_recover_every_section_of_every_included_thesis(
    records: tuple[ChunkRecord, ...], fichas: tuple[Ficha, ...], config: AppConfig
) -> None:
    by_section: dict[tuple[str, str], list[ChunkRecord]] = defaultdict(list)
    for record in records:
        by_section[(record.doc_code, record.section)].append(record)
    texts = {
        (ficha.doc_code, section): text
        for ficha in fichas
        if ficha.include
        for section, text in section_texts(ficha).items()
    }
    expected = config.fichas.expected
    assert len(texts) == 2 * expected.included + expected.objectives_extracted
    assert by_section.keys() == texts.keys()
    for key, text in texts.items():
        chunks = by_section[key]
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
        cursor = 0
        for chunk in chunks:
            assert text[cursor : chunk.char_start].strip() == ""
            assert text[chunk.char_start : chunk.char_end] == chunk.text
            cursor = chunk.char_end
        assert text[cursor:].strip() == ""


def test_the_chunk_counts_are_those_declared_for_the_snapshot(
    records: tuple[ChunkRecord, ...], config: AppConfig
) -> None:
    expected = config.chunking.expected.model_dump(exclude={"snapshot_id"})

    assert count_chunks(records) == expected


def test_both_tokenizers_count_every_section_alike(
    fichas: tuple[Ficha, ...], tokenizers: dict[str, HuggingFaceTokenCounter], config: AppConfig
) -> None:
    # Compared even when both tokenizer.json files are byte-identical, as on this corpus, so
    # that a tokenizer that counts differently for any other reason fails here.
    texts = [text for ficha in fichas if ficha.include for text in section_texts(ficha).values()]
    differing = sum(
        len({tokenizer.count(text) for tokenizer in tokenizers.values()}) > 1 for text in texts
    )

    expected = config.fichas.expected
    assert len(texts) == 2 * expected.included + expected.objectives_extracted
    assert differing == 0


def test_the_manifest_records_the_sha256_of_each_tokenizer_file_on_disk(
    interim_dir: Path, config: AppConfig
) -> None:
    if not (interim_dir / CHUNKS_MANIFEST_FILE_NAME).is_file():
        pytest.skip("the chunk table of the configured snapshot has not been built")
    from transformers.utils import cached_file

    tokenizers = chunks_manifest(interim_dir).tokenizers

    assert [tokenizer.model_id for tokenizer in tokenizers] == list(config.embedding.models)
    for tokenizer in tokenizers:
        try:
            found = cached_file(tokenizer.model_id, TOKENIZER_FILE_NAME, local_files_only=True)
        except OSError:
            # Some transformers releases raise instead of returning None.
            found = None
        if found is None:
            pytest.skip("a configured tokenizer is not in the local Hugging Face cache")
        on_disk = hashlib.sha256(Path(found).read_bytes()).hexdigest()
        assert tokenizer.tokenizer_sha256 == on_disk


def test_a_saved_chunk_table_equals_a_fresh_chunking(
    records: tuple[ChunkRecord, ...], interim_dir: Path
) -> None:
    if not (interim_dir / CHUNKS_MANIFEST_FILE_NAME).is_file():
        pytest.skip("the chunk table of the configured snapshot has not been built")

    assert load_chunks(interim_dir) == records
