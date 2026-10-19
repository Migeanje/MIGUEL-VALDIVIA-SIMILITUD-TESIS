"""The embeddings of a snapshot: chunk vectors cached, pooled, checked and saved (T14).

The chunk table is synthetic and saved on disk as the chunk build does, so the build reads it
through ``load_chunks``. The encoders are fakes: each text gets a vector drawn from a generator
seeded with the text's SHA-256, so a text always gets the same vector, and the fakes record
every text that they encode.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
import yaml

import thematic_redundancy.representation.build_embeddings as embeddings_module
from thematic_redundancy.representation.build_chunks import BUILD_COMMAND as CHUNKS_BUILD_COMMAND
from thematic_redundancy.representation.build_chunks import (
    CHUNKER_VERSION,
    CHUNKS_FILE_NAME,
    CHUNKS_MANIFEST_FILE_NAME,
    ChunkRecord,
    ChunksManifest,
    chunks_manifest,
)
from thematic_redundancy.representation.build_embeddings import (
    BUILD_COMMAND,
    DOCUMENT_FILE_NAMES,
    EMBEDDER_VERSION,
    EMBEDDINGS_DIR_NAME,
    EMBEDDINGS_MANIFEST_FILE_NAME,
    SECTION_VECTORS_FILE_NAME,
    SECTIONS_FILE_NAME,
    EmbeddingsCheckError,
    build_embeddings,
    load_embeddings,
    main,
)
from thematic_redundancy.representation.vector_cache import ChunkVectorCache, cache_key
from thematic_redundancy.representation.vectors import (
    EncoderIdentity,
    EncoderSettings,
    pool_chunk_vectors,
)
from thematic_redundancy.shared.config import (
    AppConfig,
    EmbeddingConfig,
    EmbeddingCountsConfig,
    load_config,
)
from thematic_redundancy.shared.storage import ParquetTableStore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
SNAPSHOT = "20261002T150405Z"
RUN_TIME = datetime(2026, 10, 16, 9, 0, 0, tzinfo=UTC)
MODELS = {"org/small": 4, "org/large": 6}
REVISIONS = {"org/small": "1" * 40, "org/large": "2" * 40}
FULL = ("title", "abstract", "objectives")
METADATA_ONLY = ("title", "abstract")


def chunk(doc_code: str, section: str, index: int, text: str, used: Sequence[str]) -> ChunkRecord:
    return ChunkRecord(
        doc_code=doc_code,
        section=section,  # type: ignore[arg-type]
        chunk_index=index,
        char_start=0,
        char_end=len(text),
        n_tokens=len(text.split()) + 2,
        cut_kind="sentence",
        sections_used=tuple(used),  # type: ignore[arg-type]
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
    )


# Three theses, the second metadata-only (C4); the third repeats the first one's title, so the
# 10 chunks hold 9 distinct texts.
RECORDS = (
    chunk("DOC-a", "title", 0, "Mejora del proceso de una planta", FULL),
    chunk("DOC-a", "abstract", 0, "Se estudia el proceso de una planta textil.", FULL),
    chunk("DOC-a", "abstract", 1, "Se propone un plan de mejora continua.", FULL),
    chunk("DOC-a", "objectives", 0, "Diseñar un prototipo de control.", FULL),
    chunk("DOC-b", "title", 0, "Modelo de control para una red", METADATA_ONLY),
    chunk("DOC-b", "abstract", 0, "Se analiza la red de datos de la planta.", METADATA_ONLY),
    chunk("DOC-b", "abstract", 1, "Se mide la calidad del servicio.", METADATA_ONLY),
    chunk("DOC-c", "title", 0, "Mejora del proceso de una planta", FULL),
    chunk("DOC-c", "abstract", 0, "Se evalúa un sistema de mantenimiento.", FULL),
    chunk("DOC-c", "objectives", 0, "Reducir las paradas no planificadas.", FULL),
)
DISTINCT_TEXTS = len({record.text for record in RECORDS})


def fake_vector(text: str, dimension: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return (np.random.default_rng(seed).standard_normal(dimension) * 3).astype(np.float32)


@dataclass
class FakeEncoder:
    identity: EncoderIdentity
    interrupt_on_call: int | None = None
    shape_error: bool = False
    calls: list[list[str]] = field(default_factory=list)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if len(self.calls) == self.interrupt_on_call:
            raise KeyboardInterrupt
        dimension = self.identity.dimension + self.shape_error
        return np.stack([fake_vector(text, dimension) for text in texts])


@dataclass
class FakeLoader:
    """Loads a fake encoder per model, and records the settings that it was given."""

    max_seq_length: int = 128
    interrupt: dict[str, int] = field(default_factory=dict)
    shape_error: bool = False
    encoders: dict[str, FakeEncoder] = field(default_factory=dict)
    settings: list[EncoderSettings] = field(default_factory=list)

    def __call__(self, model_id: str, settings: EncoderSettings) -> FakeEncoder:
        self.settings.append(settings)
        identity = EncoderIdentity(
            model_id, REVISIONS[model_id], self.max_seq_length, MODELS[model_id]
        )
        encoder = FakeEncoder(identity, self.interrupt.get(model_id), self.shape_error)
        self.encoders[model_id] = encoder
        return encoder

    def encoded(self, model_id: str) -> list[str]:
        return [text for call in self.encoders[model_id].calls for text in call]


def interim(project_root: Path) -> Path:
    return project_root / "data" / "interim" / SNAPSHOT


def outputs(project_root: Path) -> Path:
    return interim(project_root) / EMBEDDINGS_DIR_NAME


def cache_file(project_root: Path, model_id: str) -> Path:
    name = model_id.replace("/", "--")
    return project_root / "data" / "cache" / "chunk_vectors" / f"{name}.npy"


def write_chunk_outputs(interim_dir: Path, records: Sequence[ChunkRecord] = RECORDS) -> str:
    """Save ``records`` as the chunk build does, over any earlier chunk outputs, and return the
    table's SHA-256."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    ParquetTableStore().write(list(records), interim_dir / CHUNKS_FILE_NAME, model_cls=ChunkRecord)
    manifest = ChunksManifest(
        snapshot_id=SNAPSHOT,
        created_at=RUN_TIME,
        chunker_version=CHUNKER_VERSION,
        fichas_sha256="a" * 64,
        max_tokens=128,
        tokenizers=(),
        chunks_file=CHUNKS_FILE_NAME,
        chunks_sha256=hashlib.sha256((interim_dir / CHUNKS_FILE_NAME).read_bytes()).hexdigest(),
        counts={},
        cut_kinds={},
        sections_used={},
        sections_recovered=0,
        seconds=0.1,
    )
    (interim_dir / CHUNKS_MANIFEST_FILE_NAME).write_text(manifest.model_dump_json(), "utf-8")
    return manifest.chunks_sha256


def embedding_config(**changes: Any) -> EmbeddingConfig:
    values: dict[str, Any] = {
        "models": tuple(MODELS),
        "batch_size": 4,
        "device": "cpu",
        "threads": 2,
        "document_vector": "section_mean",
        "sensitivity_cosine_threshold": 0.95,
        "expected": EmbeddingCountsConfig(snapshot_id=SNAPSHOT, documents=3, chunks=10),
    }
    return EmbeddingConfig(**{**values, **changes})


def app_config(**changes: Any) -> AppConfig:
    return load_config(DEFAULT_CONFIG_PATH).model_copy(
        update={"embedding": embedding_config(**changes)}
    )


def build(
    project_root: Path, loader: FakeLoader, config: AppConfig | None = None, **seams: Any
) -> Any:
    return build_embeddings(
        config or app_config(),
        project_root,
        snapshot_id=SNAPSHOT,
        load_encoder=loader,
        clock=lambda: RUN_TIME,
        timer=lambda: 0.0,
        **seams,
    )


def pooled(model_id: str) -> Any:
    vectors = np.stack([fake_vector(record.text, MODELS[model_id]) for record in RECORDS])
    return pool_chunk_vectors(RECORDS, vectors)


def output_files(project_root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(outputs(project_root)).as_posix(): path.read_bytes()
        for path in sorted(outputs(project_root).rglob("*"))
        if path.is_file()
    }


def test_the_build_embeds_each_distinct_text_once_per_model_and_saves_the_vectors(
    tmp_path: Path,
) -> None:
    chunks_sha256 = write_chunk_outputs(interim(tmp_path))
    loader = FakeLoader()

    built = build(tmp_path, loader)

    assert loader.settings == [EncoderSettings(batch_size=4, device="cpu", threads=2)] * 2
    manifest = built.manifest
    assert manifest.counts == {"documents": 3, "sections": 8, "chunks": 10}
    assert (manifest.chunker_version, manifest.chunks_sha256) == (CHUNKER_VERSION, chunks_sha256)
    assert manifest.document_variant == "section_mean"
    assert manifest.encoder.model_dump() == {"batch_size": 4, "device": "cpu", "threads": 2}
    assert manifest.cache_dir == "cache/chunk_vectors"
    for name in ("max_seq_length", "model_id", "model_revision", "normalize", "text"):
        assert name in manifest.cache_key_scheme
    for record, (model_id, dimension) in zip(manifest.models, MODELS.items(), strict=True):
        assert sorted(loader.encoded(model_id)) == sorted({r.text for r in RECORDS})
        assert (record.model_id, record.revision) == (model_id, REVISIONS[model_id])
        assert (record.dimension, record.max_seq_length, record.normalize) == (
            dimension,
            128,
            False,
        )
        assert (record.distinct_texts, record.from_cache, record.embedded) == (9, 0, 9)
        assert record.cache_file == f"cache/chunk_vectors/{model_id.replace('/', '--')}.npy"
        # The cache holds the raw vector of each distinct text under its D27 C6 key.
        cache = ChunkVectorCache.open(cache_file(tmp_path, model_id), dimension)
        keys = [
            cache_key(
                model_id=model_id,
                model_revision=REVISIONS[model_id],
                max_seq_length=128,
                normalize=False,
                text=r.text,
            )
            for r in RECORDS
        ]
        assert len(cache) == 9 and cache.missing(keys) == []
        raw = np.stack([fake_vector(r.text, dimension) for r in RECORDS])
        assert np.array_equal(cache.vectors(keys), raw)
        directory = outputs(tmp_path) / record.directory
        assert record.files == {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (*DOCUMENT_FILE_NAMES.values(), SECTION_VECTORS_FILE_NAME)
        }
        expected = pooled(model_id)
        loaded = load_embeddings(interim(tmp_path), model_id)
        assert (loaded.model_id, loaded.revision, loaded.variant) == (
            model_id,
            REVISIONS[model_id],
            "section_mean",
        )
        assert loaded.doc_codes == ("DOC-a", "DOC-b", "DOC-c")
        assert loaded.vectors.dtype == np.float32 and loaded.vectors.shape == (3, dimension)
        np.testing.assert_allclose(loaded.vectors, expected.section_mean, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0, atol=1e-6)
        assert loaded.sections == expected.sections
        np.testing.assert_allclose(loaded.section_vectors, expected.section_vectors, atol=1e-6)
    assert (
        manifest.sections_sha256
        == hashlib.sha256((outputs(tmp_path) / SECTIONS_FILE_NAME).read_bytes()).hexdigest()
    )
    saved = (outputs(tmp_path) / EMBEDDINGS_MANIFEST_FILE_NAME).read_text("utf-8")
    assert json.loads(saved) == json.loads(manifest.model_dump_json())
    assert not any(record.text in saved for record in RECORDS)


def test_the_reader_gives_the_variant_asked_for_or_the_configured_one(tmp_path: Path) -> None:
    write_chunk_outputs(interim(tmp_path))
    build(tmp_path, FakeLoader(), app_config(document_vector="plain_mean"))
    expected = pooled("org/large")

    configured = load_embeddings(interim(tmp_path), "org/large")
    asked = load_embeddings(interim(tmp_path), "org/large", variant="section_mean")

    assert configured.variant == "plain_mean"
    np.testing.assert_allclose(configured.vectors, expected.plain_mean, rtol=0, atol=1e-6)
    assert asked.variant == "section_mean"
    np.testing.assert_allclose(asked.vectors, expected.section_mean, rtol=0, atol=1e-6)


def test_the_manifest_summarizes_the_cosine_between_the_two_variants(tmp_path: Path) -> None:
    write_chunk_outputs(interim(tmp_path))

    built = build(tmp_path, FakeLoader(), app_config(sensitivity_cosine_threshold=0.999999))

    for record in built.manifest.models:
        expected = pooled(record.model_id)
        cosines = np.einsum("ij,ij->i", expected.section_mean, expected.plain_mean)
        agreement = record.variant_agreement
        assert agreement.threshold == 0.999999
        assert agreement.minimum == pytest.approx(cosines.min(), abs=1e-6)
        assert agreement.median == pytest.approx(np.median(cosines), abs=1e-6)
        # DOC-c has one chunk per section, so its two document vectors are the same.
        assert cosines[2] == pytest.approx(1.0, abs=1e-12)
        assert agreement.below == 2


def test_a_rerun_embeds_nothing_and_writes_the_same_vectors(tmp_path: Path) -> None:
    write_chunk_outputs(interim(tmp_path))
    build(tmp_path, FakeLoader())
    first = output_files(tmp_path)
    caches = {model: cache_file(tmp_path, model).read_bytes() for model in MODELS}
    loader = FakeLoader()

    again = build(tmp_path, loader)

    assert all(loader.encoded(model) == [] for model in MODELS)
    for record in again.manifest.models:
        assert (record.distinct_texts, record.from_cache, record.embedded) == (9, 9, 0)
    second = output_files(tmp_path)
    assert second.keys() == first.keys()
    for name in first.keys() - {EMBEDDINGS_MANIFEST_FILE_NAME}:
        assert second[name] == first[name], name
    assert {model: cache_file(tmp_path, model).read_bytes() for model in MODELS} == caches


def test_an_interrupted_embedding_keeps_its_finished_checkpoints_in_the_cache(
    tmp_path: Path,
) -> None:
    write_chunk_outputs(interim(tmp_path))

    # With checkpoints of 4 texts, the second call of the first model is interrupted.
    with pytest.raises(KeyboardInterrupt):
        build(tmp_path, FakeLoader(interrupt={"org/small": 2}), checkpoint=4)

    assert not outputs(tmp_path).exists()
    assert not cache_file(tmp_path, "org/large").exists()
    loader = FakeLoader()
    rerun = build(tmp_path, loader, checkpoint=4)
    assert [len(call) for call in loader.encoders["org/small"].calls] == [4, 1]
    assert [len(call) for call in loader.encoders["org/large"].calls] == [4, 4, 1]
    small = rerun.manifest.models[0]
    assert (small.from_cache, small.embedded) == (4, 5)
    np.testing.assert_allclose(
        load_embeddings(interim(tmp_path), "org/small").vectors,
        pooled("org/small").section_mean,
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"documents": 4}, "documents is 3, but embedding.expected declares 4"),
        ({"chunks": 11}, "chunks is 10, but embedding.expected declares 11"),
    ],
)
def test_a_count_off_the_expectation_stops_the_build_before_any_model_loads(
    tmp_path: Path, counts: dict[str, int], message: str
) -> None:
    write_chunk_outputs(interim(tmp_path))
    expected = EmbeddingCountsConfig(snapshot_id=SNAPSHOT, documents=3, chunks=10)
    loader = FakeLoader()

    with pytest.raises(EmbeddingsCheckError, match=message):
        build(tmp_path, loader, app_config(expected=expected.model_copy(update=counts)))

    assert loader.settings == []
    assert not outputs(tmp_path).exists()
    assert not (tmp_path / "data" / "cache").exists()


def test_an_expectation_of_another_snapshot_is_refused(tmp_path: Path) -> None:
    write_chunk_outputs(interim(tmp_path))
    expected = EmbeddingCountsConfig(snapshot_id="20261002T224412Z", documents=3, chunks=10)

    with pytest.raises(ValueError, match="declares the counts of snapshot '20261002T224412Z'"):
        build(tmp_path, FakeLoader(), app_config(expected=expected))


def test_a_model_that_does_not_read_128_tokens_is_refused_before_it_embeds(
    tmp_path: Path,
) -> None:
    write_chunk_outputs(interim(tmp_path))
    loader = FakeLoader(max_seq_length=256)

    with pytest.raises(EmbeddingsCheckError, match="org/small reads up to 256 tokens, not 128"):
        build(tmp_path, loader)

    assert loader.encoded("org/small") == []
    assert not outputs(tmp_path).exists()


def test_a_chunk_longer_than_the_model_reads_is_refused_before_it_embeds(
    tmp_path: Path,
) -> None:
    # The model would truncate it silently.
    write_chunk_outputs(
        interim(tmp_path), (*RECORDS[:-1], RECORDS[-1].model_copy(update={"n_tokens": 129}))
    )
    loader = FakeLoader()

    with pytest.raises(EmbeddingsCheckError, match="a chunk of 129 tokens, more than org/small"):
        build(tmp_path, loader)

    assert loader.encoded("org/small") == []


def test_vectors_of_another_shape_from_the_encoder_are_refused(tmp_path: Path) -> None:
    write_chunk_outputs(interim(tmp_path))

    with pytest.raises(ValueError, match="of 5 dimensions, not 4"):
        build(tmp_path, FakeLoader(shape_error=True))

    assert not outputs(tmp_path).exists()


MARKER = "MARKER 7f3a"
"""A string that a message must never quote, standing for the content of a damaged file."""


@pytest.mark.parametrize(
    "damage",
    [
        "changed-vectors",
        "changed-sections",
        "truncated-manifest",
        "not-json-manifest",
        "refused-manifest-value",
        "missing-vectors",
        "unknown-model",
    ],
)
def test_damaged_or_missing_outputs_are_refused_without_quoting_the_files(
    tmp_path: Path, damage: str
) -> None:
    write_chunk_outputs(interim(tmp_path))
    built = build(tmp_path, FakeLoader())
    directory = outputs(tmp_path)
    manifest = directory / EMBEDDINGS_MANIFEST_FILE_NAME
    content = manifest.read_bytes()
    vectors = directory / built.manifest.models[0].directory / DOCUMENT_FILE_NAMES["section_mean"]
    model_id = "org/small"
    if damage == "changed-vectors":
        vectors.write_bytes(vectors.read_bytes() + b"\0")
    elif damage == "changed-sections":
        sections = directory / SECTIONS_FILE_NAME
        sections.write_bytes(sections.read_bytes() + b"\0")
    elif damage == "truncated-manifest":
        manifest.write_bytes(content[: len(content) // 2])
    elif damage == "not-json-manifest":
        manifest.write_bytes(f"{MARKER} ".encode() + content)
    elif damage == "refused-manifest-value":
        manifest.write_bytes(content.replace(f'"{SNAPSHOT}"'.encode(), f'"{MARKER}"'.encode(), 1))
    elif damage == "missing-vectors":
        vectors.unlink()
    else:
        model_id = "org/unknown"

    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        load_embeddings(interim(tmp_path), model_id)

    message = str(caught.value)
    if damage == "missing-vectors":
        assert caught.type is FileNotFoundError
        assert message == f"{vectors} is missing; run {BUILD_COMMAND}"
    elif damage == "unknown-model":
        assert caught.type is ValueError
        assert message == f"{manifest} has no embeddings of org/unknown; run {BUILD_COMMAND}"
    elif damage.startswith("changed"):
        assert caught.type is ValueError
        assert message.endswith(f"differs from {manifest.name}; run {BUILD_COMMAND} again")
    else:
        assert caught.type is ValueError
        assert message == f"{manifest} is damaged; run {BUILD_COMMAND} again"
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert MARKER not in message


# Embeddings of the current chunks only

STALE_FIELDS = ("snapshot_id", "chunker_version", "chunks_sha256", "embedder_version")
"""The fields of the embeddings manifest that must still match the chunks and the code."""

OTHER_SNAPSHOT = "20261003T120000Z"


def rewrite_chunks_manifest(interim_dir: Path, **changes: Any) -> None:
    """Write the chunk manifest again with ``changes``, as a later chunk build would."""
    path = interim_dir / CHUNKS_MANIFEST_FILE_NAME
    manifest = ChunksManifest.model_validate_json(path.read_bytes())
    path.write_text(manifest.model_copy(update=changes).model_dump_json(), "utf-8")


class UnreadStore:
    """A table store that fails the test if the reader reads any table through it."""

    def read_verified(self, *_: object, **__: object) -> NoReturn:
        pytest.fail("the reader read the outputs before it checked them against the chunks")


@pytest.mark.parametrize(
    ("drift", "fields"),
    [
        ("rebuilt-chunk-table", ("chunks_sha256",)),
        ("new-chunker-version", ("chunker_version",)),
        ("new-chunker-version-and-table", ("chunker_version", "chunks_sha256")),
        ("other-snapshot", ("snapshot_id",)),
        ("new-embedder-version", ("embedder_version",)),
    ],
)
def test_embeddings_that_the_current_chunks_or_code_would_not_give_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str, fields: tuple[str, ...]
) -> None:
    write_chunk_outputs(interim(tmp_path))
    saved = build(tmp_path, FakeLoader()).manifest
    if drift.endswith("table"):
        # The chunk build ran again and wrote another table, which matches its new manifest.
        changed = chunk("DOC-c", "objectives", 0, "Reducir las paradas.", FULL)
        write_chunk_outputs(interim(tmp_path), (*RECORDS[:-1], changed))
    if drift.startswith("new-chunker-version"):
        # As in T13a, a new chunker version may write the same table.
        rewrite_chunks_manifest(interim(tmp_path), chunker_version=CHUNKER_VERSION + 1)
    if drift == "other-snapshot":
        rewrite_chunks_manifest(interim(tmp_path), snapshot_id=OTHER_SNAPSHOT)
    if drift == "new-embedder-version":
        # Code whose pooling rules differ from those of the code that wrote the vectors.
        monkeypatch.setattr(embeddings_module, "EMBEDDER_VERSION", EMBEDDER_VERSION + 1)
    current = chunks_manifest(interim(tmp_path)).model_dump()
    current["embedder_version"] = embeddings_module.EMBEDDER_VERSION
    loaded = None

    with pytest.raises(ValueError) as caught:
        loaded = load_embeddings(interim(tmp_path), "org/small", store=UnreadStore())

    assert caught.type is ValueError and loaded is None
    message = str(caught.value)
    assert message.startswith(f"the embeddings in {outputs(tmp_path)} are stale: ")
    assert message.endswith(f"; run {BUILD_COMMAND} again")
    for name in STALE_FIELDS:
        if name in fields:
            source = "the code" if name == "embedder_version" else CHUNKS_MANIFEST_FILE_NAME
            old, new = getattr(saved, name), current[name]
            assert old != new
            assert (
                f"{name} is {old} in {EMBEDDINGS_MANIFEST_FILE_NAME} but {new} in {source}"
                in message
            )
        else:
            assert name not in message
    assert not any(record.text in message for record in RECORDS)


def test_embeddings_without_their_chunk_manifest_are_refused(tmp_path: Path) -> None:
    write_chunk_outputs(interim(tmp_path))
    build(tmp_path, FakeLoader())
    (interim(tmp_path) / CHUNKS_MANIFEST_FILE_NAME).unlink()
    loaded = None

    with pytest.raises(FileNotFoundError) as caught:
        loaded = load_embeddings(interim(tmp_path), "org/small", store=UnreadStore())

    assert loaded is None
    assert str(caught.value) == (
        f"no {CHUNKS_MANIFEST_FILE_NAME} in {interim(tmp_path)}; run {CHUNKS_BUILD_COMMAND} first"
    )


def test_embeddings_stay_valid_when_the_chunk_build_writes_the_same_table_again(
    tmp_path: Path,
) -> None:
    write_chunk_outputs(interim(tmp_path))
    build(tmp_path, FakeLoader())
    # Only the fields of the run change: its time and its seconds.
    rewrite_chunks_manifest(
        interim(tmp_path), created_at=datetime(2026, 10, 17, 8, 0, 0, tzinfo=UTC), seconds=2.5
    )

    loaded = load_embeddings(interim(tmp_path), "org/small")

    np.testing.assert_allclose(loaded.vectors, pooled("org/small").section_mean, rtol=0, atol=1e-6)


# Command line


def write_project(project_root: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["embedding"] = embedding_config().model_dump(mode="json")
    (project_root / "config").mkdir()
    (project_root / "config" / "default.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )


def run_main(project_root: Path, **seams: Any) -> int:
    values: dict[str, Any] = {
        "clock": lambda: RUN_TIME,
        "timer": lambda: 0.0,
        "load_encoder": FakeLoader(),
        **seams,
    }
    return main(["--project-root", str(project_root), "--snapshot-id", SNAPSHOT], **values)


def test_the_command_reports_counts_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    write_chunk_outputs(interim(tmp_path))

    assert run_main(tmp_path) == 0

    captured = capsys.readouterr()
    assert f"Snapshot {SNAPSHOT}: 3 documents, 8 sections, 10 chunks" in captured.out
    assert (
        "org/small at 111111111111: 4 dimensions; 9 distinct texts, 0 from the cache, 9 embedded"
        in (captured.out)
    )
    assert "Document vector read downstream: section_mean" in captured.out
    assert "org/large: embedded 9 of 9 new texts" in captured.err
    words = {word for record in RECORDS for word in record.text.split() if len(word) > 4}
    assert not words & set((captured.out + captured.err).split())


def test_the_command_reports_a_missing_chunk_table_on_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)

    assert run_main(tmp_path) == 1

    err = capsys.readouterr().err
    assert err.startswith(f"error: no {CHUNKS_MANIFEST_FILE_NAME}") and err.count("\n") == 1
    assert "build_chunks" in err


def interrupt(*_: object) -> NoReturn:
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    "seam",
    ["load_encoder", "clock"],
    ids=["while-loading-a-model", "between-the-outputs-and-their-manifest"],
)
def test_an_interrupted_command_exits_130_and_asks_for_a_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seam: str
) -> None:
    write_project(tmp_path)
    write_chunk_outputs(interim(tmp_path))

    code = run_main(tmp_path, **{seam: interrupt})

    assert code == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith(
        "interrupted: the cache keeps the chunk vectors saved at each checkpoint, and outputs "
        "that do not match their manifest are refused by readers; run the command again\n"
    )
    # The clock stamps the manifest, so the vectors were already written there, and no reader
    # accepts them without their manifest.
    assert (outputs(tmp_path) / SECTIONS_FILE_NAME).exists() == (seam == "clock")
    with pytest.raises(FileNotFoundError, match=f"no {EMBEDDINGS_MANIFEST_FILE_NAME}"):
        load_embeddings(interim(tmp_path), "org/small")
