"""The chunk table of a snapshot: built from the saved fichas, checked, and saved (T13).

The fichas are synthetic and saved on disk as the fichas build does, so the build reads them
through ``load_fichas``. The tokenizers are stand-ins that count two special tokens plus one
per word, so every chunk can be worked out by hand: with ``max_tokens`` 16, a chunk holds at
most 14 words.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID

import pytest
import yaml

from thematic_redundancy.corpus.build_fichas import (
    EXCLUSIONS_FILE_NAME,
    FICHAS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    Exclusion,
    FichasManifest,
)
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.representation.build_chunks import (
    BUILD_COMMAND,
    CHUNKS_FILE_NAME,
    CHUNKS_MANIFEST_FILE_NAME,
    ChunksCheckError,
    ChunksManifest,
    build_chunk_table,
    check_chunks,
    chunk_fichas,
    count_sections_used,
    load_chunks,
    main,
    section_texts,
)
from thematic_redundancy.representation.chunker import MaxTokenCounter
from thematic_redundancy.representation.hf_tokenizers import TokenizerIdentity
from thematic_redundancy.shared.config import (
    AppConfig,
    ChunkCountsConfig,
    ChunkingConfig,
    load_config,
)
from thematic_redundancy.shared.storage import ParquetTableStore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
SNAPSHOT = "20261002T150405Z"
RUN_TIME = datetime(2026, 10, 14, 9, 0, 0, tzinfo=UTC)
EXPECTED = ChunkCountsConfig(
    snapshot_id=SNAPSHOT, documents=2, chunks=7, title_chunks=2, body_chunks=5
)


@dataclass(frozen=True)
class FakeTokenizer:
    """Counts two special tokens plus one per word, and ``extra`` more."""

    identity: TokenizerIdentity
    extra: int = 0

    def count(self, text: str) -> int:
        return len(text.split()) + 2 + self.extra


def fake_tokenizer(model_id: str, extra: int = 0) -> FakeTokenizer:
    digest = hashlib.sha256(model_id.encode()).hexdigest()
    return FakeTokenizer(TokenizerIdentity(model_id, "0" * 40, digest, 250_002), extra)


def item(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def ficha(number: int, **overrides: Any) -> Ficha:
    values: dict[str, Any] = {
        "item_uuid": item(number),
        "doc_code": doc_code_for(item(number)),
        "handle_url": f"https://hdl.handle.net/123456789/{1000 + number}",
        "snapshot_id": SNAPSHOT,
        "document_type": "tesis",
        "renati_type_raw": "https://purl.org/pe-repo/renati/type#tesis",
        "title": "Mejora del proceso de una planta, Arequipa 2023",
        # A hard-wrapped abstract: the light cleaner joins its first two lines.
        "abstract": (
            "Se estudia el proceso de\nuna planta textil local. Se mide el tiempo de cada "
            "etapa. Se propone un plan de mejora continua."
        ),
        "objectives": "Diseñar un prototipo.\n\nEvaluar el prototipo.",
        "objectives_status": "extracted",
        "program_key": "sistemas",
        "program_name": "Ingeniería de Sistemas",
        "issue_date_raw": "2023-05-10",
        "issue_year": 2023,
        "issued_after_snapshot": False,
        "rights": "open",
        "pdf_status": "downloaded",
        "pdf_sha256": f"{number:064x}",
        "source_format": "digital",
        "section_source": "title_abstract_objectives",
        "include": True,
    }
    return Ficha(**{**values, **overrides})


FICHAS = (
    ficha(1),
    ficha(
        2,
        title="Modelo de control para una red",
        abstract=(
            "Se analiza la red de datos de la planta con un modelo simple para medir la calidad "
            "del servicio en cada turno."
        ),
        objectives=None,
        objectives_status="no_pdf",
        pdf_status="restricted",
        pdf_sha256=None,
        source_format="none",
        section_source="title_abstract",
    ),
    ficha(3, include=False, exclusion_reason="duplicate", duplicate_of=item(1)),
)
DOC1, DOC2 = doc_code_for(item(1)), doc_code_for(item(2))


def write_fichas_outputs(interim_dir: Path, fichas: Sequence[Ficha] = FICHAS) -> str:
    """Save ``fichas`` as the fichas build does, and return the table's SHA-256."""
    interim_dir.mkdir(parents=True)
    store = ParquetTableStore()
    store.write(list(fichas), interim_dir / FICHAS_FILE_NAME, model_cls=Ficha)
    store.write([], interim_dir / EXCLUSIONS_FILE_NAME, model_cls=Exclusion)

    def digest(name: str) -> str:
        return hashlib.sha256((interim_dir / name).read_bytes()).hexdigest()

    manifest = FichasManifest(
        snapshot_id=SNAPSHOT,
        created_at=RUN_TIME,
        metadata_sha256="a" * 64,
        objectives_table_sha256="b" * 64,
        fichas_file=FICHAS_FILE_NAME,
        fichas_sha256=digest(FICHAS_FILE_NAME),
        exclusions_file=EXCLUSIONS_FILE_NAME,
        exclusions_sha256=digest(EXCLUSIONS_FILE_NAME),
        counts={},
        seconds=0.1,
    )
    (interim_dir / FICHAS_MANIFEST_FILE_NAME).write_text(manifest.model_dump_json(), "utf-8")
    return manifest.fichas_sha256


def chunking_config(expected: ChunkCountsConfig = EXPECTED) -> AppConfig:
    return load_config(DEFAULT_CONFIG_PATH).model_copy(
        update={"chunking": ChunkingConfig(max_tokens=16, expected=expected)}
    )


def interim(project_root: Path) -> Path:
    return project_root / "data" / "interim" / SNAPSHOT


def build(project_root: Path, config: AppConfig | None = None) -> Any:
    return build_chunk_table(
        config or chunking_config(),
        project_root,
        snapshot_id=SNAPSHOT,
        load_counter=fake_tokenizer,
        clock=lambda: RUN_TIME,
        timer=lambda: 0.0,
    )


def test_the_build_chunks_each_included_section_and_saves_the_table(tmp_path: Path) -> None:
    fichas_sha256 = write_fichas_outputs(interim(tmp_path))

    built = build(tmp_path)

    rows = [(r.doc_code, r.section, r.chunk_index, r.cut_kind, r.text) for r in built.records]
    assert rows == [
        (DOC1, "title", 0, "sentence", "Mejora del proceso de una planta"),
        (DOC1, "abstract", 0, "sentence", "Se estudia el proceso de una planta textil local."),
        (
            DOC1,
            "abstract",
            1,
            "sentence",
            "Se mide el tiempo de cada etapa. Se propone un plan de mejora continua.",
        ),
        (DOC1, "objectives", 0, "sentence", "Diseñar un prototipo.\n\nEvaluar el prototipo."),
        (DOC2, "title", 0, "sentence", "Modelo de control para una red"),
        (
            DOC2,
            "abstract",
            0,
            "window",
            "Se analiza la red de datos de la planta con un modelo simple para",
        ),
        (DOC2, "abstract", 1, "sentence", "medir la calidad del servicio en cada turno."),
    ]
    texts = {ficha.doc_code: section_texts(ficha) for ficha in FICHAS}
    for record in built.records:
        assert texts[record.doc_code][record.section][record.char_start : record.char_end] == (
            record.text
        )
        assert record.n_tokens == len(record.text.split()) + 2
        assert record.text_sha256 == hashlib.sha256(record.text.encode()).hexdigest()
        assert record.sections_used == (
            ("title", "abstract", "objectives")
            if record.doc_code == DOC1
            else ("title", "abstract")
        )
    assert load_chunks(interim(tmp_path)) == built.records
    manifest = ChunksManifest.model_validate_json(
        (interim(tmp_path) / CHUNKS_MANIFEST_FILE_NAME).read_bytes()
    )
    assert manifest == built.manifest
    assert manifest.fichas_sha256 == fichas_sha256
    assert (
        manifest.chunks_sha256
        == hashlib.sha256((interim(tmp_path) / CHUNKS_FILE_NAME).read_bytes()).hexdigest()
    )
    assert manifest.counts == {"documents": 2, "chunks": 7, "title_chunks": 2, "body_chunks": 5}
    assert manifest.cut_kinds == {"sentence": 6, "clause": 0, "comma": 0, "window": 1}
    assert manifest.sections_used == {"title+abstract+objectives": 1, "title+abstract": 1}
    assert manifest.sections_recovered == 5
    assert manifest.max_tokens == 16
    assert [(t.model_id, t.largest_chunk_tokens) for t in manifest.tokenizers] == [
        (model, 16) for model in chunking_config().embedding.models
    ]
    assert (
        manifest.tokenizers[0].tokenizer_sha256
        == fake_tokenizer(manifest.tokenizers[0].model_id).identity.tokenizer_sha256
    )


def test_section_texts_are_light_cleaned_and_the_title_loses_its_suffix() -> None:
    assert section_texts(FICHAS[0]) == {
        "title": "Mejora del proceso de una planta",
        "abstract": (
            "Se estudia el proceso de una planta textil local. Se mide el tiempo de cada etapa. "
            "Se propone un plan de mejora continua."
        ),
        "objectives": "Diseñar un prototipo.\n\nEvaluar el prototipo.",
    }
    assert list(section_texts(FICHAS[1])) == ["title", "abstract"]


def test_a_count_off_the_expectation_stops_the_build_before_anything_is_written(
    tmp_path: Path,
) -> None:
    write_fichas_outputs(interim(tmp_path))
    expected = EXPECTED.model_copy(update={"chunks": 8, "body_chunks": 6})

    with pytest.raises(ChunksCheckError, match="body_chunks is 5, but chunking.expected"):
        build(tmp_path, chunking_config(expected))

    assert not (interim(tmp_path) / CHUNKS_FILE_NAME).exists()
    assert not (interim(tmp_path) / CHUNKS_MANIFEST_FILE_NAME).exists()


# A thesis whose title is blank once light-cleaned: the cleaner drops a zero-width space.
UNTITLED = FICHAS[0].model_copy(update={"title": "​"})
# A title of 22 words, which the C2 fallback cuts into two title chunks of 14 and 8 words.
LONG_TITLED = FICHAS[1].model_copy(
    update={
        "title": (
            "Modelo de control de calidad para una red de datos de una planta textil en la "
            "ciudad con un enfoque de mejora"
        )
    }
)


def test_the_sections_used_are_counted_once_per_thesis() -> None:
    records = chunk_fichas((UNTITLED, LONG_TITLED), fake_tokenizer("org/first"), 16)

    # The first thesis has no title row, and the second has two.
    assert [(r.doc_code, r.section) for r in records] == [
        *[(DOC1, "abstract")] * 2,
        (DOC1, "objectives"),
        *[(DOC2, "title")] * 2,
        *[(DOC2, "abstract")] * 2,
    ]
    assert count_sections_used(records) == {"abstract+objectives": 1, "title+abstract": 1}


def test_an_included_thesis_without_a_title_chunk_stops_the_build(tmp_path: Path) -> None:
    # The split title makes up for the missing one, so every declared count still matches.
    write_fichas_outputs(interim(tmp_path), (UNTITLED, LONG_TITLED))

    with pytest.raises(
        ChunksCheckError, match=r"fail 1 check: 1 included thesis has no title chunk$"
    ):
        build(tmp_path)

    assert not (interim(tmp_path) / CHUNKS_FILE_NAME).exists()
    assert not (interim(tmp_path) / CHUNKS_MANIFEST_FILE_NAME).exists()


def test_an_expectation_of_another_snapshot_is_refused(tmp_path: Path) -> None:
    write_fichas_outputs(interim(tmp_path))
    expected = EXPECTED.model_copy(update={"snapshot_id": "20261002T224412Z"})

    with pytest.raises(ValueError, match="declares the counts of snapshot '20261002T224412Z'"):
        build(tmp_path, chunking_config(expected))


def test_the_check_recounts_each_tokenizer_alone() -> None:
    # Chunked under the first tokenizer only, the two 14-word chunks break the second's limit.
    first, second = fake_tokenizer("org/first"), fake_tokenizer("org/second", extra=1)
    records = chunk_fichas(FICHAS, MaxTokenCounter((first,)), 16)

    with pytest.raises(ChunksCheckError, match="2 chunks above 16 tokens under org/second"):
        check_chunks(records, FICHAS, {"org/first": first, "org/second": second}, 16, EXPECTED)


@pytest.mark.parametrize("change", ["shorten-a-chunk", "drop-the-last-chunk"])
def test_the_check_refuses_chunks_that_do_not_recover_their_section(change: str) -> None:
    counter = fake_tokenizer("org/first")
    records = list(chunk_fichas(FICHAS, counter, 16))
    if change == "shorten-a-chunk":
        # The first abstract chunk loses its final period, which no chunk then holds.
        records[1] = records[1].model_copy(
            update={"char_end": records[1].char_end - 1, "text": records[1].text[:-1]}
        )
    else:
        del records[2]  # The second and last chunk of the same abstract.

    with pytest.raises(ChunksCheckError, match="1 section is not recovered"):
        check_chunks(records, FICHAS, {"org/first": counter}, 16, EXPECTED)


def test_a_chunk_table_that_differs_from_its_manifest_is_refused(tmp_path: Path) -> None:
    write_fichas_outputs(interim(tmp_path))
    build(tmp_path)
    table = interim(tmp_path) / CHUNKS_FILE_NAME
    table.write_bytes(table.read_bytes() + b"\0")

    with pytest.raises(ValueError, match=f"{CHUNKS_FILE_NAME} differs from"):
        load_chunks(interim(tmp_path))


MARKER = "MARKER 7f3a"
"""A string that a message must never quote, standing for the content of a damaged file; with
its space, it is no valid snapshot id."""


@pytest.mark.parametrize(
    "damage", ["truncated-manifest", "not-json-manifest", "refused-manifest-value", "no-table"]
)
def test_a_damaged_manifest_or_a_missing_table_is_refused_without_quoting_the_file(
    tmp_path: Path, damage: str
) -> None:
    write_fichas_outputs(interim(tmp_path))
    build(tmp_path)
    directory = interim(tmp_path)
    manifest = directory / CHUNKS_MANIFEST_FILE_NAME
    content = manifest.read_bytes()
    if damage == "truncated-manifest":
        manifest.write_bytes(content[: len(content) // 2])
    elif damage == "not-json-manifest":
        manifest.write_bytes(f"{MARKER} ".encode() + content)
    elif damage == "refused-manifest-value":
        manifest.write_bytes(content.replace(f'"{SNAPSHOT}"'.encode(), f'"{MARKER}"'.encode(), 1))
    else:
        (directory / CHUNKS_FILE_NAME).unlink()

    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        load_chunks(directory)

    if damage == "no-table":
        assert caught.type is FileNotFoundError
        assert str(caught.value) == (
            f"{CHUNKS_FILE_NAME} is missing from {directory}; run {BUILD_COMMAND}"
        )
    else:
        assert caught.type is ValueError
        assert str(caught.value) == (
            f"{CHUNKS_MANIFEST_FILE_NAME} in {directory} is damaged; run {BUILD_COMMAND} again"
        )
        # Not chained, so that a traceback does not quote the file either.
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert MARKER not in str(caught.value)


# Command line


def write_project(project_root: Path, expected: ChunkCountsConfig = EXPECTED) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["chunking"] = {"max_tokens": 16, "expected": expected.model_dump()}
    (project_root / "config").mkdir()
    (project_root / "config" / "default.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )


def run_main(project_root: Path) -> int:
    return main(
        ["--project-root", str(project_root), "--snapshot-id", SNAPSHOT],
        clock=lambda: RUN_TIME,
        timer=lambda: 0.0,
        load_counter=fake_tokenizer,
    )


def test_the_command_reports_counts_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    write_fichas_outputs(interim(tmp_path))

    assert run_main(tmp_path) == 0

    out = capsys.readouterr().out
    assert f"Snapshot {SNAPSHOT}: 2 documents, 7 chunks (2 title, 5 body)" in out
    assert "0 chunks above 16 tokens under any tokenizer" in out
    assert "text recovered for all 5 sections" in out
    words = {
        word for ficha in FICHAS for text in section_texts(ficha).values() for word in text.split()
    }
    assert not {word for word in words if len(word) > 4} & set(out.split())
    assert json.loads((interim(tmp_path) / CHUNKS_MANIFEST_FILE_NAME).read_text("utf-8"))


def test_the_command_reports_a_missing_fichas_table_on_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)

    assert run_main(tmp_path) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: no fichas_manifest.json") and err.count("\n") == 1
    assert "build_fichas" in err


def interrupt(*_: object) -> NoReturn:
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    "seam",
    ["load_counter", "clock"],
    ids=["while-loading-a-tokenizer", "between-the-table-and-its-manifest"],
)
def test_an_interrupted_command_exits_130_and_asks_for_a_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seam: str
) -> None:
    write_project(tmp_path)
    write_fichas_outputs(interim(tmp_path))
    seams: dict[str, Any] = {
        "clock": lambda: RUN_TIME,
        "timer": lambda: 0.0,
        "load_counter": fake_tokenizer,
        seam: interrupt,
    }

    code = main(["--project-root", str(tmp_path), "--snapshot-id", SNAPSHOT], **seams)

    assert code == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "interrupted: a chunk table that does not match its manifest is refused by readers; "
        "run the command again\n"
    )
    # The clock stamps the manifest, so the table was already written there, and no reader
    # accepts it without its manifest.
    assert (interim(tmp_path) / CHUNKS_FILE_NAME).exists() == (seam == "clock")
    with pytest.raises(FileNotFoundError, match=f"no {CHUNKS_MANIFEST_FILE_NAME}"):
        load_chunks(interim(tmp_path))
