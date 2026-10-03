"""The table store port and its Parquet adapter, checked through real files on disk.

The records are synthetic fichas, plus a small synthetic model for the column types that a
ficha does not use. Every test writes into its own temporary directory.
"""

import os
from datetime import date
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, ConfigDict

from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.shared.storage import (
    ParquetTableStore,
    SchemaMismatchError,
    TableStore,
)

STORE = ParquetTableStore()
SHA256 = "0123456789abcdef" * 4
DECOMPOSED_TITLE = "Señal con ñ descompuesta y ñ compuesta"
"""A title that spells ñ both decomposed and composed; the store must keep both as given."""
TWO_AUTHORS = ("AUT-abcdef0123", "AUT-0123456789")
"""Two author codes out of sorted order, as the repository may list them."""
ONE_AUTHOR = "AUT-00000000aa"


def item_uuid(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def ficha(number: int, **overrides: Any) -> Ficha:
    """Return a valid, included synthetic thesis, with ``overrides`` applied."""
    values: dict[str, Any] = {
        "item_uuid": item_uuid(number),
        "doc_code": doc_code_for(item_uuid(number)),
        "handle_url": f"https://repo.example.edu/handle/123456789/{1000 + number}",
        "snapshot_id": "20261002T224412Z",
        "document_type": "tesis",
        "renati_type_raw": "https://purl.org/pe-repo/renati/type#tesis",
        "title": f"Título sintético {number:03d}",
        "program_key": "sistemas",
        "program_name": "Ingeniería de Sistemas",
        "issue_date_raw": "2023-05-10",
        "issued_after_snapshot": False,
        "rights": "open",
        "include": True,
        **overrides,
    }
    return Ficha(**values)


def edge_fichas() -> list[Ficha]:
    """Return fichas that hold every kind of edge value a ficha can hold."""
    return [
        ficha(
            1,
            abstract="Diseño de un sistema de información; año, señal y pingüino.",
            objectives="Objetivo sintético: reducir el tiempo de atención.",
            objectives_status="extracted",
            issue_year=2023,
            keywords=("ñandú", "información", "sistema"),
            ocde_codes=("2.02.04",),
            language="spa",
            embargo_end=date(2024, 6, 30),
            advisor_code="ADV-0123456789",
            author_codes=TWO_AUTHORS,
            pdf_status="downloaded",
            pdf_sha256=SHA256,
            source_format="mixed",
            section_source="title_abstract_objectives",
            quality_notes=("abstract_hard_wrapped", "título con año"),
        ),
        ficha(2),  # Every optional field empty: None values and empty lists.
        ficha(
            3,
            title=DECOMPOSED_TITLE,
            issued_after_snapshot=True,
            include=False,
            exclusion_reason="duplicate",
            duplicate_of=item_uuid(1),
            author_codes=(ONE_AUTHOR,),
            quality_notes=("duplicado exacto",),
        ),
    ]


class Sample(BaseModel):
    """Synthetic model with the column types that a ficha does not use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int
    share: float
    day: date
    flag: bool
    level: Literal["low", "high"]
    rank: Literal[1, 2, 3] | None
    links: tuple[UUID, ...]
    tags: list[str]
    note: str | None = None


class StrictSample(BaseModel):
    """Synthetic strict model: it accepts a UUID or a tuple only as such, never as text or a
    list, so reading it back needs the stored values turned into those types."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    key: UUID
    parent: UUID | None
    names: tuple[str, ...]
    day: date


class Unstorable(BaseModel):
    """Synthetic model with a field that has no column type."""

    name: str
    extra: dict[str, int]


class NotedSample(Sample):
    """Subclass of :class:`Sample` with one more field, which a Sample table has no room for."""

    remark: str = "sintético"


def listing(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


# Round trips


def test_fichas_round_trip_with_their_edge_values(tmp_path: Path) -> None:
    fichas = edge_fichas()
    path = tmp_path / "fichas.parquet"

    STORE.write(fichas, path)

    stored = STORE.read(Ficha, path)
    assert stored == tuple(fichas)
    assert [record.author_codes for record in stored] == [TWO_AUTHORS, (), (ONE_AUTHOR,)]


def test_text_is_stored_exactly_as_given(tmp_path: Path) -> None:
    fichas = edge_fichas()
    path = tmp_path / "fichas.parquet"
    STORE.write(fichas, path)

    titles = pq.read_table(path).column("title").to_pylist()

    assert titles[2] == DECOMPOSED_TITLE
    assert STORE.read(Ficha, path)[2].title == DECOMPOSED_TITLE
    assert STORE.read(Ficha, path)[0].keywords == ("ñandú", "información", "sistema")


def test_uuids_are_stored_as_canonical_strings(tmp_path: Path) -> None:
    path = tmp_path / "fichas.parquet"
    STORE.write(edge_fichas(), path)

    table = pq.read_table(path)

    assert table.schema.field("item_uuid").type == pa.string()
    assert table.schema.field("duplicate_of").type == pa.string()
    assert table.column("item_uuid").to_pylist() == [str(item_uuid(n)) for n in (1, 2, 3)]
    assert table.column("duplicate_of").to_pylist() == [None, None, str(item_uuid(1))]


def test_lists_none_and_booleans_keep_their_shape_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "fichas.parquet"
    STORE.write(edge_fichas(), path)

    table = pq.read_table(path)

    assert pa.types.is_list(table.schema.field("keywords").type)
    assert table.column("keywords").to_pylist() == [["ñandú", "información", "sistema"], [], []]
    assert table.column("author_codes").to_pylist() == [list(TWO_AUTHORS), [], [ONE_AUTHOR]]
    assert table.column("abstract").null_count == 2
    assert table.schema.field("include").type == pa.bool_()
    assert table.column("include").to_pylist() == [True, True, False]
    assert table.schema.field("embargo_end").type == pa.date32()
    assert table.schema.field("issue_year").type == pa.int64()


def test_columns_follow_the_model_fields_and_their_optionality(tmp_path: Path) -> None:
    path = tmp_path / "fichas.parquet"
    STORE.write(edge_fichas(), path)

    schema = pq.read_table(path).schema

    assert schema.names == list(Ficha.model_fields)
    assert not schema.field("title").nullable
    assert schema.field("abstract").nullable


def test_other_column_types_round_trip(tmp_path: Path) -> None:
    samples = [
        Sample(
            count=-3,
            share=0.25,
            day=date(2026, 12, 4),
            flag=True,
            level="high",
            rank=2,
            links=(item_uuid(1), item_uuid(2)),
            tags=["año", ""],
            note="ñ",
        ),
        Sample(
            count=0,
            share=0.0,
            day=date(2021, 1, 6),
            flag=False,
            level="low",
            rank=None,
            links=(),
            tags=[],
        ),
    ]
    path = tmp_path / "samples.parquet"

    STORE.write(samples, path)

    assert STORE.read(Sample, path) == tuple(samples)


def test_a_strict_model_gets_back_its_uuids_and_tuples(tmp_path: Path) -> None:
    samples = [
        StrictSample(
            key=item_uuid(1), parent=item_uuid(2), names=("año", "ñ"), day=date(2026, 1, 1)
        ),
        StrictSample(key=item_uuid(2), parent=None, names=(), day=date(2021, 1, 6)),
    ]
    path = tmp_path / "strict.parquet"

    STORE.write(samples, path)

    assert STORE.read(StrictSample, path) == tuple(samples)


def test_an_empty_table_keeps_its_columns(tmp_path: Path) -> None:
    path = tmp_path / "fichas.parquet"

    STORE.write([], path, model_cls=Ficha)

    assert STORE.read(Ficha, path) == ()
    assert pq.read_table(path).schema.names == list(Ficha.model_fields)


def test_the_parquet_adapter_implements_the_table_store_port() -> None:
    assert isinstance(STORE, TableStore)


# Refused writes


def test_an_empty_table_needs_its_model_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_cls"):
        STORE.write([], tmp_path / "fichas.parquet")

    assert listing(tmp_path) == []


def test_records_of_another_class_are_refused(tmp_path: Path) -> None:
    sample = Sample(
        count=1, share=1.0, day=date(2026, 1, 1), flag=True, level="low", rank=1, links=(), tags=[]
    )

    with pytest.raises(TypeError, match="record 2 is a Sample, not a Ficha"):
        STORE.write([ficha(1), sample], tmp_path / "fichas.parquet")  # type: ignore[list-item]

    assert listing(tmp_path) == []


def test_records_of_a_subclass_are_refused_rather_than_cut_down(tmp_path: Path) -> None:
    fields: dict[str, Any] = {
        "count": 1,
        "share": 1.0,
        "day": date(2026, 1, 1),
        "flag": True,
        "level": "low",
        "rank": 1,
        "links": (),
        "tags": [],
    }

    with pytest.raises(TypeError, match="record 2 is a NotedSample, not a Sample"):
        STORE.write([Sample(**fields), NotedSample(**fields)], tmp_path / "samples.parquet")

    assert listing(tmp_path) == []


def test_a_field_without_a_column_type_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="Unstorable.extra"):
        STORE.write([Unstorable(name="x", extra={"a": 1})], tmp_path / "x.parquet")

    assert listing(tmp_path) == []


# Atomic writes


def test_a_write_leaves_only_the_table(tmp_path: Path) -> None:
    STORE.write(edge_fichas(), tmp_path / "out" / "fichas.parquet")

    assert listing(tmp_path / "out") == ["fichas.parquet"]


def test_a_write_replaces_the_previous_table(tmp_path: Path) -> None:
    path = tmp_path / "fichas.parquet"
    STORE.write(edge_fichas(), path)

    STORE.write([ficha(9)], path)

    assert STORE.read(Ficha, path) == (ficha(9),)
    assert listing(tmp_path) == ["fichas.parquet"]


def fail_on_replace(monkeypatch: pytest.MonkeyPatch, interruption: BaseException) -> None:
    def replace(source: object, target: object) -> None:
        raise interruption

    monkeypatch.setattr(os, "replace", replace)


def fail_while_writing(monkeypatch: pytest.MonkeyPatch, interruption: BaseException) -> None:
    def write_table(table: pa.Table, where: Any, **options: Any) -> None:
        where.write(b"PAR1 partial content")
        raise interruption

    monkeypatch.setattr(pq, "write_table", write_table)


@pytest.mark.parametrize("failure", [fail_on_replace, fail_while_writing])
@pytest.mark.parametrize("interruption", [OSError("simulated disk failure"), KeyboardInterrupt()])
def test_a_failed_write_keeps_the_previous_table_and_leaves_no_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Any,
    interruption: BaseException,
) -> None:
    path = tmp_path / "fichas.parquet"
    STORE.write(edge_fichas(), path)
    before = path.read_bytes()
    failure(monkeypatch, interruption)

    with pytest.raises(type(interruption)):
        STORE.write([ficha(9)], path)

    assert listing(tmp_path) == ["fichas.parquet"]
    assert path.read_bytes() == before


# Refused reads


def write_raw(path: Path, table: pa.Table) -> None:
    """Write ``table`` as it is, bypassing the store."""
    pq.write_table(table, path)


def stored_table(tmp_path: Path) -> pa.Table:
    path = tmp_path / "source.parquet"
    STORE.write(edge_fichas(), path)
    return pq.read_table(path)


def test_a_table_with_missing_and_unexpected_columns_is_refused(tmp_path: Path) -> None:
    table = stored_table(tmp_path).drop_columns(["abstract", "language"])
    table = table.append_column("author_name", pa.array(["x", "y", "z"]))
    path = tmp_path / "fichas.parquet"
    write_raw(path, table)

    with pytest.raises(SchemaMismatchError) as caught:
        STORE.read(Ficha, path)

    message = str(caught.value)
    assert "missing columns: abstract, language" in message
    assert "unexpected columns: author_name" in message
    assert "Ficha" in message


def test_a_column_of_another_type_is_refused(tmp_path: Path) -> None:
    table = stored_table(tmp_path)
    position = table.schema.get_field_index("issue_year")
    table = table.set_column(position, "issue_year", pa.array(["2023", None, None]))
    path = tmp_path / "fichas.parquet"
    write_raw(path, table)

    with pytest.raises(SchemaMismatchError, match="issue_year has type string, not int64"):
        STORE.read(Ficha, path)


def test_a_row_that_breaks_the_model_is_refused_with_its_number(tmp_path: Path) -> None:
    table = stored_table(tmp_path)
    position = table.schema.get_field_index("doc_code")
    codes = table.column("doc_code").to_pylist()
    table = table.set_column(
        position, table.schema.field("doc_code"), pa.array([codes[0], "DOC-00000000", codes[2]])
    )
    path = tmp_path / "fichas.parquet"
    write_raw(path, table)

    with pytest.raises(ValueError, match="row 2 of .* is not a valid Ficha"):
        STORE.read(Ficha, path)


def test_a_file_that_is_not_parquet_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fichas.parquet"
    path.write_bytes(b"not a parquet file")

    with pytest.raises(ValueError):
        STORE.read(Ficha, path)


def test_a_missing_table_is_reported_as_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        STORE.read(Ficha, tmp_path / "fichas.parquet")
