"""Table storage: a port that saves and loads tables of pydantic records, and a Parquet adapter.

:class:`TableStore` is the port that the rest of the code depends on, so that a database
adapter can replace the files later. :class:`ParquetTableStore` keeps one table per Parquet
file, with one column per model field, named after it and in field order. Columns take their
type from the field annotation:

- ``str`` and ``UUID`` are strings; a UUID is stored as its canonical text, such as
  ``00000000-0000-4000-8000-000000000001``;
- ``bool`` is a boolean, ``int`` a 64-bit integer, ``float`` a double, and ``date`` a date;
  a ``Literal`` takes the type of its values;
- a ``tuple[X, ...]`` or ``list[X]`` of those is a list, which may be empty;
- a field that accepts ``None`` is a nullable column, and ``None`` is stored as null.

Any other annotation, such as a dict or a nested model, is refused before anything is
written. Text is stored as given, accents and ñ included, without any normalization.
"""

import os
import tempfile
import types
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

_ARROW_TYPES: dict[type, pa.DataType] = {
    str: pa.string(),
    UUID: pa.string(),
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    date: pa.date32(),
}
"""Column type of each supported scalar type."""

_UNION_ORIGINS = (Union, types.UnionType)


class SchemaMismatchError(ValueError):
    """A stored table does not have the columns of the model that it is read as."""


@runtime_checkable
class TableStore(Protocol):
    """Port that saves a sequence of pydantic records as one table, and loads it back."""

    def write[M: BaseModel](
        self, models: Sequence[M], path: Path, *, model_cls: type[M] | None = None
    ) -> None:
        """Save ``models`` as the table at ``path``, replacing any previous one as a whole.

        ``model_cls`` gives the columns; it defaults to the class of the first model, so
        it is needed only for an empty table.
        """
        ...

    def read[M: BaseModel](self, model_cls: type[M], path: Path) -> tuple[M, ...]:
        """Load the table at ``path`` as validated ``model_cls`` records, in stored order."""
        ...


@dataclass(frozen=True)
class _Column:
    """How one model field is stored."""

    name: str
    scalar: type
    """Key of :data:`_ARROW_TYPES`: the type of the value, or of each list item."""
    nullable: bool
    sequence: type | None
    """``tuple`` or ``list`` for a field that holds several values, otherwise ``None``."""

    @property
    def arrow_type(self) -> pa.DataType:
        scalar = _ARROW_TYPES[self.scalar]
        if self.sequence is None:
            return scalar
        return pa.list_(pa.field("item", scalar, nullable=False))

    def to_arrow(self, value: Any) -> Any:
        if value is None:
            return None
        if self.sequence is None:
            return self._scalar_to_arrow(value)
        return [self._scalar_to_arrow(item) for item in value]

    def from_arrow(self, value: Any) -> Any:
        if value is None:
            return None
        if self.sequence is None:
            return self._scalar_from_arrow(value)
        return self.sequence(self._scalar_from_arrow(item) for item in value)

    def _scalar_to_arrow(self, value: Any) -> Any:
        return str(value) if self.scalar is UUID else value

    def _scalar_from_arrow(self, value: Any) -> Any:
        return UUID(value) if self.scalar is UUID else value


class ParquetTableStore:
    """:class:`TableStore` that keeps each table in one Parquet file."""

    def write[M: BaseModel](
        self, models: Sequence[M], path: Path, *, model_cls: type[M] | None = None
    ) -> None:
        """Save ``models`` as the Parquet file at ``path``, atomically.

        The table goes to a hidden temporary file in the same directory, is flushed to the
        disk, and then replaces ``path`` in one step, so readers see the previous table or
        the new one, never a partial one. Any failure removes the temporary file and leaves
        ``path`` as it was. The directory is created when it is missing.

        Raises:
            ValueError: if ``models`` is empty and ``model_cls`` is not given.
            TypeError: if a model is not exactly of ``model_cls``, or if a field of
                ``model_cls`` has no column type.
        """
        if model_cls is None:
            if not models:
                raise ValueError(
                    "an empty table has no model to take its columns from; pass model_cls"
                )
            model_cls = type(models[0])
        columns = _columns(model_cls)
        for number, model in enumerate(models, start=1):
            if type(model) is not model_cls:
                raise TypeError(
                    f"record {number} is a {type(model).__name__}, not a {model_cls.__name__}"
                )
        table = pa.Table.from_pydict(
            {
                column.name: [column.to_arrow(getattr(model, column.name)) for model in models]
                for column in columns
            },
            schema=_schema(columns),
        )
        _replace_atomically(Path(path), table)

    def read[M: BaseModel](self, model_cls: type[M], path: Path) -> tuple[M, ...]:
        """Load the Parquet file at ``path`` as ``model_cls`` records, each one validated.

        Raises:
            FileNotFoundError: if there is no file at ``path``.
            SchemaMismatchError: if the file's columns are not those of ``model_cls``, by
                name or by type.
            ValueError: if the file is not Parquet, or if a row is not a valid record. The
                message numbers the rows from 1.
            TypeError: if a field of ``model_cls`` has no column type.
        """
        columns = _columns(model_cls)
        with Path(path).open("rb") as file:
            table = pq.read_table(file)
        _check_columns(table.schema, columns, model_cls.__name__, path)
        records = []
        for number, row in enumerate(table.to_pylist(), start=1):
            try:
                values = {column.name: column.from_arrow(row[column.name]) for column in columns}
                records.append(model_cls.model_validate(values))
            except ValueError as error:  # pydantic's ValidationError is a ValueError.
                raise ValueError(
                    f"row {number} of {path} is not a valid {model_cls.__name__}"
                ) from error
        return tuple(records)


def _columns(model_cls: type[BaseModel]) -> tuple[_Column, ...]:
    """Return the columns of ``model_cls``, one per field, in field order.

    Raises:
        TypeError: if a field has no column type.
    """
    return tuple(
        _column(model_cls.__name__, name, field.annotation)
        for name, field in model_cls.model_fields.items()
    )


def _column(model_name: str, name: str, annotation: Any) -> _Column:
    """Return how the field ``name`` with type ``annotation`` is stored.

    The type may be wrapped in ``X | None``, then in ``tuple[X, ...]`` or ``list[X]``.
    """
    inner = _bare(annotation)
    nullable = False
    if get_origin(inner) in _UNION_ORIGINS:
        members = [member for member in get_args(inner) if member is not type(None)]
        if len(members) == 1 and len(get_args(inner)) == 2:
            nullable, inner = True, _bare(members[0])
    sequence = None
    origin = get_origin(inner)
    arguments = get_args(inner)
    if origin is list and len(arguments) == 1:
        sequence, inner = list, _bare(arguments[0])
    elif origin is tuple and len(arguments) == 2 and arguments[1] is Ellipsis:
        sequence, inner = tuple, _bare(arguments[0])
    scalar = _scalar(inner)
    if scalar is None:
        raise TypeError(
            f"{model_name}.{name} has the type {annotation!r}, which has no column type; use "
            "str, a Literal, UUID, bool, int, float or date, maybe in a tuple or a list, and "
            "maybe with None"
        )
    return _Column(name=name, scalar=scalar, nullable=nullable, sequence=sequence)


def _bare(annotation: Any) -> Any:
    """Return ``annotation`` without its ``Annotated`` metadata and type aliases."""
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        else:
            return annotation


def _scalar(annotation: Any) -> type | None:
    """Return the key of :data:`_ARROW_TYPES` for ``annotation``, or ``None``.

    A ``Literal`` counts as the type of its values, which must all share one type.
    """
    if get_origin(annotation) is Literal:
        kinds = {type(value) for value in get_args(annotation)}
        annotation = kinds.pop() if len(kinds) == 1 else None
    return annotation if annotation in _ARROW_TYPES else None


def _schema(columns: Sequence[_Column]) -> pa.Schema:
    return pa.schema(
        [pa.field(column.name, column.arrow_type, nullable=column.nullable) for column in columns]
    )


def _check_columns(
    schema: pa.Schema, columns: Sequence[_Column], model_name: str, path: Path
) -> None:
    """Raise :class:`SchemaMismatchError` unless ``schema`` has exactly ``columns``."""
    expected = {column.name: column for column in columns}
    found = schema.names
    missing = [name for name in expected if name not in found]
    unexpected = [name for name in found if name not in expected]
    repeated = sorted({name for name in found if found.count(name) > 1})
    mistyped = [
        f"{name} has type {schema.field(name).type}, not {expected[name].arrow_type}"
        for name in found
        if name in expected
        and name not in repeated
        and not _same_type(schema.field(name).type, expected[name].arrow_type)
    ]
    problems = [
        *([f"missing columns: {', '.join(missing)}"] if missing else []),
        *([f"unexpected columns: {', '.join(unexpected)}"] if unexpected else []),
        *([f"repeated columns: {', '.join(repeated)}"] if repeated else []),
        *mistyped,
    ]
    if problems:
        raise SchemaMismatchError(
            f"{path} does not hold {model_name} records: {'; '.join(problems)}"
        )


def _same_type(found: pa.DataType, expected: pa.DataType) -> bool:
    """Compare two column types; list types compare by item type, whatever their item name."""
    if pa.types.is_list(expected):
        return pa.types.is_list(found) and found.value_type == expected.value_type
    return found == expected


def _replace_atomically(path: Path, table: pa.Table) -> None:
    """Write ``table`` to a temporary file beside ``path``, then move it onto ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".part"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            pq.write_table(table, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
