"""The consensus labels of the labeled set: the contract between the importer that writes them
(T20c) and the run of T22 that reads them, its reader, and its match against T20a's draw.

**Contract.** One Parquet file, ``<labeling.records_dir>/<snapshot_id>/labels/consensus.parquet``
(:func:`consensus_labels_path`), inside the gitignored ``data/``, beside T20a's
``pairs.parquet``. It holds one :class:`ConsensusLabel` per pair of T20a's labeled set (phase
``labeled``), every one of them exactly once, and no pilot pair (D34 (7)). Its columns, exactly
these, in this order (:data:`COLUMNS`):

- ``pair_id`` (string): T20a's pair id, ``L-`` and 8 hex digits
  (:func:`~thematic_redundancy.labeling.pairs.pair_id`); a pilot id starts with ``P-``.
- ``label_3`` (string): the consensus level, one of :data:`LEVELS` (D34 (3)).
- ``label`` (64-bit integer): the binary label that T22 uses, 1 when ``label_3`` is
  :data:`REDUNDANT` and 0 otherwise (D31 (b), D34 (3)).
- ``source`` (string): :data:`AGREEMENT` when the two annotators gave the same level, or
  :data:`ADJUDICATION` when a third annotator settled it (D34 (9)).

No column is nullable. :class:`ConsensusLabel` refuses a row outside these domains, so a writer
that builds its rows with it cannot break the contract.

**Reading** (:func:`read_consensus_labels`) refuses a missing file, a file whose columns are not
exactly these, and a row that breaks the contract. **Matching** (:func:`labels_for_pairs`)
refuses labels that are not exactly one per labeled pair of the draw: a pilot pair, a pair id
that is not one of the labeled set's, a pair id given twice, or a labeled pair without a label.
The messages give counts, never pair ids.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from thematic_redundancy.shared.storage import ParquetTableStore, TableStore

LABELS_DIR_NAME = "labels"
CONSENSUS_FILE_NAME = "consensus.parquet"
REDUNDANT = "Redundante"
"""The top level, "mismo problema y mismo enfoque": the only one whose binary label is 1."""
RELATED = "Relacionado"
UNRELATED = "No relacionado"
LEVELS = (REDUNDANT, RELATED, UNRELATED)
"""The three levels of D34 (3), from the top down."""
AGREEMENT = "agreement"
ADJUDICATION = "adjudication"
SOURCES = (AGREEMENT, ADJUDICATION)
COLUMNS = ("pair_id", "label_3", "label", "source")
LABELED_PREFIX = "L-"
PILOT_PREFIX = "P-"

Level = Literal["Redundante", "Relacionado", "No relacionado"]
Source = Literal["agreement", "adjudication"]


class LabelsError(ValueError):
    """The consensus labels break the contract, so nothing was computed from them."""


class ConsensusLabel(BaseModel):
    """One row of ``consensus.parquet``: the consensus label of one pair of the labeled set."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    pair_id: Annotated[StrictStr, Field(min_length=1)]
    label_3: Level
    label: Annotated[StrictInt, Field(ge=0, le=1)]
    """1 when :attr:`label_3` is :data:`REDUNDANT`, else 0."""
    source: Source

    @model_validator(mode="after")
    def _label_follows_the_level(self) -> Self:
        if self.label != binary_label(self.label_3):
            raise ValueError(
                f"label must be 1 when label_3 is {REDUNDANT} and 0 otherwise (D31 (b), D34 (3))"
            )
        return self


def binary_label(level: str) -> int:
    """Return the binary label of ``level``: 1 for :data:`REDUNDANT`, 0 for the other levels.

    Raises:
        ValueError: if ``level`` is not one of :data:`LEVELS`.
    """
    if level not in LEVELS:
        raise ValueError(f"the level {level!r} is not one of {list(LEVELS)}")
    return int(level == REDUNDANT)


def consensus_labels_path(records: Path) -> Path:
    """Return the contract's path under ``records``, T20a's ``<records_dir>/<snapshot_id>``."""
    return records / LABELS_DIR_NAME / CONSENSUS_FILE_NAME


def read_consensus_labels(
    path: Path, *, store: TableStore | None = None
) -> tuple[ConsensusLabel, ...]:
    """Return the rows of the consensus labels at ``path``, each checked against the contract.

    Raises:
        FileNotFoundError: if there is no file at ``path``.
        LabelsError: if its columns are not exactly :data:`COLUMNS` with their types, it is
            not Parquet, or a row breaks the contract.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"the consensus labels are missing: there is no file at {path}; T20c writes them "
            "there once the labeling round is imported"
        )
    store = store or ParquetTableStore()
    try:
        return store.read(ConsensusLabel, path)
    except ValueError as error:  # the store's schema and row errors, and pyarrow's
        raise LabelsError(f"{_described(error)} breaks the labels contract{_fields(error)}") from (
            error
        )


def labels_for_pairs(
    labels: Sequence[ConsensusLabel], pair_ids: Sequence[str]
) -> tuple[ConsensusLabel, ...]:
    """Return the label of each of ``pair_ids``, the labeled set's pairs, in their order.

    Raises:
        LabelsError: if a label is of a pilot pair, names a pair that is not one of
            ``pair_ids``, or is given twice for one pair, or a pair has no label.
    """
    expected = set(pair_ids)
    given = Counter(label.pair_id for label in labels)
    pilot = sum(1 for pair in given if pair.startswith(PILOT_PREFIX))
    unknown = sum(1 for pair in given if pair not in expected and not pair.startswith(PILOT_PREFIX))
    repeated = sum(1 for count in given.values() if count > 1)
    missing = sum(1 for pair in expected if pair not in given)
    problems = [
        *([f"{_count(pilot, 'pilot pair')} (P- ids; the pilot is never part of the labeled set, "
           "D34 (7))"] if pilot else []),
        *([f"{_count(unknown, 'pair id')} that {_be(unknown)} not a pair of the labeled set"]
          if unknown else []),
        *([f"{_count(repeated, 'pair id')} given more than once"] if repeated else []),
        *([f"{missing} of the {len(expected)} labeled pairs {'has' if missing == 1 else 'have'} "
           "no label"] if missing else []),
    ]  # fmt: skip
    if problems:
        raise LabelsError(
            "the consensus labels are not one per pair of T20a's labeled set: "
            f"{'; '.join(problems)}; nothing was computed"
        )
    by_pair = {label.pair_id: label for label in labels}
    return tuple(by_pair[pair] for pair in pair_ids)


def _described(error: ValueError) -> str:
    """The store's message up to its verdict, such as ``row 3 of <path>``."""
    text = " ".join(str(error).split())
    for verdict in (" is not a valid ", " does not hold "):
        if verdict in text:
            return text.split(verdict, 1)[0]
    return f"the file ({text})"


def _fields(error: ValueError) -> str:
    """The fields that a row's validation refused, with pydantic's reasons, never the values."""
    cause = error.__cause__
    if not isinstance(cause, ValidationError):
        problems = str(error).split(" records: ", 1)
        return f": {problems[1]}" if len(problems) == 2 else ""
    reasons = []
    for item in cause.errors():
        where = ".".join(str(part) for part in item["loc"])
        reasons.append(f"{where}: {item['msg']}" if where else item["msg"])
    return f": {'; '.join(reasons)}"


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _be(count: int) -> str:
    return "is" if count == 1 else "are"
