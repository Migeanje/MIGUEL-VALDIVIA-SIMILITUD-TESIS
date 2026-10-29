"""The consensus labels of the labeled set, as the importer (T20c) writes them for T22.

**Contract** (declared by the parent with T22's run, which reads it; the same names and rules as
its reader, so that the two units agree): one Parquet file,
``<labeling.records_dir>/<snapshot_id>/labels/consensus.parquet`` (:func:`consensus_labels_path`),
inside the gitignored ``data/``, beside T20a's ``pairs.parquet``. It holds one
:class:`ConsensusLabel` per pair of T20a's labeled set (phase ``labeled``), every one exactly
once, and no pilot pair (D34 (7)). Its columns, exactly these, in this order (:data:`COLUMNS`),
none nullable:

- ``pair_id`` (string): T20a's pair id, ``L-`` and 8 hex digits; a pilot id starts with ``P-``.
- ``label_3`` (string): the consensus level, one of :data:`LEVELS` (D34 (3)).
- ``label`` (64-bit integer): T22's binary label, 1 when ``label_3`` is :data:`REDUNDANT` and 0
  otherwise (D31 (b), D34 (3)).
- ``source`` (string): :data:`AGREEMENT` when the two annotators gave the same level, or
  :data:`ADJUDICATION` when a third annotator settled it (D34 (9)).

**Final label of a disputed pair** (:func:`final_level`): the majority of its three labels (D34
(9)). Three labels of three levels may all differ; the final level is then the median of the
three on the scale's order, which is the majority whenever there is one and always agrees with
the majority of the three binary labels.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

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


def final_level(levels: Sequence[str]) -> str:
    """Return the final level of a disputed pair from its three labels: their majority, or, when
    all three differ, their median on the scale's order (the module docstring).

    Raises:
        ValueError: if there are not exactly three labels, or one is not one of :data:`LEVELS`.
    """
    if len(levels) != 3:
        raise ValueError(f"a disputed pair's final level needs three labels, not {len(levels)}")
    for level in levels:
        binary_label(level)
    return sorted(levels, key=LEVELS.index)[1]
