"""The consensus labels' contract (T20c writes the file, T22 reads it): its path, columns and
domains, its reader, and its match against the labeled set's pairs."""

from pathlib import Path
from typing import Any, get_args

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from thematic_redundancy.redundancy.labels import (
    ADJUDICATION,
    AGREEMENT,
    COLUMNS,
    CONSENSUS_FILE_NAME,
    LABELS_DIR_NAME,
    LEVELS,
    REDUNDANT,
    RELATED,
    SOURCES,
    UNRELATED,
    ConsensusLabel,
    LabelsError,
    Level,
    Source,
    binary_label,
    consensus_labels_path,
    labels_for_pairs,
    read_consensus_labels,
)
from thematic_redundancy.shared.storage import ParquetTableStore

PAIRS = ("L-0000000a", "L-0000000b", "L-0000000c", "L-0000000d")


def label(pair_id: str, level: str = RELATED, source: str = AGREEMENT) -> ConsensusLabel:
    return ConsensusLabel(
        pair_id=pair_id, label_3=level, label=int(level == REDUNDANT), source=source
    )


def test_the_contract_names_its_path_columns_and_domains() -> None:
    records = Path("data") / "labels" / "labeled_set" / "S1"

    assert consensus_labels_path(records) == records / "labels" / "consensus.parquet"
    assert (LABELS_DIR_NAME, CONSENSUS_FILE_NAME) == ("labels", "consensus.parquet")
    assert (
        COLUMNS == ("pair_id", "label_3", "label", "source") == tuple(ConsensusLabel.model_fields)
    )
    assert LEVELS == ("Redundante", "Relacionado", "No relacionado") == get_args(Level)
    assert (REDUNDANT, RELATED, UNRELATED) == LEVELS
    assert SOURCES == ("agreement", "adjudication") == get_args(Source)
    assert (AGREEMENT, ADJUDICATION) == SOURCES


@pytest.mark.parametrize(("level", "binary"), [(REDUNDANT, 1), (RELATED, 0), (UNRELATED, 0)])
def test_only_the_top_level_is_redundant(level: str, binary: int) -> None:
    assert binary_label(level) == binary
    assert label("L-0000000a", level).label == binary


@pytest.mark.parametrize("level", ["redundante", "Redundant", "", "No Relacionado"])
def test_a_level_outside_the_scale_has_no_binary_label(level: str) -> None:
    with pytest.raises(ValueError, match="is not one of"):
        binary_label(level)


@pytest.mark.parametrize(
    "fields",
    [
        {"label_3": REDUNDANT, "label": 0},
        {"label_3": RELATED, "label": 1},
        {"label_3": UNRELATED, "label": 1},
        {"label_3": "Redundant", "label": 1},
        {"label": True, "label_3": REDUNDANT},
        {"label": 2},
        {"source": "majority"},
        {"pair_id": ""},
        {"note": "x"},
    ],
)
def test_a_row_that_breaks_the_contract_cannot_be_built(fields: dict[str, Any]) -> None:
    values = {"pair_id": "L-0000000a", "label_3": RELATED, "label": 0, "source": AGREEMENT}

    with pytest.raises(ValidationError):
        ConsensusLabel(**{**values, **fields})


def test_the_labels_round_trip_through_the_table_store(tmp_path: Path) -> None:
    rows = [label(PAIRS[0], REDUNDANT, ADJUDICATION), label(PAIRS[1]), label(PAIRS[2], UNRELATED)]
    path = consensus_labels_path(tmp_path)
    ParquetTableStore().write(rows, path, model_cls=ConsensusLabel)

    assert read_consensus_labels(path) == tuple(rows)
    schema = pq.read_schema(path)
    assert tuple(schema.names) == COLUMNS
    assert [str(field.type) for field in schema] == ["string", "string", "int64", "string"]
    assert not any(field.nullable for field in schema)


def test_a_missing_file_is_refused_with_its_path(tmp_path: Path) -> None:
    path = consensus_labels_path(tmp_path)

    with pytest.raises(FileNotFoundError, match="the consensus labels are missing") as caught:
        read_consensus_labels(path)

    assert str(path) in str(caught.value)


def test_a_file_that_is_not_parquet_is_refused(tmp_path: Path) -> None:
    path = tmp_path / CONSENSUS_FILE_NAME
    path.write_bytes(b"pair_id,label_3\n")

    with pytest.raises(LabelsError, match="breaks the labels contract"):
        read_consensus_labels(path)


def test_a_bad_row_is_named_by_its_number_and_field(tmp_path: Path) -> None:
    broken = ConsensusLabel.model_construct(**{**label(PAIRS[1]).model_dump(), "source": "x"})
    rows = [label(PAIRS[0]), broken]
    path = tmp_path / CONSENSUS_FILE_NAME
    ParquetTableStore().write(rows, path, model_cls=ConsensusLabel)

    with pytest.raises(LabelsError, match=r"row 2 of .* breaks the labels contract: source: Input"):
        read_consensus_labels(path)


def test_the_labels_come_back_in_the_order_of_the_pairs() -> None:
    labels = [label(PAIRS[2]), label(PAIRS[0], REDUNDANT), label(PAIRS[3]), label(PAIRS[1])]

    matched = labels_for_pairs(labels, PAIRS)

    assert [row.pair_id for row in matched] == list(PAIRS)
    assert [row.label for row in matched] == [1, 0, 0, 0]


def test_every_mismatch_with_the_labeled_set_is_counted_in_one_refusal() -> None:
    labels = [
        label(PAIRS[0]),
        label(PAIRS[0]),
        label(PAIRS[1]),
        label("P-0000000e"),
        label("P-0000000f"),
        label("L-ffffffff"),
    ]

    with pytest.raises(LabelsError) as caught:
        labels_for_pairs(labels, PAIRS)

    message = str(caught.value)
    for fragment in (
        "2 pilot pairs",
        "1 pair id that is not a pair of the labeled set",
        "1 pair id given more than once",
        "2 of the 4 labeled pairs have no label",
    ):
        assert fragment in message
    assert not [pair for pair in (*PAIRS, "P-0000000e", "L-ffffffff") if pair in message]


def test_labels_that_match_exactly_pass() -> None:
    labels = [label(pair) for pair in PAIRS]

    assert labels_for_pairs(labels, PAIRS) == tuple(labels)
