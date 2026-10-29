"""The consensus labels' contract with T22's run (T20c, S4) and the final label of a disputed
pair (D34 (9)).

The contract's reader lives in T22's unit (``redundancy/labels.py``), outside this tree; these
tests pin the names, values, columns and Parquet types that both units declare, so that a change
on either side fails here.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from thematic_redundancy.labeling import consensus
from thematic_redundancy.labeling.consensus import (
    ConsensusLabel,
    binary_label,
    consensus_labels_path,
    final_level,
)
from thematic_redundancy.shared.storage import ParquetTableStore


def test_the_contract_declares_the_names_and_values_that_t22_reads() -> None:
    assert consensus.LABELS_DIR_NAME == "labels"
    assert consensus.CONSENSUS_FILE_NAME == "consensus.parquet"
    assert (consensus.REDUNDANT, consensus.RELATED, consensus.UNRELATED) == (
        "Redundante",
        "Relacionado",
        "No relacionado",
    )
    assert consensus.LEVELS == ("Redundante", "Relacionado", "No relacionado")
    assert (consensus.AGREEMENT, consensus.ADJUDICATION) == ("agreement", "adjudication")
    assert consensus.SOURCES == ("agreement", "adjudication")
    assert consensus.COLUMNS == ("pair_id", "label_3", "label", "source")
    assert (consensus.LABELED_PREFIX, consensus.PILOT_PREFIX) == ("L-", "P-")
    assert tuple(ConsensusLabel.model_fields) == consensus.COLUMNS


def test_the_file_lies_in_the_labels_folder_of_the_snapshots_records(tmp_path: Path) -> None:
    records = tmp_path / "data" / "labels" / "labeled_set" / "S1"

    assert consensus_labels_path(records) == records / "labels" / "consensus.parquet"


def test_the_file_has_exactly_the_contracts_columns_and_types_none_nullable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consensus.parquet"
    rows = [
        ConsensusLabel(pair_id="L-0000abcd", label_3="Redundante", label=1, source="agreement"),
        ConsensusLabel(
            pair_id="L-0000abce", label_3="No relacionado", label=0, source="adjudication"
        ),
    ]

    ParquetTableStore().write(rows, path, model_cls=ConsensusLabel)

    schema = pq.read_schema(path)
    assert [(field.name, field.type, field.nullable) for field in schema] == [
        ("pair_id", pa.string(), False),
        ("label_3", pa.string(), False),
        ("label", pa.int64(), False),
        ("source", pa.string(), False),
    ]
    assert ParquetTableStore().read(ConsensusLabel, path) == tuple(rows)


@pytest.mark.parametrize(
    "values",
    [
        {"label_3": "Redundante", "label": 0},
        {"label_3": "Relacionado", "label": 1},
        {"label_3": "Parecido", "label": 0},
        {"label_3": "Relacionado", "label": 0, "source": "vote"},
        {"label_3": "Relacionado", "label": True},
        {"label_3": "Relacionado", "label": 0, "pair_id": ""},
    ],
    ids=["redundant-0", "related-1", "unknown-level", "unknown-source", "boolean", "empty-id"],
)
def test_a_row_outside_the_contract_is_refused(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ConsensusLabel(**{"pair_id": "L-0000abcd", "source": "agreement", **values})  # type: ignore[arg-type]


def test_the_binary_label_is_1_for_redundante_only() -> None:
    assert [binary_label(level) for level in consensus.LEVELS] == [1, 0, 0]
    with pytest.raises(ValueError, match="not one of"):
        binary_label("redundant")


@pytest.mark.parametrize(
    ("labels", "final"),
    [
        (("Redundante", "Relacionado", "Redundante"), "Redundante"),
        (("Relacionado", "No relacionado", "No relacionado"), "No relacionado"),
        (("Redundante", "No relacionado", "No relacionado"), "No relacionado"),
        (("Relacionado", "Redundante", "Relacionado"), "Relacionado"),
        (("Redundante", "Relacionado", "No relacionado"), "Relacionado"),
        (("No relacionado", "Redundante", "Relacionado"), "Relacionado"),
    ],
)
def test_the_final_level_is_the_majority_or_the_median_of_three_different_ones(
    labels: tuple[str, str, str], final: str
) -> None:
    assert final_level(labels) == final
    # It always agrees with the majority of the three binary labels.
    assert binary_label(final) == int(sum(binary_label(level) for level in labels) >= 2)


@pytest.mark.parametrize(
    "labels", [("Redundante", "Relacionado"), ("Redundante",) * 4, ("Redundante", "Sí", "No")]
)
def test_a_final_level_needs_three_labels_of_the_scale(labels: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        final_level(labels)
