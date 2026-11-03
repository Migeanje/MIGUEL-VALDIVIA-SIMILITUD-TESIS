"""The consensus labels of the labeled set, as the importer (T20c) writes them for T22.

**Contract**: declared once, in T22's reader (:mod:`thematic_redundancy.redundancy.labels`),
and imported here, so that the importer writes with the declaration T22 reads: one Parquet file,
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

from thematic_redundancy.redundancy.labels import (
    ADJUDICATION,
    AGREEMENT,
    COLUMNS,
    CONSENSUS_FILE_NAME,
    LABELED_PREFIX,
    LABELS_DIR_NAME,
    LEVELS,
    PILOT_PREFIX,
    REDUNDANT,
    RELATED,
    SOURCES,
    UNRELATED,
    ConsensusLabel,
    Level,
    Source,
    binary_label,
    consensus_labels_path,
)

__all__ = [
    "ADJUDICATION",
    "AGREEMENT",
    "COLUMNS",
    "CONSENSUS_FILE_NAME",
    "LABELED_PREFIX",
    "LABELS_DIR_NAME",
    "LEVELS",
    "PILOT_PREFIX",
    "REDUNDANT",
    "RELATED",
    "SOURCES",
    "UNRELATED",
    "ConsensusLabel",
    "Level",
    "Source",
    "binary_label",
    "consensus_labels_path",
    "final_level",
]


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
