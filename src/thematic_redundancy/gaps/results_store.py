"""The records of the gap run (T24a, D33), and the privacy guard they pass before any is written.

Under ``results/gaps/<snapshot_id>/``, which git tracks, aggregates only:

- ``cuts.json``: rule (a)'s size cut and NPMI floor over the winner's seed runs (D32);
- ``gaps.parquet``: one row per gap of every run, its topics as ``(run_id, topic)``, its programs
  and years with their counts, its rule's evidence and, for the reference run's gaps, their
  support and stability; no doc code and no term;
- ``sensitivity.parquet``: the table of :mod:`.sensitivity`;
- ``summary.md``: the stable gaps with the T19 terms of their topics, and the counts per rule;
- ``manifest.json``, written last, as the commit point: the provenance, the values used and the
  SHA-256 of every other file.

Under the gitignored ``data/processed/<snapshot_id>/gaps/``, as T19 keeps its per-document
labels there: ``members.parquet``, the doc codes of each gap's theses, joined to the tracked
rows by ``gap_id``.

Every tracked output, the tables as the JSON of their rows, passes :func:`check_privacy`
(``corpus/profile.py``: the snapshot's titles, abstracts and their long sentences, person names
and ORCID iDs) before anything is written; a hit refuses every write.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from thematic_redundancy.corpus.profile import leaked_strings
from thematic_redundancy.gaps.rules import (
    Gap,
    InterstitialPair,
    SmallCoherentTopic,
    TemporalDecline,
    ThesisMetadata,
    TransferableTopic,
)
from thematic_redundancy.gaps.sensitivity import SensitivityRow
from thematic_redundancy.gaps.stability import GapSupport
from thematic_redundancy.shared.filesystem import write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore

GAPS_DIR_NAME = "gaps"
PROCESSED_DIR_NAME = "processed"
CUTS_FILE_NAME = "cuts.json"
GAPS_FILE_NAME = "gaps.parquet"
SENSITIVITY_FILE_NAME = "sensitivity.parquet"
SUMMARY_FILE_NAME = "summary.md"
MANIFEST_FILE_NAME = "manifest.json"
MEMBERS_FILE_NAME = "members.parquet"


def results_dir(results_root: Path, snapshot_id: str) -> Path:
    """Return ``<results_root>/gaps/<snapshot_id>``."""
    return results_root / GAPS_DIR_NAME / snapshot_id


def processed_dir(data_root: Path, snapshot_id: str) -> Path:
    """Return ``<data_root>/processed/<snapshot_id>/gaps``."""
    return data_root / PROCESSED_DIR_NAME / snapshot_id / GAPS_DIR_NAME


class PrivacyError(ValueError):
    """A tracked output holds a string of the snapshot, so nothing was written."""


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class GapRow(_Row):
    """One row of ``gaps.parquet``: one gap of one run, with no doc code and no term (D33).

    The columns of a rule's evidence hold ``None`` in the rows of the other rules."""

    gap_id: str
    run_id: str
    seed: int
    reference: bool
    rule: str
    topics: tuple[int, ...]
    """The gap's topics in its run: each one is ``(run_id, topic)``."""
    theses: int
    """The theses of the gap's topics, each counted once."""
    programs: tuple[str, ...]
    program_counts: tuple[int, ...]
    """The theses of each program of :attr:`programs`."""
    years: tuple[int, ...]
    year_counts: tuple[int, ...]
    """The theses of each year of :attr:`years`."""
    size_cut: float | None
    npmi: float | None
    npmi_floor: float | None
    centroid_cosine: float | None
    neighbor_rank: int | None
    bridge_theses: int | None
    source: str | None
    target: str | None
    affinity: int | None
    """The U4 score of the source and target programs (rule (c))."""
    source_count: int | None
    source_theses: int | None
    source_share: float | None
    target_count: int | None
    target_theses: int | None
    target_share: float | None
    early_count: int | None
    early_theses: int | None
    early_share: float | None
    late_count: int | None
    late_theses: int | None
    late_share: float | None
    decline: float | None
    support: int | None
    """For the reference run's gaps, the runs in which the gap is present."""
    stable: bool | None
    """For the reference run's gaps, whether the stability filter keeps it."""
    reason: str


class MemberRow(_Row):
    """One row of the gitignored ``members.parquet``: one thesis of one gap."""

    gap_id: str
    run_id: str
    doc_code: str
    part: str
    """``topic`` for a gap of one topic; ``first_topic``, ``second_topic`` or ``bridge`` for a
    rule (b) pair."""


class GapManifest(BaseModel):
    """Content of ``manifest.json``: provenance, values, counts and time, without text."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    snapshot_id: str
    git_commit: str
    experiment: dict[str, str]
    """SHA-256 of each T19 record read, ``manifest.json`` included, by file name."""
    inputs: dict[str, str]
    """SHA-256 of every other input, by file name."""
    winner: str
    representation: str
    provisional: bool
    """T19's flag: the winner stays provisional until T22 closes stage 2."""
    runs: tuple[str, ...]
    reference_run: str
    settings: dict[str, int | float]
    """The declared values used (``gaps`` and the snapshot's years)."""
    sensitivity: dict[str, int | float]
    """The values of the sensitivity table's variants."""
    counts: dict[str, int]
    versions: dict[str, str]
    cpu: dict[str, Any]
    files: dict[str, str]
    """SHA-256 of each file written to the results directory, by name."""
    data_files: dict[str, str]
    """SHA-256 of each file written to the data directory, by name."""
    started_at: datetime
    finished_at: datetime
    seconds: float


def gap_id(run_id: str, gap: Gap) -> str:
    """Return the id of ``gap`` of the run ``run_id``: the run, the rule and what tells the
    run's gaps of that rule apart, such as ``<run>__c_t3_minas-mecanica``."""
    if isinstance(gap, InterstitialPair):
        return f"{run_id}__b_t{gap.first_topic}-t{gap.second_topic}"
    if isinstance(gap, TransferableTopic):
        return f"{run_id}__c_t{gap.topic}_{gap.source}-{gap.target}"
    return f"{run_id}__{gap.rule}_t{gap.topic}"


def gap_rows(
    run_id: str,
    seed: int,
    gaps: Sequence[Gap],
    metadata: ThesisMetadata,
    *,
    supports: Sequence[GapSupport] | None = None,
) -> tuple[GapRow, ...]:
    """Return the rows of ``gaps``, the gaps of one run in order.

    Only the reference run's gaps are assessed by the stability filter: for that run,
    ``supports`` holds the verdict on each gap, and its rows are marked as the reference's.

    Raises:
        ValueError: if ``supports`` does not hold one verdict per gap.
    """
    if supports is not None and len(supports) != len(gaps):
        raise ValueError(f"{len(supports)} supports were given for {len(gaps)} gaps")
    codes = {code: index for index, code in enumerate(metadata.doc_codes)}
    programs = np.asarray(metadata.programs, dtype=object)
    years = np.asarray(metadata.years, dtype=np.int64)
    rows = []
    for index, gap in enumerate(gaps):
        rows_of = [codes[code] for code in sorted(frozenset().union(*gap.topic_members))]
        program_counts = sorted(Counter(programs[rows_of].tolist()).items())
        year_counts = sorted(Counter(int(year) for year in years[rows_of]).items())
        verdict = None if supports is None else supports[index]
        rows.append(
            GapRow(
                gap_id=gap_id(run_id, gap),
                run_id=run_id,
                seed=seed,
                reference=supports is not None,
                rule=gap.rule,
                topics=gap_topics(gap),
                theses=len(rows_of),
                programs=tuple(program for program, _ in program_counts),
                program_counts=tuple(count for _, count in program_counts),
                years=tuple(year for year, _ in year_counts),
                year_counts=tuple(count for _, count in year_counts),
                **_evidence(gap),
                support=None if verdict is None else verdict.support,
                stable=None if verdict is None else verdict.stable,
                reason=gap.reason,
            )
        )
    return tuple(rows)


def member_rows(run_id: str, gaps: Sequence[Gap]) -> tuple[MemberRow, ...]:
    """Return the theses of each of ``gaps``, the gaps of one run, by gap and part, doc codes
    sorted."""
    rows = []
    for gap in gaps:
        identifier = gap_id(run_id, gap)
        if isinstance(gap, InterstitialPair):
            parts = (
                ("first_topic", gap.first_members),
                ("second_topic", gap.second_members),
                ("bridge", gap.bridges),
            )
        else:
            parts = (("topic", gap.members),)
        rows += [
            MemberRow(gap_id=identifier, run_id=run_id, doc_code=code, part=part)
            for part, codes in parts
            for code in sorted(codes)
        ]
    return tuple(rows)


def tracked_texts(
    *,
    gaps: Sequence[GapRow],
    sensitivity: Sequence[SensitivityRow],
    cuts: Mapping[str, Any],
    summary: str,
) -> dict[str, str]:
    """Return the text of each tracked output but the manifest, by file name: a table as the
    JSON of its rows, one per line."""
    return {
        GAPS_FILE_NAME: "\n".join(row.model_dump_json() for row in gaps),
        SENSITIVITY_FILE_NAME: "\n".join(row.model_dump_json() for row in sensitivity),
        CUTS_FILE_NAME: _cuts_json(cuts),
        SUMMARY_FILE_NAME: summary,
    }


def check_privacy(contents: Mapping[str, str], sensitive: frozenset[str]) -> None:
    """Refuse ``contents``, texts by file name, if any holds a string of ``sensitive``.

    The message names each file and how many strings it holds, never the strings.

    Raises:
        PrivacyError: if a text holds a sensitive string.
    """
    found = {name: len(leaked_strings(text, sensitive)) for name, text in contents.items()}
    leaks = {name: count for name, count in found.items() if count}
    if leaks:
        listed = ", ".join(f"{name} ({count})" for name, count in leaks.items())
        raise PrivacyError(
            f"outputs hold strings taken from the snapshot (titles, abstracts, names or ORCID "
            f"iDs): {listed}; nothing was written"
        )


def write_outputs(
    *,
    results: Path,
    processed: Path,
    gaps: Sequence[GapRow],
    members: Sequence[MemberRow],
    sensitivity: Sequence[SensitivityRow],
    cuts: Mapping[str, Any],
    summary: str,
    manifest: GapManifest,
    sensitive: frozenset[str],
    store: TableStore | None = None,
    write_text: Callable[[Path, str], None] = write_text_atomically,
) -> GapManifest:
    """Write the records once every tracked output passes :func:`check_privacy`: the
    gitignored members first, then the tracked tables, ``cuts.json`` and ``summary.md``, then
    ``manifest.json`` last, and return the manifest.

    ``manifest`` is complete but for :attr:`~GapManifest.files` and
    :attr:`~GapManifest.data_files`, which this fills with the SHA-256 of the files written.

    Raises:
        PrivacyError: if a tracked output, the manifest included, holds a string of
            ``sensitive``; nothing is written then.
        OSError: if writing fails; the previous manifest then stays, and no longer matches
            the files that were replaced.
    """
    store = store or ParquetTableStore()
    texts = tracked_texts(gaps=gaps, sensitivity=sensitivity, cuts=cuts, summary=summary)
    check_privacy({**texts, MANIFEST_FILE_NAME: manifest.model_dump_json(indent=2)}, sensitive)
    results.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    store.write(members, processed / MEMBERS_FILE_NAME, model_cls=MemberRow)
    store.write(gaps, results / GAPS_FILE_NAME, model_cls=GapRow)
    store.write(sensitivity, results / SENSITIVITY_FILE_NAME, model_cls=SensitivityRow)
    write_text(results / CUTS_FILE_NAME, texts[CUTS_FILE_NAME])
    write_text(results / SUMMARY_FILE_NAME, summary.rstrip("\n"))
    final = manifest.model_copy(
        update={
            "files": _digests(
                results, (GAPS_FILE_NAME, SENSITIVITY_FILE_NAME, CUTS_FILE_NAME, SUMMARY_FILE_NAME)
            ),
            "data_files": _digests(processed, (MEMBERS_FILE_NAME,)),
        }
    )
    text = final.model_dump_json(indent=2)
    check_privacy({MANIFEST_FILE_NAME: text}, sensitive)
    # The commit point: until it is written, the manifest describes the previous files.
    write_text(results / MANIFEST_FILE_NAME, text)
    return final


def gap_topics(gap: Gap) -> tuple[int, ...]:
    """Return the topic ids of ``gap`` in its run: two for a rule (b) pair, one otherwise."""
    if isinstance(gap, InterstitialPair):
        return (gap.first_topic, gap.second_topic)
    return (gap.topic,)


def _evidence(gap: Gap) -> dict[str, Any]:
    """Return the evidence columns of ``gap``'s rule, every other one ``None``."""
    values: dict[str, Any] = dict.fromkeys(_EVIDENCE_COLUMNS)
    if isinstance(gap, SmallCoherentTopic):
        values.update(size_cut=gap.size_threshold, npmi=gap.npmi, npmi_floor=gap.npmi_floor)
    elif isinstance(gap, InterstitialPair):
        values.update(
            centroid_cosine=gap.centroid_cosine,
            neighbor_rank=gap.neighbor_rank,
            bridge_theses=gap.bridge_documents,
        )
    elif isinstance(gap, TransferableTopic):
        values.update({name: getattr(gap, name) for name in _TRANSFER_COLUMNS})
    elif isinstance(gap, TemporalDecline):
        values.update({name: getattr(gap, name) for name in _DECLINE_COLUMNS})
    return values


_TRANSFER_COLUMNS = (
    "source",
    "target",
    "affinity",
    "source_count",
    "source_theses",
    "source_share",
    "target_count",
    "target_theses",
    "target_share",
)
"""Rule (c)'s columns, named as the fields of :class:`~.rules.TransferableTopic`."""
_DECLINE_COLUMNS = (
    "early_count",
    "early_theses",
    "early_share",
    "late_count",
    "late_theses",
    "late_share",
    "decline",
)
"""Rule (d)'s columns, named as the fields of :class:`~.rules.TemporalDecline`."""
_EVIDENCE_COLUMNS = (
    "size_cut",
    "npmi",
    "npmi_floor",
    "centroid_cosine",
    "neighbor_rank",
    "bridge_theses",
    *_TRANSFER_COLUMNS,
    *_DECLINE_COLUMNS,
)


def _cuts_json(cuts: Mapping[str, Any]) -> str:
    return json.dumps(dict(cuts), indent=2, sort_keys=True, ensure_ascii=False)


def _digests(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
