"""The records of the gap run (T24a, D33): tracked aggregates, gitignored members, and the
privacy guard that every tracked output passes before anything is written.

Under ``results/gaps/<snapshot>/``, which git tracks: ``cuts.json``, ``gaps.parquet`` (one row
per gap of every run, its topics as ``(run_id, topic)`` and no doc code), ``sensitivity.parquet``,
``summary.md`` and ``manifest.json``, written last. Under the gitignored
``data/processed/<snapshot>/gaps/``: ``members.parquet``, the theses of each gap, joined to the
tracked rows by ``gap_id``.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from thematic_redundancy.gaps.results_store import (
    CUTS_FILE_NAME,
    GAPS_FILE_NAME,
    MANIFEST_FILE_NAME,
    MEMBERS_FILE_NAME,
    SENSITIVITY_FILE_NAME,
    SUMMARY_FILE_NAME,
    GapManifest,
    GapRow,
    MemberRow,
    PrivacyError,
    check_privacy,
    gap_id,
    gap_rows,
    member_rows,
    processed_dir,
    results_dir,
    tracked_texts,
    write_outputs,
)
from thematic_redundancy.gaps.sensitivity import SensitivityRow
from thematic_redundancy.gaps.stability import gap_support
from thematic_redundancy.shared.storage import ParquetTableStore

REFERENCE = 3
MOMENT = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def test_the_directories_follow_d33(tmp_path: Path) -> None:
    assert results_dir(tmp_path / "results", "S1") == tmp_path / "results" / "gaps" / "S1"
    assert processed_dir(tmp_path / "data", "S1") == tmp_path / "data" / "processed" / "S1" / "gaps"


def test_gap_ids_are_deterministic_unique_and_name_the_run_rule_and_topics(
    kit: SimpleNamespace,
) -> None:
    run = kit.runs[REFERENCE]
    ids = [gap_id(run, gap) for gap in kit.gaps[REFERENCE]]

    assert len(set(ids)) == len(ids)
    assert ids == [gap_id(run, gap) for gap in kit.gaps[REFERENCE]]
    assert all(identifier.startswith(f"{run}__") for identifier in ids)
    small = next(gap for gap in kit.gaps[REFERENCE] if gap.rule == "a")
    assert gap_id(run, small) == f"{run}__a_t{small.topic}"
    pair = next(gap for gap in kit.gaps[REFERENCE] if gap.rule == "b")
    assert gap_id(run, pair) == f"{run}__b_t{pair.first_topic}-t{pair.second_topic}"
    transfer = next(gap for gap in kit.gaps[REFERENCE] if gap.rule == "c")
    assert gap_id(run, transfer) == (
        f"{run}__c_t{transfer.topic}_{transfer.source}-{transfer.target}"
    )
    decline = next(gap for gap in kit.gaps[REFERENCE] if gap.rule == "d")
    assert gap_id(run, decline) == f"{run}__d_t{decline.topic}"


@pytest.fixture(scope="module")
def reference_rows(scenario: Any, kit: SimpleNamespace) -> tuple[GapRow, ...]:
    supports = gap_support(kit.gaps, reference=REFERENCE, min_runs=4, min_overlap=0.5)
    return gap_rows(
        kit.runs[REFERENCE],
        42,
        kit.gaps[REFERENCE],
        scenario.metadata,
        supports=supports,
    )


def test_gap_rows_hold_aggregates_and_no_doc_code(
    reference_rows: tuple[GapRow, ...], scenario: Any, kit: SimpleNamespace
) -> None:
    gaps = kit.gaps[REFERENCE]
    text = "\n".join(row.model_dump_json() for row in reference_rows)

    assert [row.gap_id for row in reference_rows] == [gap_id(kit.runs[3], g) for g in gaps]
    assert not any(code in text for code in scenario.doc_codes)
    assert "zq" not in text
    for row, gap in zip(reference_rows, gaps, strict=True):
        theses = frozenset().union(*gap.topic_members)
        assert (row.run_id, row.seed, row.reference, row.rule) == (kit.runs[3], 42, True, gap.rule)
        assert row.theses == len(theses) == sum(row.program_counts) == sum(row.year_counts)
        assert list(row.programs) == sorted(row.programs)
        assert list(row.years) == sorted(row.years)
        assert row.reason == gap.reason


def test_gap_rows_carry_each_rules_evidence_in_its_own_columns(
    reference_rows: tuple[GapRow, ...], kit: SimpleNamespace
) -> None:
    rows = {row.rule: row for row in reversed(reference_rows)}
    gaps = {gap.rule: gap for gap in reversed(kit.gaps[REFERENCE])}

    small, pair, transfer, decline = (rows[rule] for rule in "abcd")
    assert (small.topics, small.npmi, small.size_cut) == (
        (gaps["a"].topic,),
        gaps["a"].npmi,
        gaps["a"].size_threshold,
    )
    assert small.npmi_floor == gaps["a"].npmi_floor
    assert pair.topics == (gaps["b"].first_topic, gaps["b"].second_topic)
    assert (pair.centroid_cosine, pair.neighbor_rank, pair.bridge_theses) == (
        gaps["b"].centroid_cosine,
        gaps["b"].neighbor_rank,
        gaps["b"].bridge_documents,
    )
    c = gaps["c"]
    assert (transfer.source, transfer.target, transfer.affinity) == (c.source, c.target, c.affinity)
    assert (transfer.source_count, transfer.source_theses, transfer.source_share) == (
        c.source_count,
        c.source_theses,
        c.source_share,
    )
    assert (transfer.target_count, transfer.target_theses, transfer.target_share) == (
        c.target_count,
        c.target_theses,
        c.target_share,
    )
    d = gaps["d"]
    assert (decline.early_count, decline.early_theses, decline.early_share) == (
        d.early_count,
        d.early_theses,
        d.early_share,
    )
    assert (decline.late_count, decline.late_theses, decline.late_share, decline.decline) == (
        d.late_count,
        d.late_theses,
        d.late_share,
        d.decline,
    )
    assert small.source is None and small.centroid_cosine is None and small.decline is None
    assert pair.npmi is None and pair.affinity is None and pair.early_count is None
    assert transfer.size_cut is None and decline.target is None


def test_only_the_reference_runs_rows_carry_support_and_stability(
    reference_rows: tuple[GapRow, ...], scenario: Any, kit: SimpleNamespace
) -> None:
    supports = gap_support(kit.gaps, reference=REFERENCE, min_runs=4, min_overlap=0.5)
    assert [(row.support, row.stable) for row in reference_rows] == [
        (s.support, s.stable) for s in supports
    ]
    other = gap_rows(kit.runs[0], 7, kit.gaps[0], scenario.metadata)
    assert {(row.support, row.stable, row.reference) for row in other} == {(None, None, False)}
    with pytest.raises(ValueError):
        gap_rows(kit.runs[3], 42, kit.gaps[3], scenario.metadata, supports=supports[:-1])


def test_member_rows_name_the_theses_of_each_gap_by_part(kit: SimpleNamespace) -> None:
    run = kit.runs[REFERENCE]
    rows = member_rows(run, kit.gaps[REFERENCE])

    for gap in kit.gaps[REFERENCE]:
        mine = [row for row in rows if row.gap_id == gap_id(run, gap)]
        if gap.rule == "b":
            expected = {
                *(("first_topic", c) for c in gap.first_members),
                *(("second_topic", c) for c in gap.second_members),
                *(("bridge", c) for c in gap.bridges),
            }
        else:
            expected = {("topic", code) for code in gap.members}
        assert {(row.part, row.doc_code) for row in mine} == expected
        assert {row.run_id for row in mine} == {run}
    assert rows == member_rows(run, kit.gaps[REFERENCE])


def test_the_privacy_guard_names_the_file_and_never_the_string() -> None:
    sensitive = frozenset({"pérez inventado", "un título de tesis"})

    check_privacy({"summary.md": "no leak here", "gaps.parquet": "{}"}, sensitive)
    with pytest.raises(PrivacyError) as caught:
        check_privacy({"summary.md": "Terms: PÉREZ   Inventado", "cuts.json": "{}"}, sensitive)

    message = str(caught.value)
    assert "summary.md" in message and "1" in message
    assert "cuts.json" not in message
    assert "inventado" not in message.casefold()


# Writing


def sensitivity_row(setting: str = "declared values") -> SensitivityRow:
    return SensitivityRow(
        variant="primary",
        setting=setting,
        rule="a",
        run_id="run-1",
        seed=7,
        reference=True,
        gaps=1,
        theses=9,
        noise_theses=None,
        stable=1,
        jaccard_to_reference=None,
    )


def manifest_draft() -> GapManifest:
    return GapManifest(
        snapshot_id="S1",
        git_commit="c" * 40,
        experiment={"manifest.json": "a" * 64},
        inputs={"default.yaml": "d" * 64},
        winner="C4",
        representation="org/model",
        provisional=True,
        runs=("run-1",),
        reference_run="run-1",
        settings={"min_decline": 0.025},
        sensitivity={"rare_max_share": 0.01},
        counts={"documents": 148},
        versions={"numpy": "2"},
        cpu={"logical_cpus": 16},
        files={},
        data_files={},
        started_at=MOMENT,
        finished_at=MOMENT,
        seconds=1.5,
    )


class RecordingStore(ParquetTableStore):
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def write(self, models: Any, path: Path, *, model_cls: Any = None) -> None:
        self.log.append(path.name)
        super().write(models, path, model_cls=model_cls)


def write(
    tmp_path: Path,
    reference_rows: tuple[GapRow, ...],
    kit: SimpleNamespace,
    *,
    summary: str = "# Gaps\n",
    setting: str = "declared values",
    log: list[str] | None = None,
) -> GapManifest:
    log = [] if log is None else log

    def write_text(path: Path, text: str) -> None:
        log.append(path.name)
        path.write_text(f"{text}\n", encoding="utf-8")

    return write_outputs(
        results=tmp_path / "results",
        processed=tmp_path / "data",
        gaps=reference_rows,
        members=member_rows(kit.runs[REFERENCE], kit.gaps[REFERENCE]),
        sensitivity=(sensitivity_row(setting),),
        cuts={"size_cut": 17.25},
        summary=summary,
        manifest=manifest_draft(),
        sensitive=frozenset({"pérez inventado"}),
        store=RecordingStore(log),
        write_text=write_text,
    )


def test_outputs_are_written_members_first_and_the_manifest_last(
    tmp_path: Path, reference_rows: tuple[GapRow, ...], kit: SimpleNamespace
) -> None:
    log: list[str] = []

    manifest = write(tmp_path, reference_rows, kit, log=log)

    assert log == [
        MEMBERS_FILE_NAME,
        GAPS_FILE_NAME,
        SENSITIVITY_FILE_NAME,
        CUTS_FILE_NAME,
        SUMMARY_FILE_NAME,
        MANIFEST_FILE_NAME,
    ]
    results, data = tmp_path / "results", tmp_path / "data"
    assert manifest.files == {
        name: hashlib.sha256((results / name).read_bytes()).hexdigest()
        for name in (GAPS_FILE_NAME, SENSITIVITY_FILE_NAME, CUTS_FILE_NAME, SUMMARY_FILE_NAME)
    }
    assert manifest.data_files == {
        MEMBERS_FILE_NAME: hashlib.sha256((data / MEMBERS_FILE_NAME).read_bytes()).hexdigest()
    }
    stored = GapManifest.model_validate_json((results / MANIFEST_FILE_NAME).read_text("utf-8"))
    assert stored == manifest
    assert manifest.model_dump(exclude={"files", "data_files"}) == manifest_draft().model_dump(
        exclude={"files", "data_files"}
    )
    assert json.loads((results / CUTS_FILE_NAME).read_text("utf-8")) == {"size_cut": 17.25}
    store = ParquetTableStore()
    assert store.read(GapRow, results / GAPS_FILE_NAME) == reference_rows
    members = store.read(MemberRow, data / MEMBERS_FILE_NAME)
    assert {row.gap_id for row in members} == {row.gap_id for row in reference_rows}


@pytest.mark.parametrize(
    ("summary", "setting", "named"),
    [
        ("# Gaps\nterms: pérez inventado\n", "declared values", "summary.md"),
        ("# Gaps\n", "a setting naming Pérez Inventado", "sensitivity.parquet"),
    ],
)
def test_a_leak_in_any_tracked_output_refuses_every_write(
    tmp_path: Path,
    reference_rows: tuple[GapRow, ...],
    kit: SimpleNamespace,
    summary: str,
    setting: str,
    named: str,
) -> None:
    log: list[str] = []

    with pytest.raises(PrivacyError, match=named):
        write(tmp_path, reference_rows, kit, summary=summary, setting=setting, log=log)

    assert log == []
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_the_tracked_texts_hold_every_row_and_the_documents(
    reference_rows: tuple[GapRow, ...],
) -> None:
    texts = tracked_texts(
        gaps=reference_rows,
        sensitivity=(sensitivity_row(),),
        cuts={"size_cut": 17.25},
        summary="# Gaps",
    )

    assert list(texts) == [GAPS_FILE_NAME, SENSITIVITY_FILE_NAME, CUTS_FILE_NAME, SUMMARY_FILE_NAME]
    assert all(row.gap_id in texts[GAPS_FILE_NAME] for row in reference_rows)
    assert "declared values" in texts[SENSITIVITY_FILE_NAME]
    assert json.loads(texts[CUTS_FILE_NAME]) == {"size_cut": 17.25}
    assert texts[SUMMARY_FILE_NAME] == "# Gaps"
