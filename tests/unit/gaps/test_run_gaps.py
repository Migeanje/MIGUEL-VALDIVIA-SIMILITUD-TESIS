"""The gap run of T24a through its command line, on the synthetic T19 experiment of
``conftest.py``, written in pytest's temporary directory.

Only the corpus loader (T19's document vectors, the fichas and the snapshot's sensitive strings)
and git are replaced; T19's records, the configuration and the career-affinity matrix are real
files, read and checked as on the real snapshot.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
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
)
from thematic_redundancy.gaps.run_gaps import (
    RUN_COMMAND,
    GapRunError,
    build_maps,
    main,
    read_experiment,
    thesis_metadata,
)
from thematic_redundancy.gaps.sensitivity import SensitivityRow
from thematic_redundancy.gaps.stability import gap_support
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.results_store import cpu_record, library_versions
from thematic_redundancy.topic_map.run_experiment import GitState

TRACKED = {CUTS_FILE_NAME, GAPS_FILE_NAME, SENSITIVITY_FILE_NAME, SUMMARY_FILE_NAME}
STARTED = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def clean(root: Path, path: str) -> GitState:
    return GitState(commit="c" * 40, dirty=())


def clock() -> Any:
    moments = iter((STARTED, datetime(2026, 10, 21, 9, 1, tzinfo=UTC)))
    return lambda: next(moments)


def timer() -> Any:
    ticks = iter((10.0, 12.5))
    return lambda: next(ticks)


def run(project: Any, kit: SimpleNamespace, *extra: str, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "load": kit.loader(project.corpus),
        "git": clean,
        "clock": clock(),
        "timer": timer(),
    }
    arguments.update(overrides)
    return main(["--project-root", str(project.root), "--snapshot-id", "S1", *extra], **arguments)


def written(project: Any) -> set[str]:
    """Return the gap run's files, both directories together."""
    found = {f"results/{name}" for name in files_under(project.results)}
    return found | {f"data/{name}" for name in files_under(project.processed)}


def files_under(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def read_rows(project: Any) -> tuple[GapRow, ...]:
    return ParquetTableStore().read(GapRow, project.results / GAPS_FILE_NAME)


# A real run


def test_a_real_run_writes_the_tracked_records_and_the_gitignored_members(
    project: Any, kit: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(project, kit)

    out = capsys.readouterr().out
    assert code == 0
    assert written(project) == {
        *(f"results/{name}" for name in TRACKED | {MANIFEST_FILE_NAME, ".lock"}),
        f"data/{MEMBERS_FILE_NAME}",
    }
    assert "Stable gaps of the reference run" in out
    assert "17.25" in out


def test_the_manifest_records_provenance_settings_and_every_digest(
    project: Any, kit: SimpleNamespace
) -> None:
    run(project, kit)
    manifest = GapManifest.model_validate_json(
        (project.results / MANIFEST_FILE_NAME).read_text("utf-8")
    )

    assert (manifest.snapshot_id, manifest.git_commit) == ("S1", "c" * 40)
    assert manifest.experiment == {
        name: kit.sha256_of(directory / name)
        for directory, name in (
            (project.t19_results, "manifest.json"),
            (project.t19_results, "selection.json"),
            (project.t19_results, "topics.parquet"),
            (project.t19_processed, "labels.parquet"),
        )
    }
    assert manifest.inputs == {
        **kit.t19_inputs,
        "metadata.jsonl": "m" * 64,
        "default.yaml": kit.sha256_of(project.root / "config" / "default.yaml"),
        "career_affinity.yaml": kit.sha256_of(project.root / "config" / "career_affinity.yaml"),
    }
    assert (manifest.winner, manifest.provisional) == ("C4", True)
    assert manifest.runs == kit.runs
    assert manifest.reference_run == kit.runs[3]
    assert manifest.settings["stability_min_overlap"] == 0.5
    assert manifest.settings["npmi_floor_quantile"] == 0.25
    assert manifest.settings["rare_max_share"] == 0.0
    assert manifest.sensitivity == {
        "literal_small_percent": 5,
        "literal_distance_quantile": 0.75,
        "rare_max_share": 0.01,
        "min_overlap": 0.75,
    }
    assert manifest.counts["documents"] == 148
    assert manifest.counts["stable_a"] == manifest.counts["stable_c"] == 1
    assert manifest.versions == library_versions()
    assert manifest.cpu == cpu_record()
    assert manifest.files == {name: kit.sha256_of(project.results / name) for name in TRACKED}
    assert manifest.data_files == {
        MEMBERS_FILE_NAME: kit.sha256_of(project.processed / MEMBERS_FILE_NAME)
    }
    assert (manifest.started_at, manifest.seconds) == (STARTED, 2.5)
    assert manifest.finished_at == datetime(2026, 10, 21, 9, 1, tzinfo=UTC)


def test_t19s_provisional_flag_is_carried(
    project: Any, kit: SimpleNamespace, tmp_path: Path, scenario: Any
) -> None:
    for path in (*project.t19_results.iterdir(), *project.t19_processed.iterdir()):
        path.unlink()
    project.t19_results.rmdir()
    project.t19_processed.rmdir()
    kit.write_experiment(project.root, scenario, provisional=False)

    assert run(project, kit) == 0
    manifest = json.loads((project.results / MANIFEST_FILE_NAME).read_text("utf-8"))
    assert manifest["provisional"] is False
    assert "provisional" not in (project.results / SUMMARY_FILE_NAME).read_text("utf-8").casefold()


def test_the_cuts_come_from_the_five_runs_with_the_primary_npmi(
    project: Any, kit: SimpleNamespace
) -> None:
    run(project, kit)
    cuts = json.loads((project.results / CUTS_FILE_NAME).read_text("utf-8"))

    assert cuts["size_cut"] == 17.25
    assert cuts["npmi_floor"] == pytest.approx(0.115)
    assert (cuts["npmi_floor_quantile"], cuts["size_quantile"]) == (0.25, 0.25)
    assert (cuts["runs"], cuts["topics"], cuts["npmi_topics"]) == (list(kit.runs), 40, 40)


def test_every_gap_of_every_run_is_a_row_as_the_rules_give_it(
    project: Any, kit: SimpleNamespace
) -> None:
    run(project, kit)
    rows = read_rows(project)

    for index, (run_id, seed, gaps) in enumerate(zip(kit.runs, kit.seeds, kit.gaps, strict=True)):
        mine = [row for row in rows if row.run_id == run_id]
        assert [(row.rule, row.reason) for row in mine] == [(g.rule, g.reason) for g in gaps]
        assert {(row.seed, row.reference) for row in mine} == {(seed, index == 3)}
    counts = {
        rule: [sum(row.rule == rule and row.run_id == r for row in rows) for r in kit.runs]
        for rule in "acd"
    }
    assert counts == {"a": [2, 2, 2, 2, 2], "c": [3, 3, 3, 2, 3], "d": [1, 1, 1, 1, 0]}


def test_the_stable_gaps_are_those_of_the_medoid_run_in_four_of_five_runs(
    project: Any, kit: SimpleNamespace, scenario: Any
) -> None:
    run(project, kit)
    rows = [row for row in read_rows(project) if row.reference]
    supports = gap_support(kit.gaps, reference=3, min_runs=4, min_overlap=0.5)

    assert [(row.support, row.stable) for row in rows] == [(s.support, s.stable) for s in supports]
    small = {row.theses: (row.support, row.stable) for row in rows if row.rule == "a"}
    # C (9 theses) is in seeds 7, 13, 21 and 42; D (7) in 42 and 73 only; E is noise in 42.
    assert small == {9: (4, True), 7: (2, False)}
    assert {(row.source, row.target) for row in rows if row.rule == "c" and row.stable} == {
        ("minas", "mecanica")
    }
    others = [row for row in read_rows(project) if not row.reference]
    assert {(row.support, row.stable) for row in others} == {(None, None)}


def test_the_reference_is_the_representative_run_of_selection_json(
    project: Any, kit: SimpleNamespace, scenario: Any
) -> None:
    for path in (*project.t19_results.iterdir(), *project.t19_processed.iterdir()):
        path.unlink()
    project.t19_results.rmdir()
    project.t19_processed.rmdir()
    kit.write_experiment(project.root, scenario, representative_seed=73)

    assert run(project, kit) == 0
    rows = [row for row in read_rows(project) if row.reference]
    assert {row.seed for row in rows} == {73}
    # On seed 73, E (6 theses) is in 4 runs and D (7) in 2; C is noise there.
    assert {row.theses: row.stable for row in rows if row.rule == "a"} == {6: True, 7: False}


def test_the_sensitivity_table_is_written_and_never_selects(
    project: Any, kit: SimpleNamespace
) -> None:
    run(project, kit)
    rows = ParquetTableStore().read(SensitivityRow, project.results / SENSITIVITY_FILE_NAME)

    variants = {row.variant for row in rows}
    assert variants == {"primary", "literal_a", "literal_b", "rare_max_share", "min_overlap"}
    rare = [row for row in rows if row.variant == "rare_max_share"]
    assert [row.gaps for row in rare] == [4, 4, 4, 3, 3]
    assert len(read_rows(project)) == sum(len(gaps) for gaps in kit.gaps)


def test_tracked_outputs_hold_no_doc_code_and_only_the_summary_holds_terms(
    project: Any, kit: SimpleNamespace, scenario: Any
) -> None:
    run(project, kit)

    for name in TRACKED | {MANIFEST_FILE_NAME}:
        path = project.results / name
        if name.endswith(".parquet"):
            content = str(pq.read_table(path).to_pylist())
        else:
            content = path.read_text("utf-8")
        assert "DOC0" not in content, name
        assert ("zq" in content) == (name == SUMMARY_FILE_NAME), name
    summary = (project.results / SUMMARY_FILE_NAME).read_text("utf-8")
    assert ", ".join(kit.terms("C")) in summary
    assert kit.terms("D")[0] not in summary  # D is not stable
    columns = pq.read_schema(project.results / GAPS_FILE_NAME).names
    assert "doc_code" not in columns and "terms" not in columns


def test_the_members_join_each_tracked_gap_to_its_theses(
    project: Any, kit: SimpleNamespace, scenario: Any
) -> None:
    run(project, kit)
    members = ParquetTableStore().read(MemberRow, project.processed / MEMBERS_FILE_NAME)
    rows = read_rows(project)

    assert {row.gap_id for row in members} == {row.gap_id for row in rows}
    small = next(row for row in rows if row.reference and row.rule == "a" and row.theses == 9)
    codes = {row.doc_code for row in members if row.gap_id == small.gap_id}
    assert codes == {
        code for code, n in zip(scenario.doc_codes, scenario.names, strict=True) if n == "C"
    }


def test_the_summary_lists_the_stable_gaps_and_the_counts_per_rule(
    project: Any, kit: SimpleNamespace
) -> None:
    run(project, kit)
    summary = (project.results / SUMMARY_FILE_NAME).read_text("utf-8")
    rows = read_rows(project)

    stable = [row for row in rows if row.stable]
    assert all(f"`{row.gap_id}`" in summary for row in stable)
    assert not any(f"`{row.gap_id}`" in summary for row in rows if not row.stable)
    assert "provisional" in summary.casefold()
    assert "17.25" in summary and "0.115" in summary
    assert "high-confidence group of D32 (5): 0; affinity 1: 1." in summary
    assert "## Sensitivity" in summary


# Refusals


def test_the_dry_run_computes_everything_and_writes_nothing(
    project: Any, kit: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    before = kit.files_under(project.root)
    calls: list[Any] = []
    dirty = lambda root, path: GitState(commit="c" * 40, dirty=("odd/notes.md",))  # noqa: E731

    code = run(project, kit, "--dry-run", load=kit.loader(project.corpus, calls), git=dirty)

    out = capsys.readouterr().out
    assert code == 0
    assert kit.files_under(project.root) == before
    assert len(calls) == 1
    for text in (
        "Dry run",
        "nothing is written",
        "1 changed path",
        "17.25",
        "0.115",
        "Stable gaps of the reference run",
        "literal_a",
        "Privacy guard",
        f"{RUN_COMMAND} --snapshot-id S1",
    ):
        assert text in out


def test_a_dirty_tree_stops_a_real_run_before_anything_is_read(
    project: Any, kit: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[Any] = []
    seen: list[str] = []

    def dirty(root: Path, path: str) -> GitState:
        seen.append(path)
        return GitState(commit="c" * 40, dirty=("src/x.py", "odd/y.md"))

    code = run(project, kit, load=kit.loader(project.corpus, calls), git=dirty)

    assert code == 1
    assert "2 changed or untracked paths" in capsys.readouterr().err
    assert seen == ["results/gaps/S1"]
    assert not calls
    assert written(project) == set()


def test_a_second_run_is_refused_while_one_holds_the_lock(
    project: Any, kit: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    project.results.mkdir(parents=True)
    calls: list[Any] = []

    with exclusive_run(project.results, OSError("held")):
        code = run(project, kit, load=kit.loader(project.corpus, calls))

    assert code == 1
    assert "another gap run" in capsys.readouterr().err
    assert not calls
    assert written(project) == {"results/.lock"}


@pytest.mark.parametrize(
    ("directory", "name"),
    [
        ("t19_results", "selection.json"),
        ("t19_results", "topics.parquet"),
        ("t19_processed", "labels.parquet"),
    ],
)
def test_a_t19_file_that_differs_from_its_manifest_refuses_the_run(
    project: Any,
    kit: SimpleNamespace,
    directory: str,
    name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = getattr(project, directory) / name
    path.write_bytes(path.read_bytes() + b" ")

    code = run(project, kit)

    err = capsys.readouterr().err
    assert code == 1
    assert name in err and "SHA-256" in err
    assert written(project) == {"results/.lock"}


@pytest.mark.parametrize("change", ["inputs", "declaration", "snapshot"])
def test_inputs_other_than_t19s_refuse_the_run(
    project: Any, kit: SimpleNamespace, change: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project.t19_results / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    if change == "inputs":
        manifest["inputs"]["embeddings_manifest.json"] = "0" * 64
    elif change == "declaration":
        manifest["declaration_sha256"] = "0" * 64
    else:
        manifest["snapshot_id"] = "S2"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code = run(project, kit)

    assert code == 1
    assert "error:" in capsys.readouterr().err
    assert written(project) == {"results/.lock"}


def test_a_term_of_the_snapshot_refuses_the_whole_write(
    project: Any, kit: SimpleNamespace, scenario: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # C's first term, which the summary shows, is taken here for a string of the snapshot.
    corpus = kit.make_corpus(scenario, sensitive=frozenset({kit.terms("C")[0]}))

    code = run(project, kit, load=kit.loader(corpus))

    err = capsys.readouterr().err
    assert code == 1
    assert SUMMARY_FILE_NAME in err and "nothing was written" in err
    assert kit.terms("C")[0] not in err
    assert written(project) == {"results/.lock"}


def test_the_dry_run_applies_the_privacy_guard_too(
    project: Any, kit: SimpleNamespace, scenario: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = kit.make_corpus(scenario, sensitive=frozenset({kit.terms("C")[0]}))

    code = run(project, kit, "--dry-run", load=kit.loader(corpus))

    assert code == 1
    assert SUMMARY_FILE_NAME in capsys.readouterr().err
    assert written(project) == set()


@pytest.mark.parametrize(("error", "code"), [(KeyboardInterrupt(), 130), (OSError("missing"), 1)])
def test_exit_codes_and_the_rerun_hint(
    project: Any, kit: SimpleNamespace, error: BaseException, code: int, capsys: Any
) -> None:
    def load(*_: Any) -> Any:
        raise error

    assert run(project, kit, load=load) == code
    err = capsys.readouterr().err
    assert ("run the command again" in err) == (code == 130)
    assert MANIFEST_FILE_NAME not in {Path(p).name for p in written(project)}


def test_the_help_names_the_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    out = " ".join(capsys.readouterr().out.split())
    assert stopped.value.code == 0
    for text in ("results/gaps/<snapshot>/", "data/processed/<snapshot>/gaps/", "--dry-run"):
        assert text in out


# T19's records against each other


def test_t19s_topics_and_labels_must_agree_with_each_other_and_the_theses(
    project: Any, kit: SimpleNamespace, scenario: Any
) -> None:
    records = read_experiment(kit.config, project.root, "S1")
    run = records.run_ids[0]
    assert records.run_ids == kit.runs and records.reference == 3
    assert kit.other_run not in records.topics
    maps = build_maps(records, scenario.doc_codes, scenario.vectors)
    assert [m.topics for m in maps] == [m.topics for m in kit.maps]

    first, *others = records.topics[run]
    resized = replace(
        records, topics={**records.topics, run: (first.model_copy(update={"size": 1}), *others)}
    )
    with pytest.raises(GapRunError, match="theses in topics.parquet"):
        build_maps(resized, scenario.doc_codes, scenario.vectors)
    with pytest.raises(GapRunError, match="do not cover"):
        build_maps(records, (*scenario.doc_codes, "DOC9999"), scenario.vectors)


# The fichas


def test_the_metadata_follows_the_order_of_the_document_vectors() -> None:
    def ficha(code: str, program: str, year: int | None) -> SimpleNamespace:
        return SimpleNamespace(doc_code=code, program_key=program, issue_year=year)

    fichas = [ficha("B", "minas", 2024), ficha("A", "sistemas", 2021), ficha("X", "minas", 2022)]

    found = thesis_metadata(fichas, ("A", "B"))

    assert (found.doc_codes, found.programs, found.years) == (
        ("A", "B"),
        ("sistemas", "minas"),
        (2021, 2024),
    )
    with pytest.raises(GapRunError, match="1 thesis"):
        thesis_metadata(fichas, ("A", "B", "C"))
    with pytest.raises(GapRunError, match="year"):
        thesis_metadata([ficha("A", "sistemas", None)], ("A",))
