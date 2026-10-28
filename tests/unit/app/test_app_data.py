"""The app's data layer (T28): it reads the saved artifacts, checked, and shows no person."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from app_artifacts import (
    BRIDGE,
    DUPLICATE,
    GAPS,
    MAP_RUN,
    OTHER_RUN_TERMS,
    PERSON_CODE_PREFIXES,
    PLANTED_PERSON,
    PROGRAM_NAMES,
    SNAPSHOT,
    STABLE_GAPS,
    THESES,
    TOPIC_TERMS,
    WINNER_MODEL,
    Project,
    files_under,
    make_project,
)

from thematic_redundancy.app import data
from thematic_redundancy.app.data import (
    THESIS_COLUMNS,
    AppSettings,
    ArtifactError,
    SettingsError,
    filter_gaps,
    filter_points,
    load_experiment,
    load_gaps,
    load_theses,
    load_topic_map,
    locate_sources,
    members_of,
    parse_arguments,
)
from thematic_redundancy.corpus.build_fichas import FICHAS_FILE_NAME, FICHAS_MANIFEST_FILE_NAME
from thematic_redundancy.gaps import results_store as gap_store
from thematic_redundancy.gaps.run_gaps import RUN_COMMAND as GAPS_COMMAND
from thematic_redundancy.topic_map import results_store as t19_store
from thematic_redundancy.topic_map.run_experiment import RUN_COMMAND as T19_COMMAND


def sources_of(project: Project):
    return locate_sources(AppSettings(project.root, SNAPSHOT))


def everything(project: Project):
    sources = sources_of(project)
    experiment = load_experiment(sources)
    theses = load_theses(sources, experiment)
    return sources, experiment, theses


def cells(frame: pd.DataFrame) -> str:
    """Return every column name and cell of ``frame`` as one text."""
    return " ".join(map(str, frame.columns)) + " " + frame.to_csv(index=False)


def assert_no_person(text: str) -> None:
    assert PLANTED_PERSON not in text
    for prefix in PERSON_CODE_PREFIXES:
        assert prefix not in text


def refused(problem: str, artifact: str, call, *, command: str | None = None) -> ArtifactError:
    with pytest.raises(ArtifactError) as caught:
        call()
    error = caught.value
    assert (error.problem, error.artifact) == (problem, artifact)
    if command is not None:
        assert error.command == command
    return error


def rewrite_json(path: Path, **changes) -> None:
    content = json.loads(path.read_text(encoding="utf-8"))
    content.update(changes)
    path.write_text(json.dumps(content), encoding="utf-8")


# --- Arguments --------------------------------------------------------------------------------


def test_the_arguments_default_to_the_current_directory_and_the_latest_snapshot() -> None:
    assert parse_arguments([]) == AppSettings(Path(), None)


def test_the_arguments_name_the_project_root_and_the_snapshot(tmp_path: Path) -> None:
    settings = parse_arguments(["--project-root", str(tmp_path), "--snapshot-id", "S1"])

    assert settings == AppSettings(tmp_path, "S1")


@pytest.mark.parametrize(
    "argv",
    [["--unknown"], ["--snapshot-id"], ["--snapshot-id", "../S1"], ["stray"], ["--help"]],
)
def test_an_argument_that_is_not_valid_is_refused_without_exiting(argv: list[str]) -> None:
    with pytest.raises(SettingsError):
        parse_arguments(argv)


# --- Sources ----------------------------------------------------------------------------------


def test_the_sources_are_the_snapshots_folders_under_the_project_root(project: Project) -> None:
    sources = sources_of(project)

    assert sources.snapshot_id == SNAPSHOT
    assert sources.t19_results == project.t19_results.resolve()
    assert sources.t19_processed == project.t19_processed.resolve()
    assert sources.gaps_results == project.gaps_results.resolve()
    assert sources.gaps_processed == project.gaps_processed.resolve()
    assert sources.fichas_dir == project.fichas_dir.resolve()
    assert sources.program_names == PROGRAM_NAMES


def test_a_missing_configuration_is_reported_as_a_missing_artifact(tmp_path: Path) -> None:
    refused("missing", "default.yaml", lambda: locate_sources(AppSettings(tmp_path, SNAPSHOT)))


def test_a_configuration_that_is_not_valid_is_reported(project: Project) -> None:
    (project.root / "config" / "default.yaml").write_text("paths: [1\n", encoding="utf-8")

    refused("invalid", "default.yaml", lambda: sources_of(project))


def test_without_a_snapshot_id_and_no_harvest_the_snapshot_is_reported(project: Project) -> None:
    refused("missing", "raw", lambda: locate_sources(AppSettings(project.root, None)))


# --- T19's experiment ---------------------------------------------------------------------------


def test_the_experiment_names_the_provisional_winner_and_its_map_run(project: Project) -> None:
    experiment = load_experiment(sources_of(project))

    assert experiment.snapshot_id == SNAPSHOT
    assert experiment.provisional is True
    assert experiment.winner.configuration == "C4"
    assert experiment.winner.representation == WINNER_MODEL
    assert (experiment.winner.algorithm, experiment.winner.parameter) == ("hdbscan", 10)
    assert (experiment.map_run, experiment.map_seed) == (MAP_RUN, 42)


def test_a_confirmed_winner_is_not_provisional(tmp_path: Path) -> None:
    project = Project(tmp_path / "project")
    make_project(project.root)
    results = project.t19_results
    rewrite_json(results / t19_store.SELECTION_FILE_NAME, provisional=False)
    manifest_path = results / t19_store.MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provisional"] = False
    manifest["files"][t19_store.SELECTION_FILE_NAME] = _sha256(
        results / t19_store.SELECTION_FILE_NAME
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_experiment(sources_of(project)).provisional is False


def test_a_missing_experiment_manifest_names_t19s_command(project: Project) -> None:
    (project.t19_results / t19_store.MANIFEST_FILE_NAME).unlink()

    refused(
        "missing",
        t19_store.MANIFEST_FILE_NAME,
        lambda: load_experiment(sources_of(project)),
        command=T19_COMMAND,
    )


def test_a_selection_that_differs_from_t19s_manifest_is_refused(project: Project) -> None:
    rewrite_json(project.t19_results / t19_store.SELECTION_FILE_NAME, provisional=False)

    refused("invalid", t19_store.SELECTION_FILE_NAME, lambda: load_experiment(sources_of(project)))


def test_a_damaged_experiment_manifest_is_refused(project: Project) -> None:
    (project.t19_results / t19_store.MANIFEST_FILE_NAME).write_text("{", encoding="utf-8")

    refused("invalid", t19_store.MANIFEST_FILE_NAME, lambda: load_experiment(sources_of(project)))


def test_an_experiment_of_another_snapshot_is_refused(project: Project) -> None:
    rewrite_json(project.t19_results / t19_store.MANIFEST_FILE_NAME, snapshot_id="S2")

    refused(
        "inconsistent",
        t19_store.MANIFEST_FILE_NAME,
        lambda: load_experiment(sources_of(project)),
    )


# --- The theses -------------------------------------------------------------------------------


def test_the_theses_are_the_included_fichas_display_fields_only(project: Project) -> None:
    _, _, theses = everything(project)

    assert tuple(theses.columns) == THESIS_COLUMNS
    assert list(theses["doc_code"]) == [thesis.doc_code for thesis in THESES]
    assert DUPLICATE.doc_code not in set(theses["doc_code"])
    first = theses.iloc[0]
    assert first["title"] == THESES[0].title
    assert (first["program"], first["program_name"]) == ("industrial", "Ingeniería Industrial")
    assert (first["year"], first["url"]) == (2021, THESES[0].url)


def test_no_author_or_advisor_reaches_the_theses_table(project: Project) -> None:
    _, _, theses = everything(project)

    assert_no_person(cells(theses))
    assert not {"author_codes", "advisor_code", "abstract", "keywords"} & set(theses.columns)


def test_fichas_other_than_t19s_are_reported_as_stale(project: Project) -> None:
    path = project.fichas_dir / FICHAS_MANIFEST_FILE_NAME
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    sources = sources_of(project)
    experiment = load_experiment(sources)

    refused("stale", FICHAS_MANIFEST_FILE_NAME, lambda: load_theses(sources, experiment))


def test_a_fichas_table_that_differs_from_its_manifest_is_refused(project: Project) -> None:
    with (project.fichas_dir / FICHAS_FILE_NAME).open("ab") as table:
        table.write(b"x")
    sources = sources_of(project)
    experiment = load_experiment(sources)

    refused("invalid", FICHAS_FILE_NAME, lambda: load_theses(sources, experiment))


def test_missing_fichas_are_reported(project: Project) -> None:
    (project.fichas_dir / FICHAS_FILE_NAME).unlink()
    sources = sources_of(project)
    experiment = load_experiment(sources)

    refused("missing", FICHAS_FILE_NAME, lambda: load_theses(sources, experiment))


# --- The 2-D map ------------------------------------------------------------------------------


def test_the_map_places_every_thesis_with_its_topic_and_display_fields(project: Project) -> None:
    sources, experiment, theses = everything(project)

    topic_map = load_topic_map(sources, experiment, theses)

    points = topic_map.points
    assert topic_map.run_id == MAP_RUN
    assert len(points) == len(THESES)
    by_code = points.set_index("doc_code")
    for thesis in THESES:
        row = by_code.loc[thesis.doc_code]
        assert (row["topic"], row["x"], row["y"]) == (thesis.topic, thesis.x, thesis.y)
        assert (row["title"], row["program"], row["year"]) == (
            thesis.title,
            thesis.program,
            thesis.year,
        )
        assert row["url"] == thesis.url
    assert_no_person(cells(points))


def test_the_maps_topics_are_its_runs_with_their_terms_and_sizes(project: Project) -> None:
    sources, experiment, theses = everything(project)

    topics = load_topic_map(sources, experiment, theses).topics

    assert [(topic.topic, topic.size, topic.terms) for topic in topics] == [
        (0, 5, TOPIC_TERMS[0]),
        (1, 4, TOPIC_TERMS[1]),
        (2, 3, TOPIC_TERMS[2]),
    ]
    assert all(topic.terms != OTHER_RUN_TERMS for topic in topics)


def test_a_map_that_differs_from_t19s_manifest_is_refused(project: Project) -> None:
    with (project.t19_processed / t19_store.MAP_FILE_NAME).open("ab") as table:
        table.write(b"x")
    sources, experiment, theses = everything(project)

    refused(
        "invalid",
        t19_store.MAP_FILE_NAME,
        lambda: load_topic_map(sources, experiment, theses),
        command=T19_COMMAND,
    )


def test_a_missing_map_names_t19s_command(project: Project) -> None:
    (project.t19_processed / t19_store.MAP_FILE_NAME).unlink()
    sources, experiment, theses = everything(project)

    refused(
        "missing",
        t19_store.MAP_FILE_NAME,
        lambda: load_topic_map(sources, experiment, theses),
        command=T19_COMMAND,
    )


def test_a_map_of_another_run_than_the_selection_is_refused(project: Project) -> None:
    rewrite_json(
        project.t19_results / t19_store.SELECTION_FILE_NAME,
        map={"run_id": "another-run", "seed": 42, "method": "umap"},
    )
    _refresh_selection_digest(project)
    sources, experiment, theses = everything(project)

    refused(
        "inconsistent", t19_store.MAP_FILE_NAME, lambda: load_topic_map(sources, experiment, theses)
    )


def test_a_map_thesis_without_a_ficha_is_refused(project: Project) -> None:
    sources, experiment, theses = everything(project)

    refused(
        "inconsistent",
        t19_store.MAP_FILE_NAME,
        lambda: load_topic_map(sources, experiment, theses.iloc[1:]),
    )


# --- The gaps ---------------------------------------------------------------------------------


def test_the_gaps_are_the_reference_runs_stable_gaps_in_order(project: Project) -> None:
    sources, experiment, theses = everything(project)

    gaps = load_gaps(sources, experiment, theses)

    assert gaps.reference_run == MAP_RUN
    assert [gap.row.gap_id for gap in gaps.gaps] == [row.gap_id for row in STABLE_GAPS]
    assert [gap.number for gap in gaps.gaps] == [1, 2, 3, 4, 5]
    assert (gaps.runs, gaps.min_runs) == (5, 4)
    assert gaps.windows == ((2021, 2023), (2024, 2026))
    assert gaps.program_names == PROGRAM_NAMES


def test_each_gap_carries_its_topics_terms_and_sizes(project: Project) -> None:
    sources, experiment, theses = everything(project)

    gaps = load_gaps(sources, experiment, theses).gaps

    pair = gaps[1]
    assert pair.row.topics == (0, 1)
    assert pair.topic_terms == (TOPIC_TERMS[0], TOPIC_TERMS[1])
    assert pair.topic_sizes == (5, 4)


def test_a_gaps_members_are_its_theses_display_fields_and_parts(project: Project) -> None:
    sources, experiment, theses = everything(project)
    gaps = load_gaps(sources, experiment, theses)

    members = members_of(gaps, GAPS[1].gap_id)

    expected = [t.doc_code for t in THESES if t.topic in (0, 1)] + [BRIDGE.doc_code]
    assert sorted(members["doc_code"]) == sorted(expected)
    assert set(members["part"]) == {"first_topic", "second_topic", "bridge"}
    bridge = members.set_index("doc_code").loc[BRIDGE.doc_code]
    assert (bridge["title"], bridge["url"]) == (BRIDGE.title, BRIDGE.url)
    assert_no_person(cells(gaps.members))


def test_the_unstable_and_other_runs_gaps_have_no_members_loaded(project: Project) -> None:
    sources, experiment, theses = everything(project)

    members = load_gaps(sources, experiment, theses).members

    assert set(members["gap_id"]) == {row.gap_id for row in STABLE_GAPS}


def test_gaps_computed_on_another_t19_record_are_reported_as_stale(project: Project) -> None:
    path = project.t19_results / t19_store.MANIFEST_FILE_NAME
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    sources, experiment, theses = everything(project)

    refused(
        "stale",
        gap_store.MANIFEST_FILE_NAME,
        lambda: load_gaps(sources, experiment, theses),
        command=GAPS_COMMAND,
    )


def test_missing_gaps_name_the_gap_runs_command(project: Project) -> None:
    (project.gaps_results / gap_store.MANIFEST_FILE_NAME).unlink()
    sources, experiment, theses = everything(project)

    refused(
        "missing",
        gap_store.MANIFEST_FILE_NAME,
        lambda: load_gaps(sources, experiment, theses),
        command=GAPS_COMMAND,
    )


@pytest.mark.parametrize(
    ("folder", "name"),
    [("gaps_results", gap_store.GAPS_FILE_NAME), ("gaps_processed", gap_store.MEMBERS_FILE_NAME)],
)
def test_gap_records_that_differ_from_their_manifest_are_refused(
    project: Project, folder: str, name: str
) -> None:
    with (getattr(project, folder) / name).open("ab") as table:
        table.write(b"x")
    sources, experiment, theses = everything(project)

    refused("invalid", name, lambda: load_gaps(sources, experiment, theses))


def test_a_gap_member_without_a_ficha_is_refused(project: Project) -> None:
    sources, experiment, theses = everything(project)
    without_bridge = theses[theses["doc_code"] != BRIDGE.doc_code]

    refused(
        "inconsistent",
        gap_store.MEMBERS_FILE_NAME,
        lambda: load_gaps(sources, experiment, without_bridge),
    )


# --- Filters ----------------------------------------------------------------------------------


def test_an_empty_map_filter_keeps_every_thesis(project: Project) -> None:
    sources, experiment, theses = everything(project)
    points = load_topic_map(sources, experiment, theses).points

    assert len(filter_points(points)) == len(THESES)


def test_the_map_filters_by_topic_program_and_noise(project: Project) -> None:
    sources, experiment, theses = everything(project)
    points = load_topic_map(sources, experiment, theses).points

    by_topic = filter_points(points, topics=[1])
    by_program = filter_points(points, programs=["mecanica"])
    both = filter_points(points, topics=[0, 1], programs=["sistemas"])
    without_noise = filter_points(points, include_noise=False)

    assert set(by_topic["topic"]) == {1}
    assert len(by_topic) == 4
    assert set(by_program["program"]) == {"mecanica"}
    assert len(by_program) == 3
    assert sorted(both["doc_code"]) == sorted(
        t.doc_code for t in THESES if t.topic in (0, 1) and t.program == "sistemas"
    )
    assert -1 not in set(without_noise["topic"])
    assert len(without_noise) == len(THESES) - 2


def test_a_topic_filter_keeps_the_noise_only_when_it_is_asked_for(project: Project) -> None:
    sources, experiment, theses = everything(project)
    points = load_topic_map(sources, experiment, theses).points

    assert set(filter_points(points, topics=[2])["topic"]) == {2}
    assert set(filter_points(points, topics=[2, -1])["topic"]) == {2, -1}


def test_the_gaps_filter_by_rule_program_and_affinity_group(project: Project) -> None:
    sources, experiment, theses = everything(project)
    gaps = load_gaps(sources, experiment, theses).gaps

    def ids(selected) -> list[str]:
        return [gap.row.gap_id.split("__")[-1] for gap in selected]

    assert ids(filter_gaps(gaps)) == [row.gap_id.split("__")[-1] for row in STABLE_GAPS]
    assert ids(filter_gaps(gaps, rules=["c"])) == [
        "c_t0_industrial-mecanica",
        "c_t1_sistemas-mecanica",
    ]
    assert ids(filter_gaps(gaps, rules=["a", "d"])) == ["a_t2", "d_t0"]
    assert ids(filter_gaps(gaps, high_affinity_only=True)) == ["c_t0_industrial-mecanica"]
    # A program matches the programs of a gap's theses, and the source or target of rule (c).
    assert ids(filter_gaps(gaps, programs=["mecanica"])) == [
        "a_t2",
        "c_t0_industrial-mecanica",
        "c_t1_sistemas-mecanica",
    ]
    assert ids(filter_gaps(gaps, rules=["b"], programs=["mecanica"])) == []


# --- Read only --------------------------------------------------------------------------------


def test_loading_every_artifact_writes_no_file(project: Project) -> None:
    before = files_under(project.root)

    sources, experiment, theses = everything(project)
    load_topic_map(sources, experiment, theses)
    load_gaps(sources, experiment, theses)

    assert files_under(project.root) == before


def test_the_data_layer_imports_no_streamlit() -> None:
    code = (
        "import sys, thematic_redundancy.app.data, thematic_redundancy.app.texts, "
        "thematic_redundancy.app.tables, thematic_redundancy.app.charts, "
        "thematic_redundancy.app.registry; print('streamlit' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "False"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_selection_digest(project: Project) -> None:
    results = project.t19_results
    manifest_path = results / t19_store.MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][t19_store.SELECTION_FILE_NAME] = _sha256(
        results / t19_store.SELECTION_FILE_NAME
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # The gaps record the T19 manifest they ran on; follow it, as a new gap run would.
    gaps_manifest = project.gaps_results / gap_store.MANIFEST_FILE_NAME
    content = json.loads(gaps_manifest.read_text(encoding="utf-8"))
    content["experiment"][t19_store.MANIFEST_FILE_NAME] = _sha256(manifest_path)
    gaps_manifest.write_text(json.dumps(content), encoding="utf-8")


def test_the_module_names_its_commands_constants() -> None:
    assert data.T19_COMMAND == T19_COMMAND
    assert data.GAPS_COMMAND == GAPS_COMMAND
