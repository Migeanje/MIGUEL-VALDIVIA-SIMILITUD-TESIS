"""The app's pages under Streamlit's AppTest (T28), on synthetic artifacts.

Each test runs the real entry point, ``app/main.py``, with its arguments in ``sys.argv`` as
``streamlit run main.py -- --project-root ... --snapshot-id ...`` passes them.
"""

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from app_artifacts import (
    PERSON_CODE_PREFIXES,
    PLANTED_PERSON,
    REPOSITORY_ROOT,
    STABLE_GAPS,
    THESES,
    Project,
    WriteRecorder,
    files_under,
    rendered_text,
)
from streamlit.testing.v1 import AppTest

from thematic_redundancy.app import texts
from thematic_redundancy.app.registry import APP_DIR, MAIN_SCRIPT, PAGES
from thematic_redundancy.app.tables import LINK_COLUMN
from thematic_redundancy.gaps import results_store as gap_store
from thematic_redundancy.topic_map import results_store as t19_store

TIMEOUT = 120
"""Seconds for one run: the first one imports the stack, which takes a few seconds."""
MAP_PAGE = "map_view.py"
GAPS_PAGE = "gaps_view.py"


def started(project: Project, monkeypatch: pytest.MonkeyPatch, *, argv=None) -> AppTest:
    monkeypatch.setattr("sys.argv", argv or project.argv())
    app = AppTest.from_file(str(APP_DIR / MAIN_SCRIPT), default_timeout=TIMEOUT)
    app.run()
    return app


def on_gaps_page(app: AppTest) -> AppTest:
    app.switch_page(GAPS_PAGE)
    app.run()
    return app


def chart_points(app: AppTest) -> dict[str, int]:
    """Return the points of each trace of the page's map, by trace name."""
    (chart,) = app.get("plotly_chart")
    spec = json.loads(chart.proto.spec)
    return {trace["name"]: len(trace["x"]) for trace in spec["data"]}


def assert_no_person(app: AppTest) -> None:
    text = rendered_text(app)
    assert PLANTED_PERSON not in text
    for prefix in PERSON_CODE_PREFIXES:
        assert prefix not in text


def messages(elements: Any) -> list[str]:
    return [element.value for element in elements]


# --- Navigation -------------------------------------------------------------------------------


def test_the_registry_lists_the_map_query_and_gaps_pages_in_spanish() -> None:
    assert [(page.script, page.title, page.url_path) for page in PAGES] == [
        (MAP_PAGE, "Mapa temático", "mapa"),
        ("query_view.py", "Consulta de una propuesta", "consulta"),
        (GAPS_PAGE, "Vacíos de investigación", "vacios"),
    ]
    assert all((APP_DIR / page.script).is_file() for page in PAGES)


def test_the_app_opens_on_the_map_page(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    app = started(project, monkeypatch)

    assert not app.exception
    assert messages(app.title) == ["Mapa temático"]
    assert texts.APP_TITLE in rendered_text(app)


# --- Map page ---------------------------------------------------------------------------------


def test_the_map_page_shows_the_map_its_topics_and_the_provisional_notice(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = started(project, monkeypatch)

    assert not app.exception
    assert not app.error
    assert any(text.startswith("Resultados provisionales.") for text in messages(app.warning))
    assert sum(chart_points(app).values()) == len(THESES)
    topic_table, thesis_table = app.dataframe
    assert len(topic_table.value) == 3
    assert sorted(thesis_table.value[LINK_COLUMN]) == sorted(t.url for t in THESES)
    assert [metric.value for metric in app.metric] == ["14", "3", "2"]
    assert_no_person(app)


def test_the_map_page_filters_by_topic(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    app = started(project, monkeypatch)

    app.multiselect(key="map_topics").select(1).run()

    assert not app.exception
    assert list(chart_points(app).values()) == [4]
    assert len(app.dataframe[1].value) == 4


def test_the_map_page_filters_by_program_and_hides_the_noise(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = started(project, monkeypatch)

    app.multiselect(key="map_programs").select("mecanica").run()
    with_noise = chart_points(app)
    app.toggle(key="map_noise").set_value(False).run()

    assert not app.exception
    assert sum(with_noise.values()) == 3
    assert texts.NOISE_LABEL in with_noise
    assert texts.NOISE_LABEL not in chart_points(app)
    assert sum(chart_points(app).values()) == 2


def test_a_topic_filter_shows_its_topics_only_and_disables_the_noise_switch(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = started(project, monkeypatch)

    app.multiselect(key="map_topics").select(2).run()

    assert not app.exception
    assert texts.NOISE_LABEL not in chart_points(app)
    assert app.toggle(key="map_noise").disabled


def test_the_map_page_reports_a_missing_map_in_spanish(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project.t19_processed / t19_store.MAP_FILE_NAME).unlink()

    app = started(project, monkeypatch)

    assert not app.exception
    (error,) = messages(app.error)
    assert error.startswith("Falta el archivo «map.parquet»")
    assert not app.get("plotly_chart")


# --- Gaps page --------------------------------------------------------------------------------


def test_the_gaps_page_lists_the_stable_gaps_and_the_provisional_notice(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = on_gaps_page(started(project, monkeypatch))

    assert not app.exception
    assert not app.error
    assert messages(app.title) == ["Vacíos de investigación"]
    assert any(text.startswith("Resultados provisionales.") for text in messages(app.warning))
    gaps, members = app.dataframe
    assert list(gaps.value["Regla"]) == [texts.RULE_LABELS[row.rule] for row in STABLE_GAPS]
    first_topic = [t for t in THESES if t.topic == 2]
    assert sorted(members.value[LINK_COLUMN]) == sorted(t.url for t in first_topic)
    assert_no_person(app)


def test_the_gaps_page_filters_by_rule_program_and_affinity_group(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = on_gaps_page(started(project, monkeypatch))

    app.multiselect(key="gap_rules").select("c").run()
    by_rule = list(app.dataframe[0].value["N.º"])
    app.toggle(key="gap_high_affinity").set_value(True).run()
    high = list(app.dataframe[0].value["N.º"])
    app.multiselect(key="gap_rules").unselect("c").run()
    app.toggle(key="gap_high_affinity").set_value(False).run()
    app.multiselect(key="gap_programs").select("mecanica").run()
    by_program = list(app.dataframe[0].value["N.º"])

    assert not app.exception
    assert by_rule == [3, 4]
    assert high == [3]
    assert by_program == [1, 3, 4]


def test_the_gaps_page_shows_the_theses_of_the_chosen_gap(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = on_gaps_page(started(project, monkeypatch))

    app.selectbox(key="gap_selected").set_value(STABLE_GAPS[1].gap_id).run()

    assert not app.exception
    members = app.dataframe[1].value
    assert len(members) == 4 + 5 + 1
    assert set(members["Parte"]) == {"Primer tema", "Segundo tema", "Tesis puente"}


def test_the_gaps_page_shows_an_empty_filter_result_without_error(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = on_gaps_page(started(project, monkeypatch))

    app.multiselect(key="gap_rules").select("b").run()
    app.multiselect(key="gap_programs").select("mecanica").run()

    assert not app.exception
    assert not app.error
    assert texts.NO_GAPS in messages(app.info)
    assert len(app.dataframe) == 0


def test_the_gaps_page_reports_stale_gaps_while_the_map_still_works(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = project.gaps_results / gap_store.MANIFEST_FILE_NAME
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content["experiment"][t19_store.MANIFEST_FILE_NAME] = "0" * 64
    manifest.write_text(json.dumps(content), encoding="utf-8")

    app = started(project, monkeypatch)
    map_errors = messages(app.error)
    on_gaps_page(app)

    assert not app.exception
    assert map_errors == []
    (error,) = messages(app.error)
    assert error.startswith("El archivo «manifest.json» se generó")


# --- Arguments and persistence ----------------------------------------------------------------


def test_an_argument_that_is_not_valid_is_reported_in_spanish(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = started(project, monkeypatch, argv=["main.py", "--unknown"])

    assert not app.exception
    (error,) = messages(app.error)
    assert error.startswith("Los argumentos de la aplicación no son válidos")


def test_using_every_page_and_filter_writes_no_file(
    project: Project, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    before = files_under(project.root)
    recorder = WriteRecorder()
    app = started(project, monkeypatch)
    recorder.install(monkeypatch)

    app.toggle(key="map_noise").set_value(False).run()
    app.multiselect(key="map_programs").select("industrial").run()
    app.multiselect(key="map_topics").select(0).run()
    on_gaps_page(app)
    app.multiselect(key="gap_rules").select("c").run()
    app.multiselect(key="gap_programs").select("sistemas").run()
    app.selectbox(key="gap_selected").set_value(STABLE_GAPS[3].gap_id).run()

    assert not app.exception
    assert recorder.paths == []
    assert files_under(project.root) == before
    assert list(working.iterdir()) == []


# --- Streamlit settings -----------------------------------------------------------------------


def test_the_project_settings_turn_usage_statistics_off_and_run_headless() -> None:
    settings = tomllib.loads(
        (REPOSITORY_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert settings["browser"]["gatherUsageStats"] is False
    assert settings["server"]["headless"] is True
    assert settings["server"]["address"] == "localhost"
