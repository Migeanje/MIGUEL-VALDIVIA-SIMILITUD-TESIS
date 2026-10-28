"""The query page under Streamlit's AppTest (T28), on synthetic artifacts, with the stand-in
encoder in place of the real model."""

import logging
from pathlib import Path

import pytest
from app_artifacts import (
    PERSON_CODE_PREFIXES,
    PLANTED_PERSON,
    Project,
    WriteRecorder,
    files_under,
    rendered_text,
)
from streamlit.testing.v1 import AppTest

from thematic_redundancy.app import texts
from thematic_redundancy.app.registry import APP_DIR, MAIN_SCRIPT, PAGES
from thematic_redundancy.app.tables import (
    ANTECEDENT_TABLE_COLUMNS,
    NEARBY_GAP_TABLE_COLUMNS,
    NEIGHBOR_TABLE_COLUMNS,
    SIMILAR_TABLE_COLUMNS,
)

TIMEOUT = 120
QUERY_PAGE = "query_view.py"
MARKER = "zqmarcadorinventado"
"""An invented word typed into the proposal, which must reach no log, file or later session."""


def on_query_page(project: Project, monkeypatch: pytest.MonkeyPatch, *, argv=None) -> AppTest:
    monkeypatch.setattr("sys.argv", argv or project.argv())
    app = AppTest.from_file(str(APP_DIR / MAIN_SCRIPT), default_timeout=TIMEOUT)
    app.run()
    app.switch_page(QUERY_PAGE)
    app.run()
    return app


def submitted(app: AppTest, title: str, abstract: str = "", objectives: str = "") -> AppTest:
    app.text_input(key="query_title").input(title)
    app.text_area(key="query_abstract").input(abstract)
    app.text_area(key="query_objectives").input(objectives)
    app.button(key="query_submit").click()
    app.run()
    return app


def maintenance(app: AppTest) -> AppTest:
    return submitted(
        app,
        "Plan de mantenimiento preventivo para una planta de prueba",
        f"Se propone un plan de mantenimiento preventivo para cada equipo {MARKER}.",
        "Reducir cada falla del equipo de la planta.",
    )


def messages(elements) -> list[str]:
    return [element.value for element in elements]


def frame(app: AppTest, columns: tuple[str, ...]):
    (found,) = [f.value for f in app.dataframe if tuple(f.value.columns) == columns]
    return found


def test_the_query_page_is_registered_after_the_map() -> None:
    assert [(page.script, page.title, page.url_path) for page in PAGES][1] == (
        QUERY_PAGE,
        "Consulta de una propuesta",
        "consulta",
    )


def test_the_query_page_shows_its_form_before_any_query(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    app = on_query_page(project, monkeypatch)

    assert not app.exception
    assert not app.error
    assert messages(app.title) == ["Consulta de una propuesta"]
    assert any(text.startswith("Resultados provisionales.") for text in messages(app.warning))
    assert app.text_input(key="query_title").label == texts.QUERY_TITLE_FIELD
    assert texts.QUERY_WAITING in messages(app.info)
    assert not app.dataframe
    assert fake_models.texts == []


def test_a_proposal_gets_its_topic_similar_theses_antecedents_and_lines(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    app = maintenance(on_query_page(project, monkeypatch))

    assert not app.exception
    assert not app.error
    assert messages(app.subheader) == [
        texts.QUERY_TOPIC_HEADER,
        texts.QUERY_SIMILAR_HEADER,
        texts.QUERY_ANTECEDENTS_HEADER,
        texts.QUERY_LINES_HEADER,
    ]
    shown = rendered_text(app)
    assert "Tema 0: mantenimiento, planta, preventivo" in shown
    assert any(text.startswith("Propuesta típica para su tema") for text in messages(app.success))
    assert "cuartil Q4" in shown
    assert any(text.startswith("Pendiente de calibración") for text in messages(app.info))
    similar = frame(app, SIMILAR_TABLE_COLUMNS)
    assert len(similar) == 10
    assert list(similar["N.º"]) == list(range(1, 11))
    assert similar["Similitud"].is_monotonic_decreasing
    assert len(frame(app, ANTECEDENT_TABLE_COLUMNS)) == 5
    assert len(frame(app, NEIGHBOR_TABLE_COLUMNS)) == 2
    assert len(frame(app, NEARBY_GAP_TABLE_COLUMNS)) == 3


def test_no_redundancy_verdict_is_shown_anywhere(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    app = maintenance(on_query_page(project, monkeypatch))

    for found in app.dataframe:
        assert not {"Redundante", "Veredicto", "Redundancia"} & set(found.value.columns)
        cells = set(found.value.astype(str).to_numpy().ravel())
        assert not {"Redundante", "No redundante", "Sí", "No"} & cells
    assert "pendiente de calibración" in rendered_text(app).casefold()


def test_an_unrelated_proposal_is_flagged_atypical(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    app = submitted(
        on_query_page(project, monkeypatch),
        "Un título sin palabras de ningún tema",
        "Un resumen que tampoco comparte palabras.",
    )

    assert not app.exception
    assert any(text.startswith("Propuesta atípica para su tema") for text in messages(app.warning))


def test_a_proposal_without_a_title_or_body_is_refused_in_spanish(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    app = submitted(on_query_page(project, monkeypatch), "", "")

    assert not app.exception
    assert messages(app.error) == [texts.MISSING_FIELDS["title"], texts.MISSING_FIELDS["body"]]
    assert not app.dataframe
    assert fake_models.texts == []


def test_a_missing_model_is_reported_in_spanish(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    def missing(model_id: str, settings) -> None:
        raise OSError(f"modules.json of {model_id} is not in the local cache")

    monkeypatch.setattr("thematic_redundancy.app.query.load_sentence_encoder", missing)
    app = maintenance(on_query_page(project, monkeypatch))

    assert not app.exception
    (error,) = messages(app.error)
    assert error.startswith("El modelo «sentence-transformers/paraphrase-multilingual")
    assert "no está en la caché local" in error
    assert not app.dataframe


def test_a_query_is_written_nowhere_logged_nowhere_and_forgotten_by_a_new_session(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    fake_models,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    caplog.set_level(logging.DEBUG)
    before = files_under(project.root)
    app = on_query_page(project, monkeypatch)
    recorder = WriteRecorder()
    recorder.install(monkeypatch)

    maintenance(app)
    out, err = capfd.readouterr()

    assert not app.exception
    assert any(MARKER in text for text in fake_models.texts)
    assert recorder.paths == []
    assert files_under(project.root) == before
    assert list(working.iterdir()) == []
    assert MARKER not in caplog.text
    assert MARKER not in out + err
    later = on_query_page(project, monkeypatch)
    assert later.text_area(key="query_abstract").value == ""
    assert MARKER not in rendered_text(later)


def test_the_query_results_show_no_person(
    project: Project, monkeypatch: pytest.MonkeyPatch, fake_models
) -> None:
    app = maintenance(on_query_page(project, monkeypatch))

    shown = rendered_text(app)
    assert PLANTED_PERSON not in shown
    for prefix in PERSON_CODE_PREFIXES:
        assert prefix not in shown


def test_each_ranking_shows_its_similarity_next_to_its_rank() -> None:
    assert SIMILAR_TABLE_COLUMNS[:2] == ("N.º", "Similitud")
    assert ANTECEDENT_TABLE_COLUMNS[:3] == (
        "N.º",
        "Similitud con la propuesta",
        "Parecido con los anteriores",
    )
    assert NEIGHBOR_TABLE_COLUMNS[:2] == ("Tema", "Similitud con la propuesta")
    assert NEARBY_GAP_TABLE_COLUMNS[:2] == ("N.º del vacío", "Similitud con la propuesta")
