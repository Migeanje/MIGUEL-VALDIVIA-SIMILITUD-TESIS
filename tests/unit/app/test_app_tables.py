"""The app's Spanish copy (O01), display tables and map figure (T28), on synthetic artifacts."""

import json
from pathlib import Path

import pytest
from app_artifacts import (
    BRIDGE,
    PLANTED_PERSON,
    PROGRAM_NAMES,
    SNAPSHOT,
    STABLE_GAPS,
    THESES,
    TOPIC_TERMS,
    Project,
)

from thematic_redundancy.app import charts, texts
from thematic_redundancy.app.data import (
    AppSettings,
    ArtifactError,
    SettingsError,
    Winner,
    filter_points,
    load_experiment,
    load_gaps,
    load_theses,
    load_topic_map,
    locate_sources,
    members_of,
)
from thematic_redundancy.app.tables import (
    GAP_TABLE_COLUMNS,
    LINK_COLUMN,
    MEMBER_TABLE_COLUMNS,
    THESIS_TABLE_COLUMNS,
    TOPIC_TABLE_COLUMNS,
    gap_table,
    member_table,
    thesis_table,
    topic_table,
)


def loaded(project: Project):
    sources = locate_sources(AppSettings(project.root, SNAPSHOT))
    experiment = load_experiment(sources)
    theses = load_theses(sources, experiment)
    return (
        experiment,
        load_topic_map(sources, experiment, theses),
        load_gaps(sources, experiment, theses),
    )


# --- Copy -------------------------------------------------------------------------------------


def test_a_topic_is_named_by_its_number_and_first_terms() -> None:
    assert texts.topic_label(3, ("diseño", "estructura", "molino", "máquina")) == (
        "Tema 3: diseño, estructura, molino"
    )
    assert texts.topic_label(3, ("diseño",), count=1) == "Tema 3: diseño"
    assert texts.topic_label(-1, ()) == texts.NOISE_LABEL == "Sin tema asignado"


def test_every_gap_rule_has_a_spanish_label() -> None:
    assert list(texts.RULE_LABELS) == ["a", "b", "c", "d"]
    assert texts.RULE_LABELS["a"] == "(a) Tema pequeño y coherente"
    assert texts.RULE_LABELS["b"] == "(b) Región intersticial entre temas vecinos"
    assert texts.RULE_LABELS["c"] == "(c) Tema transferible a una carrera afín"
    assert texts.RULE_LABELS["d"] == "(d) Tema en declive"


def test_the_provisional_notice_names_the_winner_and_what_may_change() -> None:
    winner = Winner(
        configuration="C4",
        representation="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        algorithm="hdbscan",
        parameter=10,
    )

    notice = texts.provisional_notice(winner)

    assert notice.startswith("Resultados provisionales.")
    assert "C4" in notice
    assert "paraphrase-multilingual-mpnet-base-v2" in notice
    assert "HDBSCAN, tamaño mínimo de grupo 10" in notice
    assert "conjunto de pares etiquetados" in notice
    kmeans = Winner("C3", "x/model", "kmeans", 12)
    assert "K-means, K = 12" in texts.provisional_notice(kmeans)


@pytest.mark.parametrize(
    ("problem", "opening"),
    [
        ("missing", "Falta el archivo «map.parquet»"),
        ("invalid", "El archivo «map.parquet» no se puede usar"),
        ("inconsistent", "El archivo «map.parquet» no concuerda"),
        ("stale", "El archivo «map.parquet» se generó"),
    ],
)
def test_an_artifact_error_reads_in_spanish_with_its_file_and_command(
    problem: str, opening: str
) -> None:
    error = ArtifactError(
        problem,
        "map.parquet",
        Path("data/processed"),
        command="uv run python -m thematic_redundancy.topic_map.run_experiment",
        detail="an English detail that the user never sees",
    )

    message = texts.error_message(error)

    assert message.startswith(opening)
    assert "uv run python -m thematic_redundancy.topic_map.run_experiment" in message
    assert "English detail" not in message


def test_an_artifact_error_without_a_command_says_what_to_pass() -> None:
    error = ArtifactError("missing", "raw", Path("data/raw"), command=None, detail="no snapshot")

    message = texts.error_message(error)

    assert message.startswith("No hay ninguna instantánea")
    assert "--snapshot-id" in message


def test_an_argument_error_reads_in_spanish() -> None:
    message = texts.settings_message(SettingsError("unrecognized arguments: --x"))

    assert message.startswith("Los argumentos de la aplicación no son válidos")
    assert "--project-root" in message
    assert "--snapshot-id" in message


def test_each_gap_is_described_in_spanish_with_its_counts_and_shares(project: Project) -> None:
    _, _, gaps = loaded(project)
    described = [
        texts.gap_description(gap, program_names=gaps.program_names, windows=gaps.windows)
        for gap in gaps.gaps
    ]

    small, pair, high, low, decline = described
    assert small == (
        "Tema pequeño y coherente: 3 tesis (el corte del cuartil inferior de tamaño es 3) "
        "y una coherencia NPMI de 0.200 (el mínimo es 0.150)."
    )
    assert pair == ("Temas vecinos (coseno entre centroides 0.610) con 1 tesis puente entre ellos.")
    assert high == (
        "Presente en Ingeniería Industrial: 4 de 6 tesis (66.7 %). Ausente en "
        "Ingeniería Mecánica, Mecánica-Eléctrica y Mecatrónica: 0 de 3 tesis (0.0 %)."
    )
    assert low.startswith("Presente en Ingeniería de Sistemas: 3 de 5 tesis (60.0 %).")
    assert decline == (
        "Su participación bajó de 57.1 % (4 de 7 tesis, 2021–2023) a 14.3 % (1 de 7 tesis, "
        "2024–2026): una caída de 42.9 puntos porcentuales."
    )


# --- Tables -----------------------------------------------------------------------------------


def test_the_topic_table_lists_each_topic_with_its_size_and_terms(project: Project) -> None:
    _, topic_map, _ = loaded(project)

    table = topic_table(topic_map)

    assert tuple(table.columns) == TOPIC_TABLE_COLUMNS
    assert table.to_dict("records") == [
        {"Tema": topic, "Tesis": size, "Términos principales": ", ".join(TOPIC_TERMS[topic])}
        for topic, size in ((0, 5), (1, 4), (2, 3))
    ]


def test_the_thesis_table_links_each_shown_thesis_to_the_repository(project: Project) -> None:
    _, topic_map, _ = loaded(project)
    points = filter_points(topic_map.points, topics=[2])

    table = thesis_table(points, topic_map.topics)

    assert tuple(table.columns) == THESIS_TABLE_COLUMNS
    expected = [t for t in THESES if t.topic == 2]
    assert sorted(table["Título"]) == sorted(t.title for t in expected)
    assert sorted(table[LINK_COLUMN]) == sorted(t.url for t in expected)
    assert set(table["Tema"]) == {texts.topic_label(2, TOPIC_TERMS[2])}
    assert set(table["Carrera"]) == {PROGRAM_NAMES["mecanica"], PROGRAM_NAMES["sistemas"]}


def test_the_thesis_table_lists_the_theses_without_a_topic_last(project: Project) -> None:
    _, topic_map, _ = loaded(project)

    table = thesis_table(topic_map.points, topic_map.topics)

    assert list(table.columns[:2]) == ["Título", LINK_COLUMN]
    assert list(table["Tema"].iloc[-2:]) == [texts.NOISE_LABEL, texts.NOISE_LABEL]
    assert texts.NOISE_LABEL not in set(table["Tema"].iloc[:-2])
    assert list(table["Tema"].iloc[:5]) == [texts.topic_label(0, TOPIC_TERMS[0])] * 5


def test_the_gap_table_shows_rule_topics_programs_years_affinity_and_stability(
    project: Project,
) -> None:
    _, _, gaps = loaded(project)

    table = gap_table(gaps.gaps, gaps)

    assert tuple(table.columns) == GAP_TABLE_COLUMNS
    assert list(table["N.º"]) == [1, 2, 3, 4, 5]
    assert list(table["Regla"]) == [texts.RULE_LABELS[row.rule] for row in STABLE_GAPS]
    pair = table.iloc[1]
    assert pair["Temas"] == (
        f"{texts.topic_label(0, TOPIC_TERMS[0])} · {texts.topic_label(1, TOPIC_TERMS[1])}"
    )
    assert pair["Carreras"] == "Ingeniería Industrial (5); Ingeniería de Sistemas (4)"
    assert pair["Años"] == "2021 (2), 2022 (2), 2023 (1), 2024 (1), 2025 (2), 2026 (1)"
    assert pair["Tesis"] == 9
    assert pair["Estabilidad"] == "En 4 de 5 corridas"
    assert list(table["Afinidad"]) == ["—", "—", "Alta confianza (afinidad 2)", "Afinidad 1", "—"]
    assert "English reason" not in table.to_csv()


def test_the_member_table_names_each_part_and_links_each_thesis(project: Project) -> None:
    _, _, gaps = loaded(project)

    table = member_table(members_of(gaps, STABLE_GAPS[1].gap_id))

    assert tuple(table.columns) == MEMBER_TABLE_COLUMNS
    assert list(table.columns[:2]) == ["Título", LINK_COLUMN]
    bridge = table[table[LINK_COLUMN] == BRIDGE.url].iloc[0]
    assert (bridge["Título"], bridge["Parte"]) == (BRIDGE.title, "Tesis puente")
    assert set(table["Parte"]) == {"Primer tema", "Segundo tema", "Tesis puente"}
    assert PLANTED_PERSON not in table.to_csv()


# --- Map figure -------------------------------------------------------------------------------


def test_each_topic_keeps_one_color_and_shape_whatever_the_filter() -> None:
    assert charts.topic_style(0, dark=False) == (charts.PALETTE_LIGHT[0], "circle")
    assert charts.topic_style(7, dark=False) == (charts.PALETTE_LIGHT[7], "circle")
    assert charts.topic_style(8, dark=False) == (charts.PALETTE_LIGHT[0], "square")
    assert charts.topic_style(17, dark=False) == (charts.PALETTE_LIGHT[1], "diamond")
    assert charts.topic_style(3, dark=True) == (charts.PALETTE_DARK[3], "circle")
    assert charts.topic_style(-1, dark=False) == (charts.NOISE_COLOR, "circle")


def test_the_figure_draws_one_trace_per_topic_and_the_noise(project: Project) -> None:
    _, topic_map, _ = loaded(project)

    figure = charts.topic_map_figure(topic_map.points, topic_map.topics)

    names = [trace.name for trace in figure.data]
    assert names == [texts.topic_label(t, TOPIC_TERMS[t]) for t in (0, 1, 2)] + [texts.NOISE_LABEL]
    assert [len(trace.x) for trace in figure.data] == [5, 4, 3, 2]
    assert [annotation.text for annotation in figure.layout.annotations] == ["0", "1", "2"]


def test_a_filtered_figure_keeps_the_colors_of_the_full_one(project: Project) -> None:
    _, topic_map, _ = loaded(project)
    full = charts.topic_map_figure(topic_map.points, topic_map.topics)

    filtered = charts.topic_map_figure(
        filter_points(topic_map.points, topics=[1]), topic_map.topics
    )

    assert len(filtered.data) == 1
    assert filtered.data[0].marker.color == full.data[1].marker.color


def test_the_hover_shows_title_program_and_year_only(project: Project) -> None:
    _, topic_map, _ = loaded(project)

    figure = charts.topic_map_figure(topic_map.points, topic_map.topics, dark=True)

    spec = json.dumps(figure.to_plotly_json(), ensure_ascii=False)
    first = THESES[0]
    assert first.title in spec
    assert PROGRAM_NAMES[first.program] in spec
    assert first.doc_code not in spec
    assert first.url not in spec
    assert PLANTED_PERSON not in spec
    assert "AUT-" not in spec and "ADV-" not in spec
    assert figure.data[0].marker.color == charts.PALETTE_DARK[0]
