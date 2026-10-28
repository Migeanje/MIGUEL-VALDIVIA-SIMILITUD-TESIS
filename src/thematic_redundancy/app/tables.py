"""The app's display tables (T28): plain pandas tables with Spanish headers (O01), built from
the data layer's records, with no Streamlit import.

A table holds only what the app shows of a thesis: its title, program, year, topic and
repository link (:data:`LINK_COLUMN`), never a person, a doc code, an abstract or a keyword.
"""

from collections.abc import Sequence

import pandas as pd

from thematic_redundancy.app import texts
from thematic_redundancy.app.data import NOISE_TOPIC, GapRecords, StableGap, TopicMapView
from thematic_redundancy.topic_map.experiment import TopicRow

LINK_COLUMN = "Repositorio"
TOPIC_TABLE_COLUMNS = ("Tema", "Tesis", "Términos principales")
THESIS_TABLE_COLUMNS = ("Título", LINK_COLUMN, "Carrera", "Año", "Tema")
GAP_TABLE_COLUMNS = (
    "N.º",
    "Regla",
    "Temas",
    "Descripción",
    "Carreras",
    "Años",
    "Tesis",
    "Afinidad",
    "Estabilidad",
)
MEMBER_TABLE_COLUMNS = ("Título", LINK_COLUMN, "Carrera", "Año", "Parte")


def topic_table(topic_map: TopicMapView) -> pd.DataFrame:
    """Return one row per topic of the map: its number, its theses and all its top terms."""
    rows = [(topic.topic, topic.size, ", ".join(topic.terms)) for topic in topic_map.topics]
    return pd.DataFrame(rows, columns=list(TOPIC_TABLE_COLUMNS))


def thesis_table(points: pd.DataFrame, topics: Sequence[TopicRow]) -> pd.DataFrame:
    """Return the theses of ``points``, by topic, the theses without a topic last, then by year
    (latest first) and title."""
    terms = {topic.topic: topic.terms for topic in topics}
    ordered = points.assign(_noise=points["topic"] == NOISE_TOPIC).sort_values(
        ["_noise", "topic", "year", "title"], ascending=[True, True, False, True]
    )
    table = pd.DataFrame(
        {
            "Título": ordered["title"],
            LINK_COLUMN: ordered["url"],
            "Carrera": ordered["program_name"],
            "Año": ordered["year"],
            "Tema": [texts.topic_label(t, terms.get(t, ())) for t in ordered["topic"]],
        }
    )
    return table.reset_index(drop=True)


def gap_table(gaps: Sequence[StableGap], records: GapRecords) -> pd.DataFrame:
    """Return one row per gap of ``gaps``: its number, rule, topics, a description of its
    evidence, its programs and years with their theses, rule (c)'s affinity group and its
    stability."""
    names = records.program_names
    rows = [
        (
            gap.number,
            texts.RULE_LABELS.get(gap.row.rule, gap.row.rule),
            texts.gap_topics_text(gap),
            texts.gap_description(gap, program_names=names, windows=records.windows),
            texts.program_counts_text(gap.row.programs, gap.row.program_counts, names),
            texts.year_counts_text(gap.row.years, gap.row.year_counts),
            gap.row.theses,
            texts.affinity_text(gap),
            texts.support_text(gap, records.runs),
        )
        for gap in gaps
    ]
    return pd.DataFrame(rows, columns=list(GAP_TABLE_COLUMNS))


def member_table(members: pd.DataFrame) -> pd.DataFrame:
    """Return the theses of one gap, by part, then by year (latest first) and title."""
    order = {part: index for index, part in enumerate(texts.PART_LABELS)}
    ordered = members.assign(_order=members["part"].map(order)).sort_values(
        ["_order", "year", "title"], ascending=[True, False, True]
    )
    table = pd.DataFrame(
        {
            "Título": ordered["title"],
            LINK_COLUMN: ordered["url"],
            "Carrera": ordered["program_name"],
            "Año": ordered["year"],
            "Parte": [texts.PART_LABELS.get(part, part) for part in ordered["part"]],
        }
    )
    return table.reset_index(drop=True)
