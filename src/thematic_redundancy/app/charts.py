"""The app's topic map figure (T28): T19's 2-D UMAP layout as a plotly scatter, colored by topic,
with no Streamlit import.

Colors follow the topic, never its rank: topic ``t`` takes slot ``t mod 8`` of an eight-hue
categorical palette, validated for color-vision deficiencies on adjacent pairs in light and in
dark mode, and a marker shape from ``t div 8`` (circle, square, diamond), so a map of up to 24
topics keeps every topic apart by hue and shape together, and a filter never repaints the
topics it keeps. Each topic is also labeled by its number at its median position, and the
tables of the page list the same values, so no reading depends on color alone. The noise
(-1) is drawn small and gray, as context.

The hover shows a thesis's title, program and year, and its topic; no person, doc code or link.
"""

import html
import textwrap
from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from thematic_redundancy.app import texts
from thematic_redundancy.app.data import NOISE_TOPIC
from thematic_redundancy.topic_map.experiment import TopicRow

PALETTE_LIGHT = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
"""The categorical slots on a light surface: blue, orange, aqua, yellow, magenta, green, violet
and red, in this order."""
PALETTE_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)
"""The same eight hues, stepped for a dark surface."""
SYMBOLS = ("circle", "square", "diamond")
NOISE_COLOR = "#898781"
MARKER_SIZE = 9
NOISE_MARKER_SIZE = 6
HOVER_WIDTH = 70
"""Characters per hover line; a long title wraps."""
LABEL_BACKGROUND = {False: "rgba(255, 255, 255, 0.75)", True: "rgba(14, 17, 23, 0.75)"}


def topic_style(topic: int, *, dark: bool) -> tuple[str, str]:
    """Return the marker color and shape of ``topic``; the noise is gray circles."""
    if topic == NOISE_TOPIC:
        return NOISE_COLOR, SYMBOLS[0]
    palette = PALETTE_DARK if dark else PALETTE_LIGHT
    return palette[topic % len(palette)], SYMBOLS[(topic // len(palette)) % len(SYMBOLS)]


def topic_map_figure(
    points: pd.DataFrame, topics: Sequence[TopicRow], *, dark: bool = False
) -> go.Figure:
    """Return the scatter of ``points`` (the data layer's map points): one trace per topic in
    ascending order, then the noise, each topic labeled by its number."""
    terms = {topic.topic: topic.terms for topic in topics}
    figure = go.Figure()
    shown = set(points["topic"].tolist())
    order = sorted(shown - {NOISE_TOPIC}) + ([NOISE_TOPIC] if NOISE_TOPIC in shown else [])
    for topic in order:
        members = points[points["topic"] == topic]
        label = texts.topic_label(topic, terms.get(topic, ()))
        color, symbol = topic_style(topic, dark=dark)
        noise = topic == NOISE_TOPIC
        figure.add_trace(
            go.Scatter(
                x=members["x"].tolist(),
                y=members["y"].tolist(),
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "symbol": symbol,
                    "size": NOISE_MARKER_SIZE if noise else MARKER_SIZE,
                    "opacity": 0.55 if noise else 0.9,
                },
                customdata=[
                    [_hover_title(title), program, _year(year), label]
                    for title, program, year in zip(
                        members["title"], members["program_name"], members["year"], strict=True
                    )
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
        if not noise:
            figure.add_annotation(
                x=float(members["x"].median()),
                y=float(members["y"].median()),
                text=str(topic),
                showarrow=False,
                font={"size": 13},
                bgcolor=LABEL_BACKGROUND[dark],
                borderpad=2,
            )
    axis = {"showgrid": False, "zeroline": False, "showticklabels": False, "title": None}
    figure.update_layout(
        height=620,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis=axis,
        yaxis=axis,
        hovermode="closest",
        hoverdistance=24,
        legend={"title": {"text": texts.MAP_TOPICS_FILTER}, "itemsizing": "constant"},
    )
    return figure


def _hover_title(title: str) -> str:
    """Return ``title`` escaped for plotly's hover markup and wrapped into short lines."""
    return "<br>".join(textwrap.wrap(html.escape(title, quote=False), HOVER_WIDTH))


def _year(year: object) -> str:
    return "" if pd.isna(year) else str(year)
