"""The map page (T28): T19's 2-D UMAP map of the theses, colored by topic (T16, D30 E5).

It shows how many theses, topics and noise theses the map holds, the map filtered by topic and
by program, the topics with their top terms and sizes, and the theses shown, each linked to its
repository item. The hover gives a thesis's title, program and year only. Streamlit runs this
script as a page of ``main.py``.
"""

import streamlit as st

from thematic_redundancy.app import charts, tables, texts, ui
from thematic_redundancy.app.data import NOISE_TOPIC, filter_points


def render() -> None:
    """Show the map page."""
    st.title(texts.MAP_TITLE)
    ui.provisional_notice(ui.experiment())
    topic_map = ui.topic_map()
    points, topics = topic_map.points, topic_map.topics
    st.caption(texts.MAP_INTRO)
    counts = st.columns(3)
    counts[0].metric(texts.MAP_THESES_METRIC, len(points))
    counts[1].metric(texts.MAP_TOPICS_METRIC, len(topics))
    counts[2].metric(texts.MAP_NOISE_METRIC, int((points["topic"] == NOISE_TOPIC).sum()))

    terms = {topic.topic: topic.terms for topic in topics}
    names = dict(zip(points["program"], points["program_name"], strict=True))
    filters = st.columns([3, 3, 2], vertical_alignment="bottom")
    chosen_topics = filters[0].multiselect(
        texts.MAP_TOPICS_FILTER,
        list(terms),
        format_func=lambda topic: texts.topic_label(topic, terms[topic]),
        placeholder=texts.MAP_TOPICS_PLACEHOLDER,
        key="map_topics",
    )
    chosen_programs = filters[1].multiselect(
        texts.MAP_PROGRAMS_FILTER,
        sorted(names, key=lambda program: names[program]),
        format_func=lambda program: names[program],
        placeholder=texts.MAP_PROGRAMS_PLACEHOLDER,
        key="map_programs",
    )
    noise = filters[2].toggle(
        texts.MAP_NOISE_TOGGLE, value=True, disabled=bool(chosen_topics), key="map_noise"
    )
    shown = filter_points(
        points, topics=chosen_topics, programs=chosen_programs, include_noise=noise
    )
    st.caption(texts.MAP_SHOWN.format(count=len(shown)))
    if shown.empty:
        st.info(texts.MAP_EMPTY)
    else:
        st.plotly_chart(
            charts.topic_map_figure(shown, topics, dark=ui.is_dark()),
            key="map_chart",
            config={"displaylogo": False},
        )

    st.subheader(texts.MAP_TOPICS_HEADER)
    st.caption(texts.MAP_TOPICS_CAPTION)
    st.dataframe(tables.topic_table(topic_map), hide_index=True)
    if not shown.empty:
        st.subheader(texts.MAP_THESES_HEADER)
        st.dataframe(
            tables.thesis_table(shown, topics), hide_index=True, column_config=ui.link_column()
        )


if __name__ == "__main__":
    render()
