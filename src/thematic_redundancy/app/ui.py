"""Streamlit helpers shared by the app's pages (T28): the arguments, the cached loaders, the
Spanish error messages and the provisional notice.

Streamlit's cache wraps the data layer's loaders and nothing else: it holds their results in
the server's memory, keyed by the project root and the snapshot, and never on disk. Nothing a
user types or selects is kept beyond the widgets of their session. A loader whose artifact
cannot be used shows its Spanish message (:func:`thematic_redundancy.app.texts.error_message`)
and stops the page, and the server's log gets the English detail; no traceback reaches the
page.
"""

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from thematic_redundancy.app import data, texts
from thematic_redundancy.app.data import (
    AppSettings,
    AppSources,
    ArtifactError,
    Experiment,
    GapRecords,
    SettingsError,
    TopicMapView,
)
from thematic_redundancy.app.tables import LINK_COLUMN

LOGGER = logging.getLogger(__name__)


def settings() -> AppSettings:
    """Return the app's settings from the arguments after ``--`` of ``streamlit run``, or show
    why they are not valid and stop the page."""
    try:
        return data.parse_arguments(sys.argv[1:])
    except SettingsError as error:
        LOGGER.warning("the app's arguments were refused: %s", error)
        st.error(texts.settings_message(error))
        st.stop()


@st.cache_data(show_spinner=texts.LOADING)
def _sources(root: str, snapshot_id: str | None) -> AppSources:
    return data.locate_sources(AppSettings(Path(root), snapshot_id))


@st.cache_data(show_spinner=texts.LOADING)
def _experiment(root: str, snapshot_id: str | None) -> Experiment:
    return data.load_experiment(_sources(root, snapshot_id))


@st.cache_data(show_spinner=texts.LOADING)
def _theses(root: str, snapshot_id: str | None) -> pd.DataFrame:
    return data.load_theses(_sources(root, snapshot_id), _experiment(root, snapshot_id))


@st.cache_data(show_spinner=texts.LOADING)
def _topic_map(root: str, snapshot_id: str | None) -> TopicMapView:
    sources, experiment = _sources(root, snapshot_id), _experiment(root, snapshot_id)
    return data.load_topic_map(sources, experiment, _theses(root, snapshot_id))


@st.cache_data(show_spinner=texts.LOADING)
def _gap_records(root: str, snapshot_id: str | None) -> GapRecords:
    sources, experiment = _sources(root, snapshot_id), _experiment(root, snapshot_id)
    return data.load_gaps(sources, experiment, _theses(root, snapshot_id))


def _loaded[T](loader: Callable[[str, str | None], T]) -> T:
    """Return what ``loader`` reads for the app's settings, or show why it cannot and stop."""
    chosen = settings()
    try:
        return loader(str(Path(chosen.project_root).resolve()), chosen.snapshot_id)
    except ArtifactError as error:
        LOGGER.warning("a saved artifact cannot be used: %s", error)
        st.error(texts.error_message(error))
        st.stop()


def experiment() -> Experiment:
    """Return T19's experiment (the winner and its map run), or stop the page."""
    return _loaded(_experiment)


def topic_map() -> TopicMapView:
    """Return the winner's 2-D map with its topics, or stop the page."""
    return _loaded(_topic_map)


def gap_records() -> GapRecords:
    """Return the gap run's stable gaps with their theses, or stop the page."""
    return _loaded(_gap_records)


def provisional_notice(loaded: Experiment) -> None:
    """Show that the results are provisional while T19's winner waits for T22."""
    if loaded.provisional:
        st.warning(texts.provisional_notice(loaded.winner), icon=":material/hourglass_top:")


def link_column() -> dict[str, object]:
    """Return the column settings that show :data:`LINK_COLUMN` as a link to the repository."""
    return {LINK_COLUMN: st.column_config.LinkColumn(LINK_COLUMN, display_text=texts.LINK_TEXT)}


def is_dark() -> bool:
    """Tell whether the viewer's theme is dark, so the map takes its dark palette."""
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) == "dark"


def sidebar() -> None:
    """Show the app's name, its privacy note and the button that reloads the saved results."""
    st.sidebar.markdown(f"**{texts.APP_TITLE}**")
    st.sidebar.caption(texts.APP_CAPTION)
    st.sidebar.caption(texts.PRIVACY_NOTE)
    if st.sidebar.button(texts.RELOAD_BUTTON, key="reload"):
        st.cache_data.clear()
