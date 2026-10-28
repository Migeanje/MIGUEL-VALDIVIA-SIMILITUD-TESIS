"""Fixtures of the app's tests (T28): a synthetic project of saved artifacts (``app_artifacts``)
under pytest's temporary directory, and Streamlit's caches emptied around each test."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from app_artifacts import Project, make_project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return make_project(tmp_path / "project")


@pytest.fixture(autouse=True)
def _empty_streamlit_caches() -> Iterator[None]:
    """Keep the cached loaders of one test out of the next, which may reuse a path."""
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()
