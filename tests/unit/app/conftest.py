"""Fixtures of the app's tests (T28): a synthetic project of saved artifacts (``app_artifacts``)
under pytest's temporary directory, the stand-in encoder and tokenizers in place of the real
models, and Streamlit's caches emptied around each test."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from app_artifacts import FakeEncoder, Project, load_fake_counter, load_fake_encoder, make_project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return make_project(tmp_path / "project")


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> type[FakeEncoder]:
    """Load the stand-in encoder and tokenizers wherever the app loads the real ones."""
    monkeypatch.setattr("thematic_redundancy.app.query.load_sentence_encoder", load_fake_encoder)
    monkeypatch.setattr("thematic_redundancy.app.query.load_token_counter", load_fake_counter)
    FakeEncoder.texts.clear()
    return FakeEncoder


@pytest.fixture(autouse=True)
def _empty_streamlit_caches() -> Iterator[None]:
    """Keep the cached loaders of one test out of the next, which may reuse a path."""
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()
