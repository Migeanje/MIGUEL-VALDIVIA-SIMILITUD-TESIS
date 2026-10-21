"""The smoke selection of ``tests/conftest.py``, checked through its functions.

The conftest is loaded as a plain module, and pytest's configuration and test items are
stand-ins, so no inner pytest run (pytester) is needed.
"""

import importlib.util
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"


def load_conftest() -> ModuleType:
    """Import the shared conftest under a name of its own, so pytest's copy stays apart."""
    spec = importlib.util.spec_from_file_location("smoke_selection_conftest", CONFTEST_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conftest = load_conftest()


@pytest.mark.parametrize(
    ("expression", "names"),
    [
        ("", set()),
        ("smoke", {"smoke"}),
        ("not smoke", {"smoke"}),
        ("not pre-smoke", {"pre-smoke"}),
        ("smoke and not slow", {"smoke", "slow"}),
        ("(smoke or unit) and not slow", {"smoke", "unit", "slow"}),
        ('x(name="smoke")', {"x"}),
        ("x(name='smoke', other=smoke)", {"x"}),
        ("x(smoke=1)", {"x"}),
    ],
)
def test_marker_names_are_the_words_that_name_a_marker(expression: str, names: set[str]) -> None:
    assert conftest.marker_names(expression) == names


class StandInConfig:
    """The parts of ``pytest.Config`` that the selection reads, with a deselection log."""

    def __init__(self, *, smoke: bool = False, markexpr: str | None = "") -> None:
        self._options = {"--smoke": smoke, "markexpr": markexpr}
        self.deselected: list[Any] = []
        self.hook = SimpleNamespace(pytest_deselected=self._deselect)

    def getoption(self, name: str) -> Any:
        return self._options[name]

    def _deselect(self, items: Iterable[Any]) -> None:
        self.deselected.extend(items)


@pytest.mark.parametrize(
    ("smoke", "markexpr", "requested"),
    [
        (False, "", False),
        (False, None, False),
        (False, "not pre-smoke", False),
        (False, 'x(name="smoke")', False),
        (False, "smoke", True),
        (False, "not smoke", True),
        (True, "", True),
        (True, "unit", True),
    ],
)
def test_the_smoke_tests_are_asked_for_by_the_flag_or_by_an_expression_naming_them(
    smoke: bool, markexpr: str | None, requested: bool
) -> None:
    config = StandInConfig(smoke=smoke, markexpr=markexpr)

    assert conftest.smoke_requested(config) is requested


class StandInItem:
    """A collected test that carries the markers named in ``markers``."""

    def __init__(self, name: str, *markers: str) -> None:
        self.name = name
        self._markers = markers

    def get_closest_marker(self, name: str) -> object | None:
        return SimpleNamespace(name=name) if name in self._markers else None


def collected() -> list[StandInItem]:
    return [
        StandInItem("unit"),
        StandInItem("smoke", "smoke"),
        StandInItem("slow", "slow"),
        StandInItem("slow smoke", "slow", "smoke"),
    ]


def names(items: Iterable[StandInItem]) -> list[str]:
    return [item.name for item in items]


def test_a_run_leaves_out_the_smoke_tests_and_reports_them_as_deselected() -> None:
    config = StandInConfig(markexpr="not slow")
    items = collected()

    conftest.pytest_collection_modifyitems(config, items)

    assert names(items) == ["unit", "slow"]
    assert names(config.deselected) == ["smoke", "slow smoke"]


@pytest.mark.parametrize(
    ("smoke", "markexpr"), [(True, ""), (False, "smoke and not slow")], ids=["flag", "expression"]
)
def test_a_run_that_asks_for_the_smoke_tests_keeps_every_test_for_the_other_filters(
    smoke: bool, markexpr: str
) -> None:
    config = StandInConfig(smoke=smoke, markexpr=markexpr)
    items = collected()

    conftest.pytest_collection_modifyitems(config, items)

    assert names(items) == names(collected())
    assert config.deselected == []
