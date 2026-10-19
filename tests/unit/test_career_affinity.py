"""The career-affinity matrix of input U4 (``config/career_affinity.yaml``), read by gap rule (c).

The matrix was agreed before any topic result existed, so these tests check its shape and pin
its values. The file is read here as plain YAML; its loader belongs to the gap rules (T24).
"""

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AFFINITY_PATH = PROJECT_ROOT / "config" / "career_affinity.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

DECLARED_PROGRAMS = ["sistemas", "industrial", "electronica", "mecanica", "minas"]
"""Program keys in the order that ``config/default.yaml`` declares them."""

U4_PAIRS_SHA256 = "cd6b9b3087d1d180b6b439b9b89301a0ba7ff839566914e1e47ef5b999cea959"
"""SHA-256 of the pairs decided on 2026-10-19: sorted ``a-b=value`` lines joined by line feeds."""


@pytest.fixture(scope="module")
def affinity() -> dict[str, Any]:
    return yaml.safe_load(AFFINITY_PATH.read_text(encoding="utf-8"))


def configured_program_keys() -> list[str]:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    return [program["key"] for program in config["snapshot"]["programs"]]


def pair_keys_as_written() -> list[str]:
    """Return the keys of ``pairs`` as they appear in the file, repeated keys included.

    ``yaml.safe_load`` keeps only the last value of a repeated key, so the node tree is read.
    """
    root = yaml.compose(AFFINITY_PATH.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    pairs_node = next(value for key, value in root.value if key.value == "pairs")
    return [key.value for key, _ in pairs_node.value]


def test_the_programs_are_the_configured_programs_in_their_declared_order(
    affinity: dict[str, Any],
) -> None:
    assert affinity["programs"] == configured_program_keys()
    assert affinity["programs"] == DECLARED_PROGRAMS


def test_every_pair_of_distinct_programs_appears_exactly_once_and_nothing_else(
    affinity: dict[str, Any],
) -> None:
    expected = [f"{a}-{b}" for a, b in combinations(affinity["programs"], 2)]

    keys = pair_keys_as_written()

    assert sorted(keys) == sorted(expected)
    assert len(keys) == len(set(keys)) == 10


def test_every_affinity_is_a_score_of_the_scale(affinity: dict[str, Any]) -> None:
    assert set(affinity["scale"]) == {0, 1, 2}
    for pair, value in affinity["pairs"].items():
        assert type(value) is int, pair
        assert value in affinity["scale"], pair


def test_the_excluded_gap_targets_are_programs(affinity: dict[str, Any]) -> None:
    assert set(affinity["gap_target_excluded"]) <= set(affinity["programs"])


def test_electronica_is_never_a_gap_target_as_decided_in_d29(affinity: dict[str, Any]) -> None:
    assert affinity["gap_target_excluded"] == ["electronica"]


def test_the_matrix_is_frozen_as_decided_on_2026_10_19(affinity: dict[str, Any]) -> None:
    """The matrix was frozen on 2026-10-19, before any topic result existed (T16 had not run).

    The pin covers the pairs and their values, sorted, so a change to a comment or to the
    order of the lines keeps it. A deliberate change must be recorded in the plan as a new
    decision and update this value with it.
    """
    lines = sorted(f"{pair}={value}" for pair, value in affinity["pairs"].items())

    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    assert digest == U4_PAIRS_SHA256
