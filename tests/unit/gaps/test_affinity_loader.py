"""The loader of the career-affinity matrix (input U4) that gap rule (c) reads (T24).

The loader must accept ``config/career_affinity.yaml`` exactly as committed and refuse a
malformed copy. Every malformed copy is written to pytest's temporary directory, outside the
repository.
"""

from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from thematic_redundancy.gaps.affinity import CareerAffinity, load_career_affinity

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AFFINITY_PATH = PROJECT_ROOT / "config" / "career_affinity.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

PROGRAMS = ("sistemas", "industrial", "electronica", "mecanica", "minas")
"""The program keys in the order that ``config/default.yaml`` declares them."""


@pytest.fixture(scope="module")
def committed_text() -> str:
    return AFFINITY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def affinity() -> CareerAffinity:
    return load_career_affinity(AFFINITY_PATH, programs=PROGRAMS)


def write_copy(directory: Path, text: str) -> Path:
    path = directory / "career_affinity.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- The committed file -------------------------------------------------------------------


def test_the_committed_file_loads_with_its_programs_scale_and_excluded_targets(
    affinity: CareerAffinity,
) -> None:
    configured = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    configured_keys = tuple(program["key"] for program in configured["snapshot"]["programs"])

    assert affinity.programs == PROGRAMS == configured_keys
    assert affinity.scale == (0, 1, 2)
    assert affinity.gap_target_excluded == frozenset({"electronica"})


def test_every_pair_holds_the_committed_score_in_both_orders(
    affinity: CareerAffinity, committed_text: str
) -> None:
    written = yaml.safe_load(committed_text)["pairs"]

    for first, second in combinations(PROGRAMS, 2):
        expected = written[f"{first}-{second}"]
        assert affinity.score(first, second) == expected
        assert affinity.score(second, first) == expected
    assert len(affinity.pairs) == 10


def test_the_scores_named_in_the_u4_record(affinity: CareerAffinity) -> None:
    assert affinity.score("industrial", "mecanica") == 2
    assert affinity.score("mecanica", "electronica") == 2
    assert affinity.score("sistemas", "minas") == 0
    assert affinity.score("minas", "industrial") == 1


def test_electronica_can_never_be_a_gap_target_as_decided_in_d29(affinity: CareerAffinity) -> None:
    assert not affinity.can_be_target("electronica")
    for program in ("sistemas", "industrial", "mecanica", "minas"):
        assert affinity.can_be_target(program)


def test_a_program_has_no_score_with_itself_nor_an_unknown_program(
    affinity: CareerAffinity,
) -> None:
    with pytest.raises(ValueError, match="itself"):
        affinity.score("minas", "minas")
    with pytest.raises(ValueError, match="quimica"):
        affinity.score("minas", "quimica")
    with pytest.raises(ValueError, match="quimica"):
        affinity.can_be_target("quimica")


def test_the_loaded_matrix_cannot_be_changed(affinity: CareerAffinity) -> None:
    with pytest.raises(TypeError):
        affinity.pairs[("sistemas", "minas")] = 2  # type: ignore[index]


def test_the_programs_may_be_omitted_and_are_then_read_from_the_file() -> None:
    assert load_career_affinity(AFFINITY_PATH).programs == PROGRAMS


def test_the_configured_programs_must_match_the_file_in_order() -> None:
    reordered = ("industrial", "sistemas", "electronica", "mecanica", "minas")

    with pytest.raises(ValueError, match="programs"):
        load_career_affinity(AFFINITY_PATH, programs=reordered)
    with pytest.raises(ValueError, match="programs"):
        load_career_affinity(AFFINITY_PATH, programs=PROGRAMS[:4])


def test_a_copy_with_its_pairs_in_another_line_order_loads_the_same_matrix(
    tmp_path: Path, committed_text: str, affinity: CareerAffinity
) -> None:
    head, pairs_block = committed_text.split("pairs:\n")
    lines = [line for line in pairs_block.splitlines() if line.strip()]
    shuffled = head + "pairs:\n" + "\n".join(reversed(lines)) + "\n"

    assert load_career_affinity(write_copy(tmp_path, shuffled), programs=PROGRAMS) == affinity


# --- Malformed copies -----------------------------------------------------------------------


def replace_once(old: str, new: str) -> Callable[[str], str]:
    def transform(text: str) -> str:
        assert text.count(old) == 1, old
        return text.replace(old, new)

    return transform


def append(line: str) -> Callable[[str], str]:
    return lambda text: text + line


MALFORMED: dict[str, Callable[[str], str]] = {
    "a value outside the scale": replace_once("industrial-mecanica: 2", "industrial-mecanica: 3"),
    "a negative value": replace_once("sistemas-minas: 0", "sistemas-minas: -1"),
    "a boolean value": replace_once("sistemas-minas: 0", "sistemas-minas: false"),
    "a quoted value": replace_once("sistemas-minas: 0", 'sistemas-minas: "0"'),
    "a float value": replace_once("industrial-minas: 1", "industrial-minas: 1.0"),
    "a repeated pair key": append("  mecanica-minas: 1\n"),
    "a reversed pair key": replace_once("sistemas-industrial: 1", "industrial-sistemas: 1"),
    "a missing pair": replace_once("  electronica-minas: 0\n", ""),
    "a pair with an unknown program": append("  sistemas-quimica: 1\n"),
    "a program paired with itself": append("  minas-minas: 2\n"),
    "a pair key without a separator": replace_once("mecanica-minas: 1", "mecanicaminas: 1"),
    "an unknown excluded target": replace_once(
        "gap_target_excluded: [electronica]", "gap_target_excluded: [quimica]"
    ),
    "a repeated excluded target": replace_once(
        "gap_target_excluded: [electronica]", "gap_target_excluded: [electronica, electronica]"
    ),
    "a missing excluded-target list": replace_once("gap_target_excluded: [electronica]\n", ""),
    "a repeated program": replace_once(
        "programs: [sistemas, industrial, electronica, mecanica, minas]",
        "programs: [sistemas, industrial, electronica, mecanica, minas, minas]",
    ),
    "a program key with a dash": replace_once(
        "programs: [sistemas, industrial, electronica, mecanica, minas]",
        "programs: [sistemas, industrial, electronica, mecanica, minas, mina-x]",
    ),
    "a scale without its high score": replace_once("  2: high affinity\n", ""),
    "a scale with an extra score": replace_once("  2: high affinity\n", "  2: high\n  3: total\n"),
    "another version": replace_once("version: 1", "version: 2"),
    "a quoted decision date": replace_once("decided_on: 2026-10-19", 'decided_on: "2026-10-19"'),
    "an unknown top-level key": append("weights: {}\n"),
    "a list instead of a mapping": lambda text: "- sistemas\n- industrial\n",
    "broken YAML": append("pairs: [\n"),
}


@pytest.mark.parametrize("defect", sorted(MALFORMED))
def test_a_malformed_copy_is_refused(defect: str, tmp_path: Path, committed_text: str) -> None:
    text = MALFORMED[defect](committed_text)
    assert text != committed_text

    with pytest.raises((ValueError, yaml.YAMLError)):
        load_career_affinity(write_copy(tmp_path, text))


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_career_affinity(tmp_path / "absent.yaml")
