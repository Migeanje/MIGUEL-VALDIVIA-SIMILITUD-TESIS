"""Structural checks of the reproducibility package (T33), ``docs/reproducibility/``.

The package describes the repository as it is: its commands, its configuration, its tracked
results and the counts of its corpus. These tests read the package as plain text and compare
it with the repository, so that a renamed command, a new configuration block, a new tracked
result or a changed count fails here instead of leaving the package silently out of date.
They read no thesis data: only Markdown, source files, ``config/default.yaml`` and the tracked
manifests under ``results/``.
"""

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "docs" / "reproducibility"
README_PATH = PROJECT_ROOT / "README.md"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
SOURCE_DIR = PROJECT_ROOT / "src"

INDEX_NAME = "README.md"
DOCUMENT_NAMES = (
    "data_card.md",
    "environment.md",
    "configuration.md",
    "results_index.md",
    "how_to_run.md",
)
"""The documents of the package besides its index, which links each one."""

T19_MANIFEST = RESULTS_DIR / "experiments" / "20261002T224412Z" / "t19" / "manifest.json"
"""A tracked manifest whose ``versions`` the environment document must state."""

MODULE_COMMAND = re.compile(r"python -m (thematic_redundancy(?:\.\w+)+)")
SCRIPT_COMMAND = re.compile(r"python (experiments/\w+\.py)")
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z-]*")
LINK = re.compile(r"\]\(([^)\s]+)\)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
FENCED_BLOCK = re.compile(r"^```[a-z]*\n(.*?)^```", re.MULTILINE | re.DOTALL)

TRACKED_ROOTS = ("config/", "docs/", "experiments/", "odd/", "results/", "src/", "tests/")
"""Path prefixes of tracked repository content; ``data/`` and ``tessdata/`` are gitignored."""

SEED_KEYS = (
    "objectives.verification_sample.seed",
    "experiment.selection.f1.bootstrap_seed",
    "redundancy.cv_seed",
    "labeling.seed",
    "temporal.sample.seed",
    "expert_samples.seed",
)
"""The configured seeds besides the five of ``seeds``, by their dotted configuration keys."""


def read(name: str) -> str:
    return (PACKAGE_DIR / name).read_text(encoding="utf-8")


def commands(text: str) -> set[str]:
    """Return the modules and scripts that ``text`` runs with ``python -m`` or ``python``."""
    return set(MODULE_COMMAND.findall(text)) | set(SCRIPT_COMMAND.findall(text))


def command_source(command: str) -> Path:
    """Return the source file of a module path or of an ``experiments/`` script."""
    if command.startswith("experiments/"):
        return PROJECT_ROOT / command
    return SOURCE_DIR.joinpath(*command.split(".")).with_suffix(".py")


def command_lines(text: str) -> list[str]:
    """Return each command of the fenced code blocks of ``text``, continuation lines joined."""
    lines: list[str] = []
    for block in FENCED_BLOCK.findall(text):
        for line in block.replace("\\\n", " ").splitlines():
            if commands(line):
                lines.append(line)
    return lines


def has_main_guard(tree: ast.Module) -> bool:
    """Return whether ``tree`` holds a top-level ``if __name__ == "__main__":``."""
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(right, ast.Constant) and right.value == "__main__"
            for right in node.test.comparators
        )
        for node in tree.body
    )


def value_at(config: dict, key: str) -> object:
    """Return the value of the dotted ``key`` of ``config``."""
    value = config
    for part in key.split("."):
        value = value[part]
    return value


def section(text: str, heading: str) -> str:
    """Return the body of the ``## heading`` section of ``text``, up to the next ``## ``."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"the section '## {heading}' is missing"
    return match.group(1)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_index_links_every_document_of_the_package() -> None:
    index = read(INDEX_NAME)
    for name in DOCUMENT_NAMES:
        assert (PACKAGE_DIR / name).is_file(), f"docs/reproducibility/{name} is missing"
        assert f"]({name})" in index, f"the package index does not link {name}"


@pytest.mark.parametrize("name", (INDEX_NAME, *DOCUMENT_NAMES))
def test_every_relative_link_resolves(name: str) -> None:
    for target in LINK.findall(read(name)):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = (PACKAGE_DIR / target.split("#", 1)[0]).resolve()
        assert path.exists(), f"{name} links {target}, which does not exist"


@pytest.mark.parametrize("name", (INDEX_NAME, *DOCUMENT_NAMES))
def test_every_repository_path_named_exists(name: str) -> None:
    for span in CODE_SPAN.findall(read(name)):
        if not span.startswith(TRACKED_ROOTS) or any(mark in span for mark in "<*{ "):
            continue
        assert (PROJECT_ROOT / span).exists(), f"{name} names {span}, which does not exist"


def test_every_command_of_the_guide_is_a_module_or_script_with_main() -> None:
    guide = commands(read("how_to_run.md"))
    assert guide, "the guide names no command"
    for command in sorted(guide):
        source = command_source(command)
        assert source.is_file(), f"the guide runs {command}, which has no source file"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert "main" in names, f"{command} defines no top-level main()"
        assert has_main_guard(tree), f"{command} has no __main__ guard"


def test_every_option_of_the_guide_is_declared_by_its_command() -> None:
    lines = command_lines(read("how_to_run.md"))
    assert lines, "the guide's code blocks hold no command"
    for line in lines:
        (command,) = commands(line)
        source = command_source(command).read_text(encoding="utf-8")
        for option in OPTION.findall(line):
            assert f'"{option}"' in source, f"{command} declares no option {option}: {line}"


def test_the_guide_covers_every_command_of_the_readme() -> None:
    missing = commands(README_PATH.read_text(encoding="utf-8")) - commands(read("how_to_run.md"))
    assert not missing, f"how_to_run.md lacks the README's commands: {sorted(missing)}"


def test_the_results_index_names_every_tracked_result() -> None:
    index = set(CODE_SPAN.findall(read("results_index.md")))
    files = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in RESULTS_DIR.rglob("*")
        if path.is_file() and path.name != ".lock"
    )
    assert files, "results/ holds no file"
    missing = [path for path in files if path not in index]
    assert not missing, f"results_index.md lacks a row for: {missing}"


def test_the_configuration_guide_names_every_block_and_every_seed(config: dict) -> None:
    guide = read("configuration.md")
    for block in config:
        assert f"`{block}`" in guide, f"configuration.md does not name the block `{block}`"
    listed = section(guide, "Seeds")
    for seed in config["seeds"]:
        assert re.search(rf"\b{seed}\b", listed), f"the Seeds section does not list {seed}"
    for key in SEED_KEYS:
        seed = value_at(config, key)
        row = rf"\|\s*{seed}\s*\|\s*`{re.escape(key)}`\s*\|"
        assert re.search(row, listed), f"the Seeds section has no row {seed} | `{key}`"


def test_the_data_card_states_the_declared_fichas_counts(config: dict) -> None:
    card = read("data_card.md")
    expected = config["fichas"]["expected"]
    assert f"`{expected['snapshot_id']}`" in card
    for key, value in expected.items():
        if key == "snapshot_id":
            continue
        row = rf"\|\s*`{key}`\s*\|\s*{value:,}\s*\|"
        assert re.search(row, card), f"the data card has no row `{key}` | {value:,}"


def test_the_environment_states_the_recorded_library_versions() -> None:
    environment = read("environment.md")
    versions = json.loads(T19_MANIFEST.read_text(encoding="utf-8"))["versions"]
    for name, version in versions.items():
        row = rf"\|\s*`?{re.escape(name)}`?\s*\|\s*{re.escape(version)}\s*\|"
        assert re.search(row, environment), f"environment.md has no row {name} | {version}"
