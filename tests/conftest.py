"""Test selection shared by every test module: the smoke suite runs only when asked for.

Tests marked ``smoke`` check that the third-party stack works together on this machine; they
are slow and need cached models. A run leaves them out, whatever other ``-m`` filter it uses,
unless the command asks for them:

- with a ``-m`` expression that names the ``smoke`` marker, such as ``-m smoke``, which then
  decides alone which smoke tests run;
- or with ``--smoke``, which stops this hook from leaving them out. Every other filter still
  applies to them as to any test: ``--smoke -m unit`` selects no smoke test, since none is
  marked ``unit``.
"""

import re

import pytest

SMOKE_MARKER = "smoke"

_MARK_TOKEN = re.compile(
    r"""(?P<space>\s+)
    |(?P<string>'[^']*'|"[^"]*")
    |(?P<punctuation>[()=,])
    |(?P<word>[\w:+\-.\[\]\\/]+)
    |(?P<other>.)""",
    re.VERBOSE,
)
"""One token of a ``-m`` expression, as pytest's mark expressions read them: a word holds
word characters and ``:+-.[]\\/``."""

_KEYWORDS = frozenset({"and", "or", "not"})


def marker_names(expression: str) -> set[str]:
    """Return the marker names that a ``-m`` expression tests.

    A word is a marker name unless it is ``and``, ``or`` or ``not``, or it stands next to
    ``=`` as the name or the value of a keyword argument, as in ``marker(reason="x")``.
    Quoted strings are values, never names.
    """
    tokens = [
        (match.lastgroup, match.group())
        for match in _MARK_TOKEN.finditer(expression)
        if match.lastgroup != "space"
    ]
    names = set()
    for index, (kind, text) in enumerate(tokens):
        if kind != "word" or text in _KEYWORDS:
            continue
        neighbours = {tokens[at][1] for at in (index - 1, index + 1) if 0 <= at < len(tokens)}
        if "=" not in neighbours:
            names.add(text)
    return names


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            f"do not leave out the tests marked {SMOKE_MARKER}, as a run does by default; "
            "-m and -k still filter them"
        ),
    )


def smoke_requested(config: pytest.Config) -> bool:
    """Tell whether the command asks for the smoke tests: ``--smoke``, or ``-m`` naming them."""
    return bool(config.getoption("--smoke")) or SMOKE_MARKER in marker_names(
        config.getoption("markexpr") or ""
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if smoke_requested(config):
        return
    smoke = [item for item in items if item.get_closest_marker(SMOKE_MARKER) is not None]
    if smoke:
        config.hook.pytest_deselected(items=smoke)
        items[:] = [item for item in items if item.get_closest_marker(SMOKE_MARKER) is None]
