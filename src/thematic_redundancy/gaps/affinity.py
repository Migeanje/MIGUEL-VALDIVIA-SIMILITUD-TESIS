"""The career-affinity matrix of input U4, which gap rule (c) reads (T24).

``config/career_affinity.yaml`` was fixed on 2026-10-19, before any topic result existed, and
``tests/unit/test_career_affinity.py`` pins its pairs. This module is its loader: it accepts the
file exactly as committed and refuses a malformed one instead of guessing. The file holds:

- ``version`` 1 and ``decided_on``, a YAML date;
- ``scale``: the scores 0, 1 and 2, each with its meaning;
- ``programs``: the program keys, unique, in the order of ``config/default.yaml``;
- ``gap_target_excluded``: programs that are never the target of a rule (c) gap (D29:
  ``electronica``), unique, each a program;
- ``pairs``: one key ``"<a>-<b>"`` per unordered pair of distinct programs, ``a`` before ``b`` in
  the program order, each pair exactly once and nothing else, with an integer score of the scale.

A key written twice in one mapping is refused (the project's YAML loader); a boolean, a float or
a quoted number is not a score.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from thematic_redundancy.shared.config import _UniqueKeyLoader

AFFINITY_SCALE = (0, 1, 2)
"""The scores of U4: 0 no affinity, 1 partial, 2 high."""

PAIR_SEPARATOR = "-"
"""Joins the two program keys of a pair; a program key cannot hold it."""

ProgramKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


@dataclass(frozen=True)
class CareerAffinity:
    """The symmetric affinity of each pair of distinct programs, on the 0/1/2 scale."""

    programs: tuple[str, ...]
    """The program keys, in their declared order."""
    scale: tuple[int, ...]
    """The scores, ascending."""
    gap_target_excluded: frozenset[str]
    """Programs that are never the target of a rule (c) gap (D29)."""
    pairs: Mapping[tuple[str, str], int]
    """The score of each pair, keyed ``(a, b)`` with ``a`` before ``b`` in :attr:`programs`;
    read-only."""

    def score(self, first: str, second: str) -> int:
        """Return the affinity of two distinct programs, in either order.

        Raises:
            ValueError: if a program is unknown or both are the same.
        """
        self._require_program(first)
        self._require_program(second)
        if first == second:
            raise ValueError(f"a program has no affinity score with itself: '{first}'")
        position = self.programs.index
        key = (first, second) if position(first) < position(second) else (second, first)
        return self.pairs[key]

    def can_be_target(self, program: str) -> bool:
        """Tell whether ``program`` may be the target of a rule (c) gap.

        Raises:
            ValueError: if the program is unknown.
        """
        self._require_program(program)
        return program not in self.gap_target_excluded

    def _require_program(self, program: str) -> None:
        if program not in self.programs:
            raise ValueError(f"unknown program '{program}'; known: {list(self.programs)}")


class _AffinityFile(BaseModel):
    """The file's schema; see the module."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: StrictInt
    decided_on: Annotated[date, Field(strict=True)]
    scale: dict[StrictInt, Annotated[str, Field(min_length=1)]]
    programs: Annotated[tuple[ProgramKey, ...], Field(min_length=2)]
    gap_target_excluded: tuple[ProgramKey, ...]
    pairs: dict[str, StrictInt]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.version != 1:
            raise ValueError(f"version must be 1, not {self.version}")
        if tuple(sorted(self.scale)) != AFFINITY_SCALE:
            raise ValueError(f"scale must hold exactly {list(AFFINITY_SCALE)}, not {self.scale}")
        _require_unique(self.programs, "programs")
        _require_unique(self.gap_target_excluded, "gap_target_excluded")
        unknown = sorted(set(self.gap_target_excluded) - set(self.programs))
        if unknown:
            raise ValueError(f"gap_target_excluded names unknown programs: {unknown}")
        expected = {
            f"{first}{PAIR_SEPARATOR}{second}" for first, second in combinations(self.programs, 2)
        }
        missing, extra = sorted(expected - set(self.pairs)), sorted(set(self.pairs) - expected)
        if missing or extra:
            raise ValueError(
                "pairs must hold each pair of distinct programs once, the first program "
                f"earlier in the program order; missing {missing}, unexpected {extra}"
            )
        outside = {pair: value for pair, value in self.pairs.items() if value not in self.scale}
        if outside:
            raise ValueError(f"pairs with a score outside the scale: {outside}")
        return self


def _require_unique(values: Sequence[str], label: str) -> None:
    repeated = sorted({value for value in values if values.count(value) > 1})
    if repeated:
        raise ValueError(f"{label} must be unique; repeated: {repeated}")


def load_career_affinity(path: Path, *, programs: Sequence[str] | None = None) -> CareerAffinity:
    """Read and validate the career-affinity file at ``path``.

    Args:
        path: the YAML file, ``config/career_affinity.yaml`` in the project.
        programs: the configured program keys (``snapshot.programs`` of the configuration);
            when given, the file's programs must equal them in the same order.

    Raises:
        OSError: if the file cannot be read.
        yaml.YAMLError: if it is not valid YAML or a mapping repeats a key.
        ValueError: if its content breaks the contract of the module (a
            ``pydantic.ValidationError`` is one), or its programs differ from ``programs``.
    """
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    content = _AffinityFile.model_validate(raw)
    if programs is not None and tuple(programs) != content.programs:
        raise ValueError(
            f"the file's programs {list(content.programs)} differ from the configured "
            f"programs {list(programs)}"
        )
    pairs = {}
    for key, value in content.pairs.items():
        first, second = key.split(PAIR_SEPARATOR)
        pairs[first, second] = value
    return CareerAffinity(
        programs=content.programs,
        scale=tuple(sorted(content.scale)),
        gap_target_excluded=frozenset(content.gap_target_excluded),
        pairs=MappingProxyType(dict(sorted(pairs.items()))),
    )
