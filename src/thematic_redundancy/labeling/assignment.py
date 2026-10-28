"""Spread the labeled set over the pairs of annotators (T20b, D34 (8), D31 (b)).

With ``n`` annotators there are ``n (n - 1) / 2`` pairs of annotators, each called a group here:
3 annotators give 3 groups, 4 give 6. Every pair of the labeled set goes to exactly one group, and
both of its annotators label it, so each group shares about the same number of pairs for the
Cohen's κ diagnostic (D31 (a)): the 120 pairs give 3 groups of 40 or 6 of 20.

The pairs are taken pool by pool, in the draw's pool order, and within SBERT-high model by model;
within each, sorted by pair id and shuffled with ``random.Random("<seed>:assignment:<pool>")``
(``":<model>"`` added for SBERT-high). They are then dealt to the groups in turn, one after the
other, the count continuing from one pool to the next, so every group's total differs by at
most one pair (none when the groups divide the pairs), and so does its share of every pool and
model. Annotators are known by their number only, 1 to ``n``.
"""

import itertools
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from thematic_redundancy.labeling.pairs import POOLS


class DrawnPair(Protocol):
    """What the assignment reads of a drawn pair (a ``PairRow`` of the draw)."""

    @property
    def pair_id(self) -> str: ...

    @property
    def pool(self) -> str: ...

    @property
    def source(self) -> str | None: ...


@dataclass(frozen=True)
class Assignment:
    """The group of one pair: the two annotators who label it."""

    pair_id: str
    group: int
    """1 for the first pair of annotators, in the order of :func:`annotator_pairs`."""
    annotators: tuple[int, int]


def annotator_pairs(annotators: int) -> tuple[tuple[int, int], ...]:
    """Return every pair of the annotators 1 to ``annotators``, in order: (1, 2), (1, 3), ...

    Raises:
        ValueError: if there are fewer than two annotators.
    """
    if annotators < 2:
        raise ValueError(f"a pair of annotators needs at least 2 annotators, got {annotators}")
    return tuple(itertools.combinations(range(1, annotators + 1), 2))


def assign_pairs(
    pairs: Sequence[DrawnPair], annotators: int, *, seed: int
) -> tuple[Assignment, ...]:
    """Give each of ``pairs`` to one pair of annotators, as the module docstring describes, in
    the order dealt.

    Raises:
        ValueError: if there are fewer than two annotators, a pair id repeats, or a pair comes
            from a pool the draw does not have.
    """
    groups = annotator_pairs(annotators)
    identifiers = [pair.pair_id for pair in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("a pair id repeats among the pairs to assign")
    unknown = sorted({pair.pool for pair in pairs} - set(POOLS))
    if unknown:
        raise ValueError(f"pairs of unknown pools: {unknown}")
    dealt = []
    for pool in POOLS:
        sources = sorted({pair.source for pair in pairs if pair.pool == pool}, key=_source_key)
        for source in sources:
            members = [p.pair_id for p in pairs if p.pool == pool and p.source == source]
            key = f"{seed}:assignment:{pool}" + (f":{source}" if source else "")
            ordered = sorted(members)
            random.Random(key).shuffle(ordered)
            dealt += ordered
    return tuple(
        Assignment(
            pair_id=identifier,
            group=index % len(groups) + 1,
            annotators=groups[index % len(groups)],
        )
        for index, identifier in enumerate(dealt)
    )


def pairs_of(assignments: Iterable[Assignment], annotator: int) -> tuple[str, ...]:
    """Return the pair ids that ``annotator`` labels, in the order dealt."""
    return tuple(a.pair_id for a in assignments if annotator in a.annotators)


def _source_key(source: str | None) -> tuple[bool, str]:
    return (source is not None, source or "")
