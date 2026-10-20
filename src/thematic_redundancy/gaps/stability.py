"""The cross-seed stability filter of the gap rules (T24, D16: present in at least 4 of 5 seeds).

Each seed has its own topic map (D30: T24's filter uses all five runs), and topic ids differ
from seed to seed, so gaps are matched by the theses involved, never by topic id:

- two gaps can match only when they come from the same rule and, for rule (c), have the same
  source and target programs (:attr:`match_key`);
- the overlap of two topics is the Jaccard index of their members' doc codes;
- the overlap of two gaps is that of their topic, or, for a rule (b) pair, the overlap of the
  pair's weaker topic, in whichever order of the two topics gives the higher value;
- a gap is present in a run when some gap of that run overlaps it by at least ``min_overlap``.

The filter reads the gaps of one reference run, the run whose map is shown (D30: the medoid
run), and keeps those present in at least ``min_runs`` runs, the reference included. A gap that
is absent from the reference run is not reported, since its topics are not on the shown map.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

from thematic_redundancy.gaps.rules import Gap, GapInputError


@dataclass(frozen=True)
class GapSupport:
    """A gap of the reference run and the runs in which it is present."""

    gap: Gap
    overlaps: tuple[float | None, ...]
    """For each run, the highest overlap of a gap of that run with :attr:`gap`; ``None`` when
    the run has no gap of the same rule and programs."""
    support: int
    """The runs whose overlap is at least :attr:`min_overlap`."""
    min_runs: int
    min_overlap: float
    stable: bool
    """Whether :attr:`support` is at least :attr:`min_runs`."""

    @property
    def reason(self) -> str:
        verdict = "kept" if self.stable else "dropped"
        return (
            f"rule ({self.gap.rule}) gap found in {self.support} of {len(self.overlaps)} runs "
            f"(at least {self.min_runs} needed, overlap at least {self.min_overlap:g}), "
            f"{verdict}: {self.gap.reason}"
        )


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Return the Jaccard index of two sets of doc codes.

    Raises:
        GapInputError: if both sets are empty.
    """
    union = len(first | second)
    if union == 0:
        raise GapInputError("the Jaccard index of two empty sets is undefined")
    return len(first & second) / union


def gap_overlap(first: Gap, second: Gap) -> float | None:
    """Return the overlap of two gaps (see the module), or ``None`` when they cannot match."""
    if first.rule != second.rule or first.match_key != second.match_key:
        return None
    reference, other = first.topic_members, second.topic_members
    return max(
        min(jaccard(left, right) for left, right in zip(reference, order, strict=True))
        for order in permutations(other)
    )


def gap_support(
    runs: Sequence[Sequence[Gap]], *, reference: int, min_runs: int, min_overlap: float
) -> tuple[GapSupport, ...]:
    """Return the support of each gap of ``runs[reference]``, in that run's order.

    Args:
        runs: the gaps of each seed's run, every rule together or one rule at a time.
        reference: the index of the run whose gaps are assessed.
        min_runs: the runs in which a gap must be present (``gaps.stability_min_runs``).
        min_overlap: the overlap at which two gaps match, in (0, 1].

    Raises:
        GapInputError: if there is no run, or an argument is out of range.
    """
    if not runs:
        raise GapInputError("at least one run is needed")
    if not 0 <= reference < len(runs):
        raise GapInputError(f"the reference run {reference} is not one of the {len(runs)} runs")
    if not 1 <= min_runs <= len(runs):
        raise GapInputError(f"min_runs must lie in [1, {len(runs)}], not {min_runs}")
    if not 0.0 < min_overlap <= 1.0:
        raise GapInputError(f"min_overlap must lie in (0, 1], not {min_overlap}")
    results = []
    for gap in runs[reference]:
        overlaps = tuple(_best_overlap(gap, run) for run in runs)
        support = sum(1 for value in overlaps if value is not None and value >= min_overlap)
        results.append(
            GapSupport(
                gap=gap,
                overlaps=overlaps,
                support=support,
                min_runs=min_runs,
                min_overlap=min_overlap,
                stable=support >= min_runs,
            )
        )
    return tuple(results)


def stable_gaps(
    runs: Sequence[Sequence[Gap]], *, reference: int, min_runs: int, min_overlap: float
) -> tuple[GapSupport, ...]:
    """Return the gaps of :func:`gap_support` that are stable, in the reference run's order."""
    supports = gap_support(runs, reference=reference, min_runs=min_runs, min_overlap=min_overlap)
    return tuple(result for result in supports if result.stable)


def _best_overlap(gap: Gap, run: Sequence[Gap]) -> float | None:
    values = [value for other in run if (value := gap_overlap(gap, other)) is not None]
    return max(values) if values else None
