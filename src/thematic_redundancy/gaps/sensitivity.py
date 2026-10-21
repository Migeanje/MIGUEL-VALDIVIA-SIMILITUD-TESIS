"""The sensitivity table of the gap run (T24a, D32 (1)): variants reported beside the declared
rules, never used to select a gap.

Each variant changes one thing against the declared rules and stability filter (``gaps`` of the
configuration), which the table carries as its baseline:

- ``primary``: the declared rules (a) to (d) and stability filter;
- ``literal_a``: the thesis plan's literal rule (a), the topics with fewer than
  :data:`LITERAL_SMALL_PERCENT` % of the corpus's theses, with no NPMI condition;
- ``literal_b``: the thesis plan's literal rule (b), the theses whose mean cosine distance to the
  run's topic centroids is above the run's :data:`LITERAL_DISTANCE_QUANTILE` quantile; it flags
  theses, not gaps, so its rows count theses and their Jaccard overlap with the reference run's;
- ``rare_max_share``: rule (c) with ``rare_max_share`` :data:`RARE_MAX_SHARE`;
- ``min_overlap``: the stability filter with ``stability_min_overlap`` :data:`MIN_OVERLAP`.

D32 gives these four values in its text, not in the configuration, so they are module constants.
The gaps of ``literal_a`` and ``rare_max_share`` go through the declared stability filter.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL
from thematic_redundancy.gaps.affinity import CareerAffinity
from thematic_redundancy.gaps.rules import (
    Gap,
    GapInputError,
    GapMap,
    GapSettings,
    SmallCoherentTopic,
    ThesisMetadata,
    _unit_rows,
    small_coherent_topics,
    transferable_topics,
)
from thematic_redundancy.gaps.stability import GapSupport, gap_support, jaccard

LITERAL_SMALL_PERCENT = 5
"""Literal rule (a): a topic is small when it holds fewer than this percent of the theses."""
LITERAL_DISTANCE_QUANTILE = 0.75
"""Literal rule (b): the quantile of the theses' mean distance above which a thesis is flagged."""
RARE_MAX_SHARE = 0.01
"""Rule (c)'s variant: the largest target share for a topic to be rare."""
MIN_OVERLAP = 0.75
"""The stability filter's variant: the Jaccard overlap at which two gaps match."""
VARIANTS = ("primary", "literal_a", "literal_b", "rare_max_share", "min_overlap")
RULES = ("a", "b", "c", "d")


class SensitivityRow(BaseModel):
    """One row of ``sensitivity.parquet``: one variant, one rule, one run."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    variant: str
    setting: str
    """The variant's value, as a reader would state it."""
    rule: str
    run_id: str
    seed: int
    reference: bool
    """Whether the run is the reference run, whose gaps the stability filter assesses."""
    gaps: int | None
    """The run's gaps of the rule; ``None`` for ``literal_b``, which flags theses."""
    theses: int
    """The theses of those gaps' topics, each counted once, or the theses ``literal_b`` flags."""
    noise_theses: int | None
    """For ``literal_b``, the flagged theses that are noise in the run."""
    stable: int | None
    """For the reference run, its gaps of the rule that the stability filter keeps."""
    jaccard_to_reference: float | None
    """For ``literal_b``, the Jaccard index of the run's flagged theses and the reference run's."""


@dataclass(frozen=True)
class DistantTheses:
    """The theses that the literal rule (b) flags in one run."""

    flagged: frozenset[str]
    noise: int
    """The flagged theses that are noise in the run."""
    threshold: float
    """The run's :data:`LITERAL_DISTANCE_QUANTILE` quantile of the mean distance."""


def literal_small_cut(theses: int) -> int:
    """Return the largest topic size below :data:`LITERAL_SMALL_PERCENT` % of ``theses``.

    The comparison is exact, in integers: at 744 theses 5 % is 37.2, so the cut is 37; at 740 it
    is exactly 37, which is not fewer, so the cut is 36.
    """
    return max((LITERAL_SMALL_PERCENT * theses - 1) // 100, 0)


def literal_small_topics(
    gap_map: GapMap, metadata: ThesisMetadata
) -> tuple[SmallCoherentTopic, ...]:
    """Return the literal rule (a)'s gaps: the topics with fewer than
    :data:`LITERAL_SMALL_PERCENT` % of the map's theses, whatever their NPMI.

    They are rule (a)'s gaps at :func:`literal_small_cut` with the NPMI floor at -1, NPMI's
    lowest value, so the floor is no condition; a topic whose NPMI is undefined is left out, as
    rule (a) leaves it out.
    """
    return small_coherent_topics(
        gap_map, metadata, size_cut=float(literal_small_cut(gap_map.labels.size)), npmi_floor=-1.0
    )


def literal_distant_theses(
    gap_map: GapMap, vectors: Any, metadata: ThesisMetadata
) -> DistantTheses:
    """Return the literal rule (b)'s theses: those, noise included, whose mean cosine distance to
    the map's topic centroids is strictly above the :data:`LITERAL_DISTANCE_QUANTILE` quantile of
    the theses' means (linear interpolation, ``numpy.quantile``'s default).

    Raises:
        GapInputError: if the map has no topic, or the vectors do not match it or the metadata.
    """
    if not gap_map.topics:
        raise GapInputError("the literal rule (b) needs at least one topic")
    if len(metadata.doc_codes) != gap_map.labels.size:
        raise GapInputError(
            f"the metadata describes {len(metadata.doc_codes)} theses, the map "
            f"{gap_map.labels.size}"
        )
    points = _unit_rows(vectors, gap_map)
    centroids = gap_map.centroids / np.linalg.norm(gap_map.centroids, axis=1, keepdims=True)
    mean = (1.0 - points @ centroids.T).mean(axis=1)
    threshold = float(np.quantile(mean, LITERAL_DISTANCE_QUANTILE))
    flagged = mean > threshold
    codes = np.asarray(metadata.doc_codes, dtype=object)
    return DistantTheses(
        flagged=frozenset(codes[flagged].tolist()),
        noise=int(np.count_nonzero(flagged & (gap_map.labels == NOISE_LABEL))),
        threshold=threshold,
    )


def sensitivity_rows(
    maps: Sequence[GapMap],
    vectors: Any,
    metadata: ThesisMetadata,
    affinity: CareerAffinity,
    settings: GapSettings,
    *,
    run_ids: Sequence[str],
    seeds: Sequence[int],
    reference: int,
    primary: Sequence[Sequence[Gap]],
) -> tuple[SensitivityRow, ...]:
    """Return the sensitivity table: per variant of :data:`VARIANTS`, per rule it touches and per
    run, in that order.

    Args:
        maps: the seed runs' maps, in the order of ``run_ids``.
        vectors: the original document vectors, in the maps' thesis order.
        metadata: the theses' metadata.
        affinity: the career-affinity matrix (U4).
        settings: the declared values.
        run_ids: the runs' ids.
        seeds: the runs' seeds.
        reference: the index of the reference run.
        primary: each run's gaps under the declared rules, rules (a) to (d).

    Raises:
        GapInputError: if the runs, seeds, maps and gaps differ in number, or a rule refuses its
            input.
    """
    if not len(maps) == len(run_ids) == len(seeds) == len(primary):
        raise GapInputError(
            f"{len(maps)} maps, {len(run_ids)} run ids, {len(seeds)} seeds and {len(primary)} "
            "runs of gaps were given"
        )
    table = _Table(run_ids, seeds, reference)

    def support(runs: Sequence[Sequence[Gap]], min_overlap: float) -> tuple[GapSupport, ...]:
        return gap_support(
            runs, reference=reference, min_runs=settings.stability_min_runs, min_overlap=min_overlap
        )

    declared = support(primary, settings.stability_min_overlap)
    for rule in RULES:
        table.gap_rows("primary", "declared values (gaps)", rule, primary, declared)
    literal = [literal_small_topics(gap_map, metadata) for gap_map in maps]
    cut = literal_small_cut(len(metadata.doc_codes))
    table.gap_rows(
        "literal_a",
        f"size below {LITERAL_SMALL_PERCENT} % of {len(metadata.doc_codes)} theses (at most "
        f"{cut}), no NPMI floor",
        "a",
        literal,
        support(literal, settings.stability_min_overlap),
    )
    distant = [literal_distant_theses(gap_map, vectors, metadata) for gap_map in maps]
    table.distant_rows(
        f"mean cosine distance to the centroids above the P{LITERAL_DISTANCE_QUANTILE * 100:g}",
        distant,
    )
    rare = [
        transferable_topics(
            gap_map,
            metadata,
            affinity,
            min_affinity=settings.min_affinity,
            present_min_count=settings.present_min_count,
            present_min_share=settings.present_min_share,
            rare_max_share=RARE_MAX_SHARE,
        )
        for gap_map in maps
    ]
    table.gap_rows(
        "rare_max_share",
        f"rare_max_share {RARE_MAX_SHARE:g} (declared {settings.rare_max_share:g})",
        "c",
        rare,
        support(rare, settings.stability_min_overlap),
    )
    strict = support(primary, MIN_OVERLAP)
    for rule in RULES:
        table.gap_rows(
            "min_overlap",
            f"stability_min_overlap {MIN_OVERLAP:g} (declared {settings.stability_min_overlap:g})",
            rule,
            primary,
            strict,
        )
    return tuple(table.rows)


class _Table:
    """The rows of :func:`sensitivity_rows`, as they are added."""

    def __init__(self, run_ids: Sequence[str], seeds: Sequence[int], reference: int) -> None:
        self.run_ids, self.seeds, self.reference = tuple(run_ids), tuple(seeds), reference
        self.rows: list[SensitivityRow] = []

    def gap_rows(
        self,
        variant: str,
        setting: str,
        rule: str,
        runs: Sequence[Sequence[Gap]],
        supports: Sequence[GapSupport],
    ) -> None:
        stable = sum(1 for item in supports if item.gap.rule == rule and item.stable)
        for index, run in enumerate(runs):
            gaps = [gap for gap in run if gap.rule == rule]
            theses = frozenset().union(*(topic for gap in gaps for topic in gap.topic_members))
            self._add(
                variant,
                setting,
                rule,
                index,
                gaps=len(gaps),
                theses=len(theses),
                stable=stable if index == self.reference else None,
            )

    def distant_rows(self, setting: str, runs: Sequence[DistantTheses]) -> None:
        reference = runs[self.reference].flagged
        for index, run in enumerate(runs):
            overlap = jaccard(run.flagged, reference) if run.flagged or reference else None
            self._add(
                "literal_b",
                setting,
                "b",
                index,
                theses=len(run.flagged),
                noise_theses=run.noise,
                jaccard_to_reference=overlap,
            )

    def _add(self, variant: str, setting: str, rule: str, index: int, **values: Any) -> None:
        self.rows.append(
            SensitivityRow(
                **{
                    "gaps": None,
                    "noise_theses": None,
                    "stable": None,
                    "jaccard_to_reference": None,
                    **values,
                },
                variant=variant,
                setting=setting,
                rule=rule,
                run_id=self.run_ids[index],
                seed=self.seeds[index],
                reference=index == self.reference,
            )
        )
