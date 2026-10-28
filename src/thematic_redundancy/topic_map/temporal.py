"""The temporal split of the topic map (T27, O04): the winner's map fitted on the earlier
period, the later period's theses assigned to it, and the sample that experts rate.

O04 measures success on the temporal split by "assignment consistency between the 2021–2024 map
and the full map, plus a sample rated by experts". The pieces, each reusing T16's, T19's and
T23's code:

- **Periods** (:func:`split_periods`): a thesis is of the earlier period when its issue year is
  from ``first_year`` to ``split_year - 1``, and of the later one when it is from ``split_year``
  to ``last_year``; any other year is refused.
- **The earlier map** (:func:`fit_temporal_map`): the winner's configuration on the earlier
  theses alone, as T19 ran it and with no new sweep (D30 E3: "T27 reuses the winner's
  configuration and hyperparameters on 2021–2024 with no new sweep"): one unsupervised UMAP fit
  with the run's seed (:func:`~.reduction.fit_reduction`), the run's clustering of that reduction
  (:func:`~.build.cluster`), the c-TF-IDF terms of each topic (:func:`~.ctfidf.ctfidf_topics`)
  over the earlier theses' vocabulary (:func:`period_vocabulary`, T16 note 4), and the
  centroids, each the mean of its members' original vectors (:func:`~.build.topic_centroids`,
  D30 E5). O06's threshold is taken on this map (:func:`~.assignment.atypical_threshold`).
- **Assignment** (:func:`assign_period`): each later thesis goes to the topic whose centroid is
  nearest by cosine in the original space (D14, :func:`~.assignment.assign_proposals`) and is
  flagged atypical when its similarity is strictly below the threshold
  (:func:`~.assignment.check_atypical`). Nothing is re-clustered.
- **The expert sample** (:func:`draw_expert_sample`): a seeded draw of later theses, from two
  pools, the atypical ones first (:data:`SAMPLE_POOLS`), each pool's quota spread over the
  programs in proportion to its theses with a floor per program (T20a's
  :func:`~thematic_redundancy.labeling.pairs.allocate_with_floor`), and a seeded presentation
  order, so no rater can tell a pool by the position.

Nothing here reads or writes a file or holds a thesis text. An invalid input raises
:class:`~thematic_redundancy.topic_map.TopicMapError`.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from thematic_redundancy.evaluation.stability import NoiseHandling
from thematic_redundancy.labeling.pairs import allocate_with_floor
from thematic_redundancy.representation.tfidf import section_terms
from thematic_redundancy.shared.config import AppConfig
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.assignment import (
    ATYPICAL_PERCENTILE,
    AtypicalCheck,
    AtypicalThreshold,
    PercentileMethod,
    ThresholdScope,
    TopicAssignment,
    TopicCentroids,
    assign_proposals,
    atypical_threshold,
    check_atypical,
)
from thematic_redundancy.topic_map.build import (
    TopicMapPorts,
    TopicMapSettings,
    cluster,
    topic_centroids,
)
from thematic_redundancy.topic_map.clustering import ClusteringRun
from thematic_redundancy.topic_map.ctfidf import Topic, ctfidf_topics
from thematic_redundancy.topic_map.reduction import dense_vectors, fit_reduction

ATYPICAL_POOL = "atypical"
TYPICAL_POOL = "typical"
SAMPLE_POOLS = (ATYPICAL_POOL, TYPICAL_POOL)
"""The expert sample's pools, in the order they are drawn."""


@dataclass(frozen=True)
class TemporalSettings:
    """Everything the temporal split takes from the configuration, besides the topic map's."""

    first_year: int
    split_year: int
    last_year: int
    percentile: float
    """O06's percentile, :data:`~.assignment.ATYPICAL_PERCENTILE`."""
    method: PercentileMethod
    scope: ThresholdScope
    noise_handling: NoiseHandling
    """D30 E4's noise handling of the ARI, the primary one here too."""
    noise_handling_sensitivity: NoiseHandling
    programs: tuple[str, ...]
    """The configured programs, in order: the sample's strata."""
    sample_seed: int
    sample_size: int
    sample_atypical: int
    sample_program_floor: int

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        """Gather ``snapshot``'s years and programs, ``temporal``, ``proposal`` and the ARI noise
        handlings of ``experiment.ari``."""
        sample = config.temporal.sample
        return cls(
            first_year=config.snapshot.year_start,
            split_year=config.temporal.split_year,
            last_year=config.snapshot.year_end,
            percentile=ATYPICAL_PERCENTILE,
            method=config.proposal.atypical_method,
            scope=config.proposal.atypical_scope,
            noise_handling=config.experiment.ari.noise_handling,
            noise_handling_sensitivity=config.experiment.ari.sensitivity,
            programs=tuple(program.key for program in config.snapshot.programs),
            sample_seed=sample.seed,
            sample_size=sample.size,
            sample_atypical=sample.atypical,
            sample_program_floor=sample.program_floor,
        )

    @property
    def noise_handlings(self) -> tuple[NoiseHandling, NoiseHandling]:
        """The primary noise handling, then the sensitivity one."""
        return (self.noise_handling, self.noise_handling_sensitivity)

    def record(self) -> dict[str, int | float | str]:
        """Return the values as the manifest records them."""
        return {
            "first_year": self.first_year,
            "split_year": self.split_year,
            "last_year": self.last_year,
            "atypical_percentile": self.percentile,
            "atypical_scope": self.scope,
            "atypical_method": self.method,
            "noise_handling": self.noise_handling,
            "noise_handling_sensitivity": self.noise_handling_sensitivity,
            "sample_seed": self.sample_seed,
            "sample_size": self.sample_size,
            "sample_atypical": self.sample_atypical,
            "sample_program_floor": self.sample_program_floor,
        }


@dataclass(frozen=True)
class Periods:
    """The rows of the theses of each period, ascending and read-only."""

    first_year: int
    split_year: int
    last_year: int
    fitted: np.ndarray
    """The earlier period's theses: the map is fitted on them."""
    assigned: np.ndarray
    """The later period's theses: they are assigned to the map."""

    @property
    def fitted_years(self) -> tuple[int, int]:
        return (self.first_year, self.split_year - 1)

    @property
    def assigned_years(self) -> tuple[int, int]:
        return (self.split_year, self.last_year)


@dataclass(frozen=True)
class TemporalMap:
    """The winner's map of the earlier period, for one seed."""

    seed: int
    clustering: ClusteringRun
    labels: np.ndarray
    """One label per earlier thesis, in the order given, -1 for noise; read-only."""
    topics: tuple[Topic, ...]
    """One per topic, by ascending label, with its top c-TF-IDF terms."""
    centroids: TopicCentroids
    """The topics and their centroids in the original space (D14, D30 E5)."""
    threshold: AtypicalThreshold
    """O06's threshold of this map."""
    vocabulary: int
    """The terms of the c-TF-IDF fit: the earlier theses' share of T15's vocabulary."""
    seconds: dict[str, float]
    """The time of each step: ``umap``, ``cluster``, ``ctfidf`` and ``threshold``."""

    @property
    def noise_points(self) -> int:
        return int(np.count_nonzero(self.labels == -1))


@dataclass(frozen=True)
class PeriodAssignment:
    """The later theses' topics on the earlier map, with their atypical flags, in order."""

    assignments: tuple[TopicAssignment, ...]
    checks: tuple[AtypicalCheck, ...]

    @property
    def labels(self) -> np.ndarray:
        return np.array([assignment.topic for assignment in self.assignments], dtype=np.int64)

    @property
    def atypical(self) -> np.ndarray:
        return np.array([check.atypical for check in self.checks], dtype=bool)

    @property
    def atypical_count(self) -> int:
        return sum(check.atypical for check in self.checks)


@dataclass(frozen=True)
class SampleStratum:
    """One pool and program of the expert sample: its theses and the quota drawn from them."""

    pool: str
    program: str
    candidates: int
    quota: int


@dataclass(frozen=True)
class ExpertSample:
    """The later theses drawn for the experts, in draw order."""

    rows: tuple[int, ...]
    """Each drawn thesis's index in the theses given."""
    pools: tuple[str, ...]
    """Each drawn thesis's pool."""
    positions: tuple[int, ...]
    """Each drawn thesis's place, from 1, in the order the experts see the sample."""
    strata: tuple[SampleStratum, ...]
    """Every pool and program, in draw order, with or without theses."""


def split_periods(
    years: Sequence[int] | np.ndarray, *, first_year: int, split_year: int, last_year: int
) -> Periods:
    """Return the rows of the earlier period (``first_year`` to ``split_year - 1``) and of the
    later one (``split_year`` to ``last_year``).

    Raises:
        TopicMapError: if the years are not integers, ``split_year`` leaves a period without
            years, a thesis was issued outside both periods, or a period has no thesis.
    """
    if not first_year < split_year <= last_year:
        raise TopicMapError(
            f"the split year {split_year} must be after {first_year} and not after {last_year}"
        )
    values = np.asarray(years)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise TopicMapError("the issue years must be a 1-D sequence of integers")
    outside = int(np.count_nonzero((values < first_year) | (values > last_year)))
    if outside:
        verb = "thesis was" if outside == 1 else "theses were"
        raise TopicMapError(f"{outside} {verb} issued outside {first_year} to {last_year}")
    fitted = np.flatnonzero(values < split_year)
    assigned = np.flatnonzero(values >= split_year)
    for rows, (start, end) in (
        (fitted, (first_year, split_year - 1)),
        (assigned, (split_year, last_year)),
    ):
        if rows.size == 0:
            raise TopicMapError(f"there is no thesis of {start} to {end}")
        rows.setflags(write=False)
    return Periods(first_year, split_year, last_year, fitted, assigned)


def period_vocabulary(
    vocabulary: Sequence[str], documents: Sequence[Sequence[Sequence[str]]], *, max_ngram: int
) -> tuple[str, ...]:
    """Return the terms of ``vocabulary`` that ``documents`` form, in the vocabulary's order.

    T15's vocabulary is fitted on every thesis, so some of its terms occur only in the later
    theses, and :func:`~.ctfidf.ctfidf_topics` refuses a term that the theses it fits do not
    form (T16 note 4). Terms are formed inside one section, as T15 forms them.
    """
    formed = {
        term
        for document in documents
        for section in document
        for term in section_terms(list(section), max_ngram)
    }
    return tuple(term for term in vocabulary if term in formed)


def fit_temporal_map(
    vectors: Any,
    documents: Sequence[Sequence[Sequence[str]]],
    *,
    vocabulary: Sequence[str],
    clustering: ClusteringRun,
    seed: int,
    settings: TopicMapSettings,
    ports: TopicMapPorts | None = None,
    percentile: float,
    method: PercentileMethod,
    scope: ThresholdScope,
    timer: Callable[[], float] = time.perf_counter,
) -> TemporalMap:
    """Return the map of ``clustering`` on the earlier theses' ``vectors`` for ``seed``.

    Args:
        vectors: the earlier theses' original document vectors, one row each.
        documents: their full-clean section tokens, in the same order.
        vocabulary: :func:`period_vocabulary` of T15's terms over ``documents``.
        clustering: the winner's run, its algorithm and its K or ``min_cluster_size``.
        seed: the seed of UMAP and of K-means.
        settings: the configured settings of the topic map.
        ports: the libraries; the real ones by default.
        percentile: O06's percentile, :data:`~.assignment.ATYPICAL_PERCENTILE`.
        method: the percentile method of the threshold.
        scope: the scope of the threshold.
        timer: the clock of the steps' times.

    Raises:
        TopicMapError: if an input is invalid, a library breaks its contract, or the map has no
            topic.
    """
    ports = ports or TopicMapPorts()
    matrix = dense_vectors(vectors)
    seconds: dict[str, float] = {}
    started = timer()
    reduction = fit_reduction(matrix, settings.umap, seed, umap_factory=ports.umap)
    seconds["umap"] = timer() - started
    started = timer()
    labels = cluster(reduction.embedding, clustering, seed=seed, settings=settings, ports=ports)
    seconds["cluster"] = timer() - started
    started = timer()
    topic_set = ctfidf_topics(
        labels,
        documents,
        vocabulary=vocabulary,
        max_ngram=settings.max_ngram,
        top_n=settings.top_n,
        include_noise_class=settings.include_noise_class,
        backend=ports.ctfidf,
    )
    seconds["ctfidf"] = timer() - started
    topics = tuple(topic.topic for topic in topic_set.topics)
    if not topics:
        raise TopicMapError("the earlier period's map has no topic: every thesis is noise")
    started = timer()
    centroids = TopicCentroids(topics=topics, centroids=topic_centroids(matrix, labels, topics))
    threshold = atypical_threshold(
        matrix, labels, centroids, percentile=percentile, method=method, scope=scope
    )
    seconds["threshold"] = timer() - started
    return TemporalMap(
        seed=seed,
        clustering=clustering,
        labels=labels,
        topics=topic_set.topics,
        centroids=centroids,
        threshold=threshold,
        vocabulary=len(vocabulary),
        seconds=seconds,
    )


def assign_period(vectors: Any, temporal_map: TemporalMap) -> PeriodAssignment:
    """Return each later thesis's nearest topic on ``temporal_map`` and its atypical flag.

    Raises:
        TopicMapError: if ``vectors`` is not a matrix of finite, non-zero rows of the map's
            dimensions.
    """
    assignments = assign_proposals(vectors, temporal_map.centroids)
    return PeriodAssignment(
        assignments=assignments,
        checks=tuple(check_atypical(item, temporal_map.threshold) for item in assignments),
    )


def draw_expert_sample(
    programs: Sequence[str],
    atypical: Sequence[bool] | np.ndarray,
    *,
    program_order: Sequence[str],
    size: int,
    atypical_quota: int,
    program_floor: int,
    seed: int,
) -> ExpertSample:
    """Draw ``size`` of the theses whose ``programs`` and ``atypical`` flags are given.

    The atypical pool gives ``atypical_quota`` theses, or all it holds when that is less, and
    the typical pool the rest. Each pool's quota is spread over ``program_order`` by
    :func:`~thematic_redundancy.labeling.pairs.allocate_with_floor` with ``program_floor``, and
    each stratum's theses are drawn uniformly without replacement, pools and programs in that
    order, from one generator seeded with ``seed``, which then gives the presentation order.

    Raises:
        TopicMapError: if the lengths differ, a program is not in ``program_order``, ``size``
            is below 1, the atypical quota exceeds it, or the floors exceed a pool's quota.
    """
    flags = np.asarray(atypical, dtype=bool)
    if flags.ndim != 1 or flags.size != len(programs):
        raise TopicMapError(f"{flags.size} atypical flags were given for {len(programs)} theses")
    unknown = sorted(set(programs) - set(program_order))
    if unknown:
        raise TopicMapError(f"programs outside the configured ones: {unknown}")
    if size < 1:
        raise TopicMapError(f"the sample size must be at least 1, not {size}")
    if not 0 <= atypical_quota <= size:
        raise TopicMapError(f"the atypical quota {atypical_quota} exceeds the size {size}")
    keys = np.asarray(programs, dtype=object)
    pools = {ATYPICAL_POOL: np.flatnonzero(flags), TYPICAL_POOL: np.flatnonzero(~flags)}
    taken = min(atypical_quota, pools[ATYPICAL_POOL].size)
    quotas = {ATYPICAL_POOL: taken, TYPICAL_POOL: min(size - taken, pools[TYPICAL_POOL].size)}
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    drawn_pools: list[str] = []
    strata: list[SampleStratum] = []
    for pool in SAMPLE_POOLS:
        candidates = {
            program: pools[pool][keys[pools[pool]] == program] for program in program_order
        }
        sizes = {program: int(found.size) for program, found in candidates.items()}
        try:
            allocation = (
                allocate_with_floor(sizes, quotas[pool], program_floor)
                if quotas[pool]
                else dict.fromkeys(program_order, 0)
            )
        except ValueError as error:
            raise TopicMapError(f"the {pool} pool: {error}") from error
        for program in program_order:
            quota = allocation[program]
            strata.append(SampleStratum(pool, program, sizes[program], quota))
            if quota:
                chosen = rng.choice(candidates[program], size=quota, replace=False)
                rows += [int(row) for row in chosen]
                drawn_pools += [pool] * quota
    positions = rng.permutation(len(rows)) + 1
    return ExpertSample(
        rows=tuple(rows),
        pools=tuple(drawn_pools),
        positions=tuple(int(position) for position in positions),
        strata=tuple(strata),
    )
