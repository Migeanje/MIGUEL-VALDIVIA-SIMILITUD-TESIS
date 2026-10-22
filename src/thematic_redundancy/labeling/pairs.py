"""The labeled set's pair sampler (T20a, D34): the universe of pairs, the pools with their strata,
the pilot and the labeled set. Nothing here reads or writes a file.

- **Universe** (D34 (4)): every unordered pair of the included theses (D24). The theses are put
  in doc-code order, and the pairs ``(a, b)``, ``a`` before ``b``, in row-major order, so the
  universe does not depend on the order of any input. A pair's stratum (D34 (6)) is the program
  that both theses share, or :data:`BETWEEN` when they belong to two programs.
- **Scores and ranks**: each representation scores a pair by the cosine of its two document
  vectors (:func:`~thematic_redundancy.redundancy.search.pair_cosines`). A pair's percentile rank
  under a representation is its average rank among all the pairs in ascending score order (tied
  pairs share the mean of their ranks) over the number of pairs, so the highest pair has 1.0.
- **Top**: a representation's top is the ``ceil(top_share × pairs)`` pairs with the highest
  scores (D28 C10: by rank, not by a cosine cut-off); a tie at the cut goes to the earlier pair.
  The share is read as the decimal written in the configuration, so the count is exact.
- **Mid-range**: the pairs whose mean percentile rank over every representation is at least
  ``mid_low`` and below ``mid_high``, compared in exact arithmetic.
- **Draw** (D34 (5) and (7)), in this order, a pair drawn once never drawn again: the pilot, then
  the labeled set. The pilot spreads ``pilot_size`` pairs over the four pools, at least
  ``pilot_floor`` each and the rest by largest remainder of the pools' sizes, and SBERT-high's
  share over the models by largest remainder, so an odd share gives the earlier model one more
  pair; within a pool it is unstratified. The labeled set draws
  ``sbert_high_per_model`` pairs from each model's top, in the order of ``embedding.models``,
  then ``tfidf_high`` from the TF-IDF top, ``mid_range`` from the mid-range and ``random`` from
  the whole universe, unstratified. The high and mid-range pools of the labeled set spread their
  quota over the strata with :func:`allocate_with_floor`, by the candidates each stratum has
  left.
- **Seeds**: every random choice is ``random.Random(key).sample(candidates, quota)``, the
  candidates in universe order and ``key`` the string ``"<seed>:<phase>:<pool>"``, followed by
  ``":<model>"`` for SBERT-high and by ``":<stratum>"`` for a stratum of a stratified pool, as
  :func:`~thematic_redundancy.labeling.objectives_check.make_sample` derives its seeds; the same
  seed and universe give the same draw.
"""

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import rankdata

from thematic_redundancy.labeling.objectives_check import allocate
from thematic_redundancy.redundancy.search import Vectors, pair_cosines
from thematic_redundancy.shared.config import LabelingConfig

BETWEEN = "between"
"""The stratum of a pair whose theses belong to two programs."""
TFIDF = "tfidf"
"""The name of the TF-IDF baseline among the representations."""
PILOT = "pilot"
LABELED = "labeled"
PHASES = (PILOT, LABELED)
SBERT_HIGH = "sbert_high"
TFIDF_HIGH = "tfidf_high"
MID_RANGE = "mid_range"
RANDOM = "random"
POOLS = (SBERT_HIGH, TFIDF_HIGH, MID_RANGE, RANDOM)
"""The four pools of D34 (5), in the order they are drawn."""


class SamplerError(ValueError):
    """The universe cannot be built or the draw cannot be made as declared."""


@dataclass(frozen=True)
class DocumentVectors:
    """One representation's document vectors: one row per doc code, dense or sparse."""

    name: str
    doc_codes: tuple[str, ...]
    vectors: Vectors


@dataclass(frozen=True)
class PairUniverse:
    """Every pair of the theses, with its stratum and its score and rank under each
    representation; see the module docstring."""

    doc_codes: tuple[str, ...]
    """The theses, in doc-code order."""
    programs: tuple[str, ...]
    """The program of each thesis of :attr:`doc_codes`."""
    strata: tuple[str, ...]
    """The strata, in the order of the configured programs, then :data:`BETWEEN`."""
    first: np.ndarray
    """The row of each pair's first thesis in :attr:`doc_codes`."""
    second: np.ndarray
    stratum: np.ndarray
    """The index in :attr:`strata` of each pair's stratum."""
    models: tuple[str, ...]
    """The embedding models, in the configured order."""
    representations: tuple[str, ...]
    """:attr:`models`, then :data:`TFIDF`: the order of :attr:`scores` and :attr:`ranks`."""
    scores: tuple[np.ndarray, ...]
    """Each representation's cosine of every pair."""
    ranks: tuple[np.ndarray, ...]
    """Each representation's percentile rank of every pair."""

    @property
    def size(self) -> int:
        return int(self.first.size)

    def codes(self, pair: int) -> tuple[str, str]:
        """Return the doc codes of ``pair``, the earlier one first."""
        return self.doc_codes[int(self.first[pair])], self.doc_codes[int(self.second[pair])]

    def stratum_of(self, pair: int) -> str:
        return self.strata[int(self.stratum[pair])]

    def mean_ranks(self) -> np.ndarray:
        """Return each pair's mean percentile rank over the representations."""
        return np.mean(np.vstack(self.ranks), axis=0)

    def counts(self, pairs: Sequence[int] | np.ndarray) -> dict[str, int]:
        """Return how many of ``pairs`` fall in each stratum, every stratum listed."""
        chosen = self.stratum[np.asarray(pairs, dtype=np.int64)]
        tally = np.bincount(chosen, minlength=len(self.strata))
        return {name: int(count) for name, count in zip(self.strata, tally, strict=True)}


@dataclass(frozen=True)
class Step:
    """One pool's draw within a phase."""

    phase: str
    pool: str
    source: str | None
    """The embedding model of an SBERT-high step; ``None`` for the other pools."""
    quota: int
    stratified: bool
    candidates: dict[str, int]
    """The pool's candidates in each stratum that no earlier step had drawn."""
    quotas: dict[str, int] | None
    """The quota of each stratum, for a stratified step."""
    pairs: tuple[int, ...]
    """The pairs drawn, in draw order."""


@dataclass(frozen=True)
class LabeledSetDraw:
    """The pilot and the labeled set drawn from a universe, with the pools they came from."""

    universe: PairUniverse
    seed: int
    top_count: int
    tops: tuple[np.ndarray, ...]
    """Each representation's top, best first, in the order of the universe's representations."""
    mid_range: np.ndarray
    """The mid-range's pairs, in universe order."""
    steps: tuple[Step, ...]

    @property
    def pairs(self) -> tuple[int, ...]:
        """Every drawn pair, in draw order."""
        return tuple(pair for step in self.steps for pair in step.pairs)


# Ranks, the top and the mid-range


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Return each score's average rank in ascending order (tied scores share the mean of their
    ranks) over the number of scores, so the highest score has 1.0."""
    return _twice_ranks(scores) / (2.0 * len(scores))


def top_count(pairs: int, share: float) -> int:
    """Return ``ceil(share × pairs)``, with ``share`` read as the decimal it is written as, so
    that 0.07 of 100 pairs is 7, not the 8 of binary floating point."""
    return math.ceil(_decimal(share) * pairs)


def top_pairs(scores: np.ndarray, count: int) -> np.ndarray:
    """Return the ``count`` pairs with the highest scores, best first; of pairs with the same
    score, the earlier pair comes first."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=np.float64)))
    return order[:count]


def mid_range_mask(scores: Sequence[np.ndarray], *, low: float, high: float) -> np.ndarray:
    """Return whether each pair's mean percentile rank over ``scores``, one array per
    representation, is at least ``low`` and below ``high``.

    The comparison is exact: twice a pair's average rank is a whole number, so the sum of those
    over the representations is compared with the bounds read as decimals, and a pair whose
    mean is exactly ``low`` is in the band, one exactly at ``high`` out of it."""
    count, pairs = len(scores), len(scores[0])
    total = np.sum([_twice_ranks(values) for values in scores], axis=0)
    scale = 2 * count * pairs
    lower, upper = _decimal(low), _decimal(high)
    return (total * lower.denominator >= lower.numerator * scale) & (
        total * upper.denominator < upper.numerator * scale
    )


def _twice_ranks(scores: np.ndarray) -> np.ndarray:
    """Return twice each score's average ascending rank, a whole number."""
    return np.rint(2.0 * rankdata(np.asarray(scores, dtype=np.float64), method="average")).astype(
        np.int64
    )


def _decimal(value: float) -> Fraction:
    """Return ``value`` as the decimal of its shortest text, such as 9/10 for 0.9."""
    return Fraction(repr(float(value)))


# Allocation


def allocate_with_floor[K](sizes: Mapping[K, int], total: int, floor: int) -> dict[K, int]:
    """Split ``total`` over the keys of ``sizes`` (D34 (6) and (7)): each key with a positive size
    first gets ``floor``, or all it holds when that is less; the rest is then spread in
    proportion to what each key holds beyond that, by largest remainder (ties go to the earlier
    key) with :func:`~thematic_redundancy.labeling.objectives_check.allocate`. A key without
    candidates gets nothing, and one with fewer than its share gives all it holds, the rest
    going to the others. Sizes that hold no more than ``total`` give all they hold.

    Raises:
        ValueError: if a size or ``floor`` is negative, or the floors exceed ``total``.
    """
    if floor < 0 or any(size < 0 for size in sizes.values()):
        raise ValueError(f"sizes and the floor must not be negative: {dict(sizes)}, {floor}")
    if sum(sizes.values()) <= total:
        return dict(sizes)
    floors = {key: min(floor, size) for key, size in sizes.items()}
    rest = total - sum(floors.values())
    if rest < 0:
        raise ValueError(
            f"the floor of {floor} for each of {sum(1 for s in sizes.values() if s)} keys exceeds "
            f"the quota of {total}"
        )
    extra = allocate({key: size - floors[key] for key, size in sizes.items()}, rest)
    quotas = {key: floors[key] + extra[key] for key in sizes}
    if any(quotas[key] > size for key, size in sizes.items()):
        raise ValueError(f"the allocation {quotas} exceeds the sizes {dict(sizes)}")
    return quotas


def pilot_allocation(
    settings: LabelingConfig, models: Sequence[str]
) -> tuple[tuple[str, str | None, int], ...]:
    """Return the pilot's ``(pool, model, quota)`` in draw order: ``pilot_size`` pairs over the
    four pools by :func:`allocate_with_floor` of the pools' sizes with ``pilot_floor``, and
    SBERT-high's share over ``models`` by largest remainder of their equal sizes, so an odd
    share gives the earlier model one more pair (D34 (7))."""
    sizes = {
        SBERT_HIGH: settings.sbert_high_per_model * len(models),
        TFIDF_HIGH: settings.tfidf_high,
        MID_RANGE: settings.mid_range,
        RANDOM: settings.random,
    }
    quotas = allocate_with_floor(sizes, settings.pilot_size, settings.pilot_floor)
    per_model = allocate(dict.fromkeys(models, settings.sbert_high_per_model), quotas[SBERT_HIGH])
    return (
        *((SBERT_HIGH, model, per_model[model]) for model in models),
        (TFIDF_HIGH, None, quotas[TFIDF_HIGH]),
        (MID_RANGE, None, quotas[MID_RANGE]),
        (RANDOM, None, quotas[RANDOM]),
    )


# The universe


def build_universe(
    theses: Sequence[tuple[str, str]],
    semantic: Sequence[DocumentVectors],
    tfidf: DocumentVectors,
    *,
    programs: Sequence[str],
) -> PairUniverse:
    """Return the universe of ``theses``, each a ``(doc_code, program)``, scored and ranked under
    each of ``semantic`` (the embedding models, in the configured order) and ``tfidf``.

    Each representation's rows are matched to the theses by doc code, so no input order matters.

    Raises:
        SamplerError: if the theses repeat a doc code, are fewer than two, or have a program
            that is not one of ``programs``, or a representation's doc codes are not exactly the
            theses'.
        ValueError: if a document vector has no direction.
    """
    by_code = dict(theses)
    if len(by_code) != len(theses):
        raise SamplerError(f"the theses repeat {len(theses) - len(by_code)} doc codes")
    if len(by_code) < 2:
        raise SamplerError("the universe needs at least two theses")
    unknown = sorted({program for program in by_code.values() if program not in programs})
    if unknown:
        raise SamplerError(
            f"the program '{unknown[0]}' of a thesis is not one of the configured programs "
            f"{list(programs)}"
        )
    doc_codes = tuple(sorted(by_code))
    representations = (*semantic, tfidf)
    names = tuple(representation.name for representation in representations)
    if len(set(names)) != len(names):
        raise SamplerError(f"the representations repeat a name: {list(names)}")
    first, second = np.triu_indices(len(doc_codes), k=1)
    first, second = first.astype(np.int64), second.astype(np.int64)
    scores = tuple(
        _pair_scores(_aligned(representation, doc_codes), first, second)
        for representation in representations
    )
    strata = (*programs, BETWEEN)
    of_thesis = np.array([programs.index(by_code[code]) for code in doc_codes], dtype=np.int64)
    same = of_thesis[first] == of_thesis[second]
    return PairUniverse(
        doc_codes=doc_codes,
        programs=tuple(by_code[code] for code in doc_codes),
        strata=strata,
        first=first,
        second=second,
        stratum=np.where(same, of_thesis[first], len(programs)).astype(np.int64),
        models=tuple(representation.name for representation in semantic),
        representations=names,
        scores=scores,
        ranks=tuple(percentile_ranks(values) for values in scores),
    )


_PAIR_BATCH = 16_384
"""Pairs scored at once: two batches of 768-dimensional float64 rows take about 200 MB."""


def _aligned(representation: DocumentVectors, doc_codes: Sequence[str]) -> Vectors:
    """Return ``representation``'s rows in the order of ``doc_codes``."""
    rows = {code: row for row, code in enumerate(representation.doc_codes)}
    if len(rows) != len(representation.doc_codes) or rows.keys() != set(doc_codes):
        raise SamplerError(
            f"the theses of {representation.name} differ from the included theses: "
            f"{len(representation.doc_codes)} rows, {len(set(rows) & set(doc_codes))} of the "
            f"{len(doc_codes)} theses"
        )
    return representation.vectors[[rows[code] for code in doc_codes]]


def _pair_scores(vectors: Vectors, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the cosine of every pair, scored in batches by
    :func:`~thematic_redundancy.redundancy.search.pair_cosines`."""
    scores = np.empty(first.size, dtype=np.float64)
    for start in range(0, first.size, _PAIR_BATCH):
        stop = start + _PAIR_BATCH
        scores[start:stop] = pair_cosines(
            vectors, np.column_stack((first[start:stop], second[start:stop]))
        )
    return scores


# The draw


def draw_labeled_set(universe: PairUniverse, settings: LabelingConfig) -> LabeledSetDraw:
    """Draw the pilot and the labeled set from ``universe``, as the module docstring tells.

    Raises:
        SamplerError: if a pool has fewer pairs left than its quota.
    """
    count = top_count(universe.size, settings.top_share)
    tops = tuple(top_pairs(scores, count) for scores in universe.scores)
    mid = np.flatnonzero(
        mid_range_mask(universe.scores, low=settings.mid_low, high=settings.mid_high)
    )
    tfidf = universe.representations.index(TFIDF)
    candidates = {
        **{
            (SBERT_HIGH, model): np.sort(tops[universe.representations.index(model)])
            for model in universe.models
        },
        (TFIDF_HIGH, None): np.sort(tops[tfidf]),
        (MID_RANGE, None): mid,
        (RANDOM, None): np.arange(universe.size, dtype=np.int64),
    }
    labeled = (
        *((SBERT_HIGH, model, settings.sbert_high_per_model) for model in universe.models),
        (TFIDF_HIGH, None, settings.tfidf_high),
        (MID_RANGE, None, settings.mid_range),
        (RANDOM, None, settings.random),
    )
    taken = np.zeros(universe.size, dtype=bool)
    steps = []
    for phase, plan in ((PILOT, pilot_allocation(settings, universe.models)), (LABELED, labeled)):
        for pool, model, quota in plan:
            step = _draw_step(
                universe,
                phase=phase,
                pool=pool,
                source=model,
                pairs=candidates[(pool, model)],
                taken=taken,
                quota=quota,
                floor=settings.stratum_floor if phase == LABELED and pool != RANDOM else None,
                seed=settings.seed,
            )
            taken[list(step.pairs)] = True
            steps.append(step)
    return LabeledSetDraw(
        universe=universe,
        seed=settings.seed,
        top_count=count,
        tops=tops,
        mid_range=mid,
        steps=tuple(steps),
    )


def _draw_step(
    universe: PairUniverse,
    *,
    phase: str,
    pool: str,
    source: str | None,
    pairs: np.ndarray,
    taken: np.ndarray,
    quota: int,
    floor: int | None,
    seed: int,
) -> Step:
    """Draw ``quota`` of ``pairs`` that are not ``taken``: over the strata with ``floor`` when it
    is given, unstratified otherwise."""
    left = pairs[~taken[pairs]]
    available = universe.counts(left)
    if left.size < quota:
        raise SamplerError(
            f"the {phase} pool {pool}{f' of {source}' if source else ''} has {left.size} pairs "
            f"left, fewer than its quota of {quota}"
        )
    key = f"{seed}:{phase}:{pool}" + (f":{source}" if source else "")
    if floor is None:
        drawn = random.Random(key).sample(left.tolist(), quota)
        quotas = None
    else:
        quotas = allocate_with_floor(available, quota, floor)
        drawn = []
        for index, stratum in enumerate(universe.strata):
            own = left[universe.stratum[left] == index].tolist()
            drawn += random.Random(f"{key}:{stratum}").sample(own, quotas[stratum])
    return Step(
        phase=phase,
        pool=pool,
        source=source,
        quota=quota,
        stratified=floor is not None,
        candidates=available,
        quotas=quotas,
        pairs=tuple(int(pair) for pair in drawn),
    )


def pair_id(phase: str, first: str, second: str) -> str:
    """Return the id of the pair of the doc codes ``first`` and ``second``, in either order,
    drawn in ``phase``: ``P-`` for the pilot or ``L-`` for the labeled set, then the first 8
    hex digits of the SHA-256 of the two codes in order, joined by ``:``. It tells neither the
    pool nor the draw order."""
    low, high = sorted((first, second))
    prefix = {PILOT: "P", LABELED: "L"}[phase]
    return f"{prefix}-{hashlib.sha256(f'{low}:{high}'.encode()).hexdigest()[:8]}"
