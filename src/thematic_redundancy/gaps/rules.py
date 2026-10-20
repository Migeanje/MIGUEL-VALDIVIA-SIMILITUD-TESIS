"""Gap rules (a) to (d) of thesis plan section 3.6.6 over one topic map (T24, D16).

Each rule is a pure function of one seed's topic map (:class:`GapMap`), the theses' metadata
(:class:`ThesisMetadata`) and, for rule (c), the career-affinity matrix of U4. It returns typed
gaps, ordered, each with the evidence counts that made it fire and a ``reason``; no gap holds a
thesis text, only doc codes, program keys, years and numbers. The cross-seed stability filter is
in :mod:`.stability`; experts are the final filter (D16).

The plan declares (initial configuration, ``gaps``):

- (a) small coherent topics: "bottom quartile of topic size, plus an NPMI floor";
- (b) interstitial regions: "pairs among each topic's 3 nearest topics with ≤1 bridge
  document";
- (c) transferable topics: "per ``config/career_affinity.yaml``" (U4), "temáticas presentes en
  una carrera y ausentes en una carrera afín donde resultarían transferibles" (D29), with
  Electrónica never a target (D29) and the rules normalized per program (R5);
- (d) temporal decline: "change in share between 2021–2023 and 2024–2026, minimum count 3".

D16 speaks of five gap types, while the plan lists only these four; no fifth rule is built here.
The OpenAlex "never touched" reference of D17 belongs to T25.

Every value the plan does not declare is a keyword argument with no default, so the caller
states it: the NPMI floor (a); the affinity that counts as affine and what "present" and
"absent or rare" mean (c); the decline that counts (d). The definitions the plan leaves open
are fixed here and documented on each rule:

- topic size: a topic's theses; the noise (-1) is no topic and is left out of the sizes;
- the bottom quartile: sizes at or below the first quartile of the map's topic sizes, by
  linear interpolation (``numpy.quantile``'s default), so every topic tied at it is included;
- a bridge thesis (b): see :func:`interstitial_pairs`;
- a program's share of a topic (c): the topic's theses of that program over all the program's
  theses in the map, noise included;
- a window's share of a topic (d): the topic's theses of that window over all the window's
  theses, noise included.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array
from thematic_redundancy.gaps.affinity import CareerAffinity

if TYPE_CHECKING:
    from thematic_redundancy.evaluation.coherence import CoherenceResult
    from thematic_redundancy.shared.config import AppConfig
    from thematic_redundancy.topic_map.build import TopicMap

SIZE_QUANTILE = 0.25
"""The bottom quartile of rule (a): sizes at or below this quantile of the topic sizes."""


class GapInputError(ValueError):
    """An input of a gap rule breaks its contract, so no gap was computed."""


# --- Inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class GapSettings:
    """The gap parameters that the configuration declares (``gaps`` and ``snapshot``)."""

    nearest_topics: int
    bridge_max_docs: int
    temporal_first_year: int
    """First year of the earlier window: ``snapshot.year_start``."""
    temporal_split_year: int
    """First year of the later window."""
    temporal_last_year: int
    """Last year of the later window: ``snapshot.year_end``."""
    temporal_min_count: int
    stability_min_runs: int

    @classmethod
    def from_config(cls, config: "AppConfig") -> Self:
        gaps = config.gaps
        return cls(
            nearest_topics=gaps.nearest_topics,
            bridge_max_docs=gaps.bridge_max_docs,
            temporal_first_year=config.snapshot.year_start,
            temporal_split_year=gaps.temporal_split_year,
            temporal_last_year=config.snapshot.year_end,
            temporal_min_count=gaps.temporal_min_count,
            stability_min_runs=gaps.stability_min_runs,
        )


@dataclass(frozen=True, eq=False)
class GapMap:
    """What the gap rules read from one seed's topic map; the arrays are read-only copies.

    Two maps compare by identity, as their arrays have no single truth value."""

    labels: np.ndarray
    """One label per thesis, -1 for noise."""
    topics: tuple[int, ...]
    """The topic labels, ascending: exactly the labels present other than the noise."""
    centroids: np.ndarray
    """``topics × dimensions``, row ``i`` for ``topics[i]``, in the original vector space."""
    npmi: tuple[float | None, ...]
    """Each topic's NPMI, in topic order; ``None`` when it is undefined."""

    def __post_init__(self) -> None:
        labels = label_array(self.labels)
        present = tuple(sorted({int(label) for label in labels if label != NOISE_LABEL}))
        topics = tuple(int(topic) for topic in self.topics)
        if topics != present:
            raise GapInputError(f"the topics {list(topics)} differ from the labels' {present}")
        centroids = np.array(self.centroids, dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[0] != len(topics):
            raise GapInputError(
                f"the centroids have shape {centroids.shape}, not one row per topic ({len(topics)})"
            )
        if not np.all(np.isfinite(centroids)) or np.any(np.linalg.norm(centroids, axis=1) == 0):
            raise GapInputError("every centroid must be finite and nonzero")
        npmi = tuple(self.npmi)
        if len(npmi) != len(topics):
            raise GapInputError(f"{len(npmi)} NPMI values were given for {len(topics)} topics")
        for value in npmi:
            if value is not None and not (math.isfinite(value) and -1.0 <= value <= 1.0):
                raise GapInputError(f"an NPMI value must lie in [-1, 1], not {value}")
        labels.setflags(write=False)
        centroids.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "npmi", npmi)

    @classmethod
    def from_topic_map(cls, topic_map: "TopicMap", coherence: "CoherenceResult") -> Self:
        """Gather a :class:`~thematic_redundancy.topic_map.build.TopicMap` and the NPMI of its
        topics, computed on its terms in topic order.

        Raises:
            GapInputError: if the coherence does not cover the map's topics in order.
        """
        if len(coherence.topics) != len(topic_map.topics):
            raise GapInputError(
                f"the coherence has {len(coherence.topics)} topics, the map {len(topic_map.topics)}"
            )
        for topic, scored in zip(topic_map.topics, coherence.topics, strict=True):
            if topic.terms[: len(scored.terms)] != scored.terms:
                raise GapInputError(
                    f"the coherence terms {scored.terms} are not the first terms of topic "
                    f"{topic.topic}"
                )
        return cls(
            labels=topic_map.labels,
            topics=tuple(topic.topic for topic in topic_map.topics),
            centroids=topic_map.centroids,
            npmi=tuple(scored.npmi for scored in coherence.topics),
        )

    def members(self, metadata: "ThesisMetadata", topic: int) -> frozenset[str]:
        """Return the doc codes of ``topic``'s theses."""
        return frozenset(np.asarray(metadata.doc_codes, dtype=object)[self.labels == topic])


@dataclass(frozen=True)
class ThesisMetadata:
    """The metadata of the theses, in the row order of the maps' labels."""

    doc_codes: tuple[str, ...]
    """Unique identifiers; the stability filter matches gaps across seeds by them."""
    programs: tuple[str, ...]
    """Each thesis's program key."""
    years: tuple[int, ...]
    """Each thesis's year (``dc.date.issued``)."""

    def __post_init__(self) -> None:
        doc_codes, programs, years = tuple(self.doc_codes), tuple(self.programs), tuple(self.years)
        if not len(doc_codes) == len(programs) == len(years):
            raise GapInputError(
                f"the metadata lengths differ: {len(doc_codes)} doc codes, {len(programs)} "
                f"programs, {len(years)} years"
            )
        if any(not isinstance(code, str) or not code for code in doc_codes):
            raise GapInputError("every doc code must be a non-empty string")
        if len(set(doc_codes)) != len(doc_codes):
            raise GapInputError("the doc codes must be unique")
        if any(not isinstance(program, str) or not program for program in programs):
            raise GapInputError("every program must be a non-empty key")
        if any(type(year) is not int for year in years):
            raise GapInputError("every year must be an integer")
        object.__setattr__(self, "doc_codes", doc_codes)
        object.__setattr__(self, "programs", programs)
        object.__setattr__(self, "years", years)


# --- Gaps -----------------------------------------------------------------------------------


class _OneTopicGap:
    """What a gap of one topic shares: its members, and no program in its match key."""

    members: frozenset[str]

    @property
    def topic_members(self) -> tuple[frozenset[str], ...]:
        """The members of each topic of the gap, which the stability filter matches."""
        return (self.members,)

    @property
    def match_key(self) -> tuple[str, ...]:
        """What two gaps of one rule must share to match across seeds."""
        return ()


@dataclass(frozen=True)
class SmallCoherentTopic(_OneTopicGap):
    """A rule (a) gap: a topic in the bottom quartile of size whose NPMI reaches the floor."""

    rule: ClassVar[str] = "a"
    topic: int
    size: int
    size_threshold: float
    """The first quartile of the map's topic sizes."""
    topics_in_map: int
    npmi: float
    npmi_floor: float
    members: frozenset[str]
    program_counts: tuple[tuple[str, int], ...]
    """The topic's theses per program, by program key (R5)."""

    @property
    def reason(self) -> str:
        return (
            f"topic {self.topic} has {self.size} theses, at most the bottom-quartile size "
            f"{self.size_threshold:g} of the {self.topics_in_map} topics, and its NPMI "
            f"{self.npmi:.4f} reaches the floor {self.npmi_floor:g}"
        )


@dataclass(frozen=True)
class InterstitialPair:
    """A rule (b) gap: two nearby topics with few bridge theses between them."""

    rule: ClassVar[str] = "b"
    first_topic: int
    second_topic: int
    """Greater than :attr:`first_topic`."""
    centroid_cosine: float
    neighbor_rank: int
    """The better rank, from 1, of either topic among the other's nearest topics."""
    nearest_topics: int
    bridge_documents: int
    bridge_max_docs: int
    bridges: frozenset[str]
    """The doc codes of the bridge theses."""
    first_members: frozenset[str]
    second_members: frozenset[str]

    @property
    def topic_members(self) -> tuple[frozenset[str], ...]:
        return (self.first_members, self.second_members)

    @property
    def match_key(self) -> tuple[str, ...]:
        return ()

    @property
    def reason(self) -> str:
        noun = "thesis" if self.bridge_documents == 1 else "theses"
        return (
            f"topics {self.first_topic} and {self.second_topic} are among each other's "
            f"{self.nearest_topics} nearest topics (rank {self.neighbor_rank}, centroid cosine "
            f"{self.centroid_cosine:.4f}) and have {self.bridge_documents} bridge {noun}, at "
            f"most {self.bridge_max_docs}"
        )


@dataclass(frozen=True)
class TransferableTopic(_OneTopicGap):
    """A rule (c) gap: a topic present in a source program and rare in an affine target."""

    rule: ClassVar[str] = "c"
    topic: int
    source: str
    target: str
    affinity: int
    min_affinity: int
    source_count: int
    source_theses: int
    """All the source program's theses in the map, noise included."""
    source_share: float
    target_count: int
    target_theses: int
    target_share: float
    present_min_count: int
    present_min_share: float
    rare_max_share: float
    members: frozenset[str]

    @property
    def match_key(self) -> tuple[str, ...]:
        return (self.source, self.target)

    @property
    def reason(self) -> str:
        return (
            f"topic {self.topic} holds {self.source_count} of {self.source_theses} {self.source} "
            f"theses ({self.source_share:.1%}) and {self.target_count} of {self.target_theses} "
            f"{self.target} theses ({self.target_share:.1%}, at most {self.rare_max_share:.1%}); "
            f"affinity {self.affinity}, at least {self.min_affinity}"
        )


@dataclass(frozen=True)
class TemporalDecline(_OneTopicGap):
    """A rule (d) gap: a topic whose share of the theses fell from the earlier window."""

    rule: ClassVar[str] = "d"
    topic: int
    early_years: tuple[int, int]
    late_years: tuple[int, int]
    early_count: int
    early_theses: int
    """All the theses of the earlier window, noise included."""
    early_share: float
    late_count: int
    late_theses: int
    late_share: float
    decline: float
    """``early_share - late_share``."""
    min_count: int
    min_decline: float
    members: frozenset[str]
    program_counts: tuple[tuple[str, int], ...]
    """The topic's theses per program, by program key (R5)."""

    @property
    def reason(self) -> str:
        early, late = self.early_years, self.late_years
        return (
            f"topic {self.topic} fell from {self.early_count} of {self.early_theses} theses in "
            f"{early[0]}-{early[1]} ({self.early_share:.1%}) to {self.late_count} of "
            f"{self.late_theses} in {late[0]}-{late[1]} ({self.late_share:.1%}), a decline of "
            f"{self.decline:.1%}, at least {self.min_decline:.1%}"
        )


Gap = SmallCoherentTopic | InterstitialPair | TransferableTopic | TemporalDecline
"""Any gap of rules (a) to (d)."""


# --- Rules ----------------------------------------------------------------------------------


def small_coherent_topics(
    gap_map: GapMap, metadata: ThesisMetadata, *, npmi_floor: float
) -> tuple[SmallCoherentTopic, ...]:
    """Rule (a): the topics in the bottom quartile of size whose NPMI reaches ``npmi_floor``.

    A topic is in the bottom quartile when its size is at most the first quartile of the map's
    topic sizes (linear interpolation; the noise is no topic). A topic with an undefined NPMI
    never reaches the floor. The gaps come in topic order.

    Raises:
        GapInputError: if the metadata does not describe the map's theses or the floor is not
            a finite value in [-1, 1].
    """
    _require_rows(gap_map, metadata)
    if not (math.isfinite(npmi_floor) and -1.0 <= npmi_floor <= 1.0):
        raise GapInputError(f"the NPMI floor must lie in [-1, 1], not {npmi_floor}")
    if not gap_map.topics:
        return ()
    sizes = [int(np.count_nonzero(gap_map.labels == topic)) for topic in gap_map.topics]
    threshold = float(np.quantile(sizes, SIZE_QUANTILE))
    gaps = []
    for topic, size, npmi in zip(gap_map.topics, sizes, gap_map.npmi, strict=True):
        if size <= threshold and npmi is not None and npmi >= npmi_floor:
            gaps.append(
                SmallCoherentTopic(
                    topic=topic,
                    size=size,
                    size_threshold=threshold,
                    topics_in_map=len(gap_map.topics),
                    npmi=npmi,
                    npmi_floor=npmi_floor,
                    members=gap_map.members(metadata, topic),
                    program_counts=_program_counts(gap_map, metadata, topic),
                )
            )
    return tuple(gaps)


def interstitial_pairs(
    gap_map: GapMap,
    vectors: Any,
    metadata: ThesisMetadata,
    *,
    nearest_topics: int,
    bridge_max_docs: int,
) -> tuple[InterstitialPair, ...]:
    """Rule (b): pairs of nearby topics with at most ``bridge_max_docs`` bridge theses.

    Each topic's nearest topics are the ``nearest_topics`` others with the highest cosine
    between centroids (a tie goes to the lower topic label); a pair is a candidate when either
    topic is among the other's nearest.

    A thesis, noise included, is a bridge of the pair ``(t, u)`` when ``t`` and ``u`` are its
    two most similar centroids (cosine on the original vectors) and it is more similar to the
    pair's midpoint direction, the normalized sum of the two unit centroids, than to either
    centroid: it lies in the middle half of the arc between them, measured in the plane of the
    two centroids. The definition has no parameter; antipodal centroids have no midpoint and so
    no bridge. The gaps come ordered by the pair's topic labels.

    Args:
        gap_map: the map, whose centroids are in the space of ``vectors``.
        vectors: one row per thesis, the original document vectors, in the labels' order.
        metadata: the theses' metadata.
        nearest_topics: how many nearest topics each topic pairs with (``gaps.nearest_topics``).
        bridge_max_docs: the most bridge theses a gap may have (``gaps.bridge_max_docs``).

    Raises:
        GapInputError: if an argument is out of range, or the vectors do not match the map.
    """
    _require_rows(gap_map, metadata)
    if nearest_topics < 1:
        raise GapInputError(f"nearest_topics must be at least 1, not {nearest_topics}")
    if bridge_max_docs < 0:
        raise GapInputError(f"bridge_max_docs must be at least 0, not {bridge_max_docs}")
    points = _unit_rows(vectors, gap_map)
    if len(gap_map.topics) < 2:
        return ()
    centroids = gap_map.centroids / np.linalg.norm(gap_map.centroids, axis=1, keepdims=True)
    between = centroids @ centroids.T
    candidates = _nearest_pairs(between, nearest_topics)
    bridges = _bridges(points @ centroids.T, between)
    doc_codes = metadata.doc_codes
    gaps = []
    for (first, second), rank in sorted(candidates.items()):
        found = bridges.get((first, second), [])
        if len(found) <= bridge_max_docs:
            first_topic, second_topic = gap_map.topics[first], gap_map.topics[second]
            gaps.append(
                InterstitialPair(
                    first_topic=first_topic,
                    second_topic=second_topic,
                    centroid_cosine=float(between[first, second]),
                    neighbor_rank=rank,
                    nearest_topics=nearest_topics,
                    bridge_documents=len(found),
                    bridge_max_docs=bridge_max_docs,
                    bridges=frozenset(doc_codes[row] for row in found),
                    first_members=gap_map.members(metadata, first_topic),
                    second_members=gap_map.members(metadata, second_topic),
                )
            )
    return tuple(gaps)


def transferable_topics(
    gap_map: GapMap,
    metadata: ThesisMetadata,
    affinity: CareerAffinity,
    *,
    min_affinity: int,
    present_min_count: int,
    present_min_share: float,
    rare_max_share: float,
) -> tuple[TransferableTopic, ...]:
    """Rule (c): topics present in a source program and absent or rare in an affine target.

    Shares are per program (R5): a topic's theses of a program over all that program's theses
    in the map, noise included. A topic is present in the source with at least
    ``present_min_count`` theses and a share of at least ``present_min_share``; it is absent or
    rare in the target with a share of at most ``rare_max_share``. The two programs are affine
    when their score is at least ``min_affinity``. A program excluded as a target (D29:
    Electrónica) can still be a source. A program with no thesis in the map is neither. The
    gaps come by topic, then source, then target, in the affinity's program order.

    Raises:
        GapInputError: if a threshold is out of range, ``rare_max_share`` is not below
            ``present_min_share``, or a thesis has a program the matrix does not know.
    """
    _require_rows(gap_map, metadata)
    if min_affinity < 1 or min_affinity not in affinity.scale:
        raise GapInputError(
            f"min_affinity must be a score of the scale {list(affinity.scale)} above 0, "
            f"not {min_affinity}"
        )
    if present_min_count < 1:
        raise GapInputError(f"present_min_count must be at least 1, not {present_min_count}")
    if not 0.0 < present_min_share <= 1.0:
        raise GapInputError(f"present_min_share must lie in (0, 1], not {present_min_share}")
    if not 0.0 <= rare_max_share < present_min_share:
        raise GapInputError(
            f"rare_max_share must lie in [0, present_min_share), not {rare_max_share}"
        )
    unknown = sorted(set(metadata.programs) - set(affinity.programs))
    if unknown:
        raise GapInputError(f"the theses have programs the affinity does not know: {unknown}")
    programs = np.asarray(metadata.programs, dtype=object)
    totals = {program: int(np.count_nonzero(programs == program)) for program in affinity.programs}
    represented = [program for program in affinity.programs if totals[program] > 0]
    gaps = []
    for topic in gap_map.topics:
        in_topic = gap_map.labels == topic
        counts = {p: int(np.count_nonzero(in_topic & (programs == p))) for p in represented}
        shares = {program: counts[program] / totals[program] for program in represented}
        members = gap_map.members(metadata, topic)
        for source in represented:
            if counts[source] < present_min_count or shares[source] < present_min_share:
                continue
            for target in represented:
                if target == source or not affinity.can_be_target(target):
                    continue
                score = affinity.score(source, target)
                if score < min_affinity or shares[target] > rare_max_share:
                    continue
                gaps.append(
                    TransferableTopic(
                        topic=topic,
                        source=source,
                        target=target,
                        affinity=score,
                        min_affinity=min_affinity,
                        source_count=counts[source],
                        source_theses=totals[source],
                        source_share=shares[source],
                        target_count=counts[target],
                        target_theses=totals[target],
                        target_share=shares[target],
                        present_min_count=present_min_count,
                        present_min_share=present_min_share,
                        rare_max_share=rare_max_share,
                        members=members,
                    )
                )
    return tuple(gaps)


def temporal_declines(
    gap_map: GapMap,
    metadata: ThesisMetadata,
    *,
    first_year: int,
    split_year: int,
    last_year: int,
    min_count: int,
    min_decline: float,
) -> tuple[TemporalDecline, ...]:
    """Rule (d): topics whose share fell by at least ``min_decline`` from the earlier window.

    The windows are ``first_year`` to ``split_year - 1`` and ``split_year`` to ``last_year``
    (2021–2023 and 2024–2026 as configured). A topic's share of a window is its theses of that
    window over all the window's theses, noise included. A topic declines when its earlier
    count is at least ``min_count`` and its earlier share minus its later share is at least
    ``min_decline``. The gaps come in topic order.

    Raises:
        GapInputError: if the windows or thresholds are invalid, a thesis lies outside them, or
            a window has no thesis.
    """
    _require_rows(gap_map, metadata)
    if not first_year < split_year <= last_year:
        raise GapInputError(
            f"split_year ({split_year}) must be after first_year ({first_year}) and not after "
            f"last_year ({last_year})"
        )
    if min_count < 1:
        raise GapInputError(f"min_count must be at least 1, not {min_count}")
    if not (math.isfinite(min_decline) and 0.0 < min_decline <= 1.0):
        raise GapInputError(f"min_decline must lie in (0, 1], not {min_decline}")
    years = np.asarray(metadata.years, dtype=np.int64)
    outside = sorted({int(year) for year in years if not first_year <= year <= last_year})
    if outside:
        raise GapInputError(f"theses from {outside} lie outside {first_year}-{last_year}")
    early = years < split_year
    late = ~early
    early_theses, late_theses = int(np.count_nonzero(early)), int(np.count_nonzero(late))
    if early_theses == 0 or late_theses == 0:
        raise GapInputError(
            f"a window has no thesis: {early_theses} in {first_year}-{split_year - 1}, "
            f"{late_theses} in {split_year}-{last_year}"
        )
    gaps = []
    for topic in gap_map.topics:
        in_topic = gap_map.labels == topic
        early_count = int(np.count_nonzero(in_topic & early))
        late_count = int(np.count_nonzero(in_topic & late))
        early_share, late_share = early_count / early_theses, late_count / late_theses
        if early_count >= min_count and early_share - late_share >= min_decline:
            gaps.append(
                TemporalDecline(
                    topic=topic,
                    early_years=(first_year, split_year - 1),
                    late_years=(split_year, last_year),
                    early_count=early_count,
                    early_theses=early_theses,
                    early_share=early_share,
                    late_count=late_count,
                    late_theses=late_theses,
                    late_share=late_share,
                    decline=early_share - late_share,
                    min_count=min_count,
                    min_decline=min_decline,
                    members=gap_map.members(metadata, topic),
                    program_counts=_program_counts(gap_map, metadata, topic),
                )
            )
    return tuple(gaps)


# --- Helpers --------------------------------------------------------------------------------


def _require_rows(gap_map: GapMap, metadata: ThesisMetadata) -> None:
    if len(metadata.doc_codes) != gap_map.labels.size:
        raise GapInputError(
            f"the metadata describes {len(metadata.doc_codes)} theses, the map "
            f"{gap_map.labels.size}"
        )


def _program_counts(
    gap_map: GapMap, metadata: ThesisMetadata, topic: int
) -> tuple[tuple[str, int], ...]:
    programs = np.asarray(metadata.programs, dtype=object)[gap_map.labels == topic]
    return tuple(sorted(Counter(programs.tolist()).items()))


def _unit_rows(vectors: Any, gap_map: GapMap) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != gap_map.labels.size:
        raise GapInputError(
            f"the vectors have shape {matrix.shape}, not one row per thesis ({gap_map.labels.size})"
        )
    if matrix.shape[1] != gap_map.centroids.shape[1]:
        raise GapInputError(
            f"the vectors have {matrix.shape[1]} dimensions, the centroids "
            f"{gap_map.centroids.shape[1]}"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or np.any(norms == 0):
        raise GapInputError("every vector must be finite and nonzero")
    return matrix / norms


def _nearest_pairs(between: np.ndarray, nearest_topics: int) -> dict[tuple[int, int], int]:
    """Return each candidate pair of topic rows, lower row first, with its better rank."""
    count = between.shape[0]
    pairs: dict[tuple[int, int], int] = {}
    for row in range(count):
        others = sorted(
            (other for other in range(count) if other != row),
            key=lambda other: (-between[row, other], other),
        )
        for rank, other in enumerate(others[:nearest_topics], start=1):
            key = (min(row, other), max(row, other))
            pairs[key] = min(rank, pairs.get(key, rank))
    return pairs


def _bridges(similarity: np.ndarray, between: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Return the bridge theses of each pair of topic rows, lower row first (see rule (b))."""
    order = np.argsort(-similarity, axis=1, kind="stable")
    bridges: dict[tuple[int, int], list[int]] = {}
    for thesis in range(similarity.shape[0]):
        first, second = int(order[thesis, 0]), int(order[thesis, 1])
        span = math.sqrt(max(2.0 + 2.0 * float(between[first, second]), 0.0))
        if span == 0.0:
            continue
        to_first, to_second = similarity[thesis, first], similarity[thesis, second]
        if (to_first + to_second) / span > max(to_first, to_second):
            bridges.setdefault((min(first, second), max(first, second)), []).append(thesis)
    return bridges
