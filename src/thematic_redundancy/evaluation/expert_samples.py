"""The anonymized result samples of the expert session (T31; specified in
``docs/expert_session/muestras_anonimizadas.md``), drawn, rendered and guarded.

Four samples, each a set of cases printed for the panel, in neutral Spanish:

- **A, assignments of the temporal split** (:func:`draw_assignments`): up to
  ``assignments_per_program`` theses of the later period per program, drawn from T27's expert
  sample, each with its assigned group on the earlier period's map, the group's terms,
  ``group_examples`` of the group's theses, O06's atypical flag and the group's saturation
  quartile (T23);
- **B, simulated queries** (:func:`simulate_queries`): ``queries_per_program`` of sample A's
  cases per program play a proposal. Its existing document vector is the query (no encoder):
  T22's cosine search ranks the earlier period's theses, T23's assignment (through T26's
  :func:`~thematic_redundancy.recommendation.alternatives.alternative_lines`) gives its group on
  the earlier map, and T26's recommenders give its antecedents, neighbor groups and nearby
  stable gaps. The candidates are the earlier theses only, so the query thesis, a later one,
  is never in its own ranking. No redundancy verdict is shown, since τ is not calibrated;
- **C, topics** (:func:`draw_topics`): ``topics`` groups of T19's reference map, noise left
  out, each with its size per program, its terms and ``topic_examples`` random members;
- **D, gaps** (:func:`draw_gaps`): up to ``gaps_per_rule`` stable gaps of T24a's reference run
  per rule, each with the rule's fixed plain-language sentence, its programs and years, its
  groups' terms, its theses and ``gap_examples`` of them.

**Draws.** Sample ``k`` draws from its own generator, ``numpy.random.default_rng([seed, k])``
with ``k`` 1 to 4 for A to D (:func:`sample_generator`), so changing one sample's sizes leaves
the others' draws alone. Within a stratum (a program, or a rule) the candidates, sorted by doc
code (or gap id, or topic), are put in a random order and taken in that order; a candidate that
cannot be used (in B, a thesis without an abstract) is skipped, the next one is taken, and the
skip is recorded as a :class:`Replacement`. The examples of each case are then drawn without
replacement from its group's members, sorted by doc code, in the order the cases were drawn.
Last, each case gets a code (``A01``, ``B03``, ...) from a random permutation, and the cases
are shown in code order, so the code says nothing about the stratum or the draw.

**Anonymization.** The texts are titles, and in B's proposals the abstract, objectives and
keywords (A's cases show keywords too); never a person, a doc code, a uuid, a handle or a link.
:func:`check_texts` refuses any rendered text that holds a person's name form or ORCID of the
snapshot (:func:`person_strings`, searched without word boundaries, D34 (1)) or an identifier
of a thesis, so a hit stops the generation. Nothing here reads or writes a file.
"""

import csv
import io
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from thematic_redundancy.corpus.profile import PERSON_KEYS, leaked_strings, sensitive_strings
from thematic_redundancy.corpus.snapshot import SnapshotRecord
from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL
from thematic_redundancy.gaps.results_store import GapRow
from thematic_redundancy.gaps.stability import GapSupport
from thematic_redundancy.recommendation.alternatives import (
    NeighborReference,
    PairAnchor,
    alternative_lines,
)
from thematic_redundancy.recommendation.antecedents import rank_antecedents
from thematic_redundancy.redundancy.saturation import SaturationQuartiles
from thematic_redundancy.redundancy.search import cosine_top_k
from thematic_redundancy.shared.config import AppConfig, PercentileMethod, QuartileBoundary
from thematic_redundancy.topic_map.build import TopicMap

SAMPLES = ("A", "B", "C", "D")
"""The samples, in the order they are drawn."""
SAMPLE_FILE_NAMES = {sample: f"muestra_{sample.lower()}.md" for sample in SAMPLES}
"""The printable Markdown of each sample."""
GAP_RULES = ("a", "b", "c", "d")
RULE_TEXTS = {
    "a": "Grupo pequeño pero coherente: tiene pocas tesis y sus términos son consistentes "
    "entre sí.",
    "b": "Zona entre dos grupos vecinos: casi ninguna tesis conecta sus temas.",
    "c": "Tema presente en una carrera y ausente en una carrera afín, donde podría ser "
    "transferible.",
    "d": "Tema en descenso: su participación bajó entre 2021–2023 y 2024–2026.",
}
"""The fixed sentence that the experts read for each gap rule (D32), from the kit."""
VERDICT_PENDING = "No disponible: el umbral de redundancia aún no está calibrado."
"""Sample B's redundancy verdict while τ is not calibrated (T22)."""
NO_ABSTRACT = "no_abstract"
"""Why a thesis cannot be a simulated query: its ficha has no abstract."""
PROVISIONAL_NOTICE = (
    "> **Resultados provisionales.** La configuración del mapa temático que usa el sistema aún "
    "es provisional; si cambia, estas muestras se generan de nuevo con las mismas reglas."
)
"""The notice every sample carries while T19's winner is provisional."""


class ExpertSampleError(ValueError):
    """The samples cannot be drawn: an input breaks the samples' contract."""


class NameGuardError(ExpertSampleError):
    """A rendered sample holds a person's name, an ORCID or an identifier, so nothing was
    written."""


@dataclass(frozen=True)
class Thesis:
    """What the samples may show of one thesis, from its ficha."""

    doc_code: str
    program: str
    """The program's key."""
    year: int
    title: str
    abstract: str | None
    objectives: str | None
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class SampleSettings:
    """Everything the samples take from the configuration."""

    seed: int
    programs: tuple[str, ...]
    """The configured programs' keys, in order: the strata of A and B."""
    program_names: dict[str, str]
    assignments_per_program: int
    group_examples: int
    queries_per_program: int
    ranking_size: int
    topics: int
    topic_examples: int
    gaps_per_rule: int
    gap_examples: int
    terms: int
    saturation_method: PercentileMethod
    saturation_boundary: QuartileBoundary
    top_k: int
    """T26's antecedents per proposal (``recommender.top_k``)."""
    mmr_lambda: float
    neighbor_topics: int
    nearby_gaps: int

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        """Gather ``expert_samples``, the programs and ``recommender``."""
        block, recommender = config.expert_samples, config.recommender
        return cls(
            seed=block.seed,
            programs=tuple(program.key for program in config.snapshot.programs),
            program_names={program.key: program.name for program in config.snapshot.programs},
            assignments_per_program=block.assignments_per_program,
            group_examples=block.group_examples,
            queries_per_program=block.queries_per_program,
            ranking_size=block.ranking_size,
            topics=block.topics,
            topic_examples=block.topic_examples,
            gaps_per_rule=block.gaps_per_rule,
            gap_examples=block.gap_examples,
            terms=block.terms,
            saturation_method=block.saturation_method,
            saturation_boundary=block.saturation_boundary,
            top_k=recommender.top_k,
            mmr_lambda=recommender.mmr_lambda,
            neighbor_topics=recommender.neighbor_topics,
            nearby_gaps=recommender.nearby_gaps,
        )

    def record(self) -> dict[str, Any]:
        """Return the values as the manifest records them."""
        return {
            name: getattr(self, name)
            for name in (
                "seed",
                "assignments_per_program",
                "group_examples",
                "queries_per_program",
                "ranking_size",
                "topics",
                "topic_examples",
                "gaps_per_rule",
                "gap_examples",
                "terms",
                "saturation_method",
                "saturation_boundary",
                "top_k",
                "mmr_lambda",
                "neighbor_topics",
                "nearby_gaps",
            )
        }


def sample_generator(seed: int, sample: str) -> np.random.Generator:
    """Return the generator of ``sample`` (``"A"`` to ``"D"``): ``default_rng([seed, k])``,
    ``k`` its place in :data:`SAMPLES` from 1.

    Raises:
        ExpertSampleError: if ``sample`` is not one of :data:`SAMPLES`.
    """
    if sample not in SAMPLES:
        raise ExpertSampleError(f"the sample must be one of {SAMPLES}, not {sample!r}")
    return np.random.default_rng([seed, SAMPLES.index(sample) + 1])


@dataclass(frozen=True)
class Replacement:
    """A drawn candidate that could not be used and was replaced by the next one drawn."""

    sample: str
    doc_code: str
    program: str
    reason: str


# --- Sample A -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A thesis of T27's expert sample: a later thesis and its group on the earlier map."""

    doc_code: str
    program: str
    year: int
    topic: int
    atypical: bool


@dataclass(frozen=True)
class AssignmentCase:
    """One case of sample A."""

    code: str
    doc_code: str
    program: str
    year: int
    topic: int
    terms: tuple[str, ...]
    examples: tuple[str, ...]
    """Doc codes of the group's earlier theses shown, in draw order."""
    atypical: bool
    saturation: str | None
    """``"Q1"`` to ``"Q4"``; ``None`` when the quartiles are undefined."""


def draw_assignments(
    candidates: Sequence[Candidate],
    *,
    members: Mapping[int, Sequence[str]],
    topic_terms: Mapping[int, Sequence[str]],
    saturation: SaturationQuartiles,
    settings: SampleSettings,
) -> tuple[AssignmentCase, ...]:
    """Draw sample A from T27's expert sample (see the module).

    Args:
        candidates: T27's expert sample.
        members: each group's earlier theses (doc codes) on the earlier map.
        topic_terms: each group's terms, best first.
        saturation: T23's quartiles of the earlier map's group sizes.
        settings: the samples' settings.

    Raises:
        ExpertSampleError: if a candidate's program is not configured or its group is not one
            of the earlier map's.
    """
    for candidate in candidates:
        if candidate.topic not in members or candidate.topic not in topic_terms:
            raise ExpertSampleError(
                f"a thesis of T27's expert sample is assigned to group {candidate.topic}, which "
                "is not a group of the earlier map"
            )
    rng = sample_generator(settings.seed, "A")
    drawn: list[Candidate] = []
    for pool in _strata(candidates, settings.programs, _program_of):
        chosen, _ = _draw(pool, settings.assignments_per_program, rng)
        drawn += chosen
    examples = [_examples(members[case.topic], settings.group_examples, rng) for case in drawn]
    codes = _codes("A", len(drawn), rng)
    cases = [
        AssignmentCase(
            code=code,
            doc_code=case.doc_code,
            program=case.program,
            year=case.year,
            topic=case.topic,
            terms=tuple(topic_terms[case.topic])[: settings.terms],
            examples=shown,
            atypical=case.atypical,
            saturation=saturation.of_topic(case.topic).level,
        )
        for code, case, shown in zip(codes, drawn, examples, strict=True)
    ]
    return tuple(sorted(cases, key=_code))


# --- Sample B -------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySpace:
    """Where the simulated queries run: every thesis's vector and the two maps."""

    doc_codes: tuple[str, ...]
    """Every thesis, in the vectors' order."""
    vectors: np.ndarray
    """The winner's document vectors, one dense row per thesis."""
    earlier: np.ndarray
    """The rows of the earlier period's theses: the only candidates of a ranking."""
    earlier_map: TopicMap
    """The earlier period's map (T27), whose groups samples A and B show."""
    full_map: TopicMap
    """T19's reference map, whose stable gaps T24a found."""
    gaps: tuple[GapSupport, ...]
    """The reference run's stable gaps, in the run's order."""
    full_terms: Mapping[int, Sequence[str]]
    """Each group's terms on the full map."""


@dataclass(frozen=True)
class RankedWork:
    """One earlier thesis of a query's similarity ranking."""

    position: int
    doc_code: str
    similarity: float


@dataclass(frozen=True)
class NeighborLine:
    """A neighbor group of the proposal's group on the earlier map (T26)."""

    topic: int
    terms: tuple[str, ...]


@dataclass(frozen=True)
class GapLine:
    """A stable gap near the proposal (T26), with its groups' terms on the full map."""

    rule: str
    terms: tuple[tuple[str, ...], ...]
    """The terms of each of the gap's groups: one, or two for rule (b)."""


@dataclass(frozen=True)
class QueryCase:
    """One case of sample B."""

    code: str
    doc_code: str
    source: str
    """The code of the sample A case it was drawn from."""
    program: str
    year: int
    topic: int
    terms: tuple[str, ...]
    ranking: tuple[RankedWork, ...]
    antecedents: tuple[str, ...]
    """Doc codes, in T26's order."""
    neighbors: tuple[NeighborLine, ...]
    gaps: tuple[GapLine, ...]


def simulate_queries(
    cases: Sequence[AssignmentCase],
    space: QuerySpace,
    *,
    theses: Mapping[str, Thesis],
    settings: SampleSettings,
    neighbor_reference: NeighborReference,
    pair_anchor: PairAnchor,
    max_distance: float | None,
    pool_size: int | None,
) -> tuple[tuple[QueryCase, ...], tuple[Replacement, ...]]:
    """Draw sample B from sample A's ``cases`` and run each as a query (see the module).

    Args:
        cases: sample A.
        space: the vectors and maps the queries run on.
        theses: every thesis's texts, by doc code.
        settings: the samples' settings, with the recommender's values.
        neighbor_reference: T26's ``neighbor_reference``.
        pair_anchor: T26's ``pair_anchor``.
        max_distance: T26's ``max_distance``.
        pool_size: T26's antecedent ``pool_size``.

    Raises:
        ExpertSampleError: if a query thesis is among the candidates, or T23's assignment
            disagrees with the group T27 recorded for it.
    """
    rng = sample_generator(settings.seed, "B")
    drawn: list[AssignmentCase] = []
    replaced: list[Replacement] = []
    for pool in _strata(cases, settings.programs, _program_of):
        chosen, skipped = _draw(
            pool, settings.queries_per_program, rng, lambda case: _unusable(theses[case.doc_code])
        )
        drawn += chosen
        replaced += [
            Replacement("B", case.doc_code, case.program, reason) for case, reason in skipped
        ]
    codes = _codes("B", len(drawn), rng)
    rows = {code: row for row, code in enumerate(space.doc_codes)}
    candidates = {int(row) for row in space.earlier}
    documents = space.vectors[space.earlier]
    queries = []
    for code, case in zip(codes, drawn, strict=True):
        row = rows.get(case.doc_code)
        if row is None:
            raise ExpertSampleError("a case of sample A is not one of the theses of the vectors")
        if row in candidates:
            raise ExpertSampleError(
                "a query thesis is among the candidates of its own ranking; only the earlier "
                "period's theses may be"
            )
        query = space.vectors[row]
        ranking = cosine_top_k(documents, query, k=settings.ranking_size, exclude=None)
        antecedents = rank_antecedents(
            documents,
            query,
            k=settings.top_k,
            mmr_lambda=settings.mmr_lambda,
            pool_size=pool_size,
            exclude=None,
        )
        lines = alternative_lines(
            query,
            space.earlier_map,
            (),
            neighbor_topics=settings.neighbor_topics,
            nearby_gaps=0,
            neighbor_reference=neighbor_reference,
            pair_anchor=pair_anchor,
            max_distance=max_distance,
        )
        if lines.assignment.topic != case.topic:
            raise ExpertSampleError(
                f"T23's assignment puts a query in group {lines.assignment.topic} of the earlier "
                f"map, while T27 recorded group {case.topic}"
            )
        nearby = alternative_lines(
            query,
            space.full_map,
            space.gaps,
            neighbor_topics=0,
            nearby_gaps=settings.nearby_gaps,
            neighbor_reference=neighbor_reference,
            pair_anchor=pair_anchor,
            max_distance=max_distance,
        )
        queries.append(
            QueryCase(
                code=code,
                doc_code=case.doc_code,
                source=case.code,
                program=case.program,
                year=case.year,
                topic=case.topic,
                terms=lines.assigned_terms[: settings.terms],
                ranking=tuple(
                    RankedWork(
                        position, space.doc_codes[int(space.earlier[item.index])], item.score
                    )
                    for position, item in enumerate(ranking, start=1)
                ),
                antecedents=tuple(
                    space.doc_codes[int(space.earlier[item.index])]
                    for item in antecedents.antecedents
                ),
                neighbors=tuple(
                    NeighborLine(item.topic, item.terms[: settings.terms])
                    for item in lines.neighbors
                ),
                gaps=tuple(
                    GapLine(
                        item.rule,
                        tuple(
                            tuple(space.full_terms[topic])[: settings.terms]
                            for topic in item.topics
                        ),
                    )
                    for item in nearby.gaps
                ),
            )
        )
    return tuple(sorted(queries, key=_code)), tuple(replaced)


# --- Sample C -------------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicCase:
    """One case of sample C."""

    code: str
    topic: int
    size: int
    programs: tuple[tuple[str, int], ...]
    """Each program with members and their count, in the configured order."""
    terms: tuple[str, ...]
    examples: tuple[str, ...]
    """Doc codes of the members shown, in draw order."""


def draw_topics(
    labels: Mapping[str, int],
    topic_terms: Mapping[int, Sequence[str]],
    *,
    theses: Mapping[str, Thesis],
    settings: SampleSettings,
) -> tuple[TopicCase, ...]:
    """Draw sample C from T19's reference map: its ``labels`` by doc code, -1 for noise, and
    each group's terms.

    Raises:
        ExpertSampleError: if a group of the labels has no terms.
    """
    groups: dict[int, list[str]] = {}
    for code, label in labels.items():
        if label != NOISE_LABEL:
            groups.setdefault(int(label), []).append(code)
    for topic in sorted(groups):
        if topic not in topic_terms:
            raise ExpertSampleError(f"group {topic} of the reference map has no terms")
    rng = sample_generator(settings.seed, "C")
    drawn, _ = _draw(sorted(groups), settings.topics, rng)
    examples = [_examples(groups[topic], settings.topic_examples, rng) for topic in drawn]
    codes = _codes("C", len(drawn), rng)
    cases = []
    for code, topic, shown in zip(codes, drawn, examples, strict=True):
        counts = Counter(theses[member].program for member in groups[topic])
        cases.append(
            TopicCase(
                code=code,
                topic=topic,
                size=len(groups[topic]),
                programs=tuple(
                    (program, counts[program]) for program in settings.programs if counts[program]
                ),
                terms=tuple(topic_terms[topic])[: settings.terms],
                examples=shown,
            )
        )
    return tuple(sorted(cases, key=_code))


# --- Sample D -------------------------------------------------------------------------------


@dataclass(frozen=True)
class GapCase:
    """One case of sample D."""

    code: str
    gap_id: str
    rule: str
    topics: tuple[int, ...]
    topic_terms: tuple[tuple[str, ...], ...]
    programs: tuple[tuple[str, int], ...]
    years: tuple[tuple[int, int], ...]
    source: str | None
    """Rule (c): the program where the topic is present."""
    target: str | None
    """Rule (c): the affine program where it is absent."""
    theses: int
    examples: tuple[str, ...]


def draw_gaps(
    rows: Sequence[GapRow],
    members: Mapping[str, Sequence[str]],
    *,
    topic_terms: Mapping[int, Sequence[str]],
    settings: SampleSettings,
) -> tuple[GapCase, ...]:
    """Draw sample D from T24a's ``rows`` of the reference run's stable gaps; ``members`` holds
    each gap's theses (its topics' members, no bridge), by gap id.

    Raises:
        ExpertSampleError: if a row is not a stable gap of the reference run, or a gap has no
            member or names a group without terms.
    """
    for row in rows:
        if not row.reference:
            raise ExpertSampleError(f"gap {row.gap_id} is not a gap of the reference run")
        if row.stable is not True:
            raise ExpertSampleError(f"gap {row.gap_id} is not stable (D16)")
        if not members.get(row.gap_id):
            raise ExpertSampleError(f"gap {row.gap_id} has no member")
        missing = [topic for topic in row.topics if topic not in topic_terms]
        if missing:
            raise ExpertSampleError(f"gap {row.gap_id} names groups {missing} without terms")
    rng = sample_generator(settings.seed, "D")
    drawn: list[GapRow] = []
    for rule in GAP_RULES:
        pool = sorted((row for row in rows if row.rule == rule), key=_gap_id)
        chosen, _ = _draw(pool, settings.gaps_per_rule, rng)
        drawn += chosen
    examples = [_examples(members[row.gap_id], settings.gap_examples, rng) for row in drawn]
    codes = _codes("D", len(drawn), rng)
    cases = [
        GapCase(
            code=code,
            gap_id=row.gap_id,
            rule=row.rule,
            topics=tuple(row.topics),
            topic_terms=tuple(tuple(topic_terms[topic])[: settings.terms] for topic in row.topics),
            programs=tuple(zip(row.programs, row.program_counts, strict=True)),
            years=tuple(zip(row.years, row.year_counts, strict=True)),
            source=row.source,
            target=row.target,
            theses=row.theses,
            examples=shown,
        )
        for code, row, shown in zip(codes, drawn, examples, strict=True)
    ]
    return tuple(sorted(cases, key=_code))


# --- Text -----------------------------------------------------------------------------------


def render_samples(
    assignments: Sequence[AssignmentCase],
    queries: Sequence[QueryCase],
    topics: Sequence[TopicCase],
    gaps: Sequence[GapCase],
    *,
    theses: Mapping[str, Thesis],
    settings: SampleSettings,
    earlier_years: tuple[int, int],
    later_years: tuple[int, int],
    provisional: bool,
) -> dict[str, str]:
    """Return the printable Markdown of each sample, by file name (:data:`SAMPLE_FILE_NAMES`)."""
    text = _Text(theses, settings)
    return {
        SAMPLE_FILE_NAMES["A"]: _render_assignments(
            assignments, text, earlier_years, later_years, provisional
        ),
        SAMPLE_FILE_NAMES["B"]: _render_queries(
            queries, text, earlier_years, later_years, provisional
        ),
        SAMPLE_FILE_NAMES["C"]: _render_topics(topics, text, provisional),
        SAMPLE_FILE_NAMES["D"]: _render_gaps(gaps, text, provisional),
    }


@dataclass(frozen=True)
class _Text:
    """The theses and the program names, for the lines that name a thesis or a program."""

    theses: Mapping[str, Thesis]
    settings: SampleSettings

    def program(self, key: str) -> str:
        return self.settings.program_names.get(key, key)

    def title(self, code: str) -> str:
        return _flat(self.theses[code].title)

    def work(self, code: str) -> str:
        thesis = self.theses[code]
        return f"{self.title(code)} ({self.program(thesis.program)}, {thesis.year})"

    def keywords(self, code: str) -> str:
        keywords = [_flat(keyword) for keyword in self.theses[code].keywords if keyword.strip()]
        return "; ".join(keywords) or "Sin palabras clave en la ficha"


def _header(title: str, intro: str, provisional: bool) -> list[str]:
    lines = [f"# {title}", "", "Juego n.° ________", ""]
    if provisional:
        lines += [PROVISIONAL_NOTICE, ""]
    return [*lines, intro]


def _render_assignments(
    cases: Sequence[AssignmentCase],
    text: _Text,
    earlier_years: tuple[int, int],
    later_years: tuple[int, int],
    provisional: bool,
) -> str:
    earlier = _span(earlier_years)
    lines = _header(
        f"Muestra A. Asignaciones de tesis de {_span(later_years)} al mapa de {earlier}",
        f"Cada caso es una tesis de {_either(later_years)}, tratada como propuesta nueva, y el "
        f"grupo temático del mapa de {earlier} al que el sistema la asigna. Una propuesta es "
        "atípica cuando se parece a su grupo menos que casi todas las tesis del grupo. La "
        "saturación ubica el tamaño del grupo entre los del mapa: Q1 reúne a los grupos más "
        "pequeños y Q4 a los más grandes.",
        provisional,
    )
    for case in cases:
        thesis = text.theses[case.doc_code]
        lines += [
            "",
            f"## Caso {case.code}",
            "",
            f"- **Carrera y año:** {text.program(case.program)}, {case.year}",
            f"- **Título:** {text.title(case.doc_code)}",
            f"- **Palabras clave:** {text.keywords(thesis.doc_code)}",
            f"- **Grupo asignado:** {case.topic}. Términos: {', '.join(case.terms)}",
            f"- **Tesis del grupo ({earlier}):**",
            *(f"  - {text.work(code)}" for code in case.examples),
            f"- **Propuesta atípica:** {'Sí' if case.atypical else 'No'}",
            f"- **Saturación del grupo:** {case.saturation or 'No definida'}",
        ]
    return "\n".join(lines)


def _render_queries(
    cases: Sequence[QueryCase],
    text: _Text,
    earlier_years: tuple[int, int],
    later_years: tuple[int, int],
    provisional: bool,
) -> str:
    lines = _header(
        "Muestra B. Consultas simuladas",
        f"Cada caso es una tesis de {_either(later_years)} ingresada como propuesta en la vista "
        f"de consulta y comparada con las tesis de {earlier_years[0]} a {earlier_years[1]}. La "
        "similitud compara los textos de la propuesta y de cada tesis: cuanto más alta, más "
        "parecidos son.",
        provisional,
    )
    for case in cases:
        thesis = text.theses[case.doc_code]
        lines += [
            "",
            f"## Caso {case.code}",
            "",
            f"**Propuesta** ({text.program(case.program)}, {case.year})",
            "",
            f"- **Título:** {text.title(case.doc_code)}",
            f"- **Resumen:** {_flat(thesis.abstract or '') or 'No disponible en la ficha'}",
            f"- **Objetivos:** {_flat(thesis.objectives or '') or 'No disponibles en la ficha'}",
            f"- **Palabras clave:** {text.keywords(case.doc_code)}",
            "",
            f"**Grupo asignado:** {case.topic}. Términos: {', '.join(case.terms)}",
            "",
            "**Trabajos previos más similares**",
            "",
            "| Posición | Título | Carrera | Año | Similitud |",
            "|:-:|---|---|:-:|:-:|",
            *(
                f"| {work.position} | {_cell(text.title(work.doc_code))} | "
                f"{text.program(text.theses[work.doc_code].program)} | "
                f"{text.theses[work.doc_code].year} | {work.similarity:.2f} |"
                for work in case.ranking
            ),
            "",
            f"**Veredicto de redundancia:** {VERDICT_PENDING}",
            "",
            "**Antecedentes sugeridos:**",
            "",
            *(
                [
                    f"{rank}. {text.work(code)}"
                    for rank, code in enumerate(case.antecedents, start=1)
                ]
                or ["Ninguno."]
            ),
            "",
            "**Líneas alternativas:**",
            "",
            *(
                [
                    *(
                        f"- Grupo vecino {line.topic}. Términos: {', '.join(line.terms)}"
                        for line in case.neighbors
                    ),
                    *(
                        f"- Vacío cercano. {RULE_TEXTS[line.rule]} Términos: "
                        + " / ".join(", ".join(terms) for terms in line.terms)
                        for line in case.gaps
                    ),
                ]
                or ["Ninguna."]
            ),
        ]
    return "\n".join(lines)


def _render_topics(cases: Sequence[TopicCase], text: _Text, provisional: bool) -> str:
    lines = _header(
        "Muestra C. Grupos temáticos",
        "Cada grupo fue formado automáticamente por el sistema con las tesis del corpus. Los "
        "grupos y sus tesis de ejemplo se eligieron al azar, no entre los mejores.",
        provisional,
    )
    for case in cases:
        programs = "; ".join(f"{text.program(key)} {count}" for key, count in case.programs)
        lines += [
            "",
            f"## Grupo {case.code}",
            "",
            f"- **Tesis en el grupo:** {case.size}. Por carrera: {programs}",
            f"- **Términos representativos:** {', '.join(case.terms)}",
            "",
            "| Tesis de ejemplo | Carrera | Año |",
            "|---|---|:-:|",
            *(
                f"| {_cell(text.title(code))} | {text.program(text.theses[code].program)} | "
                f"{text.theses[code].year} |"
                for code in case.examples
            ),
        ]
    return "\n".join(lines)


def _render_gaps(cases: Sequence[GapCase], text: _Text, provisional: bool) -> str:
    lines = _header(
        "Muestra D. Vacíos de investigación",
        "Cada caso es un área que el sistema señala como poco explorada. Los vacíos se eligieron "
        "al azar entre los que el sistema encuentra de forma estable, hasta dos por tipo.",
        provisional,
    )
    for case in cases:
        programs = ", ".join(f"{text.program(key)} ({count})" for key, count in case.programs)
        if case.rule == "c" and case.source and case.target:
            programs += (
                f"; presente en {text.program(case.source)} y ausente en "
                f"{text.program(case.target)}"
            )
        years = ", ".join(f"{year} ({count})" for year, count in case.years)
        groups = "; ".join(
            f"Grupo {topic}: {', '.join(terms)}"
            for topic, terms in zip(case.topics, case.topic_terms, strict=True)
        )
        lines += [
            "",
            f"## Vacío {case.code}",
            "",
            f"- **Tipo:** {RULE_TEXTS[case.rule]}",
            f"- **Carreras:** {programs}. **Años:** {years}",
            f"- **Grupos y términos:** {groups}",
            f"- **Tesis en el vacío:** {case.theses}",
            "- **Tesis de ejemplo:**",
            *(f"  - {text.work(code)}" for code in case.examples),
        ]
    return "\n".join(lines)


KEY_COLUMNS = ("sample", "code", "doc_code", "topic", "gap_id", "source", "shown", "note")
"""The key's columns, in order."""


@dataclass(frozen=True)
class KeyRow:
    """One row of the key that links a case to what it shows, kept under ``data/``."""

    sample: str
    code: str
    """The case's code; empty for a replaced candidate."""
    doc_code: str
    """The case's thesis (A, B, or a replaced one); empty for C and D."""
    topic: str
    """The case's group(s), ``;``-separated; on the earlier map for A and B."""
    gap_id: str
    source: str
    """For B, the sample A case it came from."""
    shown: str
    """The doc codes of every other thesis the case shows, ``;``-separated."""
    note: str
    """For a replaced candidate, why it could not be used."""


def key_rows(
    assignments: Sequence[AssignmentCase],
    queries: Sequence[QueryCase],
    topics: Sequence[TopicCase],
    gaps: Sequence[GapCase],
    replacements: Sequence[Replacement],
) -> tuple[KeyRow, ...]:
    """Return the key: one row per case, in sample and code order, then one per replacement."""
    rows = [
        KeyRow("A", case.code, case.doc_code, str(case.topic), "", "", ";".join(case.examples), "")
        for case in assignments
    ]
    for query in queries:
        shown = dict.fromkeys([*(work.doc_code for work in query.ranking), *query.antecedents])
        rows.append(
            KeyRow(
                "B",
                query.code,
                query.doc_code,
                str(query.topic),
                "",
                query.source,
                ";".join(shown),
                "",
            )
        )
    rows += [
        KeyRow("C", case.code, "", str(case.topic), "", "", ";".join(case.examples), "")
        for case in topics
    ]
    rows += [
        KeyRow(
            "D",
            case.code,
            "",
            ";".join(str(topic) for topic in case.topics),
            case.gap_id,
            "",
            ";".join(case.examples),
            "",
        )
        for case in gaps
    ]
    rows += [
        KeyRow(item.sample, "", item.doc_code, "", "", "", "", item.reason) for item in replacements
    ]
    return tuple(rows)


def key_csv(rows: Sequence[KeyRow]) -> str:
    """Return the key as CSV text with a header row."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(KEY_COLUMNS)
    writer.writerows([getattr(row, name) for name in KEY_COLUMNS] for row in rows)
    return buffer.getvalue().rstrip("\n")


def composition(
    assignments: Sequence[AssignmentCase],
    queries: Sequence[QueryCase],
    topics: Sequence[TopicCase],
    gaps: Sequence[GapCase],
    replacements: Sequence[Replacement],
    *,
    programs: Sequence[str],
) -> dict[str, Any]:
    """Return the samples' composition in counts only: cases per sample, per program (A, B)
    and per rule (D), and the replacements."""

    def by_program(cases: Sequence[AssignmentCase] | Sequence[QueryCase]) -> dict[str, int]:
        counts = Counter(case.program for case in cases)
        return {program: counts[program] for program in programs}

    rules = Counter(case.rule for case in gaps)
    return {
        "A": {"cases": len(assignments), "programs": by_program(assignments)},
        "B": {
            "cases": len(queries),
            "programs": by_program(queries),
            "replaced": sum(item.sample == "B" for item in replacements),
        },
        "C": {"cases": len(topics)},
        "D": {"cases": len(gaps), "rules": {rule: rules[rule] for rule in GAP_RULES}},
    }


# --- Guards ---------------------------------------------------------------------------------


def person_strings(records: Iterable[SnapshotRecord]) -> frozenset[str]:
    """Return the snapshot's person-name forms and ORCID iDs, case folded:
    :func:`~thematic_redundancy.corpus.profile.sensitive_strings` of the records reduced to
    their author, advisor and juror names and their ORCID fields, so without the titles and
    abstracts that the samples show."""
    reduced = []
    for record in records:
        kept = {
            key: values
            for key, values in record.metadata.items()
            if key in PERSON_KEYS or key.casefold().endswith(".orcid")
        }
        reduced.append(record.model_copy(update={"metadata": kept}))
    return sensitive_strings(reduced)


def check_texts(
    texts: Mapping[str, str], *, persons: frozenset[str], identifiers: frozenset[str]
) -> None:
    """Refuse ``texts``, by file name, if any holds a string of ``persons`` or of
    ``identifiers`` (both searched ignoring case and spacing, without word boundaries), a
    ``DOC-`` code, a uuid or a link. The message names the files and counts, never a string.

    Raises:
        NameGuardError: if a text holds any of them.
    """
    names: dict[str, int] = {}
    found: dict[str, int] = {}
    for name, text in texts.items():
        if count := len(leaked_strings(text, persons)):
            names[name] = count
        count = len(leaked_strings(text, identifiers)) + sum(
            len(pattern.findall(text)) for pattern in _IDENTIFIER_PATTERNS
        )
        if count:
            found[name] = count
    problems = []
    if names:
        problems.append(f"a person's name or ORCID iD of the snapshot: {_listed(names)}")
    if found:
        problems.append(f"a thesis's doc code, uuid or handle, or a link: {_listed(found)}")
    if problems:
        raise NameGuardError(f"the samples hold {'; and '.join(problems)}; nothing was written")


_IDENTIFIER_PATTERNS = (
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"doc-[0-9a-f]{8}", re.IGNORECASE),
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE),
)
"""A link, a doc code and a uuid, whichever thesis they belong to."""


# --- Helpers ----------------------------------------------------------------------------------


def _strata[T: (Candidate, AssignmentCase)](
    items: Sequence[T], keys: Sequence[str], key_of: Callable[[T], str]
) -> list[list[T]]:
    """Return ``items`` grouped by ``key_of`` in the order of ``keys``, each group sorted by
    doc code.

    Raises:
        ExpertSampleError: if an item's key is not one of ``keys``.
    """
    groups: dict[str, list[T]] = {key: [] for key in keys}
    for item in items:
        key = key_of(item)
        if key not in groups:
            raise ExpertSampleError(f"the program {key} is not one of the configured programs")
        groups[key].append(item)
    return [sorted(groups[key], key=_doc_code) for key in keys]


def _draw[T](
    pool: Sequence[T],
    quota: int,
    rng: np.random.Generator,
    unusable: Callable[[T], str | None] = lambda _: None,
) -> tuple[list[T], list[tuple[T, str]]]:
    """Put ``pool`` in a random order and take its first ``quota`` usable items; return them
    and the unusable ones met before the quota was reached, with the reason."""
    chosen: list[T] = []
    skipped: list[tuple[T, str]] = []
    for index in rng.permutation(len(pool)):
        if len(chosen) == quota:
            break
        item = pool[int(index)]
        reason = unusable(item)
        if reason is None:
            chosen.append(item)
        else:
            skipped.append((item, reason))
    return chosen, skipped


def _examples(members: Sequence[str], count: int, rng: np.random.Generator) -> tuple[str, ...]:
    """Draw ``count`` of ``members`` (all when fewer), sorted first, without replacement."""
    group = sorted(members)
    if not group:
        return ()
    picked = rng.choice(len(group), size=min(count, len(group)), replace=False)
    return tuple(group[int(index)] for index in picked)


def _codes(prefix: str, count: int, rng: np.random.Generator) -> list[str]:
    """Return a code for each of ``count`` cases, in their order, from a random permutation."""
    return [f"{prefix}{int(position) + 1:02d}" for position in rng.permutation(count)]


def _unusable(thesis: Thesis) -> str | None:
    return None if thesis.abstract and thesis.abstract.strip() else NO_ABSTRACT


def _flat(text: str) -> str:
    return " ".join(text.split())


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _span(years: tuple[int, int]) -> str:
    return str(years[0]) if years[0] == years[1] else f"{years[0]}–{years[1]}"


def _either(years: tuple[int, int]) -> str:
    if years[0] == years[1]:
        return str(years[0])
    if years[1] == years[0] + 1:
        return f"{years[0]} o {years[1]}"
    return f"{years[0]} a {years[1]}"


def _listed(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} ({count})" for name, count in counts.items())


def _program_of(item: Candidate | AssignmentCase) -> str:
    return item.program


def _doc_code(item: Candidate | AssignmentCase) -> str:
    return item.doc_code


def _code(case: AssignmentCase | QueryCase | TopicCase | GapCase) -> str:
    return case.code


def _gap_id(row: GapRow) -> str:
    return row.gap_id
