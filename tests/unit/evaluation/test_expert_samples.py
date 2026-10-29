"""The anonymized result samples of the expert session (T31): the draws, the simulated queries,
the Spanish texts, the key and the guards, on small synthetic inputs. Every name, title and code
is invented; the doc codes follow the ``DOC-`` pattern only where a guard must catch them."""

import csv
import dataclasses
import io
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from thematic_redundancy.corpus.snapshot import SnapshotRecord
from thematic_redundancy.evaluation.expert_samples import (
    GAP_RULES,
    NO_ABSTRACT,
    RULE_TEXTS,
    SAMPLE_FILE_NAMES,
    VERDICT_PENDING,
    AssignmentCase,
    Candidate,
    ExpertSampleError,
    GapCase,
    GapLine,
    NameGuardError,
    NeighborLine,
    QueryCase,
    QuerySpace,
    RankedWork,
    Replacement,
    SampleSettings,
    Thesis,
    TopicCase,
    check_texts,
    composition,
    draw_assignments,
    draw_gaps,
    draw_topics,
    key_csv,
    key_rows,
    person_strings,
    render_samples,
    sample_generator,
    simulate_queries,
)
from thematic_redundancy.gaps.results_store import GapRow
from thematic_redundancy.recommendation.antecedents import rank_antecedents
from thematic_redundancy.redundancy.saturation import saturation_quartiles
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map.build import TopicMap
from thematic_redundancy.topic_map.ctfidf import Topic

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(PROJECT_ROOT / "config" / "default.yaml")
SETTINGS = SampleSettings.from_config(CONFIG)
PROGRAMS = tuple(program.key for program in CONFIG.snapshot.programs)
NAMES = {program.key: program.name for program in CONFIG.snapshot.programs}
MECANICA = NAMES["mecanica"]


def settings(**changes: object) -> SampleSettings:
    return dataclasses.replace(SETTINGS, **changes)


def words(code: str) -> str:
    """A code spelled without its hyphen, so a text never holds the code itself."""
    return code.lower().replace("-", " ")


def title(code: str) -> str:
    return f"Estudio inventado {words(code)}"


def thesis(code: str, program: str, year: int, **texts: object) -> Thesis:
    values: dict[str, object] = {
        "title": title(code),
        "abstract": f"Resumen inventado {words(code)}.",
        "objectives": f"Objetivos inventados {words(code)}.",
        "keywords": (f"clave {words(code)}", "inventada"),
    }
    values.update(texts)
    return Thesis(doc_code=code, program=program, year=year, **values)  # type: ignore[arg-type]


# --- Settings and generators -----------------------------------------------------------------


def test_the_settings_gather_the_block_the_programs_and_the_recommender() -> None:
    block = CONFIG.expert_samples

    assert SETTINGS.seed == block.seed == 20261031
    assert SETTINGS.programs == PROGRAMS
    assert SETTINGS.program_names == NAMES
    assert (SETTINGS.assignments_per_program, SETTINGS.group_examples) == (2, 3)
    assert (SETTINGS.queries_per_program, SETTINGS.ranking_size) == (1, 5)
    assert (SETTINGS.topics, SETTINGS.topic_examples) == (6, 5)
    assert (SETTINGS.gaps_per_rule, SETTINGS.gap_examples, SETTINGS.terms) == (2, 3, 10)
    assert (SETTINGS.saturation_method, SETTINGS.saturation_boundary) == ("linear", "lower")
    assert (SETTINGS.top_k, SETTINGS.mmr_lambda) == (5, 0.7)
    assert (SETTINGS.neighbor_topics, SETTINGS.nearby_gaps) == (3, 3)
    assert SETTINGS.record() == {
        **block.model_dump(),
        "top_k": 5,
        "mmr_lambda": 0.7,
        "neighbor_topics": 3,
        "nearby_gaps": 3,
    }


def test_each_sample_draws_from_its_own_generator() -> None:
    for index, sample in enumerate("ABCD", start=1):
        expected = np.random.default_rng([11, index]).integers(0, 10**9, 5)
        assert np.array_equal(sample_generator(11, sample).integers(0, 10**9, 5), expected)
    with pytest.raises(ExpertSampleError, match="sample"):
        sample_generator(11, "E")


# --- Sample A ---------------------------------------------------------------------------------


def a_inputs() -> tuple[list[Candidate], dict[int, list[str]], dict[int, tuple[str, ...]]]:
    # Three programs with 3, 1 and 0 candidates; groups 0 to 3 with 5, 2, 1 and 4 members.
    candidates = [
        Candidate("L-in-1", "industrial", 2025, 0, False),
        Candidate("L-in-2", "industrial", 2026, 1, True),
        Candidate("L-in-3", "industrial", 2025, 3, False),
        Candidate("L-si-1", "sistemas", 2026, 2, False),
        Candidate("L-mi-1", "minas", 2025, 0, True),
    ]
    members = {
        0: [f"E-0-{index}" for index in range(5)],
        1: ["E-1-0", "E-1-1"],
        2: ["E-2-0"],
        3: [f"E-3-{index}" for index in range(4)],
    }
    terms = {topic: tuple(f"term{topic}{index}" for index in range(12)) for topic in members}
    return candidates, members, terms


def a_saturation(members: dict[int, list[str]]) -> object:
    labels = [topic for topic, codes in members.items() for _ in codes]
    return saturation_quartiles(labels, method="linear", boundary="lower")


def a_oracle(
    candidates: list[Candidate],
    members: dict[int, list[str]],
    chosen_settings: SampleSettings,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """The documented scheme: per program in order, the candidates sorted by doc code in a
    random order, the first ``assignments_per_program``; then each case's examples; then the
    codes."""
    rng = np.random.default_rng([chosen_settings.seed, 1])
    drawn: list[Candidate] = []
    for program in chosen_settings.programs:
        pool = sorted((c for c in candidates if c.program == program), key=lambda c: c.doc_code)
        order = rng.permutation(len(pool))
        drawn += [pool[index] for index in order[: chosen_settings.assignments_per_program]]
    examples = []
    for case in drawn:
        group = sorted(members[case.topic])
        size = min(chosen_settings.group_examples, len(group))
        picked = rng.choice(len(group), size=size, replace=False)
        examples.append(tuple(group[index] for index in picked))
    positions = rng.permutation(len(drawn))
    cases = [
        (f"A{position + 1:02d}", case.doc_code, shown)
        for position, case, shown in zip(positions, drawn, examples, strict=True)
    ]
    return sorted(cases)


@pytest.mark.parametrize("per_program", [1, 2, 3])
def test_sample_a_takes_up_to_the_quota_per_program_by_the_documented_draw(
    per_program: int,
) -> None:
    candidates, members, terms = a_inputs()
    chosen = settings(seed=5, assignments_per_program=per_program)

    cases = draw_assignments(
        candidates,
        members=members,
        topic_terms=terms,
        saturation=a_saturation(members),
        settings=chosen,
    )

    by_program = {p: sum(case.program == p for case in cases) for p in PROGRAMS}
    assert by_program == {
        "sistemas": 1,
        "industrial": min(per_program, 3),
        "electronica": 0,
        "mecanica": 0,
        "minas": 1,
    }
    assert [(c.code, c.doc_code, c.examples) for c in cases] == a_oracle(
        candidates, members, chosen
    )


def test_a_case_shows_its_group_terms_examples_flag_and_saturation() -> None:
    candidates, members, terms = a_inputs()
    saturation = a_saturation(members)
    by_code = {c.doc_code: c for c in candidates}

    cases = draw_assignments(
        candidates,
        members=members,
        topic_terms=terms,
        saturation=saturation,
        settings=settings(seed=3, assignments_per_program=3, group_examples=3, terms=4),
    )

    assert len(cases) == 5
    assert [case.code for case in cases] == [f"A{index:02d}" for index in range(1, 6)]
    for case in cases:
        candidate = by_code[case.doc_code]
        assert (case.program, case.year, case.topic, case.atypical) == (
            candidate.program,
            candidate.year,
            candidate.topic,
            candidate.atypical,
        )
        assert case.terms == terms[case.topic][:4]
        assert len(case.examples) == min(3, len(members[case.topic]))
        assert len(set(case.examples)) == len(case.examples)
        assert set(case.examples) <= set(members[case.topic])
        assert case.saturation == saturation.of_topic(case.topic).level


def test_sample_a_is_reproducible_from_the_seed_and_changes_with_it() -> None:
    candidates = [
        Candidate(f"L-{index:03d}", PROGRAMS[index % 5], 2025, 0, False) for index in range(60)
    ]
    members = {0: [f"E-{index:03d}" for index in range(40)]}
    terms = {0: ("uno", "dos")}

    def draw(seed: int) -> tuple[AssignmentCase, ...]:
        return draw_assignments(
            candidates,
            members=members,
            topic_terms=terms,
            saturation=a_saturation(members),
            settings=settings(seed=seed),
        )

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)
    assert len(draw(1)) == 10


def test_sample_a_refuses_an_unknown_program_or_group() -> None:
    _, members, terms = a_inputs()
    saturation = a_saturation(members)
    for candidate, message in (
        (Candidate("L-x", "quimica", 2025, 0, False), "program"),
        (Candidate("L-y", "minas", 2025, 9, False), "group 9"),
    ):
        with pytest.raises(ExpertSampleError, match=message):
            draw_assignments(
                [candidate],
                members=members,
                topic_terms=terms,
                saturation=saturation,
                settings=SETTINGS,
            )


# --- Sample B ---------------------------------------------------------------------------------


def b_space() -> QuerySpace:
    """Six earlier theses on two groups (axes 0 and 1, 3 dimensions), four later ones."""
    codes = ("E0", "E1", "E2", "E3", "E4", "E5", "L0", "L1", "L2", "L3")
    vectors = np.array(
        [
            [1.0, 0.05, 0.0],
            [1.0, 0.10, 0.1],
            [1.0, 0.00, 0.3],
            [0.05, 1.0, 0.0],
            [0.10, 1.0, 0.2],
            [0.00, 1.0, 0.4],
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.1],
            [0.9, 0.3, 0.6],
            [0.1, 1.0, 0.5],
        ]
    )
    earlier = np.arange(6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    centroids = np.vstack([vectors[:3].mean(axis=0), vectors[3:6].mean(axis=0)])
    topic_map = TopicMap(
        seed=0,
        clustering=None,
        labels=labels,
        topics=(
            Topic(0, 3, ("cero", "nulo"), (0.2, 0.1)),
            Topic(1, 3, ("uno", "unidad"), (0.2, 0.1)),
        ),
        noise=None,
        centroids=centroids,
        coordinates=np.zeros((6, 2)),
        reduction=None,
    )
    return QuerySpace(
        doc_codes=codes,
        vectors=vectors,
        earlier=earlier,
        earlier_map=topic_map,
        full_map=topic_map,
        gaps=(),
        full_terms={0: ("cero",), 1: ("uno",)},
    )


def b_cases() -> list[AssignmentCase]:
    def case(code: str, doc: str, program: str, topic: int) -> AssignmentCase:
        return AssignmentCase(code, doc, program, 2025, topic, ("t",), (), False, "Q1")

    return [
        case("A01", "L0", "sistemas", 0),
        case("A02", "L1", "sistemas", 1),
        case("A03", "L2", "industrial", 0),
        case("A04", "L3", "electronica", 1),
    ]


def b_theses(**without_abstract: bool) -> dict[str, Thesis]:
    years = {"E": 2022, "L": 2025}
    result = {}
    for code in b_space().doc_codes:
        abstract = None if without_abstract.get(code) else f"Resumen inventado de {code}."
        result[code] = thesis(code, "industrial", years[code[0]], abstract=abstract)
    return result


def run_queries(
    theses: dict[str, Thesis], *, seed: int = 4, **changes: object
) -> tuple[tuple[QueryCase, ...], tuple[Replacement, ...]]:
    return simulate_queries(
        b_cases(),
        b_space(),
        theses=theses,
        settings=settings(seed=seed, ranking_size=3, top_k=2, neighbor_topics=1, **changes),
        neighbor_reference="proposal",
        pair_anchor="midpoint",
        max_distance=None,
        pool_size=None,
    )


def b_oracle(theses: dict[str, Thesis], seed: int) -> tuple[list[str], list[str]]:
    """The documented draw, one query per program: sample A's cases of the program, sorted by
    doc code, in a random order; the first with an abstract is taken, those before it are
    replaced."""
    rng = np.random.default_rng([seed, 2])
    chosen, replaced = [], []
    for program in PROGRAMS:
        pool = sorted(c.doc_code for c in b_cases() if c.program == program)
        for index in rng.permutation(len(pool)):
            code = pool[index]
            if theses[code].abstract is None:
                replaced.append(code)
                continue
            chosen.append(code)
            break
    return chosen, replaced


@pytest.mark.parametrize("seed", range(6))
def test_sample_b_takes_one_usable_case_per_program_and_records_the_skipped(seed: int) -> None:
    theses = b_theses(L1=True)

    queries, replacements = run_queries(theses, seed=seed)

    chosen, replaced = b_oracle(theses, seed)
    assert sorted(query.doc_code for query in queries) == sorted(chosen)
    assert [r.doc_code for r in replacements] == replaced
    assert all(r == Replacement("B", "L1", "sistemas", NO_ABSTRACT) for r in replacements)
    assert {query.source for query in queries} <= {"A01", "A02", "A03", "A04"}
    assert sorted(query.code for query in queries) == ["B01", "B02", "B03"]


def test_a_program_whose_cases_lack_abstracts_has_no_query() -> None:
    queries, replacements = run_queries(b_theses(L0=True, L1=True))

    assert {query.program for query in queries} == {"industrial", "electronica"}
    assert sorted(r.doc_code for r in replacements) == ["L0", "L1"]


def test_a_query_ranks_only_the_earlier_theses_by_cosine_never_itself() -> None:
    space = b_space()
    queries, _ = run_queries(b_theses(), queries_per_program=1)

    unit = space.vectors / np.linalg.norm(space.vectors, axis=1, keepdims=True)
    for query in queries:
        row = space.doc_codes.index(query.doc_code)
        scores = unit[:6] @ unit[row]
        best = np.argsort(-scores, kind="stable")[:3]
        assert [work.doc_code for work in query.ranking] == [space.doc_codes[i] for i in best]
        assert [work.position for work in query.ranking] == [1, 2, 3]
        assert [work.similarity for work in query.ranking] == pytest.approx(scores[best].tolist())
        assert query.doc_code not in {work.doc_code for work in query.ranking}
        assert all(work.doc_code.startswith("E") for work in query.ranking)


def test_a_query_reuses_t26s_antecedents_and_neighbors_on_the_earlier_map() -> None:
    space = b_space()
    queries, _ = run_queries(b_theses())

    for query in queries:
        row = space.doc_codes.index(query.doc_code)
        ranking = rank_antecedents(
            space.vectors[:6], space.vectors[row], k=2, mmr_lambda=0.7, pool_size=None, exclude=None
        )
        assert query.antecedents == tuple(space.doc_codes[a.index] for a in ranking.antecedents)
        terms = {0: ("cero", "nulo"), 1: ("uno", "unidad")}
        assert query.terms == terms[query.topic]
        assert query.neighbors == (NeighborLine(1 - query.topic, terms[1 - query.topic]),)
        assert query.gaps == ()


def test_a_query_whose_group_differs_from_t27s_is_refused() -> None:
    cases = b_cases()
    cases[2] = dataclasses.replace(cases[2], topic=1)

    with pytest.raises(ExpertSampleError, match="T23"):
        simulate_queries(
            cases,
            b_space(),
            theses=b_theses(),
            settings=settings(queries_per_program=1),
            neighbor_reference="proposal",
            pair_anchor="midpoint",
            max_distance=None,
            pool_size=None,
        )


def test_a_query_thesis_among_the_candidates_is_refused() -> None:
    space = dataclasses.replace(b_space(), earlier=np.arange(8))

    with pytest.raises(ExpertSampleError, match="candidates"):
        simulate_queries(
            b_cases(),
            space,
            theses=b_theses(),
            settings=SETTINGS,
            neighbor_reference="proposal",
            pair_anchor="midpoint",
            max_distance=None,
            pool_size=None,
        )


# --- Sample C ---------------------------------------------------------------------------------


def c_inputs() -> tuple[dict[str, int], dict[int, tuple[str, ...]], dict[str, Thesis]]:
    labels: dict[str, int] = {}
    theses: dict[str, Thesis] = {}
    sizes = {0: 7, 1: 3, 2: 9, 3: 4, 4: 6, 5: 2, 6: 5, 7: 8, -1: 6}
    for topic, size in sizes.items():
        for index in range(size):
            code = f"T{topic + 1}-{index:02d}"
            labels[code] = topic
            theses[code] = thesis(code, PROGRAMS[(index + topic) % 5], 2021 + index % 6)
    terms = {topic: tuple(f"tema{topic}_{i}" for i in range(10)) for topic in sizes if topic >= 0}
    return labels, terms, theses


def test_sample_c_draws_groups_but_never_the_noise_with_random_members() -> None:
    labels, terms, theses = c_inputs()
    chosen = settings(seed=9, topics=6, topic_examples=5, terms=3)

    cases = draw_topics(labels, terms, theses=theses, settings=chosen)

    rng = np.random.default_rng([9, 3])
    topics = list(range(8))
    drawn = [topics[i] for i in rng.permutation(8)[:6]]
    examples = []
    for topic in drawn:
        group = sorted(code for code, label in labels.items() if label == topic)
        picked = rng.choice(len(group), size=min(5, len(group)), replace=False)
        examples.append(tuple(group[i] for i in picked))
    positions = rng.permutation(6)
    expected = sorted(
        (f"C{position + 1:02d}", topic, shown)
        for position, topic, shown in zip(positions, drawn, examples, strict=True)
    )
    assert [(case.code, case.topic, case.examples) for case in cases] == expected
    for case in cases:
        group = [code for code, label in labels.items() if label == case.topic]
        assert case.size == len(group)
        counts = {p: sum(theses[c].program == p for c in group) for p in PROGRAMS}
        assert case.programs == tuple((p, n) for p, n in counts.items() if n)
        assert case.terms == terms[case.topic][:3]


def test_sample_c_takes_every_group_when_fewer_than_asked() -> None:
    labels, terms, theses = c_inputs()

    cases = draw_topics(labels, terms, theses=theses, settings=settings(topics=20))

    assert sorted(case.topic for case in cases) == list(range(8))


def test_sample_c_refuses_a_group_without_terms() -> None:
    labels, terms, theses = c_inputs()
    del terms[3]

    with pytest.raises(ExpertSampleError, match="group 3"):
        draw_topics(labels, terms, theses=theses, settings=settings(topics=8))


# --- Sample D ---------------------------------------------------------------------------------


def gap_row(rule: str, gap_id: str, topics: tuple[int, ...], **changes: object) -> GapRow:
    values: dict[str, object] = {
        "gap_id": gap_id,
        "run_id": "run42",
        "seed": 42,
        "reference": True,
        "rule": rule,
        "topics": topics,
        "theses": 12,
        "programs": ("industrial", "minas"),
        "program_counts": (4, 8),
        "years": (2022, 2023),
        "year_counts": (5, 7),
        "support": 5,
        "stable": True,
        "reason": "an invented reason",
        # The evidence columns of every rule, empty: the samples never read them.
        **{
            name: None
            for name in GapRow.model_fields
            if name
            not in {
                "gap_id",
                "run_id",
                "seed",
                "reference",
                "rule",
                "topics",
                "theses",
                "programs",
                "program_counts",
                "years",
                "year_counts",
                "support",
                "stable",
                "reason",
            }
        },
    }
    if rule == "c":
        values.update(source="minas", target="mecanica")
    values.update(changes)
    return GapRow(**values)  # type: ignore[arg-type]


def d_inputs() -> tuple[list[GapRow], dict[str, list[str]], dict[int, tuple[str, ...]]]:
    rows = [
        gap_row("a", "g-a-1", (1,)),
        gap_row("b", "g-b-1", (2, 3)),
        gap_row("c", "g-c-1", (4,)),
        gap_row("c", "g-c-2", (5,)),
        gap_row("c", "g-c-3", (6,)),
        gap_row("d", "g-d-1", (1,)),
    ]
    members = {row.gap_id: [f"M-{row.gap_id}-{i}" for i in range(6)] for row in rows}
    members["g-a-1"] = ["M-only"]
    terms = {topic: tuple(f"vacio{topic}_{i}" for i in range(10)) for topic in range(1, 7)}
    return rows, members, terms


def test_sample_d_takes_up_to_the_quota_per_rule_by_the_documented_draw() -> None:
    rows, members, terms = d_inputs()
    chosen = settings(seed=12, gaps_per_rule=2, gap_examples=3, terms=2)

    cases = draw_gaps(rows, members, topic_terms=terms, settings=chosen)

    rng = np.random.default_rng([12, 4])
    drawn: list[GapRow] = []
    for rule in GAP_RULES:
        pool = sorted((row for row in rows if row.rule == rule), key=lambda row: row.gap_id)
        drawn += [pool[i] for i in rng.permutation(len(pool))[:2]]
    examples = []
    for row in drawn:
        group = sorted(members[row.gap_id])
        picked = rng.choice(len(group), size=min(3, len(group)), replace=False)
        examples.append(tuple(group[i] for i in picked))
    positions = rng.permutation(len(drawn))
    expected = sorted(
        (f"D{position + 1:02d}", row.gap_id, shown)
        for position, row, shown in zip(positions, drawn, examples, strict=True)
    )
    assert [(case.code, case.gap_id, case.examples) for case in cases] == expected
    assert {rule: sum(c.rule == rule for c in cases) for rule in GAP_RULES} == {
        "a": 1,
        "b": 1,
        "c": 2,
        "d": 1,
    }
    for case in cases:
        row = next(r for r in rows if r.gap_id == case.gap_id)
        assert case.topics == row.topics
        assert case.topic_terms == tuple(terms[topic][:2] for topic in row.topics)
        assert case.programs == (("industrial", 4), ("minas", 8))
        assert case.years == ((2022, 5), (2023, 7))
        assert case.theses == 12
        assert (case.source, case.target) == (
            ("minas", "mecanica") if case.rule == "c" else (None, None)
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [({"stable": False}, "stable"), ({"stable": None}, "stable"), ({"reference": False}, "ref")],
)
def test_sample_d_refuses_a_gap_that_is_not_a_stable_gap_of_the_reference_run(
    changes: dict[str, object], message: str
) -> None:
    rows, members, terms = d_inputs()
    rows[0] = rows[0].model_copy(update=changes)

    with pytest.raises(ExpertSampleError, match=message):
        draw_gaps(rows, members, topic_terms=terms, settings=SETTINGS)


def test_sample_d_refuses_a_gap_without_members() -> None:
    rows, members, terms = d_inputs()
    members["g-b-1"] = []

    with pytest.raises(ExpertSampleError, match="g-b-1"):
        draw_gaps(rows, members, topic_terms=terms, settings=SETTINGS)


# --- Texts ------------------------------------------------------------------------------------


def text_inputs() -> dict[str, object]:
    theses = {
        "L-1": thesis(
            "L-1",
            "industrial",
            2025,
            title="Una propuesta | con barra   y\n  espacios",
            keywords=("logística", "almacén"),
        ),
        "L-2": thesis("L-2", "minas", 2026, objectives=None, keywords=()),
        "E-1": thesis("E-1", "mecanica", 2022),
        "E-2": thesis("E-2", "sistemas", 2023),
        "E-3": thesis("E-3", "electronica", 2021),
    }
    assignments = [
        AssignmentCase(
            "A01", "L-1", "industrial", 2025, 4, ("cadena", "suministro"), ("E-1",), True, "Q2"
        ),
        AssignmentCase("A02", "L-2", "minas", 2026, 0, ("mineral",), ("E-2", "E-3"), False, None),
    ]
    queries = [
        QueryCase(
            code="B01",
            doc_code="L-2",
            source="A02",
            program="minas",
            year=2026,
            topic=0,
            terms=("mineral",),
            ranking=(RankedWork(1, "E-2", 0.81234), RankedWork(2, "E-1", 0.5)),
            antecedents=("E-2", "E-3"),
            neighbors=(NeighborLine(3, ("vecino", "tema")),),
            gaps=(GapLine("c", (("transferible",),)), GapLine("b", (("lado",), ("otro",)))),
        )
    ]
    topics = [TopicCase("C01", 2, 9, (("industrial", 7), ("minas", 2)), ("grupo",), ("E-1",))]
    gaps = [
        GapCase(
            code="D01",
            gap_id="g-c",
            rule="c",
            topics=(5,),
            topic_terms=(("vacío", "tema"),),
            programs=(("minas", 8), ("industrial", 1)),
            years=((2022, 9),),
            source="minas",
            target="mecanica",
            theses=9,
            examples=("E-3",),
        )
    ]
    return {
        "assignments": assignments,
        "queries": queries,
        "topics": topics,
        "gaps": gaps,
        "theses": theses,
    }


def render(provisional: bool = True) -> dict[str, str]:
    data = text_inputs()
    return render_samples(
        data["assignments"],  # type: ignore[arg-type]
        data["queries"],  # type: ignore[arg-type]
        data["topics"],  # type: ignore[arg-type]
        data["gaps"],  # type: ignore[arg-type]
        theses=data["theses"],  # type: ignore[arg-type]
        settings=SETTINGS,
        earlier_years=(2021, 2024),
        later_years=(2025, 2026),
        provisional=provisional,
    )


def test_each_sample_has_its_own_printable_file() -> None:
    texts = render()

    assert (
        set(texts)
        == set(SAMPLE_FILE_NAMES.values())
        == {
            "muestra_a.md",
            "muestra_b.md",
            "muestra_c.md",
            "muestra_d.md",
        }
    )
    for text in texts.values():
        assert "Juego n.° ________" in text
        assert "Resultados provisionales" in text


def test_sample_a_shows_each_case_as_the_kit_specifies() -> None:
    text = render()["muestra_a.md"]

    assert "# Muestra A. Asignaciones de tesis de 2025–2026 al mapa de 2021–2024" in text
    assert "## Caso A01" in text and "## Caso A02" in text
    assert text.index("## Caso A01") < text.index("## Caso A02")
    assert "- **Carrera y año:** Ingeniería Industrial, 2025" in text
    assert "- **Título:** Una propuesta | con barra y espacios" in text
    assert "- **Palabras clave:** logística; almacén" in text
    assert "- **Grupo asignado:** 4. Términos: cadena, suministro" in text
    assert "- **Tesis del grupo (2021–2024):**" in text
    assert f"  - {title('E-1')} ({MECANICA}, 2022)" in text
    assert "- **Propuesta atípica:** Sí" in text and "- **Propuesta atípica:** No" in text
    assert "- **Saturación del grupo:** Q2" in text
    assert "- **Saturación del grupo:** No definida" in text
    assert "- **Palabras clave:** Sin palabras clave en la ficha" in text
    assert "Resumen inventado" not in text and "Objetivos inventados" not in text
    assert text.index("Una propuesta") < text.index(title("E-1"))


def test_sample_b_shows_the_ranking_without_a_verdict() -> None:
    text = render()["muestra_b.md"]

    assert "## Caso B01" in text
    assert "**Propuesta** (Ingeniería de Minas, 2026)" in text
    assert "- **Resumen:** Resumen inventado l 2." in text
    assert "- **Palabras clave:** Sin palabras clave en la ficha" in text
    assert "- **Objetivos:** No disponibles en la ficha" in text
    assert "**Grupo asignado:** 0. Términos: mineral" in text
    assert f"| 1 | {title('E-2')} | Ingeniería de Sistemas | 2023 | 0.81 |" in text
    assert f"| 2 | {title('E-1')} | {MECANICA} | 2022 | 0.50 |" in text
    assert f"**Veredicto de redundancia:** {VERDICT_PENDING}" in text
    assert f"1. {title('E-2')} (Ingeniería de Sistemas, 2023)" in text
    assert f"2. {title('E-3')} (Ingeniería Electrónica, 2021)" in text
    assert "- Grupo vecino 3. Términos: vecino, tema" in text
    assert f"- Vacío cercano. {RULE_TEXTS['c']} Términos: transferible" in text
    assert f"- Vacío cercano. {RULE_TEXTS['b']} Términos: lado / otro" in text
    for word in ("Redundante", "No redundante", "Relacionado", "No relacionado", "τ ="):
        assert word not in text


def test_a_title_in_a_table_cannot_break_its_row() -> None:
    data = text_inputs()
    theses = dict(data["theses"])  # type: ignore[call-overload]
    theses["E-2"] = thesis("E-2", "sistemas", 2023, title="Dos | columnas\ny una línea")
    texts = render_samples(
        data["assignments"],  # type: ignore[arg-type]
        data["queries"],  # type: ignore[arg-type]
        data["topics"],  # type: ignore[arg-type]
        data["gaps"],  # type: ignore[arg-type]
        theses=theses,
        settings=SETTINGS,
        earlier_years=(2021, 2024),
        later_years=(2025, 2026),
        provisional=False,
    )

    row = "| 1 | Dos \\| columnas y una línea | Ingeniería de Sistemas | 2023 | 0.81 |"
    assert row in texts["muestra_b.md"]


def test_samples_c_and_d_show_groups_and_gaps_as_the_kit_specifies() -> None:
    texts = render()
    topics, gaps = texts["muestra_c.md"], texts["muestra_d.md"]

    assert "## Grupo C01" in topics
    assert (
        "- **Tesis en el grupo:** 9. Por carrera: Ingeniería Industrial 7; Ingeniería de Minas 2"
    ) in topics
    assert "- **Términos representativos:** grupo" in topics
    assert f"| {title('E-1')} | {MECANICA} | 2022 |" in topics
    assert "## Vacío D01" in gaps
    assert f"- **Tipo:** {RULE_TEXTS['c']}" in gaps
    assert (
        "- **Carreras:** Ingeniería de Minas (8), Ingeniería Industrial (1); presente en "
        f"Ingeniería de Minas y ausente en {MECANICA}. **Años:** 2022 (9)"
    ) in gaps
    assert "- **Grupos y términos:** Grupo 5: vacío, tema" in gaps
    assert "- **Tesis en el vacío:** 9" in gaps
    assert f"  - {title('E-3')} (Ingeniería Electrónica, 2021)" in gaps


def test_only_sample_b_shows_abstracts_and_no_sample_shows_a_code() -> None:
    texts = render()

    for name, text in texts.items():
        assert ("Resumen inventado" in text) == (name == "muestra_b.md")
        assert ("Objetivos" in text) == (name == "muestra_b.md")
        for code in ("L-1", "L-2", "E-1", "E-2", "E-3", "g-c"):
            assert code not in text


def test_the_provisional_notice_appears_only_while_the_winner_is_provisional() -> None:
    assert all("Resultados provisionales" not in text for text in render(False).values())


def test_the_rule_texts_are_the_kits_fixed_sentences() -> None:
    kit = (PROJECT_ROOT / "docs" / "expert_session" / "muestras_anonimizadas.md").read_text(
        encoding="utf-8"
    )

    for rule, text in RULE_TEXTS.items():
        assert f"| ({rule}) | {text} |" in kit
    assert f'"{VERDICT_PENDING}"' in kit.replace("\n", " ")


# --- Key and composition ----------------------------------------------------------------------


def test_the_key_links_every_case_to_what_it_shows_and_records_the_replacements() -> None:
    data = text_inputs()
    replaced = [Replacement("B", "L-9", "mecanica", NO_ABSTRACT)]

    rows = key_rows(
        data["assignments"],  # type: ignore[arg-type]
        data["queries"],  # type: ignore[arg-type]
        data["topics"],  # type: ignore[arg-type]
        data["gaps"],  # type: ignore[arg-type]
        replaced,
    )
    table = list(csv.DictReader(io.StringIO(key_csv(rows))))

    assert [(row["sample"], row["code"]) for row in table] == [
        ("A", "A01"),
        ("A", "A02"),
        ("B", "B01"),
        ("C", "C01"),
        ("D", "D01"),
        ("B", ""),
    ]
    assert table[0] == {
        "sample": "A",
        "code": "A01",
        "doc_code": "L-1",
        "topic": "4",
        "gap_id": "",
        "source": "",
        "shown": "E-1",
        "note": "",
    }
    assert table[2]["source"] == "A02"
    assert table[2]["shown"] == "E-2;E-1;E-3"
    assert (table[3]["topic"], table[3]["shown"]) == ("2", "E-1")
    assert (table[4]["gap_id"], table[4]["topic"]) == ("g-c", "5")
    assert table[5]["doc_code"] == "L-9" and table[5]["note"] == NO_ABSTRACT


def test_the_composition_holds_counts_only() -> None:
    data = text_inputs()

    result = composition(
        data["assignments"],  # type: ignore[arg-type]
        data["queries"],  # type: ignore[arg-type]
        data["topics"],  # type: ignore[arg-type]
        data["gaps"],  # type: ignore[arg-type]
        [Replacement("B", "L-9", "mecanica", NO_ABSTRACT)],
        programs=PROGRAMS,
    )

    assert result == {
        "A": {"cases": 2, "programs": dict.fromkeys(PROGRAMS, 0) | {"industrial": 1, "minas": 1}},
        "B": {
            "cases": 1,
            "programs": dict.fromkeys(PROGRAMS, 0) | {"minas": 1},
            "replaced": 1,
        },
        "C": {"cases": 1},
        "D": {"cases": 1, "rules": {"a": 0, "b": 0, "c": 1, "d": 0}},
    }


# --- Guards -----------------------------------------------------------------------------------


def snapshot_record(metadata: dict[str, list[dict[str, object]]]) -> SnapshotRecord:
    return SnapshotRecord(
        uuid=UUID(int=7),
        handle="20.500.0/1",
        program_key="industrial",
        collection_uuid=UUID(int=8),
        harvested_at=datetime(2026, 10, 2, tzinfo=UTC),
        metadata=metadata,  # type: ignore[arg-type]
    )


def test_the_person_strings_are_the_names_and_orcids_without_titles_or_abstracts() -> None:
    record = snapshot_record(
        {
            "dc.title": [{"value": "Un título inventado de prueba"}],
            "dc.description.abstract": [{"value": "Un resumen inventado que no se busca."}],
            "dc.contributor.author": [{"value": "Plantado Ficticio, Nombre Inventado"}],
            "dc.contributor.advisor": [{"value": "Asesora Supuesta, Otra"}],
            "renati.juror": [{"value": "Jurado Imaginario, Tercero"}],
            "renati.advisor.orcid": [{"value": "https://orcid.org/0000-0002-1825-0097"}],
        }
    )

    found = person_strings([record])

    assert "nombre inventado plantado ficticio" in found
    assert "plantado ficticio, nombre inventado" in found
    assert "plantado ficticio" in found
    assert "otra asesora supuesta" in found and "tercero jurado imaginario" in found
    assert any("0000-0002-1825-0097" in item for item in found)
    assert not any("título inventado" in item or "resumen inventado" in item for item in found)


def test_the_guard_stops_on_a_name_even_glued_to_other_letters() -> None:
    persons = frozenset({"nombre inventado plantado"})
    texts = {"muestra_a.md": "Título: estudio deNombre  Inventado Plantados", "muestra_b.md": "ok"}

    with pytest.raises(NameGuardError) as caught:
        check_texts(texts, persons=persons, identifiers=frozenset())

    message = str(caught.value)
    assert "muestra_a.md (1)" in message and "muestra_b.md" not in message
    assert "inventado" not in message.casefold()


@pytest.mark.parametrize(
    "planted",
    [
        "DOC-0a1b2c3d",
        "doc-0A1B2C3D",
        "123e4567-e89b-12d3-a456-426614174000",
        "https://repositorio.example/handle/1",
        "http://otro.example",
        "www.example.org",
        "codigo-planteado-1",
    ],
)
def test_the_guard_stops_on_an_identifier_or_a_link(planted: str) -> None:
    texts = {"muestra_c.md": f"Un texto con {planted} dentro."}

    with pytest.raises(NameGuardError, match=r"muestra_c.md \(1\)"):
        check_texts(texts, persons=frozenset(), identifiers=frozenset({"codigo-planteado-1"}))


def test_the_guard_lets_clean_texts_pass() -> None:
    texts = render()

    check_texts(
        texts,
        persons=frozenset({"nombre inventado plantado"}),
        identifiers=frozenset({"DOC-0a1b2c3d"}),
    )
