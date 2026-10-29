"""The expert samples' command (T31), through ``main``, on a synthetic project written under
pytest's temporary directory, outside the repository.

The scenario holds 148 theses in 10 dimensions, each near the axis of its topic (a seeded jitter
of 0.01): six big topics (A, B, F, G, H, I) of 20 theses, 16 Industrial and one of each other
program, half of 2022 and half of 2025, the four other programs in one order for A, F and H and
in the reverse order for B, G and I, so every program has later theses; three small topics, C (8
Minas and 1 Industrial, 2022), D (5 Sistemas and 2 Mecánica, 2021 and 2024) and E (6
Electrónica, 2025); and 6 Industrial bridge theses midway between A and B (2023 and 2026).
Every Mecánica thesis of 2025-2026 lacks an abstract, so sample B can never draw one.

- T19's records: the winner C4's five seed runs share the partition, except that seeds 7, 13
  and 21 merge D into B, the medoid run 42 leaves E as noise and seed 73 leaves C as noise;
  seeds 13 and 73 number their topics in reverse. Their terms start with ``zq``.
- T24a's records: written by T24a's own command on T19's records.
- T27's records: written with T27's store. The earlier map labels the earlier theses by their
  topic, numbered in reverse (I is 0, A is 7), bridges as noise; its terms start with ``zr``. The
  later theses are assigned to it with T23's functions, flagged atypical with O06's threshold,
  and T27's own draw gives an expert sample of 40, 10 atypical, at least 2 per program and pool.
  A second temporal run (seed 7) labels every thesis 0, so a reader that took it would fail.
"""

import csv
import dataclasses
import hashlib
import io
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from scipy import sparse

from thematic_redundancy.evaluation.expert_samples import (
    NO_ABSTRACT,
    RULE_TEXTS,
    SAMPLE_FILE_NAMES,
    VERDICT_PENDING,
    Thesis,
)
from thematic_redundancy.evaluation.run_expert_samples import (
    COMPOSITION_FILE_NAME,
    KEY_FILE_NAME,
    MANIFEST_FILE_NAME,
    SampleInputs,
    SamplesManifest,
    TemporalRecords,
    main,
    read_temporal,
)
from thematic_redundancy.gaps.results_store import GapRow, MemberRow
from thematic_redundancy.gaps.rules import ThesisMetadata
from thematic_redundancy.gaps.run_gaps import CorpusInputs, read_experiment
from thematic_redundancy.gaps.run_gaps import main as run_gaps
from thematic_redundancy.recommendation.antecedents import rank_antecedents
from thematic_redundancy.redundancy.saturation import saturation_quartiles
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.assignment import (
    ATYPICAL_PERCENTILE,
    TopicCentroids,
    assign_proposals,
    atypical_threshold,
    check_atypical,
)
from thematic_redundancy.topic_map.build import topic_centroids
from thematic_redundancy.topic_map.experiment import (
    ExperimentInputs,
    SemanticInput,
    TopicRow,
    run_id,
)
from thematic_redundancy.topic_map.results_store import (
    LABELS_FILE_NAME,
    SELECTION_FILE_NAME,
    TOPICS_FILE_NAME,
    ExperimentManifest,
    LabelRow,
    processed_dir,
    results_dir,
)
from thematic_redundancy.topic_map.results_store import (
    MANIFEST_FILE_NAME as T19_MANIFEST,
)
from thematic_redundancy.topic_map.run_experiment import GitState
from thematic_redundancy.topic_map.temporal import draw_expert_sample
from thematic_redundancy.topic_map.temporal_store import (
    ASSIGNMENTS_FILE_NAME,
    SAMPLE_FILE_NAME,
    AssignmentRow,
    SampleRow,
    TemporalManifest,
    TemporalTopicRow,
    write_outputs,
)
from thematic_redundancy.topic_map.temporal_store import (
    processed_dir as temporal_processed_dir,
)
from thematic_redundancy.topic_map.temporal_store import results_dir as temporal_results_dir

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(PROJECT_ROOT / "config" / "default.yaml")
MODELS = CONFIG.embedding.models
WINNER_MODEL = MODELS[1]
SEEDS = CONFIG.seeds
SNAPSHOT = "S1"
DIMENSIONS = 10
PROGRAMS = tuple(program.key for program in CONFIG.snapshot.programs)
PROGRAM_NAMES = {program.key: program.name for program in CONFIG.snapshot.programs}
NAMES = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
BIG = ("A", "B", "F", "G", "H", "I")
OTHERS = ("sistemas", "electronica", "mecanica", "minas")
NPMI = {"A": 0.08, "B": 0.10, "F": 0.12, "G": 0.14, "H": 0.16, "I": 0.18}
T19_INPUTS = {
    "embeddings_manifest.json": "e" * 64,
    "tfidf_manifest.json": "t" * 64,
    "fichas_manifest.json": "f" * 64,
}
PERSON = "Nombre Inventado Plantado"
SENSITIVE = frozenset({"un título que no aparece en el corpus", PERSON.casefold()})
MOMENT = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
TEMPORAL_SUFFIX = "__fit2021-2024"
SAMPLES_DIR = Path("data") / "expert_session" / SNAPSHOT / "muestras"
OUTPUTS = {*SAMPLE_FILE_NAMES.values(), KEY_FILE_NAME, COMPOSITION_FILE_NAME, MANIFEST_FILE_NAME}


def winner_runs() -> tuple[str, ...]:
    return tuple(run_id(WINNER_MODEL, "hdbscan", 10, seed) for seed in SEEDS)


REFERENCE_RUN = winner_runs()[SEEDS.index(42)]


def full_terms(name: str) -> tuple[str, ...]:
    return tuple(f"zq{name.lower()}{index}" for index in range(12))


def earlier_terms(name: str) -> tuple[str, ...]:
    return tuple(f"zr{name.lower()}{index}" for index in range(12))


# --- The scenario -----------------------------------------------------------------------------


@dataclass
class Scenario:
    doc_codes: tuple[str, ...]
    names: tuple[str | None, ...]
    programs: tuple[str, ...]
    years: tuple[int, ...]
    vectors: np.ndarray
    other_vectors: np.ndarray
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    topic_names: dict[str, dict[int, str]] = field(default_factory=dict)

    @property
    def earlier(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.years) < 2025)

    @property
    def later(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.years) >= 2025)

    @property
    def earlier_ids(self) -> dict[str, int]:
        """The earlier map's topic of each base topic: reverse order, E left out."""
        present = [name for name in reversed(NAMES) if name != "E"]
        return {name: topic for topic, name in enumerate(present)}

    def earlier_labels(self) -> np.ndarray:
        ids = self.earlier_ids
        return np.array(
            [-1 if self.names[row] is None else ids[self.names[row]] for row in self.earlier]
        )


def make_scenario() -> Scenario:
    rng = np.random.default_rng(31)
    names: list[str | None] = []
    programs: list[str] = []
    years: list[int] = []
    axes: list[np.ndarray] = []
    eye = np.eye(DIMENSIONS)

    def add(name: str | None, program: str, year: int, axis: np.ndarray) -> None:
        names.append(name)
        programs.append(program)
        years.append(year)
        axes.append(axis)

    for position, name in enumerate(BIG):
        others = OTHERS if position % 2 == 0 else OTHERS[::-1]
        for index, program in enumerate(("industrial",) * 16 + others):
            add(name, program, 2022 if index % 2 == 0 else 2025, eye[NAMES.index(name)])
    for index in range(9):
        add("C", "minas" if index < 8 else "industrial", 2022, eye[NAMES.index("C")])
    for index in range(7):
        add("D", "sistemas" if index < 5 else "mecanica", 2021 if index < 4 else 2024, eye[3])
    for _ in range(6):
        add("E", "electronica", 2025, eye[NAMES.index("E")])
    for index in range(6):
        add(None, "industrial", 2023 if index < 3 else 2026, (eye[0] + eye[1]) / np.sqrt(2))
    vectors = np.vstack(axes) + rng.normal(0, 0.01, (len(axes), DIMENSIONS))
    scenario = Scenario(
        doc_codes=tuple(f"DOC-{index:08x}" for index in range(len(names))),
        names=tuple(names),
        programs=tuple(programs),
        years=tuple(years),
        vectors=vectors.astype(np.float32),
        other_vectors=rng.normal(size=(len(names), 12)).astype(np.float32),
    )
    for seed, run in zip(SEEDS, winner_runs(), strict=True):
        assigned = [run_name(name, seed) for name in names]
        present = [name for name in NAMES if name in assigned]
        if seed in (13, 73):
            present.reverse()
        ids = {name: topic for topic, name in enumerate(present)}
        scenario.labels[run] = np.array([-1 if n is None else ids[n] for n in assigned])
        scenario.topic_names[run] = {topic: name for name, topic in ids.items()}
    return scenario


def run_name(name: str | None, seed: int) -> str | None:
    if name == "D" and seed in (7, 13, 21):
        return "B"
    if (name == "E" and seed == 42) or (name == "C" and seed == 73):
        return None
    return name


SCENARIO = make_scenario()


def title_of(row: int) -> str:
    return f"Estudio sintético número {row} del corpus inventado"


def theses_of(scenario: Scenario, **titles: str) -> dict[str, Thesis]:
    result = {}
    for row, code in enumerate(scenario.doc_codes):
        later_mecanica = scenario.programs[row] == "mecanica" and scenario.years[row] >= 2025
        result[code] = Thesis(
            doc_code=code,
            program=scenario.programs[row],
            year=scenario.years[row],
            title=titles.get(code, title_of(row)),
            abstract=None if later_mecanica else f"Resumen sintético del estudio {row}.",
            objectives=None if row % 3 == 0 else f"Objetivos sintéticos del estudio {row}.",
            keywords=(f"clave{row}", "sintética") if row % 4 else (),
        )
    return result


def experiment_inputs(scenario: Scenario) -> ExperimentInputs:
    count = len(scenario.doc_codes)
    return ExperimentInputs(
        snapshot_id=SNAPSHOT,
        doc_codes=scenario.doc_codes,
        models=(
            SemanticInput(MODELS[0], "rev-a", scenario.other_vectors),
            SemanticInput(WINNER_MODEL, "rev-b", scenario.vectors),
        ),
        tfidf=sparse.csr_matrix(np.ones((count, 1))),
        vocabulary=("x",),
        documents=tuple(((("x",),),) * count),
        embedding_variant="section_mean",
        tfidf_variant="section_mean",
        input_sha256=dict(T19_INPUTS),
    )


def metadata_of(scenario: Scenario) -> ThesisMetadata:
    return ThesisMetadata(scenario.doc_codes, scenario.programs, scenario.years)


def sample_inputs(scenario: Scenario = SCENARIO, **overrides: Any) -> SampleInputs:
    arguments: dict[str, Any] = {
        "experiment": experiment_inputs(scenario),
        "theses": theses_of(scenario),
        "metadata": metadata_of(scenario),
        "persons": frozenset({PERSON.casefold()}),
        "sensitive": SENSITIVE,
        "identifiers": frozenset(
            [*scenario.doc_codes, *(f"20.500.0/{row}" for row in range(len(scenario.doc_codes)))]
        ),
        "input_sha256": {"metadata.jsonl": "m" * 64},
    }
    arguments.update(overrides)
    return SampleInputs(**arguments)


# --- The records ------------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_experiment(root: Path, scenario: Scenario, *, provisional: bool = True) -> None:
    """Write T19's records of ``scenario`` under ``root``, as T19's store writes them."""
    store = ParquetTableStore()
    results = results_dir(root / "results", SNAPSHOT)
    processed = processed_dir(root / "data", SNAPSHOT)
    results.mkdir(parents=True)
    processed.mkdir(parents=True)
    topic_rows, label_rows = [], []
    for run in winner_runs():
        labels = scenario.labels[run]
        for topic, name in sorted(scenario.topic_names[run].items()):
            topic_rows.append(
                TopicRow(
                    run_id=run,
                    topic=topic,
                    size=int(np.count_nonzero(labels == topic)),
                    terms=full_terms(name)[:10],
                    scores=tuple(0.1 - 0.001 * index for index in range(10)),
                    npmi=NPMI.get(name, 0.30),
                    npmi_sensitivity=NPMI.get(name, -0.5),
                    absent_terms=(),
                )
            )
        label_rows += [
            LabelRow(run_id=run, doc_code=code, label=int(label))
            for code, label in zip(scenario.doc_codes, labels, strict=True)
        ]
    store.write(topic_rows, results / TOPICS_FILE_NAME)
    store.write(label_rows, processed / LABELS_FILE_NAME)
    selection = {
        "winner": {
            "configuration": "C4",
            "representation": WINNER_MODEL,
            "algorithm": "hdbscan",
            "parameter": 10,
            "provisional": provisional,
        },
        "representative": {"seed": 42, "run_id": REFERENCE_RUN},
    }
    (results / SELECTION_FILE_NAME).write_text(json.dumps(selection), encoding="utf-8")
    manifest = ExperimentManifest(
        snapshot_id=SNAPSHOT,
        declaration_sha256=CONFIG.experiment_declaration_sha256(),
        git_commit="b" * 40,
        inputs=dict(T19_INPUTS),
        documents=len(scenario.doc_codes),
        provisional=provisional,
        winner="C4",
        matched="sweep",
        counts={},
        repeatability={},
        versions={},
        cpu={},
        files={name: sha256_of(results / name) for name in (TOPICS_FILE_NAME, SELECTION_FILE_NAME)},
        data_files={LABELS_FILE_NAME: sha256_of(processed / LABELS_FILE_NAME)},
        started_at=MOMENT,
        finished_at=MOMENT,
        seconds=1.0,
    )
    (results / T19_MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def clean_git(root: Path, relative: str) -> GitState:
    return GitState(commit="c" * 40, dirty=())


def write_gaps(root: Path, scenario: Scenario) -> None:
    """Run T24a's own command on T19's records."""
    corpus = CorpusInputs(
        experiment=experiment_inputs(scenario),
        metadata=metadata_of(scenario),
        sensitive=SENSITIVE,
        input_sha256={"metadata.jsonl": "m" * 64},
    )
    code = run_gaps(
        ["--project-root", str(root), "--snapshot-id", SNAPSHOT],
        load=lambda *arguments: corpus,
        git=clean_git,
        clock=lambda: MOMENT,
    )
    assert code == 0


@dataclass(frozen=True)
class Temporal:
    """T27's records of the scenario, as written: the oracle of samples A and B."""

    sample: tuple[SampleRow, ...]
    labels: dict[str, int]
    """The earlier map's label of each earlier thesis."""
    centroids: TopicCentroids


def write_temporal(root: Path, scenario: Scenario, *, provisional: bool = True) -> Temporal:
    """Write T27's records of the scenario with T27's store (see the module)."""
    vectors = scenario.vectors.astype(np.float64)
    earlier, later = scenario.earlier, scenario.later
    labels = scenario.earlier_labels()
    topics = tuple(sorted(set(labels.tolist()) - {-1}))
    centroids = TopicCentroids(
        topics=topics, centroids=topic_centroids(vectors[earlier], labels, topics)
    )
    threshold = atypical_threshold(
        vectors[earlier],
        labels,
        centroids,
        percentile=ATYPICAL_PERCENTILE,
        method="linear",
        scope="global",
    )
    assignments = assign_proposals(vectors[later], centroids)
    checks = [check_atypical(item, threshold) for item in assignments]
    drawn = draw_expert_sample(
        [scenario.programs[row] for row in later],
        [check.atypical for check in checks],
        program_order=PROGRAMS,
        size=40,
        atypical_quota=10,
        program_floor=2,
        seed=7,
    )
    reference = REFERENCE_RUN + TEMPORAL_SUFFIX
    other = winner_runs()[0] + TEMPORAL_SUFFIX
    names = {topic: name for name, topic in scenario.earlier_ids.items()}
    full = scenario.labels[REFERENCE_RUN]
    sample_rows = []
    for index, pool, position in zip(drawn.rows, drawn.pools, drawn.positions, strict=True):
        row = int(later[index])
        item, check = assignments[index], checks[index]
        sample_rows.append(
            SampleRow(
                position=position,
                doc_code=scenario.doc_codes[row],
                title="Un título que T27 guardó y que las muestras no usan",
                program=scenario.programs[row],
                year=scenario.years[row],
                issued_after_snapshot=False,
                pool=pool,
                temporal_run_id=reference,
                seed=42,
                topic=item.topic,
                topic_size=int(np.count_nonzero(labels == item.topic)),
                topic_terms=earlier_terms(names[item.topic])[:10],
                similarity=item.similarity,
                threshold=check.threshold,
                atypical=check.atypical,
                runner_up=item.runner_up,
                runner_up_similarity=item.runner_up_similarity,
                margin=item.margin,
                full_label=int(full[row]),
                counterpart_agrees=False,
                provisional=provisional,
            )
        )
    common = {"run_id": REFERENCE_RUN, "seed": 42, "provisional": provisional}
    assignment_rows = [
        AssignmentRow(
            temporal_run_id=reference,
            doc_code=scenario.doc_codes[row],
            period="fitted",
            label=int(label),
            full_label=int(full[row]),
            similarity=None,
            runner_up=None,
            runner_up_similarity=None,
            margin=None,
            threshold=None,
            atypical=None,
            **common,
        )
        for row, label in zip(earlier, labels, strict=True)
    ]
    assignment_rows += [
        AssignmentRow(
            temporal_run_id=reference,
            doc_code=scenario.doc_codes[row],
            period="assigned",
            label=item.topic,
            full_label=int(full[row]),
            similarity=item.similarity,
            runner_up=item.runner_up,
            runner_up_similarity=item.runner_up_similarity,
            margin=item.margin,
            threshold=check.threshold,
            atypical=check.atypical,
            **common,
        )
        for row, item, check in zip(later, assignments, checks, strict=True)
    ]
    assignment_rows += [
        AssignmentRow(
            temporal_run_id=other,
            run_id=winner_runs()[0],
            seed=SEEDS[0],
            doc_code=code,
            period="fitted" if scenario.years[row] < 2025 else "assigned",
            label=0,
            full_label=0,
            similarity=None,
            runner_up=None,
            runner_up_similarity=None,
            margin=None,
            threshold=None,
            atypical=None,
            provisional=provisional,
        )
        for row, code in enumerate(scenario.doc_codes)
    ]
    topic_rows = [
        TemporalTopicRow(
            temporal_run_id=reference,
            seed=42,
            reference=True,
            provisional=provisional,
            topic=topic,
            size=int(np.count_nonzero(labels == topic)),
            terms=earlier_terms(names[topic])[:10],
            scores=(0.1,) * 10,
            atypical_threshold=threshold.value_for(topic),
            counterpart_own_label=None,
            counterpart_exclude_either=None,
            assigned=0,
            atypical=0,
        )
        for topic in topics
    ]
    topic_rows.append(
        TemporalTopicRow(
            temporal_run_id=other,
            seed=SEEDS[0],
            reference=False,
            provisional=provisional,
            topic=0,
            size=len(earlier),
            terms=("zz",),
            scores=(0.1,),
            atypical_threshold=0.5,
            counterpart_own_label=None,
            counterpart_exclude_either=None,
            assigned=0,
            atypical=0,
        )
    )
    manifest = TemporalManifest(
        snapshot_id=SNAPSHOT,
        git_commit="d" * 40,
        experiment=dict(read_experiment(CONFIG, root, SNAPSHOT).sha256),
        inputs={},
        winner="C4",
        representation=WINNER_MODEL,
        algorithm="hdbscan",
        parameter=10,
        provisional=provisional,
        runs=winner_runs(),
        temporal_runs=tuple(run + TEMPORAL_SUFFIX for run in winner_runs()),
        reference_run=REFERENCE_RUN,
        settings={"first_year": 2021, "split_year": 2025, "last_year": 2026},
        counts={},
        versions={},
        cpu={},
        files={},
        data_files={},
        started_at=MOMENT,
        finished_at=MOMENT,
        seconds=1.0,
        timings={},
    )
    write_outputs(
        results=temporal_results_dir(root / "results", SNAPSHOT),
        processed=temporal_processed_dir(root / "data", SNAPSHOT),
        runs=(),
        consistency=(),
        topics=topic_rows,
        composition=(),
        assignments=assignment_rows,
        sample=sample_rows,
        summary="A synthetic summary.",
        manifest=manifest,
        guarded=frozenset(),
    )
    by_code = {
        scenario.doc_codes[row]: int(label) for row, label in zip(earlier, labels, strict=True)
    }
    return Temporal(tuple(sample_rows), by_code, centroids)


def build_project(root: Path, *, provisional: bool = True) -> Temporal:
    (root / "config").mkdir(parents=True)
    for name in ("default.yaml", "career_affinity.yaml"):
        shutil.copyfile(PROJECT_ROOT / "config" / name, root / "config" / name)
    write_experiment(root, SCENARIO, provisional=provisional)
    write_gaps(root, SCENARIO)
    return write_temporal(root, SCENARIO, provisional=provisional)


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("template") / "project"
    temporal = build_project(root)
    return SimpleNamespace(root=root, temporal=temporal)


@pytest.fixture
def project(template: SimpleNamespace, tmp_path: Path) -> Path:
    root = tmp_path / "project"
    shutil.copytree(template.root, root)
    return root


def run_main(root: Path, *extra: str, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "load": lambda *_: sample_inputs(),
        "git": clean_git,
        "clock": lambda: MOMENT,
    }
    arguments.update(overrides)
    return main(["--project-root", str(root), "--snapshot-id", SNAPSHOT, *extra], **arguments)


def files_under(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_of(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def text_of(root: Path, sample: str) -> str:
    return (root / SAMPLES_DIR / SAMPLE_FILE_NAMES[sample]).read_text(encoding="utf-8")


def key_of(root: Path) -> list[dict[str, str]]:
    text = (root / SAMPLES_DIR / KEY_FILE_NAME).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def cases_of(root: Path, sample: str) -> list[dict[str, str]]:
    return [row for row in key_of(root) if row["sample"] == sample and row["code"]]


def manifest_of(root: Path) -> SamplesManifest:
    return SamplesManifest.model_validate_json(
        (root / SAMPLES_DIR / MANIFEST_FILE_NAME).read_text(encoding="utf-8")
    )


def section(text: str, code: str) -> str:
    """Return the lines of one case, from its heading to the next one."""
    start = text.index(f" {code}\n")
    end = text.find("\n## ", start)
    return text[start : end if end >= 0 else len(text)]


def row_of(code: str) -> int:
    return SCENARIO.doc_codes.index(code)


def stable_rows(root: Path) -> list[GapRow]:
    rows = ParquetTableStore().read(GapRow, root / "results" / "gaps" / SNAPSHOT / "gaps.parquet")
    return [row for row in rows if row.reference and row.stable]


# --- A real run -------------------------------------------------------------------------------


def test_a_run_writes_the_samples_the_key_and_the_manifest_under_data_only(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = files_under(project)

    assert run_main(project) == 0

    after = files_under(project)
    added = {path for path in after if path not in before}
    assert added == {f"{SAMPLES_DIR.as_posix()}/{name}" for name in OUTPUTS} | {
        f"{SAMPLES_DIR.as_posix()}/.lock"
    }
    assert {path: digest for path, digest in after.items() if path in before} == before
    out = capsys.readouterr().out
    assert "Wrote data/expert_session/S1/muestras/" in out
    for row in range(len(SCENARIO.doc_codes)):
        assert title_of(row) not in out


def test_sample_a_draws_two_theses_per_program_from_t27s_expert_sample(
    project: Path, template: SimpleNamespace
) -> None:
    assert run_main(project) == 0

    temporal: Temporal = template.temporal
    sample = {row.doc_code: row for row in temporal.sample}
    cases = cases_of(project, "A")
    assert [row["code"] for row in cases] == [f"A{index:02d}" for index in range(1, 11)]
    assert {
        p: sum(SCENARIO.programs[row_of(r["doc_code"])] == p for r in cases) for p in PROGRAMS
    } == dict.fromkeys(PROGRAMS, 2)
    text = text_of(project, "A")
    labels = list(temporal.labels.values())
    quartiles = saturation_quartiles(labels, method="linear", boundary="lower")
    for case in cases:
        drawn = sample[case["doc_code"]]
        assert int(case["topic"]) == drawn.topic
        shown = case["shown"].split(";")
        assert len(shown) == 3
        assert all(temporal.labels[code] == drawn.topic for code in shown)
        block = section(text, case["code"])
        row = row_of(case["doc_code"])
        assert f"- **Carrera y año:** {PROGRAM_NAMES[drawn.program]}, {drawn.year}" in block
        assert f"- **Título:** {title_of(row)}" in block
        terms = ", ".join(drawn.topic_terms[:10])
        assert f"- **Grupo asignado:** {drawn.topic}. Términos: {terms}" in block
        assert terms.startswith("zr")
        for code in shown:
            other = row_of(code)
            expected = f"  - {title_of(other)} ({PROGRAM_NAMES[SCENARIO.programs[other]]}, "
            assert expected + f"{SCENARIO.years[other]})" in block
        assert f"- **Propuesta atípica:** {'Sí' if drawn.atypical else 'No'}" in block
        level = quartiles.of_topic(drawn.topic).level
        assert f"- **Saturación del grupo:** {level}" in block


def unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


def full_centroids() -> dict[int, np.ndarray]:
    labels = SCENARIO.labels[REFERENCE_RUN]
    vectors = SCENARIO.vectors.astype(np.float64)
    return {topic: vectors[labels == topic].mean(axis=0) for topic in set(labels.tolist()) - {-1}}


def nearby_gap_lines(root: Path, query: np.ndarray) -> list[str]:
    """T26's nearby gaps, recomputed by hand: each stable gap's anchor on T19's reference map
    (a topic's centroid, or the midpoint of a pair's unit centroids), by cosine distance."""
    centroids, names = full_centroids(), SCENARIO.topic_names[REFERENCE_RUN]
    rows = stable_rows(root)
    distances = []
    for row in rows:
        anchor = sum(unit(centroids[topic]) for topic in row.topics)
        distances.append(1.0 - float(unit(anchor) @ unit(query)))
    order = sorted(range(len(rows)), key=lambda index: (distances[index], index))[:3]
    return [
        f"- Vacío cercano. {RULE_TEXTS[rows[index].rule]} Términos: "
        + " / ".join(", ".join(full_terms(names[topic])[:10]) for topic in rows[index].topics)
        for index in order
    ]


def test_sample_b_simulates_one_query_per_program_without_a_verdict(
    project: Path, template: SimpleNamespace
) -> None:
    assert run_main(project) == 0

    temporal: Temporal = template.temporal
    a_cases = {row["code"]: row for row in cases_of(project, "A")}
    queries = cases_of(project, "B")
    assert [row["code"] for row in queries] == ["B01", "B02", "B03", "B04"]
    programs = [SCENARIO.programs[row_of(row["doc_code"])] for row in queries]
    assert sorted(programs) == sorted(set(PROGRAMS) - {"mecanica"})
    replaced = [row for row in key_of(project) if row["sample"] == "B" and not row["code"]]
    mecanica = sorted(
        case["doc_code"]
        for case in a_cases.values()
        if SCENARIO.programs[row_of(case["doc_code"])] == "mecanica"
    )
    assert len(mecanica) == 2
    assert sorted(row["doc_code"] for row in replaced) == mecanica
    assert {row["note"] for row in replaced} == {NO_ABSTRACT}
    text = text_of(project, "B")
    vectors = SCENARIO.vectors.astype(np.float64)
    earlier = SCENARIO.earlier
    names = {topic: name for name, topic in SCENARIO.earlier_ids.items()}
    for query in queries:
        source = a_cases[query["source"]]
        assert source["doc_code"] == query["doc_code"]
        assert query["topic"] == source["topic"]
        row = row_of(query["doc_code"])
        block = section(text, query["code"])
        program = PROGRAM_NAMES[SCENARIO.programs[row]]
        assert f"**Propuesta** ({program}, {SCENARIO.years[row]})" in block
        assert f"- **Resumen:** Resumen sintético del estudio {row}." in block
        topic = int(query["topic"])
        terms = ", ".join(earlier_terms(names[topic])[:10])
        assert f"**Grupo asignado:** {topic}. Términos: {terms}" in block
        # The ranking: the earlier theses by cosine, never the query itself.
        scores = unit(vectors[earlier]) @ unit(vectors[row])
        best = [int(earlier[index]) for index in np.argsort(-scores, kind="stable")[:5]]
        shown = query["shown"].split(";")
        assert shown[:5] == [SCENARIO.doc_codes[index] for index in best]
        assert query["doc_code"] not in shown
        assert all(SCENARIO.years[row_of(code)] < 2025 for code in shown)
        for position, index in enumerate(best, start=1):
            similarity = float(unit(vectors[index]) @ unit(vectors[row]))
            cells = (
                f"| {position} | {title_of(index)} | "
                f"{PROGRAM_NAMES[SCENARIO.programs[index]]} | {SCENARIO.years[index]} | "
                f"{similarity:.2f} |"
            )
            assert cells in block
        # The antecedents: T26's MMR over the earlier theses.
        ranking = rank_antecedents(
            vectors[earlier], vectors[row], k=5, mmr_lambda=0.7, pool_size=None, exclude=None
        )
        for rank, item in enumerate(ranking.antecedents, start=1):
            index = int(earlier[item.index])
            program = PROGRAM_NAMES[SCENARIO.programs[index]]
            assert f"{rank}. {title_of(index)} ({program}, {SCENARIO.years[index]})" in block
        # The alternative lines: neighbor groups on the earlier map, gaps on T19's map.
        similarities = unit(temporal.centroids.centroids) @ unit(vectors[row])
        topics = temporal.centroids.topics
        order = sorted(
            (index for index in range(len(topics)) if topics[index] != topic),
            key=lambda index: (-similarities[index], topics[index]),
        )[:3]
        lines = [
            f"- Grupo vecino {topics[index]}. Términos: "
            + ", ".join(earlier_terms(names[topics[index]])[:10])
            for index in order
        ]
        lines += nearby_gap_lines(project, vectors[row])
        assert "\n".join(lines) in block
        assert f"**Veredicto de redundancia:** {VERDICT_PENDING}" in block
    for word in ("Redundante", "No redundante", "Relacionado"):
        assert word not in text


def test_sample_c_draws_six_groups_of_t19s_reference_map(project: Path) -> None:
    assert run_main(project) == 0

    labels = SCENARIO.labels[REFERENCE_RUN]
    names = SCENARIO.topic_names[REFERENCE_RUN]
    cases = cases_of(project, "C")
    assert [row["code"] for row in cases] == [f"C{index:02d}" for index in range(1, 7)]
    assert len({row["topic"] for row in cases}) == 6
    text = text_of(project, "C")
    for case in cases:
        topic = int(case["topic"])
        members = [SCENARIO.doc_codes[row] for row in np.flatnonzero(labels == topic)]
        shown = case["shown"].split(";")
        assert len(shown) == min(5, len(members)) and set(shown) <= set(members)
        block = section(text, case["code"])
        assert f"- **Tesis en el grupo:** {len(members)}. Por carrera: " in block
        terms = ", ".join(full_terms(names[topic])[:10])
        assert f"- **Términos representativos:** {terms}" in block
        for code in shown:
            index = row_of(code)
            program = PROGRAM_NAMES[SCENARIO.programs[index]]
            assert f"| {title_of(index)} | {program} | {SCENARIO.years[index]} |" in block


def test_sample_d_draws_up_to_two_stable_gaps_per_rule(project: Path) -> None:
    assert run_main(project) == 0

    stable = {row.gap_id: row for row in stable_rows(project)}
    members = ParquetTableStore().read(
        MemberRow, project / "data" / "processed" / SNAPSHOT / "gaps" / "members.parquet"
    )
    names = SCENARIO.topic_names[REFERENCE_RUN]
    cases = cases_of(project, "D")
    rules = [stable[row["gap_id"]].rule for row in cases]
    assert sorted(rules) == ["a", "b", "b", "c", "d"]
    text = text_of(project, "D")
    for case in cases:
        gap = stable[case["gap_id"]]
        theses = {m.doc_code for m in members if m.gap_id == gap.gap_id and m.part != "bridge"}
        shown = case["shown"].split(";")
        assert len(shown) == 3 and set(shown) <= theses
        assert case["topic"] == ";".join(str(topic) for topic in gap.topics)
        block = section(text, case["code"])
        assert f"- **Tipo:** {RULE_TEXTS[gap.rule]}" in block
        assert f"- **Tesis en el vacío:** {gap.theses}" in block
        groups = "; ".join(
            f"Grupo {topic}: {', '.join(full_terms(names[topic])[:10])}" for topic in gap.topics
        )
        assert f"- **Grupos y términos:** {groups}" in block
        if gap.rule == "c":
            assert (
                f"presente en {PROGRAM_NAMES['minas']} y ausente en {PROGRAM_NAMES['mecanica']}"
            ) in block


def test_no_sample_shows_a_doc_code_or_an_identifier_and_only_b_shows_abstracts(
    project: Path,
) -> None:
    assert run_main(project) == 0

    identifiers = [*SCENARIO.doc_codes, *(f"20.500.0/{row}" for row in range(148))]
    for sample in "ABCD":
        text = text_of(project, sample)
        assert ("Resumen sintético" in text) == (sample == "B")
        assert not any(code in text for code in identifiers)
        assert "http" not in text and "DOC-" not in text
    assert all(row["doc_code"] for row in cases_of(project, "A"))


def test_every_sample_is_marked_provisional_while_the_winner_is(project: Path) -> None:
    assert run_main(project) == 0

    for sample in "ABCD":
        assert "Resultados provisionales" in text_of(project, sample)
    assert manifest_of(project).provisional is True


def test_a_settled_winner_leaves_the_notice_out(tmp_path: Path) -> None:
    root = tmp_path / "settled"
    build_project(root, provisional=False)

    assert run_main(root) == 0

    for sample in "ABCD":
        assert "Resultados provisionales" not in text_of(root, sample)
    assert manifest_of(root).provisional is False


def test_the_samples_repeat_from_the_seed_and_change_with_it(
    project: Path, template: SimpleNamespace, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    shutil.copytree(template.root, again)
    other = tmp_path / "other"
    shutil.copytree(template.root, other)
    config_path = other / "config" / "default.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["expert_samples"]["seed"] = 1
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    assert run_main(project) == 0
    assert run_main(again) == 0
    assert run_main(other) == 0

    for name in (*SAMPLE_FILE_NAMES.values(), KEY_FILE_NAME, COMPOSITION_FILE_NAME):
        first = (project / SAMPLES_DIR / name).read_bytes()
        assert first == (again / SAMPLES_DIR / name).read_bytes()
    assert key_of(project) != key_of(other)
    assert manifest_of(other).settings["seed"] == 1


def test_the_manifest_records_the_provenance_values_and_outputs(project: Path) -> None:
    assert run_main(project) == 0

    manifest = manifest_of(project)
    directory = project / SAMPLES_DIR
    assert manifest.snapshot_id == SNAPSHOT
    assert manifest.git_commit == "c" * 40
    assert manifest.files == {
        name: sha256_of(directory / name) for name in OUTPUTS - {MANIFEST_FILE_NAME}
    }
    assert manifest.experiment == dict(read_experiment(CONFIG, project, SNAPSHOT).sha256)
    temporal = temporal_results_dir(project / "results", SNAPSHOT)
    assert manifest.temporal["manifest.json"] == sha256_of(temporal / "manifest.json")
    assert manifest.temporal[SAMPLE_FILE_NAME] == sha256_of(
        temporal_processed_dir(project / "data", SNAPSHOT) / SAMPLE_FILE_NAME
    )
    gaps = project / "results" / "gaps" / SNAPSHOT
    assert manifest.gaps["manifest.json"] == sha256_of(gaps / "manifest.json")
    assert manifest.inputs["metadata.jsonl"] == "m" * 64
    assert manifest.inputs["default.yaml"] == sha256_of(project / "config" / "default.yaml")
    assert manifest.settings["seed"] == CONFIG.expert_samples.seed
    assert manifest.settings["neighbor_reference"] == "proposal"
    assert manifest.settings["pair_anchor"] == "midpoint"
    assert manifest.reference_run == REFERENCE_RUN
    assert manifest.temporal_run == REFERENCE_RUN + TEMPORAL_SUFFIX
    composition = json.loads((directory / COMPOSITION_FILE_NAME).read_text(encoding="utf-8"))
    assert manifest.composition == composition
    assert composition["A"]["cases"] == 10 and composition["B"]["replaced"] == 2
    assert composition["D"]["rules"] == {"a": 1, "b": 2, "c": 1, "d": 1}
    assert manifest.guard["persons"] == 1


# --- The guards -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "planted",
    [f"de {PERSON}", f"de{PERSON.replace(' ', ' ')}s", "ver 20.500.0/3"],
    ids=["name", "glued-name", "handle"],
)
def test_a_name_or_identifier_in_a_text_stops_every_write(
    project: Path, planted: str, capsys: pytest.CaptureFixture[str]
) -> None:
    titles = {code: f"{title_of(row)} {planted}" for row, code in enumerate(SCENARIO.doc_codes)}
    inputs = sample_inputs(theses=theses_of(SCENARIO, **titles))

    assert run_main(project, load=lambda *_: inputs) == 1

    assert not any((project / SAMPLES_DIR / name).exists() for name in OUTPUTS)
    error = capsys.readouterr().err
    assert "nothing was written" in error
    assert "muestra_a.md" in error
    assert "Plantado" not in error and "20.500.0" not in error


def test_the_dry_run_applies_the_guard_too(project: Path) -> None:
    titles = {code: f"{title_of(row)} de {PERSON}" for row, code in enumerate(SCENARIO.doc_codes)}
    inputs = sample_inputs(theses=theses_of(SCENARIO, **titles))

    assert run_main(project, "--dry-run", load=lambda *_: inputs) == 1


# --- The refusals -----------------------------------------------------------------------------


def test_a_missing_temporal_split_is_refused_with_its_command(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.rmtree(temporal_results_dir(project / "results", SNAPSHOT))
    shutil.rmtree(temporal_processed_dir(project / "data", SNAPSHOT))

    for extra in ((), ("--dry-run",)):
        assert run_main(project, *extra) == 1
        error = capsys.readouterr().err
        assert "T27" in error and "thematic_redundancy.topic_map.run_temporal" in error
    assert not any((project / SAMPLES_DIR / name).exists() for name in OUTPUTS)


def test_a_missing_expert_sample_alone_is_refused(project: Path) -> None:
    (temporal_processed_dir(project / "data", SNAPSHOT) / SAMPLE_FILE_NAME).unlink()

    assert run_main(project, "--dry-run") == 1


def test_a_missing_gap_run_is_refused_with_its_command(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.rmtree(project / "results" / "gaps")

    assert run_main(project, "--dry-run") == 1
    assert "thematic_redundancy.gaps.run_gaps" in capsys.readouterr().err


def test_a_temporal_split_of_other_t19_records_is_refused(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = temporal_results_dir(project / "results", SNAPSHOT) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["experiment"][SELECTION_FILE_NAME] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert run_main(project, "--dry-run") == 1
    assert "T19" in capsys.readouterr().err


def test_a_changed_temporal_record_is_refused(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = temporal_processed_dir(project / "data", SNAPSHOT) / ASSIGNMENTS_FILE_NAME
    rows = ParquetTableStore().read(AssignmentRow, path)
    ParquetTableStore().write(rows[1:], path, model_cls=AssignmentRow)

    assert run_main(project, "--dry-run") == 1
    assert ASSIGNMENTS_FILE_NAME in capsys.readouterr().err


def test_gap_records_that_the_gap_rules_no_longer_give_are_refused(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = project / "config" / "default.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["gaps"]["stability_min_overlap"] = 0.99
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    assert run_main(project, "--dry-run") == 1
    assert "T24a" in capsys.readouterr().err


def test_earlier_samples_are_never_overwritten(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(project) == 0
    before = files_under(project / SAMPLES_DIR)

    assert run_main(project) == 1

    assert files_under(project / SAMPLES_DIR) == before
    assert "never overwritten" in capsys.readouterr().err


def test_a_dirty_tree_is_refused_before_any_input_is_read(project: Path) -> None:
    calls: list[Any] = []

    def load(*arguments: Any) -> SampleInputs:
        calls.append(arguments)
        return sample_inputs()

    code = run_main(project, load=load, git=lambda *_: GitState("c" * 40, ("src/x.py",)))

    assert code == 1
    assert calls == []
    assert not (project / SAMPLES_DIR).exists()


def test_a_held_lock_is_refused(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = project / SAMPLES_DIR
    directory.mkdir(parents=True)

    with exclusive_run(directory, RuntimeError("held")):
        assert run_main(project) == 1

    assert "lock" in capsys.readouterr().err
    assert not any((directory / name).exists() for name in OUTPUTS)


def test_an_interruption_exits_130(project: Path) -> None:
    def load(*_: Any) -> SampleInputs:
        raise KeyboardInterrupt

    assert run_main(project, load=load) == 130


# --- The dry run, the records from memory and the help ----------------------------------------


def test_the_dry_run_computes_everything_and_writes_nothing(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = files_under(project)

    assert run_main(project, "--dry-run") == 0

    assert files_under(project) == before
    assert not (project / "data" / "expert_session").exists()
    out = capsys.readouterr().out
    assert "nothing is written" in out
    assert "A: 10 cases" in out and "B: 4 cases, 2 replaced" in out
    assert "C: 6 cases" in out and "D: 5 cases" in out
    assert "Name guard" in out
    for row in range(len(SCENARIO.doc_codes)):
        assert title_of(row) not in out
    assert not any(code in out for code in SCENARIO.doc_codes)


def test_the_dry_run_reports_earlier_samples(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(project) == 0
    capsys.readouterr()

    assert run_main(project, "--dry-run") == 0
    assert "refuses to overwrite" in capsys.readouterr().out


def test_t27s_records_can_come_from_memory(project: Path) -> None:
    results = temporal_results_dir(project / "results", SNAPSHOT)
    processed = temporal_processed_dir(project / "data", SNAPSHOT)
    store = ParquetTableStore()
    sample = store.read(SampleRow, processed / SAMPLE_FILE_NAME)
    assignments = store.read(AssignmentRow, processed / ASSIGNMENTS_FILE_NAME)
    topics = store.read(TemporalTopicRow, results / "topics.parquet")
    expected = read_temporal(CONFIG, project, SNAPSHOT, read_experiment(CONFIG, project, SNAPSHOT))
    shutil.rmtree(results)
    shutil.rmtree(processed)
    records = TemporalRecords.from_rows(
        provisional=True,
        run_id=REFERENCE_RUN,
        temporal_run_id=REFERENCE_RUN + TEMPORAL_SUFFIX,
        first_year=2021,
        split_year=2025,
        last_year=2026,
        sample=sample,
        assignments=assignments,
        topics=topics,
        sha256={"in_memory": "-"},
    )

    assert dataclasses.replace(records, sha256=expected.sha256) == expected
    assert run_main(project, "--dry-run", read_temporal=lambda *_: records) == 0


def test_the_help_names_the_output_and_the_one_time_rule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    assert stopped.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "data/expert_session/<snapshot>/muestras/" in text
    assert "never overwritten" in text
    assert "--dry-run" in text and "--snapshot-id" in text and "--project-root" in text
