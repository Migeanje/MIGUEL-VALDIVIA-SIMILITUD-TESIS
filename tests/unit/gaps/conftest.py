"""A synthetic T19 experiment for the gap run's tests (T24a), written under pytest's temporary
directory, outside the repository.

The scenario holds 148 theses in 10 dimensions, each near the axis of its topic (a seeded jitter
of 0.01 breaks exact ties): six big topics of 20 theses (16 Industrial and one of each other
program, half of them 2021-2023 and half 2024-2026), three small topics and 6 Industrial noise
theses midway between topics A and B, so A and B have 6 bridges. The small topics are C (8 Minas
and 1 Industrial, all 2022), D (5 Sistemas and 2 Mecánica, 4 early and 3 late) and E (6
Electrónica, all 2025). The winner C4's five seed runs share the partition, except that seeds 7,
13 and 21 merge D into B, seed 42 (the medoid run) leaves E as noise and seed 73 leaves C as
noise; seeds 13 and 73 number their topics in reverse. Worked by hand from the rules:

- rule (a): the sizes of the 40 topics give a first quartile of 17.25, and their per-topic NPMI
  (A 0.08 to I 0.18, the small topics 0.30) a 0.25 quantile of 0.115; the small topics are the
  gaps, so C is stable on seed 42 (4 of 5 runs) and D is not (2 of 5);
- rule (c): C's Minas theses lack Mecánica (affinity 1), and its one Industrial thesis is a
  share of 1/103, rare only under a ``rare_max_share`` of 0.01; D's Sistemas theses lack
  Industrial; E's Electrónica theses lack Sistemas and Mecánica;
- rule (d): C falls from 9 of the 76 early theses to none of the 72 late ones.

``npmi_sensitivity`` holds -0.5 for the small topics, so a run that read it instead of ``npmi``
would find no rule (a) gap. The MiniLM vectors are random, so a run that read them instead of the
winner's would find other gaps.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from thematic_redundancy.gaps.affinity import CareerAffinity, load_career_affinity
from thematic_redundancy.gaps.rules import (
    Gap,
    GapMap,
    GapSettings,
    ThesisMetadata,
    interstitial_pairs,
    small_coherent_topics,
    small_topic_cuts,
    temporal_declines,
    transferable_topics,
)
from thematic_redundancy.gaps.run_gaps import CorpusInputs
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.build import topic_centroids
from thematic_redundancy.topic_map.experiment import (
    ExperimentInputs,
    SemanticInput,
    TopicRow,
    run_id,
)
from thematic_redundancy.topic_map.results_store import (
    LABELS_FILE_NAME,
    MANIFEST_FILE_NAME,
    SELECTION_FILE_NAME,
    TOPICS_FILE_NAME,
    ExperimentManifest,
    LabelRow,
    processed_dir,
    results_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(PROJECT_ROOT / "config" / "default.yaml")
MODELS = CONFIG.embedding.models
WINNER_MODEL = MODELS[1]
SEEDS = CONFIG.seeds
SNAPSHOT = "S1"
DIMENSIONS = 10
NAMES = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
BIG = ("A", "B", "F", "G", "H", "I")
NPMI = {"A": 0.08, "B": 0.10, "F": 0.12, "G": 0.14, "H": 0.16, "I": 0.18}
SMALL_NPMI = 0.30
T19_INPUTS = {
    "embeddings_manifest.json": "e" * 64,
    "tfidf_manifest.json": "t" * 64,
    "fichas_manifest.json": "f" * 64,
}
OTHER_RUN = run_id(MODELS[0], "kmeans", 8, 7)
"""A run of another configuration, which the gap run must leave out."""


def winner_runs() -> tuple[str, ...]:
    return tuple(run_id(WINNER_MODEL, "hdbscan", 10, seed) for seed in SEEDS)


def terms(name: str) -> tuple[str, ...]:
    return tuple(f"zq{name.lower()}{index}" for index in range(10))


@dataclass
class Scenario:
    doc_codes: tuple[str, ...]
    names: tuple[str | None, ...]
    """Each thesis's topic name in the base partition; ``None`` for the bridge theses."""
    programs: tuple[str, ...]
    years: tuple[int, ...]
    vectors: np.ndarray
    other_vectors: np.ndarray
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    """Each winner run's labels, in thesis order."""
    topic_names: dict[str, dict[int, str]] = field(default_factory=dict)
    """Each winner run's topic ids, with their names."""

    @property
    def metadata(self) -> ThesisMetadata:
        return ThesisMetadata(self.doc_codes, self.programs, self.years)


def make_scenario() -> Scenario:
    rng = np.random.default_rng(24)
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

    others = ("sistemas", "electronica", "mecanica", "minas")
    for name in BIG:
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
        doc_codes=tuple(f"DOC{index:04d}" for index in range(len(names))),
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
    """Return the topic of a thesis of base topic ``name`` in the run of ``seed``."""
    if name == "D" and seed in (7, 13, 21):
        return "B"
    if (name == "E" and seed == 42) or (name == "C" and seed == 73):
        return None
    return name


def npmi_of(name: str, *, sensitivity: bool = False) -> float:
    if name in NPMI:
        return NPMI[name]
    return -0.5 if sensitivity else SMALL_NPMI


@dataclass
class Project:
    root: Path
    scenario: Scenario
    corpus: CorpusInputs
    t19_results: Path
    t19_processed: Path

    @property
    def results(self) -> Path:
        return self.root / "results" / "gaps" / SNAPSHOT

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed" / SNAPSHOT / "gaps"


def write_experiment(
    root: Path, scenario: Scenario, *, provisional: bool = True, representative_seed: int = 42
) -> tuple[Path, Path]:
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
                    terms=terms(name),
                    scores=tuple(0.1 - 0.001 * index for index in range(10)),
                    npmi=npmi_of(name),
                    npmi_sensitivity=npmi_of(name, sensitivity=True),
                    absent_terms=(),
                )
            )
        label_rows += [
            LabelRow(run_id=run, doc_code=code, label=int(label))
            for code, label in zip(scenario.doc_codes, labels, strict=True)
        ]
    topic_rows.append(
        TopicRow(
            run_id=OTHER_RUN,
            topic=0,
            size=len(scenario.doc_codes),
            terms=terms("z"),
            scores=(0.1,) * 10,
            npmi=0.9,
            npmi_sensitivity=0.9,
            absent_terms=(),
        )
    )
    label_rows += [LabelRow(run_id=OTHER_RUN, doc_code=c, label=0) for c in scenario.doc_codes]
    store.write(topic_rows, results / TOPICS_FILE_NAME)
    store.write(label_rows, processed / LABELS_FILE_NAME)
    representative = run_id(WINNER_MODEL, "hdbscan", 10, representative_seed)
    selection = {
        "snapshot_id": SNAPSHOT,
        "provisional": provisional,
        "winner": {
            "configuration": "C4",
            "representation": WINNER_MODEL,
            "algorithm": "hdbscan",
            "parameter": 10,
            "provisional": provisional,
        },
        "representative": {"seed": representative_seed, "run_id": representative},
    }
    (results / SELECTION_FILE_NAME).write_text(json.dumps(selection), encoding="utf-8")
    moment = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)
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
        started_at=moment,
        finished_at=moment,
        seconds=1.0,
    )
    (results / MANIFEST_FILE_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return results, processed


def make_corpus(scenario: Scenario, *, sensitive: frozenset[str] = frozenset()) -> CorpusInputs:
    count = len(scenario.doc_codes)
    return CorpusInputs(
        experiment=ExperimentInputs(
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
        ),
        metadata=scenario.metadata,
        sensitive=sensitive or frozenset({"un título que no aparece", "nombre inventado plantado"}),
        input_sha256={"metadata.jsonl": "m" * 64},
    )


def scenario_maps(scenario: Scenario) -> tuple[GapMap, ...]:
    """Return the winner runs' maps, built in memory, with no file: the tests' oracle."""
    maps = []
    for run in winner_runs():
        labels = scenario.labels[run]
        topics = tuple(sorted(scenario.topic_names[run]))
        maps.append(
            GapMap(
                labels=labels,
                topics=topics,
                centroids=topic_centroids(scenario.vectors, labels, topics),
                npmi=tuple(npmi_of(scenario.topic_names[run][topic]) for topic in topics),
            )
        )
    return tuple(maps)


def scenario_gaps(
    scenario: Scenario, maps: tuple[GapMap, ...], settings: GapSettings, affinity: CareerAffinity
) -> tuple[tuple[Gap, ...], ...]:
    """Return each run's gaps of rules (a) to (d), straight from the rules: the oracle."""
    metadata = scenario.metadata
    cuts = small_topic_cuts(maps, npmi_floor_quantile=settings.npmi_floor_quantile)
    return tuple(
        (
            *small_coherent_topics(
                gap_map, metadata, size_cut=cuts.size_cut, npmi_floor=cuts.npmi_floor
            ),
            *interstitial_pairs(
                gap_map,
                scenario.vectors,
                metadata,
                nearest_topics=settings.nearest_topics,
                bridge_max_docs=settings.bridge_max_docs,
            ),
            *transferable_topics(
                gap_map,
                metadata,
                affinity,
                min_affinity=settings.min_affinity,
                present_min_count=settings.present_min_count,
                present_min_share=settings.present_min_share,
                rare_max_share=settings.rare_max_share,
            ),
            *temporal_declines(
                gap_map,
                metadata,
                first_year=settings.temporal_first_year,
                split_year=settings.temporal_split_year,
                last_year=settings.temporal_last_year,
                min_count=settings.temporal_min_count,
                min_decline=settings.min_decline,
            ),
        )
        for gap_map in maps
    )


def affinity() -> CareerAffinity:
    return load_career_affinity(
        PROJECT_ROOT / "config" / "career_affinity.yaml",
        programs=[program.key for program in CONFIG.snapshot.programs],
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return make_scenario()


@pytest.fixture(scope="session")
def kit(scenario: Scenario) -> SimpleNamespace:
    """The scenario's constants, builders and oracle, for the test modules of this folder."""
    settings = GapSettings.from_config(CONFIG)
    matrix = affinity()
    maps = scenario_maps(scenario)
    return SimpleNamespace(
        config=CONFIG,
        snapshot=SNAPSHOT,
        seeds=SEEDS,
        runs=winner_runs(),
        other_run=OTHER_RUN,
        t19_inputs=T19_INPUTS,
        settings=settings,
        affinity=matrix,
        maps=maps,
        gaps=scenario_gaps(scenario, maps, settings, matrix),
        terms=terms,
        files_under=files_under,
        loader=loader,
        make_corpus=make_corpus,
        write_experiment=write_experiment,
        sha256_of=sha256_of,
    )


@pytest.fixture
def project(tmp_path: Path, scenario: Scenario) -> Project:
    (tmp_path / "config").mkdir()
    for name in ("default.yaml", "career_affinity.yaml"):
        shutil.copyfile(PROJECT_ROOT / "config" / name, tmp_path / "config" / name)
    t19_results, t19_processed = write_experiment(tmp_path, scenario)
    return Project(tmp_path, scenario, make_corpus(scenario), t19_results, t19_processed)


def loader(corpus: CorpusInputs, calls: list[Any] | None = None) -> Any:
    def load(*arguments: Any) -> CorpusInputs:
        if calls is not None:
            calls.append(arguments)
        return corpus

    return load
