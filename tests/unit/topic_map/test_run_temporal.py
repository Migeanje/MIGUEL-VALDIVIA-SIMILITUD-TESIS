"""The temporal-split command (T27, O04), through ``main``, on a synthetic T19 experiment
written under pytest's temporary directory, outside the repository.

The scenario holds 86 theses of five programs in 8 dimensions: five topics on the first five
axes, each with 12 earlier theses (2021-2024) and 4 later ones (2025-2026) lying exactly on
their axis; 3 earlier and 2 later bridge theses midway between topics 0 and 1; and one later
outlier near topic 2's axis but mostly off every topic. The UMAP stand-in keeps the first five
coordinates (rolled by the seed, so each seed numbers its topics differently) and the HDBSCAN
stand-in labels a point by its largest coordinate when that exceeds 0.8, else noise, so the
earlier map's topics are the five planted ones with the 3 earlier bridges as noise. The later
bridges and the outlier are the atypical theses. On seed 7 the stand-in halves topic 4's
coordinate, so that map leaves topic 4 as noise and its 4 later theses are atypical too: the
seeds' flags differ.

T19's full map (``labels.parquet``) numbers the topics in reverse and leaves every bridge as
noise; seed 13 merges topics 3 and 4, seed 42 (the medoid run) leaves 7 of topic 4's 12 earlier
theses as noise, so that topic's counterpart is the noise under ``own_label`` and its label
under ``exclude_either``, and seed 73 leaves one later thesis of topic 0 as noise. Worked by
hand, for the later theses: under ``exclude_either`` the agreement is 1 in every seed but 7 and the
ARI 1 except on seed 13; under ``own_label`` the two later bridges never agree, nor, on seed
42, topic 4's later theses.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicCtfidf
from thematic_redundancy.topic_map.build import TopicMapPorts
from thematic_redundancy.topic_map.clustering import HdbscanRun, KMeansRun
from thematic_redundancy.topic_map.experiment import (
    ExperimentInputs,
    SemanticInput,
    TopicRow,
    run_id,
)
from thematic_redundancy.topic_map.results_store import (
    LABELS_FILE_NAME,
    SELECTION_FILE_NAME,
    ExperimentManifest,
    LabelRow,
)
from thematic_redundancy.topic_map.results_store import MANIFEST_FILE_NAME as T19_MANIFEST
from thematic_redundancy.topic_map.results_store import TOPICS_FILE_NAME as T19_TOPICS
from thematic_redundancy.topic_map.results_store import processed_dir as t19_processed_dir
from thematic_redundancy.topic_map.results_store import results_dir as t19_results_dir
from thematic_redundancy.topic_map.run_experiment import GitState
from thematic_redundancy.topic_map.run_temporal import (
    TemporalInputs,
    TemporalRunError,
    Theses,
    main,
    thesis_records,
    winner_clustering,
)
from thematic_redundancy.topic_map.temporal_store import (
    ASSIGNMENTS_FILE_NAME,
    COMPOSITION_FILE_NAME,
    CONSISTENCY_FILE_NAME,
    MANIFEST_FILE_NAME,
    RUNS_FILE_NAME,
    SAMPLE_FILE_NAME,
    SUMMARY_FILE_NAME,
    TOPICS_FILE_NAME,
    AssignmentRow,
    CompositionRow,
    ConsistencyRow,
    SampleRow,
    TemporalManifest,
    TemporalRunRow,
    TemporalTopicRow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(PROJECT_ROOT / "config" / "default.yaml")
MODELS = CONFIG.embedding.models
WINNER_MODEL = MODELS[1]
SEEDS = CONFIG.seeds
SNAPSHOT = "S1"
DIMENSIONS = 8
PROGRAMS = tuple(program.key for program in CONFIG.snapshot.programs)
T19_INPUTS = {
    "embeddings_manifest.json": "e" * 64,
    "tfidf_manifest.json": "t" * 64,
    "fichas_manifest.json": "f" * 64,
}
TRACKED = {
    RUNS_FILE_NAME,
    CONSISTENCY_FILE_NAME,
    TOPICS_FILE_NAME,
    COMPOSITION_FILE_NAME,
    SUMMARY_FILE_NAME,
    MANIFEST_FILE_NAME,
}
GITIGNORED = {ASSIGNMENTS_FILE_NAME, SAMPLE_FILE_NAME}
SENSITIVE = frozenset({"un título que no aparece", "nombre inventado plantado"})


def winner_runs() -> tuple[str, ...]:
    return tuple(run_id(WINNER_MODEL, "hdbscan", 10, seed) for seed in SEEDS)


# The scenario


@dataclass(frozen=True)
class Scenario:
    doc_codes: tuple[str, ...]
    kinds: tuple[str, ...]
    """``topic``, ``bridge`` or ``outlier``."""
    topics: tuple[int, ...]
    """The planted topic; -1 for a bridge or the outlier."""
    programs: tuple[str, ...]
    years: tuple[int, ...]
    vectors: np.ndarray
    documents: tuple[tuple[tuple[str, ...], ...], ...]
    vocabulary: tuple[str, ...]

    @property
    def later(self) -> np.ndarray:
        return np.asarray(self.years) >= 2025

    def full_labels(self, seed: int) -> np.ndarray:
        """T19's full-map labels of ``seed`` (see the module)."""
        topics = np.asarray(self.topics)
        labels = np.where(topics >= 0, 4 - topics, -1)
        outlier = np.asarray(self.kinds) == "outlier"
        labels[outlier] = 4 - 2
        if seed == 13:
            labels[labels == 1] = 0
        if seed == 73:
            labels[np.flatnonzero(self.later & (topics == 0))[0]] = -1
        if seed == 42:
            labels[np.flatnonzero(~self.later & (topics == 4))[:7]] = -1
        return labels


def make_scenario() -> Scenario:
    rng = np.random.default_rng(27)
    eye = np.eye(DIMENSIONS)
    rows: list[tuple[str, int, int, np.ndarray, tuple]] = []

    def words(name: str) -> tuple[tuple[str, ...], ...]:
        return ((f"{name}uno", f"{name}dos", "comun"), (f"{name}tres", f"{name}uno"))

    for topic in range(5):
        for index in range(12):
            vector = eye[topic] + rng.normal(0, 0.01, DIMENSIONS)
            rows.append(("topic", topic, 2021 + index % 4, vector, words(f"t{topic}")))
        for index in range(4):
            later = words(f"t{topic}") + (("reciente", "nuevo"),)
            rows.append(("topic", topic, 2025 + index % 2, eye[topic].copy(), later))
    bridge = (eye[0] + eye[1]) / np.sqrt(2)
    for index in range(3):
        rows.append(
            ("bridge", -1, 2022 + index, bridge + rng.normal(0, 0.01, DIMENSIONS), words("p"))
        )
    for index in range(2):
        rows.append(("bridge", -1, 2025 + index, bridge.copy(), words("p") + (("nuevo",),)))
    rows.append(("outlier", -1, 2026, eye[5] + 0.3 * eye[2], words("o")))
    documents = tuple(row[4] for row in rows)
    vocabulary = tuple(
        sorted(
            {token for doc in documents for section in doc for token in section}
            | {
                f"{a} {b}"
                for doc in documents
                for section in doc
                for a, b in zip(section, section[1:], strict=False)
            }
        )
    )
    return Scenario(
        doc_codes=tuple(f"DOC-{index:08x}" for index in range(len(rows))),
        kinds=tuple(row[0] for row in rows),
        topics=tuple(row[1] for row in rows),
        programs=tuple(PROGRAMS[index % 5] for index in range(len(rows))),
        years=tuple(row[2] for row in rows),
        vectors=np.vstack([row[3] for row in rows]).astype(np.float32),
        documents=documents,
        vocabulary=vocabulary,
    )


SCENARIO = make_scenario()


class RolledUmap:
    """A UMAP factory: the first five coordinates, rolled by the seed; it records each fit."""

    def __init__(self) -> None:
        self.fits: list[tuple[int, tuple[int, ...]]] = []

    def __call__(self, settings: Any, seed: int) -> Any:
        factory = self

        class Model:
            def fit_transform(self, vectors: Any, **kwargs: Any) -> np.ndarray:
                assert kwargs == {}
                matrix = np.asarray(vectors)
                factory.fits.append((seed, matrix.shape))
                reduced = matrix[:, : settings.n_components].copy()
                if seed == 7:
                    reduced[:, 4] *= 0.5
                return np.roll(reduced, seed % 5, axis=1)

        return Model()


class ArgmaxHdbscan:
    """An HDBSCAN factory: a point's label is its largest coordinate's index when that exceeds
    0.8, else noise."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, size: int, settings: Any) -> Any:
        self.sizes.append(size)

        class Model:
            def fit(self, points: Any) -> "Model":
                matrix = np.asarray(points)
                self.labels_ = np.where(matrix.max(axis=1) > 0.8, matrix.argmax(axis=1), -1)
                return self

        return Model()


def make_ports(fakes: Any) -> SimpleNamespace:
    umap, hdbscan = RolledUmap(), ArgmaxHdbscan()
    ports = TopicMapPorts(umap=umap, hdbscan=hdbscan, ctfidf=BertopicCtfidf(fakes.components()))
    return SimpleNamespace(ports=ports, umap=umap, hdbscan=hdbscan)


def make_inputs(scenario: Scenario = SCENARIO, **overrides: Any) -> TemporalInputs:
    count = len(scenario.doc_codes)
    theses = Theses(
        doc_codes=scenario.doc_codes,
        programs=scenario.programs,
        years=scenario.years,
        titles=tuple(f"Título sintético número {index}" for index in range(count)),
        after_snapshot=tuple(index == count - 1 for index in range(count)),
    )
    arguments: dict[str, Any] = {
        "experiment": ExperimentInputs(
            snapshot_id=SNAPSHOT,
            doc_codes=scenario.doc_codes,
            models=(
                SemanticInput(MODELS[0], "rev-a", np.ones((count, 3), dtype=np.float32)),
                SemanticInput(WINNER_MODEL, "rev-b", scenario.vectors),
            ),
            tfidf=sparse.csr_matrix(np.ones((count, len(scenario.vocabulary)))),
            vocabulary=scenario.vocabulary,
            documents=scenario.documents,
            embedding_variant="section_mean",
            tfidf_variant="section_mean",
            input_sha256=dict(T19_INPUTS),
        ),
        "theses": theses,
        "sensitive": SENSITIVE,
        "input_sha256": {"metadata.jsonl": "m" * 64},
    }
    arguments.update(overrides)
    return TemporalInputs(**arguments)


# The project


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_experiment(
    root: Path, *, provisional: bool = True, representative_seed: int = 42
) -> None:
    """Write T19's records of the scenario under ``root``, as T19's store writes them."""
    store = ParquetTableStore()
    results = t19_results_dir(root / "results", SNAPSHOT)
    processed = t19_processed_dir(root / "data", SNAPSHOT)
    results.mkdir(parents=True)
    processed.mkdir(parents=True)
    topic_rows, label_rows = [], []
    for seed, run in zip(SEEDS, winner_runs(), strict=True):
        labels = SCENARIO.full_labels(seed)
        for topic in sorted(set(labels.tolist()) - {-1}):
            topic_rows.append(
                TopicRow(
                    run_id=run,
                    topic=topic,
                    size=int(np.count_nonzero(labels == topic)),
                    terms=(f"zq{topic}",),
                    scores=(0.1,),
                    npmi=0.2,
                    npmi_sensitivity=0.1,
                    absent_terms=(),
                )
            )
        label_rows += [
            LabelRow(run_id=run, doc_code=code, label=int(label))
            for code, label in zip(SCENARIO.doc_codes, labels, strict=True)
        ]
    store.write(topic_rows, results / T19_TOPICS)
    store.write(label_rows, processed / LABELS_FILE_NAME)
    selection = {
        "winner": {
            "configuration": "C4",
            "representation": WINNER_MODEL,
            "algorithm": "hdbscan",
            "parameter": 10,
            "provisional": provisional,
        },
        "representative": {
            "seed": representative_seed,
            "run_id": run_id(WINNER_MODEL, "hdbscan", 10, representative_seed),
        },
    }
    (results / SELECTION_FILE_NAME).write_text(json.dumps(selection), encoding="utf-8")
    moment = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)
    manifest = ExperimentManifest(
        snapshot_id=SNAPSHOT,
        declaration_sha256=CONFIG.experiment_declaration_sha256(),
        git_commit="b" * 40,
        inputs=dict(T19_INPUTS),
        documents=len(SCENARIO.doc_codes),
        provisional=provisional,
        winner="C4",
        matched="sweep",
        counts={},
        repeatability={},
        versions={},
        cpu={},
        files={name: sha256_of(results / name) for name in (T19_TOPICS, SELECTION_FILE_NAME)},
        data_files={LABELS_FILE_NAME: sha256_of(processed / LABELS_FILE_NAME)},
        started_at=moment,
        finished_at=moment,
        seconds=1.0,
    )
    (results / T19_MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def make_project(root: Path, *, sample: dict[str, int] | None = None, **experiment: Any) -> Path:
    (root / "config").mkdir(parents=True)
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["temporal"]["sample"] = sample or {"seed": 7, "size": 8, "atypical": 2, "program_floor": 0}
    (root / "config" / "default.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    write_experiment(root, **experiment)
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_project(tmp_path / "project")


def results_of(root: Path) -> Path:
    return root / "results" / "temporal" / SNAPSHOT


def processed_of(root: Path) -> Path:
    return root / "data" / "processed" / SNAPSHOT / "temporal"


def files_under(path: Path) -> set[str]:
    return {item.name for item in path.iterdir() if item.is_file()} if path.exists() else set()


def clean_git(root: Path, relative: str) -> GitState:
    return GitState(commit="c" * 40, dirty=())


def loader(inputs: TemporalInputs, calls: list | None = None) -> Any:
    def load(*arguments: Any) -> TemporalInputs:
        if calls is not None:
            calls.append(arguments)
        return inputs

    return load


def run_main(root: Path, fakes: Any, *extra: str, **overrides: Any) -> tuple[int, SimpleNamespace]:
    recorders = make_ports(fakes)
    arguments: dict[str, Any] = {
        "load": loader(make_inputs()),
        "git": clean_git,
        "clock": lambda: datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
        "ports": recorders.ports,
    }
    arguments.update(overrides)
    code = main(["--project-root", str(root), "--snapshot-id", SNAPSHOT, *extra], **arguments)
    return code, recorders


def read(path: Path, model: type) -> tuple:
    return ParquetTableStore().read(model, path)


def manifest_of(root: Path) -> TemporalManifest:
    return TemporalManifest.model_validate_json(
        (results_of(root) / MANIFEST_FILE_NAME).read_text(encoding="utf-8")
    )


# A real run


def test_a_real_run_writes_the_tracked_aggregates_and_the_gitignored_records(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run_main(project, fakes)

    assert code == 0
    assert files_under(results_of(project)) == TRACKED | {".lock"}
    assert files_under(processed_of(project)) == GITIGNORED
    out = capsys.readouterr().out
    assert f"Snapshot {SNAPSHOT}: commit {'c' * 40}" in out
    assert "results/temporal/S1/" in out


def test_each_seed_fits_its_own_map_on_the_earlier_theses_alone(project: Path, fakes: Any) -> None:
    earlier = int(np.count_nonzero(~SCENARIO.later))

    _, recorders = run_main(project, fakes)

    assert recorders.umap.fits == [(seed, (earlier, DIMENSIONS)) for seed in SEEDS]
    assert recorders.hdbscan.sizes == [10] * len(SEEDS)


def test_the_runs_hold_each_earlier_map_and_its_atypical_flags(project: Path, fakes: Any) -> None:
    run_main(project, fakes)

    rows = read(results_of(project) / RUNS_FILE_NAME, TemporalRunRow)
    assert [row.seed for row in rows] == list(SEEDS)
    assert [row.reference for row in rows] == [seed == 42 for seed in SEEDS]
    assert [row.run_id for row in rows] == list(winner_runs())
    assert [row.temporal_run_id for row in rows] == [
        f"{run}__fit2021-2024" for run in winner_runs()
    ]
    for row in rows:
        assert (row.fitted_theses, row.assigned_theses) == (63, 23)
        topics, noise, atypical = (4, 15, 7) if row.seed == 7 else (5, 3, 3)
        assert (row.topics, row.noise_theses) == (topics, noise)
        assert row.noise_share == pytest.approx(noise / 63)
        # The two later bridges and the outlier: below the P5 of the members' similarities.
        assert (row.atypical_theses, row.atypical_share) == (atypical, pytest.approx(atypical / 23))
        assert row.full_map_noise_assigned == (3 if row.seed == 73 else 2)
        assert row.provisional is True
        assert row.vocabulary_terms < len(SCENARIO.vocabulary)


def earlier_map(seed: int) -> np.ndarray:
    """The earlier map's labels of the earlier theses, by planted topic (see the module)."""
    labels = np.asarray(SCENARIO.topics)[~SCENARIO.later].copy()
    if seed == 7:
        labels[labels == 4] = -1
    return labels


def expected_agreement(seed: int, noise_handling: str) -> tuple[int, float, float, int]:
    """The points, ARI, NMI and agreeing theses of the later period, from the scenario."""
    later = SCENARIO.later
    # Each later thesis goes to the planted topic whose earlier centroid is nearest by cosine.
    labels, earlier_vectors = earlier_map(seed), SCENARIO.vectors[~later].astype(np.float64)
    topics = sorted(set(labels.tolist()) - {-1})
    centroids = np.array([earlier_vectors[labels == t].mean(axis=0) for t in topics])
    units = SCENARIO.vectors[later] / np.linalg.norm(SCENARIO.vectors[later], axis=1)[:, None]
    cosines = units @ (centroids / np.linalg.norm(centroids, axis=1)[:, None]).T
    assigned = np.asarray(topics)[cosines.argmax(axis=1)]
    full = SCENARIO.full_labels(seed)[later]
    if noise_handling == "exclude_either":
        kept = full != -1
        assigned, full = assigned[kept], full[kept]
    noise_counterpart = (seed, noise_handling) == (42, "own_label")
    counterparts = {0: 4, 1: 3, 2: 2, 3: 0 if seed == 13 else 1}
    if seed != 7:
        counterparts[4] = -1 if noise_counterpart else 0
    agreeing = sum(counterparts[a] == f for a, f in zip(assigned, full, strict=True))
    return (
        int(assigned.size),
        adjusted_rand_score(assigned, full),
        normalized_mutual_info_score(assigned, full),
        int(agreeing),
    )


def test_the_later_theses_assignment_is_compared_with_the_same_seeds_full_map(
    project: Path, fakes: Any
) -> None:
    run_main(project, fakes)

    rows = read(results_of(project) / CONSISTENCY_FILE_NAME, ConsistencyRow)
    later = [row for row in rows if row.period == "assigned"]
    assert [(row.seed, row.noise_handling) for row in later] == [
        (seed, handling) for seed in SEEDS for handling in ("own_label", "exclude_either")
    ]
    for row in later:
        points, ari, nmi, agreeing = expected_agreement(row.seed, row.noise_handling)
        assert (row.points, row.agreeing) == (points, agreeing)
        assert row.ari == pytest.approx(ari)
        assert row.nmi == pytest.approx(nmi)
        assert row.agreement == pytest.approx(agreeing / points)
        assert row.temporal_run_id == f"{row.run_id}__fit2021-2024"
    by_key = {(row.seed, row.noise_handling): row for row in later}
    assert by_key[(42, "exclude_either")].agreement == 1.0
    assert by_key[(42, "exclude_either")].ari == pytest.approx(1.0)
    assert by_key[(13, "exclude_either")].agreement == 1.0
    assert by_key[(13, "exclude_either")].ari < 1.0
    # Seed 7's map has no topic 4, so its 4 later theses cannot agree.
    assert by_key[(7, "exclude_either")].agreeing == 17
    # 20 typical theses less topic 4's 4, plus the outlier; neither bridge agrees.
    assert by_key[(42, "own_label")].agreeing == 17


def test_the_earlier_theses_compare_both_maps_with_ari_and_nmi_only(
    project: Path, fakes: Any
) -> None:
    run_main(project, fakes)

    rows = read(results_of(project) / CONSISTENCY_FILE_NAME, ConsistencyRow)
    earlier = [row for row in rows if row.period == "fitted"]
    assert len(earlier) == 2 * len(SEEDS)
    for row in earlier:
        # The earlier map's labels: the planted topics, the 3 earlier bridges as noise.
        first = earlier_map(row.seed)
        second = SCENARIO.full_labels(row.seed)[~SCENARIO.later]
        if row.noise_handling == "exclude_either":
            kept = (first != -1) & (second != -1)
            first, second = first[kept], second[kept]
        excluded = {42: 53, 7: 48}.get(row.seed, 60)
        assert row.points == (63 if row.noise_handling == "own_label" else excluded)
        assert (row.agreement, row.agreeing) == (None, None)
        assert row.ari == pytest.approx(adjusted_rand_score(first, second))
        assert row.nmi == pytest.approx(normalized_mutual_info_score(first, second))
        split = {(42, "own_label"), (7, "own_label")}
        same = row.seed != 13 and (row.seed, row.noise_handling) not in split
        assert (row.ari == pytest.approx(1.0)) == same


def test_the_topics_hold_their_terms_counterparts_and_later_theses(
    project: Path, fakes: Any
) -> None:
    run_main(project, fakes)

    rows = read(results_of(project) / TOPICS_FILE_NAME, TemporalTopicRow)
    reference = [row for row in rows if row.seed == 42]
    assert [row.topic for row in reference] == [0, 1, 2, 3, 4]
    assert all(row.size == 12 for row in reference)
    # Seed 42 rolls the axes by 2: topic id t holds the planted topic (t - 2) mod 5.
    planted = [(row.topic - 42 % 5) % 5 for row in reference]
    # Most of topic 4's earlier members are full-map noise on seed 42.
    assert [row.counterpart_own_label for row in reference] == [
        -1 if p == 4 else 4 - p for p in planted
    ]
    assert [row.counterpart_exclude_either for row in reference] == [4 - p for p in planted]
    assert sum(row.assigned for row in reference) == 23
    assert sum(row.atypical for row in reference) == 3
    for row in reference:
        assert row.terms and row.terms[0].startswith(f"t{planted[reference.index(row)]}")
        assert "nuevo" not in row.terms and "reciente" not in row.terms


def test_the_gitignored_assignments_hold_every_seed_and_thesis_with_t19s_labels(
    project: Path, fakes: Any
) -> None:
    run_main(project, fakes)

    rows = read(processed_of(project) / ASSIGNMENTS_FILE_NAME, AssignmentRow)
    assert len(rows) == len(SEEDS) * len(SCENARIO.doc_codes)
    for seed in SEEDS:
        mine = [row for row in rows if row.seed == seed]
        assert [row.doc_code for row in mine] == list(SCENARIO.doc_codes)
        assert [row.full_label for row in mine] == SCENARIO.full_labels(seed).tolist()
        assert [row.period == "assigned" for row in mine] == SCENARIO.later.tolist()
        later = [row for row in mine if row.period == "assigned"]
        assert all(row.similarity is not None and row.atypical is not None for row in later)
        assert sum(row.atypical for row in later) == (7 if seed == 7 else 3)
        earlier = [row for row in mine if row.period == "fitted"]
        assert all(row.similarity is None and row.atypical is None for row in earlier)
        assert sum(row.label == -1 for row in earlier) == (15 if seed == 7 else 3)


def test_the_expert_sample_is_drawn_from_the_medoid_runs_map(project: Path, fakes: Any) -> None:
    run_main(project, fakes)

    rows = read(processed_of(project) / SAMPLE_FILE_NAME, SampleRow)
    assert [row.position for row in rows] == list(range(1, 9))
    assert {row.seed for row in rows} == {42}
    assert sum(row.pool == "atypical" for row in rows) == 2
    assert all(row.atypical for row in rows if row.pool == "atypical")
    assert not any(row.atypical for row in rows if row.pool == "typical")
    codes = {code: index for index, code in enumerate(SCENARIO.doc_codes)}
    for row in rows:
        index = codes[row.doc_code]
        assert SCENARIO.years[index] >= 2025
        assert (row.program, row.year) == (SCENARIO.programs[index], SCENARIO.years[index])
        assert row.title == f"Título sintético número {index}"
        assert row.issued_after_snapshot == (index == len(SCENARIO.doc_codes) - 1)
        assert row.topic_terms and row.provisional is True
        # Seed 42's own_label counterparts, by the planted topic of the row's topic id.
        planted = (row.topic - 42 % 5) % 5
        assert row.counterpart_agrees == ((-1 if planted == 4 else 4 - planted) == row.full_label)
    # A later thesis of topic 4 agrees under exclude_either only, and the sample holds one.
    assert any((row.topic - 42 % 5) % 5 == 4 and not row.counterpart_agrees for row in rows)
    composition = read(results_of(project) / COMPOSITION_FILE_NAME, CompositionRow)
    assert sum(row.quota for row in composition) == 8
    assert {(row.pool, row.program) for row in composition if row.quota} == {
        (row.pool, row.program) for row in rows
    }


def test_the_same_configuration_and_inputs_draw_the_same_sample(tmp_path: Path, fakes: Any) -> None:
    first, second = make_project(tmp_path / "a"), make_project(tmp_path / "b")
    run_main(first, fakes)
    run_main(second, fakes)

    assert read(processed_of(first) / SAMPLE_FILE_NAME, SampleRow) == read(
        processed_of(second) / SAMPLE_FILE_NAME, SampleRow
    )
    third = make_project(
        tmp_path / "c", sample={"seed": 8, "size": 8, "atypical": 2, "program_floor": 0}
    )
    run_main(third, fakes)
    assert [r.doc_code for r in read(processed_of(third) / SAMPLE_FILE_NAME, SampleRow)] != [
        r.doc_code for r in read(processed_of(first) / SAMPLE_FILE_NAME, SampleRow)
    ]


def test_the_manifest_records_the_provenance_the_values_and_every_output(
    project: Path, fakes: Any
) -> None:
    run_main(project, fakes)

    manifest = manifest_of(project)
    assert manifest.snapshot_id == SNAPSHOT
    assert manifest.git_commit == "c" * 40
    assert (manifest.winner, manifest.representation) == ("C4", WINNER_MODEL)
    assert (manifest.algorithm, manifest.parameter) == ("hdbscan", 10)
    assert manifest.provisional is True
    assert manifest.runs == winner_runs()
    assert manifest.reference_run == run_id(WINNER_MODEL, "hdbscan", 10, 42)
    t19 = t19_results_dir(project / "results", SNAPSHOT)
    assert manifest.experiment[T19_MANIFEST] == sha256_of(t19 / T19_MANIFEST)
    assert set(manifest.experiment) == {
        T19_MANIFEST,
        SELECTION_FILE_NAME,
        T19_TOPICS,
        LABELS_FILE_NAME,
    }
    assert manifest.inputs == {
        **T19_INPUTS,
        "metadata.jsonl": "m" * 64,
        "default.yaml": sha256_of(project / "config" / "default.yaml"),
    }
    assert manifest.settings == {
        "first_year": 2021,
        "split_year": 2025,
        "last_year": 2026,
        "atypical_percentile": 5.0,
        "atypical_scope": "global",
        "atypical_method": "linear",
        "noise_handling": "own_label",
        "noise_handling_sensitivity": "exclude_either",
        "sample_seed": 7,
        "sample_size": 8,
        "sample_atypical": 2,
        "sample_program_floor": 0,
    }
    assert manifest.counts["fitted_theses"] == 63
    assert manifest.counts["assigned_theses"] == 23
    assert manifest.counts["assigned_issued_after_snapshot"] == 1
    assert manifest.counts["sample"] == 8
    assert manifest.counts["t15_vocabulary_terms"] == len(SCENARIO.vocabulary)
    for name in TRACKED - {MANIFEST_FILE_NAME}:
        assert manifest.files[name] == sha256_of(results_of(project) / name)
    for name in GITIGNORED:
        assert manifest.data_files[name] == sha256_of(processed_of(project) / name)
    assert manifest.versions and manifest.cpu
    assert {f"seed{seed}_umap" for seed in SEEDS} <= set(manifest.timings)


def test_a_final_winner_is_carried_as_final(tmp_path: Path, fakes: Any) -> None:
    root = make_project(tmp_path / "final", provisional=False)

    run_main(root, fakes)

    assert manifest_of(root).provisional is False
    assert not any(
        row.provisional for row in read(results_of(root) / RUNS_FILE_NAME, TemporalRunRow)
    )
    assert "provisional" not in (results_of(root) / SUMMARY_FILE_NAME).read_text(encoding="utf-8")


def test_a_provisional_winner_marks_every_output(project: Path, fakes: Any) -> None:
    run_main(project, fakes)

    for name, model in (
        (RUNS_FILE_NAME, TemporalRunRow),
        (CONSISTENCY_FILE_NAME, ConsistencyRow),
        (TOPICS_FILE_NAME, TemporalTopicRow),
        (COMPOSITION_FILE_NAME, CompositionRow),
    ):
        assert all(row.provisional for row in read(results_of(project) / name, model))
    for name, model in ((ASSIGNMENTS_FILE_NAME, AssignmentRow), (SAMPLE_FILE_NAME, SampleRow)):
        assert all(row.provisional for row in read(processed_of(project) / name, model))
    assert "provisional" in (results_of(project) / SUMMARY_FILE_NAME).read_text(encoding="utf-8")


def test_another_medoid_run_draws_from_its_own_map(tmp_path: Path, fakes: Any) -> None:
    root = make_project(tmp_path / "medoid", representative_seed=73)

    run_main(root, fakes)

    assert {row.seed for row in read(processed_of(root) / SAMPLE_FILE_NAME, SampleRow)} == {73}
    rows = read(results_of(root) / RUNS_FILE_NAME, TemporalRunRow)
    assert [row.seed for row in rows if row.reference] == [73]


def test_no_tracked_output_holds_a_doc_code_or_a_title(project: Path, fakes: Any) -> None:
    run_main(project, fakes)

    for name in TRACKED:
        text = (results_of(project) / name).read_bytes().decode("utf-8", errors="replace")
        assert "DOC-" not in text
        assert "sintético" not in text
    summary = (results_of(project) / SUMMARY_FILE_NAME).read_text(encoding="utf-8")
    assert "2021 to 2024" in summary and "2025 to 2026" in summary


# The dry run


def test_the_dry_run_computes_everything_and_writes_nothing(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    code, recorders = run_main(project, fakes, "--dry-run")

    assert code == 0
    assert not results_of(project).exists()
    assert not processed_of(project).exists()
    assert len(recorders.umap.fits) == len(SEEDS)
    out = capsys.readouterr().out
    assert "nothing is written" in out
    assert "63 theses of 2021 to 2024" in out and "23 theses of 2025 to 2026" in out
    # The snapshot's 2 strings and the 86 theses' doc codes.
    assert "Privacy guard: 5 tracked outputs checked against 88 strings" in out
    assert "--snapshot-id S1" in out


def test_the_dry_run_reports_an_earlier_temporal_split(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    run_main(project, fakes)
    capsys.readouterr()

    code, _ = run_main(project, fakes, "--dry-run")

    assert code == 0
    assert "a real run refuses to overwrite it" in capsys.readouterr().out


# Refusals


def test_a_dirty_tree_is_refused_before_any_input_is_read(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []

    def dirty(root: Path, relative: str) -> GitState:
        assert relative == "results/temporal/S1"
        return GitState(commit="c" * 40, dirty=("src/x.py",))

    code, _ = run_main(project, fakes, git=dirty, load=loader(make_inputs(), calls))

    assert code == 1
    assert calls == []
    assert not results_of(project).exists()
    assert "1 changed or untracked paths outside results/temporal/S1" in capsys.readouterr().err


@pytest.mark.parametrize("earlier", [MANIFEST_FILE_NAME, SAMPLE_FILE_NAME])
def test_an_earlier_temporal_split_is_never_overwritten(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str], earlier: str
) -> None:
    run_main(project, fakes)
    kept = {
        path.name: sha256_of(path)
        for directory in (results_of(project), processed_of(project))
        for path in directory.iterdir()
    }
    if earlier == SAMPLE_FILE_NAME:
        (results_of(project) / MANIFEST_FILE_NAME).unlink()
        del kept[MANIFEST_FILE_NAME]
    capsys.readouterr()

    code, recorders = run_main(project, fakes)

    assert code == 1
    assert recorders.umap.fits == []
    assert "already exists" in capsys.readouterr().err
    assert {
        path.name: sha256_of(path)
        for directory in (results_of(project), processed_of(project))
        for path in directory.iterdir()
    } == kept


def test_a_run_holding_the_lock_is_waited_for_by_no_one(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    results_of(project).mkdir(parents=True)

    with exclusive_run(results_of(project), RuntimeError("held")):
        code, recorders = run_main(project, fakes)

    assert code == 1
    assert recorders.umap.fits == []
    assert "holds the lock" in capsys.readouterr().err
    assert files_under(results_of(project)) == {".lock"}


def test_a_changed_t19_record_is_refused(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = t19_processed_dir(project / "data", SNAPSHOT) / LABELS_FILE_NAME
    rows = ParquetTableStore().read(LabelRow, labels)
    ParquetTableStore().write([rows[1], rows[0], *rows[2:]], labels)

    code, _ = run_main(project, fakes)

    assert code == 1
    assert "labels.parquet differs from the SHA-256" in capsys.readouterr().err
    assert files_under(results_of(project)) <= {".lock"}
    assert not processed_of(project).exists()


def test_inputs_other_than_t19s_are_refused(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = make_inputs().experiment
    other = ExperimentInputs(
        **{**experiment.__dict__, "input_sha256": {**T19_INPUTS, "tfidf_manifest.json": "x" * 64}}
    )

    code, _ = run_main(project, fakes, load=loader(make_inputs(experiment=other)))

    assert code == 1
    assert "tfidf_manifest.json" in capsys.readouterr().err
    assert not processed_of(project).exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_tracked_output_holding_a_snapshot_string_refuses_every_write(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str], dry_run: bool
) -> None:
    # A topic's top term equal to a person's name of the snapshot reaches topics.parquet.
    inputs = make_inputs(sensitive=SENSITIVE | {"t3uno"})

    code, _ = run_main(project, fakes, *(["--dry-run"] if dry_run else []), load=loader(inputs))

    assert code == 1
    err = capsys.readouterr().err
    assert "topics.parquet" in err and "t3uno" not in err
    assert files_under(results_of(project)) <= {".lock"}
    assert not processed_of(project).exists()


def test_an_interruption_exits_with_130(
    project: Path, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(*arguments: Any) -> TemporalInputs:
        raise KeyboardInterrupt

    code, _ = run_main(project, fakes, load=interrupted)

    assert code == 130
    assert "interrupted" in capsys.readouterr().err


def test_the_help_names_the_options_and_the_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for text in ("--project-root", "--snapshot-id", "--dry-run", "results/temporal/<snapshot>/"):
        assert text in out
    assert "never overwritten" in " ".join(out.split())


# The helpers


def test_the_winners_clustering_follows_its_algorithm_and_parameter() -> None:
    assert winner_clustering({"algorithm": "hdbscan", "parameter": 10}) == HdbscanRun(10)
    assert winner_clustering({"algorithm": "kmeans", "parameter": 18}) == KMeansRun(18)
    with pytest.raises(TemporalRunError, match="spectral"):
        winner_clustering({"algorithm": "spectral", "parameter": 3})


def test_the_theses_records_follow_the_document_vectors_order() -> None:
    fichas = [
        SimpleNamespace(
            doc_code=f"DOC-{index:08x}",
            program_key=program,
            issue_year=year,
            title=f"Título {index}",
            issued_after_snapshot=index == 2,
        )
        for index, (program, year) in enumerate(
            [("minas", 2021), ("sistemas", 2025), ("industrial", 2026)]
        )
    ]
    codes = ("DOC-00000002", "DOC-00000000", "DOC-00000001")

    theses = thesis_records(fichas, codes)

    assert theses == Theses(
        doc_codes=codes,
        programs=("industrial", "minas", "sistemas"),
        years=(2026, 2021, 2025),
        titles=("Título 2", "Título 0", "Título 1"),
        after_snapshot=(True, False, False),
    )
    with pytest.raises(ValueError, match="lack 1 thesis"):
        thesis_records(fichas, (*codes, "DOC-0000000f"))
