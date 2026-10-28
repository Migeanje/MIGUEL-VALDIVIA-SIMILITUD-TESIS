"""A synthetic project for T22's run on the labeled set, under pytest's temporary directory.

The corpus is the labeled-set draw's (``tests/unit/labeling/conftest.py``, loaded by path, since
pytest gives every conftest the module name ``conftest``): 130 theses of the five programs,
with both embedding models' vectors and a TF-IDF matrix. T20a's own command draws the pilot and
the labeled set from it once per session, so ``pairs.parquet`` and the draw's ``manifest.json``
are the real records; each test copies them into its own project root with the real
configuration, then writes T19's records and the consensus labels it needs.

T19's records are built with T19's selection rules on hand-made candidates, as in the real run:
MiniLM's stage 1 is won by C1 (K-means) and mpnet's by C4 (HDBSCAN), both pass every gate, and
mpnet's NPMI beats MiniLM's, so the provisional winner is C4 at stage 2's NPMI step. In the
``gates`` scenario both MiniLM configurations fail G3, so stage 2's first step names mpnet
before any F1 is read.
"""

import hashlib
import importlib.util
import itertools
import json
import random
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from thematic_redundancy.labeling import draw_pairs
from thematic_redundancy.labeling.draw_pairs import DrawInputs, PairRow
from thematic_redundancy.labeling.pairs import LABELED
from thematic_redundancy.redundancy.labels import (
    ADJUDICATION,
    AGREEMENT,
    REDUNDANT,
    RELATED,
    UNRELATED,
    ConsensusLabel,
    consensus_labels_path,
)
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.results_store import MANIFEST_FILE_NAME as T19_MANIFEST
from thematic_redundancy.topic_map.results_store import SELECTION_FILE_NAME as T19_SELECTION
from thematic_redundancy.topic_map.results_store import ExperimentManifest
from thematic_redundancy.topic_map.results_store import results_dir as t19_results_dir
from thematic_redundancy.topic_map.run_experiment import GitState
from thematic_redundancy.topic_map.selection import (
    Candidate,
    Gates,
    criterion_one_npmi,
    evaluate_gates,
    stage_one,
    stage_two_provisional,
    summarize,
)


def _load_labeling_corpus() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "labeling" / "conftest.py"
    spec = importlib.util.spec_from_file_location("labeling_synthetic_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LABELING = _load_labeling_corpus()
CONFIG = LABELING.CONFIG
MODELS: tuple[str, str] = CONFIG.embedding.models
MINILM, MPNET = MODELS
SNAPSHOT = LABELING.SNAPSHOT
PROJECT_ROOT = LABELING.PROJECT_ROOT
GATES = Gates(
    min_mean_ari=CONFIG.experiment.selection.gates.min_mean_ari,
    max_mean_noise_fraction=CONFIG.experiment.selection.gates.max_mean_noise_fraction,
)
T19_INPUTS = {
    name: LABELING.INPUT_SHA256[name]
    for name in ("embeddings_manifest.json", "tfidf_manifest.json", "fichas_manifest.json")
}
MOMENT = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)


def clean(root: Path, path: str) -> GitState:
    return GitState(commit="c" * 40, dirty=())


def clock() -> Any:
    moments = iter((MOMENT, datetime(2026, 10, 27, 9, 1, tzinfo=UTC)))
    return lambda: next(moments)


def timer() -> Any:
    ticks = itertools.count(10.0, 0.5)
    return lambda: next(ticks)


def loader(inputs: DrawInputs) -> Any:
    def load(*arguments: Any) -> DrawInputs:
        return inputs

    return load


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its SHA-256, by its path relative to ``root``."""
    return {
        path.relative_to(root).as_posix(): sha256_of(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- T19's records ----------------------------------------------------------------------------


def seeded(mean: float, sd: float) -> Any:
    """Five seed values whose sample SD is exactly ``sd``."""
    return summarize([mean - sd, mean - sd, mean, mean + sd, mean + sd])


def candidate(
    name: str,
    model: str,
    algorithm: str,
    parameter: int,
    *,
    silhouette: float,
    npmi: float,
    npmi_sensitivity: float,
    ari: float,
    noise: float = 0.0,
) -> Candidate:
    hdbscan = algorithm == "hdbscan"
    return Candidate(
        configuration=name,
        representation=model,
        algorithm=algorithm,  # type: ignore[arg-type]
        parameter=parameter,
        eligible=True,
        metrics_defined=True,
        silhouette=seeded(silhouette, 0.005),
        noise_fraction=seeded(noise, 0.01) if hdbscan else seeded(0.0, 0.0),
        clusters=seeded(18.0, 1.0) if hdbscan else seeded(float(parameter), 0.0),
        npmi=seeded(npmi, 0.005),
        ari_mean=ari,
        npmi_sensitivity=seeded(npmi_sensitivity, 0.004),
        ari_mean_sensitivity=min(1.0, ari + 0.05),
        silhouette_noise_reassigned=seeded(silhouette - 0.04, 0.01) if hdbscan else None,
    )


def candidates(scenario: str) -> dict[str, Candidate]:
    minilm_ari = 0.60 if scenario == "gates" else 0.89
    return {
        "C1": candidate(
            "C1", MINILM, "kmeans", 14, silhouette=0.09, npmi=0.17, npmi_sensitivity=0.10,
            ari=minilm_ari,
        ),
        "C2": candidate(
            "C2", MINILM, "hdbscan", 20, silhouette=0.12, npmi=0.16, npmi_sensitivity=0.09,
            ari=0.60, noise=0.20,
        ),
        "C3": candidate(
            "C3", MPNET, "kmeans", 12, silhouette=0.088, npmi=0.18, npmi_sensitivity=0.11,
            ari=0.85,
        ),
        "C4": candidate(
            "C4", MPNET, "hdbscan", 10, silhouette=0.143, npmi=0.213, npmi_sensitivity=0.13,
            ari=0.69, noise=0.26,
        ),
        "B0": candidate(
            "B0", "tfidf", "kmeans", 48, silhouette=0.03, npmi=0.205, npmi_sensitivity=0.12,
            ari=0.28,
        ),
    }  # fmt: skip


def selection_record(scenario: str) -> dict[str, Any]:
    """T19's ``selection.json`` for ``scenario``, with the keys that T19 writes and T22 reads."""
    found = candidates(scenario)
    fallback = CONFIG.experiment.selection.fallbacks
    stage_one_records, winners = {}, []
    for model, pair in ((MINILM, ("C1", "C2")), (MPNET, ("C3", "C4"))):
        decision = stage_one(found[pair[0]], found[pair[1]], gates=GATES, fallback="kmeans")
        stage_one_records[model] = {"decision": asdict(decision), "sensitivities": []}
        winners.append(found[decision.winner])
    assert fallback.algorithm == "kmeans"
    two = stage_two_provisional(*winners, gates=GATES, fallback_model=fallback.model)
    winner = next(c for c in winners if c.representation == two.winner)
    matched = candidate(
        "B0", "tfidf", "kmeans", 18, silhouette=0.03, npmi=0.219, npmi_sensitivity=0.12, ari=0.3
    )
    record = {
        "snapshot_id": SNAPSHOT,
        "provisional": True,
        "configurations": {
            name: {**asdict(c), "gates_failed": list(evaluate_gates(c, GATES).failed)}
            for name, c in found.items()
        },
        "stage_one": stage_one_records,
        "stage_two": {"decision": asdict(two), "sensitivities": []},
        "winner": {
            "configuration": winner.configuration,
            "representation": winner.representation,
            "algorithm": winner.algorithm,
            "parameter": winner.parameter,
            "provisional": True,
        },
        "criterion_one": asdict(criterion_one_npmi(winner, found["B0"], matched)),
        "findings": [],
    }
    return json.loads(json.dumps(record))


def write_t19(
    root: Path, *, scenario: str = "npmi", selection: dict[str, Any] | None = None
) -> Path:
    """Write T19's ``selection.json`` and ``manifest.json`` under ``root``; return the folder."""
    results = t19_results_dir(root / "results", SNAPSHOT)
    results.mkdir(parents=True, exist_ok=True)
    content = selection if selection is not None else selection_record(scenario)
    (results / T19_SELECTION).write_text(json.dumps(content), encoding="utf-8")
    manifest = ExperimentManifest(
        snapshot_id=SNAPSHOT,
        declaration_sha256=CONFIG.experiment_declaration_sha256(),
        git_commit="b" * 40,
        inputs=dict(T19_INPUTS),
        documents=130,
        provisional=True,
        winner=str(content["winner"]["configuration"]),
        matched="sweep",
        counts={},
        repeatability={},
        versions={},
        cpu={},
        files={T19_SELECTION: sha256_of(results / T19_SELECTION)},
        data_files={},
        started_at=MOMENT,
        finished_at=MOMENT,
        seconds=1.0,
    )
    (results / T19_MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return results


# --- The labels -------------------------------------------------------------------------------


def labeled_pairs(root: Path) -> tuple[PairRow, ...]:
    """The labeled-set pairs of the draw under ``root``, in draw order."""
    records = draw_pairs.records_dir(CONFIG, root, SNAPSHOT)
    rows = ParquetTableStore().read(PairRow, records / draw_pairs.PAIRS_FILE_NAME)
    return tuple(row for row in rows if row.phase == LABELED)


def top_ids(pairs: tuple[PairRow, ...], representation: int, count: int) -> set[str]:
    """The ids of the ``count`` pairs with the highest scores under ``representation``."""
    ranked = sorted(pairs, key=lambda row: -row.scores[representation])
    return {row.pair_id for row in ranked[:count]}


def random_ids(pairs: tuple[PairRow, ...], count: int, *, seed: int) -> set[str]:
    return set(random.Random(seed).sample([row.pair_id for row in pairs], count))


def consensus(pairs: tuple[PairRow, ...], redundant: set[str]) -> list[ConsensusLabel]:
    """Labels that put ``redundant`` at the top level and spread the others over the two
    lower levels; every seventh pair was adjudicated."""
    rows = []
    for index, row in enumerate(pairs):
        level = REDUNDANT if row.pair_id in redundant else (RELATED, UNRELATED)[index % 2]
        rows.append(
            ConsensusLabel(
                pair_id=row.pair_id,
                label_3=level,
                label=int(level == REDUNDANT),
                source=ADJUDICATION if index % 7 == 0 else AGREEMENT,
            )
        )
    return rows


def labels_path(root: Path) -> Path:
    return consensus_labels_path(draw_pairs.records_dir(CONFIG, root, SNAPSHOT))


def write_labels(root: Path, rows: list[ConsensusLabel], *, path: Path | None = None) -> Path:
    target = path or labels_path(root)
    ParquetTableStore().write(rows, target, model_cls=ConsensusLabel)
    return target


# --- Fixtures ---------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus() -> Any:
    return LABELING.make_corpus()


@pytest.fixture(scope="session")
def inputs(corpus: Any) -> DrawInputs:
    return LABELING.make_inputs(corpus)


@pytest.fixture(scope="session")
def drawn(tmp_path_factory: pytest.TempPathFactory, inputs: DrawInputs) -> Path:
    """A project root holding T20a's draw of the synthetic corpus, made by T20a's command."""
    root = tmp_path_factory.mktemp("drawn")
    (root / "config").mkdir()
    shutil.copyfile(PROJECT_ROOT / "config" / "default.yaml", root / "config" / "default.yaml")
    code = draw_pairs.main(
        ["--project-root", str(root), "--snapshot-id", SNAPSHOT],
        load=loader(inputs),
        git=clean,
        clock=clock(),
        timer=timer(),
    )
    assert code == 0
    return root


@pytest.fixture
def project(tmp_path: Path, drawn: Path) -> Path:
    """A project root with the real configuration, T20a's draw and T19's records (C4
    provisional at stage 2's NPMI step), and no labels yet."""
    root = tmp_path / "project"
    shutil.copytree(drawn, root)
    write_t19(root)
    return root


@pytest.fixture(scope="session")
def kit() -> SimpleNamespace:
    """The scenario's constants and builders, for the test modules of this folder."""
    return SimpleNamespace(
        config=CONFIG,
        models=MODELS,
        snapshot=SNAPSHOT,
        sensitive=LABELING.SENSITIVE,
        gates=GATES,
        clean=clean,
        clock=clock,
        timer=timer,
        loader=loader,
        sha256_of=sha256_of,
        files_under=files_under,
        candidate=candidate,
        candidates=candidates,
        selection_record=selection_record,
        write_t19=write_t19,
        labeled_pairs=labeled_pairs,
        top_ids=top_ids,
        random_ids=random_ids,
        consensus=consensus,
        labels_path=labels_path,
        write_labels=write_labels,
    )
