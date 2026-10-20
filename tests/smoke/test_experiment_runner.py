"""Smoke check of the experiment runner (T19) end to end, with the real UMAP, K-means, HDBSCAN
and BERTopic, on a tiny synthetic project in a temporary directory.

The project holds a copy of ``config/default.yaml``, so the declared grid runs in full (22 K,
4 sizes, 5 seeds, 2 models and the TF-IDF baseline) and the declaration check passes; the
inputs are synthetic: theses drawn from fixed theme word lists, two sets of vectors around one
axis per theme, and T15's TF-IDF over the theses. Only the input loader and git are replaced.
One thesis holds a marker word that occurs nowhere else, so it is not in the vocabulary; no
output may hold it. A second run must reuse the ten cached reductions and repeat every record
except the run-time fields.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from thematic_redundancy.representation.tfidf import TfidfSettings, fit_tfidf
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map.experiment import (
    DECLARED_SHA256,
    RUNTIME_FIELDS,
    ExperimentInputs,
    SemanticInput,
)
from thematic_redundancy.topic_map.run_experiment import GitState, main

pytestmark = pytest.mark.smoke

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
SNAPSHOT = "S20261019"
MARKER = "zzmarcadorprivado"
THEMES = (
    "voladura explosivo fragmentación macizo perforación detonador tajo roca",
    "mantenimiento rodamiento disponibilidad falla motor repuesto lubricación turbina",
    "inventario módulo datos facturación usuario aplicación servidor web",
    "concreto viga columna sismo losa cimentación acero refuerzo",
    "logística almacén ruta transporte demanda pedido flota despacho",
    "energía solar panel batería inversor eficiencia radiación consumo",
)
SHARED = ("empresa", "proceso", "análisis", "propuesta", "mejora")
PER_THEME = 18


def make_inputs(models: tuple[str, ...]) -> ExperimentInputs:
    rng = np.random.default_rng(20261019)
    documents, first = [], []
    for index, words in enumerate(THEMES):
        for _ in range(PER_THEME):
            documents.append(
                tuple(
                    tuple(
                        str(rng.choice(words.split()))
                        if rng.random() < 0.8
                        else str(rng.choice(SHARED))
                        for _ in range(length)
                    )
                    for length in (5, 24, 10)
                )
            )
            first.append(np.eye(32)[index] * 10 + rng.normal(0, 1.0, 32))
    title, abstract, objectives = documents[0]
    documents[0] = (title, (*abstract, MARKER), objectives)
    vectors = np.vstack(first)
    second = vectors @ rng.normal(size=(32, 48)) + rng.normal(0, 2.0, (len(vectors), 48))
    tfidf = fit_tfidf(
        documents, TfidfSettings(max_ngram=2, min_df=2, max_df=1.0, sublinear_tf=True)
    )
    assert MARKER not in tfidf.terms
    return ExperimentInputs(
        snapshot_id=SNAPSHOT,
        doc_codes=tuple(f"T{index:04d}" for index in range(len(documents))),
        models=(
            SemanticInput(models[0], "rev-a", vectors.astype(np.float32)),
            SemanticInput(models[1], "rev-b", second.astype(np.float32)),
        ),
        tfidf=tfidf.section_mean,
        vocabulary=tfidf.terms,
        documents=tuple(documents),
        embedding_variant="section_mean",
        tfidf_variant="section_mean",
        input_sha256={"embeddings_manifest.json": "e" * 64, "tfidf_manifest.json": "t" * 64},
    )


def strings_in(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        table = pq.read_table(path).to_pylist()
        return [json.dumps(row, ensure_ascii=False) for row in table]
    if path.suffix == ".npy":
        return []
    return [path.read_text(encoding="utf-8")]


def test_the_runner_end_to_end_and_its_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    shutil.copyfile(DEFAULT_CONFIG_PATH, project / "config" / "default.yaml")
    inputs = make_inputs(load_config(DEFAULT_CONFIG_PATH).embedding.models)
    argv = ["--project-root", str(project), "--snapshot-id", SNAPSHOT]
    clean = lambda root, path: GitState(commit="c" * 40, dirty=())  # noqa: E731

    assert main(argv, load=lambda *_: inputs, git=clean) == 0
    first_out = capsys.readouterr().out

    results = project / "results" / "experiments" / SNAPSHOT / "t19"
    processed = project / "data" / "processed" / SNAPSHOT / "experiments" / "t19"
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["declaration_sha256"] == DECLARED_SHA256
    for folder, files in ((results, manifest["files"]), (processed, manifest["data_files"])):
        for name, digest in files.items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    assert manifest["counts"]["clusterings"] in (371, 376)
    assert manifest["repeatability"]["umap_identical"]
    assert manifest["repeatability"]["kmeans_identical"]
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    assert selection["provisional"] is True
    assert selection["winner"]["configuration"] in ("C1", "C2", "C3", "C4")
    runs = pq.read_table(results / "runs.parquet").to_pylist()
    assert len(runs) == manifest["counts"]["internal_metrics"]
    assert sum(row["final"] for row in runs) == manifest["counts"]["ctfidf_fits"]
    assert len(list((processed / "umap").glob("*.npy"))) == 10
    every = [p for p in project.rglob("*") if p.is_file()]
    assert not any(MARKER in text for path in every for text in strings_in(path))
    assert MARKER not in first_out
    assert "Provisional winner" in first_out

    before = {p: p.read_bytes() for p in every if p.name not in ("manifest.json", "runs.parquet")}
    assert main(argv, load=lambda *_: inputs, git=clean) == 0
    again = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert again["counts"]["reductions_reused"] == 10
    assert again["files"] == {**manifest["files"], "runs.parquet": again["files"]["runs.parquet"]}
    assert all(path.read_bytes() == content for path, content in before.items())
    drop = lambda rows: [{k: v for k, v in r.items() if k not in RUNTIME_FIELDS} for r in rows]  # noqa: E731
    assert drop(pq.read_table(results / "runs.parquet").to_pylist()) == drop(runs)
