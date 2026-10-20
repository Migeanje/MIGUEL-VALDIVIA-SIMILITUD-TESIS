"""The records of the experiment (T19, D30 E6): every file atomic, ``manifest.json`` last, and
the cache of the UMAP reductions, on hand-made results."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import NoneType
from typing import Any, get_args

import numpy as np
import pytest
from pydantic import BaseModel

from thematic_redundancy.shared.filesystem import write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.experiment import (
    ConfigurationRow,
    ExperimentResult,
    MapRow,
    ReductionKey,
    Repeatability,
    RunCounts,
    RunRow,
    TopicRow,
)
from thematic_redundancy.topic_map.results_store import (
    FileReductionCache,
    LabelRow,
    cpu_record,
    library_versions,
    write_outputs,
)

STARTED = datetime(2026, 10, 19, 22, 0, tzinfo=UTC)
VERSIONS = {"umap-learn": "0.5.12", "numba": "0.62.0"}
RESULT_FILES = [
    "runs.parquet",
    "configurations.parquet",
    "topics.parquet",
    "selection.json",
    "manifest.json",
]


def row(model: type[BaseModel], **values: Any) -> Any:
    """Return ``model`` with ``values`` and a plain value for every other field."""
    plain = {bool: False, int: 0, float: 0.0, str: "x"}
    for name, field in model.model_fields.items():
        if name not in values:
            annotation = field.annotation
            values[name] = None if NoneType in get_args(annotation) else plain.get(annotation, ())
    return model(**values)


def make_result() -> ExperimentResult:
    run_ids = ("tfidf__kmeans_k2__seed7", "tfidf__kmeans_k2__seed13")
    return ExperimentResult(
        snapshot_id="S1",
        doc_codes=("D1", "D2", "D3"),
        runs=tuple(row(RunRow, run_id=r, k=2, final=True, npmi=0.25) for r in run_ids),
        configurations=(row(ConfigurationRow, configuration="B0", chosen=True, ari_mean=0.9),),
        topics=(row(TopicRow, run_id=run_ids[0], terms=("voladura", "roca"), scores=(0.5, 0.25)),),
        labels={r: np.array([0, 1, 1]) for r in run_ids},
        map=tuple(
            row(MapRow, run_id=run_ids[0], doc_code=d, map_x=1.5) for d in ("D1", "D2", "D3")
        ),
        selection={"provisional": True, "winner": {"configuration": "C1"}},
        repeatability=Repeatability("m/a", 7, 8, ("u", "u"), ("k", "k")),
        counts=RunCounts(12, 1, 371, 25, 370, 40, 50, 148),
        matched="baseline",
        reductions_reused=3,
    )


def write(tmp_path: Path, **kwargs: Any) -> Any:
    return write_outputs(
        make_result(),
        results=tmp_path / "results",
        processed=tmp_path / "data",
        declaration_sha256="d" * 64,
        git_commit="c" * 40,
        started_at=STARTED,
        finished_at=STARTED,
        seconds=12.5,
        versions=VERSIONS,
        cpu={"logical_cpus": 16},
        input_sha256={"tfidf_manifest.json": "t" * 64},
        **kwargs,
    )


class RecordingStore(ParquetTableStore):
    def __init__(self, written: list[str], fail_on: str | None = None) -> None:
        self.written, self.fail_on = written, fail_on

    def write(self, models: Any, path: Path, *, model_cls: Any = None) -> None:
        if path.name == self.fail_on:
            raise OSError("disk full")
        super().write(models, path, model_cls=model_cls)
        self.written.append(path.name)


def test_every_record_is_written_and_the_manifest_last(tmp_path: Path) -> None:
    written: list[str] = []

    def write_text(path: Path, text: str) -> None:
        write_text_atomically(path, text)
        written.append(path.name)

    manifest = write(tmp_path, store=RecordingStore(written), write_text=write_text)

    assert written == ["labels.parquet", "map.parquet", *RESULT_FILES]
    results = tmp_path / "results"
    assert sorted(p.name for p in results.iterdir()) == sorted(RESULT_FILES)
    for name, digest in {**manifest.files, **manifest.data_files}.items():
        folder = results if name in manifest.files else tmp_path / "data"
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    assert set(manifest.files) == set(RESULT_FILES) - {"manifest.json"}
    assert set(manifest.data_files) == {"labels.parquet", "map.parquet"}
    saved = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert saved == json.loads(manifest.model_dump_json())


def test_the_tables_read_back_as_written(tmp_path: Path) -> None:
    write(tmp_path)
    store, result = ParquetTableStore(), make_result()

    assert store.read(RunRow, tmp_path / "results" / "runs.parquet") == result.runs
    assert store.read(TopicRow, tmp_path / "results" / "topics.parquet") == result.topics
    configurations = store.read(ConfigurationRow, tmp_path / "results" / "configurations.parquet")
    assert configurations == result.configurations
    labels = store.read(LabelRow, tmp_path / "data" / "labels.parquet")
    assert [(r.run_id, r.doc_code, r.label) for r in labels][:3] == [
        ("tfidf__kmeans_k2__seed7", "D1", 0),
        ("tfidf__kmeans_k2__seed7", "D2", 1),
        ("tfidf__kmeans_k2__seed7", "D3", 1),
    ]
    selection = json.loads((tmp_path / "results" / "selection.json").read_text(encoding="utf-8"))
    assert selection == result.selection


def test_the_manifest_records_the_provenance(tmp_path: Path) -> None:
    manifest = write(tmp_path)

    assert (manifest.declaration_sha256, manifest.git_commit) == ("d" * 64, "c" * 40)
    assert manifest.inputs == {"tfidf_manifest.json": "t" * 64}
    assert (manifest.snapshot_id, manifest.documents, manifest.provisional) == ("S1", 3, True)
    assert manifest.winner == "C1"
    assert manifest.counts["clusterings"] == 371
    assert manifest.counts["reductions_reused"] == 3
    assert manifest.matched == "baseline"
    assert manifest.repeatability == {
        "model_id": "m/a",
        "seed": 7,
        "k": 8,
        "umap_sha256": ["u", "u"],
        "kmeans_sha256": ["k", "k"],
        "umap_identical": True,
        "kmeans_identical": True,
    }
    assert (manifest.versions, manifest.cpu, manifest.seconds) == (
        VERSIONS,
        {"logical_cpus": 16},
        12.5,
    )


def test_a_failed_write_keeps_the_previous_manifest(tmp_path: Path) -> None:
    write(tmp_path)
    before = (tmp_path / "results" / "manifest.json").read_bytes()

    with pytest.raises(OSError):
        write(tmp_path, store=RecordingStore([], fail_on="topics.parquet"))

    assert (tmp_path / "results" / "manifest.json").read_bytes() == before


KEY = ReductionKey(model_id="org/model-a", seed=7, sha256="a" * 64)


def test_the_cache_returns_a_reduction_it_saved(tmp_path: Path) -> None:
    embedding = np.arange(12, dtype=np.float32).reshape(4, 3)
    FileReductionCache(tmp_path / "umap", VERSIONS).put(KEY, embedding)

    got = FileReductionCache(tmp_path / "umap", VERSIONS).get(KEY)

    assert got is not None and got.dtype == np.float32
    assert np.array_equal(got, embedding)
    assert sorted(p.name for p in (tmp_path / "umap").iterdir()) == [
        "org--model-a__seed7.json",
        "org--model-a__seed7.npy",
    ]


def test_the_cache_misses_another_key_other_versions_or_changed_bytes(tmp_path: Path) -> None:
    cache = FileReductionCache(tmp_path / "umap", VERSIONS)
    cache.put(KEY, np.ones((4, 3), dtype=np.float32))
    other = FileReductionCache(tmp_path / "umap", {**VERSIONS, "numba": "0.63.0"})

    assert cache.get(ReductionKey("org/model-a", 7, "b" * 64)) is None
    assert cache.get(ReductionKey("org/model-a", 13, "a" * 64)) is None
    assert other.get(KEY) is None
    (tmp_path / "umap" / "org--model-a__seed7.npy").write_bytes(b"changed")
    assert cache.get(KEY) is None


def test_reading_an_empty_cache_creates_nothing(tmp_path: Path) -> None:
    assert FileReductionCache(tmp_path / "umap", VERSIONS).get(KEY) is None
    assert not (tmp_path / "umap").exists()


def test_the_versions_and_the_cpu_are_recorded() -> None:
    versions = library_versions()

    assert {"python", "numpy", "scikit-learn", "umap-learn", "numba", "hdbscan", "bertopic"} <= set(
        versions
    )
    assert cpu_record()["logical_cpus"] >= 1
