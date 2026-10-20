"""The records of the experiment (T19, D30 E6) and the cache of its UMAP reductions.

Under ``results/experiments/<snapshot_id>/t19/``, which git tracks: ``runs.parquet``,
``configurations.parquet``, ``topics.parquet`` (vocabulary lemmas, no thesis text),
``selection.json`` and ``manifest.json``. Under the gitignored
``data/processed/<snapshot_id>/experiments/t19/``: ``labels.parquet`` (``run_id`` × ``doc_code``),
``map.parquet`` (the representative run's 2-D and PCA coordinates) and the reductions,
``umap/<model>__seed<seed>.npy``, each with its key in a ``.json`` beside it.

Every file is replaced atomically, and :func:`write_outputs` writes ``manifest.json`` last, as
the commit point: it lists the SHA-256 of every other file, so a reader can tell whether the
files on disk are the ones it describes.
"""

import hashlib
import io
import json
import os
import platform
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from thematic_redundancy.representation.vector_cache import model_slug
from thematic_redundancy.shared.filesystem import write_bytes_atomically, write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore
from thematic_redundancy.topic_map.experiment import (
    ConfigurationRow,
    ExperimentResult,
    MapRow,
    ReductionKey,
    RunRow,
    TopicRow,
)

EXPERIMENTS_DIR_NAME = "experiments"
EXPERIMENT_DIR_NAME = "t19"
PROCESSED_DIR_NAME = "processed"
RUNS_FILE_NAME = "runs.parquet"
CONFIGURATIONS_FILE_NAME = "configurations.parquet"
TOPICS_FILE_NAME = "topics.parquet"
SELECTION_FILE_NAME = "selection.json"
MANIFEST_FILE_NAME = "manifest.json"
LABELS_FILE_NAME = "labels.parquet"
MAP_FILE_NAME = "map.parquet"
UMAP_DIR_NAME = "umap"


def results_dir(results_root: Path, snapshot_id: str) -> Path:
    """Return ``<results_root>/experiments/<snapshot_id>/t19``."""
    return results_root / EXPERIMENTS_DIR_NAME / snapshot_id / EXPERIMENT_DIR_NAME


def processed_dir(data_root: Path, snapshot_id: str) -> Path:
    """Return ``<data_root>/processed/<snapshot_id>/experiments/t19``."""
    return data_root / PROCESSED_DIR_NAME / snapshot_id / EXPERIMENTS_DIR_NAME / EXPERIMENT_DIR_NAME


class LabelRow(BaseModel):
    """One row of ``labels.parquet``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    doc_code: str
    label: int


class ExperimentManifest(BaseModel):
    """Content of ``manifest.json``: provenance, counts and time, without text."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    snapshot_id: str
    declaration_sha256: str
    git_commit: str
    inputs: dict[str, str]
    """SHA-256 of the input manifests, by file name."""
    documents: int
    provisional: bool
    winner: str
    matched: str
    counts: dict[str, int]
    repeatability: dict[str, Any]
    versions: dict[str, str]
    cpu: dict[str, Any]
    files: dict[str, str]
    """SHA-256 of each file written to the results directory, by name."""
    data_files: dict[str, str]
    """SHA-256 of each file written to the data directory, by name."""
    started_at: datetime
    finished_at: datetime
    seconds: float


class FileReductionCache:
    """The :class:`~.experiment.ReductionCache` adapter: one ``.npy`` per (model, seed) and a
    ``.json`` beside it with its key, the library versions and the array's SHA-256.

    A reduction is reused only when all three match; the ``.json`` is written after the array,
    so an interrupted write is a miss, never a wrong reduction.
    """

    def __init__(self, directory: Path, versions: dict[str, str]) -> None:
        self._directory = directory
        self._versions = dict(versions)

    def get(self, key: ReductionKey) -> np.ndarray | None:
        """Return the saved reduction of ``key``, or ``None``; never creates a file."""
        array_path, key_path = self._paths(key)
        if not (array_path.is_file() and key_path.is_file()):
            return None
        content = array_path.read_bytes()
        try:
            record = json.loads(key_path.read_text(encoding="utf-8"))
            if record != self._record(key, hashlib.sha256(content).hexdigest()):
                return None
            return np.load(io.BytesIO(content), allow_pickle=False)
        except ValueError:
            return None

    def put(self, key: ReductionKey, embedding: np.ndarray) -> None:
        """Save the reduction of ``key``: the array first, then its key."""
        array_path, key_path = self._paths(key)
        self._directory.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(embedding), allow_pickle=False)
        content = buffer.getvalue()
        write_bytes_atomically(array_path, content)
        record = self._record(key, hashlib.sha256(content).hexdigest())
        write_text_atomically(key_path, json.dumps(record, sort_keys=True, indent=2))

    def _paths(self, key: ReductionKey) -> tuple[Path, Path]:
        stem = f"{model_slug(key.model_id)}__seed{key.seed}"
        return self._directory / f"{stem}.npy", self._directory / f"{stem}.json"

    def _record(self, key: ReductionKey, array_sha256: str) -> dict[str, Any]:
        return {
            "model_id": key.model_id,
            "seed": key.seed,
            "key_sha256": key.sha256,
            "versions": self._versions,
            "npy_sha256": array_sha256,
        }


LIBRARIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "numba",
    "pynndescent",
    "hdbscan",
    "bertopic",
    "pyarrow",
)
"""The libraries whose versions the records name: the bytes of the results depend on them."""


def library_versions() -> dict[str, str]:
    """Return the versions of Python and the libraries the results depend on."""
    return {"python": platform.python_version(), **{name: version(name) for name in LIBRARIES}}


def cpu_record() -> dict[str, Any]:
    """Return the processor, the machine type and the logical CPU count."""
    return {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "logical_cpus": os.cpu_count() or 0,
    }


def write_outputs(
    result: ExperimentResult,
    *,
    results: Path,
    processed: Path,
    declaration_sha256: str,
    git_commit: str,
    input_sha256: dict[str, str],
    started_at: datetime,
    finished_at: datetime,
    seconds: float,
    versions: dict[str, str],
    cpu: dict[str, Any],
    store: TableStore | None = None,
    write_text: Callable[[Path, str], None] = write_text_atomically,
) -> ExperimentManifest:
    """Write the records of ``result``: the data files, then the tables and ``selection.json``,
    then ``manifest.json`` last, and return the manifest.

    Raises:
        OSError: if writing fails; the previous manifest then stays, and no longer matches
            the files that were replaced.
    """
    store = store or ParquetTableStore()
    results.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    labels = [
        LabelRow.model_construct(run_id=run, doc_code=code, label=int(label))
        for run, values in result.labels.items()
        for code, label in zip(result.doc_codes, values.tolist(), strict=True)
    ]
    store.write(labels, processed / LABELS_FILE_NAME, model_cls=LabelRow)
    store.write(result.map, processed / MAP_FILE_NAME, model_cls=MapRow)
    store.write(result.runs, results / RUNS_FILE_NAME, model_cls=RunRow)
    store.write(
        result.configurations, results / CONFIGURATIONS_FILE_NAME, model_cls=ConfigurationRow
    )
    store.write(result.topics, results / TOPICS_FILE_NAME, model_cls=TopicRow)
    text = json.dumps(result.selection, indent=2, sort_keys=True, ensure_ascii=False)
    write_text(results / SELECTION_FILE_NAME, text)
    repeatability = result.repeatability
    manifest = ExperimentManifest(
        snapshot_id=result.snapshot_id,
        declaration_sha256=declaration_sha256,
        git_commit=git_commit,
        inputs=input_sha256,
        documents=len(result.doc_codes),
        provisional=bool(result.selection["provisional"]),
        winner=str(result.selection["winner"]["configuration"]),
        matched=result.matched,
        counts={**asdict(result.counts), "reductions_reused": result.reductions_reused},
        repeatability={
            **asdict(repeatability),
            "umap_sha256": list(repeatability.umap_sha256),
            "kmeans_sha256": list(repeatability.kmeans_sha256),
            "umap_identical": repeatability.umap_identical,
            "kmeans_identical": repeatability.kmeans_identical,
        },
        versions=versions,
        cpu=cpu,
        files=_digests(
            results,
            (RUNS_FILE_NAME, CONFIGURATIONS_FILE_NAME, TOPICS_FILE_NAME, SELECTION_FILE_NAME),
        ),
        data_files=_digests(processed, (LABELS_FILE_NAME, MAP_FILE_NAME)),
        started_at=started_at,
        finished_at=finished_at,
        seconds=round(max(seconds, 0.0), 3),
    )
    # The commit point: until it is written, the manifest describes the previous files.
    write_text(results / MANIFEST_FILE_NAME, manifest.model_dump_json(indent=2))
    return manifest


def _digests(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
