"""Behavior of the application configuration loader."""

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from thematic_redundancy.shared.config import AppConfig, SnapshotConfig, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@pytest.fixture
def raw_default() -> dict[str, Any]:
    """Return the default configuration as plain YAML data, ready to be modified."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def load_variant(tmp_path: Path, data: dict[str, Any]) -> AppConfig:
    """Write ``data`` to a YAML file and load it through the public loader."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(config_path)


def set_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Replace the value at a dotted key such as ``"umap.n_neighbors"``."""
    *parents, leaf = dotted_key.split(".")
    node = data
    for parent in parents:
        node = node[parent]
    node[leaf] = value


def test_default_config_loads_with_the_declared_plan_values() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config.paths.data_dir == Path("data")
    assert str(config.repository.base_url).rstrip("/") == "https://repositorio.ucsm.edu.pe"
    assert config.repository.faculty_community_uuid == UUID("3e706f8d-cba0-4d4a-8d51-dd317c3e7652")
    assert (config.snapshot.year_start, config.snapshot.year_end) == (2021, 2026)
    assert config.snapshot.thesis_type == "tesis"
    assert [program.key for program in config.snapshot.programs] == [
        "sistemas",
        "industrial",
        "electronica",
        "mecanica",
        "minas",
    ]
    assert config.snapshot.programs[3].name == (
        "Ingeniería Mecánica, Mecánica-Eléctrica y Mecatrónica"
    )
    assert (config.ocr.languages, config.ocr.dpi) == ("spa+eng", 300)
    assert config.chunking.max_tokens == 128
    assert config.embedding.models == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    assert (config.umap.n_neighbors, config.umap.n_components) == (15, 5)
    assert (config.umap.min_dist, config.umap.metric) == (0.0, "cosine")
    assert (config.kmeans.k_start, config.kmeans.k_stop, config.kmeans.k_step) == (8, 50, 2)
    assert config.hdbscan.min_cluster_sizes == (5, 10, 15, 20)
    assert config.seeds == (7, 13, 21, 42, 73)
    assert config.redundancy.cv_folds == 5
    assert config.recommender.top_k == 5
    assert config.recommender.mmr_lambda == pytest.approx(0.7)
    assert (config.recommender.neighbor_topics, config.recommender.nearby_gaps) == (3, 3)
    assert config.gaps.temporal_split_year == 2024
    assert config.gaps.temporal_min_count == 3
    assert config.gaps.bridge_max_docs == 1
    assert config.gaps.nearest_topics == 3
    assert config.gaps.stability_min_runs == 4


def test_max_tokens_above_the_model_sequence_limit_is_rejected(
    raw_default: dict[str, Any], tmp_path: Path
) -> None:
    raw_default["chunking"]["max_tokens"] = 256

    with pytest.raises(ValidationError, match="max_tokens"):
        load_variant(tmp_path, raw_default)


@pytest.mark.parametrize(
    "absolute_path",
    ["/srv/thesis-data", "C:/thesis-data", "C:\\thesis-data", "\\\\server\\share\\data"],
)
def test_absolute_paths_are_rejected(
    raw_default: dict[str, Any], tmp_path: Path, absolute_path: str
) -> None:
    raw_default["paths"]["data_dir"] = absolute_path

    with pytest.raises(ValidationError, match="data_dir"):
        load_variant(tmp_path, raw_default)


def test_duplicate_seeds_are_rejected(raw_default: dict[str, Any], tmp_path: Path) -> None:
    raw_default["seeds"] = [7, 7, 21, 42, 73]

    with pytest.raises(ValidationError, match="seeds"):
        load_variant(tmp_path, raw_default)


def test_stability_min_runs_above_the_seed_count_is_rejected(
    raw_default: dict[str, Any], tmp_path: Path
) -> None:
    raw_default["gaps"]["stability_min_runs"] = len(raw_default["seeds"]) + 1

    with pytest.raises(ValidationError, match="stability_min_runs"):
        load_variant(tmp_path, raw_default)


@pytest.mark.parametrize("dotted_key", ["unexpected_key", "umap.unexpected_key"])
def test_unknown_keys_are_rejected(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str
) -> None:
    set_value(raw_default, dotted_key, 1)

    with pytest.raises(ValidationError, match="unexpected_key"):
        load_variant(tmp_path, raw_default)


def test_paths_resolve_against_the_given_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config(DEFAULT_CONFIG_PATH)

    resolved = config.paths.resolve_against(Path("project"))

    project_root = tmp_path.resolve() / "project"
    assert resolved.data_dir == project_root / "data"
    assert resolved.results_dir == project_root / "results"
    assert resolved.tessdata_dir == project_root / "tessdata"
    assert resolved.data_dir.is_absolute()


def test_loaded_config_is_immutable() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    with pytest.raises(ValidationError):
        config.chunking.max_tokens = 64


def test_snapshot_years_must_be_ordered(raw_default: dict[str, Any]) -> None:
    snapshot = raw_default["snapshot"] | {"year_start": 2027}

    with pytest.raises(ValidationError, match="year_start"):
        SnapshotConfig.model_validate(snapshot)


@pytest.mark.parametrize(
    ("dotted_key", "value"),
    [
        ("snapshot.programs", [{"key": "minas", "name": "A"}, {"key": "minas", "name": "B"}]),
        ("ocr.dpi", 71),
        ("ocr.dpi", 601),
        ("chunking.max_tokens", 15),
        ("embedding.models", []),
        ("embedding.models", ["org/model", "org/model"]),
        ("kmeans.k_start", 1),
        ("kmeans.k_stop", 8),
        ("kmeans.k_step", 0),
        ("hdbscan.min_cluster_sizes", [1, 5]),
        ("hdbscan.min_cluster_sizes", [5, 5]),
        ("seeds", [7, 13, 21, 42]),
        ("seeds", [7, 13, 21, 42, 73, 99]),
        ("redundancy.cv_folds", 1),
        ("recommender.top_k", 0),
        ("recommender.mmr_lambda", -0.1),
        ("recommender.mmr_lambda", 1.1),
        ("gaps.temporal_split_year", 2021),
        ("gaps.temporal_split_year", 2027),
    ],
)
def test_values_outside_the_contract_are_rejected(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str, value: Any
) -> None:
    set_value(raw_default, dotted_key, value)

    with pytest.raises(ValidationError):
        load_variant(tmp_path, raw_default)


@pytest.mark.parametrize(
    ("dotted_key", "value"),
    [
        ("ocr.dpi", 72),
        ("ocr.dpi", 600),
        ("chunking.max_tokens", 16),
        ("kmeans.k_start", 2),
        ("hdbscan.min_cluster_sizes", [2]),
        ("redundancy.cv_folds", 2),
        ("recommender.top_k", 1),
        ("recommender.mmr_lambda", 0.0),
        ("recommender.mmr_lambda", 1.0),
        ("gaps.temporal_split_year", 2026),
        ("gaps.stability_min_runs", 5),
    ],
)
def test_boundary_values_are_accepted(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str, value: Any
) -> None:
    set_value(raw_default, dotted_key, value)

    assert isinstance(load_variant(tmp_path, raw_default), AppConfig)
