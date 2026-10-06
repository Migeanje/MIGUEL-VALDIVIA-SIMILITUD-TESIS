"""Behavior of the application configuration loader."""

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from thematic_redundancy.shared.config import (
    AppConfig,
    KMeansConfig,
    PathsConfig,
    SnapshotConfig,
    load_config,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

SISTEMAS_COLLECTION = "695b14ab-e5b2-49b5-9edf-f709e883b73b"
"""Collection of the first declared program, ``sistemas``."""


@pytest.fixture
def raw_default() -> dict[str, Any]:
    """Return the default configuration as plain YAML data, ready to be modified."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def load_text(tmp_path: Path, text: str) -> AppConfig:
    """Write ``text`` to a YAML file and load it through the public loader."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return load_config(config_path)


def load_variant(tmp_path: Path, data: dict[str, Any]) -> AppConfig:
    """Dump ``data`` as YAML and load it through the public loader."""
    return load_text(tmp_path, yaml.safe_dump(data, allow_unicode=True))


def set_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Replace the value at a dotted key such as ``"umap.n_neighbors"``."""
    *parents, leaf = dotted_key.split(".")
    node = data
    for parent in parents:
        node = node[parent]
    node[leaf] = value


def load_with_literal(
    tmp_path: Path, data: dict[str, Any], dotted_key: str, literal: str
) -> AppConfig:
    """Load ``data`` with the value at ``dotted_key`` written as the YAML text ``literal``.

    This reaches spellings that a dump does not produce, such as merge keys or plain
    scientific notation: the literal replaces a placeholder value in the dumped text.
    """
    placeholder = "LITERAL_VALUE_PLACEHOLDER"
    set_value(data, dotted_key, placeholder)
    text = yaml.safe_dump(data, allow_unicode=True)
    assert text.count(placeholder) == 1
    return load_text(tmp_path, text.replace(placeholder, literal))


def test_default_config_loads_with_the_declared_plan_values() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config.paths.data_dir == Path("data")
    assert str(config.repository.base_url).rstrip("/") == "https://repositorio.ucsm.edu.pe"
    assert config.repository.faculty_community_uuid == UUID("3e706f8d-cba0-4d4a-8d51-dd317c3e7652")
    assert (config.repository.page_size, config.repository.max_retries) == (100, 3)
    assert config.repository.request_interval_seconds == 1.0
    assert (config.snapshot.year_start, config.snapshot.year_end) == (2021, 2026)
    assert config.snapshot.thesis_type == "tesis"
    assert [(program.key, program.collection_uuid) for program in config.snapshot.programs] == [
        ("sistemas", UUID(SISTEMAS_COLLECTION)),
        ("industrial", UUID("f0217548-ce48-4bae-b163-21b5e2504ef6")),
        ("electronica", UUID("a8bdd7c0-f1aa-4797-b04e-112e27f60da3")),
        ("mecanica", UUID("cf0ca97e-9b81-484d-a325-611b8a8d8223")),
        ("minas", UUID("9d9ecbe5-56c5-4cdf-81a3-b9a3736a47b3")),
    ]
    assert config.snapshot.programs[3].name == (
        "Ingeniería Mecánica, Mecánica-Eléctrica y Mecatrónica"
    )
    assert (config.ocr.languages, config.ocr.dpi) == ("spa+eng", 300)
    assert (config.extraction.min_text_chars, config.extraction.ocr_window_pages) == (50, 100)
    header_footer = config.extraction.header_footer
    assert (header_footer.edge_lines, header_footer.min_pages) == (3, 5)
    assert header_footer.min_share == pytest.approx(0.3)
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


@pytest.mark.parametrize("climbing_path", ["../outside", "data/../../x", "data\\..\\..\\x"])
def test_paths_with_parent_segments_are_rejected(
    raw_default: dict[str, Any], tmp_path: Path, climbing_path: str
) -> None:
    raw_default["paths"]["data_dir"] = climbing_path

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


@pytest.mark.parametrize("k_step", [3, 9])
def test_kmeans_sweep_that_would_miss_k_stop_is_rejected(k_step: int) -> None:
    # The sweep spans 11 - 3 = 8, which neither step divides.
    with pytest.raises(ValidationError, match="k_step"):
        KMeansConfig(k_start=3, k_stop=11, k_step=k_step)


@pytest.mark.parametrize(
    ("k_start", "k_stop", "k_step", "expected"),
    [(2, 3, 1, (2, 3)), (3, 11, 4, (3, 7, 11)), (5, 20, 15, (5, 20))],
    ids=["smallest-k_start", "k_stop-reached-in-steps", "k_step-equal-to-span"],
)
def test_kmeans_sweep_includes_both_k_start_and_k_stop(
    k_start: int, k_stop: int, k_step: int, expected: tuple[int, ...]
) -> None:
    sweep = KMeansConfig(k_start=k_start, k_stop=k_stop, k_step=k_step)

    assert sweep.k_values() == expected


# Each sweep breaks exactly one rule, so a missing check cannot hide behind another one.
@pytest.mark.parametrize(
    ("k_start", "k_stop", "k_step", "field"),
    [(1, 3, 1, "k_start"), (3, 11, 0, "k_step"), (10, 10, 1, "k_start"), (11, 10, 1, "k_start")],
    ids=["k_start-below-2", "zero-k_step", "k_start-equal-to-k_stop", "k_start-above-k_stop"],
)
def test_kmeans_sweep_with_bounds_outside_the_contract_is_rejected(
    k_start: int, k_stop: int, k_step: int, field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        KMeansConfig(k_start=k_start, k_stop=k_stop, k_step=k_step)


@pytest.mark.parametrize(
    "dotted_key",
    [
        "unexpected_key",
        "umap.unexpected_key",
        "extraction.unexpected_key",
        "extraction.header_footer.unexpected_key",
    ],
)
def test_unknown_keys_are_rejected(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str
) -> None:
    set_value(raw_default, dotted_key, 1)

    with pytest.raises(ValidationError, match="unexpected_key"):
        load_variant(tmp_path, raw_default)


@pytest.mark.parametrize(
    ("written_line", "repeated_line", "key"),
    [
        ("seeds: [7, 13, 21, 42, 73]\n", "seeds: [1, 2, 3, 4, 5]\n", "seeds"),
        ("  n_neighbors: 15\n", "  n_neighbors: 30\n", "n_neighbors"),
        ("    min_share: 0.3\n", "    min_share: 0.9\n", "min_share"),
    ],
    ids=["seeds", "umap.n_neighbors", "extraction.header_footer.min_share"],
)
def test_keys_written_twice_are_rejected(
    tmp_path: Path, written_line: str, repeated_line: str, key: str
) -> None:
    default_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert default_text.count(written_line) == 1
    text = default_text.replace(written_line, written_line + repeated_line)

    with pytest.raises(yaml.YAMLError, match=f"duplicate key '{key}'"):
        load_text(tmp_path, text)


def test_a_written_key_overrides_a_merged_key(raw_default: dict[str, Any], tmp_path: Path) -> None:
    umap_literal = (
        "{<<: {n_neighbors: 10, n_components: 2, min_dist: 0.5, metric: euclidean},"
        " n_neighbors: 30}"
    )

    umap = load_with_literal(tmp_path, raw_default, "umap", umap_literal).umap

    assert umap.n_neighbors == 30
    assert (umap.n_components, umap.min_dist, umap.metric) == (2, 0.5, "euclidean")


@pytest.mark.parametrize(
    ("dotted_key", "literal", "key"),
    [
        (
            "umap",
            "{<<: {n_neighbors: 10, n_neighbors: 20, n_components: 2, min_dist: 0.5,"
            " metric: euclidean}}",
            "n_neighbors",
        ),
        (
            "umap",
            "{<<: [{n_neighbors: 10, n_neighbors: 20},"
            " {n_components: 2, min_dist: 0.5, metric: euclidean}]}",
            "n_neighbors",
        ),
        (
            "snapshot.programs",
            "[&base {key: sistemas, key: minas, name: Engineering}, {<<: *base, key: industrial}]",
            "key",
        ),
    ],
    ids=["inline-source", "source-in-a-list", "anchored-source"],
)
def test_keys_written_twice_inside_a_merge_source_are_rejected(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str, literal: str, key: str
) -> None:
    with pytest.raises(yaml.YAMLError, match=f"duplicate key '{key}'"):
        load_with_literal(tmp_path, raw_default, dotted_key, literal)


# Each program keeps its own collection uuid, so only the name comes from the anchor.
@pytest.mark.parametrize(
    "programs_literal",
    [
        "[&base {key: sistemas, name: Engineering,"
        " collection_uuid: '00000000-0000-4000-8000-000000000001'},"
        " {<<: *base, key: industrial, collection_uuid: '00000000-0000-4000-8000-000000000002'},"
        " {<<: *base, key: minas, collection_uuid: '00000000-0000-4000-8000-000000000003'}]",
        "[{<<: &base {name: Engineering}, key: sistemas,"
        " collection_uuid: '00000000-0000-4000-8000-000000000001'},"
        " {<<: *base, key: industrial, collection_uuid: '00000000-0000-4000-8000-000000000002'},"
        " {<<: *base, key: minas, collection_uuid: '00000000-0000-4000-8000-000000000003'}]",
    ],
    ids=["anchored-item", "anchored-merge-source"],
)
def test_an_anchor_reused_by_several_mappings_loads(
    raw_default: dict[str, Any], tmp_path: Path, programs_literal: str
) -> None:
    config = load_with_literal(tmp_path, raw_default, "snapshot.programs", programs_literal)

    assert [(program.key, program.name) for program in config.snapshot.programs] == [
        ("sistemas", "Engineering"),
        ("industrial", "Engineering"),
        ("minas", "Engineering"),
    ]


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


@pytest.mark.parametrize("nested_path", ["data/sub", "data/v1..v2"])
def test_nested_paths_resolve_inside_the_project_root(
    raw_default: dict[str, Any], tmp_path: Path, nested_path: str
) -> None:
    raw_default["paths"]["data_dir"] = nested_path

    resolved = load_variant(tmp_path, raw_default).paths.resolve_against(tmp_path)

    assert resolved.data_dir == tmp_path.resolve() / nested_path


def test_resolution_refuses_a_location_outside_the_project_root(tmp_path: Path) -> None:
    # model_construct skips validation. It stands in for a location that only leaves
    # the root once resolved, such as a symbolic link that points elsewhere.
    unchecked = PathsConfig.model_construct(
        data_dir=Path("../outside"), results_dir=Path("results"), tessdata_dir=Path("tessdata")
    )

    with pytest.raises(ValueError, match="data_dir"):
        unchecked.resolve_against(tmp_path / "project")


def test_resolution_refuses_a_symbolic_link_that_leaves_the_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    outside = tmp_path / "outside"
    project_root.mkdir()
    outside.mkdir()
    try:
        (project_root / "data").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(
            "cannot create a directory symlink here; on Windows this needs Developer Mode "
            f"or an elevated shell ({error})"
        )
    paths = PathsConfig(
        data_dir=Path("data"), results_dir=Path("results"), tessdata_dir=Path("tessdata")
    )

    with pytest.raises(ValueError, match="data_dir resolves to .*, outside the project root"):
        paths.resolve_against(project_root)


def test_loaded_config_is_immutable() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    with pytest.raises(ValidationError):
        config.chunking.max_tokens = 64


def test_snapshot_years_must_be_ordered(raw_default: dict[str, Any]) -> None:
    snapshot = raw_default["snapshot"] | {"year_start": 2027}

    with pytest.raises(ValidationError, match="year_start"):
        SnapshotConfig.model_validate(snapshot)


def test_a_program_without_a_collection_uuid_is_rejected(
    raw_default: dict[str, Any], tmp_path: Path
) -> None:
    del raw_default["snapshot"]["programs"][1]["collection_uuid"]

    with pytest.raises(ValidationError, match="collection_uuid"):
        load_variant(tmp_path, raw_default)


@pytest.mark.parametrize(
    ("collection_uuid", "message"),
    [("not-a-uuid", "collection_uuid"), (SISTEMAS_COLLECTION, "collection uuids must be unique")],
    ids=["malformed", "shared-with-another-program"],
)
def test_a_program_collection_that_is_malformed_or_shared_is_rejected(
    raw_default: dict[str, Any], tmp_path: Path, collection_uuid: str, message: str
) -> None:
    raw_default["snapshot"]["programs"][1]["collection_uuid"] = collection_uuid

    with pytest.raises(ValidationError, match=message):
        load_variant(tmp_path, raw_default)


def test_harvest_settings_take_their_defaults_when_omitted(
    raw_default: dict[str, Any], tmp_path: Path
) -> None:
    for key in ("page_size", "request_interval_seconds", "max_retries"):
        del raw_default["repository"][key]

    repository = load_variant(tmp_path, raw_default).repository

    assert (repository.page_size, repository.max_retries) == (100, 3)
    assert repository.request_interval_seconds == 1.0


@pytest.mark.parametrize(
    ("dotted_key", "value"),
    [
        (
            "snapshot.programs",
            [
                {
                    "key": "minas",
                    "name": "A",
                    "collection_uuid": "00000000-0000-4000-8000-00000000000a",
                },
                {
                    "key": "minas",
                    "name": "B",
                    "collection_uuid": "00000000-0000-4000-8000-00000000000b",
                },
            ],
        ),
        ("repository.page_size", 0),
        ("repository.page_size", 1001),
        ("repository.request_interval_seconds", 0.5),
        ("repository.request_interval_seconds", 60.5),
        ("repository.max_retries", -1),
        ("repository.max_retries", 11),
        ("ocr.dpi", 71),
        ("ocr.dpi", 601),
        ("extraction.min_text_chars", 0),
        ("extraction.ocr_window_pages", -1),
        ("extraction.header_footer.edge_lines", 0),
        ("extraction.header_footer.edge_lines", 11),
        ("extraction.header_footer.min_share", 0.0),
        ("extraction.header_footer.min_share", 1.1),
        ("extraction.header_footer.min_pages", 1),
        ("chunking.max_tokens", 15),
        ("embedding.models", []),
        ("embedding.models", ["org/model", "org/model"]),
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


# Lax validation would turn "15" into 15 and true into 1, so these values would either load
# or fail a range check. The error type shows that the input type itself is refused.
@pytest.mark.parametrize(
    ("dotted_key", "value", "error_type"),
    [
        ("umap.n_neighbors", "15", "int_type"),
        ("repository.page_size", "100", "int_type"),
        ("repository.max_retries", 3.0, "int_type"),
        ("repository.request_interval_seconds", "1.0", "float_type"),
        ("ocr.dpi", True, "int_type"),
        ("recommender.top_k", True, "int_type"),
        ("kmeans.k_step", 2.0, "int_type"),
        ("seeds", ["7", 13, 21, 42, 73], "int_type"),
        ("recommender.mmr_lambda", "0.7", "float_type"),
        ("extraction.min_text_chars", "50", "int_type"),
        ("extraction.ocr_window_pages", 100.0, "int_type"),
        ("extraction.header_footer.edge_lines", True, "int_type"),
        ("extraction.header_footer.min_share", "0.3", "float_type"),
    ],
)
def test_numeric_fields_refuse_values_of_another_type(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str, value: Any, error_type: str
) -> None:
    set_value(raw_default, dotted_key, value)

    with pytest.raises(ValidationError) as caught:
        load_variant(tmp_path, raw_default)

    assert error_type in {error["type"] for error in caught.value.errors()}


# Under YAML 1.1 a float needs a dot, and an exponent needs a sign, so PyYAML reads 1e-3
# as a string. The loader reads every YAML 1.2 exponent form as a float.
@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("1e-3", 0.001),
        ("7e-1", 0.7),
        ("7E-1", 0.7),
        ("+70e-2", 0.7),
        (".7e0", 0.7),
        ("0.07e1", 0.7),
        ("1E0", 1.0),
        ("7.0e-1", 0.7),
    ],
)
def test_float_fields_accept_scientific_notation(
    raw_default: dict[str, Any], tmp_path: Path, notation: str, expected: float
) -> None:
    config = load_with_literal(tmp_path, raw_default, "recommender.mmr_lambda", notation)

    assert config.recommender.mmr_lambda == pytest.approx(expected)


def test_scientific_notation_above_a_float_bound_fails_the_range_check(
    raw_default: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(ValidationError) as caught:
        load_with_literal(tmp_path, raw_default, "recommender.mmr_lambda", "1E3")

    assert {error["type"] for error in caught.value.errors()} == {"less_than_equal"}


@pytest.mark.parametrize("notation", ["300.0", "3e2", "3.0e+2"])
def test_integer_fields_refuse_floats_in_any_notation(
    raw_default: dict[str, Any], tmp_path: Path, notation: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        load_with_literal(tmp_path, raw_default, "ocr.dpi", notation)

    assert {error["type"] for error in caught.value.errors()} == {"int_type"}


def test_the_global_safe_loader_still_reads_plain_exponents_as_text() -> None:
    assert yaml.safe_load("1e-3") == "1e-3"


@pytest.mark.parametrize(
    ("dotted_key", "value"),
    [
        ("repository.page_size", 1),
        ("repository.page_size", 1000),
        ("repository.request_interval_seconds", 1),
        ("repository.request_interval_seconds", 60.0),
        ("repository.max_retries", 0),
        ("repository.max_retries", 10),
        ("ocr.dpi", 72),
        ("ocr.dpi", 600),
        ("extraction.min_text_chars", 1),
        ("extraction.ocr_window_pages", 0),
        ("extraction.header_footer.edge_lines", 1),
        ("extraction.header_footer.edge_lines", 10),
        ("extraction.header_footer.min_share", 1.0),
        ("extraction.header_footer.min_share", 1),
        ("extraction.header_footer.min_pages", 2),
        ("chunking.max_tokens", 16),
        ("umap.min_dist", 0),
        ("hdbscan.min_cluster_sizes", [2]),
        ("redundancy.cv_folds", 2),
        ("recommender.top_k", 1),
        ("recommender.mmr_lambda", 0.0),
        ("recommender.mmr_lambda", 1.0),
        ("recommender.mmr_lambda", 1),
        ("gaps.temporal_split_year", 2026),
        ("gaps.stability_min_runs", 5),
    ],
)
def test_boundary_values_are_accepted(
    raw_default: dict[str, Any], tmp_path: Path, dotted_key: str, value: Any
) -> None:
    set_value(raw_default, dotted_key, value)

    assert isinstance(load_variant(tmp_path, raw_default), AppConfig)
