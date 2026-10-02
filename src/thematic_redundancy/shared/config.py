"""Typed, immutable run configuration loaded from YAML."""

import re
from collections import Counter
from collections.abc import Hashable, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Self
from uuid import UUID

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError

MODEL_MAX_SEQ_LENGTH = 128
"""Verified ``max_seq_length`` of both multilingual SBERT models; no chunk may exceed it."""

SEED_COUNT = 5
"""Number of fixed seeds that every stochastic step is repeated with."""

NonEmptyStr = Annotated[str, Field(min_length=1)]

# Numeric fields use pydantic's strict types: a quoted "15" or a boolean is refused instead
# of coerced, while StrictFloat still accepts an integer such as 0. Model-wide strict mode is
# avoided because it would also refuse the strings that YAML gives for paths and UUIDs, and
# the lists that it gives for tuples.
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
Seed = Annotated[StrictInt, Field(ge=0, lt=2**32)]


def _require_unique(values: Sequence[Hashable], label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} must be unique; duplicated: {duplicates}")


def _is_anchored(raw_path: str) -> bool:
    """Tell whether a path is absolute or rooted under either Windows or POSIX rules."""
    return bool(PureWindowsPath(raw_path).anchor or PurePosixPath(raw_path).anchor)


def _has_parent_segment(raw_path: str) -> bool:
    """Tell whether a path has a ``..`` segment under either Windows or POSIX rules."""
    return ".." in PureWindowsPath(raw_path).parts or ".." in PurePosixPath(raw_path).parts


def _resolve_inside(root: Path, location: Path, field_name: str) -> Path:
    """Resolve ``location`` under the resolved ``root`` and refuse any result outside it."""
    resolved = (root / location).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"paths.{field_name} resolves to '{resolved}', outside the project root '{root}'"
        )
    return resolved


class _FrozenModel(BaseModel):
    """Base for configuration models: immutable and closed to unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ResolvedPaths(_FrozenModel):
    """Absolute locations derived from :class:`PathsConfig` for one project root."""

    data_dir: Path
    results_dir: Path
    tessdata_dir: Path


class PathsConfig(_FrozenModel):
    """Locations relative to the project root, so one file works on every machine."""

    data_dir: Path
    results_dir: Path
    tessdata_dir: Path

    @field_validator("data_dir", "results_dir", "tessdata_dir")
    @classmethod
    def _require_relative_without_parent_segments(cls, value: Path) -> Path:
        """Refuse absolute or rooted paths and ``..`` segments, judging the text alone.

        Nothing touches the file system here, so a symbolic link can still lead outside
        the project root. :meth:`resolve_against` enforces containment once the root is known.
        """
        raw_path = str(value)
        if _is_anchored(raw_path):
            raise ValueError(f"must be relative to the project root, got '{value}'")
        if _has_parent_segment(raw_path):
            raise ValueError(f"must not contain '..' segments, got '{value}'")
        return value

    def resolve_against(self, project_root: Path) -> ResolvedPaths:
        """Join every location onto ``project_root`` and return absolute paths.

        Raises:
            ValueError: if a location resolves outside the project root, for example
                through a symbolic link that points elsewhere.
        """
        root = Path(project_root).resolve()
        return ResolvedPaths(
            data_dir=_resolve_inside(root, self.data_dir, "data_dir"),
            results_dir=_resolve_inside(root, self.results_dir, "results_dir"),
            tessdata_dir=_resolve_inside(root, self.tessdata_dir, "tessdata_dir"),
        )


class RepositoryConfig(_FrozenModel):
    """Public DSpace repository, the faculty community that holds the programs, and the
    pace of the metadata harvest."""

    base_url: HttpUrl
    faculty_community_uuid: UUID
    page_size: Annotated[StrictInt, Field(ge=1, le=1000)] = 100
    """Items requested per search page."""
    request_interval_seconds: Annotated[StrictFloat, Field(ge=1.0, le=60.0)] = 1.0
    """Shortest time between the starts of two requests; at least one second, to stay polite."""
    max_retries: Annotated[StrictInt, Field(ge=0, le=10)] = 3
    """Retries of one request after HTTP 429, HTTP 5xx, a timeout or a network error."""


class ProgramConfig(_FrozenModel):
    """Academic program: a stable short key, its official name, and its repository collection."""

    key: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    name: NonEmptyStr
    collection_uuid: UUID
    """DSpace collection that holds the program's items; an item's program comes from it."""


class SnapshotConfig(_FrozenModel):
    """Filter that selects the theses of the frozen corpus snapshot."""

    year_start: StrictInt
    year_end: StrictInt
    thesis_type: NonEmptyStr
    programs: Annotated[tuple[ProgramConfig, ...], Field(min_length=1)]

    @field_validator("programs")
    @classmethod
    def _require_distinct_programs(
        cls, value: tuple[ProgramConfig, ...]
    ) -> tuple[ProgramConfig, ...]:
        _require_unique([program.key for program in value], "program keys")
        # Two programs on one collection would list, and count, the same items twice.
        _require_unique([program.collection_uuid for program in value], "program collection uuids")
        return value

    @model_validator(mode="after")
    def _require_ordered_years(self) -> Self:
        if self.year_start > self.year_end:
            raise ValueError(
                f"year_start ({self.year_start}) must not be after year_end ({self.year_end})"
            )
        return self


class OcrConfig(_FrozenModel):
    """Tesseract settings for pages without a text layer."""

    languages: Annotated[str, Field(pattern=r"^[A-Za-z_]+(\+[A-Za-z_]+)*$")]
    """Tesseract language spec, for example ``spa+eng``."""
    dpi: Annotated[StrictInt, Field(ge=72, le=600)]


class ChunkingConfig(_FrozenModel):
    """Sentence-aligned chunk budget, special tokens included, under both tokenizers."""

    max_tokens: Annotated[StrictInt, Field(ge=16, le=MODEL_MAX_SEQ_LENGTH)]


class EmbeddingConfig(_FrozenModel):
    """Sentence-transformers model ids compared in the experiment."""

    models: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]

    @field_validator("models")
    @classmethod
    def _require_unique_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(value, "embedding models")
        return value


class UmapConfig(_FrozenModel):
    """UMAP reduction applied before clustering."""

    n_neighbors: Annotated[StrictInt, Field(ge=2)]
    n_components: PositiveInt
    min_dist: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    """Bounded by UMAP's default ``spread`` of 1.0."""
    metric: NonEmptyStr


class KMeansConfig(_FrozenModel):
    """K sweep for K-means: from ``k_start`` to ``k_stop`` (inclusive) every ``k_step``."""

    k_start: Annotated[StrictInt, Field(ge=2)]
    k_stop: StrictInt
    k_step: PositiveInt

    @model_validator(mode="after")
    def _check_sweep_rules(self) -> Self:
        if self.k_start >= self.k_stop:
            raise ValueError(f"k_start ({self.k_start}) must be below k_stop ({self.k_stop})")
        span = self.k_stop - self.k_start
        if span % self.k_step != 0:
            raise ValueError(
                f"k_step ({self.k_step}) must divide k_stop - k_start ({span}), "
                "so that the inclusive sweep ends exactly at k_stop"
            )
        return self

    def k_values(self) -> tuple[int, ...]:
        """Return every K of the sweep, from ``k_start`` to ``k_stop`` inclusive."""
        return tuple(range(self.k_start, self.k_stop + 1, self.k_step))


class HdbscanConfig(_FrozenModel):
    """Grid of ``min_cluster_size`` values; ``min_samples`` keeps the HDBSCAN default."""

    min_cluster_sizes: Annotated[tuple[Annotated[StrictInt, Field(ge=2)], ...], Field(min_length=1)]

    @field_validator("min_cluster_sizes")
    @classmethod
    def _require_unique_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _require_unique(value, "min_cluster_sizes")
        return value


class RedundancyConfig(_FrozenModel):
    """Calibration of the direct-redundancy threshold."""

    cv_folds: Annotated[StrictInt, Field(ge=2)]


class RecommenderConfig(_FrozenModel):
    """Antecedent ranking (top-k with MMR) and alternative research lines."""

    top_k: PositiveInt
    mmr_lambda: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    neighbor_topics: NonNegativeInt
    nearby_gaps: NonNegativeInt


class GapsConfig(_FrozenModel):
    """Parameters of the gap rules and of their cross-seed stability filter."""

    temporal_split_year: StrictInt
    """First year of the later period compared by the temporal-shift rule."""
    temporal_min_count: PositiveInt
    bridge_max_docs: NonNegativeInt
    """Most bridge documents a pair of nearby topics may share and still count as a gap."""
    nearest_topics: PositiveInt
    stability_min_runs: PositiveInt
    """Seeds in which a gap must reappear to be kept."""


class AppConfig(_FrozenModel):
    """Complete, validated configuration of one analysis run."""

    paths: PathsConfig
    repository: RepositoryConfig
    snapshot: SnapshotConfig
    ocr: OcrConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    umap: UmapConfig
    kmeans: KMeansConfig
    hdbscan: HdbscanConfig
    seeds: Annotated[tuple[Seed, ...], Field(min_length=SEED_COUNT, max_length=SEED_COUNT)]
    redundancy: RedundancyConfig
    recommender: RecommenderConfig
    gaps: GapsConfig

    @field_validator("seeds")
    @classmethod
    def _require_unique_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _require_unique(value, "seeds")
        return value

    @model_validator(mode="after")
    def _check_cross_section_rules(self) -> Self:
        if self.gaps.stability_min_runs > len(self.seeds):
            raise ValueError(
                f"gaps.stability_min_runs ({self.gaps.stability_min_runs}) exceeds "
                f"the number of seeds ({len(self.seeds)})"
            )
        split_year = self.gaps.temporal_split_year
        if not self.snapshot.year_start < split_year <= self.snapshot.year_end:
            raise ValueError(
                f"gaps.temporal_split_year ({split_year}) must be after snapshot.year_start "
                f"({self.snapshot.year_start}) and not after snapshot.year_end "
                f"({self.snapshot.year_end})"
            )
        return self


_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"
_YAML_FLOAT_TAG = "tag:yaml.org,2002:float"

_EXPONENT_FLOAT = re.compile(r"^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+$")
"""YAML 1.2 core-schema float written with an exponent, such as ``1e-3`` or ``1E3``."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects a key written twice in one mapping, at any depth.

    ``yaml.safe_load`` silently keeps the last duplicate. The check also covers the
    sources of a ``<<`` merge, while keys brought in through a merge may still be
    overridden, as YAML intends. Plain scientific notation such as ``1e-3`` loads as a
    float, as in YAML 1.2, instead of as a string.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[yaml.Node] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # Every mapping is flattened before it is constructed, and so is every merge
        # source, even an inline one that is never constructed on its own. Flattening
        # folds the merged keys into node.value, so take the written keys before it.
        if node in self._checked_mappings:
            super().flatten_mapping(node)
            return
        self._checked_mappings.add(node)
        written_keys = [key_node for key_node, _ in node.value if key_node.tag != _YAML_MERGE_TAG]
        super().flatten_mapping(node)
        self._reject_repeated_keys(node, written_keys)

    def _reject_repeated_keys(self, node: yaml.MappingNode, key_nodes: list[yaml.Node]) -> None:
        seen: set[Any] = set()
        for key_node in key_nodes:
            key = self.construct_object(key_node)
            if not isinstance(key, Hashable):
                continue  # PyYAML refuses unhashable keys itself when it builds the mapping.
            if key in seen:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key '{key}'",
                    key_node.start_mark,
                )
            seen.add(key)


# PyYAML follows YAML 1.1: a float needs a dot, and an exponent needs a sign, so 1e-3 and
# 7.0e1 load as strings. On a subclass, add_implicit_resolver first copies the inherited
# table, so yaml.SafeLoader keeps its rules. PyYAML's own resolvers run first, so this one
# only claims the scalars that YAML 1.1 leaves as strings.
_UniqueKeyLoader.add_implicit_resolver(_YAML_FLOAT_TAG, _EXPONENT_FLOAT, list("-+.0123456789"))


def load_config(path: Path) -> AppConfig:
    """Read a YAML configuration file and return it validated.

    Raises:
        yaml.YAMLError: if the file is not valid YAML or a mapping repeats a key.
        pydantic.ValidationError: if the content breaks the configuration contract.
    """
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    return AppConfig.model_validate(raw)
