"""Typed, immutable run configuration loaded from YAML."""

from collections import Counter
from collections.abc import Hashable, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Self
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

MODEL_MAX_SEQ_LENGTH = 128
"""Verified ``max_seq_length`` of both multilingual SBERT models; no chunk may exceed it."""

SEED_COUNT = 5
"""Number of fixed seeds that every stochastic step is repeated with."""

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Seed = Annotated[int, Field(ge=0, lt=2**32)]


def _require_unique(values: Sequence[Hashable], label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} must be unique; duplicated: {duplicates}")


def _is_anchored(raw_path: str) -> bool:
    """Tell whether a path is absolute or rooted under either Windows or POSIX rules."""
    return bool(PureWindowsPath(raw_path).anchor or PurePosixPath(raw_path).anchor)


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
    def _require_relative(cls, value: Path) -> Path:
        if _is_anchored(str(value)):
            raise ValueError(f"must be relative to the project root, got '{value}'")
        return value

    def resolve_against(self, project_root: Path) -> ResolvedPaths:
        """Join every location onto ``project_root`` and return absolute paths."""
        root = Path(project_root).resolve()
        return ResolvedPaths(
            data_dir=(root / self.data_dir).resolve(),
            results_dir=(root / self.results_dir).resolve(),
            tessdata_dir=(root / self.tessdata_dir).resolve(),
        )


class RepositoryConfig(_FrozenModel):
    """Public DSpace repository and the faculty community that holds the programs."""

    base_url: HttpUrl
    faculty_community_uuid: UUID


class ProgramConfig(_FrozenModel):
    """Academic program: a stable short key plus its official collection name."""

    key: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    name: NonEmptyStr


class SnapshotConfig(_FrozenModel):
    """Filter that selects the theses of the frozen corpus snapshot."""

    year_start: int
    year_end: int
    thesis_type: NonEmptyStr
    programs: Annotated[tuple[ProgramConfig, ...], Field(min_length=1)]

    @field_validator("programs")
    @classmethod
    def _require_unique_program_keys(
        cls, value: tuple[ProgramConfig, ...]
    ) -> tuple[ProgramConfig, ...]:
        _require_unique([program.key for program in value], "program keys")
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
    dpi: Annotated[int, Field(ge=72, le=600)]


class ChunkingConfig(_FrozenModel):
    """Sentence-aligned chunk budget, special tokens included, under both tokenizers."""

    max_tokens: Annotated[int, Field(ge=16, le=MODEL_MAX_SEQ_LENGTH)]


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

    n_neighbors: Annotated[int, Field(ge=2)]
    n_components: PositiveInt
    min_dist: Annotated[float, Field(ge=0.0, le=1.0)]
    """Bounded by UMAP's default ``spread`` of 1.0."""
    metric: NonEmptyStr


class KMeansConfig(_FrozenModel):
    """K sweep for K-means: from ``k_start`` to ``k_stop`` (inclusive) every ``k_step``."""

    k_start: Annotated[int, Field(ge=2)]
    k_stop: int
    k_step: PositiveInt

    @model_validator(mode="after")
    def _require_increasing_sweep(self) -> Self:
        if self.k_start >= self.k_stop:
            raise ValueError(f"k_start ({self.k_start}) must be below k_stop ({self.k_stop})")
        return self


class HdbscanConfig(_FrozenModel):
    """Grid of ``min_cluster_size`` values; ``min_samples`` keeps the HDBSCAN default."""

    min_cluster_sizes: Annotated[tuple[Annotated[int, Field(ge=2)], ...], Field(min_length=1)]

    @field_validator("min_cluster_sizes")
    @classmethod
    def _require_unique_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _require_unique(value, "min_cluster_sizes")
        return value


class RedundancyConfig(_FrozenModel):
    """Calibration of the direct-redundancy threshold."""

    cv_folds: Annotated[int, Field(ge=2)]


class RecommenderConfig(_FrozenModel):
    """Antecedent ranking (top-k with MMR) and alternative research lines."""

    top_k: PositiveInt
    mmr_lambda: Annotated[float, Field(ge=0.0, le=1.0)]
    neighbor_topics: NonNegativeInt
    nearby_gaps: NonNegativeInt


class GapsConfig(_FrozenModel):
    """Parameters of the gap rules and of their cross-seed stability filter."""

    temporal_split_year: int
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


def load_config(path: Path) -> AppConfig:
    """Read a YAML configuration file and return it validated."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
