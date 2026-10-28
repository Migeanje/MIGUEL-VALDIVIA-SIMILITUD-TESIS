"""Typed, immutable run configuration loaded from YAML."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Hashable, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal, Self
from uuid import UUID

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictBool,
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
SnapshotIdText = Annotated[str, Field(pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$")]
"""A snapshot id as the corpus writes it: one plain directory name."""


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


def _resolve_inside(root: Path, location: Path, label: str) -> Path:
    """Resolve ``location`` under the resolved ``root`` and refuse any result outside it;
    ``label`` names the setting in the error."""
    resolved = (root / location).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} resolves to '{resolved}', outside the project root '{root}'")
    return resolved


def _resolve_inside_data_dir(
    project_root: Path, data_dir: Path, location: Path, label: str, content: str
) -> Path:
    """Resolve ``location`` under the resolved ``project_root`` and refuse any result outside
    the resolved ``data_dir``, which git ignores; ``label`` names the setting and ``content``
    what its folder holds, in the error."""
    root = Path(project_root).resolve()
    resolved_data_dir = _resolve_inside(root, data_dir, "paths.data_dir")
    resolved = _resolve_inside(root, location, label)
    if not resolved.is_relative_to(resolved_data_dir):
        raise ValueError(
            f"{label} resolves to '{resolved}', outside paths.data_dir "
            f"'{resolved_data_dir}', which git ignores; {content}, so no link under "
            "paths.data_dir may lead it elsewhere"
        )
    return resolved


def _require_relative_path(value: Path) -> Path:
    """Refuse an absolute or rooted path and ``..`` segments, judging the text alone.

    Nothing touches the file system here, so a symbolic link can still lead outside the
    project root; resolving the path against the root enforces containment.
    """
    raw_path = str(value)
    if _is_anchored(raw_path):
        raise ValueError(f"must be relative to the project root, got '{value}'")
    if _has_parent_segment(raw_path):
        raise ValueError(f"must not contain '..' segments, got '{value}'")
    return value


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

        :meth:`resolve_against` enforces containment once the root is known.
        """
        return _require_relative_path(value)

    def resolve_against(self, project_root: Path) -> ResolvedPaths:
        """Join every location onto ``project_root`` and return absolute paths.

        Raises:
            ValueError: if a location resolves outside the project root, for example
                through a symbolic link that points elsewhere.
        """
        root = Path(project_root).resolve()
        return ResolvedPaths(
            data_dir=_resolve_inside(root, self.data_dir, "paths.data_dir"),
            results_dir=_resolve_inside(root, self.results_dir, "paths.results_dir"),
            tessdata_dir=_resolve_inside(root, self.tessdata_dir, "paths.tessdata_dir"),
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


class HeaderFooterConfig(_FrozenModel):
    """Running header and footer removal: lines at the top or the bottom of the pages of one
    document that repeat on many of its pages."""

    edge_lines: Annotated[StrictInt, Field(ge=1, le=10)]
    """Non-empty lines inspected at the top, and again at the bottom, of each page."""
    min_share: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """Smallest share of a document's pages with text on which a line must repeat."""
    min_pages: Annotated[StrictInt, Field(ge=2)]
    """Fewest pages with text that a document needs before repeated lines are looked for."""


class ExtractionConfig(_FrozenModel):
    """PDF text extraction: low-text pages, the OCR window (D25), and header/footer removal."""

    min_text_chars: PositiveInt
    """A page whose text layer has fewer characters, stripped of surrounding whitespace, is
    low-text."""
    ocr_window_pages: NonNegativeInt
    """Only the low-text pages among the first this many pages of a document are OCR'd;
    0 turns OCR off."""
    header_footer: HeaderFooterConfig


_WORKBOOK_DIR_SETTING = "objectives.verification_sample.workbook_dir"


class VerificationSampleConfig(_FrozenModel):
    """Manual verification sample of the objectives locator (O05): how many theses per
    program, the seed that draws them, and where the verification workbook goes."""

    per_program: PositiveInt
    seed: Seed
    workbook_dir: Path
    """Directory, relative to the project root, that holds one directory per snapshot with
    the workbook. The workbook holds thesis text, so it must lie inside ``paths.data_dir``."""

    @field_validator("workbook_dir")
    @classmethod
    def _require_relative_without_parent_segments(cls, value: Path) -> Path:
        return _require_relative_path(value)

    def resolve_workbook_dir(self, project_root: Path, data_dir: Path) -> Path:
        """Join ``workbook_dir`` onto ``project_root`` and return the absolute path.

        ``data_dir`` is ``paths.data_dir``, relative to the project root. The configuration
        only checks the text of both paths, so a symbolic link or a junction under
        ``data_dir`` could still lead the workbook, which holds thesis text, into a folder
        that git tracks; the resolved workbook directory must therefore lie inside the
        resolved ``data_dir``.

        Raises:
            ValueError: if it resolves outside the project root or outside ``data_dir``, for
                example through a symbolic link that points elsewhere.
        """
        return _resolve_inside_data_dir(
            project_root,
            data_dir,
            self.workbook_dir,
            _WORKBOOK_DIR_SETTING,
            "the workbook holds thesis text",
        )


class ObjectivesConfig(_FrozenModel):
    """Objectives locator (T10): bounds on the extracted texts, and the verification sample."""

    general_max_chars: PositiveInt
    """The general objective is cut at this many characters, and flagged as too long."""
    specific_max_chars: PositiveInt
    """The specific objectives are cut at this many characters, and flagged as too long."""
    min_chars: PositiveInt
    """A general objective shorter than this is flagged as too short."""
    verification_sample: VerificationSampleConfig

    @model_validator(mode="after")
    def _require_min_below_the_cap(self) -> Self:
        if self.min_chars >= self.general_max_chars:
            raise ValueError(
                f"min_chars ({self.min_chars}) must be below general_max_chars "
                f"({self.general_max_chars})"
            )
        return self


class WrongPdfConfig(_FrozenModel):
    """A thesis whose PDF on disk is a copy of another item's PDF, as found by hand (T05)."""

    item_handle: NonEmptyStr
    pdf_of_handle: NonEmptyStr
    """Item whose PDF the file really is."""

    @model_validator(mode="after")
    def _require_another_item(self) -> Self:
        if self.item_handle == self.pdf_of_handle:
            raise ValueError(f"wrong PDF of '{self.item_handle}' must name another item")
        return self


class FichasCountsConfig(_FrozenModel):
    """Counts that the fichas of one snapshot must reach, declared after inspecting it (T12)."""

    snapshot_id: SnapshotIdText
    records: NonNegativeInt
    """Every harvested record: one ficha each."""
    included: NonNegativeInt
    duplicates: NonNegativeInt
    """Theses excluded as a D24 duplicate of another record."""
    non_thesis: NonNegativeInt
    """Records excluded because their ``renati.type`` is not a thesis (D01)."""
    objectives_extracted: NonNegativeInt
    """Included theses with objectives found in their own PDF."""
    metadata_only: NonNegativeInt
    """Included theses represented by their title and abstract only."""

    @model_validator(mode="after")
    def _require_counts_that_add_up(self) -> Self:
        if self.records != self.included + self.duplicates + self.non_thesis:
            raise ValueError(
                f"records ({self.records}) must equal included + duplicates + non_thesis "
                f"({self.included + self.duplicates + self.non_thesis})"
            )
        if self.included != self.objectives_extracted + self.metadata_only:
            raise ValueError(
                f"included ({self.included}) must equal objectives_extracted + metadata_only "
                f"({self.objectives_extracted + self.metadata_only})"
            )
        return self


class FichasConfig(_FrozenModel):
    """Fichas dataset (T12): known wrong PDFs, and the counts that its build must reach."""

    wrong_pdfs: tuple[WrongPdfConfig, ...] = ()
    expected: FichasCountsConfig

    @field_validator("wrong_pdfs")
    @classmethod
    def _require_one_declaration_per_item(
        cls, value: tuple[WrongPdfConfig, ...]
    ) -> tuple[WrongPdfConfig, ...]:
        _require_unique([wrong.item_handle for wrong in value], "wrong-PDF item handles")
        return value


class ChunkCountsConfig(_FrozenModel):
    """Counts that the chunk table of one snapshot must reach, measured on it first (T13, D27)."""

    snapshot_id: SnapshotIdText
    documents: NonNegativeInt
    """Included theses: each has a title, so each has chunks."""
    chunks: NonNegativeInt
    title_chunks: NonNegativeInt
    body_chunks: NonNegativeInt
    """Chunks of the abstracts and the objectives."""

    @model_validator(mode="after")
    def _require_counts_that_add_up(self) -> Self:
        if self.chunks != self.title_chunks + self.body_chunks:
            raise ValueError(
                f"chunks ({self.chunks}) must equal title_chunks + body_chunks "
                f"({self.title_chunks + self.body_chunks})"
            )
        if self.title_chunks < self.documents:
            raise ValueError(
                f"title_chunks ({self.title_chunks}) must be at least documents "
                f"({self.documents}), since every document has a title"
            )
        return self


class ChunkingConfig(_FrozenModel):
    """Sentence-aligned chunk budget, special tokens included, under both tokenizers, and the
    counts that the chunk build must reach."""

    max_tokens: Annotated[StrictInt, Field(ge=16, le=MODEL_MAX_SEQ_LENGTH)]
    expected: ChunkCountsConfig


DocumentVariant = Literal["section_mean", "plain_mean"]
"""Document vector of a thesis (T14): ``section_mean``, the equal-weight mean of its section
vectors (D09), or ``plain_mean``, the mean of all its chunk vectors (the sensitivity variant)."""


class EmbeddingCountsConfig(_FrozenModel):
    """Counts that the embeddings of one snapshot must reach, per model (T14)."""

    snapshot_id: SnapshotIdText
    documents: NonNegativeInt
    """Included theses: one document vector each."""
    chunks: NonNegativeInt
    """Chunk vectors, one per row of the chunk table."""

    @model_validator(mode="after")
    def _require_a_chunk_per_document(self) -> Self:
        if self.chunks < self.documents:
            raise ValueError(
                f"chunks ({self.chunks}) must be at least documents ({self.documents}), since "
                "every document has its title chunk"
            )
        return self


class EmbeddingConfig(_FrozenModel):
    """Sentence-transformers models compared in the experiment, how they encode the chunks, and
    which document vector later steps read (T14)."""

    models: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    batch_size: Annotated[StrictInt, Field(ge=1, le=4096)]
    """Chunks encoded together."""
    device: Annotated[str, Field(pattern=r"^(cpu|cuda(:[0-9]+)?)$")]
    threads: Annotated[StrictInt, Field(ge=1, le=256)]
    """CPU threads of torch, pinned so that runs on one machine repeat."""
    document_vector: DocumentVariant
    sensitivity_cosine_threshold: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    """A thesis whose two document vectors have a lower cosine is counted in the summary."""
    expected: EmbeddingCountsConfig

    @field_validator("models")
    @classmethod
    def _require_unique_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(value, "embedding models")
        return value


TfidfVariant = Literal["section_mean", "concatenated"]
"""Document vector of the TF-IDF baseline (T15, D28): ``section_mean``, the equal-weight mean
of the L2-normalized section vectors (as D09), or ``concatenated``, one vector over all the
sections (the sensitivity variant)."""


class TfidfCountsConfig(_FrozenModel):
    """Counts that the TF-IDF baseline of one snapshot must reach, measured on it first (T15)."""

    snapshot_id: SnapshotIdText
    documents: NonNegativeInt
    """Included theses: one row of each matrix."""
    sections: NonNegativeInt
    """Sections with at least one token once cleaned."""
    terms: PositiveInt
    """Columns of each matrix: the vocabulary."""
    nonzeros: NonNegativeInt
    """Stored values of each matrix; the two variants have the same."""

    @model_validator(mode="after")
    def _require_counts_that_can_hold(self) -> Self:
        if self.sections < self.documents:
            raise ValueError(
                f"sections ({self.sections}) must be at least documents ({self.documents}), "
                "since every document has its title"
            )
        if self.nonzeros < self.documents:
            raise ValueError(
                f"nonzeros ({self.nonzeros}) must be at least documents ({self.documents}), "
                "since a document without a nonzero has no cosine"
            )
        return self


class TfidfConfig(_FrozenModel):
    """Vectorizer of the TF-IDF baseline, the document vector that later steps read, and the
    counts that its build must reach (T15, D28). The norm (L2) and the smooth idf stay in code,
    since cosine needs the one and nothing in the plan varies the other."""

    max_ngram: Annotated[StrictInt, Field(ge=1, le=2)]
    """1 for unigrams; 2 adds the bigrams inside each section."""
    min_df: PositiveInt
    """Fewest theses that a term must occur in."""
    max_df: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """Largest share of the theses that a term may occur in; 1.0 turns the cap off."""
    sublinear_tf: StrictBool
    """Whether a term's weight takes 1 + log of its count instead of the count."""
    document_vector: TfidfVariant
    sensitivity_cosine_threshold: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    """A thesis whose two document vectors have a lower cosine is counted in the summary."""
    expected: TfidfCountsConfig


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
    """Grid of ``min_cluster_size`` values, unique and ascending; ``min_samples`` keeps the
    HDBSCAN default."""

    min_cluster_sizes: Annotated[tuple[Annotated[StrictInt, Field(ge=2)], ...], Field(min_length=1)]

    @field_validator("min_cluster_sizes")
    @classmethod
    def _require_unique_ascending_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _require_unique(value, "min_cluster_sizes")
        if list(value) != sorted(value):
            raise ValueError(f"min_cluster_sizes must be ascending, got {list(value)}")
        return value


NpmiUnit = Literal["document", "window"]
"""NPMI co-occurrence unit, as :func:`thematic_redundancy.evaluation.coherence.npmi_coherence`
names it."""

AriNoiseHandling = Literal["own_label", "exclude_either"]
"""How the pairwise ARI treats HDBSCAN noise, as
:func:`thematic_redundancy.evaluation.stability.seed_stability` names it."""


class ExperimentKMeansConfig(_FrozenModel):
    """K-means arguments of the experiment (D30); the K sweep is :class:`KMeansConfig`, and each
    run's ``random_state`` is its seed. The other arguments keep scikit-learn 1.9.1's defaults."""

    n_init: PositiveInt
    """Initializations per run, explicit instead of scikit-learn's ``"auto"`` (one, under
    k-means++), so that the spread across seeds reflects UMAP and not one initialization."""
    init: Literal["k-means++", "random"]


class ExperimentHdbscanConfig(_FrozenModel):
    """HDBSCAN arguments of the experiment (D30); the ``min_cluster_size`` grid is
    :class:`HdbscanConfig`. HDBSCAN is deterministic, so it takes no seed."""

    min_samples: PositiveInt | None
    """``None`` keeps hdbscan's default, which is the run's ``min_cluster_size``."""
    metric: Literal["euclidean"]
    """On the UMAP output, as BERTopic's default model."""
    cluster_selection_method: Literal["eom", "leaf"]


class BaselineConfig(_FrozenModel):
    """The lexical baseline of plan §3.7: K-means on the sparse TF-IDF matrix (D28 C4), with no
    UMAP. It is never a candidate; it enters acceptance criterion 1 only."""

    representation: Literal["tfidf"]
    algorithm: Literal["kmeans"]
    space: Literal["sparse_l2"]
    """The L2-normalized ``tfidf.document_vector`` matrix itself."""
    matched_k: StrictBool
    """Whether the baseline also runs at the winner's number of topics, so that the NPMI
    comparison holds the number of topics constant."""


class CtfidfConfig(_FrozenModel):
    """How the topic terms are computed, the same way for every configuration (D30 E4)."""

    vocabulary: Literal["tfidf_terms"]
    """T15's vocabulary, fixed for every configuration, with no class-level ``min_df`` or
    ``max_df``."""
    counts: Literal["raw_within_section"]
    """Raw counts of each term per class; bigrams only within one section (D28 C3)."""
    weighting: Literal["bertopic_0.17.4_defaults"]
    """``ClassTfidfTransformer()`` as BERTopic 0.17.4 ships it: no BM25, no frequent-word
    reduction."""
    include_noise_class: StrictBool
    """Whether the noise label is a class of the c-TF-IDF fit; its topic is never scored."""
    tie_order: Literal["term_ascending"]
    """Terms by score descending, then by term ascending (code-point order)."""


def _check_npmi_setting(unit: NpmiUnit, window_size: int | None) -> None:
    if unit == "document" and window_size is not None:
        raise ValueError(f"window_size must be null for document units, not {window_size}")
    if unit == "window" and window_size is None:
        raise ValueError("window_size is required for window units")


class NpmiSettingConfig(_FrozenModel):
    """One NPMI co-occurrence setting: the unit and, for windows, the window size."""

    unit: NpmiUnit
    window_size: Annotated[StrictInt, Field(ge=2)] | None

    @model_validator(mode="after")
    def _require_a_window_size_for_windows_only(self) -> Self:
        _check_npmi_setting(self.unit, self.window_size)
        return self


class NpmiConfig(_FrozenModel):
    """NPMI coherence of the topics (D12, D30 E4): the primary setting, which the selection rule
    uses, and a sensitivity setting, which is only reported."""

    top_n: Annotated[StrictInt, Field(ge=2)]
    """Each topic's first terms that are scored."""
    unit: NpmiUnit
    window_size: Annotated[StrictInt, Field(ge=2)] | None
    reference: Literal["thesis_sections"]
    """All the included theses, each one document whose segments are its full-clean sections."""
    sensitivity: NpmiSettingConfig
    absent_terms: Literal["error"]
    """A term in no unit is a pipeline mismatch, so the run stops."""

    @model_validator(mode="after")
    def _require_two_distinct_settings(self) -> Self:
        _check_npmi_setting(self.unit, self.window_size)
        if (self.sensitivity.unit, self.sensitivity.window_size) == (self.unit, self.window_size):
            raise ValueError("npmi.sensitivity must differ from the primary setting")
        return self


class AriConfig(_FrozenModel):
    """Noise handling of the pairwise ARI across seeds: one for the stability gate and
    acceptance criterion 4, and another reported as sensitivity."""

    noise_handling: AriNoiseHandling
    sensitivity: AriNoiseHandling

    @model_validator(mode="after")
    def _require_two_distinct_handlings(self) -> Self:
        if self.sensitivity == self.noise_handling:
            raise ValueError("ari.sensitivity must differ from ari.noise_handling")
        return self


class GatesConfig(_FrozenModel):
    """Eligibility gates of a configuration, over its seed runs (D30 E3)."""

    min_mean_ari: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """Lowest mean pairwise ARI over the seed pairs; a gate at or below 0 would admit
    chance-level agreement."""
    max_mean_noise_fraction: Annotated[StrictFloat, Field(ge=0.0, lt=1.0)]
    """Highest mean share of theses left as noise; also the noise cap of the
    ``min_cluster_size`` choice. A cap of 1 would gate nothing."""


class F1ComparisonConfig(_FrozenModel):
    """Paired bootstrap over the labeled pairs that compares two F1 values (D30 E3)."""

    bootstrap_resamples: PositiveInt
    bootstrap_seed: Seed
    interval: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]
    """Coverage of the percentile interval of the difference; a difference whose interval
    excludes 0 decides."""


class FallbacksConfig(_FrozenModel):
    """What an exact tie at the end of each stage of the selection rule picks."""

    algorithm: Literal["kmeans", "hdbscan"]
    model: NonEmptyStr
    """One of ``embedding.models``. The loader does not require it, since a configuration may
    replace the models with others (the unit tests do), so the experiment checks it before any
    run."""


class SelectionConfig(_FrozenModel):
    """The winner-selection rule (O03, D30 E3): gates, then the algorithm within each model's
    space, then the model."""

    gates: GatesConfig
    seeded_tie_tolerance: Literal["max_seed_sd"]
    """A beats B on a seeded metric when the difference of their seed means exceeds the larger
    of their seed SDs; otherwise the comparison is a tie."""
    f1: F1ComparisonConfig
    fallbacks: FallbacksConfig
    representative_run: Literal["max_mean_ari_to_others"]
    """The winner's seed run that the map uses: the one with the highest mean ARI to the
    others; a tie goes to the seed listed first."""


class AcceptanceConfig(_FrozenModel):
    """Thresholds of the feature's acceptance criteria that D30 declares."""

    stability_max_cv: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """Highest coefficient of variation (SD / |mean|) over the seeds, for silhouette and for NPMI,
    in criterion 4."""


class ExperimentConfig(_FrozenModel):
    """The experiment declaration (T18, D30), fixed before any run. The values that it reuses
    from other blocks (UMAP, the K sweep, the ``min_cluster_size`` grid, the seeds, the embedding
    models and the document vectors) are not repeated here;
    :meth:`AppConfig.experiment_declaration` gathers them all."""

    clustering_space: Literal["umap"]
    """The semantic runs cluster the UMAP output: one fit per (model, seed), with the seed as
    ``random_state``, shared by K-means and HDBSCAN."""
    kmeans: ExperimentKMeansConfig
    hdbscan: ExperimentHdbscanConfig
    baseline: BaselineConfig
    k_choice: Literal["max_seed_mean_silhouette"]
    """K is the sweep value with the highest mean silhouette over the seeds; an exact tie goes
    to the smaller K, and the sweep is never extended after results."""
    mcs_choice: Literal["max_seed_mean_silhouette_under_noise_cap"]
    """Among the sizes under the noise cap whose metrics are defined in every seed, the highest
    mean silhouette; ties go to the lower mean noise share, then the smaller size."""
    ctfidf: CtfidfConfig
    npmi: NpmiConfig
    ari: AriConfig
    selection: SelectionConfig
    acceptance: AcceptanceConfig


class ProjectionConfig(_FrozenModel):
    """The 2-D projection of the original document vectors for the visualization map (T16).

    No metric, choice or selection reads it, so it lies outside the experiment declaration.
    ``umap`` fits a UMAP of its own with two components and the run's seed; ``pca`` (thesis
    plan section 3.6.3) uses scikit-learn's PCA with the run's seed and ignores the other keys.
    """

    method: Literal["umap", "pca"]
    n_neighbors: Annotated[StrictInt, Field(ge=2)]
    min_dist: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    """Bounded by UMAP's default ``spread`` of 1.0."""
    metric: NonEmptyStr


class TopicMapConfig(_FrozenModel):
    """Settings of the topic map that the experiment declaration does not hold (T16)."""

    projection: ProjectionConfig


DocCode = Annotated[str, Field(pattern=r"^DOC-[0-9a-f]{8}$")]
"""A thesis's short report code, ``DOC-`` and 8 lowercase hex digits."""


F1Tie = Literal["highest_tau", "lowest_tau"]
"""Which threshold a tie of the highest F1 picks, as
:func:`thematic_redundancy.redundancy.threshold.calibrate_threshold` names the rules."""

FinalTau = Literal["all_pairs", "fold_mean"]
"""How the final threshold follows the folds, as
:func:`thematic_redundancy.redundancy.threshold.calibrate_threshold` names the rules."""


class RedundancyConfig(_FrozenModel):
    """Calibration of the direct-redundancy threshold, and the pairs known to be duplicates."""

    cv_folds: Annotated[StrictInt, Field(ge=2)]
    cv_seed: Seed
    """The ``random_state`` of the fold split."""
    cv_stratified: StrictBool
    """Stratified folds, which keep the share of redundant pairs in each fold."""
    f1_tie: F1Tie
    final_tau: FinalTau
    known_duplicates: tuple[tuple[DocCode, DocCode], ...]
    """Pairs of included theses known to be one thesis deposited twice (O09). They stay in the
    labeled pairs and the search, and the results flag them; the order within a pair does not
    matter."""

    @field_validator("known_duplicates")
    @classmethod
    def _distinct_pairs(cls, pairs: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
        for first, second in pairs:
            if first == second:
                raise ValueError(f"a known duplicate must join two different theses: {first}")
        _require_unique([frozenset(pair) for pair in pairs], "known_duplicates")
        return pairs


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
    npmi_floor_quantile: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    """Rule (a): the quantile of the per-topic NPMI of every seed run of the winner that a small
    topic must reach (D32)."""
    min_affinity: PositiveInt
    """Rule (c): the lowest career-affinity score (U4) at which two programs are affine."""
    present_min_count: PositiveInt
    """Rule (c): the fewest theses of the source program for a topic to be present in it."""
    present_min_share: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """Rule (c): the smallest share of the source program's theses for a topic to be present."""
    rare_max_share: Annotated[StrictFloat, Field(ge=0.0, lt=1.0)]
    """Rule (c): the largest share of the target program's theses for a topic to be absent from
    it; below ``present_min_share``."""
    min_decline: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """Rule (d): the smallest fall in a topic's share from the earlier to the later window."""
    stability_min_overlap: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """The Jaccard overlap at or above which a gap of another run matches (D32)."""

    @model_validator(mode="after")
    def _require_rare_below_present(self) -> Self:
        if self.rare_max_share >= self.present_min_share:
            raise ValueError(
                f"gaps.rare_max_share ({self.rare_max_share}) must be below "
                f"gaps.present_min_share ({self.present_min_share})"
            )
        return self


_RECORDS_DIR_SETTING = "labeling.records_dir"
LABELING_POOLS = 4
"""The labeled set's pools (D34 (5)): SBERT-high, TF-IDF-high, mid-range and random."""


class LabelingConfig(_FrozenModel):
    """The labeled set's sampler (T20, D34): its seed, the pools' sizes, the share of the pairs
    that forms a representation's top, the mid-range band of mean percentile ranks, the strata's
    floor, the pilot, and where the drawn pairs go."""

    seed: Seed
    sbert_high_per_model: PositiveInt
    """Pairs drawn from each embedding model's top (SBERT-high)."""
    tfidf_high: PositiveInt
    """Pairs drawn from the TF-IDF baseline's top (TF-IDF-high)."""
    mid_range: PositiveInt
    random: PositiveInt
    """Pairs drawn uniformly from the whole universe, unstratified."""
    top_share: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]
    """A representation's top is this share of the pairs, rounded up, by rank."""
    mid_low: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]
    """The mid-range holds the pairs whose mean percentile rank is at least this..."""
    mid_high: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    """...and below this."""
    stratum_floor: NonNegativeInt
    """Fewest pairs a high or mid-range pool draws from each stratum that has candidates."""
    pilot_size: PositiveInt
    pilot_floor: PositiveInt
    """Fewest pilot pairs drawn from each pool."""
    records_dir: Path
    """Directory, relative to the project root, that holds one directory per snapshot with the
    drawn pairs. They name theses, so it must lie inside ``paths.data_dir``."""

    @field_validator("records_dir")
    @classmethod
    def _require_relative_without_parent_segments(cls, value: Path) -> Path:
        return _require_relative_path(value)

    @model_validator(mode="after")
    def _require_a_band_and_a_pilot_that_hold(self) -> Self:
        if self.mid_low >= self.mid_high:
            raise ValueError(
                f"labeling.mid_low ({self.mid_low}) must be below labeling.mid_high "
                f"({self.mid_high})"
            )
        if self.pilot_floor * LABELING_POOLS > self.pilot_size:
            raise ValueError(
                f"labeling.pilot_size ({self.pilot_size}) cannot give each of the "
                f"{LABELING_POOLS} pools labeling.pilot_floor ({self.pilot_floor}) pairs"
            )
        return self

    def resolve_records_dir(self, project_root: Path, data_dir: Path) -> Path:
        """Join ``records_dir`` onto ``project_root`` and return the absolute path, as
        :meth:`VerificationSampleConfig.resolve_workbook_dir` does.

        Raises:
            ValueError: if it resolves outside the project root or outside ``data_dir``.
        """
        return _resolve_inside_data_dir(
            project_root,
            data_dir,
            self.records_dir,
            _RECORDS_DIR_SETTING,
            "the drawn pairs name theses",
        )


class AppConfig(_FrozenModel):
    """Complete, validated configuration of one analysis run."""

    paths: PathsConfig
    repository: RepositoryConfig
    snapshot: SnapshotConfig
    ocr: OcrConfig
    extraction: ExtractionConfig
    objectives: ObjectivesConfig
    fichas: FichasConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    tfidf: TfidfConfig
    umap: UmapConfig
    kmeans: KMeansConfig
    hdbscan: HdbscanConfig
    seeds: Annotated[tuple[Seed, ...], Field(min_length=SEED_COUNT, max_length=SEED_COUNT)]
    experiment: ExperimentConfig
    topic_map: TopicMapConfig
    redundancy: RedundancyConfig
    recommender: RecommenderConfig
    gaps: GapsConfig
    labeling: LabelingConfig

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
        workbook_dir = self.objectives.verification_sample.workbook_dir
        if not workbook_dir.is_relative_to(self.paths.data_dir):
            raise ValueError(
                f"objectives.verification_sample.workbook_dir ('{workbook_dir}') must lie inside "
                f"paths.data_dir ('{self.paths.data_dir}'), which git ignores, because the "
                "workbook holds thesis text"
            )
        labeling = self.labeling
        if not labeling.records_dir.is_relative_to(self.paths.data_dir):
            raise ValueError(
                f"labeling.records_dir ('{labeling.records_dir}') must lie inside "
                f"paths.data_dir ('{self.paths.data_dir}'), which git ignores, because the "
                "drawn pairs name theses"
            )
        strata = len(self.snapshot.programs) + 1
        smallest = min(labeling.sbert_high_per_model, labeling.tfidf_high, labeling.mid_range)
        if labeling.stratum_floor * strata > smallest:
            raise ValueError(
                f"labeling.stratum_floor ({labeling.stratum_floor}) over the {strata} strata "
                f"(each program and the pairs between programs) exceeds the smallest stratified "
                f"pool ({smallest})"
            )
        return self

    def experiment_declaration(self) -> dict[str, Any]:
        """Return the experiment declaration (D30) as plain JSON data: the whole ``experiment``
        block, and the values it reuses from the other blocks, namely ``umap``, ``kmeans`` (the
        K sweep), ``hdbscan`` (the size grid), ``seeds``, ``embedding.models`` and the
        ``document_vector`` of ``embedding`` and ``tfidf``."""
        return {
            "embedding": self.embedding.model_dump(
                mode="json", include={"models", "document_vector"}
            ),
            "experiment": self.experiment.model_dump(mode="json"),
            "hdbscan": self.hdbscan.model_dump(mode="json"),
            "kmeans": self.kmeans.model_dump(mode="json"),
            "seeds": list(self.seeds),
            "tfidf": self.tfidf.model_dump(mode="json", include={"document_vector"}),
            "umap": self.umap.model_dump(mode="json"),
        }

    def experiment_declaration_sha256(self) -> str:
        """Return the SHA-256 of :meth:`experiment_declaration` as canonical JSON (keys sorted,
        no whitespace, UTF-8). A unit test pins it, and the experiment's records carry it."""
        text = json.dumps(
            self.experiment_declaration(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
