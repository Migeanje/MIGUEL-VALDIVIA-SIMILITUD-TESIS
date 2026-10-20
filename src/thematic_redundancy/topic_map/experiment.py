"""The experiment of T19: the D30 grid, its metrics and the selection, without file I/O.

:func:`run_experiment` runs the declaration of D30 (E1 to E6) on the loaded inputs:

- **Sweeps** (E1). For each model and seed, one UMAP reduction (:func:`~.reduction.fit_reduction`,
  or the cached one of an earlier run, so a rerun resumes per (model, seed)), shared through
  :func:`~.build.cluster` by every K of the sweep and every HDBSCAN size; the TF-IDF baseline is
  K-means on its sparse matrix (:func:`~.clustering.kmeans_labels`), with no UMAP (D28 C4).
  Not :func:`~.build.build_topic_maps`, which would compute the c-TF-IDF and the 2-D map of
  every sweep run, while E6 keeps them for the final runs.
- **Metrics** (E5), per run, on the original vectors: :func:`~thematic_redundancy.evaluation.
  clustering_metrics.internal_metrics`, noise left out and counted; for HDBSCAN, the silhouette
  with each noise thesis sent to its nearest centroid by cosine. ARI over the seeds of every
  configuration, under the primary and the sensitivity noise handling (E4).
- **Choices and selection** (E2, E3) through :mod:`.selection`: K* and min_cluster_size*, the
  five final configurations C1 to C4 and B0, their c-TF-IDF topics and NPMI at both settings
  (E4; an absent term stops the run), stage 1 per model, the provisional stage 2 (UD3), the
  matched-K baseline runs, criteria 1 (its NPMI part) and 4, and the representative run.
- **Map**: the 2-D coordinates of the representative run, with the configured projection, and
  the PCA coordinates of the thesis's inspection figure.
- **Repeatability** (E6): one UMAP fit and one K-means run done again, compared by SHA-256.

:func:`plan_experiment` counts the same work without doing it, for the dry run. The libraries
sit behind :class:`~.build.TopicMapPorts`, the reduction cache behind :class:`ReductionCache`.
Nothing here reads or writes a file or holds a thesis text: the tables carry doc codes, labels,
numbers and vocabulary lemmas only.
"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import sparse
from sklearn.preprocessing import normalize

from thematic_redundancy.evaluation.clustering_metrics import (
    NOISE_LABEL,
    InternalMetrics,
    internal_metrics,
)
from thematic_redundancy.evaluation.coherence import CoherenceResult, npmi_coherence
from thematic_redundancy.evaluation.stability import NoiseHandling, SeedStability, seed_stability
from thematic_redundancy.representation.build_embeddings import EMBEDDER_VERSION
from thematic_redundancy.representation.tfidf import TFIDF_VERSION
from thematic_redundancy.representation.vector_cache import model_slug
from thematic_redundancy.shared.config import AppConfig, NpmiUnit
from thematic_redundancy.topic_map.build import (
    TopicMapPorts,
    TopicMapSettings,
    cluster,
    topic_centroids,
)
from thematic_redundancy.topic_map.clustering import HdbscanRun, KMeansRun, kmeans_labels
from thematic_redundancy.topic_map.ctfidf import TopicSet, ctfidf_topics
from thematic_redundancy.topic_map.reduction import (
    Reduction,
    UmapSettings,
    dense_vectors,
    fit_reduction,
    project,
)
from thematic_redundancy.topic_map.selection import (
    BASELINE_REPRESENTATION,
    Algorithm,
    Candidate,
    Gates,
    SettingSummary,
    choose_k,
    choose_min_cluster_size,
    criterion_four,
    criterion_one_npmi,
    evaluate_gates,
    matched_k,
    representative_seed,
    stage_one,
    stage_one_sensitivities,
    stage_two_provisional,
    stage_two_sensitivities,
    summarize,
)

DECLARED_SHA256 = "b974c54c3378c8fe02f7983323f0d58fa7672e4ca7d1999df15f708d2b76850b"
"""The D30 declaration as decided on 2026-10-19 (pinned by
``tests/unit/test_experiment_declaration.py``); the runner refuses any other."""

RUNTIME_FIELDS = (
    "reduction_reused",
    "umap_seconds",
    "cluster_seconds",
    "metrics_seconds",
    "ctfidf_seconds",
)
"""The columns of :class:`RunRow` that a rerun on the same inputs may change."""

MatchedSource = Literal["baseline", "sweep", "new", "off"]
"""Where the matched-K baseline runs come from: B0's own K*, the sweep's runs at that K, five
new runs, or nowhere when ``experiment.baseline.matched_k`` is false."""


class ExperimentError(ValueError):
    """The declaration, the inputs or a result break a rule of D30, so the run stops."""


def check_declaration(config: AppConfig) -> str:
    """Return the declaration's SHA-256 once it is the pinned one and the fallback model is one
    of ``embedding.models`` (D30, T18).

    Raises:
        ExperimentError: otherwise; a changed declaration needs a new decision first.
    """
    digest = config.experiment_declaration_sha256()
    if digest != DECLARED_SHA256:
        raise ExperimentError(
            f"the experiment declaration has SHA-256 {digest}, not the D30 pin {DECLARED_SHA256}; "
            "a change needs a new decision in the plan before any run"
        )
    if config.experiment.selection.fallbacks.model not in config.embedding.models:
        raise ExperimentError("the fallback model is not one of embedding.models")
    return digest


@dataclass(frozen=True)
class NpmiSetting:
    unit: NpmiUnit
    window_size: int | None


@dataclass(frozen=True)
class ExperimentSettings:
    """Everything the experiment takes from the configuration (D30)."""

    topic_map: TopicMapSettings
    models: tuple[str, ...]
    """``embedding.models``: the inputs must hold these, in this order."""
    k_values: tuple[int, ...]
    min_cluster_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    npmi: NpmiSetting
    npmi_sensitivity: NpmiSetting
    ari: NoiseHandling
    ari_sensitivity: NoiseHandling
    gates: Gates
    fallback_algorithm: Algorithm
    fallback_model: str
    max_cv: float
    matched_k: bool

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        """Return the settings of ``config`` after :func:`check_declaration`."""
        check_declaration(config)
        experiment = config.experiment
        npmi = experiment.npmi
        selection = experiment.selection
        return cls(
            topic_map=TopicMapSettings.from_config(config),
            models=tuple(config.embedding.models),
            k_values=config.kmeans.k_values(),
            min_cluster_sizes=tuple(config.hdbscan.min_cluster_sizes),
            seeds=tuple(config.seeds),
            npmi=NpmiSetting(unit=npmi.unit, window_size=npmi.window_size),
            npmi_sensitivity=NpmiSetting(
                unit=npmi.sensitivity.unit, window_size=npmi.sensitivity.window_size
            ),
            ari=experiment.ari.noise_handling,
            ari_sensitivity=experiment.ari.sensitivity,
            gates=Gates(
                min_mean_ari=selection.gates.min_mean_ari,
                max_mean_noise_fraction=selection.gates.max_mean_noise_fraction,
            ),
            fallback_algorithm=selection.fallbacks.algorithm,
            fallback_model=selection.fallbacks.model,
            max_cv=experiment.acceptance.stability_max_cv,
            matched_k=experiment.baseline.matched_k,
        )


@dataclass(frozen=True)
class SemanticInput:
    """The document vectors of one embedding model, one row per thesis."""

    model_id: str
    revision: str
    vectors: np.ndarray


@dataclass(frozen=True)
class ExperimentInputs:
    """The loaded inputs, every row in the order of :attr:`doc_codes`."""

    snapshot_id: str
    doc_codes: tuple[str, ...]
    models: tuple[SemanticInput, ...]
    """In the order of ``embedding.models``: the first gives C1 and C2, the second C3 and C4."""
    tfidf: sparse.csr_matrix
    vocabulary: tuple[str, ...]
    documents: tuple[tuple[tuple[str, ...], ...], ...]
    """Each thesis's full-clean section tokens: the c-TF-IDF counts and the NPMI reference."""
    embedding_variant: str
    tfidf_variant: str
    input_sha256: dict[str, str] = field(default_factory=dict)
    """The SHA-256 of the input manifests, for the records."""


@dataclass(frozen=True)
class ReductionKey:
    model_id: str
    seed: int
    sha256: str
    """Of the canonical JSON of the model, its revision, its vectors, the UMAP settings and the
    seed."""


class ReductionCache(Protocol):
    """Port: the UMAP reductions of earlier runs, by key."""

    def get(self, key: ReductionKey) -> np.ndarray | None: ...

    def put(self, key: ReductionKey, embedding: np.ndarray) -> None: ...


class NoReductionCache:
    """A cache that keeps nothing."""

    def get(self, key: ReductionKey) -> np.ndarray | None:
        return None

    def put(self, key: ReductionKey, embedding: np.ndarray) -> None:
        pass


def reduction_key(model: SemanticInput, settings: UmapSettings, seed: int) -> ReductionKey:
    """Return the cache key of the reduction of ``model``'s vectors for ``seed``."""
    vectors = np.ascontiguousarray(model.vectors)
    payload = {
        "model_id": model.model_id,
        "revision": model.revision,
        "vectors": {
            "dtype": vectors.dtype.str,
            "shape": list(vectors.shape),
            "sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        },
        "umap": asdict(settings),
        "seed": seed,
    }
    return ReductionKey(model_id=model.model_id, seed=seed, sha256=_json_sha256(payload))


def _json_sha256(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class RunCounts:
    """The work of one run of the experiment, as the dry run forecasts it."""

    umap_fits: int
    pca_fits: int
    clusterings: int
    ctfidf_fits: int
    internal_metrics: int
    reassigned_silhouettes: int
    npmi_evaluations: int
    ari_evaluations: int


@dataclass(frozen=True)
class ExperimentPlan:
    """What a run would do, without doing it."""

    documents: int
    models: int
    seeds: int
    k_values: int
    min_cluster_sizes: int
    reductions: int
    reductions_cached: int
    projection_method: str
    matched_k: bool

    def counts(self, matched: MatchedSource) -> RunCounts:
        """Return the counts once the matched-K source is known; ``baseline`` adds nothing,
        ``sweep`` the c-TF-IDF of five sweep runs, ``new`` five new runs as well."""
        new = matched == "new"
        extra_runs = self.seeds if new else 0
        grid = self.k_values + self.min_cluster_sizes
        sweeps = self.models * self.seeds * grid + self.seeds * self.k_values
        finals = (2 * self.models + 1) * self.seeds + (
            self.seeds if matched in ("sweep", "new") else 0
        )
        return RunCounts(
            umap_fits=self.reductions
            - self.reductions_cached
            + 1
            + (self.projection_method == "umap"),
            pca_fits=1 + (self.projection_method == "pca"),
            clusterings=sweeps + 1 + extra_runs,
            ctfidf_fits=finals,
            internal_metrics=sweeps + extra_runs,
            reassigned_silhouettes=self.models * self.seeds * self.min_cluster_sizes,
            npmi_evaluations=2 * finals,
            ari_evaluations=2 * (self.models * grid + self.k_values + new),
        )


def plan_experiment(
    inputs: ExperimentInputs, settings: ExperimentSettings, *, cache: ReductionCache | None = None
) -> ExperimentPlan:
    """Return the plan of a run on ``inputs``, checking them, and counting the reductions that
    ``cache`` already holds.

    Raises:
        ExperimentError: if the inputs are inconsistent.
    """
    _check_inputs(inputs, settings)
    cache = cache or NoReductionCache()
    cached = sum(
        _cached(cache, model, seed, settings.topic_map.umap) is not None
        for model in inputs.models
        for seed in settings.seeds
    )
    return ExperimentPlan(
        documents=len(inputs.doc_codes),
        models=len(inputs.models),
        seeds=len(settings.seeds),
        k_values=len(settings.k_values),
        min_cluster_sizes=len(settings.min_cluster_sizes),
        reductions=len(inputs.models) * len(settings.seeds),
        reductions_cached=cached,
        projection_method=settings.topic_map.projection.method,
        matched_k=settings.matched_k,
    )


def _check_inputs(inputs: ExperimentInputs, settings: ExperimentSettings) -> None:
    points = len(inputs.doc_codes)
    problems = []
    if len(set(inputs.doc_codes)) != points:
        problems.append("the doc codes repeat")
    models = tuple(model.model_id for model in inputs.models)
    if models != settings.models or len(models) != 2:
        problems.append(f"the models are {list(models)}, not the two of embedding.models")
    for model in inputs.models:
        if np.shape(model.vectors)[0] != points:
            problems.append(f"{model.model_id} has {np.shape(model.vectors)[0]} rows")
    if not sparse.issparse(inputs.tfidf) or inputs.tfidf.shape != (points, len(inputs.vocabulary)):
        problems.append("the TF-IDF matrix is not sparse, one row per thesis, one column per term")
    if len(inputs.documents) != points:
        problems.append(f"there are {len(inputs.documents)} token documents")
    if problems:
        raise ExperimentError(
            f"the inputs of {points} theses are inconsistent: {'; '.join(problems)}"
        )


def _cached(
    cache: ReductionCache, model: SemanticInput, seed: int, settings: UmapSettings
) -> np.ndarray | None:
    """Return the cached reduction when it has the expected shape and finite values."""
    embedding = cache.get(reduction_key(model, settings, seed))
    expected = (np.shape(model.vectors)[0], settings.n_components)
    if embedding is None or embedding.shape != expected or not np.isfinite(embedding).all():
        return None
    return embedding


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class RunRow(_Row):
    """One row of ``runs.parquet``: one clustering run (D30 E6)."""

    run_id: str
    config_hash: str
    representation: str
    model_revision: str | None
    document_vector: str
    algorithm: str
    k: int | None
    min_cluster_size: int | None
    seed: int
    points: int
    clusters: int
    noise_points: int
    noise_fraction: float | None
    silhouette_cosine: float | None
    davies_bouldin_l2: float | None
    calinski_harabasz_l2: float | None
    undefined_reason: str | None
    silhouette_noise_reassigned: float | None
    labels_sha256: str
    final: bool
    matched_k: bool
    npmi: float | None
    npmi_sensitivity: float | None
    reduction_reused: bool | None
    umap_seconds: float | None
    cluster_seconds: float
    metrics_seconds: float
    ctfidf_seconds: float | None


class ConfigurationRow(_Row):
    """One row of ``configurations.parquet``: one configuration and hyperparameter over the
    seeds, mean and sample SD; ARI matrices as JSON."""

    representation: str
    algorithm: str
    k: int | None
    min_cluster_size: int | None
    configuration: str | None
    chosen: bool
    final: bool
    matched_k: bool
    seeds: int
    metrics_defined: bool
    silhouette_mean: float | None
    silhouette_sd: float | None
    davies_bouldin_mean: float | None
    davies_bouldin_sd: float | None
    calinski_harabasz_mean: float | None
    calinski_harabasz_sd: float | None
    noise_fraction_mean: float | None
    noise_fraction_sd: float | None
    clusters_mean: float | None
    clusters_sd: float | None
    silhouette_noise_reassigned_mean: float | None
    silhouette_noise_reassigned_sd: float | None
    npmi_mean: float | None
    npmi_sd: float | None
    npmi_sensitivity_mean: float | None
    npmi_sensitivity_sd: float | None
    ari_mean: float | None
    ari_minimum: float | None
    ari_matrix: str
    ari_sensitivity_mean: float | None
    ari_sensitivity_minimum: float | None
    ari_sensitivity_matrix: str


class TopicRow(_Row):
    """One row of ``topics.parquet``: one topic of a final run, its terms vocabulary lemmas."""

    run_id: str
    topic: int
    size: int
    terms: tuple[str, ...]
    scores: tuple[float, ...]
    npmi: float | None
    npmi_sensitivity: float | None
    absent_terms: tuple[str, ...]


class MapRow(_Row):
    """One thesis of the representative run's map: its label and both 2-D coordinates."""

    run_id: str
    doc_code: str
    label: int
    map_x: float
    map_y: float
    pca_x: float
    pca_y: float


@dataclass(frozen=True)
class Repeatability:
    """E6: one UMAP fit and one K-means run done again, compared by SHA-256."""

    model_id: str
    seed: int
    k: int
    umap_sha256: tuple[str, str]
    kmeans_sha256: tuple[str, str]

    @property
    def umap_identical(self) -> bool:
        return self.umap_sha256[0] == self.umap_sha256[1]

    @property
    def kmeans_identical(self) -> bool:
        return self.kmeans_sha256[0] == self.kmeans_sha256[1]


@dataclass(frozen=True)
class ExperimentResult:
    """Everything a run gives, ready to be written."""

    snapshot_id: str
    doc_codes: tuple[str, ...]
    runs: tuple[RunRow, ...]
    configurations: tuple[ConfigurationRow, ...]
    topics: tuple[TopicRow, ...]
    labels: dict[str, np.ndarray]
    """Each run's labels, by run id, in the order of :attr:`doc_codes`."""
    map: tuple[MapRow, ...]
    selection: dict
    """The content of ``selection.json``: plain JSON data."""
    repeatability: Repeatability
    counts: RunCounts
    matched: MatchedSource
    reductions_reused: int


def matched_source(
    k: int, baseline_k: int, k_values: Sequence[int], enabled: bool
) -> MatchedSource:
    """Return where the matched-K runs come from (D30 E3): nowhere when disabled, B0's own runs
    at its K*, the sweep's runs at ``k``, or five new runs."""
    if not enabled:
        return "off"
    if k == baseline_k:
        return "baseline"
    return "sweep" if k in k_values else "new"


def require_terms_present(coherence: CoherenceResult) -> None:
    """Stop when a top term is absent from the NPMI reference: a pipeline mismatch (D30 E4).

    Raises:
        ExperimentError: naming how many terms are absent, never which.
    """
    absent = sum(len(topic.absent_terms) for topic in coherence.topics)
    if absent:
        raise ExperimentError(
            f"{absent} top terms are absent from the NPMI reference, which the declared inputs "
            "cannot give: a pipeline mismatch (D30 E4)"
        )


def run_id(representation: str, algorithm: str, parameter: int, seed: int) -> str:
    """Return the id of a run, such as ``tfidf__kmeans_k24__seed7``."""
    name = "k" if algorithm == "kmeans" else "mcs"
    return f"{model_slug(representation)}__{algorithm}_{name}{parameter}__seed{seed}"


def run_experiment(
    inputs: ExperimentInputs,
    settings: ExperimentSettings,
    *,
    ports: TopicMapPorts | None = None,
    cache: ReductionCache | None = None,
    timer: Callable[[], float] = time.perf_counter,
    report: Callable[[str], None] = lambda _: None,
) -> ExperimentResult:
    """Run the experiment of D30 on ``inputs`` (see the module).

    Raises:
        ExperimentError: if the inputs are inconsistent, or a top term is absent from the NPMI
            reference (E4).
        TopicMapError: if a library breaks its contract, or the vocabulary does not belong to
            the documents.
    """
    _check_inputs(inputs, settings)
    return _Experiment(
        inputs, settings, ports or TopicMapPorts(), cache or NoReductionCache(), timer, report
    ).run()


Config = tuple[str, Algorithm, int]
"""A configuration and hyperparameter: the representation, the algorithm, K or the size."""
RunKey = tuple[str, Algorithm, int, int]
"""A run: its configuration and its seed."""


@dataclass
class _Run:
    labels: np.ndarray
    metrics: InternalMetrics
    reassigned: float | None
    reused: bool | None
    umap_seconds: float | None
    cluster_seconds: float
    metrics_seconds: float
    matched: bool = False


@dataclass(frozen=True)
class _Topics:
    topic_set: TopicSet
    primary: CoherenceResult
    sensitivity: CoherenceResult
    seconds: float


def _seconds(value: float | None) -> float | None:
    return None if value is None else round(max(value, 0.0), 3)


def _reassigned_silhouette(vectors: np.ndarray, labels: np.ndarray) -> float | None:
    """Return the silhouette over every thesis, each noise thesis sent to the topic whose
    centroid (the mean of its members) is nearest by cosine (D14, D30 E5)."""
    topics = sorted(set(labels.tolist()) - {NOISE_LABEL})
    if not topics:
        return None
    reassigned = np.array(labels)
    noise = reassigned == NOISE_LABEL
    if noise.any():
        centroids = topic_centroids(vectors, reassigned, topics)
        nearest = (normalize(vectors[noise]) @ normalize(centroids).T).argmax(axis=1)
        reassigned[noise] = np.asarray(topics)[nearest]
    return internal_metrics(vectors, reassigned).silhouette_cosine


def _json_default(value: object) -> object:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON data")


class _Experiment:
    """One run of the experiment: its state while the steps of :func:`run_experiment` run."""

    def __init__(
        self,
        inputs: ExperimentInputs,
        settings: ExperimentSettings,
        ports: TopicMapPorts,
        cache: ReductionCache,
        timer: Callable[[], float],
        report: Callable[[str], None],
    ) -> None:
        self.inputs, self.settings, self.ports = inputs, settings, ports
        self.cache, self.timer, self.report = cache, timer, report
        self.models = {model.model_id: model for model in inputs.models}
        self.runs: dict[RunKey, _Run] = {}
        self.topics: dict[RunKey, _Topics] = {}
        self.stability: dict[Config, tuple[SeedStability, SeedStability]] = {}
        self.names: dict[Config, str] = {}
        """The final configurations' names; the matched one is ``B0`` too."""
        self.chosen: set[Config] = set()
        self.counts = dict.fromkeys(RunCounts.__dataclass_fields__, 0)
        self.reused = 0
        self.first_reduction: Reduction | None = None

    def vectors(self, representation: str) -> np.ndarray | sparse.csr_matrix:
        if representation == BASELINE_REPRESENTATION:
            return self.inputs.tfidf
        return self.models[representation].vectors

    def run(self) -> ExperimentResult:
        self.sweep_semantic()
        repeatability = self.repeatability()
        self.sweep_tfidf(self.settings.k_values)
        configs = self.configurations()
        for config in configs:
            self.score_stability(config)
        selection, matched, extra, map_rows = self.select()
        return ExperimentResult(
            snapshot_id=self.inputs.snapshot_id,
            doc_codes=self.inputs.doc_codes,
            runs=tuple(self.run_row(key, run) for key, run in self.runs.items()),
            configurations=tuple(self.configuration_row(c) for c in configs + extra),
            topics=tuple(
                row for key in self.runs if key in self.topics for row in self.topic_rows(key)
            ),
            labels={run_id(*key): run.labels for key, run in self.runs.items()},
            map=map_rows,
            selection=selection,
            repeatability=repeatability,
            counts=RunCounts(**self.counts),
            matched=matched,
            reductions_reused=self.reused,
        )

    # The sweeps (E1) and their metrics (E5)

    def reduction(self, model: SemanticInput, seed: int) -> tuple[Reduction, float | None, bool]:
        umap = self.settings.topic_map.umap
        cached = _cached(self.cache, model, seed, umap)
        if cached is not None:
            self.reused += 1
            embedding = np.array(cached)
            embedding.setflags(write=False)
            return Reduction(seed=seed, settings=umap, embedding=embedding), None, True
        started = self.timer()
        reduction = fit_reduction(model.vectors, umap, seed, umap_factory=self.ports.umap)
        seconds = self.timer() - started
        self.counts["umap_fits"] += 1
        self.cache.put(reduction_key(model, umap, seed), reduction.embedding)
        return reduction, seconds, False

    def sweep_semantic(self) -> None:
        settings = self.settings
        runs = [
            *(KMeansRun(n_clusters=k) for k in settings.k_values),
            *(HdbscanRun(min_cluster_size=size) for size in settings.min_cluster_sizes),
        ]
        for model in self.inputs.models:
            for seed in settings.seeds:
                reduction, seconds, reused = self.reduction(model, seed)
                self.first_reduction = self.first_reduction or reduction
                state = "reused the cached reduction" if reused else f"UMAP in {seconds:.1f} s"
                self.report(f"{model.model_id} seed {seed}: {state}; clustering {len(runs)} runs")
                for run in runs:
                    started = self.timer()
                    labels = cluster(
                        reduction.embedding,
                        run,
                        seed=seed,
                        settings=settings.topic_map,
                        ports=self.ports,
                    )
                    if isinstance(run, KMeansRun):
                        key: RunKey = (model.model_id, "kmeans", run.n_clusters, seed)
                    else:
                        key = (model.model_id, "hdbscan", run.min_cluster_size, seed)
                    self.measure(key, labels, self.timer() - started, reused, seconds)

    def sweep_tfidf(self, k_values: Sequence[int], *, matched: bool = False) -> None:
        for seed in self.settings.seeds:
            self.report(f"TF-IDF seed {seed}: {len(k_values)} K-means runs on the sparse matrix")
            for k in k_values:
                started = self.timer()
                labels = kmeans_labels(
                    self.inputs.tfidf,
                    k,
                    seed,
                    self.settings.topic_map.kmeans,
                    model_factory=self.ports.kmeans,
                )
                key: RunKey = (BASELINE_REPRESENTATION, "kmeans", k, seed)
                self.measure(key, labels, self.timer() - started, None, None, matched=matched)

    def measure(
        self,
        key: RunKey,
        labels: np.ndarray,
        cluster_seconds: float,
        reused: bool | None,
        umap_seconds: float | None,
        *,
        matched: bool = False,
    ) -> None:
        self.counts["clusterings"] += 1
        self.counts["internal_metrics"] += 1
        started = self.timer()
        vectors = self.vectors(key[0])
        metrics = internal_metrics(vectors, labels)
        reassigned = None
        if key[1] == "hdbscan":
            self.counts["reassigned_silhouettes"] += 1
            reassigned = _reassigned_silhouette(dense_vectors(vectors), labels)
        self.runs[key] = _Run(
            labels=labels,
            metrics=metrics,
            reassigned=reassigned,
            reused=reused,
            umap_seconds=umap_seconds,
            cluster_seconds=cluster_seconds,
            metrics_seconds=self.timer() - started,
            matched=matched,
        )

    def repeatability(self) -> Repeatability:
        """Fit the first (model, seed) reduction again and rerun its first K-means (E6)."""
        model, seed, k = self.inputs.models[0], self.settings.seeds[0], self.settings.k_values[0]
        first = self.first_reduction
        if first is None:
            raise ExperimentError("the repeatability check needs a reduction")
        fresh = fit_reduction(
            model.vectors, self.settings.topic_map.umap, seed, umap_factory=self.ports.umap
        )
        again = cluster(
            first.embedding,
            KMeansRun(n_clusters=k),
            seed=seed,
            settings=self.settings.topic_map,
            ports=self.ports,
        )
        self.counts["umap_fits"] += 1
        self.counts["clusterings"] += 1
        labels = self.runs[(model.model_id, "kmeans", k, seed)].labels
        return Repeatability(
            model_id=model.model_id,
            seed=seed,
            k=k,
            umap_sha256=(_array_sha256(first.embedding), _array_sha256(fresh.embedding)),
            kmeans_sha256=(_array_sha256(labels), _array_sha256(again)),
        )

    def configurations(self) -> list[Config]:
        settings = self.settings
        configs: list[Config] = []
        for model in self.inputs.models:
            configs += [(model.model_id, "kmeans", k) for k in settings.k_values]
            configs += [(model.model_id, "hdbscan", s) for s in settings.min_cluster_sizes]
        return configs + [(BASELINE_REPRESENTATION, "kmeans", k) for k in settings.k_values]

    def seed_runs(self, config: Config) -> list[_Run]:
        return [self.runs[(*config, seed)] for seed in self.settings.seeds]

    def seed_topics(self, config: Config) -> list[_Topics | None]:
        return [self.topics.get((*config, seed)) for seed in self.settings.seeds]

    def score_stability(self, config: Config) -> None:
        labelings = [run.labels for run in self.seed_runs(config)]
        self.counts["ari_evaluations"] += 2
        self.stability[config] = (
            seed_stability(labelings, noise_handling=self.settings.ari),
            seed_stability(labelings, noise_handling=self.settings.ari_sensitivity),
        )

    # The final runs: c-TF-IDF and NPMI (E4)

    def score_topics(self, config: Config) -> None:
        topic_map, documents = self.settings.topic_map, self.inputs.documents
        for seed in self.settings.seeds:
            key = (*config, seed)
            if key in self.topics:
                continue
            started = self.timer()
            topic_set = ctfidf_topics(
                self.runs[key].labels,
                documents,
                vocabulary=self.inputs.vocabulary,
                max_ngram=topic_map.max_ngram,
                top_n=topic_map.top_n,
                include_noise_class=topic_map.include_noise_class,
                backend=self.ports.ctfidf,
            )
            seconds = self.timer() - started
            scores = [
                npmi_coherence(
                    topic_set.terms(),
                    documents,
                    top_n=topic_map.top_n,
                    unit=setting.unit,
                    window_size=setting.window_size,
                )
                for setting in (self.settings.npmi, self.settings.npmi_sensitivity)
            ]
            for coherence in scores:
                require_terms_present(coherence)
            self.counts["ctfidf_fits"] += 1
            self.counts["npmi_evaluations"] += 2
            self.topics[key] = _Topics(topic_set, scores[0], scores[1], seconds)

    # The choices (E2) and the selection (E3)

    def setting_summary(self, config: Config) -> SettingSummary:
        runs = self.seed_runs(config)
        return SettingSummary(
            parameter=config[2],
            silhouette=summarize([run.metrics.silhouette_cosine for run in runs]),
            noise_fraction=summarize([run.metrics.noise_fraction for run in runs]),
            metrics_defined=all(run.metrics.undefined_reason is None for run in runs),
        )

    def candidate(self, name: str, config: Config, eligible: bool) -> Candidate:
        runs, topics = self.seed_runs(config), self.seed_topics(config)
        primary, sensitivity = self.stability[config]
        return Candidate(
            configuration=name,
            representation=config[0],
            algorithm=config[1],
            parameter=config[2],
            eligible=eligible,
            metrics_defined=all(run.metrics.undefined_reason is None for run in runs),
            silhouette=summarize([run.metrics.silhouette_cosine for run in runs]),
            noise_fraction=summarize([run.metrics.noise_fraction for run in runs]),
            clusters=summarize([run.metrics.clusters for run in runs]),
            npmi=summarize([t.primary.mean if t else None for t in topics]),
            ari_mean=primary.mean,
            npmi_sensitivity=summarize([t.sensitivity.mean if t else None for t in topics]),
            ari_mean_sensitivity=sensitivity.mean,
            silhouette_noise_reassigned=(
                summarize([run.reassigned for run in runs]) if config[1] == "hdbscan" else None
            ),
        )

    def choose(self) -> tuple[dict[str, dict[str, Any]], dict[str, Candidate]]:
        """E2's choices, then the final configurations' topics and their candidates."""
        settings = self.settings
        models = [model.model_id for model in self.inputs.models]
        choices: dict[str, dict[str, Any]] = {}
        finals: dict[str, tuple[Config, bool]] = {}
        for index, model in enumerate([*models, BASELINE_REPRESENTATION]):
            k = choose_k([self.setting_summary((model, "kmeans", k)) for k in settings.k_values])
            choices[model] = {"kmeans": k}
            if model == BASELINE_REPRESENTATION:
                finals["B0"] = ((model, "kmeans", k.parameter), True)
                continue
            size = choose_min_cluster_size(
                [self.setting_summary((model, "hdbscan", s)) for s in settings.min_cluster_sizes],
                max_mean_noise_fraction=settings.gates.max_mean_noise_fraction,
            )
            choices[model]["hdbscan"] = size
            finals[f"C{2 * index + 1}"] = ((model, "kmeans", k.parameter), True)
            finals[f"C{2 * index + 2}"] = ((model, "hdbscan", size.parameter), size.eligible)
        for name, (config, _) in finals.items():
            self.report(f"c-TF-IDF and NPMI: {name} at {config[2]}, {len(settings.seeds)} seeds")
            self.score_topics(config)
            self.names[config] = name
            self.chosen.add(config)
        return choices, {name: self.candidate(name, *final) for name, final in finals.items()}

    def select(self) -> tuple[dict, MatchedSource, list[Config], tuple[MapRow, ...]]:
        settings, gates = self.settings, self.settings.gates
        choices, candidates = self.choose()
        findings = [
            f"{rep}, {algorithm}: {choice.reason}"
            for rep, items in choices.items()
            for algorithm, choice in items.items()
            if choice.at_bound or not choice.eligible
        ]
        stage_one_records, winners = {}, []
        for index, model in enumerate(settings.models):
            pair = (candidates[f"C{2 * index + 1}"], candidates[f"C{2 * index + 2}"])
            fallback = settings.fallback_algorithm
            decision = stage_one(*pair, gates=gates, fallback=fallback)
            flips = stage_one_sensitivities(*pair, gates=gates, fallback=fallback)
            stage_one_records[model] = {
                "decision": asdict(decision),
                "sensitivities": [asdict(item) for item in flips],
            }
            winners.append(next(c for c in pair if c.configuration == decision.winner))
            findings += [f"{model}, stage 1: {finding}" for finding in decision.findings]
            findings += [f"{model}, stage 1: {s.reason}" for s in flips if s.flipped]
        model_fallback = settings.fallback_model
        stage_two = stage_two_provisional(*winners, gates=gates, fallback_model=model_fallback)
        two_flips = stage_two_sensitivities(*winners, gates=gates, fallback_model=model_fallback)
        findings += [f"stage 2: {finding}" for finding in stage_two.findings]
        findings += [f"stage 2: {s.reason}" for s in two_flips if s.flipped]
        winner = next(c for c in winners if c.representation == stage_two.winner)

        baseline, extra = candidates["B0"], []
        k = matched_k(winner) if settings.matched_k else baseline.parameter
        source = matched_source(k, baseline.parameter, settings.k_values, settings.matched_k)
        matched_config: Config = (BASELINE_REPRESENTATION, "kmeans", k)
        if source == "new":
            self.sweep_tfidf((k,), matched=True)
            self.score_stability(matched_config)
            extra.append(matched_config)
        matched = baseline
        if source in ("sweep", "new"):
            for run in self.seed_runs(matched_config):
                run.matched = True
            self.report(f"c-TF-IDF and NPMI: B0 at the matched K {k}")
            self.score_topics(matched_config)
            self.names[matched_config] = "B0"
            matched = self.candidate("B0", matched_config, True)
        criterion_one = criterion_one_npmi(winner, baseline, matched)
        criterion_4 = criterion_four(winner, gates=gates, max_cv=settings.max_cv)
        if criterion_one.met is False:
            findings.append(f"criterion 1: {criterion_one.reason}")
        if not criterion_4.met:
            findings.append(f"criterion 4 is not met: {criterion_4.reason}")
        winner_config: Config = (winner.representation, winner.algorithm, winner.parameter)
        representative = representative_seed(
            self.stability[winner_config][0].matrix, settings.seeds
        )
        representative_run = run_id(*winner_config, representative.seed)
        self.report(f"map of the representative run {representative_run}")
        selection = {
            "snapshot_id": self.inputs.snapshot_id,
            "provisional": True,
            "choices": {
                rep: {algorithm: asdict(choice) for algorithm, choice in items.items()}
                for rep, items in choices.items()
            },
            "configurations": {
                name: {**asdict(c), "gates_failed": list(evaluate_gates(c, gates).failed)}
                for name, c in candidates.items()
            },
            "stage_one": stage_one_records,
            "stage_two": {
                "decision": asdict(stage_two),
                "sensitivities": [asdict(item) for item in two_flips],
            },
            "winner": {
                "configuration": winner.configuration,
                "representation": winner.representation,
                "algorithm": winner.algorithm,
                "parameter": winner.parameter,
                "provisional": True,
            },
            "matched_k": {"k": k, "source": source},
            "criterion_one": asdict(criterion_one),
            "criterion_four": asdict(criterion_4),
            "representative": {**asdict(representative), "run_id": representative_run},
            "map": {
                "run_id": representative_run,
                "seed": representative.seed,
                "method": settings.topic_map.projection.method,
                "inspection": "pca",
            },
            "findings": findings,
        }
        plain = json.loads(json.dumps(selection, default=_json_default))
        return plain, source, extra, self.map_rows(winner_config, representative.seed)

    def map_rows(self, config: Config, seed: int) -> tuple[MapRow, ...]:
        """The representative run's 2-D map and PCA inspection coordinates."""
        vectors = self.vectors(config[0])
        projection = self.settings.topic_map.projection
        coordinates = [
            project(vectors, setting, seed, umap_factory=self.ports.umap)
            for setting in (projection, replace(projection, method="pca"))
        ]
        self.counts["umap_fits"] += projection.method == "umap"
        self.counts["pca_fits"] += 1 + (projection.method == "pca")
        identifier = run_id(*config, seed)
        labels = self.runs[(*config, seed)].labels
        return tuple(
            MapRow(
                run_id=identifier,
                doc_code=code,
                label=int(label),
                map_x=float(xy[0]),
                map_y=float(xy[1]),
                pca_x=float(pca[0]),
                pca_y=float(pca[1]),
            )
            for code, label, xy, pca in zip(
                self.inputs.doc_codes, labels, *coordinates, strict=True
            )
        )

    # The records (E6)

    def config_hash(self, key: RunKey) -> str:
        representation, algorithm, parameter, seed = key
        topic_map = self.settings.topic_map
        semantic = representation != BASELINE_REPRESENTATION
        if algorithm == "kmeans":
            arguments = {"n_clusters": parameter, **asdict(topic_map.kmeans)}
        else:
            arguments = {"min_cluster_size": parameter, **asdict(topic_map.hdbscan)}
        return _json_sha256(
            {
                "representation": representation,
                "model_revision": self.models[representation].revision if semantic else None,
                "document_vector": (
                    self.inputs.embedding_variant if semantic else self.inputs.tfidf_variant
                ),
                "space": "umap" if semantic else "sparse_l2",
                "umap": asdict(topic_map.umap) if semantic else None,
                "algorithm": algorithm,
                "arguments": arguments,
                "seed": seed,
                "code_versions": (
                    {"embedder_version": EMBEDDER_VERSION}
                    if semantic
                    else {"tfidf_version": TFIDF_VERSION}
                ),
            }
        )

    def run_row(self, key: RunKey, run: _Run) -> RunRow:
        representation, algorithm, parameter, seed = key
        semantic = representation != BASELINE_REPRESENTATION
        metrics, topics = run.metrics, self.topics.get(key)
        return RunRow(
            run_id=run_id(*key),
            config_hash=self.config_hash(key),
            representation=representation,
            model_revision=self.models[representation].revision if semantic else None,
            document_vector=(
                self.inputs.embedding_variant if semantic else self.inputs.tfidf_variant
            ),
            algorithm=algorithm,
            k=parameter if algorithm == "kmeans" else None,
            min_cluster_size=parameter if algorithm == "hdbscan" else None,
            seed=seed,
            points=metrics.points,
            clusters=metrics.clusters,
            noise_points=metrics.noise_points,
            noise_fraction=metrics.noise_fraction,
            silhouette_cosine=metrics.silhouette_cosine,
            davies_bouldin_l2=metrics.davies_bouldin_l2,
            calinski_harabasz_l2=metrics.calinski_harabasz_l2,
            undefined_reason=metrics.undefined_reason,
            silhouette_noise_reassigned=run.reassigned,
            labels_sha256=_array_sha256(run.labels),
            final=topics is not None,
            matched_k=run.matched,
            npmi=topics.primary.mean if topics else None,
            npmi_sensitivity=topics.sensitivity.mean if topics else None,
            reduction_reused=run.reused,
            umap_seconds=_seconds(run.umap_seconds),
            cluster_seconds=_seconds(run.cluster_seconds),
            metrics_seconds=_seconds(run.metrics_seconds),
            ctfidf_seconds=_seconds(topics.seconds) if topics else None,
        )

    def configuration_row(self, config: Config) -> ConfigurationRow:
        runs, topics = self.seed_runs(config), self.seed_topics(config)
        primary, sensitivity = self.stability[config]
        series = {
            "silhouette": [run.metrics.silhouette_cosine for run in runs],
            "davies_bouldin": [run.metrics.davies_bouldin_l2 for run in runs],
            "calinski_harabasz": [run.metrics.calinski_harabasz_l2 for run in runs],
            "noise_fraction": [run.metrics.noise_fraction for run in runs],
            "clusters": [run.metrics.clusters for run in runs],
            "silhouette_noise_reassigned": [run.reassigned for run in runs],
            "npmi": [t.primary.mean if t else None for t in topics],
            "npmi_sensitivity": [t.sensitivity.mean if t else None for t in topics],
        }
        values: dict[str, float | None] = {}
        for prefix, items in series.items():
            summary = summarize(items)
            values[f"{prefix}_mean"], values[f"{prefix}_sd"] = summary.mean, summary.sd
        return ConfigurationRow(
            representation=config[0],
            algorithm=config[1],
            k=config[2] if config[1] == "kmeans" else None,
            min_cluster_size=config[2] if config[1] == "hdbscan" else None,
            configuration=self.names.get(config),
            chosen=config in self.chosen,
            final=all(topics),
            matched_k=all(run.matched for run in runs),
            seeds=len(runs),
            metrics_defined=all(run.metrics.undefined_reason is None for run in runs),
            **values,
            ari_mean=primary.mean,
            ari_minimum=primary.minimum,
            ari_matrix=json.dumps([list(row) for row in primary.matrix]),
            ari_sensitivity_mean=sensitivity.mean,
            ari_sensitivity_minimum=sensitivity.minimum,
            ari_sensitivity_matrix=json.dumps([list(row) for row in sensitivity.matrix]),
        )

    def topic_rows(self, key: RunKey) -> list[TopicRow]:
        scored = self.topics[key]
        return [
            TopicRow(
                run_id=run_id(*key),
                topic=topic.topic,
                size=topic.size,
                terms=topic.terms,
                scores=topic.scores,
                npmi=primary.npmi,
                npmi_sensitivity=sensitivity.npmi,
                absent_terms=primary.absent_terms,
            )
            for topic, primary, sensitivity in zip(
                scored.topic_set.topics,
                scored.primary.topics,
                scored.sensitivity.topics,
                strict=True,
            )
        ]
