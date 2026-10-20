"""The topic map of one configuration and one seed (T16, D30).

:func:`build_topic_maps` serves the semantic runs of one (model, seed): it fits the UMAP
reduction once and the 2-D projection once, then, for each clustering run, clusters that one
reduction and builds a :class:`TopicMap`, so K-means and HDBSCAN share the same space (D30 E1).
:func:`build_topic_map` builds one map from labels already known, such as a saved run's; it fits
no reduction. The TF-IDF baseline, which has no UMAP and no map, takes its topic terms from its
labels through :func:`~.ctfidf.ctfidf_topics` alone (D28 C4).

A map holds:

- the labels, as the clustering gave them, -1 for HDBSCAN's noise, never relabelled (D12, D16);
- the topics, one per cluster label, ascending, with their top c-TF-IDF terms over the fixed
  vocabulary (D30 E4); the noise's terms apart, when the noise is a class of the fit;
- the centroids, one per topic in the same order: the mean of the topic's members' original
  vectors (D14, D30 E5), never of the UMAP space; noise has none;
- the 2-D coordinates of every thesis, noise included, for the visualization map;
- the reduction that the labels came from, when they came from one.

The seed is passed explicitly to every step that has randomness: UMAP, the projection and
K-means; HDBSCAN is deterministic. The libraries sit behind :class:`TopicMapPorts`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from thematic_redundancy.evaluation.clustering_metrics import NOISE_LABEL, label_array
from thematic_redundancy.shared.config import AppConfig
from thematic_redundancy.topic_map import TopicMapError
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicCtfidf
from thematic_redundancy.topic_map.clustering import (
    ClusteringRun,
    HdbscanFactory,
    HdbscanRun,
    HdbscanSettings,
    KMeansFactory,
    KMeansRun,
    KMeansSettings,
    hdbscan_labels,
    hdbscan_model,
    kmeans_labels,
    kmeans_model,
)
from thematic_redundancy.topic_map.ctfidf import CtfidfBackend, Topic, ctfidf_topics
from thematic_redundancy.topic_map.reduction import (
    PROJECTION_COMPONENTS,
    ProjectionSettings,
    Reduction,
    UmapFactory,
    UmapSettings,
    dense_vectors,
    fit_reduction,
    project,
    umap_model,
)


@dataclass(frozen=True)
class TopicMapSettings:
    """Everything a topic map takes from the configuration, besides the seed."""

    umap: UmapSettings
    projection: ProjectionSettings
    kmeans: KMeansSettings
    hdbscan: HdbscanSettings
    top_n: int
    """The top terms kept per topic: NPMI's ``top_n`` (D12, D30 E4)."""
    max_ngram: int
    """T15's longest term, which the c-TF-IDF analyzer forms."""
    include_noise_class: bool

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        """Gather the ``umap`` block, ``topic_map.projection``, the experiment's K-means,
        HDBSCAN, NPMI ``top_n`` and c-TF-IDF noise class, and ``tfidf.max_ngram``."""
        experiment = config.experiment
        return cls(
            umap=UmapSettings.from_config(config.umap),
            projection=ProjectionSettings.from_config(config.topic_map.projection),
            kmeans=KMeansSettings.from_config(experiment.kmeans),
            hdbscan=HdbscanSettings.from_config(experiment.hdbscan),
            top_n=experiment.npmi.top_n,
            max_ngram=config.tfidf.max_ngram,
            include_noise_class=experiment.ctfidf.include_noise_class,
        )


@dataclass(frozen=True)
class TopicMapPorts:
    """The libraries behind the topic map; the defaults are the real ones."""

    umap: UmapFactory = umap_model
    kmeans: KMeansFactory = kmeans_model
    hdbscan: HdbscanFactory = hdbscan_model
    ctfidf: CtfidfBackend = field(default_factory=BertopicCtfidf)


@dataclass(frozen=True)
class TopicMap:
    """The topic map of one configuration and one seed (see the module)."""

    seed: int
    clustering: ClusteringRun | None
    """The run that gave the labels; ``None`` when they were given."""
    labels: np.ndarray
    """One int64 label per thesis, -1 for noise; read-only."""
    topics: tuple[Topic, ...]
    noise: Topic | None
    """The noise's terms when the noise is a class of the c-TF-IDF fit and has members."""
    centroids: np.ndarray
    """``topics × dimensions``, row ``i`` for ``topics[i]``, in the original space; read-only."""
    coordinates: np.ndarray
    """``theses × 2``, the visualization map; read-only."""
    reduction: Reduction | None
    """The UMAP fit that the labels came from; ``None`` when they were given."""

    @property
    def noise_points(self) -> int:
        return int(np.count_nonzero(self.labels == NOISE_LABEL))

    def terms(self) -> tuple[tuple[str, ...], ...]:
        """Return each topic's terms in topic order, as NPMI reads them."""
        return tuple(topic.terms for topic in self.topics)


def topic_centroids(vectors: Any, labels: Any, topics: Sequence[int]) -> np.ndarray:
    """Return, for each label of ``topics`` in order, the mean of its members' ``vectors``."""
    matrix = np.asarray(vectors, dtype=np.float64)
    label_values = np.asarray(labels)
    centroids = np.zeros((len(topics), matrix.shape[1]))
    for row, topic in enumerate(topics):
        centroids[row] = matrix[label_values == topic].mean(axis=0)
    return centroids


def cluster(
    points: Any, run: ClusteringRun, *, seed: int, settings: TopicMapSettings, ports: TopicMapPorts
) -> np.ndarray:
    """Return the labels of ``run`` on ``points``, with the run's seed for K-means."""
    if isinstance(run, KMeansRun):
        return kmeans_labels(
            points, run.n_clusters, seed, settings.kmeans, model_factory=ports.kmeans
        )
    if isinstance(run, HdbscanRun):
        return hdbscan_labels(
            points, run.min_cluster_size, settings.hdbscan, model_factory=ports.hdbscan
        )
    raise TopicMapError(f"unknown clustering run: {type(run).__name__}")


def build_topic_map(
    vectors: Any,
    labels: Sequence[int] | np.ndarray,
    documents: Sequence[Sequence[Sequence[str]]],
    *,
    vocabulary: Sequence[str],
    seed: int,
    settings: TopicMapSettings,
    ports: TopicMapPorts | None = None,
    clustering: ClusteringRun | None = None,
    reduction: Reduction | None = None,
    coordinates: np.ndarray | None = None,
) -> TopicMap:
    """Return the topic map of ``labels`` over the theses' original ``vectors``.

    Args:
        vectors: one dense row per thesis, the original document vectors.
        labels: one label per thesis, -1 for noise; kept as given.
        documents: each thesis as the full-clean token lists of its sections.
        vocabulary: T15's terms (see :func:`~.ctfidf.ctfidf_topics`).
        seed: the run's seed, for the projection when ``coordinates`` is not given.
        settings: the configured settings.
        ports: the libraries; the real ones by default.
        clustering: the run that gave the labels, recorded with the map.
        reduction: the UMAP fit that the labels came from, recorded with the map.
        coordinates: the 2-D coordinates already computed for this seed; projected otherwise.

    Raises:
        TopicMapError: if the inputs are invalid or of different lengths, or the reduction or
            the coordinates do not belong to these theses and this seed.
        ValueError: if the labels are not 1-D integers of at least -1.
    """
    ports = ports or TopicMapPorts()
    matrix = dense_vectors(vectors)
    label_values = label_array(labels)
    points = matrix.shape[0]
    if label_values.size != points:
        raise TopicMapError(f"{label_values.size} labels were given for {points} vectors")
    if len(documents) != points:
        raise TopicMapError(f"{len(documents)} documents were given for {points} vectors")
    if reduction is not None:
        if reduction.seed != seed:
            raise TopicMapError(f"the reduction has seed {reduction.seed}, not {seed}")
        if reduction.embedding.shape[0] != points:
            raise TopicMapError(
                f"the reduction has {reduction.embedding.shape[0]} rows for {points} vectors"
            )
    if coordinates is None:
        coordinates = project(matrix, settings.projection, seed, umap_factory=ports.umap)
    elif np.shape(coordinates) != (points, PROJECTION_COMPONENTS):
        raise TopicMapError(
            f"the coordinates have shape {np.shape(coordinates)}, not "
            f"{(points, PROJECTION_COMPONENTS)}"
        )
    topic_set = ctfidf_topics(
        label_values,
        documents,
        vocabulary=vocabulary,
        max_ngram=settings.max_ngram,
        top_n=settings.top_n,
        include_noise_class=settings.include_noise_class,
        backend=ports.ctfidf,
    )
    centroids = topic_centroids(matrix, label_values, [topic.topic for topic in topic_set.topics])
    return TopicMap(
        seed=seed,
        clustering=clustering,
        labels=_read_only(label_values),
        topics=topic_set.topics,
        noise=topic_set.noise,
        centroids=_read_only(centroids),
        coordinates=_shared(coordinates),
        reduction=reduction,
    )


def build_topic_maps(
    vectors: Any,
    documents: Sequence[Sequence[Sequence[str]]],
    runs: Sequence[ClusteringRun],
    *,
    vocabulary: Sequence[str],
    seed: int,
    settings: TopicMapSettings,
    ports: TopicMapPorts | None = None,
) -> tuple[TopicMap, ...]:
    """Return one topic map per run, all on one UMAP fit and one 2-D projection of
    ``vectors`` for ``seed``.

    Raises:
        TopicMapError: as :func:`build_topic_map` does, or for an invalid run.
    """
    ports = ports or TopicMapPorts()
    matrix = dense_vectors(vectors)
    reduction = fit_reduction(matrix, settings.umap, seed, umap_factory=ports.umap)
    coordinates = project(matrix, settings.projection, seed, umap_factory=ports.umap)
    maps = []
    for run in runs:
        labels = cluster(reduction.embedding, run, seed=seed, settings=settings, ports=ports)
        maps.append(
            build_topic_map(
                matrix,
                labels,
                documents,
                vocabulary=vocabulary,
                seed=seed,
                settings=settings,
                ports=ports,
                clustering=run,
                reduction=reduction,
                coordinates=coordinates,
            )
        )
    return tuple(maps)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _shared(array: Any) -> np.ndarray:
    """Return a read-only array as it is, so maps of one seed share it; copy anything else."""
    if isinstance(array, np.ndarray) and not array.flags.writeable:
        return array
    return _read_only(np.array(array))
