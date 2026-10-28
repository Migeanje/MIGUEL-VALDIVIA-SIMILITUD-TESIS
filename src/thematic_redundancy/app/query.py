"""The query view's data layer (T28): one proposal, represented as the corpus is, and what the
saved artifacts say about it, with no Streamlit import.

**Representation (S2).** A proposal holds a title, an abstract and objectives, the sections that
represent a thesis (D09). It becomes a ficha (:func:`proposal_ficha`) and goes through the
corpus's own steps for the provisional winner's model: T13's
:func:`~thematic_redundancy.representation.build_chunks.chunk_fichas` (light cleaning, the title's
place and year suffix removed, sentence-aligned chunks under the largest count of every
configured tokenizer), the model's raw chunk vectors, and T14's
:func:`~thematic_redundancy.representation.vectors.pool_chunk_vectors` (the ``section_mean``
document vector, L2-normalized). The model and the tokenizers load from the local Hugging Face
cache only, and must be those that embedded and chunked the corpus: the model's revision is the
one the embeddings manifest records, and each tokenizer's SHA-256 the one the chunk manifest
records. No chunk vector is cached: the proposal's texts are encoded in memory and dropped.

**The index (saved artifacts).** The winner's document vectors (T14, through ``load_embeddings``,
whose manifest must be the one T19 ran on), the map run's labels (``map.parquet``), its topics
(``topics.parquet``), its centroids recomputed from those vectors and labels (D30 E5, as the gap
run does), and from them O06's atypical threshold and the saturation quartiles (T23, with the
values the user confirmed on 2026-10-20), and the stable gaps of :mod:`.gap_supports`.

**The answer.** T23's assignment with O06's flag and the topic's saturation quartile, through
T26's :func:`~thematic_redundancy.recommendation.alternatives.alternative_lines`, which also gives
the neighbor topics and the nearby stable gaps; T22's top-k by cosine; and T26's antecedents by
MMR. The similarity ranking carries no redundancy verdict: the threshold τ waits for T22's
labeled set.

Nothing here writes a file, logs a proposal or keeps it: a proposal lives only in the call.
"""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np

from thematic_redundancy.app.data import (
    AppSources,
    ArtifactError,
    ArtifactProblem,
    Experiment,
    GapRecords,
    StableGap,
    TopicMapView,
    read_gap_run,
    read_topic_rows,
)
from thematic_redundancy.app.gap_supports import stable_supports
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.gaps.stability import GapSupport
from thematic_redundancy.recommendation.alternatives import (
    AlternativeLines,
    NearbyGap,
    NeighborReference,
    PairAnchor,
    alternative_lines,
)
from thematic_redundancy.recommendation.antecedents import AntecedentRanking, rank_antecedents
from thematic_redundancy.redundancy.saturation import (
    QuartileBoundary,
    SaturationQuartiles,
    TopicSaturation,
    saturation_quartiles,
)
from thematic_redundancy.redundancy.search import Neighbor, cosine_top_k
from thematic_redundancy.representation.build_chunks import BUILD_COMMAND as CHUNKS_COMMAND
from thematic_redundancy.representation.build_chunks import (
    CHUNKS_MANIFEST_FILE_NAME,
    chunk_fichas,
    chunks_manifest,
)
from thematic_redundancy.representation.build_embeddings import BUILD_COMMAND as EMBED_COMMAND
from thematic_redundancy.representation.build_embeddings import (
    EMBEDDINGS_DIR_NAME,
    EMBEDDINGS_MANIFEST_FILE_NAME,
    load_embeddings,
)
from thematic_redundancy.representation.chunker import MaxTokenCounter, TokenCounter
from thematic_redundancy.representation.hf_tokenizers import load_token_counter
from thematic_redundancy.representation.st_encoder import load_sentence_encoder
from thematic_redundancy.representation.vectors import (
    EncoderSettings,
    SentenceEncoder,
    pool_chunk_vectors,
)
from thematic_redundancy.shared.config import MODEL_MAX_SEQ_LENGTH
from thematic_redundancy.topic_map.assignment import (
    ATYPICAL_PERCENTILE,
    AtypicalCheck,
    AtypicalThreshold,
    PercentileMethod,
    ThresholdScope,
    TopicCentroids,
    atypical_threshold,
    check_atypical,
)
from thematic_redundancy.topic_map.build import TopicMap, topic_centroids
from thematic_redundancy.topic_map.ctfidf import Topic
from thematic_redundancy.topic_map.run_experiment import RUN_COMMAND as T19_COMMAND

SIMILAR_THESES = 10
"""How many of the most similar theses the view lists (T22's top-k)."""
ATYPICAL_SCOPE: ThresholdScope = "global"
PERCENTILE_METHOD: PercentileMethod = "linear"
SATURATION_BOUNDARY: QuartileBoundary = "lower"
"""T23's values, confirmed by the user on 2026-10-20 for T26, T27 and T28."""
NEIGHBOR_REFERENCE: NeighborReference = "proposal"
PAIR_ANCHOR: PairAnchor = "midpoint"
MAX_GAP_DISTANCE: float | None = None
ANTECEDENT_POOL: int | None = None
"""T26's recommended values (its implementation notes 1, 5, 6 and 7)."""
PROPOSAL_UUID = UUID(int=0)
"""The placeholder identity of a proposal's ficha, which no repository item has."""

ProposalProblem = Literal["title", "body"]
"""What a proposal lacks: its title, or both its abstract and its objectives."""
ModelProblem = Literal["missing", "mismatch"]
LoadEncoder = Callable[[str, EncoderSettings], SentenceEncoder]
LoadCounter = Callable[[str], TokenCounter]


class ModelError(Exception):
    """The model or a tokenizer that represents a proposal is missing from the local cache, or is
    not the one that represented the corpus."""

    def __init__(self, problem: ModelProblem, model_id: str, *, detail: str) -> None:
        self.problem = problem
        self.model_id = model_id
        self.detail = detail
        """What is wrong, in English, for the server's log; never shown to the user."""
        super().__init__(f"{model_id} is {problem}: {detail}")


# --- The proposal -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """A thesis proposal as a user types it: the sections that represent a thesis (D09)."""

    title: str
    abstract: str = ""
    objectives: str = ""

    def missing(self) -> tuple[ProposalProblem, ...]:
        """Return what the proposal lacks: a title, and an abstract or objectives, as an included
        thesis of the corpus has."""
        problems: list[ProposalProblem] = []
        if not self.title.strip():
            problems.append("title")
        if not self.abstract.strip() and not self.objectives.strip():
            problems.append("body")
        return tuple(problems)


def proposal_ficha(proposal: Proposal) -> Ficha:
    """Return a ficha that holds the proposal's texts as a thesis's ficha holds them; its other
    fields are placeholders that only satisfy the ficha's rules, and nothing reads them.

    Raises:
        ValueError: if the proposal has no title.
    """
    objectives = proposal.objectives if proposal.objectives.strip() else None
    return Ficha(
        item_uuid=PROPOSAL_UUID,
        doc_code=doc_code_for(PROPOSAL_UUID),
        handle_url="https://propuesta.invalid/",
        snapshot_id="propuesta",
        document_type="tesis",
        renati_type_raw=None,
        title=proposal.title,
        abstract=proposal.abstract if proposal.abstract.strip() else None,
        objectives=objectives,
        objectives_status="manual" if objectives else "pending",
        program_key="propuesta",
        program_name="Propuesta",
        issue_date_raw="propuesta",
        issued_after_snapshot=False,
        rights="unknown",
        include=True,
    )


# --- The index of saved artifacts ---------------------------------------------------------------


@dataclass(frozen=True)
class QuerySettings:
    """The values the answer uses: the configured recommender's and the duplicates' (O09)."""

    similar: int
    top_k: int
    mmr_lambda: float
    neighbor_topics: int
    nearby_gaps: int
    known_duplicates: frozenset[str]
    """The doc codes of the theses known to be deposited twice (``redundancy.known_duplicates``)."""


@dataclass(frozen=True, eq=False)
class QueryIndex:
    """What answering a proposal needs from the saved artifacts (see the module)."""

    model_id: str
    revision: str
    """The model revision that embedded the corpus; the encoder must be this one."""
    doc_codes: tuple[str, ...]
    vectors: np.ndarray
    """The winner's document vectors, one row per thesis of :attr:`doc_codes`; read-only."""
    topic_map: TopicMap
    centroids: TopicCentroids
    threshold: AtypicalThreshold
    saturation: SaturationQuartiles
    supports: tuple[GapSupport, ...]
    """The stable gaps of the shown map, in the order of the gaps page's stable gaps."""
    tokenizers: tuple[tuple[str, str], ...]
    """Each tokenizer that chunked the corpus: its model and its ``tokenizer.json`` SHA-256."""
    max_tokens: int
    settings: QuerySettings


def load_query_index(
    sources: AppSources, experiment: Experiment, topic_map: TopicMapView, records: GapRecords
) -> QueryIndex:
    """Return the index of the saved artifacts that a proposal is compared with.

    Raises:
        ArtifactError: if the embeddings or the chunk manifest are missing, changed or other
            than those T19 and the configuration name, or the records disagree.
    """
    config = sources.config
    interim = sources.fichas_dir
    embeddings_manifest = interim / EMBEDDINGS_DIR_NAME / EMBEDDINGS_MANIFEST_FILE_NAME
    model_id = experiment.winner.representation
    chunks_path = interim / CHUNKS_MANIFEST_FILE_NAME
    try:
        chunks = chunks_manifest(interim)
    except FileNotFoundError as error:
        raise _artifact("missing", chunks_path, CHUNKS_COMMAND, error) from None
    except ValueError as error:
        raise _artifact("invalid", chunks_path, CHUNKS_COMMAND, error) from None
    try:
        content = embeddings_manifest.read_bytes()
        embedded = load_embeddings(interim, model_id, variant=config.embedding.document_vector)
    except FileNotFoundError as error:
        raise _artifact("missing", embeddings_manifest, EMBED_COMMAND, error) from None
    except (OSError, ValueError) as error:
        raise _artifact("invalid", embeddings_manifest, EMBED_COMMAND, error) from None
    if hashlib.sha256(content).hexdigest() != experiment.inputs.get(embeddings_manifest.name):
        raise _artifact("stale", embeddings_manifest, T19_COMMAND, "not the embeddings of T19")
    if chunks.max_tokens != config.chunking.max_tokens:
        raise _artifact("stale", chunks_path, CHUNKS_COMMAND, "another chunk size")
    points = topic_map.points.set_index("doc_code")
    if set(points.index) != set(embedded.doc_codes):
        raise _artifact(
            "inconsistent", embeddings_manifest, T19_COMMAND, "other theses than the map's"
        )
    # The labels and the coordinates follow the rows of the document vectors.
    in_rows = points.loc[list(embedded.doc_codes)]
    labels = in_rows["topic"].to_numpy(dtype=np.int64)
    labels.flags.writeable = False
    topics = tuple(
        Topic(topic=row.topic, size=row.size, terms=row.terms, scores=row.scores)
        for row in topic_map.topics
    )
    vectors = embedded.vectors
    vectors.flags.writeable = False
    centroids = topic_centroids(vectors, labels, [topic.topic for topic in topics])
    shown = TopicMap(
        seed=experiment.map_seed,
        clustering=None,
        labels=labels,
        topics=topics,
        noise=None,
        centroids=centroids,
        coordinates=in_rows[["x", "y"]].to_numpy(dtype=np.float64),
        reduction=None,
    )
    targets = TopicCentroids.from_topic_map(shown)
    supports = stable_supports(
        read_gap_run(sources, experiment), read_topic_rows(sources, experiment), records
    )
    recommender = config.recommender
    return QueryIndex(
        model_id=model_id,
        revision=embedded.revision,
        doc_codes=embedded.doc_codes,
        vectors=vectors,
        topic_map=shown,
        centroids=targets,
        threshold=atypical_threshold(
            vectors,
            labels,
            targets,
            percentile=ATYPICAL_PERCENTILE,
            method=PERCENTILE_METHOD,
            scope=ATYPICAL_SCOPE,
        ),
        saturation=saturation_quartiles(
            labels, method=PERCENTILE_METHOD, boundary=SATURATION_BOUNDARY
        ),
        supports=supports,
        tokenizers=tuple((t.model_id, t.tokenizer_sha256) for t in chunks.tokenizers),
        max_tokens=chunks.max_tokens,
        settings=QuerySettings(
            similar=SIMILAR_THESES,
            top_k=recommender.top_k,
            mmr_lambda=recommender.mmr_lambda,
            neighbor_topics=recommender.neighbor_topics,
            nearby_gaps=recommender.nearby_gaps,
            known_duplicates=frozenset(
                code for pair in config.redundancy.known_duplicates for code in pair
            ),
        ),
    )


# --- The model --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalEncoder:
    """The winner's model and the corpus's tokenizers, checked, ready to represent proposals."""

    encoder: SentenceEncoder
    counter: TokenCounter
    """The largest count of every configured tokenizer, as T13 chunks."""
    max_tokens: int
    seconds: float
    """How long the model and the tokenizers took to load."""


def load_proposal_encoder(
    sources: AppSources,
    index: QueryIndex,
    *,
    load_encoder: LoadEncoder | None = None,
    load_counter: LoadCounter | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> ProposalEncoder:
    """Load the winner's model and every configured tokenizer from the local cache, and check
    that they are the ones that represented the corpus.

    Raises:
        ModelError: if the model or a tokenizer is missing from the local cache, or is another
            revision or file than the corpus's.
    """
    started = timer()
    settings = sources.config.embedding
    load_encoder = load_encoder or load_sentence_encoder
    load_counter = load_counter or load_token_counter
    try:
        encoder = load_encoder(
            index.model_id,
            EncoderSettings(
                batch_size=settings.batch_size, device=settings.device, threads=settings.threads
            ),
        )
    except OSError as error:
        raise ModelError("missing", index.model_id, detail=type(error).__name__) from None
    identity = encoder.identity
    if identity.revision != index.revision or identity.max_seq_length != MODEL_MAX_SEQ_LENGTH:
        raise ModelError(
            "mismatch",
            index.model_id,
            detail=f"revision {identity.revision}, {identity.max_seq_length} tokens; the corpus "
            f"was embedded by revision {index.revision} at {MODEL_MAX_SEQ_LENGTH} tokens",
        )
    recorded = dict(index.tokenizers)
    counters = []
    for model_id in settings.models:
        try:
            counter = load_counter(model_id)
        except OSError as error:
            raise ModelError("missing", model_id, detail=type(error).__name__) from None
        if recorded.get(model_id) != counter.identity.tokenizer_sha256:
            raise ModelError("mismatch", model_id, detail="another tokenizer.json than T13's")
        counters.append(counter)
    return ProposalEncoder(
        encoder=encoder,
        counter=MaxTokenCounter(tuple(counters)),
        max_tokens=index.max_tokens,
        seconds=max(timer() - started, 0.0),
    )


def represent(proposal: Proposal, encoder: ProposalEncoder) -> np.ndarray:
    """Return the proposal's ``section_mean`` document vector, as T13 and T14 give a thesis's.

    Raises:
        ValueError: if the proposal has no title, or its sections give no direction.
    """
    records = chunk_fichas([proposal_ficha(proposal)], encoder.counter, encoder.max_tokens)
    vectors = encoder.encoder.encode([record.text for record in records])
    return pool_chunk_vectors(records, vectors).section_mean[0]


# --- The answer -------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryAnswer:
    """What the saved artifacts say about one proposal."""

    lines: AlternativeLines
    """T26: the assigned topic (T23), its neighbor topics and the nearby stable gaps."""
    atypical: AtypicalCheck
    saturation: TopicSaturation
    similar: tuple[Neighbor, ...]
    antecedents: AntecedentRanking


def answer_query(vector: np.ndarray, index: QueryIndex) -> QueryAnswer:
    """Return the proposal's topic with O06's flag and its saturation, its most similar theses,
    its antecedents and its alternative lines (see the module)."""
    settings = index.settings
    lines = alternative_lines(
        vector,
        index.topic_map,
        index.supports,
        neighbor_topics=settings.neighbor_topics,
        nearby_gaps=settings.nearby_gaps,
        neighbor_reference=NEIGHBOR_REFERENCE,
        pair_anchor=PAIR_ANCHOR,
        max_distance=MAX_GAP_DISTANCE,
    )
    return QueryAnswer(
        lines=lines,
        atypical=check_atypical(lines.assignment, index.threshold),
        saturation=index.saturation.of_topic(lines.assignment.topic),
        similar=cosine_top_k(index.vectors, vector, k=settings.similar, exclude=None),
        antecedents=rank_antecedents(
            index.vectors,
            vector,
            k=settings.top_k,
            mmr_lambda=settings.mmr_lambda,
            pool_size=ANTECEDENT_POOL,
            exclude=None,
        ),
    )


def nearby_stable_gaps(
    answer: QueryAnswer, index: QueryIndex, records: GapRecords
) -> tuple[tuple[NearbyGap, StableGap], ...]:
    """Return each nearby gap of the answer with the gaps page's stable gap it is."""
    position = {id(support): number for number, support in enumerate(index.supports)}
    return tuple(
        (nearby, records.gaps[position[id(nearby.support)]]) for nearby in answer.lines.gaps
    )


def _artifact(
    problem: ArtifactProblem, path: Path, command: str, detail: str | BaseException
) -> ArtifactError:
    """Return the error about ``path``; an exception gives its kind only, never its message."""
    text = detail if isinstance(detail, str) else type(detail).__name__
    return ArtifactError(problem, path.name, path.parent, command=command, detail=text)
