"""Synthetic saved artifacts for the app's tests (T28), written under pytest's temporary
directory, outside the repository.

The project holds the repository's ``config/default.yaml`` and, for snapshot ``S1``:

- the fichas of 14 included theses of three programs (Sistemas, Industrial and Mecánica,
  2021-2026) and one duplicate that the corpus excludes, each with author and advisor codes,
  and an invented person-like string in its abstract and keywords that the app must never show;
- the chunk manifest and the winner's document vectors (T13, T14): each thesis lies on its
  topic's axis of an 8-dimensional space (each noise thesis on its own axis), with a small jitter
  and a background, so the stand-in encoder (:class:`FakeEncoder`), which counts a text's topic
  words on those axes, places a proposal near the theses of its words' topic;
- T19's records: the winner C4's five seed runs with three topics each, the map run (seed 42)
  with the 2-D coordinates of every thesis, its topics (0 with 5 theses, 1 with 4, 2 with 3) and
  2 noise theses; the other runs' terms must not show;
- T24a's records, written by T24a's own ``gap_rows`` and ``member_rows`` from real gap objects:
  six gaps of the reference run (five stable, one per rule and a second of rule (c), of affinity
  2 and 1, and one found in 2 runs only), and the other runs' copies that give those supports.

Every file is written as its own command writes it, with the SHA-256 that its manifest records.
"""

import builtins
import hashlib
import io
import json
import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from thematic_redundancy.corpus.build_fichas import (
    EXCLUSIONS_FILE_NAME,
    FICHAS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    Exclusion,
    FichasManifest,
)
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.gaps import results_store as gap_store
from thematic_redundancy.gaps.results_store import GapManifest, GapRow, MemberRow, gap_rows
from thematic_redundancy.gaps.results_store import member_rows as gap_member_rows
from thematic_redundancy.gaps.rules import (
    InterstitialPair,
    SmallCoherentTopic,
    TemporalDecline,
    ThesisMetadata,
    TransferableTopic,
)
from thematic_redundancy.gaps.stability import gap_support
from thematic_redundancy.representation.build_chunks import (
    CHUNKER_VERSION,
    CHUNKS_MANIFEST_FILE_NAME,
    ChunksManifest,
    TokenizerRecord,
)
from thematic_redundancy.representation.build_embeddings import (
    DOCUMENT_FILE_NAMES,
    EMBEDDER_VERSION,
    EMBEDDINGS_DIR_NAME,
    EMBEDDINGS_MANIFEST_FILE_NAME,
    SECTION_VECTORS_FILE_NAME,
    SECTIONS_FILE_NAME,
    EmbeddedSection,
    EmbeddingsManifest,
    EncoderSettingsRecord,
    ModelEmbeddingsRecord,
    VariantAgreementRecord,
)
from thematic_redundancy.representation.hf_tokenizers import TokenizerIdentity
from thematic_redundancy.representation.vector_cache import model_slug
from thematic_redundancy.representation.vectors import EncoderIdentity
from thematic_redundancy.shared.config import MODEL_MAX_SEQ_LENGTH, load_config
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map import results_store as t19_store
from thematic_redundancy.topic_map.experiment import MapRow, TopicRow, run_id
from thematic_redundancy.topic_map.results_store import ExperimentManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(REPOSITORY_ROOT / "config" / "default.yaml")
WINNER_MODEL = CONFIG.embedding.models[1]
SNAPSHOT = "S1"
SEEDS = (7, 13, 21, 42, 73)
RUNS = tuple(run_id(WINNER_MODEL, "hdbscan", 10, seed) for seed in SEEDS)
MAP_RUN = RUNS[3]
OTHER_RUN = RUNS[0]

PLANTED_PERSON = "Nombre Inventado Plantado"
"""A person-like string held only by fields the app never shows (abstract and keywords)."""
PERSON_CODE_PREFIXES = ("AUT-", "ADV-")

TOPIC_TERMS = {
    0: ("mantenimiento", "planta", "preventivo", "falla", "equipo"),
    1: ("logística", "almacén", "inventario", "demanda", "costo"),
    2: ("aprendizaje", "estudiante", "virtual", "educativo", "plataforma"),
}
OTHER_RUN_TERMS = ("zqotra", "zqcorrida")
"""Terms of a topic of another run, which the app must not show."""


@dataclass(frozen=True)
class Thesis:
    index: int
    program: str
    year: int
    topic: int
    x: float
    y: float

    @property
    def item_uuid(self) -> UUID:
        return UUID(int=1000 + self.index)

    @property
    def doc_code(self) -> str:
        return doc_code_for(self.item_uuid)

    @property
    def title(self) -> str:
        return (
            f"Tesis sintética {self.index:02d} sobre {TOPIC_TERMS.get(self.topic, ('ruido',))[0]}"
        )

    @property
    def url(self) -> str:
        return f"https://repo.example.edu/handle/123456789/{2000 + self.index}"


THESES = (
    Thesis(1, "industrial", 2021, 0, 1.0, 1.0),
    Thesis(2, "industrial", 2022, 0, 1.2, 0.9),
    Thesis(3, "industrial", 2022, 0, 0.8, 1.1),
    Thesis(4, "industrial", 2023, 0, 1.1, 1.3),
    Thesis(5, "sistemas", 2025, 0, 0.9, 0.8),
    Thesis(6, "sistemas", 2021, 1, 5.0, 5.0),
    Thesis(7, "sistemas", 2024, 1, 5.2, 4.8),
    Thesis(8, "sistemas", 2025, 1, 4.9, 5.1),
    Thesis(9, "industrial", 2026, 1, 5.1, 5.2),
    Thesis(10, "mecanica", 2023, 2, -3.0, 2.0),
    Thesis(11, "mecanica", 2024, 2, -3.2, 2.1),
    Thesis(12, "sistemas", 2026, 2, -2.9, 1.8),
    Thesis(13, "mecanica", 2022, -1, 2.0, -4.0),
    Thesis(14, "industrial", 2025, -1, -1.0, -3.0),
)
DUPLICATE = Thesis(15, "industrial", 2022, 0, 0.0, 0.0)
"""An excluded duplicate: it has a ficha, but no point on the map."""

PROGRAM_NAMES = {program.key: program.name for program in CONFIG.snapshot.programs}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(root: Path) -> dict[str, str]:
    """Return every file under ``root`` with its SHA-256, by relative path."""
    return {
        path.relative_to(root).as_posix(): sha256_of(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def ficha(thesis: Thesis, *, include: bool = True, duplicate_of: UUID | None = None) -> Ficha:
    return Ficha(
        item_uuid=thesis.item_uuid,
        doc_code=thesis.doc_code,
        handle_url=thesis.url,
        snapshot_id=SNAPSHOT,
        document_type="tesis",
        renati_type_raw="https://purl.org/pe-repo/renati/type#tesis",
        title=thesis.title,
        abstract=f"Resumen sintético que menciona a {PLANTED_PERSON}.",
        program_key=thesis.program,
        program_name=PROGRAM_NAMES[thesis.program],
        issue_date_raw=f"{thesis.year}-06-01",
        issue_year=thesis.year,
        issued_after_snapshot=False,
        keywords=(PLANTED_PERSON, "palabra clave"),
        rights="open",
        advisor_code=f"ADV-{thesis.index:010x}",
        author_codes=(f"AUT-{thesis.index:010x}", f"AUT-{100 + thesis.index:010x}"),
        include=include,
        exclusion_reason=None if include else "duplicate",
        duplicate_of=duplicate_of,
    )


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def fichas_dir(self) -> Path:
        return self.root / "data" / INTERIM_DIR_NAME / SNAPSHOT

    @property
    def t19_results(self) -> Path:
        return t19_store.results_dir(self.root / "results", SNAPSHOT)

    @property
    def t19_processed(self) -> Path:
        return t19_store.processed_dir(self.root / "data", SNAPSHOT)

    @property
    def gaps_results(self) -> Path:
        return gap_store.results_dir(self.root / "results", SNAPSHOT)

    @property
    def gaps_processed(self) -> Path:
        return gap_store.processed_dir(self.root / "data", SNAPSHOT)

    @property
    def embeddings_dir(self) -> Path:
        return self.fichas_dir / EMBEDDINGS_DIR_NAME

    @property
    def embeddings_manifest(self) -> Path:
        return self.embeddings_dir / EMBEDDINGS_MANIFEST_FILE_NAME

    def argv(self) -> list[str]:
        return ["main.py", "--project-root", str(self.root), "--snapshot-id", SNAPSHOT]


# --- The winner's document vectors, and the stand-in encoder and tokenizers ---------------------

DIMENSIONS = 8
BACKGROUND_AXIS = 7
BACKGROUND = 0.01
"""Every vector, of a thesis or of a text, holds this much of the last axis, so none is 0."""
NOISE_AXES = {13: 3, 14: 4}
MODEL_REVISION = "rev-synthetic"


def thesis_vector(thesis: Thesis) -> np.ndarray:
    """A thesis's document vector: its topic's axis (a noise thesis has its own), a small
    jitter on axis 5 and the background, L2-normalized."""
    vector = np.zeros(DIMENSIONS)
    vector[thesis.topic if thesis.topic >= 0 else NOISE_AXES[thesis.index]] = 1.0
    vector[5] = 0.05 * (thesis.index % 3 - 1)
    vector[BACKGROUND_AXIS] = BACKGROUND
    return vector / np.linalg.norm(vector)


def tokenizer_sha256(model_id: str) -> str:
    return hashlib.sha256(model_id.encode()).hexdigest()


class FakeCounter:
    """A stand-in tokenizer: a text's words plus two special tokens."""

    def __init__(self, model_id: str) -> None:
        self.identity = TokenizerIdentity(
            model_id=model_id,
            revision=MODEL_REVISION,
            tokenizer_sha256=tokenizer_sha256(model_id),
            vocabulary_size=100,
        )

    def count(self, text: str) -> int:
        return len(text.split()) + 2


class FakeEncoder:
    """A stand-in for the winner's model: a text's vector counts the words of each topic on that
    topic's axis, plus the background; ``texts`` records every text it encoded."""

    texts: list[str] = []

    def __init__(self, model_id: str, revision: str = MODEL_REVISION) -> None:
        self.identity = EncoderIdentity(
            model_id=model_id,
            revision=revision,
            max_seq_length=MODEL_MAX_SEQ_LENGTH,
            dimension=DIMENSIONS,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            FakeEncoder.texts.append(text)
            words = text.casefold().split()
            vector = np.zeros(DIMENSIONS)
            for topic, terms in TOPIC_TERMS.items():
                vector[topic] = sum(words.count(term) for term in terms)
            vector[BACKGROUND_AXIS] = BACKGROUND
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


def load_fake_encoder(model_id: str, settings: Any) -> FakeEncoder:
    return FakeEncoder(model_id)


def load_fake_counter(model_id: str) -> FakeCounter:
    return FakeCounter(model_id)


def npy(vectors: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    return buffer.getvalue()


def write_embeddings(project: Project) -> None:
    """Write the chunk manifest and the winner's embeddings as T13 and T14 record them: one title
    section per thesis, the document vectors of :func:`thesis_vector`."""
    store = ParquetTableStore()
    interim, embeddings = project.fichas_dir, project.embeddings_dir
    moment = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    chunks = ChunksManifest(
        snapshot_id=SNAPSHOT,
        created_at=moment,
        chunker_version=CHUNKER_VERSION,
        fichas_sha256=sha256_of(interim / FICHAS_FILE_NAME),
        max_tokens=CONFIG.chunking.max_tokens,
        tokenizers=tuple(
            TokenizerRecord(
                model_id=model,
                revision=MODEL_REVISION,
                tokenizer_sha256=tokenizer_sha256(model),
                vocabulary_size=100,
                largest_chunk_tokens=10,
            )
            for model in CONFIG.embedding.models
        ),
        chunks_file="chunks.parquet",
        chunks_sha256="d" * 64,
        counts={},
        cut_kinds={},
        sections_used={},
        sections_recovered=len(THESES),
        seconds=1.0,
    )
    (interim / CHUNKS_MANIFEST_FILE_NAME).write_text(chunks.model_dump_json(), encoding="utf-8")
    model_dir = embeddings / model_slug(WINNER_MODEL)
    model_dir.mkdir(parents=True)
    vectors = np.asarray([thesis_vector(thesis) for thesis in THESES], dtype=np.float32)
    files = {
        DOCUMENT_FILE_NAMES["section_mean"]: npy(vectors),
        SECTION_VECTORS_FILE_NAME: npy(vectors),
    }
    for name, content in files.items():
        (model_dir / name).write_bytes(content)
    sections = [
        EmbeddedSection(doc_code=thesis.doc_code, document_row=row, section="title", n_chunks=1)
        for row, thesis in enumerate(THESES)
    ]
    store.write(sections, embeddings / SECTIONS_FILE_NAME)
    manifest = EmbeddingsManifest(
        snapshot_id=SNAPSHOT,
        created_at=moment,
        embedder_version=EMBEDDER_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunks_sha256=chunks.chunks_sha256,
        cache_key_scheme="synthetic",
        cache_dir="cache/chunk_vectors",
        encoder=EncoderSettingsRecord(batch_size=8, device="cpu", threads=1),
        document_variant="section_mean",
        sections_file=SECTIONS_FILE_NAME,
        sections_sha256=sha256_of(embeddings / SECTIONS_FILE_NAME),
        counts={},
        models=(
            ModelEmbeddingsRecord(
                model_id=WINNER_MODEL,
                revision=MODEL_REVISION,
                max_seq_length=MODEL_MAX_SEQ_LENGTH,
                dimension=DIMENSIONS,
                normalize=False,
                directory=model_slug(WINNER_MODEL),
                files={
                    name: hashlib.sha256(content).hexdigest() for name, content in files.items()
                },
                cache_file="cache/chunk_vectors/synthetic.npy",
                distinct_texts=len(THESES),
                from_cache=0,
                embedded=len(THESES),
                embed_seconds=0.0,
                variant_agreement=VariantAgreementRecord(
                    threshold=0.95, minimum=1.0, median=1.0, below=0
                ),
            ),
        ),
        seconds=1.0,
    )
    project.embeddings_manifest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def write_fichas(project: Project) -> None:
    store = ParquetTableStore()
    interim = project.fichas_dir
    interim.mkdir(parents=True)
    fichas = [ficha(thesis) for thesis in THESES]
    fichas.append(ficha(DUPLICATE, include=False, duplicate_of=THESES[1].item_uuid))
    store.write(fichas, interim / FICHAS_FILE_NAME)
    exclusion = Exclusion(
        item_uuid=DUPLICATE.item_uuid,
        doc_code=DUPLICATE.doc_code,
        handle=None,
        program_key=DUPLICATE.program,
        document_type="tesis",
        exclusion_reason="duplicate",
        duplicate_of=THESES[1].item_uuid,
    )
    store.write([exclusion], interim / EXCLUSIONS_FILE_NAME)
    manifest = FichasManifest(
        snapshot_id=SNAPSHOT,
        created_at=datetime(2026, 10, 11, 12, 0, tzinfo=UTC),
        metadata_sha256="a" * 64,
        objectives_table_sha256="b" * 64,
        fichas_file=FICHAS_FILE_NAME,
        fichas_sha256=sha256_of(interim / FICHAS_FILE_NAME),
        exclusions_file=EXCLUSIONS_FILE_NAME,
        exclusions_sha256=sha256_of(interim / EXCLUSIONS_FILE_NAME),
        counts={"included": len(THESES)},
        seconds=1.0,
    )
    path = interim / FICHAS_MANIFEST_FILE_NAME
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def write_experiment(project: Project, *, provisional: bool = True) -> None:
    store = ParquetTableStore()
    results, processed = project.t19_results, project.t19_processed
    results.mkdir(parents=True)
    processed.mkdir(parents=True)
    # Every seed run of the winner has the map's three topics; only the map run's terms show.
    topics = [
        TopicRow(
            run_id=run,
            topic=topic,
            size=sum(thesis.topic == topic for thesis in THESES),
            terms=terms if run == MAP_RUN else (*OTHER_RUN_TERMS, f"zqtema{topic}"),
            scores=tuple(0.5 - 0.01 * index for index in range(len(terms))),
            npmi=0.2,
            npmi_sensitivity=0.1,
            absent_terms=(),
        )
        for run in RUNS
        for topic, terms in TOPIC_TERMS.items()
    ]
    store.write(topics, results / t19_store.TOPICS_FILE_NAME)
    points = [
        MapRow(
            run_id=MAP_RUN,
            doc_code=thesis.doc_code,
            label=thesis.topic,
            map_x=thesis.x,
            map_y=thesis.y,
            pca_x=-thesis.x,
            pca_y=-thesis.y,
        )
        for thesis in THESES
    ]
    store.write(points, processed / t19_store.MAP_FILE_NAME)
    selection = {
        "snapshot_id": SNAPSHOT,
        "provisional": provisional,
        "winner": {
            "configuration": "C4",
            "representation": WINNER_MODEL,
            "algorithm": "hdbscan",
            "parameter": 10,
            "provisional": provisional,
        },
        "representative": {"seed": 42, "run_id": MAP_RUN},
        "map": {"run_id": MAP_RUN, "seed": 42, "method": "umap", "inspection": "pca"},
    }
    (results / t19_store.SELECTION_FILE_NAME).write_text(json.dumps(selection), encoding="utf-8")
    moment = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)
    manifest = ExperimentManifest(
        snapshot_id=SNAPSHOT,
        declaration_sha256=CONFIG.experiment_declaration_sha256(),
        git_commit="b" * 40,
        inputs={
            EMBEDDINGS_MANIFEST_FILE_NAME: sha256_of(project.embeddings_manifest),
            "tfidf_manifest.json": "t" * 64,
            FICHAS_MANIFEST_FILE_NAME: sha256_of(project.fichas_dir / FICHAS_MANIFEST_FILE_NAME),
        },
        documents=len(THESES),
        provisional=provisional,
        winner="C4",
        matched="sweep",
        counts={},
        repeatability={},
        versions={},
        cpu={},
        files={
            name: sha256_of(results / name)
            for name in (t19_store.TOPICS_FILE_NAME, t19_store.SELECTION_FILE_NAME)
        },
        data_files={t19_store.MAP_FILE_NAME: sha256_of(processed / t19_store.MAP_FILE_NAME)},
        started_at=moment,
        finished_at=moment,
        seconds=1.0,
    )
    path = results / t19_store.MANIFEST_FILE_NAME
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


METADATA = ThesisMetadata(
    doc_codes=tuple(thesis.doc_code for thesis in THESES),
    programs=tuple(thesis.program for thesis in THESES),
    years=tuple(thesis.year for thesis in THESES),
)
BRIDGE = THESES[12]
"""The bridge thesis of the rule (b) gap: a noise thesis."""


def members(topic: int) -> frozenset[str]:
    return frozenset(thesis.doc_code for thesis in THESES if thesis.topic == topic)


def program_counts(topic: int) -> tuple[tuple[str, int], ...]:
    found = sorted(thesis.program for thesis in THESES if thesis.topic == topic)
    return tuple((program, found.count(program)) for program in sorted(set(found)))


def transferable(topic: int, source: str, target: str, affinity: int, *counts: float) -> Any:
    source_count, source_theses, target_count, target_theses = counts
    return TransferableTopic(
        topic=topic,
        source=source,
        target=target,
        affinity=affinity,
        min_affinity=GAP_SETTINGS["min_affinity"],
        source_count=source_count,
        source_theses=source_theses,
        source_share=source_count / source_theses,
        target_count=target_count,
        target_theses=target_theses,
        target_share=target_count / target_theses,
        present_min_count=GAP_SETTINGS["present_min_count"],
        present_min_share=GAP_SETTINGS["present_min_share"],
        rare_max_share=GAP_SETTINGS["rare_max_share"],
        members=members(topic),
    )


GAP_SETTINGS = {
    "nearest_topics": 3,
    "bridge_max_docs": 1,
    "temporal_first_year": 2021,
    "temporal_split_year": 2024,
    "temporal_last_year": 2026,
    "temporal_min_count": 3,
    "stability_min_runs": 4,
    "npmi_floor_quantile": 0.25,
    "min_affinity": 1,
    "present_min_count": 5,
    "present_min_share": 0.05,
    "rare_max_share": 0.0,
    "min_decline": 0.025,
    "stability_min_overlap": 0.5,
}
REFERENCE_GAPS = (
    SmallCoherentTopic(
        topic=2,
        size=3,
        size_threshold=3.0,
        topics_in_map=len(TOPIC_TERMS),
        npmi=0.2,
        npmi_floor=0.15,
        members=members(2),
        program_counts=program_counts(2),
    ),
    InterstitialPair(
        first_topic=0,
        second_topic=1,
        centroid_cosine=0.61,
        neighbor_rank=2,
        nearest_topics=GAP_SETTINGS["nearest_topics"],
        bridge_documents=1,
        bridge_max_docs=GAP_SETTINGS["bridge_max_docs"],
        bridges=frozenset({BRIDGE.doc_code}),
        first_members=members(0),
        second_members=members(1),
    ),
    transferable(0, "industrial", "mecanica", 2, 4, 6, 0, 3),
    transferable(1, "sistemas", "mecanica", 1, 3, 5, 0, 3),
    TemporalDecline(
        topic=0,
        early_years=(2021, 2023),
        late_years=(2024, 2026),
        early_count=4,
        early_theses=7,
        early_share=4 / 7,
        late_count=1,
        late_theses=7,
        late_share=1 / 7,
        decline=4 / 7 - 1 / 7,
        min_count=GAP_SETTINGS["temporal_min_count"],
        min_decline=GAP_SETTINGS["min_decline"],
        members=members(0),
        program_counts=program_counts(0),
    ),
    transferable(2, "mecanica", "industrial", 2, 2, 3, 0, 6),
)
"""The reference run's gaps: the first five are stable, the last one is found in 2 runs."""
PRESENCE = {7: (0, 1, 2, 3, 4, 5), 13: (0, 1, 2, 3, 4), 21: (0, 1, 2, 4), 73: (0, 2, 3, 4)}
"""Which reference gaps each other seed's run also has, so their support is 5, 4, 5, 4, 5, 2."""


def run_gaps() -> dict[str, tuple[Any, ...]]:
    """Return each run's gaps, by run id, in the order of the runs."""
    return {
        run: REFERENCE_GAPS
        if run == MAP_RUN
        else tuple(REFERENCE_GAPS[i] for i in PRESENCE[SEEDS[RUNS.index(run)]])
        for run in RUNS
    }


def gap_records() -> tuple[tuple[GapRow, ...], tuple[MemberRow, ...]]:
    """Return the gap run's rows and members, written by T24a's own record functions."""
    gaps = run_gaps()
    supports = gap_support(
        [gaps[run] for run in RUNS],
        reference=RUNS.index(MAP_RUN),
        min_runs=GAP_SETTINGS["stability_min_runs"],
        min_overlap=GAP_SETTINGS["stability_min_overlap"],
    )
    rows, member_list = [], []
    for run in (MAP_RUN, *(r for r in RUNS if r != MAP_RUN)):
        seed = SEEDS[RUNS.index(run)]
        found = gap_rows(
            run, seed, gaps[run], METADATA, supports=supports if run == MAP_RUN else None
        )
        rows += found
        member_list += gap_member_rows(run, gaps[run])
    return tuple(rows), tuple(member_list)


GAPS, MEMBERS = gap_records()
"""Every run's gap rows, the reference run's six first, and every gap's members."""
STABLE_GAPS = GAPS[:5]


def write_gaps(project: Project) -> None:
    store = ParquetTableStore()
    results, processed = project.gaps_results, project.gaps_processed
    results.mkdir(parents=True)
    processed.mkdir(parents=True)
    store.write(list(GAPS), results / gap_store.GAPS_FILE_NAME)
    store.write(list(MEMBERS), processed / gap_store.MEMBERS_FILE_NAME)
    t19_manifest = project.t19_results / t19_store.MANIFEST_FILE_NAME
    moment = datetime(2026, 10, 21, 22, 0, tzinfo=UTC)
    manifest = GapManifest(
        snapshot_id=SNAPSHOT,
        git_commit="c" * 40,
        experiment={t19_store.MANIFEST_FILE_NAME: sha256_of(t19_manifest)},
        inputs={},
        winner="C4",
        representation=WINNER_MODEL,
        provisional=True,
        runs=RUNS,
        reference_run=MAP_RUN,
        settings=GAP_SETTINGS,
        sensitivity={},
        counts={},
        versions={},
        cpu={},
        files={gap_store.GAPS_FILE_NAME: sha256_of(results / gap_store.GAPS_FILE_NAME)},
        data_files={
            gap_store.MEMBERS_FILE_NAME: sha256_of(processed / gap_store.MEMBERS_FILE_NAME)
        },
        started_at=moment,
        finished_at=moment,
        seconds=1.0,
    )
    path = results / gap_store.MANIFEST_FILE_NAME
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def make_project(root: Path) -> Project:
    project = Project(root)
    (root / "config").mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ROOT / "config" / "default.yaml", root / "config" / "default.yaml")
    write_fichas(project)
    write_embeddings(project)
    write_experiment(project)
    write_gaps(project)
    return project


# --- What the app shows and writes, for the AppTest tests -------------------------------------


def nodes(node: Any) -> Iterator[Any]:
    yield node
    for child in getattr(node, "children", {}).values():
        yield from nodes(child)


def rendered_text(app: AppTest) -> str:
    """Return everything the app rendered: each element's message, every table's cells and
    every chart's specification."""
    parts = []
    for element in nodes(app._tree):
        proto = getattr(element, "proto", None)
        if proto is not None:
            parts.append(str(proto))
    parts += [frame.value.to_csv() for frame in app.dataframe]
    parts += [chart.proto.spec for chart in app.get("plotly_chart")]
    return "\n".join(parts)


class WriteRecorder:
    """Record every file that Python opens for writing while it is installed."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_open, original_os_open = builtins.open, os.open

        def recording_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if any(flag in mode for flag in "wax+"):
                self.paths.append(str(file))
            return original_open(file, mode, *args, **kwargs)

        def recording_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND):
                self.paths.append(str(path))
            return original_os_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", recording_open)
        monkeypatch.setattr(io, "open", recording_open)
        monkeypatch.setattr(os, "open", recording_os_open)
