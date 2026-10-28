"""Synthetic inputs of the labeled set's commands, written to disk in the formats that their real
loaders read: a snapshot, the fichas outputs, the chunk table, the document vectors of both
configured models (from a stand-in encoder) and the TF-IDF matrices (from a stand-in cleaner).

Every uuid, text and name here is made up. The builders call the real writers and builds, so a
test that reads these files goes through the real loaders and their checks.
"""

import hashlib
import itertools
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock
from uuid import UUID

import numpy as np

import thematic_redundancy.representation.build_tfidf as tfidf_module
from thematic_redundancy.corpus.build_fichas import (
    EXCLUSIONS_FILE_NAME,
    FICHAS_FILE_NAME,
    FICHAS_MANIFEST_FILE_NAME,
    Exclusion,
    FichasManifest,
)
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.labeling.draw_pairs import DrawManifest, PairRow, records_dir, results_dir
from thematic_redundancy.labeling.pairs import BETWEEN, LABELED, PILOT, pair_id
from thematic_redundancy.preprocessing.stopwords import DOMAIN_STOPWORDS_FILE
from thematic_redundancy.representation.build_chunks import (
    CHUNKER_VERSION,
    CHUNKS_FILE_NAME,
    CHUNKS_MANIFEST_FILE_NAME,
    ChunkRecord,
    ChunksManifest,
    chunk_fichas,
)
from thematic_redundancy.representation.build_embeddings import build_embeddings
from thematic_redundancy.representation.build_tfidf import build_tfidf
from thematic_redundancy.representation.tfidf import CleanerIdentity
from thematic_redundancy.representation.vectors import EncoderIdentity
from thematic_redundancy.shared.config import (
    AppConfig,
    EmbeddingCountsConfig,
    TfidfCountsConfig,
)
from thematic_redundancy.shared.storage import ParquetTableStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_ID = "20261002T150405Z"
HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
RUN_TIME = datetime(2026, 10, 21, 9, 0, 0, tzinfo=UTC)
FACULTY = UUID("00000000-0000-4000-8000-0000000000fa")
DIMENSIONS = (8, 12)
"""The stand-in encoder's dimension for each configured model, in order."""


def item(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interim(root: Path) -> Path:
    return root / "data" / INTERIM_DIR_NAME / SNAPSHOT_ID


@dataclass(frozen=True)
class Thesis:
    """One synthetic thesis: its ficha's texts and the people of its snapshot record."""

    number: int
    program: str
    title: str
    abstract: str | None
    objectives: str | None
    keywords: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    advisor: str | None = None
    advisor_orcid: str | None = None
    include: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_code(self) -> str:
        return doc_code_for(item(self.number))


def ficha(thesis: Thesis, program_names: dict[str, str]) -> Ficha:
    with_objectives = thesis.objectives is not None
    values: dict[str, Any] = {
        "item_uuid": item(thesis.number),
        "doc_code": thesis.doc_code,
        "handle_url": f"https://hdl.handle.net/123456789/{1000 + thesis.number}",
        "snapshot_id": SNAPSHOT_ID,
        "document_type": "tesis",
        "renati_type_raw": "https://purl.org/pe-repo/renati/type#tesis",
        "title": thesis.title,
        "abstract": thesis.abstract,
        "objectives": thesis.objectives,
        "objectives_status": "extracted" if with_objectives else "no_pdf",
        "program_key": thesis.program,
        "program_name": program_names[thesis.program],
        "issue_date_raw": "2023-05-10",
        "issue_year": 2023,
        "issued_after_snapshot": False,
        "keywords": thesis.keywords,
        "rights": "open" if with_objectives else "restricted",
        "pdf_status": "downloaded" if with_objectives else "restricted",
        "pdf_sha256": f"{thesis.number:064x}" if with_objectives else None,
        "source_format": "digital" if with_objectives else "none",
        "section_source": "title_abstract_objectives" if with_objectives else "title_abstract",
        "include": thesis.include,
        **({} if thesis.include else {"exclusion_reason": "other"}),
        **thesis.extra,
    }
    return Ficha(**values)


def snapshot_record(thesis: Thesis, config: AppConfig) -> SnapshotRecord:
    collection = next(
        p.collection_uuid for p in config.snapshot.programs if p.key == thesis.program
    )
    fields: dict[str, list[str]] = {
        "renati.type": ["https://purl.org/pe-repo/renati/type#tesis"],
        "dc.title": [thesis.title],
        "dc.description.abstract": [thesis.abstract] if thesis.abstract else [],
        "dc.subject": list(thesis.keywords),
        "dc.contributor.author": list(thesis.authors),
        "dc.contributor.advisor": [thesis.advisor] if thesis.advisor else [],
        "renati.advisor.orcid": [thesis.advisor_orcid] if thesis.advisor_orcid else [],
    }
    return SnapshotRecord(
        uuid=item(thesis.number),
        handle=f"123456789/{1000 + thesis.number}",
        program_key=thesis.program,
        collection_uuid=collection,
        harvested_at=HARVEST_TIME,
        metadata={key: [{"value": value} for value in values] for key, values in fields.items()},
    )


def write_test_snapshot(root: Path, records: Sequence[SnapshotRecord], config: AppConfig) -> None:
    """Write ``records`` as snapshot :data:`SNAPSHOT_ID` under ``root``, as the harvest does."""
    content = encode_records(records)
    programs = {
        program.key: ProgramSummary(
            collection_uuid=program.collection_uuid,
            items=sum(record.program_key == program.key for record in records),
        )
        for program in config.snapshot.programs
    }
    manifest = SnapshotManifest(
        snapshot_id=SNAPSHOT_ID,
        harvested_at=HARVEST_TIME,
        base_url="https://repo.example.edu",
        search_url="https://repo.example.edu/server/api/discover/search/objects",
        query_parameters={},
        programs=programs,
        total_items=len(records),
        faculty_community_uuid=FACULTY,
        faculty_total=len(records),
        items_by_type={},
        future_dated_items=0,
        unparsed_issue_dates=0,
        dropped_metadata_keys=(),
        metadata_sha256=hashlib.sha256(content).hexdigest(),
    )
    write_snapshot(root / "data" / "raw" / SNAPSHOT_ID, content, manifest)


def write_fichas_outputs(interim_dir: Path, fichas: Sequence[Ficha]) -> None:
    """Save ``fichas`` as the fichas build does, its manifest last."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    store = ParquetTableStore()
    store.write(list(fichas), interim_dir / FICHAS_FILE_NAME, model_cls=Ficha)
    store.write([], interim_dir / EXCLUSIONS_FILE_NAME, model_cls=Exclusion)
    manifest = FichasManifest(
        snapshot_id=SNAPSHOT_ID,
        created_at=RUN_TIME,
        metadata_sha256="a" * 64,
        objectives_table_sha256="b" * 64,
        fichas_file=FICHAS_FILE_NAME,
        fichas_sha256=sha256(interim_dir / FICHAS_FILE_NAME),
        exclusions_file=EXCLUSIONS_FILE_NAME,
        exclusions_sha256=sha256(interim_dir / EXCLUSIONS_FILE_NAME),
        counts={},
        seconds=0.1,
    )
    (interim_dir / FICHAS_MANIFEST_FILE_NAME).write_bytes(manifest.model_dump_json().encode())


@dataclass(frozen=True)
class _WordCounter:
    """Counts two special tokens plus one per word."""

    def count(self, text: str) -> int:
        return len(text.split()) + 2


def write_chunk_outputs(interim_dir: Path, fichas: Sequence[Ficha]) -> list[ChunkRecord]:
    """Chunk ``fichas`` and save the table and its manifest, as the chunk build does."""
    records = list(chunk_fichas(fichas, _WordCounter(), 128))
    ParquetTableStore().write(records, interim_dir / CHUNKS_FILE_NAME, model_cls=ChunkRecord)
    manifest = ChunksManifest(
        snapshot_id=SNAPSHOT_ID,
        created_at=RUN_TIME,
        chunker_version=CHUNKER_VERSION,
        fichas_sha256=sha256(interim_dir / FICHAS_FILE_NAME),
        max_tokens=128,
        tokenizers=(),
        chunks_file=CHUNKS_FILE_NAME,
        chunks_sha256=sha256(interim_dir / CHUNKS_FILE_NAME),
        counts={},
        cut_kinds={},
        sections_used={},
        sections_recovered=0,
        seconds=0.1,
    )
    (interim_dir / CHUNKS_MANIFEST_FILE_NAME).write_bytes(manifest.model_dump_json().encode())
    return records


def _vector(text: str, dimension: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(dimension).astype(np.float32)


@dataclass
class _Encoder:
    identity: EncoderIdentity

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([_vector(text, self.identity.dimension) for text in texts])


@dataclass
class _Cleaner:
    identity: CleanerIdentity

    def clean(self, texts: Sequence[str]) -> list[list[str]]:
        words = ((word.strip(".,;:()") for word in text.casefold().split()) for text in texts)
        return [[word for word in text if len(word) > 3] for text in words]


def build_representations(root: Path, config: AppConfig, fichas: Sequence[Ficha]) -> AppConfig:
    """Embed and vectorize the included ``fichas`` with the real builds and stand-in models;
    return the configuration they ran with (their counts declared, and TF-IDF's ``min_df`` 1)."""
    interim_dir = interim(root)
    chunks = write_chunk_outputs(interim_dir, fichas)
    included = sum(f.include for f in fichas)
    dimensions = dict(zip(config.embedding.models, DIMENSIONS, strict=True))
    built = config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={
                    "expected": EmbeddingCountsConfig(
                        snapshot_id=SNAPSHOT_ID, documents=included, chunks=len(chunks)
                    ),
                    "threads": 1,
                }
            ),
            "tfidf": config.tfidf.model_copy(
                update={
                    "min_df": 1,
                    "expected": TfidfCountsConfig(
                        snapshot_id=SNAPSHOT_ID, documents=1, sections=1, terms=1, nonzeros=1
                    ),
                }
            ),
        }
    )
    build_embeddings(
        built,
        root,
        snapshot_id=SNAPSHOT_ID,
        load_encoder=lambda model, settings: _Encoder(
            EncoderIdentity(model, "0" * 40, 128, dimensions[model])
        ),
        clock=lambda: RUN_TIME,
        timer=lambda: 0.0,
    )
    (root / DOMAIN_STOPWORDS_FILE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / DOMAIN_STOPWORDS_FILE, root / DOMAIN_STOPWORDS_FILE)
    identity = CleanerIdentity(
        spacy_version="3.8.0",
        model_name="xx_stand_in",
        model_version="0.0.1",
        spacy_stopwords=0,
        spacy_stopwords_sha256=hashlib.sha256(b"").hexdigest(),
    )
    # The counts are those of these synthetic theses, not declared ahead: the count check of
    # the build is not what the loaders' tests are about.
    with mock.patch.object(tfidf_module, "check_tfidf"):
        build_tfidf(
            built,
            root,
            snapshot_id=SNAPSHOT_ID,
            load_cleaner=lambda domain: _Cleaner(identity),
            clock=lambda: RUN_TIME,
            timer=lambda: 0.0,
        )
    return built


# A drawn labeled set, for the workbooks (T20b)

AUTHOR = "Inventadoapellido Ficticiosegundo, Personaejemplo Sintética"
ADVISOR = "Supuestoapellido Imaginario, Asesorejemplo"
ADVISOR_ORCID = "https://orcid.org/0000-0002-1825-0097"
PER_PROGRAM = 8
WITHOUT_OBJECTIVES = frozenset({1, 9, 23})
"""Theses with metadata only: no objectives. Thesis 1 is in the first pilot pair."""
WITHOUT_KEYWORDS = frozenset({2, 30})
WITHOUT_ABSTRACT = frozenset({31})
PILOT_POOLS = (
    ("sbert_high", 0, 2),
    ("sbert_high", 1, 1),
    ("tfidf_high", None, 2),
    ("mid_range", None, 3),
    ("random", None, 2),
)
LABELED_POOLS = (
    ("sbert_high", 0, 20),
    ("sbert_high", 1, 20),
    ("tfidf_high", None, 20),
    ("mid_range", None, 30),
    ("random", None, 30),
)
"""``(pool, index of the model in embedding.models or None, pairs)`` of each step, as D34 (5)
and (7) draw them."""


def labeling_theses(config: AppConfig) -> list[Thesis]:
    """Return 8 theses of each configured program, some without objectives, keywords or an
    abstract, all with the same synthetic author and advisor; and one excluded thesis."""
    programs = [program.key for program in config.snapshot.programs]
    theses = []
    for index in range(PER_PROGRAM * len(programs)):
        number = index + 1
        theses.append(
            Thesis(
                number=number,
                program=programs[index % len(programs)],
                title=f"Título sintético {number} sobre una planta de prueba",
                abstract=None
                if number in WITHOUT_ABSTRACT
                else f"Resumen sintético {number}. Se estudia un proceso inventado.",
                objectives=None
                if number in WITHOUT_OBJECTIVES
                else f"Objetivo general sintético {number}.\n\nObjetivo específico {number}.",
                keywords=() if number in WITHOUT_KEYWORDS else (f"clave {number}", "prueba"),
                authors=(AUTHOR,),
                advisor=ADVISOR,
                advisor_orcid=ADVISOR_ORCID,
            )
        )
    theses.append(
        Thesis(99, programs[0], "Tesis excluida sintética", "Excluida.", "Excluir.", include=False)
    )
    return theses


def labeling_draw(config: AppConfig, theses: Sequence[Thesis]) -> tuple[PairRow, ...]:
    """Return a draw of 10 pilot and 120 labeled pairs of the included ``theses``, in draw order,
    as T20a records it; the first pilot pair holds thesis 1, which has no objectives."""
    included = [thesis for thesis in theses if thesis.include]
    program = {thesis.doc_code: thesis.program for thesis in included}
    first = (included[0].doc_code, included[5].doc_code)
    candidates = [
        (a.doc_code, b.doc_code)
        for a, b in itertools.combinations(included, 2)
        if {a.doc_code, b.doc_code} != set(first)
    ]
    chosen = [first, *random.Random(34).sample(candidates, 129)]
    models = config.embedding.models
    steps = [(PILOT, *pool) for pool in PILOT_POOLS] + [(LABELED, *pool) for pool in LABELED_POOLS]
    rows = []
    for phase, pool, model, size in steps:
        for _ in range(size):
            low, high = sorted(chosen[len(rows)])
            same = program[low] == program[high]
            rows.append(
                PairRow(
                    pair_id=pair_id(phase, low, high),
                    phase=phase,
                    draw_order=len(rows) + 1,
                    pool=pool,
                    source=None if model is None else models[model],
                    stratum=program[low] if same else BETWEEN,
                    doc_code_a=low,
                    doc_code_b=high,
                    program_a=program[low],
                    program_b=program[high],
                    representations=(*models, "tfidf"),
                    scores=(0.81, 0.79, 0.25),
                    ranks=(0.995, 0.993, 0.991),
                    mean_rank=0.993,
                    known_duplicate=False,
                )
            )
    return tuple(rows)


def write_draw(root: Path, config: AppConfig, pairs: Sequence[PairRow]) -> None:
    """Write ``pairs`` as T20a's draw of :data:`SNAPSHOT_ID`: the gitignored ``pairs.parquet``
    and the tracked manifest that records its SHA-256."""
    records = records_dir(config, root, SNAPSHOT_ID)
    ParquetTableStore().write(list(pairs), records / "pairs.parquet", model_cls=PairRow)
    manifest = DrawManifest(
        snapshot_id=SNAPSHOT_ID,
        git_commit="d" * 40,
        seed=config.labeling.seed,
        settings={},
        representations=(*config.embedding.models, "tfidf"),
        inputs={},
        counts={"pilot": 10, "labeled": 120},
        versions={},
        cpu={},
        files={},
        data_files={"pairs.parquet": sha256(records / "pairs.parquet")},
        started_at=RUN_TIME,
        finished_at=RUN_TIME,
        seconds=1.0,
        timings={},
    )
    results = results_dir(root / "results", SNAPSHOT_ID)
    results.mkdir(parents=True)
    (results / "manifest.json").write_text(manifest.model_dump_json(indent=2), "utf-8")


@dataclass(frozen=True)
class LabelingProject:
    """A project root with the configuration, the snapshot, the fichas and T20a's draw."""

    root: Path
    theses: tuple[Thesis, ...]
    fichas: dict[str, Ficha]
    """The included fichas, by doc code."""
    records: tuple[SnapshotRecord, ...]
    pairs: tuple[PairRow, ...]


def write_labeling_project(
    root: Path, config: AppConfig, theses: Sequence[Thesis] | None = None
) -> LabelingProject:
    """Write the inputs of the workbooks under ``root``, whose configuration is ``config``."""
    theses = list(theses or labeling_theses(config))
    names = {program.key: program.name for program in config.snapshot.programs}
    fichas = [ficha(thesis, names) for thesis in theses]
    records = [snapshot_record(thesis, config) for thesis in theses]
    write_test_snapshot(root, records, config)
    write_fichas_outputs(interim(root), fichas)
    pairs = labeling_draw(config, theses)
    write_draw(root, config, pairs)
    return LabelingProject(
        root=root,
        theses=tuple(theses),
        fichas={f.doc_code: f for f in fichas if f.include},
        records=tuple(records),
        pairs=pairs,
    )
