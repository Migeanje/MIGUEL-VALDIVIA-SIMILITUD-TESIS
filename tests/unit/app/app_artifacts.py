"""Synthetic saved artifacts for the app's tests (T28), written under pytest's temporary
directory, outside the repository.

The project holds the repository's ``config/default.yaml`` and, for snapshot ``S1``:

- the fichas of 14 included theses of three programs (Sistemas, Industrial and Mecánica,
  2021-2026) and one duplicate that the corpus excludes, each with author and advisor codes,
  and a planted person name in its abstract and keywords that the app must never show;
- T19's records: the winner C4's map run (seed 42) with the 2-D coordinates of every thesis,
  its three topics (0 with 5 theses, 1 with 4, 2 with 3) and 2 noise theses, plus a topic of
  another run whose terms the app must not show;
- T24a's records: five stable gaps of the reference run, one per rule and a second of rule (c)
  (affinity 2 and 1), one unstable gap of the reference run and one gap of another run, which
  the app must leave out, with the members of each gap.

Every file is written as its own command writes it, with the SHA-256 that its manifest records.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

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
from thematic_redundancy.gaps.results_store import GapManifest, GapRow, MemberRow
from thematic_redundancy.shared.config import load_config
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

    def argv(self) -> list[str]:
        return ["main.py", "--project-root", str(self.root), "--snapshot-id", SNAPSHOT]


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
    topics = [
        TopicRow(
            run_id=MAP_RUN,
            topic=topic,
            size=sum(thesis.topic == topic for thesis in THESES),
            terms=terms,
            scores=tuple(0.5 - 0.01 * index for index in range(len(terms))),
            npmi=0.2,
            npmi_sensitivity=0.1,
            absent_terms=(),
        )
        for topic, terms in TOPIC_TERMS.items()
    ]
    topics.append(
        TopicRow(
            run_id=OTHER_RUN,
            topic=0,
            size=len(THESES),
            terms=OTHER_RUN_TERMS,
            scores=(0.5, 0.4),
            npmi=0.1,
            npmi_sensitivity=0.1,
            absent_terms=(),
        )
    )
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
            "embeddings_manifest.json": "e" * 64,
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


EVIDENCE_COLUMNS = (
    "size_cut",
    "npmi",
    "npmi_floor",
    "centroid_cosine",
    "neighbor_rank",
    "bridge_theses",
    "source",
    "target",
    "affinity",
    "source_count",
    "source_theses",
    "source_share",
    "target_count",
    "target_theses",
    "target_share",
    "early_count",
    "early_theses",
    "early_share",
    "late_count",
    "late_theses",
    "late_share",
    "decline",
)


def gap_row(
    gap_id: str, rule: str, topics: tuple[int, ...], *, run: str = MAP_RUN, **values: Any
) -> GapRow:
    members = [thesis for thesis in THESES if thesis.topic in topics]
    programs = sorted({thesis.program for thesis in members})
    years = sorted({thesis.year for thesis in members})
    fields: dict[str, Any] = dict.fromkeys(EVIDENCE_COLUMNS)
    fields.update(
        support=None,
        stable=None,
        reference=run == MAP_RUN,
        reason="English reason that the app never shows",
    )
    fields.update(values)
    return GapRow(
        gap_id=f"{run}__{gap_id}",
        run_id=run,
        seed=SEEDS[RUNS.index(run)],
        rule=rule,
        topics=topics,
        theses=len(members),
        programs=tuple(programs),
        program_counts=tuple(sum(t.program == p for t in members) for p in programs),
        years=tuple(years),
        year_counts=tuple(sum(t.year == y for t in members) for y in years),
        **fields,
    )


GAPS = (
    gap_row("a_t2", "a", (2,), size_cut=3.0, npmi=0.2, npmi_floor=0.15, support=5, stable=True),
    gap_row(
        "b_t0-t1",
        "b",
        (0, 1),
        centroid_cosine=0.61,
        neighbor_rank=2,
        bridge_theses=1,
        support=4,
        stable=True,
    ),
    gap_row(
        "c_t0_industrial-mecanica",
        "c",
        (0,),
        source="industrial",
        target="mecanica",
        affinity=2,
        source_count=4,
        source_theses=6,
        source_share=4 / 6,
        target_count=0,
        target_theses=3,
        target_share=0.0,
        support=5,
        stable=True,
    ),
    gap_row(
        "c_t1_sistemas-mecanica",
        "c",
        (1,),
        source="sistemas",
        target="mecanica",
        affinity=1,
        source_count=3,
        source_theses=5,
        source_share=0.6,
        target_count=0,
        target_theses=3,
        target_share=0.0,
        support=4,
        stable=True,
    ),
    gap_row(
        "d_t0",
        "d",
        (0,),
        early_count=4,
        early_theses=7,
        early_share=4 / 7,
        late_count=1,
        late_theses=7,
        late_share=1 / 7,
        decline=3 / 7,
        support=5,
        stable=True,
    ),
    gap_row(
        "c_t2_mecanica-industrial",
        "c",
        (2,),
        source="mecanica",
        target="industrial",
        affinity=2,
        source_count=2,
        source_theses=3,
        source_share=2 / 3,
        target_count=0,
        target_theses=6,
        target_share=0.0,
        support=2,
        stable=False,
    ),
    gap_row("a_t0", "a", (0,), run=OTHER_RUN, size_cut=3.0, npmi=0.3, npmi_floor=0.15),
)
STABLE_GAPS = GAPS[:5]
BRIDGE = THESES[12]
"""The bridge thesis of the rule (b) gap: a noise thesis."""


def member_rows() -> list[MemberRow]:
    rows = []
    for gap in GAPS:
        if gap.rule == "b":
            parts = [("first_topic", 0), ("second_topic", 1)]
        else:
            parts = [("topic", gap.topics[0])]
        for part, topic in parts:
            rows += [
                MemberRow(gap_id=gap.gap_id, run_id=gap.run_id, doc_code=t.doc_code, part=part)
                for t in THESES
                if t.topic == topic
            ]
        if gap.rule == "b":
            rows.append(
                MemberRow(
                    gap_id=gap.gap_id, run_id=gap.run_id, doc_code=BRIDGE.doc_code, part="bridge"
                )
            )
    return rows


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


def write_gaps(project: Project) -> None:
    store = ParquetTableStore()
    results, processed = project.gaps_results, project.gaps_processed
    results.mkdir(parents=True)
    processed.mkdir(parents=True)
    store.write(list(GAPS), results / gap_store.GAPS_FILE_NAME)
    store.write(member_rows(), processed / gap_store.MEMBERS_FILE_NAME)
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
    write_experiment(project)
    write_gaps(project)
    return project
