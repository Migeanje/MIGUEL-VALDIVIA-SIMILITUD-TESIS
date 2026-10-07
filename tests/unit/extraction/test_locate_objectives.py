"""Behavior of the objectives locator's use case and command line, on synthetic page files.

Each test builds a small project: the default configuration, a snapshot of synthetic
records, and page files with a text manifest as the text extraction writes them. Every
objective holds the marker ``SYNTHETICOBJECTIVE``, which must never reach the console or
the summary file.
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.extraction.locate_objectives import (
    ObjectivesManifest,
    locate_all,
    main,
)
from thematic_redundancy.extraction.objectives import LOCATOR_VERSION, ObjectivesRecord
from thematic_redundancy.extraction.page_text import PageText
from thematic_redundancy.extraction.text_manifest import (
    ExtractionSettings,
    TextEntry,
    TextManifest,
    summarize,
    write_text_manifest,
)
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.storage import ParquetTableStore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
DECLARED = load_config(DEFAULT_CONFIG_PATH)
COLLECTIONS = {program.key: program.collection_uuid for program in DECLARED.snapshot.programs}
SNAPSHOT_ID = "20261002T150405Z"
HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
RUN_TIME = datetime(2026, 10, 7, 9, 0, 0, tzinfo=UTC)
MARKER = "SYNTHETICOBJECTIVE"

WITH_OBJECTIVES = [
    "CAPÍTULO I\n\nPLANTEAMIENTO DEL PROBLEMA",
    "1.3.1. Objetivo general\n\n"
    f"Determinar el efecto {MARKER} de la temperatura en el proceso de prueba.\n\n"
    "1.3.2. Objetivos específicos\n\n"
    f"• Medir la variable {MARKER} principal.\n\n1.4. Justificación",
]
WITHOUT_OBJECTIVES = ["Introducción\n\nTexto sintético sin objetivos.", "Conclusiones"]
ABSTRACT_ONLY = [
    "RESUMEN\n\n"
    f"Objetivo general: Evaluar el rendimiento {MARKER} del equipo de prueba.\n\n"
    "Palabras clave: prueba.",
    "Introducción\n\nTexto sintético.",
]


def item_uuid(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def handle(number: int) -> str:
    return f"123456789/{1000 + number}"


def page_records(texts: Sequence[str]) -> list[PageText]:
    records = []
    for index, raw in enumerate(texts):
        text = raw.strip()
        records.append(
            PageText(
                page_index=index,
                text=text,
                source="text_layer" if text else "empty",
                char_count=len(text),
                low_text=len(text) < 50,
                ocr_attempted=False,
                ocr_failed=False,
                lines_removed=0,
            )
        )
    return records


def pages_dir(root: Path) -> Path:
    return root / "data" / "interim" / SNAPSHOT_ID / "pages"


def write_project(
    root: Path, documents: dict[int, tuple[str, list[str] | None]], *, text_manifest: bool = True
) -> None:
    """Write a project whose theses have the given program and page texts; ``None`` stands
    for a PDF whose extraction failed."""
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "default.yaml").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    records = [
        SnapshotRecord(
            uuid=item_uuid(number),
            handle=handle(number),
            program_key=program,
            collection_uuid=COLLECTIONS[program],
            harvested_at=HARVEST_TIME,
            metadata={
                "dc.title": [{"value": f"Título sintético {number}"}],
                "dc.description.abstract": [{"value": f"Resumen sintético {number}"}],
                "dc.contributor.author": [{"value": "Apellido Sintético, Nombre"}],
            },
        )
        for number, (program, _) in documents.items()
    ]
    content = encode_records(records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.program_key] = counts.get(record.program_key, 0) + 1
    write_snapshot(
        root / "data" / "raw" / SNAPSHOT_ID,
        content,
        SnapshotManifest(
            snapshot_id=SNAPSHOT_ID,
            harvested_at=HARVEST_TIME,
            base_url="https://repo.example.edu",
            search_url="https://repo.example.edu/server/api/discover/search/objects",
            query_parameters={},
            programs={
                key: ProgramSummary(collection_uuid=COLLECTIONS[key], items=count)
                for key, count in counts.items()
            },
            total_items=len(records),
            faculty_community_uuid=UUID("00000000-0000-4000-8000-0000000000f0"),
            faculty_total=len(records),
            items_by_type={},
            future_dated_items=0,
            unparsed_issue_dates=0,
            dropped_metadata_keys=(),
            metadata_sha256=hashlib.sha256(content).hexdigest(),
        ),
    )
    if not text_manifest:
        return
    settings = ExtractionSettings.from_config(DECLARED)
    fingerprint = settings.fingerprint()
    store = ParquetTableStore()
    entries: dict[UUID, TextEntry] = {}
    for number, (program, texts) in documents.items():
        found = {
            "program_key": program,
            "pdf_sha256": f"{number:064x}",
            "config_fingerprint": fingerprint,
            "extracted_at": RUN_TIME,
        }
        if texts is None:
            entries[item_uuid(number)] = TextEntry(
                **found, status="error", reason="unreadable_pdf", detail="synthetic failure"
            )
            continue
        path = pages_dir(root) / f"{item_uuid(number)}.parquet"
        store.write(page_records(texts), path, model_cls=PageText)
        entries[item_uuid(number)] = TextEntry(
            **found,
            status="ok",
            file=path.name,
            output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pages=len(texts),
            text_layer_pages=sum(bool(text.strip()) for text in texts),
            empty_pages=sum(not text.strip() for text in texts),
        )
    pages_dir(root).mkdir(parents=True, exist_ok=True)
    write_text_manifest(
        pages_dir(root) / "manifest.json",
        TextManifest(
            snapshot_id=SNAPSHOT_ID,
            settings=settings,
            config_fingerprint=fingerprint,
            updated_at=RUN_TIME,
            summary=summarize(entries, fingerprint, entries.keys()),
            items=entries,
        ),
    )


THREE = {
    1: ("sistemas", WITH_OBJECTIVES),
    2: ("industrial", WITHOUT_OBJECTIVES),
    3: ("minas", ABSTRACT_ONLY),
}


class FakeTimer:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


def table(root: Path) -> tuple[ObjectivesRecord, ...]:
    return ParquetTableStore().read(
        ObjectivesRecord, root / "data" / "interim" / SNAPSHOT_ID / "objectives.parquet"
    )


def test_the_command_writes_one_row_per_page_file_and_a_numbers_only_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path, THREE)
    summary_path = tmp_path / "results" / "objectives" / SNAPSHOT_ID / "summary.json"

    code = main(["--project-root", str(tmp_path), "--summary-out", str(summary_path)])

    assert code == 0
    rows = table(tmp_path)
    assert [(row.item_uuid, row.handle, row.program, row.status) for row in rows] == [
        (item_uuid(1), handle(1), "sistemas", "extracted"),
        (item_uuid(2), handle(2), "industrial", "not_found"),
        (item_uuid(3), handle(3), "minas", "extracted"),
    ]
    first = rows[0]
    assert first.objective_general is not None and MARKER in first.objective_general
    assert (first.page_start, first.page_end, first.pattern) == (1, 1, "objetivo_general")
    assert first.general_chars == len(first.objective_general) and first.pages == 2
    assert (
        first.pages_sha256
        == hashlib.sha256(
            (pages_dir(tmp_path) / f"{item_uuid(1)}.parquet").read_bytes()
        ).hexdigest()
    )
    assert rows[2].flags == ("specific_missing", "abstract_only")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["settings"]["locator_version"] == LOCATOR_VERSION
    counts = summary["summary"]
    assert counts["documents"] == 3
    assert counts["by_status"] == {"extracted": 2, "not_found": 1}
    assert counts["by_program"] == {
        "sistemas": {"extracted": 1, "not_found": 0},
        "industrial": {"extracted": 0, "not_found": 1},
        "minas": {"extracted": 1, "not_found": 0},
    }
    assert counts["by_pattern"]["objetivo_general"] == 2
    assert counts["flags"]["abstract_only"] == 1 and counts["flags"]["ocr_page"] == 0
    assert summary["seconds"] >= 0
    output = capsys.readouterr()
    for text in (summary_path.read_text(encoding="utf-8"), output.out, output.err):
        assert MARKER not in text
    assert "extracted 2" in output.out


def test_the_use_case_records_its_settings_time_and_table_digest(tmp_path: Path) -> None:
    write_project(tmp_path, THREE)

    report = locate_all(DECLARED, tmp_path, timer=FakeTimer(), clock=lambda: RUN_TIME)

    saved = ObjectivesManifest.model_validate_json(report.manifest_path.read_bytes())
    assert saved == report.manifest
    assert saved.seconds == pytest.approx(0.5)
    assert saved.created_at == RUN_TIME
    assert saved.table_sha256 == hashlib.sha256(report.table_path.read_bytes()).hexdigest()
    assert saved.settings.general_max_chars == DECLARED.objectives.general_max_chars
    assert len(saved.settings_fingerprint) == 64


def test_a_pdf_whose_extraction_failed_is_counted_but_left_out(tmp_path: Path) -> None:
    write_project(tmp_path, THREE | {4: ("mecanica", None)})

    report = locate_all(DECLARED, tmp_path)

    assert [row.item_uuid for row in table(tmp_path)] == [item_uuid(n) for n in (1, 2, 3)]
    assert (report.manifest.summary.documents, report.manifest.summary.text_errors) == (3, 1)


def test_a_page_file_changed_since_its_extraction_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path, THREE)
    ParquetTableStore().write(
        page_records(["Otro texto"]), pages_dir(tmp_path) / f"{item_uuid(2)}.parquet"
    )

    code = main(["--project-root", str(tmp_path)])

    assert code == 1
    assert "changed since its extraction" in capsys.readouterr().err
    assert not (tmp_path / "data" / "interim" / SNAPSHOT_ID / "objectives.parquet").exists()


def test_without_page_texts_the_command_asks_for_the_text_extraction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path, THREE, text_manifest=False)

    code = main(["--project-root", str(tmp_path)])

    assert code == 1
    assert "thematic_redundancy.extraction.extract_text" in capsys.readouterr().err


def test_make_sample_writes_the_workbook_once_and_never_overwrites_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path, THREE)
    workbook = (
        tmp_path / "data" / "labels" / "objectives_check" / SNAPSHOT_ID / "objectives_check.xlsx"
    )

    assert main(["--project-root", str(tmp_path), "--make-sample"]) == 0
    before = workbook.read_bytes()
    sample = json.loads((workbook.parent / "sample.json").read_text(encoding="utf-8"))
    assert len(sample["items"]) == 3
    assert sample["provenance"]["locator_version"] == LOCATOR_VERSION
    assert "Verification sample: 3 theses" in capsys.readouterr().out

    assert main(["--project-root", str(tmp_path), "--make-sample"]) == 1

    assert "already exists" in capsys.readouterr().err
    assert workbook.read_bytes() == before
