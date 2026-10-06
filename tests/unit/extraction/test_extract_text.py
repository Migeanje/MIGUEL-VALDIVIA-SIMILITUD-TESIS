"""Behavior of the PDF text extraction: the use case, its manifest, and its command line.

The PDFs are synthetic (see ``synthetic_pdfs``) and are read by the real PyMuPDF adapter,
while OCR goes through a fake engine that answers instantly and moves a fake clock forward,
so the recorded times are exact. No test reads a thesis or needs language files.
"""

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pymupdf
import pytest
from synthetic_pdfs import (
    MARKER,
    ImagePage,
    PageSpec,
    body_lines,
    thesis_page,
    write_corrupt_file,
    write_pdf,
)

from thematic_redundancy.corpus.pdf_manifest import PdfEntry, PdfManifest, write_pdf_manifest
from thematic_redundancy.corpus.pdf_manifest import summarize as summarize_pdfs
from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.extraction.extract_text import (
    ExtractionInProgressError,
    ExtractionReport,
    extract_texts,
    main,
)
from thematic_redundancy.extraction.page_text import PageText
from thematic_redundancy.extraction.pdf_reader import PdfPages, PdfReadError, TextLayer
from thematic_redundancy.extraction.pymupdf_reader import PyMuPdfReader
from thematic_redundancy.extraction.text_manifest import ExtractionSettings
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.shared.storage import ParquetTableStore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
DECLARED = load_config(DEFAULT_CONFIG_PATH)
COLLECTIONS = {program.key: program.collection_uuid for program in DECLARED.snapshot.programs}

SNAPSHOT_ID = "20261002T150405Z"
HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
RUN_TIME = datetime(2026, 10, 4, 9, 0, 0, tzinfo=UTC)

OCR_TEXT = f"{MARKER} texto leido por OCR en una pagina escaneada"
OCR_SECONDS = 2.0
"""Time that every fake OCR call takes on the fake clock."""


def item_uuid(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeClock:
    """Timer that stands still, except when the fake OCR moves it forward."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeOcr:
    """OCR engine that answers :data:`OCR_TEXT`, taking :data:`OCR_SECONDS` on the fake
    clock. ``hook`` runs before each answer with the item uuid and the page number."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.fail = False
        self.hook: Callable[[UUID, int], None] | None = None
        self.calls: list[tuple[UUID, int]] = []

    def __call__(self, page: pymupdf.Page) -> str:
        item = UUID(Path(page.parent.name).stem)
        self.calls.append((item, page.number))
        if self.hook is not None:
            self.hook(item, page.number)
        self.clock.now += OCR_SECONDS
        if self.fail:
            raise RuntimeError("synthetic OCR failure")
        return OCR_TEXT


class SpyReader:
    """Real PyMuPDF reader that records which PDFs it opens, and can fail one page."""

    def __init__(self, ocr: FakeOcr) -> None:
        self._reader = PyMuPdfReader(Path("tessdata"), "spa+eng", 300, ocr_engine=ocr)
        self.opened: list[UUID] = []
        self.unreadable_page: tuple[UUID, int] | None = None

    @contextmanager
    def open(self, path: Path) -> Iterator[PdfPages]:
        item = UUID(path.stem)
        self.opened.append(item)
        with self._reader.open(path) as pages:
            yield _FailingPage(pages, item, self.unreadable_page)


class _FailingPage:
    def __init__(self, pages: PdfPages, item: UUID, unreadable: tuple[UUID, int] | None):
        self._pages = pages
        self._item = item
        self._unreadable = unreadable

    @property
    def page_count(self) -> int:
        return self._pages.page_count

    def text_layer(self, index: int) -> TextLayer:
        if self._unreadable == (self._item, index):
            raise PdfReadError(f"cannot read page {index}: synthetic failure")
        return self._pages.text_layer(index)

    def ocr_text(self, index: int) -> str:
        return self._pages.ocr_text(index)

    def warning_count(self) -> int:
        return self._pages.warning_count()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ocr(clock: FakeClock) -> FakeOcr:
    return FakeOcr(clock)


@pytest.fixture
def reader(ocr: FakeOcr) -> SpyReader:
    return SpyReader(ocr)


@pytest.fixture
def config() -> AppConfig:
    return DECLARED


def with_extraction(config: AppConfig, **changes: Any) -> AppConfig:
    """Return ``config`` with some extraction settings, or header/footer settings, changed."""
    header_footer = config.extraction.header_footer.model_copy(
        update={key: value for key, value in changes.items() if key in {"min_share"}}
    )
    extraction = config.extraction.model_copy(
        update={key: value for key, value in changes.items() if key not in {"min_share"}}
        | {"header_footer": header_footer}
    )
    return config.model_copy(update={"extraction": extraction})


def raw_dir(project_root: Path, snapshot_id: str = SNAPSHOT_ID) -> Path:
    return project_root / "data" / "raw" / snapshot_id


def pages_dir(project_root: Path, snapshot_id: str = SNAPSHOT_ID) -> Path:
    return project_root / "data" / "interim" / snapshot_id / "pages"


def pdf_path(project_root: Path, number: int) -> Path:
    return raw_dir(project_root) / "pdfs" / f"{item_uuid(number)}.pdf"


def pages_path(project_root: Path, number: int) -> Path:
    return pages_dir(project_root) / f"{item_uuid(number)}.parquet"


def snapshot_record(number: int, program: str) -> SnapshotRecord:
    return SnapshotRecord(
        uuid=item_uuid(number),
        handle=f"123456789/{1000 + number}",
        program_key=program,
        collection_uuid=COLLECTIONS[program],
        harvested_at=HARVEST_TIME,
        metadata={},
    )


def write_test_snapshot(
    project_root: Path,
    pdfs: dict[int, Sequence[PageSpec] | bytes | None],
    *,
    programs: dict[int, str] | None = None,
    snapshot_id: str = SNAPSHOT_ID,
) -> None:
    """Write a snapshot whose theses have the given PDFs, and its PDF manifest.

    A value of ``None`` stands for a restricted thesis without a PDF, and ``bytes`` for a
    file written as is, such as a damaged one.
    """
    programs = programs or {}
    records = [snapshot_record(number, programs.get(number, "sistemas")) for number in pdfs]
    content = encode_records(records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.program_key] = counts.get(record.program_key, 0) + 1
    write_snapshot(
        raw_dir(project_root, snapshot_id),
        content,
        SnapshotManifest(
            snapshot_id=snapshot_id,
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
    entries: dict[UUID, PdfEntry] = {}
    for record in records:
        number = int(record.uuid.hex[-12:])
        spec = pdfs[number]
        target = raw_dir(project_root, snapshot_id) / "pdfs" / f"{record.uuid}.pdf"
        if spec is None:
            entries[record.uuid] = PdfEntry(
                program_key=record.program_key,
                status="restricted",
                reason="rights_restricted",
                checked_at=RUN_TIME,
            )
            continue
        if isinstance(spec, bytes):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(spec)
        else:
            write_pdf(target, spec)
        entries[record.uuid] = pdf_entry(record.program_key, target)
    write_pdf_manifest(
        raw_dir(project_root, snapshot_id) / "pdfs" / "manifest.json",
        PdfManifest(
            snapshot_id=snapshot_id,
            base_url="https://repo.example.edu",
            thesis_type="tesis",
            selection_rule="synthetic",
            updated_at=RUN_TIME,
            summary=summarize_pdfs(entries),
            items=entries,
        ),
    )


def pdf_entry(program: str, target: Path) -> PdfEntry:
    content = target.read_bytes()
    return PdfEntry(
        program_key=program,
        status="already_present",
        file=target.name,
        bitstream_uuid=UUID(int=len(content)),
        size_bytes=len(content),
        checksum_algorithm="MD5",
        checksum=hashlib.md5(content).hexdigest(),
        md5=hashlib.md5(content).hexdigest(),
        sha256=hashlib.sha256(content).hexdigest(),
        checked_at=RUN_TIME,
    )


def rewrite_pdf(project_root: Path, number: int, spec: Sequence[PageSpec]) -> None:
    """Replace the PDF of thesis ``number`` and its PDF manifest entry, as a new download."""
    target = pdf_path(project_root, number)
    write_pdf(target, spec)
    manifest_path = raw_dir(project_root) / "pdfs" / "manifest.json"
    manifest = PdfManifest.model_validate_json(manifest_path.read_bytes())
    items = dict(manifest.items)
    items[item_uuid(number)] = pdf_entry(items[item_uuid(number)].program_key, target)
    write_pdf_manifest(manifest_path, manifest.model_copy(update={"items": items}))


def thesis(pages: int = 6) -> list[PageSpec]:
    return [thesis_page(number) for number in range(1, pages + 1)]


def run(
    config: AppConfig, project_root: Path, reader: SpyReader, clock: FakeClock, **options: Any
) -> ExtractionReport:
    options.setdefault("snapshot_id", SNAPSHOT_ID)
    return extract_texts(
        config, project_root, reader, clock=lambda: RUN_TIME, timer=clock, **options
    )


def stored_pages(project_root: Path, number: int) -> tuple[PageText, ...]:
    return ParquetTableStore().read(PageText, pages_path(project_root, number))


def text_manifest(project_root: Path) -> dict[str, Any]:
    return json.loads((pages_dir(project_root) / "manifest.json").read_text(encoding="utf-8"))


def entry_of(project_root: Path, number: int) -> dict[str, Any]:
    return text_manifest(project_root)["items"][str(item_uuid(number))]


def outcomes(report: ExtractionReport) -> list[tuple[UUID, str, bool]]:
    return [
        (outcome.item_uuid, outcome.entry.status, outcome.skipped) for outcome in report.outcomes
    ]


def leftovers(project_root: Path) -> list[str]:
    return sorted(path.name for path in pages_dir(project_root).glob("*.part"))


def lines_of(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


# --- One PDF, page by page ---------------------------------------------------------------


def test_each_pdf_gets_its_page_records_without_running_lines(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: [*thesis(6), ImagePage]})

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", False)]
    pages = stored_pages(tmp_path, 1)
    assert [page.page_index for page in pages] == list(range(7))
    assert lines_of(pages[0].text) == body_lines(1)
    assert lines_of(pages[5].text) == body_lines(6)
    assert [page.source for page in pages] == ["text_layer"] * 6 + ["ocr"]
    assert pages[6].text == OCR_TEXT
    assert [page.lines_removed for page in pages] == [3] * 6 + [0]
    assert [page.low_text for page in pages] == [False] * 6 + [True]
    assert [page.char_count for page in pages] == [len(page.text) for page in pages]


def test_the_manifest_records_counts_hashes_and_times_but_no_text(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: [*thesis(6), ImagePage, ImagePage]})

    run(config, tmp_path, reader, clock)

    entry = entry_of(tmp_path, 1)
    fingerprint = ExtractionSettings.from_config(config).fingerprint()
    assert entry == {
        "program_key": "sistemas",
        "status": "ok",
        "reason": None,
        "detail": None,
        "pdf_sha256": sha256(pdf_path(tmp_path, 1)),
        "config_fingerprint": fingerprint,
        "file": f"{item_uuid(1)}.parquet",
        "output_sha256": sha256(pages_path(tmp_path, 1)),
        "pages": 8,
        "text_layer_pages": 6,
        "ocr_pages": 2,
        "empty_pages": 0,
        "low_text_pages": 2,
        "ocr_attempted": 2,
        "ocr_failed": 0,
        "unreadable_pages": 0,
        "lines_removed": 18,
        "pdf_warnings": 0,
        "seconds": 2 * OCR_SECONDS,
        "ocr_seconds": 2 * OCR_SECONDS,
        "extracted_at": "2026-10-04T09:00:00Z",
    }
    manifest = text_manifest(tmp_path)
    assert manifest["config_fingerprint"] == fingerprint
    assert manifest["settings"]["ocr_window_pages"] == 100
    assert manifest["summary"]["ocr_attempted"] == 2
    assert manifest["summary"]["by_program"] == {"sistemas": {"ok": 1}}
    assert MARKER not in json.dumps(manifest)


def test_low_text_pages_outside_the_ocr_window_are_not_ocrd(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, ocr: FakeOcr
) -> None:
    write_test_snapshot(tmp_path, {1: [ImagePage, *thesis(5), ImagePage]})

    run(with_extraction(config, ocr_window_pages=2), tmp_path, reader, clock)

    assert ocr.calls == [(item_uuid(1), 0)]
    pages = stored_pages(tmp_path, 1)
    assert (pages[0].source, pages[0].ocr_attempted) == ("ocr", True)
    assert (pages[6].source, pages[6].low_text, pages[6].ocr_attempted) == ("empty", True, False)
    assert entry_of(tmp_path, 1)["empty_pages"] == 1


def test_an_ocr_failure_leaves_the_page_without_ocr_and_the_pdf_ok(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, ocr: FakeOcr
) -> None:
    write_test_snapshot(tmp_path, {1: [*thesis(5), ImagePage]})
    ocr.fail = True

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", False)]
    page = stored_pages(tmp_path, 1)[5]
    assert (page.source, page.ocr_attempted, page.ocr_failed) == ("empty", True, True)
    entry = entry_of(tmp_path, 1)
    assert (entry["ocr_attempted"], entry["ocr_failed"], entry["ocr_pages"]) == (1, 1, 0)


def test_a_page_whose_text_layer_cannot_be_read_is_counted_and_ocrd(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(6)})
    reader.unreadable_page = (item_uuid(1), 2)

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", False)]
    assert stored_pages(tmp_path, 1)[2].source == "ocr"
    entry = entry_of(tmp_path, 1)
    assert (entry["unreadable_pages"], entry["ocr_attempted"]) == (1, 1)


# --- Resuming and redoing ----------------------------------------------------------------


def test_a_second_run_skips_every_pdf_already_extracted(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: thesis()})
    run(config, tmp_path, reader, clock)
    written = pages_path(tmp_path, 1).read_bytes()
    reader.opened.clear()

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", True), (item_uuid(2), "ok", True)]
    assert reader.opened == []
    assert pages_path(tmp_path, 1).read_bytes() == written


def test_changed_settings_extract_every_pdf_again(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: thesis()})
    run(config, tmp_path, reader, clock)
    reader.opened.clear()
    changed = with_extraction(config, min_share=0.9)

    report = run(changed, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", False), (item_uuid(2), "ok", False)]
    assert reader.opened == [item_uuid(1), item_uuid(2)]
    fingerprint = ExtractionSettings.from_config(changed).fingerprint()
    assert entry_of(tmp_path, 1)["config_fingerprint"] == fingerprint
    assert text_manifest(tmp_path)["summary"]["stale"] == 0


def test_a_new_download_of_a_pdf_is_extracted_again(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: thesis()})
    run(config, tmp_path, reader, clock)
    reader.opened.clear()
    rewrite_pdf(tmp_path, 2, thesis(7))

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", True), (item_uuid(2), "ok", False)]
    assert entry_of(tmp_path, 2)["pages"] == 7
    assert entry_of(tmp_path, 2)["pdf_sha256"] == sha256(pdf_path(tmp_path, 2))


@pytest.mark.parametrize("damage", ["delete", "change"])
def test_a_missing_or_changed_pages_file_is_written_again(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, damage: str
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: thesis()})
    run(config, tmp_path, reader, clock)
    reader.opened.clear()
    if damage == "delete":
        pages_path(tmp_path, 1).unlink()
    else:
        pages_path(tmp_path, 1).write_bytes(b"not the file that was written")

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", False), (item_uuid(2), "ok", True)]
    assert len(stored_pages(tmp_path, 1)) == 6


def test_a_pdf_that_no_longer_matches_the_pdf_manifest_is_an_error(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: thesis()})
    run(config, tmp_path, reader, clock)
    rewrite_pdf(tmp_path, 1, thesis(7))
    write_pdf(pdf_path(tmp_path, 1), thesis(8))  # Changed after its download was recorded.

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "error", False), (item_uuid(2), "ok", True)]
    assert entry_of(tmp_path, 1)["reason"] == "pdf_changed"
    assert not pages_path(tmp_path, 1).exists()  # Stale page records do not survive.


# --- Failures and interruptions ----------------------------------------------------------


def test_an_unreadable_pdf_is_recorded_and_the_run_goes_on(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    corrupt = write_corrupt_file(tmp_path / "corrupt.pdf").read_bytes()
    write_test_snapshot(tmp_path, {1: thesis(), 2: corrupt, 3: thesis()})

    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [
        (item_uuid(1), "ok", False),
        (item_uuid(2), "error", False),
        (item_uuid(3), "ok", False),
    ]
    entry = entry_of(tmp_path, 2)
    assert (entry["reason"], entry["file"], entry["pages"]) == ("unreadable_pdf", None, 0)
    assert not pages_path(tmp_path, 2).exists()
    summary = text_manifest(tmp_path)["summary"]
    assert (summary["ok"], summary["error"]) == (2, 1)

    reader.opened.clear()
    report = run(config, tmp_path, reader, clock)

    assert reader.opened == [item_uuid(2)]  # Only the failed PDF is tried again.
    assert outcomes(report)[1] == (item_uuid(2), "error", False)


def test_an_interrupted_run_keeps_finished_pdfs_and_leaves_no_partial_files(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, ocr: FakeOcr
) -> None:
    write_test_snapshot(tmp_path, {1: [*thesis(5), ImagePage], 2: [*thesis(5), ImagePage]})

    def interrupt(item: UUID, page: int) -> None:
        if item == item_uuid(2):
            raise KeyboardInterrupt

    ocr.hook = interrupt

    with pytest.raises(KeyboardInterrupt):
        run(config, tmp_path, reader, clock)

    assert list(text_manifest(tmp_path)["items"]) == [str(item_uuid(1))]
    assert not pages_path(tmp_path, 2).exists()
    assert leftovers(tmp_path) == []

    ocr.hook = None
    reader.opened.clear()
    report = run(config, tmp_path, reader, clock)

    assert outcomes(report) == [(item_uuid(1), "ok", True), (item_uuid(2), "ok", False)]
    assert reader.opened == [item_uuid(2)]


def test_partial_files_left_by_a_killed_run_are_removed(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis()})
    pages_dir(tmp_path).mkdir(parents=True)
    (pages_dir(tmp_path) / f".{item_uuid(7)}.parquet.abc123.part").write_bytes(b"partial")
    (pages_dir(tmp_path) / "manifest.json.part").write_bytes(b"{")

    run(config, tmp_path, reader, clock)

    assert leftovers(tmp_path) == []


def test_a_second_run_cannot_start_while_one_is_in_progress(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, ocr: FakeOcr
) -> None:
    write_test_snapshot(tmp_path, {1: [*thesis(5), ImagePage]})
    refused: list[Exception] = []

    def start_another_run(item: UUID, page: int) -> None:
        rival = SpyReader(FakeOcr(FakeClock()))
        try:
            run(config, tmp_path, rival, FakeClock())
        except ExtractionInProgressError as error:
            refused.append(error)
        assert rival.opened == []

    ocr.hook = start_another_run

    report = run(config, tmp_path, reader, clock)

    assert len(refused) == 1
    assert outcomes(report) == [(item_uuid(1), "ok", False)]


# --- Selection and inputs ----------------------------------------------------------------


def test_only_theses_with_a_pdf_are_selected_in_snapshot_order(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: None, 3: thesis(), 4: thesis()})
    plans: list[Any] = []

    report = run(
        config, tmp_path, reader, clock, only=[item_uuid(4), item_uuid(1)], on_start=plans.append
    )

    assert [outcome.item_uuid for outcome in report.outcomes] == [item_uuid(1), item_uuid(4)]
    assert plans[0].pdfs == 3
    assert [selected.item_uuid for selected in plans[0].selected] == [item_uuid(1), item_uuid(4)]
    assert [
        outcome.item_uuid for outcome in run(config, tmp_path, reader, clock, limit=2).outcomes
    ] == [
        item_uuid(1),
        item_uuid(3),
    ]


@pytest.mark.parametrize("number", [2, 9], ids=["restricted", "unknown"])
def test_only_refuses_an_item_without_a_pdf(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, number: int
) -> None:
    write_test_snapshot(tmp_path, {1: thesis(), 2: None})

    with pytest.raises(ValueError, match=str(item_uuid(number))):
        run(config, tmp_path, reader, clock, only=[item_uuid(number)])

    assert reader.opened == []


def test_a_limit_below_one_is_refused(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    with pytest.raises(ValueError, match="limit"):
        run(config, tmp_path, reader, clock, limit=0)


def test_the_latest_snapshot_is_used_unless_one_is_named(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis()})

    report = run(config, tmp_path, reader, clock, snapshot_id=None)

    assert report.snapshot_id == SNAPSHOT_ID
    assert pages_path(tmp_path, 1).exists()


def test_a_snapshot_without_a_pdf_manifest_is_reported(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock
) -> None:
    write_test_snapshot(tmp_path, {1: thesis()})
    (raw_dir(tmp_path) / "pdfs" / "manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="download_pdfs"):
        run(config, tmp_path, reader, clock)


@pytest.mark.parametrize("content", ["{", "other-snapshot"], ids=["damaged", "other-snapshot"])
def test_a_text_manifest_that_does_not_fit_is_refused(
    config: AppConfig, tmp_path: Path, reader: SpyReader, clock: FakeClock, content: str
) -> None:
    write_test_snapshot(tmp_path, {1: thesis()})
    run(config, tmp_path, reader, clock)
    manifest_path = pages_dir(tmp_path) / "manifest.json"
    if content == "other-snapshot":
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.write_text(json.dumps(saved | {"snapshot_id": "other"}), encoding="utf-8")
    else:
        manifest_path.write_text(content, encoding="utf-8")
    reader.opened.clear()

    with pytest.raises(ValueError, match="manifest"):
        run(config, tmp_path, reader, clock)

    assert reader.opened == []


# --- Command line ------------------------------------------------------------------------


@pytest.fixture
def cli_root(tmp_path: Path) -> Path:
    """Return a project root that holds a copy of the declared configuration."""
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    return tmp_path


def cli(root: Path, reader: SpyReader, clock: FakeClock, *arguments: str) -> int:
    return main(
        ["--project-root", str(root), *arguments],
        reader=reader,
        clock=lambda: RUN_TIME,
        timer=clock,
    )


def test_cli_prints_one_line_of_counts_per_pdf_and_a_summary_without_any_text(
    cli_root: Path, reader: SpyReader, clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(
        cli_root, {1: [*thesis(5), ImagePage], 2: None, 3: thesis(6)}, programs={3: "minas"}
    )

    code = cli(cli_root, reader, clock)

    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0].startswith(f"Snapshot {SNAPSHOT_ID}: 2 of 2 thesis PDFs")
    assert re.search(
        rf"^\[1/2\] sistemas +{item_uuid(1)}  ok  6 pages \(5 text layer, 1 OCR, 0 empty\), "
        rf"OCR 1 tried / 0 failed, 15 lines removed, 2\.0 s, ETA 0:00:02$",
        out,
        re.M,
    )
    assert re.search(rf"^\[2/2\] minas +{item_uuid(3)}  ok  6 pages .* ETA 0:00:00$", out, re.M)
    assert "This run: 2 PDFs, 2 extracted, 0 skipped, 0 failed" in out
    assert "OCR: 1 pages tried, 0 failed, 2.0 s, 2.00 s per page" in out
    assert MARKER not in out

    assert cli(cli_root, reader, clock) == 0
    out = capsys.readouterr().out
    assert re.search(rf"^\[1/2\] sistemas +{item_uuid(1)}  unchanged, skipped$", out, re.M)
    assert "This run: 2 PDFs, 0 extracted, 2 skipped, 0 failed" in out


def test_cli_writes_a_summary_file_of_numbers_only(
    cli_root: Path, reader: SpyReader, clock: FakeClock, tmp_path: Path
) -> None:
    write_test_snapshot(cli_root, {1: [*thesis(5), ImagePage], 2: thesis(6)})
    summary_path = tmp_path / "out" / "summary.json"

    assert cli(cli_root, reader, clock, "--summary-out", str(summary_path)) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["snapshot_id"] == SNAPSHOT_ID
    assert summary["settings"]["ocr_window_pages"] == 100
    assert summary["summary"]["items"] == 2
    assert summary["summary"]["pages"] == 12
    assert summary["ocr_seconds_per_page"] == OCR_SECONDS
    assert MARKER not in summary_path.read_text(encoding="utf-8")
    assert str(item_uuid(1)) not in summary_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--snapshot-id", "missing"], "missing"),
        (["--snapshot-id", "../escape"], "snapshot id"),
        (["--only", str(item_uuid(9))], str(item_uuid(9))),
    ],
    ids=["unknown-snapshot", "unsafe-snapshot-id", "unknown-item"],
)
def test_cli_reports_a_fatal_error_on_one_line_with_exit_code_1(
    cli_root: Path,
    reader: SpyReader,
    clock: FakeClock,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    write_test_snapshot(cli_root, {1: thesis()})

    code = cli(cli_root, reader, clock, *arguments)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert message in captured.err
    assert len(captured.err.splitlines()) == 1
    assert reader.opened == []


def test_cli_reports_missing_language_files_before_reading_any_pdf(
    cli_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, {1: thesis()})

    code = main(["--project-root", str(cli_root)])

    captured = capsys.readouterr()
    assert code == 1
    assert "traineddata" in captured.err
    assert not pages_dir(cli_root).exists()


def test_cli_exits_with_1_when_a_pdf_failed(
    cli_root: Path, reader: SpyReader, clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt = write_corrupt_file(cli_root / "corrupt.pdf").read_bytes()
    write_test_snapshot(cli_root, {1: thesis(), 2: corrupt})

    code = cli(cli_root, reader, clock)

    out = capsys.readouterr().out
    assert code == 1
    assert f"{item_uuid(2)}  error (unreadable_pdf)" in out
    assert "1 PDFs failed" in out


def test_cli_stops_cleanly_on_an_interrupt(
    cli_root: Path,
    reader: SpyReader,
    clock: FakeClock,
    ocr: FakeOcr,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_test_snapshot(cli_root, {1: [*thesis(5), ImagePage], 2: [*thesis(5), ImagePage]})

    def interrupt(item: UUID, page: int) -> None:
        if item == item_uuid(2):
            raise KeyboardInterrupt

    ocr.hook = interrupt

    code = cli(cli_root, reader, clock)

    captured = capsys.readouterr()
    assert code == 130
    assert "interrupted" in captured.err
    assert list(text_manifest(cli_root)["items"]) == [str(item_uuid(1))]
    assert leftovers(cli_root) == []


@pytest.mark.parametrize(
    "arguments", [["--limit", "0"], ["--limit", "two"], ["--only", "not-a-uuid"], ["--only"]]
)
def test_cli_rejects_malformed_arguments(
    cli_root: Path, reader: SpyReader, clock: FakeClock, arguments: list[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli(cli_root, reader, clock, *arguments)

    assert caught.value.code == 2
    assert reader.opened == []
