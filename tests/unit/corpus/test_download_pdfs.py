"""Behavior of the thesis PDF download: the use case, its manifest, and its command line.

An in-memory repository serves synthetic files, and a fixed clock sets the run time, so no
test opens a connection or waits. The last tests run the real DSpace adapter over
``httpx.MockTransport`` and the synthetic item in ``tests/fixtures/dspace``. Every file name
holds the marker ``SYNTHETICNAME``, which must never reach the disk.
"""

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import HttpUrl

from thematic_redundancy.corpus.download_pdfs import (
    MAX_CONSECUTIVE_FAILURES,
    DownloadInProgressError,
    DownloadReport,
    DownloadStoppedError,
    download_theses,
    main,
)
from thematic_redundancy.corpus.dspace import DSpaceRestRepository
from thematic_redundancy.corpus.repository import (
    Bitstream,
    Bundle,
    ByteSink,
    Checksum,
    RepositoryError,
    ResourceUnavailableError,
)
from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.shared.config import AppConfig, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "dspace"
DECLARED = load_config(DEFAULT_CONFIG_PATH)
COLLECTIONS = {program.key: program.collection_uuid for program in DECLARED.snapshot.programs}
FACULTY = UUID("00000000-0000-4000-8000-0000000000f0")

HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
SNAPSHOT_ID = "20261002T150405Z"
RUN_TIME = datetime(2026, 10, 3, 9, 0, 0, tzinfo=UTC)
LATER = RUN_TIME + timedelta(hours=2)

TESIS = "https://purl.org/pe-repo/renati/type#tesis"
SUFICIENCIA = "https://purl.org/pe-repo/renati/type#trabajoDeSuficienciaProfesional"
OPEN = "https://purl.org/coar/access_right/c_abf2"
EMBARGOED = "https://purl.org/coar/access_right/c_f1cf"
RESTRICTED = "https://purl.org/coar/access_right/c_16ec"

THESIS_KIND, REPORT_KIND, AUTHORIZATION_KIND = 1, 2, 3


def item_uuid(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def file_uuid(number: int, kind: int = THESIS_KIND) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{kind:04d}{number:08d}")


def pdf_bytes(number: int) -> bytes:
    """Return the synthetic body of thesis ``number``: a PDF header and fake text."""
    return b"%PDF-1.7\n" + f"Synthetic thesis body {number:03d}\n".encode() * 40 + b"%%EOF\n"


def digest(algorithm: str, content: bytes) -> str:
    return hashlib.new(algorithm.replace("-", "").lower(), content).hexdigest()


def entry(value: str) -> dict[str, Any]:
    return {"value": value, "language": None, "authority": None, "confidence": -1, "place": 0}


def record(
    number: int,
    program: str = "sistemas",
    *,
    types: Sequence[str] = (TESIS,),
    rights: str = OPEN,
) -> SnapshotRecord:
    """Return snapshot record ``number``: a synthetic item of ``program``."""
    return SnapshotRecord(
        uuid=item_uuid(number),
        handle=f"123456789/{1000 + number}",
        program_key=program,
        collection_uuid=COLLECTIONS[program],
        harvested_at=HARVEST_TIME,
        metadata={
            "dc.rights": [entry(rights)],
            "dc.title": [entry(f"Título sintético {number:03d}")],
            "renati.type": [entry(value) for value in types],
        },
    )


def write_test_snapshot(
    project_root: Path,
    records: Sequence[SnapshotRecord],
    *,
    snapshot_id: str = SNAPSHOT_ID,
    harvested_at: datetime = HARVEST_TIME,
) -> Path:
    """Write ``records`` as a snapshot, as the harvest would, and return its directory."""
    content = encode_records(records)
    counts: dict[str, int] = {}
    for item in records:
        counts[item.program_key] = counts.get(item.program_key, 0) + 1
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        harvested_at=harvested_at,
        base_url="https://repo.example.edu",
        search_url="https://repo.example.edu/server/api/discover/search/objects",
        query_parameters={},
        programs={
            key: ProgramSummary(collection_uuid=COLLECTIONS[key], items=count)
            for key, count in counts.items()
        },
        total_items=len(records),
        faculty_community_uuid=FACULTY,
        faculty_total=len(records),
        items_by_type={},
        future_dated_items=0,
        unparsed_issue_dates=0,
        dropped_metadata_keys=(),
        metadata_sha256=hashlib.sha256(content).hexdigest(),
    )
    snapshot_dir = project_root / "data" / "raw" / snapshot_id
    write_snapshot(snapshot_dir, content, manifest)
    return snapshot_dir


class FakeFiles:
    """In-memory bitstream repository that serves items, refuses or fails some of them on
    request, and records every call."""

    def __init__(self) -> None:
        self.bundles: dict[UUID, tuple[Bundle, ...]] = {}
        self.contents: dict[UUID, bytes] = {}
        self.refusals: dict[UUID, int] = {}
        """HTTP status that refuses an item listing or a file download, by uuid."""
        self.failures: dict[UUID, str] = {}
        """Message of a repository failure, by item or file uuid."""
        self.during_download: Callable[[UUID], None] | None = None
        """Hook called halfway through every download, with the file uuid."""
        self.calls: list[tuple[str, UUID]] = []

    def serve_thesis(
        self,
        number: int,
        *,
        body: bytes | None = None,
        listed_size: int | None = None,
        listed_checksum: str | None = None,
        algorithm: str = "MD5",
    ) -> None:
        """Serve item ``number`` with its thesis, a similarity report and an authorization."""
        content = pdf_bytes(number) if body is None else body
        thesis = Bitstream(
            uuid=file_uuid(number),
            name=f"70.{number:04d}.SYNTHETICNAME.pdf",
            size_bytes=len(content) if listed_size is None else listed_size,
            checksum=Checksum(
                algorithm=algorithm,
                value=digest(algorithm, content) if listed_checksum is None else listed_checksum,
            ),
        )
        extras = [
            (REPORT_KIND, f"70.{number:04d}.SYNTHETICNAME.RT.pdf"),
            (AUTHORIZATION_KIND, f"Autorización_70.{number:04d}.SYNTHETICNAME.pdf"),
        ]
        others = tuple(
            Bitstream(
                uuid=file_uuid(number, kind),
                name=name,
                size_bytes=len(b"%PDF-other"),
                checksum=Checksum(algorithm="MD5", value=digest("MD5", b"%PDF-other")),
            )
            for kind, name in extras
        )
        self.serve_files(number, (thesis, *others))
        self.contents[thesis.uuid] = content
        for other in others:
            self.contents[other.uuid] = b"%PDF-other"

    def serve_files(self, number: int, original: Sequence[Bitstream]) -> None:
        """Serve item ``number`` with ``original`` as its ORIGINAL bundle, plus a license."""
        license_file = Bitstream(
            uuid=file_uuid(number, 9),
            name="license.txt",
            size_bytes=7,
            checksum=Checksum(algorithm="MD5", value=digest("MD5", b"license")),
        )
        self.bundles[item_uuid(number)] = (
            Bundle(name="ORIGINAL", bitstreams=tuple(original)),
            Bundle(name="LICENSE", bitstreams=(license_file,)),
        )

    def list_bundles(self, item: UUID) -> tuple[Bundle, ...]:
        self.calls.append(("list", item))
        self._refuse(item)
        return self.bundles.get(item, ())

    def download(self, bitstream: UUID, sink: ByteSink) -> None:
        self.calls.append(("download", bitstream))
        self._refuse(bitstream)
        sink.restart()
        content = self.contents[bitstream]
        middle = len(content) // 2
        sink.write(content[:middle])
        if self.during_download is not None:
            self.during_download(bitstream)
        sink.write(content[middle:])

    def _refuse(self, uuid: UUID) -> None:
        if uuid in self.refusals:
            status = self.refusals[uuid]
            raise ResourceUnavailableError(f"GET {uuid} returned HTTP {status}", status_code=status)
        if uuid in self.failures:
            raise RepositoryError(self.failures[uuid])


@pytest.fixture
def config() -> AppConfig:
    """Return the declared configuration pointed at a synthetic repository."""
    repository = DECLARED.repository.model_copy(
        update={"base_url": HttpUrl("https://repo.example.edu")}
    )
    return DECLARED.model_copy(update={"repository": repository})


@pytest.fixture
def files() -> FakeFiles:
    return FakeFiles()


def run(
    config: AppConfig,
    project_root: Path,
    files: FakeFiles,
    *,
    at: datetime = RUN_TIME,
    **options: Any,
) -> DownloadReport:
    """Run the use case with the clock fixed at ``at``."""
    return download_theses(config, project_root, files, clock=lambda: at, **options)


def pdf_dir(project_root: Path, snapshot_id: str = SNAPSHOT_ID) -> Path:
    return project_root / "data" / "raw" / snapshot_id / "pdfs"


def read_pdf_manifest(project_root: Path) -> dict[str, Any]:
    return json.loads((pdf_dir(project_root) / "manifest.json").read_text(encoding="utf-8"))


def statuses(report: DownloadReport) -> list[tuple[UUID, str, str | None]]:
    return [
        (outcome.item_uuid, outcome.entry.status, outcome.entry.reason)
        for outcome in report.outcomes
    ]


def saved_pdfs(project_root: Path) -> list[str]:
    return sorted(path.name for path in pdf_dir(project_root).glob("*.pdf"))


def leftovers(project_root: Path) -> list[str]:
    """Return the partial files left in the PDF directory."""
    return sorted(path.name for path in pdf_dir(project_root).glob("*.part"))


def test_the_thesis_of_each_item_is_saved_under_the_item_uuid(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1), record(2, "minas")])
    files.serve_thesis(1)
    files.serve_thesis(2)

    report = run(config, tmp_path, files)

    assert report.pdf_dir == pdf_dir(tmp_path).resolve()
    assert statuses(report) == [
        (item_uuid(1), "downloaded", None),
        (item_uuid(2), "downloaded", None),
    ]
    assert saved_pdfs(tmp_path) == [f"{item_uuid(1)}.pdf", f"{item_uuid(2)}.pdf"]
    assert (pdf_dir(tmp_path) / f"{item_uuid(2)}.pdf").read_bytes() == pdf_bytes(2)
    assert leftovers(tmp_path) == []
    # One listing and one download per item: the report and the authorization are never fetched.
    assert files.calls == [
        ("list", item_uuid(1)),
        ("download", file_uuid(1)),
        ("list", item_uuid(2)),
        ("download", file_uuid(2)),
    ]


def test_the_manifest_records_each_file_and_its_checksums_but_no_file_name(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1), record(2, "minas")])
    files.serve_thesis(1)
    files.serve_thesis(2)

    run(config, tmp_path, files)

    manifest = read_pdf_manifest(tmp_path)
    body = pdf_bytes(1)
    assert manifest["items"][str(item_uuid(1))] == {
        "program_key": "sistemas",
        "status": "downloaded",
        "reason": None,
        "detail": None,
        "file": f"{item_uuid(1)}.pdf",
        "bitstream_uuid": str(file_uuid(1)),
        "size_bytes": len(body),
        "checksum_algorithm": "MD5",
        "checksum": digest("MD5", body),
        "md5": digest("MD5", body),
        "sha256": digest("SHA256", body),
        "candidate_count": 1,
        "downloaded_at": "2026-10-03T09:00:00Z",
        "checked_at": "2026-10-03T09:00:00Z",
    }
    assert list(manifest["items"]) == [str(item_uuid(1)), str(item_uuid(2))]
    assert manifest["summary"] == {
        "items": 2,
        "files": 2,
        "total_bytes": len(pdf_bytes(1)) + len(pdf_bytes(2)),
        "by_status": {"downloaded": 2},
        "by_program": {"sistemas": {"downloaded": 1}, "minas": {"downloaded": 1}},
    }
    assert (manifest["snapshot_id"], manifest["base_url"]) == (
        SNAPSHOT_ID,
        "https://repo.example.edu",
    )
    assert (manifest["thesis_type"], manifest["updated_at"]) == ("tesis", "2026-10-03T09:00:00Z")
    assert ".RT.pdf" in manifest["selection_rule"]
    assert b"SYNTHETICNAME" not in (pdf_dir(tmp_path) / "manifest.json").read_bytes()


@pytest.mark.parametrize(
    ("serve", "reason"),
    [
        ({"body": b"<html>Maintenance page</html>"}, "not_a_pdf"),
        ({"listed_size": len(pdf_bytes(1)) + 10}, "size_mismatch"),
        ({"listed_size": len(pdf_bytes(1)) - 10}, "size_mismatch"),
        ({"listed_checksum": "0" * 32}, "checksum_mismatch"),
    ],
    ids=["not-a-pdf", "shorter-than-listed", "longer-than-listed", "other-checksum"],
)
def test_a_file_that_fails_a_check_is_discarded_as_an_integrity_error(
    config: AppConfig,
    tmp_path: Path,
    files: FakeFiles,
    serve: dict[str, Any],
    reason: str,
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1, **serve)

    report = run(config, tmp_path, files)

    assert statuses(report) == [(item_uuid(1), "integrity_error", reason)]
    assert report.outcomes[0].entry.detail
    assert saved_pdfs(tmp_path) == []
    assert leftovers(tmp_path) == []
    assert files.calls.count(("download", file_uuid(1))) == 1  # One attempt, no loop.
    assert read_pdf_manifest(tmp_path)["summary"]["by_status"] == {"integrity_error": 1}


def test_a_checksum_in_an_unknown_algorithm_is_refused_before_downloading(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1, algorithm="MD5", listed_checksum="1a2b3c4d")
    thesis = files.bundles[item_uuid(1)][0].bitstreams[0]
    unknown = Bitstream(
        uuid=thesis.uuid,
        name=thesis.name,
        size_bytes=thesis.size_bytes,
        checksum=Checksum(algorithm="CRC32", value="1a2b3c4d"),
    )
    files.serve_files(1, [unknown])

    report = run(config, tmp_path, files)

    assert statuses(report) == [(item_uuid(1), "integrity_error", "unsupported_checksum_algorithm")]
    assert files.calls == [("list", item_uuid(1))]


@pytest.mark.parametrize("algorithm", ["SHA-256", "sha1", "md5"])
def test_a_checksum_in_another_known_algorithm_is_verified_too(
    config: AppConfig, tmp_path: Path, files: FakeFiles, algorithm: str
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1, algorithm=algorithm)

    report = run(config, tmp_path, files)

    [outcome] = report.outcomes
    assert (outcome.entry.status, outcome.entry.checksum_algorithm) == ("downloaded", algorithm)
    assert outcome.entry.checksum == digest(algorithm, pdf_bytes(1))


def test_a_second_run_finds_the_files_already_present_without_any_request(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1), record(2, "minas")])
    files.serve_thesis(1)
    files.serve_thesis(2)
    run(config, tmp_path, files)
    calls = list(files.calls)

    second = run(config, tmp_path, files, at=LATER)

    assert [status for _, status, _ in statuses(second)] == ["already_present"] * 2
    assert files.calls == calls
    entry_1 = read_pdf_manifest(tmp_path)["items"][str(item_uuid(1))]
    assert entry_1["status"] == "already_present"
    assert entry_1["downloaded_at"] == "2026-10-03T09:00:00Z"
    assert entry_1["checked_at"] == "2026-10-03T11:00:00Z"
    assert entry_1["sha256"] == digest("SHA256", pdf_bytes(1))
    assert read_pdf_manifest(tmp_path)["summary"]["by_status"] == {"already_present": 2}


def truncated(body: bytes) -> bytes:
    return body[: len(body) // 2]


def with_one_byte_changed(body: bytes) -> bytes:
    """Return ``body`` with the same size and header but one byte flipped."""
    return body[:-3] + bytes([body[-3] ^ 0x01]) + body[-2:]


@pytest.mark.parametrize("damage", [truncated, with_one_byte_changed])
def test_a_file_changed_on_disk_is_downloaded_again(
    config: AppConfig, tmp_path: Path, files: FakeFiles, damage: Callable[[bytes], bytes]
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1)
    run(config, tmp_path, files)
    saved = pdf_dir(tmp_path) / f"{item_uuid(1)}.pdf"
    saved.write_bytes(damage(pdf_bytes(1)))

    second = run(config, tmp_path, files, at=LATER)

    assert statuses(second) == [(item_uuid(1), "downloaded", None)]
    assert saved.read_bytes() == pdf_bytes(1)
    assert files.calls[2:] == [("list", item_uuid(1)), ("download", file_uuid(1))]


@pytest.mark.parametrize(
    ("chunk", "reason", "chunks_taken"),
    [(b"%PDF" * 256, "size_mismatch", 2), (b"<htm" * 256, "not_a_pdf", 0)],
    ids=["longer-than-listed", "not-a-pdf"],
)
def test_a_body_that_cannot_be_the_listed_pdf_is_cut_off_early(
    config: AppConfig,
    tmp_path: Path,
    files: FakeFiles,
    chunk: bytes,
    reason: str,
    chunks_taken: int,
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1, listed_size=2 * len(chunk))
    taken: list[int] = []

    def endless_body(bitstream: UUID, sink: ByteSink) -> None:
        files.calls.append(("download", bitstream))
        sink.restart()
        for number in range(10_000):
            sink.write(chunk)
            taken.append(number)

    files.download = endless_body  # type: ignore[method-assign]

    report = run(config, tmp_path, files)

    assert statuses(report) == [(item_uuid(1), "integrity_error", reason)]
    assert len(taken) == chunks_taken  # The rest of the body was never read.
    assert leftovers(tmp_path) == []


def test_a_file_missing_from_the_manifest_is_kept_when_it_matches_the_repository(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1)
    run(config, tmp_path, files)
    (pdf_dir(tmp_path) / "manifest.json").unlink()

    second = run(config, tmp_path, files, at=LATER)

    assert statuses(second) == [(item_uuid(1), "already_present", None)]
    assert files.calls[2:] == [("list", item_uuid(1))]  # Checked against the listing only.
    assert second.outcomes[0].entry.sha256 == digest("SHA256", pdf_bytes(1))


@pytest.mark.parametrize(("rights", "code"), [(EMBARGOED, "c_f1cf"), (RESTRICTED, "c_16ec")])
def test_items_closed_by_their_access_rights_are_skipped_without_any_request(
    config: AppConfig, tmp_path: Path, files: FakeFiles, rights: str, code: str
) -> None:
    write_test_snapshot(tmp_path, [record(1, rights=rights), record(2)])
    files.serve_thesis(1)
    files.serve_thesis(2)

    report = run(config, tmp_path, files)

    assert statuses(report) == [
        (item_uuid(1), "restricted", "rights_restricted"),
        (item_uuid(2), "downloaded", None),
    ]
    assert code in (report.outcomes[0].entry.detail or "")
    assert [uuid for _, uuid in files.calls] == [item_uuid(2), file_uuid(2)]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ("restricted", "http_401")),
        (403, ("restricted", "http_403")),
        (404, ("not_found", "http_404")),
    ],
)
@pytest.mark.parametrize("refused", ["listing", "file"])
def test_a_refused_item_is_recorded_and_never_asked_again(
    config: AppConfig,
    tmp_path: Path,
    files: FakeFiles,
    status: int,
    expected: tuple[str, str],
    refused: str,
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1)
    files.refusals[item_uuid(1) if refused == "listing" else file_uuid(1)] = status

    first = run(config, tmp_path, files)
    calls = list(files.calls)
    second = run(config, tmp_path, files, at=LATER)

    assert statuses(first) == statuses(second) == [(item_uuid(1), *expected)]
    assert files.calls == calls  # The second run asked nothing.
    assert saved_pdfs(tmp_path) == []
    assert leftovers(tmp_path) == []


def only_excluded_files(files: FakeFiles) -> None:
    files.serve_thesis(1)
    report, authorization = files.bundles[item_uuid(1)][0].bitstreams[1:]
    files.serve_files(1, [report, authorization])


def no_original_bundle(files: FakeFiles) -> None:
    files.bundles[item_uuid(1)] = ()


@pytest.mark.parametrize(
    ("serve", "reason"),
    [(only_excluded_files, "no_candidate_pdf"), (no_original_bundle, "no_original_bundle")],
)
def test_an_item_without_a_thesis_pdf_is_recorded_as_no_thesis_file(
    config: AppConfig,
    tmp_path: Path,
    files: FakeFiles,
    serve: Callable[[FakeFiles], None],
    reason: str,
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    serve(files)

    report = run(config, tmp_path, files)

    assert statuses(report) == [(item_uuid(1), "no_thesis_file", reason)]
    assert report.outcomes[0].entry.candidate_count == 0
    assert files.calls == [("list", item_uuid(1))]


def test_an_item_with_several_candidate_pdfs_is_recorded_as_ambiguous(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1)
    thesis, report_file, authorization = files.bundles[item_uuid(1)][0].bitstreams
    second_volume = Bitstream(
        uuid=file_uuid(1, 4),
        name="70.0001.SYNTHETICNAME.Tomo2.pdf",
        size_bytes=10,
        checksum=Checksum(algorithm="MD5", value="0" * 32),
    )
    files.serve_files(1, [thesis, report_file, second_volume, authorization])

    report = run(config, tmp_path, files)

    assert statuses(report) == [(item_uuid(1), "ambiguous", "multiple_candidates")]
    assert report.outcomes[0].entry.candidate_count == 2
    assert files.calls == [("list", item_uuid(1))]
    assert saved_pdfs(tmp_path) == []


def test_an_interrupted_run_keeps_the_items_it_finished_and_the_next_run_resumes(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1), record(2), record(3)])
    for number in (1, 2, 3):
        files.serve_thesis(number)

    def interrupt(bitstream: UUID) -> None:
        if bitstream == file_uuid(2):
            raise KeyboardInterrupt

    files.during_download = interrupt

    with pytest.raises(KeyboardInterrupt):
        run(config, tmp_path, files)

    manifest = read_pdf_manifest(tmp_path)
    assert list(manifest["items"]) == [str(item_uuid(1))]
    assert manifest["summary"]["by_status"] == {"downloaded": 1}
    assert saved_pdfs(tmp_path) == [f"{item_uuid(1)}.pdf"]
    assert leftovers(tmp_path) == []

    files.during_download = None
    calls = len(files.calls)
    resumed = run(config, tmp_path, files, at=LATER)

    assert [status for _, status, _ in statuses(resumed)] == [
        "already_present",
        "downloaded",
        "downloaded",
    ]
    assert files.calls[calls:] == [
        ("list", item_uuid(2)),
        ("download", file_uuid(2)),
        ("list", item_uuid(3)),
        ("download", file_uuid(3)),
    ]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, [1, 3, 4, 5, 6]),
        ({"limit": 2}, [1, 3]),
        ({"per_program": 1}, [1, 5]),
        ({"per_program": 2, "limit": 3}, [1, 3, 5]),
        ({"limit": 50}, [1, 3, 4, 5, 6]),
    ],
    ids=["all", "limit", "per-program", "both", "limit-above-total"],
)
def test_limit_and_per_program_take_the_first_theses_in_snapshot_order(
    config: AppConfig,
    tmp_path: Path,
    files: FakeFiles,
    options: Mapping[str, int],
    expected: list[int],
) -> None:
    records = [
        record(1),
        record(2, types=(SUFICIENCIA,)),
        record(3),
        record(4),
        record(5, "minas"),
        record(6, "minas"),
    ]
    write_test_snapshot(tmp_path, records)
    for number in (1, 3, 4, 5, 6):
        files.serve_thesis(number)

    report = run(config, tmp_path, files, **options)

    assert [outcome.item_uuid for outcome in report.outcomes] == [item_uuid(n) for n in expected]
    assert report.theses == 5
    assert ("list", item_uuid(2)) not in files.calls  # Not a thesis.


@pytest.mark.parametrize("option", ["limit", "per_program"])
def test_limits_below_one_are_rejected(
    config: AppConfig, tmp_path: Path, files: FakeFiles, option: str
) -> None:
    write_test_snapshot(tmp_path, [record(1)])

    with pytest.raises(ValueError, match=option):
        run(config, tmp_path, files, **{option: 0})

    assert files.calls == []


def test_the_latest_snapshot_is_used_unless_one_is_named(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(
        tmp_path, [record(1)], snapshot_id="zeta-older", harvested_at=HARVEST_TIME - timedelta(1)
    )
    write_test_snapshot(tmp_path, [record(2)], snapshot_id="alpha-newer")
    (tmp_path / "data" / "raw" / ".omega.123.partial").mkdir()  # An unfinished harvest.
    (tmp_path / "data" / "raw" / "notes.txt").write_text("not a snapshot", encoding="utf-8")
    files.serve_thesis(1)
    files.serve_thesis(2)

    latest = run(config, tmp_path, files)
    named = run(config, tmp_path, files, snapshot_id="zeta-older")

    assert (latest.snapshot_id, [outcome.item_uuid for outcome in latest.outcomes]) == (
        "alpha-newer",
        [item_uuid(2)],
    )
    assert (named.snapshot_id, [outcome.item_uuid for outcome in named.outcomes]) == (
        "zeta-older",
        [item_uuid(1)],
    )
    assert named.pdf_dir == pdf_dir(tmp_path, "zeta-older").resolve()


def test_a_snapshot_whose_metadata_does_not_match_its_digest_is_refused(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    snapshot_dir = write_test_snapshot(tmp_path, [record(1)])
    with (snapshot_dir / "metadata.jsonl").open("ab") as metadata:
        metadata.write(b"\n")
    files.serve_thesis(1)

    with pytest.raises(ValueError, match="SHA-256"):
        run(config, tmp_path, files)

    assert files.calls == []
    assert not pdf_dir(tmp_path).exists()


@pytest.mark.parametrize(
    ("prepare", "snapshot_id", "message"),
    [
        (lambda root: None, None, "no snapshot"),
        (lambda root: write_test_snapshot(root, [record(1)]), "missing", "missing"),
        (lambda root: write_test_snapshot(root, [record(1)]), "../escape", "snapshot id"),
    ],
    ids=["no-snapshot-at-all", "unknown-id", "unsafe-id"],
)
def test_a_snapshot_that_cannot_be_found_is_reported_before_any_request(
    config: AppConfig,
    tmp_path: Path,
    files: FakeFiles,
    prepare: Callable[[Path], object],
    snapshot_id: str | None,
    message: str,
) -> None:
    prepare(tmp_path)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        run(config, tmp_path, files, snapshot_id=snapshot_id)

    assert files.calls == []


def test_a_damaged_pdf_manifest_is_refused_before_any_request(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    pdf_dir(tmp_path).mkdir()
    (pdf_dir(tmp_path) / "manifest.json").write_text("{not json", encoding="utf-8")
    files.serve_thesis(1)

    with pytest.raises(ValueError, match="manifest"):
        run(config, tmp_path, files)

    assert files.calls == []


def test_a_failed_item_is_recorded_the_run_goes_on_and_the_next_run_retries_it(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1), record(2)])
    files.serve_thesis(1)
    files.serve_thesis(2)
    files.failures[item_uuid(1)] = "GET items/1 gave up after 4 attempts: HTTP 503"

    first = run(config, tmp_path, files)
    del files.failures[item_uuid(1)]
    second = run(config, tmp_path, files, at=LATER)

    assert statuses(first) == [
        (item_uuid(1), "error", "repository_error"),
        (item_uuid(2), "downloaded", None),
    ]
    assert "HTTP 503" in (first.outcomes[0].entry.detail or "")
    assert statuses(second) == [
        (item_uuid(1), "downloaded", None),
        (item_uuid(2), "already_present", None),
    ]


def test_failures_in_a_row_stop_the_run_and_keep_what_was_recorded(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    count = MAX_CONSECUTIVE_FAILURES + 1
    write_test_snapshot(tmp_path, [record(number) for number in range(1, count + 1)])
    for number in range(1, count + 1):
        files.serve_thesis(number)
        files.failures[item_uuid(number)] = "GET gave up after 4 attempts: HTTP 503"

    with pytest.raises(DownloadStoppedError, match=str(MAX_CONSECUTIVE_FAILURES)):
        run(config, tmp_path, files)

    assert files.calls == [("list", item_uuid(n)) for n in range(1, count)]
    manifest = read_pdf_manifest(tmp_path)
    assert manifest["summary"]["by_status"] == {"error": MAX_CONSECUTIVE_FAILURES}


def test_a_success_between_failures_keeps_the_run_going(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    numbers = range(1, 2 * MAX_CONSECUTIVE_FAILURES + 1)
    write_test_snapshot(tmp_path, [record(number) for number in numbers])
    for number in numbers:
        files.serve_thesis(number)
        if number != MAX_CONSECUTIVE_FAILURES:
            files.failures[item_uuid(number)] = "GET gave up after 4 attempts: HTTP 503"

    with pytest.raises(DownloadStoppedError):
        run(config, tmp_path, files)

    assert len(read_pdf_manifest(tmp_path)["items"]) == 2 * MAX_CONSECUTIVE_FAILURES


def test_a_second_run_cannot_start_while_one_is_in_progress(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    files.serve_thesis(1)
    refused: list[Exception] = []

    def start_another_run(bitstream: UUID) -> None:
        rival = FakeFiles()
        try:
            run(config, tmp_path, rival)
        except DownloadInProgressError as error:
            refused.append(error)
        assert rival.calls == []

    files.during_download = start_another_run

    report = run(config, tmp_path, files)

    assert len(refused) == 1
    assert statuses(report) == [(item_uuid(1), "downloaded", None)]
    files.during_download = None
    assert statuses(run(config, tmp_path, files, at=LATER)) == [
        (item_uuid(1), "already_present", None)
    ]


def test_partial_files_left_by_a_killed_run_are_removed(
    config: AppConfig, tmp_path: Path, files: FakeFiles
) -> None:
    write_test_snapshot(tmp_path, [record(1)])
    pdf_dir(tmp_path).mkdir()
    (pdf_dir(tmp_path) / f"{item_uuid(7)}.pdf.part").write_bytes(b"%PDF-1.7 partial")
    files.serve_thesis(1)

    run(config, tmp_path, files)

    assert leftovers(tmp_path) == []
    assert saved_pdfs(tmp_path) == [f"{item_uuid(1)}.pdf"]


@pytest.fixture
def cli_root(tmp_path: Path) -> Path:
    """Return a project root that holds a copy of the declared configuration."""
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    return tmp_path


def test_cli_prints_one_line_per_thesis_and_a_summary(
    cli_root: Path, files: FakeFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(
        cli_root,
        [record(1), record(2), record(3, rights=EMBARGOED), record(4, "minas"), record(5, "minas")],
    )
    for number in (1, 2, 4, 5):
        files.serve_thesis(number)

    code = main(
        ["--project-root", str(cli_root), "--per-program", "2"],
        repository=files,
        clock=lambda: RUN_TIME,
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0].startswith(f"Snapshot {SNAPSHOT_ID}: 4 of 5 theses")
    size = f"{len(pdf_bytes(1)):,} bytes"
    assert re.search(rf"^\[1/4\] sistemas +{item_uuid(1)}  downloaded +{size}$", out, re.M)
    assert re.search(rf"^\[3/4\] minas +{item_uuid(4)}  downloaded", out, re.M)
    assert str(item_uuid(3)) not in out  # The third sistemas thesis is beyond --per-program 2.
    assert re.search(r"^  downloaded +4$", out, re.M)
    assert re.search(r"^  sistemas +downloaded 2$", out, re.M)
    total = sum(len(pdf_bytes(number)) for number in (1, 2, 4, 5))
    assert f"4 files, {total:,} bytes" in out
    assert str((pdf_dir(cli_root) / "manifest.json").resolve()) in out
    assert "SYNTHETICNAME" not in out


def test_cli_reports_restricted_items_with_their_reason(
    cli_root: Path, files: FakeFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, [record(1, rights=EMBARGOED), record(2)])
    files.serve_thesis(2)
    files.refusals[file_uuid(2)] = 403

    code = main(["--project-root", str(cli_root)], repository=files, clock=lambda: RUN_TIME)

    out = capsys.readouterr().out
    assert code == 0
    assert f"{item_uuid(1)}  restricted (rights_restricted)" in out
    assert f"{item_uuid(2)}  restricted (http_403)" in out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--snapshot-id", "missing"], "missing"),
        (["--snapshot-id", "../escape"], "snapshot id"),
    ],
    ids=["unknown-snapshot", "unsafe-snapshot-id"],
)
def test_cli_reports_a_fatal_error_on_one_line_with_exit_code_1(
    cli_root: Path,
    files: FakeFiles,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    write_test_snapshot(cli_root, [record(1)])

    code = main(
        ["--project-root", str(cli_root), *arguments], repository=files, clock=lambda: RUN_TIME
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert message in captured.err
    assert len(captured.err.splitlines()) == 1
    assert files.calls == []


def test_cli_exits_with_1_when_an_item_failed_and_says_how_to_retry(
    cli_root: Path, files: FakeFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, [record(1), record(2)])
    files.serve_thesis(1)
    files.serve_thesis(2, listed_checksum="0" * 32)

    code = main(["--project-root", str(cli_root)], repository=files, clock=lambda: RUN_TIME)

    out = capsys.readouterr().out
    assert code == 1
    assert f"{item_uuid(2)}  integrity_error (checksum_mismatch)" in out
    assert "run the command again" in out


def test_cli_stops_cleanly_on_an_interrupt_and_keeps_the_manifest(
    cli_root: Path, files: FakeFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, [record(1), record(2)])
    files.serve_thesis(1)
    files.serve_thesis(2)

    def interrupt(bitstream: UUID) -> None:
        if bitstream == file_uuid(2):
            raise KeyboardInterrupt

    files.during_download = interrupt

    code = main(["--project-root", str(cli_root)], repository=files, clock=lambda: RUN_TIME)

    captured = capsys.readouterr()
    assert code == 130
    assert "interrupted" in captured.err
    assert list(read_pdf_manifest(cli_root)["items"]) == [str(item_uuid(1))]
    assert leftovers(cli_root) == []


@pytest.mark.parametrize("option", ["--limit", "--per-program"])
@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_cli_rejects_counts_that_are_not_positive_integers(
    cli_root: Path, files: FakeFiles, option: str, value: str
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--project-root", str(cli_root), option, value], repository=files)

    assert caught.value.code == 2
    assert files.calls == []


def item_response(number: int, body: bytes) -> dict[str, Any]:
    """Return the synthetic item fixture as item ``number``, whose thesis holds ``body``."""
    payload = json.loads((FIXTURES_DIR / "item_files.json").read_text(encoding="utf-8"))
    payload["uuid"] = payload["id"] = str(item_uuid(number))
    original = payload["_embedded"]["bundles"]["_embedded"]["bundles"][0]
    thesis, report_file, authorization = original["_embedded"]["bitstreams"]["_embedded"][
        "bitstreams"
    ]
    for kind, listed in [(1, thesis), (2, report_file), (3, authorization)]:
        listed["uuid"] = listed["id"] = str(file_uuid(number, kind))
    thesis["sizeBytes"] = len(body)
    thesis["checkSum"] = {"checkSumAlgorithm": "MD5", "value": digest("MD5", body)}
    return payload


def test_cli_downloads_through_the_dspace_adapter_at_a_polite_pace(
    cli_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, [record(1), record(2, "minas"), record(3, types=(SUFICIENCIA,))])
    bodies = {str(file_uuid(number)): pdf_bytes(number) for number in (1, 2)}
    paths: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        segments = request.url.path.split("/")
        if segments[-2] == "items":
            number = int(segments[-1][-12:])  # Inverse of item_uuid().
            return httpx.Response(200, json=item_response(number, pdf_bytes(number)))
        if segments[-1] == "content" and segments[-2] in bodies:
            return httpx.Response(200, content=bodies[segments[-2]])
        return httpx.Response(404)

    def run_cli() -> tuple[int, DSpaceRestRepository, list[float]]:
        sleeps: list[float] = []
        with httpx.Client(transport=httpx.MockTransport(serve)) as client:
            adapter = DSpaceRestRepository.from_config(
                client, DECLARED.repository, clock=lambda: 0.0, sleeper=sleeps.append
            )
            code = main(
                ["--project-root", str(cli_root)], repository=adapter, clock=lambda: RUN_TIME
            )
        return code, adapter, sleeps

    code, adapter, sleeps = run_cli()

    out = capsys.readouterr().out
    assert code == 0
    assert paths == [
        f"/server/api/core/items/{item_uuid(1)}",
        f"/server/api/core/bitstreams/{file_uuid(1)}/content",
        f"/server/api/core/items/{item_uuid(2)}",
        f"/server/api/core/bitstreams/{file_uuid(2)}/content",
    ]
    assert (adapter.request_count, sleeps) == (4, [1.0, 1.0, 1.0])
    assert "HTTP requests: 4" in out
    assert saved_pdfs(cli_root) == [f"{item_uuid(1)}.pdf", f"{item_uuid(2)}.pdf"]
    assert b"SYNTHETICNAME" not in (pdf_dir(cli_root) / "manifest.json").read_bytes()

    code, adapter, _ = run_cli()

    out = capsys.readouterr().out
    assert (code, adapter.request_count) == (0, 0)
    assert "HTTP requests: 0" in out
    assert out.count("already_present") >= 2
