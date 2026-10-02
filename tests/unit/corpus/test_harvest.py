"""Behavior of the metadata harvest: the use case that writes a snapshot, and its command line.

An in-memory repository stands in for DSpace and a fixed clock sets the harvest time, so no
test opens a connection or waits. The last test runs the real DSpace adapter over
``httpx.MockTransport`` and the synthetic pages in ``tests/fixtures/dspace``.
"""

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import HttpUrl

from thematic_redundancy.corpus.dspace import DSpaceRestRepository
from thematic_redundancy.corpus.harvest import HarvestError, HarvestReport, harvest_snapshot, main
from thematic_redundancy.corpus.repository import (
    IssuedYears,
    RepositoryError,
    RepositoryItem,
    ScopeListing,
    SearchDescription,
)
from thematic_redundancy.corpus.snapshot import SnapshotExistsError, SnapshotManifest
from thematic_redundancy.shared.config import AppConfig, ProgramConfig, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "dspace"
DECLARED = load_config(DEFAULT_CONFIG_PATH)

HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
SNAPSHOT_ID = "20261002T150405Z"
"""Default snapshot id for :data:`HARVEST_TIME`."""

SISTEMAS = UUID("00000000-0000-4000-8000-0000000000c1")
MINAS = UUID("00000000-0000-4000-8000-0000000000c2")
FACULTY = UUID("00000000-0000-4000-8000-0000000000f0")
YEARS = IssuedYears(first=2021, last=2026)
SEARCH_URL = "https://repo.example.edu/server/api/discover/search/objects"
TESIS = "https://purl.org/pe-repo/renati/type#tesis"
SUFICIENCIA = "https://purl.org/pe-repo/renati/type#trabajoDeSuficienciaProfesional"


def entry(value: str, *, language: str | None = None, place: int = 0) -> dict[str, Any]:
    """Return one metadata value in the shape that DSpace serves."""
    return {
        "value": value,
        "language": language,
        "authority": None,
        "confidence": -1,
        "place": place,
    }


def item_uuid(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def item(
    number: int,
    *,
    issued: str | None = "2022-05-10",
    types: Sequence[str] = (TESIS,),
    extra: Mapping[str, list[dict[str, Any]]] | None = None,
) -> RepositoryItem:
    """Return a synthetic item whose metadata holds both DNI fields, as UCSM items do."""
    metadata: dict[str, list[dict[str, Any]]] = {
        "dc.contributor.author": [entry(f"Autor Sintético, {number:03d}")],
        "dc.title": [entry(f"Título sintético {number:03d}", language="es")],
        "renati.advisor.dni": [entry(f"SYNTHETIC-DNI-ADVISOR-{number:03d}")],
        "renati.author.dni": [entry(f"SYNTHETIC-DNI-AUTHOR-{number:03d}")],
    }
    if issued is not None:
        metadata["dc.date.issued"] = [entry(issued)]
    if types:
        metadata["renati.type"] = [entry(value, place=place) for place, value in enumerate(types)]
    metadata.update(extra or {})
    return RepositoryItem(
        uuid=item_uuid(number), handle=f"123456789/{1000 + number}", metadata=metadata
    )


class FakeRepository:
    """In-memory item repository that serves fixed listings and records every search."""

    def __init__(
        self,
        listings: Mapping[UUID, Sequence[RepositoryItem]],
        *,
        reported: Mapping[UUID, int] | None = None,
        faculty_total: int | None = None,
        failing_scope: UUID | None = None,
        on_list: Callable[[], None] | None = None,
    ) -> None:
        self.listings = listings
        self.reported = reported or {}
        self.faculty_total = faculty_total
        self.failing_scope = failing_scope
        self.on_list = on_list
        self.calls: list[tuple[str, UUID, IssuedYears]] = []

    def describe_search(self, years: IssuedYears) -> SearchDescription:
        return SearchDescription(
            search_url=SEARCH_URL,
            query_parameters={
                "dsoType": "ITEM",
                "size": "100",
                "sort": "dc.date.issued,ASC",
                "f.dateIssued": f"[{years.first} TO {years.last}],equals",
            },
        )

    def list_items(self, scope: UUID, years: IssuedYears) -> ScopeListing:
        self.calls.append(("list", scope, years))
        if self.on_list is not None:
            self.on_list()
        if scope == self.failing_scope:
            raise RepositoryError(f"GET {SEARCH_URL} gave up after 4 attempts: HTTP 503")
        items = tuple(self.listings.get(scope, ()))
        return ScopeListing(items=items, reported_total=self.reported.get(scope, len(items)))

    def count_items(self, scope: UUID, years: IssuedYears) -> int:
        self.calls.append(("count", scope, years))
        if self.faculty_total is not None:
            return self.faculty_total
        return sum(len(items) for items in self.listings.values())


@pytest.fixture
def config() -> AppConfig:
    """Return the declared configuration narrowed to two programs on synthetic collections."""
    programs = (
        ProgramConfig(key="sistemas", name="Ingeniería de Sistemas", collection_uuid=SISTEMAS),
        ProgramConfig(key="minas", name="Ingeniería de Minas", collection_uuid=MINAS),
    )
    repository = DECLARED.repository.model_copy(
        update={"base_url": HttpUrl("https://repo.example.edu"), "faculty_community_uuid": FACULTY}
    )
    snapshot = DECLARED.snapshot.model_copy(update={"programs": programs})
    return DECLARED.model_copy(update={"repository": repository, "snapshot": snapshot})


@pytest.fixture
def cli_root(tmp_path: Path) -> Path:
    """Return a project root that holds a copy of the declared configuration."""
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    return tmp_path


def harvest(
    config: AppConfig,
    project_root: Path,
    repository: FakeRepository,
    *,
    snapshot_id: str | None = None,
) -> HarvestReport:
    """Run the use case with the clock fixed at :data:`HARVEST_TIME`."""
    return harvest_snapshot(
        config, project_root, repository, snapshot_id=snapshot_id, clock=lambda: HARVEST_TIME
    )


def raw_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw"


def raw_listing(project_root: Path) -> list[str]:
    """Return the sorted names in ``data/raw``, hidden staging directories included."""
    directory = raw_dir(project_root)
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def raw_state(project_root: Path) -> dict[str, bytes | None]:
    """Map every path under ``data/raw`` to its bytes, or to ``None`` for a directory."""
    directory = raw_dir(project_root)
    if not directory.exists():
        return {}
    return {
        path.relative_to(directory).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(directory.rglob("*"))
    }


def read_records(snapshot_dir: Path) -> list[dict[str, Any]]:
    lines = (snapshot_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def read_manifest(snapshot_dir: Path) -> dict[str, Any]:
    return json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def occupy(directory: Path) -> None:
    """Create a snapshot directory that already holds a file, as an earlier run would."""
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_bytes(b"original")


def test_each_program_is_listed_from_its_own_collection(config: AppConfig, tmp_path: Path) -> None:
    repository = FakeRepository({SISTEMAS: [item(1), item(2)], MINAS: [item(3)]})

    report = harvest(config, tmp_path, repository)

    assert repository.calls == [
        ("list", SISTEMAS, YEARS),
        ("list", MINAS, YEARS),
        ("count", FACULTY, YEARS),
    ]
    records = read_records(report.snapshot_dir)
    assert [
        (record["uuid"], record["program_key"], record["collection_uuid"]) for record in records
    ] == [
        (str(item_uuid(1)), "sistemas", str(SISTEMAS)),
        (str(item_uuid(2)), "sistemas", str(SISTEMAS)),
        (str(item_uuid(3)), "minas", str(MINAS)),
    ]


def test_the_searches_cover_the_configured_issue_years(config: AppConfig, tmp_path: Path) -> None:
    snapshot = config.snapshot.model_copy(update={"year_start": 2023, "year_end": 2024})
    repository = FakeRepository({})

    report = harvest(config.model_copy(update={"snapshot": snapshot}), tmp_path, repository)

    assert {years for _, _, years in repository.calls} == {IssuedYears(first=2023, last=2024)}
    assert report.manifest.query_parameters["f.dateIssued"] == "[2023 TO 2024],equals"


def test_a_record_holds_the_item_its_program_its_collection_and_the_harvest_time(
    config: AppConfig, tmp_path: Path
) -> None:
    subjects = [entry("Gestión", language="es"), entry("Procesos", language="es", place=1)]
    harvested = item(1, extra={"dc.subject": subjects})

    report = harvest(config, tmp_path, FakeRepository({SISTEMAS: [harvested]}))

    [record] = read_records(report.snapshot_dir)
    public = {key: values for key, values in harvested.metadata.items() if not key.endswith(".dni")}
    assert record == {
        "uuid": str(harvested.uuid),
        "handle": "123456789/1001",
        "program_key": "sistemas",
        "collection_uuid": str(SISTEMAS),
        "harvested_at": "2026-10-02T15:04:05Z",
        "metadata": public,
    }
    assert list(record) == [
        "uuid",
        "handle",
        "program_key",
        "collection_uuid",
        "harvested_at",
        "metadata",
    ]
    assert list(record["metadata"]) == sorted(public)


def test_identity_numbers_never_reach_the_disk(config: AppConfig, tmp_path: Path) -> None:
    upper_case = item(2, extra={"local.juror.DNI": [entry("SYNTHETIC-DNI-JUROR-002")]})
    repository = FakeRepository({SISTEMAS: [item(1), upper_case], MINAS: [item(3)]})

    report = harvest(config, tmp_path, repository)

    written = {path.name: path.read_bytes() for path in report.snapshot_dir.iterdir()}
    assert sorted(written) == ["manifest.json", "metadata.jsonl"]
    assert not any(b"SYNTHETIC-DNI" in content for content in written.values())
    keys = {key for record in read_records(report.snapshot_dir) for key in record["metadata"]}
    assert not [key for key in keys if key.casefold().endswith(".dni")]
    assert {"dc.contributor.author", "dc.date.issued", "dc.title", "renati.type"} <= keys
    assert read_manifest(report.snapshot_dir)["dropped_metadata_keys"] == [
        "local.juror.DNI",
        "renati.advisor.dni",
        "renati.author.dni",
    ]


def test_the_manifest_counts_items_per_program_and_per_type(
    config: AppConfig, tmp_path: Path
) -> None:
    repository = FakeRepository(
        {
            SISTEMAS: [item(1), item(2, types=(SUFICIENCIA,)), item(3, types=())],
            MINAS: [item(4, types=(TESIS, SUFICIENCIA)), item(5)],
        }
    )

    manifest = read_manifest(harvest(config, tmp_path, repository).snapshot_dir)

    assert {key: program["items"] for key, program in manifest["programs"].items()} == {
        "sistemas": 3,
        "minas": 2,
    }
    assert manifest["total_items"] == 5
    # An item with several types counts once under each of them.
    assert manifest["items_by_type"] == {TESIS: 3, SUFICIENCIA: 2, "(none)": 1}


@pytest.mark.parametrize(
    ("issued", "future", "unparsed"),
    [
        ("2021-01-20", 0, 0),
        ("2026-10-02", 0, 0),
        ("2026-10-03", 1, 0),
        ("2026-12", 1, 0),
        ("2026-10", 0, 0),
        ("2026", 0, 0),
        ("2026-12-04T05:00:00Z", 1, 0),
        ("2027", 1, 0),
        ("sin fecha", 0, 1),
        ("2026-13-01", 0, 1),
        (None, 0, 1),
    ],
    ids=[
        "past-day",
        "harvest-day",
        "next-day",
        "later-month",
        "harvest-month",
        "harvest-year",
        "later-timestamp",
        "later-year",
        "text",
        "impossible-month",
        "missing",
    ],
)
def test_items_issued_after_the_harvest_date_are_counted(
    config: AppConfig, tmp_path: Path, issued: str | None, future: int, unparsed: int
) -> None:
    repository = FakeRepository({SISTEMAS: [item(1, issued=issued)]})

    manifest = read_manifest(harvest(config, tmp_path, repository).snapshot_dir)

    assert (manifest["future_dated_items"], manifest["unparsed_issue_dates"]) == (future, unparsed)


def test_the_manifest_records_the_source_the_query_and_the_metadata_digest(
    config: AppConfig, tmp_path: Path
) -> None:
    repository = FakeRepository({SISTEMAS: [item(1), item(2)], MINAS: [item(3)]}, faculty_total=4)

    report = harvest(config, tmp_path, repository)

    manifest = read_manifest(report.snapshot_dir)
    content = (report.snapshot_dir / "metadata.jsonl").read_bytes()
    assert manifest == {
        "snapshot_id": SNAPSHOT_ID,
        "harvested_at": "2026-10-02T15:04:05Z",
        "base_url": "https://repo.example.edu",
        "search_url": SEARCH_URL,
        "query_parameters": {
            "dsoType": "ITEM",
            "size": "100",
            "sort": "dc.date.issued,ASC",
            "f.dateIssued": "[2021 TO 2026],equals",
        },
        "programs": {
            "sistemas": {"collection_uuid": str(SISTEMAS), "items": 2},
            "minas": {"collection_uuid": str(MINAS), "items": 1},
        },
        "total_items": 3,
        "faculty_community_uuid": str(FACULTY),
        "faculty_total": 4,
        "items_by_type": {TESIS: 3},
        "future_dated_items": 0,
        "unparsed_issue_dates": 0,
        "dropped_metadata_keys": ["renati.advisor.dni", "renati.author.dni"],
        "metadata_file": "metadata.jsonl",
        "metadata_sha256": hashlib.sha256(content).hexdigest(),
    }
    assert list(manifest["programs"]) == ["sistemas", "minas"]
    assert report.manifest == SnapshotManifest.model_validate(manifest)


def test_the_metadata_file_holds_one_utf8_json_object_per_line(
    config: AppConfig, tmp_path: Path
) -> None:
    repository = FakeRepository({SISTEMAS: [item(1), item(2)], MINAS: [item(3)]})

    content = (harvest(config, tmp_path, repository).snapshot_dir / "metadata.jsonl").read_bytes()

    assert content.count(b"\n") == 3 and content.endswith(b"\n")
    assert b"\r" not in content
    assert "Título sintético 001".encode() in content  # UTF-8 text, not \u escapes.


@pytest.mark.parametrize(
    "existing",
    [occupy, Path.mkdir, lambda path: path.write_bytes(b"original")],
    ids=["directory-with-files", "empty-directory", "file"],
)
def test_an_existing_snapshot_is_refused_before_any_request(
    config: AppConfig, tmp_path: Path, existing: Callable[[Path], object]
) -> None:
    raw_dir(tmp_path).mkdir(parents=True)
    existing(raw_dir(tmp_path) / "kept")
    before = raw_state(tmp_path)
    repository = FakeRepository({SISTEMAS: [item(1)]})

    with pytest.raises(SnapshotExistsError, match="kept"):
        harvest(config, tmp_path, repository, snapshot_id="kept")

    assert repository.calls == []
    assert raw_state(tmp_path) == before


def test_a_snapshot_created_during_the_harvest_is_not_overwritten(
    config: AppConfig, tmp_path: Path
) -> None:
    rival = raw_dir(tmp_path) / SNAPSHOT_ID

    def occupy_once() -> None:
        if not rival.exists():
            occupy(rival)

    repository = FakeRepository({SISTEMAS: [item(1)]}, on_list=occupy_once)

    with pytest.raises(SnapshotExistsError, match=SNAPSHOT_ID):
        harvest(config, tmp_path, repository)

    assert raw_state(tmp_path) == {SNAPSHOT_ID: None, f"{SNAPSHOT_ID}/manifest.json": b"original"}


def test_the_snapshot_appears_complete_in_a_single_rename(
    config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_rename = os.rename
    seen: dict[str, Any] = {}

    def observe(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        seen["target_existed"] = Path(target).exists()
        seen["staged"] = {path.name: path.read_bytes() for path in Path(source).iterdir()}
        real_rename(source, target)

    monkeypatch.setattr(os, "rename", observe)

    report = harvest(config, tmp_path, FakeRepository({SISTEMAS: [item(1)]}))

    assert seen["target_existed"] is False
    assert sorted(seen["staged"]) == ["manifest.json", "metadata.jsonl"]
    assert seen["staged"] == {
        path.name: path.read_bytes() for path in report.snapshot_dir.iterdir()
    }
    assert raw_listing(tmp_path) == [SNAPSHOT_ID]  # No staging directory is left behind.


@pytest.mark.parametrize("interruption", [OSError("simulated rename failure"), KeyboardInterrupt()])
def test_an_interrupted_write_leaves_no_snapshot_and_no_partial_files(
    config: AppConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    def interrupt(source: object, target: object) -> None:
        raise interruption

    monkeypatch.setattr(os, "rename", interrupt)

    with pytest.raises(type(interruption)):
        harvest(config, tmp_path, FakeRepository({SISTEMAS: [item(1)]}))

    assert raw_listing(tmp_path) == []


def test_a_failed_search_writes_nothing(config: AppConfig, tmp_path: Path) -> None:
    repository = FakeRepository({SISTEMAS: [item(1)]}, failing_scope=MINAS)

    with pytest.raises(RepositoryError, match="HTTP 503"):
        harvest(config, tmp_path, repository)

    assert raw_listing(tmp_path) == []


def test_a_listing_short_of_the_reported_total_is_refused(
    config: AppConfig, tmp_path: Path
) -> None:
    repository = FakeRepository({SISTEMAS: [item(1), item(2)]}, reported={SISTEMAS: 3})

    with pytest.raises(HarvestError, match=r"'sistemas'.* 2 items.* reported 3"):
        harvest(config, tmp_path, repository)

    assert raw_listing(tmp_path) == []


@pytest.mark.parametrize(
    "listings",
    [{SISTEMAS: [item(1), item(1)]}, {SISTEMAS: [item(1)], MINAS: [item(1)]}],
    ids=["twice-in-one-collection", "in-two-collections"],
)
def test_an_item_listed_twice_is_refused(
    config: AppConfig, tmp_path: Path, listings: Mapping[UUID, Sequence[RepositoryItem]]
) -> None:
    with pytest.raises(HarvestError, match=str(item_uuid(1))):
        harvest(config, tmp_path, FakeRepository(listings))

    assert raw_listing(tmp_path) == []


def test_the_default_snapshot_id_is_the_utc_harvest_time(config: AppConfig, tmp_path: Path) -> None:
    report = harvest(config, tmp_path, FakeRepository({}))

    assert report.snapshot_dir == (raw_dir(tmp_path) / SNAPSHOT_ID).resolve()
    assert report.manifest.snapshot_id == SNAPSHOT_ID


def test_a_clock_in_another_time_zone_is_read_as_utc(config: AppConfig, tmp_path: Path) -> None:
    lima_time = HARVEST_TIME.astimezone(timezone(timedelta(hours=-5)))

    report = harvest_snapshot(config, tmp_path, FakeRepository({}), clock=lambda: lima_time)

    assert report.manifest.snapshot_id == SNAPSHOT_ID
    assert read_manifest(report.snapshot_dir)["harvested_at"] == "2026-10-02T15:04:05Z"


def test_a_clock_without_a_time_zone_is_rejected(config: AppConfig, tmp_path: Path) -> None:
    repository = FakeRepository({})

    with pytest.raises(ValueError, match="time zone"):
        harvest_snapshot(
            config, tmp_path, repository, clock=lambda: HARVEST_TIME.replace(tzinfo=None)
        )

    assert repository.calls == []


@pytest.mark.parametrize("snapshot_id", ["baseline", "2026-10-02_v1", "a", "x" * 64])
def test_a_chosen_snapshot_id_names_the_directory(
    config: AppConfig, tmp_path: Path, snapshot_id: str
) -> None:
    report = harvest(config, tmp_path, FakeRepository({}), snapshot_id=snapshot_id)

    assert report.snapshot_dir.name == snapshot_id
    assert read_manifest(report.snapshot_dir)["snapshot_id"] == snapshot_id


@pytest.mark.parametrize(
    "snapshot_id",
    [
        "",
        "../escape",
        "nested/id",
        "nested\\id",
        ".hidden",
        "trailing.",
        "C:",
        "with space",
        "x" * 65,
    ],
)
def test_a_snapshot_id_that_is_not_one_plain_name_is_rejected(
    config: AppConfig, tmp_path: Path, snapshot_id: str
) -> None:
    repository = FakeRepository({})

    with pytest.raises(ValueError, match="snapshot id"):
        harvest(config, tmp_path, repository, snapshot_id=snapshot_id)

    assert repository.calls == []
    assert raw_listing(tmp_path) == []
    assert not (tmp_path / "data" / "escape").exists()


def test_cli_writes_the_snapshot_and_prints_its_summary(
    cli_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second, *_ = DECLARED.snapshot.programs
    repository = FakeRepository(
        {
            first.collection_uuid: [item(1), item(2, issued="2026-12-04")],
            second.collection_uuid: [item(3, types=(SUFICIENCIA,))],
        }
    )

    code = main(
        ["--project-root", str(cli_root), "--snapshot-id", "cli-run"],
        repository=repository,
        clock=lambda: HARVEST_TIME,
    )

    out = capsys.readouterr().out
    snapshot_dir = cli_root / "data" / "raw" / "cli-run"
    assert code == 0
    assert sorted(path.name for path in snapshot_dir.iterdir()) == [
        "manifest.json",
        "metadata.jsonl",
    ]
    assert str(snapshot_dir.resolve()) in out
    counts = dict(re.findall(r"^  (\w+) +(\d+)$", out, flags=re.MULTILINE))
    assert counts == {
        "sistemas": "2",
        "industrial": "1",
        "electronica": "0",
        "mecanica": "0",
        "minas": "0",
        "total": "3",
    }
    assert re.search(rf"^  {re.escape(TESIS)} +2$", out, flags=re.MULTILINE)
    assert "Items issued after the harvest date: 1" in out
    assert "Dropped metadata keys: renati.advisor.dni, renati.author.dni" in out


@pytest.mark.parametrize(
    ("arguments", "failing_scope", "message"),
    [
        (["--snapshot-id", "taken"], None, "already exists"),
        (["--snapshot-id", "../escape"], None, "snapshot id"),
        ([], DECLARED.snapshot.programs[1].collection_uuid, "HTTP 503"),
    ],
    ids=["existing-snapshot", "unsafe-snapshot-id", "repository-failure"],
)
def test_cli_reports_a_failure_on_one_error_line_with_exit_code_1(
    cli_root: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    failing_scope: UUID | None,
    message: str,
) -> None:
    occupy(raw_dir(cli_root) / "taken")
    repository = FakeRepository({}, failing_scope=failing_scope)

    code = main(
        ["--project-root", str(cli_root), *arguments],
        repository=repository,
        clock=lambda: HARVEST_TIME,
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert message in captured.err
    assert len(captured.err.splitlines()) == 1
    assert captured.out == ""
    assert raw_listing(cli_root) == ["taken"]


def test_cli_harvests_recorded_pages_through_the_dspace_adapter(
    cli_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sistemas = str(DECLARED.snapshot.programs[0].collection_uuid)
    faculty = str(DECLARED.repository.faculty_community_uuid)

    def serve(request: httpx.Request) -> httpx.Response:
        scope = request.url.params["scope"]
        if scope == sistemas:
            payload = fixture(f"search_page_{request.url.params['page']}.json")
        elif scope == faculty:
            payload = fixture("search_page_0.json")
        else:
            payload = fixture("search_empty.json")
        payload["scope"] = scope  # A real server echoes the scope that it searched.
        return httpx.Response(200, json=payload)

    sleeps: list[float] = []
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        adapter = DSpaceRestRepository.from_config(
            client, DECLARED.repository, clock=lambda: 0.0, sleeper=sleeps.append
        )
        code = main(
            ["--project-root", str(cli_root), "--snapshot-id", "recorded"],
            repository=adapter,
            clock=lambda: HARVEST_TIME,
        )

    out = capsys.readouterr().out
    snapshot_dir = cli_root / "data" / "raw" / "recorded"
    manifest = read_manifest(snapshot_dir)
    assert code == 0
    assert [record["handle"] for record in read_records(snapshot_dir)] == [
        f"123456789/{1000 + number}" for number in range(1, 6)
    ]
    assert {record["program_key"] for record in read_records(snapshot_dir)} == {"sistemas"}
    assert not any(b"SYNTHETIC-DNI" in path.read_bytes() for path in snapshot_dir.iterdir())
    assert manifest["dropped_metadata_keys"] == ["renati.advisor.dni", "renati.author.dni"]
    assert manifest["items_by_type"] == {TESIS: 4, SUFICIENCIA: 1}
    assert (manifest["total_items"], manifest["faculty_total"]) == (5, 5)
    assert manifest["future_dated_items"] == 1  # Issued 2026-12-04, after the harvest date.
    assert manifest["search_url"] == (
        "https://repositorio.ucsm.edu.pe/server/api/discover/search/objects"
    )
    # Three pages for sistemas, one empty page for each other program, one faculty count.
    assert adapter.request_count == 8
    assert sleeps == [1.0] * 7
    assert "HTTP requests: 8" in out
