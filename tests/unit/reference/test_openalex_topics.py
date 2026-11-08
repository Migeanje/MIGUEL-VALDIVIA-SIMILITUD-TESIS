"""Behavior of the one-time download of the OpenAlex Topics list, through its command line.

Every test runs :func:`main` on a project root that holds a copy of the declared configuration,
with ``openalex.per_page`` lowered to 2 so that a few synthetic topics span several pages. A
fake transport serves the pages (:mod:`openalex_fakes`), a fixed clock sets the retrieval time,
a fake git reader gives the code state and a fake monotonic clock stands in for time, so no test
opens a connection, runs git or sleeps.
"""

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from openalex_fakes import FakeServer, ServerFactory, numbered_pages, page, topic, topics

from thematic_redundancy.corpus.snapshot import SnapshotManifest, latest_snapshot_id
from thematic_redundancy.reference import openalex_topics as command
from thematic_redundancy.reference.openalex import USER_AGENT
from thematic_redundancy.reference.openalex_topics import main
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.topic_map.run_experiment import GitState

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
PER_PAGE = 2
RETRIEVED_AT = datetime(2026, 11, 5, 14, 3, 9, tzinfo=UTC)
DOWNLOAD_ID = "20261105T140309Z"
"""Default download id for :data:`RETRIEVED_AT`."""
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TOPICS_URL = "https://api.openalex.org/topics"


class FakeGit:
    """Stands in for ``read_git_state``: records its calls and returns ``state``."""

    def __init__(self, state: GitState | None = None, error: Exception | None = None) -> None:
        self.state = state or GitState(commit=COMMIT, dirty=())
        self.error = error
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, project_root: Path, results_path: str) -> GitState:
        self.calls.append((project_root, results_path))
        if self.error is not None:
            raise self.error
        return self.state


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Return a project root holding the declared configuration with ``per_page`` 2."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    declared = "  per_page: 200\n"
    assert text.count(declared) == 1
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "default.yaml").write_bytes(
        text.replace(declared, f"  per_page: {PER_PAGE}\n").encode()
    )
    return tmp_path


def run(
    root: Path,
    fake: FakeServer,
    *arguments: str,
    clock: Any = lambda: RETRIEVED_AT,
    git: FakeGit | None = None,
) -> int:
    return main(
        ["--project-root", str(root), *arguments],
        transport=httpx.MockTransport(fake),
        clock=clock,
        git=git or FakeGit(),
        monotonic=fake.time.clock,
        sleeper=fake.time.sleep,
    )


def data_file(root: Path, download_id: str = DOWNLOAD_ID) -> Path:
    return root / "data" / "raw" / "openalex" / "topics" / download_id / "topics.jsonl"


def manifest_file(root: Path, download_id: str = DOWNLOAD_ID) -> Path:
    return root / "results" / "reference" / "openalex" / "topics" / download_id / "manifest.json"


def written(root: Path) -> list[str]:
    """Return every file under ``data/`` and ``results/``, run locks left out."""
    return sorted(
        path.relative_to(root).as_posix()
        for top in ("data", "results")
        for path in (root / top).rglob("*")
        if path.is_file() and path.name != ".lock"
    )


def read_manifest(root: Path, download_id: str = DOWNLOAD_ID) -> dict[str, Any]:
    return json.loads(manifest_file(root, download_id).read_text(encoding="utf-8"))


def test_the_download_writes_every_topic_on_its_own_line_as_served(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    records = topics(5)
    fake = server(by_page=numbered_pages(records, PER_PAGE))

    code = run(root, fake)

    out = capsys.readouterr().out
    assert code == 0
    assert written(root) == [
        f"data/raw/openalex/topics/{DOWNLOAD_ID}/topics.jsonl",
        f"results/reference/openalex/topics/{DOWNLOAD_ID}/manifest.json",
    ]
    content = data_file(root).read_bytes()
    assert b"\r" not in content
    assert content.endswith(b"\n")
    lines = content.decode("utf-8").split("\n")[:-1]
    assert [json.loads(line) for line in lines] == records
    assert [list(json.loads(line)) for line in lines] == [list(record) for record in records]
    assert "diseño de señales pequeñas" in lines[0]  # UTF-8 text, not \u escapes.
    assert fake.sent("page") == ["1", "2", "3"]
    assert f"OpenAlex Topics download {DOWNLOAD_ID}" in out
    assert "Topics: 5 written, 5 distinct ids, meta.count 5" in out
    assert "Pages: 3; HTTP requests: 3" in out
    assert str(data_file(root).resolve()) in out
    assert str(manifest_file(root).resolve()) in out


def test_the_manifest_records_the_source_the_counts_the_digests_and_the_code(
    root: Path, server: ServerFactory
) -> None:
    fake = server(by_page=numbered_pages(topics(3), PER_PAGE))
    git = FakeGit(GitState(commit=COMMIT, dirty=("README.md", "notas inventadas.txt")))

    assert run(root, fake, git=git) == 0

    manifest = read_manifest(root)
    content = data_file(root).read_bytes()
    assert manifest["download_id"] == DOWNLOAD_ID
    assert manifest["retrieved_at"] == "2026-11-05T14:03:09Z"
    assert manifest["base_url"] == "https://api.openalex.org"
    assert manifest["url"] == TOPICS_URL
    assert manifest["query_parameters"] == {"per-page": "2", "page": "1"}
    assert manifest["user_agent"] == USER_AGENT
    assert (manifest["meta_count"], manifest["records_written"], manifest["distinct_ids"]) == (
        3,
        3,
        3,
    )
    assert (manifest["pages"], manifest["requests"]) == (2, 2)
    assert manifest["data_files"] == {
        "topics.jsonl": {
            "path": f"data/raw/openalex/topics/{DOWNLOAD_ID}/topics.jsonl",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    }
    config_bytes = (root / "config" / "default.yaml").read_bytes()
    assert manifest["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert manifest["settings"] == {
        "base_url": "https://api.openalex.org/",
        "per_page": 2,
        "request_interval_seconds": 1.0,
        "timeout_seconds": 30.0,
        "max_retries": 3,
        "backoff_seconds": 2.0,
    }
    assert (manifest["git_commit"], manifest["git_dirty"]) == (
        COMMIT,
        ["README.md", "notas inventadas.txt"],
    )
    assert manifest["versions"] == {
        "python": manifest["versions"]["python"],
        "httpx": version("httpx"),
        "pydantic": version("pydantic"),
    }
    assert manifest["seconds"] == pytest.approx(1.25)  # 2 requests 1 s apart, 0.25 s each.
    assert git.calls == [
        (root.resolve(), f"results/reference/openalex/topics/{DOWNLOAD_ID}"),
    ]
    raw = manifest_file(root).read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n")


def test_the_summary_names_a_dirty_tree_that_the_manifest_records(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server(by_page=numbered_pages(topics(1), PER_PAGE))

    code = run(root, fake, git=FakeGit(GitState(commit=COMMIT, dirty=("README.md",))))

    assert code == 0
    assert f"Git: commit {COMMIT}, 1 changed path (recorded in the manifest)" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("moment", "download_id"),
    [
        (RETRIEVED_AT, DOWNLOAD_ID),
        (datetime(2026, 11, 5, 9, 3, 9, tzinfo=timezone(timedelta(hours=-5))), DOWNLOAD_ID),
    ],
    ids=["utc", "lima-time"],
)
def test_the_default_download_id_is_the_utc_retrieval_time(
    root: Path, server: ServerFactory, moment: datetime, download_id: str
) -> None:
    fake = server(by_page=numbered_pages(topics(1), PER_PAGE))

    assert run(root, fake, clock=lambda: moment) == 0

    assert data_file(root, download_id).is_file()
    assert read_manifest(root, download_id)["retrieved_at"] == "2026-11-05T14:03:09Z"


def test_a_chosen_download_id_names_both_folders(root: Path, server: ServerFactory) -> None:
    fake = server(by_page=numbered_pages(topics(1), PER_PAGE))

    assert run(root, fake, "--download-id", "topics-2026") == 0

    assert data_file(root, "topics-2026").is_file()
    assert read_manifest(root, "topics-2026")["download_id"] == "topics-2026"


def test_a_clock_without_a_time_zone_is_refused_before_any_request(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()

    code = run(root, fake, clock=lambda: datetime(2026, 11, 5, 14, 3, 9))

    assert code == 1
    assert "time zone" in capsys.readouterr().err
    assert fake.requests == []


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        (numbered_pages(topics(3), PER_PAGE, count=4), "holds 3 topics, but OpenAlex reported 4"),
        (
            numbered_pages([*topics(2), topic(1)], PER_PAGE),
            "https://openalex.org/T90001 appears 2 times",
        ),
    ],
    ids=["short-of-meta-count", "duplicate-id"],
)
def test_an_incomplete_listing_is_refused_and_nothing_is_written(
    root: Path,
    server: ServerFactory,
    capsys: pytest.CaptureFixture[str],
    pages: list[dict[str, Any]],
    message: str,
) -> None:
    fake = server(by_page=pages)

    code = run(root, fake)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert message in captured.err
    assert "nothing was written" in captured.err
    assert len(captured.err.splitlines()) == 1
    assert captured.out == ""
    assert written(root) == []
    assert not data_file(root).parent.exists()
    assert not manifest_file(root).parent.exists()


@pytest.mark.parametrize("existing", ["data", "manifest"])
def test_an_existing_download_id_is_refused_before_any_request(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str], existing: str
) -> None:
    target = data_file(root) if existing == "data" else manifest_file(root)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original")
    fake = server()

    code = run(root, fake, "--download-id", DOWNLOAD_ID)

    captured = capsys.readouterr()
    assert code == 1
    assert "already exists" in captured.err
    assert "--download-id" in captured.err
    assert fake.requests == []
    assert target.read_bytes() == b"original"
    assert written(root) == [target.relative_to(root).as_posix()]


@pytest.mark.parametrize("download_id", ["../escape", ".hidden", "a/b", ""])
def test_a_download_id_that_is_not_one_plain_name_is_refused(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str], download_id: str
) -> None:
    fake = server()

    code = run(root, fake, "--download-id", download_id)

    assert code == 1
    assert "download id" in capsys.readouterr().err
    assert fake.requests == []
    assert not (root / "data").exists()
    assert not (root / "results").exists()


def test_one_download_runs_at_a_time(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    lock_dir = root / "data" / "raw" / "openalex" / "topics"
    lock_dir.mkdir(parents=True)
    fake = server()

    with exclusive_run(lock_dir, RuntimeError("held by the test")):
        code = run(root, fake)

    assert code == 1
    assert "another download holds the lock" in capsys.readouterr().err
    assert fake.requests == []
    assert written(root) == []


def test_a_git_state_that_cannot_be_read_stops_before_any_request(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = server()

    code = run(root, fake, git=FakeGit(error=OSError("git rev-parse failed: not a repository")))

    assert code == 1
    assert "not a repository" in capsys.readouterr().err
    assert fake.requests == []
    assert written(root) == []


@pytest.mark.parametrize("status", [401, 403, 429])
def test_a_refusal_names_its_status_and_nothing_is_written(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    fake = server([httpx.Response(status)] * 4)

    code = run(root, fake)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert f"HTTP {status}" in captured.err
    assert "no email, API key or cookie" in captured.err
    assert len(captured.err.splitlines()) == 1
    assert len(fake.requests) == (4 if status == 429 else 1)  # max_retries is 3.
    assert written(root) == []


def test_an_interruption_during_the_download_exits_130_and_writes_nothing(
    root: Path, server: ServerFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = numbered_pages(topics(5), PER_PAGE)
    fake = server([pages[0], KeyboardInterrupt()], by_page=pages)

    code = run(root, fake)

    captured = capsys.readouterr()
    assert code == 130
    assert captured.err.startswith("interrupted: ")
    assert "nothing was written" in captured.err
    assert written(root) == []
    assert not data_file(root).parent.exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("disk full")])
def test_a_manifest_that_cannot_be_written_takes_the_data_file_back(
    root: Path,
    server: ServerFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    def fail(path: Path, text: str) -> None:
        assert data_file(root).is_file()  # The data file is written first.
        raise error

    monkeypatch.setattr(command, "write_text_atomically", fail)
    fake = server(by_page=numbered_pages(topics(3), PER_PAGE))

    code = run(root, fake)

    err = capsys.readouterr().err
    assert code == (130 if isinstance(error, KeyboardInterrupt) else 1)
    assert "nothing was written" in err
    assert written(root) == []
    assert not data_file(root).parent.exists()
    assert not manifest_file(root).parent.exists()


def test_a_second_download_needs_a_new_id_and_leaves_the_first_alone(
    root: Path, server: ServerFactory
) -> None:
    pages = numbered_pages(topics(3), PER_PAGE)
    assert run(root, server(by_page=pages)) == 0
    first = {path: (root / path).read_bytes() for path in written(root)}

    assert run(root, server(by_page=pages)) == 1
    later = datetime(2026, 11, 6, tzinfo=UTC)
    assert run(root, server(by_page=pages), clock=lambda: later) == 0

    assert {path: (root / path).read_bytes() for path in first} == first
    assert data_file(root, "20261106T000000Z").read_bytes() == data_file(root).read_bytes()


def test_a_snapshot_beside_a_download_is_still_the_latest_snapshot(
    root: Path, server: ServerFactory
) -> None:
    snapshot = root / "data" / "raw" / "20261002T224412Z"
    snapshot.mkdir(parents=True)
    manifest = SnapshotManifest(
        snapshot_id="20261002T224412Z",
        harvested_at=datetime(2026, 10, 2, 22, 44, 12, tzinfo=UTC),
        base_url="https://repo.example.edu",
        search_url="https://repo.example.edu/server/api/discover/search/objects",
        query_parameters={},
        programs={},
        total_items=0,
        faculty_community_uuid=UUID("00000000-0000-4000-8000-0000000000f0"),
        faculty_total=0,
        items_by_type={},
        future_dated_items=0,
        unparsed_issue_dates=0,
        dropped_metadata_keys=(),
        metadata_sha256="0" * 64,
    )
    (snapshot / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    assert run(root, server(by_page=numbered_pages(topics(1), PER_PAGE))) == 0

    assert latest_snapshot_id(root / "data" / "raw") == "20261002T224412Z"


def test_an_openalex_download_alone_is_not_a_snapshot(root: Path, server: ServerFactory) -> None:
    assert run(root, server(by_page=numbered_pages(topics(1), PER_PAGE))) == 0

    with pytest.raises(FileNotFoundError, match="no snapshot"):
        latest_snapshot_id(root / "data" / "raw")


def test_the_help_describes_the_download_and_its_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])

    out = " ".join(capsys.readouterr().out.split())
    assert caught.value.code == 0
    assert "--project-root" in out
    assert "--download-id" in out
    assert "OpenAlex Topics" in out
    assert re.search(r"no (e-?mail|personal data)", out)
    assert "authoriz" in out


def test_an_empty_list_matching_its_count_is_written(root: Path, server: ServerFactory) -> None:
    fake = server([page([], count=0)])

    assert run(root, fake) == 0

    assert data_file(root).read_bytes() == b""
    assert read_manifest(root)["records_written"] == 0


def test_a_data_file_that_cannot_be_taken_back_is_named(
    root: Path,
    server: ServerFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_unlink = Path.unlink

    def held_open(self: Path, missing_ok: bool = False) -> None:
        if self.name == "topics.jsonl":
            raise PermissionError("held open by another program")
        real_unlink(self, missing_ok=missing_ok)

    def fail(path: Path, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "unlink", held_open)
    monkeypatch.setattr(command, "write_text_atomically", fail)
    fake = server(by_page=numbered_pages(topics(1), PER_PAGE))

    code = run(root, fake)

    err = capsys.readouterr().err
    assert code == 1
    assert "disk full" in err
    assert str(data_file(root).resolve()) in " ".join(err.split())
    assert "move them away" in err
    assert written(root) == [f"data/raw/openalex/topics/{DOWNLOAD_ID}/topics.jsonl"]


def test_an_interruption_after_the_manifest_keeps_the_complete_download(
    root: Path,
    server: ServerFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_write = command.write_text_atomically

    def write_then_interrupt(path: Path, text: str) -> None:
        real_write(path, text)
        raise KeyboardInterrupt

    monkeypatch.setattr(command, "write_text_atomically", write_then_interrupt)
    fake = server(by_page=numbered_pages(topics(3), PER_PAGE))

    code = run(root, fake)

    err = capsys.readouterr().err
    assert code == 130
    assert "complete before the interruption" in err
    assert written(root) == [
        f"data/raw/openalex/topics/{DOWNLOAD_ID}/topics.jsonl",
        f"results/reference/openalex/topics/{DOWNLOAD_ID}/manifest.json",
    ]
    assert read_manifest(root)["records_written"] == 3
