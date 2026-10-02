"""Behavior of the OCR assets step, which fetches and verifies the Tesseract language files.

A fake downloader stands in for the network, and the HTTP adapter runs over
``httpx.MockTransport``, so no test opens a connection.
"""

import hashlib
import io
import json
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from thematic_redundancy.extraction.ocr_assets import (
    MIN_TRAINEDDATA_BYTES,
    ByteSink,
    HttpxDownloader,
    TessdataDownloadError,
    TessdataReport,
    ensure_tessdata,
    language_codes,
    main,
)
from thematic_redundancy.shared.config import AppConfig, OcrConfig, PathsConfig, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

SPA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata"
ENG_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"
FAST_SPA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/spa.traineddata"
RAW_SPA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/spa.traineddata"

DOWNLOAD_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
LATER_TIME = datetime(2026, 10, 3, 8, 30, 0, tzinfo=UTC)

SMALL_FLOOR = 1_024
"""Size floor for direct calls, which keeps the fake payloads small."""


def payload_for(url: str, size: int) -> bytes:
    """Return ``size`` deterministic bytes that differ from one URL to another."""
    block = hashlib.sha256(url.encode()).digest()
    return (block * (size // len(block) + 1))[:size]


class FakeDownloader:
    """Serve one payload per URL and record every request; a failing URL breaks midway."""

    def __init__(self, payloads: dict[str, bytes], failing: tuple[str, ...] = ()) -> None:
        self.payloads = payloads
        self.failing = failing
        self.requests: list[str] = []

    def __call__(self, url: str, sink: ByteSink, /) -> None:
        self.requests.append(url)
        payload = self.payloads[url]
        if url in self.failing:
            sink.write(payload[: len(payload) // 2])
            raise ConnectionResetError(f"connection reset while fetching {url}")
        sink.write(payload)


@pytest.fixture
def config() -> AppConfig:
    """Return the declared configuration: languages ``spa+eng``, files under ``tessdata/``."""
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def payloads() -> dict[str, bytes]:
    """Return distinct payloads, above :data:`SMALL_FLOOR`, for both configured files."""
    return {SPA_URL: payload_for(SPA_URL, 4_096), ENG_URL: payload_for(ENG_URL, 3_000)}


@pytest.fixture
def cli_root(tmp_path: Path) -> Path:
    """Return a project root that holds a copy of the declared configuration."""
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    return tmp_path


def fetch(
    config: AppConfig,
    project_root: Path,
    downloader: FakeDownloader,
    *,
    force: bool = False,
    at: datetime = DOWNLOAD_TIME,
) -> TessdataReport:
    """Run the step with the small size floor and a clock fixed at ``at``."""
    return ensure_tessdata(
        config, project_root, downloader, force=force, min_bytes=SMALL_FLOOR, clock=lambda: at
    )


def listing(directory: Path) -> list[str]:
    """Return the sorted names in ``directory``, temporary files included."""
    return sorted(path.name for path in directory.iterdir())


def statuses(output: str) -> dict[str, str]:
    """Map each file named in a command-line summary to the status printed beside it."""
    return dict(re.findall(r"(\w+\.traineddata)\s+(downloaded|already present)", output))


def rewrite_manifest(tessdata_dir: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    """Load ``manifest.json``, apply ``edit`` to its data, and save it back."""
    path = tessdata_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def alter_spanish_file(tessdata_dir: Path) -> None:
    path = tessdata_dir / "spa.traineddata"
    path.write_bytes(bytes(path.stat().st_size))  # Same size, different content.


def truncate_spanish_file(tessdata_dir: Path) -> None:
    path = tessdata_dir / "spa.traineddata"
    path.write_bytes(path.read_bytes()[:-1])


def delete_spanish_file(tessdata_dir: Path) -> None:
    (tessdata_dir / "spa.traineddata").unlink()


def record_spanish_file_from_another_source(tessdata_dir: Path) -> None:
    rewrite_manifest(
        tessdata_dir, lambda data: data["files"]["spa.traineddata"].update(source_url=FAST_SPA_URL)
    )


def delete_manifest(tessdata_dir: Path) -> None:
    (tessdata_dir / "manifest.json").unlink()


def corrupt_manifest(tessdata_dir: Path) -> None:
    (tessdata_dir / "manifest.json").write_bytes(b"\xff{not json")


def record_another_variant(tessdata_dir: Path) -> None:
    rewrite_manifest(tessdata_dir, lambda data: data.update(variant="tessdata_fast"))


@pytest.mark.parametrize(
    ("spec", "codes"),
    [("spa+eng", ("spa", "eng")), ("spa", ("spa",)), ("chi_sim+eng", ("chi_sim", "eng"))],
)
def test_language_codes_follow_the_order_of_the_tesseract_spec(
    spec: str, codes: tuple[str, ...]
) -> None:
    assert language_codes(spec) == codes


@pytest.mark.parametrize("spec", ["", "spa+", "+eng", "spa++eng", "spa eng", "../spa", "spa+spa"])
def test_malformed_or_repeated_language_codes_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError, match="language code"):
        language_codes(spec)


def test_configured_files_are_downloaded_in_order_into_the_tessdata_dir(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    downloader = FakeDownloader(payloads)

    report = fetch(config, tmp_path, downloader)

    tessdata_dir = tmp_path / "tessdata"
    assert downloader.requests == [SPA_URL, ENG_URL]
    assert report.downloaded == ("spa", "eng")
    assert report.tessdata_dir == tessdata_dir.resolve()
    assert (tessdata_dir / "spa.traineddata").read_bytes() == payloads[SPA_URL]
    assert (tessdata_dir / "eng.traineddata").read_bytes() == payloads[ENG_URL]
    assert listing(tessdata_dir) == ["eng.traineddata", "manifest.json", "spa.traineddata"]


def test_languages_and_directory_come_from_the_config(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    spanish_only = config.model_copy(
        update={
            "ocr": OcrConfig(languages="spa", dpi=config.ocr.dpi),
            "paths": PathsConfig(
                data_dir=Path("data"),
                results_dir=Path("results"),
                tessdata_dir=Path("assets") / "ocr",
            ),
        }
    )
    downloader = FakeDownloader(payloads)

    report = fetch(spanish_only, tmp_path, downloader)

    assert downloader.requests == [SPA_URL]
    assert report.tessdata_dir == (tmp_path / "assets" / "ocr").resolve()
    assert listing(report.tessdata_dir) == ["manifest.json", "spa.traineddata"]
    assert report.manifest.languages == ("spa",)


def test_manifest_records_the_variant_the_languages_and_each_file(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    fetch(config, tmp_path, FakeDownloader(payloads))

    manifest = json.loads((tmp_path / "tessdata" / "manifest.json").read_text(encoding="utf-8"))

    def record(url: str) -> dict[str, object]:
        return {
            "source_url": url,
            "sha256": hashlib.sha256(payloads[url]).hexdigest(),
            "size_bytes": len(payloads[url]),
            "downloaded_at": "2026-10-02T15:04:05Z",
        }

    assert manifest == {
        "variant": "tessdata_best",
        "languages": ["spa", "eng"],
        "files": {"spa.traineddata": record(SPA_URL), "eng.traineddata": record(ENG_URL)},
    }


def test_files_that_match_the_manifest_are_not_downloaded_again(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    fetch(config, tmp_path, FakeDownloader(payloads))
    manifest_path = tmp_path / "tessdata" / "manifest.json"
    first_manifest = manifest_path.read_bytes()
    second_run = FakeDownloader(payloads)

    report = fetch(config, tmp_path, second_run, at=LATER_TIME)

    assert second_run.requests == []
    assert report.downloaded == ()
    assert manifest_path.read_bytes() == first_manifest  # The first download times are kept.


def test_force_downloads_every_file_again_and_records_the_new_time(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    fetch(config, tmp_path, FakeDownloader(payloads))
    forced_run = FakeDownloader(payloads)

    report = fetch(config, tmp_path, forced_run, force=True, at=LATER_TIME)

    assert forced_run.requests == [SPA_URL, ENG_URL]
    assert report.downloaded == ("spa", "eng")
    assert {record.downloaded_at for record in report.manifest.files.values()} == {LATER_TIME}


def test_a_shorter_language_list_updates_the_manifest_without_downloading(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    fetch(config, tmp_path, FakeDownloader(payloads))
    spanish_only = config.model_copy(update={"ocr": OcrConfig(languages="spa", dpi=config.ocr.dpi)})
    later_run = FakeDownloader(payloads)

    report = fetch(spanish_only, tmp_path, later_run)

    saved = json.loads((tmp_path / "tessdata" / "manifest.json").read_text(encoding="utf-8"))
    assert later_run.requests == []
    assert report.manifest.languages == ("spa",)
    assert (saved["languages"], list(saved["files"])) == (["spa"], ["spa.traineddata"])


@pytest.mark.parametrize(
    ("tamper", "expected_requests"),
    [
        pytest.param(alter_spanish_file, [SPA_URL], id="file-altered"),
        pytest.param(truncate_spanish_file, [SPA_URL], id="file-truncated"),
        pytest.param(delete_spanish_file, [SPA_URL], id="file-deleted"),
        pytest.param(record_spanish_file_from_another_source, [SPA_URL], id="other-source"),
        pytest.param(delete_manifest, [SPA_URL, ENG_URL], id="manifest-missing"),
        pytest.param(corrupt_manifest, [SPA_URL, ENG_URL], id="manifest-corrupt"),
        pytest.param(record_another_variant, [SPA_URL, ENG_URL], id="other-variant"),
    ],
)
def test_files_that_no_longer_match_the_manifest_are_downloaded_again(
    config: AppConfig,
    payloads: dict[str, bytes],
    tmp_path: Path,
    tamper: Callable[[Path], None],
    expected_requests: list[str],
) -> None:
    tessdata_dir = tmp_path / "tessdata"
    fetch(config, tmp_path, FakeDownloader(payloads))
    tamper(tessdata_dir)
    repair_run = FakeDownloader(payloads)

    fetch(config, tmp_path, repair_run)

    assert repair_run.requests == expected_requests
    assert (tessdata_dir / "spa.traineddata").read_bytes() == payloads[SPA_URL]
    assert (tessdata_dir / "eng.traineddata").read_bytes() == payloads[ENG_URL]
    settled_run = FakeDownloader(payloads)
    fetch(config, tmp_path, settled_run)
    assert settled_run.requests == []


def test_an_undersized_download_is_rejected_and_leaves_no_file(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    payloads[SPA_URL] = payload_for(SPA_URL, SMALL_FLOOR - 1)

    with pytest.raises(TessdataDownloadError, match="bytes"):
        fetch(config, tmp_path, FakeDownloader(payloads))

    assert listing(tmp_path / "tessdata") == []


def test_an_interrupted_download_leaves_no_partial_file(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    with pytest.raises(ConnectionResetError):
        fetch(config, tmp_path, FakeDownloader(payloads, failing=(SPA_URL,)))

    assert listing(tmp_path / "tessdata") == []


def test_a_failed_forced_download_keeps_the_previous_files(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    fetch(config, tmp_path, FakeDownloader(payloads))
    tessdata_dir = tmp_path / "tessdata"
    before = {name: (tessdata_dir / name).read_bytes() for name in listing(tessdata_dir)}

    with pytest.raises(ConnectionResetError):
        fetch(config, tmp_path, FakeDownloader(payloads, failing=(SPA_URL,)), force=True)

    assert {name: (tessdata_dir / name).read_bytes() for name in listing(tessdata_dir)} == before


def test_files_fetched_before_a_failure_stay_recorded(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    with pytest.raises(ConnectionResetError):
        fetch(config, tmp_path, FakeDownloader(payloads, failing=(ENG_URL,)))
    assert listing(tmp_path / "tessdata") == ["manifest.json", "spa.traineddata"]
    retry_run = FakeDownloader(payloads)

    report = fetch(config, tmp_path, retry_run)

    assert retry_run.requests == [ENG_URL]
    assert report.downloaded == ("eng",)


def test_httpx_downloader_follows_the_redirect_and_streams_the_body() -> None:
    body = payload_for(RAW_SPA_URL, 200_000)
    seen: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": RAW_SPA_URL})
        return httpx.Response(200, content=body)

    sink = io.BytesIO()
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        HttpxDownloader(client)(SPA_URL, sink)

    assert seen == [SPA_URL, RAW_SPA_URL]
    assert sink.getvalue() == body


@pytest.mark.parametrize("status", [404, 500])
def test_httpx_downloader_rejects_any_status_other_than_ok(status: int) -> None:
    sink = io.BytesIO()
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=b"<html/>"))

    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TessdataDownloadError, match=f"HTTP {status}"),
    ):
        HttpxDownloader(client)(SPA_URL, sink)

    assert sink.getvalue() == b""


def test_httpx_downloader_reports_a_network_failure_with_the_url() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(refuse)) as client,
        pytest.raises(TessdataDownloadError, match=re.escape(SPA_URL)),
    ):
        HttpxDownloader(client)(SPA_URL, io.BytesIO())


def test_cli_downloads_once_and_again_only_when_forced(
    cli_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    full_size = {url: payload_for(url, MIN_TRAINEDDATA_BYTES) for url in (SPA_URL, ENG_URL)}
    downloader = FakeDownloader(full_size)
    arguments = ["--project-root", str(cli_root)]

    assert main(arguments, downloader=downloader) == 0
    first = capsys.readouterr().out
    assert main(arguments, downloader=downloader) == 0
    second = capsys.readouterr().out
    assert main([*arguments, "--force"], downloader=downloader) == 0
    forced = capsys.readouterr().out

    assert downloader.requests == [SPA_URL, ENG_URL, SPA_URL, ENG_URL]
    fetched = {"spa.traineddata": "downloaded", "eng.traineddata": "downloaded"}
    assert statuses(first) == fetched
    assert statuses(second) == dict.fromkeys(fetched, "already present")
    assert statuses(forced) == fetched
    assert hashlib.sha256(full_size[SPA_URL]).hexdigest() in first


def test_cli_rejects_an_undersized_download_with_an_error_exit(
    cli_root: Path, payloads: dict[str, bytes], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--project-root", str(cli_root)], downloader=FakeDownloader(payloads))

    error = capsys.readouterr().err
    assert exit_code == 1
    assert error.startswith("error:")
    assert SPA_URL in error
    assert listing(cli_root / "tessdata") == []
