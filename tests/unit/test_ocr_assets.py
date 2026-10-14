"""Behavior of the OCR assets step, which fetches and verifies the Tesseract language files.

A fake downloader stands in for the network, and the HTTP adapter runs over
``httpx.MockTransport``, so no test opens a connection. Most tests pin the configured
languages to small fake payloads; a few use the real pins, which no fake payload matches.
"""

import hashlib
import io
import json
import re
import shutil
import struct
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from thematic_redundancy.extraction.ocr_assets import (
    PINNED_TRAINEDDATA,
    ByteSink,
    HttpxDownloader,
    PinnedFile,
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
POR_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/por.traineddata"
FAST_SPA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/spa.traineddata"
RAW_SPA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/spa.traineddata"

DOWNLOAD_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
LATER_TIME = datetime(2026, 10, 3, 8, 30, 0, tzinfo=UTC)


def payload_for(url: str, size: int) -> bytes:
    """Return ``size`` deterministic bytes that differ from one URL to another."""
    block = hashlib.sha256(url.encode()).digest()
    return (block * (size // len(block) + 1))[:size]


def pin_of(content: bytes) -> PinnedFile:
    return PinnedFile(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


PAYLOADS = {SPA_URL: payload_for(SPA_URL, 4_096), ENG_URL: payload_for(ENG_URL, 3_000)}
"""Distinct fake payloads for both configured files."""

PINS = {"spa": pin_of(PAYLOADS[SPA_URL]), "eng": pin_of(PAYLOADS[ENG_URL])}
"""Pins that the fake payloads match, in place of the real ones."""


def traineddata(
    size: int, *, entries: int = 24, offsets: Sequence[int] | None = None, header: bytes = b""
) -> bytes:
    """Return ``size`` bytes laid out as a Tesseract language file: an entry count, an offset
    table with two entries present by default, then filler. ``header`` replaces the start."""
    table_end = 4 + 8 * entries
    if offsets is None:
        offsets = [table_end, table_end + 100, *[-1] * (entries - 2)]
    start = header or struct.pack(f"<i{len(offsets)}q", entries, *offsets)
    return (start + payload_for(POR_URL, size))[:size]


class FakeDownloader:
    """Serve one payload per URL, in one write or in chunks of ``chunk_size`` bytes, and
    record every request; a failing URL breaks midway."""

    def __init__(
        self,
        payloads: dict[str, bytes],
        failing: tuple[str, ...] = (),
        chunk_size: int | None = None,
    ) -> None:
        self.payloads = payloads
        self.failing = failing
        self.chunk_size = chunk_size
        self.requests: list[str] = []

    def __call__(self, url: str, sink: ByteSink, /) -> None:
        self.requests.append(url)
        payload = self.payloads[url]
        if url in self.failing:
            sink.write(payload[: len(payload) // 2])
            raise ConnectionResetError(f"connection reset while fetching {url}")
        step = self.chunk_size or max(len(payload), 1)
        for start in range(0, len(payload), step):
            sink.write(payload[start : start + step])


@pytest.fixture
def config() -> AppConfig:
    """Return the declared configuration: languages ``spa+eng``, files under ``tessdata/``."""
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def payloads() -> dict[str, bytes]:
    """Return a fresh copy of :data:`PAYLOADS`, which a test may change."""
    return dict(PAYLOADS)


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
    pins: Mapping[str, PinnedFile] = PINS,
) -> TessdataReport:
    """Run the step with the fake pins and a clock fixed at ``at``."""
    return ensure_tessdata(
        config, project_root, downloader, force=force, pins=pins, clock=lambda: at
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


@pytest.mark.parametrize(
    "served",
    [PAYLOADS[SPA_URL][:-1], bytes(len(PAYLOADS[SPA_URL]))],
    ids=["shorter", "same-size-other-bytes"],
)
def test_a_download_that_differs_from_its_pin_is_rejected_and_leaves_no_file(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path, served: bytes
) -> None:
    payloads[SPA_URL] = served

    with pytest.raises(TessdataDownloadError) as caught:
        fetch(config, tmp_path, FakeDownloader(payloads))

    message = str(caught.value)
    assert message.startswith(f"{SPA_URL} returned {len(served):,} bytes")
    assert hashlib.sha256(served).hexdigest() in message
    assert f"pinned tessdata_best file of {PINS['spa'].size_bytes:,} bytes" in message
    assert PINS["spa"].sha256 in message
    assert listing(tmp_path / "tessdata") == []


def test_the_real_pins_are_the_files_fetched_for_the_ocr_on_2026_10_02() -> None:
    assert dict(PINNED_TRAINEDDATA) == {
        "spa": PinnedFile(
            size_bytes=13_570_187,
            sha256="e2c1ffdad8b30f26c45d4017a9183d3a7f9aa69e59918be4f88b126fac99ab2c",
        ),
        "eng": PinnedFile(
            size_bytes=15_400_601,
            sha256="8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
        ),
    }


def test_a_kept_file_of_a_pinned_language_must_also_match_its_pin(
    config: AppConfig, payloads: dict[str, bytes], tmp_path: Path
) -> None:
    earlier = payload_for(FAST_SPA_URL, 5_000)
    fetch(
        config,
        tmp_path,
        FakeDownloader({**payloads, SPA_URL: earlier}),
        pins={**PINS, "spa": pin_of(earlier)},
    )
    later_run = FakeDownloader(payloads)

    report = fetch(config, tmp_path, later_run, at=LATER_TIME)

    assert later_run.requests == [SPA_URL]
    assert report.downloaded == ("spa",)
    assert (tmp_path / "tessdata" / "spa.traineddata").read_bytes() == payloads[SPA_URL]


@pytest.fixture
def portuguese_too(config: AppConfig) -> AppConfig:
    """Return the configuration with a third language that has no pin."""
    return config.model_copy(update={"ocr": OcrConfig(languages="spa+eng+por", dpi=300)})


@pytest.mark.parametrize(
    ("content", "chunk_size"),
    [
        (traineddata(5_000), None),
        (traineddata(5_000), 100),  # The header arrives over three writes.
        (traineddata(700, entries=1, offsets=[12]), None),
    ],
    ids=["24-entries", "24-entries-in-chunks", "one-entry"],
)
def test_a_file_without_a_pin_is_kept_when_it_is_laid_out_as_a_language_file(
    portuguese_too: AppConfig,
    payloads: dict[str, bytes],
    tmp_path: Path,
    content: bytes,
    chunk_size: int | None,
) -> None:
    downloader = FakeDownloader({**payloads, POR_URL: content}, chunk_size=chunk_size)

    report = fetch(portuguese_too, tmp_path, downloader)

    assert downloader.requests == [SPA_URL, ENG_URL, POR_URL]
    assert (tmp_path / "tessdata" / "por.traineddata").read_bytes() == content
    record = report.manifest.files["por.traineddata"]
    assert (record.size_bytes, record.sha256) == (len(content), hashlib.sha256(content).hexdigest())


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (b"", "too short"),
        (b"<!DOCTYPE html><html><body>Sign in to continue</body></html>" * 20, "entry count"),
        (traineddata(5_000, header=struct.pack("<i", 0)), "entry count of 0"),
        (traineddata(5_000, header=struct.pack("<i", 65)), "entry count of 65"),
        (traineddata(100), "ends inside its offset table"),
        (traineddata(5_000, offsets=[-1] * 24), "lists no entry"),
        (traineddata(5_000, offsets=[196, 6_000, *[-1] * 22]), "past the end of the file"),
        (traineddata(5_000, offsets=[196, 5_000, *[-1] * 22]), "past the end of the file"),
        (traineddata(5_000, offsets=[150, 300, *[-1] * 22]), "into the table itself"),
        (traineddata(5_000, offsets=[400, 300, *[-1] * 22]), "out of order"),
    ],
    ids=[
        "empty",
        "html-page",
        "no-entries",
        "too-many-entries",
        "cut-in-its-table",
        "all-entries-absent",
        "offset-past-the-end",
        "offset-at-the-end",
        "offset-inside-the-table",
        "offsets-out-of-order",
    ],
)
def test_a_file_without_a_pin_is_rejected_when_it_is_not_a_language_file(
    portuguese_too: AppConfig,
    payloads: dict[str, bytes],
    tmp_path: Path,
    content: bytes,
    problem: str,
) -> None:
    downloader = FakeDownloader({**payloads, POR_URL: content})

    with pytest.raises(TessdataDownloadError) as caught:
        fetch(portuguese_too, tmp_path, downloader)

    message = str(caught.value)
    assert message.startswith(
        f"{POR_URL} returned {len(content):,} bytes that are not a Tesseract language file"
    )
    assert problem in message
    assert listing(tmp_path / "tessdata") == ["eng.traineddata", "manifest.json", "spa.traineddata"]


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
    cli_root: Path, payloads: dict[str, bytes], capsys: pytest.CaptureFixture[str]
) -> None:
    downloader = FakeDownloader(payloads)
    arguments = ["--project-root", str(cli_root)]

    assert main(arguments, downloader=downloader, pins=PINS) == 0
    first = capsys.readouterr().out
    assert main(arguments, downloader=downloader, pins=PINS) == 0
    second = capsys.readouterr().out
    assert main([*arguments, "--force"], downloader=downloader, pins=PINS) == 0
    forced = capsys.readouterr().out

    assert downloader.requests == [SPA_URL, ENG_URL, SPA_URL, ENG_URL]
    fetched = {"spa.traineddata": "downloaded", "eng.traineddata": "downloaded"}
    assert statuses(first) == fetched
    assert statuses(second) == dict.fromkeys(fetched, "already present")
    assert statuses(forced) == fetched
    assert hashlib.sha256(payloads[SPA_URL]).hexdigest() in first


def test_cli_rejects_a_download_that_is_not_the_pinned_file_with_an_error_exit(
    cli_root: Path, payloads: dict[str, bytes], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--project-root", str(cli_root)], downloader=FakeDownloader(payloads))

    error = capsys.readouterr().err
    assert exit_code == 1
    assert error.startswith("error:")
    assert SPA_URL in error
    assert PINNED_TRAINEDDATA["spa"].sha256 in error
    assert len(error.splitlines()) == 1
    assert listing(cli_root / "tessdata") == []


def edit_config(cli_root: Path, old: str, new: str) -> None:
    path = cli_root / "config" / "default.yaml"
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path.write_bytes(text.replace(old, new).encode("utf-8"))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("languages: spa+eng", "languages: spa+spa", "must be unique"),
        ("dpi: 300", "dpi: 30", "greater than or equal to 72"),
        ("tessdata_dir: tessdata", "tessdata_dir: ../outside", "outside"),
        ("ocr:", "ocr: [", "while parsing"),
    ],
    ids=["repeated-language", "invalid-setting", "directory-outside-the-project", "not-yaml"],
)
def test_cli_reports_a_configuration_error_on_one_line_with_exit_code_1(
    cli_root: Path, capsys: pytest.CaptureFixture[str], old: str, new: str, message: str
) -> None:
    edit_config(cli_root, old, new)
    downloader = FakeDownloader({})

    exit_code = main(["--project-root", str(cli_root)], downloader=downloader)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert message in captured.err
    assert len(captured.err.splitlines()) == 1
    assert (captured.out, downloader.requests) == ("", [])
    assert not (cli_root / "tessdata").exists()
