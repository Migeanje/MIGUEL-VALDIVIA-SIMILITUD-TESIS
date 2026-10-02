"""Fetch and verify the Tesseract language files that the OCR fallback reads.

PyMuPDF bundles Tesseract, so OCR needs only one ``<lang>.traineddata`` file per configured
language and no Tesseract installation (D07). This module downloads those files once from
the ``tessdata_best`` models into the configured ``paths.tessdata_dir`` and records the
source, size, SHA-256 and download time of each one in ``manifest.json``. Later runs check
the files against the manifest and download only the ones that are missing or changed.

Run it once per machine::

    uv run python -m thematic_redundancy.extraction.ocr_assets [--project-root PATH] [--force]

The OCR itself belongs to the PDF text extraction; this module only manages its assets.
"""

import argparse
import hashlib
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, BinaryIO, Protocol

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
)

from thematic_redundancy.shared.config import AppConfig, load_config

TESSDATA_VARIANT = "tessdata_best"
"""Model family: Tesseract's most accurate LSTM models, slower than ``tessdata_fast``."""

TESSDATA_BASE_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main"
"""GitHub answers with a redirect to the file on raw.githubusercontent.com."""

MIN_TRAINEDDATA_BYTES = 5_000_000
"""Smallest plausible ``tessdata_best`` file; an HTML error page is far smaller."""

MANIFEST_FILE_NAME = "manifest.json"

CONFIG_PATH = Path("config") / "default.yaml"
"""Configuration that the command line loads, relative to the project root."""

FETCH_COMMAND = "uv run python -m thematic_redundancy.extraction.ocr_assets"
"""One-time command that fetches the language files; error messages point to it."""

_LANGUAGE_CODE = re.compile(r"[A-Za-z_]+")
_CHUNK_BYTES = 64 * 1024
_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


class TessdataDownloadError(RuntimeError):
    """A language file could not be fetched, or the fetched bytes failed a sanity check."""


class ByteSink(Protocol):
    """Destination of a download, such as a binary file."""

    def write(self, data: bytes, /) -> int: ...


class Downloader(Protocol):
    """Write the body of ``url`` into ``sink``; raise if the server does not deliver it."""

    def __call__(self, url: str, sink: ByteSink, /) -> None: ...


class HttpxDownloader:
    """:class:`Downloader` that streams over an ``httpx.Client`` and follows redirects."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def __call__(self, url: str, sink: ByteSink, /) -> None:
        try:
            with self._client.stream("GET", url, follow_redirects=True) as response:
                if response.status_code != httpx.codes.OK:
                    raise TessdataDownloadError(f"GET {url} returned HTTP {response.status_code}")
                for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                    sink.write(chunk)
        except httpx.HTTPError as error:
            raise TessdataDownloadError(f"GET {url} failed: {error}") from error


class TraineddataRecord(BaseModel):
    """Provenance of one language file, as the manifest records it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: Annotated[str, Field(pattern=r"^https://\S+$")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: Annotated[StrictInt, Field(ge=1)]
    downloaded_at: AwareDatetime

    @field_validator("downloaded_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"must be a UTC time, got '{value.isoformat()}'")
        return value


class TessdataManifest(BaseModel):
    """Content of ``manifest.json``: the model family, the languages, and each file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    languages: tuple[str, ...]
    files: dict[str, TraineddataRecord]
    """Records keyed by file name, such as ``spa.traineddata``."""


@dataclass(frozen=True)
class TessdataReport:
    """Outcome of :func:`ensure_tessdata`."""

    tessdata_dir: Path
    manifest: TessdataManifest
    downloaded: tuple[str, ...]
    """Language codes fetched by this call; the other configured languages were present."""

    @property
    def manifest_path(self) -> Path:
        return self.tessdata_dir / MANIFEST_FILE_NAME


def language_codes(spec: str) -> tuple[str, ...]:
    """Split a Tesseract language spec such as ``spa+eng`` into ``("spa", "eng")``.

    Raises:
        ValueError: if a code is empty or malformed, or if a code appears twice.
    """
    codes = tuple(spec.split("+"))
    malformed = [code for code in codes if not _LANGUAGE_CODE.fullmatch(code)]
    if malformed:
        raise ValueError(f"malformed Tesseract language code(s) {malformed} in '{spec}'")
    repeated = sorted({code for code in codes if codes.count(code) > 1})
    if repeated:
        raise ValueError(f"Tesseract language codes must be unique; repeated: {repeated}")
    return codes


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def ensure_tessdata(
    config: AppConfig,
    project_root: Path,
    downloader: Downloader,
    *,
    force: bool = False,
    min_bytes: int = MIN_TRAINEDDATA_BYTES,
    clock: Callable[[], datetime] = _utc_now,
) -> TessdataReport:
    """Make sure the tessdata directory holds a verified file for each configured language.

    The directory is ``paths.tessdata_dir`` under ``project_root``, and the languages come
    from ``ocr.languages``. A file is kept when its size and SHA-256 still match its
    manifest record for the same source; any other file is downloaded, and ``force``
    downloads every file again. A download goes to a temporary file beside its target and
    replaces the target only once it passes the checks, so a failure never leaves a partial
    file. The manifest is saved after each download, so a later failure keeps the files
    already fetched, and an unchanged manifest is not rewritten.

    Raises:
        TessdataDownloadError: if a download fails or is smaller than ``min_bytes``.
        ValueError: if ``ocr.languages`` repeats a language, or if the directory resolves
            outside the project root.
    """
    codes = language_codes(config.ocr.languages)
    tessdata_dir = config.paths.resolve_against(project_root).tessdata_dir
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tessdata_dir / MANIFEST_FILE_NAME
    on_disk = _read_manifest(manifest_path)
    records: dict[str, TraineddataRecord] = {}
    if on_disk is not None and on_disk.variant == TESSDATA_VARIANT:
        records.update(on_disk.files)
    downloaded: list[str] = []
    for code in codes:
        file_name, source_url = _file_name(code), _source_url(code)
        target = tessdata_dir / file_name
        if not force and _is_intact(target, records.get(file_name), source_url):
            continue
        sha256, size_bytes = _download(downloader, source_url, target, min_bytes)
        records[file_name] = TraineddataRecord(
            source_url=source_url, sha256=sha256, size_bytes=size_bytes, downloaded_at=clock()
        )
        downloaded.append(code)
        on_disk = _write_manifest(
            manifest_path,
            TessdataManifest(variant=TESSDATA_VARIANT, languages=codes, files=dict(records)),
        )
    manifest = TessdataManifest(
        variant=TESSDATA_VARIANT,
        languages=codes,
        files={_file_name(code): records[_file_name(code)] for code in codes},
    )
    if manifest != on_disk:  # Also saves a changed language list that needed no download.
        _write_manifest(manifest_path, manifest)
    return TessdataReport(
        tessdata_dir=tessdata_dir, manifest=manifest, downloaded=tuple(downloaded)
    )


def _file_name(code: str) -> str:
    return f"{code}.traineddata"


def _source_url(code: str) -> str:
    return f"{TESSDATA_BASE_URL}/{_file_name(code)}"


def _sha256_of(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _is_intact(path: Path, record: TraineddataRecord | None, source_url: str) -> bool:
    """Tell whether ``path`` still holds the bytes that ``record`` describes for ``source_url``."""
    if record is None or record.source_url != source_url or not path.is_file():
        return False
    return path.stat().st_size == record.size_bytes and _sha256_of(path) == record.sha256


def _read_manifest(path: Path) -> TessdataManifest | None:
    """Return the saved manifest, or ``None`` when it is missing or damaged.

    A damaged manifest is not an error: it only means that every file is downloaded again.
    """
    try:
        return TessdataManifest.model_validate_json(path.read_bytes())
    except (FileNotFoundError, ValidationError):
        return None


def _write_manifest(path: Path, manifest: TessdataManifest) -> TessdataManifest:
    """Save ``manifest`` atomically and return it."""
    with _staged(path) as staged:
        staged.write(f"{manifest.model_dump_json(indent=2)}\n".encode())
    return manifest


def _download(downloader: Downloader, url: str, target: Path, min_bytes: int) -> tuple[str, int]:
    """Fetch ``url`` into ``target`` atomically and return the file's SHA-256 and size."""
    with _staged(target) as staged:
        downloader(url, staged)
        if staged.size < min_bytes:
            raise TessdataDownloadError(
                f"{url} returned {staged.size:,} bytes, fewer than the {min_bytes:,} bytes "
                f"expected of a {TESSDATA_VARIANT} file"
            )
    return staged.sha256, staged.size


class _StagedFile:
    """Byte sink over a temporary file that tracks the size and SHA-256 of what it wrote."""

    def __init__(self, file: BinaryIO) -> None:
        self._file = file
        self._digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes, /) -> int:
        self._file.write(data)
        self._digest.update(data)
        self.size += len(data)
        return len(data)

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


@contextmanager
def _staged(target: Path) -> Iterator[_StagedFile]:
    """Collect bytes in a temporary file beside ``target``, then move it onto ``target``.

    The move happens only when the block succeeds, and it replaces ``target`` atomically,
    so readers see the old file or the new one, never a partial one. When the block
    raises, the temporary file is removed.
    """
    descriptor, staged_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part"
    )
    staged_path = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            yield _StagedFile(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(staged_path, target)
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise


def _summary(report: TessdataReport) -> str:
    """Describe the outcome: one line per language file, then the manifest location."""
    lines = [f"{TESSDATA_VARIANT} language files in {report.tessdata_dir}:"]
    for code in report.manifest.languages:
        file_name = _file_name(code)
        record = report.manifest.files[file_name]
        status = "downloaded" if code in report.downloaded else "already present"
        lines.append(
            f"  {file_name}  {status:<15}  {record.size_bytes:>12,} bytes  sha256 {record.sha256}"
        )
    lines.append(f"Manifest: {report.manifest_path}")
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.extraction.ocr_assets",
        description=(
            f"Download the {TESSDATA_VARIANT} Tesseract language files that OCR needs into "
            "the configured tessdata directory, and verify the ones already there."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download every file again, even when it matches the manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, downloader: Downloader | None = None) -> int:
    """Run the command line and return its exit code; ``downloader`` replaces HTTP in tests."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    try:
        config = load_config(project_root / CONFIG_PATH)
        with ExitStack() as stack:
            if downloader is None:
                client = stack.enter_context(httpx.Client(timeout=_HTTP_TIMEOUT))
                downloader = HttpxDownloader(client)
            report = ensure_tessdata(config, project_root, downloader, force=arguments.force)
    except (TessdataDownloadError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
