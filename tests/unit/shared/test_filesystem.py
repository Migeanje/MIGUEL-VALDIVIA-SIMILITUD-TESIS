"""The file-system helpers that the commands share: the run lock and the atomic writes.

Every test works in its own temporary directory. The lock is checked from a second handle in
this process and, where the operating system's behavior across processes matters, from a
child Python process.
"""

import errno
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from thematic_redundancy.shared.filesystem import (
    LOCK_FILE_NAME,
    exclusive_run,
    fsync_directory,
    write_bytes_atomically,
    write_files_atomically,
    write_text_atomically,
)


class BusyError(RuntimeError):
    """The error that a caller of the lock asks for when another run holds it."""


def busy() -> BusyError:
    return BusyError("another run is using the directory; wait until it ends")


def listing(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


# --- The run lock ------------------------------------------------------------------------


def test_the_lock_is_a_lock_file_in_the_guarded_directory_that_stays_after_the_run(
    tmp_path: Path,
) -> None:
    with exclusive_run(tmp_path, busy()):
        assert listing(tmp_path) == [".lock"]

    assert LOCK_FILE_NAME == ".lock"
    # Only the system's lock on the file comes and goes; the file stays, empty.
    assert (tmp_path / ".lock").read_bytes() == b""


def test_a_second_holder_is_refused_with_the_error_its_caller_gave(tmp_path: Path) -> None:
    refusal = busy()
    entered: list[bool] = []

    with (
        exclusive_run(tmp_path, busy()),
        pytest.raises(BusyError) as caught,
        exclusive_run(tmp_path, refusal),
    ):
        entered.append(True)

    assert caught.value is refusal
    assert str(caught.value) == "another run is using the directory; wait until it ends"
    assert isinstance(caught.value.__cause__, OSError)
    assert entered == []


def test_the_lock_is_released_when_the_block_ends(tmp_path: Path) -> None:
    with exclusive_run(tmp_path, busy()):
        pass

    with exclusive_run(tmp_path, busy()):
        pass


@pytest.mark.parametrize(
    "error", [ValueError("synthetic failure"), KeyboardInterrupt()], ids=["error", "interrupt"]
)
def test_the_lock_is_released_when_the_block_raises_and_the_error_goes_through(
    tmp_path: Path, error: BaseException
) -> None:
    with pytest.raises(type(error)) as caught, exclusive_run(tmp_path, busy()):
        raise error

    assert caught.value is error
    with exclusive_run(tmp_path, busy()):
        pass


def test_a_lock_file_left_by_an_earlier_run_neither_blocks_a_run_nor_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / LOCK_FILE_NAME).write_bytes(b"left by a killed run")

    with exclusive_run(tmp_path, busy()):
        pass

    assert (tmp_path / LOCK_FILE_NAME).read_bytes() == b"left by a killed run"


def test_runs_in_different_directories_do_not_block_each_other(tmp_path: Path) -> None:
    first, second = tmp_path / "pdfs", tmp_path / "pages"
    first.mkdir()
    second.mkdir()

    with exclusive_run(first, busy()), exclusive_run(second, busy()):
        assert listing(first) == listing(second) == [".lock"]


def test_a_missing_directory_is_a_disk_error_not_a_busy_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError), exclusive_run(tmp_path / "missing", busy()):
        pass


ATTEMPT = """
import sys
import time
from pathlib import Path

from thematic_redundancy.shared.filesystem import exclusive_run

started = time.monotonic()
try:
    with exclusive_run(Path(sys.argv[1]), RuntimeError("busy")):
        print("acquired")
except RuntimeError as error:
    print("refused" if isinstance(error.__cause__, OSError) else "refused without a cause")
print(f"{time.monotonic() - started:.3f}")
"""
"""Takes the lock of a directory once, without holding it, and prints the outcome and the
seconds the attempt took."""

HOLD = """
import sys
from pathlib import Path

from thematic_redundancy.shared.filesystem import exclusive_run

with exclusive_run(Path(sys.argv[1]), RuntimeError("busy")):
    print("holding", flush=True)
    sys.stdin.read()
"""
"""Takes the lock of a directory and holds it until its standard input closes or it is
killed."""


def attempt_in_another_process(directory: Path) -> tuple[str, float]:
    """Try the lock of ``directory`` from a child process; return the outcome and seconds."""
    child = subprocess.run(
        [sys.executable, "-c", ATTEMPT, str(directory)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    outcome, seconds = child.stdout.splitlines()
    return outcome, float(seconds)


def acquire_within(directory: Path, seconds: float) -> None:
    """Take and release the lock of ``directory``, trying again for up to ``seconds``.

    A system frees the locks of a process that ends, but Windows may take a moment to do it.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            with exclusive_run(directory, busy()):
                return
        except BusyError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_another_process_is_refused_at_once_while_a_run_holds_the_lock(tmp_path: Path) -> None:
    with exclusive_run(tmp_path, busy()):
        outcome, seconds = attempt_in_another_process(tmp_path)

    assert outcome == "refused"
    # Taken without waiting: a waiting lock would only give up after about 10 s on Windows,
    # and never on POSIX.
    assert seconds < 2
    assert attempt_in_another_process(tmp_path)[0] == "acquired"


def test_a_killed_run_leaves_no_stale_lock(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == "holding\n"
        with pytest.raises(BusyError), exclusive_run(tmp_path, busy()):
            pass
    finally:
        holder.kill()
        holder.communicate(timeout=60)

    assert holder.returncode != 0
    acquire_within(tmp_path, seconds=10)


# --- Atomic text writes ------------------------------------------------------------------


def test_the_text_is_written_in_utf8_and_ends_with_one_line_feed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_text_atomically(path, '{\n  "título": "Diseño — ñandú €"\n}')

    # Line breaks stay line feeds on every system, and nothing else is added.
    assert path.read_bytes() == '{\n  "título": "Diseño — ñandú €"\n}\n'.encode()


def test_an_existing_file_is_replaced_and_no_staged_file_is_left(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_bytes(b"previous report, longer than the new one\n")

    write_text_atomically(path, "# Report")

    assert path.read_bytes() == b"# Report\n"
    assert listing(tmp_path) == ["report.md"]


def test_a_staged_file_left_by_a_killed_run_is_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    (tmp_path / "manifest.json.part").write_bytes(b'{"partial": "and longer than the new one"')

    write_text_atomically(path, "{}")

    assert path.read_bytes() == b"{}\n"
    assert listing(tmp_path) == ["manifest.json"]


def test_the_directory_of_the_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_text_atomically(tmp_path / "missing" / "manifest.json", "{}")

    assert listing(tmp_path) == []


def record_file_calls(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    directory: Path,
    *,
    directory_opens: bool = True,
    directory_flush_error: OSError | None = None,
) -> list[str]:
    """Record the flushes, moves and directory steps of a write, in order, and return the log.

    Moves are shown relative to ``root``. Opening ``directory`` gives a stand-in descriptor
    that is never flushed for real, since Windows cannot open a directory; with
    ``directory_opens`` false the open fails as on Windows, and ``directory_flush_error``
    makes the flush of the directory fail. Every other call goes through to :mod:`os`.
    """
    calls: list[str] = []
    real_open, real_fsync, real_close, real_replace = os.open, os.fsync, os.close, os.replace
    stand_in = root / "directory-stand-in"
    stand_in.write_bytes(b"")
    stand_ins: set[int] = set()

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) != directory:
            return real_open(path, flags, *args, **kwargs)
        calls.append("open directory")
        if not directory_opens:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        descriptor = real_open(stand_in, os.O_RDONLY)
        stand_ins.add(descriptor)
        return descriptor

    def fsync(descriptor: int) -> None:
        if descriptor in stand_ins:
            calls.append("fsync directory")
            if directory_flush_error is not None:
                raise directory_flush_error
            return
        calls.append(f"fsync file of {os.fstat(descriptor).st_size} bytes")
        real_fsync(descriptor)

    def close(descriptor: int) -> None:
        if descriptor in stand_ins:
            calls.append("close directory")
        real_close(descriptor)

    def replace(source: Any, target: Any) -> None:
        moved = (Path(source).relative_to(root), Path(target).relative_to(root))
        calls.append(f"replace {moved[0].as_posix()} -> {moved[1].as_posix()}")
        real_replace(source, target)

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(os, "replace", replace)
    return calls


MOVE = "replace out/manifest.json.part -> out/manifest.json"
"""The one move of a write: from the staged file beside the target onto the target."""


def test_the_staged_file_is_flushed_then_moved_onto_the_target_then_the_directory_flushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out" / "manifest.json"
    path.parent.mkdir()
    calls = record_file_calls(monkeypatch, tmp_path, path.parent)

    write_text_atomically(path, "{}")

    assert calls == [
        "fsync file of 3 bytes",
        MOVE,
        "open directory",
        "fsync directory",
        "close directory",
    ]
    assert path.read_bytes() == b"{}\n"


def test_a_file_is_published_where_its_directory_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out" / "manifest.json"
    path.parent.mkdir()
    path.write_bytes(b"old\n")
    calls = record_file_calls(monkeypatch, tmp_path, path.parent, directory_opens=False)

    write_text_atomically(path, "{}")

    assert calls == ["fsync file of 3 bytes", MOVE, "open directory"]
    assert path.read_bytes() == b"{}\n"
    assert listing(path.parent) == ["manifest.json"]


@pytest.mark.parametrize(
    "error",
    [OSError(errno.EINVAL, "Invalid argument"), OSError(errno.EIO, "Input/output error")],
    ids=["unsupported", "disk-failure"],
)
def test_a_file_is_published_where_its_directory_cannot_be_flushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    path = tmp_path / "out" / "manifest.json"
    path.parent.mkdir()
    calls = record_file_calls(monkeypatch, tmp_path, path.parent, directory_flush_error=error)

    write_text_atomically(path, "{}")

    # The file is flushed and in place by then; the directory's descriptor is still closed.
    assert calls == [
        "fsync file of 3 bytes",
        MOVE,
        "open directory",
        "fsync directory",
        "close directory",
    ]
    assert path.read_bytes() == b"{}\n"


@pytest.mark.parametrize("step", ["fsync", "replace"])
@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk-full", "interrupt"],
)
def test_a_failed_write_keeps_the_previous_file_and_removes_the_staged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, step: str
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"previous\n")

    def fail(*args: Any) -> None:
        raise error

    monkeypatch.setattr(os, step, fail)

    with pytest.raises(type(error)) as caught:
        write_text_atomically(path, '{"new": true}')

    assert caught.value is error
    assert path.read_bytes() == b"previous\n"
    assert listing(tmp_path) == ["manifest.json"]


@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk-full", "interrupt"],
)
def test_a_staged_file_that_cannot_be_removed_does_not_hide_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"previous\n")

    def fail(*args: Any) -> None:
        raise error

    def held_open(self: Path, missing_ok: bool = False) -> None:
        # As on Windows while an antivirus or an indexer holds the file open.
        raise PermissionError(errno.EACCES, "The process cannot access the file", str(self))

    monkeypatch.setattr(os, "replace", fail)
    monkeypatch.setattr(Path, "unlink", held_open)

    with pytest.raises(type(error)) as caught:
        write_text_atomically(path, '{"new": true}')

    assert caught.value is error
    assert path.read_bytes() == b"previous\n"
    assert listing(tmp_path) == ["manifest.json", "manifest.json.part"]
    # The next write takes over the staged file that was left.
    monkeypatch.undo()
    write_text_atomically(path, "{}")
    assert listing(tmp_path) == ["manifest.json"]
    assert path.read_bytes() == b"{}\n"


def test_flushing_a_real_directory_never_fails(tmp_path: Path) -> None:
    # Windows cannot open a directory, so there the step is skipped; elsewhere it flushes.
    fsync_directory(tmp_path)


# --- Atomic byte writes, of one file or of several -----------------------------------------

PNG_LIKE = b"\x89PNG\r\n\x1a\n\x00\xff binary \r\n data"
"""Bytes that a text write would change: line breaks of both kinds and no final line feed."""


def test_bytes_are_written_exactly_as_given(tmp_path: Path) -> None:
    path = tmp_path / "figure.png"
    path.write_bytes(b"previous figure, longer than the new one")

    write_bytes_atomically(path, PNG_LIKE)

    assert path.read_bytes() == PNG_LIKE
    assert listing(tmp_path) == ["figure.png"]


def test_several_files_are_written_and_other_files_of_the_directory_stay(tmp_path: Path) -> None:
    (tmp_path / "profile.json").write_bytes(b"previous profile\n")
    (tmp_path / "candidates.json").write_bytes(b"written by another command\n")

    write_files_atomically(tmp_path, {"profile.json": b"{}\n", "figure.png": PNG_LIKE})

    assert (tmp_path / "profile.json").read_bytes() == b"{}\n"
    assert (tmp_path / "figure.png").read_bytes() == PNG_LIKE
    assert (tmp_path / "candidates.json").read_bytes() == b"written by another command\n"
    assert listing(tmp_path) == ["candidates.json", "figure.png", "profile.json"]


def test_every_file_is_staged_and_flushed_before_the_first_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    calls = record_file_calls(monkeypatch, tmp_path, directory)

    write_files_atomically(directory, {"profile.json": b"{}\n", "figure.png": PNG_LIKE})

    assert calls == [
        "fsync file of 3 bytes",
        f"fsync file of {len(PNG_LIKE)} bytes",
        "replace out/profile.json.part -> out/profile.json",
        "replace out/figure.png.part -> out/figure.png",
        "open directory",
        "fsync directory",
        "close directory",
    ]


@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk-full", "interrupt"],
)
def test_a_failure_while_staging_keeps_every_previous_file_and_removes_the_staged_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    names = ("profile.json", "summary.md", "figure.png")
    for name in names:
        (tmp_path / name).write_bytes(f"previous {name}\n".encode())
    real_fsync = os.fsync
    flushed: list[int] = []

    def fsync_failing_on_the_second_file(descriptor: int) -> None:
        flushed.append(descriptor)
        if len(flushed) == 2:
            raise error
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync_failing_on_the_second_file)

    with pytest.raises(type(error)) as caught:
        write_files_atomically(tmp_path, {name: b"new\n" for name in names})

    assert caught.value is error
    for name in names:
        assert (tmp_path / name).read_bytes() == f"previous {name}\n".encode()
    assert listing(tmp_path) == sorted(names)


def test_a_failed_move_leaves_the_files_moved_before_it_and_removes_the_staged_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ("profile.json", "summary.md", "figure.png")
    for name in names:
        (tmp_path / name).write_bytes(b"previous\n")
    real_replace = os.replace

    def replace_failing_on_the_summary(source: Any, target: Any) -> None:
        if Path(target).name == "summary.md":
            raise PermissionError(errno.EACCES, "The process cannot access the file", target)
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace_failing_on_the_summary)

    with pytest.raises(PermissionError):
        write_files_atomically(tmp_path, {name: b"new\n" for name in names})

    # Moves cannot be undone, so a failed move is the one case that mixes new and old files.
    assert (tmp_path / "profile.json").read_bytes() == b"new\n"
    assert (tmp_path / "summary.md").read_bytes() == b"previous\n"
    assert (tmp_path / "figure.png").read_bytes() == b"previous\n"
    assert listing(tmp_path) == sorted(names)


def test_the_directory_of_the_files_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_files_atomically(tmp_path / "missing", {"profile.json": b"{}\n"})

    assert listing(tmp_path) == []


@pytest.mark.parametrize("name", ["", ".", "..", "eda/profile.json", "../profile.json"])
def test_a_file_name_must_name_a_file_of_the_directory(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="file name"):
        write_files_atomically(tmp_path, {"profile.json": b"{}\n", name: b"{}\n"})

    # The names are checked before anything is written.
    assert listing(tmp_path) == []


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        pytest.param(
            {"profile.json.part": b"staged name\n", "profile.json": b"{}\n"},
            "staged files",
            id="the staged name of another file",
        ),
        pytest.param({"summary.PART": b"# Report\n"}, "staged files", id="a staged name"),
        pytest.param(
            {"Profile.json": b"upper\n", "profile.json": b"lower\n"},
            "same file where letter case is ignored",
            id="names equal but for their case",
        ),
    ],
)
def test_names_that_could_name_the_same_file_are_refused_before_anything_is_written(
    tmp_path: Path, contents: dict[str, bytes], reason: str
) -> None:
    # A staged name taken by another file, or two names that one file holds where letter case
    # is ignored (Windows, macOS), would make one write overwrite another.
    with pytest.raises(ValueError, match=reason):
        write_files_atomically(tmp_path, contents)

    assert listing(tmp_path) == []
