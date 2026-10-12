"""File-system helpers that the commands share: a run lock and atomic text writes.

- :func:`exclusive_run` lets one run at a time use a directory, such as the PDF directory of
  the download or the pages directory of the text extraction. It takes the operating
  system's lock on the directory's :data:`LOCK_FILE_NAME` without waiting, and the system
  releases it when the process ends, so a killed run leaves no stale lock behind.
- :func:`write_text_atomically` replaces a text file, such as a JSON manifest or a Markdown
  report, so that readers see the previous file or the new one, never a partial one, and so
  that the new one survives a crash.
- :func:`fsync_directory` flushes the entries of a directory to the disk, where the system
  allows it.
"""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

LOCK_FILE_NAME = ".lock"
"""File that :func:`exclusive_run` locks in the directory it guards. The file stays after a
run; only the system's lock on it comes and goes."""


@contextmanager
def exclusive_run(directory: Path, busy_error: Exception) -> Iterator[None]:
    """Hold the lock of ``directory`` while the block runs.

    The lock is the operating system's lock on :data:`LOCK_FILE_NAME` in ``directory``, which
    is created when missing and never deleted. It is taken without waiting, and the system
    releases it when the process ends, so a killed run leaves no stale lock behind. Each call
    opens the file anew, so a second holder is refused even within one process. The lock is
    released when the block ends, also when it raises.

    Raises:
        Exception: ``busy_error`` itself, from the system's error, if another run holds the
            lock; the block does not run then.
        OSError: if the lock file cannot be opened, such as when ``directory`` is missing.
    """
    with (directory / LOCK_FILE_NAME).open("a+b") as lock_file:
        try:
            _lock(lock_file)
        except OSError as error:
            raise busy_error from error
        try:
            yield
        finally:
            _unlock(lock_file)


if sys.platform == "win32":

    def _lock(file: BinaryIO) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(file: BinaryIO) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)

else:

    def _lock(file: BinaryIO) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(file: BinaryIO) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def write_text_atomically(path: Path, text: str) -> None:
    """Replace the file at ``path`` with ``text`` atomically and durably.

    The file holds ``text`` encoded in UTF-8 and followed by one line feed, and nothing else:
    line breaks inside ``text`` stay line feeds on every system. The content goes to
    ``<path>.part`` beside ``path`` first and is flushed to the disk, then one rename puts it
    in place, so readers see the previous file or the new one, never a partial one. The
    directory is flushed last (:func:`fsync_directory`), so the new name survives a crash.

    When a step up to the rename fails, or is interrupted, the previous file stays, the
    original error or interruption is raised, and the staged file is removed where the system
    allows it. On Windows the removal fails while another process, such as an antivirus or an
    indexer, holds the file open; the staged file then stays. A staged file that a killed or
    failed run left behind is overwritten by the next write; the resumable commands also
    delete such files when they start.

    Raises:
        OSError: if writing fails, such as when the directory of ``path`` is missing.
    """
    staged = path.with_name(f"{path.name}.part")
    try:
        with staged.open("wb") as file:
            file.write(f"{text}\n".encode())
            file.flush()
            os.fsync(file.fileno())
        os.replace(staged, path)
    except BaseException:
        # A failed removal must not replace the error that matters.
        with suppress(OSError):
            staged.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def fsync_directory(directory: Path) -> None:
    """Flush the entries of ``directory`` to the disk, so that a file just named in it stays.

    POSIX systems need this after a rename. A failure to open or to flush the directory is
    skipped: Windows cannot open a directory as a file, and some network, FUSE or overlay
    file systems refuse to flush one (EINVAL). The file is already flushed and in place by
    then, so this step can only make its name more durable; raising would fail a write that
    readers can already see.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
