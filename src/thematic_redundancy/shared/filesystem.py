"""File-system helpers that the commands share: a run lock and atomic writes.

- :func:`exclusive_run` lets one run at a time use a directory, such as the PDF directory of
  the download or the pages directory of the text extraction. It takes the operating
  system's lock on the directory's :data:`LOCK_FILE_NAME` without waiting, and the system
  releases it when the process ends, so a killed run leaves no stale lock behind.
- :func:`write_text_atomically` replaces a text file, such as a JSON manifest or a Markdown
  report, so that readers see the previous file or the new one, never a partial one, and so
  that the new one survives a crash. :func:`write_bytes_atomically` does the same for any
  bytes, such as a figure, and :func:`write_files_atomically` for several files of one
  directory, which it stages all before it moves any into place.
- :func:`fsync_directory` flushes the entries of a directory to the disk, where the system
  allows it.
"""

import os
import sys
from collections.abc import Iterator, Mapping
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

STAGED_SUFFIX = ".part"
"""Suffix of the staged file that an atomic write fills beside its target before the rename."""


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
        ValueError: if the name of ``path`` ends in :data:`STAGED_SUFFIX`.
        OSError: if writing fails, such as when the directory of ``path`` is missing.
    """
    write_bytes_atomically(path, f"{text}\n".encode())


def write_bytes_atomically(path: Path, content: bytes) -> None:
    """Replace the file at ``path`` with ``content``, exactly, atomically and durably.

    It works as :func:`write_text_atomically` does, through ``<path>.part``, but writes the
    bytes as given, with nothing added.

    Raises:
        ValueError: if the name of ``path`` ends in :data:`STAGED_SUFFIX`.
        OSError: if writing fails, such as when the directory of ``path`` is missing.
    """
    write_files_atomically(path.parent, {path.name: content})


def write_files_atomically(directory: Path, contents: Mapping[str, bytes]) -> None:
    """Replace the files of ``directory`` named in ``contents`` with their bytes.

    Each file is written to ``<name>.part`` in ``directory`` and flushed to the disk, all of
    them before any is moved into place; then one rename per file, in the order of
    ``contents``, puts each in place, and the directory is flushed last
    (:func:`fsync_directory`). Each file is therefore the previous one or the new one, never
    a partial one, and the other files of ``directory`` stay as they are.

    When writing a staged file fails or is interrupted, every file keeps its previous
    content, the original error or interruption is raised, and the staged files are removed
    where the system allows it, as :func:`write_text_atomically` describes. A rename cannot be
    undone, so a rename that fails, as on Windows while another program holds the target
    open, or an interruption between two renames leaves a mix: the files renamed before it
    are new, and the others keep their previous content. Writing them all again fixes that.

    Raises:
        ValueError: if a key of ``contents`` is not the name of a file in ``directory``, such
            as ``""``, ``..`` or a name with a directory part; if it ends in
            :data:`STAGED_SUFFIX`, in any letter case, so that its staged file or itself could
            be another key's; or if two keys are equal once their letter case is ignored, as
            they name one file on Windows and macOS. Nothing is written then.
        OSError: if writing fails, such as when ``directory`` is missing.
    """
    names_by_folded_case: dict[str, str] = {}
    for name in contents:
        if name in ("", ".", "..") or Path(name).name != name:
            raise ValueError(f"{name!r} is not a file name; give the name of a file only")
        if name.casefold().endswith(STAGED_SUFFIX):
            raise ValueError(f"{name!r} ends in {STAGED_SUFFIX}, the suffix of the staged files")
        other = names_by_folded_case.setdefault(name.casefold(), name)
        if other != name:
            raise ValueError(
                f"{other!r} and {name!r} may name the same file where letter case is ignored"
            )
    staged = {name: directory / f"{name}{STAGED_SUFFIX}" for name in contents}
    try:
        for name, content in contents.items():
            with staged[name].open("wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        for name in contents:
            os.replace(staged[name], directory / name)
    except BaseException:
        # A failed removal must not replace the error that matters.
        for path in staged.values():
            with suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    fsync_directory(directory)


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
