"""Keyed pseudonyms for people, the identities they come from, and the secret key file.

Every ORCID iD here is synthetic: each lies below the block from which ORCID assigns iDs,
which starts at 0000-0001-5000-0007, so none can belong to a person. The names are
synthetic as well, and the test key is simply the bytes 0 to 31.
"""

import errno
import io
import os
import re
import secrets
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from thematic_redundancy.corpus import anonymize
from thematic_redundancy.corpus.anonymize import (
    KEY_SIZE,
    default_key_path,
    load_or_create_key,
    normalize_name,
    normalize_orcid,
    person_identity,
    pseudonym,
)
from thematic_redundancy.shared.config import PathsConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DECLARED = load_config(PROJECT_ROOT / "config" / "default.yaml")

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
ORCID = "0000-0000-1234-5672"
NAME = "Sintético Ñuñez, Ángel Uno"
NAME_IDENTITY = "name:sintetico nunez, angel uno"


def listing(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


# Pseudonyms


def test_a_pseudonym_is_the_prefix_and_ten_hex_digits() -> None:
    code = pseudonym("orcid:0000-0000-0000-0001", KEY, "ADV-")

    assert re.fullmatch(r"ADV-[0-9a-f]{10}", code)


def test_the_pseudonym_construction_is_pinned() -> None:
    # HMAC-SHA256 under KEY of "ADV-", a NUL, then the identity, cut to 10 hex digits.
    # A change here would silently change every code already stored.
    assert pseudonym("orcid:0000-0000-0000-0001", KEY, "ADV-") == "ADV-dcc7a11187"
    assert pseudonym("orcid:0000-0000-0000-0001", KEY, "AUT-") == "AUT-e3028df100"


def test_a_pseudonym_is_deterministic() -> None:
    assert pseudonym(NAME_IDENTITY, KEY, "ADV-") == pseudonym(NAME_IDENTITY, bytes(KEY), "ADV-")


def test_another_key_gives_an_unrelated_code() -> None:
    assert pseudonym(NAME_IDENTITY, KEY, "ADV-") != pseudonym(NAME_IDENTITY, OTHER_KEY, "ADV-")


def test_codes_of_two_kinds_for_one_person_cannot_be_linked() -> None:
    advisor = pseudonym(NAME_IDENTITY, KEY, "ADV-")
    author = pseudonym(NAME_IDENTITY, KEY, "AUT-")

    assert advisor.removeprefix("ADV-") != author.removeprefix("AUT-")


def test_different_identities_give_different_codes() -> None:
    codes = {
        pseudonym(identity, KEY, "ADV-")
        for identity in ("orcid:0000-0000-0000-0001", "orcid:0000-0000-1234-5672", NAME_IDENTITY)
    }

    assert len(codes) == 3


@pytest.mark.parametrize("identity", ["", "   "])
def test_a_pseudonym_needs_an_identity(identity: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        pseudonym(identity, KEY, "ADV-")


@pytest.mark.parametrize("key", [b"", b"short", bytes(31), bytes(33)])
def test_a_pseudonym_needs_a_key_of_the_right_size(key: bytes) -> None:
    with pytest.raises(ValueError, match=f"{KEY_SIZE} bytes"):
        pseudonym(NAME_IDENTITY, key, "ADV-")


@pytest.mark.parametrize("prefix", ["", "ADV ", "A\x00", "Ñ-", "-ADV", "A" * 17])
def test_a_pseudonym_needs_a_plain_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        pseudonym(NAME_IDENTITY, KEY, prefix)


# ORCID iDs


@pytest.mark.parametrize(
    "value",
    [
        ORCID,
        "0000000012345672",
        "https://orcid.org/0000-0000-1234-5672",
        "http://orcid.org/0000-0000-1234-5672",
        "https://www.orcid.org/0000-0000-1234-5672/",
        "orcid.org/0000-0000-1234-5672",
        "HTTPS://ORCID.ORG/0000-0000-1234-5672",
        "  0000-0000-1234-5672\n",
    ],
    ids=["bare", "no-hyphens", "https", "http", "www-slash", "no-scheme", "upper-case", "padded"],
)
def test_an_orcid_is_normalized_from_a_url_or_a_bare_id(value: str) -> None:
    assert normalize_orcid(value) == ORCID


@pytest.mark.parametrize("value", ["0000-0000-0000-001X", "0000-0000-0000-001x"])
def test_an_orcid_may_end_with_the_check_character_x(value: str) -> None:
    assert normalize_orcid(value) == "0000-0000-0000-001X"


@pytest.mark.parametrize(
    "value",
    [
        "0000-0000-1234-5673",
        "0000-0000-1234-567",
        "0000-0000-1234-56722",
        "0000-0000-1234-567Y",
        "0000 0000 1234 5672",
        "https://example.org/0000-0000-1234-5672",
        "ORCID 0000-0000-1234-5672",
        "",
    ],
    ids=[
        "wrong-check-digit",
        "short",
        "long",
        "bad-check-character",
        "spaced",
        "other-site",
        "labelled",
        "empty",
    ],
)
def test_a_value_with_a_wrong_check_digit_or_shape_is_no_orcid(value: str) -> None:
    assert normalize_orcid(value) is None


# Names and identities


@pytest.mark.parametrize(
    "variant",
    [
        NAME,
        "SINTÉTICO ÑUÑEZ, ÁNGEL UNO",
        "Sintetico Nunez, Angel Uno",
        "  Sintético   Ñuñez ,Ángel\tUno. ",
        "Sintético Ñuñez, Ángel Uno",
        "Sintético Ñuñez, Ángel Uno",
        "“Sintético Ñuñez, Ángel Uno”",
        "Sintético Ñuñez,\nÁngel Uno;",
    ],
    ids=[
        "as-written",
        "upper-case",
        "no-accents",
        "spacing-and-period",
        "decomposed-accents",
        "no-break-spaces",
        "quoted",
        "line-break-and-semicolon",
    ],
)
def test_spelling_variants_of_a_name_share_one_identity(variant: str) -> None:
    assert normalize_name(variant) == "sintetico nunez, angel uno"
    assert person_identity(None, variant) == NAME_IDENTITY


def test_different_names_keep_different_identities() -> None:
    assert person_identity(None, NAME) != person_identity(None, "Sintético Ñuñez, Ángel Dos")


def test_inner_punctuation_of_a_name_is_kept() -> None:
    assert normalize_name("Sintético-Ñuñez, Á. Uno") == "sintetico-nunez, a. uno"


def test_the_orcid_identifies_a_person_before_the_name() -> None:
    identity = person_identity(f"https://orcid.org/{ORCID}", NAME)

    assert identity == f"orcid:{ORCID}"
    assert person_identity(ORCID, "Sintético Ñuñez, Á. U.") == identity
    assert person_identity(ORCID, None) == identity


@pytest.mark.parametrize("orcid", [None, "", "0000-0000-1234-5673", "no registrado"])
def test_without_a_valid_orcid_the_name_identifies_the_person(orcid: str | None) -> None:
    assert person_identity(orcid, NAME) == NAME_IDENTITY


@pytest.mark.parametrize(
    ("orcid", "name"),
    [(None, None), ("", "   "), (" ", None), (None, "..."), ("0000-0000-1234-5673", None)],
    ids=["both-missing", "both-blank", "blank-orcid", "punctuation-only", "invalid-orcid"],
)
def test_without_a_valid_orcid_or_a_name_there_is_no_identity(
    orcid: str | None, name: str | None
) -> None:
    assert person_identity(orcid, name) is None


def test_name_variants_get_one_code() -> None:
    codes = {
        pseudonym(identity, KEY, "ADV-")
        for variant in (NAME, "SINTETICO NUNEZ, ANGEL UNO", "Sintético  Ñuñez, Ángel Uno.")
        if (identity := person_identity(None, variant)) is not None
    }

    assert len(codes) == 1


# The secret key


def test_a_new_key_is_saved_once_and_then_reused(tmp_path: Path) -> None:
    path = tmp_path / "pseudonym.key"

    key = load_or_create_key(path)

    assert len(key) == KEY_SIZE
    assert path.read_bytes() == key
    assert load_or_create_key(path) == key
    assert path.read_bytes() == key
    assert listing(tmp_path) == ["pseudonym.key"]  # No temporary file is left behind.


def test_a_new_key_comes_from_the_secrets_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []

    def token_bytes(size: int) -> bytes:
        sizes.append(size)
        return KEY

    monkeypatch.setattr(secrets, "token_bytes", token_bytes)

    assert load_or_create_key(tmp_path / "pseudonym.key") == KEY
    assert sizes == [KEY_SIZE]


def test_two_new_keys_differ(tmp_path: Path) -> None:
    assert load_or_create_key(tmp_path / "a.key") != load_or_create_key(tmp_path / "b.key")


def test_a_new_key_creates_its_directory(tmp_path: Path) -> None:
    path = tmp_path / "data" / "interim" / "pseudonym.key"

    assert load_or_create_key(path) == path.read_bytes()


class FakeClock:
    """A clock that moves only when it is slept on, and runs ``on_sleep`` after each sleep."""

    def __init__(self, on_sleep: Callable[[], None] | None = None) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._on_sleep = on_sleep

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self._on_sleep is not None:
            self._on_sleep()


@pytest.mark.parametrize("size", [0, 16, KEY_SIZE - 1, KEY_SIZE + 1, 64])
def test_a_key_file_of_the_wrong_size_is_refused_and_left_untouched(
    tmp_path: Path, size: int
) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(bytes(range(size)))
    clock = FakeClock()

    with pytest.raises(ValueError, match=f"{size} bytes.*{KEY_SIZE}"):
        load_or_create_key(path, clock=clock, sleeper=clock.sleep)

    assert path.read_bytes() == bytes(range(size))
    assert listing(tmp_path) == ["pseudonym.key"]


def test_a_short_key_file_is_refused_only_once_the_settle_time_has_passed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(OTHER_KEY[:16])
    clock = FakeClock()

    with pytest.raises(ValueError, match=f"16 bytes.*{KEY_SIZE}"):
        load_or_create_key(path, clock=clock, sleeper=clock.sleep)

    assert clock.now >= anonymize.KEY_SETTLE_SECONDS
    assert clock.now - clock.sleeps[-1] < anonymize.KEY_SETTLE_SECONDS  # No sleep past it.
    assert path.read_bytes() == OTHER_KEY[:16]


def test_a_long_key_file_is_refused_without_waiting(tmp_path: Path) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(KEY + b"\n")
    clock = FakeClock()

    with pytest.raises(ValueError, match=f"{KEY_SIZE + 1} bytes"):
        load_or_create_key(path, clock=clock, sleeper=clock.sleep)

    assert clock.sleeps == []


def test_a_key_file_that_another_run_is_still_writing_is_waited_for(tmp_path: Path) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(OTHER_KEY[:10])
    clock = FakeClock(on_sleep=lambda: path.write_bytes(OTHER_KEY))

    assert load_or_create_key(path, clock=clock, sleeper=clock.sleep) == OTHER_KEY
    assert 0 < clock.now < anonymize.KEY_SETTLE_SECONDS
    assert path.read_bytes() == OTHER_KEY


def test_a_key_file_of_zero_bytes_only_is_refused_and_left_untouched(tmp_path: Path) -> None:
    # A crash while a key was written in place on FAT or exFAT can leave such a file.
    path = tmp_path / "pseudonym.key"
    path.write_bytes(bytes(KEY_SIZE))
    clock = FakeClock()

    with pytest.raises(ValueError, match=f"{KEY_SIZE} zero bytes"):
        load_or_create_key(path, clock=clock, sleeper=clock.sleep)

    assert clock.now >= anonymize.KEY_SETTLE_SECONDS
    assert path.read_bytes() == bytes(KEY_SIZE)
    assert listing(tmp_path) == ["pseudonym.key"]


def test_a_zero_key_file_that_another_run_is_still_writing_is_waited_for(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(bytes(KEY_SIZE))
    clock = FakeClock(on_sleep=lambda: path.write_bytes(OTHER_KEY))

    assert load_or_create_key(path, clock=clock, sleeper=clock.sleep) == OTHER_KEY
    assert 0 < clock.now < anonymize.KEY_SETTLE_SECONDS


def test_a_short_key_file_that_disappears_while_waiting_gives_way_to_a_new_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(OTHER_KEY[:16])
    clock = FakeClock(on_sleep=path.unlink)  # Its writer failed and removed it.

    key = load_or_create_key(path, clock=clock, sleeper=clock.sleep)

    assert len(key) == KEY_SIZE
    assert path.read_bytes() == key
    assert len(clock.sleeps) == 1
    assert listing(tmp_path) == ["pseudonym.key"]


def test_an_interrupted_link_leaves_neither_a_key_nor_a_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(source: object, target: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", interrupt)

    with pytest.raises(KeyboardInterrupt):
        load_or_create_key(tmp_path / "pseudonym.key")

    assert listing(tmp_path) == []


def test_a_key_created_meanwhile_by_another_run_is_kept_and_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pseudonym.key"

    def lose_the_race(source: object, target: object) -> None:
        path.write_bytes(OTHER_KEY)
        raise FileExistsError(target)

    monkeypatch.setattr(os, "link", lose_the_race)

    assert load_or_create_key(path) == OTHER_KEY
    assert path.read_bytes() == OTHER_KEY
    assert listing(tmp_path) == ["pseudonym.key"]


# The secret key on file systems without hard links

NO_HARD_LINKS = [
    PermissionError(errno.EPERM, "Operation not permitted"),
    OSError(errno.EOPNOTSUPP, "Operation not supported"),
    OSError(errno.EINVAL, "Incorrect function", None, 1),
    OSError(errno.EINVAL, "The request is not supported", None, 50),
]
"""What a hard link gives where the file system has none.

A FAT or exFAT volume gives EPERM on Linux, and ERROR_INVALID_FUNCTION (winerror 1, which
Python reports as EINVAL) on Windows. Some network shares and container mounts give
EOPNOTSUPP, or ERROR_NOT_SUPPORTED (winerror 50, EINVAL as well) on Windows.
"""
NO_HARD_LINK_IDS = ["posix-fat", "posix-share", "windows-fat", "windows-share"]


def without_hard_links(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    def link(source: object, target: object) -> None:
        raise error

    monkeypatch.setattr(os, "link", link)


@pytest.mark.parametrize("error", NO_HARD_LINKS, ids=NO_HARD_LINK_IDS)
def test_a_key_is_created_once_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    path = tmp_path / "pseudonym.key"
    without_hard_links(monkeypatch, error)

    key = load_or_create_key(path)

    assert len(key) == KEY_SIZE
    assert path.read_bytes() == key
    assert load_or_create_key(path) == key
    assert listing(tmp_path) == ["pseudonym.key"]  # No temporary file is left behind.


def test_without_hard_links_the_key_file_is_created_exclusively_and_privately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pseudonym.key"
    without_hard_links(monkeypatch, NO_HARD_LINKS[0])
    opens: list[tuple[int, int]] = []
    real_open = os.open

    def open_(target: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        if Path(target) == path:
            opens.append((flags, mode))
        return real_open(target, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", open_)

    key = load_or_create_key(path)

    required = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    assert [(flags & required, mode) for flags, mode in opens] == [(required, 0o600)]
    assert path.read_bytes() == key


def test_without_hard_links_a_key_created_meanwhile_by_another_run_is_kept_and_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pseudonym.key"

    def lose_the_race(source: object, target: object) -> None:
        path.write_bytes(OTHER_KEY)
        raise NO_HARD_LINKS[0]

    monkeypatch.setattr(os, "link", lose_the_race)

    assert load_or_create_key(path) == OTHER_KEY
    assert path.read_bytes() == OTHER_KEY
    assert listing(tmp_path) == ["pseudonym.key"]


def test_without_hard_links_a_key_that_the_winning_run_is_still_writing_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pseudonym.key"

    def lose_the_race(source: object, target: object) -> None:
        path.write_bytes(OTHER_KEY[:16])  # The winner has created the file, not filled it.
        raise NO_HARD_LINKS[0]

    monkeypatch.setattr(os, "link", lose_the_race)
    clock = FakeClock(on_sleep=lambda: path.write_bytes(OTHER_KEY))

    assert load_or_create_key(path, clock=clock, sleeper=clock.sleep) == OTHER_KEY
    assert path.read_bytes() == OTHER_KEY
    assert listing(tmp_path) == ["pseudonym.key"]


def test_without_hard_links_a_key_that_the_winning_run_gives_up_is_created_by_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pseudonym.key"

    def lose_the_race(source: object, target: object) -> None:
        path.write_bytes(OTHER_KEY[:16])  # The winner has created the file, not filled it.
        raise NO_HARD_LINKS[0]

    monkeypatch.setattr(os, "link", lose_the_race)
    clock = FakeClock(on_sleep=path.unlink)  # The winner fails and removes its file.

    with pytest.raises(FileNotFoundError):
        load_or_create_key(path, clock=clock, sleeper=clock.sleep)

    assert listing(tmp_path) == []
    without_hard_links(monkeypatch, NO_HARD_LINKS[0])
    key = load_or_create_key(path)
    assert path.read_bytes() == key


def create_together(path: Path, runs: int) -> list[bytes]:
    """Let ``runs`` threads ask for the key at ``path`` at the same moment; return their keys."""
    start = threading.Barrier(runs)
    keys: list[bytes] = []

    def run() -> None:
        start.wait()
        keys.append(load_or_create_key(path))

    threads = [threading.Thread(target=run) for _ in range(runs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return keys


@pytest.mark.parametrize("hard_links", [True, False], ids=["hard-links", "no-hard-links"])
def test_runs_that_create_the_key_at_the_same_time_all_use_one_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hard_links: bool
) -> None:
    # Real races never reach the settle loop, the wait for a key file that is still short or
    # all zero bytes: a probe of 200 rounds in each mode saw several creators in every round
    # but never a short read (T08a). That loop is covered with a fake clock by the tests
    # above, such as test_a_key_file_that_another_run_is_still_writing_is_waited_for and
    # test_without_hard_links_a_key_that_the_winning_run_is_still_writing_is_waited_for.
    if not hard_links:
        without_hard_links(monkeypatch, NO_HARD_LINKS[0])

    for round_number in range(20):
        path = tmp_path / f"round-{round_number}" / "pseudonym.key"

        keys = create_together(path, runs=4)

        assert keys == [path.read_bytes()] * 4
        assert listing(path.parent) == ["pseudonym.key"]


def fail_the_exclusive_open(
    monkeypatch: pytest.MonkeyPatch, path: Path, error: BaseException
) -> None:
    real_open = os.open

    def open_(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(target) == path:
            raise error
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_)


class FailingKeyFile(io.FileIO):
    """The key file opened for writing, whose write puts ``visible`` bytes in it, then fails."""

    def __init__(self, descriptor: int, visible: int, error: BaseException) -> None:
        super().__init__(descriptor, "wb")
        self._visible = visible
        self._error = error

    def write(self, data: Any) -> int:
        super().write(bytes(data)[: self._visible])
        raise self._error


def fail_the_write(visible: int) -> Callable[[pytest.MonkeyPatch, Path, BaseException], None]:
    """Return a failure that lets the write of the key file put ``visible`` bytes in it."""

    def failure(monkeypatch: pytest.MonkeyPatch, path: Path, error: BaseException) -> None:
        real_open, real_fdopen = os.open, os.fdopen
        key_descriptors: set[int] = set()

        def open_(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            descriptor = real_open(target, flags, *args, **kwargs)
            if Path(target) == path:
                key_descriptors.add(descriptor)
            return descriptor

        def fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
            if descriptor in key_descriptors:
                return FailingKeyFile(descriptor, visible, error)
            return real_fdopen(descriptor, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_)
        monkeypatch.setattr(os, "fdopen", fdopen)

    return failure


@pytest.mark.parametrize(
    "failure",
    [fail_the_exclusive_open, fail_the_write(0), fail_the_write(16)],
    ids=["open", "write-nothing", "write-half"],
)
@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk-full", "interrupted"],
)
def test_without_hard_links_a_failed_creation_leaves_neither_a_key_nor_a_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Any, error: BaseException
) -> None:
    path = tmp_path / "pseudonym.key"
    without_hard_links(monkeypatch, NO_HARD_LINKS[0])
    failure(monkeypatch, path, error)

    with pytest.raises(type(error)) as caught:
        load_or_create_key(path)

    assert caught.value is error
    assert listing(tmp_path) == []


@pytest.mark.parametrize(
    "error",
    [OSError(errno.EIO, "Input/output error"), KeyboardInterrupt()],
    ids=["disk-failure", "interrupted"],
)
def test_without_hard_links_a_complete_key_that_may_be_in_use_is_kept_when_its_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    path = tmp_path / "pseudonym.key"
    without_hard_links(monkeypatch, NO_HARD_LINKS[0])
    real_fsync = os.fsync
    flushes: list[int] = []
    read_meanwhile: list[bytes] = []

    def fsync(descriptor: int) -> None:
        flushes.append(descriptor)
        if len(flushes) == 2:  # The key file itself, after the temporary file.
            read_meanwhile.append(load_or_create_key(path))  # Another run reads the key.
            raise error
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)

    with pytest.raises(type(error)) as caught:
        load_or_create_key(path)

    assert caught.value is error
    assert path.read_bytes() == read_meanwhile[0]
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert load_or_create_key(path) == read_meanwhile[0]
    assert listing(tmp_path) == ["pseudonym.key"]


def test_without_hard_links_a_failed_removal_does_not_hide_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pseudonym.key"
    without_hard_links(monkeypatch, NO_HARD_LINKS[0])
    error = OSError(errno.ENOSPC, "No space left on device")
    fail_the_write(0)(monkeypatch, path, error)
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == path:  # As on Windows while another run holds the file open.
            raise PermissionError(errno.EACCES, "The file is being used by another process")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(OSError) as caught:
        load_or_create_key(path)

    assert caught.value is error
    assert path.read_bytes() == b""  # Left behind, and refused by the next run.
    clock = FakeClock()
    with pytest.raises(ValueError, match=f"0 bytes.*{KEY_SIZE}"):
        load_or_create_key(path, clock=clock, sleeper=clock.sleep)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("hard_links", [True, False], ids=["hard-links", "no-hard-links"])
def test_a_new_key_file_is_readable_and_writable_by_its_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hard_links: bool
) -> None:
    path = tmp_path / "pseudonym.key"
    if not hard_links:
        without_hard_links(monkeypatch, NO_HARD_LINKS[0])

    load_or_create_key(path)

    assert path.stat().st_mode & 0o777 == 0o600


# The secret key: making its name durable


def record_key_calls(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    link_error: OSError | None = None,
    directory_opens: bool = True,
    directory_flush_error: OSError | None = None,
) -> list[str]:
    """Record the flushes, links, key opens and directory steps of a key creation, in order.

    ``link_error`` makes every hard link fail with it, as on a file system without hard
    links. Opening the key's directory gives a stand-in descriptor that is never flushed for
    real, since Windows cannot open a directory; with ``directory_opens`` false the open fails
    as on Windows, and ``directory_flush_error`` makes the flush of the directory fail. Every
    other call goes through to :mod:`os`.
    """
    calls: list[str] = []
    real_open, real_fsync, real_close, real_link = os.open, os.fsync, os.close, os.link
    stand_in = path.parent.parent / "directory-stand-in"
    stand_in.write_bytes(b"")
    stand_ins: set[int] = set()

    def open_(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(target) == path:
            calls.append("open key")
        elif Path(target) == path.parent:
            calls.append("open directory")
            if not directory_opens:
                raise PermissionError(errno.EACCES, "Permission denied", str(target))
            descriptor = real_open(stand_in, os.O_RDONLY)
            stand_ins.add(descriptor)
            return descriptor
        return real_open(target, flags, *args, **kwargs)

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

    def link(source: Any, target: Any) -> None:
        calls.append("link")
        if link_error is not None:
            raise link_error
        real_link(source, target)

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(os, "link", link)
    return calls


FLUSHED_DIRECTORY = ["open directory", "fsync directory", "close directory"]


def test_a_new_key_is_flushed_then_linked_then_its_directory_flushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    path.parent.mkdir()
    calls = record_key_calls(monkeypatch, path)

    key = load_or_create_key(path)

    assert calls == ["fsync file of 32 bytes", "link", *FLUSHED_DIRECTORY]
    assert path.read_bytes() == key


def test_without_hard_links_a_new_key_is_written_then_its_directory_flushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    path.parent.mkdir()
    calls = record_key_calls(monkeypatch, path, link_error=NO_HARD_LINKS[0])

    key = load_or_create_key(path)

    assert calls == [
        "fsync file of 32 bytes",
        "link",
        "open key",
        "fsync file of 32 bytes",
        *FLUSHED_DIRECTORY,
    ]
    assert path.read_bytes() == key


def test_a_new_key_is_kept_where_its_directory_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    path.parent.mkdir()
    calls = record_key_calls(monkeypatch, path, directory_opens=False)

    key = load_or_create_key(path)

    assert calls == ["fsync file of 32 bytes", "link", "open directory"]
    assert path.read_bytes() == key
    assert listing(path.parent) == ["pseudonym.key"]


@pytest.mark.parametrize(
    "error",
    [OSError(errno.EINVAL, "Invalid argument"), OSError(errno.EIO, "Input/output error")],
    ids=["unsupported", "failed"],
)
def test_a_new_key_is_kept_where_its_directory_cannot_be_flushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    path.parent.mkdir()
    calls = record_key_calls(monkeypatch, path, directory_flush_error=error)

    key = load_or_create_key(path)

    assert calls == ["fsync file of 32 bytes", "link", *FLUSHED_DIRECTORY]
    assert path.read_bytes() == key
    assert load_or_create_key(path) == key


def test_an_existing_key_is_read_without_creating_or_flushing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    path.parent.mkdir()
    path.write_bytes(OTHER_KEY)
    calls = record_key_calls(monkeypatch, path)

    assert load_or_create_key(path) == OTHER_KEY
    assert calls == []


def test_a_run_that_loses_the_race_leaves_the_flushing_to_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    path.parent.mkdir()
    calls = record_key_calls(monkeypatch, path, link_error=FileExistsError(str(path)))
    recorded_link = os.link

    def lose_the_race(source: Any, target: Any) -> None:
        path.write_bytes(OTHER_KEY)
        recorded_link(source, target)

    monkeypatch.setattr(os, "link", lose_the_race)

    assert load_or_create_key(path) == OTHER_KEY
    assert calls == ["fsync file of 32 bytes", "link"]


# The secret key: where it lives


def test_the_default_key_lives_in_the_interim_data_directory(tmp_path: Path) -> None:
    expected = tmp_path.resolve() / "data" / "interim" / "pseudonym.key"

    assert default_key_path(DECLARED.paths, tmp_path) == expected


def test_the_default_key_follows_the_configured_data_directory(tmp_path: Path) -> None:
    paths = PathsConfig(
        data_dir=Path("store/data"), results_dir=Path("results"), tessdata_dir=Path("tessdata")
    )

    expected = tmp_path.resolve() / "store" / "data" / "interim" / "pseudonym.key"
    assert default_key_path(paths, tmp_path) == expected


@pytest.mark.parametrize("ignore_file", [".gitignore", ".dockerignore"])
def test_the_default_key_location_is_kept_out_of_git_and_docker(ignore_file: str) -> None:
    location = default_key_path(DECLARED.paths, PROJECT_ROOT).relative_to(PROJECT_ROOT)
    lines = (PROJECT_ROOT / ignore_file).read_text(encoding="utf-8").splitlines()
    ignored = {line.strip().strip("/") for line in lines if line.strip()}

    assert location.parts[0] in ignored
