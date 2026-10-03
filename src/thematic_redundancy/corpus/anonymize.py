"""Keyed pseudonyms for people: stable codes that neither hold nor reveal a name.

A person is identified first by their ORCID iD and otherwise by their name, as
:func:`person_identity` describes. :func:`pseudonym` turns that identity into a code such as
``ADV-3f9a0c12bd``: a prefix that tells the kind of person, then 10 hex digits of an
HMAC-SHA256 under a secret key. The prefix is part of the MAC input, so the advisor code and
the author code of one person are unrelated.

The secret key is 32 random bytes, kept in one file that :func:`load_or_create_key` creates
on first use. Its default location, from :func:`default_key_path`, is
``<data_dir>/interim/pseudonym.key``, inside the gitignored data directory.

- **Never commit the key or share it.** Anyone who holds it can test a guessed name or
  ORCID against the codes.
- **Without the key, the codes can be neither re-derived nor reversed.** Losing it means that
  the next run gives every person a new code, unrelated to the codes stored before. The
  key is never replaced automatically, so a key file of the wrong size stops the run.
"""

import hashlib
import hmac
import os
import re
import secrets
import tempfile
import unicodedata
from pathlib import Path

from thematic_redundancy.shared.config import PathsConfig

KEY_SIZE = 32
"""Bytes in a pseudonym key."""

CODE_HEX_DIGITS = 10
"""Hex digits after the prefix of a code: 40 bits, so ~1,000 people almost never collide."""

INTERIM_DIR_NAME = "interim"
"""Directory under ``paths.data_dir`` for derived data, such as the fichas and the key."""

KEY_FILE_NAME = "pseudonym.key"

ORCID_IDENTITY_PREFIX = "orcid:"
NAME_IDENTITY_PREFIX = "name:"

_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,15}")
_ORCID = re.compile(
    r"(?:(?:https?://)?(?:www\.)?orcid\.org/)?"
    r"([0-9]{4})-?([0-9]{4})-?([0-9]{4})-?([0-9]{3}[0-9X])/?",
    re.IGNORECASE,
)
_COMMA = re.compile(r"\s*,\s*")
_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")
"""Leading or trailing characters that are neither letters nor digits."""


def pseudonym(identity: str, key: bytes, prefix: str) -> str:
    """Return the code of ``identity``: ``prefix`` and 10 hex digits of an HMAC-SHA256.

    The MAC covers ``prefix``, a NUL character and ``identity``, all in UTF-8, so the same
    identity under two prefixes gives unrelated digits. The same arguments always give the
    same code.

    Raises:
        ValueError: if ``identity`` is blank, if ``key`` does not have :data:`KEY_SIZE`
            bytes, or if ``prefix`` is not 1 to 16 ASCII letters, digits, ``-`` or ``_``
            that start with a letter.
    """
    if not _PREFIX.fullmatch(prefix):
        raise ValueError(
            f"the prefix must be 1 to 16 ASCII letters, digits, '-' or '_', starting with a "
            f"letter; got {prefix!r}"
        )
    if not identity.strip():
        raise ValueError("a pseudonym needs a non-blank identity")
    if len(key) != KEY_SIZE:
        raise ValueError(f"the key must have exactly {KEY_SIZE} bytes, got {len(key)}")
    message = f"{prefix}\x00{identity}".encode()
    digest = hmac.new(key, message, hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:CODE_HEX_DIGITS]}"


def person_identity(orcid: str | None, name: str | None) -> str | None:
    """Return the identity of a person, from which :func:`pseudonym` makes their code.

    A valid ORCID iD wins, as ``orcid:0000-0000-1234-5672``, so every spelling of the name of
    a person with an ORCID shares one identity. Otherwise the name, normalized by
    :func:`normalize_name`, gives ``name:<normalized name>``. An ORCID iD that is malformed or
    fails its check digit counts as missing. Without either, there is no identity.
    """
    if orcid is not None and (normalized := normalize_orcid(orcid)) is not None:
        return f"{ORCID_IDENTITY_PREFIX}{normalized}"
    if name is not None and (normalized := normalize_name(name)):
        return f"{NAME_IDENTITY_PREFIX}{normalized}"
    return None


def normalize_orcid(value: str) -> str | None:
    """Return an ORCID iD as ``0000-0000-1234-5672``, or ``None`` if ``value`` is not one.

    ``value`` may be a bare iD, with or without hyphens, or an ``orcid.org`` URL, in any
    letter case. Its last character must be the ISO 7064 MOD 11-2 check character of the
    other 15 digits, which may be ``X``.
    """
    match = _ORCID.fullmatch(value.strip())
    if match is None:
        return None
    digits = "".join(match.groups()).upper()
    if _orcid_check_character(digits[:15]) != digits[15]:
        return None
    return "-".join(digits[start : start + 4] for start in range(0, 16, 4))


def normalize_name(name: str) -> str:
    """Return the form of a person's name used to match its spellings, or ``""``.

    The name is decomposed (Unicode NFKD), its accents are dropped, and it is case folded,
    so ``Ñuñez``, ``ÑUÑEZ`` and ``Nunez`` all give ``nunez``. Whitespace is collapsed, the
    spacing around commas is evened out, and punctuation is trimmed from both ends. Inner
    punctuation, such as a hyphen or the comma between family and given names, is kept.
    The result only serves matching; it is never stored or shown.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    collapsed = " ".join(unaccented.casefold().split())
    return _EDGE_PUNCTUATION.sub("", _COMMA.sub(", ", collapsed))


def default_key_path(paths: PathsConfig, project_root: Path) -> Path:
    """Return where the pseudonym key of the project at ``project_root`` lives.

    It is ``interim/pseudonym.key`` under ``paths.data_dir``, which git and Docker ignore.

    Raises:
        ValueError: if the data directory resolves outside the project root.
    """
    return paths.resolve_against(project_root).data_dir / INTERIM_DIR_NAME / KEY_FILE_NAME


def load_or_create_key(path: Path) -> bytes:
    """Return the key saved at ``path``, creating it first if there is none.

    A new key is :data:`KEY_SIZE` random bytes from :mod:`secrets`. They go to a temporary
    file beside ``path`` and are flushed to the disk, and a hard link then gives them their
    name, which fails rather than replace a file. So the key appears complete or not at all,
    and an existing key is never overwritten: if another run creates one first, that one is
    used.

    Raises:
        ValueError: if the file at ``path`` does not hold exactly :data:`KEY_SIZE` bytes. It
            is left as it is, because the codes made with it depend on it.
    """
    try:
        return _read_key(path)
    except FileNotFoundError:
        pass
    key = secrets.token_bytes(KEY_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".part"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(key)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            return _read_key(path)
        return key
    finally:
        staged.unlink(missing_ok=True)


def _read_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) != KEY_SIZE:
        raise ValueError(
            f"the pseudonym key file {path} holds {len(key)} bytes, but a key has exactly "
            f"{KEY_SIZE}; it is not replaced automatically, because every code depends on it. "
            "Restore the original key, or move this file away on purpose to start new codes"
        )
    return key


def _orcid_check_character(base_digits: str) -> str:
    """Return the ISO 7064 MOD 11-2 check character of the first 15 digits of an ORCID iD."""
    total = 0
    for digit in base_digits:
        total = (total + int(digit)) * 2
    result = (12 - total % 11) % 11
    return "X" if result == 10 else str(result)
