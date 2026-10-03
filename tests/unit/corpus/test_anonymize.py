"""Keyed pseudonyms for people, the identities they come from, and the secret key file.

Every ORCID iD here is synthetic: each lies below the block from which ORCID assigns iDs,
which starts at 0000-0001-5000-0007, so none can belong to a person. The names are
synthetic as well, and the test key is simply the bytes 0 to 31.
"""

import os
import re
import secrets
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("size", [0, 16, KEY_SIZE - 1, KEY_SIZE + 1, 64])
def test_a_key_file_of_the_wrong_size_is_refused_and_left_untouched(
    tmp_path: Path, size: int
) -> None:
    path = tmp_path / "pseudonym.key"
    path.write_bytes(bytes(range(size)))

    with pytest.raises(ValueError, match=f"{size} bytes.*{KEY_SIZE}"):
        load_or_create_key(path)

    assert path.read_bytes() == bytes(range(size))
    assert listing(tmp_path) == ["pseudonym.key"]


@pytest.mark.parametrize("interruption", [OSError("simulated link failure"), KeyboardInterrupt()])
def test_a_failed_creation_leaves_neither_a_key_nor_a_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: BaseException
) -> None:
    def fail(source: object, target: object) -> None:
        raise interruption

    monkeypatch.setattr(os, "link", fail)

    with pytest.raises(type(interruption)):
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
