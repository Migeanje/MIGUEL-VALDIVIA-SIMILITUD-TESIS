"""The chunk-vector cache: its key (D27, C6) and its file (T14)."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thematic_redundancy.representation.vector_cache import (
    ChunkVectorCache,
    VectorCacheError,
    cache_key,
    canonical_key_json,
    model_slug,
)

FIELDS: dict[str, Any] = {
    "model_id": "org/model",
    "model_revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    "max_seq_length": 128,
    "normalize": False,
    "text": "Diseño de un sistema, «año» 2023",
}


def test_the_key_is_the_sha256_of_the_canonical_json() -> None:
    canonical = (
        '{"max_seq_length":128,"model_id":"org/model",'
        '"model_revision":"e8f8c211226b894fcb81acc59f3b34ba3efd5f42","normalize":false,'
        '"text":"Diseño de un sistema, «año» 2023"}'
    )

    assert canonical_key_json(**FIELDS) == canonical
    assert cache_key(**FIELDS) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_the_key_does_not_depend_on_the_order_of_the_arguments() -> None:
    assert cache_key(**dict(reversed(FIELDS.items()))) == cache_key(**FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "org/other-model"),
        ("model_revision", "4328cf26390c98c5e3c738b4460a05b95f4911f5"),
        ("max_seq_length", 256),
        ("normalize", True),
        ("text", "Diseño de un sistema, «año» 2024"),
        ("text", "Diseño de un sistema, «año» 2023 "),
    ],
    ids=["model", "revision", "length", "normalize", "text", "trailing-space"],
)
def test_each_field_changes_the_key(field: str, value: Any) -> None:
    assert cache_key(**{**FIELDS, field: value}) != cache_key(**FIELDS)


@pytest.mark.parametrize(
    ("model_id", "slug"),
    [
        (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
        ),
        ("local-model", "local-model"),
    ],
)
def test_a_model_id_becomes_one_file_name(model_id: str, slug: str) -> None:
    assert model_slug(model_id) == slug


@pytest.mark.parametrize("model_id", ["..", "org/../x", "org\\model", "a/b/c", "", "org/"])
def test_a_model_id_that_could_leave_its_directory_is_refused(model_id: str) -> None:
    with pytest.raises(ValueError, match="is not a model id"):
        model_slug(model_id)


def keys(*texts: str) -> list[str]:
    return [cache_key(**{**FIELDS, "text": text}) for text in texts]


def test_a_saved_cache_gives_back_its_vectors(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "org--model.npy"
    cache = ChunkVectorCache.open(path, 3)
    first, second = keys("uno", "dos")
    assert len(cache) == 0 and cache.missing([first, second, first]) == [first, second]

    cache.add([first, second], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    cache.save()
    reopened = ChunkVectorCache.open(path, 3)

    assert len(reopened) == 2 and reopened.missing([second, first]) == []
    vectors = reopened.vectors([second, first, second])
    assert vectors.dtype == np.float32
    assert np.array_equal(vectors, np.array([[4, 5, 6], [1, 2, 3], [4, 5, 6]], dtype=np.float32))
    assert b"uno" not in path.read_bytes()


def test_the_file_does_not_depend_on_the_order_in_which_vectors_were_added(
    tmp_path: Path,
) -> None:
    first, second = keys("uno", "dos")
    vectors = {first: [1.0, 0.0], second: [0.0, 1.0]}
    contents = []
    for order in ([first, second], [second, first]):
        cache = ChunkVectorCache.open(tmp_path / f"{len(contents)}.npy", 2)
        for key in order:
            cache.add([key], np.array([vectors[key]]))
        cache.save()
        contents.append(cache.path.read_bytes())

    assert contents[0] == contents[1]


@pytest.mark.parametrize(
    ("keys_given", "vectors", "message"),
    [
        (1, np.zeros((2, 3)), "2 vectors for 1 key"),
        (2, np.zeros((2, 4)), "of 4 dimensions, not 3"),
        (2, np.array([[0, 0, np.inf], [0, 0, 0]]), "not finite"),
    ],
    ids=["count", "dimension", "not-finite"],
)
def test_vectors_that_do_not_fit_the_cache_are_refused(
    tmp_path: Path, keys_given: int, vectors: np.ndarray, message: str
) -> None:
    cache = ChunkVectorCache.open(tmp_path / "org--model.npy", 3)

    with pytest.raises(ValueError, match=message):
        cache.add(keys("uno", "dos")[:keys_given], vectors)

    assert len(cache) == 0


def test_a_key_without_a_vector_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        ChunkVectorCache.open(tmp_path / "org--model.npy", 3).vectors(keys("uno"))


def write_saved_cache(path: Path) -> bytes:
    cache = ChunkVectorCache.open(path, 3)
    cache.add(keys("uno", "dos"), np.array([[1, 2, 3], [4, 5, 6]]))
    cache.save()
    return path.read_bytes()


@pytest.mark.parametrize(
    "damage", ["truncated", "garbage", "other-dimension", "repeated-key", "not-finite", "bad-key"]
)
def test_a_damaged_cache_file_is_refused(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "org--model.npy"
    content = write_saved_cache(path)
    dimension = 3
    if damage == "truncated":
        path.write_bytes(content[: len(content) - 5])
    elif damage == "garbage":
        path.write_bytes(b"not a cache " * 10)
    elif damage == "other-dimension":
        dimension = 4
    else:
        array = np.load(path, allow_pickle=False)
        if damage == "repeated-key":
            array["key"][1] = array["key"][0]
        elif damage == "not-finite":
            array["vector"][0, 0] = np.nan
        else:
            array["key"][0] = b"Z" * 64
        np.save(path, array, allow_pickle=False)

    with pytest.raises(VectorCacheError) as caught:
        ChunkVectorCache.open(path, dimension)

    assert str(caught.value) == (
        f"{path} is not a cache of {dimension}-dimensional chunk vectors; delete it and run the "
        "command again"
    )
