"""Cache of raw chunk vectors, keyed as D27 C6 fixes it (T14).

A chunk vector depends only on the model and the chunk's text, so :func:`cache_key` keys it by
the SHA-256 of the canonical JSON of ``{model_id, model_revision, max_seq_length, normalize,
text}`` (:func:`canonical_key_json`): sorted keys, no spaces, and the text's characters kept
as they are, in UTF-8. ``normalize`` is ``false``, since the cache holds the model's raw
output, which the pooling averages before it normalizes (D09). Another revision of the model,
another sequence length or another text gives another key, so a stale vector is never reused.

Each model has one cache file, ``<model id, with "/" as "--">.npy`` in :data:`CACHE_DIR` under
the gitignored ``paths.data_dir``, outside any snapshot, so that a later snapshot reuses the
vectors of unchanged chunks. The file is a NumPy structured array with one record per key,
``key`` (its 64 hex digits, in ASCII) and ``vector`` (float32), sorted by key, so the same
vectors always give the same bytes. It holds no text. It is replaced atomically, and a file
that is not such an array is refused rather than used.
"""

import hashlib
import io
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from thematic_redundancy.shared.filesystem import write_bytes_atomically

CACHE_DIR = Path("cache") / "chunk_vectors"
"""Directory of the cache files, relative to ``paths.data_dir``."""

CACHE_KEY_SCHEME = (
    "SHA-256 of the canonical JSON of {max_seq_length, model_id, model_revision, normalize, "
    "text}: keys sorted, separators ',' and ':', characters unescaped, UTF-8"
)
"""How :func:`cache_key` keys a vector, as the manifests record it."""

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
"""A Hugging Face model id: a name, maybe after an owner and a slash."""

_KEY = re.compile(rb"^[0-9a-f]{64}$")


class VectorCacheError(ValueError):
    """A cache file cannot be read as a vector cache, so it is not used."""


def canonical_key_json(
    *, model_id: str, model_revision: str, max_seq_length: int, normalize: bool, text: str
) -> str:
    """Return the canonical JSON that :func:`cache_key` hashes."""
    fields = {
        "model_id": model_id,
        "model_revision": model_revision,
        "max_seq_length": max_seq_length,
        "normalize": normalize,
        "text": text,
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(
    *, model_id: str, model_revision: str, max_seq_length: int, normalize: bool, text: str
) -> str:
    """Return the cache key of a chunk vector: the SHA-256 of :func:`canonical_key_json`."""
    canonical = canonical_key_json(
        model_id=model_id,
        model_revision=model_revision,
        max_seq_length=max_seq_length,
        normalize=normalize,
        text=text,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_slug(model_id: str) -> str:
    """Return ``model_id`` as one file or directory name, with ``/`` as ``--``.

    Raises:
        ValueError: if ``model_id`` is not a model id, such as one with ``..`` or a backslash,
            which could name a path outside its directory.
    """
    if not _MODEL_ID.match(model_id) or ".." in model_id:
        raise ValueError(f"{model_id!r} is not a model id")
    return model_id.replace("/", "--")


def _record_type(dimension: int) -> np.dtype:
    return np.dtype([("key", "S64"), ("vector", "<f4", (dimension,))])


class ChunkVectorCache:
    """The cached vectors of one model, held in memory and saved to one file."""

    def __init__(self, path: Path, dimension: int) -> None:
        self.path = path
        self.dimension = dimension
        self._vectors: dict[str, np.ndarray] = {}

    @classmethod
    def open(cls, path: Path, dimension: int) -> "ChunkVectorCache":
        """Return the cache saved at ``path``, or an empty one when there is no file.

        Raises:
            VectorCacheError: if the file is not a cache of ``dimension``-dimensional vectors
                with distinct, well-formed keys and finite values.
        """
        cache = cls(path, dimension)
        if path.is_file():
            cache._vectors = _read(path, dimension)
        return cache

    def __len__(self) -> int:
        return len(self._vectors)

    def missing(self, keys: Iterable[str]) -> list[str]:
        """Return the distinct ``keys`` without a vector, in the order of their first use."""
        return [key for key in dict.fromkeys(keys) if key not in self._vectors]

    def add(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        """Hold ``vectors``, one row per key, as float32.

        Raises:
            ValueError: if there is not one row per key, of :attr:`dimension` finite values.
        """
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(keys):
            raise ValueError(f"{len(matrix)} vectors for {len(keys)} keys")
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"vectors of {matrix.shape[1]} dimensions, not {self.dimension}")
        if not np.isfinite(matrix).all():
            raise ValueError("a vector with a value that is not finite")
        self._vectors.update(zip(keys, matrix, strict=True))

    def vectors(self, keys: Sequence[str]) -> np.ndarray:
        """Return the float32 vectors of ``keys``, one row each, in order.

        Raises:
            KeyError: if a key has no vector.
        """
        rows = [self._vectors[key] for key in keys]
        return np.stack(rows) if rows else np.zeros((0, self.dimension), dtype=np.float32)

    def save(self) -> None:
        """Replace the file with every vector held, sorted by key, creating its directory."""
        keys = sorted(self._vectors)
        records = np.zeros(len(keys), dtype=_record_type(self.dimension))
        records["key"] = [key.encode("ascii") for key in keys]
        if keys:
            records["vector"] = np.stack([self._vectors[key] for key in keys])
        buffer = io.BytesIO()
        np.save(buffer, records, allow_pickle=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomically(self.path, buffer.getvalue())


def _read(path: Path, dimension: int) -> dict[str, np.ndarray]:
    """Return the vectors of the cache file at ``path`` by key."""
    refused = VectorCacheError(
        f"{path} is not a cache of {dimension}-dimensional chunk vectors; delete it and run "
        "the command again"
    )
    try:
        records = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    except (ValueError, EOFError):
        raise refused from None
    if records.dtype != _record_type(dimension) or records.ndim != 1:
        raise refused
    keys = [key for key in records["key"].tolist() if _KEY.match(key)]
    vectors = records["vector"]
    if len(set(keys)) != len(records) or not np.isfinite(vectors).all():
        raise refused
    return {key.decode("ascii"): vector for key, vector in zip(keys, vectors, strict=True)}
