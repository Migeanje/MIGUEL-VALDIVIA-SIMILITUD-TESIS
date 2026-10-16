"""The sentence-transformers adapter of the sentence-encoder port (T14), without a model.

:func:`load_sentence_encoder` imports torch, transformers and sentence-transformers when it is
called, which takes many seconds, so these tests put stand-in modules in their place.
"""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from thematic_redundancy.representation.st_encoder import (
    MODULES_FILE_NAME,
    load_sentence_encoder,
)
from thematic_redundancy.representation.vectors import EncoderIdentity, EncoderSettings

COMMIT = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SNAPSHOT_DIR = Path("hub") / "models--org--model" / "snapshots" / COMMIT
SETTINGS = EncoderSettings(batch_size=8, device="cpu", threads=3)


class StandInModel:
    """Records how it was loaded and called, and returns float64 vectors of 4 dimensions."""

    created: list[tuple[str, dict[str, Any]]] = []
    encoded: list[tuple[list[str], dict[str, Any]]] = []

    def __init__(self, path: str, **options: Any) -> None:
        self.created.append((path, options))
        self.max_seq_length = 128

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        self.encoded.append((texts, options))
        return np.arange(4 * len(texts), dtype=np.float64).reshape(len(texts), 4)


def install_stand_ins(
    monkeypatch: pytest.MonkeyPatch, cached_file: Any, threads: list[int]
) -> None:
    StandInModel.created, StandInModel.encoded = [], []
    utils = ModuleType("transformers.utils")
    utils.__dict__["cached_file"] = cached_file
    transformers = ModuleType("transformers")
    transformers.__dict__["utils"] = utils
    torch = ModuleType("torch")
    torch.__dict__["set_num_threads"] = threads.append
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.__dict__["SentenceTransformer"] = StandInModel
    for name, module in [
        ("transformers", transformers),
        ("transformers.utils", utils),
        ("torch", torch),
        ("sentence_transformers", sentence_transformers),
    ]:
        monkeypatch.setitem(sys.modules, name, module)


def test_the_encoder_loads_the_cached_snapshot_and_returns_raw_float32_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups: list[tuple[str, str, dict[str, Any]]] = []
    threads: list[int] = []

    def cached_file(model_id: str, file_name: str, **options: Any) -> str:
        lookups.append((model_id, file_name, options))
        return str(SNAPSHOT_DIR / file_name)

    install_stand_ins(monkeypatch, cached_file, threads)

    encoder = load_sentence_encoder("org/model", SETTINGS)
    vectors = encoder.encode(("uno", "dos"))

    assert lookups == [("org/model", MODULES_FILE_NAME, {"local_files_only": True})]
    assert threads == [3]
    assert StandInModel.created == [
        (str(SNAPSHOT_DIR), {"device": "cpu", "local_files_only": True})
    ]
    assert encoder.identity == EncoderIdentity(
        model_id="org/model", revision=COMMIT, max_seq_length=128, dimension=4
    )
    assert StandInModel.encoded == [
        (
            ["uno", "dos"],
            {
                "batch_size": 8,
                "normalize_embeddings": False,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )
    ]
    assert vectors.dtype == np.float32
    assert np.array_equal(vectors, np.arange(8, dtype=np.float32).reshape(2, 4))


def no_file(*_: Any, **__: Any) -> None:
    return None


def file_error(*_: Any, **__: Any) -> None:
    raise OSError("not cached")


def outside_the_cache(_: str, file_name: str, **__: Any) -> str:
    return str(Path("models") / "org-model" / file_name)


@pytest.mark.parametrize(
    ("cached_file", "message"),
    [
        (no_file, r"^modules\.json of org/model is not in the local cache$"),
        (file_error, r"^modules\.json of org/model is not in the local cache$"),
        (outside_the_cache, r"^org/model is not in the Hugging Face cache layout"),
    ],
    ids=["lookup-returns-none", "lookup-raises", "no-revision"],
)
def test_a_model_without_a_cached_revision_is_refused_before_any_load(
    monkeypatch: pytest.MonkeyPatch, cached_file: Any, message: str
) -> None:
    install_stand_ins(monkeypatch, cached_file, [])

    with pytest.raises(OSError, match=message):
        load_sentence_encoder("org/model", SETTINGS)

    assert StandInModel.created == []
