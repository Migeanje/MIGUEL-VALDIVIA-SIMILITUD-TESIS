"""The Hugging Face adapter of the token-counter port (D27, C5), without loading a model.

:func:`load_token_counter` imports ``transformers`` when it is called, and that import takes
many seconds, so the refusal test puts stand-in modules in its place: the cache lookup finds
nothing, and loading a tokenizer would fail the test.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from thematic_redundancy.representation.hf_tokenizers import (
    TOKENIZER_FILE_NAME,
    load_token_counter,
    snapshot_revision,
)

COMMIT = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_CACHE = Path("hub") / "models--org--model"


def stand_in_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (MODEL_CACHE / "snapshots" / COMMIT / TOKENIZER_FILE_NAME, COMMIT),
        (Path("models") / "minilm" / TOKENIZER_FILE_NAME, None),
        (Path("backup") / COMMIT / TOKENIZER_FILE_NAME, None),
        (Path(TOKENIZER_FILE_NAME), None),
    ],
    ids=["cache-snapshot", "local-folder", "commit-folder-outside-snapshots", "bare-name"],
)
def test_the_revision_is_the_snapshot_folder_of_the_cache(path: Path, expected: str | None) -> None:
    assert snapshot_revision(path) == expected


def test_a_cache_without_the_tokenizer_file_is_refused_before_any_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups = []

    def cached_file(model_id: str, file_name: str, **options: Any) -> None:
        lookups.append((model_id, file_name, options))

    def from_pretrained(*_: Any, **__: Any) -> None:
        raise AssertionError("no tokenizer may be loaded")

    utils = stand_in_module("transformers.utils", cached_file=cached_file)
    transformers = stand_in_module(
        "transformers",
        utils=utils,
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
        logging=SimpleNamespace(set_verbosity_error=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)

    with pytest.raises(OSError, match=r"^tokenizer\.json of org/model is not in the local cache$"):
        load_token_counter("org/model")

    assert lookups == [("org/model", TOKENIZER_FILE_NAME, {"local_files_only": True})]
