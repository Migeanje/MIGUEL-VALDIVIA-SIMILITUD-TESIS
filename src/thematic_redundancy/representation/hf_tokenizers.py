"""Hugging Face adapter of the token-counter port (D27, C5).

:func:`load_token_counter` loads the tokenizer of an embedding model from the local Hugging Face
cache only, so nothing is downloaded, and counts as the model reads: special tokens included,
without truncation. It loads the tokenizer from the folder of the cached ``tokenizer.json`` and
fingerprints that file (:class:`TokenizerIdentity`), so the chunk manifest names the very
tokenizer that counted. A cache without ``tokenizer.json``, such as a partial download, is
refused instead of loading a stand-in tokenizer.

``transformers`` is imported when a tokenizer is loaded, so importing this module stays light.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKENIZER_FILE_NAME = "tokenizer.json"


@dataclass(frozen=True)
class TokenizerIdentity:
    """Which tokenizer counted: the model, the cached revision and the tokenizer file."""

    model_id: str
    revision: str | None
    """Commit of the model in the Hugging Face cache, or ``None`` outside that cache."""
    tokenizer_sha256: str
    """SHA-256 of the ``tokenizer.json`` that was loaded."""
    vocabulary_size: int


class HuggingFaceTokenCounter:
    """:class:`~thematic_redundancy.representation.chunker.TokenCounter` over a Hugging Face
    tokenizer."""

    def __init__(self, tokenizer: Any, identity: TokenizerIdentity) -> None:
        self._tokenizer = tokenizer
        self.identity = identity

    def count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def load_token_counter(model_id: str) -> HuggingFaceTokenCounter:
    """Return a counter of the tokens that ``model_id`` reads, from the local cache only.

    Raises:
        OSError: if the model's ``tokenizer.json`` is not in the local cache.
    """
    from transformers import AutoTokenizer
    from transformers import logging as transformers_logging
    from transformers.utils import cached_file

    # Counting a text longer than the model's limit is the point here, so silence the warning.
    transformers_logging.set_verbosity_error()
    found = cached_file(model_id, TOKENIZER_FILE_NAME, local_files_only=True)
    if found is None:
        raise OSError(f"{TOKENIZER_FILE_NAME} of {model_id} is not in the local cache")
    path = Path(found)
    tokenizer = AutoTokenizer.from_pretrained(path.parent, local_files_only=True)
    identity = TokenizerIdentity(
        model_id=model_id,
        revision=snapshot_revision(path),
        tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        vocabulary_size=tokenizer.vocab_size,
    )
    return HuggingFaceTokenCounter(tokenizer, identity)


def snapshot_revision(path: Path) -> str | None:
    """Return the commit of a file in the Hugging Face cache, which keeps each revision of a
    model in ``snapshots/<commit>/``; ``None`` for a file outside that layout."""
    return path.parent.name if path.parent.parent.name == "snapshots" else None
