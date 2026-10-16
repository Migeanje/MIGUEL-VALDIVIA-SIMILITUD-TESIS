"""Sentence-transformers adapter of the sentence-encoder port (T14).

:func:`load_sentence_encoder` loads an embedding model from the local Hugging Face cache only,
so nothing is downloaded. It finds the cached ``modules.json`` of the model, takes the
revision from the snapshot folder that holds it, and loads the model from that folder, so the
revision in the cache key is the one that encodes. A model outside the cache layout has no
revision, so it is refused. torch runs on the configured device with the configured CPU
threads, and :meth:`SentenceTransformerEncoder.encode` returns the model's output as it is: the
mean of its token vectors, not normalized (both configured models end with a pooling module).

torch, transformers and sentence-transformers are imported when a model is loaded, so
importing this module stays light.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from thematic_redundancy.representation.hf_tokenizers import snapshot_revision
from thematic_redundancy.representation.vectors import EncoderIdentity, EncoderSettings

MODULES_FILE_NAME = "modules.json"
"""File that every sentence-transformers model has, found to locate the cached snapshot."""


class SentenceTransformerEncoder:
    """:class:`~thematic_redundancy.representation.vectors.SentenceEncoder` over a
    sentence-transformers model."""

    def __init__(self, model: Any, identity: EncoderIdentity, settings: EncoderSettings) -> None:
        self._model = model
        self.identity = identity
        self.settings = settings

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def load_sentence_encoder(model_id: str, settings: EncoderSettings) -> SentenceTransformerEncoder:
    """Return an encoder of ``model_id``, loaded from the local cache only.

    Raises:
        OSError: if the model's ``modules.json`` is not in the local cache, or is outside the
            cache layout, so that its revision is unknown.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers.utils import cached_file

    torch.set_num_threads(settings.threads)
    missing = OSError(f"{MODULES_FILE_NAME} of {model_id} is not in the local cache")
    try:
        found = cached_file(model_id, MODULES_FILE_NAME, local_files_only=True)
    except OSError:
        # Some transformers releases raise instead of returning None.
        raise missing from None
    if found is None:
        raise missing
    folder = Path(found).parent
    revision = snapshot_revision(Path(found))
    if revision is None:
        raise OSError(
            f"{model_id} is not in the Hugging Face cache layout, so its revision, which keys "
            "the cached vectors, is unknown"
        )
    model = SentenceTransformer(str(folder), device=settings.device, local_files_only=True)
    identity = EncoderIdentity(
        model_id=model_id,
        revision=revision,
        max_seq_length=int(model.max_seq_length),
        dimension=int(model.get_embedding_dimension()),
    )
    return SentenceTransformerEncoder(model, identity, settings)
