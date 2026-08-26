"""Local sentence-transformer embeddings. No API calls, no cost."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import settings


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: Sequence[str], *, batch_size: int = 256, show_progress: bool = False) -> list[list[float]]:
    """Embed a batch of texts into unit-normalised vectors."""
    if not texts:
        return []
    model = _model(settings().embedding_model)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [[float(x) for x in row] for row in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def dimension() -> int:
    return int(_model(settings().embedding_model).get_sentence_embedding_dimension())
