"""Local sentence-transformer embeddings (no network calls at inference).

Uses BAAI/bge-small-en-v1.5 by default (384-dim). bge models are trained with
a query instruction prefix, so `embed_query` adds it while `embed_texts`
(for documents) does not.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loaded lazily and cached so the model is read from disk only once.
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed document chunks. Returns float32 array of shape (n, dim)."""
    vecs = _model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query. Returns float32 array of shape (1, dim)."""
    vec = _model().encode([QUERY_PREFIX + text], convert_to_numpy=True,
                          show_progress_bar=False)
    return np.asarray(vec, dtype="float32")
