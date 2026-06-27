"""FAISS-backed vector store with a JSON metadata sidecar.

FAISS stores only the vectors, so we keep a parallel list of metadata dicts
(text + source) whose positions line up with the FAISS row ids. Cosine
similarity is achieved with an inner-product index over L2-normalized vectors.
"""
import json
import os
from typing import Dict, List

import faiss
import numpy as np

from .config import settings


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata: List[Dict] = []

    @property
    def index_path(self) -> str:
        return os.path.join(settings.index_dir, "wiki.faiss")

    @property
    def meta_path(self) -> str:
        return os.path.join(settings.index_dir, "wiki.meta.json")

    def build(self, embeddings: np.ndarray, metadatas: List[Dict]) -> "VectorStore":
        embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index
        self.metadata = metadatas
        self.save()
        return self

    def save(self) -> None:
        os.makedirs(settings.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self) -> "VectorStore":
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                "No index found. Run `python -m src.ingest` first."
            )
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        return self

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        query_embedding = np.asarray(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(query_embedding, k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
