"""Scan the markdown vault, chunk + embed every note, and build the FAISS index.

Run as: python -m src.ingest

Files whose name begins with `exclude-` are skipped (handy for drafts you
don't want retrieved, without deleting them).
"""
import glob
import os
import re
from typing import Dict, Iterator, List, Tuple

from .chunking import split_text
from .config import settings
from .embeddings import embed_texts
from .store import VectorStore


def _iter_markdown(vault_dir: str) -> Iterator[str]:
    pattern = os.path.join(vault_dir, "**", "*.md")
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.basename(path).startswith("exclude-"):
            continue
        yield path


def _title_from(content: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return os.path.splitext(os.path.basename(fallback))[0]


def _collect() -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    for path in _iter_markdown(settings.vault_dir):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        rel = os.path.relpath(path, settings.vault_dir)
        title = _title_from(content, rel)
        for i, chunk in enumerate(split_text(content)):
            texts.append(chunk)
            metas.append({"source": title, "path": rel, "chunk": i, "text": chunk})
    return texts, metas


def ingest() -> int:
    texts, metas = _collect()
    if not texts:
        print(f"No notes found in vault: {settings.vault_dir}")
        return 0

    embeddings = embed_texts(texts)
    VectorStore().build(embeddings, metas)

    note_count = len({m["path"] for m in metas})
    print(f"Indexed {len(texts)} chunks from {note_count} note(s).")
    return len(texts)


if __name__ == "__main__":
    ingest()
