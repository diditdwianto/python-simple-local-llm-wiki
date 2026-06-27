"""Recursive character text splitter.

Splits on progressively finer separators so chunks stay close to a target
size while respecting paragraph / sentence boundaries where possible. A small
overlap is carried between chunks to preserve context across boundaries.
"""
from typing import List

from .config import settings

# Tried in order: paragraphs, then lines, then sentences, then words, then chars.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if len(text) <= chunk_size or not separators:
        return [text]

    sep, rest = separators[0], separators[1:]
    parts = list(text) if sep == "" else text.split(sep)

    pieces: List[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_recursive_split(part, rest, chunk_size))
    return pieces


def split_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split `text` into overlapping chunks of roughly `chunk_size` characters."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    text = (text or "").strip()
    if not text:
        return []

    pieces = _recursive_split(text, SEPARATORS, chunk_size)

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(current) + len(piece) + 1 <= chunk_size:
            current = f"{current} {piece}".strip()
            continue

        if current:
            chunks.append(current)
        # Seed the next chunk with the tail of the previous one for overlap.
        if overlap and chunks:
            tail = chunks[-1][-overlap:]
            current = f"{tail} {piece}".strip()
        else:
            current = piece

    if current:
        chunks.append(current)
    return chunks
