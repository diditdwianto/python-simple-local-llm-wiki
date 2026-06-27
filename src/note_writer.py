"""Persist an LLM answer back into the vault as a new markdown note.

This is the "wiki grows itself" loop: a generated answer becomes a note,
which the watcher then re-indexes so it is searchable next time.
"""
import datetime
import os
import re
from typing import Dict, List

from .config import settings


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", text)[:60].strip("-")
    return slug or "note"


def save_note(question: str, answer: str, sources: List[Dict] = None) -> str:
    """Write a `# question` note with the answer + cited sources. Returns path."""
    os.makedirs(settings.vault_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{slugify(question)}.md"
    path = os.path.join(settings.vault_dir, filename)

    lines = [f"# {question}", "", answer.strip(), ""]
    if sources:
        unique = sorted({s.get("source", "note") for s in sources})
        lines += ["", "---", f"*Generated from notes: {', '.join(unique)}*"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
