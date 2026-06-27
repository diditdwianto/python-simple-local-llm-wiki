"""Ask the wiki a question from the command line.

Run as: python -m src.query "How do I configure the embedding model?"
"""
import sys
from typing import Dict, List, Tuple

from .config import settings
from .embeddings import embed_query
from .generate import generate
from .store import VectorStore


def answer(question: str, k: int = None) -> Tuple[str, List[Dict]]:
    """Retrieve context for `question` and generate a grounded answer."""
    k = k or settings.top_k
    store = VectorStore().load()
    contexts = store.search(embed_query(question), k)
    response = generate(question, contexts)
    return response, contexts


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question"')
        return

    question = " ".join(sys.argv[1:])
    response, contexts = answer(question)

    print(response)
    if contexts:
        print("\nSources:")
        for c in contexts:
            print(f"  - {c['source']} (score {c['score']:.3f})")


if __name__ == "__main__":
    main()
