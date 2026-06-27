"""Ask the wiki a question from the command line.

Run as: python -m src.query "How do I configure the embedding model?"
"""
import sys
import time
from typing import Dict, Iterator, List, Tuple

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


def _llm_label() -> str:
    model = settings.groq_model if settings.llm_provider == "groq" else settings.ollama_model
    return f"Waiting for LLM ({settings.llm_provider}: {model})"


def answer_streamed(question: str, k: int = None) -> Iterator[Tuple[str, Dict]]:
    """Run the pipeline stage by stage, yielding (event, data) as each finishes.

    Events: "stage_start" / "stage_done" per stage, then a final "result".
    Each stage_done carries elapsed milliseconds so the UI can show timings.
    """
    k = k or settings.top_k
    overall = time.perf_counter()

    # 1) Load the FAISS index + metadata
    name = "Loading index"
    yield "stage_start", {"name": name}
    t = time.perf_counter()
    store = VectorStore().load()
    yield "stage_done", {"name": name, "ms": (time.perf_counter() - t) * 1000}

    # 2) Embed the query (also loads the embedding model on first call)
    name = "Embedding query"
    yield "stage_start", {"name": name}
    t = time.perf_counter()
    query_embedding = embed_query(question)
    yield "stage_done", {"name": name, "ms": (time.perf_counter() - t) * 1000}

    # 3) Vector search over the vault
    name = "Searching vault"
    yield "stage_start", {"name": name}
    t = time.perf_counter()
    contexts = store.search(query_embedding, k)
    yield "stage_done", {
        "name": name,
        "ms": (time.perf_counter() - t) * 1000,
        "hits": len(contexts),
    }

    # 4) Generate the grounded answer with the LLM (the slow part)
    name = _llm_label()
    yield "stage_start", {"name": name}
    t = time.perf_counter()
    response = generate(question, contexts)
    yield "stage_done", {"name": name, "ms": (time.perf_counter() - t) * 1000}

    yield "result", {
        "answer": response,
        "contexts": contexts,
        "total_ms": (time.perf_counter() - overall) * 1000,
    }


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
