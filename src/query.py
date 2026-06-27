"""Ask the wiki a question from the command line.

Run as: python -m src.query "How do I configure the embedding model?"
"""
import sys
import time
from typing import Dict, Iterator, List, Tuple

from .config import settings
from .embeddings import embed_query
from .generate import generate, generate_stream
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

    # 4) Generate the grounded answer with the LLM, streaming token by token.
    name = _llm_label()
    yield "stage_start", {"name": name}
    t = time.perf_counter()
    first_token_at = None
    pieces: List[str] = []
    stats: Dict = {}

    for kind, payload in generate_stream(question, contexts):
        if kind == "token":
            if first_token_at is None:
                first_token_at = time.perf_counter()
                yield "llm_first_token", {"ms": (first_token_at - t) * 1000}
            pieces.append(payload)
            yield "llm_token", {"text": payload}
        elif kind == "stats":
            stats = payload

    response = "".join(pieces)
    elapsed_ms = (time.perf_counter() - t) * 1000

    done = {"name": name, "ms": elapsed_ms}
    if first_token_at is not None:
        done["ttft_ms"] = (first_token_at - t) * 1000
    if stats.get("prompt_tokens") is not None:
        done["prompt_tokens"] = stats["prompt_tokens"]
        done["prompt_ms"] = stats.get("prompt_ms")
    if stats.get("gen_tokens") is not None:
        done["gen_tokens"] = stats["gen_tokens"]
        done["gen_ms"] = stats.get("gen_ms")
        gen_ms = stats.get("gen_ms") or 0
        if gen_ms > 0:
            done["tok_per_sec"] = round(stats["gen_tokens"] / (gen_ms / 1000), 1)
    if stats.get("load_ms"):
        done["load_ms"] = stats["load_ms"]
    yield "stage_done", done

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
