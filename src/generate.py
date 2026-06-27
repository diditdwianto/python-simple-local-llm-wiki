"""Pluggable LLM generation.

Switch backends with LLM_PROVIDER in your .env:
  - "ollama"  -> fully local, offline (default). Requires Ollama running.
  - "groq"    -> cloud free tier. Requires GROQ_API_KEY.

The retrieval + prompt assembly is identical for both; only the transport
to the model differs.
"""
import json
from typing import Dict, Iterator, List, Tuple

import requests

from .config import settings


def _ns_to_ms(value) -> float:
    """Ollama reports durations in nanoseconds; convert to ms (None -> 0)."""
    return (value or 0) / 1e6


def _sec_to_ms(value) -> float:
    """Groq reports durations in seconds; convert to ms (None -> 0)."""
    return (value or 0) * 1000

SYSTEM_PROMPT = (
    "You are a local wiki assistant. Answer the user's question using ONLY the "
    "context provided from their personal notes. If the context does not contain "
    "the answer, say so plainly instead of guessing. Cite the note titles you used "
    "in [square brackets]. Respond in clean, concise markdown."
)


def _build_user_prompt(question: str, contexts: List[Dict]) -> str:
    if contexts:
        blocks = "\n\n".join(
            f"[{c.get('source', 'note')}]\n{c.get('text', '')}" for c in contexts
        )
    else:
        blocks = "(no relevant notes found)"
    return f"Context from the wiki:\n\n{blocks}\n\nQuestion: {question}"


def generate(question: str, contexts: List[Dict]) -> str:
    """Generate an answer grounded in `contexts` using the configured provider."""
    user_prompt = _build_user_prompt(question, contexts)
    provider = settings.llm_provider

    if provider == "groq":
        return _generate_groq(user_prompt)
    if provider == "ollama":
        return _generate_ollama(user_prompt)
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'ollama' or 'groq'."
    )


def _generate_groq(user_prompt: str) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (required for LLM_PROVIDER=groq).")

    from groq import Groq  # imported lazily so Ollama users don't need the SDK

    client = Groq(api_key=settings.groq_api_key)
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _generate_ollama(user_prompt: str) -> str:
    try:
        resp = requests.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.ollama_host}. "
            "Is it running? Try `ollama serve` and `ollama pull "
            f"{settings.ollama_model}`."
        ) from exc
    return resp.json()["message"]["content"].strip()


# --- Streaming variants (token-by-token, with timing/usage stats) -----------

def generate_stream(question: str, contexts: List[Dict]) -> Iterator[Tuple[str, object]]:
    """Stream the answer as it is produced.

    Yields ("token", text) for each chunk, then a final ("stats", dict) with
    token counts and per-phase durations (when the backend reports them).
    """
    user_prompt = _build_user_prompt(question, contexts)
    provider = settings.llm_provider

    if provider == "groq":
        yield from _stream_groq(user_prompt)
    elif provider == "ollama":
        yield from _stream_ollama(user_prompt)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Use 'ollama' or 'groq'."
        )


def _stream_ollama(user_prompt: str) -> Iterator[Tuple[str, object]]:
    try:
        resp = requests.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
                "options": {"temperature": 0.2},
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.ollama_host}. "
            "Is it running? Try `ollama serve` and `ollama pull "
            f"{settings.ollama_model}`."
        ) from exc

    for raw in resp.iter_lines():
        if not raw:
            continue
        obj = json.loads(raw)
        content = (obj.get("message") or {}).get("content", "")
        if content:
            yield "token", content
        if obj.get("done"):
            yield "stats", {
                "load_ms": _ns_to_ms(obj.get("load_duration")),
                "prompt_tokens": obj.get("prompt_eval_count"),
                "prompt_ms": _ns_to_ms(obj.get("prompt_eval_duration")),
                "gen_tokens": obj.get("eval_count"),
                "gen_ms": _ns_to_ms(obj.get("eval_duration")),
            }


def _stream_groq(user_prompt: str) -> Iterator[Tuple[str, object]]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (required for LLM_PROVIDER=groq).")

    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    stream = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield "token", chunk.choices[0].delta.content
        if getattr(chunk, "usage", None):
            usage = chunk.usage

    if usage is not None:
        yield "stats", {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "prompt_ms": _sec_to_ms(getattr(usage, "prompt_time", None)),
            "gen_tokens": getattr(usage, "completion_tokens", None),
            "gen_ms": _sec_to_ms(getattr(usage, "completion_time", None)),
        }
