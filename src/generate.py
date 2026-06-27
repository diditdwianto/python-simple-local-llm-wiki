"""Pluggable LLM generation.

Switch backends with LLM_PROVIDER in your .env:
  - "ollama"  -> fully local, offline (default). Requires Ollama running.
  - "groq"    -> cloud free tier. Requires GROQ_API_KEY.

The retrieval + prompt assembly is identical for both; only the transport
to the model differs.
"""
from typing import Dict, List

import requests

from .config import settings

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
