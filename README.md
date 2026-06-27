# python-simple-local-llm-wiki

> Local LLM Wiki. Confidential documents never leave your computer.

A small proof-of-concept "second brain": a markdown vault you can **ask
questions of** and that can **grow itself** by saving generated answers back as
notes. Retrieval is fully local (FAISS + local embeddings); the LLM is
pluggable between a fully-offline **Ollama** backend and the **Groq** free tier.

## How it works

```
vault/*.md ──▶ chunk ──▶ local embeddings (bge-small) ──▶ FAISS index
                                                              │
question ──▶ embed query ──▶ FAISS search ──▶ top-k chunks ───┤
                                                              ▼
                                          LLM (Ollama | Groq) ──▶ answer
                                                              │
                                          save answer ──▶ vault/*.md (re-indexed)
```

## Stack

| Concern        | Choice                                                |
| -------------- | ----------------------------------------------------- |
| Notes          | Plain markdown in `vault/` (Obsidian-compatible)      |
| Embeddings     | `BAAI/bge-small-en-v1.5` (local, 384-dim)             |
| Vector store   | FAISS (`IndexFlatIP`, cosine via normalized vectors)  |
| LLM            | Ollama (offline, default) **or** Groq (cloud)         |
| Auto re-index  | Watchdog watches the vault                            |
| Interfaces     | Flask web UI **and** CLI                              |

## Project layout

```
src/
  config.py        env-driven configuration
  chunking.py      recursive character text splitter
  embeddings.py    local sentence-transformer embeddings
  store.py         FAISS index + JSON metadata sidecar
  generate.py      pluggable LLM (ollama | groq)
  ingest.py        scan vault -> chunk -> embed -> build index
  query.py         CLI: ask a question
  note_writer.py   save an answer back to the vault
  watcher.py       auto-reindex on file changes
  app.py           Flask web UI
vault/             your markdown notes (source of truth)
```

## Setup

> **Python:** use 3.10–3.12 for the smoothest `faiss-cpu` wheel install.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # adjust if you want Groq instead of Ollama
```

### Pick an LLM backend

**Ollama (offline, default)**
```bash
# install from https://ollama.com, then:
ollama pull llama3
ollama serve          # usually already running
```

**Groq (cloud free tier)** — set in `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

## Usage

```bash
# 1) Build the index from the vault
python -m src.ingest

# 2a) Ask from the CLI
python -m src.query "How does the wiki grow itself?"

# 2b) …or launch the web UI
python -m src.app      # http://127.0.0.1:5555

# 3) (optional) auto-reindex while you edit notes
python -m src.watcher
```

In the web UI you can browse notes, ask questions, and click **Save as note** to
persist a good answer back into the vault (it is re-indexed immediately).

## Notes

- Prefix a filename with `exclude-` to skip it during ingestion without deleting.
- Tune retrieval/chunking via `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` in `.env`.
- The FAISS index lives in `.index/` (git-ignored) and is rebuilt by `ingest`.
