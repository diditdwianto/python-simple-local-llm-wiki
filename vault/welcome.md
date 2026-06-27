# Welcome to your Local LLM Wiki

This vault holds your notes as plain markdown files. Everything stays on your
machine — embeddings and (optionally) the LLM run locally, so confidential
documents never leave your computer.

## How it works

1. **Notes** live here in the `vault/` folder as `.md` files. Edit them with any
   editor, including Obsidian.
2. **Ingest** chunks every note, embeds it locally with a sentence-transformer,
   and stores the vectors in a FAISS index.
3. **Ask** a question: the wiki retrieves the most relevant chunks and feeds them
   to a local (Ollama) or cloud (Groq) LLM to write a grounded answer.
4. **Save** good answers back as new notes — the wiki grows itself.

## Tips

- Prefix a filename with `exclude-` to keep it out of search without deleting it.
- Run the watcher (`python -m src.watcher`) to auto-reindex as you edit.
- Switch the LLM backend any time with `LLM_PROVIDER` in your `.env`.
