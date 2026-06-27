"""Central configuration, driven by environment variables (.env supported).

Everything the wiki needs is resolved here so the rest of the code never
touches os.environ directly. Import `settings` and read attributes.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Settings:
    # --- Embeddings (runs locally, no API needed) ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- LLM backend: "ollama" (offline) or "groq" (cloud free tier) ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    # Groq (cloud) settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Ollama (local) settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")

    # --- Storage ---
    vault_dir: str = os.getenv("VAULT_DIR", os.path.join(_project_root(), "vault"))
    index_dir: str = os.getenv("INDEX_DIR", os.path.join(_project_root(), ".index"))

    # --- Retrieval / chunking (character-based for a simple PoC) ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    # --- Web UI ---
    flask_host: str = os.getenv("FLASK_HOST", "127.0.0.1")
    flask_port: int = int(os.getenv("FLASK_PORT", "5566"))


settings = Settings()
