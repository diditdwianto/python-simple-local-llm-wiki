"""Watch the vault and re-index automatically when markdown files change.

Run as: python -m src.watcher

Keeps the FAISS index in sync as you edit notes in Obsidian (or any editor)
without having to re-run ingest by hand. Events are debounced so a burst of
saves triggers a single rebuild.
"""
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import settings
from .ingest import ingest

DEBOUNCE_SECONDS = 1.5


class VaultHandler(FileSystemEventHandler):
    def __init__(self):
        self._last_run = 0.0

    def on_any_event(self, event):
        if event.is_directory or not str(event.src_path).endswith(".md"):
            return
        now = time.time()
        if now - self._last_run < DEBOUNCE_SECONDS:
            return
        self._last_run = now
        print(f"[watcher] change detected ({event.event_type}): re-indexing…")
        try:
            ingest()
        except Exception as exc:  # keep the watcher alive on transient errors
            print(f"[watcher] reindex failed: {exc}")


def main() -> None:
    handler = VaultHandler()
    observer = Observer()
    observer.schedule(handler, settings.vault_dir, recursive=True)
    observer.start()
    print(f"[watcher] watching {settings.vault_dir} (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
