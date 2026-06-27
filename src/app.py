"""Flask web UI for the local LLM wiki.

Run as: python -m src.app   (then open http://127.0.0.1:5566)

Features:
  - Browse notes in the vault (left sidebar)
  - Read a note rendered as HTML
  - Ask a question -> RAG answer grounded in the vault, with sources
  - Save a generated answer as a new note (re-indexed on the next save/watch)
"""
import glob
import os

import markdown as md
from flask import Flask, redirect, render_template_string, request, url_for

from .config import settings
from .ingest import ingest
from .note_writer import save_note
from .query import answer

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Local LLM Wiki</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0;
           color: #1a1a1a; background: #fafafa; }
    header { background: #1f2937; color: #fff; padding: 14px 22px; }
    header b { font-size: 18px; }
    header span { color: #9ca3af; font-size: 13px; margin-left: 10px; }
    .layout { display: flex; min-height: calc(100vh - 52px); }
    .sidebar { width: 260px; border-right: 1px solid #e5e7eb; padding: 18px;
               background: #fff; }
    .sidebar a { display: block; padding: 6px 8px; color: #2563eb;
                 text-decoration: none; border-radius: 6px; font-size: 14px; }
    .sidebar a:hover { background: #eff6ff; }
    .main { flex: 1; padding: 26px 34px; max-width: 820px; }
    textarea { width: 100%; padding: 10px; border: 1px solid #d1d5db;
               border-radius: 8px; font-size: 15px; resize: vertical; }
    button { background: #2563eb; color: #fff; border: 0; padding: 9px 16px;
             border-radius: 8px; font-size: 14px; cursor: pointer; }
    button.secondary { background: #6b7280; }
    .answer { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
              padding: 18px 22px; margin-top: 18px; }
    .sources { color: #6b7280; font-size: 13px; margin-top: 14px; }
    .sources code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }
    .note-body { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
                 padding: 22px 26px; }
    pre { background: #f3f4f6; padding: 12px; border-radius: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <header>
    <b>📚 Local LLM Wiki</b>
    <span>provider: {{ provider }} &middot; {{ note_count }} notes indexed</span>
  </header>
  <div class="layout">
    <div class="sidebar">
      <form action="{{ url_for('reindex') }}" method="post" style="margin-bottom:14px">
        <button class="secondary" type="submit">↻ Re-index vault</button>
      </form>
      <strong style="font-size:13px;color:#6b7280">NOTES</strong>
      {% for n in notes %}
        <a href="{{ url_for('view_note', name=n) }}">{{ n }}</a>
      {% else %}
        <p style="color:#9ca3af;font-size:13px">Vault is empty.</p>
      {% endfor %}
    </div>
    <div class="main">
      <form action="{{ url_for('ask') }}" method="post">
        <textarea name="question" rows="3"
          placeholder="Ask your wiki anything…">{{ question or '' }}</textarea>
        <div style="margin-top:10px"><button type="submit">Ask</button></div>
      </form>

      {% if response %}
        <div class="answer">
          {{ response_html | safe }}
          {% if sources %}
          <div class="sources">
            Sources:
            {% for s in sources %}<code>{{ s }}</code> {% endfor %}
          </div>
          {% endif %}
          <form action="{{ url_for('save') }}" method="post" style="margin-top:14px">
            <input type="hidden" name="question" value="{{ question }}">
            <input type="hidden" name="answer" value="{{ response }}">
            <input type="hidden" name="sources" value="{{ sources_csv }}">
            <button type="submit">💾 Save as note</button>
          </form>
        </div>
      {% endif %}

      {% if note_html %}
        <h2>{{ note_name }}</h2>
        <div class="note-body">{{ note_html | safe }}</div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""


def _list_notes():
    pattern = os.path.join(settings.vault_dir, "**", "*.md")
    return [
        os.path.relpath(p, settings.vault_dir)
        for p in sorted(glob.glob(pattern, recursive=True))
    ]


def _render(**kwargs):
    notes = _list_notes()
    base = dict(
        provider=settings.llm_provider,
        notes=notes,
        note_count=len(notes),
        question=None,
        response=None,
        response_html=None,
        sources=None,
        sources_csv="",
        note_html=None,
        note_name=None,
    )
    base.update(kwargs)
    return render_template_string(PAGE, **base)


@app.route("/")
def index():
    return _render()


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.form.get("question") or "").strip()
    if not question:
        return redirect(url_for("index"))

    try:
        response, contexts = answer(question)
    except FileNotFoundError:
        response = "⚠️ No index found yet. Click **Re-index vault** first."
        contexts = []
    except Exception as exc:
        response = f"⚠️ {exc}"
        contexts = []

    sources = sorted({c["source"] for c in contexts}) if contexts else []
    return _render(
        question=question,
        response=response,
        response_html=md.markdown(response, extensions=["fenced_code", "tables"]),
        sources=sources,
        sources_csv=", ".join(sources),
    )


@app.route("/save", methods=["POST"])
def save():
    question = request.form.get("question", "").strip()
    answer_text = request.form.get("answer", "").strip()
    sources = [{"source": s.strip()} for s in
               request.form.get("sources", "").split(",") if s.strip()]
    if question and answer_text:
        save_note(question, answer_text, sources)
        ingest()  # rebuild so the new note is immediately searchable
    return redirect(url_for("index"))


@app.route("/note/<path:name>")
def view_note(name):
    path = os.path.join(settings.vault_dir, name)
    if not os.path.isfile(path) or not os.path.abspath(path).startswith(
        os.path.abspath(settings.vault_dir)
    ):
        return redirect(url_for("index"))
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return _render(
        note_name=name,
        note_html=md.markdown(content, extensions=["fenced_code", "tables"]),
    )


@app.route("/reindex", methods=["POST"])
def reindex():
    ingest()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host=settings.flask_host, port=settings.flask_port, debug=True)
