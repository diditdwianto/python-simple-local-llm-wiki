"""Flask web UI for the local LLM wiki.

Run as: python -m src.app   (then open http://127.0.0.1:5566)

Features:
  - Browse notes in the vault (left sidebar)
  - Read a note rendered as HTML
  - Ask a question -> RAG answer grounded in the vault, with sources
  - Save a generated answer as a new note (re-indexed on the next save/watch)
"""
import glob
import json
import os

import markdown as md
from flask import (Flask, Response, redirect, render_template_string, request,
                   stream_with_context, url_for)

from .config import settings
from .ingest import ingest
from .note_writer import save_note
from .query import answer, answer_streamed

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
    .activity { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
                padding: 14px 18px; margin-top: 18px; display: none; }
    .activity h4 { margin: 0 0 8px; font-size: 12px; color: #6b7280;
                   text-transform: uppercase; letter-spacing: .05em; }
    .stage { display: flex; align-items: center; gap: 9px; padding: 4px 0;
             font-size: 14px; color: #6b7280; }
    .stage.done { color: #111827; }
    .stage .ms { margin-left: auto; color: #6b7280;
                 font-variant-numeric: tabular-nums; }
    .total { margin-top: 8px; font-size: 13px; color: #374151; font-weight: 600; }
    .check { color: #16a34a; font-weight: 700; width: 13px; text-align: center; }
    .spinner { display: inline-block; width: 13px; height: 13px; box-sizing: border-box;
               border: 2px solid #c7d2fe; border-top-color: #2563eb;
               border-radius: 50%; animation: spin .7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    button:disabled { opacity: .55; cursor: default; }
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
      <form id="ask-form">
        <textarea id="question" rows="3"
          placeholder="Ask your wiki anything…">{{ question or '' }}</textarea>
        <div style="margin-top:10px">
          <button id="ask-btn" type="submit">Ask</button>
        </div>
      </form>

      <div class="activity" id="activity">
        <h4>Activity</h4>
        <div id="stages"></div>
        <div class="total" id="total"></div>
      </div>

      <div id="answer"></div>

      {% if note_html %}
        <h2>{{ note_name }}</h2>
        <div class="note-body">{{ note_html | safe }}</div>
      {% endif %}
    </div>
  </div>
  <script>
    const form = document.getElementById('ask-form');
    const qEl = document.getElementById('question');
    const btn = document.getElementById('ask-btn');
    const activityEl = document.getElementById('activity');
    const stagesEl = document.getElementById('stages');
    const totalEl = document.getElementById('total');
    const answerEl = document.getElementById('answer');
    let timer = null, t0 = 0, currentQ = '';

    function fmt(ms) {
      return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';
    }

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const q = qEl.value.trim();
      if (q) ask(q);
    });

    function ask(q) {
      currentQ = q;
      btn.disabled = true;
      answerEl.innerHTML = '';
      stagesEl.innerHTML = '';
      totalEl.textContent = '';
      activityEl.style.display = 'block';
      t0 = performance.now();
      clearInterval(timer);
      timer = setInterval(function () {
        totalEl.textContent = 'Elapsed: ' + fmt(performance.now() - t0);
      }, 100);

      const rows = {};
      let done = false;
      const es = new EventSource('/ask_stream?question=' + encodeURIComponent(q));

      es.addEventListener('stage_start', function (e) {
        const d = JSON.parse(e.data);
        const row = document.createElement('div');
        row.className = 'stage';
        row.innerHTML = '<span class="spinner"></span>' +
          '<span class="label">' + d.name + '</span><span class="ms"></span>';
        stagesEl.appendChild(row);
        rows[d.name] = row;
      });

      es.addEventListener('stage_done', function (e) {
        const d = JSON.parse(e.data);
        const row = rows[d.name];
        if (!row) return;
        row.classList.add('done');
        const sp = row.querySelector('.spinner');
        if (sp) sp.outerHTML = '<span class="check">✓</span>';
        const extra = (d.hits !== undefined) ? ' · ' + d.hits + ' hits' : '';
        row.querySelector('.ms').textContent = fmt(d.ms) + extra;
      });

      es.addEventListener('result', function (e) {
        const d = JSON.parse(e.data);
        done = true;
        clearInterval(timer);
        totalEl.textContent = 'Total: ' + fmt(d.total_ms);
        es.close();
        btn.disabled = false;
        renderAnswer(d);
      });

      es.addEventListener('fail', function (e) {
        done = true;
        clearInterval(timer);
        es.close();
        btn.disabled = false;
        let msg = 'Request failed.';
        try { msg = JSON.parse(e.data).message; } catch (_) {}
        answerEl.innerHTML = '<div class="answer">⚠️ ' + msg + '</div>';
      });

      es.onerror = function () {
        if (done) return;
        done = true;
        clearInterval(timer);
        es.close();
        btn.disabled = false;
        answerEl.innerHTML = '<div class="answer">⚠️ Connection lost.</div>';
      };
    }

    function renderAnswer(d) {
      const wrap = document.createElement('div');
      wrap.className = 'answer';
      wrap.innerHTML = d.answer_html;
      if (d.sources && d.sources.length) {
        const s = document.createElement('div');
        s.className = 'sources';
        s.innerHTML = 'Sources: ' + d.sources.map(function (x) {
          return '<code>' + x + '</code>';
        }).join(' ');
        wrap.appendChild(s);
      }
      const f = document.createElement('form');
      f.action = '/save';
      f.method = 'post';
      f.style.marginTop = '14px';
      f.innerHTML = '<input type="hidden" name="question">' +
        '<input type="hidden" name="answer">' +
        '<input type="hidden" name="sources">' +
        '<button type="submit">💾 Save as note</button>';
      f.question.value = currentQ;
      f.answer.value = d.answer;
      f.sources.value = (d.sources || []).join(', ');
      wrap.appendChild(f);
      answerEl.appendChild(wrap);
    }
  </script>
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/ask_stream")
def ask_stream():
    """Stream the pipeline stage by stage as Server-Sent Events.

    The browser opens this with EventSource and renders each stage live with
    its elapsed time, so the user sees what is happening during the LLM wait.
    """
    question = (request.args.get("question") or "").strip()

    def gen():
        if not question:
            yield _sse("fail", {"message": "Empty question."})
            return
        try:
            for event, data in answer_streamed(question):
                if event == "result":
                    sources = sorted({c["source"] for c in data["contexts"]})
                    yield _sse("result", {
                        "answer": data["answer"],
                        "answer_html": md.markdown(
                            data["answer"], extensions=["fenced_code", "tables"]
                        ),
                        "sources": sources,
                        "total_ms": data["total_ms"],
                    })
                else:
                    yield _sse(event, data)
        except FileNotFoundError:
            yield _sse("fail", {"message": "No index found yet. Click "
                                "“Re-index vault” first."})
        except Exception as exc:  # surface the error in the activity panel
            yield _sse("fail", {"message": str(exc)})

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
