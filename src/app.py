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
    .llmstats { margin-top: 6px; font-size: 12px; color: #6b7280;
                font-variant-numeric: tabular-nums; }
    .live { white-space: pre-wrap; }
    .live::after { content: "▍"; color: #2563eb; animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .disc { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
            margin-top: 12px; padding: 0 16px; }
    .disc > summary { cursor: pointer; padding: 12px 0; font-size: 14px;
                      font-weight: 600; color: #374151; list-style: none; }
    .disc > summary::-webkit-details-marker { display: none; }
    .disc > summary::before { content: "▸"; display: inline-block; margin-right: 8px;
                              color: #9ca3af; transition: transform .15s; }
    .disc[open] > summary::before { transform: rotate(90deg); }
    .disc-body { padding: 0 0 14px; }
    .hit { border-top: 1px solid #f3f4f6; padding: 10px 0; font-size: 13px; }
    .hit-head { display: flex; justify-content: space-between; color: #111827; }
    .hit-head .score { color: #6b7280; font-variant-numeric: tabular-nums; }
    .hit-text { color: #4b5563; margin-top: 4px; white-space: pre-wrap;
                max-height: 130px; overflow: auto; }
    .disc-body pre { white-space: pre-wrap; font-size: 12px; margin: 0;
                     max-height: 360px; overflow: auto; }
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
        <div class="llmstats" id="llmstats"></div>
        <div class="total" id="total"></div>
      </div>

      <div id="extras"></div>
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
    const llmstatsEl = document.getElementById('llmstats');
    const totalEl = document.getElementById('total');
    const extrasEl = document.getElementById('extras');
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
      extrasEl.innerHTML = '';
      llmstatsEl.textContent = '';
      totalEl.textContent = '';
      activityEl.style.display = 'block';
      t0 = performance.now();
      clearInterval(timer);
      timer = setInterval(function () {
        totalEl.textContent = 'Elapsed: ' + fmt(performance.now() - t0);
      }, 100);

      const rows = {};
      let done = false, activeRow = null, ttftMs = null;
      let liveEl = null, liveText = '', promptSummary = null;
      const es = new EventSource('/ask_stream?question=' + encodeURIComponent(q));

      es.addEventListener('stage_start', function (e) {
        const d = JSON.parse(e.data);
        const row = document.createElement('div');
        row.className = 'stage';
        row.innerHTML = '<span class="spinner"></span>' +
          '<span class="label">' + d.name + '</span><span class="ms"></span>';
        stagesEl.appendChild(row);
        rows[d.name] = row;
        activeRow = row;
      });

      es.addEventListener('retrieval', function (e) {
        const d = JSON.parse(e.data);
        const det = document.createElement('details');
        det.className = 'disc';
        const sum = document.createElement('summary');
        sum.textContent = '🔍 Search vault hits (' + d.hits.length + ')';
        det.appendChild(sum);
        const body = document.createElement('div');
        body.className = 'disc-body';
        d.hits.forEach(function (h) {
          const hit = document.createElement('div');
          hit.className = 'hit';
          const head = document.createElement('div');
          head.className = 'hit-head';
          const src = document.createElement('span');
          src.textContent = (h.source || '(untitled)') +
            (h.path ? '  ·  ' + h.path + ' #' + h.chunk : '');
          const sc = document.createElement('span');
          sc.className = 'score';
          sc.textContent = 'score ' + h.score;
          head.appendChild(src);
          head.appendChild(sc);
          const txt = document.createElement('div');
          txt.className = 'hit-text';
          txt.textContent = h.text;
          hit.appendChild(head);
          hit.appendChild(txt);
          body.appendChild(hit);
        });
        det.appendChild(body);
        extrasEl.appendChild(det);
      });

      es.addEventListener('llm_input', function (e) {
        const d = JSON.parse(e.data);
        const det = document.createElement('details');
        det.className = 'disc';
        const sum = document.createElement('summary');
        sum.textContent = '📤 Prompt sent to LLM (~' + d.approx_tokens +
          ' tokens, ' + d.chars + ' chars)';
        promptSummary = sum;
        det.appendChild(sum);
        const body = document.createElement('div');
        body.className = 'disc-body';
        const pre = document.createElement('pre');
        pre.textContent = 'SYSTEM:\\n' + d.system + '\\n\\nUSER:\\n' + d.user;
        body.appendChild(pre);
        det.appendChild(body);
        extrasEl.appendChild(det);
      });

      es.addEventListener('llm_first_token', function (e) {
        const d = JSON.parse(e.data);
        ttftMs = d.ms;
        if (activeRow) {
          activeRow.querySelector('.ms').textContent =
            'first token ' + fmt(d.ms) + ' · generating…';
        }
        liveText = '';
        liveEl = document.createElement('div');
        liveEl.className = 'answer live';
        answerEl.innerHTML = '';
        answerEl.appendChild(liveEl);
      });

      es.addEventListener('llm_token', function (e) {
        const d = JSON.parse(e.data);
        liveText += d.text;
        if (liveEl) liveEl.textContent = liveText;
      });

      es.addEventListener('stage_done', function (e) {
        const d = JSON.parse(e.data);
        const row = rows[d.name];
        if (row) {
          row.classList.add('done');
          const sp = row.querySelector('.spinner');
          if (sp) sp.outerHTML = '<span class="check">✓</span>';
          let extra = '';
          if (d.hits !== undefined) {
            extra = ' · ' + d.hits + ' hits';
          } else if (d.gen_tokens) {
            extra = ' · ' + d.gen_tokens + ' tok' +
              (d.tok_per_sec ? ' @ ' + d.tok_per_sec + ' tok/s' : '');
          }
          row.querySelector('.ms').textContent = fmt(d.ms) + extra;
        }
        if (d.gen_tokens || d.prompt_tokens) {
          const bits = [];
          if (ttftMs != null) bits.push('first token ' + fmt(ttftMs));
          if (d.prompt_tokens) bits.push('prompt ' + d.prompt_tokens + ' tok' +
            (d.prompt_ms ? ' in ' + fmt(d.prompt_ms) : ''));
          if (d.gen_tokens) bits.push('generated ' + d.gen_tokens + ' tok' +
            (d.gen_ms ? ' in ' + fmt(d.gen_ms) : '') +
            (d.tok_per_sec ? ' (' + d.tok_per_sec + ' tok/s)' : ''));
          if (d.load_ms) bits.push('model load ' + fmt(d.load_ms));
          llmstatsEl.textContent = '⚡ ' + bits.join(' · ');
        }
        if (promptSummary && d.prompt_tokens) {
          promptSummary.textContent =
            '📤 Prompt sent to LLM (' + d.prompt_tokens + ' tokens)';
        }
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
      answerEl.innerHTML = '';
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
