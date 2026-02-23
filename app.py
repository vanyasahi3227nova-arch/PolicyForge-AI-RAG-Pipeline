"""
PolicyForge AI — Gradio Application
Entry point for Hugging Face Spaces.

HF Spaces runs this file automatically.
Set HF_TOKEN as a Space secret for the Inference API.
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Generator

import gradio as gr

from rag_pipeline import PolicyForgePipeline

# ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# Initialise pipeline & pre-load framework data
# ──────────────────────────────────────────────────
logger.info("Initialising PolicyForge AI pipeline…")
pipe = PolicyForgePipeline()

DATA_DIR = Path(__file__).parent / "data"
for fname, label in [("nist_800_53.txt", "NIST_800_53"), ("iso_27001.txt", "ISO_27001_2022")]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        n = pipe.ingest_file(str(fpath))
        logger.info(f"Pre-loaded {label}: {n} chunks")
    else:
        logger.warning(f"Data file not found: {fpath}")

logger.info(f"Pipeline ready. Status: {pipe.status()}")

# ──────────────────────────────────────────────────
# Helper: format sources for display
# ──────────────────────────────────────────────────
def fmt_sources(sources: list) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for i, s in enumerate(sources, 1):
        bar_filled = "█" * int(s["confidence"] / 10)
        bar_empty  = "░" * (10 - int(s["confidence"] / 10))
        lines.append(
            f"**{i}. [{s['framework']}] {s['label']}**"
            + (f" · pg. {s['page']}" if s.get("page") is not None else "")
            + f"\n`Confidence: {s['confidence']}%` {bar_filled}{bar_empty}\n"
            + f"> {s['snippet'][:180]}…\n"
        )
    return "\n---\n".join(lines)

# ──────────────────────────────────────────────────
# Core query function (streaming)
# ──────────────────────────────────────────────────
def run_query(
    question: str,
    framework: str,
    history: list,
) -> Generator:
    """Yields (history, sources_md, status_md) tuples for streaming."""

    if not question.strip():
        yield history, "", "⚠ Please enter a question."
        return

    status = pipe.status()
    if not status["ready"]:
        yield history, "", "⚠ Knowledge base not loaded. Please wait and retry."
        return

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "⟳ Retrieving relevant controls…"})
    yield history, "", "🔍 Running hybrid retrieval (BM25 + ChromaDB)…"

    try:
        fw_map = {
            "Auto-detect": "auto",
            "NIST 800-53": "nist",
            "ISO 27001:2022": "iso",
            "Both Frameworks": "both",
        }
        result = pipe.query(question, framework=fw_map.get(framework, "auto"))
    except Exception as e:
        err = f"⚠ Error: {e}"
        history[-1]["content"] = err
        yield history, "", f"❌ Failed: {e}"
        return

    answer   = result["answer"]
    sources  = result["sources"]
    ret_ms   = result["retrieval_ms"]
    n_chunks = result["chunks_retrieved"]

    history[-1]["content"] = answer
    sources_md = fmt_sources(sources)
    status_md  = (
        f"✅ Done · **{n_chunks} chunks** retrieved · "
        f"**{ret_ms}ms** retrieval · "
        f"Framework: **{result['framework'].upper()}**"
    )
    yield history, sources_md, status_md


def upload_pdf(file, history):
    """Handle user PDF uploads."""
    if file is None:
        return history, "No file uploaded."
    try:
        n = pipe.ingest_file(file.name)
        msg = f"✅ Indexed **{Path(file.name).stem}** — {n} chunks added to knowledge base."
    except Exception as e:
        msg = f"⚠ Upload failed: {e}"
    history = history or []
    history.append({"role": "assistant", "content": msg})
    return history, msg


def clear_chat():
    return [], "", "Ready."

# ──────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────
CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --navy:   #070F1E;
    --navy2:  #0D1B31;
    --navy3:  #112240;
    --blue:   #1D6AE5;
    --teal:   #05C3D4;
    --teal2:  #0891B2;
    --violet: #7C3AED;
    --t1: #F0F4FF;
    --t2: #94A3C4;
    --t3: #4B5878;
    --glass: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --font-d: 'Syne', sans-serif;
    --font-b: 'DM Sans', sans-serif;
    --font-m: 'IBM Plex Mono', monospace;
}

/* ── Global ── */
body, .gradio-container {
    background: var(--navy) !important;
    font-family: var(--font-b) !important;
    color: var(--t1) !important;
}

/* Grid texture */
.gradio-container::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(5,195,212,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(5,195,212,.025) 1px, transparent 1px);
    background-size: 44px 44px;
}

/* ── App title ── */
#app-title {
    font-family: var(--font-d) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(92deg, #F0F4FF 30%, #05C3D4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 4px !important;
}
#app-subtitle {
    color: var(--t2) !important;
    font-size: 0.875rem !important;
    margin-bottom: 0 !important;
}

/* ── Panels ── */
.panel-box {
    background: rgba(13,27,49,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    backdrop-filter: blur(20px) !important;
}

/* ── Chatbot ── */
.chatbot-wrap { border-radius: 12px !important; overflow: hidden !important; }

.message.user {
    background: linear-gradient(135deg, #1D6AE5, #1558C0) !important;
    border-radius: 13px 4px 13px 13px !important;
    color: #fff !important;
    font-size: 13.5px !important;
    line-height: 1.65 !important;
    box-shadow: 0 4px 18px rgba(29,106,229,.25) !important;
    border: none !important;
}
.message.bot {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px 13px 13px 13px !important;
    color: var(--t1) !important;
    font-size: 13.5px !important;
    line-height: 1.7 !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Text inputs ── */
textarea, input[type="text"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--t1) !important;
    font-family: var(--font-b) !important;
    font-size: 13.5px !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(29,106,229,0.5) !important;
    box-shadow: 0 0 0 3px rgba(29,106,229,0.1) !important;
    outline: none !important;
}
textarea::placeholder { color: rgba(148,163,196,0.35) !important; }

/* ── Buttons ── */
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, var(--blue), var(--teal2)) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(29,106,229,0.3) !important;
    font-family: var(--font-b) !important;
}
button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(29,106,229,.45) !important; }

button.secondary, button[variant="secondary"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    color: var(--t2) !important;
    border-radius: 10px !important;
    font-family: var(--font-b) !important;
    transition: all 0.2s !important;
}
button.secondary:hover { background: rgba(255,255,255,.08) !important; color: var(--t1) !important; }

/* ── Dropdown ── */
select, .dropdown {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    color: var(--t2) !important;
    border-radius: 8px !important;
    font-family: var(--font-b) !important;
}

/* ── Labels ── */
label span, .label-wrap span {
    color: var(--t2) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Accordion ── */
.accordion {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.accordion-header {
    color: var(--teal) !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Status bar ── */
#status-bar {
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    color: var(--teal) !important;
    padding: 6px 12px !important;
    background: rgba(5,195,212,0.07) !important;
    border: 1px solid rgba(5,195,212,0.18) !important;
    border-radius: 8px !important;
}

/* ── Sources panel ── */
#sources-panel {
    font-size: 12px !important;
    line-height: 1.65 !important;
    color: var(--t2) !important;
}
#sources-panel strong { color: var(--t1) !important; }
#sources-panel code {
    background: rgba(5,195,212,.12) !important;
    color: var(--teal) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-family: var(--font-m) !important;
    font-size: 10px !important;
}
#sources-panel blockquote {
    border-left: 2px solid rgba(5,195,212,.3) !important;
    color: var(--t3) !important;
    padding-left: 10px !important;
    margin: 4px 0 !important;
    font-size: 11px !important;
}

/* ── Framework chips ── */
.fw-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 8px 0 4px;
}
.fc {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: var(--font-m);
}
.fc-n { background: rgba(29,106,229,.15); border: 1px solid rgba(29,106,229,.3); color: #93C5FD; }
.fc-i { background: rgba(124,58,237,.15); border: 1px solid rgba(124,58,237,.3); color: #C4B5FD; }
.fc-k { background: rgba(16,185,129,.1);  border: 1px solid rgba(16,185,129,.25); color: #6EE7B7; }

/* ── Upload area ── */
.upload-area {
    background: var(--glass) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    color: var(--t3) !important;
    transition: border-color 0.2s !important;
}
.upload-area:hover { border-color: rgba(29,106,229,.4) !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.1); border-radius: 2px; }

/* ── Logo pulse ── */
@keyframes lp { 0%,100%{box-shadow:0 0 20px rgba(5,195,212,.3)} 50%{box-shadow:0 0 35px rgba(5,195,212,.55)} }
#logo-shield { animation: lp 3s ease-in-out infinite !important; }
"""

# ──────────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Generate 10 device management policies for a finance company aligned with NIST 800-53",
    "What ISO 27001 controls apply to access management for a healthcare organization?",
    "Create an incident response policy for a SaaS startup covering detection, containment, and recovery",
    "Generate a cryptography and encryption policy referencing both NIST and ISO 27001 controls",
    "What are the key personnel security controls under NIST 800-53 AT and PS families?",
    "Create a remote work security policy aligned with ISO 27001 A.6.7 and NIST AC-17",
]

with gr.Blocks(
    css=CSS,
    title="PolicyForge AI — Cybersecurity Knowledge Engine",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["DM Sans", "sans-serif"],
    ),
) as demo:

    # ── Header ──────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div style="display:flex;align-items:center;gap:14px;padding:8px 0 4px">
              <div id="logo-shield" style="width:48px;height:48px;border-radius:13px;
                background:linear-gradient(140deg,#1D6AE5,#0891B2);
                display:flex;align-items:center;justify-content:center;flex-shrink:0">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.25C17.25 22.15 21 17.25 21 12V7L12 2z"
                    fill="rgba(255,255,255,.92)"/>
                  <circle cx="9" cy="12" r="1.4" fill="#0D1B31"/>
                  <circle cx="15" cy="12" r="1.4" fill="#0D1B31"/>
                  <path d="M9.5 15s1 1.5 2.5 1.5 2.5-1.5 2.5-1.5"
                    stroke="#0D1B31" stroke-width="1.3" stroke-linecap="round"/>
                </svg>
              </div>
              <div>
                <div id="app-title">PolicyForge AI</div>
                <div id="app-subtitle">
                  RAG-powered cybersecurity policy generation &nbsp;·&nbsp;
                  <span class="fc fc-n">NIST 800-53</span>&nbsp;
                  <span class="fc fc-i">ISO 27001:2022</span>&nbsp;
                  <span class="fc fc-k">● Live</span>
                </div>
              </div>
            </div>
            """)

    gr.HTML("<hr style='border-color:rgba(255,255,255,.07);margin:4px 0 12px'/>")

    # ── Main layout ─────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT — chat column
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                value=[],
                label="Policy Generation Chat",
                height=480,
                type="messages",
                avatar_images=(None, None),
                show_label=True,
                bubble_full_width=False,
                elem_classes=["chatbot-wrap"],
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Describe your organization's cybersecurity policy needs… (Shift+Enter for new line)",
                    lines=2,
                    max_lines=5,
                    label="Your query",
                    scale=5,
                    show_label=False,
                )
                with gr.Column(scale=1, min_width=110):
                    submit_btn = gr.Button("Generate →", variant="primary", size="lg")
                    clear_btn  = gr.Button("Clear", variant="secondary", size="sm")

            with gr.Row():
                framework_dd = gr.Dropdown(
                    choices=["Auto-detect", "NIST 800-53", "ISO 27001:2022", "Both Frameworks"],
                    value="Auto-detect",
                    label="Framework",
                    scale=2,
                )
                status_box = gr.Markdown(
                    value="✅ Knowledge base loaded — ready to generate policies.",
                    elem_id="status-bar",
                )

            # Example queries
            with gr.Accordion("💡 Example queries — click to use", open=False):
                for eq in EXAMPLE_QUERIES:
                    gr.Button(eq, variant="secondary", size="sm").click(
                        fn=lambda x=eq: x, outputs=question_box
                    )

        # RIGHT — knowledge panel
        with gr.Column(scale=2):

            gr.HTML("""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 1l1.7 3.8H14l-3.4 2.6 1.3 4L8 9.5l-3.9 1.9 1.3-4L2 5.8h4.3z"
                  stroke="#05C3D4" stroke-width="1.2" stroke-linejoin="round"/>
              </svg>
              <span style="font-family:'Syne',sans-serif;font-size:13px;font-weight:800;color:#F0F4FF;letter-spacing:.01em">
                Framework Intelligence
              </span>
              <span style="font-size:9px;color:#05C3D4;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                margin-left:auto;background:rgba(5,195,212,.1);border:1px solid rgba(5,195,212,.2);
                padding:2px 8px;border-radius:20px">● BM25 + Vector Active</span>
            </div>
            """)

            sources_panel = gr.Markdown(
                value="_Retrieved framework controls will appear here after your first query._",
                label="Retrieved Knowledge Sources",
                elem_id="sources-panel",
            )

            gr.HTML("<hr style='border-color:rgba(255,255,255,.07);margin:12px 0'/>")

            gr.HTML("""
            <div style="font-family:'Syne',sans-serif;font-size:12px;font-weight:800;
              color:#F0F4FF;letter-spacing:.01em;margin-bottom:8px">
              📄 Upload Custom Document
            </div>
            <div style="font-size:11px;color:#4B5878;margin-bottom:8px">
              Add your own PDF (policy docs, standards, audit reports) to extend the knowledge base.
            </div>
            """)

            pdf_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                elem_classes=["upload-area"],
            )
            upload_btn = gr.Button("Index Document", variant="secondary", size="sm")

    # ── Pipeline status accordion ────────────────────
    with gr.Accordion("🔬 Pipeline Status & Architecture", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
**Architecture:**
```
Query → BM25 Retriever (lexical)  ┐
                                   ├→ RRF Fusion → Top-k Context → Mistral-7B (HF API) → Policy
Query → ChromaDB (semantic)       ┘
```
**Models:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, CPU)
- LLM: `mistralai/Mistral-7B-Instruct-v0.3` (HF Inference API, serverless)
- Vector Store: ChromaDB (in-memory)

**RRF formula:** `score = Σ 1 / (0.4 + rank + 1)`
                """)
            with gr.Column():
                def get_status_md():
                    s = pipe.status()
                    return f"""
**Status:** {"✅ Ready" if s["ready"] else "⚠ Not ready"}
**Chunks indexed:** `{s["total_chunks"]}`
**Sources loaded:**
{chr(10).join(f"- `{src}`" for src in s["loaded_sources"]) or "_none_"}
**Embed model:** `{s["embed_model"]}`
**LLM:** `{s["llm_model"]}`
                    """
                pipeline_status = gr.Markdown(get_status_md())
                gr.Button("Refresh Status", variant="secondary", size="sm").click(
                    fn=get_status_md, outputs=pipeline_status
                )

    # ── Event wiring ─────────────────────────────────

    def submit(q, fw, hist):
        for h, s, st in run_query(q, fw, hist):
            yield h, s, st, ""   # clear input after submit

    submit_btn.click(
        fn=submit,
        inputs=[question_box, framework_dd, chatbot],
        outputs=[chatbot, sources_panel, status_box, question_box],
    )

    question_box.submit(
        fn=submit,
        inputs=[question_box, framework_dd, chatbot],
        outputs=[chatbot, sources_panel, status_box, question_box],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, sources_panel, status_box],
    )

    upload_btn.click(
        fn=upload_pdf,
        inputs=[pdf_upload, chatbot],
        outputs=[chatbot, status_box],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
