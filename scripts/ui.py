"""Comprehensive Gradio web interface for ResearchGraph AI.

Tabs:
1. Ingest Papers (ArXiv Search OR Local PDF Upload -> PDF in data/raw/ -> Chunks in data/processed/)
2. Extract Entities (LLM -> Neo4j Graph + Qdrant Vectors)
3. GraphRAG Q&A (Hybrid Retrieval + Grounded Answer Generation)
"""
from __future__ import annotations

import datetime
import json
import logging
import shutil
import sys
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gradio as gr

from researchgraph.config import get_settings
from researchgraph.extraction import run_full_extraction
from researchgraph.generation import AnswerGenerator
from researchgraph.ingestion import PaperMeta, extract_chunks, load_all_metadata, run_ingestion
from researchgraph.logging_setup import get_logger, setup_logging
from researchgraph.retrieval import GraphRetriever, HybridRetriever
from researchgraph.vectorstore import VectorStore

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Logging Capture
# ---------------------------------------------------------------------------

class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.buffer = StringIO()
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.write(self.format(record) + "\n")

    def getvalue(self) -> str:
        return self.buffer.getvalue()


# ---------------------------------------------------------------------------
# Tab 1: Ingestion Handlers
# ---------------------------------------------------------------------------

def run_ingestion_ui(query: str, max_papers: int, chunk_size: int):
    setup_logging(level="INFO")
    capture = _LogCapture()
    logging.getLogger().addHandler(capture)

    try:
        chunks = run_ingestion(
            query=query,
            max_papers=int(max_papers),
            chunk_size=int(chunk_size),
        )
    except Exception as exc:
        logging.getLogger().removeHandler(capture)
        return capture.getvalue() + f"\n\nERROR: {exc}", [], f"❌ Ingestion failed: {exc}"

    logging.getLogger().removeHandler(capture)
    metas = load_all_metadata()
    table = [
        [m.arxiv_id, m.title[:65] + ("…" if len(m.title) > 65 else ""), m.published, ", ".join(m.authors[:2])]
        for m in metas
    ]
    summary = f"✅ Ingested {len(chunks)} chunks from {len(metas)} total papers"
    return capture.getvalue(), table, summary


def run_pdf_upload_ui(uploaded_files: Any, chunk_size: int = 1000):
    """Save uploaded PDFs to data/raw/, chunk them into data/processed/, and update metadata."""
    if not uploaded_files:
        return "No files selected.", [], "⚠️ Please upload at least one PDF file."

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    setup_logging(level="INFO")
    capture = _LogCapture()
    logging.getLogger().addHandler(capture)

    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Processing %d uploaded PDF file(s) ===", len(uploaded_files))
    saved_count = 0
    total_chunks = 0

    for file_item in uploaded_files:
        file_path_str = getattr(file_item, "name", str(file_item))
        src_path = Path(file_path_str)

        orig_filename = src_path.name
        stem = src_path.stem.replace(" ", "_")
        dest_pdf = settings.raw_dir / orig_filename

        try:
            # 1. Save uploaded PDF to data/raw/
            shutil.copy2(src_path, dest_pdf)
            log.info("Saved local PDF → %s", dest_pdf)

            # 2. Extract text chunks
            chunks = extract_chunks(
                pdf_path=dest_pdf,
                arxiv_id=stem,
                chunk_size=int(chunk_size),
            )

            # 3. Save JSON text chunks into data/processed/
            chunks_path = settings.processed_dir / f"{stem}_chunks.json"
            chunks_path.write_text(
                json.dumps([asdict(c) for c in chunks], indent=2),
                encoding="utf-8",
            )
            log.info("Saved %d chunks → %s", len(chunks), chunks_path.name)
            total_chunks += len(chunks)

            # 4. Save metadata JSON so paper appears in dashboard
            preview_abstract = chunks[0].text[:400] if chunks else "No text extracted"
            meta = PaperMeta(
                arxiv_id=stem,
                title=src_path.stem.replace("_", " ").title(),
                authors=["Local Upload"],
                abstract=preview_abstract,
                published=datetime.date.today().isoformat(),
                pdf_url="",
                primary_category="Local",
                local_pdf=dest_pdf.name,
            )
            meta_path = settings.metadata_dir / f"{stem}.json"
            meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
            saved_count += 1

        except Exception as exc:
            log.error("Failed to process %s: %s", orig_filename, exc)

    logging.getLogger().removeHandler(capture)

    # Refresh cached papers
    metas = load_all_metadata()
    table = [
        [m.arxiv_id, m.title[:65] + ("…" if len(m.title) > 65 else ""), m.published, ", ".join(m.authors[:2])]
        for m in metas
    ]
    summary = f"✅ Processed {saved_count} uploaded PDF(s) ({total_chunks} text chunks generated in data/processed/)"
    return capture.getvalue(), table, summary


def refresh_papers_ui():
    metas = load_all_metadata()
    table = [
        [m.arxiv_id, m.title[:65] + ("…" if len(m.title) > 65 else ""), m.published, ", ".join(m.authors[:2])]
        for m in metas
    ]
    return table, f"📚 {len(metas)} papers cached"


# ---------------------------------------------------------------------------
# Tab 2: Extraction Handlers
# ---------------------------------------------------------------------------

def run_extraction_ui(max_papers: int):
    setup_logging(level="INFO")
    capture = _LogCapture()
    logging.getLogger().addHandler(capture)

    try:
        summary_dict = run_full_extraction(max_papers=int(max_papers) if max_papers > 0 else None)
        total_exps = sum(summary_dict.values())
        msg = f"✅ Extracted {total_exps} experiments across {len(summary_dict)} papers"
    except Exception as exc:
        logging.getLogger().removeHandler(capture)
        return capture.getvalue() + f"\n\nERROR: {exc}", f"❌ Extraction failed: {exc}"

    logging.getLogger().removeHandler(capture)
    return capture.getvalue(), msg


# ---------------------------------------------------------------------------
# Tab 3: GraphRAG Q&A Handlers
# ---------------------------------------------------------------------------

def run_qa_ui(question: str):
    if not question.strip():
        return "Please enter a question.", [], []

    try:
        settings = get_settings()

        graph_retriever = None
        try:
            graph_retriever = GraphRetriever(settings=settings)
        except Exception:
            pass

        vector_store = None
        try:
            vector_store = VectorStore(settings=settings)
        except Exception:
            pass

        hybrid = HybridRetriever(
            settings=settings,
            graph_retriever=graph_retriever,
            vector_store=vector_store,
        )
        generator = AnswerGenerator(retriever=hybrid)
        ans = generator.generate(question)

        # Graph facts table
        facts_table = []
        for f in ans.context.graph_facts:
            metrics_str = ", ".join(
                f"{m.get('metric')}={m.get('value')}"
                for m in f.get("results", [])
                if m.get("metric")
            )
            facts_table.append([
                f.get("arxiv_id", ""),
                f.get("paper_title", "")[:50],
                f.get("algorithm", ""),
                f.get("dataset", ""),
                f.get("xai_method", "") or "None",
                metrics_str,
            ])

        # Vector chunks table
        chunks_table = [
            [c.get("arxiv_id", ""), str(c.get("page_start", 0)), round(float(c.get("score", 0)), 3), c.get("text", "")[:120] + "…"]
            for c in ans.context.vector_chunks
        ]

        citations_str = "\n".join(f"- ArXiv: {c}" for c in ans.citations) if ans.citations else "None identified"
        full_response = f"{ans.answer}\n\n### 📑 Cited Papers:\n{citations_str}"

        return full_response, facts_table, chunks_table

    except Exception as exc:
        return f"❌ Query failed: {exc}", [], []


# ---------------------------------------------------------------------------
# UI Construction Factory
# ---------------------------------------------------------------------------

PRESET_QUERIES = [
    "SHAP LIME Random Forest XGBoost tabular data explainability",
    "SHAP feature importance machine learning interpretability",
    "LIME local interpretable model-agnostic explanations",
    "XGBoost gradient boosting tabular classification",
]

PRESET_QUESTIONS = [
    "Which papers evaluate SHAP on Random Forest and what performance metrics did they report?",
    "What datasets are commonly used for tabular explainability experiments?",
    "How does XGBoost compare with Logistic Regression in terms of accuracy and explainability?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ResearchGraph AI") as demo:
        gr.Markdown(
            """
            # 🔬 ResearchGraph AI Cockpit
            **End-to-End GraphRAG pipeline for Machine Learning Research on Tabular Data & Explainability (XAI)**
            """
        )

        with gr.Tabs():
            # TAB 1: INGESTION
            with gr.Tab("📥 1. Ingest Papers"):
                gr.Markdown("### Search ArXiv or Upload Local PDFs into `data/raw/` and extract chunks into `data/processed/`")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🌐 Option A: Search ArXiv")
                        query_box = gr.Textbox(
                            label="ArXiv Search Query",
                            value=PRESET_QUERIES[0],
                            lines=2,
                        )
                        preset_drop = gr.Dropdown(choices=PRESET_QUERIES, label="Quick Presets", interactive=True)
                        preset_drop.change(lambda q: q, inputs=preset_drop, outputs=query_box)
                        max_papers = gr.Slider(1, 20, value=3, step=1, label="Max ArXiv papers")
                        run_ingest_btn = gr.Button("▶ Download & Ingest from ArXiv", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 Option B: Upload Local PDF(s)")
                        pdf_upload_input = gr.File(
                            label="Select PDF file(s) from your computer",
                            file_count="multiple",
                            file_types=[".pdf"],
                        )
                        chunk_size = gr.Slider(500, 2000, value=1000, step=100, label="Chunk size (chars)")
                        run_upload_btn = gr.Button("📤 Process Uploaded PDF(s)", variant="secondary")

                ingest_summary = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📚 Cached Papers")
                        refresh_btn = gr.Button("🔄 Refresh Papers")
                        papers_table = gr.Dataframe(
                            headers=["ArXiv ID", "Title", "Published", "Authors"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Column():
                        gr.Markdown("#### 📜 Ingestion Logs")
                        ingest_logs = gr.Textbox(lines=14, interactive=False, autoscroll=True)

                # Events
                run_ingest_btn.click(
                    fn=run_ingestion_ui,
                    inputs=[query_box, max_papers, chunk_size],
                    outputs=[ingest_logs, papers_table, ingest_summary],
                )
                run_upload_btn.click(
                    fn=run_pdf_upload_ui,
                    inputs=[pdf_upload_input, chunk_size],
                    outputs=[ingest_logs, papers_table, ingest_summary],
                )
                refresh_btn.click(fn=refresh_papers_ui, outputs=[papers_table, ingest_summary])

            # TAB 2: EXTRACTION
            with gr.Tab("🧠 2. Extract & Build Graph"):
                gr.Markdown(
                    """
                    ### LLM Entity & Relation Extraction
                    Reads paper chunks, identifies **(Paper)-[:RUNS]->(Experiment)-[:USES_ALGORITHM]->(Algorithm)**,
                    **[:ON_DATASET]->(Dataset)**, **[:EXPLAINED_WITH]->(ExplainabilityMethod)**, and **[:PRODUCES]->(Result)**,
                    and indexes vectors in **Qdrant** & Knowledge Graph nodes in **Neo4j**.
                    """
                )
                with gr.Row():
                    extract_limit = gr.Slider(1, 20, value=3, step=1, label="Papers to process (0 = all)")
                    run_extract_btn = gr.Button("⚡ Run LLM Extraction & Indexing", variant="primary")

                extract_summary = gr.Textbox(label="Extraction Status", interactive=False)
                extract_logs = gr.Textbox(label="Extraction Logs", lines=12, interactive=False, autoscroll=True)

                run_extract_btn.click(
                    fn=run_extraction_ui,
                    inputs=[extract_limit],
                    outputs=[extract_logs, extract_summary],
                )

            # TAB 3: GraphRAG Q&A
            with gr.Tab("💬 3. GraphRAG Q&A"):
                gr.Markdown("### Ask questions against the hybrid Knowledge Graph + Vector Store")
                with gr.Row():
                    question_box = gr.Textbox(
                        label="Scientific Question",
                        placeholder="e.g. Which papers evaluate SHAP on Random Forest?",
                        value=PRESET_QUESTIONS[0],
                        lines=2,
                        scale=3,
                    )
                    ask_btn = gr.Button("🔍 Answer Question", variant="primary", scale=1)

                qa_preset_drop = gr.Dropdown(choices=PRESET_QUESTIONS, label="Sample Questions")
                qa_preset_drop.change(lambda q: q, inputs=qa_preset_drop, outputs=question_box)

                answer_box = gr.Markdown(label="Synthesized Grounded Answer")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🕸️ Retrieved Graph Facts (Neo4j)")
                        graph_table = gr.Dataframe(
                            headers=["ArXiv ID", "Paper Title", "Algorithm", "Dataset", "XAI", "Results"],
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("#### 📄 Retrieved Text Excerpts (Qdrant)")
                        vector_table = gr.Dataframe(
                            headers=["ArXiv ID", "Page", "Score", "Snippet"],
                            interactive=False,
                        )

                ask_btn.click(
                    fn=run_qa_ui,
                    inputs=[question_box],
                    outputs=[answer_box, graph_table, vector_table],
                )

        demo.load(fn=refresh_papers_ui, outputs=[papers_table, ingest_summary])

    return demo


if __name__ == "__main__":
    app_demo = build_ui()
    app_demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
