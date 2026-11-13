"""FastAPI application for ResearchGraph AI pipeline."""
from __future__ import annotations

from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from researchgraph.config import get_settings
from researchgraph.extraction import process_paper, run_full_extraction
from researchgraph.generation import AnswerGenerator
from researchgraph.graph import GraphWriter
from researchgraph.ingestion import load_all_metadata, run_ingestion
from researchgraph.retrieval import GraphRetriever, HybridRetriever
from researchgraph.vectorstore import VectorStore

app = FastAPI(
    title="ResearchGraph AI API",
    description="GraphRAG Pipeline API for Tabular ML and Explainable AI Research Papers",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    query: str = "SHAP LIME Random Forest XGBoost tabular data explainability"
    max_papers: int = 5
    chunk_size: int = 1000


class ExtractRequest(BaseModel):
    arxiv_id: Optional[str] = None
    max_papers: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    top_chunks: int = 5
    top_graph: int = 10


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "ResearchGraph AI"}


@app.get("/papers")
def list_papers() -> List[dict[str, Any]]:
    metas = load_all_metadata()
    return [
        {
            "arxiv_id": m.arxiv_id,
            "title": m.title,
            "authors": m.authors,
            "published": m.published,
            "pdf_url": m.pdf_url,
        }
        for m in metas
    ]


@app.post("/ingest")
def ingest_papers(req: IngestRequest) -> dict[str, Any]:
    try:
        chunks = run_ingestion(
            query=req.query,
            max_papers=req.max_papers,
            chunk_size=req.chunk_size,
        )
        return {
            "status": "success",
            "papers_downloaded": len(set(c.arxiv_id for c in chunks)),
            "total_chunks": len(chunks),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract")
def extract_entities(req: ExtractRequest) -> dict[str, Any]:
    try:
        if req.arxiv_id:
            writer = None
            try:
                writer = GraphWriter()
                writer.setup_schema()
            except Exception:
                pass

            vector_store = None
            try:
                vector_store = VectorStore()
            except Exception:
                pass

            exps = process_paper(
                arxiv_id=req.arxiv_id,
                writer=writer,
                vector_store=vector_store,
            )
            if writer:
                writer.close()
            return {"arxiv_id": req.arxiv_id, "experiments_extracted": len(exps)}

        results = run_full_extraction(max_papers=req.max_papers)
        return {"status": "success", "summary": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query")
def query_graphrag(req: QueryRequest) -> dict[str, Any]:
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
        answer = generator.generate(req.question)

        return {
            "question": answer.question,
            "answer": answer.answer,
            "citations": answer.citations,
            "graph_facts_count": len(answer.context.graph_facts),
            "vector_chunks_count": len(answer.context.vector_chunks),
            "graph_facts": answer.context.graph_facts,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
