"""Hybrid retriever combining Neo4j Knowledge Graph facts and Qdrant vector chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from researchgraph.config import Settings, get_settings
from researchgraph.logging_setup import get_logger
from researchgraph.retrieval.graph_retriever import GraphRetriever
from researchgraph.vectorstore import VectorStore

log = get_logger(__name__)


@dataclass
class RetrievalContext:
    """Consolidated context containing both structured facts and text chunks."""
    query: str
    graph_facts: List[dict[str, Any]] = field(default_factory=list)
    vector_chunks: List[dict[str, Any]] = field(default_factory=list)

    def to_formatted_context(self) -> str:
        """Produce a formatted string suitable for prompting an answer generation LLM."""
        lines: list[str] = []

        if self.graph_facts:
            lines.append("=== STRUCTURED KNOWLEDGE GRAPH FACTS ===")
            for f in self.graph_facts:
                metrics_str = ", ".join(
                    f"{m['metric']}={m['value']}" + (f" {m['unit']}" if m.get("unit") else "")
                    for m in f.get("results", [])
                    if m.get("metric") and m.get("value") is not None
                ) or "N/A"
                lines.append(
                    f"- Paper [{f.get('arxiv_id')}]: \"{f.get('paper_title')}\" | "
                    f"Algorithm: {f.get('algorithm')} | Dataset: {f.get('dataset')} | "
                    f"XAI: {f.get('xai_method') or 'None'} | Results: {metrics_str}"
                )
            lines.append("")

        if self.vector_chunks:
            lines.append("=== RETRIEVED TEXT EXCERPTS ===")
            for i, c in enumerate(self.vector_chunks, 1):
                arxiv_id = c.get("arxiv_id", "Unknown")
                page = c.get("page_start", 0)
                text = c.get("text", "").strip()
                lines.append(f"[{i}] [Paper {arxiv_id}, Page {page}]: {text}")
            lines.append("")

        if not lines:
            return "No matching scientific facts or text passages found."

        return "\n".join(lines)


class HybridRetriever:
    """Orchestrates combined Graph + Vector retrieval."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.settings = settings or get_settings()
        self.graph = graph_retriever
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_chunks: int = 5,
        top_graph: int = 10,
    ) -> RetrievalContext:
        """Execute hybrid search across graph and vector stores."""
        graph_facts: List[dict[str, Any]] = []
        vector_chunks: List[dict[str, Any]] = []

        # 1. Heuristic entity keyword extraction for graph filter
        algo_filter = None
        for cand in ("random forest", "xgboost", "logistic regression", "lightgbm", "catboost", "neural network"):
            if cand in query.lower():
                algo_filter = cand
                break

        xai_filter = None
        for cand in ("shap", "lime", "treeshap", "kernelshap", "permutation"):
            if cand in query.lower():
                xai_filter = cand
                break

        # 2. Query Knowledge Graph
        if self.graph:
            try:
                graph_facts = self.graph.query_experiments(
                    algorithm=algo_filter,
                    xai_method=xai_filter,
                    limit=top_graph,
                )
            except Exception as exc:
                log.warning("Graph retrieval failed: %s", exc)

        # 3. Query Vector Store
        if self.vector_store:
            try:
                vector_chunks = self.vector_store.search(query, top_k=top_chunks)
            except Exception as exc:
                log.warning("Vector retrieval failed: %s", exc)

        return RetrievalContext(
            query=query,
            graph_facts=graph_facts,
            vector_chunks=vector_chunks,
        )
