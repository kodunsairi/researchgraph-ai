"""Unit tests for GraphRAG hybrid retrieval."""
from __future__ import annotations

from unittest.mock import MagicMock

from researchgraph.retrieval import GraphRetriever, HybridRetriever, RetrievalContext


def test_retrieval_context_formatting():
    ctx = RetrievalContext(
        query="What is the AUC of Random Forest?",
        graph_facts=[
            {
                "arxiv_id": "2301.00001",
                "paper_title": "Explainable Tabular ML",
                "algorithm": "Random Forest",
                "dataset": "Adult",
                "xai_method": "SHAP",
                "results": [{"metric": "AUC", "value": 0.91, "unit": None}],
            }
        ],
        vector_chunks=[
            {
                "arxiv_id": "2301.00001",
                "page_start": 3,
                "text": "Random Forest achieved an AUC score of 0.91 on Adult Census.",
            }
        ],
    )
    formatted = ctx.to_formatted_context()
    assert "Explainable Tabular ML" in formatted
    assert "AUC=0.91" in formatted
    assert "Page 3" in formatted


def test_hybrid_retriever_delegation():
    mock_graph = MagicMock(spec=GraphRetriever)
    mock_graph.query_experiments.return_value = [{"paper_title": "Paper A"}]

    mock_vec = MagicMock()
    mock_vec.search.return_value = [{"text": "Chunk text"}]

    retriever = HybridRetriever(graph_retriever=mock_graph, vector_store=mock_vec)
    ctx = retriever.retrieve("SHAP with Random Forest", top_chunks=2, top_graph=3)

    assert len(ctx.graph_facts) == 1
    assert len(ctx.vector_chunks) == 1
    assert mock_graph.query_experiments.called
    assert mock_vec.search.called
