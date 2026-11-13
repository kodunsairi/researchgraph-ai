"""Unit tests for answer generation."""
from __future__ import annotations

from unittest.mock import MagicMock

from researchgraph.generation import AnswerGenerator
from researchgraph.llm.provider import LLMProvider
from researchgraph.retrieval import RetrievalContext


def test_answer_generator():
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete.return_value = "According to [Paper 2301.00001], Random Forest achieved an AUC of 0.91."

    generator = AnswerGenerator(llm=mock_llm)
    ctx = RetrievalContext(
        query="What is the AUC of Random Forest?",
        graph_facts=[
            {
                "arxiv_id": "2301.00001",
                "paper_title": "Explainable Tabular ML",
                "algorithm": "Random Forest",
                "dataset": "Adult",
                "xai_method": "SHAP",
                "results": [{"metric": "AUC", "value": 0.91}],
            }
        ],
    )

    result = generator.generate("What is the AUC of Random Forest?", context=ctx)
    assert result.question == "What is the AUC of Random Forest?"
    assert "2301.00001" in result.answer
    assert "2301.00001" in result.citations
    assert mock_llm.complete.called
