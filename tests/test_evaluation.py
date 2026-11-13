"""Unit tests for evaluation metrics."""
from __future__ import annotations

from researchgraph.evaluation import evaluate_citation_coverage, evaluate_faithfulness
from researchgraph.generation import GeneratedAnswer
from researchgraph.retrieval import RetrievalContext


def test_evaluate_faithfulness():
    ctx = RetrievalContext(
        query="What is the AUC of Random Forest?",
        graph_facts=[
            {
                "arxiv_id": "2301.00001",
                "results": [{"metric": "AUC", "value": 0.91}],
            }
        ],
    )
    # Faithful answer containing the true number 0.91
    ans_faithful = GeneratedAnswer(
        question="What is the AUC?",
        answer="The reported AUC is 0.91.",
        context=ctx,
        citations=["2301.00001"],
    )
    score_f = evaluate_faithfulness(ans_faithful)
    assert score_f == 1.0

    # Unfaithful answer inventing 0.77
    ans_hallucinated = GeneratedAnswer(
        question="What is the AUC?",
        answer="The reported AUC is 0.77 and accuracy is 0.55.",
        context=ctx,
        citations=["2301.00001"],
    )
    score_h = evaluate_faithfulness(ans_hallucinated)
    assert score_h == 0.0


def test_evaluate_citation_coverage():
    ctx = RetrievalContext(
        query="test",
        graph_facts=[{"arxiv_id": "2301.00001"}, {"arxiv_id": "2301.00002"}],
    )
    ans = GeneratedAnswer(
        question="test",
        answer="According to 2301.00001...",
        context=ctx,
        citations=["2301.00001"],
    )
    cov = evaluate_citation_coverage(ans)
    assert cov == 0.5
