"""Evaluation module for GraphRAG quality checks."""
from researchgraph.evaluation.faithfulness import (
    evaluate_citation_coverage,
    evaluate_faithfulness,
)

__all__ = [
    "evaluate_faithfulness",
    "evaluate_citation_coverage",
]
