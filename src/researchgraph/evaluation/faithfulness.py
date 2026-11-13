"""Evaluation metrics for GraphRAG retrieval and generation."""
from __future__ import annotations

import re
from typing import Any, List
from researchgraph.generation.generator import GeneratedAnswer


def evaluate_faithfulness(answer: GeneratedAnswer) -> float:
    """Heuristic assessment of answer faithfulness against retrieved facts and chunks.

    Returns a score between 0.0 and 1.0 based on overlap between numbers/metrics
    asserted in the answer and those actually present in the context.
    """
    context_text = answer.context.to_formatted_context()
    if not answer.answer.strip():
        return 0.0

    # Extract all numbers/decimals from the generated answer
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer.answer))
    if not answer_numbers:
        return 1.0  # Qualitative answer without hallucinated numbers

    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context_text))

    # Calculate proportion of answer numbers that are grounded in context
    grounded = answer_numbers.intersection(context_numbers)
    return round(len(grounded) / len(answer_numbers), 2)


def evaluate_citation_coverage(answer: GeneratedAnswer) -> float:
    """Measure the ratio of cited papers against available papers in context."""
    available_papers = {f.get("arxiv_id") for f in answer.context.graph_facts if f.get("arxiv_id")}
    available_papers.update({c.get("arxiv_id") for c in answer.context.vector_chunks if c.get("arxiv_id")})

    if not available_papers:
        return 1.0

    cited = set(answer.citations)
    overlap = cited.intersection(available_papers)
    return round(len(overlap) / len(available_papers), 2)
