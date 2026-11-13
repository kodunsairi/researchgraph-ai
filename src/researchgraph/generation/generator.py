"""Grounded answer generation using GraphRAG retrieved context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from researchgraph.llm import LLMProvider, get_provider
from researchgraph.logging_setup import get_logger
from researchgraph.retrieval import HybridRetriever, RetrievalContext

log = get_logger(__name__)

GENERATION_SYSTEM_PROMPT = """You are ResearchGraph AI, an academic research assistant specializing in Explainable Machine Learning (XAI) and Tabular Data.

Your objective is to provide precise, faithful, and comprehensive scientific answers to user questions based STRICTLY on the provided context (Knowledge Graph facts and paper excerpts).

Guidelines:
1. Cite sources explicitly whenever stating experimental results, metrics, datasets, or conclusions (e.g. "[Paper 2301.12345, Page 2]").
2. Present quantitative comparisons clearly (e.g., using Markdown tables or bullet points).
3. If the context does not contain enough evidence to answer a question, clearly acknowledge the limitation rather than guessing.
4. Ground your assertions in the structured facts and excerpts provided.
"""

GENERATION_USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

USER QUESTION:
{question}

Provide a well-structured, cited, and accurate answer based on the context above:
"""


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with citations and retrieval context."""
    question: str
    answer: str
    context: RetrievalContext
    citations: List[str] = field(default_factory=list)


class AnswerGenerator:
    """Generates cited scientific answers given user queries and GraphRAG context."""

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.llm = llm or get_provider()
        self.retriever = retriever

    def generate(self, question: str, context: Optional[RetrievalContext] = None) -> GeneratedAnswer:
        """Retrieve context (if not provided) and generate grounded answer."""
        if context is None:
            if self.retriever is None:
                raise ValueError("Retriever must be provided if context is None")
            context = self.retriever.retrieve(question)

        formatted_context = context.to_formatted_context()
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GENERATION_USER_PROMPT_TEMPLATE.format(
                    context=formatted_context,
                    question=question,
                ),
            },
        ]

        reply = self.llm.complete(messages, temperature=0.1)

        # Extract paper citations mentioned
        cited_papers = set()
        for fact in context.graph_facts:
            aid = fact.get("arxiv_id")
            if aid and aid in reply:
                cited_papers.add(aid)
        for chunk in context.vector_chunks:
            aid = chunk.get("arxiv_id")
            if aid and aid in reply:
                cited_papers.add(aid)

        return GeneratedAnswer(
            question=question,
            answer=reply,
            context=context,
            citations=sorted(list(cited_papers)),
        )
