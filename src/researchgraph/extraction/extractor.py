"""Entity and relationship extractor using LLM."""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from researchgraph.extraction.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
)
from researchgraph.graph.schema import (
    Algorithm,
    Dataset,
    Experiment,
    ExplainabilityMethod,
    Metric,
)
from researchgraph.llm import LLMProvider, get_provider
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


def _extract_json_block(text: str) -> str:
    """Extract JSON string from potential markdown fences or noisy text."""
    # Try finding markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Otherwise look for first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


class EntityExtractor:
    """Extracts structured GraphRAG schema v2 entities from paper text."""

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm or get_provider()

    def extract_from_text(self, text: str, arxiv_id: str) -> List[Experiment]:
        """Send chunk/section text to LLM and parse into list of Experiment instances."""
        if not text or len(text.strip()) < 50:
            return []

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXTRACTION_USER_PROMPT_TEMPLATE.format(
                    arxiv_id=arxiv_id,
                    text=text[:3500],  # safeguard context window
                ),
            },
        ]

        try:
            raw_reply = self.llm.complete(messages, temperature=0.0)
            cleaned_json = _extract_json_block(raw_reply)
            data = json.loads(cleaned_json)
        except Exception as exc:
            log.warning("Failed to extract or parse JSON for %s: %s", arxiv_id, exc)
            return []

        experiments_data = data.get("experiments", [])
        if not isinstance(experiments_data, list):
            return []

        parsed_experiments: List[Experiment] = []
        for exp_dict in experiments_data:
            if not isinstance(exp_dict, dict):
                continue

            algo_name = exp_dict.get("algorithm_name")
            dataset_name = exp_dict.get("dataset_name")
            if not algo_name or not dataset_name:
                continue

            algo = Algorithm(
                name=str(algo_name).strip(),
                family=exp_dict.get("algorithm_family"),
            )
            dataset = Dataset(
                name=str(dataset_name).strip(),
                domain=exp_dict.get("dataset_domain"),
                is_tabular=bool(exp_dict.get("is_tabular", True)),
            )

            xai_name = exp_dict.get("explainability_method")
            xai = ExplainabilityMethod(name=str(xai_name).strip()) if xai_name else None

            metrics_list: List[Metric] = []
            for m in exp_dict.get("metrics", []):
                if isinstance(m, dict) and "name" in m and "value" in m:
                    try:
                        val = float(m["value"])
                        metrics_list.append(
                            Metric(
                                name=str(m["name"]).strip(),
                                value=val,
                                unit=m.get("unit"),
                            )
                        )
                    except (ValueError, TypeError):
                        continue

            experiment = Experiment(
                paper_arxiv_id=arxiv_id,
                algorithm=algo,
                dataset=dataset,
                explainability_method=xai,
                metrics=metrics_list,
            )
            parsed_experiments.append(experiment)

        return parsed_experiments
