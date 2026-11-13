"""Unit tests for entity extraction with mock LLM provider."""
from __future__ import annotations

from unittest.mock import MagicMock

from researchgraph.extraction.extractor import EntityExtractor, _extract_json_block
from researchgraph.llm.provider import LLMProvider


def test_extract_json_block():
    raw_markdown = """Here is the result:
```json
{
  "experiments": []
}
```
Hope this helps!"""
    extracted = _extract_json_block(raw_markdown)
    assert extracted.strip() == '{\n  "experiments": []\n}'


def test_entity_extractor_parses_experiments():
    sample_json = """{
  "experiments": [
    {
      "algorithm_name": "Random Forest",
      "algorithm_family": "tree_ensemble",
      "dataset_name": "Adult Census",
      "dataset_domain": "demographics",
      "is_tabular": true,
      "explainability_method": "SHAP",
      "metrics": [
        {"name": "AUC", "value": 0.89, "unit": null},
        {"name": "Accuracy", "value": 0.85, "unit": "%"}
      ]
    }
  ]
}"""

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete.return_value = sample_json

    extractor = EntityExtractor(llm=mock_llm)
    experiments = extractor.extract_from_text(
        "We trained a Random Forest model on the Adult Census tabular dataset and explained predictions using SHAP, achieving an AUC of 0.89 and accuracy of 85%.",
        arxiv_id="2301.00001",
    )

    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.paper_arxiv_id == "2301.00001"
    assert exp.algorithm.name == "Random Forest"
    assert exp.algorithm.family == "tree_ensemble"
    assert exp.dataset.name == "Adult Census"
    assert exp.dataset.is_tabular is True
    assert exp.explainability_method is not None
    assert exp.explainability_method.name == "SHAP"
    assert len(exp.metrics) == 2
    assert exp.metrics[0].name == "AUC"
    assert exp.metrics[0].value == 0.89
