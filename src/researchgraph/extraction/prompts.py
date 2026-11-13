"""Prompts for LLM-based entity and relation extraction."""
from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """You are an expert scientific data extraction system specializing in Machine Learning research papers on Tabular Data and Explainable AI (XAI).

Your task is to analyze scientific text and extract all reported machine learning experiments, models, datasets, explainability methods, and quantitative evaluation results.

You MUST respond with valid JSON ONLY (no conversational preamble or postscript), adhering to this JSON structure:
{
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
        {"name": "Accuracy", "value": 0.84, "unit": "%"}
      ]
    }
  ]
}

Extraction Rules:
1. algorithm_name: Specific ML model (e.g., "Random Forest", "XGBoost", "Logistic Regression", "LightGBM", "CatBoost", "Neural Network").
2. algorithm_family: e.g. "tree_ensemble", "linear_model", "deep_learning", or null if unknown.
3. dataset_name: The name of the benchmark or real-world dataset (e.g. "Adult", "COMPAS", "German Credit", "HELOC").
4. is_tabular: true if tabular/relational features, false otherwise.
5. explainability_method: "SHAP", "LIME", "TreeSHAP", "KernelSHAP", "Permutation Importance", or null if no XAI method is applied.
6. metrics: Any quantitative scores with numeric values (e.g. AUC, Accuracy, F1, Precision, Recall, MSE, Brier Score).
7. If the text does not contain any concrete experiment or results, return {"experiments": []}.
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Text from paper (ArXiv ID: {arxiv_id}):
\"\"\"
{text}
\"\"\"

Extract all experimental setups and reported metrics as specified. Return valid JSON only.
"""
