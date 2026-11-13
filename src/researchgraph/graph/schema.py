"""Pydantic models representing the Knowledge Graph schema (v2)."""
from __future__ import annotations

import hashlib
from typing import Optional
from pydantic import BaseModel, Field


class Metric(BaseModel):
    """Metric measurement reported by an experiment."""
    name: str  # e.g. "AUC", "Accuracy", "F1", "RMSE"
    value: float
    unit: Optional[str] = None


class Algorithm(BaseModel):
    """Machine learning algorithm / model."""
    name: str  # e.g. "Random Forest", "XGBoost", "Logistic Regression"
    family: Optional[str] = None  # e.g. "tree_ensemble", "linear"


class Dataset(BaseModel):
    """Dataset used in experiments."""
    name: str  # e.g. "Adult Census", "COMPAS", "German Credit"
    domain: Optional[str] = None  # e.g. "finance", "healthcare", "tabular"
    is_tabular: bool = True


class ExplainabilityMethod(BaseModel):
    """Explainability or interpretability technique."""
    name: str  # e.g. "SHAP", "LIME", "TreeSHAP"


class Experiment(BaseModel):
    """A specific experimental run linking algorithm, dataset, explainability, and metrics."""
    id: Optional[str] = None
    paper_arxiv_id: str
    algorithm: Algorithm
    dataset: Dataset
    explainability_method: Optional[ExplainabilityMethod] = None
    metrics: list[Metric] = Field(default_factory=list)

    def get_experiment_id(self) -> str:
        """Compute unique deterministic identifier for this experiment."""
        if self.id:
            return self.id
        xai_name = self.explainability_method.name if self.explainability_method else "none"
        raw_key = f"{self.paper_arxiv_id}::{self.algorithm.name}::{self.dataset.name}::{xai_name}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


class Paper(BaseModel):
    """Paper node representing an academic publication."""
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published: str = ""
