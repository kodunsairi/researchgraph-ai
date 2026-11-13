"""Unit tests for graph schema and writer (using mock Neo4j driver)."""
from __future__ import annotations

from unittest.mock import MagicMock
from pydantic import SecretStr

from researchgraph.config import Settings
from researchgraph.graph import (
    Algorithm,
    Dataset,
    Experiment,
    ExplainabilityMethod,
    GraphWriter,
    Metric,
    Paper,
)


def test_schema_models():
    paper = Paper(
        arxiv_id="2301.00001",
        title="Interpretable ML on Tabular Data",
        authors=["Alice", "Bob"],
        abstract="We evaluate SHAP and LIME on Random Forest.",
        published="2023-01-01",
    )
    assert paper.arxiv_id == "2301.00001"
    assert len(paper.authors) == 2

    exp = Experiment(
        paper_arxiv_id="2301.00001",
        algorithm=Algorithm(name="Random Forest", family="tree_ensemble"),
        dataset=Dataset(name="Adult Census", domain="census", is_tabular=True),
        explainability_method=ExplainabilityMethod(name="SHAP"),
        metrics=[Metric(name="AUC", value=0.89, unit=None)],
    )
    exp_id = exp.get_experiment_id()
    assert isinstance(exp_id, str)
    assert len(exp_id) == 16


def test_graph_writer_upsert(monkeypatch):
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    settings = Settings(
        neo4j_password=SecretStr("mockpass123"),
    )
    writer = GraphWriter(settings=settings, driver=mock_driver)

    paper = Paper(
        arxiv_id="2301.00001",
        title="Test Title",
        authors=["Alice"],
    )
    writer.upsert_paper(paper)
    assert mock_session.run.called

    exp = Experiment(
        paper_arxiv_id="2301.00001",
        algorithm=Algorithm(name="XGBoost"),
        dataset=Dataset(name="Credit"),
        explainability_method=ExplainabilityMethod(name="LIME"),
        metrics=[Metric(name="Accuracy", value=0.92)],
    )
    exp_id = writer.upsert_experiment(exp)
    assert exp_id is not None
    assert mock_session.execute_write.called
