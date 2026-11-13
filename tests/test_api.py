"""Unit tests for FastAPI endpoints (direct function calls)."""
from __future__ import annotations

import researchgraph.api.app as api_module
from researchgraph.ingestion.arxiv_client import PaperMeta


def test_health_endpoint():
    res = api_module.health()
    assert res["status"] == "ok"
    assert res["app"] == "ResearchGraph AI"


def test_list_papers(monkeypatch):
    mock_metas = [
        PaperMeta(
            arxiv_id="2301.00001",
            title="Sample Paper",
            authors=["Alice"],
            abstract="Sample Abstract",
            published="2023-01-01",
            pdf_url="http://arxiv.org/pdf/2301.00001",
            primary_category="cs.LG",
            local_pdf="2301.00001.pdf",
        )
    ]
    monkeypatch.setattr(api_module, "load_all_metadata", lambda: mock_metas)

    papers = api_module.list_papers()
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "2301.00001"
    assert papers[0]["title"] == "Sample Paper"
