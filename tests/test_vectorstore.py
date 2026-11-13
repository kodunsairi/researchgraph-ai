"""Unit tests for vectorstore and embedder (offline with mocks)."""
from __future__ import annotations

from unittest.mock import MagicMock

from researchgraph.ingestion.pdf_extractor import TextChunk
from researchgraph.vectorstore import Embedder, VectorStore


def test_embedder_dimension_and_embed(monkeypatch):
    embedder = Embedder()
    # Mock _load_model to set a dummy model so it doesn't download weights over the network
    embedder._model = "dummy"

    assert embedder.dimension == 384
    vecs = embedder.embed(["Hello world", "Machine learning"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def test_vectorstore_upsert_and_search():
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.payload = {
        "arxiv_id": "2301.12345",
        "chunk_index": 1,
        "text": "SHAP values on Random Forest show feature importance.",
        "page_start": 2,
        "page_end": 2,
    }
    mock_client.search.return_value = [mock_hit]

    embedder = Embedder()
    embedder._model = "dummy"

    store = VectorStore(client=mock_client, embedder=embedder)

    chunks = [
        TextChunk(
            arxiv_id="2301.12345",
            chunk_index=1,
            text="SHAP values on Random Forest show feature importance.",
            page_start=2,
            page_end=2,
            char_start=500,
            char_end=560,
        )
    ]

    count = store.upsert_chunks(chunks)
    assert count == 1
    assert mock_client.upsert.called

    results = store.search("explainability in random forest", top_k=1)
    assert len(results) == 1
    assert results[0]["arxiv_id"] == "2301.12345"
    assert results[0]["score"] == 0.95
