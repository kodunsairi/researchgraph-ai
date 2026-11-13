"""Unit test for local PDF upload and chunk processing in UI."""
from __future__ import annotations

import sys
from pathlib import Path
from pypdf import PdfWriter

# Add project root to sys.path so scripts can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ui import run_pdf_upload_ui


def test_run_pdf_upload_ui(tmp_path: Path, monkeypatch):
    # Create a small dummy valid PDF
    sample_pdf = tmp_path / "test_paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(sample_pdf, "wb") as f:
        writer.write(f)

    # Mock extract_chunks to return mock chunk
    from researchgraph.ingestion.pdf_extractor import TextChunk
    mock_chunks = [
        TextChunk(
            arxiv_id="test_paper",
            chunk_index=0,
            text="This is a test chunk from local paper.",
            page_start=0,
            page_end=0,
            char_start=0,
            char_end=37,
        )
    ]
    monkeypatch.setattr("scripts.ui.extract_chunks", lambda **kwargs: mock_chunks)

    # Run upload handler
    logs, table, summary = run_pdf_upload_ui([str(sample_pdf)], chunk_size=1000)

    assert "Processed 1 uploaded PDF(s)" in summary
    assert "data/processed/" in summary
