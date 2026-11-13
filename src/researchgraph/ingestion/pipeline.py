"""Ingestion pipeline — ties ArXiv download and PDF extraction together.

Typical usage
-------------
    from researchgraph.ingestion.pipeline import run_ingestion

    chunks = run_ingestion(
        query="SHAP LIME Random Forest tabular data explainability",
        max_papers=10,
    )
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from researchgraph.config import get_settings
from researchgraph.ingestion.arxiv_client import PaperMeta, search_and_download
from researchgraph.ingestion.pdf_extractor import TextChunk, extract_chunks
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


def run_ingestion(
    query: str = "SHAP LIME Random Forest XGBoost tabular data explainability",
    max_papers: int = 20,
    chunk_size: int = 1_000,
    overlap: int = 150,
    *,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> list[TextChunk]:
    """Download papers from ArXiv and extract text chunks.

    Args:
        query:          ArXiv query string.
        max_papers:     Maximum number of papers to fetch.
        chunk_size:     Character size of each text chunk.
        overlap:        Overlap between consecutive chunks.
        raw_dir:        Directory for PDFs (defaults to settings.raw_dir).
        processed_dir:  Directory for chunk JSON files.
        metadata_dir:   Directory for paper metadata JSON files.

    Returns:
        All :class:`TextChunk` objects across every paper.
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.raw_dir
    processed_dir = processed_dir or settings.processed_dir
    metadata_dir = metadata_dir or settings.metadata_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Ingestion start | query=%r | max_papers=%d ===", query, max_papers)

    # 1. Download papers -------------------------------------------------------
    papers: list[PaperMeta] = search_and_download(
        query=query,
        max_results=max_papers,
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
    )
    log.info("Downloaded/cached %d papers", len(papers))

    # 2. Extract chunks --------------------------------------------------------
    all_chunks: list[TextChunk] = []

    for paper in papers:
        if not paper.local_pdf:
            log.warning("No local PDF for %s — skipping extraction", paper.arxiv_id)
            continue

        pdf_path = raw_dir / paper.local_pdf
        chunks = extract_chunks(
            pdf_path=pdf_path,
            arxiv_id=paper.arxiv_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # 3. Persist chunks as JSON (one file per paper) -----------------------
        if chunks:
            safe_id = paper.arxiv_id.replace("/", "_")
            out_path = processed_dir / f"{safe_id}_chunks.json"
            out_path.write_text(
                json.dumps([asdict(c) for c in chunks], indent=2),
                encoding="utf-8",
            )
            log.info("Saved %d chunks → %s", len(chunks), out_path.name)
            all_chunks.extend(chunks)

    log.info(
        "=== Ingestion complete | papers=%d | chunks=%d ===",
        len(papers),
        len(all_chunks),
    )
    return all_chunks


def load_chunks(arxiv_id: str, processed_dir: Path | None = None) -> list[TextChunk]:
    """Load persisted chunks for a single paper by ArXiv ID."""
    settings = get_settings()
    processed_dir = processed_dir or settings.processed_dir
    safe_id = arxiv_id.replace("/", "_")
    path = processed_dir / f"{safe_id}_chunks.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TextChunk(**item) for item in data]
