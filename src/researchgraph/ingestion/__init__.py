"""Public API for the ingestion module."""
from researchgraph.ingestion.arxiv_client import PaperMeta, load_all_metadata, search_and_download
from researchgraph.ingestion.pdf_extractor import TextChunk, extract_chunks
from researchgraph.ingestion.pipeline import load_chunks, run_ingestion

__all__ = [
    # ArXiv
    "PaperMeta",
    "search_and_download",
    "load_all_metadata",
    # PDF extraction
    "TextChunk",
    "extract_chunks",
    # Pipeline
    "run_ingestion",
    "load_chunks",
]
