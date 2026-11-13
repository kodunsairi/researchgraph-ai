"""Entity and relation extraction module."""
from researchgraph.extraction.extractor import EntityExtractor
from researchgraph.extraction.pipeline import process_paper, run_full_extraction

__all__ = [
    "EntityExtractor",
    "process_paper",
    "run_full_extraction",
]
