"""Run the ingestion pipeline from the command line.

Usage
-----
    python scripts/ingest.py
    python scripts/ingest.py --query "SHAP XGBoost tabular" --max-papers 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the src package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from researchgraph.ingestion import run_ingestion
from researchgraph.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers from ArXiv.")
    parser.add_argument(
        "--query",
        default="SHAP LIME Random Forest XGBoost tabular data explainability",
        help="ArXiv search query (default: SHAP/LIME/RF/XGB tabular)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum number of papers to download (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Characters per chunk (default: 1000)",
    )
    args = parser.parse_args()

    setup_logging()

    chunks = run_ingestion(
        query=args.query,
        max_papers=args.max_papers,
        chunk_size=args.chunk_size,
    )

    print(f"\n✅  Done — {len(chunks)} chunks from up to {args.max_papers} papers.")
    print(f"    PDFs   → data/raw/")
    print(f"    Chunks → data/processed/")
    print(f"    Meta   → data/metadata/")


if __name__ == "__main__":
    main()
