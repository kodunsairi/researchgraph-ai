"""ArXiv search and PDF download client.

Uses the ``arxiv`` package (wraps the ArXiv API v2).
Downloads PDFs to ``settings.raw_dir`` and writes per-paper JSON metadata
to ``settings.metadata_dir``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv  # type: ignore[import-untyped]

from researchgraph.config import get_settings
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class PaperMeta:
    """Lightweight, serialisable metadata for one ArXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str          # ISO-8601 date string
    pdf_url: str
    primary_category: str
    local_pdf: str          # relative to raw_dir


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def search_and_download(
    query: str,
    max_results: int = 20,
    *,
    raw_dir: Path | None = None,
    metadata_dir: Path | None = None,
    delay_seconds: float = 3.0,
) -> list[PaperMeta]:
    """Search ArXiv and download PDFs that are not already cached.

    Args:
        query:          ArXiv search query string.
        max_results:    Maximum number of papers to fetch.
        raw_dir:        Override for PDF storage directory.
        metadata_dir:   Override for metadata JSON directory.
        delay_seconds:  Polite delay between successive downloads.

    Returns:
        List of :class:`PaperMeta` for every paper found (including cached ones).
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.raw_dir
    metadata_dir = metadata_dir or settings.metadata_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=delay_seconds)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[PaperMeta] = []

    for result in client.results(search):
        arxiv_id = result.get_short_id()          # e.g. "2301.00001v1"
        safe_id = arxiv_id.replace("/", "_")
        pdf_path = raw_dir / f"{safe_id}.pdf"
        meta_path = metadata_dir / f"{safe_id}.json"

        meta = PaperMeta(
            arxiv_id=arxiv_id,
            title=result.title,
            authors=[str(a) for a in result.authors],
            abstract=result.summary,
            published=result.published.date().isoformat(),
            pdf_url=result.pdf_url,
            primary_category=result.primary_category,
            local_pdf=pdf_path.name,
        )

        # Download only if PDF is missing
        if not pdf_path.exists():
            log.info("Downloading %s — %s", arxiv_id, result.title[:60])
            try:
                result.download_pdf(dirpath=str(raw_dir), filename=pdf_path.name)
                log.info("  saved → %s", pdf_path.name)
            except Exception as exc:
                log.warning("  download failed for %s: %s", arxiv_id, exc)
                meta.local_pdf = ""   # mark as unavailable
        else:
            log.info("Cache hit: %s", pdf_path.name)

        # Persist metadata JSON
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        papers.append(meta)

    log.info("search_and_download done — %d papers", len(papers))
    return papers


def load_all_metadata(metadata_dir: Path | None = None) -> list[PaperMeta]:
    """Load every persisted :class:`PaperMeta` from *metadata_dir*."""
    settings = get_settings()
    metadata_dir = metadata_dir or settings.metadata_dir
    metas: list[PaperMeta] = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metas.append(PaperMeta(**data))
        except Exception as exc:
            log.warning("Could not load %s: %s", path.name, exc)
    return metas
