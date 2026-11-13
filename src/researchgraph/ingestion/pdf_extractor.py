"""PDF text extraction and chunking.

Extracts plain text from a PDF file using *pypdf* and splits it into
overlapping chunks suitable for downstream embedding / LLM processing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader  # type: ignore[import-untyped]

from researchgraph.logging_setup import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """One chunk of text extracted from a paper."""

    arxiv_id: str
    chunk_index: int
    text: str
    page_start: int   # first PDF page contributing to this chunk (0-based)
    page_end: int     # last  PDF page contributing to this chunk (0-based)
    char_start: int   # character offset in the full concatenated text
    char_end: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(page_number, cleaned_text)`` for each PDF page."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean(raw)
        if cleaned:
            pages.append((i, cleaned))
    return pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chunks(
    pdf_path: Path,
    arxiv_id: str,
    chunk_size: int = 1_000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Extract text from *pdf_path* and split into overlapping chunks.

    Args:
        pdf_path:   Path to the PDF file.
        arxiv_id:   ArXiv identifier (stored in every chunk for traceability).
        chunk_size: Target chunk size in characters.
        overlap:    Number of characters to repeat between consecutive chunks.

    Returns:
        Ordered list of :class:`TextChunk`.
    """
    if not pdf_path.exists():
        log.warning("PDF not found, skipping: %s", pdf_path)
        return []

    try:
        pages = _extract_pages(pdf_path)
    except Exception as exc:
        log.error("Failed to read %s: %s", pdf_path.name, exc)
        return []

    if not pages:
        log.warning("No extractable text in %s", pdf_path.name)
        return []

    # Build a full-text string with page boundary markers so we can recover
    # page numbers for each character offset.
    full_text = ""
    # Maps character offset → page number
    page_map: list[tuple[int, int]] = []   # (start_char, page_no)
    for page_no, text in pages:
        page_map.append((len(full_text), page_no))
        full_text += text + " "

    def _page_at(char_offset: int) -> int:
        """Return the page number for a given character offset."""
        result = 0
        for start, page_no in page_map:
            if start <= char_offset:
                result = page_no
            else:
                break
        return result

    # Sliding-window chunking
    chunks: list[TextChunk] = []
    pos = 0
    total = len(full_text)

    while pos < total:
        end = min(pos + chunk_size, total)
        chunk_text = full_text[pos:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    arxiv_id=arxiv_id,
                    chunk_index=len(chunks),
                    text=chunk_text,
                    page_start=_page_at(pos),
                    page_end=_page_at(end - 1),
                    char_start=pos,
                    char_end=end,
                )
            )
        pos += chunk_size - overlap   # advance with overlap
        if end == total:
            break

    log.info(
        "Extracted %d chunks from %s (%d pages, %d chars)",
        len(chunks),
        pdf_path.name,
        len(pages),
        total,
    )
    return chunks
