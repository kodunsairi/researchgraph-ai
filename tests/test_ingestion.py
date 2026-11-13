"""Tests for the ingestion module (offline — no network or Docker required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchgraph.ingestion.pdf_extractor import TextChunk, _clean, extract_chunks
from researchgraph.ingestion.pipeline import load_chunks


# ---------------------------------------------------------------------------
# _clean helper
# ---------------------------------------------------------------------------

def test_clean_collapses_whitespace():
    assert _clean("hello   world\n\t!") == "hello world !"


def test_clean_strips():
    assert _clean("  hi  ") == "hi"


# ---------------------------------------------------------------------------
# extract_chunks — missing file
# ---------------------------------------------------------------------------

def test_extract_chunks_missing_pdf(tmp_path: Path):
    chunks = extract_chunks(tmp_path / "ghost.pdf", arxiv_id="0000.00000")
    assert chunks == []


# ---------------------------------------------------------------------------
# load_chunks — missing processed file
# ---------------------------------------------------------------------------

def test_load_chunks_missing(tmp_path: Path):
    chunks = load_chunks("9999.99999", processed_dir=tmp_path)
    assert chunks == []


# ---------------------------------------------------------------------------
# load_chunks — round-trip
# ---------------------------------------------------------------------------

def test_load_chunks_roundtrip(tmp_path: Path):
    sample = [
        TextChunk(
            arxiv_id="1234.56789",
            chunk_index=0,
            text="Hello world",
            page_start=0,
            page_end=0,
            char_start=0,
            char_end=11,
        )
    ]
    out = tmp_path / "1234.56789_chunks.json"
    import dataclasses
    out.write_text(json.dumps([dataclasses.asdict(c) for c in sample]), encoding="utf-8")

    loaded = load_chunks("1234.56789", processed_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].text == "Hello world"
    assert loaded[0].chunk_index == 0
