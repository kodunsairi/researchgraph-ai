"""Full extraction pipeline linking Ingestion -> LLM Extraction -> GraphWriter & VectorStore."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from researchgraph.config import Settings, get_settings
from researchgraph.extraction.extractor import EntityExtractor
from researchgraph.graph import GraphWriter, Paper
from researchgraph.graph.schema import Experiment
from researchgraph.ingestion import load_all_metadata, load_chunks
from researchgraph.logging_setup import get_logger
from researchgraph.vectorstore import VectorStore

log = get_logger(__name__)


def process_paper(
    arxiv_id: str,
    extractor: Optional[EntityExtractor] = None,
    writer: Optional[GraphWriter] = None,
    vector_store: Optional[VectorStore] = None,
    settings: Optional[Settings] = None,
) -> List[Experiment]:
    """Process a single ingested paper: extract entities, index vectors, and write to Neo4j."""
    s = settings or get_settings()
    extractor = extractor or EntityExtractor()

    # 1. Load paper metadata
    meta_path = s.metadata_dir / f"{arxiv_id.replace('/', '_')}.json"
    paper_node: Optional[Paper] = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            paper_node = Paper(
                arxiv_id=m["arxiv_id"],
                title=m.get("title", ""),
                authors=m.get("authors", []),
                abstract=m.get("abstract", ""),
                published=m.get("published", ""),
            )
        except Exception as exc:
            log.warning("Could not parse metadata for %s: %s", arxiv_id, exc)

    if paper_node is None:
        paper_node = Paper(arxiv_id=arxiv_id, title=f"Paper {arxiv_id}")

    # 2. Upsert Paper to Neo4j
    if writer:
        writer.upsert_paper(paper_node)

    # 3. Load chunks
    chunks = load_chunks(arxiv_id, processed_dir=s.processed_dir)

    # 4. Upsert chunks to VectorStore
    if vector_store and chunks:
        try:
            vector_store.upsert_chunks(chunks)
        except Exception as exc:
            log.warning("Vector indexing skipped/failed for %s: %s", arxiv_id, exc)

    # 5. Extract experiments from text chunks (focusing on abstract and experimental sections)
    experiments: List[Experiment] = []
    seen_ids: set[str] = set()

    # Always extract from abstract first
    if paper_node.abstract:
        abs_exps = extractor.extract_from_text(paper_node.abstract, arxiv_id)
        for e in abs_exps:
            eid = e.get_experiment_id()
            if eid not in seen_ids:
                seen_ids.add(eid)
                experiments.append(e)

    # Extract from chunks containing keywords
    keywords = ("accuracy", "auc", "result", "metric", "experiment", "shap", "lime", "table", "performance")
    for chunk in chunks:
        lower = chunk.text.lower()
        if any(k in lower for k in keywords):
            chunk_exps = extractor.extract_from_text(chunk.text, arxiv_id)
            for e in chunk_exps:
                eid = e.get_experiment_id()
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    experiments.append(e)

    # 6. Upsert experiments into Neo4j
    if writer:
        for exp in experiments:
            try:
                writer.upsert_experiment(exp)
            except Exception as exc:
                log.warning("Failed to write experiment %s to Neo4j: %s", exp.id, exc)

    log.info("Finished processing %s: %d experiments extracted", arxiv_id, len(experiments))
    return experiments


def run_full_extraction(
    max_papers: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> dict[str, int]:
    """Run extraction across all ingested papers."""
    s = settings or get_settings()
    extractor = EntityExtractor()

    writer: Optional[GraphWriter] = None
    try:
        writer = GraphWriter(settings=s)
        writer.setup_schema()
    except Exception as exc:
        log.warning("Neo4j writer could not connect: %s. Proceeding in offline mode.", exc)
        writer = None

    vector_store: Optional[VectorStore] = None
    try:
        vector_store = VectorStore(settings=s)
        vector_store.ensure_collection()
    except Exception as exc:
        log.warning("VectorStore could not connect: %s. Proceeding without vector indexing.", exc)
        vector_store = None

    metas = load_all_metadata(s.metadata_dir)
    if max_papers:
        metas = metas[:max_papers]

    summary = {}
    for meta in metas:
        exps = process_paper(
            arxiv_id=meta.arxiv_id,
            extractor=extractor,
            writer=writer,
            vector_store=vector_store,
            settings=s,
        )
        summary[meta.arxiv_id] = len(exps)

    if writer:
        writer.close()

    return summary
