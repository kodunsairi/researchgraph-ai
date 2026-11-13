"""Qdrant vector store implementation for paper text chunks."""
from __future__ import annotations

import uuid
from typing import Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from researchgraph.config import Settings, get_settings
from researchgraph.ingestion.pdf_extractor import TextChunk
from researchgraph.logging_setup import get_logger
from researchgraph.vectorstore.embedder import Embedder

log = get_logger(__name__)


class VectorStore:
    """Manages Qdrant vector index for paper text chunks."""

    COLLECTION_NAME = "research_chunks"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[Embedder] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings.embedding_model)
        self.client = client or QdrantClient(url=self.settings.qdrant_url)

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.COLLECTION_NAME for c in collections)
            if not exists:
                log.info("Creating Qdrant collection: %s", self.COLLECTION_NAME)
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.embedder.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            log.warning("Could not verify/create Qdrant collection: %s", exc)

    def upsert_chunks(self, chunks: List[TextChunk]) -> int:
        """Embed and upsert text chunks into Qdrant."""
        if not chunks:
            return 0

        self.ensure_collection()
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        points = []
        for chunk, vec in zip(chunks, embeddings):
            # Generate deterministic UUID from arxiv_id + chunk_index
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.arxiv_id}_{chunk.chunk_index}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "arxiv_id": chunk.arxiv_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    },
                )
            )

        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
        log.info("Upserted %d chunk vectors into Qdrant", len(points))
        return len(points)

    def search(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        """Perform semantic similarity search on text chunks."""
        self.ensure_collection()
        query_vec = self.embedder.embed_one(query)
        hits = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "arxiv_id": payload.get("arxiv_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                    "page_start": payload.get("page_start", 0),
                    "page_end": payload.get("page_end", 0),
                }
            )
        return results
