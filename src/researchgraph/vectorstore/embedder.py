"""Embedding abstraction for vector store indexing and query encoding."""
from __future__ import annotations

from typing import Any, List
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


class Embedder:
    """Generates vector embeddings for text chunks and queries."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model: Any = None
        self._dim = 384  # default for BAAI/bge-small-en-v1.5

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
            log.info("Loading FastEmbed model: %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Loading SentenceTransformer model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                log.warning("Neither fastembed nor sentence_transformers installed. Using dummy embedder.")
                self._model = "dummy"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        if not texts:
            return []
        self._load_model()
        if self._model == "dummy":
            # Deterministic dummy vector for testing
            return [[0.01 * (hash(t + str(i)) % 100) for i in range(self._dim)] for t in texts]

        # FastEmbed returns a generator of numpy arrays
        if hasattr(self._model, "embed"):
            embeddings = list(self._model.embed(texts))
            return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]

        # SentenceTransformers encode
        if hasattr(self._model, "encode"):
            embeddings = self._model.encode(texts)
            return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]

        return [[0.0] * self._dim for _ in texts]

    def embed_one(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed([text])[0]
