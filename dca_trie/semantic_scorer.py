"""
Semantic Scorer for DCA-Trie.

Encodes question-context and KG paths into 384-dim embeddings
using sentence-transformers/all-MiniLM-L6-v2 and computes
cosine similarity as a relevance score.

Optimizations:
  - Batch encoding (up to 10x faster on GPU)
  - Configurable dictionary cache for repeated paths
  - Auto-detects CUDA if PyTorch is available
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticScorer:
    """
    Lightweight semantic similarity scorer with batch support.

    Usage:
        scorer = SemanticScorer()
        score = scorer.score("entity -> relation -> entity", "question text")
        scores = scorer.score_batch(["path1", "path2"], "question text")
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device=None, cache_size=50000):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self._cache_size = cache_size
        self._path_cache = {}

    def encode_path(self, path_str: str) -> np.ndarray:
        """Encode a single path string. Results cached up to cache_size entries."""
        if self._cache_size > 0:
            cached = self._path_cache.get(path_str)
            if cached is not None:
                return cached
            emb = self.model.encode(path_str, convert_to_numpy=True)
            if len(self._path_cache) < self._cache_size:
                self._path_cache[path_str] = emb
            return emb
        return self.model.encode(path_str, convert_to_numpy=True)

    def encode_query(self, question: str, partial_gen: str = "") -> np.ndarray:
        """Encode question + optional partial generation context."""
        text = question + " " + partial_gen if partial_gen else question
        return self.model.encode(text, convert_to_numpy=True)

    def encode_paths_batch(self, path_strs, batch_size=64) -> np.ndarray:
        """
        Encode multiple path strings in a single batch call.

        For GPU this is ~10x faster than encoding paths one-by-one.

        Args:
            path_strs: iterable of path strings
            batch_size: batch size for SentenceTransformer (default 64)

        Returns:
            np.ndarray of shape (num_paths, 384)
        """
        paths = list(path_strs)
        embs = self.model.encode(
            paths, batch_size=batch_size, convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self._cache_size > 0:
            for p, e in zip(paths, embs):
                if len(self._path_cache) < self._cache_size:
                    self._path_cache[p] = e
        return embs

    def score_path(self, path_str: str, query_emb: np.ndarray) -> float:
        """Cosine similarity between a single path and a pre-computed query embedding."""
        path_emb = self.encode_path(path_str)
        return float(
            np.dot(path_emb, query_emb)
            / (np.linalg.norm(path_emb) * np.linalg.norm(query_emb))
        )

    def score(self, path_str: str, question: str, partial_gen: str = "") -> float:
        query_emb = self.encode_query(question, partial_gen)
        return self.score_path(path_str, query_emb)

    def score_batch(self, path_strs, question: str, partial_gen: str = "",
                    batch_size=64) -> np.ndarray:
        """
        Score multiple paths against a question in batch.

        Returns np.ndarray of cosine similarity scores, shape (num_paths,).
        """
        query_emb = self.encode_query(question, partial_gen)
        path_embs = self.encode_paths_batch(path_strs, batch_size=batch_size)
        query_norm = np.linalg.norm(query_emb)
        return np.dot(path_embs, query_emb) / (
            np.linalg.norm(path_embs, axis=1) * query_norm
        )

    def clear_cache(self):
        """Clear the internal path embedding cache."""
        self._path_cache.clear()
