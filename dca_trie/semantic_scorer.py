"""
Semantic Scorer for DCA-Trie.

Encodes question-context and KG paths into 384-dim embeddings
using sentence-transformers/all-MiniLM-L6-v2 and computes
cosine similarity as a relevance score.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache


class SemanticScorer:
    """
    Lightweight semantic similarity scorer.

    Usage:
        scorer = SemanticScorer(device='cpu')
        score = scorer.score("entity -> relation -> entity", "question text")
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    @lru_cache(maxsize=50000)
    def encode_path(self, path_str: str) -> np.ndarray:
        """Encode a KG path string to 384-dim embedding. Results cached."""
        return self.model.encode(path_str, convert_to_numpy=True)

    def encode_query(self, question: str, partial_gen: str = "") -> np.ndarray:
        """Encode question + optional partial generation context."""
        text = question + " " + partial_gen if partial_gen else question
        return self.model.encode(text, convert_to_numpy=True)

    def score_path(self, path_str: str, query_emb: np.ndarray) -> float:
        """Cosine similarity between a path and a pre-computed query embedding."""
        path_emb = self.encode_path(path_str)
        return float(
            np.dot(path_emb, query_emb)
            / (np.linalg.norm(path_emb) * np.linalg.norm(query_emb))
        )

    def score(self, path_str: str, question: str, partial_gen: str = "") -> float:
        """Convenience: encode query and score in one call."""
        query_emb = self.encode_query(question, partial_gen)
        return self.score_path(path_str, query_emb)
