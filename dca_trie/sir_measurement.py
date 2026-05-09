"""
SIR (Semantic Irrelevance Ratio) Measurement.

SIR(q, t) = 1 - max_{p in P_valid(t)} cos(E(p), E(q, y_<t))

Where:
  - P_valid(t): set of valid paths in the trie at step t
  - E: sentence transformer encoder (384-dim embeddings)
  - cos: cosine similarity

SIR = 0: all admitted paths are relevant
SIR = 1: no admitted path is relevant
"""

import numpy as np
from collections import defaultdict
from dca_trie.semantic_scorer import SemanticScorer


class SIRMeasurer:
    """
    Measures oracle permissiveness via SIR.

    Usage:
        scorer = SemanticScorer()
        measurer = SIRMeasurer(scorer)
        result = measurer.measure_from_trie(trie, "question text")
    """

    def __init__(self, scorer: SemanticScorer, tokenizer=None):
        self.scorer = scorer
        self.tokenizer = tokenizer

    def measure_from_trie(self, trie, question: str, partial_gen: str = ""):
        """
        Compute SIR by iterating all paths in a trie.

        Args:
            trie: Iterable of paths (strings or token sequences)
            question: Input question
            partial_gen: Partial generation context

        Returns:
            dict with sir, max_similarity, avg_similarity, num_paths
        """
        query_emb = self.scorer.encode_query(question, partial_gen)

        max_sim = -1.0
        total_sim = 0.0
        num_paths = 0

        for token_seq in trie:
            num_paths += 1
            if isinstance(token_seq, (list, tuple)) and self.tokenizer:
                path_str = self.tokenizer.decode(token_seq, skip_special_tokens=True)
            else:
                path_str = str(token_seq)

            path_emb = self.scorer.encode_path(path_str)
            sim = float(
                np.dot(path_emb, query_emb)
                / (np.linalg.norm(path_emb) * np.linalg.norm(query_emb))
            )
            max_sim = max(max_sim, sim)
            total_sim += sim

        if num_paths == 0:
            return {
                "sir": 0.0,
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "num_paths": 0,
            }

        return {
            "sir": 1.0 - max_sim,
            "max_similarity": max_sim,
            "avg_similarity": total_sim / num_paths,
            "num_paths": num_paths,
        }

    def measure_per_hop(self, trie, question: str, hop_depths=None):
        """
        Measure SIR stratified by hop depth.

        Args:
            trie: Iterable of paths
            question: Input question
            hop_depths: Depths to analyze (default [1, 2, 3, 4])

        Returns:
            dict mapping hop_depth -> SIR stats
        """
        if hop_depths is None:
            hop_depths = [1, 2, 3, 4]

        query_emb = self.scorer.encode_query(question)
        hop_paths = defaultdict(list)

        for token_seq in trie:
            if isinstance(token_seq, (list, tuple)) and self.tokenizer:
                path_str = self.tokenizer.decode(token_seq, skip_special_tokens=True)
            else:
                path_str = str(token_seq)

            num_hops = path_str.count(" -> ")
            if num_hops in hop_depths:
                hop_paths[num_hops].append(path_str)

        results = {}
        for hop in hop_depths:
            paths = hop_paths.get(hop, [])
            if not paths:
                results[hop] = {
                    "sir": 0.0,
                    "num_paths": 0,
                    "max_similarity": 0.0,
                    "avg_similarity": 0.0,
                }
                continue

            similarities = []
            for p in paths:
                path_emb = self.scorer.encode_path(p)
                sim = float(
                    np.dot(path_emb, query_emb)
                    / (np.linalg.norm(path_emb) * np.linalg.norm(query_emb))
                )
                similarities.append(sim)

            results[hop] = {
                "sir": 1.0 - max(similarities),
                "num_paths": len(paths),
                "max_similarity": max(similarities),
                "avg_similarity": sum(similarities) / len(similarities),
            }

        return results
