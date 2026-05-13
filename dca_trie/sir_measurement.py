"""
SIR (Semantic Irrelevance Ratio) Measurement.

SIR(q, t) = 1 - max_{p in P_valid(t)} cos(E(p), E(q, y_<t))

Where:
  - P_valid(t): set of valid paths in the trie at step t
  - E: sentence transformer encoder (384-dim embeddings)
  - cos: cosine similarity

SIR = 0: all admitted paths are relevant
SIR = 1: no admitted path is relevant
SIR = None: no paths to measure (empty trie)
"""

import numpy as np
from collections import defaultdict
from dca_trie.semantic_scorer import SemanticScorer


class SIRMeasurer:
    """
    Measures oracle permissiveness via SIR with batch-encoding support.

    Usage:
        scorer = SemanticScorer()
        measurer = SIRMeasurer(scorer)
        result = measurer.measure_from_trie(trie, "question text")
    """

    def __init__(self, scorer: SemanticScorer, tokenizer=None):
        self.scorer = scorer
        self.tokenizer = tokenizer

    def _resolve_paths(self, trie):
        """Convert trie/token sequences to path strings."""
        paths = []
        for token_seq in trie:
            if isinstance(token_seq, (list, tuple)) and self.tokenizer:
                path_str = self.tokenizer.decode(token_seq, skip_special_tokens=True)
            else:
                path_str = str(token_seq)
            paths.append(path_str)
        return paths

    def measure_from_trie(self, trie, question: str, partial_gen: str = "",
                          batch_size=64):
        """
        Compute SIR using batch encoding for speed.

        Args:
            trie: Iterable of paths (strings or token sequences)
            question: Input question
            partial_gen: Partial generation context
            batch_size: Batch size for SentenceTransformer encoding

        Returns:
            dict with sir, max_similarity, avg_similarity, num_paths
            sir is None if the trie is empty
        """
        path_strs = self._resolve_paths(trie)
        num_paths = len(path_strs)

        if num_paths == 0:
            return {
                "sir": None,
                "max_similarity": None,
                "avg_similarity": None,
                "num_paths": 0,
            }

        query_emb = self.scorer.encode_query(question, partial_gen)
        path_embs = self.scorer.encode_paths_batch(path_strs, batch_size=batch_size)
        query_norm = np.linalg.norm(query_emb)

        similarities = np.dot(path_embs, query_emb) / (
            np.linalg.norm(path_embs, axis=1) * query_norm
        )

        return {
            "sir": 1.0 - float(np.max(similarities)),
            "max_similarity": float(np.max(similarities)),
            "avg_similarity": float(np.mean(similarities)),
            "num_paths": num_paths,
        }

    def measure_per_hop(self, trie, question: str, partial_gen: str = "",
                        hop_depths=None, batch_size=64):
        """
        Measure SIR stratified by hop depth using batch encoding.

        Args:
            trie: Iterable of paths
            question: Input question
            partial_gen: Partial generation context
            hop_depths: Depths to analyze (default [1, 2, 3, 4])
            batch_size: Batch size for SentenceTransformer encoding

        Returns:
            dict mapping hop_depth -> SIR stats
        """
        if hop_depths is None:
            hop_depths = [1, 2, 3, 4]
        hop_depths = set(hop_depths)

        query_emb = self.scorer.encode_query(question, partial_gen)

        # Collect all path strings with their hop depths
        hop_paths = defaultdict(list)
        for token_seq in trie:
            if isinstance(token_seq, (list, tuple)) and self.tokenizer:
                path_str = self.tokenizer.decode(token_seq, skip_special_tokens=True)
            else:
                path_str = str(token_seq)

            num_hops = path_str.count(" -> ")
            if num_hops in hop_depths:
                hop_paths[num_hops].append(path_str)

        # Batch-encode all paths at once (more efficient than per-hop batches)
        all_paths = []
        hop_indices = {}  # hop -> (start, end) into all_paths
        for hop in sorted(hop_paths.keys()):
            hop_indices[hop] = (len(all_paths), len(all_paths) + len(hop_paths[hop]))
            all_paths.extend(hop_paths[hop])

        if not all_paths:
            return {h: {"sir": None, "num_paths": 0, "max_similarity": None,
                        "avg_similarity": None} for h in hop_depths}

        path_embs = self.scorer.encode_paths_batch(all_paths, batch_size=batch_size)
        query_norm = np.linalg.norm(query_emb)

        results = {}
        for hop in hop_depths:
            paths = hop_paths.get(hop, [])
            if not paths:
                results[hop] = {
                    "sir": None,
                    "num_paths": 0,
                    "max_similarity": None,
                    "avg_similarity": None,
                }
                continue

            start, end = hop_indices[hop]
            hop_embs = path_embs[start:end]
            similarities = np.dot(hop_embs, query_emb) / (
                np.linalg.norm(hop_embs, axis=1) * query_norm
            )

            results[hop] = {
                "sir": 1.0 - float(np.max(similarities)),
                "num_paths": len(paths),
                "max_similarity": float(np.max(similarities)),
                "avg_similarity": float(np.mean(similarities)),
            }

        return results
