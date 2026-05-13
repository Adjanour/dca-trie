"""
DCA-Trie v1: Static Semantic Filtering at Trie Construction Time.

Wraps GCR's DFS-based path enumeration with semantic relevance scoring.
Only paths above threshold tau are admitted into the KG-Trie.

Key difference from GCR:
  GCR:     trie = MarisaTrie(tokenize(dfs(graph, q_entity, max_len)))
  DCA-Trie v1:  trie = MarisaTrie(tokenize(filter_score(dfs(graph, q_entity, max_len), question)))

Integration: Can be used standalone or patched into GCR's pipeline
via v1_gcr_integration.patch_prompt_builder().
"""

from typing import List, Optional, Callable
import numpy as np
from gcr.src.trie import MarisaTrie
from gcr.src.utils.graph_utils import build_graph, dfs
from gcr.src.utils import path_to_string as _default_path_to_str
from dca_trie.semantic_scorer import SemanticScorer


class V1TrieBuilder:
    """
    Builds a semantically filtered KG-Trie.

    Usage:
        builder = V1TrieBuilder(tokenizer, scorer, tau=0.3)
        trie = builder.build_filtered_trie(question_dict)

    The resulting trie contains only paths whose semantic similarity
    to the question is >= tau.

    For MID resolution, pass path_to_str_fn=resolver.resolve_path
    (wraps path_to_string with MID-to-readable-name conversion).
    """

    def __init__(
        self,
        tokenizer,
        scorer: SemanticScorer,
        tau: float = 0.3,
        index_path_length: int = 2,
        undirected: bool = False,
        path_to_str_fn: Optional[Callable] = None,
    ):
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.tau = tau
        self.index_path_length = index_path_length
        self.undirected = undirected
        self._path_to_str = path_to_str_fn or _default_path_to_str

    def build_filtered_trie(self, question_dict):
        all_paths = self._enumerate_paths(question_dict)
        if not all_paths:
            return None

        query_emb = self.scorer.encode_query(question_dict["question"])

        filtered_strs = []
        for p in all_paths:
            path_str = self._path_to_str(p)
            score = self.scorer.score_path(path_str, query_emb)
            if score >= self.tau:
                filtered_strs.append(path_str)

        if not filtered_strs:
            return None

        tokenized = self.tokenizer(
            filtered_strs, padding=False, add_special_tokens=False
        ).input_ids
        tokenized = [ids + [self.tokenizer.eos_token_id] for ids in tokenized]

        return MarisaTrie(tokenized, max_token_id=len(self.tokenizer) + 1)

    def build_filtered_trie_with_scores(self, question_dict):
        all_paths = self._enumerate_paths(question_dict)
        if not all_paths:
            return None, []

        query_emb = self.scorer.encode_query(question_dict["question"])

        scores = []
        filtered_strs = []
        for p in all_paths:
            path_str = self._path_to_str(p)
            score = self.scorer.score_path(path_str, query_emb)
            scores.append((path_str, float(score)))
            if score >= self.tau:
                filtered_strs.append(path_str)

        if not filtered_strs:
            return None, scores

        tokenized = self.tokenizer(
            filtered_strs, padding=False, add_special_tokens=False
        ).input_ids
        tokenized = [ids + [self.tokenizer.eos_token_id] for ids in tokenized]

        trie = MarisaTrie(tokenized, max_token_id=len(self.tokenizer) + 1)
        return trie, scores

    def _enumerate_paths(self, question_dict):
        if "paths" in question_dict:
            return question_dict["paths"]

        g = build_graph(question_dict["graph"], self.undirected)
        return dfs(g, question_dict["q_entity"], self.index_path_length)

    def filter_paths_only(self, question_dict):
        """Return filtered path strings without building a trie."""
        all_paths = self._enumerate_paths(question_dict)
        if not all_paths:
            return []

        query_emb = self.scorer.encode_query(question_dict["question"])

        filtered = []
        for p in all_paths:
            path_str = self._path_to_str(p)
            score = self.scorer.score_path(path_str, query_emb)
            if score >= self.tau:
                filtered.append(path_str)

        return filtered
