"""
DCA-Trie v1: Static Semantic Filtering at Trie Construction Time.

Wraps GCR's DFS-based path enumeration with semantic relevance scoring.
Only paths above threshold τ are admitted into the KG-Trie.

Key difference from GCR:
  GCR:     trie = MarisaTrie(tokenize(dfs(graph, q_entity, max_len)))
  DCA-Trie v1:  trie = MarisaTrie(tokenize(filter_score(dfs(graph, q_entity, max_len), question)))

Integration point: Replaces GraphConstrainedPromptBuilder.get_graph_index()
in the Step 1 (predict_paths_and_answers.py) pipeline.
"""

from typing import List, Tuple, Optional
import numpy as np
from gcr.src.trie import MarisaTrie
from gcr.src.utils.graph_utils import build_graph, dfs
import gcr.src.utils as _gcr_utils
from dca_trie.semantic_scorer import SemanticScorer


class V1TrieBuilder:
    """
    Builds a semantically filtered KG-Trie.

    Usage:
        builder = V1TrieBuilder(tokenizer, scorer, tau=0.3)
        trie = builder.build_filtered_trie(question_dict)

    The resulting trie contains only paths whose semantic similarity
    to the question is >= tau.
    """

    def __init__(
        self,
        tokenizer,
        scorer: SemanticScorer,
        tau: float = 0.3,
        index_path_length: int = 2,
        undirected: bool = False,
    ):
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.tau = tau
        self.index_path_length = index_path_length
        self.undirected = undirected

    def build_filtered_trie(self, question_dict):
        """
        Main entry point: build a MarisaTrie with only semantically relevant paths.

        Args:
            question_dict: dict with keys 'question', 'q_entity', 'graph'.
                           Compatible with HuggingFace WebQSP/CWQ format.

        Returns:
            MarisaTrie containing only filtered paths, or None if no paths remain.
        """
        # Step 1: Get all structural paths (same as GCR)
        all_paths = self._enumerate_paths(question_dict)
        if not all_paths:
            return None

        # Step 2: Encode question once
        query_emb = self.scorer.encode_query(question_dict["question"])

        # Step 3: Score and filter
        filtered_strs = []
        for p in all_paths:
            path_str = _gcr_utils.path_to_string(p)
            score = self.scorer.score_path(path_str, query_emb)
            if score >= self.tau:
                filtered_strs.append(path_str)

        if not filtered_strs:
            return None

        # Step 4: Tokenize and build trie
        tokenized = self.tokenizer(
            filtered_strs, padding=False, add_special_tokens=False
        ).input_ids
        tokenized = [ids + [self.tokenizer.eos_token_id] for ids in tokenized]

        return MarisaTrie(tokenized, max_token_id=len(self.tokenizer) + 1)

    def build_filtered_trie_with_scores(self, question_dict):
        """
        Like build_filtered_trie, but returns (trie, scores) for analysis.
        scores is a list of (path_str, score) for all paths (both kept and pruned).
        """
        all_paths = self._enumerate_paths(question_dict)
        if not all_paths:
            return None, []

        query_emb = self.scorer.encode_query(question_dict["question"])

        scores = []
        filtered_strs = []
        for p in all_paths:
            path_str = _gcr_utils.path_to_string(p)
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
        """Same path enumeration as GCR's get_graph_index()."""
        if "paths" in question_dict:
            return question_dict["paths"]

        g = build_graph(question_dict["graph"], self.undirected)
        return dfs(g, question_dict["q_entity"], self.index_path_length)

    def filter_paths_only(self, question_dict):
        """
        Return filtered path strings without building a trie.
        Useful for threshold sweep FNR computation.
        """
        all_paths = self._enumerate_paths(question_dict)
        if not all_paths:
            return []

        query_emb = self.scorer.encode_query(question_dict["question"])

        filtered = []
        for p in all_paths:
            path_str = _gcr_utils.path_to_string(p)
            score = self.scorer.score_path(path_str, query_emb)
            if score >= self.tau:
                filtered.append(path_str)

        return filtered
