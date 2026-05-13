"""
DCA-Trie v2: Step-wise Dynamic Trie Expansion During Beam Search.

Extends GCR's constrained decoding by semantically expanding the KG-Trie
at each reasoning step, per beam hypothesis.

Key design:
  - Uses mutable ``Trie`` (dict-based) for per-beam dynamic tries
    (MarisaTrie is immutable and cannot be expanded mid-generation).
  - ``TrieStateManager`` tracks trie state by hypothesis content
    (tuple of token IDs), not beam index — handles HuggingFace's
    beam reordering correctly.
  - Entity boundary detection uses token-level matching of the
    `` -> `` separator pattern.

Usage:
    decoder = V2Decoder(model, tokenizer, scorer, tau=0.3)
    outputs = decoder.generate(input_query, initial_trie, question_dict,
                               start_token_ids, end_token_ids)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch
import copy
import numpy as np

from gcr.src.trie import Trie
from gcr.src.utils.graph_utils import build_graph
from dca_trie.semantic_scorer import SemanticScorer


@dataclass
class BeamTrieState:
    """Mutable per-beam trie state for dynamic expansion."""
    trie: Trie
    last_entity: Optional[str] = None
    committed_hops: int = 0


class TrieStateManager:
    """
    Manages per-beam trie states keyed by hypothesis content.

    HuggingFace's beam search reorders beams between steps, so indexing
    by `batch_id` is unreliable. Instead we key by the full token sequence
    (hypothesis content), which is deterministic.
    """

    def __init__(self, initial_trie: Trie, k: int):
        self.initial_trie = initial_trie
        self.k = k
        self._states = {}

    def get_state(self, sent) -> BeamTrieState:
        """Get (or create) the trie state for a given hypothesis."""
        key = tuple(sent.tolist())
        if key in self._states:
            return self._states[key]

        # New hypothesis: copy parent's state if we can find it
        parent = list(key[:-1]) if len(key) > 1 else []
        parent_key = tuple(parent)
        if parent_key in self._states:
            parent_state = self._states[parent_key]
            state = BeamTrieState(
                trie=copy.deepcopy(parent_state.trie),
                last_entity=parent_state.last_entity,
                committed_hops=parent_state.committed_hops,
            )
        else:
            state = BeamTrieState(trie=copy.deepcopy(self.initial_trie))

        self._states[key] = state
        return state

    def cleanup_old_states(self, current_keys):
        """Remove states for hypotheses no longer in the beam."""
        keep = set(tuple(k) for k in current_keys)
        drop = [k for k in self._states if k not in keep]
        for k in drop:
            del self._states[k]

    def reset(self):
        self._states = {}


class V2Decoder:
    """
    DCA-Trie v2 decoder with step-wise dynamic trie expansion.

    Wraps HuggingFace's ``model.generate()`` with a ``prefix_allowed_tokens_fn``
    that maintains per-beam trie state and expands the trie when entity
    boundaries are detected.
    """

    def __init__(
        self,
        model,
        tokenizer,
        scorer: SemanticScorer,
        tau: float = 0.3,
        k: int = 5,
        path_to_str_fn: Optional[Callable] = None,
        separator: str = " -> ",
        max_hops: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.tau = tau
        self.k = k
        self._path_to_str = path_to_str_fn or (lambda x: x)
        self.separator = separator
        self.max_hops = max_hops
        self.separator_token_ids = self._encode_separator(separator)

    def _encode_separator(self, sep: str) -> List[int]:
        """Tokenize the separator string (e.g. ' -> ') for pattern matching."""
        return self.tokenizer.encode(sep, add_special_tokens=False)

    def _count_separators(self, sent_ids: List[int], input_len: int) -> int:
        """
        Count occurrences of the separator token pattern in the generated prefix.

        Uses sliding window matching on the token IDs to avoid fragile
        string-based heuristics.
        """
        gen = sent_ids[input_len:]
        sep = self.separator_token_ids
        count = 0
        for i in range(len(gen) - len(sep) + 1):
            if gen[i:i + len(sep)] == sep:
                count += 1
        return count

    def detect_entity_commit(self, sent, input_len: int,
                              current_state: BeamTrieState) -> Optional[str]:
        """
        Detect when an entity boundary is crossed.

        Returns the newly committed entity name if detected, else None.

        Strategy: count `` -> `` separators in the generated prefix.
        If the count increased, an entity just got committed.
        """
        new_count = self._count_separators(sent.tolist(), input_len)
        if new_count > current_state.committed_hops:
            current_state.committed_hops = new_count
            text = self.tokenizer.decode(sent[input_len:].tolist(), skip_special_tokens=True)
            parts = text.split(self.separator)
            if len(parts) >= 2:
                return parts[-2].strip()
        return None

    def _build_allowed_tokens_fn(self, input_len: int, graph,
                                  question_dict, state_manager: TrieStateManager,
                                  start_token_ids: List[int],
                                  end_token_ids: List[int],
                                  constrained_flag_fn: Callable):
        """Build the ``prefix_allowed_tokens_fn`` closure."""

        def allowed_tokens_fn(batch_id, sent):
            # Basic constrained decoding check
            constrained, L_input = constrained_flag_fn(sent, start_token_ids, end_token_ids)
            if not constrained:
                return list(range(len(self.tokenizer)))

            state = state_manager.get_state(sent)

            path_prefix = sent.tolist()[L_input:]
            valid_tokens = state.trie.get(path_prefix)

            # Check for entity boundary
            entity = self.detect_entity_commit(sent, input_len, state)
            if entity is not None:
                self.expand_trie(state, entity, question_dict, path_prefix, graph)

            return valid_tokens if valid_tokens else list(range(len(self.tokenizer)))

        return allowed_tokens_fn

    def expand_trie(self, state: BeamTrieState, entity: str,
                     question_dict, partial_gen,
                     graph):
        """
        Expand the trie with semantically relevant neighbors of the
        just-committed entity.

        Only paths whose semantic similarity to the question >= tau
        are added.
        """
        if not graph.has_node(entity):
            return

        query_emb = self.scorer.encode_query(
            question_dict["question"],
            self.tokenizer.decode(partial_gen, skip_special_tokens=True),
        )

        new_paths = []
        for neighbor in graph.successors(entity):
            rel = graph[entity][neighbor].get("relation", "")
            path_str = self._path_to_str(f"{entity}{self.separator}{rel}{self.separator}{neighbor}")

            score = self.scorer.score_path(path_str, query_emb)
            if score >= self.tau:
                tokenized = self.tokenizer.encode(path_str, add_special_tokens=False)
                new_paths.append(tokenized)

        if new_paths:
            for seq in new_paths:
                state.trie.add(seq)

    @torch.inference_mode()
    def generate(self, input_query: str, initial_trie: Trie,
                  question_dict, start_token_ids, end_token_ids,
                  constrained_flag_fn: Optional[Callable] = None,
                  **gen_kwargs):
        """
        Run beam search with DCA-Trie v2 dynamic expansion.

        Args:
            input_query: Formatted prompt string.
            initial_trie: Initial KG-Trie (all structurally valid paths).
            question_dict: Question dict with 'question', 'graph', 'q_entity'.
            start_token_ids: Token IDs for the constrained region start.
            end_token_ids: Token IDs for the constrained region end.
            constrained_flag_fn: Callable returning (is_constrained, L_input).
                                Defaults to GCR's ``check_constrained_flag``.
            **gen_kwargs: Passed to ``model.generate()`` (e.g., num_beams).

        Returns:
            List of decoded output strings.
        """
        if constrained_flag_fn is None:
            from gcr.src.graph_constrained_decoding import check_constrained_flag
            constrained_flag_fn = check_constrained_flag

        graph = build_graph(question_dict["graph"])

        state_manager = TrieStateManager(
            initial_trie=copy.deepcopy(initial_trie),
            k=self.k,
        )

        inputs = self.tokenizer(input_query, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.model.device)
        attention_mask = inputs.attention_mask.to(self.model.model.device)
        input_len = input_ids.shape[1]

        allowed_tokens_fn = self._build_allowed_tokens_fn(
            input_len=input_len,
            graph=graph,
            question_dict=question_dict,
            state_manager=state_manager,
            start_token_ids=start_token_ids,
            end_token_ids=end_token_ids,
            constrained_flag_fn=constrained_flag_fn,
        )

        default_gen_kwargs = dict(
            num_beams=self.k,
            num_beam_groups=1,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        default_gen_kwargs.update(gen_kwargs)

        res = self.model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.model.generation_cfg,
            prefix_allowed_tokens_fn=allowed_tokens_fn,
            **default_gen_kwargs,
        )

        return [
            self.tokenizer.decode(r[input_len:], skip_special_tokens=True)
            for r in res.sequences
        ]
