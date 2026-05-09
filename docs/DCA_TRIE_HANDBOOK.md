# DCA-Trie Implementation Handbook

> **🆕 CONTRIBUTION:** This document describes the DCA-Trie implementation,
> which builds on the GCR baseline (in `gcr/`). All code referenced with
> `gcr/` prefix is baseline; code in `dca_trie/` is contribution.

This document explains **exactly** how to implement DCA-Trie on top of the GCR codebase. It identifies the extension points in GCR and shows how your code plugs into them.

---

## Architecture Overview

```bash
┌─────────────────────────────────────────────────────────────┐
│                     DCA-Trie Code                           │
│                                                             │
│  semantic_scorer.py     sir_measurement.py                  │
│  v1_trie_builder.py     v2_decoder.py                       │
│  evaluation.py          prototype.py                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Extends / Wraps                         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      GCR Codebase                           │
│                                                             │
│  src/trie.py           src/graph_constrained_decoding.py    │
│  src/qa_prompt_builder.py   src/llms/                       │
│  workflow/predict_paths_and_answers.py                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Extension Point 1: Trie Construction (`get_graph_index`)

**Location:** `gcr/src/qa_prompt_builder.py`, method `GraphConstrainedPromptBuilder.get_graph_index()`

**What GCR does:**

```python
def get_graph_index(self, question_dict):
    paths_list = dfs(g, question_dict["q_entity"], self.index_path_length)
    paths_list_str = [path_to_string(p) for p in paths_list]
    tokenized_paths = self.tokenizer(paths_list_str, ...)
    return MarisaTrie(tokenized_paths, ...)
```

**DCA-Trie v1 hook:** Filter `paths_list` before building the trie.

```python
# In v1_trie_builder.py:
def build_filtered_trie(self, question_dict, tau):
    paths_list = dfs(g, question_dict["q_entity"], self.index_path_length)
    query_emb = self.scorer.encode_query(question_dict["question"])
    
    filtered_paths = []
    for p in paths_list:
        path_str = path_to_string(p)
        path_emb = self.scorer.encode_path(path_str)
        score = cosine_similarity(query_emb, path_emb)
        if score >= tau:
            filtered_paths.append(p)
    
    # Only paths above threshold go into the trie
    paths_list_str = [path_to_string(p) for p in filtered_paths]
    tokenized_paths = self.tokenizer(paths_list_str, ...)
    return MarisaTrie(tokenized_paths, ...)
```

**Where to intercept:** You have two options:

1. **Subclass `GraphConstrainedPromptBuilder`** and override `get_graph_index()`
2. **Wrap the builder** in `predict_paths_and_answers.py` — call your `build_filtered_trie()` instead of `process_input()`

---

## Extension Point 2: The `allowed_tokens_fn` Hook

**Location:** `gcr/src/graph_constrained_decoding.py`, method `GraphConstrainedDecoding.allowed_tokens_fn()`

**What GCR does:**

```python
def allowed_tokens_fn(self, batch_id, sent):
    if constrained_flag:
        allow_tokens = self.trie.get(sent.tolist()[L_input:])
    return allow_tokens
```

**What it receives:**

- `batch_id`: which beam in the beam search (0 to K-1)
- `sent`: the full token sequence generated so far (input + output)

**What it must return:** A list of valid token IDs for the next step.

---

## Extension Point 3: The `generate_sentence` Method

**Location:** `gcr/src/llms/graph_constrained_decoding_model.py`

```python
def generate_sentence(self, llm_input, trie, start_token_ids, end_token_ids, enable_constrained_by_default):
    gcr = GraphConstrainedDecoding(self.tokenizer, trie, ...)
    res = self.model.generate(
        prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
        ...
    )
```

**DCA-Trie v2 hook:** Replace the single-trie `GraphConstrainedDecoding` with a per-beam multi-trie version.

---

## Implementation: DCA-Trie v1 (Static Filtering)

### File: `dca_trie/semantic_scorer.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

class SemanticScorer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.tau_ref = 0.3  # default threshold
    
    @lru_cache(maxsize=10000)
    def encode_path(self, path_str: str) -> np.ndarray:
        """Cache path embeddings for repeated queries."""
        return self.model.encode(path_str, convert_to_numpy=True)
    
    def encode_query(self, question: str, partial_gen: str = "") -> np.ndarray:
        """Encode question + optional partial generation."""
        text = question + " " + partial_gen if partial_gen else question
        return self.model.encode(text, convert_to_numpy=True)
    
    def score(self, path_str: str, query_emb: np.ndarray) -> float:
        """Cosine similarity between path and query."""
        path_emb = self.encode_path(path_str)
        return np.dot(path_emb, query_emb) / (
            np.linalg.norm(path_emb) * np.linalg.norm(query_emb)
        )
```

### File: `dca_trie/v1_trie_builder.py`

The key insight: v1 operates at trie construction time, not during decoding.

```python
class V1TrieBuilder:
    def __init__(self, tokenizer, scorer: SemanticScorer, tau: float):
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.tau = tau
    
    def build_filtered_trie(self, question_dict):
        """Build a MarisaTrie with only semantically relevant paths."""
        # Step 1: Get all structural paths (same as GCR)
        g = build_graph(question_dict["graph"])
        all_paths = dfs(g, question_dict["q_entity"], max_length=2)
        
        # Step 2: Encode question once
        query_emb = self.scorer.encode_query(question_dict["question"])
        
        # Step 3: Filter paths
        filtered_paths = []
        for p in all_paths:
            path_str = path_to_string(p)
            score = self.scorer.score(path_str, query_emb)
            if score >= self.tau:
                filtered_paths.append(p)
        
        # Step 4: Build trie from filtered paths (same as GCR)
        paths_str = [path_to_string(p) for p in filtered_paths]
        tokenized = self.tokenizer(paths_str, padding=False, add_special_tokens=False).input_ids
        tokenized = [ids + [self.tokenizer.eos_token_id] for ids in tokenized]
        return MarisaTrie(tokenized, max_token_id=len(self.tokenizer) + 1)
```

### Integration into Step 1 Pipeline

In `predict_paths_and_answers.py`, you intercept the trie construction:

```python
# Original GCR:
input_query, ground_paths, trie = input_builder.process_input(data)

# DCA-Trie v1:
input_query, ground_paths, _ = input_builder.process_input(data)
trie = v1_builder.build_filtered_trie(data)

# Rest is identical:
prediction = model.generate_sentence(input, trie, ...)
```

### Threshold Sweep Procedure (Phase 2)

```python
def sweep_threshold(dataset, scorer):
    for tau in np.arange(0.1, 0.65, 0.05):
        false_negatives = 0
        total_sizes = []
        
        for data in dataset:
            builder = V1TrieBuilder(tokenizer, scorer, tau)
            filtered_paths = builder.filter_paths(data)
            
            # Check FNR: does any gold path survive filtering?
            gold_paths = data["ground_truth_paths"]
            survives = any(
                scorer.score(gp, query_emb) >= tau 
                for gp in gold_paths
            )
            if not survives:
                false_negatives += 1
            
            total_sizes.append(len(filtered_paths))
        
        fnr = false_negatives / len(dataset)
        avg_size = np.mean(total_sizes)
        print(f"tau={tau:.2f}: FNR={fnr:.3f}, avg_trie_size={avg_size:.0f}")
```

Select the highest `tau` where `FNR < 0.05`.

---

## Implementation: DCA-Trie v2 (Dynamic Expansion)

### The Core Challenge

v2 must maintain **one trie per beam**. When beam search forks/copies a hypothesis, the trie must be copied too. When an entity is committed, the trie must be expanded with the entity's neighbors (filtered by semantic score).

### File: `dca_trie/v2_decoder.py`

```python
from dataclasses import dataclass, field
from src.trie import MarisaTrie
from src.graph_constrained_decoding import GraphConstrainedDecoding
import copy

@dataclass
class BeamTrieState:
    """Holds one trie per beam."""
    trie: MarisaTrie
    last_entity: str = None

class V2Decoder:
    def __init__(self, model, tokenizer, scorer: SemanticScorer, tau: float, k: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.tau = tau
        self.k = k
    
    def generate(self, input_query, initial_trie, question_dict, 
                 start_token_ids, end_token_ids):
        """
        DCA-Trie v2: Step-wise dynamic expansion.
        
        The tricky part: HuggingFace's generate() with prefix_allowed_tokens_fn
        supports a single function. We need to maintain per-beam trie state.
        
        Strategy: Use a wrapper function that tracks beam states internally.
        """
        # Initialize per-beam states (all start with the same trie)
        beam_states = [
            BeamTrieState(trie=copy.deepcopy(initial_trie))
            for _ in range(self.k)
        ]
        
        # Build the graph for neighbor lookups
        graph = build_graph(question_dict["graph"])
        
        # We need to track which beam we're in. HuggingFace passes batch_id
        # to prefix_allowed_tokens_fn. We'll use a closure that captures
        # the beam_states list.
        
        def allowed_tokens_fn(batch_id, sent):
            # Basic constrained decoding logic first
            constrained_flag, L_input = check_constrained_flag(sent, start_token_ids, end_token_ids)
            
            if not constrained_flag:
                return list(range(len(self.tokenizer)))
            
            # Get current beam's trie
            current_state = beam_states[batch_id]
            
            # Look up valid tokens from the trie
            path_prefix = sent.tolist()[L_input:]
            valid_tokens = current_state.trie.get(path_prefix)
            
            # Check if we just committed an entity (entity boundary detected)
            # This depends on the tokenization pattern — study GCR's output format
            newly_committed = self.detect_entity_commit(sent, path_prefix)
            if newly_committed:
                entity = newly_committed
                # Expand trie with scored neighbors
                self.expand_trie(current_state, entity, 
                               question_dict["question"], 
                               partial_gen=path_prefix,
                               graph=graph)
            
            return valid_tokens if valid_tokens else list(range(len(self.tokenizer)))
        
        # Now run generation with our per-beam-aware function
        return self.run_beam_search(input_query, allowed_tokens_fn)
    
    def expand_trie(self, state, entity, question, partial_gen, graph):
        """Add semantically relevant neighbors of entity to the trie."""
        query_emb = self.scorer.encode_query(question, partial_gen)
        
        new_paths = []
        for neighbor in graph.neighbors(entity):
            rel = graph[entity][neighbor]["relation"]
            path_str = f"{entity} -> {rel} -> {neighbor}"
            
            score = self.scorer.score(path_str, query_emb)
            if score >= self.tau:
                # Tokenize and add to trie
                tokenized = self.tokenizer.encode(path_str)
                new_paths.append(tokenized)
        
        # Add new paths to existing trie
        # NOTE: MarisaTrie is IMMUTABLE! This is a key challenge.
        # You need to build a new trie combining old + new paths.
        # Or use the simple Trie class instead of MarisaTrie for dynamic use.
        combined = self.merge_tries(state.trie, new_paths)
        state.trie = combined
    
    def detect_entity_commit(self, sent, path_prefix):
        """
        Detect when the model has finished generating an entity name.
        
        In GCR's output format, entities are separated by ' -> ' (tokenized).
        An entity is "committed" when the model generates ' -> ' after a span.
        
        This is implementation-specific — study GCR's actual output format
        to determine the exact boundary tokens.
        """
        # Decode the path prefix to text
        text = self.tokenizer.decode(path_prefix)
        
        # Check if the latest tokens complete an entity
        # Heuristic: entity is committed when we see ' -> ' or end of path
        # (This needs refinement based on actual GCR output patterns)
        if " -> " in text or text.endswith(">"):
            parts = text.split(" -> ")
            if len(parts) >= 2:
                return parts[-1]  # return the newly committed entity
        return None
    
    def run_beam_search(self, input_query, allowed_tokens_fn):
        """Wrapper around model.generate() with custom prefix_allowed_tokens_fn."""
        inputs = self.tokenizer(input_query, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        res = self.model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.model.generation_cfg,
            prefix_allowed_tokens_fn=allowed_tokens_fn,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return [
            self.tokenizer.decode(r[input_ids.shape[1]:], skip_special_tokens=True)
            for r in res.sequences
        ]
    
    def merge_tries(self, old_trie, new_sequences):
        """Build a new MarisaTrie combining old paths and new sequences."""
        # Collect all existing paths from old trie
        all_seqs = list(old_trie)  # MarisaTrie.__iter__ yields paths
        all_seqs.extend(new_sequences)
        return MarisaTrie(all_seqs, max_token_id=len(self.tokenizer) + 1)
```

### The Beam State Copy Problem

**Critical:** When HuggingFace's beam search forks a hypothesis (keeps top K out of K×V candidates), beams can be reordered or duplicated. You need to ensure the trie follows its beam.

HuggingFace calls `prefix_allowed_tokens_fn(batch_id, sent)` where `batch_id` is the beam index in the current step. However, between steps, HuggingFace internally reorders beams. This means you can't rely on `batch_id` being stable across steps.

**Solution:** Track state by hypothesis content instead of beam index:

```python
class TrieStateManager:
    def __init__(self, initial_trie, k):
        self.states = {}  # keyed by tuple of token IDs
        self.k = k
    
    def get_state(self, sent):
        key = tuple(sent.tolist())
        if key not in self.states:
            # New hypothesis needs a copy of its parent's state
            # Find parent by removing last token
            parent_key = key[:-1]
            if parent_key in self.states:
                parent_state = self.states[parent_key]
                self.states[key] = BeamTrieState(trie=copy.deepcopy(parent_state.trie))
            else:
                self.states[key] = BeamTrieState(trie=copy.deepcopy(self.initial_trie))
        return self.states[key]
```

---

## Implementation: SIR Measurement

### File: `dca_trie/sir_measurement.py`

```python
class SIRMeasurer:
    def __init__(self, scorer: SemanticScorer):
        self.scorer = scorer
    
    def measure(self, trie, question, partial_gen=""):
        """
        Compute SIR at a given generation step.
        
        SIR(q, t) = 1 - max_{p in P_valid(t)} cos(E(p), E(q, y_<t))
        """
        query_emb = self.scorer.encode_query(question, partial_gen)
        
        # Iterate all paths in the trie
        max_similarity = -1.0
        total_paths = 0
        
        for path_seq in trie:
            total_paths += 1
            path_str = self.tokenizer.decode(path_seq)
            sim = self.scorer.score(path_str, query_emb)
            max_similarity = max(max_similarity, sim)
        
        if total_paths == 0:
            return 0.0
        
        return 1.0 - max_similarity
    
    def measure_per_hop(self, trie, question, hop_depths):
        """
        Measure SIR stratified by hop depth.
        Hop depth = number of relation steps in the path.
        """
        query_emb = self.scorer.encode_query(question)
        results = {}
        
        for hop in hop_depths:
            paths_at_hop = self.filter_by_hop(trie, hop)
            if not paths_at_hop:
                results[hop] = {"sir": 0.0, "count": 0}
                continue
            
            similarities = []
            for p in paths_at_hop:
                sim = self.scorer.score(p, query_emb)
                similarities.append(sim)
            
            results[hop] = {
                "sir": 1.0 - max(similarities),
                "avg_sim": sum(similarities) / len(similarities),
                "count": len(paths_at_hop),
            }
        
        return results
```

---

## File Structure for DCA-Trie

```
dca_trie/
├── __init__.py
├── semantic_scorer.py       # MiniLM encoder + cosine scoring + caching
├── sir_measurement.py       # SIR metric computation
├── v1_trie_builder.py       # Static filtering at trie construction
├── v2_decoder.py            # Beam search with per-beam trie state
├── evaluation.py            # Hits@1, F1, faithfulness, SIR, trie size
└── prototype.py             # Gradio demo

experiments/
├── reproduce_gcr.py         # Phase 0
├── measure_sir_gcr.py       # Phase 1
├── threshold_sweep_v1.py    # Phase 2
└── run_eval.py              # Phase 4

data/
├── webqsp_val_100.json      # Threshold calibration split
└── question_bank.json       # Pre-computed demo examples
```

---

## Key GCR Internals Summary

| Component | File | What It Does |
|-----------|------|-------------|
| `MarisaTrie` | `gcr/src/trie.py` | Immutable prefix tree for token ID sequences |
| `GraphConstrainedDecoding` | `gcr/src/graph_constrained_decoding.py` | Connects trie to HF generate() via `prefix_allowed_tokens_fn` |
| `GraphConstrainedDecodingModel` | `gcr/src/llms/graph_constrained_decoding_model.py` | Model wrapper that calls generate() with constraint |
| `get_graph_index()` | `gcr/src/qa_prompt_builder.py` | Builds trie from graph DFS paths |
| `dfs()` | `gcr/src/utils/graph_utils.py` | Enumerates all paths up to max_length |
| `eval_path_result_w_ans()` | `gcr/src/utils/qa_utils.py` | Evaluates Hits@1, F1, path accuracy |
