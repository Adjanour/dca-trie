# DCA-Trie v3: Semantic Backtracking for KG-Constrained Decoding

## The Problem DCA-Trie v2 Does Not Solve

DCA-Trie v2 makes the constraint oracle **semantically aware**: at each entity boundary, it scores candidate neighbors and only admits paths relevant to the question. But it still has a structural limitation shared with GCR — **forward monotonicity**.

Once the model generates a token sequence committing to entity `e_t`, that decision is irreversible. If `e_t` turns out to be a dead end (no outgoing paths with semantic score ≥ τ), the beam dies. In GCR, this falls back to the full vocabulary — losing the faithfulness guarantee. In DCA-Trie v2, it means the beam produced no useful answer.

### Concrete Example

Question: *"Who is the spouse of the ex-president of the USA?"*

Step 1: Model starts from `"USA"` and must pick a relation.

Valid relations in the trie (scored by MiniLM):
- `location.country.capital` → Washington D.C. (sim: 0.55)
- `location.country.president` → Barack Obama (sim: 0.72)
- `location.country.currency` → US Dollar (sim: 0.42)
- `people.person.place_of_birth` → various (sim: 0.38)

The model (correctly) picks `location.country.president` → Barack Obama.

Step 2: From Barack Obama, valid relations:
- `people.person.spouse` → Michelle Obama (sim: 0.81) ✓
- `people.person.children` → various (sim: 0.45)
- `people.person.profession` → various (sim: 0.33)

The model picks `people.person.spouse` → Michelle Obama. ✓ Correct answer.

**Now consider a harder case:**

Question: *"What film won the Oscar for Best Picture in 2020 directed by a South Korean director?"*

Step 1: Model starts from `"Oscar"` and picks `award.award.winner` → `Parasite`.

Step 2: From `Parasite`, the model needs to pick the right outgoing edge:
- `film.film.directed_by` → Bong Joon-ho (sim: 0.62) ✓
- `film.film.country` → South Korea (sim: 0.58)
- `film.film.language` → Korean (sim: 0.51)
- `film.film.cast` → various actors (sim: 0.44)

Suppose the model picks `film.film.country` → South Korea (score 0.58, above τ = 0.3). Now from `South Korea`, the model needs to get back to the film... but it can't. South Korea has thousands of outgoing edges, and the original entity `Parasite` isn't directly reachable from `South Korea` in one hop (the relation goes the other way). The beam is stuck.

**With backtracking:** The model detects that from `South Korea`, no paths lead toward an answer entity. It backtracks to `Parasite`, tries `film.film.directed_by` → Bong Joon-ho instead, and succeeds.

---

## The Core Idea: Trie State Checkpoints with Rollback

The innovation is a **checkpoint-restore mechanism** for the trie during beam search:

```
At each entity boundary (when a new entity is committed):

1. SNAPSHOT: Save the current trie state + beam score
2. EXPAND: Score neighbors, admit relevant ones into the trie
3. COMMIT: Generate the next tokens
4. MONITOR: Track whether the beam is making progress

If beam is in a dead end (all continuations scored < τ, or trie returns empty):
  5. ROLLBACK: Restore the last snapshot
  6. MASK: Block the path that led to the dead end
  7. RETRY: Try the next-best path from the checkpoint
```

### Why This Is Novel

| Property | GCR | DoG | DCA-Trie v2 | **DCA-Trie v3** |
|----------|-----|-----|-------------|-----------------|
| Structural faithfulness | ✓ | ✓ | ✓ | ✓ |
| Question-aware constraints | ✗ | ✗ | ✓ | ✓ |
| Generation-state feedback | ✗ | ✗ | ✓ | ✓ |
| **Non-monotonic recovery** | ✗ | ✗ | ✗ | **✓** |
| Theoretical grounding | — | — | — | Connects to CRANE, AdapTrack |

No existing KG-constrained decoding method supports non-monotonic recovery. DoG expands stepwise but never retreats. GCR's trie is static. DCA-Trie v2 scores are better but still commit forward.

---

## Algorithm Design

### 1. Checkpoint Data Structure

```python
@dataclass
class TrieCheckpoint:
    """Snapshot of beam state at an entity boundary."""
    prefix_tokens: List[int]        # token sequence up to this point
    prefix_text: str                 # decoded text (for entity boundary detection)
    current_entity: str              # the entity just committed
    trie: Trie                       # snapshot of the trie BEFORE expansion
    beam_score: float                # cumulative score at this point
    used_expansions: Set[str]        # which neighbor expansions were already tried
    backtrack_depth: int             # how many times we've backtracked so far

class CheckpointStack:
    """LIFO stack of trie checkpoints for backtracking."""
    def __init__(self, max_depth=3):
        self.stack = []
        self.max_depth = max_depth
    
    def push(self, checkpoint: TrieCheckpoint):
        self.stack.append(checkpoint)
    
    def pop(self) -> Optional[TrieCheckpoint]:
        if not self.stack:
            return None
        return self.stack.pop()
    
    def peek(self) -> Optional[TrieCheckpoint]:
        if not self.stack:
            return None
        return self.stack[-1]
```

### 2. Entity Boundary Detection

The critical question: when has the model "committed" to an entity?

In GCR's output format, paths look like:
```
<PATH> USA -> location.country.president -> Barack Obama -> people.person.spouse -> Michelle Obama </PATH>
```

An entity is committed when the token sequence completes an entity name followed by ` -> `. At this point, the model must pick a relation to traverse next. This is the natural checkpoint point.

```python
def detect_entity_boundary(prefix_tokens, tokenizer):
    """
    Detect if the model just committed an entity.
    Returns the entity text if yes, None otherwise.
    """
    text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
    
    # Look for pattern: "... entity_name ->" at the end
    # The " -> " separator indicates a relation follows
    # If the text ends with " -> ", the entity before it is committed
    if text.rstrip().endswith("->"):
        # Strip the trailing " ->"
        stripped = text.rstrip()[:-2].strip()
        parts = stripped.split(" -> ")
        if len(parts) >= 2 and len(parts) % 2 == 0:
            # Even number of parts means we just finished an entity
            # Pattern: entity -> rel -> entity -> rel -> entity (just committed)
            return parts[-1]  # the newly committed entity
    
    return None
```

### 3. Condition Detection: When to Backtrack

Three conditions trigger rollback:

```python
def should_backtrack(beam_state, scorer, question, graph):
    """
    Check if the current beam state warrants backtracking.
    Returns (should_backtrack: bool, reason: str)
    """
    # Condition 1: Dead end — trie returns empty
    valid_next = beam_state.trie.get(beam_state.prefix)
    if len(valid_next) == 0:
        return True, "dead_end"
    
    # Condition 2: Semantic dead end — all valid continuations score below τ
    scores = []
    for token_id in valid_next:
        hypothetical = beam_state.prefix + [token_id]
        path_text = tokenizer.decode(hypothetical)
        score = scorer.score(path_text, question)
        scores.append(score)
    
    max_score = max(scores)
    if max_score < TAU:
        return True, "semantic_dead_end"
    
    # Condition 3: Entropy spike — model is highly uncertain
    # (Requires access to model logits at this step)
    if beam_state.entropy > ENTROPY_THRESHOLD:
        return True, "high_uncertainty"
    
    return False, ""
```

### 4. The Rollback Operation

```python
def rollback(beam, checkpoint_stack, graph, scorer, question):
    """
    Backtrack to the last checkpoint and try the next-best path.
    Returns updated (prefix, trie, checkpoint_stack) or None if max backtracking exceeded.
    """
    # Pop the last checkpoint
    checkpoint = checkpoint_stack.pop()
    if checkpoint is None:
        return None  # nothing to backtrack to
    
    # Mark the path that failed as used
    failed_path = beam.prefix_text
    checkpoint.used_expansions.add(failed_path)
    
    # Restore checkpoint state
    prefix_tokens = checkpoint.prefix_tokens
    restored_trie = checkpoint.trie
    current_entity = checkpoint.current_entity
    
    # Find the next-best neighbor not yet tried
    best_neighbor = None
    best_score = -1
    
    for neighbor in graph.neighbors(current_entity):
        rel = graph[current_entity][neighbor]["relation"]
        candidate_path = f"{checkpoint.prefix_text} -> {rel} -> {neighbor}"
        
        if candidate_path in checkpoint.used_expansions:
            continue  # already tried and failed
        
        score = scorer.score(candidate_path, question + " " + checkpoint.prefix_text)
        if score > best_score:
            best_score = score
            best_neighbor = (neighbor, rel, candidate_path)
    
    if best_neighbor is None or best_score < TAU:
        # No more options at this level — backtrack further
        return rollback(beam, checkpoint_stack, graph, scorer, question)
    
    # Expand the trie with the new path
    entity, rel, path_str = best_neighbor
    new_tokenized = tokenizer.encode(path_str)
    expanded_trie = expand_trie_with_path(restored_trie, new_tokenized)
    
    # Update beam state
    beam.prefix_tokens = tokenizer.encode(checkpoint.prefix_text + " -> " + rel + " -> " + entity)
    beam.trie = expanded_trie
    beam.checkpoint_stack = checkpoint_stack
    
    return beam
```

### 5. Complete Decoding Loop

```python
class DCATrieV3Decoder:
    def __init__(self, model, tokenizer, scorer, graph, tau=0.3, k=5, max_backtrack=3):
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.graph = graph
        self.tau = tau
        self.k = k
        self.max_backtrack = max_backtrack
    
    def generate(self, question, initial_trie, start_token_ids, end_token_ids):
        """
        Run beam search with semantic backtracking.
        
        This implements a custom generation loop that wraps model.generate()
        with checkpointing and rollback.
        """
        # Initialize beams with checkpoint stacks
        beams = self.init_beams(initial_trie, question)
        
        for step in range(MAX_STEPS):
            # --- Detection Phase ---
            for beam in beams:
                if not beam.active:
                    continue
                
                should_back, reason = should_backtrack(beam, self.scorer, question, self.graph)
                if should_back:
                    if beam.checkpoint_stack.backtrack_count < self.max_backtrack:
                        # Attempt rollback
                        restored = rollback(beam, beam.checkpoint_stack, self.graph, self.scorer, question)
                        if restored is None:
                            beam.active = False  # exhausted all options
                        else:
                            beam = restored
                    else:
                        beam.active = False  # exceeded max backtrack budget
            
            # --- Generation Phase ---
            # Run model.generate() for one step with all active beams
            outputs = self.model.generate_step([b.prefix for b in active_beams])
            
            # --- Checkpoint Phase ---
            for i, beam in enumerate(active_beams):
                new_token = outputs.next_tokens[i]
                beam.prefix = beam.prefix + [new_token]
                
                # Check if we just committed an entity
                entity = detect_entity_boundary(beam.prefix, self.tokenizer)
                if entity:
                    # Snapshot current state
                    checkpoint = TrieCheckpoint(
                        prefix_tokens=list(beam.prefix),
                        prefix_text=self.tokenizer.decode(beam.prefix),
                        current_entity=entity,
                        trie=copy_trie(beam.trie),  # deep copy the trie
                        beam_score=beam.score,
                        used_expansions=set(),
                        backtrack_depth=0,
                    )
                    beam.checkpoint_stack.push(checkpoint)
                    
                    # Expand trie with scored neighbors
                    expand_trie_semantically(
                        beam.trie, entity, self.graph, 
                        self.scorer, question, beam.prefix_text
                    )
            
            # --- Pruning Phase ---
            beams = keep_top_k(beams, self.k)
        
        return self.extract_paths(beams)
```

---

## Implementation Strategy for GCR Codebase

### Option A: Custom Decoding Loop (Recommended for Maximum Novelty)

Replace HuggingFace's `model.generate()` with a manual loop that:
1. Calls `model.forward()` step by step
2. Manages beam state, checkpoints, and trie copies
3. Implements rollback as described

**Pros:** Full control, clean architecture, publishable contribution
**Cons:** More code to write, potential performance overhead

### Option B: Multi-Stage generate() Calls (Recommended for Timeline Safety)

A practical approximation that still demonstrates the idea:

1. Run GCR's normal `model.generate()` for K beams
2. For each generated path, evaluate whether it succeeds (reaches answer entity)
3. For failed paths, identify the entity split where the path went wrong
4. Re-run generation from that split point with the bad choice masked
5. Re-evaluate

```python
def iterative_refine(question, trie, model, graph, scorer, k=5):
    """Multi-stage generation with backtracking via re-execution."""
    
    # Stage 1: Normal generation
    paths = model.generate(question, trie, k=k)
    
    # Stage 2: Evaluate each path
    for i, path in enumerate(paths):
        if reaches_answer(path):
            continue  # success
        
        # Stage 3: Find the wrong split
        entity, alternatives = find_bad_split(path, graph, scorer, question)
        
        if alternatives:
            # Stage 4: Mask the bad path and re-generate
            masked_trie = mask_path(trie, path)
            new_paths = model.generate(question, masked_trie, k=1)
            paths[i] = new_paths[0]
    
    return paths
```

**Pros:** Works with existing `model.generate()`, simpler to implement
**Cons:** Less elegant, doesn't capture the "runtime recovery" narrative as cleanly

### Option C: Prefix-Allowed-Tokens with External State (Middle Ground)

Use HuggingFace's `prefix_allowed_tokens_fn` but manage state externally:

```python
class BacktrackingState:
    def __init__(self, initial_trie, graph, scorer, question):
        self.checkpoint_stack = CheckpointStack()
        self.trie = initial_trie
        self.snapshot_schedule = []  # positions where checkpoints exist
        # ... 
    
    def allowed_tokens_fn(self, batch_id, sent):
        prefix = sent.tolist()[self.input_length:]
        
        # Check for entity boundary
        entity = detect_entity_boundary(prefix, self.tokenizer)
        if entity:
            # Checkpoint before expanding
            self.checkpoint_stack.push(TrieCheckpoint(...))
            expand_trie(self.trie, entity, ...)
        
        # Get valid tokens
        valid = self.trie.get(prefix)
        
        if not valid or self.all_below_threshold(valid):
            # Backtrack!
            checkpoint = self.checkpoint_stack.pop()
            if checkpoint and checkpoint.backtrack_depth < MAX_BACKTRACK:
                # Restore and retry
                self.trie = checkpoint.trie
                # ... this gets complex with HuggingFace's internals
                return self.trie.get(checkpoint.prefix)
        
        return valid or FULL_VOCAB
```

**Pros:** Reuses existing `model.generate()` framework
**Cons:** HuggingFace doesn't support changing the generated prefix mid-generation; this only works for lookahead at the current step, not true rollback

---

## Evaluation Plan

### Metrics

| Metric | What It Measures | Compared To |
|--------|-----------------|-------------|
| Hits@1 | Answer accuracy | GCR, DCA-Trie v1/v2 |
| Path recovery rate | % dead-end paths saved by backtracking | v2 alone |
| Avg backtrack depth | How many rollbacks per successful path | Internal |
| SIR | Oracle permissiveness | GCR, v1, v2 |
| Gen time overhead | Latency cost of backtracking | v2 |

### Ablation Questions

1. Does backtracking improve accuracy on multi-hop (depth 3+) questions more than shallow ones?
2. What's the optimal `max_backtrack` budget? (Tradeoff between recovery rate and latency)
3. Which backtrack trigger is most effective: dead-end, semantic dead-end, or entropy spike?
4. Does semantic scoring + backtracking help more than either alone? (Cross-ablation)

### Expected Results

- **WebQSP (depth 1-2):** Small improvement. Most paths are short enough that v2 already handles them.
- **CWQ (depth 3-4):** Larger improvement. Error propagation is worse on longer paths, so backtracking matters more.
- **Path recovery rate:** Expect 15-30% of failed beams to be recoverable via backtracking.

---

## Connection to the Research Landscape

| Paper | What It Does | How DCA-Trie v3 Differs |
|-------|-------------|------------------------|
| **AdapTrack** (Oct 2025) | Backtracking for API completion constraints | AdapTrack is general; DCA-Trie v3 is specialized for KG-trie structure, uses semantic scoring to guide which path to retry, not just constraint recovery |
| **CRANE** (ICML 2025) | Shows constraints damage reasoning; proposes grammar augmentation | DCA-Trie v3 shows a complementary result: when constraints DO damage reasoning, backtracking can recover the lost accuracy |
| **ATLAS-RTC** (Mar 2026) | Closed-loop runtime control with rollback | ATLAS-RTC is a general framework; DCA-Trie v3 is KG-specific, using the trie structure for efficient checkpoint/restore |
| **AWRS** | Adaptive rejection sampling for constrained gen | AWRS is forward-only (rejection, not rollback); DCA-Trie v3 adds the backward mechanism |
| **RouterKGQA** (Mar 2026) | Specialized-general routing with answer filtering | RouterKGQA repairs at the query level; DCA-Trie v3 repairs at the token level during decoding |

DCA-Trie v3 occupies a unique space: it's the first method to apply **semantically-guided backtracking** specifically to **KG-trie constrained decoding**.

---

## Proposed Timeline for Implementation

| Week | Task |
|------|------|
| 1 | Implement `CheckpointStack`, `TrieCheckpoint`, entity boundary detection |
| 2 | Build `should_backtrack()` with all three conditions |
| 3 | Implement `rollback()` with trie restoration |
| 4 | Integrate into custom decoding loop (Option A) or multi-stage pipeline (Option B) |
| 5 | Test on 100-question subset, debug boundary detection and trie restoration |
| 6 | Full evaluation on WebQSP + CWQ |
| 7 | Ablation experiments (backtrack depth, triggers, τ sensitivity) |
| 8 | Write up for Chapter 4 |

The v1, v2, and SIR measurement from the original plan are prerequisites — this builds on top of v2.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Entity boundary detection is unreliable (tokenization makes it hard to detect " -> " boundaries) | Medium | Use the trie structure itself: a token is a boundary if it triggers a transition from entity-subtrie to relation-subtrie |
| Trie copying at every checkpoint is too slow | Medium | Use copy-on-write: share the base trie, only copy the expanded portions |
| Backtracking doesn't improve accuracy much on WebQSP (shallow questions) | High | This is expected — frame the contribution around CWQ depth 3-4. If WebQSP is flat, that's still an interesting negative result |
| Backtracking loops infinitely | Low | Hard limit on `max_backtrack` per beam; also limit total backtrack steps per question |
