# Concepts Guide

## 1. Knowledge Graphs (KGs)

### What is a KG?

A knowledge graph is a structured database of facts stored as **triples**: `(subject, relation, object)`.

```
(Marie Curie,   award_received,    Nobel_Prize_Physics)
(subject)       (relation)         (object)
```

Each triple is a single verified fact. Collections of triples form a graph where entities are nodes and relations are labeled edges.

### Freebase

Freebase is the KG used by WebQSP and CWQ (the two benchmarks in this project). It contains tens of millions of triples across domains like:

- People (biographies, awards, education)
- Locations (geography, cities, countries)
- Entertainment (films, music, books)
- Science (discoveries, institutions)

Entity IDs in Freebase use the format `m.0l0j4x3` (mid format). Relation names use dotted paths like `people.person.parents` or `location.location.contains`.

### Multi-hop Reasoning

Many questions require reasoning through multiple triples. For example:

_"What is the nationality of the director of the film in which the Blue Hawaii actor starred?"_

This requires following a chain: `Blue Hawaii → starred_actor → Elvis Presley → starred_in_films → ... → director → nationality`

A **reasoning path** is a sequence of triples connecting the question entity to the answer entity:

```
Elvis Presley → film.actor.film → [film] → film.film.directed_by → [director] → people.person.nationality → [country]
```

---

## 2. Autoregressive Generation in LLMs

### How an LLM Generates Text

At each step `t`, given the input `x` and previously generated tokens `y_<t`:

```
P(y_t | y_<t, x) = softmax( logits_t )
```

The model computes a probability distribution over its entire vocabulary (~128K tokens for Llama-3.1). The token is selected by some **decoding strategy** (greedy, sampling, beam search).

### Why Hallucination Happens

The critical problem: **the model's parameters are the only source of facts**. There is no mechanism to verify a candidate token against an external database before selecting it.

1. The model has learned statistical patterns from training text
2. For common facts ("Paris is the capital of France"), the parameters encode reliable knowledge
3. For rare or compositional facts, the parameters may encode incorrect associations
4. Once an incorrect token is generated at step `t`, it becomes conditioning context for step `t+1`, propagating the error fluently

### The Scaling Paradox

- Larger models hallucinate **differently**, not **less**
- On TruthfulQA, larger models sometimes perform **worse** because they better reproduce widespread misconceptions from training data
- Multi-hop compositional questions remain difficult even for GPT-4

---

## 3. Constrained Decoding

### The Core Idea

Instead of changing what the model **sees** (prompt engineering, RAG), constrained decoding changes what the model is **allowed to generate**.

Constrained decoding works via **logit masking**:

```
allowed_tokens = [tokens that are valid at this step]
for token_id not in allowed_tokens:
    logits[token_id] = -inf

P(y_t | y_<t, x) = softmax( masked_logits_t )
```

Tokens outside the allowed set get exactly **zero probability** — not low probability, but zero. This is a hard constraint, not a soft one.

### Types of Constrained Decoding

| Type | Constraint | Example |
|------|-----------|---------|
| Grammar | Valid syntax | JSON, SQL generation |
| Entity | Valid KG entities | GCR, DoG |
| Format | Specific structure | <PATH>...</PATH> |

### Prefix-Allowed-Tokens Function

HuggingFace's `model.generate()` supports `prefix_allowed_tokens_fn(batch_id, input_ids)` which returns the list of valid token IDs at each step. This is the hook GCR uses.

---

## 4. The KG-Trie

### What is a Trie?

A **trie** (prefix tree) is a tree structure where:

- Each node represents a prefix
- Each edge is labeled with a token ID
- Paths from root to leaves represent complete sequences

For sequences:

- `[101, 205, 300]`
- `[101, 206]`

the trie looks like:

```
root → 101 → 205 → 300 (leaf)
          ↘ 206 (leaf)
```

### The GCR KG-Trie

GCR builds a trie from all valid KG paths:

1. **Start** from question entities (`q_entity`)
2. **DFS/BFS** expands outward up to L hops
3. **Each path** is a string like `"Elvis Presley -> film.actor.film -> Blue Hawaii"`
4. The **string is tokenized** into token IDs
5. These token ID sequences are inserted into a **MarisaTrie**

At decoding time, the trie answers: "Given the tokens generated so far, what tokens are valid next?"

### Why a Trie and Not a Set?

- A trie provides O(1) per-step lookup for the valid next tokens
- A flat set would require checking all paths against the partial generation
- The trie naturally handles variable-length paths

### GCR's Static Trie Problem

GCR builds the trie **once** before generation starts. The same valid-token set is used at step 1 and step 10. This means:

- Paths that are irrelevant to the question remain valid at all steps
- The constraint oracle has no awareness of the question's intent
- The model must still rely on its parametric knowledge to choose among valid options

---

## 5. The MarisaTrie Implementation

GCR uses the `marisa-trie` Python library (a wrapper around the MARISA C++ library — Matching Algorithm with Recursively Implemented StorAge).

```python
from src.trie import MarisaTrie

# Build: pass list of token ID sequences
trie = MarisaTrie(tokenized_paths, max_token_id=len(tokenizer) + 1)

# Lookup: get valid next tokens given a prefix
next_tokens = trie.get([101, 205])  # returns [300] if edge exists
```

**Key detail:** MarisaTrie internally converts token IDs to characters:

- Token IDs (0-54999) → `chr(i)`
- Token IDs (65000+) → `chr(i)` with offset
- This is an encoding trick to use string-based trie libraries

For the first-level lookup, it caches all first tokens for O(1) access at the root.

---

## 6. Beam Search

### How Beam Search Works

Beam search maintains K **hypotheses** (partial sequences) at each step:

1. Start with K copies of the input prompt
2. At each step, expand each hypothesis by V (vocab size) candidates
3. Score each candidate (log-probability of the full sequence)
4. Keep only the top-K scoring candidates
5. Repeat until an end token is generated

### Diversity-Penalized Group Beam Search

GCR uses `group-beam` mode which:

1. Divides K beams into G groups
2. Within each group, standard beam search
3. Between groups, a **diversity penalty** discourages generating the same tokens
4. Result: K diverse reasoning paths

```
num_beams = K
num_beam_groups = K
diversity_penalty = 1.0
```

This means each beam is its own group, maximizing diversity.

### Entity Boundary Detection in Beam Search

A critical detail: during beam search, GCR needs to know when an entity has been fully generated (so the next token is a relation, not another entity name). This is handled by the trie structure itself:

- Paths are stored as complete strings including entity tokens, relation tokens, and special tokens
- When a path reaches a leaf (end of entity), the only valid tokens are `eos_token_id` or the next relation token
- The `MarisaTrie.get()` method returns tokens that continue any valid path from the current prefix

---

## 7. The SIR Metric (Semantic Irrelevance Ratio)

### What SIR Measures

SIR quantifies **how permissive the constraint oracle is** — the fraction of valid paths that are semantically irrelevant to the question.

### Formal Definition

For a question `q` at step `t` with admitted path set `P_valid(t)`:

```
SIR(q, t) = 1 - max_{p in P_valid(t)} cos(E(p), E(q, y_<t))
```

where `E` is a sentence transformer encoder producing 384-dimensional embeddings, and `cos` is cosine similarity.

### What SIR Tells You

- **SIR ≈ 0**: All admitted paths are relevant to the question
- **SIR ≈ 1**: Most admitted paths are irrelevant to the question
- GCR's static trie should show **high SIR** because it admits all structurally valid paths
- DCA-Trie aims to achieve **lower SIR** by filtering semantically irrelevant paths

### Why SIR Matters

SIR is a **process metric**, not an outcome metric. It measures the quality of the constraint oracle independently of whether the final answer is correct. This is important because:

- A model can get lucky and answer correctly despite a permissive oracle
- A model can fail despite a tight oracle
- SIR isolates the constraint quality from the model's parametric knowledge

---

## 8. The Two-Step GCR Pipeline

### Step 1: Path Generation + Answer Extraction

```
Input:  Question q
        Knowledge Graph subgraph G_q
        Question entities E_q

Process:
1. Build all paths in G_q within L hops of E_q → P_all
2. Build MarisaTrie from P_all
3. Generate K paths via constrained beam search
4. For each path, extract the terminal entity as a candidate answer

Output: K reasoning paths + candidate answers
```

### Step 2: Graph Inductive Reasoning (Optional)

```
Input:  Question q
        K reasoning paths from Step 1

Process:
1. Format reasoning paths as context in the prompt
2. Query a general LLM (GPT-4o-mini) to reason over paths
3. Produce final answer

Output: Final answer
```

Step 2 is optional because Step 1 already produces answers. The reported Hits@1 of 92.6 includes both steps.

---

## 9. Dataset Structure

Each entry in the HuggingFace dataset (`rmanluo/RoG-webqsp`, `rmanluo/RoG-cwq`) contains:

```python
{
    "id": "WebQSP-1234",
    "question": "What is the nationality of the director of Inception?",
    "answer": ["Canadian", "Canada"],
    "q_entity": ["m.0c6m"],      # Freebase entity IDs
    "a_entity": ["m.0dl3s"],     # Freebase entity IDs
    "graph": [                    # KB subgraph as adjacency list
        ["m.0c6m", "film.director.film", "m.0dl3s"],
        ["m.0c6m", "people.person.nationality", "m.0dl3s"],
        ...
    ],
    "paths": [                    # Pre-computed (optional)
        [
            ["m.0c6m", "people.person.nationality", "m.02m71r1"],
            ["m.02m71r1", "location.country.national_anthem", "m.0c6m"]
        ]
    ]
}
```

The `graph` field is a **local subgraph** extracted from Freebase around the question and answer entities — it does NOT include the entire Freebase graph. This makes the DFS/BFS tractable.
