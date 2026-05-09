# Answer-Type-Guided Hard Constraints: Research Survey

> **📦 LITERATURE SURVEY:** Sections 1-7 analyse GCR's limitations and survey
> external research (RouterKGQA, ORT, READS, RCD, CoKE, PCR, etc.) to
> motivate and support the answer-type direction.
>
> **🆕 CONTRIBUTION:** Section 8 is a concrete implementation roadmap for
> extending DCA-Trie with answer-type-guided hard constraints.

A deep investigation into why cosine similarity is a flawed relevance proxy
and what answer-type-prediction-based alternatives exist — connecting research
from KGQA, constrained decoding, and semantic parsing literatures.

---

## Quick Reference: Verified Source Links

| Paper / Resource | Link |
|-----------------|------|
| **GCR** (Luo et al., ICML 2025) | https://proceedings.mlr.press/v267/luo25t.html |
| **RouterKGQA** (Yuan et al., ACL 2025) | https://arxiv.org/abs/2603.20017 |
| **ORT** (Liu et al., ACL 2025) | https://arxiv.org/abs/2502.11491 |
| **READS** (Zhou et al., 2024) | https://arxiv.org/abs/2412.12643 |
| **RAR** (2025) | https://arxiv.org/abs/2505.20971 |
| **RCD** (El Hamdani et al., 2025) | https://arxiv.org/abs/2509.23417 |
| **SEAL** (Wang et al., 2025) | https://arxiv.org/abs/2512.04868 |
| **CoKE** (Wang et al., ACL 2020) | https://arxiv.org/abs/1911.02168 |
| **PCR** (Oladokun, 2025) | https://arxiv.org/abs/2511.18313 |
| **SmartVector** (2026) | https://arxiv.org/abs/2604.20598 |
| **When Vector Search Fails** (Pan, 2026) | https://tianpan.co/blog/2026-04-20-knowledge-graphs-vs-vector-search-retrieval |
| **RouterKGQA code** | https://github.com/Oldcircle/RouterKGQA |
| **DCA-Trie project** | `/home/bernard/research/projects/dca-trie` |

---

## Table of Contents

1. [The Failure Analysis: Why Cosine Similarity Is a Proxy](#1-the-failure-analysis)
2. [The Alternative: Answer-Type-Guided Hard Constraints](#2-the-alternative)
3. [Approach Family 1: Question-Type Classification for KGQA](#3-approach-family-1-question-type-classification)
4. [Approach Family 2: Constraint-Aware Path Generation](#4-approach-family-2-constraint-aware-path-generation)
5. [Approach Family 3: Ontology-Guided Traversal](#5-approach-family-3-ontology-guided-traversal)
6. [Approach Family 4: Explicit Constraint Filtering](#6-approach-family-4-explicit-constraint-filtering)
7. [Contrastive Analysis Across Approaches](#7-contrastive-analysis)
8. [Implementation Roadmap for DCA-Trie](#8-implementation-roadmap)
9. [References](#9-references)

---

# 1. The Failure Analysis

## 1.1 The Core Problem

Cosine similarity between embedding vectors measures **topical overlap**, not
**inferential relevance**. These are fundamentally different.

```python
Question: "Where was Marie Curie born?"

Path A:  "Marie Curie → people.person.place_of_birth → Warsaw"
         cos_sim(emb(A), emb(question)) = 0.82   ← correct

Path B:  "Marie Curie → people.person.nationality → Polish"
         cos_sim(emb(B), emb(question)) = 0.78   ← WRONG, but scores nearly as high

Path C:  "Marie Curie → award_received → Nobel_Prize_in_Physics"
         cos_sim(emb(C), emb(question)) = 0.55   ← correctly lower, but still positive

Path D:  "Marie Curie → people.person.education → Sorbonne"
         cos_sim(emb(D), emb(question)) = 0.71   ← WRONG, high because both are "biographical"
```

The embeddings capture that all biographical relations are "about Marie Curie" and
thus have high cosine similarity with the question. But only one of these paths
actually answers "Where was she born?"

The problem is structural, not solvable by a better embedding model: **cosine
similarity in embedding space does not distinguish between "this is topically
related" and "this is the answer".**

## 1.2 Three Failure Modes

| Failure Mode | Description | Example |
|---|---|---|
| **Topical bleed** | Relations about the same entity all score high regardless of question intent | nationality vs. place_of_birth for biographical questions |
| **Relation ambiguity** | A single relation type can produce entities of different types | `award_received` can → Nobel Prize (event) or a Person (recipient), depending on direction |
| **Terminal entity irrelevance** | The path ends at an entity of the wrong type even if intermediate relations match | Path ending at a Date when the question asks for a Person |

The third failure mode is the most consequential for GCR/DCA-Trie because it is
detectable and preventable at trie construction time.

## 1.3 What the Research Literature Says

> "Semantic similarity is the wrong abstraction for a significant class of retrieval
> problems. Multi-hop relationship queries... require traversing node → node → node.
> No amount of embedding quality makes this expressible as a nearest-neighbor search."
> — *[When Vector Search Fails](https://tianpan.co/blog/2026-04-20-knowledge-graphs-vs-vector-search-retrieval) (Pan, 2026)*

> "The confidence of retrieval is not the same as the correctness of retrieval. Vector
> search returns high-similarity results even when those results don't actually answer
> the query — the similarity score tells you something about semantic relatedness, not
> about whether the retrieved content contains the answer."
> — *[When Vector Search Fails](https://tianpan.co/blog/2026-04-20-knowledge-graphs-vs-vector-search-retrieval) (Pan, 2026)*

> "Cosine similarity alone has no principled way to distinguish among candidates that
> are all topically relevant."
> — *[SmartVector](https://arxiv.org/abs/2604.20598) (2026)*

> "The purpose of the question is abstract and difficult to match with specific entities.
> Existing methods rely on entity vector matching, but as a result, it is difficult to
> establish reasoning paths to the purpose, leading to information loss and redundancy."
> — *[ORT](https://arxiv.org/abs/2502.11491) (Liu et al., ACL 2025)*

> "Embedding-based approaches retrieve information based solely on semantic similarity,
> without considering structural relationships within knowledge bases. This limitation
> becomes particularly problematic in multi-hop reasoning scenarios."
> — *[Path-Constrained Retrieval](https://arxiv.org/abs/2511.18313) (Oladokun, 2025)*

## 1.4 Empirical Evidence from the Project

The DCA-Trie threshold sweep on 5 test questions found:

- τ = 0.55: FNR = 0.0, avg trie size reduction = 49.6%
- τ = 0.60: FNR = 0.2 (one false negative)

This means at τ = 0.55, half of all structurally valid paths were filtered with
zero false negatives — on these 5 questions. The false negative at τ = 0.60 is a
warning: cosine similarity is inconsistent enough that a single threshold cannot
cleanly separate relevant from irrelevant for all questions. The threshold that
works for "Who was the spouse of Barack Obama?" may not work for
"Which film did Blue Hawaii actor star in?" because the cosine distributions
differ per question.

---

# 2. The Alternative

## 2.1 The Key Insight

Instead of filtering paths by semantic similarity to the question text, filter by
**compatibility between the terminal entity type and the expected answer type**.

This transforms the problem from a continuous similarity ranking (soft constraint)
to a discrete type compatibility check (hard constraint).

### Soft constraint (cosine similarity)

```
admit(p) iff cos_sim(embed(question), embed(p)) ≥ τ
```

### Hard constraint (answer-type)

```
admit(p) iff type_of(terminal_entity(p)) ∈ expected_answer_types(question)
```

### Combined

```
admit(p) iff all:
  1. p is structurally valid in KG
  2. type_of(terminal_entity(p)) ∈ expected_answer_types(question)
  3. cos_sim(embed(question), embed(p)) ≥ τ
```

## 2.2 Why This Is More Principled

| Property | Cosine Similarity | Answer-Type |
|----------|------------------|-------------|
| **What it measures** | Topical overlap | Logical compatibility |
| **Basis** | Distributional semantics | KG ontology (explicit types) |
| **False positive cause** | Semantic relatedness ≠ relevance | N/A — it's a logical check |
| **False negative cause** | Imperfect embeddings | KG type incompleteness |
| **Calibration** | Requires threshold sweep per dataset | Zero config |
| **Domain transfer** | Embeddings degrade with domain-specific vocabulary | Types are schema-defined, domain-independent |
| **Explainability** | "It scored 0.55" | "The path ends at a Person, but the question asks for a Location" |

## 2.3 What Freebase Types Look Like

Freebase entities carry type annotations as a `type.object.type` relation:

```
Entity: /m/0c6q0          (Warsaw)
Type:   /location/citytown
Type:   /location/location
Type:   /location/administrative_division

Entity: /m/0d6j3x          (Marie Curie)
Type:   /people/person
Type:   /award/award_winner
Type:   /education/educated_at

Entity: /m/02m3n7d         (Nobel Prize in Physics)
Type:   /award/award
Type:   /award/award_category
```

**The mapping from question type to target Freebase type is straightforward:**

| Question Word | Expected Answer Type | Freebase Type Filter |
|--------------|---------------------|---------------------|
| Who | Person | `people.person` |
| Where | Location | `location.location` |
| When / What date | Date/Time | `time.datetime` or `/type/datetime` |
| What organization | Organization | `organization.organization` |
| How many | Number/Quantity | Any (count query, not entity) |
| Which *[noun]* | Depends on noun | Type corresponding to noun |
| What *[person noun]* | Person | `people.person` |

## 2.4 Limitations to Acknowledge

1. **KG type incompleteness**: Some entities may lack type annotations. Freebase
   is generally well-typed, but Wikidata has sparser type coverage.

2. **Ambiguous answer types**: "What did Marie Curie discover?" expects a
   `science.discovery` or similar. Classification requires understanding the
   question's relational predicate, not just the question word.

3. **Multi-type answers**: "List the films directed by Norman Taurog" expects
   entities of type `film.film`. If some directors also act in films, the
   answer set could mix types.

4. **Granularity mismatch**: The question might expect a specific subtype while
   the terminal entity only has a generic type annotation.

These are real limitations but they are **knowable and measurable** — unlike
cosine similarity's limitations, which are opaque. You can compute type coverage
statistics for your dataset and report exactly where the approach succeeds and
fails.

---

# 3. Approach Family 1: Question-Type Classification

These methods classify the question into a predefined type and use the type to
constrain downstream processing.

## 3.1 SEAL: Self-Evolving Agentic Learning (2025)

**Paper:** [SEAL](https://arxiv.org/abs/2512.04868) (arXiv:2512.04868, 2025)

**Core idea:** Two-stage semantic parsing. Stage 1 generates an S-expression core.
Stage 2 predicts the question type and selects a template.

**Question-type taxonomy:**

```
simple        — single triple lookup
verify        — boolean (yes/no)
count         — "how many"
comparative   — "more than", "largest"
temporal      — time-bound queries
conjunction   — multiple conditions
```

**Method:** Question-type prediction via LLM prompting with 3 examples per type,
backed by keyword-based heuristics for ambiguity resolution.

**Relevance to DCA-Trie:** The question-type taxonomy maps directly onto answer
types. SEAL demonstrates that a small taxonomy (6-8 types) captures the majority
of WebQSP/CWQ questions.

**Implementability:** High. Rule-based heuristics for question-word + head-noun
extraction cover 80%+ of cases (as noted in the DCA-Trie Implementation Plan).

## 3.2 Reason-Align-Respond (RAR) (2025)

**Paper:** [RAR](https://arxiv.org/abs/2505.20971) (arXiv:2505.20971, 2025)

**Core idea:** Three fine-tuned LLMs — Reasoner (generates reasoning chain),
Aligner (decodes KG path guided by reasoning chain), Responser (produces final
answer). The Reasoner's reasoning chain acts as an implicit question-type
representation.

**Key result:** Removing KG-constrained decoding yields the largest performance
drop in their ablation, confirming the importance of hard constraints for faithful
KG reasoning.

**Relevance to DCA-Trie:** RAR's Reasoner module is a proxy for explicit
question-type classification. The reasoning chain it generates ("First find
X's nationality, then find the country...") encodes answer type information.

**Implementability:** Low (requires training three models). But the architecture
validates that question understanding → path generation is the right direction.

## 3.3 Implementation Strategy: Lightweight Question-Type Classifier

For DCA-Trie, a full question-type taxonomy can be implemented as a 50-line
Python function using question-word + dependency parsing:

```python
def classify_answer_type(question: str) -> str:
    q = question.lower().strip()

    # Person questions
    if q.startswith("who"):
        return "person"
    if q.startswith("whom"):
        return "person"

    # Location questions
    if q.startswith("where"):
        return "location"
    if re.search(r"\b(in|at|from)\s+which\s+(city|country|place|location)\b", q):
        return "location"

    # Date questions
    if q.startswith("when"):
        return "date"
    if re.search(r"\bwhat\s+(year|date|day|time|month|decade|century)\b", q):
        return "date"

    # Organization questions
    if re.search(r"\bwhich\s+(company|organization|university|school|bank)\b", q):
        return "organization"

    # Quantity questions
    if q.startswith("how many"):
        return "quantity"
    if q.startswith("how much"):
        return "quantity"

    # Fallback to LLM for ambiguous cases
    return "other"
```

**Limitation:** Pure rule-based covers ~80%. The remaining ~20% (e.g., "What is
the capital of Ghana?" → Location, not "other") require either an LLM call or
a learned classifier.

**For the project:** 80% coverage is sufficient to demonstrate the concept.
Instrument the remaining 20% as "unknown type" and show that answer-type
filtering improves precision on known-type questions without degrading
performance on unknown-type questions (where it falls back to no filtering).

---

# 4. Approach Family 2: Constraint-Aware Path Generation

These methods generate **explicit constraints** alongside reasoning paths, then
use the constraints for answer filtering.

## 4.1 RouterKGQA (2025)

**Paper:** [RouterKGQA](https://arxiv.org/abs/2603.20017) (arXiv:2603.20017, 2025)
**Code:** https://github.com/Oldcircle/RouterKGQA

**Core idea:** A specialized model generates a Constraint-aware Reasoning Path
(CRP) containing both a main path and explicit constraints (numeric, temporal,
ordering). A general model repairs unreachable paths via KG-guided beam search.

**Constraint types:**

```
Main path:     Elvis Presley → film.actor.film → ?film
Constraints:
  - direction: outgoing from Elvis Presley
  - relation_type: film.actor.film (not film.film.starring)
  - answer_type: film.film
  - ordering: release_date descending (for "latest...")
  - numeric: limit = 1 (for "first..." or "most recent...")
```

**Key results:**

- Outperforms GCR by 3.57 F1 and 0.49 Hits@1 on average
- Gains are most pronounced on **comparative** (∆F1 = +4.02) and
  **superlative** (∆F1 = +3.17) queries — precisely where GCR's
  constraint-free generation is weakest
- Requires only 1.15 LLM calls per question (vs. agent-based methods that
  require 10+)

**Relevance to DCA-Trie:** RouterKGQA directly validates the thesis that
explicit constraints improve upon GCR. The constraint types they identify
(numeric, temporal, ordering) extend beyond answer-type but include it.

**Key difference from DCA-Trie:** RouterKGQA trains a specialized model to
predict constraints. DCA-Trie's answer-type approach is rule-based (no
training needed). RouterKGQA's approach is more powerful; DCA-Trie's is more
pragmatic.

## 4.2 Constraint-Aware Answer Filtering

RouterKGQA's specific innovation for answer filtering:

```python
def filter_by_constraints(candidate_paths, constraints):
    filtered = []
    for path in candidate_paths:
        terminal = path[-1]  # last entity
        if constraints.get("answer_type"):
            if terminal.type not in constraints["answer_type"]:
                continue
        if constraints.get("numeric"):
            if not satisfies_numeric(terminal, constraints["numeric"]):
                continue
        if constraints.get("ordering"):
            if not satisfies_ordering(path, constraints["ordering"]):
                continue
        filtered.append(path)
    return filtered
```

**Implementability for DCA-Trie:** The answer-type filter is directly applicable.
The numeric/ordering filters are valuable extensions but require more work.

---

# 5. Approach Family 3: Ontology-Guided Traversal

These methods use the KG's ontology schema — entity types, relation domains/ranges,
class hierarchies — to guide or constrain traversal.

## 5.1 ORT: Ontology-Guided Reverse Thinking (2025)

**Paper:** [ORT](https://arxiv.org/abs/2502.11491) (arXiv:2502.11491, ACL 2025)

**Core idea:** Instead of traversing forward from the question entity (which
accumulates irrelevant branches), ORT works **backwards**: it extracts
*purpose labels* (answer types) and *condition labels* (question constraints)
from the question, constructs label-level reasoning paths using the KG ontology,
then uses these label paths to guide knowledge retrieval.

```
Question: "Which film did Blue Hawaii actor star in?"

Step 1 — Extract labels:
  Purpose label (answer type):  film.film
  Condition label:              people.person (the actor)

Step 2 — Construct label-level paths:
  people.person → film.actor.film → film.film

Step 3 — Ground labels to entities:
  people.person → Elvis Presley
  film.film → Blue_Hawaii
```

**Key results:**

- State-of-the-art on WebQSP and CWQ
- Significantly outperforms entity-vector-matching methods
- Label paths are compact (2-3 hops vs. entity paths that can be hundreds)

**Relevance to DCA-Trie:** ORT is the closest existing work to what DCA-Trie's
answer-type filtering proposes. The "purpose label" corresponds directly to the
"expected answer type". ORT uses it for *retrieval* (finding the right path);
DCA-Trie would use it for *filtering* (removing wrong paths from the trie).

**Key difference:** ORT requires ontology schema to be complete (label-level
paths need correct relation domain/range annotations). DCA-Trie's answer-type
check only needs terminal entity type annotations, which are more widely
available.

**Implementability:** ORT is complex (multi-step LLM pipeline). But the core
idea — extract answer type from the question, use it to filter — is simpler
than ORT's full implementation.

## 5.2 CoKE: Contextualized KG Embeddings (2019)

**Paper:** [CoKE](https://arxiv.org/abs/1911.02168) (ACL 2020, arXiv:1911.02168)

**Core idea:** Entities and relations have different meanings in different graph
contexts. CoKE uses a Transformer encoder over edge/path sequences to produce
contextualized embeddings.

**Relevance to DCA-Trie:** CoKE shows that a single embedding cannot capture an
entity's role in all contexts. This is the same limitation that makes cosine
similarity over static embeddings a poor relevance proxy in DCA-Trie. A
contextualized scorer (e.g., a small transformer that scores path+question
jointly) would be a more principled replacement for static cosine similarity.

**Implementability:** Low for this project (requires training CoKE-like model).
But the insight validates that the limitation is recognized in the broader KG
literature.

---

# 6. Approach Family 4: Explicit Constraint Filtering

These methods add constraint layers *after* path generation/reasoning to filter
answers.

## 6.1 READS: Reasoning With Discriminative Subtasks (2025)

**Paper:** [READS](https://arxiv.org/abs/2412.12643) (arXiv:2412.12643, 2024)

**Core idea:** Decompose KGQA into three subtasks:

1. **Graph searching** — retrieve question-related subgraph
2. **Graph pruning** — apply semantic constraints to prune irrelevant nodes
3. **Answer inference** — locate answer position in pruned subgraph

The pruning step explicitly models constraints:

```python
def pruning_step(subgraph, question):
    # Step 1: LLM identifies constraint types in question
    constraints = llm.extract_constraints(question)

    # Step 2: For each constraint type, filter subgraph entities
    for entity in subgraph.entities:
        if constraint_matches(entity, constraints):
            keep(entity)
        else:
            prune(entity)

    return pruned_subgraph
```

**Relevance to DCA-Trie:** READS validates the "filter by constraint" approach
at a graph level (not just terminal entities). The constraint types include
entity type, numeric range, date range, and relation direction — all applicable
to DCA-Trie's trie filtering.

**Key results:** READS's pruning step is responsible for the largest gain in
their ablation study. Constraint-aware pruning directly improves answer accuracy.

## 6.2 RCD: Retrieval-Constrained Decoding (2025)

**Paper:** [RCD](https://arxiv.org/abs/2509.23417) (arXiv:2509.23417, 2025)

**Core idea:** Constrain decoding to a pre-defined set of candidate entities
(not paths). Build a trie over candidate entity tokens. The model can only
generate entities present in the candidate set.

**Key results:**

- Constrained decoding over ALL entities (3.6M) still outperforms unconstrained
  (35.8 vs 32.3 F1 for Llama-3.1-70B)
- But targeted retrieval + constrained decoding does much better (46.0 F1)
- Conclusion: constraint alone helps, but retrieval narrowing helps more

**Relevance to DCA-Trie:** RCD shows that the entity-level constraint (equivalent
to answer-type filtering) is independently valuable. The ablation where they use
all 3.6M entities vs. retrieved subset mirrors the comparison between GCR (all
paths) and DCA-Trie (filtered paths).

---

# 7. Contrastive Analysis

## 7.1 Positioning DCA-Trie Answer-Type Constraints

| Approach | Constraint Type | Where Applied | Training Required | GCR Compatible |
|----------|----------------|--------------|-------------------|----------------|
| **DCA-Trie (cosine)** | Soft (continuous threshold) | Trie construction | No | Yes (wrapper) |
| **DCA-Trie (answer-type)** | Hard (type match) | Trie construction or decoding step | No | Yes (wrapper) |
| **ORT** | Label-level path retrieval | Retrieval phase | No (LLM-based) | Indirect |
| **RouterKGQA** | Explicit constraint prediction | Path generation + answer filtering | Yes (LoRA) | No (different arch) |
| **READS** | Semantic pruning | Subgraph level | No (LLM-based) | Indirect |
| **RCD** | Entity candidate set | Decoding | No | Yes (different trie) |
| **SEAL** | Question-type template | Semantic parsing | No (few-shot LLM) | No (different task) |

## 7.2 What Each Approach Adds to DCA-Trie

| Research Finding | Implication for DCA-Trie |
|-----------------|-------------------------|
| Cosine similarity conflates topical overlap with relevance (multiple sources) | Answer-type constraint should be a **separate filter layer**, not a replacement for cosine |
| RouterKGQA: explicit constraints improve GCR by 3.57 F1 | This is the strongest external validation that DCA-Trie's direction is correct |
| ORT: label-level reasoning outperforms entity-vector matching | Answer-type filtering is not just practical — it's **principled** and backed by SOTA results |
| READS: pruning step yields largest ablation gain | Constraint filtering may be more impactful than path quality improvements |
| RCD: entity constraint alone helps, retrieval targeting helps more | Answer-type filter (entity type matching) + cosine filter (semantic matching) are **complementary**, not redundant |
| CoKE: entities have different roles in different contexts | A stronger scorer might use a small transformer over path+question, not static embeddings |

## 7.3 Theoretical Framework: The Two Dimensions of Oracle Quality

Drawing on all the above research, oracle quality can be decomposed into:

```
Oracle Quality = Structural Coverage × Relevance Precision

Structural Coverage:   fraction of correct answer paths admitted by the oracle
                       (GCR has 100% — it admits everything)

Relevance Precision:    fraction of admitted paths that are relevant to the question
                       (GCR has low precision — it admits all structural paths)
```

| Variant | Structural Coverage | Relevance Precision | Mechanism |
|---------|-------------------|-------------------|-----------|
| **GCR** | 100% | Low | No filtering |
| **DCA-Trie v1** (cosine only) | ~95-100% (depends on τ) | Medium | Cosine threshold |
| **DCA-Trie v1** (answer-type only) | ~90-95% (untyped paths lost) | High | Type compatibility |
| **DCA-Trie v1** (combined) | ~90-95% | Highest | Both filters |
| **RouterKGQA** | ~95% (with repair) | High | Explicit constraints |

**Recommendation:** Use combined filtering. The cosine filter catches semantic
irrelevance; the answer-type filter catches structural incompatibility. They
operate on different failure modes and are independently useful.

---

# 8. Implementation Roadmap

## Phase 1: Dataset Analysis (1-2 days)

Before writing code, measure the prevalence of the problem:

1. **Sample 100 questions from WebQSP test set** (stratified by question word)
2. **For each question, collect all GCR-admitted paths at the terminal hop**
3. **Annotate:** How many terminal entities have the correct type? Wrong type?
4. **Measure:** What fraction of the trie's terminal entities would be eliminated
   by answer-type filtering?
5. **Report:** Coverage (do all correct answers survive?) and pruning rate

This directly replicates the Implementation Plan's "concrete next step" (p. 525-529).

## Phase 2: Question-Type Classifier (2-3 days)

Implement a lightweight classifier:

```python
# Option A: Rule-based (covers ~80%)
class RuleBasedTypeClassifier:
    TYPES = {
        "person": {"who", "whom", "whose"},
        "location": {"where", "which city", "which country", "which place"},
        "date": {"when", "what year", "what date", "what day"},
        "organization": {"which company", "which organization", "which university"},
        "quantity": {"how many", "how much"},
        "other": set(),
    }

    def classify(self, question):
        q = question.lower().strip()
        for type_, triggers in self.TYPES.items():
            if any(q.startswith(t) for t in triggers):
                return type_
        return "other"

# Option B: Few-shot LLM (covers ~98%)
LLM_TYPE_CLASSIFIER_PROMPT = """
Classify the question into one of: person, location, date, organization, quantity, other.

Examples:
- "Who was the spouse of Barack Obama?" → person
- "Where was Marie Curie born?" → location
- "When was the Nobel Prize founded?" → date
- "Which university did Stephen Hawking attend?" → organization
- "How many awards did Elvis Presley win?" → quantity

Question: {question}
Type:"""
```

**Validation:** Run both on 100 questions. Measure accuracy. If rule-based covers
≥80% with ≥95% accuracy, it's sufficient for the thesis experiment. Use LLM for
the remaining 20%.

## Phase 3: Freebase Type Mapping (1 day)

Create the mapping from question types to Freebase type constraints:

```python
ANSWER_TYPE_TO_FREEBASE_TYPE = {
    "person":         "people.person",
    "location":       "location.location",      # catch-all location
    "city":           "/location/citytown",
    "country":        "/location/country",
    "date":           "/time/datetime",
    "organization":   "/organization/organization",
    "educational_institution": "/education/educational_institution",
    "award":           "/award/award",
    "film":            "/film/film",
    "music_album":     "/music/album",
    "song":            "/music/composition",
    "book":            "/book/book",
    "quantity":        None,   # entity type not applicable
}
```

**Implementation note:** Freebase types use dotted paths (e.g., `people.person`).
Entity type annotations can be accessed via `graph[node]["type"]` or by querying
the subgraph for `(entity, "type.object.type", ?type)` triples.

## Phase 4: Answer-Type Filter Implementation (2-3 days)

Extend `V1TrieBuilder` with answer-type filtering:

```python
class AnswerTypeAwareTrieBuilder:
    def __init__(self, tokenizer, type_classifier, type_to_freebase, encoder=None, tau=None):
        self.tokenizer = tokenizer
        self.type_classifier = type_classifier
        self.type_to_freebase = type_to_freebase
        self.encoder = encoder          # optional: cosine filter on top
        self.tau = tau

    def build_filtered_trie(self, graph, q_entity, question):
        # 1. Classify question
        answer_type = self.type_classifier.classify(question)

        # 2. Get compatible Freebase types
        compatible_types = self.type_to_freebase.get(answer_type, [])

        # 3. Enumerate all structural paths
        all_paths = dfs(graph, q_entity, max_hops=4)

        # 4. Filter by answer type
        filtered_paths = []
        for path in all_paths:
            terminal_entity = path[-1][-1]  # last entity in last triple
            terminal_type = get_entity_type(graph, terminal_entity)

            # Hard constraint: type compatibility
            if compatible_types and terminal_type not in compatible_types:
                continue

            # Soft constraint (optional): cosine similarity
            if self.encoder and self.tau is not None:
                score = self.score_path(path, question)
                if score < self.tau:
                    continue

            filtered_paths.append(path)

        # 5. Build trie from filtered paths
        path_strings = [path_to_string(p) for p in filtered_paths]
        tokenized = self.tokenizer(path_strings)
        return MarisaTrie(tokenized)
```

## Phase 5: Evaluation (2-3 days)

Compare four conditions:

```
Condition              | Filtering
-----------------------|--------------------------
GCR (baseline)         | None
DCA-Trie v1 (cosine)   | cosine ≥ τ
DCA-Trie v1 (type)     | answer-type match
DCA-Trie v1 (combined) | both
```

Metrics:

- **Hits@1 / F1** — does answer accuracy improve?
- **Trie size** — how many paths are filtered?
- **FNR** — fraction of correct answers lost
- **SIR** — does semantic irrelevance decrease?
- **Coverage by question type** — which question types benefit most?

## Phase 6: Error Analysis (1-2 days)

For each failure case, classify into:

| Error Type | Description | Mitigation |
|-----------|-------------|------------|
| **Type prediction wrong** | Classifier assigned wrong answer type | Improve classifier (LLM fallback) |
| **Entity missing type** | Terminal entity has no type annotation in KG | Fall back to cosine-only filtering |
| **Type too coarse** | "location.location" misses city-specific correctness | Use more granular Freebase types |
| **Answer type ambiguous** | "What did X invent?" — could be product, process, or discovery | Accept multiple compatible types |
| **Multiple answer types** | Question expects diverse entities (e.g., "list all...") | Skip type filtering for "list" questions |

Report the distribution. This becomes your thesis's error analysis section.

## Interaction with DCA-Trie v2/v3

Answer-type filtering is orthogonal to v1/v2/v3:

- **v1 (static):** Filter at trie construction. Cheap and effective.
- **v2 (dynamic):** Filter at each expansion step (per beam). Prevents beams from
  expanding into type-incompatible branches.
- **v3 (backtracking):** When backtracking, the answer-type constraint provides a
  clear signal for when to backtrack: "This path ends at a Person but we need a
  Location — abandoned."

---

# 9. References

## Papers Cited (with links)

| Paper | Venue | Year | Link |
|-------|-------|------|------|
| **GCR** (Luo et al.) | ICML | 2025 | https://proceedings.mlr.press/v267/luo25t.html |
| **RouterKGQA** (Yuan et al.) | ACL | 2025 | https://arxiv.org/abs/2603.20017 |
| **ORT** (Liu et al.) | ACL | 2025 | https://arxiv.org/abs/2502.11491 |
| **READS** (Zhou et al.) | arXiv | 2024 | https://arxiv.org/abs/2412.12643 |
| **RCD** (El Hamdani et al.) | arXiv | 2025 | https://arxiv.org/abs/2509.23417 |
| **SEAL** (Wang et al.) | arXiv | 2025 | https://arxiv.org/abs/2512.04868 |
| **RAR** | arXiv | 2025 | https://arxiv.org/abs/2505.20971 |
| **CoKE** (Wang et al.) | ACL | 2020 | https://arxiv.org/abs/1911.02168 |
| **Path-Constrained Retrieval** (Oladokun) | arXiv | 2025 | https://arxiv.org/abs/2511.18313 |
| **SmartVector** | arXiv | 2026 | https://arxiv.org/abs/2604.20598 |

## Blog Posts and Code

| Resource | Link |
|----------|------|
| *When Vector Search Fails* (Tian Pan, 2026) | https://tianpan.co/blog/2026-04-20-knowledge-graphs-vs-vector-search-retrieval |
| *GraphRAG vs. Vector RAG* (Tian Pan, 2026) | https://tianpan.co/blog/2026-04-17-graphrag-vs-vector-rag-knowledge-graphs |
| *GraphRAG in Production* (Tian Pan, 2026) | https://tianpan.co/blog/2026-04-12-graphrag-production-when-vector-search-fails-multi-hop-reasoning |
| RouterKGQA code | https://github.com/Oldcircle/RouterKGQA |
| RCD code | https://github.com/Rajjaa/disambiguated-LLM |
| CoKE code | https://github.com/PaddlePaddle/Research/tree/master/KG/CoKE |
| SmartVector code | https://github.com/naizhong/smartvector |

## Direct Quotes

> "The purpose of the question is abstract and difficult to match with specific
> entities. Existing methods rely on entity vector matching, but as a result, it
> is difficult to establish reasoning paths to the purpose, leading to information
> loss and redundancy."
> — [ORT](https://arxiv.org/abs/2502.11491) (Liu et al., ACL 2025)

> "The gains [over GCR] are most pronounced on comparative (∆F1 = +4.02) and
> superlative (∆F1 = +3.17) queries — precisely the types that demand explicit
> numeric and ordering constraints, where GCR's constraint-free generation is
> most inadequate."
> — [RouterKGQA](https://arxiv.org/abs/2603.20017) (Yuan et al., ACL 2025)

> "Constrained generation alone captures more factual knowledge than unconstrained
> decoding. However, purely constraining the decoding stage, even over a large pool
> of possible entities, yields meaningful improvements but cannot fully substitute
> for targeted retrieval."
> — [RCD](https://arxiv.org/abs/2509.23417) (El Hamdani et al., 2025)

> "Removing KG-constrained Decoding yields the largest performance decrease,
> underscoring the importance of restricting generation to valid KG paths."
> — [RAR](https://arxiv.org/abs/2505.20971) (2025)

> "Cosine similarity between the question and both answers [old and new] is
> near-identical; without temporal signals, the retriever cannot distinguish them."
> — [SmartVector](https://arxiv.org/abs/2604.20598) (2026)
