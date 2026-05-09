## Implementation Plan

### Phase 0: Environment and Baseline Reproduction (Do this first, nothing else depends on success without it)

**Goal:** Reproduce GCR on WebQSP within 1-2% of Hits@1 = 92.6

Tasks:
1. Clone GCR repo, install dependencies, verify checkpoint loads from HuggingFace
2. Run GCR inference on WebQSP test set (start with a 100-question subset to validate the pipeline is working before running the full set)
3. Confirm Hits@1, F1, structural faithfulness match reported numbers
4. Log exact package versions (`requirements.txt` snapshot) — you need this for your reproducibility section

**Exit criterion:** WebQSP Hits@1 ≥ 91% on your run (within the 1-2% tolerance)

---

### Phase 1: SIR Measurement Module

**Goal:** Characterise permissiveness in GCR's static trie — this is objective (i) and generates your baseline permissiveness numbers for Chapter 4

Build one module: `sir_measurement.py`

```
inputs:  question q, GCR-built trie, partial generation y_{<t}
outputs: SIR(q, t), per-hop SIR breakdown

key components:
  - path enumeration from trie (iterate all paths admitted at step t)
  - MiniLM encoder (sentence-transformers, run on CPU)
  - cosine similarity with τ_ref = 0.3
  - aggregate over questions and hop depths
```

Run this against GCR's output (no DCA-Trie needed yet) to measure baseline SIR on WebQSP and CWQ. Plot SIR vs. hop depth — this should show a clear upward trend that motivates the whole project.

**Exit criterion:** You have a table: GCR baseline SIR at depths 1/2/3/4, with average trie size per step

---

### Phase 2: DCA-Trie v1 — Static Semantic Filtering

This is the simpler variant and should be tackled before v2. It touches only the trie construction step, not the decoding loop.

**Module structure:**

```
semantic_scorer.py        — shared between v1 and v2
  encode_path(p)          — str(p) → 384-dim vector, cached
  encode_query(q, y_lt)   — query(q, y_<t) → 384-dim vector
  score(p, q, y_lt)       — cosine similarity

v1_trie_builder.py        — wraps GCR's BuildTrie
  build_filtered_trie(G, E_q, q, τ)
    1. run GCR's BFS to get all paths P
    2. precompute query embedding u = encode_query(q, ε)
    3. for each p in P: compute score, discard if < τ
    4. call GCR's BuildTrie on P_filtered
```

**Threshold sweep procedure:**

For `τ` in `[0.1, 0.6]` step `0.05`:
- Run v1 on the 100-question validation split
- Record: FNR, trie reduction %, SIR
- Plot all three on the same axis vs. τ
- Select the highest τ where FNR < 0.05

**Important:** FNR is checked by verifying whether the gold answer path from GCR's annotation survives filtering. Make sure you have gold path annotations available for the validation split.

**Exit criterion:** Selected τ_v1, confirmed FNR < 5%, trie reduction quantified

---

### Phase 3: DCA-Trie v2 — Step-Wise Dynamic Expansion

This is the harder variant. The key challenge is per-beam trie state management.

**Module structure:**

```
v2_decoder.py             — wraps GCR's beam search loop
  BeamTrieState           — data class holding one trie per beam
    trie: current KG-Trie
    last_entity: most recently committed entity
    
  dca_beam_search(model, tokenizer, input_ids, G, E_q, q, τ, k=5)
    1. init: BeamTrieState × k, one per beam
    2. at each step t:
       a. for each beam: lookup valid tokens from beam.trie
       b. apply per-beam logit mask
       c. run beam step, get new hypotheses
       d. for each hypothesis that committed an entity e_t:
          - update query embedding: u_t = encode_query(q, y_{≤t})
          - for each (e_t, r, e') in G.neighbors(e_t):
            if score((e_t,r,e'), q, y_{≤t}) >= τ:
              add to beam.trie
```

**Critical implementation note:** When beams split/merge during beam search, you need to copy (not reference) the trie state when a beam is forked. A shallow copy of the trie node structure is enough if nodes are immutable (add-only) — verify this against GCR's trie implementation.

**Entity boundary detection:** Study GCR's decoding loop to see how it tracks when an entity has been fully generated. This is likely done via a special end-of-entity token or by tracking the trie state depth. Hook into the same mechanism.

**Exit criterion:** v2 produces correct, non-identical results across beams; faithfulness check passes

---

### Phase 4: Full Evaluation

Run the 4-system comparison (CoT, GCR, v1, v2) on WebQSP test + CWQ.

For CWQ, if cost is a concern for GPT-4o-mini synthesis: stratified sample of 500 questions per hop depth (2,000 total) is defensible.

Produce tables:
- Hits@1 and F1 by system and by hop depth
- Structural faithfulness rate per system
- SIR per system and per hop depth
- Average trie size per step per system

---

## Prototype Design

**Recommended approach: Gradio interface on Colab, with a pre-computed question bank as fallback**

This is the most practical choice because:
- Gradio works natively in Colab and generates a public shareable URL (`share=True`)
- You can demo live inference if the A100 is available, or fall back to cached results
- The visual output (reasoning chain with highlighted path) fits Gradio's component model well

---

### Prototype Architecture

```
┌─────────────────────────────────────────────┐
│                Gradio Interface             │
│                                             │
│  Question input box                         │
│  System selector (GCR / v1 / v2)            │
│  "Ask" button                               │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  Reasoning Chain (step-by-step)     │    │
│  │  Entity → Relation → Entity → ...   │    │
│  │  (each hop on its own line)         │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │  Final Answer                       │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │  Metrics: SIR | Trie Size | Hops    │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

### What the Demo Should Show

The thesis examiner wants to see the **key claim in action**: DCA-Trie produces tighter, more focused reasoning chains than GCR. Your demo should make this visible.

Design the output panel to show:
1. **The reasoning chain** as a path: `USA → ex-president → Barack Obama → spouse → Michelle Obama`
2. **Paths considered vs. paths pruned**: "GCR would have also considered 47 other paths; DCA-Trie pruned 43 of them"
3. **Final answer** from GPT-4o-mini
4. **Side-by-side mode**: let the user run the same question under GCR and v1/v2 and see the difference in trie size and pruned paths

The side-by-side comparison is the most compelling demonstration of the core contribution.

---

### Demo Implementation Sketch

```python
# prototype.py — runs in Colab

import gradio as gr
from dca_trie import DCATrieInferencePipeline

pipeline = DCATrieInferencePipeline(
    model_checkpoint="path/to/gcr-llama-3.1-8b",
    kg_path="path/to/freebase_subgraph",
    device="cuda"
)

QUESTION_BANK = {
    # pre-computed for reliability — use these as suggested examples
    "Who is the spouse of the ex-president of the USA?": {...},
    "What country was the birthplace of the director of Inception?": {...},
    # add ~10 questions across hop depths
}

def run_query(question, system):
    if question in QUESTION_BANK and not LIVE_MODE:
        result = QUESTION_BANK[question][system]
    else:
        result = pipeline.run(question, system=system)
    
    chain_display = format_chain(result["reasoning_chain"])
    pruned_info = f"Trie size: {result['avg_trie_size']:.0f} paths | "
                  f"Pruned: {result['paths_pruned']} | "
                  f"SIR: {result['sir']:.3f}"
    return chain_display, result["answer"], pruned_info

demo = gr.Interface(
    fn=run_query,
    inputs=[
        gr.Textbox(label="Question", placeholder="Ask a multi-hop question..."),
        gr.Radio(["GCR (baseline)", "DCA-Trie v1", "DCA-Trie v2"], label="System")
    ],
    outputs=[
        gr.Textbox(label="Reasoning Chain"),
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Constraint Metrics")
    ],
    examples=[
        ["Who is the spouse of the ex-president of the USA?", "DCA-Trie v2"],
        ...
    ]
)

demo.launch(share=True)
```

---

### Pre-Computed Question Bank Strategy

Curate ~15-20 questions with these properties:
- At least 4 questions at each hop depth (1, 2, 3, 4)
- Questions where GCR and DCA-Trie give **the same correct answer** (shows filtering doesn't hurt quality)
- Questions where DCA-Trie gives a **correct answer but GCR does not** (shows the benefit)
- At least one question where the pruning difference is dramatic (large SIR reduction)

Store each entry as:
```json
{
  "question": "...",
  "gcr": { "chain": [...], "answer": "...", "avg_trie_size": 127, "sir": 0.71 },
  "v1":  { "chain": [...], "answer": "...", "avg_trie_size": 23,  "sir": 0.08 },
  "v2":  { "chain": [...], "answer": "...", "avg_trie_size": 11,  "sir": 0.04 }
}
```

This means the demo works even without a GPU present during a viva, which is a significant practical advantage.

---

## Recommended File Structure

```
dca_trie/
  semantic_scorer.py     — MiniLM encoder, cosine scoring, caching
  sir_measurement.py     — SIR metric computation
  v1_trie_builder.py     — static filtering at construction
  v2_decoder.py          — beam search with per-beam trie state
  evaluation.py          — Hits@1, F1, faithfulness, SIR, trie size
  prototype.py           — Gradio demo

experiments/
  reproduce_gcr.py       — Phase 0
  measure_sir_gcr.py     — Phase 1
  threshold_sweep_v1.py  — Phase 2
  run_eval.py            — Phase 4

data/
  webqsp_val_100.json    — threshold calibration split (kept separate)
  question_bank.json     — pre-computed demo examples
```

---

What would be most useful next — working through the `v2_decoder.py` beam state logic in more detail, or designing the question bank and demo output format?

This is worth thinking about carefully. Let me engage with it honestly.

---

## What the Core Idea Actually Is

The fundamental move in DCA-Trie is:

> "The constraint oracle should know *what the question is asking*, not just *what paths structurally exist in the graph*."

That is a correct and well-motivated observation. The problem it identifies — GCR's trie is topology-blind to question intent — is real and quantifiable. But let me push on whether the *implementation* of that idea is as deep as the idea itself deserves.

---

## The Fundamental Weakness: Similarity Is Not Relevance

DCA-Trie's semantic oracle is:

```
score(p, q, y_{<t}) = cosine(MiniLM(p), MiniLM(q + y_{<t}))
```

This measures **topical overlap** between a path string and a question string. But what you actually want to know is:

> "Does this path lead toward the answer to this question?"

These are fundamentally different things.

Consider the question: *"Where was Marie Curie born?"*

- Path A: `Marie Curie → nationality → Poland`
- Path B: `Marie Curie → place_of_birth → Warsaw`

Both have high cosine similarity to the question. Only B answers it. MiniLM cannot distinguish "nationality" from "birthplace" at the semantic level that matters here because both paths are *topically about Marie Curie and a place*. The relevant signal is **inferential** — it is about the *type of relationship* the question is asking for — and cosine similarity of surface strings does not capture that.

This is not a minor issue. It is the gap between measuring *topical co-occurrence* and measuring *answer entailment*. DCA-Trie is doing the former and calling it semantic filtering. The thesis is honest about this being a heuristic, but the implications for the SIR metric and for CWQ multi-hop results are worth thinking through.

---

## What a More Principled Oracle Would Look Like

There are three directions that go deeper than cosine similarity:

**1. Answer-Type-Guided Constraint**

Every question has an implicit expected answer type: person, location, date, organization, quantity. This is deterministic from the question word (`who → person`, `where → location`, `when → date`). A constraint oracle that enforces answer-type compatibility on the *terminal entity* of each candidate path is:
- Stronger than semantic similarity (it is a hard logical constraint, not a soft score)
- More interpretable (you can explain why a path was pruned)
- Trivially composable with GCR's structural faithfulness guarantee

This alone would prune a large fraction of the irrelevant paths that DCA-Trie targets with cosine similarity, and do it more reliably. It also generalises better to new domains than a general-purpose sentence encoder.

**2. Learned Relevance Oracle (Small Trained Model)**

Instead of a heuristic scorer, train a small binary classifier:

```
input:  (path, question, y_{<t})
output: P(path is on a gold reasoning chain)
```

Training data is free — GCR's trajectories give you (path, question, gold/non-gold) triples. A 2-layer MLP over MiniLM embeddings would already be more precise than cosine similarity alone, because it learns the *decision boundary* between relevant and irrelevant paths rather than relying on a fixed reference threshold `τ_ref`. The learned oracle is also aware that the threshold itself is not symmetric — you care much more about false negatives (pruning gold paths) than false positives (admitting irrelevant ones).

**3. Uncertainty-Conditioned Constraint Tightening**

Your Chapter 1 scope section mentions this as a theoretical extension not implemented — but it is arguably the most principled approach. The intuition is:

- When the model's beam scores are spread (high uncertainty), the oracle should be *tight* — the model needs external guidance
- When the beam scores are concentrated (low uncertainty), the oracle can be *loose* — the model has a strong preference and filtering is less necessary

Formally, the admission threshold becomes `τ(t) = τ_base + λ · H(beam_scores_t)` where `H` is the entropy of the beam score distribution. This operationalises the idea that the oracle should compensate for model uncertainty rather than applying a uniform filter. It is also the natural answer to the question "how do you know when to trust the LLM's own preferences?" — you look at whether those preferences are decisive.

---

## The Long-Tail Expert Domain Problem

This is where the research becomes genuinely hard. The Freebase/WebQSP setting is arguably the *best-case scenario* for constrained decoding:

- The KG is large but clean and closed-world
- Entity and relation names are readable English
- Gold answer paths are annotated and available for calibration
- The graph is complete enough that valid paths exist for most questions

Real long-tail expert domains violate all four of these:

**Medicine (UMLS, SNOMED-CT):**
Relation names like `has_metabolite`, `contraindicated_with`, `associated_finding` are domain jargon. A general MiniLM encoder trained on web text will have low cosine similarity between "What drugs interact with warfarin?" and a path through `drug_interaction` edges because the training distribution does not contain those relation strings. The scorer breaks not because the idea is wrong but because the encoder is not calibrated for the domain.

**Legal reasoning (case law citation graphs):**
The "graph" is a citation network, not an entity-relation-entity triplet structure. Paths are chains of legal authority (precedent, overruling, distinguishing). The constraint oracle needs to understand *legal inferential structure* — which prior case supports which legal principle — not just topical overlap.

**Financial compliance (regulatory graphs):**
Rules, exceptions, definitions form a dependency graph where logical precedence matters. A path through the graph can be structurally valid but logically incoherent if a rule is applied before its definitional prerequisites are established. MiniLM cosine similarity is blind to this.

**The deeper engineering reality issue:**

Freebase has ~1.9B triplets but is static and publicly available. Expert domain KGs are typically:
- Incomplete (the open-world assumption applies — absence of an edge does not mean the fact is false)
- Dynamic (medical guidelines change, case law evolves)
- Access-controlled (hospital drug interaction databases, legal databases, financial rule systems)
- Maintained by domain specialists who did not build them for machine reasoning

The incompleteness problem is fundamental. If your constraint oracle requires that every generated token correspond to a verified KG edge, and the KG is incomplete, the oracle will block factually correct answers that do not yet have a path in the graph. This is not a calibration problem — it is a structural mismatch between the closed-world assumption your oracle requires and the open-world reality of expert knowledge.

---

## What Innovation Looks Like in This Space

Here is what I think the genuinely hard problems are if you want to make constrained decoding an engineering reality in these domains:

**Problem 1: KG incompleteness → Soft faithfulness modes**

A production oracle needs at least three operating modes:
1. Hard structural constraint (when the KG is complete and trusted — current DCA-Trie)
2. Soft semantic constraint (when the path does not exist in the KG but is semantically supported by documents in a retrieval corpus — a RAG fallback with hallucination detection)
3. Transparent failure (when neither structured nor unstructured support exists, the system refuses rather than fabricating)

The transitions between modes need to be automatic and calibrated. This is an engineering problem, not a benchmarking problem.

**Problem 2: Domain-adapted scorers without annotation**

For most expert domains, you do not have gold path annotations to calibrate a threshold. You need a scorer that is robust without domain-specific tuning. This points toward using the LLM's own token probabilities as a signal rather than an external encoder — specifically, looking at how much the LLM's confidence drops when a given path is included vs. excluded. This is related to influence estimation and is an active research area.

**Problem 3: Latency at the entity-commit boundary**

DCA-Trie v2 adds an encoder call every time the model commits an entity. On a dense Freebase node (a major country, a famous person), that entity can have thousands of neighbours. Even with caching, this can mean hundreds of encoder calls per decoding step. In production, you need either:
- A much smaller constraint model (a lookup table of pre-scored path prefixes)
- Or a different architectural choice where constraint checking is parallelised, not interleaved with decoding

**Problem 4: The tokenizer alignment problem at scale**

This problem exists in GCR too but becomes worse in expert domains where entity names are long, hyphenated, or non-English. "Acetylsalicylicacid" or "EGFR-T790M mutation" will tokenize unpredictably. A constraint trie built on character-level tokens of Freebase IDs may not be portable to clinical text entity names. A more robust oracle would operate at the span level (constraint which spans can be generated, not which subword tokens), which requires a different decoding architecture.

---

## Honest Assessment

Your thesis has a clear, well-motivated contribution within a controlled experimental scope. The innovation is real: conditioning the oracle on question semantics is better than pure structural faithfulness. The SIR metric is a useful diagnostic tool even with its limitations.

Where it is less deep: the mechanism (cosine similarity of string representations) is a reasonable engineering approximation but is not theoretically grounded in what "semantic relevance" means for question answering. The benchmarks (WebQSP, CWQ over Freebase) are the easiest version of the problem, not the hardest.

The more ambitious version of this research — the one that would be publishable as a research paper rather than a strong thesis — would combine:
1. Answer-type-guided hard constraints (logic, not heuristic)
2. A learned oracle (discriminatively trained, not threshold-tuned)
3. Graceful degradation for incomplete KGs
4. A real expert domain as validation (not just Freebase)

That is a larger scope than your current thesis. What you have is a solid, well-scoped contribution. The honest question is whether, given your timeline and available compute, you want to push further on any of these dimensions — or whether you document these as future work and deliver a clean, complete version of what you have defined.

What is your timeline and is there flexibility to extend scope in any direction?

4 months is enough to do the core work well and add one meaningful dimension that directly addresses the deepest critique. Let me give you a concrete breakdown of how I would allocate that time.

---

## The Honest Time Budget

| Phase     | Work                                  | Weeks         |
| --------- | ------------------------------------- | ------------- |
| 0         | GCR reproduction + environment setup  | 2             |
| 1         | SIR measurement on GCR baseline       | 1             |
| 2         | DCA-Trie v1 + threshold sweep         | 2             |
| 3         | DCA-Trie v2 + beam state management   | 3             |
| 4         | Full evaluation (WebQSP + CWQ sample) | 2             |
| 5         | Prototype (Gradio + question bank)    | 1             |
| 6         | Writing Chapter 4 + Chapter 5         | 3             |
| 7         | Revision + buffer                     | 2             |
| **Total** |                                       | **~16 weeks** |

That leaves you roughly 2-3 weeks of genuine slack for innovation. The question is where to spend it.

---

## The One Extension Worth Adding in 4 Months

Out of everything I described, **answer-type-guided hard constraints** are the right extension for your timeline. Here is why:

**It is cheap to implement.** It is rule-based, not learned. No training, no new data, no additional compute. You can implement it in 1 week.

**It directly addresses the deepest critique of the current approach.** The core weakness of cosine similarity scoring is that it conflates topical overlap with inferential relevance. Answer-type guidance is an actual logical constraint — not a heuristic — and it operates at the terminal entity of each candidate path, which is exactly where the oracle needs to be most precise.

**It gives you a third DCA-Trie variant for free.** You now have:
- GCR: `f(G, E_q)` — structure only
- DCA-Trie v1: `f(G, E_q, q)` — structure + static semantic similarity
- DCA-Trie v2: `f(G, E_q, q, y_{<t})` — structure + dynamic semantic similarity
- **DCA-Trie v3**: `f(G, E_q, q, type(q))` — structure + answer-type hard constraint

Or you fold it into v1/v2 as an additional filter layer, showing the complementary effect of the two mechanisms. Either way it strengthens the ablation design considerably.

**It maps onto the expert domain critique.** Answer-type compatibility is domain-transferable in a way that a MiniLM cosine threshold is not. A medical oracle that knows "which drug?" questions need paths ending in pharmaceutical compound entities does not depend on MiniLM understanding Freebase relation names. This gives your Chapter 5 a concrete, principled bridge between the thesis work and the long-tail domain problem.

---

## How Answer-Type Guidance Works

The implementation is simple. At path admission time, check whether the terminal entity type is compatible with the expected answer type of the question.

**Step 1: Classify question type.** A small rule-based or few-shot classifier maps the question to one of: `{person, location, date, organisation, quantity, other}`. Rule-based covers 80%+ of WebQSP/CWQ questions with simple heuristics (question word + head noun).

**Step 2: Map answer type to Freebase type constraint.** Freebase entities have type annotations. `people.person` for persons, `location.location` for locations, etc.

**Step 3: Filter at terminal entity.** When constructing the trie or expanding a step, discard any path whose terminal entity type does not match the expected answer type.

This is applied *before* or *in conjunction with* the MiniLM filter. The combination is:

```
admit path p if:
  1. p is structurally valid (existing KG edge)        ← GCR's condition
  2. p's terminal entity type matches answer type       ← new hard constraint
  3. score(p, q, y_{<t}) >= τ                          ← DCA-Trie's soft condition
```

Condition 2 costs nothing at inference time once entity types are indexed.

---

## The Expert Domain Argument for Your Thesis

With this addition, your Chapter 5 discussion can make a specific, concrete argument about long-tail domains rather than a general observation. The argument is:

> DCA-Trie's semantic similarity component (MiniLM cosine scoring) is a domain-general approximation that degrades when domain-specific relation vocabulary diverges from web-text training distributions. The answer-type-guided component, by contrast, is a logical constraint that is calibratable in any domain where entities have type annotations — which holds for medical ontologies (UMLS semantic types), legal taxonomies (document types, legal entities), and financial regulatory graphs (instrument types, rule categories) alike. Future work in expert domains should therefore prioritise the answer-type component over the cosine scoring component, and replace the latter with domain-adapted alternatives.

That is a specific, substantiated argument — not a generic "future work includes other domains" paragraph.

---

## What to Scope Out Firmly

Given 4 months, do not attempt:

- **New KG domain (Wikidata, UMLS):** The GCR pipeline is Freebase-specific. Adapting it to a new KG alone is a multi-week engineering task with high uncertainty.
- **Learned oracle (trained discriminator):** Training introduces a new set of hyperparameters, data requirements, and convergence risks. The payoff is not worth the risk.
- **Uncertainty-conditioned threshold:** Theoretically elegant but requires significant changes to the beam search loop and a new calibration experiment. The 2-3 week estimate is probably optimistic given the beam state management complexity you already have in v2.

Put all three of these in Chapter 5 as precisely-described future work. Examiners respect a researcher who knows what the right next steps are, even if they did not take them all.

---

## Revised Scope Statement for Your Thesis

Your scope currently says constrained decoding over Freebase-based KGQA only. I would revise the contribution claim slightly to:

> DCA-Trie contributes three things: (1) a formal characterisation of the permissiveness problem and the SIR metric to measure it; (2) two semantic filtering mechanisms — cosine-similarity-based (soft) and answer-type-guided (hard) — as complementary oracle components; (3) a principled analysis of how these components differ in their domain-transferability properties, establishing the answer-type component as the more robust foundation for future work in expert domains.

This frames the same work more ambitiously without requiring additional experiments beyond the answer-type addition.

---

## Concrete Next Step

Before writing any code, spend one day doing this:

1. Pull the WebQSP question set and manually annotate 20 questions with expected answer type (person / location / date / other)
2. Look at GCR's gold path annotations for those 20 questions
3. Check: what fraction of GCR's admitted paths at the terminal hop have the wrong entity type?

If that number is high (I expect it to be 40-70% on location and person questions), you have empirical motivation for the answer-type component that comes from your own data, not just from theory. That number goes directly into your Chapter 3 motivation section.