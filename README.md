# DCA-Trie: Dynamic Context-Aware Trie

**Knowledge Graph-Constrained LLM Generation with Semantic Filtering**

Final year project by Bernard Kirk Adjanor Katamanos, Erica Amonor, Joseph Osei Nyarko, Jessica Afua Etornam Nsafoah. Supervised by Dr. Eric Affum, UMaT.

## Overview

DCA-Trie extends [Graph-Constrained Reasoning (GCR)](https://github.com/rmanluo/graph-constrained-reasoning) with semantically-aware path filtering. The core idea: GCR's KG-Trie admits all structurally valid paths, but many are irrelevant to the question. DCA-Trie scores each path against the question using a sentence transformer and prunes irrelevant ones.

### Repository Structure

```bash
vendor/gcr/              #   SUBMODULE: exact original GCR (RManLuo/graph-constrained-reasoning)
  git submodule update --init  # clone the original

gcr/                     #   VENDORED: GCR with import path changes (src. -> gcr.src.)
├── src/                 #   Core GCR library (trie, decoding, model wrappers, utils)
├── workflow/            #   GCR pipeline scripts (path generation, inductive reasoning)
├── scripts/             #   Shell wrappers for GCR pipeline
├── accelerate_configs/  #   DeepSpeed / multi-GPU configs
└── resources/           #   Figures from the GCR paper

dca_trie/                #   CONTRIBUTION: DCA-Trie implementation
├── semantic_scorer.py   #   MiniLM encoder + cosine scoring
├── sir_measurement.py   #   Semantic Irrelevance Ratio metric
├── v1_trie_builder.py   #   Static semantic filtering at trie construction
├── mid_resolver.py      #   Freebase MID-to-name resolver
└── test_mini_freebase.py#   Test data for local development

experiments/             # DCA-Trie experiment scripts
├── threshold_sweep_v1.py

docs/                    # Documentation
├── RESEARCH_PLAYBOOK.md
├── DCA_TRIE_HANDBOOK.md
├── CODEBASE_OVERVIEW.md
├── LLM_FOUNDATIONS_GUIDE.md
├── ANSWER_TYPE_CONSTRAINTS_RESEARCH.md
└── ...
```

### Phases

| Phase | What | Status |
|-------|------|--------|
| 0 | Environment + GCR baseline reproduction | Notebook ready |
| 1 | SIR measurement | Notebook + module ready |
| 2 | DCA-Trie v1 — static semantic filtering | Module tested on local data |
| 3 | DCA-Trie v2 — step-wise dynamic expansion | Design complete |
| 4 | Full evaluation | Pending |
| 5 | Gradio prototype | Pending |
| v3 | Semantic backtracking (extension) | Design complete |

## Quick Start

```bash
# Install dependencies
poetry install

# Run threshold sweep on test data
poetry run python experiments/threshold_sweep_v1.py --data test

# Run on WebQSP (requires HF token + GPU)
poetry run python experiments/threshold_sweep_v1.py --data webqsp
```

## References

- Luo et al. *Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models.* ICML 2025.
- Li et al. *Decoding on Graphs: Faithful and Sound Reasoning on Knowledge Graphs.* ACL 2025.
- Banerjee et al. *CRANE: Reasoning with constrained LLM generation.* ICML 2025.
