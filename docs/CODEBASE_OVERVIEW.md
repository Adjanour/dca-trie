# Codebase Overview

> **📦 gcr/** = Vendored GCR baseline (Luo et al., ICML 2025)
> **🆕 dca_trie/** = DCA-Trie contribution (this project)

## Directory Structure

```bash
dca-trie/
├── 📦 gcr/                        # BASELINE: GCR (RManLuo/Graph-constrained-Reasoning)
│   ├── src/                       #   Core GCR library
│   │   ├── trie.py                #     Trie data structures
│   │   ├── graph_constrained_decoding.py  # Logit masking
│   │   ├── qa_prompt_builder.py   #     Prompt templates
│   │   ├── llms/                  #     Model wrappers
│   │   └── utils/                 #     Graph utils, QA metrics
│   ├── workflow/                  #   GCR pipeline entry points
│   ├── scripts/                   #   Shell wrappers
│   ├── accelerate_configs/        #   DeepSpeed configs
│   └── resources/                 #   GCR paper figures
│
├── 🆕 dca_trie/                   # CONTRIBUTION: DCA-Trie
│   ├── semantic_scorer.py
│   ├── sir_measurement.py
│   ├── v1_trie_builder.py
│   └── test_mini_freebase.py
│
├── 🆕 experiments/                # CONTRIBUTION: Experiments
│   └── threshold_sweep_v1.py
│
├── 🆕 docs/                       # CONTRIBUTION: Documentation
│   ├── LLM_FOUNDATIONS_GUIDE.md
│   ├── ANSWER_TYPE_CONSTRAINTS_RESEARCH.md
│   ├── DCA_TRIE_HANDBOOK.md
│   └── ...
│
├── ATTRIBUTION.md                 # Attribution for vendored GCR code
├── pyproject.toml                 # Poetry dependency management
├── README.md
└── .env.example
```

---

## Two-Step Inference Pipeline

GCR's inference is split into two separate stages:

### Step 1: Graph-Constrained Decoding

**Entry point:** `gcr/workflow/predict_paths_and_answers.py`

**What it does:**

1. Loads questions from HuggingFace (`rmanluo/RoG-webqsp` or `rmanluo/RoG-cwq`)
2. For each question, builds a **KG-Trie** containing all valid paths within L hops of the question entities
3. Runs the KG-specialized LLM with **beam search**, using the trie to mask invalid tokens
4. Generates K reasoning paths and extracts answer hypotheses
5. Evaluates Hits@1, F1 on the extracted answers

**Output:** `results/GenPaths/{dataset}/{model}/{split}/{prompt_mode}-{gen_mode}-k{K}-index_len{L}/predictions.jsonl`

### Step 2: Graph Inductive Reasoning

**Entry point:** `gcr/workflow/predict_final_answer.py`

**What it does:**

1. Takes the reasoning paths from Step 1 as input
2. Passes them to a general LLM (e.g., GPT-4o-mini) as context
3. The LLM reasons over multiple paths to produce the final answer
4. Evaluates accuracy

**Output:** `results/KGQA/{dataset}/{model}/{split}/.../predictions.jsonl`

---

## Data Flow (Step 1)

```
Question (from HF dataset)
    │
    ▼
┌──────────────────────────────┐
│  PathGenerationWithAnswer    │
│  PromptBuilder.process_input │
│                              │
│  1. Build graph (nx.DiGraph) │
│     from question_dict["graph"]  │
│                              │
│  2. DFS from q_entity        │
│     up to index_path_length  │
│     hops → list of paths     │
│                              │
│  3. Tokenize each path       │
│     → list of token IDs      │
│                              │
│  4. Build MarisaTrie         │
│     from tokenized paths     │
└──────────┬───────────────────┘
           │ input query + trie
           ▼
┌──────────────────────────────┐
│  GraphConstrainedDecoding    │
│  Model.generate_sentence     │
│                              │
│  1. Tokenize input query     │
│  2. Call model.generate()    │
│     with prefix_allowed_     │
│     tokens_fn = gcr.         │
│     allowed_tokens_fn        │
│                              │
│  3. At each step:            │
│     - Check if inside        │
│       <PATH>...</PATH>       │
│     - If yes: look up valid  │
│       tokens from trie       │
│     - Set invalid tokens     │
│       logits to -inf         │
│     - Continue generation    │
└──────────┬───────────────────┘
           │ K generated paths
           ▼
┌──────────────────────────────┐
│  Evaluation                  │
│  eval_path_result_w_ans()    │
│  → Hits@1, F1, Path F1      │
└──────────────────────────────┘
```

---

## Model Registry

Defined in `gcr/src/llms/__init__.py`:

```python
registed_language_models = {
    'gpt': ChatGPT,                          # OpenAI API models
    'others': HfCausalModel,                  # Generic HuggingFace models
    'gcr': GraphConstrainedDecodingModel,     # GCR model with trie constraints
}
```

The `get_registed_model(model_name)` function checks if `'gpt'`, `'gcr'`, or `'others'` is in the model name (case-insensitive) and returns the appropriate class.

Since the model name `"GCR-Meta-Llama-3.1-8B-Instruct"` contains `"gcr"`, it matches `GraphConstrainedDecodingModel`.

---

## Prompt Builder Hierarchy

```
GraphConstrainedPromptBuilder        (base: builds graphs, creates trie, processes input)
    ├── PathGenerationPromptBuilder   (CoT-style: generates reasoning paths)
    ├── JointReasoningPromptBuilder   (generates <PATH>...</PATH> + answer jointly)
    │   └── PathGenerationWithAnswerPromptBuilder  (used in Step 1)
    ├── RetrievalPromptBuilder        (separate entity/relation/triple retrieval)
    └── PromptBuilder                 (used in Step 2: formats paths+question for final answer)
```

Key distinction:

- **Step 1** uses `PathGenerationWithAnswerPromptBuilder` which produces paths wrapped in `<PATH>...</PATH>` tokens and extracts answers from them
- **Step 2** uses `PromptBuilder` which formats candidate paths as context for a general LLM to reason over

---

## Key Design Decisions

1. **Beam search with diversity:** Uses `group-beam` mode which partitions beams into groups and applies a diversity penalty to ensure diverse reasoning paths.

2. **Trie as constraint oracle:** The MarisaTrie is a memory-efficient prefix tree. At each decoding step, the trie returns the set of valid next token IDs given the prefix generated so far.

3. **Constrained region only:** Constraint only applies between `<PATH>` and `</PATH>` tokens. Outside this region, the model generates freely.

4. **Pre-built graph index:** During training, the shortest-path index is built offline. During inference, the graph is typically built on-the-fly from the dataset (which includes the graph triples for each question).

5. **Dataset format:** Each entry in the HuggingFace dataset contains:
   - `question`: the natural language question
   - `q_entity`: list of topic entity IDs in the KG
   - `a_entity`: list of answer entity IDs
   - `graph`: list of `[head, relation, tail]` triplets (KB subgraph)
   - `answer`: list of answer strings
   - `paths` (optional): pre-computed paths
