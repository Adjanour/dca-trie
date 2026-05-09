# DCA-Trie Research Playbook

## From Zero to Thesis — Complete Step-by-Step Guide

**Author:** Bernard Kirk Adjanor Katamanos
**Date:** May 2026

This document tells you **exactly** what to do, in what order, what to know before each step, and how to fix things when they break. Follow it sequentially.

---

## Table of Contents

1. [Environment Setup (Do This First)](#1-environment-setup-do-this-first)
2. [Phase Overview Table](#2-phase-overview-table)
3. [Phase 0: GCR Baseline Reproduction](#3-phase-0-gcr-baseline-reproduction)
4. [Phase 1: SIR Measurement](#4-phase-1-sir-measurement)
5. [Phase 2: DCA-Trie v1 — Threshold Sweep](#5-phase-2-dca-trie-v1--threshold-sweep)
6. [Phase 3: DCA-Trie v2 — Beam Search with Dynamic Expansion](#6-phase-3-dca-trie-v2--beam-search-with-dynamic-expansion)
7. [Phase 3b: DCA-Trie v3 — Answer-Type Constraints](#7-phase-3b-dca-trie-v3--answer-type-constraints)
8. [Phase 4: Full Evaluation](#8-phase-4-full-evaluation)
9. [Phase 5: Prototype Demo](#9-phase-5-prototype-demo)
10. [Phase 6: Thesis Writing](#10-phase-6-thesis-writing)
11. [Appendices](#11-appendices)

---

## 1. Environment Setup (Do This First)

### 1.1 What You Need to Know

- **Python version**: This project needs Python 3.10 (not 3.14, not 3.11 — specifically 3.10+). GCR's dependencies (PyTorch, DeepSpeed, bitsandbytes, etc.) are tested on 3.10.
- **Poetry**: The project uses Poetry for dependency management. Poetry creates an isolated virtual environment so your system Python stays clean.
- **PEP 668**: Modern Linux (Ubuntu 24.04+) blocks `pip install --system` to protect system packages. This is why `pip install` fails with "externally managed environment". Poetry bypasses this because it creates its own venv.
- **DeepSpeed + bitsandbytes**: These are tricky to install. They need your system's CUDA version to match PyTorch's CUDA version. On CPU-only machines, they still install (just won't use GPU).
- **marisa-trie**: Requires a C++ compiler and `libtool`. On Ubuntu: `apt install build-essential libtool`.

### 1.2 Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential libtool python3.10 python3.10-venv python3.10-dev

# For Colab: these are already installed
```

### 1.3 Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
# Add to PATH (put this in ~/.bashrc):
export PATH="$HOME/.local/bin:$PATH"
poetry --version  # Should show 1.8.x or 2.x
```

### 1.4 Configure Poetry for This Project

```bash
cd /path/to/graph-constrained-reasoning

# Tell Poetry to use Python 3.10 specifically:
poetry env use python3.10
# Or if python3.10 is not in PATH:
# poetry env use /usr/bin/python3.10

# Verify:
poetry env info  # Shows Python 3.10 + venv path
```

### 1.5 Install All Dependencies

```bash
# This reads pyproject.toml and poetry.lock, installs everything:
poetry install

# This will take 10-30 minutes depending on your internet.
# It downloads: torch, transformers, deepspeed, bitsandbytes,
# sentence-transformers, scikit-learn, etc.
```

**If installation fails mid-way:**

```bash
# Try with more verbose output:
poetry install -vvv

# Common fixes:
rm -rf ~/.cache/pip  # Clear pip cache
poetry install --no-cache  # Skip cache

# If bitsandbytes or deepspeed fail (they need CUDA):
poetry install --without dev  # Skip optional groups
```

### 1.6 Activate the Virtual Environment

```bash
poetry shell
# OR:
source $(poetry env info --path)/bin/activate

# Verify:
which python
python --version  # Should show 3.10.x
```

### 1.7 Verify Key Packages

Run this in the activated env:

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
from sentence_transformers import SentenceTransformer; print('Sentence-Transformers: OK')
import marisa_trie; print('marisa-trie: OK')
from datasets import load_dataset; print('Datasets: OK')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import deepspeed; print('DeepSpeed: OK')
"
```

### 1.8 Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY = sk-your-key
#   HF_TOKEN = hf_your-token
```

`.env.example` already exists in the repo. HF_TOKEN is needed to download GCR's model from HuggingFace (it's a gated model).

### 1.9 Verify the Project Code Loads

```bash
cd /path/to/graph-constrained-reasoning

poetry shell

# Test DCA-Trie imports:
python -c "from dca_trie.semantic_scorer import SemanticScorer; print('semantic_scorer: OK')"
python -c "from dca_trie.sir_measurement import SIRMeasurer; print('sir_measurement: OK')"
python -c "from dca_trie.v1_trie_builder import V1TrieBuilder; print('v1_trie_builder: OK')"
python -c "from dca_trie.mid_resolver import MidResolver; print('mid_resolver: OK')"

# Test GCR imports:
python -c "from gcr.src.trie import MarisaTrie, Trie; print('trie: OK')"
python -c "from gcr.src.utils.graph_utils import build_graph, dfs, get_truth_paths; print('graph_utils: OK')"
```

**All of the above should print "OK"** before proceeding.

### 1.10 Environment Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pip install` fails with "externally managed environment" | PEP 668 | Don't use `pip install`. Use `poetry add <package>` or activate the shell first |
| `poetry install` fails on `bitsandbytes` | Missing CUDA | `poetry install` still works (installs CPU fallback). If it truly fails, remove bitsandbytes from pyproject.toml temporarily |
| `marisa-trie` fails to build | Missing libtool | `sudo apt install build-essential libtool` |
| `ModuleNotFoundError: gcr` | Wrong working dir | Always run from the repo root, or `cd /path/to/graph-constrained-reasoning` |
| `ModuleNotFoundError: dca_trie` | Not installed | Run `poetry install` from repo root |
| CUDA not available | No NVIDIA GPU | That's expected on your local machine. Use Colab for GPU phases. The CPU-only code (Phases 1-2, MID resolver) works fine locally |
| DeepSpeed import error | CUDA version mismatch | DeepSpeed only needed for GCR inference on GPU. Comment out deepspeed from pyproject.toml for CPU-only work |

### 1.11 Colab Setup (for GPU Phases)

When you need GPU (Phases 0, 3, 4), use Google Colab:

1. Go to https://colab.research.google.com/
2. Runtime → Change runtime type → T4 GPU or A100
3. In the first cell:

```python
# Clone the repo
!git clone https://github.com/YOUR_GITHUB/graph-constrained-reasoning.git
%cd graph-constrained-reasoning

# Install Poetry
!curl -sSL https://install.python-poetry.org | python3 -
!export PATH="$HOME/.local/bin:$PATH"

# Install deps
!poetry install

# Set up HF token
import os
os.environ["HF_TOKEN"] = "hf_your-token"

# Activate
!poetry run python your_script.py
```

The notebooks (`phase0_reproduce_baseline.ipynb`, `phase1_sir_measurement.ipynb`) already handle this.

---

## 2. Phase Overview Table

| # | Phase | What You Do | Hardware | Dependencies | Est. Time | Status |
|---|-------|------------|----------|-------------|-----------|--------|
| 0 | **GCR Baseline Reproduction** | Run GCR inference on WebQSP, verify Hits@1 matches paper | GPU (A100, 40GB) | Phase 1 env setup | 1-2 weeks | Notebook ready |
| 1 | **SIR Measurement** | Measure how permissive GCR's trie is using SIR metric | CPU only | Phase 0 output | 1 week | Module exists, needs MID integration |
| 2 | **DCA-Trie v1 Threshold Sweep** | Sweep tau, pick best threshold, measure FNR/reduction | CPU only | Phase 1 modules | 1-2 weeks | Script ready, tested on synthetic data |
| 3 | **DCA-Trie v2 Dynamic Expansion** | Implement per-beam trie state during decoding | GPU (T4/A100) | Phase 0 env + Phase 2 tau | 2-3 weeks | Design complete, not implemented |
| 3b | **DCA-Trie v3 Answer-Type Constraints** | Add hard answer-type filter on terminal entities | CPU alone, GPU for eval | Phase 2 | 1 week | Research complete, not implemented |
| 4 | **Full Evaluation** | 4-system comparison on WebQSP + CWQ | GPU | Phases 0-3 complete | 2 weeks | Not started |
| 5 | **Prototype/Gradio Demo** | Interactive demo for thesis viva | GPU (or pre-computed) | Phase 4 output | 1 week | Design documented |
| 6 | **Thesis Writing** | Write Chapters 3-5, intro, conclusion | Laptop | All phases | 3-4 weeks | Outline exists |

**Critical path**: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

---

## 3. Phase 0: GCR Baseline Reproduction

### 3.1 What You Need to Know

Before starting this phase, understand:
- **What is GCR?** GCR (Graph-Constrained Reasoning) uses a KG-Trie to constrain LLM decoding so generated paths are faithful to the knowledge graph. The trie is built from a DFS of all paths within L hops of the question entity.
- **Two-step pipeline**: Step 1 generates K reasoning paths using a KG-specialized LLM. Step 2 feeds those paths to GPT-4o-mini to produce the final answer.
- **WebQSP dataset**: 1,737 test questions over Freebase. Questions have annotated gold paths.
- **Reported numbers**: Hits@1 = 92.6, F1 = 91.8 on WebQSP. Your target: within 2% of these.

### 3.2 Prerequisites

- [ ] Colab Pro (for A100 40GB — T4 will work but may OOM)
- [ ] HuggingFace token with access to `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` (gated model — request access on HF)
- [ ] OpenAI API key for GPT-4o-mini (Step 2)
- [ ] `phase0_reproduce_baseline.ipynb` ready

### 3.3 Steps

#### Step 0.1: Open the Notebook

Open `phase0_reproduce_baseline.ipynb` in Colab. Follow the cells sequentially.

#### Step 0.2: Verify GPU

The first code cell checks `torch.cuda.is_available()`. It should say:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
VRAM: 40.0 GB
```

If not: Runtime → Change runtime type → A100.

#### Step 0.3: Install Dependencies

The notebook runs `poetry install`. This downloads:
- PyTorch with CUDA (2.4.x)
- Transformers 4.44
- DeepSpeed 0.14.x
- bitsandbytes 0.43.x
- All other deps in pyproject.toml

**Time**: ~15-30 minutes depending on Colab's internet speed.

#### Step 0.4: Install System Packages for marisa-trie

```bash
!apt-get update && apt-get install -y build-essential libtool
```

#### Step 0.5: Set HF_TOKEN

```python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"
```

Also set `OPENAI_API_KEY` in `.env`.

#### Step 0.6: Run GCR Step 1 (Path Generation)

This runs `gcr/workflow/predict_paths_and_answers.py` with:
- Model: `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct`
- Dataset: WebQSP test set (or a subset first)
- k=5 beams, index_len=2

This is the most compute-intensive step. With A100:
- Full WebQSP (1,737 questions): ~4-6 hours
- 100-question subset: ~20-30 minutes

**Always start with a subset** (e.g., `--subset 100`) to validate the pipeline works.

#### Step 0.7: Run GCR Step 2 (Final Answer)

Takes the paths from Step 1, passes to GPT-4o-mini for final answer reasoning.

**Cost**: ~$1-2 for full WebQSP with GPT-4o-mini.

#### Step 0.8: Verify Hits@1

Compare against reported values:
```
Expected: Hits@1 >= 91.0, F1 >= 90.0
Your run should be within 2% of these.
```

#### Step 0.9: Snapshot Package Versions

Save your environment for the reproducibility section of your thesis:

```python
!pip freeze > requirements_snapshot.txt
```

This goes in your thesis appendix.

### 3.4 Exit Criterion

- [ ] WebQSP Hits@1 >= 91% on your run
- [ ] You understand the GCR pipeline (you can trace: question → graph → DFS → trie → constrained decoding → paths → answer)
- [ ] `requirements_snapshot.txt` saved

### 3.5 If It Fails

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| OOM (out of memory) | T4 16GB too small | Switch to A100 runtime. If A100 not available, reduce batch size in `accelerate_configs/` |
| Model not found | No HF token / no access | Request access at https://huggingface.co/rmanluo/GCR-Meta-Llama-3.1-8B-Instruct |
| DeepSpeed error | CUDA version mismatch | Install with: `!pip install deepspeed==0.14.2 --no-build-isolation` |
| Steps take too long | Too many questions | Always test with --subset 100 first |
| `path_to_string` missing | Import path wrong | The code uses `from gcr.src.utils import path_to_string` |

---

## 4. Phase 1: SIR Measurement

### 4.1 What You Need to Know

- **SIR (Semantic Irrelevance Ratio)** measures how "permissive" a constraint oracle is:
  ```
  SIR(q, t) = 1 - max_{p in P_valid(t)} cos(MiniLM(p), MiniLM(q))
  ```
- SIR = 0: all admitted paths are relevant to the question
- SIR = 1: no admitted path is relevant
- You measure SIR on GCR's **existing output** (from Phase 0). No DCA-Trie needed.
- You expect SIR to increase with hop depth (deeper paths are noisier).
- **MID problem**: WebQSP graphs contain Freebase MIDs (`m.0k8nh0b`) which MiniLM cannot score. You need `MidResolver` to convert them to readable names.

### 4.2 Prerequisites

- [ ] Phase 0 complete (GCR output exists)
- [ ] `dca_trie/semantic_scorer.py` exists and works
- [ ] `dca_trie/sir_measurement.py` exists and works
- [ ] `dca_trie/mid_resolver.py` exists and works
- [ ] `poetry shell` activated (Python 3.10)

### 4.3 Conceptual Background

**Why SIR matters for your thesis:**

Your thesis argument is: *GCR's trie admits too many irrelevant paths, especially at deeper hops. DCA-Trie fixes this by filtering semantically.*

SIR gives you the numbers to prove the "too many irrelevant paths" claim. The SIR measurement on GCR baseline goes **directly into Chapter 4** of your thesis as the motivation table.

**What you're looking for:**
- Depth 1 paths: SIR should be low (~0.2-0.3), because 1-hop paths from q_entity are strongly topical
- Depth 2 paths: SIR should be higher (~0.4-0.6), because paths start to diverge
- Depth 3+ paths: SIR should be highest (~0.6-0.8), because the trie admits many irrelevant continuations

### 4.4 Steps

#### Step 1.1: Load GCR Output

GCR saves predictions to:
```
results/GenPaths/{dataset}/{model}/{split}/{prompt_mode}-{gen_mode}-k{K}-index_len{L}/predictions.jsonl
```

Each line is a JSON object with:
```json
{
  "id": "...",
  "question": "...",
  "predictions": [...],    // K generated path strings
  "ground_truth": [...],   // gold path strings  
  "graph": [...],          // KG triples
  "answers": [...],
  "q_entity": [...],
  "a_entity": [...]
}
```

If Phase 0 output doesn't exist yet, you can still prototype SIR measurement using:
- The synthetic test data in `dca_trie/test_mini_freebase.py`
- Or load WebQSP directly from HuggingFace (no GPU needed for data loading)

#### Step 1.2: Set Up MID Resolution

```python
from dca_trie.mid_resolver import MidResolver
from datasets import load_dataset

dataset = load_dataset("rmanluo/RoG-webqsp", split="test")
questions = list(dataset)

resolver = MidResolver(cache_path="data/mid_to_name.json")
resolver.build_from_dataset(questions)
print(resolver.coverage(questions))
# Target: ~62% coverage from graph triples alone
```

#### Step 1.3: Patch path_to_string

The MID resolver needs to intercept `path_to_string()` so that all paths use readable names instead of MIDs:

```python
import gcr.src.utils as gcr_utils

_orig = gcr_utils.path_to_string

def _resolved_path_to_string(path):
    raw = _orig(path)
    return resolver.resolve_path(raw)

gcr_utils.path_to_string = _resolved_path_to_string
```

This patch is already implemented in `experiments/threshold_sweep_v1.py` (lines 266-275).

#### Step 1.4: Measure SIR on GCR Tries

```python
from dca_trie.semantic_scorer import SemanticScorer
from dca_trie.sir_measurement import SIRMeasurer

scorer = SemanticScorer(device="cpu")
measurer = SIRMeasurer(scorer)

for question in questions[:10]:  # Start small
    # Build the trie (same way GCR does)
    from gcr.src.utils.graph_utils import build_graph, dfs
    g = build_graph(question["graph"])
    all_paths = dfs(g, question["q_entity"], max_length=2)
    
    # Convert paths to strings (uses patched path_to_string)
    from gcr.src.utils import path_to_string
    path_strs = [path_to_string(p) for p in all_paths]
    
    # Measure SIR per hop depth
    results = measurer.measure_per_hop(
        StringTrie(path_strs),  # see sweep script for StringTrie helper
        question["question"]
    )
    print(f"Depth 1 SIR: {results[1]['sir']:.3f} ({results[1]['num_paths']} paths)")
    print(f"Depth 2 SIR: {results[2]['sir']:.3f} ({results[2]['num_paths']} paths)")
```

#### Step 1.5: Aggregate Results

For the thesis, produce this table:

| Hop Depth | Avg. SIR | Avg. Trie Size | Num. Questions |
|-----------|----------|----------------|----------------|
| 1 | 0.32 | 8.4 | 1,737 |
| 2 | 0.55 | 47.2 | 1,737 |
| 3 | 0.71 | 183.6 | 1,737 |
| 4 | 0.83 | 512.3 | 1,737 |

**Expected finding**: SIR increases with hop depth, confirming the thesis hypothesis.

#### Step 1.6: Plot SIR vs. Hop Depth

Create a matplotlib plot showing the upward trend. This goes in Chapter 4.

```python
import matplotlib.pyplot as plt

depths = [1, 2, 3, 4]
sir_values = [0.32, 0.55, 0.71, 0.83]

plt.figure(figsize=(6, 4))
plt.plot(depths, sir_values, 'o-', linewidth=2)
plt.xlabel("Hop Depth")
plt.ylabel("Average SIR")
plt.title("GCR Baseline: SIR Increases with Hop Depth")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.savefig("docs/figures/sir_vs_depth.png", dpi=300)
```

### 4.5 Exit Criterion

- [ ] SIR vs. hop depth table generated for GCR baseline
- [ ] Average trie size per step quantified
- [ ] Plot saved for thesis
- [ ] You understand: *this is why DCA-Trie is needed*

### 4.6 If It Fails

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| MIDs everywhere in paths | MID resolver not covering enough | `build_from_fb_names_file()` with the 248MB Freebase names dump |
| SIR values all near 1.0 | MIDs unresolved — MiniLM scores garbage | Check: `path_to_string` patch working? Print a sample path |
| SIR values all near 0.0 | Paths too short or test data too simple | Use real WebQSP data, not just test_mini_freebase |
| No Phase 0 output | GPU required | You can still measure SIR on the raw trie (Path 2 from get_graph_index), not on generated paths. This gives you trie-level SIR (not generation-level) |

---

## 5. Phase 2: DCA-Trie v1 — Threshold Sweep

### 5.1 What You Need to Know

- **DCA-Trie v1** filters paths *at trie construction time*, before decoding starts.
- **τ (tau)**: the cosine similarity threshold. Paths with score < τ are pruned.
- **FNR (False Negative Rate)**: fraction of questions where ALL gold paths are pruned. Must stay below 5%.
- You want: the **highest τ** where FNR < 5%, maximizing pruning while not losing answers.
- The sweep script already exists at `experiments/threshold_sweep_v1.py`.

### 5.2 Prerequisites

- [ ] Phase 1 modules working (SemanticScorer, SIRMeasurer, MidResolver, V1TrieBuilder)
- [ ] `poetry shell` activated
- [ ] Understanding of: cosine similarity, FNR, precision-recall tradeoff

### 5.3 Steps

#### Step 2.1: Run Threshold Sweep on Test Data First

Always validate with synthetic data:

```bash
poetry run python experiments/threshold_sweep_v1.py --dataset test
```

Expected output:
```
  τ=0.10  FNR=0.000  reduction=0.0%    filtered=5    original=5    SIR=0.0000
  τ=0.15  FNR=0.000  reduction=0.0%    filtered=5    original=5    SIR=0.0000
  ...
  τ=0.50  FNR=0.000  reduction=68.0%   filtered=2    original=5    SIR=0.1234
  τ=0.55  FNR=0.200  reduction=72.0%   filtered=1    original=5    SIR=0.0876
  τ=0.60  FNR=0.400  reduction=84.0%   filtered=0    original=5    SIR=0.0000
```

This should finish in under 1 minute (5 questions, no HuggingFace data loading).

#### Step 2.2: Run on WebQSP (100 Questions)

```bash
poetry run python experiments/threshold_sweep_v1.py --dataset webqsp --num 100 --tau_max 0.50
```

This:
1. Loads 100 questions from HuggingFace `rmanluo/RoG-webqsp`
2. Loads the GCR tokenizer (needed for trie building)
3. Builds MID resolver from the dataset
4. Sweeps τ from 0.10 to 0.50 (step 0.05)
5. Reports FNR, trie reduction, SIR per τ
6. Selects the best τ (highest with FNR < 5%)
7. Saves results to `data/threshold_sweep_results.json`

**Time**: ~15-30 minutes (MiniLM encoding per path × sweep × 100 questions).

#### Step 2.3: Run Full WebQSP

```bash
poetry run python experiments/threshold_sweep_v1.py --dataset webqsp --num -1
```

`--num -1` means all questions. This may take 2-4 hours on CPU.

#### Step 2.4: Run on CWQ

```bash
poetry run python experiments/threshold_sweep_v1.py --dataset cwq --num 100
```

CWQ questions tend to be deeper, so you may get better pruning (higher reduction).

#### Step 2.5: Interpret Results

The script prints a summary like:
```
Selected τ = 0.35 (highest with FNR < 5%)
  FNR: 0.040 (4/100)
  Trie reduction: 58.3%
  Avg filtered size: 18 (from 43)
  Avg SIR: 0.3124
```

**Record these for your thesis:**
| τ | FNR | Trie Reduction | Avg Filtered Size | Avg SIR |
|---|-----|---------------|-------------------|---------|
| 0.10 | 0.000 | 5% | 41 | 0.55 |
| 0.20 | 0.010 | 22% | 34 | 0.47 |
| 0.30 | 0.030 | 45% | 24 | 0.38 |
| **0.35** | **0.040** | **58%** | **18** | **0.31** |
| 0.40 | 0.070 | 67% | 14 | 0.27 |

The selected τ (0.35 in this example) is what you use for v2 and evaluation.

### 5.4 Exit Criterion

- [ ] Selected τ_v1 with FNR < 5% on WebQSP
- [ ] Trie reduction quantified (expected: 40-60%)
- [ ] Results saved to JSON for thesis table
- [ ] Understanding: *v1 filters statically, before decoding. This is the safe baseline.*

### 5.5 If It Fails

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| No τ with FNR < 5% | τ range too restrictive | Set `--tau_max 0.30` and `--tau_min 0.05` to find the sweet spot |
| HuggingFace connection error | No internet / HF token missing | Set `HF_TOKEN` in `.env`. Or use `--dataset test` to verify the sweep logic works |
| Tokenizer download hangs | HuggingFace rate limit | Use `HF_TOKEN` in `.env` or `os.environ["HF_TOKEN"]` |
| All paths pruned at τ=0.10 | MID resolution not working | Check: `resolver.resolve_path()` actually converts MIDs? Print a sample |

---

## 6. Phase 3: DCA-Trie v2 — Beam Search with Dynamic Expansion

### 6.1 What You Need to Know

**This is the hardest phase.**

- **v2** does NOT filter at trie construction time. Instead, it starts with an empty/small trie and **expands it dynamically** as the model commits entities.
- When the model generates `President → (commits entity) Barack Obama`, v2 looks up all of Obama's neighbors in the KG, scores them against the question, and adds only relevant ones to the beam's trie.
- **One trie per beam**: Beam search maintains K hypotheses. Each hypothesis has its own trie state because each hypothesis commits different entities.
- **Beam state management**: When HuggingFace's beam search copies/fork hypotheses, you must copy the trie too.
- **Entity boundary detection**: You need to detect when the model has finished generating an entity name (so you know when to expand).

### 6.2 Prerequisites

- [ ] Phase 0 environment working (GPU, DeepSpeed, GCR model loaded)
- [ ] Phase 2 τ value selected (e.g., τ = 0.35)
- [ ] Deep understanding of GCR's decoding loop (`gcr/src/graph_constrained_decoding.py`)
- [ ] Deep understanding of HuggingFace's `prefix_allowed_tokens_fn` mechanism

### 6.3 Conceptual Background

**How GCR's decoding works (study this first):**

1. `GraphConstrainedDecodingModel.generate_sentence()` calls `self.model.generate()`
2. It passes `prefix_allowed_tokens_fn=gcr.allowed_tokens_fn`
3. At each decoding step, HF calls `gcr.allowed_tokens_fn(batch_id, sent)` where:
   - `batch_id` = which beam (0 to K-1)
   - `sent` = full token sequence so far (input + generated)
4. The function looks up the trie for valid next tokens, returns them
5. HF masks logits of tokens not in the valid list

**What v2 must do differently:**

Instead of a single trie shared across all beams, v2 maintains a list of `K` tries. When `allowed_tokens_fn` is called for beam `batch_id`, it uses that beam's private trie. When the beam commits an entity, v2 expands that beam's trie with the entity's scored neighbors.

**The beam copy problem:**

Between decoding steps, HF internally reorders beams (keeps top K out of K×V candidates). The `batch_id` for a hypothesis can change between steps. This means you **cannot rely on `batch_id` stability**.

**Solution**: Track trie state by hypothesis content (token IDs), not by batch index.

### 6.4 Implementation Steps

#### Step 3.1: Study GCR's Decoding Code

Read these files carefully:
- `gcr/src/graph_constrained_decoding.py` — `allowed_tokens_fn()`, `GraphConstrainedDecoding` class
- `gcr/src/llms/graph_constrained_decoding_model.py` — `generate_sentence()`, how it calls `model.generate()`
- `gcr/src/qa_prompt_builder.py` — `get_graph_index()`, how the trie is built

Understand:
- How `<PATH>` and `</PATH>` tokens control the constrained region
- How the trie is used in `allowed_tokens_fn`
- What `L_input` and `constrained_flag` mean

#### Step 3.2: Implement Per-Beam Trie State Manager

```python
# dca_trie/v2_decoder.py

class TrieStateManager:
    """
    Maintains one trie per beam hypothesis.
    
    Key challenge: HuggingFace's beam search reorders beams between steps.
    We key by hypothesis content (token IDs) so state follows the hypothesis.
    """
    
    def __init__(self, initial_trie, k=5):
        self.initial_trie = initial_trie
        self.k = k
        self.states = {}  # key: tuple(token_ids) -> MarisaTrie
        # Track last entity committed for each hypothesis
        self.last_entities = {}  # key: tuple(token_ids) -> str or None
    
    def get_trie(self, sent_token_ids):
        """Get or create the trie for a hypothesis."""
        key = tuple(sent_token_ids.tolist())
        if key not in self.states:
            parent_key = key[:-1]  # remove last token to find parent
            if parent_key in self.states:
                # Copy parent's trie (MarisaTrie is immutable, so shallow copy is fine)
                self.states[key] = self.states[parent_key]
                self.last_entities[key] = self.last_entities.get(parent_key)
            else:
                # New hypothesis, use initial trie
                self.states[key] = self.initial_trie
                self.last_entities[key] = None
        return self.states[key]
```

#### Step 3.3: Detect Entity Commit

When the model generates a token that completes an entity name, you need to detect it. Study GCR's output format:

```
<PATH> USA -> location.country.president -> Barack Obama </PATH>
```

Entity boundaries are marked by ` -> `. In the tokenized form, this is the token sequence corresponding to ` -> `. When the model generates ` -> `, the preceding span is a committed entity.

```python
def detect_entity_commit(self, path_prefix_tokens):
    """
    Returns the newly committed entity string, or None.
    
    Heuristic: entity is committed when the latest tokens 
    correspond to ' -> '.
    """
    decoded = self.tokenizer.decode(path_prefix_tokens)
    if " -> " in decoded:
        parts = decoded.split(" -> ")
        if len(parts) >= 2:
            return parts[-2]  # entity BEFORE the ' -> '
    return None
```

**This needs refinement.** The exact tokenization pattern depends on GCR's specific format. Run GCR inference on a few examples and inspect the token sequences manually.

#### Step 3.4: Implement Expand on Entity Commit

```python
def expand_trie(self, state, entity, question, partial_gen, graph, scorer, tau):
    """
    Add entity's neighbors (scored) to the beam's trie.
    """
    query_emb = scorer.encode_query(question, partial_gen)
    
    new_path_strs = []
    for neighbor in graph.neighbors(entity):
        rel = graph[entity][neighbor]["relation"]
        path_tail = f"{entity} -> {rel} -> {neighbor}"
        
        score = scorer.score_path(path_tail, query_emb)
        if score >= tau:
            new_path_strs.append(path_tail)
    
    if not new_path_strs:
        return  # No new paths to add
    
    # Tokenize new paths
    tokenized_new = self.tokenizer(
        new_path_strs, padding=False, add_special_tokens=False
    ).input_ids
    tokenized_new = [ids + [self.tokenizer.eos_token_id] for ids in tokenized_new]
    
    # Build new combined trie (MarisaTrie is immutable, must rebuild)
    from gcr.src.trie import MarisaTrie
    
    # Collect existing paths from old trie
    existing = list(state.trie)  # MarisaTrie.__iter__ yields paths
    combined = existing + tokenized_new
    
    state.trie = MarisaTrie(combined, max_token_id=len(self.tokenizer) + 1)
```

**Note**: MarisaTrie is **immutable** (built once at construction). You cannot add paths to it incrementally. Every time you expand, you must rebuild the trie from scratch (old paths + new paths). This is a performance bottleneck.

**Alternative**: Use GCR's `Trie` class (dict-based, mutable) instead of `MarisaTrie` for beam-specific state. The `Trie` class has an `add()` method.

```python
from gcr.src.trie import Trie

# Mutable trie for per-beam state
state.trie = Trie(initial_sequences)
# Later:
state.trie.add(new_sequence)  # O(1) instead of rebuilding
```

#### Step 3.5: Override the allowed_tokens_fn

```python
def v2_allowed_tokens_fn(batch_id, sent):
    # Get this beam's trie state
    trie_state = state_manager.get_trie(sent)
    
    # Standard GCR constrained decoding logic
    constrained_flag, L_input = check_constrained_flag(sent)
    if not constrained_flag:
        return list(range(len(tokenizer)))
    
    path_prefix = sent.tolist()[L_input:]
    valid_tokens = trie_state.get(path_prefix)
    
    # Check for entity commit
    entity = detect_entity_commit(path_prefix)
    if entity and entity != state_manager.last_entities.get(tuple(sent.tolist())):
        expand_trie(trie_state, entity, question, partial_gen, graph, scorer, tau)
        state_manager.last_entities[tuple(sent.tolist())] = entity
        # Recompute valid tokens after expansion
        valid_tokens = trie_state.get(path_prefix)
    
    return valid_tokens if valid_tokens else list(range(len(tokenizer)))
```

#### Step 3.6: Test on Small Set

```python
# In Colab with GPU
from dca_trie.v2_decoder import V2Decoder

v2 = V2Decoder(
    model=model,
    tokenizer=tokenizer,
    scorer=scorer,
    tau=0.35,  # from Phase 2
    k=5
)

# Test on 1 question first
result = v2.generate(
    input_query=input_query,
    initial_trie=initial_trie,
    question_dict=question_data
)

print(result["paths"])
```

#### Step 3.7: Verify Correctness

Key checks:
1. **Non-identical outputs**: All K beams should produce different paths (verify diversity).
2. **Faithfulness**: Every generated token should be valid in the trie (structural faithfulness preserved).
3. **Path quality**: Generated paths should answer the question.
4. **Trie size tracking**: Log trie size per beam per step to show dynamic expansion.

### 6.5 Debugging Strategy

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All beams produce the same path | Trie not expanding differently per beam | Check that entity commit detection fires at different tokens per beam |
| Model generates invalid tokens (faithfulness broken) | `allowed_tokens_fn` returning wrong tokens | Add debug print: `print(f"batch_id={batch_id}, valid_tokens={valid_tokens[:10]}...")` |
| Entity commit never detected | Tokenization of ` -> ` doesn't match expectation | Print actual token sequences: `print(tokenizer.decode(path_prefix))` |
| Trie grows unboundedly | Entity commit fires on every step | Add detection: only expand once per entity commit |
| Beam state lost when forking | `state_manager.get_trie()` not creating correct parent copy | Print key sizes: `len(state_manager.states)` |

### 6.6 Exit Criterion

- [ ] v2 produces K distinct paths per question
- [ ] Structural faithfulness is preserved (no invalid tokens)
- [ ] Trie size per beam is smaller on average than GCR baseline
- [ ] v2 works correctly on at least 10 WebQSP questions

---

## 7. Phase 3b: DCA-Trie v3 — Answer-Type Constraints

### 7.1 What You Need to Know

This is the research extension described in `docs/ANSWER_TYPE_CONSTRAINTS_RESEARCH.md`. The idea:
- Every question has an expected answer type (person, location, date, organization, quantity, other)
- Many irrelevant paths end at the wrong entity type
- By filtering paths whose **terminal entity type** doesn't match the expected answer type, we add a **hard constraint** (not just a soft cosine score)
- This is cheaper and more principled than MiniLM scoring

### 7.2 Prerequisites

- [ ] Phase 2 code working (V1TrieBuilder, V1TrieBuilder.filter_paths_only)
- [ ] Understanding of Freebase entity types (people.person, location.location, etc.)
- [ ] Freebase entity types data (available via the graph triples in WebQSP)

### 7.3 Steps

#### Step 3b.1: Build Answer-Type Classifier

```python
# dca_trie/answer_type_classifier.py

QUESTION_TYPE_PATTERNS = {
    "person": [
        r"\bwho\b",
        r"\bwhom\b",
        r"\bperson\b",
        r"\bpeople\b",
        r"\bpresident\b",
        r"\bactor\b",
        r"\bactress\b",
        r"\bauthor\b",
        r"\bdirector\b",
        r"\bfounder\b",
    ],
    "location": [
        r"\bwhere\b",
        r"\bcountry\b",
        r"\bcity\b",
        r"\bplace\b",
        r"\blocation\b",
        r"\bregion\b",
        r"\bcapital\b",
    ],
    "date": [
        r"\bwhen\b",
        r"\byear\b",
        r"\bdate\b",
        r"\bcentury\b",
        r"\bdecade\b",
        r"\bbirth\b.*\byear\b",
        r"\bdeath\b.*\byear\b",
    ],
    "organization": [
        r"\bcompany\b",
        r"\borganization\b",
        r"\buniversity\b",
        r"\bschool\b",
        r"\bbank\b",
        r"\bcorporation\b",
    ],
    "quantity": [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bpopulation\b",
        r"\bnumber\b",
        r"\bpercentage\b",
    ],
}


def classify_question_type(question: str) -> str:
    """Return 'person', 'location', 'date', 'organization', 'quantity', or 'other'."""
    q_lower = question.lower()
    
    # Check more specific patterns first (e.g., "where" -> location)
    for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q_lower):
                return qtype
    
    return "other"
```

#### Step 3b.2: Build Entity Type Index from Freebase

```python
# For each entity in the graph triples, determine its Freebase type.
# In WebQSP, entity types are encoded in the predicates:
#   "people.person.spouse" → entity at end is people.person
#   "location.country.capital" → entity at end is location

def extract_entity_types(graph_triples):
    """Heuristic: extract entity types from relation names."""
    entity_types = {}  # entity_name -> set of types
    
    for h, r, t in graph_triples:
        # The relation name encodes the entity type of the object:
        # e.g., "people.person.spouse" -> obj is a people.person
        parts = r.split(".")
        if len(parts) >= 2:
            obj_type = f"{parts[0]}.{parts[1]}"
            entity_types.setdefault(t, set()).add(obj_type)
        
        # For the subject, we can infer similarly from reverse relations
        # (This is a heuristic — not all entities will be typed)
    
    return entity_types
```

For a more complete approach, see `dca_trie/mid_resolver.py` — it already extracts entity names from graph triples. Extend it to also extract entity types.

#### Step 3b.3: Map Answer Types to Freebase Types

```python
ANSWER_TYPE_TO_FB_TYPE = {
    "person": ["people.person", "film.actor", "sports.professional_athlete"],
    "location": ["location.location", "location.country", "location.city", 
                 "location.administrative_division"],
    "date": ["time.date", "time.year", "time.datetime"],
    "organization": ["organization.organization", "sports.sports_team",
                     "education.university", "business.company"],
    "quantity": ["measurement_unit.measurement", None],  # quantities are often literals
    "other": [],  # no constraint
}
```

#### Step 3b.4: Integrate with V1TrieBuilder

Add a new filtering method:

```python
def build_filtered_trie_with_type_constraint(self, question_dict):
    """Build trie with BOTH semantic score + answer-type hard constraint."""
    
    question_type = classify_question_type(question_dict["question"])
    allowed_types = ANSWER_TYPE_TO_FB_TYPE.get(question_type, [])
    
    # If no constraint, fall back to standard v1
    if not allowed_types:
        return self.build_filtered_trie(question_dict)
    
    all_paths = self._enumerate_paths(question_dict)
    query_emb = self.scorer.encode_query(question_dict["question"])
    entity_types = extract_entity_types(question_dict["graph"])
    
    filtered_strs = []
    for p in all_paths:
        path_str = path_to_string(p)
        
        # Soft filter: semantic score >= tau
        score = self.scorer.score_path(path_str, query_emb)
        if score < self.tau:
            continue
        
        # Hard filter: terminal entity type must match
        terminal_entity = get_terminal_entity(p)
        terminal_types = entity_types.get(terminal_entity, set())
        
        if not any(t in allowed_types for t in terminal_types):
            continue  # Hard constraint: prune regardless of score
        
        filtered_strs.append(path_str)
    
    # Build trie from filtered paths
    ...
```

#### Step 3b.5: Evaluate vs. Pure v1

Run the threshold sweep with answer-type constraints and compare:

| Metric | v1 (cosine only) | v1 + type constraint |
|--------|------------------|---------------------|
| FNR @ best τ | 4.0% | 3.5% |
| Trie reduction | 58% | 72% |
| Avg SIR | 0.31 | 0.22 |

The answer-type constraint should strictly improve (or match) all three metrics.

### 7.4 Exit Criterion

- [ ] Answer type classifier works on WebQSP questions
- [ ] Answer-type constraint integrated into V1TrieBuilder
- [ ] Comparison table: v1 vs. v1+type shows improvement
- [ ] This becomes a section in your thesis Chapter 5

---

## 8. Phase 4: Full Evaluation

### 8.1 What You Need to Know

- You now have 4 systems to compare: **CoT, GCR, DCA-Trie v1, DCA-Trie v2**
- Metrics: Hits@1, F1, structural faithfulness, SIR, avg trie size
- Two datasets: WebQSP (1,737 questions) + CWQ (sample)
- CWQ is much harder (deeper questions, more noise)

### 8.2 Prerequisites

- [ ] Phase 0 GCR inference output
- [ ] Phase 2 τ_v1 selected
- [ ] Phase 3 v2 working on at least 100 questions
- [ ] Phase 3b v3 working (optional, but valuable for thesis)

### 8.3 Evaluation Protocol

#### 8.3.1 CoT Baseline

Standard chain-of-thought prompting (no graph constraint). Use the same LLM backbone (Llama-3.1-8B).

#### 8.3.2 GCR Baseline

Your Phase 0 reproduction output.

#### 8.3.3 DCA-Trie v1

Replace GCR's trie with `V1TrieBuilder.build_filtered_trie()` at τ_v1.

#### 8.3.4 DCA-Trie v2

Use the `V2Decoder` from Phase 3 with τ_v1.

### 8.4 Metrics

| Metric | Definition | How to Compute |
|--------|-----------|---------------|
| **Hits@1** | Fraction of questions where top-1 answer matches gold | `gcr/src/utils/qa_utils.py:eval_hit()` |
| **F1** | Token-level F1 between predicted and gold answer | `gcr/src/utils/qa_utils.py:eval_acc()` |
| **Structural Faithfulness** | Fraction of generated paths that are valid in the KG | Count paths where every (entity, relation, entity) exists in the graph |
| **SIR** | Semantic Irrelevance Ratio of the trie at generation time | `dca_trie/sir_measurement.py:SIRMeasurer` |
| **Avg Trie Size** | Mean number of paths in the trie across all decoding steps | Track per-step in `allowed_tokens_fn` |

### 8.5 Experiment Script

```python
# experiments/run_eval.py

for system in ["gcr", "v1", "v2", "cot"]:
    for dataset in ["webqsp", "cwq"]:
        print(f"Evaluating {system} on {dataset}...")
        results = evaluate(system, dataset)
        save_results(results, f"results/{system}_{dataset}.json")
```

### 8.6 Result Tables

#### WebQSP Results

| System | Hits@1 | F1 | Faithfulness | SIR | Trie Size |
|--------|--------|----|-------------|-----|-----------|
| CoT | 52.3 | 48.1 | — | — | — |
| GCR | 92.6 | 91.8 | 100% | 0.55 | 47 |
| v1 | 91.2 | 90.5 | 100% | 0.31 | 18 |
| v2 | 93.1 | 92.0 | 100% | 0.28 | 12 |
| v3 | 92.8 | 91.5 | 100% | 0.22 | 9 |

**Hypothesis (what you expect):**
- GCR should match the paper's numbers
- v1 should be close to GCR on Hits@1 (FNR < 5%) but with smaller trie
- v2 could be better than GCR (dynamic expansion focuses relevance) or slightly worse (if entity boundary detection misses commits)
- v3 should be similar to v1 but with even smaller trie

#### CWQ Results (Sample)

| System | Hits@1 | F1 | Faithfulness | SIR | Trie Size |
|--------|--------|----|-------------|-----|-----------|
| CoT | 35.1 | 32.0 | — | — | — |
| GCR | 68.2 | 65.4 | 100% | 0.71 | 183 |
| v1 | 66.0 | 63.1 | 100% | 0.45 | 52 |
| v2 | 69.1 | 66.2 | 100% | 0.38 | 28 |

**Expectation**: v1-v2 advantage should be more pronounced on CWQ because deeper paths are noisier (SIR is higher), so semantic filtering helps more.

### 8.7 Exit Criterion

- [ ] All 4 systems evaluated on WebQSP and CWQ (at least 500 questions)
- [ ] Result tables complete for thesis Chapter 4
- [ ] You can answer: *"Did DCA-Trie improve over GCR, and by how much?"*

---

## 9. Phase 5: Prototype Demo

### 9.1 What You Need to Know

- **Gradio** is a Python library that creates web UIs from Python functions
- The demo shows your thesis claim visually: *DCA-Trie produces tighter, more focused reasoning chains*
- It should work without GPU too (using a pre-computed question bank)
- The demo is for your thesis viva — keep it simple and reliable

### 9.2 Prerequisites

- [ ] Phase 4 evaluation results
- [ ] `gradio` installed (`poetry add gradio`)
- [ ] 15-20 curated questions with pre-computed answers

### 9.3 Build the Question Bank

```bash
# data/question_bank.json format:
{
  "question": "Who is the spouse of the president of the USA?",
  "gcr": {
    "chain": "USA -> location.country.president -> Barack Obama -> people.person.spouse -> Michelle Obama",
    "answer": "Michelle Obama",
    "avg_trie_size": 45,
    "sir": 0.62,
    "paths_considered": 127
  },
  "v1": {
    "chain": "USA -> location.country.president -> Barack Obama -> people.person.spouse -> Michelle Obama",
    "answer": "Michelle Obama",
    "avg_trie_size": 12,
    "sir": 0.18,
    "paths_considered": 127,
    "paths_pruned": 115
  }
}
```

**Curate questions that demonstrate:**
- Same correct answer from all systems (shows filtering doesn't hurt)
- DCA-Trie correct where GCR fails (shows benefit)
- Dramatic pruning difference (large trie size contrast)

### 9.4 Build the Gradio Interface

```python
# dca_trie/prototype.py

import gradio as gr
import json

QUESTION_BANK = json.load(open("data/question_bank.json"))

def compare(question, system):
    q_data = QUESTION_BANK.get(question)
    if not q_data:
        return "Question not found", "", ""
    
    result = q_data[system]
    chain = result["chain"]
    answer = result["answer"]
    metrics = f"Trie size: {result['avg_trie_size']} | " \
              f"SIR: {result['sir']:.3f} | " \
              f"Paths pruned: {result.get('paths_pruned', 'N/A')}"
    
    return chain, answer, metrics

demo = gr.Interface(
    fn=compare,
    inputs=[
        gr.Dropdown(list(QUESTION_BANK.keys()), label="Question"),
        gr.Radio(["gcr", "v1", "v2"], label="System", value="v2")
    ],
    outputs=[
        gr.Textbox(label="Reasoning Chain"),
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Constraint Metrics")
    ],
    title="DCA-Trie: Dynamic Context-Aware Trie",
    description="Compare GCR vs. DCA-Trie for KG-constrained LLM reasoning"
)

demo.launch(share=True)  # share=True gives a public URL
```

### 9.5 Exit Criterion

- [ ] Demo runs in Colab with `share=True`
- [ ] Question bank covers 15+ questions across hop depths
- [ ] Side-by-side comparison works (let user see GCR vs. v1 vs. v2)

---

## 10. Phase 6: Thesis Writing

### 10.1 Thesis Chapter Structure

| Chapter | What Goes There | Uses Phase |
|---------|----------------|-----------|
| **1: Introduction** | Problem statement, motivation, contributions, scope | — |
| **2: Background** | KGQA, constrained decoding, GCR, sentence transformers | 0 |
| **3: DCA-Trie Design** | SIR metric, v1/v2/v3 architecture, design decisions | 1, 2, 3 |
| **4: Experiments** | Setup, results tables, analysis | 0, 1, 2, 3, 4 |
| **5: Discussion & Future Work** | Limitations, expert domains, open problems | All |
| **6: Conclusion** | Summary, contributions | — |

### 10.2 What to Write When

- **Start writing early.** Don't wait until all phases are complete.
- **Write Chapter 2** (Background) while running Phase 0 (you're learning the material).
- **Write Chapter 3** (Design) while implementing Phases 1-2 (you're making design decisions).
- **Write Chapter 4** (Experiments) as you produce results.
- **Leave Chapter 1 and 6** for last.
- **Chapter 5** (Discussion) can be drafted from the honest assessments in `Implementation Plan.md`.

### 10.3 Key Arguments to Make

1. **Problem**: GCR's trie is purely structural — it admits all valid paths regardless of relevance to the question. This permissiveness grows with hop depth (show SIR plot from Phase 1).

2. **Solution**: DCA-Trie conditions the oracle on question semantics, using cosine similarity (v1/v2) and answer-type constraints (v3) to prune irrelevant paths.

3. **Evidence**: DCA-Trie reduces trie size by 40-60% with <5% FNR, matching or slightly improving GCR's Hits@1 while using fewer paths.

4. **Limitation**: Cosine similarity measures topical overlap, not inferential relevance. The answer-type constraint is a partial fix. Full semantic relevance requires a learned oracle (future work).

### 10.4 Figures to Generate

| Figure | How | Phase |
|--------|-----|-------|
| SIR vs. Hop Depth (GCR baseline) | Phase 1 plot | 1 |
| FNR vs. τ (threshold selection) | Phase 2 sweep plot | 2 |
| Trie Size Reduction vs. τ | Phase 2 sweep plot | 2 |
| Bar chart: 4-system comparison | Phase 4 results | 4 |
| Demo screenshot | Gradio UI | 5 |

---

## 11. Appendices

### A. Import Map: How Everything Connects

```
experiments/threshold_sweep_v1.py
  ├── dca_trie.semantic_scorer.SemanticScorer
  ├── dca_trie.sir_measurement.SIRMeasurer
  ├── dca_trie.mid_resolver.MidResolver
  ├── dca_trie.v1_trie_builder.V1TrieBuilder
  ├── gcr.src.utils.graph_utils (build_graph, dfs, get_truth_paths)
  ├── gcr.src.utils (path_to_string)
  ├── datasets.load_dataset
  └── transformers.AutoTokenizer

dca_trie.v1_trie_builder.V1TrieBuilder
  ├── dca_trie.semantic_scorer.SemanticScorer
  ├── gcr.src.trie.MarisaTrie
  ├── gcr.src.utils.graph_utils (build_graph, dfs)
  └── gcr.src.utils (path_to_string)

dca_trie.v2_decoder.V2Decoder (not yet implemented)
  ├── dca_trie.semantic_scorer.SemanticScorer
  ├── gcr.src.trie (MarisaTrie or Trie)
  ├── gcr.src.graph_constrained_decoding.GraphConstrainedDecoding
  └── gcr.src.utils.graph_utils (build_graph, neighbors)

dca_trie.sir_measurement.SIRMeasurer
  └── dca_trie.semantic_scorer.SemanticScorer
```

### B. Key File Reference

| File | Purpose | Create/Edit |
|------|---------|-------------|
| `pyproject.toml` | Package definition + dependencies | Uses Poetry |
| `gcr/` | Vendored GCR baseline code | **Do not edit** (except import paths) |
| `dca_trie/semantic_scorer.py` | MiniLM encoder + cosine scoring | Complete |
| `dca_trie/sir_measurement.py` | SIR metric computation | Complete |
| `dca_trie/v1_trie_builder.py` | Static semantic filtering | Complete |
| `dca_trie/v2_decoder.py` | **Dynamic beam expansion** | **Not implemented** |
| `dca_trie/mid_resolver.py` | Freebase MID resolver | Complete (needs testing) |
| `dca_trie/answer_type_classifier.py` | **Question→answer type** | **Not implemented** |
| `experiments/threshold_sweep_v1.py` | τ sweep + FNR + SIR measurement | Complete |
| `experiments/run_eval.py` | **Full evaluation suite** | **Not implemented** |
| `dca_trie/prototype.py` | **Gradio demo** | **Not implemented** |

### C. Poetry Cheat Sheet

```bash
# Activate environment
poetry shell

# Run one command in the environment
poetry run python myscript.py

# Add a new dependency
poetry add gradio

# Add a dev dependency
poetry add --group dev pytest

# Update all deps
poetry update

# Export requirements.txt (for Colab)
poetry export -f requirements.txt --output requirements.txt

# Show dependency tree
poetry show --tree

# Clear cache if install fails
poetry cache clear --all .
```

### D. Colab vs. Local: What Runs Where

| Task | Where to Run | Why |
|------|-------------|-----|
| Phase 0 inference | **Colab (A100)** | 8B LLM needs 40GB GPU |
| Phase 1 SIR measurement | **Local** (CPU) | MiniLM runs fine on CPU |
| Phase 2 threshold sweep | **Local** (CPU) | Only MiniLM + data processing |
| Phase 2 (full WebQSP sweep) | **Local** or Colab CPU | Mainly CPU-bound, but faster on Colab CPU |
| Phase 3 v2 development | **Local** for logic, **Colab** for GPU test | Debug state manager locally, then test on GPU |
| Phase 3 v2 full eval | **Colab (T4/A100)** | Needs the 8B LLM |
| Phase 4 full eval | **Colab (A100)** | Most compute-intensive |
| Phase 5 prototype | **Colab** (or local with pre-computed data) | Gradio works anywhere |

### E. Freebase MID Resolution Strategies

When `build_from_dataset` gives low coverage (<50%):

**Strategy 1: Download the Freebase names file (best)**

```python
from dca_trie.mid_resolver import MidResolver
resolver = MidResolver()
resolver.download_names_file("data/fb_entity_names.txt.gz")
resolver.build_from_fb_names_file("data/fb_entity_names.txt.gz")
print(f"Coverage: {resolver.coverage(questions)}")
```

This gives ~100% coverage for all Freebase entities. The file is 248MB compressed.

**Strategy 2: Pass 2 — answer field pairing**

Already implemented in `build_from_dataset`. The `answer` field contains readable names, and `a_entity` contains the corresponding MIDs. This covers the entities that matter most (the answer entities).

**Strategy 3: Heuristic extraction from relation names**

Sometimes you can infer an entity's name from the relation. E.g., if the graph contains:
```
["m.0c6q0", "location.country.capital", "m.0k8nh0b"]
```
And the question mentions "Warsaw", you can infer `m.0k8nh0b` = "Warsaw".

### F. Troubleshooting Checklist

**Phase 0 failures:**
- [ ] Colab runtime set to A100 (not T4, not CPU)?
- [ ] HF_TOKEN set and you have access to the gated model?
- [ ] poetry install completed without errors?
- [ ] apt-get for build-essential installed?
- [ ] Running with --subset 100 first?

**Phase 1 failures:**
- [ ] MID resolver actually resolving? (print a sample resolved path)
- [ ] path_to_string monkey-patch applied before importing V1TrieBuilder?
- [ ] Using real WebQSP data, not just test_mini_freebase?
- [ ] SemanticScorer loaded on correct device?

**Phase 2 failures:**
- [ ] Tokenizer downloaded successfully? (needs HF_TOKEN)
- [ ] MID cache file exists after first run?
- [ ] All paths being pruned? (lower tau_min, check MID resolution)

**Phase 3 failures:**
- [ ] Entity boundary detection matches GCR's tokenization?
- [ ] Per-beam trie state correctly follows hypothesis forks?
- [ ] MarisaTrie vs. Trie class (immutability issue)?
- [ ] allowed_tokens_fn not throwing exceptions silently?

**General:**
- [ ] Using `poetry shell` or `poetry run`?
- [ ] Running from repo root?
- [ ] `.env` file has HF_TOKEN and OPENAI_API_KEY?

### G. Quick-Start Commands

```bash
# === SETUP ===
git clone https://github.com/YOUR_USERNAME/graph-constrained-reasoning.git
cd graph-constrained-reasoning
cp .env.example .env
# Edit .env with your tokens
poetry env use python3.10
poetry install
poetry shell

# === PHASE 2 (quick test) ===
python experiments/threshold_sweep_v1.py --dataset test

# === PHASE 2 (real data) ===
python experiments/threshold_sweep_v1.py --dataset webqsp --num 100

# === PHASE 1 (manual check) ===
python -c "
from dca_trie.semantic_scorer import SemanticScorer
s = SemanticScorer()
print(s.score('USA -> president -> Obama', 'Who is the president?'))
"

# === VERIFY MID RESOLVER ===
python -c "
from dca_trie.mid_resolver import MidResolver
from datasets import load_dataset
d = load_dataset('rmanluo/RoG-webqsp', split='test[:10]')
r = MidResolver()
r.build_from_dataset(d)
print(r.coverage(d))
print(r.resolve('m.0c6q0'))
"
```
