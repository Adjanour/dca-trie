# Libraries Guide

## 1. HuggingFace Transformers

**Used for:** Loading and running the LLM, tokenization, generation with constraints.

### Key Components in GCR

#### `AutoTokenizer.from_pretrained()`
```python
self.tokenizer = AutoTokenizer.from_pretrained(
    self.args.model_path, token=HF_TOKEN, trust_remote_code=True
)
```
- Loads the tokenizer matching the model checkpoint
- `token=HF_TOKEN` is needed for gated models (Llama-3.1)
- The tokenizer converts text ↔ token IDs

#### `AutoModelForCausalLM.from_pretrained()`
```python
self.model = AutoModelForCausalLM.from_pretrained(
    self.args.model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```
- Loads the model with automatic device mapping
- `torch.bfloat16` reduces memory from ~32GB to ~16GB for 8B params
- `flash_attention_2` enables memory-efficient attention

#### `model.generate()`
```python
res = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=self.generation_cfg,
    prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
)
```
- The core generation method
- `prefix_allowed_tokens_fn` is the constraint hook — called at each step with the current prefix

#### `generation_config`
- Controls beam search parameters: `num_beams`, `num_beam_groups`, `diversity_penalty`
- Loaded from the model's config, then overridden

### Model Loading Details

The `HfCausalModel` in `base_hf_causal_model.py` loads the model config first, then overrides generation parameters:

```python
self.generation_cfg = GenerationConfig.from_pretrained(self.args.model_path)
self.generation_cfg.max_new_tokens = self.args.max_new_tokens
self.generation_cfg.num_beams = self.args.k
# etc.
```

### HF_Datasets

**Used for:** Loading WebQSP and CWQ from HuggingFace hub.

```python
from datasets import load_dataset
dataset = load_dataset("rmanluo/RoG-webqsp", split="test[:100]")
```

The datasets are stored in HuggingFace's Parquet format. Each split contains the question, graph subgraph, entities, and answers.

---

## 2. Marisa Trie

**Used for:** Memory-efficient prefix tree for constraint lookup.

### What is Marisa?

MARISA = **M**atching **A**lgorithm with **R**ecursively **I**mplemented **S**tor**A**ge

It's a C++ library for building static (immutable) tries with very high memory density. The Python bindings wrap this.

### Installation
```bash
pip install marisa-trie
```

### API in GCR

```python
from marisa_trie import Trie

# Build from strings
trie = marisa_trie.Trie(["abc", "abd", "cde"])

# Lookup keys sharing a prefix
keys = trie.keys("ab")  # returns ["abc", "abd"]

# Check membership
"abc" in trie  # True
```

### The Token ID Encoding Trick

MarisaTrie works with strings, not integers. GCR converts token IDs to characters:

```python
self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
    [chr(i) for i in range(65000, max_token_id + 10000)]
    if max_token_id >= 55000
    else []
)
self.char2int = {self.int2char[i]: i for i in range(max_token_id)}
```

Token IDs are converted to characters, concatenated into strings, and stored in the trie:

```python
trie = marisa_trie.Trie(
    "".join([self.int2char[i] for i in sequence])
    for sequence in sequences
)
```

At lookup time:
```python
key = "".join([self.int2char[i] for i in prefix_sequence])
next_tokens = {
    self.char2int[e[len(key)]] 
    for e in self.trie.keys(key) 
    if len(e) > len(key)
}
```

### First-Level Cache

For O(1) lookup at the root level (most common case in generation):

```python
self.zero_iter = list({sequence[0] for sequence in sequences})
```

If `len(prefix_sequence) == 0`, return the cached set directly without consulting the trie.

### Why MarisaTrie Instead of dict

- **Memory:** A Python dict of 100K+ token sequences consumes significant memory. MarisaTrie is optimized for exactly this scenario.
- **Speed:** Prefix lookups are O(length of prefix) and very fast in C++
- **Immutability:** The trie doesn't change after construction (important for GCR's static trie, but a limitation for DCA-Trie v2 which needs dynamic updates)

---

## 3. Flash Attention 2

**Used for:** Memory-efficient and fast attention computation on GPU.

### What Flash Attention Does

Standard attention computes the full `N x N` attention matrix and stores it in HBM (high-bandwidth memory), requiring O(N²) memory. Flash Attention 2:

1. **Tiles** the attention computation so it fits in fast SRAM
2. **Avoids materializing** the full attention matrix
3. Uses the **online softmax** trick to compute attention in O(N²) time with O(N) memory

### Impact on GCR

- Llama-3.1-8B with standard attention on A100 (40GB): barely fits with batch size 1
- With flash-attention 2: comfortably fits with beam search
- The `--attn_implementation flash_attention_2` flag enables this

### Installation
```bash
pip install flash-attn --no-build-isolation
```

Requires CUDA 12.1+ and a compatible GPU (A100, H100, RTX 3090/4090).

---

## 4. Sentence-Transformers (all-MiniLM-L6-v2)

**Used for:** Computing semantic similarity between questions and KG paths (for DCA-Trie).

### What It Is

`sentence-transformers/all-MiniLM-L6-v2` is a lightweight (~80MB) sentence embedding model:
- Based on MiniLM (distilled BERT)
- 6 transformer layers
- 384-dimensional embeddings
- Outputs normalized embeddings (cosine similarity = dot product)

### Usage in DCA-Trie

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode paths and queries
path_embedding = model.encode("Elvis Presley -> film.actor.film -> Blue Hawaii")
query_embedding = model.encode("What film did Elvis Presley star in?")

# Compute similarity
similarity = model.similarity(path_embedding, query_embedding)
```

### Why MiniML-L6-v2 for DCA-Trie

- **Small and fast:** ~80MB model, ~5ms per encoding on CPU
- **Good enough:** Produces reasonable semantic similarity for path filtering
- **No fine-tuning needed:** Works zero-shot for the KGQA domain
- **Caching-friendly:** Path embeddings can be cached for repeated use

### Known Limitation (from the Plan)

MiniLM measures **topical overlap**, not **inferential relevance**:

> Path A (Marie Curie → nationality → Poland) and Path B (Marie Curie → place_of_birth → Warsaw) both have high cosine similarity to "Where was Marie Curie born?" Only B answers it.

Cosine similarity captures that both paths are "about Marie Curie and a place" but cannot distinguish "birthplace" from "nationality."

---

## 5. NetworkX

**Used for:** Graph representation and traversal.

### Usage in GCR

```python
import networkx as nx

# Build from triplets
G = nx.DiGraph()
for h, r, t in triplets:
    G.add_edge(h.strip(), t.strip(), relation=r.strip())

# Get neighbors
for neighbor in G.neighbors(node):
    relation = G[node][neighbor]["relation"]

# Shortest paths
paths = nx.all_shortest_paths(G, source, target)
```

### DFS Implementation

GCR implements its own DFS (in `graph_utils.py`) rather than using NetworkX's:

```python
def dfs_visit(node, path):
    if len(path) > max_length:
        return
    for neighbor in graph.neighbors(node):
        rel = graph[node][neighbor]["relation"]
        new_path = path + [(node, rel, neighbor)]
        if len(new_path) <= max_length:
            path_lists.add(tuple(new_path))
        dfs_visit(neighbor, new_path)
```

This collects ALL paths up to max_length, not just the shortest ones.

---

## 6. PyTorch

**Used for:** Tensor operations, device management.

### Key Patterns in GCR

```python
# Move inputs to GPU
input_ids = inputs.input_ids.to(self.model.device)
attention_mask = inputs.attention_mask.to(self.model.device)

# Inference mode (no gradient tracking)
@torch.inference_mode()
def generate_sentence(self, llm_input):
    ...

# Mixed precision
torch_dtype=torch.bfloat16
```

---

## 7. PEFT / LoRA

**Used for:** Efficient fine-tuning of the KG-specialized LLM.

GCR fine-tunes the base Llama model with LoRA (Low-Rank Adaptation):
- Only a small number of adapter parameters are trained
- The base model weights remain frozen
- The adapter is merged at inference time

The training script (`finetune_kg_specialized_llm.py`) uses `peft` for LoRA and `trl` for training with reinforcement learning.

---

## 8. Additional Tools

| Library | Purpose |
|---------|---------|
| `accelerate` | Multi-GPU training, DeepSpeed integration |
| `deepspeed` | Memory-efficient distributed training (ZeRO stages) |
| `bitsandbytes` | 4-bit/8-bit quantization for memory reduction |
| `wandb` | Experiment tracking and logging |
| `openai` | API client for GPT models (Step 2) |
| `scikit-learn` | F1/precision/recall computation |
