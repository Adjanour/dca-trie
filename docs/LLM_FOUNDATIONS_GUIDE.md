# LLM Foundations Guide

A deep, mechanistic guide to Large Language Models, built from first principles,
bridging to Graph-Constrained Reasoning and DCA-Trie.

---

> **📦 BASELINE:** Layers 0-5 describe concepts from the GCR framework
> (Luo et al., ICML 2025) and general LLM theory. These are prerequisite
> knowledge, not original contributions.
>
> **🆕 CONTRIBUTION:** Layer 6 describes DCA-Trie concepts (this project).
> When code paths are referenced: `gcr/src/` = baseline, `dca_trie/` = contribution.

---

## How to Use This Guide

| Layer | Title | Prerequisites | For the Project |
|-------|-------|--------------|-----------------|
| 0 | The Essence | None | Why LLMs reason (and hallucinate) |
| 1 | Autoregressive Transformers | Layer 0 | The engine GCR constrains |
| 2 | Training | Layer 1 | Why fine-tuning a KG-specialized LLM works |
| 3 | Decoding Strategies | Layer 2 | The interface GCR hooks into |
| 4 | Constrained Decoding | Layer 3 | The mechanism GCR is built on |
| 5 | Knowledge Graphs & LLMs | Layers 3-4 | The problem DCA-Trie improves |
| 6 | Semantic Filtering | Layer 5 | The innovation DCA-Trie introduces |
| 7 | Learning Path | All | What to study next based on your role |

Each layer contains:

- **First principles exposition** — what, why, how
- **Anchors to the project** — concrete code paths and files
- **Authoritative references** — from Karpathy, Ng, and the research literature
- **Exercises** — to verify understanding against the codebase

---

# Layer 0: The Essence — What Makes a Thought?

## 0.1 The Deepest Question

Before any architecture, training objective, or decoding strategy, ask this:

> What does it mean for a machine to "reason" about the world?

An LLM is a single, enormous function:

```
P(y_t | y_<t, x)
```

At step t, given input x and tokens generated so far y_<t, what token y_t comes next?

This is all it does. There is no internal monologue, no working memory, no database
lookup, no verifier. It is next-token prediction, all the way down.

And yet, from this apparently shallow operation, the following emerge:

- Translation between languages
- Code generation and debugging
- Multi-step mathematical reasoning
- Theory-of-mind inferences
- Creative writing

**The central mystery of LLMs:** How can a mechanism so simple produce behaviour so
complex? Karpathy frames this as *compression*: training is a lossy compression of
the entire internet into a weights file. The 10 TB of text in the training corpus is
compressed into ~140 GB of float16 parameters (for a 70B model). The result is a
*compressed world model* — not a database, but a simulator of the processes that
generated the text.

> **Karpathy, "Intro to LLMs" (2023):** "The model is not exactly a database. It's
> more like a dream simulator of the internet. It's not exactly a database because
> you can't lookup facts. But you can prompt it in the right way and it will
> simulate the fact for you."

## 0.2 The Two Crises

This compression explains two fundamental problems that GCR/DCA-Trie address:

### Crisis 1: Hallucination

Because the model is a simulator, not a database, it has no fidelity guarantee.
For common facts ("Paris is the capital of France"), the compression is lossless.
For rare facts ("the 7th largest exporter of cobalt in 2023"), the compression loses
information. The model's output is always its *best statistical guess*, not a
*verified retrieval*.

**Project anchor:** See `docs/CONCEPTS_GUIDE.md` §2.2 — "Why Hallucination Happens"

### Crisis 2: Knowledge Cutoff

The model's knowledge is frozen at the time of training. Any fact that changes
after training (e.g., "Who is the current US president?") cannot be correctly
answered without external grounding.

## 0.3 The Solution Space

Three families of solutions exist for grounding LLMs in facts:

| Approach | Changes | Examples |
|----------|---------|----------|
| **Input augmentation** | What the model *sees* | RAG, prompt engineering |
| **Model updating** | What the model *is* | Fine-tuning, LoRA |
| **Output constraint** | What the model *can say* | **GCR**, DCA-Trie, Grammar constraints |

GCR sits in the third family. It does not change the model or its input. It changes
the allowed output space during decoding.

**"The constraint oracle should know what the question is asking, not just what
paths structurally exist in the graph."** — DCA-Trie thesis statement

---

# Layer 1: The Autoregressive Transformer

## 1.1 The Big Picture

A transformer is a neural network architecture that processes sequences. It maps
a sequence of input tokens to a sequence of output logits, where each logit is
a score for every token in the vocabulary.

```
Input:  [t1, t2, t3, ..., tn]    (integers, each representing a token)
        │
        ▼
    Embedding ──► Positional Encoding
        │
        ▼
    ┌─────────────────────────────────┐
    │  N × Transformer Blocks         │
    │  ┌───────────────────────────┐  │
    │  │ Self-Attention            │  │
    │  │   (communication)         │  │
    │  │       +                   │  │
    │  │ Feed-Forward Network      │  │
    │  │   (computation)           │  │
    │  │       +                   │  │
    │  │ Residual Connections      │  │
    │  │ Layer Normalization       │  │
    │  └───────────────────────────┘  │
    └─────────────────────────────────┘
        │
        ▼
Output: [logits_1, logits_2, ..., logits_n]
```

> **Karpathy's framework:** A transformer block has two phases:
>
> 1. **Communication** (self-attention): tokens exchange information
> 2. **Computation** (feed-forward): each token processes its gathered information independently

## 1.2 Tokenization

Before the transformer sees text, the text must be converted to integers.

```
Text:    "The president of Ghana is Nana Akufo-Addo"
          │
          ▼
Tokens:  ["The", " president", " of", " Ghana", " is", " Nana", " Ak", "uf", "o",
           "-", "Ad", "do"]
          │
          ▼
IDs:     [791, 4713, 297, 4057, 338, 6941, 17637, 5952, 340, 12, 4266, 2152]
```

Key facts:

- Tokens are **subword units** (not words, not characters)
- Llama-3.1 uses a vocabulary of 128K tokens
- Tokenization is lossy: "Nana Akufo-Addo" is split into chunks the model learned
  from training data
- **This is relevant to GCR**: The trie stores token-ID sequences. The constraint
  oracle must match at the token level, not the word level.

> **Karpathy, "Let's build the GPT Tokenizer" (2024):** "Tokenization is a completely
> separate stage of the LLM pipeline. It has its own training algorithm (BPE), its
> own training data, and after training implements two functions: encode() from strings
> to tokens, and decode() back from tokens to strings. A lot of weird behaviours and
> problems of LLMs trace back to tokenization."

## 1.3 Token Embedding

Each token ID is mapped to a dense vector via an embedding table:

```
embedding_table: [vocab_size × d_model]

token_id = 791  ──►  embedding_table[791]  ──►  vector of length d_model (e.g., 4096)
```

This is the first learnable parameter. The embedding table contains one vector per
token in the vocabulary. These vectors are learned during training and encode
semantic information about each token.

## 1.4 Positional Encoding

The transformer has no inherent notion of position. The sequence "[A, B, C]" is the
same as "[C, A, B]" to the attention mechanism. Positional encodings (either learned
or sinusoidal) give each token a "position signal":

```
input_vector[t] = token_embedding[t] + positional_encoding[t]
```

For GPT-style models (including Llama), this is a learned vector per position.
For the original Transformer, it is a fixed sinusoidal function.

## 1.5 Self-Attention (Communication)

This is the core mechanism that allows tokens to "talk" to each other.

```
Given:  x = [x₁, x₂, x₃, ..., xₙ]   (each xᵢ is a d_model-dimensional vector)

For each token at position i:

1. Compute query, key, value vectors:
   qᵢ = W_Q · xᵢ    (what am I looking for?)
   kᵢ = W_K · xᵢ    (what do I contain?)
   vᵢ = W_V · xᵢ    (what do I broadcast?)

2. Compute attention scores with all tokens j (≤ i in autoregressive models):
   score(i,j) = qᵢ · kⱼ / √d_k    (how relevant is j to i?)

3. Normalize scores via softmax:
   α(i,j) = exp(score(i,j)) / Σₖ exp(score(i,k))

4. Weighted sum of values:
   outputᵢ = Σⱼ α(i,j) · vⱼ
```

**The critical property for GCR:** Attention is the mechanism that allows the model
to follow the structure of its input. When the model generates inside
`<PATH>...</PATH>` tags, attention to those tags tells the model it is in
constrained mode. The fine-tuned KG-specialized LLM has learned to recognise
these boundaries through supervised fine-tuning.

### Project anchor: How does attention relate to constrained decoding?

In `gcr/src/graph_constrained_decoding.py`, the `prefix_allowed_tokens_fn` intercepts
generation at each step. But the *model* must still attend to its prior tokens to
decide *which* valid token to choose. The constraint says "you can only pick from
this set" but the model's attention mechanism determines *which one* from that set.

This is the division of labour:

- **Constraint (oracle):** what tokens are valid → `allowed_tokens_fn`
- **Model (reasoner):** which valid token to pick → attention + FFN

## 1.6 Casual Masking (Autoregressive Constraint)

In a generative (decoder-only) transformer, token i can only attend to tokens j ≤ i.
This is enforced by a causal mask:

```
Before softmax:   attention_scores = Q · K^T + mask

mask[i,j] = 0      if j ≤ i    (allowed)
mask[i,j] = -∞     if j > i    (blocked)
```

This ensures the model never cheats by looking at future tokens — it must predict
each token using only the past.

## 1.7 Multi-Head Attention

Instead of one attention computation, the model runs H in parallel:

```
head₁:  output₁ = Attention(Q₁, K₁, V₁)
head₂:  output₂ = Attention(Q₂, K₂, V₂)
...
headₕ:  outputₕ = Attention(Qₕ, Kₕ, Vₕ)

output = Concat([output₁, ..., outputₕ]) · W_O
```

Each head learns a different attention pattern. Some heads learn syntactic patterns
(e.g., attending to the subject of a verb), others learn semantic patterns
(e.g., entity coreference), and some learn positional patterns.

> **Karpathy, "Let's build GPT" (2023):** "Multi-head attention allows the model to
> attend to different things at different positions simultaneously. One head might
> look at the previous verb, another at the subject noun, another at the beginning
> of the sentence. They all run in parallel and their outputs are concatenated."

## 1.8 Feed-Forward Network (Computation)

After attention (communication), each token vector passes through a Feed-Forward
Network (FFN):

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
```

Key insight: the FFN operates **per token, independently**. There is no communication
here. Each token processes its own gathered information.

**Analogy (Karpathy):**

- Attention: a meeting where everyone shares information
- FFN: everyone goes to their desk and processes what they learned

The FFN is where "knowledge" is primarily stored in LLMs. The parameters W₁ and W₂
of the FFN layers are where facts are compressed. Research shows that FFN layers act
as key-value memories: certain neurons activate for specific factual patterns.

## 1.9 Residual Connections

Each sublayer (attention, FFN) is wrapped in a residual connection:

```
x' = x + Sublayer(LayerNorm(x))
```

**Why this matters:** Residual connections allow gradients to flow directly through
the network during training. They also allow the model to "preserve" information
across layers. If the attention layer doesn't need to modify a token's representation,
it can learn to output near-zero and the residual connection preserves the input.

## 1.10 Layer Normalization

Applied before each sublayer (pre-norm, used in Llama):

```
LayerNorm(x) = (x - μ) / σ * γ + β
```

This stabilises training by normalising activations to have zero mean and unit
variance, then scaling/shifting by learned parameters.

## 1.11 Putting It Together: The Full Forward Pass

```
def forward(self, tokens):
    # tokens shape: (batch, seq_len)

    # 1. Embedding
    x = token_embedding(tokens)                     # (B, T, C)

    # 2. Add positional encoding
    x = x + positional_encoding[:seq_len]           # (B, T, C)

    # 3. Transformer blocks
    for block in self.blocks:
        x = x + block.attention(LayerNorm(x))        # communication
        x = x + block.ffn(LayerNorm(x))              # computation

    # 4. Final layer norm
    x = LayerNorm(x)                                 # (B, T, C)

    # 5. Project to vocabulary
    logits = lm_head(x)                              # (B, T, vocab_size)

    return logits
```

The output `logits[t]` at the final position is a vector of length `vocab_size`
where each element is the (unnormalised) score for that token.

## 1.12 From Logits to Probabilities

```
P(y_t | y_<t) = softmax(logits_t)

P(y_t = v | y_<t) = exp(logits[v]) / Σ_w exp(logits[w])
```

This is the distribution the decoding strategy samples from.

### Project anchor: Where does GCR intercept this?

In `gcr/src/graph_constrained_decoding.py`, the `prefix_allowed_tokens_fn` is called
by HuggingFace's `model.generate()` at each step. It receives the current prefix
and returns a list of allowed token IDs. HuggingFace then sets the logits of all
*disallowed* tokens to -infinity.

```
for token_id in range(vocab_size):
    if token_id not in allowed_tokens:
        logits[token_id] = -float('inf')

probabilities = softmax(logits)     # now zero for disallowed tokens
```

This is a **hard constraint**: the model cannot assign any probability to disallowed
tokens. This is different from a soft constraint (like repetition penalty) which
merely adjusts scores.

---

# Layer 2: Training

## 2.1 The Three-Stage Pipeline

Karpathy describes LLM development as three stages:

```
Stage 1: Pre-training
    Data: Internet text (10 TB+)
    Objective: Next-token prediction
    Output: Base model (document completer)
    Cost: ~$2M for a 7B model, hundreds of GPUs for weeks

Stage 2: Supervised Fine-Tuning (SFT)
    Data: Human-written instruction-response pairs
    Objective: Next-token prediction on demonstrations
    Output: Assistant model (follows instructions)

Stage 3: Reinforcement Learning from Human Feedback (RLHF)
    Data: Human preference comparisons
    Objective: Maximise reward from learned reward model
    Output: Aligned model (helpful, honest, harmless)
```

> **Ng, "Generative AI with LLMs" (2023):** "The LLM lifecycle is:
>
> 1. **Pretraining** — select data, train a foundation model
> 2. **Fine-tuning** — adapt the model to a specific task
> 3. **Evaluation** — assess performance, iterate on data/method"

## 2.2 Pre-Training: Compression as Learning

Pre-training is **lossy compression of the internet into parameters**.

The objective (next-token prediction) is deceptively simple:

```
Loss = - Σₜ log P(y_t | y_<t)
```

Minimising this loss forces the model to learn:

- **Vocabulary** (which tokens exist)
- **Syntax** (grammatical patterns)
- **World knowledge** (facts, relationships)
- **Reasoning patterns** (logical chains, multi-step processes)
- **Genre conventions** (how different types of text are structured)

**The scaling laws** (Kaplan et al., 2020; Hoffmann et al., 2022) show that
performance improves predictably with:

- More parameters (bigger models)
- More training data (more tokens)
- More compute (more FLOPs)

The Chinchilla scaling law (Hoffmann et al., 2022) established that for every
parameter, you need ~20 tokens of training data. A 7B model needs ~140B tokens.
This is why data quality and quantity both matter.

**Project anchor:** The KG-specialized LLM used in GCR (`GCR-Llama-3.1-8B` or
similar) is a **fine-tuned** model. It starts from a pre-trained base (Llama-3.1-8B)
and is further trained on KG path generation data. See
`gcr/workflow/finetune_kg_specialized_llm.py` for the fine-tuning implementation.

## 2.3 Fine-Tuning: Adaptation

Fine-tuning takes a pre-trained base model and trains it further on a specific task.

### Full Fine-Tuning

All parameters are updated. This is expensive, especially for 70B+ models.

### Parameter-Efficient Fine-Tuning (PEFT)

Only a small number of additional parameters are trained. LoRA (Low-Rank Adaptation)
is the most popular method:

```
For a weight matrix W ∈ ℝ^{d×k}:
    W' = W + A · B
where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, r << min(d, k)

Only A and B are trained (r = 8 or 16, vs. d,k = 4096+)
```

LoRA adds rank-decomposition matrices to attention layers. The original weights are
frozen. This reduces memory from ~16 GB to ~2-4 GB for a 7B model.

**Project anchor:** `gcr/workflow/finetune_kg_specialized_llm.py` configures LoRA
fine-tuning for the KG-specialized LLM. The `peft` library is used. The target
modules are typically `q_proj`, `k_proj`, `v_proj`, `o_proj` of the attention layers.
See the `PEFT_LORA` configuration in that file.

### Why Fine-Tuning Works for GCR

The KG-specialized LLM is fine-tuned to:

1. Recognise the `<PATH>...</PATH>` format
2. Generate reasoning paths that follow KG structure
3. Use the entity and relation vocabulary of the specific KG

Without fine-tuning, a base Llama model would not:

- Know that `film.film.directed_by` is a valid relation
- Know to output paths in the `entity → relation → entity → ...` format
- Respect the `<PATH>` tag boundaries that GCR uses for constraint triggering

## 2.4 The Instruction Tuning

Fine-tuning for instruction following uses the same next-token prediction objective,
but on carefully curated data:

```
Input:   "What is the capital of Ghana?"
Target:  "The capital of Ghana is Accra."

Loss:    -log P("The" | "What is the capital of Ghana?")
         -log P("capital" | "What is the capital of Ghana? The")
         ...
```

The model learns to produce helpful completions to questions, rather than continuing
documents.

## 2.5 KG-Specialized Fine-Tuning in Detail

The training data for GCR's KG-specialized LLM consists of QA pairs where the
target answer includes reasoning paths from the KG:

```
Input:   "Question: Which film did Blue Hawaii actor star in?"

Target:  "The reasoning path is:
          <PATH> Elvis Presley → film.actor.film → Blue Hawaii </PATH>
          Therefore, the answer is Blue Hawaii."
```

The model learns to generate KG-grounded reasoning paths. The fine-tuning ensures
the model stays "on the rails" of KG structure when generating within path
delimiters.

### Project anchor files

- `gcr/src/qa_prompt_builder.py` — constructs the training prompts
- `gcr/workflow/build_shortest_path_index.py` — pre-computes gold paths for training
- `gcr/workflow/finetune_kg_specialized_llm.py` — orchestrates LoRA fine-tuning

---

# Layer 3: Decoding Strategies

## 3.1 The Core Problem

After training, we have a model that computes `P(y_t | y_<t)`. But how do we
select tokens from this distribution? Different strategies give very different
results.

## 3.2 Greedy Decoding

```
y_t = argmax P(y_t | y_<t)
```

Always pick the single most likely token.

**Problem:** Greedy decoding often leads to repetitive, generic, or locally optimal
but globally poor sequences. A slightly less likely token at step 5 could unlock a
much better continuation, but greedy will never discover it.

## 3.3 Beam Search

Maintain K hypotheses (beams) at each step:

```
Step 1:  ["The"]  (score: -0.3)
          ["A"]    (score: -0.5)     ← top K=2

Step 2:  For each beam, compute top K extensions:
          "The president"     (-0.7)
          "The capital"       (-0.9)
          "A president"       (-1.0)
          "A capital"         (-1.2)
          → Keep top K:
          "The president"     (-0.7)
          "A president"       (-1.0)

Step 3:  Continue...
```

Beam search is a trade-off between exploration and computation. K=1 = greedy.
Typical values are K=4 to K=10 for generation tasks.

## 3.4 Group Beam Search (GCR's Method)

GCR uses a variant called **group beam search** where beams are organised into
groups that must be diverse. This ensures the model generates **diverse reasoning
paths** rather than K nearly-identical ones.

```
Group 1:  beam 1, beam 2    (must diverge from each other)
Group 2:  beam 3, beam 4    (must diverge from each other)
```

Diversity is enforced by modifying the scoring to penalise beams that are similar
to beams in other groups. This is crucial for GCR because the inductive reasoning
step (Step 2) benefits from diverse candidate paths.

**Project anchor:** In `gcr/src/graph_constrained_decoding.py`, look for `group_size`
and `diversity_penalty` parameters. The `HfCausalModel` in
`gcr/src/llms/base_hf_causal_model.py` configures group beam search via HuggingFace's
`generate()` method with `num_beams`, `num_beam_groups`, and `diversity_penalty`.

## 3.5 Sampling Methods

Instead of deterministic selection, sample from the distribution:

```
y_t ~ P(y_t | y_<t)
```

### Temperature Scaling

Controls the "sharpness" of the distribution:

```
P(y_t = v) = exp(logits[v] / T) / Σ_w exp(logits[w] / T)

T → 0:     deterministic (greedy)
T = 1.0:   original distribution
T → ∞:     uniform (random)
T < 1.0:   sharper (more conservative)
```

### Top-K Sampling

Only sample from the K highest-probability tokens:

```
allowed = {tokens with top K probabilities}
y_t ~ P(y_t | y_<t, y_t ∈ allowed)    (re-normalised)
```

### Top-P (Nucleus) Sampling

Only sample from tokens whose cumulative probability exceeds P:

```
sorted_tokens = sort_by_probability(descending)
cumulative = cumulative_sum(probabilities)
cutoff = first index where cumulative > P
allowed = sorted_tokens[:cutoff]
```

## 3.6 Why Decoding Strategy Matters for GCR

In GCR's Step 1 (path generation), the model uses **group beam search** with
constrained decoding. The constraint (trie) limits the token space to valid paths.
The beam search explores multiple paths. The diversity penalty ensures different
paths are discovered.

In GCR's Step 2 (inductive reasoning), an unconstrained LLM (GPT-4o-mini or
similar) reads the diverse paths and produces a final answer. Here, sampling
(top-p, temperature) is used to get varied reasoning.

**The connection:** If GCR's beam search finds only one type of path, the
inductive reasoning step has limited evidence. DCA-Trie's semantic filtering
improves the *quality* of paths in the beam, which should improve the final
answer.

---

# Layer 4: Constrained Decoding

## 4.1 The Core Mechanism

Constrained decoding modifies the token selection process to enforce structural
constraints:

```
Standard:    y_t = sample_from(P(y_t | y_<t))

Constrained: y_t = sample_from(P(y_t | y_<t)   such that   constraint(y_≤t) = True)
```

Implementation: **logit masking**

```
logits[token_id] = -inf   for all token_id ∉ allowed_tokens(y_<t)
```

This is a **hard constraint** because disallowed tokens have exactly zero
probability after softmax.

## 4.2 Types of Constraints

| Constraint Type | What It Enforces | Example |
|-----------------|------------------|---------|
| **Prefix/trie** | Next token must continue a valid prefix | GCR's KG-Trie |
| **Regular expression** | Output must match a regex | JSON schema, phone numbers |
| **Grammar** | Output must follow a context-free grammar | SQL, Python code |
| **Length** | Total length must be within bounds | Summary length limit |
| **Vocabulary** | Certain words are forbidden | Profanity filter |

## 4.3 Trie-Based Constraint (GCR's Approach)

A **trie** (prefix tree) stores all valid token sequences. At each decoding step,
the trie is queried with the current prefix to find valid next tokens.

```
Trie structure:
                    root
                   /    \
                "Elvis"  "The"
                /     \
        "Presley"   "is"
        /               \
    "→"               "→"
    /                   \
"film.actor.film"    "president"
```

At generation step t, the prefix generated so far determines which node in the
trie we are at. The children of that node are the valid next tokens.

**The efficiency insight:** Querying a trie is O(length of prefix), not O(size of
vocabulary). For a trie with millions of paths, the lookup is essentially
instantaneous because `marisa_trie` uses a compressed trie representation that
maps string prefixes to IDs.

**Project anchor:** See `gcr/src/trie.py` for the three trie implementations:

- `Trie` (Python dict-based, used in training)
- `MarisaTrie` (compressed C++ library, used in inference)
- `DummyTrieEntity` (for entity-format constraints in Step 2)

## 4.4 The GCR Constraint Flow

```
1. Question enters the system
2. Extract entities from question → E_q
3. Extract KG subgraph around E_q → G_q
4. Run DFS on G_q up to L hops → all valid paths P
5. Build trie T from tokenized P
6. Decode with prefix_allowed_tokens_fn:
     if inside <PATH>:
         valid = trie.query(prefix)
         logits[~valid] = -inf
     else:
         no constraint
```

**The critical limitation GCR does not address:** Step 4 enumerates *all* paths,
including semantically irrelevant ones. The trie contains paths like
`Marie Curie → people.person.children → Irène Joliot-Curie` when the question
asks about place of birth. Both paths are "valid" in the trie — they exist in the
KG. But only one answers the question.

**This is the precise motivation for DCA-Trie.**

## 4.5 The prefix_allowed_tokens_fn Interface

HuggingFace's `model.generate()` accepts a `prefix_allowed_tokens_fn` callback:

```python
def prefix_allowed_tokens_fn(batch_id, input_ids):
    """
    batch_id: int — which sequence in the batch
    input_ids: torch.Tensor — generated tokens so far for this sequence

    Returns: List[int] — allowed token IDs at this step
    """
    prefix = input_ids.tolist()
    valid_tokens = trie.query(prefix)
    if len(valid_tokens) == 0:
        return all_tokens    # fallback to avoid crash
    return valid_tokens
```

**Project anchor:** This is implemented in
`gcr/src/graph_constrained_decoding.py` class `GraphConstrainedDecoding`, method
`allowed_tokens_fn`.

## 4.6 The Two Modes: Constrained vs. Unconstrained

GCR only constrains decoding within `<PATH>...</PATH>` regions. Outside these tags,
the model generates freely.

Detection mechanism:

```python
def check_constrained_flag(input_ids):
    text = tokenizer.decode(input_ids)
    if '<PATH>' in text and '</PATH>' not in text:
        return True   # currently inside path generation
    return False      # outside
```

The toggle is based on tag boundaries in the generated text. This is simple and
effective, but means the constraint cannot apply to the initial part of the
generation.

## 4.7 The Beam State Challenge (DCA-Trie v2 Motivation)

When using beam search with constrained decoding, each beam may be at a different
position in the trie. In standard GCR, this doesn't matter because the trie is
**static** and shared across all beams.

But DCA-Trie v2 proposes **dynamic expansion**: different beams may discover
different paths that need to be added to the trie. This means each beam needs
its own trie state.

This is the key implementation challenge for v2. The trie must be:

1. **Per-beam**: each beam has its own set of expanded paths
2. **Copyable**: when HuggingFace clones beams during beam search, the trie must
   be cloned too
3. **Efficient**: expanding paths requires KG traversal, which is expensive

---

# Layer 5: Knowledge Graphs and LLMs

## 5.1 Knowledge Graph Fundamentals

A Knowledge Graph (KG) is a structured collection of facts:

```
Triples: (subject, relation, object)

(Elvis Presley,   people.person.place_of_birth,   Tupelo)
(Elvis Presley,   film.actor.film,                Blue_Hawaii)
(Blue_Hawaii,     film.film.directed_by,          Norman_Taurog)
(Norman_Taurog,   people.person.nationality,      United_States)
```

Entities (Elvis Presley) are nodes. Relations (film.actor.film) are labelled edges.
A multi-hop reasoning path connects the question entity to the answer entity through
a chain of triples.

## 5.2 The KG-LLM Synergy

| Capability | LLM | KG |
|------------|-----|-----|
| World knowledge | Broad, shallow, potentially wrong | Narrow, deep, verified |
| Reasoning | Flexible, creative | Rigid, structural |
| Coverage | General | Domain-specific |
| Freshness | Frozen at training | Updatable |
| Hallucination | Inherent | Impossible |

**The goal of KG-enhanced LLM reasoning:** combine LLM flexibility with KG
fidelity.

## 5.3 Approaches to KG-Enhanced LLMs

### 5.3.1 Retrieval-Augmented (RAG for KG)

Retrieve relevant triples from the KG and insert them into the LLM's context:

```
Question: "Which film did Blue Hawaii actor star in?"

Retrieved triples:
- Elvis Presley → film.actor.film → Blue_Hawaii
- Blue_Hawaii → film.film.starring → Elvis_Presley

Prompt: "Given the following facts:
1. Elvis Presley starred in Blue_Hawaii
2. Blue_Hawaii starred Elvis Presley

Answer: Blue_Hawaii"
```

**Limitation:** The retriever may miss the critical fact. The LLM may ignore
retrieved facts and rely on parametric knowledge.

### 5.3.2 Agent-Based

An LLM agent interacts with the KG through API calls:

```
LLM:   *What entity does the question refer to?* → "Elvis Presley"
LLM:   *What films did Elvis Presley star in?* → KG.query("Elvis Presley", "film.actor.film")
KG:    [Blue_Hawaii, ...]
LLM:   *Which film answers the question?* → "Blue_Hawaii"
```

**Limitation:** Requires multiple LLM calls, error propagates across steps,
agent planning is unreliable.

### 5.3.3 Graph-Constrained Decoding (GCR)

The LLM generates paths, but the decoding is constrained to valid KG paths.

**Advantage over RAG:** The constraint is hard — the model *cannot* generate an
invalid path. No retrieval step to fail.
**Advantage over agents:** Single generation pass, no iterative API calls.
**Trade-off:** Requires a fine-tuned KG-specialized LLM.

## 5.4 The Faithfulness Guarantee

GCR's claim of "zero reasoning hallucination" means:

> Every generated reasoning path corresponds to a real sequence of triples
> in the KG.

This is guaranteed by construction because:

1. The trie is built from actual KG paths
2. The decoding constraint ensures only trie-valid tokens are generated
3. The generated path is therefore a valid KG path

**But — and this is DCA-Trie's core insight — faithfulness ≠ relevance.**
A path can be faithful to the KG and irrelevant to the question simultaneously.

## 5.5 The Permissiveness Problem

GCR's trie admits **all structurally valid paths**. For a question about Marie
Curie's birthplace, the trie includes:

```
Relevant:   Marie Curie → people.person.place_of_birth → Warsaw
Irrelevant: Marie Curie → people.person.education → Sorbonne
Irrelevant: Marie Curie → award_received → Nobel_Prize
Irrelevant: Marie Curie → people.person.children → Irène Joliot-Curie
Irrelevant: Marie Curie → people.person.spouse → Pierre Curie
```

The model must navigate this increasingly large space at each step. With beam
search, beams may explore irrelevant branches, wasting capacity.

**SIR (Semantic Irrelevance Ratio):**

```
SIR = 1 - max_{p ∈ trie} cosine_similarity(embed(p), embed(q))
```

SIR = 0: all paths are relevant
SIR = 1: no path is relevant

DCA-Trie's threshold sweep found that at τ = 0.55, the trie size is reduced by
~50% with zero false negatives (on 5 test questions).

**Project anchor:** `dca_trie/sir_measurement.py` implements SIR computation.

---

# Layer 6: Semantic Filtering and DCA-Trie

## 6.1 The Design Principle

> **Ng's framework for LLM application design:**
>
> 1. What is the **input**? (question)
> 2. What is the **output**? (reasoning path)
> 3. What is the **constraint**? (must be valid in KG)
> 4. How do we **evaluate**? (answer accuracy + path faithfulness)

GCR answers 1-4. DCA-Trie adds a fifth question:
> 5. Is the constraint **tight enough**? (does it admit irrelevant paths?)

## 6.2 Semantic Similarity as a Relevance Proxy

DCA-Trie uses sentence-transformer embeddings (all-MiniLM-L6-v2) to score path
relevance:

```python
query_embedding = model.encode(question)            # 384-dim vector
path_embedding = model.encode(path_to_string(p))    # 384-dim vector
score = cosine_similarity(query_embedding, path_embedding)
```

The assumption: paths that are semantically similar to the question are likely
relevant. This is a heuristic, and the project is explicit about its limitations.

## 6.3 DCA-Trie Variants

### v1: Static Filtering (Implemented)

Filter paths at trie construction time, before decoding begins:

```python
def build_v1_trie(question, kg_subgraph, tau):
    all_paths = dfs(kg_subgraph)                 # all structural paths
    query_emb = encode(question)                  # encode once
    filtered = []
    for path in all_paths:
        score = cosine_similarity(encode(path), query_emb)
        if score >= tau:
            filtered.append(path)
    return build_marisa_trie(filtered)
```

**Effect:** The trie is smaller, so the model has fewer irrelevant branches to
explore. But the filtering is static — it doesn't adapt as the model generates.

### v2: Dynamic Expansion (Designed)

Start with a small trie and expand it step-by-step during decoding:

```python
def v2_decode(question, kg_subgraph, tau, beam_size):
    # Each beam has its own trie state
    beam_tries = [empty_trie() for _ in range(beam_size)]

    for step in range(max_steps):
        for beam_id, beam in enumerate(beams):
            prefix = beam.generated_text

            # Expand trie with paths relevant to current prefix
            new_paths = expand_paths(kg_subgraph, prefix, question, tau)
            beam_tries[beam_id].add_paths(new_paths)

            # Constrain decoding with this beam's trie
            valid_tokens = beam_tries[beam_id].query(prefix)
            beam.step(valid_tokens)
```

**Challenge:** Per-beam trie state must follow beam cloning in HuggingFace's beam
search. The `batch_id` parameter in `prefix_allowed_tokens_fn` is not stable across
steps, so state must be tracked by generated prefix content.

### v3: Semantic Backtracking (Designed)

When the model goes down an unpromising path, backtrack to the last branching
point and prune that branch from the trie. This adds a non-monotonic recovery
mechanism.

**Project anchor:** `docs/DCA_TRIE_V3_BACKTRACKING.md` has the full design.

## 6.4 The Embedding Limitation

**Critical caveat from the project itself:**

> "MiniLM cosine similarity measures topical overlap, not inferential relevance."

Example:

```
Question: "Where was Marie Curie born?"
Path A:   "Marie Curie → people.person.place_of_birth → Warsaw"
           → Cosine similarity with question: high (correct)

Path B:   "Marie Curie → people.person.nationality → Polish"
           → Cosine similarity with question: also high (WRONG, but topically related)

Path C:   "Marie Curie → award_received → Nobel_Prize_in_Physics"
           → Cosine similarity: lower (correctly identified as less relevant)
```

The embedding captures that both "birth" and "nationality" are biographical, so
both score highly. Only a deeper model of the question's intent could distinguish
them.

**This is the central open problem DCA-Trie surfaces:** relevance is not
cosine-similarity. Cosine similarity is a proxy. The project proposes
answer-type-guided hard constraints as a more principled direction.

## 6.5 The Broader Research Context

DCA-Trie is part of a family of constrained decoding approaches:

| Work | Constraint Type | Dynamic? | Semantics-Aware? |
|------|----------------|----------|-----------------|
| GCR (Luo et al., 2025) | KG-Trie (static) | No | No |
| DoG (2025) | Dynamic graph expansion | Yes | No |
| ReFactX (2025) | Prefix-tree over verbalized facts | No | No |
| RwT (2025) | MCTS over KG paths | Yes | No |
| **DCA-Trie v1** | KG-Trie + static semantic filter | No | Yes (cosine) |
| **DCA-Trie v2** | Per-beam dynamic trie | Yes | Yes (cosine) |
| **DCA-Trie v3** | With backtracking | Yes | Yes (cosine + history) |

The field is converging on the idea that constraints must be both **dynamic**
(adapt to generation context) and **semantics-aware** (use question meaning, not
just graph structure).

---

# Layer 7: Learning Path

## 7.1 Core Resources

### From Andrej Karpathy

These should be studied **in order**:

1. **"Neural Networks: Zero to Hero"** (video course)
   <https://karpathy.ai/zero-to-hero.html>
   - Start here if you're not comfortable with backpropagation and PyTorch basics
   - Builds micrograd → makemore → GPT from scratch

2. **"The spelled-out intro to language modeling: building makemore"** (video, ~2h)
   - Character-level bigram model
   - Introduces the autoregressive framework used by all LLMs
   - **Key for this project:** The concept of modelling P(y_t | y_{<t})

3. **"Let's build GPT: from scratch, in code, spelled out"** (video, ~2h)
   - Builds nanoGPT on Shakespeare
   - Walks through every component: embedding, attention, FFN, residual, layernorm
   - **Key for this project:** Understanding what exactly the LLM is doing when
     GCR constrains its output

4. **"Let's build the GPT Tokenizer"** (video, ~2h)
   - BPE tokenization in detail
   - **Key for this project:** Why the trie stores token IDs (not words), and
     how `path_to_string` conversion works

5. **"Intro to Large Language Models"** (video, 1h)
   - General audience overview
   - Training, inference, scaling laws, tool use, security
   - **Key for this project:** The "LLM as OS" analogy for understanding how
     constrained decoding is like adding a system call interface

6. **"Deep Dive into LLMs like ChatGPT"** (video, 3.5h)
   - The most comprehensive treatment
   - Covers: pretraining data, tokenization, architectures, fine-tuning, RLHF,
     evaluation, deployment
   - **Key for this project:** The sections on fine-tuning and inference are
     directly applicable to understanding GCR's pipeline

### From Andrew Ng

1. **"Generative AI with Large Language Models"** (course, ~10h)
   <https://deeplearning.ai/courses/generative-ai-with-llms/>
   - The full LLM lifecycle: data → pretraining → fine-tuning → evaluation → deployment
   - Scaling laws, RLHF, prompt engineering
   - **Key for this project:** The fine-tuning section explains why
     KG-specialized LLM training works (Layer 2 of this guide)

2. **"Building Systems with the ChatGPT API"** (course, ~2h)
   - Chain of thought, chaining prompts, classification, evaluation
   - **Key for this project:** The evaluation framework can be applied to
     evaluating DCA-Trie vs GCR output quality

3. **"Agentic AI with Andrew Ng"** (course)
   <https://deeplearning.ai/courses/agentic-ai/>
   - Reflection, tool use, planning, multi-agent
   - **Key for this project:** The reflection pattern is related to DCA-Trie v3's
     backtracking idea

### Research Papers

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
    - The original transformer paper
    - Read the Karpathy tutorial first, then read this for the formal specification

2. **"Graph-Constrained Reasoning"** (Luo et al., ICML 2025)
    - The method DCA-Trie builds on
    - Read the `gcr/src/` code alongside it

3. **"Decoding on Graphs"** (DoG, ACL 2025)
    - Dynamic graph expansion during constrained decoding
    - Directly related to DCA-Trie v2's approach

4. **"ReFactX"** (ISWC 2025)
    - Scales constrained decoding to 800M facts using prefix-tree index
    - Alternative approach to GCR's subgraph extraction

5. **"Scaling Laws for Neural Language Models"** (Kaplan et al., 2020)
    - **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022)
    - The empirical foundation for why LLMs work

## 7.2 Learning Paths by Role

### If you are implementing DCA-Trie (core engineering)

1. Karpathy's "Let's build GPT" — understand the engine
2. Read `gcr/src/trie.py` and `gcr/src/graph_constrained_decoding.py` — understand the constraint
3. Read `dca_trie/v1_trie_builder.py` — understand the current implementation
4. Read `docs/DCA_TRIE_HANDBOOK.md` — understand the full system
5. Study HuggingFace's `generate()` source (particularly beam search) — understand
   why v2's per-beam trie is challenging
6. Implement v1 → evaluate → implement v2

### If you are evaluating / benchmarking

1. Ng's "Generative AI with LLMs" — understand evaluation methodology
2. Read `gcr/src/utils/qa_utils.py` — understand the metrics (Hits@1, F1, accuracy)
3. Read `dca_trie/sir_measurement.py` — understand SIR
4. Run `phase1_sir_measurement.ipynb` — replicate the threshold sweep

### If you are extending the research

1. All of the above
2. Read DoG and ReFactX papers — understand the landscape
3. Read `docs/DCA_TRIE_V3_BACKTRACKING.md` — understand the extension path
4. Identify a limitation: cosine similarity as relevance proxy? KG incompleteness?
   Multi-hop path scoring?
5. Design and test your extension

## 7.3 Exercises to Verify Understanding

### Exercise 1: Trace the GCR Flow

Pick a question from `dca_trie/test_mini_freebase.py` (e.g., "Who was the spouse
of Barack Obama?").

1. What entities does the question contain? → `q_entity`
2. What relations connect these entities in Freebase?
3. What paths does DFS discover?
4. How large is the trie?
5. What tokens are valid at step 1 of `<PATH>` generation?

**Verify by reading:** `gcr/src/graph_utils.py` → `dfs()`, and `gcr/src/trie.py` → `Trie`.

### Exercise 2: Compute SIR

For the same question:

1. Encode the question with all-MiniLM-L6-v2
2. Encode each path in the trie
3. Compute max cosine similarity
4. Compute SIR = 1 - max_sim
5. Try different τ values and see which paths are filtered

**Verify by running:** `dca_trie/sir_measurement.py` on the test questions.

### Exercise 3: Modify the Trie

Create a modified version of `V1TrieBuilder` that uses a different similarity
metric (e.g., Euclidean distance instead of cosine, or a different embedding model).

1. How does the filtered path set change?
2. How does SIR change?
3. Does the answer accuracy change?

**Verify:** Compare against the baseline threshold sweep results in
`data/threshold_sweep_results.json`.

### Exercise 4: Model the Beam State Problem

Draw the state diagram for DCA-Trie v2's per-beam trie:

1. Beam search starts with K beams, all with the same initial trie
2. At step 1, each beam generates a different token
3. Each beam's trie expands differently (different paths are relevant given the prefix)
4. HuggingFace clones/prunes beams — trie state must follow
5. When a beam is pruned, its trie is discarded
6. When a beam is cloned, its trie is cloned

**Implementation question:** Where in HuggingFace's beam search do you hook into
to manage this state? Hint: look at the `BeamSearchScorer` and
`BeamHypotheses` classes.

## 7.4 Key Equations Reference

| Concept | Equation | Context |
|---------|----------|---------|
| Next-token prediction | P(y_t | y_<t) | The fundamental LLM operation |
| Training loss | L = -Σ_t log P(y_t | y_<t) | Cross-entropy over all positions |
| Attention score | score(i,j) = q_i · k_j / √d_k | Core attention mechanism |
| Multi-head attention | output = Concat(head_1,...,head_H) · W_O | Parallel attention computation |
| FFN computation | FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2 | Per-token processing |
| Temperature sampling | P(v) = exp(logit[v]/T) / Σ_w exp(logit[w]/T) | Decoding control |
| Cosine similarity | sim(a,b) = a·b / (‖a‖·‖b‖) | DCA-Trie relevance scoring |
| SIR | SIR(q,t) = 1 - max_{p ∈ trie_t} cos(emb(p), emb(q)) | Oracle permissiveness metric |

---

## Appendix: Mapping the Project's Architecture

### Where Each Layer Lives in the Codebase

| Layer | Concept | File(s) |
|-------|---------|---------|
| 1 | Transformer (used via HF) | `gcr/src/llms/base_hf_causal_model.py` |
| 1 | Tokenization | HuggingFace `AutoTokenizer` |
| 1 | Attention / FFN | Inside the loaded model (HF) |
| 2 | Pre-training | External (Llama-3.1-8B from Meta) |
| 2 | Fine-tuning (LoRA) | `gcr/workflow/finetune_kg_specialized_llm.py` |
| 3 | Group beam search | `gcr/src/llms/base_hf_causal_model.py` |
| 3 | Decoding strategies | HuggingFace `model.generate()` |
| 4 | Trie constraint | `gcr/src/trie.py` + `gcr/src/graph_constrained_decoding.py` |
| 4 | Logit masking | `gcr/src/graph_constrained_decoding.py` `allowed_tokens_fn` |
| 5 | Knowledge graph | `gcr/src/utils/graph_utils.py` (NetworkX) |
| 5 | KG subgraph extraction | `gcr/src/qa_prompt_builder.py` |
| 6 | Semantic scoring | `dca_trie/semantic_scorer.py` |
| 6 | SIR metric | `dca_trie/sir_measurement.py` |
| 6 | DCA-Trie v1 | `dca_trie/v1_trie_builder.py` |
| 6 | DCA-Trie v2/v3 | `docs/DCA_TRIE_HANDBOOK.md` + `docs/DCA_TRIE_V3_BACKTRACKING.md` |

### Data Flow Diagram

```
Question(q)
    │
    ▼
┌─────────────────────────────────────┐
│  Entity Extraction (from question)  │
│  → q_entity IDs                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Graph Extraction (from KG)         │
│  → G_q: subgraph around q_entities  │  Layer 5
│     (L-hop neighbourhood)           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Path Enumeration (DFS on G_q)      │
│  → all paths P up to L hops         │  Layer 5
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  DCA-Trie Filtering (v1)            │
│  score(p, q) = cos(emb(p), emb(q))  │  Layer 6
│  keep if score ≥ τ                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  MarisaTrie Construction            │
│  → tokenized, compressed prefix tree│  Layer 4
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Constrained Decoding (Step 1)      │
│  KG-specialized LLM + group beam    │  Layers 3+4
│  prefix_allowed_tokens_fn(trie)     │
│  → K diverse reasoning paths        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Inductive Reasoning (Step 2)       │
│  General LLM (GPT-4o-mini)         │
│  reads K paths → produces answer    │
└─────────────────────────────────────┘
    │
    ▼
Answer(a)
```

---

*"The constraint oracle should know what the question is asking, not just what
paths structurally exist in the graph."* — DCA-Trie thesis statement

*"An LLM is not a database. It's more like a dream simulator of the internet."*
— Andrej Karpathy

*"The key to building effective AI applications is understanding what the model
can do, what it cannot do, and designing the architecture around those
constraints."* — Andrew Ng
