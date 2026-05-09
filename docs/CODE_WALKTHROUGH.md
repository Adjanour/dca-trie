# Code Walkthrough

This document provides detailed line-by-line walkthroughs of the most important files in the GCR codebase. Read these in order.

---

## 1. `gcr/src/trie.py` — The Constraint Data Structure

### Class: `Trie` (Simple Python dict-based trie)

```python
class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
        self.append_trie = None
        self.bos_token_id = None
```

**What it is:** A recursive dictionary structure where each key is a token ID and each value is a sub-dict for the next token.

**The `append_trie` mechanism:** Allows chaining two tries together. Used for handling the transition from the path trie to the general vocabulary when the path is complete.

**`_add_to_trie(sequence, trie_dict):**
```python
@staticmethod
def _add_to_trie(sequence: List[int], trie_dict: Dict):
    if sequence:
        if sequence[0] not in trie_dict:
            trie_dict[sequence[0]] = {}
        Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])
```

Recursively inserts each token ID as a key in nested dicts. A leaf is an empty dict `{}`.

**`_get_from_trie(prefix_sequence, trie_dict):**
```python
if len(prefix_sequence) == 0:
    output = list(trie_dict.keys())
    if append_trie and bos_token_id in output:
        output.remove(bos_token_id)
        output += list(append_trie.trie_dict.keys())
    return output
```

When prefix is empty (at the root), return all valid first tokens. If an `append_trie` is linked and `bos_token_id` is in the output, it means "the path is done, now fall through to the appended trie." It removes the BOS token from output and adds the append_trie's root keys instead.

```python
elif prefix_sequence[0] in trie_dict:
    return Trie._get_from_trie(
        prefix_sequence[1:], trie_dict[prefix_sequence[0]],
        append_trie, bos_token_id,
    )
```

Follow the first token of the prefix into the next level of the dict.

```python
else:
    if append_trie:
        return append_trie.get(prefix_sequence)
    else:
        return []
```

If the prefix doesn't match, fall through to `append_trie` if available, otherwise return empty.

### Class: `MarisaTrie` (Production trie used in inference)

This is the actual trie used during inference. The simpler `Trie` class is used mostly for training data processing.

**Constructor:**
```python
def __init__(self, sequences, cache_fist_branch=True, max_token_id=256001):
    # Map token IDs to characters for storage in the string trie
    self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
        [chr(i) for i in range(65000, max_token_id + 10000)]
        if max_token_id >= 55000
        else []
    )
    self.char2int = {self.int2char[i]: i for i in range(max_token_id)}
    
    # Cache first-level tokens for O(1) root lookup
    self.cache_fist_branch = cache_fist_branch
    if self.cache_fist_branch:
        self.zero_iter = list({sequence[0] for sequence in sequences})
    
    # Build the marisa trie
    self.trie = marisa_trie.Trie(
        "".join([self.int2char[i] for i in sequence])
        for sequence in sequences
    )
```

**The char encoding trick:** Token IDs can go up to 256,000+ (Llama-3.1 has ~128K vocabulary). The `chr()` function only works for Unicode code points up to 0x10FFFF, so:
- IDs 0-54999: `chr(id)` directly
- IDs 55000-64999: **skipped** (potential conflicts)
- IDs 65000+: `chr(id + offset)` to avoid the gap

**`get(prefix_sequence)`:**
```python
def get(self, prefix_sequence):
    if self.cache_fist_branch and len(prefix_sequence) == 0:
        return self.zero_iter
    else:
        key = "".join([self.int2char[i] for i in prefix_sequence])
        # Find all strings that start with `key` and return their next character
        return list({
            self.char2int[e[len(key)]] 
            for e in self.trie.keys(key) 
            if len(e) > len(key)
        })
```

1. For root query (empty prefix), return cached first tokens
2. For deeper queries, find all stored paths starting with the prefix
3. For each match longer than the prefix, extract the next character (token ID)

### Class: `DummyTrieEntity` (Used during Step 2)

```python
class DummyTrieEntity(object):
    def __init__(self, return_values, codes):
        self._return_values = list(
            set(return_values).difference(
                set(codes[e] for e in ("start_mention_token", "end_mention_token", "start_entity_token"))
            )
        )
        self._codes = codes
    
    def get(self, indices, depth=0):
        if len(indices) == 0 and depth == 0:
            return self._codes["end_mention_token"]
        elif len(indices) == 0 and depth == 1:
            return self._codes["start_entity_token"]
        elif len(indices) == 0:
            return self._return_values
        elif len(indices) == 1 and indices[0] == self._codes["end_entity_token"]:
            return self._codes["EOS"]
        else:
            return self.get(indices[1:], depth=depth + 1)
```

This implements a special trie that enforces entity formatting: `[end_mention] [start_entity] entity_name [end_entity] [EOS]`. Used in Step 2 when GCR generates answers constrained to entity strings.

---

## 2. `gcr/src/graph_constrained_decoding.py` — The Constraint Hook

This is the file that connects the trie to the HuggingFace generation loop.

### `GraphConstrainedDecoding.__init__`

```python
def __init__(self, tokenizer, trie, start_token_ids=None, end_token_ids=None, enable_constrained_by_default=False):
    self.tokenizer = tokenizer
    self.trie = trie
    self.start_token = start_token_ids  # <PATH> token
    self.end_token = end_token_ids      # </PATH> token
    self.all_tokens = list(range(len(tokenizer)))  # full vocabulary
    self.constrained_flag = enable_constrained_by_default
    self.L_input = None
```

### `allowed_tokens_fn(batch_id, sent)` — The Core Logic

This function is called by HuggingFace's `model.generate()` at every generation step.

```python
def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
    constrained_flag = self.constrained_flag
    
    # Check if we've entered the constrained region
    if self.start_token is not None and self.end_token is not None:
        constrained_flag, L_input = self.check_constrained_flag(sent)
    else:
        if self.L_input is None:
            self.L_input = len(sent)
        L_input = self.L_input
    
    allow_tokens = self.all_tokens  # default: all tokens allowed
    
    if constrained_flag:
        # Only use trie lookup within <PATH>...</PATH>
        allow_tokens = self.trie.get(sent.tolist()[L_input:])
        if len(allow_tokens) == 0:
            return self.all_tokens  # fallback if trie returns empty
    
    return allow_tokens
```

**Step-by-step:**
1. Check if we're inside `<PATH>...</PATH>` tokens
2. If yes: look up valid next tokens in the trie based on the path prefix generated so far
3. If no: allow the full vocabulary
4. If trie returns empty (shouldn't happen, but safe): fall back to full vocab

### `check_constrained_flag(sent)` — Entity Boundary Detection

```python
def check_constrained_flag(self, sent: torch.Tensor):
    matched_start_token = torch.where(sent == self.start_token)[0]
    if len(matched_start_token) == 0:
        return False, len(sent)
    
    last_start_tokens = torch.where(sent == self.start_token)[0][-1]
    end_token_number = len(torch.where(sent[last_start_tokens:] == self.end_token)[0])
    
    if end_token_number == 0:
        self.last_start_token = last_start_tokens
        return True, last_start_tokens
    else:
        self.last_start_token = None
        return False, len(sent)
```

**Logic:**
1. Find the last `<PATH>` token in the generated sequence
2. Count `</PATH>` tokens after it
3. If 0 close tags after the last open tag → we're inside a path → constrain
4. If ≥1 close tags → the path is closed → don't constrain

---

## 3. `gcr/src/qa_prompt_builder.py` — Building Inputs and Graph Indexes

### `GraphConstrainedPromptBuilder.get_graph_index(question_dict)`

This is the method that builds the MarisaTrie from the graph subgraph:

```python
def get_graph_index(self, question_dict):
    if "paths" in question_dict:
        paths_list = question_dict["paths"]
    else:
        g = utils.build_graph(question_dict["graph"], self.undirected)
        if self.add_rule:
            rules = question_dict['predicted_paths']
            if len(rules) > 0:
                paths_list = self.apply_rules(g, rules, question_dict["q_entity"])
            else:
                paths_list = utils.dfs(g, question_dict["q_entity"], self.index_path_length)
        else:
            paths_list = utils.dfs(g, question_dict["q_entity"], self.index_path_length)
    
    paths_list_str = [utils.path_to_string(p) for p in paths_list]
    tokenized_paths = self.tokenizer(paths_list_str, padding=False, add_special_tokens=False).input_ids
    tokenized_path_list = [ids + [self.tokenizer.eos_token_id] for ids in tokenized_paths]
    return MarisaTrie(tokenized_path_list, max_token_id=len(self.tokenizer) + 1)
```

**Step-by-step:**
1. Get all paths (from pre-computed cache or DFS)
2. Convert paths to strings: `["Entity -> rel -> Entity -> ...", ...]`
3. Tokenize each path string → list of token ID lists
4. Append `eos_token_id` to each sequence (marks path boundaries)
5. Build MarisaTrie

### `JointReasoningPromptBuilder.get_graph_index` (Used in Step 1)

Same as above, but wraps each path in `<PATH>...</PATH>` tokens:

```python
paths_list_str = [f"{self.PATH_START_TOKEN}{utils.path_to_string(path)}{self.PATH_END_TOKEN}" 
                  for path in paths_list]
```

---

## 4. `gcr/src/llms/graph_constrained_decoding_model.py` — The GCR Model

```python
class GraphConstrainedDecodingModel(HfCausalModel):
    def generate_sentence(self, llm_input, trie, start_token_ids=None, 
                          end_token_ids=None, enable_constrained_by_default=True):
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, 
                                        end_token_ids, enable_constrained_by_default)
        res = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_cfg,
            prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = []
        if len(res.sequences) == 1:
            return self.tokenizer.decode(res.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        for r in res.sequences:
            response.append(self.tokenizer.decode(r[input_ids.shape[1]:], skip_special_tokens=True))
        return response
```

**Key detail:** `prefix_allowed_tokens_fn=gcr.allowed_tokens_fn` is the connection point. HuggingFace calls this function after every generated token to determine which tokens are valid next.

---

## 5. `gcr/workflow/predict_paths_and_answers.py` — Step 1 Entry Point

### The `prediction` function (per-question):

```python
def prediction(data, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None  # skip already-processed (resume support)
    
    input_query, ground_paths, trie = input_builder.process_input(data)
    if trie is None:
        return None  # no valid paths → skip
    
    # Get special tokens for constrained region
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    
    input = model.prepare_model_prompt(input_query)  # add chat template
    prediction = model.generate_sentence(
        input, trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,  # only constrain within <PATH>...</PATH>
    )
    
    return {"id": id, "question": question, "prediction": prediction, ...}
```

### Main loop:

```python
def main(args, LLM):
    dataset = load_dataset(input_file, split=args.split)
    model = LLM(args)
    model.prepare_for_inference()
    input_builder = PathGenerationWithAnswerPromptBuilder(...)
    
    for data in tqdm(dataset):
        res = prediction(data, processed_list, input_builder, model)
        if res is not None:
            fout.write(json.dumps(res) + "\n")
    
    eval_path_result_w_ans(os.path.join(output_dir, 'predictions.jsonl'))
```

---

## 6. `gcr/src/utils/graph_utils.py` — Graph Operations

### `dfs(graph, start_node_list, max_length)` — Path Enumeration

```python
def dfs(graph, start_node_list, max_length):
    def dfs_visit(node, path):
        if len(path) > max_length:
            return
        for neighbor in graph.neighbors(node):
            rel = graph[node][neighbor]["relation"]
            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_length:
                path_lists.add(tuple(new_path))
            dfs_visit(neighbor, new_path)
    
    path_lists = set()
    for start_node in start_node_list:
        dfs_visit(start_node, [])
    return list(path_lists)
```

Collects **all paths** up to `max_length` starting from the question entities. This produces the set of all structurally valid reasoning chains.

### `build_graph(graph, undirected)` — From Triplets to NetworkX

```python
def build_graph(graph: list, undirected=False):
    G = nx.Graph() if undirected else nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G
```

### `get_truth_paths(q_entity, a_entity, graph)` — Gold Path Extraction

```python
def get_truth_paths(q_entity, a_entity, graph):
    paths = []
    for h in q_entity:
        for t in a_entity:
            for p in nx.all_shortest_paths(graph, h, t):
                paths.append(p)
    # Add relations to paths
    result_paths = []
    for p in paths:
        tmp = [(p[i], graph[p[i]][p[i+1]]["relation"], p[i+1]) for i in range(len(p)-1)]
        result_paths.append(tmp)
    return result_paths
```

Finds the **shortest paths** between question and answer entities. These are the "gold" paths used for evaluation of path F1.

---

## 7. `gcr/src/utils/qa_utils.py` — Evaluation

### `eval_hit(prediction, answer)` — Hits@1

```python
def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0
```

Returns 1 if the prediction contains any of the ground truth answers (after normalization).

### `eval_f1(prediction, answer)` — F1 Score

```python
def eval_f1(prediction, answer):
    # recall: fraction of ground truth answers found in prediction string
    ans_recalled = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            ans_recalled += 1
    recall = ans_recalled / len(answer)
    
    # precision: fraction of predictions that match any ground truth
    prediction_correct = 0
    for p in prediction:
        for a in answer:
            if match(p, a):
                prediction_correct += 1
                break
    precision = prediction_correct / len(prediction)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
```

### `normalize(s)` — Answer Normalization

```python
def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)  # remove articles
    s = re.sub(r"\b(<pad>)\b", " ", s)     # remove padding tokens
    s = " ".join(s.split())
    return s
```
