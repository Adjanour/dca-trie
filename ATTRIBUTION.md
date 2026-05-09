# Attribution

## GCR Baseline Code

The code in `gcr/` is vendored from the [Graph-Constrained Reasoning (GCR)](https://github.com/RManLuo/graph-constrained-reasoning)
repository by Luo et al. (ICML 2025). It is included **unmodified** except for
mechanical import path changes (prefixing with `gcr.`) necessitated by the
directory restructure for academic clarity.

The **exact original** is preserved as a git submodule at `vendor/gcr/`
(pinned to commit `9518e8e`). Use `git submodule update --init` to clone it.

| Original (`vendor/gcr/`) | This repo (`gcr/`) |
|--------------------------|-------------------|
| `src/` | `gcr/src/` (imports prefixed with `gcr.src.`) |
| `workflow/` | `gcr/workflow/` |
| `scripts/` | `gcr/scripts/` |
| `accelerate_configs/` | `gcr/accelerate_configs/` |
| `resources/` | `gcr/resources/` |

### License

The GCR code is released under the MIT License by its authors. See `LICENSE`
for the full text.

### Citation (GCR)

```bibtex
@inproceedings{luo2025icml-graphconstrained,
  title     = {{Graph-Constrained Reasoning: Faithful Reasoning on Knowledge
                Graphs with Large Language Models}},
  author    = {Luo, Linhao and Zhao, Zicheng and Haffari, Gholamreza and
               Li, Yuan-Fang and Gong, Chen and Pan, Shirui},
  booktitle = {Proceedings of the 42nd International Conference on Machine
               Learning},
  year      = {2025},
  pages     = {41540--41565},
  volume    = {267},
}
```

---

## DCA-Trie Contribution

All code in `dca_trie/`, `experiments/`, and `docs/` (except where noted) is
original contribution by:

- Bernard Kirk Adjanor Katamanos
- Erica Amonor
- Joseph Osei Nyarko
- Jessica Afua Etornam Nsafoah

Supervised by Dr. Eric Affum, University of Mines and Technology (UMaT), Ghana.

### Citation (DCA-Trie)

```bibtex
@misc{dca-trie-2025,
  title  = {{DCA-Trie}: Dynamic Context-Aware Trie for Faithful
            {KG}-Constrained {LLM} Reasoning},
  author = {Katamanos, Bernard Kirk Adjanor and Amonor, Erica and
            Nyarko, Joseph Osei and Nsafoah, Jessica Afua Etornam},
  year   = {2025},
}
```

---

## How to Distinguish the Two

| Criteria | `gcr/` (baseline) | `dca_trie/` (contribution) |
|----------|-------------------|---------------------------|
| Origin | Vendored from RManLuo | Original work |
| License | MIT (original authors) | MIT (current authors) |
| Purpose | KG-Trie constrained decoding | Semantic path filtering |
| Key files | `trie.py`, `graph_constrained_decoding.py` | `semantic_scorer.py`, `v1_trie_builder.py` |
| Model used | KG-specialized LLM + general LLM | Sentence Transformer (MiniLM) |
| Novelty | — | SIR metric, semantic filtering, answer-type constraints |

When importing from the baseline in DCA-Trie code, the `gcr.` prefix makes
the boundary explicit:

```python
from gcr.src.trie import MarisaTrie         # ⬅ baseline
from dca_trie.semantic_scorer import SemanticScorer  # ⬅ contribution
```
