"""
Threshold Sweep for DCA-Trie v1.

Sweeps tau in [0.1, 0.6] step 0.05 and records:
  - FNR (False Negative Rate): fraction of questions where gold paths are pruned
  - Trie reduction %: (original_size - filtered_size) / original_size
  - SIR of the filtered trie
  - Hits@1 of the filtered trie (if ground truth answers available)

Usage:
    python experiments/threshold_sweep_v1.py                         # WebQSP (default, 100 questions)
    python experiments/threshold_sweep_v1.py --dataset cwq           # CWQ
    python experiments/threshold_sweep_v1.py --dataset test          # synthetic test data
    python experiments/threshold_sweep_v1.py --num 500               # more questions
    python experiments/threshold_sweep_v1.py --tau_max 0.50 --tau_step 0.02  # finer grid
"""

import argparse
import json
import sys
import os
import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Imports that don't depend on path_to_string patch
from dca_trie.semantic_scorer import SemanticScorer
from dca_trie.sir_measurement import SIRMeasurer
from dca_trie.mid_resolver import MidResolver
from dca_trie.v1_trie_builder import V1TrieBuilder
from gcr.src.utils.graph_utils import build_graph, dfs, get_truth_paths


# Local test data (fallback)
def _get_test_questions():
    from dca_trie.test_mini_freebase import get_all_test_questions

    return get_all_test_questions()


ID_PATTERN = re.compile(r"^m\.\d+")

# V1TrieBuilder imports path_to_string internally; we patch via its module
# after optionally applying MID resolution
_ORIG_PATH_TO_STR = None


def sweep_tau_on_questions(
    questions,
    tokenizer,
    scorer,
    tau_values=None,
    measure_sir=True,
):
    """
    Run threshold sweep over a list of question_dicts.

    Args:
        questions: list of dicts matching GCR's WebQSP format
        tokenizer: HF tokenizer (for building trie)
        scorer: SemanticScorer instance
        tau_values: list of tau thresholds to try
        measure_sir: whether to compute SIR for filtered trie

    Returns:
        list of dicts, one per tau value, with keys:
          tau, fnr, total, missed_ids, avg_reduction,
          avg_filtered_size, avg_original_size, avg_sir
    """
    if tau_values is None:
        tau_values = np.arange(0.10, 0.65, 0.05)

    results = []
    original_sizes = []
    all_gold_strs = []  # gold path strings per question
    all_original_paths = []

    # Pre-compute original sizes and gold paths
    for q in questions:
        g = build_graph(q["graph"])
        paths = _dfs_paths(g, q)
        original_sizes.append(len(paths))
        all_original_paths.append(paths)
        all_gold_strs.append(_get_gold_path_strs(q))

    for tau in tau_values:
        tau = round(tau, 2)
        total = 0
        fn_count = 0
        missed_ids = []
        filtered_sizes = []
        original_sizes_for_tau = []
        sir_values = []

        # Re-init builder for this tau
        builder = V1TrieBuilder(
            tokenizer=tokenizer,
            scorer=scorer,
            tau=tau,
        )

        for idx, q in enumerate(questions):
            total += 1

            # Get filtered paths
            filtered = builder.filter_paths_only(q)
            filtered_set = set(filtered)
            filtered_sizes.append(len(filtered))
            original_sizes_for_tau.append(original_sizes[idx])

            # Check FNR: do any gold paths survive?
            gold_strs = all_gold_strs[idx]
            if gold_strs:
                any_survive = any(g in filtered_set for g in gold_strs)
                if not any_survive:
                    fn_count += 1
                    missed_ids.append(q.get("id", "unknown"))

            # Measure SIR on filtered paths if requested
            if measure_sir and filtered:
                measurer = SIRMeasurer(scorer)

                # Build a simple string iterable instead of MarisaTrie
                class StringTrie:
                    def __init__(self, paths):
                        self.paths = paths

                    def __iter__(self):
                        return iter(self.paths)

                    def __len__(self):
                        return len(self.paths)

                sir_result = measurer.measure_from_trie(
                    StringTrie(filtered), q["question"]
                )
                sir_values.append(sir_result["sir"])

        fnr = fn_count / total if total > 0 else 0.0
        avg_reduction = (
            np.mean(
                [
                    (o - f) / o if o > 0 else 0
                    for o, f in zip(original_sizes_for_tau, filtered_sizes)
                ]
            )
            if original_sizes_for_tau
            else 0.0
        )

        results.append(
            {
                "tau": tau,
                "fnr": fnr,
                "total": total,
                "false_negatives": fn_count,
                "missed_ids": missed_ids,
                "avg_reduction": float(avg_reduction),
                "avg_filtered_size": float(np.mean(filtered_sizes))
                if filtered_sizes
                else 0,
                "avg_original_size": float(np.mean(original_sizes_for_tau))
                if original_sizes_for_tau
                else 0,
                "avg_sir": float(np.mean(sir_values)) if sir_values else None,
            }
        )

        _print_tau_result(results[-1])

    return results


def _dfs_paths(g, q):
    """Get all paths from question entities up to 2 hops."""
    return dfs(g, q["q_entity"], 2)


def _get_gold_path_strs(q):
    """Get ground truth path strings from question_dict."""
    g = build_graph(q["graph"])
    truth = get_truth_paths(q["q_entity"], q["a_entity"], g)
    from gcr.src.utils import path_to_string as pts

    return [pts(p) for p in truth]


def _print_tau_result(r):
    """Print a single tau result row."""
    sir_str = f"SIR={r['avg_sir']:.4f}" if r["avg_sir"] is not None else "SIR=N/A"
    print(
        f"  τ={r['tau']:.2f}  "
        f"FNR={r['fnr']:.3f}  "
        f"reduction={r['avg_reduction']:.1%}  "
        f"filtered={r['avg_filtered_size']:.0f}  "
        f"original={r['avg_original_size']:.0f}  "
        f"{sir_str}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="webqsp",
        choices=["webqsp", "cwq", "test"],
        help="Data source (default: webqsp from HuggingFace; 'test' for synthetic data)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of questions to use (default 100; use -1 for all)",
    )
    parser.add_argument("--tau_min", type=float, default=0.10)
    parser.add_argument("--tau_max", type=float, default=0.60)
    parser.add_argument("--tau_step", type=float, default=0.05)
    parser.add_argument(
        "--output", type=str, default="data/threshold_sweep_results.json"
    )
    parser.add_argument(
        "--mid_cache",
        type=str,
        default="data/mid_to_name.json",
        help="Path to MID-to-name cache (auto-built from dataset)",
    )
    args = parser.parse_args()

    tau_values = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    tau_values = [round(t, 2) for t in tau_values]

    print(f"Loading SemanticScorer (MiniLM)...")
    scorer = SemanticScorer()

    if args.dataset == "test":
        print(f"Using local test data ({len(_get_test_questions())} questions)")

        questions = _get_test_questions()
        tokenizer = None
    else:
        print(f"Loading {args.dataset} from HuggingFace...")
        from datasets import load_dataset
        from transformers import AutoTokenizer

        # Build MID resolver and patch path_to_string BEFORE importing V1TrieBuilder
        resolver = MidResolver(cache_path=args.mid_cache)

        split = f"test[:{args.num}]" if args.num > 0 else "test"
        dataset = load_dataset(f"rmanluo/RoG-{args.dataset}", split=split)
        questions = list(dataset)
        print(f"  Loaded {len(questions)} questions")

        print(
            f"  Building MID-to-name resolver (may be partial without Freebase dump)..."
        )
        resolver.build_from_dataset(questions)
        cov = resolver.coverage(questions)
        print(
            f"  MID coverage: {cov['resolved']}/{cov['total_mids']} "
            f"({cov['coverage_pct']}%)"
        )

        # Patch path_to_string so SemanticScorer sees readable names
        import gcr.src.utils as gcr_utils

        global _ORIG_PATH_TO_STR
        _ORIG_PATH_TO_STR = gcr_utils.path_to_string

        def _resolved_path_to_string(path):
            raw = _ORIG_PATH_TO_STR(path)
            return resolver.resolve_path(raw)

        gcr_utils.path_to_string = _resolved_path_to_string

        tokenizer = AutoTokenizer.from_pretrained(
            "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct",
            trust_remote_code=True,
        )

    print(f"\nSweeping τ from {args.tau_min} to {args.tau_max}...")
    print(
        f"{'τ':<6} {'FNR':<8} {'Reduction':<12} {'Filtered':<10} {'Original':<10} {'SIR':<8}"
    )
    print("-" * 60)

    results = sweep_tau_on_questions(
        questions=questions,
        tokenizer=tokenizer,
        scorer=scorer,
        tau_values=tau_values,
    )

    # Restore original path_to_string
    if _ORIG_PATH_TO_STR is not None:
        import gcr.src.utils as gcr_utils

        gcr_utils.path_to_string = _ORIG_PATH_TO_STR

    # Select best tau: highest tau where FNR < 0.05
    best = None
    for r in results:
        if r["fnr"] < 0.05:
            best = r

    print("\n" + "=" * 60)
    if best:
        print(f"Selected τ = {best['tau']:.2f} (highest with FNR < 5%)")
        print(f"  FNR: {best['fnr']:.3f} ({best['false_negatives']}/{best['total']})")
        print(f"  Trie reduction: {best['avg_reduction']:.1%}")
        print(
            f"  Avg filtered size: {best['avg_filtered_size']:.0f} (from {best['avg_original_size']:.0f})"
        )
        if best["avg_sir"] is not None:
            print(f"  Avg SIR: {best['avg_sir']:.4f}")
    else:
        print("No τ with FNR < 5% found!")

    # Save results
    os.makedirs("data", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "num_questions": len(questions),
                "tau_values": tau_values,
                "results": results,
                "selected_tau": best["tau"] if best else None,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
