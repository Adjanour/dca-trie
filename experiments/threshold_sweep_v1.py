"""
Threshold Sweep for DCA-Trie v1.

Sweeps tau in [0.1, 0.6] step 0.05 and records:
  - FNR (False Negative Rate): fraction of questions where gold paths are pruned
  - Trie reduction %: (original_size - filtered_size) / original_size
  - SIR of the filtered trie
  - SIR of the original (unfiltered) trie for comparison
  - Hits@1 of the filtered trie (if ground truth answers available)

Usage:
    python experiments/threshold_sweep_v1.py                           # WebQSP (default, 100 questions)
    python experiments/threshold_sweep_v1.py --dataset cwq             # CWQ
    python experiments/threshold_sweep_v1.py --dataset test            # synthetic test data
    python experiments/threshold_sweep_v1.py --num 500                 # more questions
    python experiments/threshold_sweep_v1.py --tau_max 0.50 --tau_step 0.02
"""

import argparse
import json
import sys
import os
import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dca_trie.semantic_scorer import SemanticScorer
from dca_trie.sir_measurement import SIRMeasurer
from dca_trie.mid_resolver import MidResolver
from dca_trie.v1_trie_builder import V1TrieBuilder
from gcr.src.utils.graph_utils import build_graph, dfs, get_truth_paths
from gcr.src.utils import path_to_string as _gcr_path_to_string


def _get_test_questions():
    from dca_trie.test_mini_freebase import get_all_test_questions
    return get_all_test_questions()


ID_PATTERN = re.compile(r"^m\.\d+")


class _StringTrie:
    """Minimal wrapper so SIRMeasurer can iterate over path strings."""
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)


def sweep_tau_on_questions(
    questions,
    tokenizer,
    scorer,
    tau_values=None,
    path_to_str_fn=None,
    all_resolved_gold=None,
):
    if tau_values is None:
        tau_values = np.arange(0.10, 0.65, 0.05)

    results = []
    original_sizes = []
    all_original_strs = []

    for q in questions:
        g = build_graph(q["graph"])
        paths = _dfs_paths(g, q)
        original_sizes.append(len(paths))

        path_strs = [_gcr_path_to_string(p) for p in paths]
        all_original_strs.append(path_strs)

    # Measure baseline SIR on unfiltered trie (resolved for meaningful scoring)
    print("Measuring baseline SIR on unfiltered tries...")
    baseline_sirs = []
    for idx, q in enumerate(questions):
        raw_strs = all_original_strs[idx]
        resolved_strs = [path_to_str_fn(s) for s in raw_strs] if path_to_str_fn else raw_strs
        if resolved_strs:
            sir = SIRMeasurer(scorer).measure_from_trie(
                _StringTrie(resolved_strs), q["question"]
            )
            baseline_sirs.append(sir["sir"])
    avg_baseline_sir = float(np.mean(baseline_sirs)) if baseline_sirs else None

    for tau in tau_values:
        tau = round(tau, 2)
        total = 0
        fn_count = 0
        missed_ids = []
        filtered_sizes = []
        filtered_sirs = []

        builder = V1TrieBuilder(
            tokenizer=tokenizer,
            scorer=scorer,
            tau=tau,
            path_to_str_fn=path_to_str_fn,
        )

        for idx, q in enumerate(questions):
            total += 1
            filtered = builder.filter_paths_only(q)
            filtered_set = set(filtered)
            filtered_sizes.append(len(filtered))

            resolved_gold = all_resolved_gold[idx] if all_resolved_gold else set()
            if resolved_gold:
                any_survive = any(g in filtered_set for g in resolved_gold)
                if not any_survive:
                    fn_count += 1
                    missed_ids.append(q.get("id", "unknown"))

            if filtered:
                sir_result = SIRMeasurer(scorer).measure_from_trie(
                    _StringTrie(filtered), q["question"]
                )
                filtered_sirs.append(sir_result["sir"])

        fnr = fn_count / total if total > 0 else 0.0
        avg_reduction = (
            np.mean([
                (o - f) / o if o > 0 else 0
                for o, f in zip(original_sizes, filtered_sizes)
            ])
            if original_sizes else 0.0
        )

        results.append({
            "tau": tau,
            "fnr": fnr,
            "total": total,
            "false_negatives": fn_count,
            "missed_ids": missed_ids,
            "avg_reduction": float(avg_reduction),
            "avg_filtered_size": float(np.mean(filtered_sizes)) if filtered_sizes else 0,
            "avg_original_size": float(np.mean(original_sizes)) if original_sizes else 0,
            "avg_sir": float(np.mean(filtered_sirs)) if filtered_sirs else None,
            "baseline_sir": avg_baseline_sir,
        })

        _print_tau_result(results[-1])

    return results


def _dfs_paths(g, q):
    return dfs(g, q["q_entity"], 2)


def _get_gold_path_strs(q):
    g = build_graph(q["graph"])
    truth = get_truth_paths(q["q_entity"], q["a_entity"], g)
    from gcr.src.utils import path_to_string as pts
    return [pts(p) for p in truth]


def _print_tau_result(r):
    sir_str = f"SIR={r['avg_sir']:.4f}" if r["avg_sir"] is not None else "SIR=N/A"
    baseline_str = f"  baseline SIR={r['baseline_sir']:.4f}" if r["baseline_sir"] is not None else ""
    print(
        f"  tau={r['tau']:.2f}  "
        f"FNR={r['fnr']:.3f}  "
        f"reduction={r['avg_reduction']:.1%}  "
        f"filtered={r['avg_filtered_size']:.0f}  "
        f"original={r['avg_original_size']:.0f}  "
        f"{sir_str}{baseline_str}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="webqsp",
        choices=["webqsp", "cwq", "test"],
        help="Data source (default: webqsp from HuggingFace; 'test' for synthetic data)",
    )
    parser.add_argument(
        "--num", type=int, default=100,
        help="Number of questions to use (default 100; use -1 for all)",
    )
    parser.add_argument("--tau_min", type=float, default=0.10)
    parser.add_argument("--tau_max", type=float, default=0.60)
    parser.add_argument("--tau_step", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="data/threshold_sweep_results.json")
    parser.add_argument(
        "--mid_cache", type=str, default="data/mid_to_name.json",
        help="Path to MID-to-name cache (auto-built from dataset + Freebase names)",
    )
    parser.add_argument(
        "--download_fb_names", action="store_true",
        help="Download Freebase entity names file (~248MB) for ~99% MID coverage",
    )
    args = parser.parse_args()

    tau_values = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    tau_values = [round(t, 2) for t in tau_values]

    print("Loading SemanticScorer (MiniLM)...")
    scorer = SemanticScorer()

    if args.dataset == "test":
        print(f"Using local test data ({len(_get_test_questions())} questions)")
        questions = _get_test_questions()
        tokenizer = None
        path_to_str_fn = None
    else:
        print(f"Loading {args.dataset} from HuggingFace...")
        from datasets import load_dataset
        from transformers import AutoTokenizer

        split = f"test[:{args.num}]" if args.num > 0 else "test"
        dataset = load_dataset(f"rmanluo/RoG-{args.dataset}", split=split)
        questions = list(dataset)
        print(f"  Loaded {len(questions)} questions")

        resolver = MidResolver(cache_path=args.mid_cache)
        resolver.build_from_dataset(questions)

        if args.download_fb_names:
            resolver.download_and_build_fb_names(target_path="data/fb_entity_names.txt.gz")

        cov = resolver.coverage(questions)
        print(f"  MID coverage: {cov['resolved']}/{cov['total_mids']} "
              f"({cov['coverage_pct']}%)")

        def _resolved_path_to_str(p):
            """If tuple, convert to string then resolve MIDs. If already string, just resolve."""
            raw = _gcr_path_to_string(p) if not isinstance(p, str) else p
            return resolver.resolve_path(raw)

        path_to_str_fn = _resolved_path_to_str
        tokenizer = AutoTokenizer.from_pretrained(
            "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct",
            trust_remote_code=True,
        )

    print(f"\nSweeping tau from {args.tau_min} to {args.tau_max}...")
    print(f"{'tau':<6} {'FNR':<8} {'Reduction':<12} {'Filtered':<10} {'Original':<10} {'SIR':<8}")
    print("-" * 60)

    # Pre-compute resolved gold paths for FNR comparison
    all_resolved_gold = None
    if resolver:
        all_resolved_gold = [
            {resolver.resolve_path(g) for g in _get_gold_path_strs(q)}
            for q in questions
        ]

    results = sweep_tau_on_questions(
        questions=questions,
        tokenizer=tokenizer,
        scorer=scorer,
        tau_values=tau_values,
        path_to_str_fn=path_to_str_fn,
        all_resolved_gold=all_resolved_gold,
    )

    best = None
    for r in results:
        if r["fnr"] < 0.05:
            best = r

    print("\n" + "=" * 60)
    if best:
        print(f"Selected tau = {best['tau']:.2f} (highest with FNR < 5%)")
        print(f"  FNR: {best['fnr']:.3f} ({best['false_negatives']}/{best['total']})")
        print(f"  Trie reduction: {best['avg_reduction']:.1%}")
        print(f"  Avg filtered size: {best['avg_filtered_size']:.0f} (from {best['avg_original_size']:.0f})")
        if best["avg_sir"] is not None:
            print(f"  Avg SIR (filtered): {best['avg_sir']:.4f}")
        if best["baseline_sir"] is not None:
            print(f"  Avg SIR (baseline): {best['baseline_sir']:.4f}")
    else:
        print("No tau with FNR < 5% found!")

    os.makedirs("data", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "num_questions": len(questions),
            "tau_values": tau_values,
            "results": results,
            "selected_tau": best["tau"] if best else None,
            "baseline_sir": best["baseline_sir"] if best else None,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
