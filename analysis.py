"""
analysis.py

Loads results/sweep_results.json and reproduces the paper's statistical
analysis:

  - per-seed effect sizes Delta = Z53_patched - X_patched (X in {Q53, C53})
    for both the embedding-PC1 and residual-PC1 protocols
  - paired bootstrap CI on the mean of Delta across seeds
  - Wilcoxon signed-rank test and sign test on paired Delta values
  - dimensionality comparison (participation ratio) across all three
    operations, matched by seed

This is the script that answers the central question raised in Phase 1:
does the brittleness effect (Q53 more disrupted than Z53) survive when
tested against C53, which is commutative like Z53 and differs only in
associativity?

Usage:
    python analysis.py --in results/sweep_results.json
"""

import argparse
import json

import numpy as np
from scipy import stats


def load_records(path):
    with open(path) as f:
        return json.load(f)


def paired_effect_sizes(records, op_a: str, op_b: str, protocol: str, n_heads: int = 1):
    """
    Returns arrays of matched-seed effect sizes Delta = op_a_patched -
    op_b_patched for the given protocol ('embedding_patch' or
    'residual_patch'), only using seeds present for both operations.
    """
    by_seed_a = {r["seed"]: r for r in records
                 if r["operation"] == op_a and r["n_heads"] == n_heads}
    by_seed_b = {r["seed"]: r for r in records
                 if r["operation"] == op_b and r["n_heads"] == n_heads}
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))

    deltas = []
    for seed in common_seeds:
        val_a = by_seed_a[seed][protocol]["mean"]
        val_b = by_seed_b[seed][protocol]["mean"]
        deltas.append(val_a - val_b)
    return common_seeds, np.array(deltas)


def bootstrap_ci_on_mean(values: np.ndarray, n_resamples: int = 10_000,
                          ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def evaluate_comparison(records, op_a: str, op_b: str, protocol: str, n_heads: int = 1):
    seeds, deltas = paired_effect_sizes(records, op_a, op_b, protocol, n_heads)
    if len(deltas) == 0:
        print(f"  No matched seeds found for {op_a} vs {op_b} ({protocol}, "
              f"heads={n_heads}).")
        return None

    mean, lo, hi = bootstrap_ci_on_mean(deltas)
    n_positive = int((deltas > 0).sum())
    n_total = len(deltas)

    # Wilcoxon signed-rank test (paired, two-sided)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(deltas)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    # Sign test (binomial test against 0.5)
    sign_result = stats.binomtest(n_positive, n_total, p=0.5)
    sign_p = sign_result.pvalue

    print(f"\n  {op_a} vs {op_b}  [{protocol}, heads={n_heads}]  "
          f"(n={n_total} matched seeds)")
    print(f"    seeds: {seeds}")
    print(f"    per-seed deltas: {np.round(deltas, 4).tolist()}")
    print(f"    mean delta = {mean:+.4f}   95% bootstrap CI = "
          f"[{lo:+.4f}, {hi:+.4f}]"
          f"{'  (excludes zero)' if lo > 0 or hi < 0 else '  (STRADDLES ZERO)'}")
    print(f"    {n_positive}/{n_total} seeds positive "
          f"(op_a more disrupted)")
    print(f"    Wilcoxon signed-rank p = {wilcoxon_p:.4f}")
    print(f"    sign test p = {sign_p:.4f}")

    return {
        "op_a": op_a, "op_b": op_b, "protocol": protocol, "n_heads": n_heads,
        "seeds": seeds, "deltas": deltas.tolist(),
        "mean": mean, "ci_lo": lo, "ci_hi": hi,
        "n_positive": n_positive, "n_total": n_total,
        "wilcoxon_p": float(wilcoxon_p), "sign_p": float(sign_p),
    }


def dimensionality_comparison(records, n_heads: int = 1):
    """Prints participation ratio and effective rank, grouped by operation."""
    print("\n  Participation ratio / effective rank by operation "
          f"(heads={n_heads}):")
    for op_name in ["Z53", "Q53", "C53"]:
        vals = [r["participation_ratio"] for r in records
                if r["operation"] == op_name and r["n_heads"] == n_heads]
        eff = [r["effective_rank"] for r in records
               if r["operation"] == op_name and r["n_heads"] == n_heads]
        if not vals:
            continue
        print(f"    {op_name}: PR = {np.mean(vals):.2f} +/- {np.std(vals):.2f}  "
              f"(n={len(vals)}),  eff.rank = {np.mean(eff):.2f} +/- {np.std(eff):.2f}")


def catapult_summary(records):
    print("\n  Catapult occurrences:")
    for r in records:
        if r["n_catapults"] > 0:
            print(f"    {r['operation']} seed={r['seed']} heads={r['n_heads']}: "
                  f"{r['n_catapults']} catapult(s) at steps "
                  f"{[c[0] for c in r['catapults']]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str,
                         default="results/sweep_results.json")
    args = parser.parse_args()

    records = load_records(args.in_path)
    heads_present = sorted(set(r["n_heads"] for r in records))

    print("=" * 70)
    print("EMBEDDING-PC1 PROTOCOL (original, illusion-prone proxy)")
    print("=" * 70)
    for heads in heads_present:
        evaluate_comparison(records, "Z53", "Q53", "embedding_patch", heads)
        evaluate_comparison(records, "Z53", "C53", "embedding_patch", heads)

    print("\n" + "=" * 70)
    print("RESIDUAL-STREAM-PC1 PROTOCOL (corrected proxy)")
    print("=" * 70)
    for heads in heads_present:
        evaluate_comparison(records, "Z53", "Q53", "residual_patch", heads)
        evaluate_comparison(records, "Z53", "C53", "residual_patch", heads)

    print("\n" + "=" * 70)
    print("DIMENSIONALITY (Priority #2: distributed vs. concentrated)")
    print("=" * 70)
    for heads in heads_present:
        dimensionality_comparison(records, heads)

    print("\n" + "=" * 70)
    print("CATAPULTS")
    print("=" * 70)
    catapult_summary(records)


if __name__ == "__main__":
    main()
