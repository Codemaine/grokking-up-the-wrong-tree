"""
figures.py

Regenerates the paper's figures from results/sweep_results.json, plus
two new figures for the Phase-1 extensions:

    fig6_c53_effect_sizes.png   -- Z53 vs C53 effect sizes (mirrors fig5,
                                    but for the unconfounded comparison)
    fig7_participation_ratio.png -- participation ratio by operation,
                                    testing the distributed-vs-concentrated
                                    hypothesis directly (Priority #2)

Usage:
    python figures.py --in results/sweep_results.json --out figures/
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_records(path):
    with open(path) as f:
        return json.load(f)


def fig_effect_sizes(records, op_a, op_b, protocol_pairs, out_path, n_heads=1):
    """
    protocol_pairs: list of (label, json_key) e.g.
        [("Original (embedding PC1)", "embedding_patch"),
         ("Improved (residual PC1)", "residual_patch")]
    Mirrors the paper's Figure 5 style: one panel per protocol.
    """
    by_seed_a = {r["seed"]: r for r in records
                 if r["operation"] == op_a and r["n_heads"] == n_heads}
    by_seed_b = {r["seed"]: r for r in records
                 if r["operation"] == op_b and r["n_heads"] == n_heads}
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
    if not common_seeds:
        print(f"  [skip] no matched seeds for {op_a} vs {op_b}")
        return

    fig, axes = plt.subplots(1, len(protocol_pairs), figsize=(6 * len(protocol_pairs), 4),
                              sharey=True)
    if len(protocol_pairs) == 1:
        axes = [axes]

    for ax, (label, key) in zip(axes, protocol_pairs):
        deltas = []
        for seed in common_seeds:
            deltas.append(by_seed_a[seed][key]["mean"] - by_seed_b[seed][key]["mean"])
        colors = ["tab:red" if d < 0 else "tab:green" for d in deltas]
        ax.barh([f"Seed {s}" for s in common_seeds], deltas, color=colors)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel(f"Delta = P(correct | {op_a}) - P(correct | {op_b})")

    fig.suptitle(f"Effect size: {op_a} vs {op_b}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_participation_ratio(records, out_path, n_heads=1):
    """Bar chart of participation ratio by operation, matched by seed."""
    ops = ["Z53", "Q53", "C53"]
    by_op = {op: [r for r in records if r["operation"] == op and r["n_heads"] == n_heads]
              for op in ops}
    if not any(by_op.values()):
        print("  [skip] no records for participation ratio figure")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    means = [np.mean([r["participation_ratio"] for r in by_op[op]]) if by_op[op] else 0
             for op in ops]
    stds = [np.std([r["participation_ratio"] for r in by_op[op]]) if by_op[op] else 0
            for op in ops]
    colors = ["tab:blue", "tab:red", "tab:green"]
    ax.bar(ops, means, yerr=stds, color=colors, capsize=5)
    ax.set_ylabel("Participation ratio (residual-stream PCA)")
    ax.set_title("Distributed vs. concentrated representation\n"
                  "(higher = more distributed across directions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_scree_comparison(model_summaries, out_path):
    """
    model_summaries: dict op_name -> list of spectrum arrays (one per seed)
    Plots mean scree curve per operation on one axis, matching the
    'concentration hypothesis predicts a steeper drop-off' framing from
    the paper's Future Directions section.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"Z53": "tab:blue", "Q53": "tab:red", "C53": "tab:green"}
    for op_name, spectra in model_summaries.items():
        if not spectra:
            continue
        arr = np.array(spectra)  # [n_seeds, d_model]
        mean_spectrum = arr.mean(axis=0)
        ax.plot(np.arange(1, len(mean_spectrum) + 1)[:20], mean_spectrum[:20],
                 label=op_name, color=colors.get(op_name), marker="o", markersize=3)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Residual-stream PCA scree plot (top 20 components)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str,
                         default="results/sweep_results.json")
    parser.add_argument("--out", dest="out_dir", type=str, default="figures/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.in_path)
    heads_present = sorted(set(r["n_heads"] for r in records))

    for heads in heads_present:
        suffix = f"_heads{heads}" if len(heads_present) > 1 else ""

        fig_effect_sizes(
            records, "Z53", "Q53",
            [("Original (embedding PC1)", "embedding_patch"),
             ("Improved (residual PC1)", "residual_patch")],
            os.path.join(args.out_dir, f"fig5_effect_sizes_Z53_Q53{suffix}.png"),
            n_heads=heads,
        )
        fig_effect_sizes(
            records, "Z53", "C53",
            [("Embedding PC1", "embedding_patch"),
             ("Residual PC1", "residual_patch")],
            os.path.join(args.out_dir, f"fig6_effect_sizes_Z53_C53{suffix}.png"),
            n_heads=heads,
        )
        fig_participation_ratio(
            records,
            os.path.join(args.out_dir, f"fig7_participation_ratio{suffix}.png"),
            n_heads=heads,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
