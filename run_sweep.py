"""
run_sweep.py

Orchestrates the full confirmatory sweep:

    3 operations (Z53, Q53, C53) x N_SEEDS seeds
    -> for each trained model: embedding-PC1 patch test, residual-PC1
       patch test, explained-variance-gap check, cosine alignment,
       and full dimensionality summary (participation ratio, effective
       rank, scree spectrum).

Also supports an optional head-count ablation (Priority #4): pass
--heads 1 4 to additionally train/test 4-head variants of every
(operation, seed) pair.

Usage:
    python run_sweep.py                      # default: 10 seeds, 1 head
    python run_sweep.py --n_seeds 5           # quick run matching the
                                               # original paper's seed count
    python run_sweep.py --heads 1 4           # add the head-count ablation
    python run_sweep.py --steps 2000 --n_seeds 2   # fast smoke test

Results are written incrementally to results/sweep_results.json so a
run can be resumed/inspected even if interrupted partway through.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ops import OPERATIONS, C53_DEFAULT_K
from train import train_one_model
from perturbation import (
    embedding_pc1_direction,
    residual_pc1_direction,
    evaluate_patch_test,
    bootstrap_ci,
)
from diagnostics import (
    explained_variance_gap,
    cosine_alignment,
    dimensionality_summary,
)

DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044, 5055, 6066]  # 10 seeds
N_PATCH_TRIALS = 200
N_BOOTSTRAP = 10_000
N_RESID_SAMPLES = 2000


def run_single_model(op_name: str, op, seed: int, n_heads: int, total_steps: int):
    """Trains one model and runs the complete diagnostic/perturbation battery."""
    t0 = time.perf_counter()
    train_result = train_one_model(op, seed=seed, n_heads=n_heads,
                                    total_steps=total_steps)
    model = train_result["model"]

    # --- directions ---
    emb_dir = embedding_pc1_direction(model)
    res_dir, resid_acts = residual_pc1_direction(model, n_samples=N_RESID_SAMPLES,
                                                  seed=0)

    # --- patch tests (both directions) ---
    emb_mean, emb_probs = evaluate_patch_test(
        model, op, emb_dir, n_trials=N_PATCH_TRIALS, seed=0
    )
    res_mean, res_probs = evaluate_patch_test(
        model, op, res_dir, n_trials=N_PATCH_TRIALS, seed=0
    )
    emb_ci = bootstrap_ci(emb_probs, n_resamples=N_BOOTSTRAP, seed=0)
    res_ci = bootstrap_ci(res_probs, n_resamples=N_BOOTSTRAP, seed=0)

    # --- diagnostics ---
    W_E = model.W_E.detach().cpu().numpy()
    var_gap = explained_variance_gap(W_E)
    alignment = cosine_alignment(emb_dir, res_dir)
    dim_summary = dimensionality_summary(model, n_samples=N_RESID_SAMPLES, seed=0)

    elapsed = time.perf_counter() - t0

    record = {
        "operation": op_name,
        "seed": seed,
        "n_heads": n_heads,
        "final_test_acc": train_result["final_test_acc"],
        "n_catapults": len(train_result["catapults"]),
        "catapults": train_result["catapults"],
        "train_wall_time_s": train_result["wall_time_s"],
        "total_wall_time_s": elapsed,
        "embedding_patch": {
            "mean": emb_ci[0], "ci_lo": emb_ci[1], "ci_hi": emb_ci[2],
        },
        "residual_patch": {
            "mean": res_ci[0], "ci_lo": res_ci[1], "ci_hi": res_ci[2],
        },
        "embedding_variance_gap": var_gap["gap"],
        "embedding_pc1_ratio": var_gap["pc1_ratio"],
        "embedding_residual_alignment": alignment,
        "participation_ratio": dim_summary["participation_ratio"],
        "effective_rank": dim_summary["effective_rank"],
        "residual_pc1_ratio": dim_summary["pc1_ratio"],
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10,
                         help="Number of seeds to use (from DEFAULT_SEEDS, "
                              "in order). Default 10, per Phase-1 Priority #3.")
    parser.add_argument("--heads", type=int, nargs="+", default=[1],
                         help="Head counts to sweep. Default [1] matches the "
                              "original confirmatory sweep; pass '1 4' to add "
                              "the Priority-#4 architectural ablation.")
    parser.add_argument("--steps", type=int, default=50_000,
                         help="Training steps per model. Use a small value "
                              "(e.g. 2000) for a fast smoke test.")
    parser.add_argument("--out", type=str, default="results/sweep_results.json")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS[: args.n_seeds]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Sweep config: operations={list(OPERATIONS.keys())}, "
          f"seeds={seeds}, heads={args.heads}, steps={args.steps}")
    print(f"C53 uses k={C53_DEFAULT_K}\n")

    results = []
    n_total = len(OPERATIONS) * len(seeds) * len(args.heads)
    i = 0
    t_start = time.perf_counter()

    for n_heads in args.heads:
        for op_name, op in OPERATIONS.items():
            for seed in seeds:
                i += 1
                print(f"[{i}/{n_total}] training {op_name} seed={seed} "
                      f"heads={n_heads} ...")
                record = run_single_model(op_name, op, seed, n_heads, args.steps)
                results.append(record)

                # write incrementally so partial progress is never lost
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)

    total_time = time.perf_counter() - t_start
    print(f"\nSweep complete: {len(results)} models in {total_time / 60:.1f} min.")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
