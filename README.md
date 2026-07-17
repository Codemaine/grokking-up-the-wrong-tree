# Grokking Up the Wrong Tree

This repository contains the complete reproducible codebase for the paper: **"Grokking Up the Wrong Tree: Interpretability Illusions in Algebraic Transformers"**.

It investigates whether non-associative structures intrinsically require "brittle" representations, or whether apparent geometric brittleness is an artifact of specific arithmetic structure, exploratory statistics, and unaligned interpretability proxies.

## Setup

```bash
pip install -r requirements.txt
```

*Note: `transformer-lens==3.5.1` and `torch==2.13.0` match the versions reported in the paper. Running on CPU is recommended to ensure fully deterministic reproduction of the paper's results.*

## Core Scripts

- `ops.py` - Defines the order-53 operations used in the study ($Z_{53}$, $Q_{53}$, $C_{53}$, $L_{53}$, $L_{53}^{\text{assoc}}$). Run directly to exhaustively verify their properties (e.g. associativity, commutativity, Latin-square).
- `run_pilot_study.py` - Replicates the initial pilot study ($Z_{53}$ vs $Q_{53}$) across 5 random seeds. Catches the "silent proxy failure" due to late-training catapult dynamics.
- `run_sweep.py` - Orchestrates the full confirmatory experiment sweep. Trains all operations across multiple seeds, calculating both the 1D PCA patching tests and dimensionality measures (Participation Ratio, Effective Rank).
- `analysis.py` - Reproduces the paper's main statistical comparisons (e.g. $Z_{53}$ vs $C_{53}$ and $Z_{53}$ vs $L_{53}$) using the results from `run_sweep.py`.
- `compute_fdr.py` - Applies the Benjamini-Hochberg False Discovery Rate (FDR) correction to the $p$-values.

## Replication Guide

### 1. The Pilot Study (Proxy Misalignment)
To reproduce the finding from Section 4.2 that the `embedding-PC1` proxy can silently fail due to catapult dynamics, run:
```bash
python run_pilot_study.py --seeds 5
```
This trains 5 seeds of $Z_{53}$ and $Q_{53}$ and evaluates them using the 1D PCA patching test on the token embedding matrix.

### 2. The Confirmatory Sweep (Dimensionality)
To run the full multi-seed suite comparing all operations (including the unstructured $L_{53}$ control):
```bash
python run_sweep.py --n_seeds 50 --out results/sweep_results_h1.json
```
For the 4-head architectural ablation:
```bash
python run_sweep.py --n_seeds 50 --heads 4 --out results/sweep_results_h4.json
```

### 3. Analysis & FDR Correction
To extract the statistical significance and dimensionality gaps:
```bash
python analysis.py --in results/sweep_results_h1.json
```
*(Repeat for the 4-head results)*.

Finally, to apply the rigorous multiple-testing correction that collapses the "dimensionality gap" artifact (Section 4.1):
```bash
python compute_fdr.py
```
*(Make sure to update `compute_fdr.py` paths to point to your generated summary files).*
