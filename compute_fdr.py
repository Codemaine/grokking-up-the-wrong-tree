#!/usr/bin/env python3
import json
import os
import numpy as np

OUTPUT_DIR = "paper_figures"

def bh_correction(p_values):
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    adjusted_p_values = np.zeros(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        adjusted_p = sorted_p_values[i] * n / (i + 1)
        running_min = min(running_min, adjusted_p)
        adjusted_p_values[sorted_indices[i]] = running_min
    return np.clip(adjusted_p_values, 0, 1)

if __name__ == "__main__":
    h1_path = os.path.join(OUTPUT_DIR, "aggregate_summary_h1.json")
    h4_path = os.path.join(OUTPUT_DIR, "aggregate_summary_h4.json")
    
    if not os.path.exists(h1_path) or not os.path.exists(h4_path):
        print("Both h1 and h4 JSON summaries must exist to compute full FDR.")
        exit(1)
        
    with open(h1_path) as f:
        h1 = json.load(f)
    with open(h4_path) as f:
        h4 = json.load(f)
        
    # The 12 primary hypotheses:
    family = [
        # 1-head
        ("h1_pr_c53", h1["participation_ratio"]["stats_c53"]["p_value"]),
        ("h1_pr_l53", h1["participation_ratio"]["stats_l53"]["p_value"]),
        ("h1_ent_c53", h1["activation_entropy"]["stats_c53"]["p_value"]),
        ("h1_ent_l53", h1["activation_entropy"]["stats_l53"]["p_value"]),
        ("h1_patch_c53", h1["subspace_illusion"]["stats_c53"]["p_value"]),
        ("h1_patch_l53", h1["subspace_illusion"]["stats_l53"]["p_value"]),
        # 4-head
        ("h4_pr_c53", h4["participation_ratio"]["stats_c53"]["p_value"]),
        ("h4_pr_l53", h4["participation_ratio"]["stats_l53"]["p_value"]),
        ("h4_ent_c53", h4["activation_entropy"]["stats_c53"]["p_value"]),
        ("h4_ent_l53", h4["activation_entropy"]["stats_l53"]["p_value"]),
        ("h4_patch_c53", h4["subspace_illusion"]["stats_c53"]["p_value"]),
        ("h4_patch_l53", h4["subspace_illusion"]["stats_l53"]["p_value"]),
    ]
    
    names = [x[0] for x in family]
    raw_p_values = [x[1] for x in family]
    
    adj_p_values = bh_correction(raw_p_values)
    
    print("=== Benjamini-Hochberg FDR Correction ===")
    print(f"Family Size: {len(family)} tests\n")
    for name, raw_p, adj_p in zip(names, raw_p_values, adj_p_values):
        sig = "*" if adj_p < 0.05 else "ns"
        print(f"{name:15s}: raw_p = {raw_p:.4f} -> fdr_p = {adj_p:.4f} ({sig})")
    
    # Save the corrected p-values
    out_dict = {k: float(v) for k, v in zip(names, adj_p_values)}
    with open(os.path.join(OUTPUT_DIR, "fdr_corrected_pvalues.json"), 'w') as f:
        json.dump(out_dict, f, indent=2)
        
    print(f"\nSaved FDR-corrected p-values to {OUTPUT_DIR}/fdr_corrected_pvalues.json")
