"""
Multi-seed replication: Z53 vs Q53, same config as the original run,
grokking check, embedding-PC single-direction test, per seed.

Resumable: safe to kill and rerun, skips completed (table, seed) runs
and finished analyses.

Usage:
    python multiseed_experiment.py --seeds 5
    python multiseed_experiment.py --seeds 5 --report_only   # rebuild summary from what's done
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utilities import get_act_name

RESULTS_DIR = Path("multiseed_results")
CKPT_DIR = RESULTS_DIR / "checkpoints"
N = 53
STEPS = 50000
LR, WD = 1e-3, 1.0
FRAC_TRAIN = 0.3
BATCH_SIZE = 512
LOG_EVERY = 500
CKPT_EVERY = 2000
N_TEST_SAMPLES = 1000
N_BOOTSTRAPS = 1000
HOOK_NAME = "blocks.0.hook_resid_mid"
SEQ_POS = 1


def make_Z53():
    return torch.tensor([[(a + b) % N for b in range(N)] for a in range(N)])


def make_Q53():
    sigma = list(range(N))
    sigma[-2], sigma[-1] = sigma[-1], sigma[-2]
    return torch.tensor([[(a + sigma[b]) % N for b in range(N)] for a in range(N)])


def is_assoc(table):
    n = table.shape[0]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if table[table[a, b], c] != table[a, table[b, c]]:
                    return False
    return True


def make_dataset(table, seed):
    pairs = torch.tensor([[i, j] for i in range(N) for j in range(N)])
    labels = torch.tensor([table[i, j] for i in range(N) for j in range(N)])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g)
    n_train = int(len(pairs) * FRAC_TRAIN)
    return (pairs[perm[:n_train]], labels[perm[:n_train]]), (pairs[perm[n_train:]], labels[perm[n_train:]])


def make_model(seed):
    cfg = HookedTransformerConfig(n_layers=1, n_heads=1, d_model=128, d_head=64, d_mlp=512,
        act_fn="relu", normalization_type=None, d_vocab=N, d_vocab_out=N, n_ctx=2, seed=seed)
    return HookedTransformer(cfg)


def train_one(tag, table, seed):
    train_data, test_data = make_dataset(table, seed)
    model = make_model(seed)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    start_step = 0
    cp = CKPT_DIR / f"{tag}.pt"
    if cp.exists():
        state = torch.load(cp, map_location="cpu")
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        start_step = state["step"]
        if start_step >= STEPS:
            print(f"[{tag}] already complete ({start_step}/{STEPS}), skipping")
            return model
        print(f"[{tag}] resuming from {start_step}/{STEPS}")
    else:
        print(f"[{tag}] starting fresh, target {STEPS} steps")

    train_pairs, train_labels = train_data
    test_pairs, test_labels = test_data
    n_train = len(train_pairs)
    log_file = RESULTS_DIR / f"{tag}_log.csv"
    write_header = not log_file.exists()
    f = open(log_file, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["step", "train_loss", "test_acc"])

    step = start_step
    try:
        for step in range(start_step + 1, STEPS + 1):
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            x, y = train_pairs[idx], train_labels[idx]
            model.train()
            logits = model(x)[:, -1, :]
            loss = nn.CrossEntropyLoss()(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % LOG_EVERY == 0 or step == STEPS:
                model.eval()
                with torch.no_grad():
                    test_logits = model(test_pairs)[:, -1, :]
                    test_acc = (test_logits.argmax(-1) == test_labels).float().mean().item()
                writer.writerow([step, loss.item(), test_acc]); f.flush()
            if step % CKPT_EVERY == 0 or step == STEPS:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, cp)
    except KeyboardInterrupt:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, cp)
        f.close()
        print(f"[{tag}] interrupted at {step}, checkpoint saved")
        raise
    f.close()
    print(f"[{tag}] training done")
    return model


def grokking_check(tag):
    log_file = RESULTS_DIR / f"{tag}_log.csv"
    steps, accs = [], []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"])); accs.append(float(row["test_acc"]))
    return {"max_acc": max(accs), "final_acc": accs[-1], "step_at_max": steps[int(np.argmax(accs))]}


def embedding_pc_test(model, table, seed):
    rng = np.random.default_rng(seed + 99999)
    clean_inputs = torch.tensor(rng.integers(0, N, size=(N_TEST_SAMPLES, 2)))
    corrupt_inputs = torch.tensor(rng.integers(0, N, size=(N_TEST_SAMPLES, 2)))
    collision = torch.all(clean_inputs == corrupt_inputs, dim=-1)
    while collision.any():
        corrupt_inputs[collision] = torch.tensor(rng.integers(0, N, size=(int(collision.sum()), 2)))
        collision = torch.all(clean_inputs == corrupt_inputs, dim=-1)
    targets = table[clean_inputs[:, 0], clean_inputs[:, 1]]

    model.eval()
    with torch.no_grad():
        _, cc = model.run_with_cache(clean_inputs, names_filter=HOOK_NAME)
        clean_resid = cc[HOOK_NAME][:, SEQ_POS, :]
        _, cr = model.run_with_cache(corrupt_inputs, names_filter=HOOK_NAME)
        corrupt_resid = cr[HOOK_NAME][:, SEQ_POS, :]
    diffs = corrupt_resid - clean_resid

    W_E = model.embed.W_E.detach().cpu().numpy()
    pca_we = PCA(n_components=1); pca_we.fit(W_E)
    v1 = torch.tensor(pca_we.components_[0], dtype=torch.float32)
    proj_scalar = torch.sum(diffs * v1, dim=-1, keepdim=True)
    diffs_projected = proj_scalar * v1

    def hook(resid, hook, diffs_projected=diffs_projected):
        resid[:, SEQ_POS, :] = resid[:, SEQ_POS, :] + diffs_projected
        return resid

    with torch.no_grad():
        patched_logits = model.run_with_hooks(clean_inputs, fwd_hooks=[(HOOK_NAME, hook)])
        probs = torch.softmax(patched_logits[:, SEQ_POS, :], dim=-1)
        correct_probs = probs[torch.arange(N_TEST_SAMPLES), targets].numpy()

    boot_means = [np.mean(np.random.choice(correct_probs, size=len(correct_probs), replace=True))
                  for _ in range(N_BOOTSTRAPS)]
    return {
        "mean": float(correct_probs.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "raw": correct_probs.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(exist_ok=True)

    Z53, Q53 = make_Z53(), make_Q53()
    assert is_assoc(Z53) and not is_assoc(Q53), "table sanity check failed"

    summary_path = RESULTS_DIR / "multiseed_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if not args.report_only:
        for seed in range(args.seeds):
            for name, table in [("Z53", Z53), ("Q53", Q53)]:
                tag = f"{name}_seed{seed}"
                t0 = time.time()
                model = train_one(tag, table, seed)
                gk = grokking_check(tag)
                emb = embedding_pc_test(model, table, seed)
                summary[tag] = {"type": name, "seed": seed, "grokking": gk,
                                 "embedding_pc_test": {k: v for k, v in emb.items() if k != "raw"},
                                 "wall_time_s": time.time() - t0}
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f"[{tag}] max_acc={gk['max_acc']:.4f} final_acc={gk['final_acc']:.4f} "
                      f"emb_patched={emb['mean']:.4f} CI=[{emb['ci_low']:.4f},{emb['ci_high']:.4f}]")

    # Aggregate across completed seeds
    z_means = [v["embedding_pc_test"]["mean"] for k, v in summary.items() if v["type"] == "Z53"]
    q_means = [v["embedding_pc_test"]["mean"] for k, v in summary.items() if v["type"] == "Q53"]
    print(f"\n=== Aggregate across {len(z_means)} completed seed-pairs ===")
    if z_means and q_means:
        print(f"Z53 patched prob: mean={np.mean(z_means):.4f} across seeds {z_means}")
        print(f"Q53 patched prob: mean={np.mean(q_means):.4f} across seeds {q_means}")


if __name__ == "__main__":
    main()
