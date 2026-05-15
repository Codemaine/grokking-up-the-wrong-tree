"""
train.py

Training loop for a single (operation, seed) model, matching the
paper's protocol exactly: AdamW, lr=1e-3, weight_decay=1.0, batch_size=512,
linear LR decay over 50,000 steps, 30/70 train/test split.

Also implements automatic catapult detection (a late-training loss spike,
Section 4.1 / Zhu et al. 2024) so that Table 2-style statistics can be
reproduced automatically rather than by eyeballing loss curves.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, build_dataset

TOTAL_STEPS = 50_000
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1.0
EVAL_EVERY = 100          # steps between eval/logging points
CATAPULT_LOSS_RATIO = 10  # a step-to-step loss ratio above this, after
                           # step 5000, is flagged as a catapult candidate


def final_token_logits(model, inputs):
    """inputs: [batch, 2] -> returns [batch, vocab] logits at final position."""
    logits = model(inputs)          # [batch, 2, vocab]
    return logits[:, -1, :]         # final token position


def evaluate(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        logits = final_token_logits(model, inputs)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        loss = F.cross_entropy(logits, labels).item()
    model.train()
    return acc, loss


def train_one_model(op, seed: int, n_heads: int = 1, n_layers: int = 1,
                     total_steps: int = TOTAL_STEPS, verbose: bool = True):
    """
    Trains a single model on `op` with the given `seed`.

    Returns a dict with:
        model            -- the trained HookedTransformer
        history          -- dict of per-eval-point lists: step, train_loss,
                             test_acc, test_loss
        catapults        -- list of (step, loss_before, loss_after) flagged
                             as late-training instabilities
        final_test_acc   -- float
        wall_time_s      -- training-only wall-clock time
    """
    torch.manual_seed(seed)
    model = build_model(seed=seed, n_heads=n_heads, n_layers=n_layers)
    train_in, train_lab, test_in, test_lab = build_dataset(seed=seed, op=op)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    shuffle_gen = torch.Generator().manual_seed(seed + 1_000_000)
    n_train = train_in.shape[0]

    history = {"step": [], "train_loss": [], "test_acc": [], "test_loss": []}
    catapults = []
    prev_train_loss = None

    t0 = time.perf_counter()
    step = 0
    while step < total_steps:
        # reshuffle each "epoch" through the fixed training subset
        perm = torch.randperm(n_train, generator=shuffle_gen)
        for i in range(0, n_train, BATCH_SIZE):
            if step >= total_steps:
                break
            idx = perm[i:i + BATCH_SIZE]
            batch_in, batch_lab = train_in[idx], train_lab[idx]

            optimizer.zero_grad()
            logits = final_token_logits(model, batch_in)
            loss = F.cross_entropy(logits, batch_lab)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss = loss.item()

            # catapult detection: large step-to-step loss ratio, late in training
            if prev_train_loss is not None and prev_train_loss > 1e-8:
                ratio = train_loss / prev_train_loss
                if step > 5000 and ratio > CATAPULT_LOSS_RATIO:
                    catapults.append((step, prev_train_loss, train_loss))
            prev_train_loss = train_loss

            if step % EVAL_EVERY == 0:
                test_acc, test_loss = evaluate(model, test_in, test_lab)
                history["step"].append(step)
                history["train_loss"].append(train_loss)
                history["test_acc"].append(test_acc)
                history["test_loss"].append(test_loss)

            step += 1

    wall_time_s = time.perf_counter() - t0
    final_test_acc, final_test_loss = evaluate(model, test_in, test_lab)

    if verbose:
        cat_str = f"{len(catapults)} catapult(s) detected" if catapults else "no catapults"
        print(f"  [{op.__name__ if hasattr(op, '__name__') else 'op'} "
              f"seed={seed} heads={n_heads}] "
              f"final_test_acc={final_test_acc:.4f}  "
              f"time={wall_time_s:.1f}s  {cat_str}")

    return {
        "model": model,
        "history": history,
        "catapults": catapults,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
        "wall_time_s": wall_time_s,
        "test_inputs": test_in,
        "test_labels": test_lab,
    }
