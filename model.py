"""
model.py

Model construction and dataset preparation. Architecture matches the
paper exactly: one-layer transformer, d_model=128, d_head=64, d_mlp=512,
ReLU, no LayerNorm, context length 2, vocab size 53.

An optional `n_heads` override is provided for the Priority-#4 ablation
(architectural generalisation check with multiple attention heads);
default is 1 head to match the original confirmatory sweep.
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

N = 53


def build_model(seed: int, n_heads: int = 1, n_layers: int = 1,
                 d_model: int = 128, d_mlp: int = 512) -> HookedTransformer:
    """
    Instantiates a HookedTransformer with the paper's architecture.
    d_head is derived as d_model // n_heads so that n_heads * d_head =
    d_model stays constant across the head-count ablation (Priority #4),
    keeping total attention capacity comparable.
    """
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_head = d_model // n_heads

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=N,
        n_ctx=2,
        act_fn="relu",
        normalization_type=None,   # no LayerNorm, matching the paper
        seed=seed,
    )
    model = HookedTransformer(cfg)
    return model


def build_dataset(seed: int, op, train_frac: float = 0.30):
    """
    Builds the full (a, b) -> op(a, b) dataset and a deterministic
    train/test split, seeded jointly with model initialisation as in
    the paper's comparability controls (Section 3.5): the seed
    determines both weight init and the split, and the SAME seed is
    reused across all three operations so that Z53/Q53/C53 seed k
    share the same underlying random state.

    Returns:
        train_inputs, train_labels, test_inputs, test_labels
        (all torch.LongTensor)
    """
    gen = torch.Generator().manual_seed(seed)

    all_pairs = torch.cartesian_prod(torch.arange(N), torch.arange(N))  # [N*N, 2]
    labels = torch.tensor([op(int(a), int(b)) for a, b in all_pairs.tolist()],
                           dtype=torch.long)

    n_total = all_pairs.shape[0]
    perm = torch.randperm(n_total, generator=gen)
    n_train = int(train_frac * n_total)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_inputs, train_labels = all_pairs[train_idx], labels[train_idx]
    test_inputs, test_labels = all_pairs[test_idx], labels[test_idx]

    return train_inputs, train_labels, test_inputs, test_labels


def batch_shuffle_generator(seed: int):
    """Dedicated generator for DataLoader-style shuffling, seeded separately
    from the split generator so that shuffle order is reproducible and
    matched across operations at a given seed (comparability control)."""
    return torch.Generator().manual_seed(seed + 1_000_000)
