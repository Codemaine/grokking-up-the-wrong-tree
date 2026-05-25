"""
perturbation.py

The single-direction causal perturbation test (Section 3.4 of the paper).

For a clean input pair (a,b) and an independently sampled corrupted pair
(a',b'), we compute the residual-stream activation at the post-attention,
pre-MLP hook point (blocks.0.hook_resid_mid) at the final token position
for both inputs, take their difference, project this difference onto a
chosen unit direction v, and add only that projected component back into
the clean activation. We then measure the softmax probability assigned to
the correct answer under this patched activation.

Two direction choices are implemented:
    - embedding_pc1_direction : top PC of the static token-embedding matrix
    - residual_pc1_direction  : top PC of ~2000 collected residual-stream
                                 activations at the same hook point

Both a "weak" statistical explanation (explained-variance gap) and the
correct geometric diagnosis (cosine alignment between the two directions)
are implemented in diagnostics.py.
"""

import numpy as np
import torch

HOOK_NAME = "blocks.0.hook_resid_mid"
N = 53


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def embedding_pc1_direction(model) -> torch.Tensor:
    """Top principal component of the static token-embedding matrix W_E."""
    W_E = model.W_E.detach().cpu().numpy()          # [d_vocab, d_model]
    W_E_centered = W_E - W_E.mean(axis=0, keepdims=True)
    # SVD-based PCA
    _, _, Vt = np.linalg.svd(W_E_centered, full_matrices=False)
    pc1 = Vt[0]
    pc1 = pc1 / np.linalg.norm(pc1)
    return torch.tensor(pc1, dtype=torch.float32)


def collect_residual_activations(model, n_samples: int = 2000,
                                  seed: int = 0) -> np.ndarray:
    """
    Collects residual-stream activations at HOOK_NAME, final token
    position, over n_samples randomly sampled input pairs.
    Returns array of shape [n_samples, d_model].
    """
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, N, (n_samples,), generator=gen)
    b = torch.randint(0, N, (n_samples,), generator=gen)
    inputs = torch.stack([a, b], dim=1)

    acts = []

    def hook_fn(tensor, hook):
        acts.append(tensor[:, -1, :].detach().cpu().numpy())
        return tensor

    model.run_with_hooks(inputs, fwd_hooks=[(HOOK_NAME, hook_fn)])
    return np.concatenate(acts, axis=0)


def residual_pc1_direction(model, n_samples: int = 2000,
                            seed: int = 0) -> torch.Tensor:
    """Top principal component of collected residual-stream activations."""
    acts = collect_residual_activations(model, n_samples=n_samples, seed=seed)
    acts_centered = acts - acts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(acts_centered, full_matrices=False)
    pc1 = Vt[0]
    pc1 = pc1 / np.linalg.norm(pc1)
    return torch.tensor(pc1, dtype=torch.float32), acts


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def _patch_hook_factory(direction: torch.Tensor, clean_act: torch.Tensor,
                         corrupt_act: torch.Tensor):
    """
    Builds a hook function that replaces the clean activation's component
    along `direction` with the corrupted activation's component along the
    same direction, leaving the orthogonal complement untouched.
    """
    direction = direction / direction.norm()

    def hook_fn(tensor, hook):
        # tensor: [batch, seq, d_model]; we only touch the final position
        clean_proj = (clean_act @ direction).unsqueeze(-1) * direction
        corrupt_proj = (corrupt_act @ direction).unsqueeze(-1) * direction
        delta = corrupt_proj - clean_proj
        tensor[:, -1, :] = tensor[:, -1, :] + delta
        return tensor

    return hook_fn


def run_patch_trial(model, direction: torch.Tensor, clean_inputs: torch.Tensor,
                     corrupt_inputs: torch.Tensor) -> np.ndarray:
    """
    Runs the patched forward pass for a batch of (clean, corrupt) input
    pairs and returns the correct-answer probability for each clean input
    under the patch (correct answer = model's own prediction structure is
    NOT assumed here; caller supplies clean_labels separately -- see
    evaluate_patch_test below).

    Returns the patched logits at the final token position, [batch, vocab].
    """
    # first, grab clean and corrupt activations at HOOK_NAME (no patch)
    store = {}

    def make_store_hook(key):
        def store_hook(tensor, hook):
            store[key] = tensor[:, -1, :].detach().clone()
            return tensor
        return store_hook

    model.run_with_hooks(
        clean_inputs, fwd_hooks=[(HOOK_NAME, make_store_hook("clean"))]
    )
    model.run_with_hooks(
        corrupt_inputs, fwd_hooks=[(HOOK_NAME, make_store_hook("corrupt"))]
    )

    patch_hook = _patch_hook_factory(direction, store["clean"], store["corrupt"])
    patched_logits = model.run_with_hooks(
        clean_inputs, fwd_hooks=[(HOOK_NAME, patch_hook)]
    )
    return patched_logits[:, -1, :]


def evaluate_patch_test(model, op, direction: torch.Tensor, n_trials: int = 200,
                         seed: int = 0):
    """
    Runs the full patch test: samples n_trials clean/corrupt input-pair
    trials, patches the given direction, and returns:
        mean_prob   -- mean correct-answer probability under the patch
        probs       -- per-trial correct-answer probability array [n_trials]
                        (for bootstrapping via bootstrap_ci)

    `op` is the ground-truth operation (e.g. from ops.OPERATIONS), used to
    compute the correct label for each clean input pair.
    """
    gen = torch.Generator().manual_seed(seed)
    clean_a = torch.randint(0, N, (n_trials,), generator=gen)
    clean_b = torch.randint(0, N, (n_trials,), generator=gen)
    corrupt_a = torch.randint(0, N, (n_trials,), generator=gen)
    corrupt_b = torch.randint(0, N, (n_trials,), generator=gen)

    clean_inputs = torch.stack([clean_a, clean_b], dim=1)
    corrupt_inputs = torch.stack([corrupt_a, corrupt_b], dim=1)

    clean_labels = torch.tensor(
        [op(int(a), int(b)) for a, b in clean_inputs.tolist()], dtype=torch.long
    )

    patched_logits = run_patch_trial(model, direction, clean_inputs, corrupt_inputs)
    probs_full = torch.softmax(patched_logits, dim=-1)
    correct_probs = probs_full[torch.arange(n_trials), clean_labels].detach().cpu().numpy()

    return float(correct_probs.mean()), correct_probs


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000,
                  ci: float = 0.95, seed: int = 0):
    """Simple percentile bootstrap CI for the mean of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)
