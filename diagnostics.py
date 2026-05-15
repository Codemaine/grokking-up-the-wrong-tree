"""
diagnostics.py

Diagnostic and dimensionality-analysis tools.

  - explained_variance_gap : PC1 vs PC2 gap of the embedding matrix
                              (the "weak PC1" explanation that was tested
                              and falsified in Section 4.3 of the paper)
  - cosine_alignment        : alignment between embedding PC1 and
                               residual-stream PC1 (the correct diagnosis)
  - residual_pca_spectrum   : full explained-variance spectrum of the
                               residual stream (for scree plots)
  - participation_ratio     : single-number summary of how "spread out"
                               the residual-stream variance is across
                               components. PR = (sum lambda_i)^2 / sum(lambda_i^2).
                               Ranges from 1 (all variance in one direction,
                               maximally concentrated) to d_model (variance
                               spread equally across all directions,
                               maximally distributed).
  - effective_rank          : exponential of the Shannon entropy of the
                               normalised eigenvalue spectrum. A second,
                               differently-weighted summary of "how many
                               directions carry the variance" -- reported
                               alongside PR since the two measures can
                               disagree for spectra with a few outlier
                               components plus a long flat tail.

This directly operationalises the Priority-#2 "distributed vs
concentrated representation" hypothesis from Section 5.1/5.5 of the
paper: it predicts PR(Q53) < PR(Z53) and PR(Q53) < PR(C53), i.e. the
non-associative operation should compress its task-relevant variance
into fewer residual-stream directions than either associative
comparison operation.
"""

import numpy as np

from perturbation import collect_residual_activations


def _pca_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of the covariance of X, [n_samples, d]."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X_centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, a_min=0, a_max=None)  # numerical safety
    return eigvals


def explained_variance_gap(embedding_matrix: np.ndarray) -> dict:
    """
    Returns explained-variance ratios of PC1 and PC2 of the embedding
    matrix, and their gap. This is the "weak/unstable PC1" explanation
    that was hypothesised and then falsified in Section 4.3 -- included
    here so the falsification is reproducible, not just narrated.
    """
    eigvals = _pca_eigenvalues(embedding_matrix)
    total = eigvals.sum()
    ratios = eigvals / total
    return {
        "pc1_ratio": float(ratios[0]),
        "pc2_ratio": float(ratios[1]),
        "gap": float(ratios[0] - ratios[1]),
        "full_spectrum": ratios,
    }


def cosine_alignment(direction_a: "torch.Tensor", direction_b: "torch.Tensor") -> float:
    """Cosine similarity between two direction vectors."""
    a = direction_a.detach().cpu().numpy()
    b = direction_b.detach().cpu().numpy()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.abs(np.dot(a, b)))  # unsigned, since PC sign is arbitrary


def residual_pca_spectrum(model, n_samples: int = 2000, seed: int = 0) -> np.ndarray:
    """Full explained-variance-ratio spectrum of residual-stream activations."""
    acts = collect_residual_activations(model, n_samples=n_samples, seed=seed)
    eigvals = _pca_eigenvalues(acts)
    ratios = eigvals / eigvals.sum()
    return ratios


def participation_ratio(eigvals_or_ratios: np.ndarray) -> float:
    """
    PR = (sum lambda_i)^2 / sum(lambda_i^2).
    Scale-invariant: works whether given raw eigenvalues or normalised
    ratios, since PR is invariant to uniform rescaling of the spectrum.
    """
    x = np.asarray(eigvals_or_ratios, dtype=np.float64)
    return float((x.sum() ** 2) / (x ** 2).sum())


def effective_rank(eigvals_or_ratios: np.ndarray, eps: float = 1e-12) -> float:
    """
    exp(Shannon entropy) of the normalised eigenvalue distribution.
    A second summary statistic for "how many directions carry variance",
    reported alongside participation ratio (see module docstring).
    """
    x = np.asarray(eigvals_or_ratios, dtype=np.float64)
    p = x / x.sum()
    p = p[p > eps]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def dimensionality_summary(model, n_samples: int = 2000, seed: int = 0) -> dict:
    """Convenience wrapper: full spectrum + PR + effective rank in one call."""
    ratios = residual_pca_spectrum(model, n_samples=n_samples, seed=seed)
    return {
        "spectrum": ratios,
        "participation_ratio": participation_ratio(ratios),
        "effective_rank": effective_rank(ratios),
        "pc1_ratio": float(ratios[0]),
    }
