"""
FAD computation and log-scale self-reference normalization (S_norm).

Implements:
  - Fréchet Distance (Eq. 1 in the paper)
  - S_norm normalization (Eq. 3 in the paper)
"""

import math
import numpy as np
from scipy.linalg import sqrtm


def compute_statistics(embs: np.ndarray) -> tuple:
    """
    Compute mean and covariance from an (N, D) embedding matrix.

    Returns:
        (mu, sigma) where mu is (D,) and sigma is (D, D).
    """
    mu = np.mean(embs, axis=0)
    sigma = np.cov(embs, rowvar=False)
    if sigma.ndim < 2:
        sigma = np.atleast_2d(sigma)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, sig1: np.ndarray,
                     mu2: np.ndarray, sig2: np.ndarray) -> float:
    """
    Fréchet Distance between two Gaussians N(mu1, sig1) and N(mu2, sig2).

    FAD = ||mu1 - mu2||^2 + Tr(sig1 + sig2 - 2 * (sig1 @ sig2)^{1/2})
    """
    diff = mu1 - mu2
    eps = 1e-6
    sig1 = sig1 + eps * np.eye(sig1.shape[0])
    sig2 = sig2 + eps * np.eye(sig2.shape[0])
    covmean, _ = sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.real(
        diff @ diff + np.trace(sig1 + sig2 - 2 * covmean)
    ))


def s_norm(fad_val: float, fad_max: float) -> float:
    """
    Log-scale self-reference normalization (Eq. 3).

        S_norm = log(1 + FAD) / log(1 + FAD_max)

    Args:
        fad_val: FAD value for a specific condition.
        fad_max: Maximum FAD observed for this encoder across all conditions.
    """
    return math.log(1 + fad_val) / math.log(1 + fad_max)


def compute_fad_max(results: list, encoder: str) -> float:
    """Find FAD_max for an encoder across all conditions and datasets."""
    vals = [d["fad"] for d in results if d["encoder"] == encoder]
    if not vals:
        raise ValueError(f"No results found for encoder '{encoder}'")
    return max(vals)
