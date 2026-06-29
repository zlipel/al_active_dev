from __future__ import annotations

import numpy as np
from scipy.stats import norm
import torch
import gpytorch
from pygmo import hypervolume


# psi function for ehvi
def exipsi_vectorized(a, b, m, s):
    """
    Vectorized psi function for EHVI.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.asarray(m)
    s = np.asarray(s)

    # guard tiny sigma
    s = np.maximum(s, 1e-12)
    z = (b - m) / s
    return s * norm.pdf(z) + (a - m) * norm.cdf(z)


def ehvi_analytic(mus1: np.ndarray | list, sigmas1: np.ndarray | list, mus2: np.ndarray | list, sigmas2: np.ndarray | list, augmented_front: np.ndarray | list) -> np.ndarray:
    """
    Analytic EHVI for 2D MINIMIZATION problems (stripe decomposition).
    augmented_front must be in MIN space and sorted by objective-2 (y2) ascending,
    with 2 sentinel points.
    """
    y1 = augmented_front[:, 0]
    y2 = augmented_front[:, 1]

    y1_prev = y1[:-1]
    y1_curr = y1[1:]
    y2_curr = y2[1:]   

    # broadcast shapes: (n_stripes, 1) vs (1, n_samples)
    y1_prev = y1_prev[:, None]
    y1_curr = y1_curr[:, None]
    y2_curr = y2_curr[:, None]

    mus1 = np.asarray(mus1)[None, :]
    sigmas1 = np.asarray(sigmas1)[None, :]
    mus2 = np.asarray(mus2)[None, :]
    sigmas2 = np.asarray(sigmas2)[None, :]

    sigmas1 = np.maximum(sigmas1, 1e-12)
    sigmas2 = np.maximum(sigmas2, 1e-12)

    term1 = (y1_prev - y1_curr) * norm.cdf((y1_curr - mus1) / sigmas1) * exipsi_vectorized(
        y2_curr, y2_curr, mus2, sigmas2
    )

    psi_prev = exipsi_vectorized(y1_prev, y1_prev, mus1, sigmas1)
    psi_curr = exipsi_vectorized(y1_prev, y1_curr, mus1, sigmas1)

    term2 = (psi_prev - psi_curr) * exipsi_vectorized(y2_curr, y2_curr, mus2, sigmas2)

    return np.sum(term1 + term2, axis=0)


# Utilities to create the augment front...
def _to_min_space(pareto_Y: np.ndarray, front: str) -> np.ndarray:
    """
    Convert to minimization space:
    - upper: max-max -> negate so we minimize
    - lower: min-min -> unchanged
    """
    Y = np.asarray(pareto_Y, dtype=np.float32)
    if front == "upper":
        return -Y
    if front == "lower":
        return Y
    raise ValueError("front must be 'upper' or 'lower'")


def _ref_point_frac(pmin: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """
    Naive reference point method in min space: fraction of of the span beyond nadir.
    0 < frac < 1 means some frac of the nadir past the nadir point.
    0.0 means the nadir point itself.
    """
    worst = pmin.max(axis=0)
    return worst + frac * np.abs(worst)


def _ref_point_on_IN_line(
    pmin: np.ndarray,
    tau: float = 0.05,
    cap_frac: float = 0.8,
) -> np.ndarray:
    """
    Reference point on ray from ideal -> nadir -> beyond, in minimization space.
    Makes sure rp is worse than nadir!
    """
    I = pmin.min(axis=0)  # ideal (best) in minimization
    N = pmin.max(axis=0)  # nadir (worst) in minimization

    d = N - I
    L = float(np.linalg.norm(d))
    if L < 1e-8:
        u = np.array([1.0, 0.0], dtype=np.float32)
        L = 1.0
    else:
        u = d / L

    s = float(np.clip(tau * L, 0.0, cap_frac * L))
    R = N + s * u

    # strict domination: ref must be worse than every point (at least worse than N)
    return np.maximum(R, N + 1e-12)


def _ref_point_halfway(pmin: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """
    Another heuristic choice: halfway between the frac and nadir point
    """
    R0 = _ref_point_frac(pmin, frac=frac) # naive frac method
    I = pmin.min(axis=0)  # ideal (best) in minimization
    N = pmin.max(axis=0)  # nadir (worst) in minimization

    d = N - I
    L = float(np.linalg.norm(d))
    u = np.array([1.0, 0.0], dtype=np.float32) if L < 1e-8 else d / L
    
    R0_on_IN = N + np.dot(R0 - N, u)*u


    return np.maximum(0.5 * (R0_on_IN + N), N + 1e-4)


def _get_ref_point(
    pmin: np.ndarray,
    ref_mode: str = "frac",   # "frac" | "in_line" | "halfway"
    frac: float = 0.5,
    tau: float = 0.05,
    cap_frac: float = 0.8,
) -> np.ndarray:
    if ref_mode == "frac":
        return _ref_point_frac(pmin, frac=frac)
    if ref_mode == "in_line":
        return _ref_point_on_IN_line(pmin, tau=tau, cap_frac=cap_frac)
    if ref_mode == "halfway":
        return _ref_point_halfway(pmin, frac=frac)
    raise ValueError(f"Unknown ref_mode={ref_mode}")


def _augment_front(
    pareto_front_min: np.ndarray,
    ref_point_min: np.ndarray,
    big: float = 1e6,
) -> np.ndarray:
    """
    Build augmented front for minimization stripe-decomposed EHVI.

    Returns: (N+2,2) array sorted by y2 ascending.
    """
    pf = np.asarray(pareto_front_min, dtype=np.float32)
    r1, r2 = map(float, ref_point_min)

    # sort by objective 2 ascending (minimization)
    pf_sorted = np.array(sorted(map(tuple, pf), key=lambda p: p[1]), dtype=np.float32)

    # sentinel points to close the stripes toward the reference rectangle
    # (r1, -inf) and (-inf, r2)
    return np.vstack([
        np.array([r1,  -1*big], dtype=np.float32),
        pf_sorted,
        np.array([-1*big, r2], dtype=np.float32),
    ])


# public api to generate augmented front (what we call in run_ga)
def front_augmentation(
    pareto_front: np.ndarray,
    front: str,
    ref_mode: str = "frac",  # default is frac
    frac: float = 0.5,
    tau: float = 0.05,
    cap_frac: float = 0.8,
    big: float = 1e6,
    return_ref: bool = False,
    mc_mode: bool = False,
):
    """
    Augment and order Pareto front for analytic EHVI.
    Parameters:
    -----------
      pareto_front: np.ndarray
        [N,2] in ORIGINAL space
      front: str
        'upper' (max-max) or 'lower' (min-min)
      ref_mode: str
        method to choose reference point
      frac: float
        fraction parameter for 'frac' ref_mode
      tau: float
        tau parameter for 'in_line' ref_mode
      cap_frac: float
        cap_frac parameter for 'in_line' ref_mode
      big: float
        large number for sentinels
    Returns:
    -----------
    augmented_front: np.ndarray
        2D array with sentinels, in MIN space, sorted by objective-2 ascending
    ref_point: np.ndarray  
        Reference point used for HV calculation (only if return_ref is True)
    """
    pmin = _to_min_space(pareto_front, front=front)
    ref_min = _get_ref_point(pmin, ref_mode=ref_mode, frac=frac, tau=tau, cap_frac=cap_frac)

    if not mc_mode:
        aug = _augment_front(pmin, ref_min, big=big)
    else:
        aug = pmin  # no sentinels for MC-EHVI
    
    if return_ref:
        return aug, ref_min
    return aug

def is_dominated_by_front(pf: np.ndarray, s: np.ndarray) -> bool:
    le = np.all(pf <= s[None, :], axis=1)
    lt = np.any(pf <  s[None, :], axis=1)
    return bool(np.any(le & lt))

def filter_nondominated(points: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask of nondominated points for 2D minimization.
    A point is dominated if there exists another point <= in both dims and < in at least one.
    """
    P = np.asarray(points, dtype=np.float32)
    n = P.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # sort by y1 ascending, then y2 ascending
    idx = np.lexsort((P[:, 1], P[:, 0]))
    P_sorted = P[idx]

    mask_sorted = np.ones(n, dtype=bool)
    best_y2 = np.inf

    # sweep from best y1 to worst y1
    for k in range(n):
        y2 = P_sorted[k, 1]
        # dominated if we've already seen a point with y2 <= current y2
        # because y1 is non-increasing in dominance due to sorting
        if y2 >= best_y2 - 0.0:   # non-strict here is correct because y1 is <= and y2 is <=
            mask_sorted[k] = False
        else:
            best_y2 = y2

    mask = np.zeros(n, dtype=bool)
    mask[idx] = mask_sorted
    return mask

def ehvi_samples(sample, base_hv, pareto_front, ref_point):
    # all args are in MIN space
    if is_dominated_by_front(pareto_front, sample):
        return 0.0
    if np.any(sample >= ref_point):
        return 0.0

    extended_front = np.vstack([pareto_front, sample])
    nd_front = extended_front[filter_nondominated(extended_front)]
    hv = hypervolume(nd_front).compute(ref_point)
    return hv - base_hv

def monte_carlo_ehvi_batch(
    candidate_tensor, model, pareto_front, ref_point, base_hv,
    front="upper", min_samples=64, max_samples=548, stderr_tol=1e-3,
    chunk_size=128,
):
    """
    Monte Carlo EHVI estimation for batch of candidates.
    Parameters:
    ----------- 
    candidate_tensor: torch.Tensor
        (B, D) tensor of candidates
    model: gpytorch model
        trained GP model
    pareto_front: np.ndarray
        (N,2) array in MIN space
    ref_point: np.ndarray
        (2,) array in MIN space
    base_hv: float
        hypervolume of current pareto front in MIN space
    front: str
        'upper' or 'lower'
    min_samples: int
        minimum MC samples per candidate
    max_samples: int
        maximum MC samples per candidate
    stderr_tol: float
        standard error tolerance for convergence
    chunk_size: int
        number of samples to draw per iteration
    Returns:
    -----------
    ehvi_vals: np.ndarray
        (B,) array of EHVI values
    """
 
    B = candidate_tensor.size(0)
    ehvi_vals = np.zeros(B, dtype=np.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(candidate_tensor)

    for i in range(B):
        improvements = []
        drawn = 0
        converged = False

        while (not converged) and (drawn < max_samples):
            S = min(chunk_size, max_samples - drawn)
            
            with torch.no_grad():
                samples = posterior.rsample(torch.Size([S]))  # (S, B, 2)

            s_np = samples[:, i, :].detach().cpu().numpy()   # (S, 2)
            if front == "upper":
                s_np *= -1.0

            for s in s_np:
                improvements.append(ehvi_samples(s, base_hv, pareto_front, ref_point))

            drawn += S

            if drawn >= min_samples:
                arr = np.asarray(improvements, dtype=np.float32)
                stderr = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.inf
                if stderr < stderr_tol:
                    converged = True

        ehvi_vals[i] = float(np.mean(improvements)) if improvements else 0.0

    return ehvi_vals