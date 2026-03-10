
from jax import numpy as jnp
from jax import random as jrnd



def safe_trace_normalize(K, eps=1e-6):
    tr = jnp.trace(K)
    tr = jnp.maximum(tr, eps)
    K = K / tr
    K = jnp.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    return K
def log_rbf_kernel(X, eps=1e-6):
    # pairwise squared distance
    diff = X[:, None, :] - X[None, :, :]
    D = jnp.sum(diff ** 2, axis=-1)

    scale = jnp.median(D)
    scale = jnp.maximum(scale, eps)

    D = D / scale
    K = 1.0 / (1.0 + D)

    return safe_trace_normalize(K, eps)

def inner_product_kernel(X, eps=1e-8):
    # X: (n, d)
    norm = jnp.linalg.norm(X, axis=1, keepdims=True)
    norm = jnp.maximum(norm, eps)

    Xn = X / norm
    K = Xn @ Xn.T   # ∈ [-1, 1]

    return safe_trace_normalize(K, eps)
def covariance_kernel(X, eps=1e-6):
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    d = Xc.shape[1]

    K = (Xc @ Xc.T) / max(d, 1)

    return safe_trace_normalize(K, eps)

def compute_kernels(X):
    Ks = jnp.stack(
        [
            log_rbf_kernel(X),
            inner_product_kernel(X),
            covariance_kernel(X),
        ],
        axis=0,
    )
    return jnp.nan_to_num(Ks, nan=0.0)



def stat_kernel_featurewise(
    X,
    eps=1e-6,
):
    D, K = X.shape

    # -------- 1. mean / std over K --------
    mu = jnp.mean(X, axis=-1)                         # (D,)
    std = jnp.std(X, axis=-1, ddof=0) + eps            # unbiased=False

    # -------- 2. t-stat --------
    t_stat = mu / std                                 # (D,)

    # -------- 3. chi-square-like --------
    chi_stat = jnp.mean((X - mu[:, None]) ** 2, axis=-1) / (std ** 2 + eps)

    # -------- 4. fisher-like (pseudo split over K) --------
    mask = X > mu[:, None]                            # (D, K), bool
    mask_sum = jnp.maximum(mask.sum(axis=-1), 1)      # clamp(min=1)

    mu1 = (X * mask).sum(axis=-1) / mask_sum
    mu2 = (X * (~mask)).sum(axis=-1) / (K - mask_sum + eps)

    var1 = ((X - mu1[:, None]) ** 2 * mask).sum(axis=-1) / mask_sum
    var2 = ((X - mu2[:, None]) ** 2 * (~mask)).sum(axis=-1) / (K - mask_sum + eps)

    fisher_stat = (mu1 - mu2) ** 2 / (var1 + var2 + eps)

    # -------- 5. stack per feature --------
    stat = jnp.stack(
        [t_stat, chi_stat, fisher_stat],
        axis=1     # (D, 3)
    )

    return stat