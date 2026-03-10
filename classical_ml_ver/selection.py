import numpy as np

def laplacian_kernel_matrix_np(X, Y=None, sigma=1.0, eps=1e-8):
    """
    X: (N, D)
    Y: (M, D) or None
    return: (N, M)
    """
    if Y is None:
        Y = X

    # L1 distance
    dist_l1 = np.sum(
        np.abs(X[:, None, :] - Y[None, :, :]),
        axis=-1
    )

    return np.exp(-dist_l1 / (sigma + eps))


def cov_mu_reject_np(m, sigma=1.0, eps=1e-2, alpha=1.96, seed=None):
    """
    m: (N, M) embedding
    return: (N,) binary selection mask
    """
    if seed is not None:
        np.random.seed(seed)

    N = m.shape[0]

    # ---------- kernel ----------
    K = laplacian_kernel_matrix_np(m, sigma=sigma)

    # ---------- mu ----------
    mu = np.maximum(np.abs(m.mean(axis=1)) + eps, eps)   # (N,) - ensure positive

    # ---------- normalized covariance ----------
    D_inv_sqrt = np.diag(mu ** (-0.5))
    H = D_inv_sqrt @ K @ D_inv_sqrt

    # ---------- cholesky ----------
    L = np.linalg.cholesky(
        H + eps * np.eye(N)
    )

    # ---------- gaussian fluctuation ----------
    z = np.random.randn(N)
    x = mu + L @ z

    # ---------- relative scale ----------
    xs = x / np.sqrt(mu)

    # ---------- confidence bounds ----------
    ones = np.ones(N)
    bound = (L @ ones) / N
    upper = np.sqrt(mu) + alpha * bound
    lower = np.sqrt(mu) - alpha * bound

    # ---------- reject condition ----------
    reject = (xs > upper) | (xs < lower)

    return reject.astype(int)

import numpy as np

def build_interaction_matrix_np(m, score_hat, tau=1.0, eps=1e-8):
    """
    m:         (N, D)
    score_hat: (N, K)
    return:    (N, N)
    """

    # feature interaction
    Km = (m @ m.T) / (m.shape[1] + eps)     # (N, N)

    # score interaction
    diff = score_hat[:, None, :] - score_hat[None, :, :]
    Ks = np.sum(diff ** 2, axis=-1)
    Ks = np.exp(-Ks / (tau + eps))

    # joint interaction
    K = Km * Ks

    return K
def interaction_to_force_np(K, normalize=True, eps=1e-8):
    """
    K: (N, N)
    return: (N,)
    """
    f = K.sum(axis=-1)

    if normalize:
        f = f / (f.sum() + eps)

    return f

def update_prior_gaussian_np(
    mu,
    Sigma,
    m,
    score_hat,
    alpha=0.1,
    beta=0.1,
    eps=1e-5,
    tau=1.0
):
    """
    mu:        (N,)
    Sigma:     (N, N)
    m:         (N, D)
    score_hat: (N, K)

    return:
        mu_new, Sigma_new
    """

    N = mu.shape[0]

    # 1. interaction
    K = build_interaction_matrix_np(m, score_hat, tau=tau)

    # 2. sample-wise force
    f = interaction_to_force_np(K)

    # 3. mean update (importance / fitness)
    mu_new = (1 - alpha) * mu + alpha * f

    # 4. covariance update
    Sigma_new = (1 - beta) * Sigma + beta * K

    # 5. stabilize
    Sigma_new = Sigma_new + eps * np.eye(N)

    return mu_new, Sigma_new
def lrt_ni_vs_n_np(mu, Sigma, eps=1e-8):
    """
    mu:    (N,)
    Sigma: (N, N)
    return:
        llr: (N,)
    """
    var_i = np.diag(Sigma)           # (N,)

    mu_g = mu.mean()                 # scalar
    var_g = var_i.mean()             # scalar

    llr = (mu - mu_g) ** 2 / (var_g + eps) \
          - np.log(var_i + eps)

    return llr

