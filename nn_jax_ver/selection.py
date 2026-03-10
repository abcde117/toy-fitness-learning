
import jax

from jax import numpy as jnp

def laplacian_kernel_matrix(X, Y=None, sigma=1.0, eps=1e-8):
    """
    X: (N, D)
    Y: (M, D) or None
    return: (N, M)
    """
    if Y is None:
        Y = X

    dist_l1 = jnp.sum(
        jnp.abs(X[:, None, :] - Y[None, :, :]),
        axis=-1
    )

    return jnp.exp(-dist_l1 / (sigma + eps))


@jax.jit
def cov_mu_reject(m, key):
    """
    m: (n, m)
    returns: (n,) int array
    """

    lap_ker = laplacian_kernel_matrix(m)

    # mu = mean over last dim
    mu = jnp.mean(m, axis=-1)

    # h = D^{-1/2} K D^{-1/2}
    mu_inv_sqrt = mu ** (-0.5)
    D_inv_sqrt = jnp.diag(mu_inv_sqrt)

    h = D_inv_sqrt @ lap_ker @ D_inv_sqrt

    # Cholesky
    dl = jnp.linalg.cholesky(h + 1e-3 * jnp.eye(h.shape[0]))

    hmu = jnp.sqrt(mu)

    # random z
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (lap_ker.shape[0],))

    x = mu + dl @ z
    xs = x / jnp.sqrt(mu)

    ones = jnp.ones(dl.shape[-1])

    upb = hmu + dl @ (ones * 1.96) / x.shape[0]
    lob = hmu - dl @ (ones * 1.96) / x.shape[0]

    upb_c = xs > upb
    lob_c = xs < lob

    return ((upb_c.astype(jnp.int32) + lob_c.astype(jnp.int32)) >= 1).astype(jnp.int32)



def build_interaction_matrix(m, score_hat, tau=1.0):
    """
    m:         (N, D)
    score_hat: (N, K)
    return:    (N, N)
    """

    Km = (m @ m.T) / m.shape[-1]   # (N, N)

    Ks = jnp.sum(
        (score_hat[:, None, :] - score_hat[None, :, :]) ** 2,
        axis=-1
    )
    Ks = jnp.exp(-Ks / tau)

    K = Km * Ks
    return K
def interaction_to_force(K, normalize=True):
    """
    K: (N, N)
    return: (N,)
    """

    f = jnp.sum(K, axis=-1)

    if normalize:
        f = f / (jnp.sum(f) + 1e-8)

    return f



def update_prior_gaussian(
    mu,
    Sigma,
    m,
    score_hat,
    alpha=0.1,
    beta=0.1,
    eps=1e-5,
    tau=1.0,
):
    """
    mu:        (N,)
    Sigma:     (N, N)
    m:         (N, D)
    score_hat: (N, K)
    """

    N = mu.shape[0]

    # 1. interaction
    K = build_interaction_matrix(m, score_hat, tau=tau)

    # 2. sample-wise force
    f = interaction_to_force(K)

    # 3. mean update
    mu_new = (1.0 - alpha) * mu + alpha * f

    # 4. covariance update
    Sigma_new = (1.0 - beta) * Sigma + beta * K

    # 5. numerical stability
    Sigma_new = Sigma_new + eps * jnp.eye(N)

    return mu_new, Sigma_new



def lrt_ni_vs_n(mu, Sigma, eps=1e-8):
    """
    mu:    (N,)
    Sigma: (N, N)
    return: (N,)
    """

    var_i = jnp.diag(Sigma)

    mu_g = jnp.mean(mu)
    var_g = jnp.mean(var_i)

    llr = (mu - mu_g) ** 2 / (var_g + eps) \
          - jnp.log(var_i + eps)

    return llr
