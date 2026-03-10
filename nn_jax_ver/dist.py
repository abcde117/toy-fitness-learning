import jax

from jax import numpy as jnp
import jax.scipy as jsp
from jax import random as jrnd




def build_featurewise_sample_gaussians(xo, eps=1e-5):
    # xo: (N, D)
    N, D = xo.shape

    # -------- mean --------
    mu = jnp.mean(xo, axis=0)            # (D,)
    mus = jnp.ones_like(xo) * mu[None]   # (N, D)
    mus = mus.T                          # (D, N)

    # -------- centered --------
    X = xo - jnp.mean(xo, axis=0, keepdims=True)  # (N, D)
    X = X.T                                       # (D, N)

    # -------- covariances --------
    covs = X[:, :, None] * X[:, None, :]          # (D, N, N)
    covs = covs / (N - 1)

    eye = jnp.eye(N)
    covs = covs + eps * eye[None, :, :]

    return mus, covs

def build_samplewise_gaussians(xo, eps=1e-5):
    # xo: (N, D)
    N, D = xo.shape

    # -------- means --------
    mus = xo                              # (N, D)

    # -------- global covariance --------
    X = xo - jnp.mean(xo, axis=0, keepdims=True)
    cov = (X.T @ X) / (N - 1)             # (D, D)
    cov = cov + eps * jnp.eye(D)

    covs = jnp.broadcast_to(cov, (N, D, D))

    return mus, covs



def mvn_log_prob(x, mu, cov):
    """
    x   : (N,)
    mu  : (N,)
    cov : (N, N)
    """
    N = x.shape[0]

    diff = x - mu
    chol = jnp.linalg.cholesky(cov)

    # solve L y = diff
    y = jsp.linalg.solve_triangular(chol, diff, lower=True)

    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    quad = jnp.dot(y, y)

    return -0.5 * (quad + log_det + N * jnp.log(2 * jnp.pi))


def gmm_log_prob(x, mus, covs, weights, eps=1e-6):
    K, N = mus.shape

    def one_component(mu, cov, w, xk):
        cov = cov + eps * jnp.eye(N)
        return jnp.log(w) + mvn_log_prob(xk, mu, cov)

    log_probs = jax.vmap(one_component)(
        mus,
        covs,
        weights,
        x.T,      
    )

    return jax.nn.logsumexp(log_probs, axis=0)