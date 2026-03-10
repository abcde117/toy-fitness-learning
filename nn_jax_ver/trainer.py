

from .data_process import build_type_graph
from .fitness import  compute_kernels,stat_kernel_featurewise
from .dist import *
import jax

from jax import numpy as jnp
from jax import random as jrnd

import equinox as eqx
import optax


# encoding train
def  enco_loss_fn(
    model,
    x,     # (N, D)
    xo,    # (N, D+T)
    t,
    q,
    lambda1=0.5,
    lambda2=0.5,
):
    # ---------- graph ----------
    g1 = build_type_graph(t)      # (N, N)
    g2 = build_type_graph(q)      # (N, N)
    g = jnp.stack([g1, g2], axis=0)

    # ---------- forward ----------
    m = model(x, g)               # (N, K)
    mim = m @ m.T                 # (N, N)

    # ---------- kernel ----------
    k = compute_kernels(xo)       # (C, N, N)

    # ---------- stat ----------
    stat = stat_kernel_featurewise(xo)  # (N, K) 或 (N, 3)

    # ---------- loss ----------
    loss_kernel = jnp.mean((mim[None, :, :] - k) ** 2)
    loss_stat = jnp.mean((m - stat) ** 2)

    loss = lambda1 * loss_kernel + lambda2 * loss_stat

    aux = {
        "loss_kernel": loss_kernel,
        "loss_stat": loss_stat,
        "loss_total": loss,
    }

    return loss, aux





@eqx.filter_jit
def exp_step(
    model,
    opt_state,
    optimizer,
    x,
    xo,
    t,
    q,
    lambda1=0.5,
    lambda2=0.5,
):
    (loss, aux), grads = eqx.filter_value_and_grad(
        enco_loss_fn,
        has_aux=True
    )(model, x, xo, t, q, lambda1, lambda2)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux

def make_batches(data, batch_size, key):
   N = data["x"].shape[0]
   perm = jax.random.permutation(key, N)
   max_i = (N // batch_size) * batch_size
   for i in range(0, max_i, batch_size):
        idx = perm[i:i + batch_size]
        yield {
            "x": data["x"][idx],
            "xo": data["xo"][idx],
            "type": data["type"][idx],
            "quality": data["quality"][idx],
        }

def encoding_train_runner(
    model,
    data,
    optimizer,
    opt_state,
    batch_size=32,
    lambda1=0.5,
    lambda2=0.5,
    epochs=1,
    seed=0,
):
    x_all  = jnp.array(data["x"])
    xo_all = jnp.array(data["xo"])
    t_all  = jnp.array(data["type"])
    q_all  = jnp.array(data["quality"])

    key = jax.random.PRNGKey(seed)

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        total_loss = 0.0
        n_batch = 0

        for batch in make_batches(
            {
                "x": x_all,
                "xo": xo_all,
                "type": t_all,
                "quality": q_all,
            },
            batch_size,
            subkey,
        ):
            model, opt_state, loss, aux = exp_step(
                model,
                opt_state,
                optimizer,
                batch["x"],
                batch["xo"],
                batch["type"],  # Changed from batch["t"]
                batch["quality"],
                lambda1,
                lambda2,
            )

            total_loss += loss
            n_batch += 1

        print(f"[epoch {epoch}] avg loss = {float(total_loss / n_batch)}")

    return model, opt_state





#decoding train

def compute_reference_scores(xo):
    """
    xo: shape (N, D)
    returns:
        score_ref:  shape (N, D)
        score_refk: shape (N, D)
    """

    N, D = xo.shape

    # ---------- feature-wise GMM ----------
    def log_p_fn(x):
        mus, covs = build_featurewise_sample_gaussians(x)
        weights = jnp.ones(D) / D
        return gmm_log_prob(x, mus, covs, weights)

    # ---------- sample-wise GMM (on x.T) ----------
    def log_pk_fn(x):
        muk, covk = build_samplewise_gaussians(x)
        weights = jnp.ones(N) / N
        return gmm_log_prob(x.T, muk, covk, weights)

    # ∇_x log p(x)
    score_ref = jax.grad(log_p_fn)(xo)

    # ∇_x log p(x^T)
    score_refk = jax.grad(log_pk_fn)(xo)

    return score_ref, score_refk

@eqx.filter_jit
def compute_decoder_score(m, decoder):
    def energy_fn(m):
        return jnp.sum(decoder(m))
    return jax.grad(energy_fn)(m)

def kl_like_loss(score_hat, score_refk, tau=1.0):
    """
    score_hat:  (N, D)
    score_refk: (N, D)
    returns: scalar
    """

    # pairwise squared distance: (N, N)
    kh = jnp.sum(
        (score_hat[:, None, :] - score_hat[None, :, :]) ** 2,
        axis=-1
    )

    kr = jnp.sum(
        (score_refk[:, None, :] - score_refk[None, :, :]) ** 2,
        axis=-1
    )

    # log-softmax over last dim
    log_p_hat = jax.nn.log_softmax(-kh / tau, axis=-1)
    log_p_ref = jax.nn.log_softmax(-kr / tau, axis=-1)

    # mean over all entries
    return jnp.mean(log_p_ref - log_p_hat)


def point_wise_proj_loss(score_hat, score_ref):
    """
    score_hat: (N, D)
    score_ref: (N, D)
    returns: scalar
    """

    # (N, D, D)
    diff = score_hat[:, :, None] - score_ref[:, None, :]

    # softmax over last dim (D)
    weights = jax.nn.softmax(-(diff ** 2), axis=-1)

    # weighted projection: (N, D)
    proj_ref = jnp.sum(weights * score_ref[:, None, :], axis=-1)

    # squared error
    score_loss = jnp.sum((score_hat - proj_ref) ** 2)

    return score_loss
def compute_losses(
    score_hat,
    score_ref,
    score_refk,
    beta=1.0,
    tau=1.0
):
    score_loss = point_wise_proj_loss(score_hat, score_ref)

    kl_loss = kl_like_loss(score_hat, score_refk, tau=tau)

    loss = score_loss + beta * kl_loss

    return loss, score_loss, kl_loss


def forward_one_batch(
    decoder,
    encoder,          # frozen
    batch,
    beta=1.0,
    tau=1.0,
):
    """
    returns: loss, (score_loss, kl_loss)
    """

    x  = batch["x"]
    xo = batch["xo"]
    t  = batch["type"]
    q  = batch["quality"]

    # ---------- build graph ----------
    g1 = build_type_graph(t)
    g2 = build_type_graph(q)
    g  = jnp.stack([g1, g2], axis=0)

    # ---------- frozen representation ----------
    m = encoder(x, g)
    m = jax.lax.stop_gradient(m)

    # ---------- reference scores ----------
    score_ref, score_refk = compute_reference_scores(xo)

    # ---------- decoder score ----------
    score_hat = compute_decoder_score(m, decoder)

    # ---------- losses ----------
    loss, score_loss, kl_loss = compute_losses(
        score_hat,
        score_ref,
        score_refk,
        beta=beta,
        tau=tau
    )

    return loss, (score_loss, kl_loss)
@eqx.filter_jit
def train_step(
    decoder,
    encoder,
    optimizer,
    opt_state,
    batch,
    beta,
    tau,
):
    """
    updates decoder only
    """

    def loss_fn(decoder):
        loss, aux = forward_one_batch(
            decoder,
            encoder,
            batch,
            beta=beta,
            tau=tau,
        )
        return loss, aux

    (loss, (score_loss, kl_loss)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(decoder)

    updates, opt_state = optimizer.update(grads, opt_state)
    decoder = eqx.apply_updates(decoder, updates)

    return decoder, opt_state, loss, score_loss, kl_loss

def decoding_train_runner(
    encoder,        # frozen representation model
    decoder,
    data,
    optimizer,
    opt_state,
    batch_size=32,
    beta=1.0,
    tau=1.0,
    epochs=1,
    seed=0,
    log_every=10,
):

    # ---------- full dataset ----------
    x_all  = jnp.array(data["x"])
    xo_all = jnp.array(data["xo"])
    t_all  = jnp.array(data["type"])
    q_all  = jnp.array(data["quality"])

    key = jax.random.PRNGKey(seed)

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)

        total_loss = 0.0
        total_score = 0.0
        total_kl = 0.0
        n_batch = 0

        for i, batch in enumerate(
            make_batches(
                {
                    "x": x_all,
                    "xo": xo_all,
                    "type": t_all,
                    "quality": q_all,
                },
                batch_size,
                subkey,
            )
        ):
            decoder, opt_state, loss, score_loss, kl_loss = train_step(
                decoder=decoder,
                encoder=encoder,      # frozen
                optimizer=optimizer,
                opt_state=opt_state,
                batch=batch,
                beta=beta,
                tau=tau,
            )

            total_loss += loss
            total_score += score_loss
            total_kl += kl_loss
            n_batch += 1

            if i % log_every == 0:
                print(
                    f"[ep {epoch} | it {i}] "
                    f"loss={float(loss):.4f} "
                    f"score={float(score_loss):.4f} "
                    f"kl={float(kl_loss):.4f}"
                )

        print(
            f"[epoch {epoch}] "
            f"avg loss={float(total_loss / n_batch):.4f} | "
            f"score={float(total_score / n_batch):.4f} | "
            f"kl={float(total_kl / n_batch):.4f}"
        )

    return decoder, opt_state
