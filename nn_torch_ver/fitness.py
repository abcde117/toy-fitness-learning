import torch
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def safe_trace_normalize(K, eps=1e-6):
    tr = torch.trace(K)
    tr = torch.clamp(tr, min=eps)
    K = K / tr
    K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    return K

def log_rbf_kernel(X, eps=1e-6):
    D = torch.cdist(X, X, p=2).pow(2)

    scale = D.median()
    scale = torch.clamp(scale, min=eps)

    D = D / scale
    K = 1.0 / (1.0 + D)

    return safe_trace_normalize(K, eps)
def cosine_kernel(X, eps=1e-8):
    norm = X.norm(dim=1, keepdim=True)
    norm = torch.clamp(norm, min=eps)

    Xn = X / norm
    K = Xn @ Xn.T

    return safe_trace_normalize(K, eps)

def covariance_kernel(X, eps=1e-6):
    Xc = X - X.mean(dim=0, keepdim=True)
    d = Xc.shape[1]

    K = Xc @ Xc.T / max(d, 1)

    return safe_trace_normalize(K, eps)
def compute_kernels(X):
    Ks = [
        log_rbf_kernel(X),
        cosine_kernel(X),
        covariance_kernel(X),
    ]
    K = torch.stack(Ks, dim=0)
    return torch.nan_to_num(K, nan=0.0)



def stat_kernel_featurewise(
    X,
    eps=1e-6,
):
    """
    X: (D, K)
       for each feature d, we compute statistics over K
    return:
        stat: (D, 3)  -> [t, chi, fisher] per feature
    """

    D, K = X.shape

    # -------- 1. mean / std over K --------
    mu = X.mean(dim=-1)                          # (D,)
    std = X.std(dim=-1, unbiased=False) + eps    # (D,)

    # -------- 2. t-stat --------
    t_stat = mu / std                            # (D,)

    # -------- 3. chi-square-like --------
    chi_stat = ((X - mu[:, None]) ** 2).mean(dim=-1) / (std ** 2 + eps)

    # -------- 4. fisher-like (pseudo split over K) --------
    mask = X > mu[:, None]                       # (D, K)
    mask_sum = mask.sum(dim=-1).clamp(min=1)

    mu1 = (X * mask).sum(dim=-1) / mask_sum
    mu2 = (X * (~mask)).sum(dim=-1) / (K - mask_sum + eps)

    var1 = ((X - mu1[:, None]) ** 2 * mask).sum(dim=-1) / mask_sum
    var2 = ((X - mu2[:, None]) ** 2 * (~mask)).sum(dim=-1) / (K - mask_sum + eps)

    fisher_stat = (mu1 - mu2) ** 2 / (var1 + var2 + eps)

    # -------- 5. stack per feature --------
    stat = torch.stack(
        [t_stat, chi_stat, fisher_stat],
        dim=1        # (D, 3)
    )

    return stat




def build_similarity_graph(xo, eps=1e-8):
    G = cosine_similarity(xo)
    G = G / (np.trace(G) + eps)
    return G
class FitnessEncoder:
    """
    xo -> m_target
    """
    def __init__(self, embed_dim=8, random_state=0):
        self.embed_dim = embed_dim
        self.random_state = random_state
        self.scaler = StandardScaler()

    def fit_transform(self, xo, quality, type_numeric):
        """
        xo: (N, T)    one-hot type
        quality: (N,)
        type_numeric: (N,)
        """

        # ---------- graph ----------
        G = build_similarity_graph(xo)

        # ---------- quality-conditioned graph ----------
        Gq = G * np.exp(
            -np.abs(quality[:, None] - quality[None, :])
        )

        # ---------- spectral embedding ----------
        spec = SpectralEmbedding(
            n_components=self.embed_dim,
            affinity="precomputed",
            random_state=self.random_state,
        )
        Z = spec.fit_transform(Gq)

        # ---------- simple stat (替代你 NN 的 stat kernel) ----------
        mu = xo.mean(axis=1, keepdims=True)
        std = xo.std(axis=1, keepdims=True) + 1e-6
        stat = np.concatenate([mu, std], axis=1)

        # ---------- final fitness target ----------
        m_target = np.concatenate([Z, stat], axis=1)
        m_target = self.scaler.fit_transform(m_target)

        return m_target