from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import numpy as np


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

        # ---------- simple stat ( 的 stat kernel) ----------
        mu = xo.mean(axis=1, keepdims=True)
        std = xo.std(axis=1, keepdims=True) + 1e-6
        stat = np.concatenate([mu, std], axis=1)

        # ---------- final fitness target ----------
        m_target = np.concatenate([Z, stat], axis=1)
        m_target = self.scaler.fit_transform(m_target)

        return m_target
class KRR_Encoder:
    """
    model(x) -> m_hat
    """
    def __init__(self, gamma=1.0, alpha=1e-2):
        self.encoder = MultiOutputRegressor(
            KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
        )

    def fit(self, x, m_target):
        self.encoder.fit(x, m_target)

    def transform(self, x):
        return self.encoder.predict(x)

    def fit_transform(self, x, m_target):
        self.fit(x, m_target)
        return self.transform(x)
    
    


class DistEncoderGMM:
    """
    xo -> dist_score_target (N, 2)
    """
    def __init__(
        self,
        n_components_sample=4,
        n_components_feature=4,
        cov_type="full",
        random_state=0,
    ):
        self.gmm_sample = GaussianMixture(
            n_components=n_components_sample,
            covariance_type=cov_type,
            random_state=random_state,
             reg_covar=1e-3, 
        )
        self.gmm_feature = GaussianMixture(
            n_components=n_components_feature,
            covariance_type=cov_type,
            random_state=random_state,
             reg_covar=1e-3, 
        )
        self.scaler = StandardScaler()

    def fit_transform(self, xo):
        """
        xo: (N, D)
        return: (N, 2)
        """
        N, D = xo.shape

        # ---- sample-wise GMM ----
        self.gmm_sample.fit(xo)
        logp_sample = self.gmm_sample.score_samples(xo)  # (N,)

        # ---- feature-wise GMM (NN-aligned) ----
        xo_T = xo.T                # (D, N)
        self.gmm_feature.fit(xo_T)
        logp_f = self.gmm_feature.score_samples(xo_T)  # (D,)

        # broadcast feature score back to samples
        logp_feature = np.full(N, logp_f.mean())

        # ---- concat & normalize ----
        score = np.stack([logp_sample, logp_feature], axis=1)
        score = self.scaler.fit_transform(score)

        return score