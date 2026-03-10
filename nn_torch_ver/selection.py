import torch
import jaxtyping
from  jaxtyping  import Array,Float,Int,Bool



def laplacian_kernel_matrix(X, Y=None, sigma=1.0, eps=1e-8):
    """
    X: (N, D)
    Y: (M, D) or None
    return: (N, M)
    """
    if Y is None:
        Y = X

    dist_l1 = torch.sum(
        torch.abs(X[:, None, :] - Y[None, :, :]),
        dim=-1
    )

    return torch.exp(-dist_l1 / (sigma + eps))

def cov_mu_reject(m:Float[Array,'n m']):
   lap_ker=laplacian_kernel_matrix(m)
   #l=torch.linalg.cholesky(lap_ker+torch.eye(lc.shape[0]))
   mu=m.mean(dim=-1)
   h=(torch.diag(mu**(-1/2))@lap_ker@torch.diag(mu**(-1/2)))
   dl=torch.linalg.cholesky(h+1e-3*torch.eye(h.shape[0]))
   hmu=torch.sqrt(mu)
   z=torch.randn(lap_ker.shape[0])
   x=mu+dl@z
   xs=x/torch.sqrt(mu)
   upb_c=xs>(hmu+dl@(torch.ones(dl.size(-1))*1.96)/x.size(0))
   lob_c=xs<(hmu-dl@(torch.ones(dl.size(-1))*1.96)/x.size(0))
   return ((upb_c.int()+lob_c.int() )>=1).int()

def build_interaction_matrix(m, score_hat, tau=1.0):
    """
    m:         (N, D)
    score_hat: (N, K)
    return:    (N, N) interaction / energy matrix
    """


    Km = (m @ m.T) / m.size(-1)          # (N, N)


    Ks = (score_hat[:, None, :] - score_hat[None, :, :]).pow(2).sum(-1)
    Ks = torch.exp(-Ks / tau)


    K = Km * Ks

    return K
def interaction_to_force(K, normalize=True):
    """
    K: (N, N)
    return: (N,)
    """
    f = K.sum(dim=-1)

    if normalize:
        f = f / (f.sum() + 1e-8)

    return f


def update_prior_gaussian(
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

    N = mu.size(0)

    # 1. interaction
    K = build_interaction_matrix(m, score_hat, tau=tau)  # (N, N)

    # 2. sample-wise force
    f = interaction_to_force(K)  # (N,)

    # 3. mean update（importance / fitness）
    mu_new = (1 - alpha) * mu + alpha * f

    # 4. covariance update
    Sigma_new = (1 - beta) * Sigma + beta * K

    # 5.
    Sigma_new = Sigma_new + eps * torch.eye(N, device=Sigma.device)

    return mu_new, Sigma_new
def lrt_ni_vs_n(mu, Sigma, eps=1e-8):
    """
    mu:    (N,)
    Sigma: (N, N)
    return:
        llr: (N,)  log-likelihood ratio for each sample
    """
    var_i = torch.diag(Sigma)                 # (N,)

    mu_g = mu.mean()                          # scalar
    var_g = var_i.mean()                      # scalar

    llr = (mu - mu_g).pow(2) / (var_g + eps) \
          - torch.log(var_i + eps)

    return llr


