
import torch

def build_featurewise_sample_gaussians(xo, eps=1e-5):
    # xo: (N, D)
    N, D = xo.shape

    mu=xo.mean(dim=0)
    mus=torch.ones_like(xo)
    mus=mus*mu[None,:]
    mus=mus.T                       # (D, N)

    X = xo - xo.mean(dim=0, keepdim=True)  # (N, D)
    X = X.T                               # (D, N)

    covs = X[:, :, None] * X[:, None, :]   # (D, N, N)
    covs = covs / (N - 1)

    eye = torch.eye(N, device=xo.device)
    covs = covs + eps * eye[None]

    return mus, covs
def build_samplewise_gaussians(xo, eps=1e-5):
    # xo: (N, D)
    N, D = xo.shape

    # sample-wise means
    mus = xo                    # (N, D)

    # global covariance (shared)
    X = xo - xo.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / (N - 1)   # (D, D)
    cov = cov + eps * torch.eye(D, device=xo.device)

    covs = cov.unsqueeze(0).expand(N, D, D)  # (N, D, D)

    return mus, covs


def gmm_log_prob(x, mus, covs,weights):
    log_probs=[]
    n,m=mus.size()
    for k in range(n):
      cov_=covs[k,:,:]+1e-6*torch.eye(m,device=x.device)
      dist=torch.distributions.MultivariateNormal(mus[k,:],cov_)
      log_probs.append(torch.log(weights[k])+dist.log_prob(x[:,k]))
    return torch.logsumexp(torch.stack(log_probs),dim=0)


