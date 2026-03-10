from .data_process import build_type_graph
from .fitness import  compute_kernels,stat_kernel_featurewise
import torch
import torch.nn.functional as F


device='cuda' if torch.cuda.is_available() else 'cpu'

 #enocding trainer
def exp_runner(
    model,
    loader,
    optimizer,
    lambda1=0.5,
    lambda2=0.5,
    device=device,
):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].float().to(device)     # (B, D)
        xo = batch["xo"].float().to(device)   # (B, D+T)
        t =batch["type"].long().to(device) # Convert list of numbers to tensor
        q = batch["quality"].float().to(device)

        # build_type_graph expects a 1D tensor, so ensure `t` is correctly shaped if needed by build_type_graph
        # Given `t` is already (B,), build_type_graph(t) will return (B,B) adjacency.
        g1=build_type_graph(t).to(device)
        g2=build_type_graph(q).to(device)
        g=torch.stack([g1,g2],dim=0)

        # ---------- forward ----------
        m = model(x,g)
        #m=F.normalize(m, dim=1)
                    # (B, K)
        mim = m @ m.T                  # (B, B)


        # ---------- kernel ----------
        k = compute_kernels(xo)            # (B, B)

        # ---------- stat ----------
        stat = stat_kernel_featurewise(xo)
        print('stat_norm',stat.norm())
        print('k_norm',k.norm())
        print('mim_norm',mim.norm())
        print('mim_norm',mim.norm())

        # ---------- loss ----------
        loss_kernel = torch.pow(mim[None, :, :] - k, 2).mean()
        print('loss_kernel',loss_kernel.item())
        loss_stat = ((m - stat) ** 2).mean()
        print('loss_stat',loss_stat.item())

        loss = lambda1 * loss_kernel + lambda2 * loss_stat
        print('loss_per',loss.item()  )


        # ---------- backward ----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print('total',total_loss)

    return total_loss / len(loader)



# decoding trainer


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


def compute_reference_scores(xo):
    mus, covs = build_featurewise_sample_gaussians(xo)
    muk, covk = build_samplewise_gaussians(xo)

    log_p = gmm_log_prob(
        xo,
        mus,
        covs,
        weights=torch.ones(xo.size(-1), device=xo.device) / xo.size(-1)
    )
    log_pk = gmm_log_prob(
        xo.T,
        muk,
        covk,
        weights=torch.ones(xo.size(0), device=xo.device) / xo.size(0)
    )

    score_ref = torch.autograd.grad(
        log_p,
        xo,
        retain_graph=True,
        create_graph=False
    )[0]

    score_refk = torch.autograd.grad(
        log_pk,
        xo,
        retain_graph=False,
        create_graph=False
    )[0]

    return score_ref, score_refk

def compute_decoder_score(m, decoder):
    e = decoder(m)
    score_hat = torch.autograd.grad(
        e.sum(),
        m,
        retain_graph=True,
        create_graph=True
    )[0]
    return score_hat


def kl_like_loss(score_hat, score_refk, tau=1.0):
  """ score_hat: (N, D) score_refk: (N, D) """
    # pairwise squared distance


  kh = (score_hat[:, None, :] - score_hat[None, :, :]).pow(2).sum(-1)
  kr = (score_refk[:, None, :] - score_refk[None, :, :]).pow(2).sum(-1)

    # turn distance into energy -> log prob
  log_p_hat = F.log_softmax(-kh / tau, dim=-1)
  log_p_ref = F.log_softmax(-kr / tau, dim=-1)

  return (log_p_ref - log_p_hat).mean()
def point_wise_proj_loss(score_hat, score_ref):
    
   diff=score_hat[:,:,None]-score_ref[:,None,:]
   weights=torch.softmax(-diff.pow(2), dim=-1)
   proj_ref = (weights * score_ref[:, None, :]).sum(dim=-1)
   score_loss = (score_hat - proj_ref).pow(2).sum()
   return score_loss

def compute_losses(score_hat, score_ref, score_refk, beta=1.0, tau=1.0):
    #score_loss = F.mse_loss(score_ref.sum(dim=-1),score_hat.sum(dim=-1)
    #)
    #score_loss = (score_ref.sum(dim=-1) - score_hat.sum(dim=-1)).pow(2).sum()
    #score_loss=(score_hat[:,:,None]-score_ref[:,None,:]).pow(2).sum()
    #score_loss=(nn.Softmax(dim=-1)(score_hat[:,:,None]-score_ref[:,None,:]).pow(2)*(score_hat[:,:,None]-score_ref[:,None,:]).pow(2)).sum()
    score_loss=point_wise_proj_loss(score_hat, score_ref)

    kl_loss = kl_like_loss(score_hat, score_refk, tau=tau)

    loss = score_loss + beta * kl_loss

    return loss, score_loss, kl_loss


def forward_one_batch(
    batch,
    model,
    decoder,
    optimizer,
    device,
    beta=1.0,
    tau=1.0,
):
    x  = batch["x"].float().to(device)
    xo = batch["xo"].float().to(device).requires_grad_(True)
    t  = batch["type"].long().to(device)
    q  = batch["quality"].float().to(device)

    # graph
    g1 = build_type_graph(t).to(device)
    g2 = build_type_graph(q).to(device)
    g  = torch.stack([g1, g2], dim=0)

    # representation model (frozen)
    with torch.no_grad():
        m = model(x, g)
    m = m.detach().requires_grad_(True)

    # GMM ( GP-like thing)
    score_ref, score_refk = compute_reference_scores(xo)

    # NN score
    #score_hat = decoder(m)
    
    score_hat = compute_decoder_score(m, decoder)


    # losses
    loss, score_loss, kl_loss = compute_losses(
        score_hat,
        score_ref,
        score_refk,
        beta=beta,
        tau=tau
    )


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "score_loss": score_loss.item(),
        "kl_loss": kl_loss.item(),
    }
def decoding_train(
    loader,
    model,
    decoder,
    optimizer,
    device,
    epochs=1,
    beta=1.0,
    tau=1.0,
    log_every=10,
):
    model.eval()      # frozen
    decoder.train()   # train score field

    for ep in range(epochs):
        for i, batch in enumerate(loader):
            stats = forward_one_batch(
                batch=batch,
                model=model,
                decoder=decoder,
                optimizer=optimizer,
                device=device,
                beta=beta,
                tau=tau,
            )

            if i % log_every == 0:
                print(
                    f"[ep {ep} | it {i}] "
                    f"loss={stats['loss']:.4f} "
                    f"score={stats['score_loss']:.4f} "
                    f"kl={stats['kl_loss']:.4f}"

                )