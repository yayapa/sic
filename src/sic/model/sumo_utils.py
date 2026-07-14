"""SuMo++ survival head (MONDE+) and its continuous-time losses.

The head models the cumulative hazard with a network ``M+`` that is
monotone-increasing in time by construction and unconstrained in the
covariates::

    Lambda(t|x) = M+(t, x) - M+(0, x),    S(t|x) = exp(-Lambda(t|x))

Subtracting the value at ``t = 0`` makes ``S(0) = 1`` hold exactly, and the
monotone construction makes ``S`` non-increasing in ``t``. The head consumes the
ICL row embeddings as ``x`` (the transformer already learns the representation),
so it is a drop-in survival head for the SIC backbone.

Reference
---------
S. Dittmer et al. "Reinterpreting Survival Analysis in the Universal
Approximator Age." arXiv:2307.13579 (2023).

The head (MONDE+ block, border fix, cumulative-hazard form) is from that paper.
The ``combo_bce`` loss below (NLL + weight * BCE) is an addition in this work,
not from the paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

# Per-task time scale for the head: it sees ``t / t_scale``, where ``t_scale`` is
# this quantile of the TRAINING durations ("max" = training maximum). Without
# this normalization the positive networks saturate on raw durations (e.g. days).
DEFAULT_TSCALE_Q = 0.95


def compute_t_scale(t_train: torch.Tensor, q=DEFAULT_TSCALE_Q) -> torch.Tensor:
    """Per-task time-normalization scale from the TRAINING durations.

    Parameters
    ----------
    t_train : Tensor
        Observed training durations of one task (any shape).
    q : float, numeric str, or "max"
        Quantile in (0, 1] of the training durations, or "max" for the maximum.

    Returns
    -------
    0-dim Tensor on ``t_train``'s device, clamped ``>= 1e-8`` (degenerate tasks).
    """
    t = t_train.float().flatten()
    scale = t.max() if str(q) == "max" else torch.quantile(t, float(q))
    return scale.clamp_min(1e-8)


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights (weight = ``log_weight ** 2``)."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight ** 2, self.bias)


class MondePlusLayer(nn.Module):
    """One MONDE+ layer (SuMo++, eq. 17-18)::

        z_out = H(z) + sigma( A( phi(a*t + b) (.) psi(G(z)) ) + B(z) + L(z0) )

    where ``(.)`` is the Hadamard product. ``H, A, B, G`` use non-negative weights
    (``PositiveLinear``) and ``a = a_raw**2 >= 0``; together with the monotone
    ``phi = softplus`` this makes the layer monotone-increasing in ``t``. ``L`` is a
    FREE ``Linear`` input-skip, leaving the layer unconstrained in ``z0`` (the
    covariates). ``sigma = tanh``, or identity for the last layer.
    """

    def __init__(self, d_in: int, d_out: int, d_z0: int, d_inter: int, last: bool = False):
        super().__init__()
        self.H = PositiveLinear(d_in, d_out, bias=True)      # residual / projection (non-neg)
        self.B = PositiveLinear(d_in, d_out, bias=True)      # hidden -> out (non-neg)
        self.G = PositiveLinear(d_in, d_inter, bias=True)    # hidden -> interaction (non-neg)
        self.A = PositiveLinear(d_inter, d_out, bias=False)  # interaction -> out (non-neg; no bias, paper B.3)
        self.L = nn.Linear(d_z0, d_out)                      # input skip (FREE -> free in x)
        self.a_raw = nn.Parameter(torch.empty(d_inter))
        self.b = nn.Parameter(torch.zeros(d_inter))
        self.last = last
        self._paper_init()

    def _paper_init(self):
        # SuMo++ appendix B.3: scale the non-negative affine weights down (factor
        # 0.2) so the cumulative hazard doesn't blow up across layers; a ~ U(0,1), b = 0.
        with torch.no_grad():
            for lin in (self.H, self.B, self.G, self.A):
                lin.log_weight.mul_(0.2 ** 0.5)                   # effective weight (log_weight**2) x 0.2
            self.a_raw.copy_(torch.rand_like(self.a_raw).sqrt())  # a = a_raw**2 ~ U(0,1)
            self.b.zero_()

    def forward(self, t, z, z0):
        a = self.a_raw ** 2                                       # (d_inter,) >= 0
        phi = nn.functional.softplus(a * t + self.b)             # (N, d_inter), increasing in t
        psi = nn.functional.softplus(self.G(z))                  # (N, d_inter), >= 0
        z_tilde = self.A(phi * psi)                              # (N, d_out)
        pre = z_tilde + self.B(z) + self.L(z0)
        act = pre if self.last else torch.tanh(pre)
        return self.H(z) + act


class MondePlus(nn.Module):
    """MONDE+ network ``M+(t, q)``: monotone-increasing in ``t``, free in ``q``.

    The last layer uses ``sigma = identity`` so ``M+`` is unbounded in ``t`` (hence
    ``S`` can reach 0).
    """

    def __init__(self, d_model: int, hidden_dims, d_inter=None):
        super().__init__()
        d_inter = d_inter or hidden_dims[0]
        dims = [d_model] + list(hidden_dims)
        self.layers = nn.ModuleList([
            MondePlusLayer(dims[k], dims[k + 1], d_z0=d_model, d_inter=d_inter)
            for k in range(len(hidden_dims))
        ])
        self.out = MondePlusLayer(hidden_dims[-1], 1, d_z0=d_model, d_inter=d_inter, last=True)

    def forward(self, t, q):
        z = q
        for layer in self.layers:
            z = layer(t, z, q)
        return self.out(t, z, q)


class SuMoHeadPlusPlus(nn.Module):
    """SuMo++ : border-fixed survival head with the MONDE+ block (eq. 17-21).

    ``Lambda(t|q) = M+(t, q) - M+(0, q);  S = exp(-Lambda),  q = x_emb``.
    Guarantees ``S(0) = 1`` exactly, ``S`` monotone non-increasing in ``t``, and
    ``S -> 0`` as ``t`` grows.
    """

    def __init__(self, d_model: int, layers_surv=None):
        super().__init__()
        if layers_surv is None:
            layers_surv = [256, 128, 64]  # architecture of the released SuMoPP checkpoint
        self.mplus = MondePlus(d_model, hidden_dims=list(layers_surv))

    def forward(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0, gradient: bool = False):
        """Return ``(S, dF/dt)`` for the query rows.

        Parameters
        ----------
        x_emb : (N, d_model)   ICL row embeddings used as covariates ``q``.
        horizon : (N,)         evaluation times ``t``.
        t_scale : per-dataset time scale. ``horizon`` is divided by this BEFORE the
            positive net, so the head always sees ~[0, 1] time and does not
            saturate. The density is still differentiated w.r.t. the ORIGINAL
            ``t`` (chain rule supplies the ``1/t_scale`` factor), so it stays in
            original time units.
        gradient : if True, also return the density ``dF/dt`` via autograd.
        """
        t = horizon.clone().detach().requires_grad_(gradient)
        t_in = (t / t_scale).unsqueeze(1)   # normalized time
        Lam = self.mplus(t_in, x_emb) - self.mplus(torch.zeros_like(t_in), x_emb)  # >= 0, = 0 at t=0
        survival = torch.exp(-Lam)
        F = 1.0 - survival
        intensity = grad(F.sum(), t, create_graph=True)[0].unsqueeze(1) if gradient else None
        return survival, intensity

    @torch.no_grad()
    def predict_survival(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0) -> torch.Tensor:
        return self.forward(x_emb, horizon, t_scale=t_scale, gradient=False)[0]


def nll_loss(model, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor, t_scale: float = 1.0, eps: float = 1e-8):
    """Right-censored continuous negative log-likelihood.

    ``-[ sum_{censored} log S(T) + sum_{event} log f(T) ] / N``, with the density
    ``f = dF/dt`` from the head's double-backward.
    """
    survival, intensity = model.forward(x, t, t_scale=t_scale, gradient=True)
    # out-of-place clamp: an in-place clamp_ corrupts autograd for S = exp(-Lambda)
    # because exp's backward needs its unmodified output. Behaviour-preserving here.
    survival = survival.clamp(eps)
    intensity = intensity.clamp(eps)

    error = torch.log(survival[e == 0]).sum()
    error += torch.log(intensity[e != 0]).sum()
    return -error / len(x)


def bce_loss(model, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor,
             t_scale=1.0, sigma_frac: float = 0.1, eps: float = 1e-7):
    """Pointwise BCE (SuMo++, eq. 9-11): "survival = infinitely many time-indexed
    classifiers", trained with BCE. Per sample we draw ONE time:

      - censored (e=0):     t' = T                -> "alive": -log S(t'|x)
      - event (e=1), 50%:   t' = clamp(T-|z|, 0)  (premortem)  -> "alive": -log S
      - event (e=1), 50%:   t' = T + |z|          (postmortem) -> "dead" : -log(1-S)

    with ``z ~ |N(0, sigma^2)|``, ``sigma = sigma_frac * t_scale``. Uses only ``S``
    (no density / double-backward).
    """
    t = t.reshape(-1)
    e = e.reshape(-1)
    n = t.shape[0]
    sigma = sigma_frac * t_scale
    z = torch.randn(n, device=t.device).abs() * sigma
    coin = torch.rand(n, device=t.device)
    is_event = e != 0
    post = is_event & (coin >= 0.5)                              # postmortem -> "dead"
    t_event_pre = (t - z).clamp_min(0.0)                         # event premortem time
    t_prime = torch.where(post, t + z, torch.where(is_event, t_event_pre, t))
    S = model.forward(x, t_prime, t_scale=t_scale, gradient=False)[0].reshape(-1).clamp(eps, 1.0 - eps)
    loss = torch.where(post, -torch.log(1.0 - S), -torch.log(S))
    return loss.mean()


def combo_bce_loss(model, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor,
                   t_scale=1.0, bce_weight: float = 1.0, sigma_frac: float = 0.1):
    """``combo_bce = NLL + bce_weight * BCE``.

    The right-censored NLL is a proper scoring rule; the pointwise BCE is not
    (Rindt et al. 2022, AISTATS), so propriety of the sum is not guaranteed.
    Empirically the NLL term favours calibration (IBS) while the BCE term
    favours ranking (C-index); the sum is robust on both. ``bce_weight = 0``
    recovers pure NLL.
    """
    return nll_loss(model, x, t, e, t_scale=t_scale) \
        + bce_weight * bce_loss(model, x, t, e, t_scale=t_scale, sigma_frac=sigma_frac)


@torch.no_grad()
def predict_surv_sumo(x_emb: torch.Tensor, time_grid: torch.Tensor,
                      head: SuMoHeadPlusPlus, t_scale: float = 1.0,
                      chunk: int | None = 1_000_000) -> torch.Tensor:
    """Survival ``S(t|x)`` on an arbitrary time grid.

    Parameters
    ----------
    x_emb : (N, d_model)
    time_grid : (M,)
    head : the SuMo++ head.
    t_scale : per-dataset time scale (must match the value used at training).
    chunk : evaluate subjects in blocks so at most ~``chunk`` (subject, time)
        pairs are materialized per head call; ``None`` evaluates the full
        ``(N * M, d_model)`` grid in one call. Chunking is exact, not an
        approximation: the head is a pure per-row function, so evaluating in
        blocks and concatenating equals the single call.

    Returns
    -------
    S : (N, M) survival evaluated at each time in ``time_grid``.
    """
    N, d = x_emb.shape
    M = time_grid.numel()
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=x_emb.device, dtype=x_emb.dtype)

    t_row = time_grid.to(device=x_emb.device, dtype=x_emb.dtype)
    bs = N if chunk is None else max(1, chunk // M)  # subjects per block
    S = torch.empty((N, M), device=x_emb.device, dtype=x_emb.dtype)
    for i in range(0, N, bs):
        xb = x_emb[i:i + bs]
        b = xb.shape[0]
        x_rep = xb[:, None, :].expand(b, M, d).reshape(b * M, d)
        t_rep = t_row[None, :].expand(b, M).reshape(b * M)
        S[i:i + bs] = head.predict_survival(x_rep, t_rep, t_scale=t_scale).reshape(b, M)
    return S
