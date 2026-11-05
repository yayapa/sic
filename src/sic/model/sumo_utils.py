"""Continuous-time SuMo / SuMo++ survival heads and losses.

SuMo (Rindt et al., AISTATS 2022 / MrHuff Sumo-Net)
    Mixed monotone-in-t net on ICL embeddings: free path for covariates,
    non-negative weights on the time path and its descendants. Models
    ``F = sigmoid(h)``, ``S = 1 - F``, density via autograd.

SuMo++ (Dittmer et al., arXiv:2307.13579)
    Cumulative hazard via MONDE+ with border fix::

        Lambda(t|x) = M+(t, x) - M+(0, x),    S(t|x) = exp(-Lambda(t|x))

Both heads consume ICL row embeddings (no separate covariate MLP) and share
the same time-normalization and ``(x, t, t_scale) -> (S, f)`` contract.

The ``combo_bce`` loss (NLL + weight * BCE) is an SIC addition for SuMo++, not
from either paper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# Per-task time norm for trunk + continuous heads. Default: p95 of TRAINING
# durations ("max" = training maximum, "minmax" = affine to ~[0, 1]).
DEFAULT_TSCALE_Q = 0.95
CONTINUOUS_SURV_TYPES = ("sumo", "sumo_pp")


def compute_t_scale(t_train: torch.Tensor, q=DEFAULT_TSCALE_Q, dim=None) -> torch.Tensor:
    """Per-task divide-scale from TRAINING durations (quantile or max).

    Any quantile (and the maximum) is positively homogeneous of degree one,
    ``s(a*t) = a*s(t)``, so ``t / s`` is exactly invariant to the unit of time.

    Parameters
    ----------
    t_train : Tensor
        Observed training durations of one task (any shape), or a batch of tasks
        when ``dim`` is given.
    q : float, numeric str, or "max"
        Quantile in (0, 1] of the training durations, or "max" for the maximum.
        Do not pass ``"minmax"`` here — use ``compute_time_norm``.
    dim : int, optional
        Reduce over this dimension only, giving one scale per task. When None
        (default) the tensor is flattened and a single scale is returned.

    Returns
    -------
    Tensor on ``t_train``'s device, clamped ``>= 1e-8`` (degenerate tasks).
    0-dim when ``dim`` is None.
    """
    if str(q) == "minmax":
        raise ValueError("compute_t_scale does not support 'minmax'; use compute_time_norm")
    t = t_train.float()
    if dim is None:
        t = t.flatten()
        scale = t.max() if str(q) == "max" else torch.quantile(t, float(q))
    else:
        scale = t.amax(dim=dim) if str(q) == "max" else torch.quantile(t, float(q), dim=dim)
    return scale.clamp_min(1e-8)


def compute_time_norm(t_train: torch.Tensor, mode=DEFAULT_TSCALE_Q, dim=None) -> dict[str, Any]:
    """Per-task time normalization from CONTEXT / training durations.

    Returns a dict consumed by ``normalize_time`` / ``time_norm_scale``:

    * ``{"kind": "div", "scale": s}`` for quantile or ``"max"``
    * ``{"kind": "minmax", "t_min": ..., "t_max": ...}`` for ``"minmax"``
    """
    mode_s = str(mode)
    if mode_s == "minmax":
        t = t_train.float()
        if dim is None:
            t = t.flatten()
            t_min, t_max = t.min(), t.max()
        else:
            t_min, t_max = t.amin(dim=dim), t.amax(dim=dim)
        # Degenerate context: fall back to a unit range so t_in stays finite.
        t_max = torch.maximum(t_max, t_min + 1e-8)
        return {"kind": "minmax", "t_min": t_min, "t_max": t_max}
    return {"kind": "div", "scale": compute_t_scale(t_train, q=mode, dim=dim)}


def _as_time_norm(t_scale_or_norm) -> dict[str, Any]:
    """Accept a legacy float/tensor scale or a ``compute_time_norm`` dict."""
    if isinstance(t_scale_or_norm, dict):
        return t_scale_or_norm
    return {"kind": "div", "scale": t_scale_or_norm}


def normalize_time(t: torch.Tensor, norm) -> torch.Tensor:
    """Map raw durations ``t`` to the head/embedding time coordinate."""
    norm = _as_time_norm(norm)
    if norm["kind"] == "minmax":
        return (t - norm["t_min"]) / (norm["t_max"] - norm["t_min"])
    return t / norm["scale"]


def time_norm_scale(norm) -> torch.Tensor | float:
    """Positive scale for jitter / sigma (``s`` or ``t_max - t_min``)."""
    norm = _as_time_norm(norm)
    if norm["kind"] == "minmax":
        return (norm["t_max"] - norm["t_min"]).clamp_min(1e-8)
    scale = norm["scale"]
    if torch.is_tensor(scale):
        return scale.clamp_min(1e-8)
    return max(float(scale), 1e-8)


def embed_time_feature(time: torch.Tensor, norm) -> torch.Tensor:
    """Scalar time feature for ``Time_to_event_Embedding`` (shape = ``time``).

    * div modes: ``log(t / s + eps)`` (scale-equivariant, spreads deciles)
    * minmax: affine ``t_in`` in ~[0, 1] (no log — near-zero would explode)
    """
    norm = _as_time_norm(norm)
    if norm["kind"] == "minmax":
        return normalize_time(time, norm)
    return torch.log(normalize_time(time, norm) + 1e-6)


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


class MixedFirstLayer(nn.Module):
    """MrHuff first mixed layer: positive path for ``t``, free path for ``x``.

    Output halves are concatenated then ``tanh``'d (``mixed_layer_2`` in Sumo-Net).
    """

    def __init__(self, d_x: int, d_out: int, d_t: int = 1):
        super().__init__()
        if d_out % 2 != 0:
            raise ValueError(f"MixedFirstLayer d_out must be even, got {d_out}")
        d_half = d_out // 2
        self.pos = PositiveLinear(d_t, d_half, bias=True)
        self.free = nn.Linear(d_x, d_half)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.cat([self.pos(t), self.free(x)], dim=1))


class SuMoHead(nn.Module):
    """Original SuMo-net mixed monotone head (Rindt et al. / MrHuff), no ``h_cov``.

    ICL embeddings replace the covariate net. Time enters with non-negative
    weights; later layers are fully positive so all descendants of ``t`` stay
    monotone. ``F = sigmoid(h)``, ``S = 1 - F``.
    """

    def __init__(self, d_model: int, layers_surv=None):
        super().__init__()
        if layers_surv is None:
            layers_surv = [256, 128, 64]  # capacity-matched to SuMoPP
        dims = list(layers_surv)
        self.first = MixedFirstLayer(d_model, dims[0])
        self.hidden = nn.ModuleList(
            [PositiveLinear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
        )
        self.out = PositiveLinear(dims[-1], 1, bias=True)
        # Sumo-Net init: v ~ N(0, 1/d_in), w = v**2 -> positive row-sums ~1 at any width.
        # PositiveLinear's |xavier| default sums to ~10-16 here and saturates the last tanh.
        with torch.no_grad():
            for lin in [self.first.pos, *self.hidden, self.out]:
                lin.log_weight.normal_(0.0, lin.in_features ** -0.5)
                lin.bias.normal_(0.0, lin.out_features ** -0.5)

    def _h(self, x_emb: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        z = self.first(x_emb, t_in)
        for layer in self.hidden:
            z = torch.tanh(layer(z))
        return self.out(z)

    def forward(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0, gradient: bool = False):
        """Return ``(S, dF/dt_in)``.

        ``horizon`` is normalized before the net (minmax / p95 / max). Density for
        NLL is ``∂F/∂t_in`` (MrHuff Sumo-Net convention: likelihood in scaled time),
        not ``∂F/∂t_raw``. SuMo++ uses the same convention.
        """
        # Leaf = scaled time (like Sumo-Net's MinMaxScaler'd y), not raw horizon.
        t_in = normalize_time(horizon.detach(), t_scale).unsqueeze(1)
        if gradient:
            t_in = t_in.requires_grad_(True)
        h = self._h(x_emb, t_in)
        F = torch.sigmoid(h)
        survival = 1.0 - F
        intensity = grad(F.sum(), t_in, create_graph=True)[0] if gradient else None
        return survival, intensity

    @torch.no_grad()
    def predict_survival(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0) -> torch.Tensor:
        return self.forward(x_emb, horizon, t_scale=t_scale, gradient=False)[0]

    def log_survival(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0) -> torch.Tensor:
        """``log S = log sigmoid(-h)`` (S = 1 - sigmoid(h)), stable in both tails."""
        t_in = normalize_time(horizon.detach(), t_scale).unsqueeze(1)
        return F.logsigmoid(-self._h(x_emb, t_in))


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
        """Return ``(S, dF/dt_in)`` for the query rows.

        Parameters
        ----------
        x_emb : (N, d_model)   ICL row embeddings used as covariates ``q``.
        horizon : (N,)         evaluation times ``t``.
        t_scale : float/tensor or TimeNorm dict. ``horizon`` is normalized BEFORE the
            positive net. Density is differentiated w.r.t. the normalized time
            ``t_in`` (same convention as SuMo), so the NLL is scale-invariant.
        gradient : if True, also return the density ``dF/dt_in`` via autograd.
        """
        t = horizon.clone().detach().requires_grad_(gradient)
        t_in = normalize_time(t, t_scale).unsqueeze(1)
        if gradient:
            t_in = t_in.requires_grad_(True)
        # Border at original t=0 (works for div and minmax; for div, t_in(0)=0).
        t0_in = normalize_time(torch.zeros_like(t), t_scale).unsqueeze(1)
        Lam = self.mplus(t_in, x_emb) - self.mplus(t0_in, x_emb)  # >= 0 when t>=0 for div
        survival = torch.exp(-Lam)
        F = 1.0 - survival
        intensity = grad(F.sum(), t_in, create_graph=True)[0] if gradient else None
        return survival, intensity

    @torch.no_grad()
    def predict_survival(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0) -> torch.Tensor:
        return self.forward(x_emb, horizon, t_scale=t_scale, gradient=False)[0]

    def log_survival(self, x_emb: torch.Tensor, horizon: torch.Tensor, t_scale=1.0) -> torch.Tensor:
        """``log S = -Lambda`` exactly (no exp round trip)."""
        t = horizon.detach()
        t_in = normalize_time(t, t_scale).unsqueeze(1)
        t0_in = normalize_time(torch.zeros_like(t), t_scale).unsqueeze(1)
        return -(self.mplus(t_in, x_emb) - self.mplus(t0_in, x_emb))


def nll_loss(model, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor, t_scale: float = 1.0, eps: float = 1e-8):
    """Right-censored continuous negative log-likelihood.

    ``-[ sum_{censored} log S(T) + sum_{event} log f(T) ] / N``, with density
    ``f = ∂F/∂t_in`` in normalized time from the head (SuMo and SuMo++).
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

    with ``z ~ |N(0, sigma^2)|``, ``sigma = sigma_frac * scale`` where ``scale``
    is ``t_scale`` (div) or ``t_max - t_min`` (minmax). Uses only ``S``.
    """
    t = t.reshape(-1)
    e = e.reshape(-1)
    n = t.shape[0]
    sigma = sigma_frac * time_norm_scale(t_scale)
    z = torch.randn(n, device=t.device).abs() * sigma
    coin = torch.rand(n, device=t.device)
    is_event = e != 0
    post = is_event & (coin >= 0.5)                              # postmortem -> "dead"
    t_event_pre = (t - z).clamp_min(0.0)                         # event premortem time
    t_prime = torch.where(post, t + z, torch.where(is_event, t_event_pre, t))
    S = model.forward(x, t_prime, t_scale=t_scale, gradient=False)[0].reshape(-1).clamp(eps, 1.0 - eps)
    loss = torch.where(post, -torch.log(1.0 - S), -torch.log(S))
    return loss.mean()


def true_targets(t_true: torch.Tensor, tau=float("inf")) -> tuple[torch.Tensor, torch.Tensor]:
    """``(t, e)`` loss targets from TRUE (uncensored) times, censored at the horizon ``tau``.

    ``t = min(t_true, tau)``, ``e = 1[t_true <= tau]``. With ``tau = inf`` (no cap),
    non-finite ``t_true`` (cure / overflow) becomes censored at the largest finite value.
    """
    if not np.isfinite(float(tau)):
        finite = torch.isfinite(t_true)
        tau = t_true[finite].max() if finite.any() else torch.ones((), device=t_true.device, dtype=t_true.dtype)
    tau = torch.as_tensor(tau, device=t_true.device, dtype=t_true.dtype)
    e = t_true <= tau
    return torch.minimum(t_true, tau), e.to(t_true.dtype)


def _log1mexp(d: torch.Tensor) -> torch.Tensor:
    """``log(1 - e^d)`` for ``d < 0`` (Maechler 2012: expm1 near 0, log1p in the tail)."""
    return torch.where(d > -0.6931471805599453, torch.log(-torch.expm1(d)), torch.log1p(-torch.exp(d)))


def curve_ce_loss(model, x: torch.Tensor, t_grid: torch.Tensor, S_true: torch.Tensor, t_scale=1.0):
    """Binned cross-entropy against the true survival curve (report §6, oracle target).

    ``t_grid`` (G,) starts at 0; ``S_true`` (N, G) is the true S at the grid. Bin masses
    ``p_k = S_{k-1} - S_k`` plus the tail ``S_{G-1}`` beyond the last knot. The model's bin
    log-probabilities stay in log space: ``log(S_a - S_b) = l_a + log1mexp(l_b - l_a)``.
    Minimised by the posterior mean of the bin probabilities, i.e. E[S_T(t) | D] at the knots.
    """
    N, G = S_true.shape
    t = t_grid.unsqueeze(0).expand(N, G).reshape(-1)
    x_rep = x.unsqueeze(1).expand(N, G, x.shape[-1]).reshape(N * G, -1)
    log_s = model.log_survival(x_rep, t, t_scale=t_scale).reshape(N, G).float()
    # Equal consecutive log S (flat model curve, float32 ties) -> large finite penalty, not -inf.
    d = (log_s[:, 1:] - log_s[:, :-1]).clamp(max=-1e-7)
    log_bin = log_s[:, :-1] + _log1mexp(d)
    p = (S_true[:, :-1] - S_true[:, 1:]).clamp_min(0.0)
    ll = (p * log_bin).sum(1) + S_true[:, -1] * log_s[:, -1]
    return -ll.mean()


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
                      head: nn.Module, t_scale: float = 1.0,
                      chunk: int | None = 1_000_000) -> torch.Tensor:
    """Survival ``S(t|x)`` on an arbitrary time grid.

    Parameters
    ----------
    x_emb : (N, d_model)
    time_grid : (M,)
    head : SuMo or SuMo++ head (``predict_survival`` contract).
    t_scale : float/tensor scale or TimeNorm dict (must match training).
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


def _self_check():
    """Minimal runnable check for SuMoHead (monotone S, finite NLL, scale-invariant NLL)."""
    torch.manual_seed(0)
    N, d = 8, 32
    head = SuMoHead(d_model=d, layers_surv=[16, 8])
    x = torch.randn(N, d)
    t = torch.rand(N) * 2 + 0.1
    e = torch.randint(0, 2, (N,))
    norm = compute_time_norm(t, mode=0.95)
    S0, _ = head.forward(x, torch.zeros(N), t_scale=norm, gradient=False)
    S1, _ = head.forward(x, t, t_scale=norm, gradient=False)
    assert (S0 > 0).all() and (S0 < 1).all()
    assert (S1 <= S0 + 1e-5).all(), "S should be nonincreasing in t"
    grid = torch.linspace(0.1, 2.0, 5)
    Sg = predict_surv_sumo(x[:2], grid, head, t_scale=norm)
    assert Sg.shape == (2, 5)
    assert (Sg[:, 1:] <= Sg[:, :-1] + 1e-5).all()
    loss = nll_loss(head, x, t, e, t_scale=norm)
    assert torch.isfinite(loss), loss
    # minmax path
    nm = compute_time_norm(t, mode="minmax")
    assert nm["kind"] == "minmax"
    loss_mm = nll_loss(head, x, t, e, t_scale=nm)
    assert torch.isfinite(loss_mm), loss_mm
    # Scaled-time NLL: pure rescaling of durations must not change loss (MrHuff).
    a = 1000.0
    t2 = t * a
    nm2 = compute_time_norm(t2, mode="minmax")
    loss_mm2 = nll_loss(head, x, t2, e, t_scale=nm2)
    assert torch.allclose(loss_mm, loss_mm2, rtol=1e-4, atol=1e-5), (
        float(loss_mm), float(loss_mm2)
    )
    # SuMo++: density shape matches S, NLL scale-invariant under the div (p95) norm.
    pp = SuMoHeadPlusPlus(d_model=d, layers_surv=[16, 8])
    S_pp, f_pp = pp.forward(x, t, t_scale=norm, gradient=True)
    assert f_pp.shape == S_pp.shape, (f_pp.shape, S_pp.shape)
    loss_pp = nll_loss(pp, x, t, e, t_scale=norm)
    loss_pp2 = nll_loss(pp, x, t2, e, t_scale=compute_time_norm(t2, mode=0.95))
    assert torch.allclose(loss_pp, loss_pp2, rtol=1e-4, atol=1e-5), (
        float(loss_pp), float(loss_pp2)
    )
    # True-time targets: capped at tau, and raw with non-finite -> censored at max finite.
    tt = torch.tensor([0.5, 1.0, 2.0, float("inf")])
    tc, ec = true_targets(tt, tau=1.0)
    assert torch.equal(tc, torch.tensor([0.5, 1.0, 1.0, 1.0])) and torch.equal(ec, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    tr, er = true_targets(tt)
    assert torch.equal(tr, torch.tensor([0.5, 1.0, 2.0, 2.0])) and torch.equal(er, torch.tensor([1.0, 1.0, 1.0, 0.0]))
    loss_true = nll_loss(head, x[:4], tr, er, t_scale=norm)
    assert torch.isfinite(loss_true), loss_true
    print("sumo_utils self-check OK", float(loss.detach()), float(loss_mm.detach()))


if __name__ == "__main__":
    _self_check()
