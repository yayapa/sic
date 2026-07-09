"""
Discrete-grid survival sampler (SUMO-style) that mirrors Reg2Surv's forward contract.

Given y of shape (N, m) (the SCM prior outputs), each column is treated as an
independent risk-score channel eta_i.  A positive monotone network maps
(eta_1, ..., eta_m, t) -> survival on a discrete time grid; event and censoring
times are obtained by inverse-CDF exponential interpolation in each grid bin.

Hyper-parameters are read from hp["reg2surv_discrete_param"] (or directly from
hp if that key is absent):

    n_grid                 : int   number of grid points in [0, 1]

    alpha_bounds           : (scale, shift), alpha = Beta(1.1, 1.1) * scale + shift
    gamma_k_bounds         : (lo, hi), uniform
    mu_t_bounds            : (lo, hi), uniform
    lambda_eta_bounds      : (lo, hi), uniform per eta channel
    n_event_layers_bounds  : (lo, hi), uniform int, inclusive
    event_layer_bounds     : (lo, hi), uniform int, inclusive
    n_cens_layers_bounds   : (lo, hi), uniform int, inclusive
    cens_layer_bounds      : (lo, hi), uniform int, inclusive
    t_max_bounds           : (lo, hi), uniform
    censoring_rate_bounds  : (lo, hi), uniform

Optional independent bounds for the censoring net (fall back to the event-net
bounds when omitted):

    alpha_cens_bounds, gamma_k_cens_bounds, mu_t_cens_bounds, lambda_eta_cens_bounds

Rejection / resampling (optional):

    event_net_max_tries         : int   max number of resample attempts
    event_net_min_median_delta  : float median |f(t_max)-f(0)| threshold
    event_net_max_flat_frac     : float max fraction of subjects with |delta| < delta_eps
    event_net_delta_eps         : float tolerance for "flat" delta
    event_net_check_n           : int   number of eta samples used for the check
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, Tensor

EPS = 1e-6


# ---------------------------------------------------------------------------
# Positive monotone network building blocks
# ---------------------------------------------------------------------------
class PositiveLinear(nn.Module):
    """Linear layer with raw (post-sampled) positive weights (no softplus)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y


def build_positive_network(input_dim: int, hidden_layers: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_layers:
        layers += [PositiveLinear(prev, h), nn.Tanh()]
        prev = h
    layers += [PositiveLinear(prev, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Weight / bias initialisation
# ---------------------------------------------------------------------------
@torch.no_grad()
def random_init(
    network: nn.Sequential,
    input_means: Tensor,              # (fan_in_first,)
    alpha: float,
    gamma_k: float,
    lambda_eta: Tensor,               # (n_etas,)
    np_rng: np.random.Generator,
) -> None:
    """Xavier-style positive-Gamma initialisation with eta-column rescaling.

    First layer:
        - weights W ~ Gamma(k, theta) with theta chosen s.t. E[W^2] = alpha^2 / fan_in
        - eta columns post-scaled by lambda_i / sqrt(n_etas) so the total eta
          variance cancels the n_etas multiplicity factor.
        - bias centred at -(W @ input_means) so per-sample pre-activation mean is 0.

    Deeper layers:
        - same Gamma scaling, bias = 0.
    """
    is_first = True
    lam_cpu = lambda_eta.detach().cpu()

    for m in network.modules():
        if not isinstance(m, PositiveLinear):
            continue

        fan_in = m.in_features
        k = float(gamma_k)
        theta = float(alpha) / math.sqrt(max(fan_in, 1) * k * (1.0 + k))

        # Reproducible Gamma draw through the numpy RNG
        w_np = np_rng.gamma(shape=k, scale=theta, size=tuple(m.weight.shape))
        w = torch.tensor(w_np, device=m.weight.device, dtype=m.weight.dtype)
        m.weight.data.copy_(w)

        if is_first:
            n_etas = fan_in - 1
            if n_etas > 0:
                lam = lam_cpu.to(device=w.device, dtype=w.dtype)
                if lam.numel() == 1:
                    lam = lam.expand(n_etas)
                elif lam.numel() != n_etas:
                    raise ValueError(
                        f"lambda_eta length {lam.numel()} != n_etas {n_etas}"
                    )
                scale = lam / math.sqrt(max(n_etas, 1))
                m.weight.data[:, :n_etas] *= scale  # broadcasts per column

        if m.bias is not None:
            if is_first:
                im = input_means.to(device=m.weight.device, dtype=m.weight.dtype)
                m.bias.data.copy_(-(m.weight.data @ im))
            else:
                m.bias.data.zero_()

        is_first = False


# ---------------------------------------------------------------------------
# Survival curve evaluation on a time grid
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_survival(net: nn.Sequential, eta: Tensor, t_grid: Tensor) -> Tensor:
    """S(t|eta) for an m-eta network.  eta: (N,) or (N, m).  Returns (N, G)."""
    if eta.ndim == 1:
        eta = eta.unsqueeze(1)
    N, M = eta.shape
    G = t_grid.shape[0]
    eta_exp = eta.unsqueeze(1).expand(N, G, M).reshape(-1, M)
    t_exp = t_grid.view(1, G, 1).expand(N, G, 1).reshape(-1, 1)
    inp = torch.cat([eta_exp, t_exp], dim=1)
    f = net(inp).reshape(N, G)
    f0, fG = f[:, 0:1], f[:, -1:]
    S = ((fG - f) / (fG - f0 + EPS)).clamp(0.0, 1.0)
    return S


@torch.no_grad()
def compute_survival_1d(net: nn.Sequential, t_grid: Tensor) -> Tensor:
    """S(t) for a pure time net (fan_in = 1).  Returns (G,)."""
    f = net(t_grid.unsqueeze(1)).squeeze(1)
    return ((f[-1] - f) / (f[-1] - f[0] + EPS)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Inverse-CDF via exponential interpolation
# ---------------------------------------------------------------------------
@torch.no_grad()
def invert_survival(S: Tensor, t_grid: Tensor, U: Tensor) -> Tensor:
    """Vectorised inverse-CDF with exponential interpolation inside each bin."""
    N, G = S.shape
    neg = -S
    # Searchsorted needs the query broadcast per row
    idx = torch.searchsorted(neg, -U.unsqueeze(1), right=False).squeeze(1) - 1
    idx = idx.clamp(0, G - 2)

    rows = torch.arange(N, device=S.device)
    s_l, s_r = S[rows, idx], S[rows, idx + 1]
    t_l, t_r = t_grid[idx], t_grid[idx + 1]

    safe = (s_l > 0) & (s_r > 0) & (s_l != s_r)
    frac = torch.zeros(N, device=S.device, dtype=S.dtype)
    frac[safe] = torch.log(s_l[safe] / U[safe]) / torch.log(s_l[safe] / s_r[safe])
    frac = frac.clamp(0.0, 1.0)

    T = t_l + frac * (t_r - t_l)
    T = torch.where(U >= S[:, 0], t_grid[0], T)
    T = torch.where(U <= S[:, -1], t_grid[-1], T)
    return T


# ---------------------------------------------------------------------------
# Rejection filter for degenerate event nets
# ---------------------------------------------------------------------------
@torch.no_grad()
def _event_net_delta_f(
    event_net: nn.Sequential,
    eta_check: Tensor,   # (N_check, m)
    t0: float,
    t1: float,
) -> Tensor:
    n = eta_check.shape[0]
    t0_t = torch.full((n, 1), float(t0), device=eta_check.device, dtype=eta_check.dtype)
    t1_t = torch.full((n, 1), float(t1), device=eta_check.device, dtype=eta_check.dtype)
    f0 = event_net(torch.cat([eta_check, t0_t], dim=1)).view(-1)
    f1 = event_net(torch.cat([eta_check, t1_t], dim=1)).view(-1)
    return (f1 - f0).abs()


# ---------------------------------------------------------------------------
# Reg2SurvDiscrete module
# ---------------------------------------------------------------------------
class Reg2SurvDiscrete(nn.Module):
    """Discrete-grid survival sampler with a Reg2Surv-compatible forward."""

    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp.get("reg2surv_discrete_param", hp)
        self.device = self.hp.get("device", "cpu")
        self.dtype = self.hp.get("dtype", torch.float32)
        # Degenerate-time rejection (optional; defaults keep it active but safe).
        self._time_max_tries = int(self.hp.get("time_max_tries", 20))
        self._time_min_spread_rel = float(self.hp.get("time_min_spread_rel", 1.0e-8))
        self._time_min_spread_abs = float(self.hp.get("time_min_spread_abs", 1.0e-12))

        seed = self.hp.get("seed", None)
        # Torch generator -> uniform draws of U, V, lambda_eta vectors.
        # Kept on CPU; samples are moved to target device where needed.
        self._g = torch.Generator(device=self.device)
        if seed is not None:
            self._g.manual_seed(int(seed))
        # numpy RNG -> Beta / Gamma draws (torch.distributions doesn't accept generators).
        self._np = np.random.default_rng(None if seed is None else int(seed))

    # ------------------------------------------------------------------
    # Per-dataset hyper-parameter sampling
    # ------------------------------------------------------------------
    def _sample_hypers(self, n_etas: int) -> dict:
        g = self._g
        h = self.hp

        def _u(lo, hi):
            return lo + (hi - lo) * float(torch.rand((), generator=g, device=self.device).cpu().item())

        def _randint(lo, hi):
            # inclusive both ends; if lo == hi we'd still like to draw lo.
            hi_excl = hi + 1 if hi >= lo else lo + 1
            return int(torch.randint(lo, hi_excl, (1,), generator=g, device=self.device).cpu().item())

        def _beta_alpha(scale, shift, a: float = 1.1, b: float = 1.1):
            return float(self._np.beta(a, b)) * scale + shift

        def _vec_u(n, lo, hi):
            if n == 0:
                return torch.empty(0, device=self.device)
            return torch.empty(n, device=self.device).uniform_(lo, hi, generator=g)

        # ---- Event-net hypers (generate5.py event block) ----
        alpha = _beta_alpha(*h["alpha_bounds"])
        gamma_k = _u(*h["gamma_k_bounds"])
        lo_mu_t, hi_mu_t = h.get("mu_t_bounds", (0.0, 1.0))
        mu_t = _u(lo_mu_t + EPS, hi_mu_t - EPS)
        lambda_eta = _vec_u(n_etas, *h["lambda_eta_bounds"] )

        n_ev = _randint(*h["n_event_layers_bounds"])
        event_layers = [_randint(*h["event_layer_bounds"]) for _ in range(n_ev)]

        # ---- Censoring-net hypers (independent draws, same dist families) ----
        alpha_c = _beta_alpha(*h.get("alpha_cens_bounds", h["alpha_bounds"]))
        gamma_k_c = _u(*h.get("gamma_k_cens_bounds", h["gamma_k_bounds"]))
        lo_mu_t_c, hi_mu_t_c = h.get("mu_t_cens_bounds", h["mu_t_bounds"])
        mu_t_c = _u(lo_mu_t_c + EPS, hi_mu_t_c - EPS)
        # cens net has fan_in = 1 -> no eta columns; draw only for API symmetry.
        _ = _vec_u(0, *h.get("lambda_eta_cens_bounds", h["lambda_eta_bounds"]))

        n_cs = _randint(*h["n_cens_layers_bounds"])
        cens_layers = [_randint(*h["cens_layer_bounds"]) for _ in range(n_cs)]

        # ---- Shared scalars (one draw used for both T and C) ----
        t_max = _u(*h["t_max_bounds"])
        C_pct = _u(*h["censoring_rate_bounds"])

        return dict(
            alpha=alpha, gamma_k=gamma_k, mu_t=mu_t, lambda_eta=lambda_eta,
            event_layers=event_layers,
            alpha_c=alpha_c, gamma_k_c=gamma_k_c, mu_t_c=mu_t_c,
            cens_layers=cens_layers,
            t_max=t_max, C_pct=C_pct,
        )
    # for testing diffetent fixed hyperparameters
    def _sample_hypers_2(self, n_etas: int) -> dict:
        g = self._g
        h = self.hp

        def _u(lo, hi):
            return lo + (hi - lo) * float(torch.rand((), generator=g).item())

        def _randint(lo, hi):
            # inclusive both ends; if lo == hi we'd still like to draw lo.
            hi_excl = hi + 1 if hi >= lo else lo + 1
            return int(torch.randint(lo, hi_excl, (1,), generator=g).item())

        def _beta_alpha(scale, shift, a: float = 1.1, b: float = 1.1):
            return float(self._np.beta(a, b)) * scale + shift

        def _vec_u(n, lo, hi):
            if n == 0:
                return torch.empty(0)
            return torch.empty(n).uniform_(lo, hi, generator=g)

        # ---- Event-net hypers (generate5.py event block) ----
        alpha = torch.tensor(1.5)
        gamma_k = torch.tensor(2.0)
        #mu_t = torch.tensor(1.0)
        mu_t = np.random.uniform(0.0, 1.0)
        lambda_eta = torch.tensor(1.0)

        n_ev = _randint(*h["n_event_layers_bounds"])
        event_layers = [_randint(*h["event_layer_bounds"]) for _ in range(n_ev)]

        # ---- Censoring-net hypers (independent draws, same dist families) ----
        alpha_c = alpha
        gamma_k_c = gamma_k
        mu_t_c = mu_t
        # cens net has fan_in = 1 -> no eta columns; draw only for API symmetry.
        _ = 0.0

        n_cs = 1
        cens_layers = [8]

        # ---- Shared scalars (one draw used for both T and C) ----
        t_max = _u(*h["t_max_bounds"])
        C_pct = _u(*h["censoring_rate_bounds"])

        return dict(
            alpha=alpha, gamma_k=gamma_k, mu_t=mu_t, lambda_eta=lambda_eta,
            event_layers=event_layers,
            alpha_c=alpha_c, gamma_k_c=gamma_k_c, mu_t_c=mu_t_c,
            cens_layers=cens_layers,
            t_max=t_max, C_pct=C_pct,
        )
    # ------------------------------------------------------------------
    # Event net construction with optional rejection loop
    # ------------------------------------------------------------------
    def _build_event_net(self, M: int, hyp: dict) -> nn.Sequential:
        dev, dt = self.device, self.dtype
        max_tries = int(self.hp.get("event_net_max_tries", 1))
        min_med = float(self.hp.get("event_net_min_median_delta", 0.0))
        max_flat = float(self.hp.get("event_net_max_flat_frac", 1.0))
        delta_eps = float(self.hp.get("event_net_delta_eps", 1e-3))
        n_check = int(self.hp.get("event_net_check_n", 256))

        input_means = torch.cat(
            [torch.zeros(M), torch.tensor([hyp["mu_t"]])]
        ).to(device=dev, dtype=dt)

        last_net = None
        for _ in range(max(max_tries, 1)):
            net = build_positive_network(M + 1, hyp["event_layers"]).to(dev, dtype=dt)
            random_init(
                net,
                input_means=input_means,
                alpha=hyp["alpha"], gamma_k=hyp["gamma_k"],
                lambda_eta=hyp["lambda_eta"],
                np_rng=self._np,
            )
            net.eval()
            last_net = net

            if max_tries <= 1:
                return net

            # Rejection check: make sure f varies with t for most eta draws.
            eta_check = torch.randn(n_check, M, device=dev, dtype=dt) if M > 0 \
                else torch.zeros(n_check, 0, device=dev, dtype=dt)
            delta = _event_net_delta_f(net, eta_check, 0.0, 1.0)
            flat_frac = float((delta < delta_eps).float().mean().item())
            med_delta = float(delta.median().item())
            if flat_frac <= max_flat and med_delta >= min_med:
                return net

        return last_net

    # ------------------------------------------------------------------
    # Contract-matching forward: (X, y) -> (X, (T_obs, E), meta)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self, X: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], dict]:
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        N, M = y.shape
        dev, dt = self.device, self.dtype

        X = X.to(device=dev, dtype=dt)
        y = y.to(device=dev, dtype=dt)

        # Per-channel z-score (matches generate5.py get_eta)
        mu = y.mean(0, keepdim=True)
        sd = y.std(0, keepdim=True).clamp_min(1e-8)
        eta = ((y - mu) / sd).clamp(-5.0, 5.0)

        # Regenerate if the observed times are degenerate (all equal / near-equal).
        last_payload = None
        max_tries = max(int(self._time_max_tries), 1)
        for _ in range(max_tries):
            hyp = self._sample_hypers(M)
            t_grid = torch.linspace(0.0, 1.0, self.hp["n_grid"], device=dev, dtype=dt)

            # ---- Event net ----
            event_net = self._build_event_net(M, hyp)

            # ---- Censoring net (fan_in == 1, no eta) ----
            cens_net = build_positive_network(1, hyp["cens_layers"]).to(dev, dtype=dt)
            random_init(
                cens_net,
                input_means=torch.tensor([hyp["mu_t_c"]], device=dev, dtype=dt),
                alpha=hyp["alpha_c"], gamma_k=hyp["gamma_k_c"],
                lambda_eta=torch.empty(0),
                np_rng=self._np,
            )
            cens_net.eval()

            # ---- Event survival & times ----
            S_event = compute_survival(event_net, eta, t_grid)                 # (N, G)
            U = torch.empty(N, device=dev).uniform_(0.01, 0.99, generator=self._g).to(device=dev, dtype=dt)
            T_norm = invert_survival(S_event, t_grid, U)
            T = T_norm * hyp["t_max"]

            # ---- Censoring survival & times ----
            S_cens_1d = compute_survival_1d(cens_net, t_grid)                   # (G,)
            S_cens = S_cens_1d.unsqueeze(0).expand(N, -1)
            V = torch.empty(N, device=dev).uniform_(0.01, 0.99, generator=self._g).to(device=dev, dtype=dt)
            C_norm = invert_survival(S_cens, t_grid, V)
            C = C_norm * hyp["t_max"]

            # ---- Admin censoring at quantile (1 - C_pct) of T ----
            C_admin = torch.quantile(T, 1.0 - hyp["C_pct"])
            E = (T <= C).to(torch.int32)
            T_obs = torch.minimum(T, C)
            E = E * (T_obs <= C_admin).to(torch.int32)
            T_obs = torch.minimum(T_obs, C_admin)

            # Degeneracy check on observed times.
            # Use relative tolerance scaled by t_max to be robust across horizons.
            spread = (T_obs.max() - T_obs.min()).abs()
            eps = (self._time_min_spread_rel * float(hyp["t_max"])) + self._time_min_spread_abs

            last_payload = (hyp, t_grid, S_event, T_obs, E, C_admin)
            if float(spread.detach().cpu().item()) > float(eps):
                break

        assert last_payload is not None
        hyp, t_grid, S_event, T_obs, E, C_admin = last_payload

        meta = {
            "mode": "discrete",
            "n_etas": int(M),
            "alpha": float(hyp["alpha"]),
            "gamma_k": float(hyp["gamma_k"]),
            "mu_t": float(hyp["mu_t"]),
            "lambda_eta": hyp["lambda_eta"].detach().cpu().tolist(),
            "event_layers": list(hyp["event_layers"]),
            "alpha_c": float(hyp["alpha_c"]),
            "gamma_k_c": float(hyp["gamma_k_c"]),
            "mu_t_c": float(hyp["mu_t_c"]),
            "cens_layers": list(hyp["cens_layers"]),
            "t_max": float(hyp["t_max"]),
            "C_percent": float(hyp["C_pct"]),
            "C": float(C_admin.detach().cpu().item()),
        }
        # Stash for offline debug / plotting helpers (generate_syn_data.py).
        # Not part of meta because the JSON serialiser in genload.save_sampled_params
        # can't serialise ndarrays; consumers read these attributes directly.
        self.last_S_event = S_event.detach().cpu().numpy()
        self.last_t_grid_scaled = (t_grid * hyp["t_max"]).detach().cpu().numpy()
        self.last_t_grid_norm = t_grid.detach().cpu().numpy()
        return X, (T_obs, E), meta
