from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from typing_extensions import Literal

from .admin_censoring import apply_admin_censoring, sample_follow_up_fractions
from .censoring_eta import (
    DEP_COVDEP,
    DEP_INDEP,
    eta_c_from_x,
    sample_censoring_dependence,
)
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

Kind = Literal["unit", "positive", "real", "bounds", "log_bounds"]
TensorLike = Union[Tensor, float, np.ndarray, List[float]]

EPS = 1e-6


def standardize(x: Tensor) -> Tensor:
    mu = x.mean()
    sd = x.std().clamp_min(1e-8)
    return (x - mu) / sd


# --------------------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------------------

class CategoricalSampler:
    """
    Samples categorical labels, but can also sample integer indices efficiently.
    """
    def __init__(
        self,
        distribution: Dict[str, float],
        seed: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ):
        self.device = device or "cpu"
        self.dtype = dtype

        labels = [str(k).lower() for k in distribution.keys()]
        probs = torch.tensor(list(distribution.values()), dtype=dtype, device=self.device)
        total = probs.sum()
        if total <= 0:
            raise ValueError("Probabilities/weights must sum to a positive value.")
        self._probs = probs / total
        self._labels = labels
        self._g = generator or torch.Generator(device=self.device)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def sample_indices(self, n: int) -> torch.LongTensor:
        idx = torch.multinomial(self._probs, n, replacement=True, generator=self._g)
        return idx.to(torch.long)

    def sample(self, n: int):
        idx = self.sample_indices(n)
        if n == 1:
            return self._labels[int(idx.item())]
        return [self._labels[int(i)] for i in idx.tolist()]


class NumericalSampler:
    """
    'unit'       -> U ∈ (0,1) (open interval; safe for logs)
    'positive'   -> Log-uniform on [1e-8, 1e8]
    'real'       -> Logistic(0,1): log(u/(1-u))
    'bounds'     -> Uniform on (lo,hi)
    'log_bounds' -> Log-uniform on (lo,hi); requires lo>0, hi>lo
    """
    def __init__(
        self,
        kind: Kind,
        *,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.LOW = 1e-8
        self.HIGH = 1e8
        kind = str(kind).lower()
        if kind not in ("unit", "positive", "real", "bounds", "log_bounds"):
            raise ValueError(
                "kind must be 'unit', 'positive', 'real', 'bounds', or 'log_bounds'"
            )
        self.kind = kind
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self._g = generator or torch.Generator(device=self.device)

        if self.kind == "positive":
            lo = torch.tensor(self.LOW, device=self.device, dtype=self.dtype)
            hi = torch.tensor(self.HIGH, device=self.device, dtype=self.dtype)
            self._log_lo = lo.log()
            self._log_hi = hi.log()

        if self.kind == "bounds":
            if bounds is None:
                raise ValueError("bounds must be provided when kind='bounds'")
            lo, hi = map(float, bounds)
            self._b_lo = torch.tensor(lo, device=self.device, dtype=self.dtype)
            self._b_hi = torch.tensor(hi, device=self.device, dtype=self.dtype)

        if self.kind == "log_bounds":
            if bounds is None:
                raise ValueError("bounds must be provided when kind='log_bounds'")
            lo, hi = map(float, bounds)
            if not (lo > 0.0 and hi > lo):
                raise ValueError(f"log_bounds requires 0 < lo < hi, got lo={lo}, hi={hi}")
            self._log_lo = torch.tensor(lo, device=self.device, dtype=self.dtype).log()
            self._log_hi = torch.tensor(hi, device=self.device, dtype=self.dtype).log()

    def _u_open01(self, shape: Tuple[int, ...]) -> torch.Tensor:
        u = torch.rand(shape, device=self.device, dtype=self.dtype, generator=self._g)
        return u.clamp_(min=EPS, max=1 - EPS)

    def sample(self, n: int) -> torch.Tensor:
        if self.kind == "unit":
            return self._u_open01((n,))
        if self.kind in ("positive", "log_bounds"):
            u = self._u_open01((n,))
            return torch.exp(self._log_lo + u * (self._log_hi - self._log_lo))
        if self.kind == "bounds":
            u = self._u_open01((n,))
            return self._b_lo + u * (self._b_hi - self._b_lo)
        # 'real'
        u = self._u_open01((n,))
        return torch.log(u / (1 - u))


# --------------------------------------------------------------------------------------
# Vectorized inverse cumulative hazard implementations (standardized to shape/scale)
# --------------------------------------------------------------------------------------

def _Hinv_weibull(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    invk = 1.0 / (shape + EPS)
    return scale * torch.pow(y, invk)

def _Hinv_gompertz(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == eta, scale == b;  H0(t) = (eta/b)(e^{bt} - 1)  =>  H0^{-1}(y) = log1p((b/eta) y) / b.
    # b == 0 is the exponential limit y / eta. No EPS: b > 0 and eta > 0 from the anchor solve,
    # and log1p is exact for small arguments.
    small = scale <= 0
    out_exp = y / shape
    out_g = torch.log1p(scale / shape * y) / scale
    return torch.where(small, out_exp, out_g)

def _Hinv_lognormal(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == sigma, scale == mu (log scale);  H0^{-1}(y) = exp(mu + sigma z),  z = Phi^{-1}(1 - e^{-y}).
    # z is evaluated in float32 without forming 1 - e^{-y} (cancellation) or e^{-y} (underflow for y > ~103):
    #   y < log 2:        z = Phi^{-1}(-expm1(-y))     -expm1(-y) == 1 - e^{-y}, exact for small y
    #   log 2 <= y < 30:  z = -Phi^{-1}(e^{-y})        symmetry Phi^{-1}(1 - p) = -Phi^{-1}(p)
    #   y >= 30:          solve log Phi(-z) = -y        <=> Phi(z) = 1 - e^{-y}, in log space;
    #                     start z0 = sqrt(2y - log(4 pi y)) from Phi(-z) ~ phi(z)/z
    # then two Newton steps on f(z) = log Phi(-z) + y, f'(z) = -phi(z)/Phi(-z) (no-op where z is already exact).
    z_left = torch.special.ndtri(-torch.expm1(-y))
    z = torch.where(y < 30, -torch.special.ndtri(torch.exp(-y)), torch.sqrt(2 * y - torch.log(4 * np.pi * y)))
    for _ in range(2):
        log_sf = torch.special.log_ndtr(-z)
        z = z + (log_sf + y) * torch.exp(log_sf + 0.5 * z * z + 0.5 * np.log(2 * np.pi))
    z = torch.where(y < np.log(2), z_left, z)
    return torch.exp(scale + shape * z)

def _Hinv_loglogistic(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == a, scale == lambda;  H0^{-1}(y) = lambda (e^y - 1)^{1/a}.
    # log(e^y - 1) = y + log(1 - e^{-y}) = y + log(-expm1(-y)): no overflow of e^y for y > 88.7,
    # so T is +inf only when the true T exceeds float32. 1/a without EPS matches the anchor (a >= 0.5).
    return scale * torch.exp((y + torch.log(-torch.expm1(-y))) / shape)

def _Hinv_birnbaum_saunders(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == alpha, scale == beta;  H0^{-1}(y) = beta r^2,  r = (az + sqrt(az^2 + 4)) / 2,  az = alpha z,
    # z = Phi^{-1}(1 - e^{-y}) evaluated by the same three cases as in _Hinv_lognormal.
    z_left = torch.special.ndtri(-torch.expm1(-y))
    z = torch.where(y < 30, -torch.special.ndtri(torch.exp(-y)), torch.sqrt(2 * y - torch.log(4 * np.pi * y)))
    for _ in range(2):
        log_sf = torch.special.log_ndtr(-z)
        z = z + (log_sf + y) * torch.exp(log_sf + 0.5 * z * z + 0.5 * np.log(2 * np.pi))
    z = torch.where(y < np.log(2), z_left, z)
    # For az < 0 the sum az + sqrt(az^2 + 4) cancels; multiplying by (sqrt - az)/(sqrt - az) gives the
    # identical r = 2 / (sqrt(az^2 + 4) - az) without cancellation.
    az = shape * z
    s = torch.sqrt(az * az + 4.0)
    r = torch.where(az < 0, 2.0 / (s - az), 0.5 * (az + s))
    return scale * r * r


# --------------------------------------------------------------------------------------
# Baseline log-survival log S0(t) of the same standard distributions (same shape/scale names)
# --------------------------------------------------------------------------------------

def _logS_weibull(t: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    return -torch.pow(t / scale, shape)

def _logS_gompertz(t: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == eta, scale == b; b == 0 is the exponential limit -eta t.
    return torch.where(scale <= 0, -shape * t, -(shape / scale) * torch.expm1(scale * t))

def _logS_lognormal(t: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == sigma, scale == mu (log scale)
    return torch.special.log_ndtr(-(torch.log(t) - scale) / shape)

def _logS_loglogistic(t: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    return -torch.log1p(torch.pow(t / scale, shape))

def _logS_birnbaum_saunders(t: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == alpha, scale == beta
    r = torch.sqrt(t / scale)
    return torch.special.log_ndtr(-(r - 1.0 / r) / shape)


# --------------------------------------------------------------------------------------
# Model transform (vectorized)
# --------------------------------------------------------------------------------------

def _model_prefactor_and_cumhaz_arg(model: str, U: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      prefactor = exp(-eta1)
      y_arg     = -log(U) * exp(eta1 - eta2)
    so that T = prefactor * H0^{-1}(y_arg)
    """
    model = model.lower()
    if model == "ph":
        eta1 = torch.zeros_like(eta[:, 0] if eta.ndim == 2 else eta)
        eta2 = eta[:, 0] if eta.ndim == 2 else eta
    elif model == "aft":
        eta1 = eta[:, 0] if eta.ndim == 2 else eta
        eta2 = eta1
    elif model == "ah":
        eta1 = eta[:, 0] if eta.ndim == 2 else eta
        eta2 = torch.zeros_like(eta1)
    elif model == "eh":
        if eta.ndim != 2 or eta.shape[1] < 2:
            raise ValueError("Model 'eh' expects y to have at least 2 columns (two risk scores).")
        eta1 = eta[:, 0]
        eta2 = eta[:, 1]
    else:
        raise ValueError(f"Unsupported model type: {model}")

    y_arg = (-torch.log(U)) * torch.exp(eta1 - eta2)
    pref = torch.exp(-eta1)
    return pref, y_arg


# --------------------------------------------------------------------------------------
# Accelerated Reg2Surv (no exponential; everything uses standardized shape/scale)
# --------------------------------------------------------------------------------------

class Reg2Surv(nn.Module):
    """
      - All distributions expose ONLY two parameters: 'shape' and 'scale'
          * Weibull:            shape=k,   scale=lambda
          * Gompertz:           shape=eta, scale=b
          * Lognormal:          shape=sigma, scale=mu (log scale)
          * Loglogistic:        shape=alpha, scale=lambda
          * Birnbaum-Saunders:  shape=alpha, scale=beta
      - Batched param sampling and batched T computation per distribution code
      - No per-row objects/closures in the hot path
    """

    INV_HAZARD_LABELS = [
        "weibull",
        "gompertz",
        "lognormal",
        "loglogistic",
        "birnbaum-saunders",
    ]
    INV_HAZARD_TO_CODE = {name: i for i, name in enumerate(INV_HAZARD_LABELS)}

    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        self.device = self.hp.get("device", "cuda")
        self.dtype = self.hp.get("dtype", torch.float32)
        self.seed = self.hp.get("seed", None)
        

        self.rng = torch.Generator(device=self.device)
        if self.seed is not None:
            self.rng.manual_seed(int(self.seed))

        # Base samplers
        self.U_sampler = NumericalSampler("unit", seed=self.seed, device=self.device, generator=self.rng, dtype=self.dtype)
        self.U_C_sampler = NumericalSampler(
            "unit",
            seed=(self.seed + 1) if self.seed is not None else None,
            device=self.device,
            generator=self.rng,
            dtype=self.dtype,
        )

        self.model_sampler = CategoricalSampler(
            self.hp["model_distribution"],
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            dtype=self.dtype,
        )

        self.inv_hazard_sampler = CategoricalSampler(
            self.hp["inv_hazard_distribution"],
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            dtype=self.dtype,
        )
        self.inv_hazard_sampler_C = CategoricalSampler(
            self.hp["inv_hazard_distribution"],
            seed=(self.seed + 2) if self.seed is not None else None,
            device=self.device,
            generator=self.rng,
            dtype=self.dtype,
        )

        # Map sampled indices -> canonical hazard codes on device (fast)
        self._map_idx_to_code = self._build_idx_to_code_map(self.inv_hazard_sampler)
        self._map_idx_to_code_C = self._build_idx_to_code_map(self.inv_hazard_sampler_C)

        # Anchors and censoring fraction samplers
        self.q_star_sampler = NumericalSampler(
            "bounds",
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            bounds=(self.hp["q_star"]["min"], self.hp["q_star"]["max"]),
            dtype=self.dtype,
        )
        # Censoring anchor level. Drawn independently of q_star so that P(C < T)
        # is not pinned near 1/2 by the two time distributions sharing an anchor.
        # t_star stays shared, which keeps C on the same time scale as T.
        q_star_C_bounds = self.hp.get("q_star_C", self.hp["q_star"])
        self.q_star_C_sampler = NumericalSampler(
            "bounds",
            seed=(self.seed + 3) if self.seed is not None else None,
            device=self.device,
            generator=self.rng,
            bounds=(q_star_C_bounds["min"], q_star_C_bounds["max"]),
            dtype=self.dtype,
        )
        self.tstar_sampler = NumericalSampler(
            "log_bounds",
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            bounds=(self.hp["t_star"]["min"], self.hp["t_star"]["max"]),
            dtype=self.dtype,
        )

        # Administrative censoring branches. censoring_rate keeps its old meaning
        # (target administrative censoring fraction) as the default for type 2.
        t1_bounds = self.hp.get("type1_quantile_bounds", [0.05, 0.95])
        t2_bounds = self.hp.get(
            "type2_quantile_bounds",
            [self.hp["censoring_rate"]["min"], self.hp["censoring_rate"]["max"]],
        )
        self.type1_q_sampler = NumericalSampler(
            "bounds",
            seed=(self.seed + 4) if self.seed is not None else None,
            device=self.device,
            generator=self.rng,
            bounds=(float(t1_bounds[0]), float(t1_bounds[1])),
            dtype=self.dtype,
        )
        self.C_sampler = NumericalSampler(
            "bounds",
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            bounds=(float(t2_bounds[0]), float(t2_bounds[1])),
            dtype=self.dtype,
        )

        self.p_type1 = float(self.hp.get("p_type1", 0.5))
        self.p_type2 = float(self.hp.get("p_type2", 0.5))
        self.p_cov_indep_cens = float(self.hp.get("p_cov_indep_cens", 0.25))
        self.p_informative_cens = float(self.hp.get("p_informative_cens", 0.0))

        # Optional true survival curve per row on linspace(0, 1, n_grid) * t_hi, with
        # t_hi = surv_curve_window_mult * max(T_obs of the context rows).
        self.save_surv_curve = bool(self.hp.get("save_surv_curve", False))
        self.n_grid = int(self.hp.get("n_grid", 100))
        self.surv_curve_window_mult = float(self.hp.get("surv_curve_window_mult", 1.0))

        # Random censoring switch: off means C = +inf.
        self.p_random = float(self.hp.get("p_random", 1.0))

        # Same SCM HP families as the event prior (threaded from SCMPrior).
        self.scm_sampled_hp = DEFAULT_SAMPLED_HP
        self.scm_fixed_hp = DEFAULT_FIXED_HP
        self.scm_prior_type = "mlp_scm"

        # Staggered entry: subject i is followed for r_i * tau, r_i = 1 - a * U^(1/k).
        self.p_stagger = float(self.hp.get("p_stagger_entry", 1.0))
        self.accrual_frac_bounds = tuple(float(v) for v in self.hp.get("accrual_frac_bounds", (0.0, 1.0)))
        self.accrual_shape_bounds = tuple(float(v) for v in self.hp.get("accrual_shape_bounds", (0.2, 1.0)))
        # numpy RNG -> uniform draws for the follow-up fractions.
        self._np = np.random.default_rng(None if self.seed is None else int(self.seed) + 7)

        self.unit_distrib_param_sampler = NumericalSampler(
            "unit",
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            dtype=self.dtype,
        )


    def _build_idx_to_code_map(self, sampler: CategoricalSampler) -> torch.LongTensor:
        labels = sampler.labels
        codes: List[int] = []
        for s in labels:
            s = str(s).lower()
            if s == "exponential":
                raise ValueError("Remove 'exponential' from hp['inv_hazard_distribution'] (it is no longer supported).")
            if s not in self.INV_HAZARD_TO_CODE:
                raise ValueError(f"inv_hazard label '{s}' not in supported set {self.INV_HAZARD_LABELS}.")
            codes.append(self.INV_HAZARD_TO_CODE[s])
        return torch.tensor(codes, device=self.device, dtype=torch.long)

    def _get_eta(self, y: Tensor) -> Tensor:
        y = y.to(device=self.device, dtype=self.dtype)
        mu = y.mean(dim=0, keepdim=True)
        sd = y.std(dim=0, keepdim=True).clamp_min(1e-8)
        return ((y - mu) / sd).clamp(-5, 5)

    def _sample_params_batched(
        self,
        haz_codes: torch.LongTensor,
        q_star: Tensor,
        t_star: Tensor,
    ) -> Dict[str, Tensor]:
        N = haz_codes.shape[0]
        device, dtype = self.device, self.dtype
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)

        shape = torch.full((N,), nan, device=device, dtype=dtype)
        scale = torch.full((N,), nan, device=device, dtype=dtype)

        # One uniform draw for all rows (open interval)
        u = self.unit_distrib_param_sampler.sample(N)  # (N,)

        q = q_star.clamp(min=EPS, max=1.0 - EPS)
        t = t_star.clamp_min(EPS)
        neglogq = -torch.log(q)

        def u_to_bounds(u_sub: Tensor, lo: float, hi: float) -> Tensor:
            lo_t = torch.tensor(lo, device=device, dtype=dtype)
            hi_t = torch.tensor(hi, device=device, dtype=dtype)
            return lo_t + u_sub * (hi_t - lo_t)

        # Masks (no Python indexing, no variable-sized sampling)
        m_weib = haz_codes == self.INV_HAZARD_TO_CODE["weibull"]
        m_gomp = haz_codes == self.INV_HAZARD_TO_CODE["gompertz"]
        m_logn = haz_codes == self.INV_HAZARD_TO_CODE["lognormal"]
        m_llog = haz_codes == self.INV_HAZARD_TO_CODE["loglogistic"]
        m_bs   = haz_codes == self.INV_HAZARD_TO_CODE["birnbaum-saunders"]

        # Weibull: shape ~ U(0.5,5), scale derived
        if m_weib.any():
            k = u_to_bounds(u[m_weib], self.hp["bounds"]["weibull"]["k"]["min"], self.hp["bounds"]["weibull"]["k"]["max"])
            lam = t / torch.pow(neglogq, 1.0 / (k + EPS))
            shape[m_weib] = k
            scale[m_weib] = lam

        # Gompertz: scale=b ~ U(0,0.3), shape=eta derived
        # note that scale and shape are taken from the wikipedia parametrization, which is not true for inverse hazard sampling;
        # do not be confused, these are just the names for the parameters; b is easier to derive compared to eta
        if m_gomp.any():
            b = u_to_bounds(u[m_gomp], self.hp["bounds"]["gompertz"]["b"]["min"], self.hp["bounds"]["gompertz"]["b"]["max"])
            # Exact anchor H0(t*) = -log q*:  eta = b (-log q*) / expm1(b t*);  b == 0 limit: eta = -log q* / t*.
            # Stored as is (no EPS): for large b t* the true eta drops far below EPS, so adding EPS broke the anchor.
            # u in [EPS, 1 - EPS] keeps b > 0, expm1(b t*) > 0 and b t* <= 30 (no overflow).
            small = b <= 0
            eta_small = neglogq / t
            eta_big = (b * neglogq) / torch.expm1(b * t)
            eta = torch.where(small, eta_small.expand_as(b), eta_big)
            shape[m_gomp] = eta
            scale[m_gomp] = b

        # Lognormal: shape=sigma ~ U(0.3,1.5), scale=mu=log(t*)-sigma*z_anchor (log scale)
        if m_logn.any():
            sigma = u_to_bounds(u[m_logn], self.hp["bounds"]["lognormal"]["sigma"]["min"], self.hp["bounds"]["lognormal"]["sigma"]["max"])
            z_anchor = torch.special.ndtri((1.0 - q).clamp(min=EPS, max=1.0 - EPS))
            #s = t * torch.exp(-sigma * z_anchor)
            s = torch.log(t) - sigma * z_anchor
            shape[m_logn] = sigma
            scale[m_logn] = s

        # Loglogistic: shape ~ U(0.5,5), scale derived
        if m_llog.any():
            a = u_to_bounds(u[m_llog], self.hp["bounds"]["loglogistic"]["a"]["min"], self.hp["bounds"]["loglogistic"]["a"]["max"])
            log_base = torch.log1p(-q) - torch.log(q)
            lam = t / torch.exp((1.0 / a) * log_base)
            shape[m_llog] = a
            scale[m_llog] = lam

        # Birnbaum–Saunders: shape ~ U(0.5,5), scale=beta derived
        if m_bs.any():
            alpha = u_to_bounds(u[m_bs], self.hp["bounds"]["birnbaum-saunders"]["alpha"]["min"], self.hp["bounds"]["birnbaum-saunders"]["alpha"]["max"])
            z_anchor = torch.special.ndtri((1.0 - q).clamp(min=EPS, max=1.0 - EPS))
            # Same r as in _Hinv_birnbaum_saunders (cancellation-free for az < 0), so H0^{-1}(-log q*) = beta r^2 = t*.
            # r > 0 always, so no EPS in the denominator.
            az = alpha * z_anchor
            s = torch.sqrt(az * az + 4.0)
            r = torch.where(az < 0, 2.0 / (s - az), 0.5 * (az + s))
            beta = t / (r * r)
            shape[m_bs] = alpha
            scale[m_bs] = beta

        return {"shape": shape, "scale": scale}


    def _compute_T_batched(
        self,
        model: str,
        haz_codes: torch.LongTensor,
        params: Dict[str, Tensor],
        U: Tensor,
        eta: Tensor,
    ) -> Tensor:
        N = haz_codes.shape[0]
        T_out = torch.empty((N,), device=self.device, dtype=self.dtype)

        pref, y_arg = _model_prefactor_and_cumhaz_arg(model, U, eta)

        def idx_for(code: int) -> torch.LongTensor:
            return torch.nonzero(haz_codes == code, as_tuple=True)[0]

        # Weibull
        code = self.INV_HAZARD_TO_CODE["weibull"]
        idx = idx_for(code)
        if idx.numel() > 0:
            T_out[idx] = pref[idx] * _Hinv_weibull(y_arg[idx], params["shape"][idx], params["scale"][idx])

        # Gompertz
        code = self.INV_HAZARD_TO_CODE["gompertz"]
        idx = idx_for(code)
        if idx.numel() > 0:
            T_out[idx] = pref[idx] * _Hinv_gompertz(y_arg[idx], params["shape"][idx], params["scale"][idx])

        # Lognormal
        code = self.INV_HAZARD_TO_CODE["lognormal"]
        idx = idx_for(code)
        if idx.numel() > 0:
            T_out[idx] = pref[idx] * _Hinv_lognormal(y_arg[idx], params["shape"][idx], params["scale"][idx])

        # Loglogistic
        code = self.INV_HAZARD_TO_CODE["loglogistic"]
        idx = idx_for(code)
        if idx.numel() > 0:
            T_out[idx] = pref[idx] * _Hinv_loglogistic(y_arg[idx], params["shape"][idx], params["scale"][idx])

        # Birnbaum-Saunders
        code = self.INV_HAZARD_TO_CODE["birnbaum-saunders"]
        idx = idx_for(code)
        if idx.numel() > 0:
            T_out[idx] = pref[idx] * _Hinv_birnbaum_saunders(y_arg[idx], params["shape"][idx], params["scale"][idx])

        return T_out

    def _log_surv_batched(
        self,
        model: str,
        haz_codes: torch.LongTensor,
        params: Dict[str, Tensor],
        eta: Tensor,
        t: Tensor,
    ) -> Tensor:
        """log S(t | row) on a time grid t (G,), shape (N, G).

        Extended-hazard form of all four models: S(t|x) = S0(t e^{eta1})^{e^{eta2 - eta1}}.
        With -log U = 1 the sampler's transform gives pref = e^{-eta1} and y_arg = e^{eta1 - eta2}.
        """
        N = haz_codes.shape[0]
        U = torch.full((N,), float(np.exp(-1.0)), device=self.device, dtype=self.dtype)
        pref, y_arg = _model_prefactor_and_cumhaz_arg(model, U, eta)
        t_base = t.unsqueeze(0) / pref.unsqueeze(1)                      # (N, G)
        out = torch.empty_like(t_base)
        log_s0 = {
            "weibull": _logS_weibull,
            "gompertz": _logS_gompertz,
            "lognormal": _logS_lognormal,
            "loglogistic": _logS_loglogistic,
            "birnbaum-saunders": _logS_birnbaum_saunders,
        }
        for name, fn in log_s0.items():
            idx = torch.nonzero(haz_codes == self.INV_HAZARD_TO_CODE[name], as_tuple=True)[0]
            if idx.numel() > 0:
                shape = params["shape"][idx].unsqueeze(1)
                scale = params["scale"][idx].unsqueeze(1)
                out[idx] = fn(t_base[idx], shape, scale) / y_arg[idx].unsqueeze(1)
        return out

    def forward(self, X: Tensor, y: Tensor, train_size: Optional[int] = None) -> tuple[Tensor, dict, dict]:
        if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"Input shapes mismatch or incorrect dims. X: {X.shape}, y: {y.shape}")

        N = X.shape[0]
        eta = self._get_eta(y)
        t_max = float(self.hp.get("t_max", 10000.0))
        max_tries = max(int(self.hp.get("time_max_tries", 50)), 1)

        # Redraw anchors/hazards/U until observed times stay on the t_star scale
        # (max(T_obs) <= t_max * t_star). eta is fixed for the dataset.
        for _ in range(max_tries):
            # anchors. t_star is shared so that T and C live on the same time scale;
            # only the anchor level differs, which is what tilts P(C < T).
            q_star = self.q_star_sampler.sample(1)[0].clamp(min=EPS, max=1.0 - EPS)
            q_star_C = self.q_star_C_sampler.sample(1)[0].clamp(min=EPS, max=1.0 - EPS)
            t_star = self.tstar_sampler.sample(1)[0].clamp_min(EPS)

            # censoring fraction and model (same structure)
            C_percent = self.C_sampler.sample(1)[0]
            model = self.model_sampler.sample(1)

            # U draws
            U = self.U_sampler.sample(N)
            U_C = self.U_C_sampler.sample(N)

            # hazards as indices -> canonical codes on device
            inv_idx = self.inv_hazard_sampler.sample_indices(N)
            inv_idx_C = self.inv_hazard_sampler_C.sample_indices(N)

            haz_codes = self._map_idx_to_code[inv_idx]
            haz_codes_C = self._map_idx_to_code_C[inv_idx_C]

            # params + times (batched)
            params_t = self._sample_params_batched(haz_codes, q_star=q_star, t_star=t_star)
            params_C_t = self._sample_params_batched(haz_codes_C, q_star=q_star_C, t_star=t_star)

            T = self._compute_T_batched(model=model, haz_codes=haz_codes, params=params_t, U=U, eta=eta)

            # Ternary dependence: indep (η_C=0) / covdep (η_C=f(X)) / info (SCM siblings).
            if eta.shape[1] < 4:
                dep = DEP_INDEP
            else:
                dep = sample_censoring_dependence(
                    self.p_cov_indep_cens,
                    self.p_informative_cens,
                    rng=self.rng,
                    device=self.device,
                )
            cov_indep_cens = dep == DEP_INDEP
            if dep == DEP_INDEP:
                eta_C = torch.zeros((N, 2), device=self.device, dtype=self.dtype)
            elif dep == DEP_COVDEP:
                eta_C = eta_c_from_x(
                    X, 2,
                    prior_type=getattr(self, "scm_prior_type", "mlp_scm"),
                    device=self.device,
                    dtype=self.dtype,
                    scm_sampled_hp=getattr(self, "scm_sampled_hp", None),
                    scm_fixed_hp=getattr(self, "scm_fixed_hp", None),
                )
            else:
                eta_C = eta[:, 2:4]
            C_times = self._compute_T_batched(model=model, haz_codes=haz_codes_C, params=params_C_t, U=U_C, eta=eta_C)
            use_random = float(torch.rand((), generator=self.rng, device=self.device)) < self.p_random
            if not use_random:
                C_times = torch.full_like(C_times, float("inf"))

            # random censoring
            E = (T <= C_times).to(torch.int32)
            T_obs = torch.minimum(T, C_times)
            random_cens_frac = float(1.0 - E.double().mean())
            n_events_random = int(E.sum())

            # Administrative censoring: two independent optional branches, earliest wins.
            use_type1 = float(torch.rand((), generator=self.rng, device=self.device)) < self.p_type1
            use_type2 = float(torch.rand((), generator=self.rng, device=self.device)) < self.p_type2
            q_type1 = self.type1_q_sampler.sample(1)[0]

            # Staggered entry spreads the cutoff over [min(r) * C_admin, C_admin]
            # instead of leaving every still-at-risk subject tied at one time.
            stagger = False
            r_frac = None
            accrual_frac, accrual_shape = 0.0, 1.0
            if use_type1 or use_type2:
                stagger = float(torch.rand((), generator=self.rng, device=self.device)) < self.p_stagger
                if stagger:
                    lo_a, hi_a = self.accrual_frac_bounds
                    lo_k, hi_k = self.accrual_shape_bounds
                    accrual_frac = lo_a + (hi_a - lo_a) * float(torch.rand((), generator=self.rng, device=self.device))
                    accrual_shape = lo_k + (hi_k - lo_k) * float(torch.rand((), generator=self.rng, device=self.device))
                    r_frac = sample_follow_up_fractions(
                        self._np, N, accrual_frac, accrual_shape, device=self.device, dtype=self.dtype
                    )

            T_obs, E, C_admin, admin_info = apply_admin_censoring(
                T_obs,
                E,
                T_event=T,
                r=r_frac,
                use_type1=use_type1,
                use_type2=use_type2,
                q_type1=q_type1,
                c_pct=C_percent,
            )
            admin_info["random_cens_frac"] = random_cens_frac
            admin_info["admin_cens_frac"] = (n_events_random - int(E.sum())) / N

            if bool((T_obs.max() <= (t_max * t_star)).item()):
                break

        # T_cens includes administrative censoring, so T_obs == min(T, T_cens).
        if C_admin is None:
            T_cens = C_times
        else:
            # Same double-precision product as apply_admin_censoring, so ties at the horizon agree with E.
            limit = C_admin if r_frac is None else r_frac.double() * C_admin
            T_cens = torch.minimum(C_times, limit.to(C_times.dtype))
        y_out = {"T_obs": T_obs, "E": E, "T_event": T, "T_cens": T_cens}
        if self.save_surv_curve:
            if train_size is None:
                raise ValueError("save_surv_curve needs train_size: the grid ends at the max observed context time")
            # Context rows only: a grid placed by hidden quantities (true times, query rows) biases the loss.
            t_hi = self.surv_curve_window_mult * T_obs[:train_size].max()
            t_grid = torch.linspace(0.0, 1.0, self.n_grid, device=self.device, dtype=self.dtype) * t_hi
            y_out["t_grid"] = t_grid
            y_out["S"] = torch.exp(self._log_surv_batched(model, haz_codes, params_t, eta, t_grid))

        # --- metadata (standardized: everything is shape/scale)
        inv_labels = self.inv_hazard_sampler.labels
        inv_labels_C = self.inv_hazard_sampler_C.labels
        inv_hazards = [inv_labels[int(i)] for i in inv_idx.detach().cpu().tolist()]
        inv_hazards_C = [inv_labels_C[int(i)] for i in inv_idx_C.detach().cpu().tolist()]

        params_cpu = {k: v.detach().cpu() for k, v in params_t.items()}
        params_C_cpu = {k: v.detach().cpu() for k, v in params_C_t.items()}

        def row_params_list(haz_list: List[str], p_cpu: Dict[str, Tensor]) -> List[Dict[str, float]]:
            out_list: List[Dict[str, float]] = []
            for i, h in enumerate(haz_list):
                h = h.lower()
                if h in self.INV_HAZARD_TO_CODE:
                    out_list.append({"shape": float(p_cpu["shape"][i]), "scale": float(p_cpu["scale"][i])})
                else:
                    raise ValueError(f"Unsupported inverse hazard distribution: {h}")
            return out_list

        sampled_params_to_save = {
            "model": model,
            "inv_hazards": inv_hazards,
            "inv_hazards_C": inv_hazards_C,
            "params": row_params_list(inv_hazards, params_cpu),
            "params_C": row_params_list(inv_hazards_C, params_C_cpu),
            "q_star": float(q_star.detach().cpu().item()),
            "q_star_C": float(q_star_C.detach().cpu().item()),
            "t_star": float(t_star.detach().cpu().item()),
            "C": float(C_admin.detach().cpu().item()) if C_admin is not None else float("inf"),
            "C_percent": float(C_percent.detach().cpu().item()),
            "cov_indep_cens": bool(cov_indep_cens),
            "censoring_dependence": str(dep),
            "admin_type1": bool(use_type1),
            "admin_type2": bool(use_type2),
            "q_type1": float(q_type1.detach().cpu().item()),
            "random_cens": bool(use_random),
            "stagger_entry": bool(stagger),
            "accrual_frac": float(accrual_frac),
            "accrual_shape": float(accrual_shape),
            "r_min": float(r_frac.min().detach().cpu().item()) if r_frac is not None else 1.0,
            "r_mean": float(r_frac.mean().detach().cpu().item()) if r_frac is not None else 1.0,
            "event_frac": float(E.double().mean()),
            **admin_info,
        }

        return X, y_out, sampled_params_to_save
