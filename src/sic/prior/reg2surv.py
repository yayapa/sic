from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from typing_extensions import Literal

Kind = Literal["unit", "positive", "real", "bounds"]
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
    'unit'     -> U ∈ (0,1) (open interval; safe for logs)
    'positive' -> Log-uniform on [1e-8, 1e8]
    'real'     -> Logistic(0,1): log(u/(1-u))
    'bounds'   -> Uniform on (lo,hi)
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
        if kind not in ("unit", "positive", "real", "bounds"):
            raise ValueError("kind must be 'unit', 'positive', 'real', or 'bounds'")
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

    def _u_open01(self, shape: Tuple[int, ...]) -> torch.Tensor:
        u = torch.rand(shape, device=self.device, dtype=self.dtype, generator=self._g)
        return u.clamp_(min=EPS, max=1 - EPS)

    def sample(self, n: int) -> torch.Tensor:
        if self.kind == "unit":
            return self._u_open01((n,))
        if self.kind == "positive":
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
    # shape == eta, scale == b
    small = torch.abs(scale) < 1e-10
    out_exp = y / (shape + EPS)
    c = scale / (shape + EPS)
    out_g = torch.log1p(c * y) / (scale + EPS)
    return torch.where(small, out_exp, out_g)

def _Hinv_lognormal(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == sigma, scale == exp(mu)

    u = 1.0 - torch.exp(-y)
    u = u.clamp(min=EPS, max=1.0 - EPS)
    z = torch.special.ndtri(u)
    return torch.exp(scale + shape * z)
def _Hinv_loglogistic(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    w = torch.expm1(y)
    inva = 1.0 / (shape + EPS)
    return scale * torch.pow(torch.clamp(w, min=0.0), inva)

def _Hinv_birnbaum_saunders(y: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # shape == alpha, scale == beta
    u = 1.0 - torch.exp(-y)
    u = u.clamp(min=EPS, max=1.0 - EPS)
    z = torch.special.ndtri(u)
    az = shape * z
    r = 0.5 * (az + torch.sqrt(az * az + 4.0))
    return scale * r * r


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
          * Lognormal:          shape=sigma, scale=exp(mu)
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
        self.tstar_sampler = NumericalSampler(
            "bounds",
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            bounds=(self.hp["t_star"]["min"], self.hp["t_star"]["max"]),
            dtype=self.dtype,
        )
        self.C_sampler = NumericalSampler(
            "bounds",
            seed=self.seed,
            device=self.device,
            generator=self.rng,
            bounds=(self.hp["censoring_rate"]["min"], self.hp["censoring_rate"]["max"]),
            dtype=self.dtype,
        )

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
        return (y - mu) / sd

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
            bt = b * t
            small = torch.abs(b) < 1e-10
            eta_small = neglogq / t
            eta_big = (b * neglogq) / (torch.expm1(bt) + EPS)
            eta = torch.where(small, eta_small.expand_as(b), eta_big)
            shape[m_gomp] = eta + EPS
            scale[m_gomp] = b + EPS

        # Lognormal: shape=sigma ~ U(0.3,1.5), scale=exp(mu)=t*exp(-sigma*z_anchor)
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
            az = alpha * z_anchor
            r = 0.5 * (az + torch.sqrt(az * az + 4.0))
            beta = t / (r * r + EPS)
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

    def forward(self, X: Tensor, y: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], dict]:
        if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"Input shapes mismatch or incorrect dims. X: {X.shape}, y: {y.shape}")

        N = X.shape[0]

        # anchors (same placement/structure)
        q_star = self.q_star_sampler.sample(1)[0].clamp(min=EPS, max=1.0 - EPS)
        t_star = self.tstar_sampler.sample(1)[0].clamp_min(EPS)

        # censoring fraction and model (same structure)
        C_percent = self.C_sampler.sample(1)[0]
        model = self.model_sampler.sample(1)

        # risk score normalization
        eta = self._get_eta(y)

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
        params_C_t = self._sample_params_batched(haz_codes_C, q_star=q_star, t_star=t_star)

        T = self._compute_T_batched(model=model, haz_codes=haz_codes, params=params_t, U=U, eta=eta)

        eta_C = torch.zeros_like(eta)
        C_times = self._compute_T_batched(model=model, haz_codes=haz_codes_C, params=params_C_t, U=U_C, eta=eta_C)

        # admin censoring quantile (GPU)
        q = (1.0 - C_percent).clamp(min=EPS, max=1.0 - EPS)
        C_admin = torch.quantile(T, q)

        # censoring
        E = (T <= C_times).to(torch.int32)
        T_obs = torch.minimum(T, C_times)

        E = E * (T_obs <= C_admin).to(torch.int32)
        T_obs = torch.minimum(T_obs, C_admin)

        y_out = (T_obs, E)

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
            "t_star": float(t_star.detach().cpu().item()),
            "C": float(C_admin.detach().cpu().item()),
            "C_percent": float(C_percent.detach().cpu().item()),
        }

        return X, y_out, sampled_params_to_save
