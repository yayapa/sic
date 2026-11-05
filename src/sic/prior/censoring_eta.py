"""Censoring-dependence helpers: regime draw + η_C = f(X) via non-causal SCM."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor

from .hp_sampling import HpSamplerList
from .mlp_scm import MLPSCM
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from .tree_scm import TreeSCM

DEP_INDEP = "indep"
DEP_COVDEP = "covdep"
DEP_INFO = "info"


def sample_censoring_dependence(
    p_indep: float,
    p_info: float,
    *,
    rng: Optional[Union[np.random.Generator, torch.Generator]] = None,
    device: str = "cpu",
) -> str:
    """Exclusive draw: indep / info / covdep (remainder).

    Requires ``p_indep + p_info <= 1``. Remainder mass is ``covdep``.
    """
    p_indep = float(max(p_indep, 0.0))
    p_info = float(max(p_info, 0.0))
    if p_indep + p_info > 1.0 + 1e-9:
        raise ValueError(f"p_indep + p_info = {p_indep + p_info} > 1")

    if isinstance(rng, np.random.Generator):
        u = float(rng.random())
    elif isinstance(rng, torch.Generator):
        u = float(torch.rand((), generator=rng, device=device).cpu().item())
    else:
        u = float(np.random.random())

    if u < p_indep:
        return DEP_INDEP
    if u < p_indep + p_info:
        return DEP_INFO
    return DEP_COVDEP


def active_feature_matrix(X: Tensor) -> Tensor:
    """Drop near-zero-std (pad) columns; keep at least one column."""
    X = torch.as_tensor(X)
    if X.ndim != 2 or X.shape[1] == 0:
        return X
    std = X.float().std(dim=0)
    idx = torch.nonzero(std > 1e-8, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return X[:, :1]
    return X[:, idx]


def zscore_clamp(y: Tensor, cap: float = 5.0) -> Tensor:
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    mu = y.mean(0, keepdim=True)
    sd = y.std(0, keepdim=True).clamp_min(1e-8)
    return ((y - mu) / sd).clamp(-cap, cap)


def _resolve_sampled(sampled: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v() if callable(v) else v) for k, v in sampled.items()}


@torch.no_grad()
def eta_c_from_x(
    X: Tensor,
    M: int,
    *,
    prior_type: str = "mlp_scm",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    scm_sampled_hp: Optional[Dict[str, Any]] = None,
    scm_fixed_hp: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Build η_C (N, M) as a fresh non-causal SCM map of observed X.

    Architecture HPs are drawn from ``scm_sampled_hp`` (same families as the
    event SCM). ``is_causal`` is forced False so ``y = f(causes)`` with
    ``causes = X_active``.
    """
    X = torch.as_tensor(X, device=device, dtype=dtype)
    N = X.shape[0]
    M = int(M)
    if M <= 0 or N == 0:
        return torch.zeros(N, max(M, 0), device=device, dtype=dtype)

    X_active = active_feature_matrix(X).to(device=device, dtype=dtype)
    d = int(X_active.shape[1])

    sampled_cfg = scm_sampled_hp if scm_sampled_hp is not None else DEFAULT_SAMPLED_HP
    fixed = dict(scm_fixed_hp if scm_fixed_hp is not None else DEFAULT_FIXED_HP)
    drawn = _resolve_sampled(HpSamplerList(sampled_cfg, device=device).sample())

    # Strip survival-only keys that are not SCM ctor args (harmless via **kwargs
    # but keep the dict lean). Force non-causal map of X.
    scm_kwargs = {**fixed, **drawn}
    for k in (
        "reg2surv", "reg2surv_param", "reg2surv_discrete", "reg2surv_discrete_param",
        "mix_probs", "balanced", "multiclass_ordered_prob", "cat_prob", "max_categories",
        "scale_by_max_features", "permute_features", "permute_labels",
    ):
        scm_kwargs.pop(k, None)

    scm_kwargs.update(
        seq_len=N,
        num_features=d,
        num_causes=d,
        num_outputs=M,
        is_causal=False,
        device=device,
    )

    use_tree = str(prior_type).lower() in ("tree_scm", "tree")
    prior_cls = TreeSCM if use_tree else MLPSCM
    scm = prior_cls(**scm_kwargs)
    _X_out, y_C = scm.forward(causes=X_active)
    if isinstance(y_C, tuple):
        y_C = y_C[0]
    y_C = torch.as_tensor(y_C, device=device, dtype=dtype)
    if y_C.ndim == 1:
        y_C = y_C.unsqueeze(-1)
    if y_C.shape[1] != M:
        # Tree/MLP edge cases: pad or trim to M
        if y_C.shape[1] < M:
            y_C = torch.nn.functional.pad(y_C, (0, M - y_C.shape[1]))
        else:
            y_C = y_C[:, :M]
    return zscore_clamp(y_C)
