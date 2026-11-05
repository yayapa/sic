"""Administrative censoring with staggered study entry.

Shared by the continuous (:mod:`reg2surv`) and discrete
(:mod:`reg2surv_discrete`) priors.

Both branches pick a single study horizon tau on the study time axis:

  type 1  a fixed closing date tau_1 = Q_q(T), set from the true event-time
          distribution, so it ignores loss to follow-up and accrual and the
          number of observed events is random;
  type 2  an event-driven stop at the d-th *observed* event, d = ceil((1 - c) N)
          fixed in advance. Randomly censored subjects never count. If fewer
          than d events are observable the study runs to the end and the branch
          does not bind.

Subjects enter the study at staggered times. With accrual fraction a and
accrual shape k,

    r_i = 1 - a * U_i^(1/k),   U_i ~ U(0, 1),
    F_i = r_i * tau,           T_obs_i = min(T_i, C_i, F_i).

k = 1 is uniform accrual; k < 1 front-loads entry so most subjects are followed
almost the whole study. 1 - a is the minimum follow-up. Subject i's event is
seen before the stop iff T_i <= r_i * tau, i.e. T_i / r_i <= tau, so the type 2
horizon is the d-th smallest T_i / r_i among observed events.

Entry lives on the study axis only: every subject is still followed from their
own time origin, so S(0) = 1 and there is no left truncation. Because r_i is
drawn independently of (T_i, C_i, X_i), F_i is an independent censoring
variable and the usual likelihood is unchanged. With r = None every subject
shares the horizon.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

EPS = 1e-6


def sample_follow_up_fractions(
    np_rng: np.random.Generator,
    n: int,
    accrual_frac: float,
    accrual_shape: float,
    device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    """Draw r_i = 1 - a * U_i^(1/k), the share of the study subject i is followed for.

    Returns None for a <= 0 (everyone enters at time 0). numpy is the sampler
    because torch.distributions does not accept a generator.
    """
    a, k = float(accrual_frac), float(accrual_shape)
    if not (0.0 <= a <= 1.0 and k > 0.0):
        raise ValueError(f"need 0 <= accrual_frac <= 1 and accrual_shape > 0, got a={a}, k={k}")
    if a <= 0.0:
        return None
    # numpy's uniform is [0, 1), so r > 1 - a >= 0 and T / r stays finite.
    U = np_rng.uniform(0.0, 1.0, size=int(n))
    return torch.as_tensor(1.0 - a * U ** (1.0 / k), device=device, dtype=dtype)


def apply_admin_censoring(
    T_obs: Tensor,
    E: Tensor,
    *,
    T_event: Optional[Tensor] = None,
    r: Optional[Tensor] = None,
    use_type1: bool = False,
    use_type2: bool = False,
    q_type1: Union[float, Tensor] = 0.5,
    c_pct: Union[float, Tensor] = 0.5,
) -> Tuple[Tensor, Tensor, Optional[Tensor], dict]:
    """Apply the optional administrative branches; the earliest cutoff wins.

    Parameters
    ----------
    T_obs, E
        Observed times and event indicators after random censoring.
    T_event
        True event times. Sets the type 1 horizon; falls back to T_obs if None.
    r
        Per-subject follow-up fractions in (0, 1]. None means every subject
        shares the horizon.
    q_type1
        Quantile of the true event times at which the type 1 study closes.
    c_pct
        Type 2 stops at the d-th observed event, d = ceil((1 - c_pct) * N).

    Returns
    -------
    (T_obs, E, tau, info). tau is the shared horizon, or None if no branch
    bound. info records which branch bound and whether type 2 was reachable.
    """
    info = {"type1_bound": False, "type2_bound": False, "type2_reached": False, "type2_d": 0}
    if not (use_type1 or use_type2):
        return T_obs, E, None, info

    obs = T_obs.double()
    ref = (T_event if T_event is not None else T_obs).double()
    r64 = r.double() if r is not None else torch.ones_like(obs)
    observed = E.bool() & torch.isfinite(obs)

    tau1 = tau2 = None
    if use_type1:
        finite = torch.isfinite(ref)
        if finite.any():
            tau1 = torch.quantile(ref[finite], float(q_type1))
    if use_type2:
        c = min(max(float(c_pct), EPS), 1.0 - EPS)
        d = max(1, math.ceil((1.0 - c) * obs.numel()))
        info["type2_d"] = d
        n_obs = int(observed.sum())
        if n_obs >= d:
            info["type2_reached"] = True
            tau2 = (obs[observed] / r64[observed]).kthvalue(d).values

    if tau1 is None and tau2 is None:
        return T_obs, E, None, info
    if tau2 is None or (tau1 is not None and tau1 <= tau2):
        tau = tau1
        info["type1_bound"] = True
    else:
        tau = tau2
        info["type2_bound"] = True

    # Ratio space is the quantity tau2 was taken from, so the d-th event cannot
    # be flipped to censored by rounding r * tau.
    keep = obs / r64 <= tau
    E = E * keep.to(E.dtype)
    T_obs = torch.where(keep, obs, r64 * tau).to(T_obs.dtype)
    return T_obs, E, tau, info
