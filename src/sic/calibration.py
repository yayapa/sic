"""Post-training conformal calibration of survival curves.

Two methods are provided, both operating purely on predicted survival curves
(numpy arrays) so they are independent of the model head:

* ``cipot`` -- CiPOT (Qi et al., NeurIPS 2024), probability-space adjustment.
* ``csd``   -- CSD   (Qi et al., ICML 2024), quantile-space adjustment.

Adapted from the SIC research code (``PostCalibrationInferencer``), which in
turn follows the reference implementations of CiPOT and CSD.
"""

from __future__ import annotations

import numpy as np

from SurvivalEVAL.Evaluations.util import (
    predict_prob_from_curve,
    quantile_to_survival,
    check_monotonicity,
    make_monotonic,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeierArea


def default_quantile_levels(n_quantiles: int = 9) -> np.ndarray:
    return np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)


def _survival_to_quantile(surv_prob, time_coords, quantile_levels):
    from scipy.interpolate import PchipInterpolator

    cdf = 1.0 - surv_prob
    slope = cdf[:, -1] / np.maximum(time_coords[:, -1], 1e-12)
    N = cdf.shape[0]
    qpred = np.empty((N, quantile_levels.shape[0]))

    for i in range(N):
        cdf_i = cdf[i, :]
        t_i = time_coords[i, :]
        cdf_unique, idx = np.unique(cdf_i, return_index=True)
        t_unique = t_i[idx]

        if len(cdf_unique) < 2:
            s = slope[i] if slope[i] > 0 else 1e-8
            qpred[i, :] = quantile_levels / s
            continue

        interp = PchipInterpolator(cdf_unique, t_unique, extrapolate=True)
        qpred[i, :] = interp(quantile_levels)

        beyond = quantile_levels > cdf_unique[-1]
        if np.any(beyond) and slope[i] > 0:
            qpred[i, beyond] = quantile_levels[beyond] / slope[i]

    return np.clip(qpred, 0, None)


def _pad_curves(S, time_grid):
    """Ensure survival curves start at (t=0, S=1). Returns (S_pad, tc_pad)."""
    N = S.shape[0]
    tc = np.repeat(time_grid[np.newaxis, :], N, axis=0)
    if time_grid[0] != 0:
        tc = np.concatenate([np.zeros((N, 1)), tc], axis=1)
        S = np.concatenate([np.ones((N, 1)), S], axis=1)
    else:
        S = S.copy()
        S[:, 0] = 1.0
    S = np.clip(S, 0, 1)
    return S, tc


def _quantiles_to_surv(quan_levels, quan_preds, time_grid, interpolation="Pchip"):
    quan_preds = np.clip(quan_preds, 0, None)
    if 0 not in quan_levels:
        quan_levels = np.insert(quan_levels, 0, 0)
        quan_preds = np.insert(quan_preds, 0, 0, axis=1)
    quan_preds = np.maximum.accumulate(quan_preds, axis=1)
    quan_preds = quan_preds + np.arange(quan_preds.shape[1]) * 1e-10

    try:
        S_adj = quantile_to_survival(quan_levels, quan_preds, time_grid, interpolation)
    except (AssertionError, ValueError):
        S_adj = quantile_to_survival(quan_levels, quan_preds, time_grid, "Linear")

    if not check_monotonicity(S_adj):
        S_adj = make_monotonic(S_adj, time_grid, method="ceil")
    return np.clip(S_adj, 0, 1)


# --------------------------------------------------------------------------- #
# CiPOT
# --------------------------------------------------------------------------- #
def cipot_calibrate(S_cal, time_grid, t_cal, e_cal, n_sample=1000, interpolation="Pchip"):
    S_pad, tc_pad = _pad_curves(S_cal, time_grid)
    N = S_pad.shape[0]
    con_scores = np.array([
        predict_prob_from_curve(S_pad[i], tc_pad[i], t_cal[i], interpolation)
        for i in range(N)
    ])
    e_bool = e_cal.astype(bool)
    con_event = np.repeat(con_scores[e_bool], n_sample)
    grid = np.linspace(0, 1, n_sample).reshape(n_sample, 1)
    con_censor = (grid * con_scores[~e_bool].reshape(1, -1)).reshape(-1)
    all_scores = np.concatenate([con_event, con_censor])
    return -np.sort(-all_scores)


def _cipot_apply_inverse(nc, quantile_levels):
    nc = np.sort(nc, axis=0)
    index = np.ceil((1 - quantile_levels) * (nc.shape[0] + 1)) - 1
    index = index.astype(int).clip(0, nc.shape[0] - 1)
    return nc[index]


def cipot_predict(S_test, time_grid, cal_scores, quantile_levels, interpolation="Pchip"):
    S_pad, tc_pad = _pad_curves(S_test, time_grid)
    adj_pct = _cipot_apply_inverse(cal_scores, quantile_levels)
    adj_quan = np.clip(1 - adj_pct, 0, 1 - 1e-8)
    quan_preds = _survival_to_quantile(S_pad, tc_pad, adj_quan)
    return _quantiles_to_surv(quantile_levels, quan_preds, time_grid, interpolation)


# --------------------------------------------------------------------------- #
# CSD
# --------------------------------------------------------------------------- #
def _csd_decensor(t_cal, e_cal, t_train, e_train):
    km = KaplanMeierArea(t_train, e_train.astype(bool))
    best_guesses = t_cal.copy().astype(float)
    censor_mask = ~e_cal.astype(bool)
    if censor_mask.any():
        censor_times = t_cal[censor_mask]
        conditional_mean = km.best_guess(censor_times)
        too_late = censor_times > km.km_linear_zero
        conditional_mean[too_late] = censor_times[too_late]
        best_guesses[censor_mask] = conditional_mean
    return best_guesses


def csd_calibrate(S_cal, time_grid, t_cal, e_cal, t_train, e_train, quantile_levels):
    t_decensored = _csd_decensor(t_cal, e_cal, t_train, e_train)
    S_pad, tc_pad = _pad_curves(S_cal, time_grid)
    quan_preds = _survival_to_quantile(S_pad, tc_pad, quantile_levels)
    scores = quan_preds - t_decensored[:, np.newaxis]
    return np.sort(scores, axis=0)[::-1]


def _csd_apply_inverse(nc, quantile_levels):
    nc_asc = np.sort(nc, axis=0)
    index = np.ceil((1 - quantile_levels) * (nc.shape[0] + 1)) - 1
    index = index.astype(int).clip(0, nc.shape[0] - 1)
    q_idx = np.arange(quantile_levels.shape[0])
    return nc_asc[index, q_idx]


def csd_predict(S_test, time_grid, cal_scores, quantile_levels, interpolation="Pchip"):
    S_pad, tc_pad = _pad_curves(S_test, time_grid)
    quan_preds = _survival_to_quantile(S_pad, tc_pad, quantile_levels)
    error_dist = _csd_apply_inverse(cal_scores, quantile_levels)
    quan_preds_adj = quan_preds - error_dist
    return _quantiles_to_surv(quantile_levels, quan_preds_adj, time_grid, interpolation)
