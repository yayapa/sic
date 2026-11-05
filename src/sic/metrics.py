"""Survival evaluation metrics.

``evaluate_survival`` computes the sabaselines-aligned metric suite on a
predicted survival-curve DataFrame:

* time-dependent concordance (adj-Antolini and Antolini)
* integrated Brier score on unique test times (pycox + SurvivalEVAL)
* q90-horizon IBS (pycox + SurvivalEVAL; re-queried ``surv_q90`` when given)
* D-calibration p-value (``S`` clipped to ``[0, 1]``)
* weighted MAE with pseudo-observations
* AUC / Brier / IPCW C-index at train event-time quantiles (``surv_pw`` when given)

``cross_validate`` re-queries the model on the unique-test, q90, and pointwise
grids (option B), matching sabaselines ``queries_arbitrary_times`` models.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from SurvivalEVAL.Evaluator import SurvivalEvaluator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .horizons import (
    clip_times_to_interval,
    make_ibs_q90_grid,
    make_pointwise_times,
    unique_sorted_times,
)


def _d_calibration_pvalue(
    surv_np: np.ndarray,
    time_index: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    train_durations: np.ndarray,
    train_events: np.ndarray,
    num_bins: int = 10,
) -> float:
    """D-calibration p-value with ``S`` clipped to ``[0, 1]`` (sabaselines)."""
    surv_clip = np.clip(np.asarray(surv_np, dtype=np.float64), 0.0, 1.0)
    eval_dcal = SurvivalEvaluator(
        surv_clip,
        time_index,
        durations,
        events,
        train_durations,
        train_events,
    )
    pvalue, _ = eval_dcal.d_calibration(num_bins=num_bins)
    return float(pvalue)


def evaluate_survival(
    surv_df: pd.DataFrame,
    t_test,
    e_test,
    t_train,
    e_train,
    surv_q90: pd.DataFrame | None = None,
    surv_pw: pd.DataFrame | None = None,
) -> dict:
    """Evaluate a survival-curve DataFrame (index = times, columns = subjects).

    When ``surv_q90`` / ``surv_pw`` are provided (sabaselines re-query path),
    ``ibs_q90`` and pointwise metrics use those frames instead of interpolating
    from ``surv_df``.
    """
    t_test = np.asarray(t_test, dtype=float)
    e_test = np.asarray(e_test, dtype=int)
    t_train = np.asarray(t_train, dtype=float)
    e_train = np.asarray(e_train, dtype=int)

    unique_test_times = unique_sorted_times(t_test)
    if len(unique_test_times) == 0:
        return {"cindex_td": np.nan, "cindex_antolini": np.nan, "ibs": np.nan}

    try:
        evaluator = EvalSurv(surv_df, t_test, e_test.astype(bool), censor_surv="km")
        cindex_td = float(evaluator.concordance_td(method="adj_antolini"))
        cindex_antolini = float(evaluator.concordance_td(method="antolini"))
        ibs = float(evaluator.integrated_brier_score(unique_test_times))
    except Exception:
        evaluator = None
        cindex_td = cindex_antolini = ibs = np.nan

    metrics: dict = {
        "cindex_td": cindex_td,
        "cindex_antolini": cindex_antolini,
        "ibs": ibs,
    }

    # Pure numpy: keep even if SurvivalEVAL / sksurv later fail.
    pointwise_times = make_pointwise_times(t_train, e_train)
    metrics["pointwise_times"] = [float(t) for t in pointwise_times]

    max_train_time = float(t_train.max()) if t_train.size else np.nan
    ibs_grid = np.asarray([], dtype=np.float64)
    if t_train.size and e_train.size and np.isfinite(max_train_time):
        ibs_grid = make_ibs_q90_grid(
            t_train, e_train, percentile=90, n_points=100, max_time=max_train_time,
        )
        ibs_grid = clip_times_to_interval(ibs_grid, None, max_train_time)

    # q90 IBS: re-queried curve when provided, else interpolate from surv_df.
    evaluator_q90 = evaluator
    if surv_q90 is not None and len(ibs_grid) > 0:
        try:
            evaluator_q90 = EvalSurv(
                surv_q90, t_test, e_test.astype(bool), censor_surv="km"
            )
        except Exception:
            evaluator_q90 = None
    if len(ibs_grid) > 0 and evaluator_q90 is not None:
        try:
            metrics["ibs_q90"] = float(evaluator_q90.integrated_brier_score(ibs_grid))
        except Exception:
            metrics["ibs_q90"] = np.nan
    else:
        metrics["ibs_q90"] = np.nan

    try:
        surv_np = surv_df.to_numpy().T  # (N, M)
        time_index = surv_df.index.to_numpy()

        labels_train = Surv.from_arrays(e_train.astype(bool), t_train)
        test_mask = t_test <= max_train_time
        T_test_trunc = t_test[test_mask]
        E_test_trunc = e_test[test_mask]
        labels_test_trunc = Surv.from_arrays(E_test_trunc.astype(bool), T_test_trunc)
        min_test_time = float(T_test_trunc.min()) if len(T_test_trunc) else np.nan
        max_test_time = float(T_test_trunc.max()) if len(T_test_trunc) else np.nan

        eval_se = SurvivalEvaluator(
            surv_np, time_index, t_test, e_test, t_train, e_train,
        )
        try:
            metrics["ibs_survivaleval"] = float(
                eval_se.integrated_brier_score(target_times=unique_test_times)
            )
        except Exception:
            metrics["ibs_survivaleval"] = np.nan
        if len(ibs_grid) > 0:
            try:
                if surv_q90 is not None:
                    eval_se_q90 = SurvivalEvaluator(
                        surv_q90.to_numpy().T,
                        surv_q90.index.to_numpy(),
                        t_test,
                        e_test,
                        t_train,
                        e_train,
                    )
                    metrics["ibs_survivaleval_q90"] = float(
                        eval_se_q90.integrated_brier_score(target_times=ibs_grid)
                    )
                else:
                    metrics["ibs_survivaleval_q90"] = float(
                        eval_se.integrated_brier_score(target_times=ibs_grid)
                    )
            except Exception:
                metrics["ibs_survivaleval_q90"] = np.nan
        else:
            metrics["ibs_survivaleval_q90"] = np.nan

        try:
            metrics["weighted_MAE_PO"] = float(eval_se.mae(method="Pseudo_obs", weighted=True))
        except Exception:
            metrics["weighted_MAE_PO"] = np.nan
        try:
            metrics["d_cal"] = _d_calibration_pvalue(
                surv_np, time_index, t_test, e_test, t_train, e_train,
            )
        except Exception:
            metrics["d_cal"] = np.nan

        surv_pw_np = surv_np
        time_index_pw = time_index
        eval_se_pw = eval_se
        if surv_pw is not None and len(pointwise_times) > 0:
            surv_pw_np = surv_pw.to_numpy().T
            time_index_pw = surv_pw.index.to_numpy()
            eval_se_pw = SurvivalEvaluator(
                surv_pw_np, time_index_pw, t_test, e_test, t_train, e_train,
            )

        auc_scores: list[float] = []
        bs_scores: list[float] = []
        c_index_scores: list[float] = []
        for eval_time in pointwise_times:
            idx = int(np.argmin(np.abs(time_index_pw - eval_time)))
            surv_at_t = surv_pw_np[test_mask, idx] if test_mask.any() else np.asarray([])
            est_risk = 1.0 - surv_at_t
            in_range = (
                test_mask.any()
                and np.isfinite(min_test_time)
                and np.isfinite(max_test_time)
                and min_test_time <= float(eval_time) < max_test_time
            )
            if in_range:
                try:
                    auc_val = cumulative_dynamic_auc(
                        labels_train, labels_test_trunc, est_risk, times=[float(eval_time)],
                    )[0][0]
                except Exception:
                    auc_val = np.nan
                try:
                    ci_val = concordance_index_ipcw(
                        labels_train, labels_test_trunc, est_risk, tau=float(eval_time),
                    )[0]
                except Exception:
                    ci_val = np.nan
            else:
                auc_val = np.nan
                ci_val = np.nan
            auc_scores.append(float(auc_val))
            try:
                bs_scores.append(float(eval_se_pw.brier_score(float(eval_time))))
            except Exception:
                bs_scores.append(np.nan)
            c_index_scores.append(float(ci_val))

        metrics["auc_scores"] = auc_scores
        metrics["bs_scores"] = bs_scores
        metrics["c_index_scores"] = c_index_scores
        metrics["td_auc"] = float(np.nanmean(auc_scores)) if auc_scores else np.nan
    except Exception:
        for key in (
            "td_auc",
            "ibs_q90",
            "ibs_survivaleval",
            "ibs_survivaleval_q90",
            "weighted_MAE_PO",
            "d_cal",
        ):
            metrics.setdefault(key, np.nan)
        for key in ("auc_scores", "bs_scores", "c_index_scores"):
            metrics.setdefault(key, [])

    return metrics


# Scalar metrics (one value per fold) and the quantile levels used for the
# per-time-point metrics returned by ``evaluate_survival``.
SCALAR_METRICS = [
    "cindex_td",
    "cindex_antolini",
    "ibs",
    "ibs_q90",
    "ibs_survivaleval",
    "ibs_survivaleval_q90",
    "td_auc",
    "d_cal",
    "weighted_MAE_PO",
]
QUANTILE_LEVELS = (0.25, 0.5, 0.75)

def _jsonable(obj):
    """Convert numpy scalars/arrays so ``json.dumps`` succeeds."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def prepare_fold(x_df, t, e, num_cols, train_idx, test_idx):
    """Median-impute and scale ``num_cols``, fitted on train only.

    Returns pandas DataFrames for X (SIC ``fit_predict_surv`` takes ArrayLike).
    Columns in ``num_cols`` that are missing from the frame are skipped.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_df = x_df.iloc[train_idx].copy()
    test_df = x_df.iloc[test_idx].copy()

    numeric_cols = [col for col in num_cols if col in train_df.columns]
    if numeric_cols:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        train_mat = pipe.fit_transform(train_df[numeric_cols])
        test_mat = pipe.transform(test_df[numeric_cols])
        train_df.loc[:, numeric_cols] = train_mat
        test_df.loc[:, numeric_cols] = test_mat

    t = np.asarray(t)
    e = np.asarray(e)
    return {
        "train": (train_df, t[train_idx], e[train_idx]),
        "test": (test_df, t[test_idx], e[test_idx]),
    }


def metrics_to_row(res: dict) -> dict:
    """Flatten an ``evaluate_survival`` result into a single flat record.

    The per-quantile lists (``auc_scores`` / ``bs_scores`` / ``c_index_scores``)
    are expanded into columns like ``auc@0.25``, ``bs@0.5``, ``cindex@0.75`` so
    the whole result fits one row of a CSV.
    """
    row = {k: res[k] for k in SCALAR_METRICS}
    for name, key in [("auc", "auc_scores"), ("bs", "bs_scores"), ("cindex", "c_index_scores")]:
        for q, val in zip(QUANTILE_LEVELS, res.get(key, [])):
            row[f"{name}@{q}"] = val
    return row


def cross_validate(
    model,
    datasets,
    k_folds: int = 5,
    seed: int = 42,
    csv_path=None,
    verbose: bool = True,
    model_name=None,
    json_out=None,
    curve_dir=None,
) -> pd.DataFrame:
    """Run k-fold CV of a fitted :class:`sic.SIC` model over several datasets.

    Sabaselines-aligned re-query: each fold predicts on (1) unique test times,
    (2) the train q90 IBS grid, and (3) train event-time quantiles for pointwise
    metrics. DeepHit heads interpolate from training bins onto each grid.

    With ``curve_dir`` every fold's predicted curves on the IBS grid are saved to
    ``<curve_dir>/<dataset>_fold<k>.npz`` (``times``, ``surv_hat``, ``test_idx``,
    ``t``, ``e``, ``tau``) so they can be redrawn later.

    Parameters
    ----------
    model : sic.SIC
        An estimator; ``fit_predict_surv`` is called per fold (up to 3 grids).
    datasets : list of str
        Dataset names understood by :func:`sic.load_dataset`.
    csv_path : path, optional
        If given, the tidy per-fold results are written here as CSV.
    model_name : str, optional
        Name stored under ``datasets[].models[].name`` in the JSON payload
        (typically the resolved survival head). Defaults to
        ``model.survival_head`` when present.
    json_out : path, optional
        If given, write sabaselines-shaped ``nested_cv_results.json``.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, fold) with all flattened metrics.
    """
    from sklearn.model_selection import KFold

    from .train.datasets import load_dataset
    from .api import make_y

    resolved_name = model_name if model_name is not None else getattr(model, "survival_head", None)
    if curve_dir is not None:
        curve_dir = Path(curve_dir)
        curve_dir.mkdir(parents=True, exist_ok=True)
    json_datasets = []
    rows = []
    for dataset in datasets:
        out = load_dataset(dataset)
        x, t, e = out["x"], out["t"], out["e"]
        num_cols = out["num_cols"]
        n_samples = int(len(x))
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        t_start = time.time()

        fold_summaries = []
        fold_metrics: dict[str, list] = {}
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x)):
            fold_data = prepare_fold(x, t, e, num_cols, train_idx, test_idx)
            x_tr, t_tr, e_tr = fold_data["train"]
            x_te, t_te, e_te = fold_data["test"]

            et = unique_sorted_times(t_te)
            max_tr = float(np.asarray(t_tr).max())
            ibs_grid = clip_times_to_interval(
                make_ibs_q90_grid(
                    t_tr, e_tr, percentile=90, n_points=100, max_time=max_tr,
                ),
                None,
                max_tr,
            )
            pw = make_pointwise_times(t_tr, e_tr)
            y_tr = make_y(t_tr, e_tr)

            try:
                surv = model.fit_predict_surv(
                    x_tr, y_tr, x_te,
                    eval_times=et.tolist(),
                )
                surv_q90 = None
                if len(ibs_grid) > 0:
                    surv_q90 = model.fit_predict_surv(
                        x_tr, y_tr, x_te,
                        eval_times=ibs_grid.tolist(),
                    )
                surv_pw = None
                if len(pw) > 0:
                    surv_pw = model.fit_predict_surv(
                        x_tr, y_tr, x_te,
                        eval_times=pw.tolist(),
                    )
                res = evaluate_survival(
                    surv, t_te, e_te, t_tr, e_tr,
                    surv_q90=surv_q90, surv_pw=surv_pw,
                )
                row = metrics_to_row(res)
                if curve_dir is not None and surv_q90 is not None:
                    np.savez_compressed(
                        curve_dir / f"{dataset}_fold{fold_idx}.npz",
                        test_idx=test_idx, times=ibs_grid, surv_hat=surv_q90.to_numpy(),
                        t=t_te, e=e_te, tau=max_tr,
                    )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[{dataset} fold {fold_idx}] failed: {exc}")
                res = {k: np.nan for k in SCALAR_METRICS}
                row = res
            row = {"dataset": dataset, "n_samples": n_samples, "fold": fold_idx, **row}
            rows.append(row)
            metrics = _jsonable(res)
            fold_summaries.append({
                "fold": fold_idx,
                "best_params": {},
                "val_score": None,
                "metrics": metrics,
            })
            # _jsonable first: np.float32/np.float64 are not Python float (NumPy 2).
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    fold_metrics.setdefault(key, []).append(value)

        if verbose:
            sub = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
            n_miscal = int((sub["d_cal"] < 0.05).sum())
            print(
                f"{dataset:11s} "
                f"C-index {np.nanmean(sub['cindex_td']):.3f}+/-{np.nanstd(sub['cindex_td']):.3f}  "
                f"IBS {np.nanmean(sub['ibs']):.3f}+/-{np.nanstd(sub['ibs']):.3f}  "
                f"D-cal {np.nanmean(sub['d_cal']):.3f}+/-{np.nanstd(sub['d_cal']):.3f}  "
                f"miscalibrated {n_miscal}/{sub['d_cal'].notna().sum()} folds"
            )

        json_datasets.append({
            "name": dataset,
            "models": [{
                "name": resolved_name,
                "time_taken": time.time() - t_start,
                "folds": fold_summaries,
                "summary": {
                    m: {
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values)),
                    }
                    for m, values in fold_metrics.items()
                },
            }],
        })

    df = pd.DataFrame(rows)
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
    if json_out is not None:
        payload = {
            "config_path": None,
            "defaults": {
                "outer_folds": k_folds,
                "random_seed": seed,
                "ibs_max_horizon_percentile": 90,
                "ibs_n_horizon_points": 100,
                "pointwise_time_percentiles": [0.25, 0.5, 0.75],
            },
            "datasets": json_datasets,
        }
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return df


def attach_n_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-dataset ``n_samples`` when missing (e.g. CSVs from older ``cross_validate``)."""
    if "n_samples" in df.columns:
        return df
    from .train.datasets import load_dataset

    out = df.copy()
    n_by_ds = {
        ds: len(load_dataset(ds)["x"])
        for ds in out["dataset"].unique()
    }
    out["n_samples"] = out["dataset"].map(n_by_ds)
    return out


def summarize(
    df: pd.DataFrame,
    metrics=("cindex_td", "d_cal", "ibs"),
    d_cal_alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate a per-fold results frame into ``mean +/- std`` per dataset.

    Returns a DataFrame indexed by dataset with one ``"mean +/- std"`` string
    column per requested metric (plus numeric ``<metric>_mean`` / ``_std``).
    When ``d_cal`` is among the metrics, a ``d_cal_miscalibrated`` column
    reports how many folds were miscalibrated (p-value below ``d_cal_alpha``)
    as ``"k/n"``.
    """
    metrics = [m for m in metrics if m in df.columns]
    # sort=False keeps first-appearance order (e.g. list_datasets() / CLI order).
    grouped = df.groupby("dataset", sort=False)[list(metrics)]
    mean, std = grouped.mean(), grouped.std()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = [f"{mu:.3f} +/- {sd:.3f}" for mu, sd in zip(mean[m], std[m])]
        out[f"{m}_mean"] = mean[m]
        out[f"{m}_std"] = std[m]
    if "d_cal" in metrics:
        by_ds = df.groupby("dataset", sort=False)["d_cal"]
        miscal = by_ds.apply(lambda s: int((s < d_cal_alpha).sum()))
        valid = by_ds.apply(lambda s: int(s.notna().sum()))
        out["d_cal_miscalibrated"] = [f"{k}/{n}" for k, n in zip(miscal, valid)]
    return out
