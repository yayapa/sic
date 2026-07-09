"""Survival evaluation metrics.

``evaluate_survival`` computes the full metric suite used in the SIC paper on a
predicted survival-curve DataFrame:

* time-dependent concordance (Antolini)
* integrated Brier score (pycox)
* D-calibration p-value
* weighted MAE with pseudo-observations
* AUC / Brier / IPCW C-index at the 0.25/0.5/0.75 time quantiles
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from SurvivalEVAL.Evaluator import SurvivalEvaluator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def evaluate_survival(surv_df: pd.DataFrame, t_test, e_test, t_train, e_train) -> dict:
    """Evaluate a survival-curve DataFrame (index = times, columns = subjects)."""
    time_grid = surv_df.index.to_numpy()
    t_test = np.asarray(t_test, dtype=float)
    e_test = np.asarray(e_test, dtype=int)
    t_train = np.asarray(t_train, dtype=float)
    e_train = np.asarray(e_train, dtype=int)

    eval_surv = EvalSurv(surv_df, t_test, e_test, censor_surv="km")
    ibs = eval_surv.integrated_brier_score(time_grid)
    c_index_td = eval_surv.concordance_td()

    surv_test = surv_df.to_numpy().T  # (N, M)

    labels_train = Surv.from_arrays(e_train.astype(bool), t_train)
    max_train_time = t_train.max()
    test_mask = t_test <= max_train_time
    labels_test = Surv.from_arrays(e_test[test_mask].astype(bool), t_test[test_mask])

    ev = SurvivalEvaluator(surv_test, time_grid, t_test, e_test, t_train, e_train)
    try:
        weighted_MAE_PO = ev.mae(method="Pseudo_obs", weighted=True)
    except Exception:
        weighted_MAE_PO = np.nan
    try:
        d_cal, _ = ev.d_calibration(num_bins=10)
    except Exception:
        d_cal = np.nan

    auc_scores, bs_scores, c_index_scores = [], [], []
    eval_time_indices = [int(len(time_grid) * q) for q in (0.25, 0.5, 0.75)]
    for eti in eval_time_indices:
        eval_time = time_grid[eti]
        est_risk = 1 - surv_test[test_mask, eti]
        try:
            auc = cumulative_dynamic_auc(labels_train, labels_test, est_risk, times=[eval_time])[0][0]
        except Exception:  # noqa: BLE001
            auc = np.nan
        auc_scores.append(auc)
        bs_scores.append(ev.brier_score(float(eval_time)))
        try:
            ci = concordance_index_ipcw(labels_train, labels_test, est_risk, tau=eval_time)[0]
        except Exception:  # noqa: BLE001
            ci = np.nan
        c_index_scores.append(ci)

    return {
        "cindex_td": c_index_td,
        "ibs": ibs,
        "d_cal": d_cal,
        "weighted_MAE_PO": weighted_MAE_PO,
        "auc_scores": auc_scores,
        "bs_scores": bs_scores,
        "c_index_scores": c_index_scores,
    }


# Scalar metrics (one value per fold) and the quantile levels used for the
# per-time-point metrics returned by ``evaluate_survival``.
SCALAR_METRICS = ["cindex_td", "ibs", "d_cal", "weighted_MAE_PO"]
QUANTILE_LEVELS = (0.25, 0.5, 0.75)


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
    post_training_calibration=None,
    csv_path=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run k-fold CV of a fitted :class:`sic.SIC` model over several datasets.

    Follows the original SIC evaluation protocol: survival curves are always
    evaluated on the per-fold **test time grid** (the unique sorted test
    durations). For DeepHit heads the model predicts on its training-time
    quantile bins (labtrans) and interpolates onto that grid; CoxPH and DeepEH
    predict on it directly.

    Parameters
    ----------
    model : sic.SIC
        An estimator; ``fit_predict_surv`` is called per fold.
    datasets : list of str
        Dataset names understood by :func:`sic.load_dataset`.
    post_training_calibration : {None, "cipot", "csd"}
        Forwarded to ``fit_predict_surv``.
    csv_path : path, optional
        If given, the tidy per-fold results are written here as CSV.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, fold) with all flattened metrics.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    from .train.datasets import load_dataset
    from .api import make_y

    rows = []
    for dataset in datasets:
        out = load_dataset(dataset, sequential=False)
        x, t, e = out["x"], out["t"], out["e"]
        num_cols = out["num_cols"]
        n_samples = int(len(x))
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x)):
            x_tr, x_te = x.iloc[train_idx].copy(), x.iloc[test_idx].copy()
            t_tr, t_te = t[train_idx], t[test_idx]
            e_tr, e_te = e[train_idx], e[test_idx]

            scaler = StandardScaler()
            x_tr[num_cols] = scaler.fit_transform(x_tr[num_cols])
            x_te[num_cols] = scaler.transform(x_te[num_cols])

            # Original protocol: evaluate on the test time grid.
            et = np.unique(np.sort(t_te)).tolist()

            try:
                surv = model.fit_predict_surv(
                    x_tr, make_y(t_tr, e_tr), x_te,
                    eval_times=et, post_training_calibration=post_training_calibration,
                )
                res = evaluate_survival(surv, t_te, e_te, t_tr, e_tr)
                row = metrics_to_row(res)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[{dataset} fold {fold_idx}] failed: {exc}")
                row = {k: np.nan for k in SCALAR_METRICS}
            row = {"dataset": dataset, "n_samples": n_samples, "fold": fold_idx, **row}
            rows.append(row)

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

    df = pd.DataFrame(rows)
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
    return df


def attach_n_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-dataset ``n_samples`` when missing (e.g. CSVs from older ``cross_validate``)."""
    if "n_samples" in df.columns:
        return df
    from .train.datasets import load_dataset

    out = df.copy()
    n_by_ds = {
        ds: len(load_dataset(ds, sequential=False)["x"])
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
    grouped = df.groupby("dataset")[list(metrics)]
    mean, std = grouped.mean(), grouped.std()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = [f"{mu:.3f} +/- {sd:.3f}" for mu, sd in zip(mean[m], std[m])]
        out[f"{m}_mean"] = mean[m]
        out[f"{m}_std"] = std[m]
    if "d_cal" in metrics:
        by_ds = df.groupby("dataset")["d_cal"]
        miscal = by_ds.apply(lambda s: int((s < d_cal_alpha).sum()))
        valid = by_ds.apply(lambda s: int(s.notna().sum()))
        out["d_cal_miscalibrated"] = [f"{k}/{n}" for k, n in zip(miscal, valid)]
    return out
