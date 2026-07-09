"""The one-liner SIC estimator for in-context survival analysis.

Example
-------
>>> from sic import SIC
>>> model = SIC(survival_head="DeepHit10")           # downloads weights on first use
>>> surv = model.fit_predict_surv(X_train, y_train, X_test, eval_times=times)
>>> surv.shape        # (len(times), n_test_subjects)

There is no gradient training at ``fit`` time: SIC is a prior-fitted network
that performs the whole survival analysis in a single forward pass, using the
training rows as in-context examples.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import build_model_config, resolve_head, SURVIVAL_HEADS
from .weights import resolve_checkpoint
from .model.tabicl import TabICL
from .train.survUtils import (
    predict_surv_deephit,
    predict_pmf_deephit,
    predict_surv_new_times_deephit,
    predict_surv_coxph_2,
    predict_surv_deepeh,
)

ArrayLike = Union[np.ndarray, pd.DataFrame, "torch.Tensor"]

# Practical cap on the in-context training set. The backbone was pre-trained
# with sequences up to ~60k rows; beyond that inference will OOM on most GPUs.
MAX_CONTEXT_ROWS = 60000


def make_y(time, event) -> np.ndarray:
    """Build a scikit-survival structured array from ``time`` and ``event``.

    Parameters
    ----------
    time : array-like of float
        Observed durations (time to event or censoring).
    event : array-like of bool/int
        Event indicator (1/True = event occurred, 0/False = censored).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    if time.shape != event.shape:
        raise ValueError(f"time and event must have the same shape, got {time.shape} and {event.shape}.")
    y = np.empty(time.shape[0], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = event
    y["time"] = time
    return y


def _parse_y(y) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the many accepted ``y`` formats into ``(time, event)`` arrays.

    Accepted:
    * scikit-survival structured array with two fields (event: bool, time: float)
    * a ``(durations, events)`` tuple/list
    * a 2-column DataFrame or ndarray (time first, event second, unless column
      names disambiguate)
    """
    # Structured array (sksurv style)
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = y.dtype.names
        if len(names) != 2:
            raise ValueError(f"Structured y must have exactly 2 fields, got {names}.")
        # Field order in sksurv is (event, time); detect by dtype (bool = event).
        f0, f1 = names
        if np.issubdtype(y.dtype[f0], np.bool_):
            event, time = y[f0], y[f1]
        else:
            time, event = y[f0], y[f1]
        # Structured-array fields are strided views; copy to contiguous arrays.
        return np.ascontiguousarray(time, dtype=float), np.ascontiguousarray(event).astype(int)

    # Tuple/list -> (durations, events)
    if isinstance(y, (tuple, list)) and len(y) == 2:
        time, event = y
        return np.asarray(time, dtype=float), np.asarray(event).astype(int)

    # DataFrame
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 2:
            raise ValueError("DataFrame y must have exactly 2 columns.")
        cols = [c.lower() for c in y.columns]
        time_col = next((c for c, lc in zip(y.columns, cols) if "time" in lc or "dur" in lc), None)
        event_col = next((c for c, lc in zip(y.columns, cols) if "event" in lc or "status" in lc or "cens" in lc), None)
        if time_col is not None and event_col is not None:
            return y[time_col].to_numpy(float), y[event_col].to_numpy().astype(int)
        return y.iloc[:, 0].to_numpy(float), y.iloc[:, 1].to_numpy().astype(int)

    # 2D ndarray (N, 2): time first, event second
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(float), arr[:, 1].astype(int)

    raise ValueError(
        "Unrecognized y format. Use sic.make_y(time, event), a (durations, events) "
        "tuple, a 2-column DataFrame, or an (N, 2) array."
    )


def _to_2d_float(X: ArrayLike) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32)
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


class SIC:
    """In-context survival analysis with a prior-fitted network.

    Parameters
    ----------
    survival_head : str, default="DeepHit10"
        One of ``DeepHit10``, ``DeepHit50``, ``DeepHit100``, ``CoxPH``, ``DeepEH``.
    device : str or torch.device, optional
        Defaults to CUDA when available, else CPU.
    model_path : str or Path, optional
        Local checkpoint to load instead of downloading from the Hub.
    repo_id : str, optional
        Hugging Face repo id to download released weights from. Defaults to
        ``$SIC_HF_REPO`` or the packaged placeholder.
    allow_auto_download : bool, default=True
        Allow downloading the checkpoint when it is not found locally.
    standardize : bool, default=True
        z-score covariates using training-set statistics before inference.
    random_state : int, default=42
        Seed used for reproducible time-bin quantiles / any randomness.
    """

    def __init__(
        self,
        survival_head: str = "DeepHit10",
        device: Optional[Union[str, torch.device]] = None,
        model_path=None,
        repo_id: Optional[str] = None,
        allow_auto_download: bool = True,
        standardize: bool = False,
        random_state: int = 42,
    ):
        resolve_head(survival_head)  # validate early
        self.survival_head = survival_head
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.repo_id = repo_id
        self.allow_auto_download = allow_auto_download
        self.standardize = standardize
        self.random_state = random_state

        self.model: Optional[TabICL] = None
        self.surv_model_type: Optional[str] = None
        self.num_bins: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _ensure_model(self, survival_head: Optional[str] = None):
        head = survival_head or self.survival_head
        if self.model is not None and head == self.survival_head:
            return
        self.survival_head = head

        ckpt_path = resolve_checkpoint(
            head,
            model_path=self.model_path,
            repo_id=self.repo_id,
            allow_auto_download=self.allow_auto_download,
        )
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint {ckpt_path} does not contain a 'state_dict'.")

        # Prefer the config stored in the (release) checkpoint; fall back to
        # the packaged defaults for the requested head.
        config = checkpoint.get("config") or build_model_config(head)
        config = dict(config)
        config.setdefault("mode", "survival")

        model = TabICL(**config)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing:
            # Released checkpoints should load cleanly; warn loudly otherwise.
            import warnings

            warnings.warn(
                f"Loaded checkpoint with {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys. Predictions may be unreliable.",
                RuntimeWarning,
            )
        model.to(self.device).eval()

        self.model = model
        self.surv_model_type = config["surv_model_type"]
        self.num_bins = config["num_bins"]

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    def _prepare(self, X_train, y_train, X_test):
        Xtr = _to_2d_float(X_train)
        Xte = _to_2d_float(X_test)
        if Xtr.shape[1] != Xte.shape[1]:
            raise ValueError(
                f"X_train and X_test must have the same number of features, "
                f"got {Xtr.shape[1]} and {Xte.shape[1]}."
            )
        if Xtr.shape[0] > MAX_CONTEXT_ROWS:
            raise ValueError(
                f"Training set has {Xtr.shape[0]} rows which exceeds the in-context "
                f"limit of {MAX_CONTEXT_ROWS}. Subsample the training set."
            )

        t, e = _parse_y(y_train)
        if t.shape[0] != Xtr.shape[0]:
            raise ValueError(
                f"y_train has {t.shape[0]} entries but X_train has {Xtr.shape[0]} rows."
            )

        if self.standardize:
            mean = Xtr.mean(axis=0, keepdims=True)
            std = Xtr.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            Xtr = (Xtr - mean) / std
            Xte = (Xte - mean) / std

        dev = self.device
        X_train_t = torch.tensor(Xtr, dtype=torch.float32, device=dev).unsqueeze(0)
        X_test_t = torch.tensor(Xte, dtype=torch.float32, device=dev).unsqueeze(0)
        T_train = torch.tensor(t, dtype=torch.float32, device=dev).unsqueeze(0)
        E_train = torch.tensor(e, dtype=torch.float32, device=dev).unsqueeze(0)
        return X_train_t, X_test_t, T_train, E_train

    def _forward(self, X_train_t, X_test_t, T_train, E_train):
        X = torch.cat([X_train_t, X_test_t], dim=1)
        y_train = torch.stack([T_train, E_train], dim=1)  # (1, 2, train_size)
        with torch.no_grad():
            return self.model(X, y_train)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_predict_surv(
        self,
        X_train: ArrayLike,
        y_train,
        X_test: ArrayLike,
        eval_times: Optional[Sequence[float]] = None,
        survival_head: Optional[str] = None,
        post_training_calibration: Optional[str] = None,
    ) -> pd.DataFrame:
        """Predict survival curves for ``X_test`` given labeled ``(X_train, y_train)``.

        Parameters
        ----------
        eval_times : sequence of float, optional
            Times at which to report survival probabilities. When ``None`` the
            model-native grid is used (DeepHit bin edges / Cox event times /
            DeepEH internal grid).
        survival_head : str, optional
            Override the head for this call (re-loads the corresponding weights).
        post_training_calibration : {None, "cipot", "csd"}
            Optional conformal calibration fitted on the training set.

        Returns
        -------
        pandas.DataFrame
            Survival probabilities with the evaluation times as the index and one
            column per test subject (the ``pycox`` / ``EvalSurv`` convention).
        """
        self._ensure_model(survival_head)
        X_train_t, X_test_t, T_train, E_train = self._prepare(X_train, y_train, X_test)

        S, time_grid = self._predict_curves(X_train_t, X_test_t, T_train, E_train, eval_times)

        if post_training_calibration is not None:
            S, time_grid = self._apply_calibration(
                post_training_calibration, X_train_t, T_train, E_train, S, time_grid, eval_times,
            )

        return pd.DataFrame(S.T, index=time_grid)

    def fit_predict_raw(
        self,
        X_train: ArrayLike,
        y_train,
        X_test: ArrayLike,
        survival_head: Optional[str] = None,
    ) -> dict:
        """Return the head-native raw outputs for ``X_test`` (no eval times).

        The returned dict is self-describing per head:

        * DeepHit*: ``{"head", "logits" (N, K), "pmf" (N, K), "bin_edges" (K,)}``
        * CoxPH:    ``{"head", "log_hazard" (N, 1), "baseline_times", "baseline_survival"}``
        * DeepEH:   ``{"head", "g" (N, 2)}``
        """
        self._ensure_model(survival_head)
        X_train_t, X_test_t, T_train, E_train = self._prepare(X_train, y_train, X_test)
        out = self._forward(X_train_t, X_test_t, T_train, E_train)

        head = self.survival_head
        if self.surv_model_type == "deephit":
            logits = out["log_hazard"].squeeze(0)  # (N, K)
            from pycox.models.utils import pad_col

            pmf = pad_col(logits).softmax(1)[:, :-1]  # (N, K)
            edges = self._deephit_bin_edges(T_train, E_train)
            return {
                "head": head,
                "logits": logits.cpu().numpy(),
                "pmf": pmf.detach().cpu().numpy(),
                "bin_edges": edges.cpu().numpy(),
            }
        elif self.surv_model_type == "coxph":
            from sic._vendor.torchsurv.loss.cox import baseline_survival_function

            log_hazard = out["log_hazard"].squeeze(0)  # (N, 1)
            log_hazard_train = out["log_hazard_train"].squeeze(0)
            bs = baseline_survival_function(
                log_hazard_train, E_train.squeeze(0).bool(), T_train.squeeze(0)
            )
            return {
                "head": head,
                "log_hazard": log_hazard.cpu().numpy(),
                "baseline_times": bs["time"].cpu().numpy(),
                "baseline_survival": bs["baseline_survival"].cpu().numpy(),
            }
        elif self.surv_model_type == "deepeh":
            # DeepEH consumes only the first two head columns (g1, g2).
            g = out["log_hazard"].squeeze(0)[:, :2]  # (N, 2)
            return {"head": head, "g": g.cpu().numpy()}
        raise ValueError(f"Unsupported surv_model_type={self.surv_model_type}.")

    # ------------------------------------------------------------------ #
    # Per-head survival reconstruction
    # ------------------------------------------------------------------ #
    def _deephit_bin_edges(self, T_train, E_train) -> torch.Tensor:
        from pycox.preprocessing.label_transforms import LabTransDiscreteTime

        num_durations = self.num_bins
        durations_train = T_train.squeeze(0).detach().cpu().numpy()
        events_train = E_train.squeeze(0).detach().cpu().numpy().astype(bool)
        labtrans = LabTransDiscreteTime(num_durations, scheme="quantiles")
        labtrans.fit(durations_train, events_train)
        if len(labtrans.cuts) != num_durations:
            labtrans = LabTransDiscreteTime(num_durations, scheme="equidistant")
            labtrans.fit(durations_train, events_train)
        return torch.tensor(labtrans.cuts, device=self.device)

    def _predict_curves(self, X_train_t, X_test_t, T_train, E_train, eval_times):
        out = self._forward(X_train_t, X_test_t, T_train, E_train)
        mt = self.surv_model_type

        if mt == "deephit":
            log_hazards = out["log_hazard"].squeeze(0)  # (N, K)
            edges = self._deephit_bin_edges(T_train, E_train)
            S = predict_surv_deephit(log_hazards).T  # (N, K) on bin edges
            if eval_times is None:
                grid = edges
            else:
                grid = torch.tensor(np.asarray(eval_times, dtype=float),
                                    device=self.device, dtype=S.dtype)
            S = predict_surv_new_times_deephit(S, edges, grid)
            return S.detach().cpu().numpy(), grid.detach().cpu().numpy()

        if mt == "coxph":
            log_hazards = out["log_hazard"].squeeze()
            log_hazards_train = out["log_hazard_train"].squeeze()
            if eval_times is None:
                grid_np = np.unique(np.sort(T_train.squeeze(0).cpu().numpy()))
            else:
                grid_np = np.asarray(eval_times, dtype=float)
            grid = torch.tensor(grid_np, device=self.device, dtype=log_hazards.dtype)
            S, _ = predict_surv_coxph_2(
                log_hazards_train, E_train.squeeze(0), T_train.squeeze(0), log_hazards, grid,
            )
            return S.detach().cpu().numpy(), grid_np

        if mt == "deepeh":
            log_hazards = out["log_hazard"].squeeze(0)
            log_hazards_train = out["log_hazard_train"].squeeze(0)
            # DeepEH reconstructs survival on the sorted test times; we then
            # interpolate onto the requested eval times if given.
            T_test_dummy = torch.arange(
                1, X_test_t.shape[1] + 1, device=self.device, dtype=torch.float32
            )
            # Use provided eval_times as the query grid when available.
            S, tt, _, _ = predict_surv_deepeh(
                log_hazards_train, T_train.squeeze(0), E_train.squeeze(0),
                log_hazards, T_test_dummy, torch.ones_like(T_test_dummy),
            )
            S = torch.as_tensor(S, device=self.device, dtype=torch.float32)
            src_times = torch.as_tensor(tt, device=self.device, dtype=torch.float32)
            order = torch.argsort(src_times)
            src_times = src_times[order]
            S = S[:, order]
            if eval_times is None:
                return S.detach().cpu().numpy(), src_times.detach().cpu().numpy()
            grid = torch.tensor(np.asarray(eval_times, dtype=float),
                                device=self.device, dtype=torch.float32)
            S = predict_surv_new_times_deephit(S, src_times, grid)
            return S.detach().cpu().numpy(), grid.detach().cpu().numpy()

        raise ValueError(f"Unsupported surv_model_type={mt}.")

    # ------------------------------------------------------------------ #
    # Calibration wiring
    # ------------------------------------------------------------------ #
    def _apply_calibration(self, method, X_train_t, T_train, E_train, S_test, time_grid, eval_times):
        from . import calibration as cal

        method = method.lower()
        if method not in ("cipot", "csd"):
            raise ValueError(f"post_training_calibration must be None, 'cipot' or 'csd', got {method!r}.")

        t_train = T_train.squeeze(0).cpu().numpy()
        e_train = E_train.squeeze(0).cpu().numpy().astype(int)

        # Calibration curves: predict on the training set as its own query set,
        # on the same evaluation grid as the test curves.
        S_cal, tg_cal = self._predict_curves(X_train_t, X_train_t, T_train, E_train, list(time_grid))

        quantile_levels = cal.default_quantile_levels()
        if method == "cipot":
            scores = cal.cipot_calibrate(S_cal, tg_cal, t_train, e_train)
            S_adj = cal.cipot_predict(S_test, time_grid, scores, quantile_levels)
        else:
            scores = cal.csd_calibrate(S_cal, tg_cal, t_train, e_train, t_train, e_train, quantile_levels)
            S_adj = cal.csd_predict(S_test, time_grid, scores, quantile_levels)
        return S_adj, time_grid
