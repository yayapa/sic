"""The one-liner SIC estimator for in-context survival analysis.

Example
-------
>>> from sic import SIC
>>> model = SIC(survival_head="SuMoPP")              # downloads weights on first use
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
    predict_surv_deepeh_at_grid,
)
from .model.sumo_utils import (
    predict_surv_sumo,
    compute_time_norm,
    nll_loss,
    DEFAULT_TSCALE_Q,
    CONTINUOUS_SURV_TYPES,
)

ArrayLike = Union[np.ndarray, pd.DataFrame, "torch.Tensor"]

# Practical cap on the in-context training set. The backbone was pre-trained
# with sequences up to ~60k rows; beyond that inference will OOM on most GPUs.
MAX_CONTEXT_ROWS = 500000#60000


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


def _feature_perms(
    n_features: int,
    n_estimators: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Column orders for test-time ensembling (identity, then cyclic shifts).

    Member 0 is the identity. Members ``1 .. min(M-1, H-1)`` are cyclic column
    shifts. If ``M > H``, remaining members are seeded random permutations.
    """
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators}.")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    base = np.arange(n_features)
    perms: list[list[int]] = [base.tolist()]
    for i in range(1, n_estimators):
        if i < n_features:
            perms.append(np.roll(base, i).tolist())
        else:
            perms.append(rng.permutation(n_features).tolist())
    return perms


class SIC:
    """In-context survival analysis with a prior-fitted network.

    Parameters
    ----------
    survival_head : str, default="SuMoPP"
        One of ``DeepHit10``, ``DeepHit50``, ``DeepHit100``, ``CoxPH``, ``DeepEH``,
        ``SuMo``, ``SuMoPP``.
    device : str or torch.device, optional
        Defaults to CUDA when available, else CPU.
    model_path : str or Path, optional
        Local checkpoint to load instead of downloading from the Hub.
    repo_id : str, optional
        Hugging Face repo id to download released weights from. Defaults to
        ``$SIC_HF_REPO`` or the packaged placeholder.
    allow_auto_download : bool, default=True
        Allow downloading the checkpoint when it is not found locally.
    standardize : bool, default=False
        z-score covariates using training-set statistics before inference.
    random_state : int, default=42
        Seed used for reproducible feature-permutation ensembles / any randomness.
    n_estimators : int, default=8
        Number of feature-permutation ensemble members. Predictions average
        survival curves ``S(t|x)`` over column orders. ``1`` disables ensembling.
    """

    def __init__(
        self,
        survival_head: str = "SuMoPP",
        device: Optional[Union[str, torch.device]] = None,
        model_path=None,
        repo_id: Optional[str] = None,
        allow_auto_download: bool = True,
        standardize: bool = False,
        random_state: int = 42,
        n_estimators: int = 8,
    ):
        resolve_head(survival_head)  # validate early
        if int(n_estimators) < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}.")
        self.survival_head = survival_head
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.repo_id = repo_id
        self.allow_auto_download = allow_auto_download
        self.standardize = standardize
        self.random_state = random_state
        self.n_estimators = int(n_estimators)

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

        # Time-norm rule for trunk embedding + continuous heads. TabICL accepts
        # sumo_tscale_q; keep a copy for predict paths that recompute the norm.
        self.sumo_tscale_q = config.get("sumo_tscale_q", DEFAULT_TSCALE_Q)
        # Legacy SuMoPP release checkpoints store this metadata but the current
        # head always uses linear / minmax time scaling via ``compute_time_norm``.
        config.pop("time_warp", None)
        # Training-only censoring head: inference uses the event head alone.
        config.pop("cens_head", None)

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
    ) -> pd.DataFrame:
        """Predict survival curves for ``X_test`` given labeled ``(X_train, y_train)``.

        Parameters
        ----------
        eval_times : sequence of float, optional
            Times at which to report survival probabilities. When ``None`` the
            model-native grid is used (DeepHit bin edges / sorted unique training
            times for CoxPH, DeepEH, SuMo, and SuMoPP).
        survival_head : str, optional
            Override the head for this call (re-loads the corresponding weights).

        Returns
        -------
        pandas.DataFrame
            Survival probabilities with the evaluation times as the index and one
            column per test subject (the ``pycox`` / ``EvalSurv`` convention).
        """
        self._ensure_model(survival_head)
        X_train_t, X_test_t, T_train, E_train = self._prepare(X_train, y_train, X_test)

        S, time_grid = self._predict_curves(X_train_t, X_test_t, T_train, E_train, eval_times)

        return pd.DataFrame(S.T, index=np.asarray(time_grid, dtype=np.float32))

    def fit_predict_density_nll(
        self,
        X_train: ArrayLike,
        y_train,
        X_test: ArrayLike,
        t,
        e,
        survival_head: Optional[str] = None,
    ) -> float:
        """Mean SuMo/SuMo++ NLL of ``(t, e)`` for ``X_test`` given the train context.

        ``t_scale`` is computed from **train** durations (same convention as
        ``fit_predict_surv``). Returns NaN for non-continuous heads.
        """
        self._ensure_model(survival_head)
        if self.surv_model_type not in CONTINUOUS_SURV_TYPES:
            return float("nan")
        X_train_t, X_test_t, T_train, E_train = self._prepare(X_train, y_train, X_test)
        out = self._forward(X_train_t, X_test_t, T_train, E_train)
        emb = out["emb_test"].squeeze(0)
        head = out["sumo_head"]
        t_scale = compute_time_norm(T_train, self.sumo_tscale_q)
        t_t = torch.as_tensor(np.asarray(t, dtype=np.float64), device=emb.device, dtype=emb.dtype)
        e_t = torch.as_tensor(np.asarray(e, dtype=np.float64), device=emb.device, dtype=emb.dtype)
        loss = nll_loss(head, emb, t_t, e_t, t_scale=t_scale)
        return float(loss.detach().cpu())

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
        * SuMo / SuMoPP: ``{"head", "survival" (N, M), "times" (M,)}`` on the native grid
        """
        self._ensure_model(survival_head)
        X_train_t, X_test_t, T_train, E_train = self._prepare(X_train, y_train, X_test)

        head = self.survival_head
        if self.surv_model_type in CONTINUOUS_SURV_TYPES:
            # Continuous head: the native grid (sorted unique training times) is
            # exactly what ``_predict_curves`` uses for ``eval_times=None``, so
            # delegate to keep a single sumo prediction path.
            S, grid_np = self._predict_curves(X_train_t, X_test_t, T_train, E_train, eval_times=None)
            return {"head": head, "survival": S, "times": np.asarray(grid_np, dtype=np.float32)}

        out = self._forward(X_train_t, X_test_t, T_train, E_train)

        if self.surv_model_type == "deephit":
            logits = out["log_hazard"].squeeze(0)  # (N, K)
            from pycox.models.utils import pad_col

            pmf = pad_col(logits).softmax(1)[:, :-1]  # (N, K)
            edges = self._deephit_bin_edges(T_train, E_train)
            return {
                "head": head,
                "logits": logits.cpu().numpy(),
                "pmf": pmf.detach().cpu().numpy(),
                "bin_edges": np.asarray(edges.cpu().numpy(), dtype=np.float32),
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
                "baseline_times": np.asarray(bs["time"].cpu().numpy(), dtype=np.float32),
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

    def _predict_curves_once(self, X_train_t, X_test_t, T_train, E_train, eval_times):
        """Single forward pass → survival curves on ``eval_times`` (or native grid)."""
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
            if eval_times is None:
                grid_np = np.unique(np.sort(T_train.squeeze(0).cpu().numpy()))
            else:
                grid_np = np.sort(np.asarray(eval_times, dtype=float))
            grid = torch.tensor(grid_np, device=self.device, dtype=torch.float32)
            S = predict_surv_deepeh_at_grid(
                log_hazards_train[:, :2], T_train.squeeze(0), E_train.squeeze(0),
                log_hazards[:, :2], grid,
            )
            return S.detach().cpu().numpy(), grid_np

        if mt in CONTINUOUS_SURV_TYPES:
            emb_test = out["emb_test"].squeeze(0)  # (N, d_model)
            head = out["sumo_head"]
            t_scale = compute_time_norm(T_train, self.sumo_tscale_q)
            if eval_times is None:
                grid_np = np.unique(np.sort(T_train.squeeze(0).cpu().numpy()))
            else:
                grid_np = np.asarray(eval_times, dtype=float)
            grid = torch.tensor(grid_np, device=self.device, dtype=emb_test.dtype)
            S = predict_surv_sumo(emb_test, grid, head, t_scale=t_scale)  # (N, M)
            return S.detach().cpu().numpy(), grid_np

        raise ValueError(f"Unsupported surv_model_type={mt}.")

    def _predict_curves(self, X_train_t, X_test_t, T_train, E_train, eval_times):
        """Predict survival curves, optionally averaging over feature permutations."""
        if self.n_estimators == 1:
            return self._predict_curves_once(
                X_train_t, X_test_t, T_train, E_train, eval_times
            )

        n_features = X_train_t.shape[-1]
        rng = np.random.default_rng(self.random_state)
        perms = _feature_perms(n_features, self.n_estimators, rng)

        curves = []
        grid_np = None
        for perm in perms:
            S_m, grid_m = self._predict_curves_once(
                X_train_t[..., perm],
                X_test_t[..., perm],
                T_train,
                E_train,
                eval_times,
            )
            curves.append(S_m)
            if grid_np is None:
                grid_np = grid_m
        return np.mean(np.stack(curves, axis=0), axis=0), grid_np
