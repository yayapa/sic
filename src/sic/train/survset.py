"""SurvSet benchmark: dataset list and loading.

Returns the same dict contract as ``sic.train.datasets`` loaders::

    {x, t, e, covariates, num_cols, cat_cols}

Load-time: drop near-constant columns, cast ``fac_*`` to category, TableVectorizer
one-hot (numeric NaNs kept). Fold-time median impute + scaling happens in
``sic.metrics.prepare_fold`` on ``num_cols``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Frozen benchmark list: all SurvSet datasets with is_td=False, n>10, n>n_features
# (n_features>3, except prostateSurvival), ordered by ascending sample size.
SURVSET_DATASETS: Tuple[str, ...] = (
    "breast",
    "micro.censure",
    "pharmacoSmoking",
    "cgd",
    "veteran",
    "nki70",
    "stagec",
    "burn",
    "d.oropha.rec",
    "wpbc",
    "Melanoma",
    "hepatoCellular",
    "cancer",
    "mgus",
    "e1684",
    "pbc",
    "Pbc3",
    "ova",
    "diabetes",
    "retinopathy",
    "zinc",
    "phpl04K8a",
    "Unemployment",
    "whas500",
    "prostate",
    "cost",
    "follic",
    "vlbw",
    "uis",
    "GBSG2",
    "dataOvarian1",
    "colon",
    "grace",
    "rdata",
    "LeukSurv",
    "actg",
    "dataDIVAT2",
    "TRACE",
    "scania",
    "Aids2",
    "rott2",
    "UnempDur",
    "divorce",
    "acath",
    "smarto",
    "nwtco",
    "dataDIVAT3",
    "Framingham",
    "rhc",
    "dataDIVAT1",
    "Dialysis",
    "flchain",
    "support2",
    "prostateSurvival",
    "hdfail",
)

META_COLS = ("pid", "event", "time", "time2")
NEAR_CONSTANT_THRESHOLD = 0.99


def list_survset_datasets() -> List[str]:
    """Return the SurvSet benchmark dataset names (ascending sample size)."""
    return list(SURVSET_DATASETS)


def is_survset_dataset(name: str) -> bool:
    """True if ``name`` is a SurvSet benchmark dataset (case-sensitive)."""
    return name in SURVSET_DATASETS


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def _mode_frequency(series: pd.Series) -> float:
    """Fraction of rows equal to the mode; NaN counts as its own level."""
    n = len(series)
    if n == 0:
        return 1.0
    # Avoid categorical fillna category errors; treat NaN as a distinct level.
    vc = series.astype("object").where(series.notna(), other="__NA__").value_counts(dropna=False)
    return float(vc.iloc[0] / n) if len(vc) else 1.0


def drop_near_constant_columns(
    X: pd.DataFrame,
    threshold: float = NEAR_CONSTANT_THRESHOLD,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns whose mode frequency is >= threshold (NaN as a value)."""
    drop = [c for c in X.columns if _mode_frequency(X[c]) >= threshold]
    if drop:
        X = X.drop(columns=drop)
    return X, drop


def _cast_fac_to_category(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if c.startswith("fac_"):
            X[c] = X[c].astype("string").astype("category")
    return X


def load_survset_dataset(ds_name: str) -> Dict:
    """Load and vectorize a SurvSet dataset into the SIC loader contract."""
    if not is_survset_dataset(ds_name):
        raise ValueError(
            f"'{ds_name}' is not a SurvSet benchmark dataset. Available: {list(SURVSET_DATASETS)}"
        )

    from skrub import TableVectorizer
    from SurvSet.data import SurvLoader

    raw = SurvLoader().load_dataset(ds_name)["df"].copy()
    feat_cols = _feature_columns(raw)
    X = raw[feat_cols].copy()
    t = raw["time"].to_numpy(dtype=np.float64)
    e = raw["event"].to_numpy(dtype=np.int32)

    num_cols_raw = [c for c in X.columns if c.startswith("num_")]
    X = _cast_fac_to_category(X)
    X, _dropped = drop_near_constant_columns(X)

    # Drop rows that are all-NaN across remaining features.
    if len(X.columns) == 0:
        raise ValueError(f"SurvSet dataset '{ds_name}' has no features after filtering.")
    row_ok = ~X.isna().all(axis=1)
    if not bool(row_ok.all()):
        X = X.loc[row_ok].reset_index(drop=True)
        t = t[row_ok.to_numpy()]
        e = e[row_ok.to_numpy()]
    else:
        X = X.reset_index(drop=True)

    tv = TableVectorizer()  # default one-hot; numeric PassThrough keeps NaNs
    X_t = tv.fit_transform(X)
    if not isinstance(X_t, pd.DataFrame):
        X_t = pd.DataFrame(X_t, columns=tv.get_feature_names_out())

    num_cols = [c for c in num_cols_raw if c in X_t.columns]
    cat_cols = [c for c in X_t.columns if c not in num_cols]

    return {
        "x": X_t,
        "t": t.astype(np.float64),
        "e": e.astype(np.int32),
        "covariates": list(X_t.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
