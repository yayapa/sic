"""Real-world survival benchmark datasets.

Every loader returns a uniform dict::

    {
        'x': pd.DataFrame,      # covariates
        't': np.ndarray,        # durations (float)
        'e': np.ndarray,        # event indicators (0/1)
        'covariates': list,
        'num_cols': list,
        'cat_cols': list,
    }

Datasets are obtained without any manual setup:

* ``WHAS500``, ``AIDS``, ``VETERAN`` ship with ``scikit-survival``.
* ``GBSG2``, ``LUNG`` ship with ``lifelines``.
* ``FRAMINGHAM``, ``PBC``, ``SUPPORT``, ``FLCHAIN``, ``NWTCO``, ``METABRIC``,
  ``GBSG`` are lightweight CSVs bundled with this package
  (``sic/train/datasets/*.csv``; METABRIC/GBSG from the DeepSurv/pycox releases).

The private SEER and UNOS loaders from the research code are intentionally not
included here (those datasets require restricted-access credentials).
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

EPS = 1e-8

# Public list of datasets that work out of the box.
ALL_DATASETS = [
    "VETERAN",
    "LUNG",
    "PBC",
    "WHAS500",
    "GBSG2",
    "AIDS",
    "METABRIC",
    "GBSG",
    "NWTCO",
    "FRAMINGHAM",
    "FLCHAIN",
    "SUPPORT",
]


def list_datasets():
    """Return the names of all datasets that :func:`load_dataset` supports."""
    return list(ALL_DATASETS)


def _bundled_csv(name: str) -> pd.DataFrame:
    """Read a CSV that is bundled as package data under ``sic/train/datasets``."""
    with resources.as_file(resources.files(__package__).joinpath("datasets", name)) as path:
        df = pd.read_csv(path)
    # Drop the unnamed R row-index column that ships with some CSVs.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def load_dataset(dataset="SUPPORT", sequential=False):
    """Load a benchmark survival dataset by name.

    Parameters
    ----------
    dataset : str
        One of :func:`list_datasets`.
    sequential : bool, default=False
        Only relevant for the longitudinal datasets (PBC, FRAMINGHAM); when
        False only the first record per subject is kept.
    """
    dataset = dataset.upper()
    if dataset == "GBSG":
        return _load_gbsg_dataset()
    elif dataset == "GBSG2":
        return _load_gbsg_2_dataset()
    elif dataset == "METABRIC":
        return _load_metabric_dataset()
    elif dataset == "FLCHAIN":
        return _load_flchain_dataset()
    elif dataset == "NWTCO":
        return _load_nwtco_dataset()
    elif dataset == "WHAS500":
        return _load_whas500_dataset()
    elif dataset == "SUPPORT":
        return _load_support_dataset()
    elif dataset == "PBC":
        return _load_pbc_dataset(sequential, competing=False)
    elif dataset == "FRAMINGHAM":
        return _load_framingham_dataset(sequential, competing=False)
    elif dataset == "LUNG":
        return _load_lung()
    elif dataset == "AIDS":
        return _load_aids()
    elif dataset == "VETERAN":
        return _load_veteran()
    else:
        raise NotImplementedError(
            f"Dataset '{dataset}' not available. Supported datasets: {ALL_DATASETS}"
        )


def _load_whas500_dataset():
    from sksurv.datasets import load_whas500

    x, data_y = load_whas500()
    cat_cols = ["afb", "av3", "cvd", "gender", "miord", "mitype", "chf", "sho"]
    num_cols = ["age", "bmi", "diasbp", "hr", "los", "sysbp"]
    x[cat_cols] = x[cat_cols].astype(int)
    x[num_cols] = x[num_cols].astype(float)
    covariates = x.columns
    e = data_y["fstat"].astype(int)
    t = data_y["lenfol"].astype(float)
    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": covariates,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_framingham_dataset(sequential, competing=False):
    data = _bundled_csv("framingham.csv")

    if not sequential:
        data = data.groupby("RANDID").first()

    cat_cols = ["SEX", "CURSMOKE", "DIABETES", "BPMEDS", "educ", "PREVCHD",
                "PREVAP", "PREVMI", "PREVSTRK", "PREVHYP"]
    num_cols = ["TOTCHOL", "AGE", "SYSBP", "DIABP", "CIGPDAY", "BMI",
                "HEARTRTE", "GLUCOSE"]
    x1 = data[cat_cols].astype("category")
    x1 = pd.get_dummies(x1)
    x1 = x1.drop(columns=["SEX_1", "CURSMOKE_0", "DIABETES_0", "educ_1.0",
                          "PREVCHD_0", "PREVAP_0", "PREVMI_0", "PREVSTRK_0",
                          "PREVHYP_0"])
    x1 = x1.astype(int)
    x2 = data[num_cols]
    imputer = SimpleImputer(strategy="mean")
    x2 = pd.DataFrame(imputer.fit_transform(x2), columns=x2.columns, index=data.index)

    x = pd.concat([x1, x2], axis=1)

    time = (data["TIMEDTH"] - data["TIME"]).values
    event = data["DEATH"].values

    if competing:
        time_cvd = (data["TIMECVD"] - data["TIME"]).values
        event[data["CVD"] == 1] = 2
        time[data["CVD"] == 1] = time_cvd[data["CVD"] == 1]

    if not sequential:
        return {
            "x": x,
            "t": time + 1,
            "e": event,
            "covariates": np.concatenate([x1.columns, x2.columns]),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }
    raise NotImplementedError("sequential=True is not implemented for Framingham.")


def _load_pbc_dataset(sequential, competing=False):
    data = _bundled_csv("pbc2.csv")
    if not sequential:
        data = data.groupby("id").first()

    cat_cols = ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema",
                "histologic"]
    num_cols = ["serBilir", "serChol", "albumin", "alkaline", "SGOT",
                "platelets", "prothrombin"]

    data["histologic"] = data["histologic"].astype(str)
    dat_cat = data[cat_cols]
    dat_num = data[num_cols]

    age = data["age"] + data["years"]

    x1 = pd.get_dummies(dat_cat)
    x1 = x1.astype(int)
    x2 = dat_num
    x3 = age.rename("age_years")
    x2 = pd.concat([x2, x3], axis=1)

    imputer = SimpleImputer(strategy="mean")
    x2 = pd.DataFrame(imputer.fit_transform(x2), columns=x2.columns, index=data.index)

    x = pd.concat([x1, x2], axis=1)

    time = (data["years"] - data["year"]).values
    event = (data["status"] == "dead").values.astype(int)
    if competing:
        event[data["status"] == "transplanted"] = 2

    if not sequential:
        return {
            "x": x,
            "t": time + 1,
            "e": event,
            "num_cols": num_cols + ["age_years"],
            "cat_cols": cat_cols,
            "covariates": list(x.columns),
        }
    raise NotImplementedError("sequential=True is not implemented for PBC.")


def _load_support_dataset():
    data = _bundled_csv("support2.csv")
    remove = data["d.time"].notna()
    data = data[remove].reset_index(drop=True)

    num_cols = ["age", "num.co", "meanbp", "wblc", "hrt", "resp", "temp",
                "pafi", "alb", "bili", "crea", "sod", "ph", "glucose", "bun",
                "urine", "adlp", "adls"]
    x1 = data[num_cols]

    cat_cols = ["sex", "dzgroup", "dzclass", "income", "race", "ca"]
    x2 = pd.get_dummies(data[cat_cols])
    x2 = x2.drop(columns=["sex_male", "dzgroup_Cirrhosis", "dzclass_Coma", "ca_yes"])
    x2 = x2.astype(int)

    imputer = SimpleImputer(strategy="mean")
    x1 = pd.DataFrame(imputer.fit_transform(x1), columns=num_cols, index=data.index)
    x = pd.concat([x1, x2], axis=1)

    t = data["d.time"].values + 1
    e = data["death"].values

    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": np.concatenate([x1.columns, x2.columns]),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_metabric_dataset():
    cat_cols = ["Hormone", "Radiotherapy", "Chemotherapy", "ER-positive"]
    num_cols = ["MKI67", "EGFR", "PGR", "ERBB2", "Age at diagnosis"]
    df = _bundled_csv("metabric.csv")
    df = df.rename(columns={"x0": "MKI67", "x1": "EGFR", "x2": "PGR",
                            "x3": "ERBB2", "x4": "Hormone", "x5": "Radiotherapy",
                            "x6": "Chemotherapy", "x7": "ER-positive",
                            "x8": "Age at diagnosis"})
    df["duration"] += EPS
    e = df["event"].values.astype(int)
    t = df["duration"].values.astype(float)
    x = df.drop(columns=["duration", "event"])
    for col in cat_cols:
        x[col] = x[col].astype(int)
    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": x.columns,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_flchain_dataset():
    df = _bundled_csv("flchain.csv")
    e = df["death"].values.astype(int)
    t = df["futime"].values.astype(float)
    x = df.drop(columns=["futime", "death"])
    if "rownames" in x.columns:
        x = x.drop(["rownames"], axis="columns")

    cat_cols = ["sex", "flc.grp", "mgus"]
    num_cols = ["age", "sample.yr", "kappa", "lambda", "creatinine"]

    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": x.columns,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_nwtco_dataset():
    df = _bundled_csv("nwtco.csv")
    e = df["rel"].values.astype(int)
    t = df["edrel"].values.astype(float)
    x = df.drop(columns=["edrel", "rel"])
    if "rownames" in x.columns:
        x = x.drop(["rownames"], axis="columns")
    cat_cols = ["in.subcohort", "instit_2", "histol_2", "stage", "study_4"]
    num_cols = ["age"]
    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": x.columns,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_gbsg_dataset():
    df = _bundled_csv("gbsg.csv")
    e = df["event"].values.astype(int)
    t = df["duration"].values.astype(float)
    x = df.drop(columns=["duration", "event"])
    cat_cols = ["x1", "x2"]
    num_cols = ["x3", "x4", "x5", "x6"]
    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": x.columns,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_gbsg_2_dataset():
    import lifelines

    df = lifelines.datasets.load_gbsg2()
    df = pd.get_dummies(df, columns=["horTh", "menostat", "tgrade"]).astype("float")
    df.drop(columns=["horTh_no", "menostat_Post", "tgrade_I"], axis=1, inplace=True)

    num_cols = ["age", "tsize", "pnodes", "progrec", "estrec"]
    cat_cols = ["horTh", "menostat", "tgrade"]

    e = df["cens"].values.astype(int)
    t = df["time"].values.astype(float)
    x = df.drop(columns=["time", "cens"])
    return {
        "x": x,
        "t": t,
        "e": e,
        "covariates": x.columns,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def _load_aids():
    from sksurv.datasets import load_aids

    x, y = load_aids()
    e = y["censor"].astype(int)
    t = y["time"].astype(float)
    num_cols = ["age", "cd4", "priorzdv"]
    cat_cols = ["hemophil", "ivdrug", "karnof", "sex", "strat2", "tx", "txgrp", "raceth"]
    for col in cat_cols:
        x[col] = x[col].astype("category").cat.codes
    return {
        "x": x,
        "t": t,
        "e": e,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "covariates": x.columns.tolist(),
    }


def _load_veteran():
    from sksurv.datasets import load_veterans_lung_cancer

    x, y = load_veterans_lung_cancer()
    e = y["Status"].astype(int)
    t = y["Survival_in_days"].astype(float)
    cat_cols = ["Celltype", "Prior_therapy", "Treatment"]
    x = pd.get_dummies(x, columns=["Celltype"], dummy_na=False, dtype=int)
    x["Prior_therapy"] = x["Prior_therapy"].astype("category").cat.codes
    x["Treatment"] = x["Treatment"].astype("category").cat.codes
    num_cols = ["Age_in_years", "Karnofsky_score", "Months_from_Diagnosis"]
    for col in num_cols:
        x[col] = x[col].fillna(x[col].median())
    return {
        "x": x,
        "t": t,
        "e": e,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "covariates": x.columns.tolist(),
    }


def _load_lung():
    import lifelines

    df = lifelines.datasets.load_lung()
    e = df["status"]
    t = df["time"]
    df = df.drop(columns=["time", "status"])
    df["ph.ecog"] = df["ph.ecog"].fillna(df["ph.ecog"].median()).astype(int)
    cat_cols = ["ph.ecog", "sex"]
    df["sex"] = df["sex"] - 1
    num_cols = ["inst", "age", "meal.cal", "wt.loss", "pat.karno", "ph.karno"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return {
        "x": df,
        "t": t.values,
        "e": e.values,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "covariates": df.columns.tolist(),
    }


if __name__ == "__main__":
    for name in ALL_DATASETS:
        print("Loading dataset:", name)
        out = load_dataset(name, sequential=False)
        print(f"  {name}: {out['x'].shape[0]} samples, {out['x'].shape[1]} features, "
              f"event rate {np.mean(out['e']) * 100:.1f}%, "
              f"time range [{np.min(out['t']):.2f}, {np.max(out['t']):.2f}]")
