"""Benchmark survival datasets.

Every loader returns a uniform dict::

    {
        'x': pd.DataFrame,      # covariates
        't': np.ndarray,        # durations (float)
        'e': np.ndarray,        # event indicators (0/1)
        'covariates': list,
        'num_cols': list,
        'cat_cols': list,
    }

The benchmark (:func:`list_datasets`) is:

* all SurvSet datasets in :data:`sic.train.survset.SURVSET_DATASETS`
  (needs the optional ``SurvSet`` and ``skrub`` packages: ``pip install sic[survset]``),
* ``METABRIC``, bundled with this package (DeepSurv/pycox release),
* ``SEER`` and ``UNOS`` when available. They are restricted-access and not
  distributed: obtain the CSVs yourself (``seernfg.csv``, ``unos.csv``) and point
  ``$SIC_DATA_DIR`` at their folder.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from .survset import is_survset_dataset, list_survset_datasets, load_survset_dataset

EPS = 1e-8

# Restricted-access CSVs (SEER / UNOS), not distributed: set $SIC_DATA_DIR to their folder.
_DATA_DIR_ENV = "SIC_DATA_DIR"
RESTRICTED_DATASETS = {"SEER": "seernfg.csv", "UNOS": "unos.csv"}


def restricted_data_dir() -> Path:
    return Path(os.environ.get(_DATA_DIR_ENV, "datasets")).expanduser()


def _restricted_csv(name: str) -> Path:
    path = restricted_data_dir() / RESTRICTED_DATASETS[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} is restricted-access and not shipped with SIC. Place "
            f"{path.name} in a folder and set ${_DATA_DIR_ENV} to it (looked for {path})."
        )
    return path


def available_restricted_datasets() -> list[str]:
    """Restricted datasets whose CSV exists under :func:`restricted_data_dir`."""
    return [n for n, f in RESTRICTED_DATASETS.items() if (restricted_data_dir() / f).is_file()]


def list_datasets() -> list[str]:
    """The benchmark: SurvSet datasets, METABRIC, and SEER / UNOS if their CSVs are found."""
    return list_survset_datasets() + ["METABRIC"] + available_restricted_datasets()


def _bundled_csv(name: str) -> pd.DataFrame:
    """Read a CSV that is bundled as package data under ``sic/train/datasets``."""
    with resources.as_file(resources.files(__package__).joinpath("datasets", name)) as path:
        df = pd.read_csv(path)
    # Drop the unnamed R row-index column that ships with some CSVs.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def load_dataset(dataset: str = "METABRIC", sequential: bool = False) -> dict:
    """Load a benchmark dataset by name (see :func:`list_datasets`).

    SurvSet names are case-sensitive (e.g. ``veteran``, ``GBSG2``). ``sequential``
    is accepted for backward compatibility and ignored.
    """
    if is_survset_dataset(dataset):
        try:
            return load_survset_dataset(dataset)
        except ImportError as exc:
            raise ImportError(
                f"SurvSet dataset '{dataset}' needs the SurvSet and skrub packages: "
                'pip install -e ".[survset]"'
            ) from exc
    name = dataset.upper()
    if name == "METABRIC":
        return _load_metabric_dataset()
    if name == "SEER":
        return _load_seer_dataset()
    if name == "UNOS":
        return _load_unos()
    raise NotImplementedError(
        f"Dataset '{dataset}' not available. Benchmark datasets: {list_datasets()}"
    )


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


def _load_unos():
    df = pd.read_csv(_restricted_csv("UNOS"))
    x = df.drop(columns=["Survival_Time_in_days", "Event"])
    t = df["Survival_Time_in_days"].values.astype("float64")
    e = df["Event"].values.astype("int32")

    continuous_1 = [
        "INIT_AGE", "HGT_CM_CALC", "WGT_KG_CALC", "NUM_PREV_TX",
        "CREAT_TRR", "TBILI", "DAYS_STAT1", "DAYS_STAT1A", "DAYS_STAT2",
        "DAYS_STAT1B", "BMI_CALC", "AGE_DON", "HGT_CM_DON_CALC", "WGT_KG_DON_CALC",
        "BMI_DON_CALC", "ISCHTIME", "TBILI_DON", "CREAT_DON", "LV_EJECT", "HLAMIS",
    ]
    continuous_2 = ["PRAMR", "PRAMR_CL1", "PRAMR_CL2", "PRAPK", "PRAPK_CL1", "PRAPK_CL2"]

    categorical_1 = [
        "GENDER", "INFECT_IV_DRUG_TRR", "TRANSFUSIONS", "PREV_TX", "ABO", "DIAL_PRIOR_TX",
        "INOTROPIC", "GENDER_DON", "ABO_DON",
        "HEP_C_ANTI_DON", "CLIN_INFECT_DON", "VENTILATOR_TRR", "ECMO_TRR", "IABP_TRR",
        "AMIS", "BMIS", "DRMIS",
    ]
    categorical_2 = ["DIAB", "VAD_TAH_TRR", "HIST_DIABETES_DON", "ABO_MAT"]
    num_cols = continuous_1 + continuous_2
    cat_cols = categorical_1 + categorical_2

    for col in x.columns:
        if type(x[col].iloc[0]) == np.bool_:
            x[col] = x[col].astype(np.int32)

    return {
        "x": x,
        "t": t,
        "e": e,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "covariates": x.columns.tolist(),
    }


def _load_seer_dataset(dropallnans=True, competing=False):
    # based on https://github.com/Jeanselme/NeuralFineGray/.../nfg/datasets.py
    df = pd.read_csv(_restricted_csv("SEER"))
    # Remove multiple visits
    df = df.groupby("Patient ID").first().drop(columns=["Site recode ICD-O-3/WHO 2008"])

    # Encode using dictionary to remove missing data
    df["RX Summ--Surg Prim Site (1998+)"].replace("126", np.nan, inplace=True)
    df["Sequence number"].replace(["88", "99"], np.nan, inplace=True)
    df["Regional nodes positive (1988+)"].replace(
        ["95", "96", "97", "98", "99", "126"], np.nan, inplace=True
    )
    df["Regional nodes examined (1988+)"].replace(
        ["95", "96", "97", "98", "99", "126"], np.nan, inplace=True
    )
    df = (
        df.replace(
            ["Blank(s)", "Unknown", "Recode not available", "NA", "null", "None"],
            np.nan,
        ).rename(columns={"Survival months": "duration"})
    )
    df = df.rename(columns={"Grade Recode (thru 2017)": "Grade"})
    if dropallnans:
        df = df.dropna()

    df = df[~df.duration.isna()]

    df["duration"] = df["duration"].astype(float)
    df["event"] = (
        df["SEER cause-specific death classification"]
        == "Dead (attributable to this cancer dx)"
    )
    df.loc[
        (df["COD to site recode"] == "Diseases of Heart")
        & (df["SEER cause-specific death classification"] == "Alive or dead of other cause"),
        "event",
    ] = 2
    df["event"] = df["event"].astype(int)
    if not competing:
        df.loc[df["event"] == 2, "event"] = 0  # treat competing events as censored

    df = df.drop(columns=["COD to site recode"])

    categorical_col = [
        "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
        "Laterality",
        "Diagnostic Confirmation",
        "Histology recode - broad groupings",
        "Chemotherapy recode (yes, no/unk)",
        "Radiation recode",
        "ER Status Recode Breast Cancer (1990+)",
        "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3",
        "ICD-O-3 Hist/behav, malignant",
        "Sequence number",
        "RX Summ--Surg Prim Site (1998+)",
        "CS extension (2004-2015)",
        "CS lymph nodes (2004-2015)",
        "CS mets at dx (2004-2015)",
        "Origin recode NHIA (Hispanic, Non-Hisp)",
    ]
    df[categorical_col] = df[categorical_col].apply(
        lambda col: col.astype("object").where(col.notna(), np.nan)
    )

    ordinal_col = ["Age recode with <1 year olds", "Grade", "Year of diagnosis"]

    imputer = SimpleImputer(strategy="most_frequent")
    df_cat_imputed = pd.DataFrame(
        imputer.fit_transform(df[categorical_col]),
        columns=categorical_col,
        index=df.index,
    )
    for col in categorical_col:
        df_cat_imputed[col] = df_cat_imputed[col].astype(str)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_cat = pd.DataFrame(
        enc.fit_transform(df_cat_imputed),
        columns=categorical_col,
        index=df.index,
    )

    df_ord = pd.DataFrame(
        imputer.fit_transform(df[ordinal_col]), columns=ordinal_col, index=df.index
    )
    df_ord = df_ord.replace(
        {
            age: number
            for number, age in enumerate(
                [
                    "01-04 years", "05-09 years", "10-14 years", "15-19 years",
                    "20-24 years", "25-29 years", "30-34 years", "35-39 years",
                    "40-44 years", "45-49 years", "50-54 years", "55-59 years",
                    "60-64 years", "65-69 years", "70-74 years", "75-79 years",
                    "80-84 years", "85+ years",
                ]
            )
        }
    ).replace(
        {
            grade: number
            for number, grade in enumerate(
                [
                    "Well differentiated; Grade I",
                    "Moderately differentiated; Grade II",
                    "Poorly differentiated; Grade III",
                    "Undifferentiated; anaplastic; Grade IV",
                ]
            )
        }
    )

    numerical_col = [
        "Total number of in situ/malignant tumors for patient",
        "Total number of benign/borderline tumors for patient",
        "CS tumor size (2004-2015)",
        "Regional nodes examined (1988+)",
        "Regional nodes positive (1988+)",
    ]
    imputer = SimpleImputer(strategy="mean")
    df_num = pd.DataFrame(
        imputer.fit_transform(df[numerical_col].astype(float)),
        columns=numerical_col,
        index=df.index,
    )

    return {
        "x": pd.concat([df_cat, df_num, df_ord], axis=1),
        "t": df["duration"].values,
        "e": df["event"].values,
        "covariates": np.concatenate([df_cat.columns, df_num.columns, df_ord.columns]),
        "num_cols": numerical_col,
        "cat_cols": categorical_col + ordinal_col,
    }


if __name__ == "__main__":
    for name in list_datasets():
        out = load_dataset(name)
        print(f"{name}: {out['x'].shape[0]} samples, {out['x'].shape[1]} features, "
              f"event rate {np.mean(out['e']) * 100:.1f}%, "
              f"time range [{np.min(out['t']):.2f}, {np.max(out['t']):.2f}]")
