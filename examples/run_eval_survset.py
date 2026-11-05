#!/usr/bin/env python
"""Evaluate a SIC checkpoint with k-fold CV and write metrics to Excel + JSON.

Default: the whole benchmark, ``sic.list_datasets()`` = all SurvSet datasets +
METABRIC (bundled) + the restricted-access SEER / UNOS when their CSVs are found
under ``--data_dir`` (or ``$SIC_DATA_DIR``). SurvSet datasets need the ``SurvSet``
and ``skrub`` packages (``pip install -e ".[survset]"``).

Protocol: ``sic.metrics.cross_validate`` (k-fold CV, per-fold median impute +
StandardScaler on numeric covariates, test-time grid).

Examples::

    # Full benchmark with the released SuMoPP weights (Hugging Face download)
    python examples/run_eval_survset.py

    # Local checkpoint, smoke run on two datasets
    python examples/run_eval_survset.py \\
        --survival_head SuMoPP --model_path /path/to/SuMoPP.ckpt \\
        --datasets hdfail METABRIC --k_folds 2

Any name accepted by ``sic.load_dataset`` works in ``--datasets``
(SurvSet ids are case-sensitive).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from sic import SIC, SURVIVAL_HEADS, list_datasets
from sic.metrics import QUANTILE_LEVELS, SCALAR_METRICS, cross_validate, summarize

_HEAD_ALIASES = {"sumo": "SuMo", "sumopp": "SuMoPP", "sumo_pp": "SuMoPP", "sumo++": "SuMoPP"}


def _all_metric_names() -> list[str]:
    names = list(SCALAR_METRICS)
    for prefix in ("auc", "bs", "cindex"):
        names += [f"{prefix}@{q}" for q in QUANTILE_LEVELS]
    return names


def _resolve_head(name: str) -> str:
    key = name.strip()
    if key in SURVIVAL_HEADS:
        return key
    alias = _HEAD_ALIASES.get(key.lower())
    if alias is None:
        raise SystemExit(f"Unknown survival_head={name!r}. Choose one of {SURVIVAL_HEADS}.")
    return alias


def _default_datasets() -> list[str]:
    try:
        import skrub  # noqa: F401
        import SurvSet  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"SurvSet and skrub are required for the default benchmark ({exc}). "
            'Install with: pip install -e ".[survset]"'
        ) from exc
    return list_datasets()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--survival_head", "--head", dest="survival_head", default="SuMoPP",
                   help=f"Survival head (default: SuMoPP). One of {SURVIVAL_HEADS}.")
    p.add_argument("--model_path", default=None,
                   help="Local checkpoint. If omitted, downloads the released weights.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Override the dataset list (default: full benchmark).")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="Folder with restricted CSVs (unos.csv, seernfg.csv); sets $SIC_DATA_DIR.")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=8,
                   help="Feature-permutation ensemble size (default: 8). Use 1 to disable.")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    p.add_argument("--out", type=Path, default=Path("eval_survset.xlsx"))
    p.add_argument("--json-out", type=Path, default=None,
                   help="Nested CV JSON (default: nested_cv_results.json beside --out).")
    return p.parse_args(argv)


def _write_xlsx(out: Path, folds: pd.DataFrame, summary: pd.DataFrame, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    mean_cols = [c for c in summary.columns if c.endswith("_mean")]
    means = summary[mean_cols].rename(columns=lambda c: c[: -len("_mean")])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_info", index=False)
        folds.to_excel(writer, sheet_name="folds", index=False)
        summary.to_excel(writer, sheet_name="summary")
        means.to_excel(writer, sheet_name="means")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.data_dir is not None:
        os.environ["SIC_DATA_DIR"] = str(args.data_dir.expanduser())
    if args.n_estimators < 1:
        raise SystemExit(f"--n_estimators must be >= 1, got {args.n_estimators}.")
    if args.model_path is not None and not Path(args.model_path).expanduser().is_file():
        raise SystemExit(f"Checkpoint not found: {args.model_path}")

    head = _resolve_head(args.survival_head)
    datasets = args.datasets or _default_datasets()

    model = SIC(
        survival_head=head,
        model_path=args.model_path,
        device=args.device,
        standardize=False,  # cross_validate imputes + scales numeric columns per fold
        allow_auto_download=args.model_path is None,
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
    print(
        f"Evaluating {head} from {args.model_path or 'released weights'} on "
        f"{len(datasets)} dataset(s), k={args.k_folds}, seed={args.seed}, "
        f"n_estimators={args.n_estimators}\nDatasets: {', '.join(datasets)}",
        flush=True,
    )

    json_out = args.json_out or args.out.parent / "nested_cv_results.json"
    folds = cross_validate(
        model,
        datasets,
        k_folds=args.k_folds,
        seed=args.seed,
        verbose=True,
        model_name=head,
        json_out=json_out,
    )
    metric_names = tuple(m for m in _all_metric_names() if m in folds.columns)
    summary = summarize(folds, metrics=metric_names)
    summary = summary.reindex([d for d in datasets if d in summary.index])

    meta = {
        "survival_head": head,
        "model_path": args.model_path or "",
        "datasets": ",".join(datasets),
        "k_folds": args.k_folds,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "device": str(model.device),
        "n_fold_rows": len(folds),
    }
    _write_xlsx(args.out, folds, summary, meta)
    print(f"Wrote {args.out.resolve()} and {Path(json_out).resolve()}")
    print(summary[[m for m in ("cindex_td", "ibs", "d_cal") if m in summary.columns]])


if __name__ == "__main__":
    main()
