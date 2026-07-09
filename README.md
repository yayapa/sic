# SIC: Survival In-Context

The official implementation of the paper [Survival In-Context: Amortized Bayesian Survival Analysis via Prior-Fitted Networks](https://arxiv.org/pdf/2603.29475)

SIC brings the PFN and ICL paradigms to survival analysis. SIC requires no hyperparameter tuning and no training, performing predictions using the training dataset as context.

## Installation

With these steps you can develop and change the model as well as execute on your own datasets. (You may want to omit some steps to use the package as-is.)


### Using Conda

```bash
git clone https://github.com/yayapa/sic
cd sic
conda env create -f env.yml      # installs the package editable with [dev] extras
conda activate sic
pip install -e .
```

Note 1: For installing pre-commit hooks you can use (not necessary for the usage): `pip install -e ".[dev]"` then run `bash scripts/install-git-hooks.sh`.

Note 2: Model weights and the bundled benchmark data are hosted at [yayapa/sic](https://huggingface.co/yayapa/sic) and downloaded automatically on first use, then cached locally. Override the source repo with the `SIC_HF_REPO` environment variable and the cache/download location with `SIC_CACHE_DIR` (or the standard `HF_HOME`).


### Using uv

```bash
git clone https://github.com/yayapa/sic
cd sic
uv sync
```


## Quick start

```python
from sic import SIC, make_y, load_dataset

data = load_dataset("METABRIC")
X, t, e = data["x"], data["t"], data["e"]

# split however you like
X_train, X_test = X.iloc[:1500], X.iloc[1500:]
y_train = make_y(t[:1500], e[:1500])          # scikit-survival style labels

model = SIC()     
surv = model.fit_predict_surv(
    X_train, y_train, X_test,
    eval_times=[1, 2, 3, 4, 5],             # years; None -> model-native grid
)
# surv: DataFrame, index = eval_times, columns = test subjects (EvalSurv convention)
```

## Examples

Everything you need to know about running the model is in the example notebooks:

- `examples/sic_walkthrough.ipynb` - notebook tour of SIC: raw outputs, calibrated curves, the metric suite, and CV results.
- `examples/sic_head_comparison.ipynb` - cross-validate all heads on every dataset; grouped C-index / IBS bar plots and a miscalibrated-folds heatmap, datasets sorted by sample size.
- `examples/inspect_prior_batch.ipynb` - load one synthetic prior batch and plot Kaplan–Meier curves and event times.

### Survival heads

SIC is not survival-head independent. The default is the `DeepHit10` head. We also have other head checkpoints one may want to try:

| `survival_head`                              | model                  | output                                                         |
| ---------------------------------------------- | ---------------------- | -------------------------------------------------------------- |
| `DeepHit10` / `DeepHit50` / `DeepHit100` | discrete-time DeepHit  | 10 / 50 / 100 bins                                             |
| `CoxPH`                                      | proportional hazards   | 1 log-hazard / subject                                         |
| `DeepEH`                                     | extended hazard        | 2 hazards / subject                                            |
| `SuMo` (TODO)                                | Neural ODE formulation | S(t,x) / subject where x and specific t are required for input |

### Post-training calibration

We found that the post-training calibration methods [CiPOT](https://arxiv.org/pdf/2410.20579v3) and [CSD](https://arxiv.org/pdf/2405.07374) may be beneficial for improving IBS and D-calibration while retaining C-index performance:

```python
surv = model.fit_predict_surv(
    X_train, y_train, X_test, eval_times=times,
    post_training_calibration="cipot",   # or "csd"
)
```

## Benchmarking

`sic.list_datasets()` returns the supported benchmarks. Real data is obtained without manual setup: some datasets are auto-downloaded by `pycox`, others ship with `scikit-survival` / `lifelines`, and a few small ones are bundled as CSVs. The restricted-access SEER and UNOS datasets are not included.

## Generating Prior and training your own model

See `examples/` for scripts and notebooks. Workflow:

```bash
bash examples/run_genload_continuous.sh   # synthetic continuous priors
bash examples/run_genload_discrete.sh     # synthetic discrete priors
bash examples/run_train_mix.sh            # train on a mix (run genload first)
```

Edit paths and hyperparameters at the top of each script. Inspect a prior batch: `examples/inspect_prior_batch.ipynb`.

**wandb:** `run_train_mix.sh` logs to [wandb](https://wandb.ai) by default (`--wandb_mode online`). Set `WANDB_API_KEY` or run `wandb login` (requires **wandb ≥ 0.22.3** for current API keys). Use `--wandb_mode offline` or `--wandb_log False` to skip cloud logging.

## Baselines

To increase readability, we plan to release the code for training baselines using hyperparameter tuning in optuna as a separate repository later. TODO

## Acknowledgments

We thank the authors of [TabICL](https://arxiv.org/pdf/2502.05564) for making PFNs open source! We used their codebase to develop our model.

Additionally, we actively used in the development the work [CiPOT](https://arxiv.org/pdf/2410.20579v3), [CSD](https://arxiv.org/pdf/2405.07374), and [torchssurv](https://github.com/novartis/torchsurv). We thank the authors for making their code public.

## LLM Note

The original research code was written by [yayapa](https://github.com/yayapa/). However, for refactoring and publishing the package, LLMs and agentic mode have been actively used. If you see any problems, please report them using issues.

## License

MIT (see `LICENSE`). Includes code derived from TabICL (BSD 3-Clause) and a
vendored copy of TorchSurv (MIT). See `THIRD_PARTY_NOTICES`.
