"""Model configuration and the survival-head registry.

``HEAD_SPECS`` maps the user-facing ``survival_head`` name to
    * the internal ``surv_model_type`` understood by the model,
    * the number of head outputs (``num_bins``),
    * the checkpoint filename to download from the Hugging Face Hub.

``DEFAULT_MODEL_CONFIG`` holds the transformer backbone hyperparameters used by
the released SIC checkpoints. When a checkpoint stores its own ``config`` dict
(the release format), that takes precedence over these defaults.
"""

from __future__ import annotations

# Backbone hyperparameters shared by all released SIC checkpoints. These match
# the configuration used to pre-train the released models.
DEFAULT_MODEL_CONFIG = {
    "max_classes": 10,
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 4,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "row_num_cls": 4,
    "row_rope_base": 100000,
    "icl_num_blocks": 12,
    "icl_nhead": 4,
    "ff_factor": 2,
    "dropout": 0.0,
    "activation": "gelu",
    "norm_first": True,
    "mode": "survival",
}

# survival_head -> (surv_model_type, num_bins, checkpoint filename on the Hub)
HEAD_SPECS = {
    "DeepHit10": {"surv_model_type": "deephit", "num_bins": 10, "filename": "DeepHit10_paper.ckpt"},
    "DeepHit50": {"surv_model_type": "deephit", "num_bins": 50, "filename": "DeepHit50_paper.ckpt"},
    "DeepHit100": {"surv_model_type": "deephit", "num_bins": 100, "filename": "DeepHit100_paper.ckpt"},
    "CoxPH": {"surv_model_type": "coxph", "num_bins": 1, "filename": "CoxPH.ckpt"},
    "DeepEH": {"surv_model_type": "deepeh", "num_bins": 2, "filename": "DeepEH.ckpt"},
    # Continuous-time SuMo++ head; num_bins is unused by the head (kept = 1).
    "SuMoPP": {"surv_model_type": "sumo_pp", "num_bins": 1, "filename": "SuMoPP.ckpt"},
}

SURVIVAL_HEADS = list(HEAD_SPECS.keys())


def resolve_head(survival_head: str) -> dict:
    """Return the spec dict for ``survival_head`` or raise a clear error."""
    if survival_head not in HEAD_SPECS:
        raise ValueError(
            f"Unknown survival_head={survival_head!r}. "
            f"Choose one of {SURVIVAL_HEADS}."
        )
    return HEAD_SPECS[survival_head]


def build_model_config(survival_head: str, overrides: dict | None = None) -> dict:
    """Compose a full model config for a given head from the defaults."""
    spec = resolve_head(survival_head)
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg["surv_model_type"] = spec["surv_model_type"]
    cfg["num_bins"] = spec["num_bins"]
    if overrides:
        cfg.update(overrides)
    return cfg
