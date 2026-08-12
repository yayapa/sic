"""Define argument parser for TabICL training."""

import argparse


def str2bool(value):
    return value.lower() == "true"


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def build_parser():
    """Build parser with all TabICL training arguments."""
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument("--wandb_log", default=False, type=str2bool, help="Log results using wandb")
    parser.add_argument("--wandb_project", type=str, default="TabICL", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb logging directory")
    parser.add_argument(
        "--wandb_mode", default="offline", type=str, help="Wandb logging mode: online, offline, or disabled"
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="Data type (supported for float16, float32) used for training"
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", help="Learning rate scheduler: see optim.py for options."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay / L2 regularization penalty")
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--cosine_lr_end", type=float, default=0, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--prior_dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "If set, load pre-generated prior datasets from multiple directories and sample between them each step. "
            "Overrides --prior_dir when provided."
        ),
    )
    parser.add_argument(
        "--prior_mix_probs",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Mixing probabilities for --prior_dirs (one per directory). "
            "If omitted, uses uniform probabilities."
        ),
    )
    parser.add_argument(
        "--prior_mix_seed",
        type=int,
        default=None,
        help="Random seed for sampling among --prior_dirs. Defaults to --np_seed if not set.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="The maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="The maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type", default="mix_scm", type=str, help="Prior type: dummy, mlp_scm, tree_scm, mix_scm"
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )

    # Column Embedding Config
    parser.add_argument("--embed_dim", type=int, default=128, help="Base embedding dimension")
    parser.add_argument("--col_num_blocks", type=int, default=3, help="Number of blocks in column embedder")
    parser.add_argument("--col_nhead", type=int, default=4, help="Number of attention heads in column embedder")
    parser.add_argument("--col_num_inds", type=int, default=128, help="Number of inducing points in column embedder")
    parser.add_argument("--freeze_col", default=False, type=str2bool, help="Whether to freeze the column embedder")

    # Row Interaction Config
    parser.add_argument("--row_num_blocks", type=int, default=3, help="Number of blocks in row interactor")
    parser.add_argument("--row_nhead", type=int, default=8, help="Number of attention heads in row interactor")
    parser.add_argument("--row_num_cls", type=int, default=4, help="Number of CLS tokens in row interactor")
    parser.add_argument("--row_rope_base", type=float, default=100000, help="RoPE base value for row interactor")
    parser.add_argument("--freeze_row", default=False, type=str2bool, help="Whether to freeze the row interactor")

    # ICL Config
    parser.add_argument("--icl_num_blocks", type=int, default=12, help="Number of transformer blocks in ICL predictor")
    parser.add_argument("--icl_nhead", type=int, default=4, help="Number of attention heads in ICL predictor")
    parser.add_argument("--freeze_icl", default=False, type=str2bool, help="Whether to freeze the ICL predictor")

    # Shared Architecture Config
    parser.add_argument("--ff_factor", type=int, default=2, help="Expansion factor for feedforward dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function type")
    parser.add_argument(
        "--norm_first", default=True, type=str2bool, help="If True, use pre-norm transformer architecture"
    )
    parser.add_argument("--mode", type=str, default="survival", help="Task mode: classification or survival")
    parser.add_argument("--num_bins", type=int, default=100, help="Number of bins for survival analysis")

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")

    parser.add_argument("--freeze_col_embedder", default=False, type=str2bool, help="Freeze column embedder")
    parser.add_argument("--freeze_row_interactor", default=False, type=str2bool, help="Freeze row interactor")
    parser.add_argument("--freeze_icl_predictor", default=False, type=str2bool, help="Freeze ICL predictor")
    parser.add_argument("--surv_loss", type=str, default="deephit", choices=["deephit", "coxph", "deepeh", "nll", "bce", "combo_bce"], help="Survival loss function (nll/bce/combo_bce are for the sumo_pp head)")
    parser.add_argument("--surv_model_type", type=str, default="deephit", choices=["deephit", "coxph", "deepeh", "sumo_pp"], help="Survival head type")
    parser.add_argument("--disable_ddp", default=False, type=str2bool, help="Force disable DDP even if RANK is set")
    parser.add_argument("--time_binning_scheme", type=str, default="quantiles", help="Time grid discretization scheme for discrete approaches for survival analysis")
    parser.add_argument("--deephit_loss_alpha", type=float, default=0.2, help="Alpha parameter for DeepHit loss")
    parser.add_argument("--deephit_loss_sigma", type=float, default=0.1, help="Sigma parameter for DeepHit loss")
    # --- sumo_pp head (nll / bce / combo_bce losses) ---
    parser.add_argument("--sumo_tscale_q", type=str, default="0.95", help="Time-scale rule for the sumo_pp head: 'max' (train maximum) or a quantile in (0,1], e.g. 0.95. Applied at training AND eval so they match; use the same value the checkpoint was trained with.")
    parser.add_argument("--bce_weight", type=float, default=0.3, help="Weight on the BCE term when surv_loss=combo_bce (loss = NLL + bce_weight*BCE; 0 = pure NLL). 0.3 is the validated value; larger weights (e.g. 1.0) destabilize training.")
    parser.add_argument("--bce_sigma", type=float, default=0.1, help="sigma_frac for the BCE time perturbation (sigma = bce_sigma*t_scale); used by surv_loss in {bce, combo_bce}")
    parser.add_argument("--use_xcal", default=False, type=str2bool, help="Whether to use X-CAL (only applied when surv_loss=deephit)")
    parser.add_argument("--xcal_bins", type=int, default=20, help="Number of bins for X-CAL")
    parser.add_argument("--xcal_gamma", type=float, default=1.0, help="Gamma parameter for X-CAL (soft membership sharpness)")
    parser.add_argument("--xcal_lam", type=float, default=1.0, help="Regularization strength for X-CAL: loss = nll + xcal_lam * d_cal")
    return parser
