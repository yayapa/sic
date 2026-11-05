from __future__ import annotations

import datetime
import os
import time
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import torch
from torch import nn, tensor
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

from sic.model.tabicl import TabICL
from sic.prior.dataset import PriorDataset
from sic.prior.genload import LoadPriorDataset, MixedLoadPriorDataset
from sic.train.optim import get_scheduler
from sic.train.train_config import build_parser
from pycox.evaluation import EvalSurv
import pandas as pd
from sic._vendor.torchsurv.loss.cox import neg_partial_log_likelihood
from sic._vendor.torchsurv.metrics.cindex import ConcordanceIndex
import lifelines
from sklearn.model_selection import train_test_split
from sic.train.survUtils import (
    predict_surv_deephit, predict_pmf_deephit, predict_surv_new_times_deephit,
    predict_surv_coxph_2, predict_surv_deepeh, deepeh_loss,
    sum_concordant_antolini, d_calibration,
)
from sic.model.sumo_utils import (
    nll_loss, bce_loss, combo_bce_loss, predict_surv_sumo, compute_time_norm,
    true_targets, curve_ce_loss, CONTINUOUS_SURV_TYPES,
)
from pycox.models.loss import DeepHitSingleLoss
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.data import pair_rank_mat
from sic.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

# disable warnings
warnings.filterwarnings("ignore")


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def _split_batch_item(t, size: int):
    """torch.split along datasets; survival targets are a dict of tensors, split per key."""
    if isinstance(t, dict):
        return [dict(zip(t, parts)) for parts in zip(*(torch.split(v, size, dim=0) for v in t.values()))]
    return torch.split(t, size, dim=0)


def _has_nan(y) -> bool:
    return any(torch.isnan(v).any() for v in y.values()) if isinstance(y, dict) else bool(torch.isnan(y).any())


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.ddp = False if self.config.disable_ddp else self.ddp
        print("DDP:", self.ddp)

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""
        wandb.login(key=os.getenv("WANDB_API_KEY", None))
        datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.config.wandb_log and self.master_process:
            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name + f"_-{datetime_str}",
                #id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )
        else:
            self.wandb_run = None

        # --- new helper ---
    def _freeze_components(self, model=None):
        model = model or self.raw_model
        # accept both flag spellings
        freeze_col = getattr(self.config, "freeze_col_embedder", None)
        freeze_row = getattr(self.config, "freeze_row_interactor", None)
        freeze_icl = getattr(self.config, "freeze_icl_predictor", None)

        if freeze_col:
            print("Freezing column embedder")
            model.col_embedder.eval()
            for p in model.col_embedder.parameters():
                p.requires_grad = False
        if freeze_row:
            print("Freezing row interactor")
            model.row_interactor.eval()
            for p in model.row_interactor.parameters():
                p.requires_grad = False
        if freeze_icl:
            print("Freezing ICL predictor")
            model.icl_predictor.eval()
            for p in model.icl_predictor.parameters():
                p.requires_grad = False

    
    def build_model(self):
        """Build and initialize the TabICL model."""

        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "mode": self.config.mode,
            "num_bins": self.config.num_bins,
            "surv_model_type": self.config.surv_model_type,
            "sumo_tscale_q": self.config.sumo_tscale_q,
            "cens_head": self.config.cens_head,
        }

        model = TabICL(**self.model_config)
        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")
        
        # Freeze requested components BEFORE optimizer creation
        self._freeze_components(model)

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self):
        """
        Sets up a tabular dataset generator that creates synthetic datasets
        during training with controllable properties and data distributions.
        Note configuring prior and training on-the-fly DOES NOT WORK NOW
        """

        # Resolve multi-prior configuration (backward compatible with --prior_dir).
        prior_dirs = getattr(self.config, "prior_dirs", None)
        if prior_dirs is None and isinstance(getattr(self.config, "prior_dir", None), str):
            # Allow comma-separated prior_dir for convenience.
            if "," in self.config.prior_dir:
                prior_dirs = [p.strip() for p in self.config.prior_dir.split(",") if p.strip()]

        if prior_dirs is None and self.config.prior_dir is None:
            # Generate prior data on the fly
            #dataset = PriorDataset(
            #    batch_size=self.config.batch_size,
            #    batch_size_per_gp=self.config.batch_size_per_gp,
            #    min_features=self.config.min_features,
            #    max_features=self.config.max_features,
            #    max_classes=self.config.max_classes,
            #    min_seq_len=self.config.min_seq_len,
            #    max_seq_len=self.config.max_seq_len,
            #    log_seq_len=self.config.log_seq_len,
            #    seq_len_per_gp=self.config.seq_len_per_gp,
            #    min_train_size=self.config.min_train_size,
            #    max_train_size=self.config.max_train_size,
            #    replay_small=self.config.replay_small,
            #    prior_type=self.config.prior_type,
            #    device=self.config.prior_device,
            #    n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
            #    reg2surv=self.config.mode == "survival",
            #)
            dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                scm_fixed_hp=DEFAULT_FIXED_HP,
                scm_sampled_hp=DEFAULT_SAMPLED_HP,
                n_jobs=1,#self.args.n_jobs,
                num_threads_per_generate=1,
                device=self.config.prior_device,
                reg2surv=True#self.config.reg2surv
            )
        else:
            # Load pre-generated prior data from disk
            print("Loading prior dataset from disk...")
            print("Rank:", self.ddp_rank, "World Size:", self.ddp_world_size)
            if prior_dirs is None:
                dataset = LoadPriorDataset(
                    data_dir=self.config.prior_dir,
                    batch_size=self.config.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.load_prior_start,
                    delete_after_load=self.config.delete_after_load,
                    device=self.config.prior_device,
                    timeout=0,
                )
            else:
                datasets = [
                    LoadPriorDataset(
                        data_dir=d,
                        batch_size=self.config.batch_size,
                        ddp_world_size=self.ddp_world_size,
                        ddp_rank=self.ddp_rank,
                        start_from=self.config.load_prior_start,
                        delete_after_load=self.config.delete_after_load,
                        device=self.config.prior_device,
                        timeout=0,
                    )
                    for d in prior_dirs
                ]

                mix_probs = getattr(self.config, "prior_mix_probs", None)
                mix_seed = getattr(self.config, "prior_mix_seed", None)
                if mix_seed is None:
                    mix_seed = int(self.config.np_seed)
                # Make mixing deterministic but different across ranks.
                mix_seed = int(mix_seed) + int(self.ddp_rank)

                dataset = MixedLoadPriorDataset(datasets=datasets, mix_probs=mix_probs, seed=mix_seed)

        if self.master_process:
            print(dataset)

        # Create dataloader for efficient loading and prefetching
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device if self.config.prior_device == "cpu" else "",
        )
    


    def configure_optimizer(self):
        """Configure optimizer and scheduler."""

                # Only optimize trainable parameters
        trainable_params = [p for p in self.raw_model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            params=trainable_params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")
        

        res = self.raw_model.load_state_dict(checkpoint["state_dict"], strict=False)
        # re-init only the tensors that didn't load
        with torch.no_grad():
            miss = set(res.missing_keys)
            for name, p in self.raw_model.named_parameters():
                if name in miss:
                    if p.ndim >= 2:
                        nn.init.xavier_uniform_(p)          # weights (2D+)
                    elif "bias" in name.lower():
                        nn.init.zeros_(p)                   # biases
                    else:
                        nn.init.ones_(p)                    # 1D weights (e.g., LayerNorm gamma)
        
        if self.master_process:
            print(f"Loaded (strict=False). Missing: {res.missing_keys}, Unexpected: {res.unexpected_keys}")
        
        if self.config.freeze_col_embedder:
            self.raw_model.col_embedder.eval()
            for p in self.raw_model.col_embedder.parameters():
                p.requires_grad = False

        if self.config.freeze_row_interactor:
            self.raw_model.row_interactor.eval()
            for p in self.raw_model.row_interactor.parameters():
                p.requires_grad = False

        if self.config.freeze_icl_predictor:
            self.raw_model.icl_predictor.eval()
            for p in self.raw_model.icl_predictor.parameters():
                p.requires_grad = False
        
        
        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            print("Load states of optimizer and scheduler as well")
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            # Ensure initial_lr is set in optimizer param_groups after loading
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")
                
        if self.ddp:
            torch.distributed.barrier()

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        config = dict(self.model_config)
        if self.config.surv_model_type in CONTINUOUS_SURV_TYPES:
            # Already in model_config; keep explicit for older readers of the ckpt.
            config["sumo_tscale_q"] = self.config.sumo_tscale_q
        checkpoint = {
            "config": config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_progress:
            # Get the next batch
            with Timer() as prior_timer:
                batch = next(dataloader)
            
            prior_time = prior_timer.elapsed

            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            self.curr_step = step + 1
            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})
                results_test_on_gbsg2 = self.test_on_gbsg2()
                results.update(results_test_on_gbsg2)

                # Update progress bar with rounded values for cleaner display
                step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()
                        

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # Add learning rate to results
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)
                
    def test_on_gbsg2(self):
        df = lifelines.datasets.load_gbsg2()
        df_onehot = pd.get_dummies(df, columns=["horTh", "menostat", "tgrade"]).astype("float")
        df_onehot.drop(
            ["horTh_no", "menostat_Post", "tgrade_I"],
            axis=1,
            inplace=True,
        )
        df_train, df_test = train_test_split(df_onehot, test_size=0.3, random_state=self.config.np_seed)
        X_train = torch.tensor(df_train.drop(columns=["time", "cens"]).values, dtype=torch.float32).unsqueeze(0)
        T_train = torch.tensor(df_train["time"].values, dtype=torch.float32).unsqueeze(0)
        E_train = torch.tensor(df_train["cens"].values, dtype=torch.float32).unsqueeze(0)
        T_train = T_train.to(self.config.device)
        E_train = E_train.to(self.config.device)
        # y_train to the tensor of shape 1,2, train_size
        y_train = torch.stack([T_train, E_train], dim=1)
        X_test = torch.tensor(df_test.drop(columns=["time", "cens"]).values, dtype=torch.float32).unsqueeze(0)
        T_test = torch.tensor(df_test["time"].values, dtype=torch.float32).unsqueeze(0)
        E_test = torch.tensor(df_test["cens"].values, dtype=torch.float32).unsqueeze(0)
        
        if self.config.surv_model_type == "deephit":
            num_durations = self.config.num_bins
            labtrans = LabTransDiscreteTime(num_durations, scheme=self.config.time_binning_scheme)
            durations_train = T_train.squeeze(0).detach().cpu().numpy()
            events_train = E_train.squeeze(0).detach().cpu().numpy().astype(bool)
            labtrans.fit(durations_train, events_train)
            
        
        self.model.eval()
        with torch.no_grad():
            X = torch.cat([X_train, X_test], dim=1)
            X = X.to(self.config.device)
            y_train = y_train.to(self.config.device)
            T_test = T_test.to(self.config.device)
            E_test = E_test.to(self.config.device)
                    
            E_test = E_test.squeeze(0)
            T_test = T_test.squeeze(0)
            
            c_index_alternative = 0.0

            time_grid = T_test.cpu().numpy()
            time_grid = np.unique(np.sort(time_grid))
            
            if self.config.surv_model_type == "coxph":
                res = self.model(X, y_train)  #
                log_hazards = res["log_hazard"]  # (1, test_size, K)
                log_hazards_train = res["log_hazard_train"]  # (1, train_size, K)
                log_hazards = log_hazards.squeeze()  # (test_size, K)
                log_hazards_train = log_hazards_train.squeeze()  # (train_size, K)
                cindex = ConcordanceIndex()
                c_index_alternative = cindex(log_hazards, E_test.bool(), T_test).item()
                S, _ = predict_surv_coxph_2(log_hazards_train, E_train.squeeze(0), T_train.squeeze(0),
                                         log_hazards, torch.tensor(time_grid, device=log_hazards.device))
            elif self.config.surv_model_type == "deepeh":
                log_hazards = self.model(X, y_train)["log_hazard"]  # (1, test_size, K)
                log_hazards = log_hazards.squeeze()  # (test_size, K)
                log_hazards_train = self.model(X, y_train)["log_hazard_train"]  # (1, train_size, K)
                log_hazards_train = log_hazards_train.squeeze()  # (train_size, K)
                S, tt, dlt, ind_sort = predict_surv_deepeh(log_hazards_train, T_train.squeeze(0), E_train.squeeze(0),
                                                            log_hazards, T_test.squeeze(), E_test.squeeze(0))
                c_index_alternative = sum_concordant_antolini(S, tt, dlt)

            elif self.config.surv_model_type == "deephit":
                log_hazards = self.model(X, y_train)["log_hazard"]  # (1, test_size, K)
                log_hazards = log_hazards.squeeze(0)  # (test_size, K)
                
                time_grid_old = torch.tensor(labtrans.cuts, device=log_hazards.device)
                time_grid_new = torch.tensor(time_grid, device=log_hazards.device)
                
                S =  predict_surv_deephit(log_hazards)
                S = S.T
                # after the discussion came to conclusion that our evaluation on new time grid is sufficient
                #class MockDeepHit:
                #    def predict_surv(self, input, *args, **kwargs): return S
                
                # Interpolate (stays on device)
                #sub = 10
                #interpolator = InterpolateDiscrete(MockDeepHit(), scheme='const_pdf', duration_index=labtrans.cuts, sub=sub)
                #S = interpolator.predict_surv(None, numpy=False)
                #time_grid_new = pycox_utils.make_subgrid(labtrans.cuts, sub)

                S = predict_surv_new_times_deephit(S, time_grid_old, time_grid_new)

            elif self.config.surv_model_type in CONTINUOUS_SURV_TYPES:
                res = self.model(X, y_train)
                emb_test = res["emb_test"].squeeze(0)        # (Tq, d_model)
                head = res["sumo_head"]
                time_grid_t = torch.tensor(time_grid, device=emb_test.device, dtype=emb_test.dtype)
                t_scale = compute_time_norm(T_train, self.config.sumo_tscale_q)
                S = predict_surv_sumo(emb_test, time_grid_t, head, t_scale=t_scale)

            S = S.detach().cpu().numpy()

            if self.config.surv_model_type == "deepeh":
                # note: the default method for deepeh performs sorting internally
                # therefore, we need to use the tt and dlt returned from predict_surv_deepeh
                # also tested to sort back the S and time, and then use T_test and E_test, gives the same results
                S_df = pd.DataFrame(S.T, index=tt)
                eval_surv = EvalSurv(S_df, tt, dlt, censor_surv="km")
            elif self.config.surv_model_type == "deephit":
                #S_df = pd.DataFrame(S.T, index=time_grid_new.cpu().numpy())
                S_df = pd.DataFrame(S.T, index=time_grid)
                eval_surv = EvalSurv(S_df, T_test.cpu().numpy(), E_test.cpu().numpy(), censor_surv="km")
            else:
                S_df = pd.DataFrame(S.T, index=time_grid) #, index=times_b.cpu().numpy())
                eval_surv = EvalSurv(S_df, T_test.cpu().numpy(), E_test.cpu().numpy(), censor_surv="km")    

            ibs_time_grid = np.linspace(T_test.min().item(), T_test.max().item(), 100)
            ibs = eval_surv.integrated_brier_score(ibs_time_grid)
            c_index_td = eval_surv.concordance_td()
        self.model.train()
        #print(f"Test on GBSG2: C-index (td): {c_index_td:.4f}, C-index (Alternative): {c_index_alternative:.4f}, IBS: {ibs:.4f}")
        return {"test_on_gbsg2_cindex_td": c_index_td, "test_on_gbsg2_ibs": ibs, "test_on_gbsg2_cindex_alternative": c_index_alternative}
    
        

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """
        Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor (micro_batch_size,)
            Sequence lengths for each dataset.

        micro_train_size : Tensor (micro_batch_size,)
            Training sizes (split positions) for each dataset.

        Returns
        -------
        tuple (int, int)
            The common (seq_len, train_size) for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")

        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """
        Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (`micro_d.max()`) respectively. This optimizes
        memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor (B, T, H)
            Input features per dataset.

        micro_y : Tensor (B, T)
            Target labels per dataset.

        micro_d : Tensor (B,)
            Number of active features per dataset.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        tuple (Tensor, Tensor)
            Truncated (micro_X, micro_y) tensors with shapes
            (B, seq_len, micro_d.max()) and (B, seq_len).
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
        if isinstance(micro_y, dict):
            # Row-indexed keys are (B, L[, G]); t_grid is per dataset and keeps its length.
            micro_y = {k: v if k == "t_grid" else v[:, :seq_len] for k, v in micro_y.items()}
        elif micro_y.shape[2] > seq_len:
            micro_y = micro_y[:, :, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y
        

    
    def _repeat_edges_as_features(self, edges: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        edges: (K+1,) -> (seq_len, K+1) by repeating across rows
        """
        return edges.unsqueeze(0).expand(seq_len, -1)

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches, skip_computation=False):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors for the micro batch

        micro_batch_idx : int
            Index of the current micro batch

        num_micro_batches : int
            Total number of micro batches

        Returns
        -------
        dict
            Result dictionary
        """
        #self.test_on_gbsg2()

        
        # Set DDP gradient sync for last micro batch only
        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1
            
        if skip_computation:
            dummy_loss = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0.0
            scaled_loss = dummy_loss / num_micro_batches
            self.scaler.scale(scaled_loss).backward()
            return {}
        
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_y = (
            {k: v.to(self.config.device) for k, v in micro_y.items()}
            if isinstance(micro_y, dict) else micro_y.to(self.config.device)
        )
        micro_d = micro_d.to(self.config.device)
            
        micro_results = {}

        if torch.isnan(micro_X).any() or _has_nan(micro_y):
            return None

        if self.config.mode == "survival":
            T = micro_y["T_obs"]
            E = micro_y["E"]

            E_train, E_test = E[:, :train_size], E[:, train_size:]
            T_train, T_test = T[:, :train_size], T[:, train_size:]  # kept for future support-conditioning if needed
            if self.config.surv_target == "true":
                if "T_event" not in micro_y:
                    raise ValueError(
                        "surv_target='true' needs y with T_event and T_cens; "
                        f"got keys {sorted(micro_y)}. Regenerate the prior data."
                    )
                T_event_test = micro_y["T_event"][:, train_size:]
                T_cens_test = micro_y["T_cens"][:, train_size:]
            elif self.config.surv_target == "curve":
                if "S" not in micro_y:
                    raise ValueError(
                        "surv_target='curve' needs survival curves (y keys S, t_grid); "
                        "regenerate the prior with save_surv_curve: true."
                    )
                if self.config.surv_loss not in ("nll", "bce", "combo_bce"):
                    raise ValueError("surv_target='curve' is only implemented for the SuMo/SuMo++ heads")
                S_test = micro_y["S"][:, train_size:]
                t_grid = micro_y["t_grid"]
            
            B       = micro_X.shape[0]
            seq_len = micro_X.shape[1]              # already aligned by align_micro_batch
            K       = self.config.num_bins          # model head uses this K



            if self.config.surv_model_type == "deephit":
                bins_list = []
                for b in range(B):
                    labtrans = LabTransDiscreteTime(K, scheme=self.config.time_binning_scheme) # 'quantiles' or 'equidist' binning
                    try: 
                        labtrans.fit(T_train[b].cpu().numpy(), E_train[b].cpu().numpy())
                    except Exception as e:
                        print(f"Error in fit for batch {b}: {e}")
                        raise torch.cuda.OutOfMemoryError()
                    bins_b = labtrans
                    bins_list.append(bins_b)
            

            with self.amp_ctx:
                # 1) forward: model returns dict with query slice
                # Observed channels only: true times must never enter the context.
                y_train = torch.stack([T_train, E_train], dim=1)
                
                out = self.model(micro_X, y_train, micro_d)

                if self.config.surv_model_type in ["coxph", "deephit", "deepeh"]:
                    log_hazard = out["log_hazard"]
                    log_hazard_train = out["log_hazard_train"]
                    dt_list, is_event_list = [], []

                if self.config.surv_model_type == "deephit":
                    idx_duration_list, event_list, rank_mat_list = [], [], []
                    for b in range(B):
                        labtrans = bins_list[b]
                        try: 
                            idx_duration_b, event_b = labtrans.transform(T_test[b].cpu().numpy(), E_test[b].cpu().numpy())
                        except Exception as e:
                            print(f"Error in transform for batch {b}: {e}")
                            raise torch.cuda.OutOfMemoryError()
                        idx_duration_list.append(torch.tensor(idx_duration_b, device=self.config.device))
                        event_list.append(torch.tensor(event_b, device=self.config.device))
                        rank_mat_list.append(torch.tensor(pair_rank_mat(idx_duration_b, event_b)).to(self.config.device))
                
                if self.config.surv_loss == "coxph":
                    loss = tensor(0.0, device=self.config.device)
                    B_eff = 0
                    for b in range(B):
                        try:
                            loss_b = neg_partial_log_likelihood(log_hazard[b].squeeze(1), E_test[b].bool(), T_test[b], reduction='mean', ties_method='efron')
                        except Exception as e:
                            print(f"Error in neg_partial_log_likelihood for batch {b}: {e}")
                            continue
                        loss += loss_b
                        B_eff += 1
                    if B_eff > 0:
                        loss = loss / B_eff
                    else:
                        loss = loss + micro_X.sum() * 0.0
                elif self.config.surv_loss == "deephit":
                    loss_dh = DeepHitSingleLoss(alpha=self.config.deephit_loss_alpha, sigma=self.config.deephit_loss_sigma)
                    loss = tensor(0.0, device=self.config.device)
                    use_xcal = getattr(self.config, "use_xcal", False)
                    xcal_loss = tensor(0.0, device=self.config.device)
                    for b in range(B):
                        loss_b = loss_dh(log_hazard[b].squeeze(1), idx_duration_list[b], event_list[b], rank_mat_list[b])
                        loss += loss_b

                        # ----------------- X-CAL -----------------
                        # Compute the model CDF evaluated at each query's observed time T
                        # using linear interpolation inside the observed bin, matching the
                        # categorical-model recipe from the reference X-CAL repo.
                        if use_xcal:
                            logits_b = log_hazard[b].squeeze(1).float()                  # (N, K_bins)
                            pmf_b = predict_pmf_deephit(logits_b).squeeze(0)              # (K_bins, N)
                            cif_b = pmf_b.cumsum(0)                                       # (K_bins, N)

                            observed_bins = idx_duration_list[b].long()                   # (N,)
                            Nq = observed_bins.numel()
                            patient_idx = torch.arange(Nq, device=self.config.device)

                            pmf_at_bin = pmf_b[observed_bins, patient_idx]                # (N,)
                            cif_at_bin = cif_b[observed_bins, patient_idx]                # (N,) = CIF(k_n)
                            cif_prev   = cif_at_bin - pmf_at_bin                          # (N,) = CIF(k_n - 1)

                            cuts_b = torch.as_tensor(
                                bins_list[b].cuts,
                                device=self.config.device,
                                dtype=T_test.dtype,
                            )                                                             # (K_bins,)
                            # right edge of last (unbounded) bin -> fall back to max observed time
                            max_t = T_test[b].max().detach()
                            right_last = torch.maximum(cuts_b[-1] + 1e-6, max_t + 1e-6)
                            left_edges = cuts_b                                           # (K_bins,)
                            right_edges = torch.cat([cuts_b[1:], right_last.view(1)], dim=0)  # (K_bins,)

                            a_k = left_edges[observed_bins]                               # (N,)
                            b_k = right_edges[observed_bins]                              # (N,)
                            ratio = ((T_test[b] - a_k) / (b_k - a_k).clamp(min=1e-8)).clamp(0.0, 1.0)

                            cdf_at_T = (cif_prev + pmf_at_bin * ratio).clamp(0.0, 1.0)    # (N,)
                            is_alive_xcal = 1.0 - E_test[b].float()                       # (N,) 1=censored

                            xcal_b = d_calibration(
                                points=cdf_at_T,
                                is_alive=is_alive_xcal,
                                nbins=getattr(self.config, "xcal_bins", 20),
                                differentiable=True,
                                gamma=getattr(self.config, "xcal_gamma", 1.0),
                                device=self.config.device,
                            )
                            xcal_loss = xcal_loss + xcal_b
                    loss = loss / B
                    if use_xcal:
                        xcal_loss = xcal_loss / B
                        loss = loss + getattr(self.config, "xcal_lam", 1.0) * xcal_loss
                elif self.config.surv_loss == "deepeh":
                    loss = 0.0
                    B_n = 0
                    for b in range(B):
                        try:
                            deep_eh_loss = deepeh_loss(log_hazard[b].squeeze(1), T_test[b], E_test[b])
                        except Exception as ex:
                            print("DeepEH loss computation error:")
                            continue
                        if torch.isnan(deep_eh_loss):
                            continue
                        loss += deep_eh_loss
                        B_n += 1
                    if B_n > 0:
                        loss = loss / B_n
                    else:
                        loss = log_hazard.sum() * 0.0
                elif self.config.surv_loss in ("nll", "bce", "combo_bce"):
                    # Continuous SuMo / SuMo++ head. Per task: embeddings + head + time norm.
                    head = out["sumo_head"]
                    if self.config.surv_loss == "nll":
                        _loss_fn = nll_loss
                    elif self.config.surv_loss == "bce":
                        _loss_fn = functools.partial(bce_loss, sigma_frac=self.config.bce_sigma)
                    else:  # combo_bce: NLL + bce_weight * BCE
                        _loss_fn = functools.partial(
                            combo_bce_loss,
                            bce_weight=self.config.bce_weight,
                            sigma_frac=self.config.bce_sigma,
                        )
                    cens_head = out.get("cens_head")
                    use_true = self.config.surv_target == "true"
                    loss = tensor(0.0, device=self.config.device)
                    B_eff = 0
                    for b in range(B):
                        Xb = out["emb_test"][b]          # (Tq, d_model)
                        Tb = T_test[b]                   # (Tq,)
                        Eb = E_test[b]                   # (Tq,)
                        if Tb.numel() == 0:
                            continue
                        try:
                            t_scale = compute_time_norm(T_train[b], self.config.sumo_tscale_q)
                            if use_true:
                                mult = self.config.true_tau_mult
                                tau = mult * T_train[b].max() if 0 < mult < float("inf") else float("inf")
                                Tb, Eb = true_targets(T_event_test[b], tau)
                                if cens_head is not None:
                                    Tc, Ec = true_targets(T_cens_test[b], tau)
                            elif cens_head is not None:
                                Tc, Ec = Tb, 1.0 - Eb
                            if self.config.surv_target == "curve":
                                ll = curve_ce_loss(head, Xb, t_grid[b], S_test[b], t_scale=t_scale)
                            else:
                                ll = _loss_fn(head, Xb, Tb, Eb, t_scale=t_scale)
                            if cens_head is not None:
                                ll = ll + self.config.cens_loss_weight * _loss_fn(cens_head, Xb, Tc, Ec, t_scale=t_scale)
                        except Exception as ex:
                            print("SuMo loss computation error:", ex)
                            continue
                        loss += ll
                        B_eff += 1
                    if B_eff > 0:
                        loss = loss / B_eff
                    else:
                        loss = loss + micro_X.sum() * 0.0

            # Scale for grad accumulation
            scaled_loss = loss / num_micro_batches
            if scaled_loss.isnan() or scaled_loss.isinf() or abs(scaled_loss.item()) < 1e-8:
                print("NaN/Inf/zero loss encountered, skipping micro batch")
                dummy_loss = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0.0
                scaled_loss = dummy_loss / num_micro_batches
            
            self.scaler.scale(scaled_loss).backward()


            with torch.no_grad():
                micro_results["nll"] = scaled_loss.item()
                if micro_batch_idx == num_micro_batches - 1:
                    cis_sum = 0.0; cis_cnt = 0
                    ibs_sum = 0.0; ibs_cnt = 0
                    concordance_index_alternative_sum = 0.0
                    concordance_index_alternative_cnt = 0
                    for b in range(B):
                        e_b = E_test[b].detach().cpu().numpy()
                        t_b = T_test[b].detach().cpu().numpy()
                        time_grid = np.unique(np.sort(t_b))

                        if self.config.surv_loss == "coxph":
                            if not E_train[b].bool().any():
                                continue
                            
                            try:
                                S_b, time_grid_coxph = predict_surv_coxph_2(log_hazard_train[b].squeeze(1), E_train[b], T_train[b],
                                                    log_hazard[b].squeeze(1), torch.tensor(time_grid, device=log_hazard.device))
                            except Exception as e:
                                print(f"Error in predict_surv_coxph_2 for batch {b}: {e}")
                                S_b = torch.zeros((len(time_grid),), device=self.config.device)
                                time_grid_coxph = torch.zeros((len(time_grid),), device=self.config.device)
                                continue

                        elif self.config.surv_loss == "deephit":
                            cuts = bins_list[b].cuts
                            S_b = predict_surv_deephit(log_hazard[b].squeeze(1))
                            #S_b = S_b.T  # (K, N)
                            #time_grid_new = torch.tensor(time_grid, device=log_hazard.device)
                            S_b = predict_surv_new_times_deephit(
                                torch.tensor(S_b.T, dtype=torch.float32, device=log_hazard.device), 
                                torch.tensor(cuts, dtype=torch.float32, device=log_hazard.device),                      
                                torch.tensor(time_grid, device=log_hazard.device)
                            )
                            # for interpolation evaluation
                            # after the discussion came to conclusion that our evaluation on new time grid is sufficient
                            #class MockDeepHit:
                            #    def predict_surv(self, input, *args, **kwargs): return S_b
                        
                            # Interpolate (stays on device)
                            #sub = 10
                            #interpolator = InterpolateDiscrete(MockDeepHit(), scheme='const_pdf', duration_index=labtrans.cuts, sub=sub)
                            #S_b = interpolator.predict_surv(None, numpy=False)
                            #time_grid_new = pycox_utils.make_subgrid(labtrans.cuts, sub)

                        elif self.config.surv_loss == "deepeh":
                            try:
                                S_b, tt, dlt, ind_sort = predict_surv_deepeh(log_hazard_train[b].squeeze(1), T_train[b], E_train[b],
                                                                        log_hazard[b].squeeze(1), T_test[b], E_test[b])
                            except Exception as e:
                                print(f"Error in predict_surv_deepeh for batch:")
                                S_b = torch.zeros((len(time_grid),), device=self.config.device)
                                tt = torch.zeros((len(time_grid),), device=self.config.device)
                                dlt = torch.zeros((len(time_grid),), device=self.config.device)
                                ind_sort = torch.zeros((len(time_grid),), device=self.config.device)
                                continue
                        elif self.config.surv_loss in ("nll", "bce", "combo_bce"):
                            head = out["sumo_head"]
                            try:
                                t_scale = compute_time_norm(T_train[b], self.config.sumo_tscale_q)
                                S_b = predict_surv_sumo(
                                    out["emb_test"][b],
                                    torch.tensor(time_grid, device=self.config.device, dtype=out["emb_test"][b].dtype),
                                    head,
                                    t_scale=t_scale,
                                )
                            except Exception as e:
                                print(f"Error in predict_surv_sumo for batch {b}: {e}")
                                continue

                        S_b = S_b.detach().cpu().numpy()
                        if self.config.surv_loss == "deepeh":
                            S_df = pd.DataFrame(S_b.T, index=tt)  # (K, Tq)
                            ev = EvalSurv(S_df, tt, dlt, censor_surv='km')
                        elif self.config.surv_loss == "deephit":
                            S_df = pd.DataFrame(S_b.T, index=time_grid)  # (K, Tq)
                            ev = EvalSurv(S_df, t_b, (e_b!=0), censor_surv='km')
                        else:
                            S_df = pd.DataFrame(S_b.T, index=time_grid)  # (K, Tq)
                            ev = EvalSurv(S_df, t_b, (e_b!=0), censor_surv='km')

                        if self.config.surv_loss == "coxph":
                            # harrels C concordance index from the torchsurv
                            risk_scores = log_hazard[b]
                            cindex = ConcordanceIndex()
                            try:
                                concordance_index_alternative_ = cindex(risk_scores, E_test[b].bool(), T_test[b]).item()
                            except Exception as e:
                                print(f"Error in ConcordanceIndex for batch {b}: {e}")
                                concordance_index_alternative_ = float('nan')
                            
                            if np.isfinite(concordance_index_alternative_):
                                concordance_index_alternative_sum += concordance_index_alternative_
                                concordance_index_alternative_cnt += 1
                        elif self.config.surv_loss == "deepeh":
                            # antolini's C concordance index
                            try:
                                concordance_index_alternative_ = sum_concordant_antolini(S_b, tt, dlt)
                            except Exception as e:
                                print(f"Error in sum_concordant_antolini for batch {b}: {e}")
                                concordance_index_alternative_ = float('nan')
                            if np.isfinite(concordance_index_alternative_):
                                concordance_index_alternative_sum += concordance_index_alternative_
                                concordance_index_alternative_cnt += 1

                        try:
                            ci = ev.concordance_td()
                            if np.isfinite(ci):
                                cis_sum += ci
                                cis_cnt += 1
                        except Exception as ex:
                            print("Concordance calculation error:", ex)
                            ci = float('nan')
                        try:
                            ibs_time_grid = np.linspace(t_b.min(), t_b.max(), 100)
                            ibs = ev.integrated_brier_score(ibs_time_grid) #change simps to simpson
                            if np.isfinite(ibs):
                                ibs_sum += ibs
                                ibs_cnt += 1
                        except Exception as ex:
                            print("IBS calculation error:", ex)
                            ibs = float('nan')
                    
                    if cis_cnt > 0:
                        micro_results["C-index (td)"] = (cis_sum / cis_cnt) #/ num_micro_batches
                    if ibs_cnt > 0:
                        micro_results["ibs"] = (ibs_sum / ibs_cnt) #/ num_micro_batches            
                    if concordance_index_alternative_cnt > 0:
                        micro_results["C-index (Alternative)"] = (concordance_index_alternative_sum / concordance_index_alternative_cnt) #/ num_micro_batches       
            
            return micro_results

        # ---------------- classification branch unchanged ----------------

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]


        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)  # (B, test_size, max_classes)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)

        # Scale loss for gradient accumulation and backpropagate
        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {}
            micro_results["ce"] = scaled_loss.item()
            accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
            micro_results["accuracy"] = accuracy.item() / num_micro_batches

        return micro_results

    def _is_main_process(self) -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
    
    def run_batch(self, batch):
        """
        Trains the model on a batch of datasets. Handles gradient accumulation by
        splitting the batch into micro-batches. Supports variable-sized datasets
        by padding. Skips micro-batches on CUDA OOM errors. Updates model
        parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch: tuple
            Contains tensors (X, y, d, seq_len, train_size) for the batch.
            X and y can be Tensors or NestedTensors (for variable sequence lengths).

        Returns
        ------
        dict
            Dictionary containing 'ce' (cross-entropy loss) and 'accuracy'.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Pad nested tensors to the same size
        batch = [t.to_padded_tensor(padding=0.0) if torch.is_tensor(t) and t.is_nested else t for t in batch]

        # Split the batch into micro-batches along the first dimension
        num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
        micro_batches = [_split_batch_item(t, self.config.micro_batch_size) for t in batch]
        micro_batches = list(zip(*micro_batches))

        if self.config.mode == "survival":
            results = {"nll": 0.0, 
                       "C-index (td)": 0.0, 
                       "ibs": 0.0, 
                       "C-index (Alternative)": 0.0,
                       "test_on_gbsg2_cindex_td": 0.0,
                       "test_on_gbsg2_ibs": 0.0,
                       "test_on_gbsg2_cindex_alternative": 0.0}
        else:
            results = {"ce": 0.0, "accuracy": 0.0}
        
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            skip_computation = False

            micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
            if torch.isnan(micro_X).any() or _has_nan(micro_y):
                skip_computation = True
                if self.master_process:
                    print(f"Warning: NaNs detected in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
            if self.ddp:
                skip_tensor = torch.tensor([1 if skip_computation else 0], device=self.config.device)
                torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)
                skip_computation = bool(skip_tensor.item())

            if skip_computation:
                failed_batches += 1
            try:    
                # Process micro-batch (OOM can occur here)
                micro_results = self.run_micro_batch(
                    micro_batch if not skip_computation else None,
                    idx,
                    num_micro_batches,
                    skip_computation=skip_computation
                )
                
                if micro_results is not None:
                    for k, v in micro_results.items():
                        results[k] += v
                elif not skip_computation:
                    failed_batches += 1
            except torch.cuda.OutOfMemoryError:
                skip_computation = True
                if self.master_process:
                    print(
                        f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                    )
                torch.cuda.empty_cache()
                failed_batches += 1
                 
                if self.ddp:
                    skip_tensor = torch.tensor([1], device=self.config.device)
                    torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)
                
                # All ranks must do a dummy backward pass to stay synchronized
                self.run_micro_batch(None, idx, num_micro_batches, skip_computation=True)

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.5:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        # Clip the gradient
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update the learning rate
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results


def main(argv: list[str] | None = None):
    """Train SIC on synthetic priors. Run via ``python -m sic.train.run``."""
    parser = build_parser()
    config = parser.parse_args(argv)

    # Continuous SuMo / SuMo++ heads and their losses only work as a pair.
    if config.surv_model_type == "sumo":
        if config.surv_loss != "nll":
            parser.error("--surv_model_type sumo requires --surv_loss nll")
    elif config.surv_model_type == "sumo_pp":
        if config.surv_loss not in ("nll", "bce", "combo_bce"):
            parser.error(
                "--surv_model_type sumo_pp requires --surv_loss in nll/bce/combo_bce"
            )
    elif config.surv_loss in ("nll", "bce", "combo_bce"):
        parser.error(
            f"--surv_loss {config.surv_loss} requires --surv_model_type sumo or sumo_pp"
        )

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
