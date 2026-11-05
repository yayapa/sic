"""
Utilities for generating and loading prior datasets from disk for training.

This module implements a workflow for generating synthetic tabular data from prior
distributions, saving it efficiently to disk in a sparse format, and loading it during
distributed training.

This module provides two main classes:
- SavePriorDataset: Generates and saves batches of prior data to disk
- LoadPriorDataset: Loads pre-generated prior data from disk for distributed training

The data is saved in a sparse format to reduce storage requirements and loaded
on demand during training. The module supports distributed training by allowing
different processes to load different batches in a coordinated way.

The saved data includes:
- X: Input features in sparse format or nested tensor (for variable-length sequences)
- y: Target labels as regular tensor or nested tensor (for variable-length sequences)
- d: Number of features per dataset
- seq_lens: Sequence length for each dataset
- train_sizes: Position at which to split training and evaluation data
- batch_size: Number of datasets in the batch
"""

from __future__ import annotations

import time
import json
import warnings
import argparse
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from torch.utils.data import IterableDataset

from sic.prior.dataset import PriorDataset
from sic.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
import yaml

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def dense2sparse(
    dense_tensor: torch.Tensor, row_lengths: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert a dense tensor with trailing zeros into a compact 1D representation.

    Parameters
    ----------
    dense_tensor : torch.Tensor
        Input tensor of shape (num_rows, num_cols) where each row may contain
        trailing zeros beyond the valid entries

    row_lengths : torch.Tensor
        Tensor of shape (num_rows,) specifying the number of valid entries
        in each row of the dense tensor

    dtype : torch.dtype, default=torch.float32
        Output data type for the sparse representation

    Returns
    -------
    torch.Tensor
        1D tensor of shape (sum(row_lengths),) containing only the valid entries
    """

    assert dense_tensor.dim() == 2, "dense_tensor must be 2D"
    num_rows, num_cols = dense_tensor.shape
    assert row_lengths.shape[0] == num_rows, "row_lengths must match number of rows"
    assert (row_lengths <= num_cols).all(), "row_lengths cannot exceed number of columns"

    indices = torch.arange(num_cols, device=dense_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    sparse = dense_tensor[mask].to(dtype)

    return sparse


def sparse2dense(
    sparse_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense tensor from its sparse representation.

    This function is the inverse of dense2sparse, reconstructing a padded dense
    tensor from a compact 1D representation and the corresponding row lengths.
    Unused entries in the output are filled with zeros.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        1D tensor containing the valid entries from the original dense tensor

    row_lengths : torch.Tensor
        Number of valid entries for each row in the output tensor

    max_len : Optional[int], default=None
        Maximum length for each row in the output. If None, uses max(row_lengths)

    dtype : torch.dtype, default=torch.float32
        Output data type for the dense representation

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (num_rows, max_len) with zeros padding
    """

    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(sparse_tensor), "data length must match sum of row_lengths"

    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)

    return dense


class SliceNestedTensor:
    """A wrapper for nested tensors that supports slicing along the first dimension.

    This class wraps PyTorch's nested tensor and provides slicing operations
    along the first dimension, which are not natively supported by nested tensors.
    It maintains compatibility with other nested tensor operations by forwarding
    attribute access to the wrapped tensor.

    Parameters
    ----------
    nested_tensor : torch.Tensor
        A nested tensor to wrap
    """

    def __init__(self, nested_tensor):
        self.nested_tensor = nested_tensor
        self.is_nested = nested_tensor.is_nested

    def __getitem__(self, idx):
        """Support slicing operations along the first dimension."""
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = self.nested_tensor.size(0) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step

            indices = list(range(start, stop, step))
            return SliceNestedTensor(torch.nested.nested_tensor([self.nested_tensor[i] for i in indices]))
        elif isinstance(idx, int):
            return self.nested_tensor[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped nested tensor."""
        return getattr(self.nested_tensor, name)

    def __len__(self):
        """Return the length of the first dimension."""
        return self.nested_tensor.size(0)

    def to(self, *args, **kwargs):
        """Support the to() method for device/dtype conversion."""
        return SliceNestedTensor(self.nested_tensor.to(*args, **kwargs))


def cat_slice_nested_tensors(tensors: List, dim=0) -> SliceNestedTensor:
    """Concatenate a list of SliceNestedTensor objects along dimension dim.

    Parameters
    ----------
    tensors : List
        List of tensors to concatenate

    dim : int, default=0
        Dimension along which to concatenate

    Returns
    -------
    SliceNestedTensor
        Concatenated tensor wrapped in SliceNestedTensor
    """
    # Extract the wrapped nested tensors
    nested_tensors = [t.nested_tensor if isinstance(t, SliceNestedTensor) else t for t in tensors]
    return SliceNestedTensor(torch.cat(nested_tensors, dim=dim))


def _cat_y(a, b):
    """Concatenate targets along the dataset dim; survival targets are dicts of tensors."""
    if isinstance(a, dict):
        return {k: torch.cat([a[k], b[k]], dim=0) for k in a}
    return torch.cat([a, b], dim=0)


def _take_y(y, sl: slice):
    return {k: v[sl] for k, v in y.items()} if isinstance(y, dict) else y[sl]


class LoadPriorDataset(IterableDataset):
    """Loads pre-generated prior data sequentially for distributed training.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the batch files

    batch_size : int, default=512
        Number of datasets to return in each iteration

    ddp_world_size : int, default=1
        Total number of distributed processes

    ddp_rank : int, default=0
        Rank of current process

    start_from : int, default=0
        Batch index to start loading from

    max_batches : int, optional
        Maximum number of batches to load. If None, load indefinitely.

    timeout : int, default=60
        Maximum time in seconds to wait for a batch file

    delete_after_load : bool, default=False
        Whether to delete batch files after loading them

    device : str, default='cpu'
        Device to load tensors to
    """

    def __init__(
        self,
        data_dir,
        batch_size=512,
        ddp_world_size=1,
        ddp_rank=0,
        start_from=0,
        max_batches=None,
        timeout=60,
        delete_after_load=False,
        device="cpu",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.current_idx = ddp_rank + start_from
        self.max_batches = max_batches
        self.timeout = timeout
        self.delete_after_load = delete_after_load
        self.device = device

        # Load metadata if available
        self.metadata = None
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load or parse metadata.json: {e}")

        # Buffer for storing datasets that haven't been returned yet
        self.buffer_X = None
        self.buffer_y = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_train_sizes = None
        self.buffer_size = 0

    def __iter__(self):
        return self

    def _load_batch_file(self):
        """Load a single batch file from disk.

        Returns
        -------
        tuple
            A tuple containing X, y, d, seq_lens, train_sizes and the size of the batch
        """
        batch_file = self.data_dir / "batch_data" / f"batch_{self.current_idx:06d}.pt"

        # Try loading the file for up to timeout seconds
        wait_time = 0
        while not batch_file.exists():
            if wait_time >= self.timeout:
                raise RuntimeError(f"Timeout waiting for batch file {batch_file}")
            time.sleep(5)
            wait_time += 5

        batch = torch.load(batch_file, map_location=self.device, weights_only=True)
        X = batch["X"]
        y = batch["y"]
        if not isinstance(y, dict) and not y.is_nested and y.ndim == 3:
            # Survival data saved before targets became a dict: channels (T_obs, E[, T_event, T_cens]).
            y = dict(zip(("T_obs", "E", "T_event", "T_cens"), y.unbind(1)))
        d = batch["d"]
        seq_lens = batch["seq_lens"]
        train_sizes = batch["train_sizes"]
        batch_size = batch["batch_size"]

        if X.is_nested:
            # Wrap nested tensors with SliceNestedTensor
            X = SliceNestedTensor(X)
            y = SliceNestedTensor(y)
        else:
            # Convert sparse tensor to dense
            X = sparse2dense(X, d.repeat_interleave(seq_lens[0]), dtype=torch.float32).view(batch_size, seq_lens[0], -1)

        # Delete file if requested
        if self.delete_after_load and batch_file.exists():
            batch_file.unlink()

        # Prepare next index for this process
        self.current_idx += self.ddp_world_size

        return X, y, d, seq_lens, train_sizes, batch_size

    def __next__(self):
        """Load datasets until we have at least batch_size, then return exactly batch_size.

        This method accumulates datasets from multiple files if necessary to return
        the exact number of datasets specified in batch_size. Any extra datasets are
        kept in a buffer for the next iteration.

        Returns
        -------
        tuple
            A tuple containing:
            - X: Input features [batch_size, seq_len, features] or nested tensor
            - y: Target labels [batch_size, seq_len] or nested tensor
            - d: Number of features per dataset
            - seq_lens: Sequence length for each dataset
            - train_sizes: Position at which to split training and evaluation data
        """
        # Check if we've reached the maximum number of batches and have no buffered data
        if self.max_batches is not None and self.current_idx >= self.max_batches and (self.buffer_size == 0):
            raise StopIteration

        # Initialize or use existing buffer
        if self.buffer_size == 0:
            # Load the first batch
            X, y, d, seq_lens, train_sizes, file_batch_size = self._load_batch_file()
            if not isinstance(X, SliceNestedTensor) and file_batch_size % self.batch_size != 0:
                # Dense files have their own seq_len, train_size and feature width, so they cannot be merged.
                raise ValueError(
                    f"batch_size={self.batch_size} must divide the saved file batch size ({file_batch_size}) "
                    f"in {self.data_dir}; datasets from different files cannot share one batch."
                )
            self.buffer_X = X
            self.buffer_y = y
            self.buffer_d = d
            self.buffer_seq_lens = seq_lens
            self.buffer_train_sizes = train_sizes
            self.buffer_size = file_batch_size

        # Keep loading files until we have enough data or no more files
        while self.buffer_size < self.batch_size:
            # Check if we've reached max_batches
            if self.max_batches is not None and self.current_idx >= self.max_batches:
                # If we can't get a full batch, return what we have
                break

            try:
                # Load another batch and append to buffer
                X, y, d, seq_lens, train_sizes, file_batch_size = self._load_batch_file()

                # Concatenate with existing buffer
                if self.buffer_X is None:
                    # If buffer is empty, directly assign
                    self.buffer_X = X
                    self.buffer_y = y
                    self.buffer_d = d
                    self.buffer_seq_lens = seq_lens
                    self.buffer_train_sizes = train_sizes
                    self.buffer_size = file_batch_size
                else:
                    # Otherwise concatenate, handling SliceNestedTensor if needed
                    if isinstance(X, SliceNestedTensor):
                        self.buffer_X = cat_slice_nested_tensors([self.buffer_X, X], dim=0)
                        self.buffer_y = cat_slice_nested_tensors([self.buffer_y, y], dim=0)
                    else:
                        self.buffer_X = torch.cat([self.buffer_X, X], dim=0)
                        self.buffer_y = _cat_y(self.buffer_y, y)

                    self.buffer_d = torch.cat([self.buffer_d, d], dim=0)
                    self.buffer_seq_lens = torch.cat([self.buffer_seq_lens, seq_lens], dim=0)
                    self.buffer_train_sizes = torch.cat([self.buffer_train_sizes, train_sizes], dim=0)
                    self.buffer_size += file_batch_size
            except Exception as e:
                # If we can't load more files, use what we have
                print(f"Warning: Could not load more files: {str(e)}")
                break

        # Extract batch_size datasets (or all if we have fewer)
        output_size = min(self.batch_size, self.buffer_size)

        # Prepare output
        X_out = self.buffer_X[:output_size]
        y_out = _take_y(self.buffer_y, slice(None, output_size))
        d_out = self.buffer_d[:output_size]
        seq_lens_out = self.buffer_seq_lens[:output_size]
        train_sizes_out = self.buffer_train_sizes[:output_size]

        # Update buffer with remaining data
        if output_size < self.buffer_size:
            self.buffer_X = self.buffer_X[output_size:]
            self.buffer_y = _take_y(self.buffer_y, slice(output_size, None))
            self.buffer_d = self.buffer_d[output_size:]
            self.buffer_seq_lens = self.buffer_seq_lens[output_size:]
            self.buffer_train_sizes = self.buffer_train_sizes[output_size:]
            self.buffer_size -= output_size
        else:
            # Buffer is now empty
            self.buffer_X = None
            self.buffer_y = None
            self.buffer_d = None
            self.buffer_seq_lens = None
            self.buffer_train_sizes = None
            self.buffer_size = 0

        if isinstance(X_out, SliceNestedTensor):
            X_out = X_out.nested_tensor
            y_out = y_out.nested_tensor

        return X_out, y_out, d_out, seq_lens_out, train_sizes_out

    def __repr__(self) -> str:
        """
        Returns a string representation of the LoadPriorDataset.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        repr_str = (
            f"LoadPriorDataset(\n"
            f"  data_dir: {self.data_dir}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  ddp_world_size: {self.ddp_world_size}\n"
            f"  ddp_rank: {self.ddp_rank}\n"
            f"  start_from: {self.current_idx - self.ddp_rank}\n"
            f"  max_batches: {self.max_batches or 'Infinite'}\n"
            f"  timeout: {self.timeout}\n"
            f"  delete_after_load: {self.delete_after_load}\n"
            f"  device: {self.device}\n"
        )
        if self.metadata:
            repr_str += "  Loaded Metadata:\n"
            repr_str += f"    prior_type: {self.metadata.get('prior_type', 'N/A')}\n"
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += f"    max classes: {self.metadata.get('max_classes', 'N/A')}\n"
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
            repr_str += f"    train_size: {self.metadata.get('min_train_size', 'N/A')} - {self.metadata.get('max_train_size', 'N/A')}\n"
            repr_str += f"    replay_small: {self.metadata.get('replay_small', 'N/A')}\n"
        repr_str += ")"

        return repr_str


class MixedLoadPriorDataset(IterableDataset):
    """
    Sample batches from multiple LoadPriorDataset instances.

    Each __next__ selects one of the underlying datasets according to mix_probs
    and returns its next batch. Intended for mixing multiple pre-generated prior
    directories during training.
    """

    def __init__(
        self,
        datasets: List[IterableDataset],
        mix_probs: Optional[List[float]] = None,
        seed: int = 42,
    ):
        super().__init__()
        if not datasets or len(datasets) < 2:
            raise ValueError("MixedLoadPriorDataset requires at least two datasets.")

        self.datasets = list(datasets)
        self.n = len(self.datasets)

        if mix_probs is None:
            probs = np.ones(self.n, dtype=np.float64) / self.n
        else:
            if len(mix_probs) != self.n:
                raise ValueError(
                    f"mix_probs must have same length as datasets ({self.n}), got {len(mix_probs)}."
                )
            probs = np.asarray(mix_probs, dtype=np.float64)
            if np.any(probs < 0):
                raise ValueError("mix_probs must be non-negative.")
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                raise ValueError("mix_probs must sum to a positive finite value.")
            probs = probs / s

        self.mix_probs = probs
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._iters = None

    def __iter__(self):
        self._iters = [iter(ds) for ds in self.datasets]
        return self

    def __next__(self):
        if self._iters is None:
            self.__iter__()
        idx = int(self.rng.choice(self.n, p=self.mix_probs))
        return next(self._iters[idx])

    def __repr__(self) -> str:
        probs_str = ", ".join(f"{p:.4f}" for p in self.mix_probs.tolist())
        return (
            "MixedLoadPriorDataset(\n"
            f"  n_datasets: {self.n}\n"
            f"  mix_probs: [{probs_str}]\n"
            f"  seed: {self.seed}\n"
            ")"
        )


# Per-dataset censoring regime record saved next to cov_indep_cens; read by
# scripts/audit_prior_nan_events.py. Keys missing from a dataset's meta are NaN.
CENSORING_META_BOOL_KEYS = (
    "random_cens", "admin_type1", "admin_type2",
    "type1_bound", "type2_bound", "type2_reached", "stagger_entry",
)
CENSORING_META_FLOAT_KEYS = (
    "C", "q_type1", "C_percent", "type2_d",
    "accrual_frac", "accrual_shape", "r_mean",
    "random_cens_frac", "admin_cens_frac", "event_frac",
)


def censoring_meta_tensors(sampled_params_list) -> dict:
    """Stack the scalar censoring fields of each dataset's meta into (B,) tensors."""
    metas = [p or {} for p in sampled_params_list]
    out = {k: torch.tensor([bool(m.get(k, False)) for m in metas]) for k in CENSORING_META_BOOL_KEYS}
    for k in CENSORING_META_FLOAT_KEYS:
        out[k] = torch.tensor([float(m.get(k, float("nan"))) for m in metas], dtype=torch.float32)
    return out


class SavePriorDataset:
    """Generates and saves batches of prior datasets to disk.

    The datasets are saved as individual batch files in the specified directory
    using an atomic file writing pattern to ensure data integrity.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration for dataset generation
    """

    def __init__(self, args):
        self.args = args
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_metadata()

        self.prior = PriorDataset(
            batch_size=self.args.batch_size,
            batch_size_per_gp=self.args.batch_size_per_gp,
            min_features=self.args.min_features,
            max_features=self.args.max_features,
            max_classes=self.args.max_classes,
            min_seq_len=self.args.min_seq_len,
            max_seq_len=self.args.max_seq_len,
            log_seq_len=self.args.log_seq_len,
            seq_len_per_gp=self.args.seq_len_per_gp,
            min_train_size=self.args.min_train_size,
            max_train_size=self.args.max_train_size,
            replay_small=self.args.replay_small,
            prior_type=self.args.prior_type,
            scm_fixed_hp=DEFAULT_FIXED_HP,
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=self.args.n_jobs,
            num_threads_per_generate=self.args.num_threads_per_generate,
            device=self.args.device,
            reg2surv=self.args.reg2surv,
            reg2surv_discrete=getattr(self.args, "reg2surv_discrete", False),
        )
        print(self.prior)

    def save_metadata(self):
        """Save metadata about the dataset generation configuration to a JSON file."""
        metadata = {
            "prior_type": self.args.prior_type,
            "batch_size": self.args.batch_size,
            "batch_size_per_gp": self.args.batch_size_per_gp,
            "min_seq_len": self.args.min_seq_len,
            "max_seq_len": self.args.max_seq_len,
            "log_seq_len": self.args.log_seq_len,
            "seq_len_per_gp": self.args.seq_len_per_gp,
            "min_features": self.args.min_features,
            "max_features": self.args.max_features,
            "max_classes": self.args.max_classes,
            "min_train_size": self.args.min_train_size,
            "max_train_size": self.args.max_train_size,
            "replay_small": self.args.replay_small,
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        # Keep the exact prior configs next to the data they generated.
        for cfg in (getattr(self.args, "reg2surv_config", None), getattr(self.args, "reg2surv_discrete_config", None)):
            if cfg is not None:
                shutil.copy2(cfg, self.save_dir / Path(cfg).name)

    def save_batch_sparse(self, batch_idx, X, y, d, seq_lens, train_sizes, cov_indep_cens=None, censoring_meta=None):
        """Save batch data in sparse format for efficient storage.

        This method handles the conversion between dense and sparse tensor formats
        when appropriate and saves the batch data to a PyTorch file. It uses an atomic
        write pattern (writing to a temporary file and then renaming) to ensure data
        integrity even if the process is interrupted during saving.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming

        X : torch.Tensor
            Input features tensor, either in dense format [batch_size, seq_len, features]
            or in nested tensor format for variable sequence lengths

        y : torch.Tensor
            Target labels tensor

        d : torch.Tensor
            Number of features for each dataset

        seq_lens : torch.Tensor
            Sequence length for each dataset

        train_sizes : torch.Tensor
            Position at which to split training and evaluation data

        cov_indep_cens : torch.Tensor, optional
            Per-dataset bool flag, True if censoring is covariate-independent

        censoring_meta : dict[str, torch.Tensor], optional
            Per-dataset censoring regime record (see CENSORING_META_KEYS)
        """
        save_dir_batch_data = self.save_dir / "batch_data"
        save_dir_batch_data.mkdir(parents=True, exist_ok=True)
        if self.args.seq_len_per_gp:
            # X and y are nested tensors and they are already sparse
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32)

        # Create temporary file first
        batch_file = save_dir_batch_data / f"batch_{batch_idx:06d}.pt"
        temp_file = save_dir_batch_data / f"batch_{batch_idx:06d}.pt.tmp"
        payload = {"X": X, "y": y, "d": d, "seq_lens": seq_lens, "train_sizes": train_sizes, "batch_size": B}
        if cov_indep_cens is not None:
            payload["cov_indep_cens"] = cov_indep_cens
        if censoring_meta is not None:
            payload["censoring_meta"] = censoring_meta
        torch.save(payload, temp_file)
        # Atomic rename to ensure file integrity
        temp_file.replace(batch_file)
    def save_sampled_params(self, batch_idx, sampled_params_to_save_list):
        """Save sampled hyperparameters for each dataset in the batch.

        This method saves the sampled hyperparameters used to generate each dataset
        in the batch to a separate JSON file. It uses an atomic write pattern (writing
        to a temporary file and then renaming) to ensure data integrity even if the
        process is interrupted during saving.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming

        sampled_params_to_save_list : List[dict]
            List of dictionaries containing sampled hyperparameters for each dataset
        """
        # Create temporary file first
        # create sampled_params directory if not exists
        self.save_dir_sampled_params = self.save_dir / "sampled_params"
        self.save_dir_sampled_params.mkdir(parents=True, exist_ok=True)
        params_file = self.save_dir_sampled_params / f"batch_{batch_idx:06d}_params.json"
        temp_file = self.save_dir_sampled_params / f"batch_{batch_idx:06d}_params.json.tmp"
        with open(temp_file, "w") as f:
            json.dump(sampled_params_to_save_list, f, indent=2)
        # Atomic rename to ensure file integrity
        temp_file.replace(params_file)
        
    def save_batch_sparse_surv(self, batch_idx, X, y, d, seq_lens, train_sizes):
        """Save batch data in sparse format for efficient storage.

        This method handles the conversion between dense and sparse tensor formats
        when appropriate and saves the batch data to a PyTorch file. It uses an atomic
        write pattern (writing to a temporary file and then renaming) to ensure data
        integrity even if the process is interrupted during saving.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming

        X : torch.Tensor
            Input features tensor, either in dense format [batch_size, seq_len, features]
            or in nested tensor format for variable sequence lengths

        y : torch.Tensor
            Target labels tensor

        d : torch.Tensor
            Number of features for each dataset

        seq_lens : torch.Tensor
            Sequence length for each dataset

        train_sizes : torch.Tensor
            Position at which to split training and evaluation data
        """

        if self.args.seq_len_per_gp:
            # X and y are nested tensors and they are already sparse
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32)

        # Create temporary file first
        batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
        temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"
        torch.save(
            {"X": X, "t": y[0], "e": y[1], "d": d, "seq_lens": seq_lens, "train_sizes": train_sizes, "batch_size": B},
            temp_file,
        )
        # Atomic rename to ensure file integrity
        temp_file.replace(batch_file)

    def run(self):
        """Generate and save batches of prior datasets."""
        print(f"Save directory: {self.save_dir}")
        print(f"Generating {self.args.num_batches} batches starting from index {self.args.resume_from}")

        for batch_idx in tqdm(
            range(self.args.resume_from, self.args.resume_from + self.args.num_batches),
            desc="Generating batches",
        ):
            X, y, d, seq_lens, train_sizes, sampled_params_to_save_list = self.prior.get_batch()
            # Move tensors to CPU before saving
            X = X.cpu()
            y = {k: v.cpu() for k, v in y.items()} if isinstance(y, dict) else y.cpu()
            #y = (y[0].cpu(), y[1].cpu()) if self.args.reg2surv else y.cpu()
            
            d = d.cpu()
            seq_lens = seq_lens.cpu()
            train_sizes = train_sizes.cpu()
            # main() sets reg2surv=False for the discrete prior, so check both flags.
            if self.args.reg2surv or getattr(self.args, "reg2surv_discrete", False):
                #self.save_batch_sparse_surv(batch_idx, X, y, d, seq_lens, train_sizes)
                cov_indep_cens = torch.tensor(
                    [bool(p.get("cov_indep_cens", False)) if p else False for p in sampled_params_to_save_list]
                )
                self.save_batch_sparse(
                    batch_idx, X, y, d, seq_lens, train_sizes,
                    cov_indep_cens=cov_indep_cens,
                    censoring_meta=censoring_meta_tensors(sampled_params_to_save_list),
                )
                #self.save_sampled_params(batch_idx, sampled_params_to_save_list)
            else:
                self.save_batch_sparse(batch_idx, X, y, d, seq_lens, train_sizes)


def main(argv: list[str] | None = None):
    """Generate synthetic prior batches. Run via ``python -m sic.prior.genload``."""

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

    parser = argparse.ArgumentParser(description="Generate training prior datasets")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the generated data")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--batch_size", type=int, default=10, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=1, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")

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
        "--min_train_size", type=train_size_type, default=0.1, help="Minimum training size position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=train_size_type, default=0.9, help="Maximum training size position/ratio"
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mix_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm"],
        help="Type of prior to use",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for generation"
    )
    parser.add_argument("--reg2surv", type=str2bool, default=True, help="Whether to generate survival analysis data using Reg2Surv")
    parser.add_argument(
        "--reg2surv_config",
        type=str,
        default=None,
        help="Path to reg2surv YAML config file (e.g., reg2surv_params.yaml)",
    )
    parser.add_argument(
        "--reg2surv_discrete",
        type=str2bool,
        default=False,
        help="Whether to generate survival analysis data using Reg2SurvDiscrete (SUMO-style positive monotone net)",
    )
    parser.add_argument(
        "--reg2surv_discrete_config",
        type=str,
        default=None,
        help="Path to reg2surv_discrete YAML config file (e.g., reg2surv_discrete_params.yaml)",
    )
    args = parser.parse_args(argv)
    if args.reg2surv_config is not None:
        with open(args.reg2surv_config, "r") as f:
            reg2surv_param = yaml.safe_load(f)
        if reg2surv_param.get("dtype") == "float32":
            import torch
            reg2surv_param["dtype"] = torch.float32
        from sic.prior import prior_config
        prior_config.DEFAULT_FIXED_HP["reg2surv_param"] = reg2surv_param
        prior_config.DEFAULT_FIXED_HP["reg2surv"] = True

    if args.reg2surv_discrete_config is not None:
        with open(args.reg2surv_discrete_config, "r") as f:
            reg2surv_discrete_param = yaml.safe_load(f)
        # dtype normalisation (yaml stores strings)
        if reg2surv_discrete_param.get("dtype") == "float32":
            import torch
            reg2surv_discrete_param["dtype"] = torch.float32
        elif reg2surv_discrete_param.get("dtype") == "float64":
            import torch
            reg2surv_discrete_param["dtype"] = torch.float64
        # YAML parses [a, b] as lists; the sampler expects tuples/pairs of floats.
        for key in (
            "alpha_bounds", "alpha_cens_bounds",
            "gamma_k_bounds", "gamma_k_cens_bounds",
            "mu_t_bounds", "mu_t_cens_bounds",
            "lambda_eta_bounds", "lambda_eta_cens_bounds",
            "n_event_layers_bounds", "event_layer_bounds",
            "n_cens_layers_bounds", "cens_layer_bounds",
            "t_max_bounds", "censoring_rate_bounds",
            "type1_quantile_bounds", "type2_quantile_bounds",
            "accrual_frac_bounds", "accrual_shape_bounds",
        ):
            if key in reg2surv_discrete_param and isinstance(reg2surv_discrete_param[key], list):
                reg2surv_discrete_param[key] = tuple(reg2surv_discrete_param[key])
        from sic.prior import prior_config
        prior_config.DEFAULT_FIXED_HP["reg2surv_discrete_param"] = reg2surv_discrete_param
        prior_config.DEFAULT_FIXED_HP["reg2surv_discrete"] = True
        args.reg2surv_discrete = True
        # When using the discrete sampler, disable the continuous one to avoid conflicts.
        args.reg2surv = False
        prior_config.DEFAULT_FIXED_HP["reg2surv"] = False
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    saver = SavePriorDataset(args)
    saver.run()


if __name__ == "__main__":
    main()
