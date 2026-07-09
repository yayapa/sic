"""Checkpoint resolution and download for SIC.

Released checkpoints are hosted on the Hugging Face Hub
(https://huggingface.co/yayapa/sic). The repo can be overridden per-call or via
the ``SIC_HF_REPO`` environment variable.

Release checkpoint schema (a plain ``torch.save`` dict)::

    {
        "format_version": 1,
        "survival_head": "DeepHit10",
        "config": {... model hyperparameters ...},   # full TabICL(**config) kwargs
        "state_dict": {... module weights ...},
    }

Older training checkpoints that only contain ``state_dict`` are still loadable;
in that case the caller supplies the config from :mod:`sic.config`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .config import resolve_head

# Official release repo. Override with SIC_HF_REPO or the repo_id arg.
DEFAULT_HF_REPO = "yayapa/sic"


def get_repo_id(repo_id: Optional[str] = None) -> str:
    if repo_id:
        return repo_id
    return os.environ.get("SIC_HF_REPO", DEFAULT_HF_REPO)


def resolve_checkpoint(
    survival_head: str,
    model_path: Optional[str | Path] = None,
    repo_id: Optional[str] = None,
    allow_auto_download: bool = True,
) -> str:
    """Return a local path to the checkpoint for ``survival_head``.

    Resolution order:
    1. If ``model_path`` is given and exists on disk, use it directly.
    2. Otherwise download the head's checkpoint from the Hugging Face Hub
       (repo ``repo_id`` / ``$SIC_HF_REPO`` / the default placeholder) into the
       local HF cache. Requires ``allow_auto_download=True``.
    """
    if model_path is not None:
        model_path = Path(model_path).expanduser()
        if model_path.exists():
            return str(model_path)
        if not allow_auto_download:
            raise FileNotFoundError(
                f"Checkpoint not found at {model_path} and allow_auto_download=False."
            )

    if not allow_auto_download:
        raise FileNotFoundError(
            "No local checkpoint provided and allow_auto_download=False. "
            "Pass model_path=<file> or enable auto-download."
        )

    from huggingface_hub import hf_hub_download

    spec = resolve_head(survival_head)
    repo = get_repo_id(repo_id)
    return hf_hub_download(repo_id=repo, filename=spec["filename"])


def convert_training_checkpoint(
    src_path: str | Path,
    survival_head: str,
    dst_path: str | Path,
    config: Optional[dict] = None,
) -> str:
    """Convert an old training ``.ckpt`` into the strict-loadable release format.

    The training checkpoints saved by :class:`sic.train.run.Trainer` already
    contain ``state_dict`` and ``config``; this rewrites them into the minimal
    release schema so they load with ``strict=True`` and carry the head name.

    Parameters
    ----------
    src_path : path to the original checkpoint.
    survival_head : one of :data:`sic.config.SURVIVAL_HEADS`.
    dst_path : where to write the release checkpoint.
    config : optional model config override. If omitted, uses the ``config``
        stored in the source checkpoint, else the packaged defaults for the head.
    """
    import torch

    from .config import build_model_config

    resolve_head(survival_head)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"Source checkpoint {src_path} has no 'state_dict'.")
    cfg = config or ckpt.get("config") or build_model_config(survival_head)
    release = {
        "format_version": 1,
        "survival_head": survival_head,
        "config": dict(cfg),
        "state_dict": ckpt["state_dict"],
    }
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, dst_path)
    return str(dst_path)
