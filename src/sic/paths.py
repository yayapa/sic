"""Filesystem locations used by SIC.

No absolute paths are hardcoded. Everything resolves to a per-user cache
directory that can be overridden with the ``SIC_CACHE_DIR`` environment
variable. Downloaded checkpoints and any generated artifacts live there.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs


def get_cache_dir() -> Path:
    """Return the SIC cache directory, creating it if necessary.

    Resolution order:
    1. ``$SIC_CACHE_DIR`` if set.
    2. The platform user cache dir (e.g. ``~/.cache/sic`` on Linux).
    """
    env = os.environ.get("SIC_CACHE_DIR")
    root = Path(env).expanduser() if env else Path(platformdirs.user_cache_dir("sic"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_checkpoint_dir() -> Path:
    """Directory where downloaded model checkpoints are cached."""
    d = get_cache_dir() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
