"""Matplotlib runtime setup helpers for Spyctres command-line tools."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def ensure_matplotlib_config_dir():
    """Set a writable Matplotlib config/cache directory when needed.

    Some shared or sandboxed systems have a non-writable ``~/.config``. If
    ``MPLCONFIGDIR`` is unset, Matplotlib prints a noisy warning and creates a
    random temporary cache directory at import time. Spyctres scripts call this
    helper before importing ``matplotlib.pyplot`` so first-run CLI output stays
    readable and Matplotlib can reuse a stable cache between runs.

    User-supplied ``MPLCONFIGDIR`` values are preserved.
    """
    if "MPLCONFIGDIR" in os.environ:
        return os.environ["MPLCONFIGDIR"]
    path = Path(tempfile.gettempdir()) / "spyctres_matplotlib_cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(path)
    return str(path)
