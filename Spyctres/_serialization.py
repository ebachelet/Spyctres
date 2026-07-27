"""Small internal helpers for strict JSON and atomic artifact writes.

The validation runners and structured result objects all need the same
low-level behavior: convert NumPy-rich payloads to native JSON types, reject
NaN/Inf in emitted JSON, create output directories, and avoid leaving
half-written files after interrupted runs. Keeping those mechanics here avoids
quietly diverging checkpoint semantics between scripts.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def json_safe(value):
    """Return ``value`` converted to strict native JSON-compatible types."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_write_json(path, payload, *, indent=2, sort_keys=False, **json_kwargs):
    """Write strict JSON through a temporary file and atomic replace."""
    json_kwargs.setdefault("allow_nan", False)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(payload),
                handle,
                indent=indent,
                sort_keys=sort_keys,
                **json_kwargs,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv_rows(path, fieldnames, rows):
    """Write CSV rows through a temporary file and atomic replace."""
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def safe_filename(value, fallback="target"):
    """Return a conservative filename stem for generated artifacts."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = text.strip("._")
    return text or str(fallback)
