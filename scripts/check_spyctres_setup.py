#!/usr/bin/env python
"""Compatibility wrapper for ``spyctres doctor``.

The implementation lives in :mod:`Spyctres.setup_check` so the same diagnostic
is available from editable installs, wheels, and source checkouts.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres.setup_check import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            prog="python scripts/check_spyctres_setup.py",
            description=(
                "Setup diagnostic for a local Spyctres + PHOENIX development "
                "environment."
            ),
        )
    )
