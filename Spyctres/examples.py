"""Small helpers for bundled Spyctres example data.

These helpers deliberately resolve only files that ship with, or sit beside, a
Spyctres source checkout.  They never fall back to a developer's Downloads or
local data directories, so tutorials remain reproducible for other users.
"""

from __future__ import annotations

from pathlib import Path


def example_data_path(relative_path, *, must_exist=True):
    """Return a path under ``examples/data`` for tutorial notebooks/scripts.

    Parameters
    ----------
    relative_path : str or path-like
        Path relative to the repository's ``examples/data`` directory, for
        example ``"gaia_benchmark/HIP79672_HARPS_1_R42KNorm.txt.gz"``.
    must_exist : bool, optional
        If True, raise a clear :class:`FileNotFoundError` when the example data
        file is not available in the current checkout/install context.

    Notes
    -----
    The numbered examples use this helper to avoid repeated path boilerplate.
    The returned path is local filesystem data for examples, not a general
    science-data downloader.
    """
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("example_data_path() expects a path relative to examples/data.")
    if any(part == ".." for part in relative.parts):
        raise ValueError("example_data_path() does not allow parent-directory components.")

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "examples" / "data" / relative
    if must_exist and not candidate.exists():
        raise FileNotFoundError(
            "Example data file {0!r} was not found under {1}. "
            "If you installed Spyctres from a wheel without example data, use "
            "a source checkout or pass your own spectrum path to read_spectrum().".format(
                str(relative),
                candidate.parent,
            )
        )
    return candidate
