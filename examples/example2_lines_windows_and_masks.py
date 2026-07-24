#!/usr/bin/env python
"""Example 2: diagnostic windows, explicit masks, and a local line fit.

This example introduces one extra concept at a time after Example 1:

1. select advisory diagnostic windows from the loaded wavelength coverage;
2. build an explicit mask/warning bundle without mutating the spectrum;
3. fit one local line as a diagnostic, not as a replacement for PHOENIX.

Example:

  python examples/example2_lines_windows_and_masks.py \
    --output-json /tmp/spyctres_example2.json \
    --output-plot /tmp/spyctres_example2_windows.png \
    --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp
import numpy as np


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 2: plot diagnostic windows, build explicit masks, and "
            "fit one local line. No PHOENIX library is required."
        ),
        epilog=(
            "Example:\n"
            "  python examples/example2_lines_windows_and_masks.py "
            "--output-json /tmp/spyctres_example2.json "
            "--output-plot /tmp/spyctres_example2_windows.png --no-show"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=str(EXAMPLE_UVB),
        help="Spectrum file. Defaults to the bundled X-SHOOTER UVB example.",
    )
    parser.add_argument("--instrument", default="xshooter")
    parser.add_argument("--line", default="Hgamma", help="Known line name.")
    parser.add_argument(
        "--mask-dibs",
        action="store_true",
        help="Actually exclude curated optical DIB regions; default is warn only.",
    )
    parser.add_argument(
        "--archive",
        choices=("warn", "mask", "ignore"),
        default="warn",
        help="Archive/product bad-region policy for the mask bundle.",
    )
    parser.add_argument(
        "--tellurics",
        choices=("warn", "fallback", "none"),
        default="warn",
        help="Telluric policy for this lightweight example.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    args = build_parser().parse_args(argv)

    print("Reading spectrum...", flush=True)
    spec = sp.read_spectrum(args.spectrum, instrument=args.instrument)

    print("Selecting diagnostic windows...", flush=True)
    windows = sp.select_diagnostic_windows(spec)

    print("Building explicit mask/warning bundle...", flush=True)
    mask = sp.build_mask(
        spec,
        archive=False if args.archive == "ignore" else args.archive,
        tellurics=False if args.tellurics == "none" else args.tellurics,
        dibs=bool(args.mask_dibs),
    )

    print("Running one local line fit...", flush=True)
    line_result = sp.fit_line(spec, args.line)
    print(
        "Line {0}: success={1}, rv={2:.3g} km/s, flags={3}".format(
            args.line,
            line_result.success,
            line_result.rv_kms,
            ",".join(line_result.flags) or "none",
        ),
        flush=True,
    )

    print("Plotting diagnostic windows...", flush=True)
    fig, _ax = sp.plot_diagnostic_windows(
        spec,
        windows,
        mask=mask,
        show_nonstellar=True,
        title="Example 2: diagnostic windows and warning regions",
    )
    if args.output_plot:
        Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_plot, bbox_inches="tight")
        print("Saved plot: {0}".format(args.output_plot), flush=True)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "example": "example2_lines_windows_and_masks",
            "diagnostic_windows": windows,
            "mask": mask.to_metadata(),
            "line_fit": line_result.to_dict(),
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print("Wrote JSON: {0}".format(args.output_json), flush=True)

    import matplotlib.pyplot as plt

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
