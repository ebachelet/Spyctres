#!/usr/bin/env python
"""Example 2: diagnostic windows, explicit masks, and a local line fit.

This example introduces one extra concept at a time after Example 1:

1. select advisory diagnostic windows from the loaded wavelength coverage;
2. build an explicit mask/warning bundle without mutating the spectrum;
3. fit one local line as a diagnostic, not as a replacement for PHOENIX.

What this demonstrates
----------------------
How Spyctres records why pixels are warned about or excluded, and how local
line fits can diagnose wavelength/line-shape issues before a full PHOENIX fit.

What this does not prove
------------------------
The local line fit is not a stellar-parameter classifier and the advisory
windows are not hidden automatic arm/continuum corrections.

Example:

  python examples/example2_lines_windows_and_masks.py \
    --output-json /tmp/spyctres_example2.json \
    --output-plot /tmp/spyctres_example2_windows.png \
    --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp
from Spyctres._serialization import atomic_write_json
from Spyctres.plotting import save_figure


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
            "--output-plot /tmp/spyctres_example2_windows.png --no-show\n\n"
            "Next:\n"
            "  python examples/example3_improving_a_phoenix_fit.py --no-show"
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
        save_figure(fig, args.output_plot)
        print("Saved plot: {0}".format(args.output_plot), flush=True)

    if args.output_json:
        payload = {
            "example": "example2_lines_windows_and_masks",
            "diagnostic_windows": windows,
            "mask": mask.to_metadata(),
            "line_fit": line_result.to_dict(),
        }
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)

    print(
        "\nScope note: Example 2 is a diagnostic-mask and local-line workflow; "
        "it does not replace the PHOENIX full-spectrum fit.",
        flush=True,
    )
    print(
        "Next: run examples/example3_improving_a_phoenix_fit.py to compare "
        "quicklook and stronger reviewed fit setups.",
        flush=True,
    )

    import matplotlib.pyplot as plt

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
