#!/usr/bin/env python
"""Example 3: compare quicklook and stronger reviewed PHOENIX setups.

This example shows how to improve a first pass without blindly expanding the
whole PHOENIX grid: compare the reviewed setup summaries first, then optionally
run the two fits and compare their structured results.

Example dry run:

  python examples/example3_improving_a_phoenix_fit.py --no-show

Example opt-in fit comparison:

  python examples/example3_improving_a_phoenix_fit.py \
    --run-fits --R 6200 \
    --output-json /tmp/spyctres_example3.json \
    --output-plot /tmp/spyctres_example3_standard.png \
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


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 3: review quicklook versus standard FitSetup objects, "
            "then optionally run and compare the PHOENIX fits."
        ),
        epilog=(
            "Dry run:\n"
            "  python examples/example3_improving_a_phoenix_fit.py --no-show\n\n"
            "Run fits:\n"
            "  python examples/example3_improving_a_phoenix_fit.py --run-fits "
            "--R 6200 --output-json /tmp/spyctres_example3.json "
            "--output-plot /tmp/spyctres_example3_standard.png --no-show"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectrum", nargs="?", default=str(EXAMPLE_UVB))
    parser.add_argument("--instrument", default="xshooter")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument("--run-fits", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _progress(event):
    message = event.get("message", str(event)) if isinstance(event, dict) else str(event)
    print(message, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)

    print("Reading spectrum...", flush=True)
    spec = sp.read_spectrum(args.spectrum, instrument=args.instrument)

    print("Reviewing quicklook setup...", flush=True)
    quick_setup = sp.suggest_fit_setup(
        spec,
        mode="quicklook",
        assumed_resolution=args.resolution_R,
    )
    print(quick_setup.summary_text(), flush=True)

    print("Reviewing stronger standard setup...", flush=True)
    standard_setup = sp.suggest_fit_setup(
        spec,
        mode="standard",
        assumed_resolution=args.resolution_R,
    )
    print(standard_setup.summary_text(), flush=True)

    payload = {
        "example": "example3_improving_a_phoenix_fit",
        "quicklook_setup": quick_setup.to_dict(),
        "standard_setup": standard_setup.to_dict(),
    }

    if args.run_fits:
        print("Running quicklook fit...", flush=True)
        quick = sp.fit_stellar_spectrum(
            spec,
            model="phoenix",
            setup=quick_setup,
            phoenix_dir=args.phoenix_dir,
            progress_callback=_progress,
        )
        print("Running standard fit...", flush=True)
        standard = sp.fit_stellar_spectrum(
            spec,
            model="phoenix",
            setup=standard_setup,
            phoenix_dir=args.phoenix_dir,
            progress_callback=_progress,
        )
        comparison = sp.compare_fits(
            quick,
            standard,
            labels=("quicklook", "standard"),
        )
        payload.update(
            {
                "quicklook_result": quick.to_dict(
                    include_arrays=False,
                    include_local_paths=True,
                ),
                "standard_result": standard.to_dict(
                    include_arrays=False,
                    include_local_paths=True,
                ),
                "comparison": comparison,
            }
        )
        print(standard.quality_report_text(), flush=True)
        if args.output_plot:
            fig, _axes = sp.plot_fit_referee(standard, savepath=args.output_plot)
            print("Saved standard-fit plot: {0}".format(args.output_plot), flush=True)
            if args.no_show:
                import matplotlib.pyplot as plt

                plt.close(fig)

    if args.output_json:
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
