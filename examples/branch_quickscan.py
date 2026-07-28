#!/usr/bin/env python
"""Plan or run bounded classification-branch quickscans.

This example is the fast "what should I try first?" workflow.  It reads a
spectrum, asks Spyctres which broad classification branches have useful feature
coverage, and writes compact JSON/CSV/PNG summaries.  The default is a dry run;
PHOENIX fits are launched only if ``--run-fits`` is supplied.

Example dry run using bundled data:

  python examples/branch_quickscan.py \
    examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
    --reader xshooter_merge1d \
    --output-json /tmp/spyctres_branch_quickscan.json \
    --output-csv /tmp/spyctres_branch_quickscan.csv \
    --output-plot /tmp/spyctres_branch_quickscan.png

Example opt-in bounded fit run:

  python examples/branch_quickscan.py \
    examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
    --reader xshooter_merge1d \
    --run-fits \
    --R 6200 \
    --max-branches 3 \
    --output-json /tmp/spyctres_branch_quickscan_fit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres import (
    plot_branch_quickscan,
    read_spectrum,
    run_branch_quickscan,
    write_branch_quickscan_csv,
    write_branch_quickscan_json,
)


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Branch quickscan scaffold. By default this only plans bounded "
            "classification-branch comparisons; use --run-fits to run PHOENIX."
        ),
        epilog=(
            "Fast dry run:\n"
            "  python examples/branch_quickscan.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--reader xshooter_merge1d --output-json /tmp/spyctres_branches.json "
            "--output-csv /tmp/spyctres_branches.csv "
            "--output-plot /tmp/spyctres_branches.png\n\n"
            "Bounded opt-in fit run:\n"
            "  python examples/branch_quickscan.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--reader xshooter_merge1d --run-fits --R 6200 --max-branches 3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=str(EXAMPLE_UVB),
        help="Spectrum to inspect. Defaults to the bundled X-SHOOTER UVB example.",
    )
    parser.add_argument(
        "--reader",
        default="xshooter_merge1d",
        help="Registered Spyctres reader. Default: xshooter_merge1d.",
    )
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_branch_quickscan.json",
        help="Atomic JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional compact CSV output path.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional compact PNG/SVG/PDF plot path.",
    )
    parser.add_argument(
        "--run-fits",
        action="store_true",
        help="Run bounded PHOENIX fits for the selected branches.",
    )
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--R",
        type=float,
        default=None,
        dest="resolution_R",
        help="Optional constant resolving power assumption for fit runs.",
    )
    parser.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Spyctres default budget used to build branch fit settings.",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=3,
        help="Maximum candidate branches to compare. Default: 3.",
    )
    parser.add_argument(
        "--rv-grid-n",
        type=int,
        default=None,
        help="Override the branch default RV-grid size for fit runs.",
    )
    parser.add_argument(
        "--multistart",
        type=int,
        default=None,
        help="Override the branch default number of local starts for fit runs.",
    )
    parser.add_argument(
        "--mdeg",
        type=int,
        default=2,
        help="Multiplicative Legendre continuum degree for fit runs.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=100,
        help="Maximum local-optimizer function evaluations per branch fit.",
    )
    parser.add_argument(
        "--coarse-decimate",
        type=int,
        default=12,
        help="Coarse-grid pixel decimation used during physical initialization.",
    )
    parser.add_argument(
        "--forward-model",
        choices=("native_interp", "interp_observed"),
        default="native_interp",
        help="Forward model used for opt-in branch fits. Default: native_interp.",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help=(
            "Reconstruct model arrays during opt-in fits. Off by default to "
            "keep branch quickscans fast."
        ),
    )
    return parser


def _progress(event):
    if isinstance(event, dict):
        print(event.get("message", str(event)), flush=True)
    else:
        print(str(event), flush=True)


def _base_fit_kwargs(args):
    # These are expert-visible fit-budget knobs.  They override only the small
    # branch defaults; the branch-specific regions and parameter bounds still
    # come from the auditable branch plan unless the code is edited explicitly.
    fit_kwargs = {
        "forward_model": args.forward_model,
        "mdeg": int(args.mdeg),
        "max_nfev": int(args.max_nfev),
        "coarse_decimate": int(args.coarse_decimate),
    }
    if args.resolution_R is not None:
        fit_kwargs["R"] = float(args.resolution_R)
    if args.rv_grid_n is not None:
        fit_kwargs["rv_grid_n"] = int(args.rv_grid_n)
    if args.multistart is not None:
        fit_kwargs["multistart"] = int(args.multistart)
    return fit_kwargs


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instrument is not None:
        if args.reader != "xshooter_merge1d":
            raise ValueError("Pass --reader or --instrument, not both.")
        args.reader = args.instrument

    # First canonicalize the input.  Everything downstream sees the same
    # SpectrumSegment/SpectrumCollection structure regardless of reader.
    print("Reading spectrum...", flush=True)
    spectrum = read_spectrum(args.spectrum, reader=args.reader)

    # Build the branch plan, and optionally run a bounded number of PHOENIX
    # fits.  The default path is cheap and does not load the PHOENIX library.
    print("Building branch quickscan plan...", flush=True)
    if args.run_fits:
        print("Running opt-in branch PHOENIX fits...", flush=True)
    payload = run_branch_quickscan(
        spectrum,
        run_fits=bool(args.run_fits),
        mode=args.defaults_mode,
        max_branches=int(args.max_branches),
        fit_call_kwargs={
            "phoenix_dir": args.phoenix_dir,
            "reconstruct": bool(args.reconstruct),
            "progress_callback": _progress,
        },
        base_fit_kwargs=_base_fit_kwargs(args),
        progress_callback=_progress,
    )

    # JSON is the authoritative product.  CSV/PNG are lightweight views for
    # quick inspection and reviewer communication.
    print("Writing branch quickscan outputs...", flush=True)
    write_branch_quickscan_json(args.output_json, payload)
    write_branch_quickscan_csv(args.output_csv, payload)
    if args.output_plot is not None:
        fig, _axes = plot_branch_quickscan(payload, savepath=args.output_plot)
        import matplotlib.pyplot as plt

        plt.close(fig)

    nplanned = len(payload.get("planned_branches", ()))
    nskipped = len(payload.get("skipped_branches", ()))
    print(
        "Done. status={0}, planned={1}, skipped={2}, output={3}".format(
            payload.get("status"),
            nplanned,
            nskipped,
            args.output_json,
        ),
        flush=True,
    )
    if not args.run_fits:
        print("Dry run only. Add --run-fits to launch bounded PHOENIX fits.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
