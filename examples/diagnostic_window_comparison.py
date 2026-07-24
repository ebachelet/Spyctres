#!/usr/bin/env python
"""Plan or run bounded diagnostic-window comparison fits.

This example uses spectra bundled under ``examples/data/`` by default.  It is
meant for "which spectral features are driving this answer?" checks, not for
automatic publication-grade model selection.  The default mode is a dry run
that selects windows, builds a small comparison plan, and writes JSON/CSV/PNG
summaries without loading PHOENIX.

Example quick dry run:

  python examples/diagnostic_window_comparison.py \
    examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
    --instrument xshooter \
    --output-json /tmp/spyctres_window_comparison.json \
    --output-csv /tmp/spyctres_window_comparison.csv \
    --output-plot /tmp/spyctres_window_comparison.png

Example opt-in bounded fit run:

  python examples/diagnostic_window_comparison.py \
    examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
    --instrument xshooter \
    --run-fits \
    --R 6200 \
    --max-comparisons 4 \
    --output-json /tmp/spyctres_window_comparison_fit.json

Fit runs write both held-out residual checks and common-window residual
summaries by default.  Use ``--no-evaluate-heldout --no-evaluate-common`` only
for timing-only runs where reconstructed model diagnostics are unnecessary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres import (
    plot_diagnostic_window_comparison,
    read_spectrum,
    run_diagnostic_window_comparison,
    write_diagnostic_window_comparison_csv,
    write_diagnostic_window_comparison_json,
)


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-window comparison scaffold. By default this only "
            "plans bounded feature-window comparisons; use --run-fits to run "
            "the expensive PHOENIX fits."
        ),
        epilog=(
            "Fast dry run:\n"
            "  python examples/diagnostic_window_comparison.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter --output-json /tmp/spyctres_windows.json "
            "--output-csv /tmp/spyctres_windows.csv "
            "--output-plot /tmp/spyctres_windows.png\n\n"
            "Bounded opt-in fit run:\n"
            "  python examples/diagnostic_window_comparison.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter --run-fits --R 6200 --max-comparisons 4"
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
        "--instrument",
        default="xshooter",
        help="Registered Spyctres instrument reader. Default: xshooter.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_diagnostic_window_comparison.json",
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
        help=(
            "Run the bounded PHOENIX comparison fits. Default is dry-run "
            "planning only."
        ),
    )
    parser.add_argument(
        "--evaluate-heldout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --run-fits is active, reconstruct each fitted model and score "
            "selected-but-held-out diagnostic windows. Default: true."
        ),
    )
    parser.add_argument(
        "--evaluate-common",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --run-fits is active, score every fit on the same union of "
            "planned comparison windows. Default: true."
        ),
    )
    parser.add_argument(
        "--holdout-min-pixels",
        type=int,
        default=3,
        help="Minimum valid unfitted pixels required to score a held-out window.",
    )
    parser.add_argument(
        "--common-min-pixels",
        type=int,
        default=None,
        help=(
            "Minimum valid pixels required to score a common-evaluation window. "
            "Default: same as --holdout-min-pixels."
        ),
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
        "--roles",
        default=None,
        help="Optional comma-separated diagnostic roles, e.g. temperature,gravity.",
    )
    parser.add_argument(
        "--initial-teff",
        type=float,
        default=None,
        help="Optional soft Teff hint for diagnostic-window applicability scoring.",
    )
    parser.add_argument(
        "--rv",
        type=float,
        default=None,
        help=(
            "Optional preliminary stellar RV used only to map canonical "
            "rest-frame windows onto observed-frame spectra."
        ),
    )
    parser.add_argument(
        "--rv-padding-kms",
        type=float,
        default=0.0,
        help="Velocity padding applied to selected windows. Default: 0.",
    )
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--max-single-windows", type=int, default=5)
    parser.add_argument("--max-comparisons", type=int, default=8)
    parser.add_argument(
        "--exclude-warn-windows",
        action="store_true",
        help=(
            "Skip windows with default_fit_policy='warn'. By default they are "
            "included but clearly labelled."
        ),
    )
    parser.add_argument(
        "--include-stress-windows",
        action="store_true",
        help=(
            "Include stress-only windows such as He/NLTE-sensitive features in "
            "fit combinations. Off by default."
        ),
    )
    parser.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="diagnostic",
        help="Spyctres default budget for opt-in fit runs.",
    )
    parser.add_argument("--rv-grid-n", type=int, default=41)
    parser.add_argument("--multistart", type=int, default=2)
    parser.add_argument("--mdeg", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser


def _parse_roles(value):
    if value is None:
        return None
    roles = [item.strip() for item in str(value).split(",") if item.strip()]
    return roles or None


def _progress(event):
    if isinstance(event, dict):
        print(event.get("message", str(event)), flush=True)
    else:
        print(str(event), flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Ingest once, then keep all downstream operations on Spyctres' canonical
    # SpectrumSegment/SpectrumCollection representation.
    print("Reading spectrum...", flush=True)
    spectrum = read_spectrum(args.spectrum, instrument=args.instrument)

    # Fit options are deliberately small and bounded.  The comparison runner
    # will override ``regions`` for each selected window combination.
    base_fit_kwargs = {
        "forward_model": "native_interp",
        "rv_init": "grid",
        "rv_grid_n": int(args.rv_grid_n),
        "multistart": int(args.multistart),
        "mdeg": int(args.mdeg),
        "max_nfev": int(args.max_nfev),
    }
    if args.resolution_R is not None:
        base_fit_kwargs["R"] = float(args.resolution_R)

    fit_call_kwargs = {
        "model": "phoenix",
        "phoenix_dir": args.phoenix_dir,
        "auto_defaults": True,
        "defaults_mode": args.defaults_mode,
        "science_case": "diagnostic_window_comparison",
        # Held-out/common residual checks need reconstructed model arrays. Turn
        # both off when scalar timing comparisons are all you need.
        "reconstruct": bool(args.evaluate_heldout or args.evaluate_common),
        "progress_callback": _progress,
    }

    # The default is a cheap plan.  Expensive PHOENIX calls only happen if the
    # user explicitly supplies --run-fits.
    print("Building diagnostic-window comparison plan...", flush=True)
    payload = run_diagnostic_window_comparison(
        spectrum,
        run_fits=bool(args.run_fits),
        evaluate_heldout=bool(args.evaluate_heldout),
        evaluate_common=bool(args.evaluate_common),
        holdout_min_pixels=int(args.holdout_min_pixels),
        common_min_pixels=args.common_min_pixels,
        fit_call_kwargs=fit_call_kwargs,
        base_fit_kwargs=base_fit_kwargs,
        roles=_parse_roles(args.roles),
        initial_teff=args.initial_teff,
        rv_kms=args.rv,
        rv_padding_kms=args.rv_padding_kms,
        max_windows=args.max_windows,
        max_single_windows=args.max_single_windows,
        max_comparisons=args.max_comparisons,
        include_warn_windows=not args.exclude_warn_windows,
        include_stress_windows=bool(args.include_stress_windows),
        progress_callback=_progress,
    )

    print("Writing diagnostic-window comparison outputs...", flush=True)
    write_diagnostic_window_comparison_json(args.output_json, payload)
    write_diagnostic_window_comparison_csv(args.output_csv, payload)
    if args.output_plot is not None:
        fig, _axes = plot_diagnostic_window_comparison(
            payload,
            savepath=args.output_plot,
        )
        # Keep command-line execution non-interactive unless the user opens the
        # saved figure separately.
        import matplotlib.pyplot as plt

        plt.close(fig)

    planned = len(payload.get("planned_comparisons", ()))
    skipped = len(payload.get("skipped_comparisons", ()))
    print(
        "Done. status={0}, planned={1}, skipped={2}, output={3}".format(
            payload.get("status"),
            planned,
            skipped,
            args.output_json,
        ),
        flush=True,
    )
    if not args.run_fits:
        print("Dry run only. Add --run-fits to launch bounded PHOENIX fits.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
