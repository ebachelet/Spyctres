#!/usr/bin/env python
"""Example 1: read, inspect, review setup, and optionally run one PHOENIX fit.

Welcome to Spyctres.

Learning goal
-------------
This is the shortest beginner path through the public Spyctres API.  It uses
the bundled 18 Sco / HIP79672 Gaia FGK Benchmark Stars spectrum by default and
deliberately separates
"review the setup" from "launch the PHOENIX fit".

What this demonstrates
----------------------
Spectrum ingestion, a first-look plot, a reviewed ``FitSetup``, and an optional
first-pass PHOENIX classification fit shown in a zoomed diagnostic-line grid.

What this does not prove
------------------------
It is not a reviewed-analysis atmospheric-parameter solution, precision
line-width measurement, abundance analysis, or final wavelength/LSF validation.

Example dry run, no PHOENIX library needed:

  python examples/example1_quickstart.py --no-show

Example observed-window plot without fitting:

  python examples/example1_quickstart.py \
    --output-plot /tmp/spyctres_example1_windows.png \
    --no-show

Example opt-in fit:

  python examples/example1_quickstart.py \
    --run-fit \
    --output-json /tmp/spyctres_example1_fit.json \
    --output-plot /tmp/spyctres_example1_fit.png \
    --no-show

Before running the PHOENIX fit, check the environment with:

  spyctres doctor --require-phoenix

or, if the console command has not been installed on PATH:

  python -m Spyctres.cli doctor --require-phoenix
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp
from Spyctres._serialization import atomic_write_json


EXAMPLE_BENCHMARK = sp.example_data_path(
    "gaia_benchmark/HIP79672_HARPS_1_R42KNorm.txt.gz"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 1 quickstart: read a clean bundled benchmark spectrum, "
            "plot/inspect it, "
            "review a FitSetup, and optionally run the reviewed PHOENIX fit."
        ),
        epilog=(
            "Fast first run:\n"
            "  python examples/example1_quickstart.py --no-show\n\n"
            "Run the fit after PHOENIX is configured:\n"
            "  python examples/example1_quickstart.py --run-fit "
            "--output-json /tmp/spyctres_example1_fit.json "
            "--output-plot /tmp/spyctres_example1_fit.png --no-show\n\n"
            "Next:\n"
            "  python examples/example2_lines_windows_and_masks.py --no-show"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=str(EXAMPLE_BENCHMARK),
        help="Spectrum file. Defaults to the bundled 18 Sco benchmark spectrum.",
    )
    parser.add_argument(
        "--reader",
        default="gbs_v3_ascii",
        help="Registered Spyctres reader. Default: gbs_v3_ascii.",
    )
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Setup/search-budget mode. Default: quicklook.",
    )
    parser.add_argument(
        "--intent",
        choices=(
            "inspect",
            "quicklook_classification",
            "atmospheric_parameters",
            "radial_velocity",
            "reviewed_analysis",
        ),
        default="quicklook_classification",
        help="Task-specific readiness policy used by suggest_fit_setup().",
    )
    parser.add_argument("--readiness-intent", dest="intent", help=argparse.SUPPRESS)
    parser.add_argument(
        "--R",
        type=float,
        default=None,
        dest="resolution_R",
        help="Optional explicit resolving-power assumption for setup/fit.",
    )
    parser.add_argument(
        "--run-fit",
        action="store_true",
        help="Launch the PHOENIX fit using exactly the reviewed setup.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--output-plot",
        default=None,
        help=(
            "Save the diagnostic-line plot. Without --run-fit this is an "
            "observed-only window plot; with --run-fit it includes the model."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive matplotlib windows.",
    )
    return parser


def _progress(event):
    message = event.get("message", str(event)) if isinstance(event, dict) else str(event)
    if message.startswith("Preparing PHOENIX interpolator/cache"):
        print("Preparing PHOENIX templates/cache...", flush=True)
        return
    shown_prefixes = (
        "Running coarse physical",
        "Coarse physical initialization selected",
        "Running coarse RV",
        "Coarse RV grid scan selected",
        "Starting local optimizer",
        "Finished local optimizer",
        "Selected best fit",
    )
    if not message.startswith(shown_prefixes):
        return
    print(message, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instrument is not None:
        if args.reader != "gbs_v3_ascii":
            raise ValueError("Pass --reader or --instrument, not both.")
        args.reader = args.instrument

    print("Reading spectrum...", flush=True)
    reader_info = sp.get_reader_info(args.reader)
    print(
        "Reader profile: {0} — {1}".format(
            reader_info.canonical_name,
            reader_info.expected_file_type,
        ),
        flush=True,
    )
    print(
        "  list options with sp.list_readers(); inspect one with "
        "sp.get_reader_info('reader_name').",
        flush=True,
    )
    # Read the spectrum with the selected reader.  Public container summaries
    # use the convention valid_mask=True means a pixel is usable.
    spec = sp.read_spectrum(args.spectrum, reader=args.reader)
    print("Spectrum summary:", flush=True)
    print(spec.summary(), flush=True)

    # First-look plot: this shows what was loaded, before any fit is attempted.
    print("Plotting/inspecting loaded spectrum...", flush=True)
    fig, _ax = sp.plot_spectrum(spec, title="Example 1: loaded spectrum")

    # A lightweight plotting tour: Spyctres can highlight broad diagnostic
    # regions that overlap the uploaded wavelength range. This helps a new user
    # decide what is worth inspecting before launching any expensive fit.
    diagnostic_selection = sp.select_diagnostic_windows(spec, max_windows=6)
    selected_windows = tuple(diagnostic_selection.selected[:4])
    print("\nInteresting regions to inspect", flush=True)
    print(diagnostic_selection.summary_text(max_rows=5), flush=True)
    print(
        "  These are plotting/review suggestions only; they are not automatic "
        "proofs of spectral type.",
        flush=True,
    )
    print(
        "  Manual windows are also accepted as (wmin, wmax) or "
        "('label', wmin, wmax) tuples.",
        flush=True,
    )
    diag_fig, _diag_ax = sp.plot_diagnostic_windows(
        spec,
        selection=diagnostic_selection,
        title="Example 1: suggested diagnostic windows",
    )
    if args.no_show:
        sp.ensure_matplotlib_config_dir()

    # A FitSetup is the reviewed plan. It is cheap: no PHOENIX templates are
    # loaded here.  The object contains a lot of provenance, but a new user only
    # needs the compact summary below.
    print("Building reviewed FitSetup...", flush=True)
    setup = sp.suggest_fit_setup(
        spec,
        mode=args.defaults_mode,
        intent=args.intent,
        assumed_resolution=args.resolution_R,
    )
    print(setup.summary_text(include_hash=False), flush=True)
    print(
        "  full reproducibility metadata is saved to JSON when --output-json is used.",
        flush=True,
    )

    payload = {
        "example": "example1_quickstart",
        "spectrum": str(args.spectrum),
        "reader": str(args.reader),
        "setup": setup.to_dict(),
    }

    result = None
    plot_fig = None
    if args.run_fit:
        print(
            "Before interpreting a PHOENIX fit, make sure "
            "`spyctres doctor --require-phoenix` passes. If the console command "
            "is unavailable, use `python -m Spyctres.cli doctor --require-phoenix`.",
            flush=True,
        )
        print("Running PHOENIX fit using the reviewed setup...", flush=True)
        result = sp.fit_stellar_spectrum(
            spec,
            model="phoenix",
            setup=setup,
            phoenix_dir=args.phoenix_dir,
            progress_callback=_progress,
        )
        print(result.summary_text(include_hash=False, max_flags=5), flush=True)
        payload["fit_result"] = result.to_dict(
            include_arrays=False,
            include_local_paths=True,
        )
        if args.output_plot or not args.no_show:
            segments = list(spec.segments) if hasattr(spec, "segments") else [spec]
            if len(segments) == 1 and result.models and selected_windows:
                plot_fig, _axes = sp.plot_model_line_windows(
                    result,
                    windows=selected_windows,
                    segment=spec,
                    savepath=args.output_plot,
                    title="Example 1: fitted diagnostic windows",
                    footer=None,
                    ncols=2,
                    figsize_per_panel=(7.2, 3.7),
                )
            else:
                plot_fig, _axes = sp.plot_fit_referee(
                    result,
                    segment=spec,
                    savepath=args.output_plot,
                )
        if args.output_plot:
            print("Saved fit plot: {0}".format(args.output_plot), flush=True)
    elif args.output_plot or not args.no_show:
        segments = list(spec.segments) if hasattr(spec, "segments") else [spec]
        if len(segments) == 1 and selected_windows:
            segment = segments[0]
            plot_fig, _axes = sp.plot_spectrum_line_windows(
                segment.wave,
                segment.flux,
                selected_windows,
                valid_mask=segment.valid_mask,
                savepath=args.output_plot,
                title="Example 1: observed diagnostic windows",
                ncols=2,
                figsize_per_panel=(7.2, 3.4),
            )
        else:
            plot_fig, _axes = sp.plot_diagnostic_windows(
                spec,
                savepath=args.output_plot,
                title="Example 1: observed diagnostic windows",
            )
        if args.output_plot:
            print("Saved observed-window plot: {0}".format(args.output_plot), flush=True)

    if args.output_json:
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)

    if args.run_fit:
        print(
            "\nScope note: Example 1 is an exploratory classification path. "
            "Treat the quality flags and setup assumptions as part of the "
            "result, not as decoration.",
            flush=True,
        )
    else:
        print(
            "\nScope note: no PHOENIX fit was run; this checked ingestion, "
            "plotting, and the reviewed setup only.",
            flush=True,
        )
    print(
        "Next: run examples/example2_lines_windows_and_masks.py to inspect "
        "diagnostic windows and explicit masks.",
        flush=True,
    )

    if args.no_show:
        import matplotlib.pyplot as plt

        if plot_fig is not None:
            plt.close(plot_fig)
        plt.close(diag_fig)
        plt.close(fig)
    else:
        import matplotlib.pyplot as plt

        if result is None or not args.output_plot:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
