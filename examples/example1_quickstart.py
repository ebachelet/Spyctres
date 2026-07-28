#!/usr/bin/env python
"""Example 1: read, inspect, review setup, and optionally run one PHOENIX fit.

Learning goal
-------------
This is the shortest beginner path through the public Spyctres API.  It uses
the bundled 18 Sco / HIP79672 Gaia FGK Benchmark Stars spectrum by default and
deliberately separates
"review the setup" from "launch the PHOENIX fit".

What this demonstrates
----------------------
Spectrum ingestion, a first-look plot, a reviewed ``FitSetup``, and an optional
first-pass PHOENIX classification fit.

What this does not prove
------------------------
It is not a publication-quality atmospheric-parameter solution, precision
line-width measurement, abundance analysis, or final wavelength/LSF validation.

Example dry run, no PHOENIX library needed:

  python examples/example1_quickstart.py --no-show

Example opt-in fit:

  python examples/example1_quickstart.py \
    --run-fit \
    --output-json /tmp/spyctres_example1_fit.json \
    --output-plot /tmp/spyctres_example1_fit.png \
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


EXAMPLE_BENCHMARK = (
    Path(__file__).resolve().parent
    / "data"
    / "gaia_benchmark"
    / "HIP79672_HARPS_1_R42KNorm.txt.gz"
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
        "--readiness-intent",
        choices=(
            "inspect",
            "quicklook_classification",
            "atmospheric_parameters",
            "radial_velocity",
            "publication",
        ),
        default="quicklook_classification",
        help="Task-specific readiness policy used by suggest_fit_setup().",
    )
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
    parser.add_argument("--output-plot", default=None)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive matplotlib windows.",
    )
    return parser


def _progress(event):
    message = event.get("message", str(event)) if isinstance(event, dict) else str(event)
    print(message, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instrument is not None:
        if args.reader != "gbs_v3_ascii":
            raise ValueError("Pass --reader or --instrument, not both.")
        args.reader = args.instrument

    print("Reading spectrum...", flush=True)
    spec = sp.read_spectrum(args.spectrum, reader=args.reader)

    # First-look plot: this shows what was loaded, before any fit is attempted.
    print("Plotting/inspecting loaded spectrum...", flush=True)
    fig, _ax = sp.plot_spectrum(spec, title="Example 1: loaded spectrum")
    if args.no_show:
        sp.ensure_matplotlib_config_dir()

    # A FitSetup is the reviewed plan.  It is cheap: no PHOENIX templates are
    # loaded here, and the setup hash later proves exactly what was fitted.
    print("Building reviewed FitSetup...", flush=True)
    setup = sp.suggest_fit_setup(
        spec,
        mode=args.defaults_mode,
        readiness_intent=args.readiness_intent,
        assumed_resolution=args.resolution_R,
    )
    print(setup.summary_text(), flush=True)

    payload = {
        "example": "example1_quickstart",
        "spectrum": str(args.spectrum),
        "reader": str(args.reader),
        "setup": setup.to_dict(),
    }

    result = None
    if args.run_fit:
        print("Running PHOENIX fit using the reviewed setup...", flush=True)
        result = sp.fit_stellar_spectrum(
            spec,
            model="phoenix",
            setup=setup,
            phoenix_dir=args.phoenix_dir,
            progress_callback=_progress,
        )
        print(result.quality_report_text(), flush=True)
        payload["fit_result"] = result.to_dict(
            include_arrays=False,
            include_local_paths=True,
        )
        if args.output_plot:
            plot_fig, _axes = sp.plot_fit_referee(result, savepath=args.output_plot)
            print("Saved fit plot: {0}".format(args.output_plot), flush=True)
            if args.no_show:
                import matplotlib.pyplot as plt

                plt.close(plot_fig)

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

        plt.close(fig)
    else:
        import matplotlib.pyplot as plt

        if result is None or not args.output_plot:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
