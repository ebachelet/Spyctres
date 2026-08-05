#!/usr/bin/env python
"""Example 8: PEPSI legacy line-window validation.

This numbered example preserves the historical PEPSI line-window validation
workflow, but presents it in the maintained Spyctres style.  The default run is
PHOENIX-free: it reads bundled PEPSI ``.nor`` spectra, builds the legacy line
windows through ``Spyctres.recipes``, and plots/prints the prepared windows.
The actual optimizer remains in ``scripts/pepsi_fit_smoketest.py`` so the
notebook and script do not duplicate fitting code.

What this demonstrates
----------------------
How the PEPSI compatibility path converts full-arm ``.nor`` spectra into the
same ``SpectrumCollection`` abstraction used elsewhere, while keeping the
wavelength-medium/frame assumptions explicit.

What this does not prove
------------------------
The legacy line-window result is a regression/compatibility check, not a
general replacement for the reviewed PHOENIX fitting workflow.  PEPSI products
from different releases can use different wavelength conventions, so users
must verify release documentation and FITS headers before changing velocity or
wavelength assumptions.

Example inspection run, no PHOENIX library needed:

  python examples/example8_pepsi_legacy_linefit_validation.py --no-show

Inspection plot:

  python examples/example8_pepsi_legacy_linefit_validation.py \
    --plot-dir /tmp/spyctres_example8_pepsi \
    --no-show

Full legacy regression fit, requires PHOENIX and creates a compact model
overlay grid:

  python examples/example8_pepsi_legacy_linefit_validation.py \
    --run-legacy-fit \
    --plot-dir /tmp/spyctres_example8_pepsi \
    --no-show
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_DIR.parent
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp


DEFAULT_PEPSI_FILES = (
    "pepsir.20230603.009.dxt.nor",
    "pepsir.20230603.010.dxt.nor",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 8: inspect PEPSI legacy line-window preparation and show "
            "the maintained command for the full legacy PHOENIX validation."
        ),
        epilog=(
            "Inspect bundled PEPSI windows:\n"
            "  python examples/example8_pepsi_legacy_linefit_validation.py --no-show\n\n"
            "Save an inspection plot:\n"
            "  python examples/example8_pepsi_legacy_linefit_validation.py "
            "--plot-dir /tmp/spyctres_example8_pepsi --no-show\n\n"
            "Run the full legacy optimizer:\n"
            "  python examples/example8_pepsi_legacy_linefit_validation.py "
            "--run-legacy-fit --plot-dir /tmp/spyctres_example8_pepsi --no-show"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectra",
        nargs="*",
        help="Optional PEPSI .nor spectra. Defaults to bundled red PEPSI files.",
    )
    parser.add_argument("--reader", default="pepsi_nor")
    parser.add_argument(
        "--wave-hypothesis",
        choices=("air", "vacuum", "unknown", "air_to_vac"),
        default="air",
        help="Explicit working wavelength-medium hypothesis for the legacy windows.",
    )
    parser.add_argument("--legacy-halfwidth", type=float, default=10.0)
    parser.add_argument("--window-pad", type=float, default=2.0)
    parser.add_argument("--legacy-flux-min", type=float, default=0.2)
    parser.add_argument("--legacy-flux-max", type=float, default=1.1)
    parser.add_argument(
        "--use-telluric-mask",
        action="store_true",
        help=(
            "Apply the high-resolution transmission-threshold telluric mask "
            "during inspection. This is opt-in because PEPSI frame conventions "
            "must be verified for each product."
        ),
    )
    parser.add_argument("--telluric-threshold", type=float, default=0.90)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument(
        "--run-legacy-fit",
        action="store_true",
        help=(
            "Run the maintained PEPSI legacy PHOENIX regression fit and write "
            "a compact data/model line-window diagnostic. Requires a configured "
            "local PHOENIX library."
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def _example_paths():
    return [sp.example_data_path(name) for name in DEFAULT_PEPSI_FILES]


def _maybe_show(no_show):
    import matplotlib.pyplot as plt

    if no_show:
        plt.close("all")
    else:
        plt.show()


def _print_reader_context(reader):
    info = sp.get_reader_info(reader).to_metadata()
    print("Reader:", info["canonical_name"])
    print("  aliases:", ", ".join(info["aliases"]))
    print("  product:", info["expected_file_type"])
    print("  default medium:", info["default_wave_medium"])
    print("  default observer frame:", info["default_observer_frame"])
    print("  stellar rest status:", info["default_stellar_rest_status"])
    print("  resolution:", info["resolving_power"])


def _build_collection(args):
    paths = [Path(path) for path in args.spectra] if args.spectra else _example_paths()
    raw_segments = []
    for path in paths:
        print("Reading PEPSI spectrum:", path, flush=True)
        raw_segments.append(sp.read_spectrum(path, reader=args.reader))

    print("Building PEPSI legacy line windows...", flush=True)
    input_segments, fit_segments, window_defs_air = sp.recipes.build_pepsi_legacy_segments(
        raw_segments,
        wave_hypothesis=args.wave_hypothesis,
        halfwidth_A=args.legacy_halfwidth,
        flux_min=args.legacy_flux_min,
        flux_max=args.legacy_flux_max,
        window_pad_A=args.window_pad,
    )

    if args.use_telluric_mask:
        print("Applying high-resolution telluric transmission mask...", flush=True)
        telluric_mask = sp.telluric_transmission_exclusion_mask(
            threshold=args.telluric_threshold
        )
        fit_segments = [
            seg.copy(mask=np.asarray(seg.mask, dtype=bool) & ~telluric_mask(seg.wave))
            for seg in fit_segments
        ]
        print(
            "  method={0}, threshold={1}, model={2}, frame_type={3}".format(
                telluric_mask.metadata.get("method"),
                telluric_mask.metadata.get("threshold"),
                telluric_mask.metadata.get("model_file"),
                telluric_mask.metadata.get("frame_type"),
            )
        )

    collection = sp.SpectrumCollection(
        fit_segments,
        name="example8_pepsi_legacy_line_windows",
        meta={
            "workflow": "example8_pepsi_legacy_linefit_validation",
            "wave_hypothesis": args.wave_hypothesis,
            "legacy_window_defs_air": [list(item) for item in window_defs_air],
            "input_files": [str(path) for path in paths],
        },
    )
    return input_segments, collection


def _print_collection_summary(collection):
    summary = collection.summary()
    print("\nPrepared PEPSI line-window collection:")
    print("  segments:", summary["n_segments"])
    print("  pixels:", summary["n_pixels"])
    print("  valid fraction: {0:.3f}".format(summary["valid_fraction"]))
    print("  wave media:", ", ".join(summary["wave_mediums"]))
    print("\n  window                     valid/total       range [Å]")
    print("  ---------------------------------------------------------")
    for segment in collection.segments:
        segment_summary = segment.summary()
        wmin, wmax = segment_summary["wavelength_range_A"]
        print(
            "  {0:<24} {1:>5}/{2:<7} {3:8.2f}-{4:8.2f}".format(
                str(segment.name)[:24],
                segment_summary["n_valid_pixels"],
                segment_summary["n_pixels"],
                wmin,
                wmax,
            )
        )


def _plot_collection(collection, plot_dir=None):
    savepath = None
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        savepath = plot_dir / "example8_pepsi_legacy_windows.png"
    fig, _axes = sp.plot_prepared_line_window_diagnostics(
        collection,
        title="Example 8: PEPSI legacy line windows; no PHOENIX fit has been run",
        footer=(
            "Blue = prepared normalized PEPSI windows; gray = pixels not used by "
            "the legacy mask. The orange model overlay appears only in the "
            "optional PHOENIX legacy fit."
        ),
        ncols=min(6, len(collection.segments)),
        savepath=savepath,
    )
    if savepath is not None:
        print("Wrote", savepath)
    return fig


def _legacy_fit_command(args, output_line_plot=None):
    paths = [Path(path) for path in args.spectra] if args.spectra else _example_paths()
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "pepsi_fit_smoketest.py"),
        "--preset",
        "pepsi_legacy_red_fast",
        "--wave-hypothesis",
        args.wave_hypothesis,
    ]
    if args.use_telluric_mask:
        command.extend(
            ["--use-telluric-mask", "--telluric-threshold", str(args.telluric_threshold)]
        )
    if output_line_plot is not None:
        command.extend(["--output-line-plot", str(output_line_plot)])
    if args.no_show:
        command.append("--no-show")
    command.extend(str(path) for path in paths)
    return command


def _maybe_run_legacy_fit(args):
    output_line_plot = None
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_line_plot = plot_dir / "example8_pepsi_legacy_fit.png"

    command = _legacy_fit_command(args, output_line_plot=output_line_plot)
    print("\nMaintained full legacy fit command:")
    print(" ".join(str(item) for item in command))

    if not args.run_legacy_fit:
        print(
            "\nNot running the optimizer by default. Add --run-legacy-fit after "
            "PHOENIX is configured to create the model-overlaid diagnostic grid."
        )
        return

    print("\nRunning full PEPSI legacy regression fit...", flush=True)
    subprocess.run(command, check=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    _print_reader_context(args.reader)
    _input_segments, collection = _build_collection(args)
    _print_collection_summary(collection)
    _plot_collection(collection, plot_dir=args.plot_dir)
    _maybe_run_legacy_fit(args)
    _maybe_show(args.no_show)
    print(
        "\nInterpretation: this example validates PEPSI window preparation and "
        "can call the shared regression runner for a compact model/data grid. "
        "Do not infer wavelength frame or stellar-rest corrections from the "
        ".dxt.nor suffix alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
