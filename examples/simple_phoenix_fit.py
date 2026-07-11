"""Minimal command-line example for the public Spyctres PHOENIX API.

Purpose
-------
This script demonstrates the recommended public `fit_stellar_spectrum()`
workflow from a reduced spectrum to a structured PHOENIX result and an
interactive diagnostic plot. The plot is a
wide, stacked diagnostic view: observed spectrum and model on top, residuals
underneath. This makes it easier to inspect broad failures, masked regions, and
line-profile mismatches than a small square pop-up.

The script does perform a full-spectrum fit over the usable input pixels, but it
is not configured as a precision line-profile analysis. In particular, it does
not fit rotational or macroturbulent broadening, detailed abundance patterns, or
instrument-specific LSF variations. The reader's nominal resolution (or ``--R``)
is the only instrumental broadening supplied. Individual observed lines can
therefore be wider or narrower than the demonstration model even when the broad
atmospheric classification is useful.

Example
-------
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_result.json \
  --output-plot /tmp/spyctres_fit.png
"""

import argparse
import json

from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt
from Spyctres import fit_stellar_spectrum, prepare_phoenix_fit_kwargs
from Spyctres.io import read_spectrum
from Spyctres.plotting import plot_fit_referee


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal fit_stellar_spectrum() PHOENIX demonstration. This is "
            "not a precision line-profile fit: rotation, macroturbulence, "
            "abundance variations, and detailed LSF structure are not fitted."
        ),
        epilog=(
            "Example:\n"
            "  python examples/simple_phoenix_fit.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter "
            "--output-json /tmp/spyctres_result.json "
            "--output-plot /tmp/spyctres_fit.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spectrum", help="Reduced one-dimensional spectrum file.")
    parser.add_argument("--instrument", required=True, help="Registered reader name.")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--auto-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use spectrum metadata/coverage to choose first-pass fit defaults. "
            "Expert CLI values still override the suggestions."
        ),
    )
    parser.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Search-budget mode used by --auto-defaults.",
    )
    parser.add_argument("--teff", type=float, default=None)
    parser.add_argument("--feh", type=float, default=None)
    parser.add_argument("--logg", type=float, default=None)
    parser.add_argument("--rv", type=float, default=None)
    parser.add_argument("--teff-min", type=float, default=None)
    parser.add_argument("--teff-max", type=float, default=None)
    parser.add_argument("--feh-min", type=float, default=None)
    parser.add_argument("--feh-max", type=float, default=None)
    parser.add_argument("--logg-min", type=float, default=None)
    parser.add_argument("--logg-max", type=float, default=None)
    parser.add_argument("--rv-min", type=float, default=None)
    parser.add_argument("--rv-max", type=float, default=None)
    parser.add_argument("--wmin", type=float, default=None, help="Override fit-window minimum wavelength in Angstrom.")
    parser.add_argument("--wmax", type=float, default=None, help="Override fit-window maximum wavelength in Angstrom.")
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument(
        "--plot-layout",
        choices=("stacked", "side_by_side"),
        default="stacked",
        help=(
            "Diagnostic plot layout. 'stacked' is the default interactive view: "
            "wide data/model panel over a wide residual panel."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive fit figure (useful for batch runs).",
    )
    return parser


def _fit_kwargs_from_args(args, spectrum):
    return prepare_phoenix_fit_kwargs(
        spectrum,
        auto_defaults=args.auto_defaults,
        defaults_mode=args.defaults_mode,
        science_case="classification",
        p0_overrides=(args.teff, args.feh, args.logg, args.rv),
        lower_bound_overrides=(
            args.teff_min,
            args.feh_min,
            args.logg_min,
            args.rv_min,
        ),
        upper_bound_overrides=(
            args.teff_max,
            args.feh_max,
            args.logg_max,
            args.rv_max,
        ),
        window=(
            args.wmin,
            args.wmax,
        ) if args.wmin is not None or args.wmax is not None else None,
        resolution_R=args.resolution_R,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("Reading spectrum...", flush=True)
    spectrum = read_spectrum(args.spectrum, instrument=args.instrument)
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, spectrum)
    if suggestion is not None:
        print("Suggested first-pass fit defaults:", flush=True)
        for reason in suggestion.reasons:
            print("  - {0}".format(reason), flush=True)
        for warning in suggestion.warnings:
            print("  WARNING: {0}".format(warning), flush=True)
    print("Running public fit_stellar_spectrum() workflow...", flush=True)
    result = fit_stellar_spectrum(
        spectrum,
        model="phoenix",
        phoenix_dir=args.phoenix_dir,
        auto_defaults=False,
        science_case="classification",
        progress_callback=lambda event: print(event, flush=True),
        **fit_kwargs,
    )
    if suggestion is not None:
        result.summary["fit_default_suggestion"] = suggestion.to_dict()
    summary = {
        key: result[key]
        for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
    }
    print(json.dumps(summary, indent=2))
    print()
    print(result.quality_report_text())
    print(
        "\nInterpretation: this example demonstrates ingestion, a native-grid "
        "full-spectrum PHOENIX fit, structured results, and plotting. It is not "
        "a precision line-width fit; line-profile mismatches can reflect stellar "
        "rotation/macroturbulence, abundance differences, or an approximate LSF."
    )

    if not result.models:
        if args.output_json:
            result.save_json(args.output_json)
        raise RuntimeError("Fit did not converge, so no model is available to plot.")
    print("Building diagnostic plot...", flush=True)
    fig, _axes = plot_fit_referee(
        result,
        segment=spectrum,
        savepath=args.output_plot,
        layout=args.plot_layout,
        figsize_per_segment=(
            (16.0, 6.4) if args.plot_layout == "stacked" else (12.0, 3.4)
        ),
        max_points_per_segment=20000,
    )
    if args.output_json:
        result.save_json(
            args.output_json,
            plot_paths=getattr(fig, "spyctres_generated_files", None),
        )
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
