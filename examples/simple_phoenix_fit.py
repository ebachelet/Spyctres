"""Minimal command-line example for the public Spyctres PHOENIX API.

Purpose
-------
This script demonstrates the shortest complete path from a reduced spectrum to
a structured PHOENIX result and an interactive diagnostic plot. The plot is a
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

import matplotlib.pyplot as plt
import numpy as np

from Spyctres import fit_phoenix_spectrum, suggest_phoenix_fit_defaults
from Spyctres.io import SpectrumCollection, read_spectrum
from Spyctres.plotting import plot_fit_referee


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal public-API full-spectrum PHOENIX demonstration. This is "
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


def _clip_grid(values, lo, hi):
    values = [float(value) for value in values if float(lo) <= float(value) <= float(hi)]
    if values:
        return values
    return [0.5 * (float(lo) + float(hi))]


def _spectrum_wave_range(spectrum):
    segments = list(spectrum.segments) if isinstance(spectrum, SpectrumCollection) else [spectrum]
    waves = []
    for segment in segments:
        wave = np.asarray(segment.wave, dtype=float)
        mask = np.asarray(segment.mask, dtype=bool)
        good = wave[mask & np.isfinite(wave)]
        if good.size:
            waves.append(good)
    if not waves:
        raise ValueError("No finite valid wavelengths are available for window selection.")
    merged = np.concatenate(waves)
    return float(np.min(merged)), float(np.max(merged))


def _fit_kwargs_from_args(args, spectrum):
    suggestion = None
    if args.auto_defaults:
        suggestion = suggest_phoenix_fit_defaults(
            spectrum,
            mode=args.defaults_mode,
            science_case="classification",
        )
        fit_kwargs = dict(suggestion.fit_kwargs)
    else:
        fit_kwargs = {
            "p0": (
                5750.0 if args.teff is None else args.teff,
                0.0 if args.feh is None else args.feh,
                4.5 if args.logg is None else args.logg,
                0.0 if args.rv is None else args.rv,
            ),
            "forward_model": "native_interp",
            "rv_init": "grid",
            "rv_grid_n": 41,
            "mdeg": 2,
        }

    p0 = list(fit_kwargs.get("p0", (5750.0, 0.0, 4.5, 0.0)))
    for index, value in enumerate((args.teff, args.feh, args.logg, args.rv)):
        if value is not None:
            p0[index] = float(value)
    fit_kwargs["p0"] = tuple(p0)

    bounds = fit_kwargs.get(
        "bounds",
        ((4500.0, -1.5, 2.5, -300.0), (10000.0, 0.5, 5.5, 300.0)),
    )
    lower = list(bounds[0])
    upper = list(bounds[1])
    lower_overrides = (args.teff_min, args.feh_min, args.logg_min, args.rv_min)
    upper_overrides = (args.teff_max, args.feh_max, args.logg_max, args.rv_max)
    for index, value in enumerate(lower_overrides):
        if value is not None:
            lower[index] = float(value)
    for index, value in enumerate(upper_overrides):
        if value is not None:
            upper[index] = float(value)
    if any(hi <= lo for lo, hi in zip(lower, upper)):
        raise ValueError("Fit bounds must have min < max for every parameter.")
    fit_kwargs["bounds"] = (tuple(lower), tuple(upper))

    if args.wmin is not None or args.wmax is not None:
        data_lo, data_hi = _spectrum_wave_range(spectrum)
        existing = fit_kwargs.get("regions", [(None, None)])
        base_lo, base_hi = existing[0]
        if base_lo is None:
            base_lo = data_lo
        if base_hi is None:
            base_hi = data_hi
        wmin = float(args.wmin) if args.wmin is not None else float(base_lo)
        wmax = float(args.wmax) if args.wmax is not None else float(base_hi)
        if wmax <= wmin:
            raise ValueError("--wmax must be greater than --wmin.")
        fit_kwargs["regions"] = [(wmin, wmax)]

    if "coarse_teff_grid" in fit_kwargs:
        fit_kwargs["coarse_teff_grid"] = _clip_grid(
            fit_kwargs["coarse_teff_grid"], lower[0], upper[0]
        )
    if "coarse_feh_grid" in fit_kwargs:
        fit_kwargs["coarse_feh_grid"] = _clip_grid(
            fit_kwargs["coarse_feh_grid"], lower[1], upper[1]
        )
    if "coarse_logg_grid" in fit_kwargs:
        fit_kwargs["coarse_logg_grid"] = _clip_grid(
            fit_kwargs["coarse_logg_grid"], lower[2], upper[2]
        )

    if args.resolution_R is not None:
        fit_kwargs["R"] = float(args.resolution_R)
    return fit_kwargs, suggestion


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
    print("Running PHOENIX fit...", flush=True)
    result = fit_phoenix_spectrum(
        spectrum,
        phoenix_dir=args.phoenix_dir,
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
