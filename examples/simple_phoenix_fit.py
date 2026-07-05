"""Minimal command-line example for the public Spyctres PHOENIX API.

Example
-------
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter --teff 6000 --feh 0.0 --logg 4.0 --rv 0.0 \
  --output-json /tmp/spyctres_result.json \
  --output-plot /tmp/spyctres_fit.png
"""

import argparse
import json

import matplotlib.pyplot as plt

from Spyctres import fit_phoenix_spectrum
from Spyctres.io import read_spectrum
from Spyctres.plotting import plot_full_spectrum_fit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Minimal public-API PHOENIX fitting example.",
        epilog=(
            "Example:\n"
            "  python examples/simple_phoenix_fit.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter --teff 6000 --feh 0.0 --logg 4.0 --rv 0.0 "
            "--output-json /tmp/spyctres_result.json "
            "--output-plot /tmp/spyctres_fit.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spectrum", help="Reduced one-dimensional spectrum file.")
    parser.add_argument("--instrument", required=True, help="Registered reader name.")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--teff", type=float, default=5750.0)
    parser.add_argument("--feh", type=float, default=0.0)
    parser.add_argument("--logg", type=float, default=4.5)
    parser.add_argument("--rv", type=float, default=0.0)
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive fit figure (useful for batch runs).",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spectrum = read_spectrum(args.spectrum, instrument=args.instrument)
    result = fit_phoenix_spectrum(
        spectrum,
        phoenix_dir=args.phoenix_dir,
        p0=(args.teff, args.feh, args.logg, args.rv),
        R=args.resolution_R,
    )
    summary = {
        key: result[key]
        for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
    }
    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(result.to_json(indent=2))

    if not result.models:
        raise RuntimeError("Fit did not converge, so no model is available to plot.")
    segment = spectrum if hasattr(spectrum, "wave") else spectrum[0]
    fig, _axes = plot_full_spectrum_fit(
        segment.wave,
        segment.flux,
        err=segment.err,
        model=result.models[0],
        used_mask=result.used_masks[0],
        excluded_mask=result.excluded_masks[0],
        title="PHOENIX fit: {0}".format(segment.name or args.spectrum),
    )
    if args.output_plot:
        fig.savefig(args.output_plot, dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
