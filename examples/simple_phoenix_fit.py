"""Minimal command-line example for the public Spyctres PHOENIX API."""

import argparse
import json

from Spyctres import fit_phoenix_spectrum
from Spyctres.io import read_spectrum
from Spyctres.plotting import plot_full_spectrum_fit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
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

    if args.output_plot:
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
        fig.savefig(args.output_plot, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
