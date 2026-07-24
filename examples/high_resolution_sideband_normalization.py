"""Example: local sideband normalization for high-resolution line windows.

This example demonstrates the preprocessing pattern recommended for UVES-like,
PEPSI-like, or other high-resolution line diagnostics: normalize a small line
window from nearby continuum sidebands before doing any local line measurement
or LSF-sensitive comparison.

It is not a PHOENIX atmospheric-parameter fit. The full-spectrum PHOENIX
workflow uses a multiplicative continuum inside the fit; this sideband path is
the complementary local-window tool for detailed line inspection.

Example
-------
python examples/high_resolution_sideband_normalization.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --line-center 4861.33 \
  --wmin 4830 --wmax 4895 \
  --sideband-left -55 -30 \
  --sideband-right 35 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres import ensure_matplotlib_config_dir

ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt

from Spyctres import normalize_segment_sidebands, read_spectrum


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate local sideband normalization for a high-resolution "
            "line window. This is preprocessing/diagnostics, not a PHOENIX fit."
        ),
        epilog=(
            "Example:\n"
            "  python examples/high_resolution_sideband_normalization.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter --line-center 4861.33 "
            "--wmin 4830 --wmax 4895 --no-show"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectrum", help="Input reduced one-dimensional spectrum.")
    parser.add_argument("--instrument", required=True, help="Registered reader name.")
    parser.add_argument("--line-center", type=float, required=True, help="Line center in the data wavelength medium, Angstrom.")
    parser.add_argument("--line-label", default="diagnostic line")
    parser.add_argument("--wmin", type=float, required=True, help="Window minimum in Angstrom.")
    parser.add_argument("--wmax", type=float, required=True, help="Window maximum in Angstrom.")
    parser.add_argument(
        "--sideband-left",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=(-60.0, -30.0),
        help="Left continuum sideband relative to --line-center, Angstrom.",
    )
    parser.add_argument(
        "--sideband-right",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=(30.0, 60.0),
        help="Right continuum sideband relative to --line-center, Angstrom.",
    )
    parser.add_argument("--sideband-order", type=int, default=1)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _coerce_one_segment(spectrum):
    if hasattr(spectrum, "segments"):
        if len(spectrum.segments) != 1:
            raise ValueError(
                "This compact example expects one segment. Pass an individual arm/order."
            )
        return spectrum.segments[0]
    return spectrum


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.wmax <= args.wmin:
        raise ValueError("--wmax must be greater than --wmin.")

    print("Reading spectrum...", flush=True)
    seg = _coerce_one_segment(read_spectrum(args.spectrum, instrument=args.instrument))
    window = seg.window(args.wmin, args.wmax, name=args.line_label)
    if np.sum(window.mask) < 6:
        raise ValueError("Selected line window has too few usable pixels.")

    window.meta["line_center_data"] = float(args.line_center)
    window.meta["cont_windows"] = (
        tuple(float(value) for value in args.sideband_left),
        tuple(float(value) for value in args.sideband_right),
    )
    window.meta["line_label"] = args.line_label

    print("Normalizing local line window with sidebands...", flush=True)
    norm, info = normalize_segment_sidebands(
        window,
        sideband_order=int(args.sideband_order),
    )
    print(
        "Sideband normalization: mode={mode}, sideband_mode={sideband_mode}, n_sideband={n_sideband}".format(
            **info
        ),
        flush=True,
    )

    wave = np.asarray(window.wave, dtype=float)
    good = np.asarray(window.mask, dtype=bool) & np.isfinite(wave)
    sideband_mask = np.zeros_like(good, dtype=bool)
    for lo, hi in window.meta["cont_windows"]:
        sideband_mask |= good & (wave > args.line_center + lo) & (wave < args.line_center + hi)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(13.0, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )
    ax0.plot(window.wave[good], window.flux[good], color="0.25", lw=0.9, label="original")
    ax0.scatter(
        window.wave[sideband_mask],
        window.flux[sideband_mask],
        s=10,
        color="tab:blue",
        alpha=0.8,
        label="sideband pixels",
    )
    ax0.axvline(args.line_center, color="tab:red", ls="--", lw=1.0)
    ax0.set_ylabel("Flux")
    ax0.legend(loc="best")

    ax1.plot(norm.wave[good], norm.flux[good], color="0.15", lw=0.9, label="normalized")
    ax1.axhline(1.0, color="0.65", lw=0.8)
    ax1.axvline(args.line_center, color="tab:red", ls="--", lw=1.0)
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel("Normalized flux")
    ax1.legend(loc="best")
    fig.suptitle(
        "{0}: sideband-normalized local window".format(args.line_label),
        y=0.98,
    )

    if args.output_plot:
        Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_plot, bbox_inches="tight")
        print("Saved plot: {0}".format(args.output_plot), flush=True)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
