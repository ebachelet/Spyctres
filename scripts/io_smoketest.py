import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from Spyctres.io import read_spectrum, concatenate_segments
from Spyctres.plotting import plot_spectrum_quicklook


def summarize(seg):
    mask = np.asarray(seg.mask, dtype=bool)

    msg = [
        "name={0}".format(seg.name),
        "N={0}".format(len(seg.wave)),
        "N_used={0}".format(int(np.sum(mask))),
        "finite_flux_frac={0:.4f}".format(float(np.mean(np.isfinite(seg.flux)))),
        "medium={0}".format(seg.wave_medium),
        "frame={0}".format(seg.wave_frame),
        "R={0}".format(seg.meta.get("resolution_R")),
    ]

    if np.any(mask):
        w = seg.wave[mask]
        msg.append("wave=[{0:.2f},{1:.2f}]".format(float(w.min()), float(w.max())))

        if seg.err is not None:
            e = seg.err[mask]
            msg.append("median_err={0:.4g}".format(float(np.nanmedian(e))))
        else:
            msg.append("err=None")
    else:
        msg.append("wave=[no good pixels]")
        msg.append("err=None" if seg.err is None else "median_err=nan")

    instrument = str(seg.meta.get("instrument", "")).strip().upper()

    if instrument == "PEPSI":
        msg.append("fiber={0}".format(seg.meta.get("fiber")))
        msg.append("cd={0}".format(seg.meta.get("cross_disperser")))

    if instrument in ["XSHOOTER", "X-SHOOTER"]:
        msg.append("arm={0}".format(seg.meta.get("arm")))
        msg.append("slit={0}".format(seg.meta.get("slit_name")))
        msg.append("telluric_corrected={0}".format(seg.meta.get("telluric_corrected")))
        msg.append("barycorr_kms={0}".format(seg.meta.get("barycorr_kms")))

    return "  ".join(msg)


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "Read one or more reduced 1D spectra with the Spyctres I/O layer and "
            "print a short summary. Use this to confirm that a file is recognized, "
            "its metadata look sensible, and the quick-look plot renders."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/io_smoketest.py --instrument xshooter path/to/xshooter_1d.fits\n"
            "  python scripts/io_smoketest.py --instrument pepsi path/to/seg1.dxt.nor path/to/seg2.dxt.nor path/to/seg3.dxt.nor --join\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main():
    parser = build_parser()
    parser.add_argument(
        "files",
        nargs="+",
        help="Input spectrum file(s)",
    )
    parser.add_argument(
        "--instrument",
        required=True,
        choices=["pepsi", "xshooter"],
        help="Instrument reader to use",
    )
    parser.add_argument(
        "--join",
        action="store_true",
        help="Concatenate segments and print a joined summary",
    )
    args = parser.parse_args()

    missing = [p for p in args.files if not os.path.isfile(p)]
    if missing:
        parser.error("Input file(s) not found: {0}".format(", ".join(missing)))

    segs = []
    for p in args.files:
        s = read_spectrum(p, instrument=args.instrument)
        segs.append(s)
        print(p)
        print(summarize(s))

        fig, ax = plot_spectrum_quicklook(s, use_mask=True, show_error=False)
        plt.show()

    if args.join and len(segs) > 1:
        joined = concatenate_segments(segs, sort=True, name="{0}_joined".format(args.instrument))
        print("\nJOINED")
        print(summarize(joined))


if __name__ == "__main__":
    main()
