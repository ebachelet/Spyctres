import os
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt

from Spyctres._serialization import save_figure
from Spyctres.io import concatenate_segments, list_readers, read_spectrum
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
        
    if instrument == "FLOYDS":
        msg.append("facility={0}".format(seg.meta.get("facility")))
        msg.append("date_obs={0}".format(seg.meta.get("date_obs")))
        
    if instrument in ["GMOS", "GEMINI"]:
        msg.append("object={0}".format(seg.meta.get("object")))
        msg.append("filename={0}".format(seg.meta.get("filename")))
        msg.append("origin={0}".format(seg.meta.get("origin")))
              
    return "  ".join(msg)


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "Setup/ingestion diagnostic for reduced 1D spectra using the Spyctres I/O layer. "
            "print a short summary. Use this to confirm that a file is recognized, "
            "its metadata look sensible, and the quick-look plot renders."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/io_smoketest.py --reader xshooter_merge1d "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits\n"
            "  python scripts/io_smoketest.py --reader pepsi_nor "
            "examples/data/pepsir.20230603.009.dxt.nor "
            "examples/data/pepsir.20230603.010.dxt.nor\n"
            "  python scripts/io_smoketest.py --reader floyds_csv "
            "examples/data/Gaia21ccu_2024_11_23_FLOYDS.csv "
            "--no-show --plot-dir io_plots\n\n"
            "  python scripts/io_smoketest.py --reader gbs_v3_ascii "
            "examples/data/gaia_benchmark/HIP79672_HARPS_1_R42KNorm.txt.gz "
            "--no-show --plot-dir io_plots\n\n"
            "UVES-POP, SDSS, Gemini, and other readers are supported for "
            "user-supplied files, but no runnable example command is shown "
            "unless that spectrum is bundled under examples/data/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )


def _safe_plot_name(path, index):
    stem = os.path.splitext(os.path.basename(path))[0] or "spectrum"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return "{0:02d}_{1}.png".format(index + 1, safe)


def main():
    parser = build_parser()
    parser.add_argument(
        "files",
        nargs="+",
        help="Input spectrum file(s)",
    )
    parser.add_argument(
        "--reader",
        default=None,
        choices=list_readers(include_aliases=True),
        help="Spectrum reader to use.",
    )
    parser.add_argument("--instrument", default=None, choices=list_readers(include_aliases=True), help=argparse.SUPPRESS)
    parser.add_argument(
        "--join",
        action="store_true",
        help="Concatenate segments and print a joined summary",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory where quick-look PNG plots are written.",
    )
    args = parser.parse_args()
    if args.reader is not None and args.instrument is not None:
        parser.error("Pass --reader or --instrument, not both.")
    if args.instrument is not None:
        args.reader = args.instrument
    if args.reader is None:
        parser.error("--reader is required.")

    missing = [p for p in args.files if not os.path.isfile(p)]
    if missing:
        parser.error("Input file(s) not found: {0}".format(", ".join(missing)))

    if args.plot_dir is not None:
        os.makedirs(args.plot_dir, exist_ok=True)

    segs = []
    for index, p in enumerate(args.files):
        s = read_spectrum(p, reader=args.reader)
        segs.append(s)
        print(p)
        print(summarize(s))

        fig, ax = plot_spectrum_quicklook(s, use_mask=True, show_error=False)
        if args.plot_dir is not None:
            out_path = os.path.join(args.plot_dir, _safe_plot_name(p, index))
            save_figure(fig, out_path, dpi=150)
            print("Wrote quick-look plot: {0}".format(out_path))
        if args.no_show:
            plt.close(fig)
        else:
            plt.show()

    if args.join and len(segs) > 1:
        joined = concatenate_segments(segs, sort=True, name="{0}_joined".format(args.reader))
        print("\nJOINED")
        print(summarize(joined))


if __name__ == "__main__":
    main()
