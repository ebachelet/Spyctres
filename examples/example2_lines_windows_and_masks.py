#!/usr/bin/env python
"""Example 2: diagnostic windows, explicit masks, and simple local line fits.

This example introduces one extra concept at a time after Example 1:

1. select advisory diagnostic windows from the loaded wavelength coverage;
2. build an explicit mask/warning bundle without mutating the spectrum;
3. run simple local line fits as diagnostics, not as replacements for PHOENIX.

What this demonstrates
----------------------
How Spyctres records why pixels are warned about or excluded, and how local
line fits can diagnose wavelength/line-shape issues before a full PHOENIX fit.
The Balmer-line examples intentionally show that numerical convergence is not
the same thing as a physically adequate line-profile model.

What this does not prove
------------------------
The local Gaussian line fits are not stellar-parameter classifiers, physical
Balmer-wing models, or hidden automatic arm/continuum corrections.

Example:

  python examples/example2_lines_windows_and_masks.py \
    --output-json /tmp/spyctres_example2.json \
    --output-plot /tmp/spyctres_example2_windows.png \
    --no-show
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


EXAMPLE_UVB = sp.example_data_path("TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits")
DEFAULT_LINES = ("Hdelta", "Hgamma", "Hbeta", "Mg II 4481")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 2: plot diagnostic windows, build explicit masks, and "
            "run simple local diagnostic line fits. No PHOENIX library is required."
        ),
        epilog=(
            "Example:\n"
            "  python examples/example2_lines_windows_and_masks.py "
            "--output-json /tmp/spyctres_example2.json "
            "--output-plot /tmp/spyctres_example2_windows.png --no-show\n\n"
            "Next:\n"
            "  python examples/example3_improving_a_phoenix_fit.py --no-show"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=str(EXAMPLE_UVB),
        help="Spectrum file. Defaults to the bundled X-SHOOTER UVB example.",
    )
    parser.add_argument("--reader", default="xshooter_merge1d")
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--line",
        action="append",
        dest="lines",
        help=(
            "Known line name to fit. May be repeated. Defaults to Hdelta, "
            "Hgamma, Hbeta, and Mg II 4481."
        ),
    )
    parser.add_argument(
        "--mask-dibs",
        action="store_true",
        help="Actually exclude curated optical DIB regions; default is warn only.",
    )
    parser.add_argument(
        "--archive",
        choices=("warn", "mask", "ignore"),
        default="mask",
        help="Archive/product bad-region policy for the mask bundle.",
    )
    parser.add_argument(
        "--tellurics",
        choices=("warn", "fallback", "none"),
        default="warn",
        help="Telluric policy for this lightweight example.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instrument is not None:
        if args.reader != "xshooter_merge1d":
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
    spec = sp.read_spectrum(args.spectrum, reader=args.reader)
    print("Spectrum summary:", flush=True)
    print(spec.summary(), flush=True)

    print("Selecting diagnostic windows...", flush=True)
    windows = sp.select_diagnostic_windows(spec, max_windows=8)
    print(windows.summary_text(max_rows=8), flush=True)
    print(
        "  Diagnostic-window overlays are visual guideposts only; they do not "
        "mask, rescale, or modify the spectrum.",
        flush=True,
    )

    print("Building warning-only mask bundle...", flush=True)
    warnings = sp.build_mask(
        spec,
        archive="warn",
        tellurics="warn",
        dibs=False,
    )
    print(warnings.summary_text(), flush=True)

    print("Building reviewed applied mask bundle...", flush=True)
    reviewed_mask = sp.build_mask(
        spec,
        archive=False if args.archive == "ignore" else args.archive,
        tellurics=False if args.tellurics == "none" else args.tellurics,
        dibs=bool(args.mask_dibs),
    )
    print(reviewed_mask.summary_text(), flush=True)
    if len(reviewed_mask) == 0:
        print(
            "  No named archive/telluric/DIB exclusion masks are active here; "
            "unusable pixels come from the spectrum's own valid-mask and "
            "finite-data checks.",
            flush=True,
        )
    else:
        print(
            "  Active named exclusion masks: {0}".format(
                ", ".join(mask.name for mask in reviewed_mask)
            ),
            flush=True,
        )

    coverage = spec.summary().get("wavelength_range_A") if hasattr(spec, "summary") else None
    if coverage:
        known_lines = sp.list_known_lines(wmin=coverage[0], wmax=coverage[1])
        print(
            "Known local-line names covered by this spectrum: {0}".format(
                ", ".join(known_lines) or "none"
            ),
            flush=True,
        )
    print(
        "Running simple local diagnostic fits with the reviewed valid mask...",
        flush=True,
    )
    print(
        "  fit_line/fit_lines use a simple local profile plus continuum. "
        "success=True means the optimizer converged; check chi2_red and flags "
        "before interpreting the result.",
        flush=True,
    )
    line_names = tuple(args.lines or DEFAULT_LINES)
    line_results = []
    for line_name in line_names:
        try:
            result = sp.fit_line(
                spec,
                line_name,
                valid_mask=reviewed_mask.valid_mask,
            )
        except ValueError as exc:
            print("  skipped {0}: {1}".format(line_name, exc), flush=True)
            continue
        line_results.append(result)
        print("  " + result.summary_text(), flush=True)
    line_comparison = sp.compare_line_fits(
        line_results,
        labels=[result.line_name for result in line_results],
    )
    print(line_comparison.summary_text(), flush=True)
    print(
        "  Balmer lines in this spectrum are broad and can be poor matches to "
        "a simple Gaussian local model. Mg II 4481 is usually the better "
        "representative narrow-feature plot in this tutorial.",
        flush=True,
    )

    target_line_name = "Hgamma" if any(
        result.line_name == "Hgamma" for result in line_results
    ) else (line_results[0].line_name if line_results else None)
    if target_line_name is not None:
        baseline = sp.fit_line(
            spec,
            target_line_name,
            config=sp.LineFitConfig(continuum_order=1),
            valid_mask=reviewed_mask.valid_mask,
        )
        refined = sp.fit_line(
            spec,
            target_line_name,
            config=sp.LineFitConfig(continuum_order=2),
            valid_mask=reviewed_mask.valid_mask,
        )
        continuum_comparison = sp.compare_line_fits(
            [baseline, refined],
            labels=["linear continuum", "quadratic continuum"],
        )
        print("\nOne-line continuum check: {0}".format(target_line_name), flush=True)
        print(continuum_comparison.summary_text(), flush=True)
        print(
            "  If chi2_red remains high after changing continuum order, the "
            "underlying local line-profile model is still inadequate.",
            flush=True,
        )
    else:
        baseline = None
        refined = None
        continuum_comparison = None

    print("Plotting diagnostic windows...", flush=True)
    fig, _ax = sp.plot_diagnostic_windows(
        spec,
        windows,
        mask=reviewed_mask,
        show_nonstellar=True,
        savepath=args.output_plot,
        title="Example 2: reviewed diagnostic windows and applied masks",
    )
    if args.output_plot:
        print("Saved plot: {0}".format(args.output_plot), flush=True)

    if args.output_json:
        payload = {
            "example": "example2_lines_windows_and_masks",
            "diagnostic_windows": windows.to_dict(),
            "warning_mask": warnings.to_metadata(),
            "reviewed_mask": reviewed_mask.to_metadata(),
            "line_fits": [result.to_dict() for result in line_results],
            "line_comparison": line_comparison.to_dict(),
            "continuum_comparison": None
            if continuum_comparison is None
            else continuum_comparison.to_dict(),
        }
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)

    print(
        "\nScope note: Example 2 is a diagnostic-mask and local-line workflow; "
        "it does not replace the PHOENIX full-spectrum fit.",
        flush=True,
    )
    print(
        "Next: run examples/example3_improving_a_phoenix_fit.py to compare "
        "quicklook and stronger reviewed fit setups.",
        flush=True,
    )

    import matplotlib.pyplot as plt

    if args.no_show:
        plt.close(fig)
    else:
        line_lookup = {result.line_name: result for result in line_results}
        sp.plot_line_fit_comparison(
            line_comparison,
            title="Example 2: local line-fit diagnostic metrics",
        )
        if "Mg II 4481" in line_lookup:
            sp.plot_line_fit(line_lookup["Mg II 4481"])
        if "Hgamma" in line_lookup:
            sp.plot_line_fit(line_lookup["Hgamma"])
        if continuum_comparison is not None:
            sp.plot_line_fit_comparison(
                continuum_comparison,
                title="Example 2: local continuum sensitivity",
            )
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
