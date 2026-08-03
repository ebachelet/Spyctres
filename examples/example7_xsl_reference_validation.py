#!/usr/bin/env python
"""Example 7: XSL DR3 reference-star validation.

This numbered example keeps the XSL Figure 1 validation workflow in the
maintained examples path.  It is intentionally conservative by default: it
loads bundled XSL DR3 spectra and a saved coarse-validation result, shows how
Spyctres preserves the XSL product metadata, and optionally renders
observed/model validation panels.  A fresh PHOENIX validation run is available
with ``--run-validation``.

What this demonstrates
----------------------
How to treat XSL DR3 as a product-specific reader, not merely as a generic
X-SHOOTER arm.  XSL DR3 files are merged library products with air wavelengths,
stellar-rest-frame status, and arm/overlap resolution provenance.  Spyctres
keeps that provenance and does not silently rescale arms or apply another RV
correction.

What this does not prove
------------------------
The bundled coarse result is a reproducibility example and review aid, not a
final calibration of Spyctres over the full stellar parameter space.  Ordinary
targets, stress targets, and unsupported PHOENIX cases must be interpreted
separately.

Example inspection run, no PHOENIX library needed:

  python examples/example7_xsl_reference_validation.py --no-show

Render a few bundled validation plots:

  python examples/example7_xsl_reference_validation.py \
    --plot-dir /tmp/spyctres_example7_xsl \
    --max-target-plots 3 \
    --no-show

Fresh validation run, requires PHOENIX:

  python examples/example7_xsl_reference_validation.py \
    --run-validation \
    --output-json /tmp/spyctres_example7_xsl_results.json \
    --plot-dir /tmp/spyctres_example7_xsl \
    --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_DIR.parent
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp


MANIFEST = _EXAMPLE_DIR / "xsl_validation_manifest.csv"
BUNDLED_RESULTS = sp.example_data_path("xsl_figure1_validation_coarse_results.json")
BUNDLED_TARGET = sp.example_data_path("xsl_spectrum_X0245_merged.fits")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 7: inspect bundled XSL DR3 validation products and "
            "optionally rerun the PHOENIX validation runner."
        ),
        epilog=(
            "No-PHOENIX inspection:\n"
            "  python examples/example7_xsl_reference_validation.py --no-show\n\n"
            "Render bundled validation plots:\n"
            "  python examples/example7_xsl_reference_validation.py "
            "--plot-dir /tmp/spyctres_example7_xsl --max-target-plots 3 --no-show\n\n"
            "Fresh PHOENIX validation:\n"
            "  python examples/example7_xsl_reference_validation.py --run-validation "
            "--output-json /tmp/spyctres_example7_xsl_results.json "
            "--plot-dir /tmp/spyctres_example7_xsl --no-show\n\n"
            "Summary-only rendering from a saved validation JSON:\n"
            "  python scripts/xsl_validation_plots.py /tmp/spyctres_example7_xsl_results.json "
            "--no-target-plots "
            "--output-summary-md /tmp/spyctres_xsl_summary.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--reader",
        default="xsl_dr3",
        help="Reader used for inspecting a bundled XSL DR3 target.",
    )
    parser.add_argument(
        "--results-json",
        default=str(BUNDLED_RESULTS),
        help=(
            "Existing validation JSON to inspect when --run-validation is not "
            "set. Defaults to the bundled coarse XSL result."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_example7_xsl_results.json",
        help="Fresh validation output path used with --run-validation.",
    )
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run scripts/xsl_validation.py. Requires a configured PHOENIX library.",
    )
    parser.add_argument(
        "--max-target-plots",
        type=int,
        default=1,
        help="Maximum observed/model validation panels to render. Default: 1.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory for rendered validation PNGs.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("global", "per_segment", "none"),
        default="global",
        help=(
            "Display scaling for validation panels. global keeps one display "
            "scale per star; per_segment is diagnostic only."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create/save plots without opening an interactive Matplotlib window.",
    )
    return parser


def _load_payload(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Expected an XSL validation JSON object with a results list.")
    return payload


def _print_reader_context(reader):
    info = sp.get_reader_info(reader).to_metadata()
    print("Reader:", info["canonical_name"])
    print("  aliases:", ", ".join(info["aliases"]))
    print("  product:", info["expected_file_type"])
    print("  wavelength:", info["wavelength_unit"], info["default_wave_medium"])
    print("  stellar rest status:", info["default_stellar_rest_status"])
    print("  resolution:", info["resolving_power"])


def _print_one_spectrum_summary(reader):
    spec = sp.read_spectrum(BUNDLED_TARGET, reader=reader)
    summary = spec.summary()
    print("\nBundled XSL target inspection:")
    print("  target:", Path(BUNDLED_TARGET).name)
    print("  type:", summary["type"])
    print("  segments:", summary["n_segments"])
    print(
        "  wavelength range: {0:.0f}-{1:.0f} Å".format(
            *summary["wavelength_range_A"]
        )
    )
    print("  wave media:", ", ".join(summary["wave_mediums"]))
    print("  stellar rest:", ", ".join(summary["stellar_rest_status"]))


def _print_validation_table(payload):
    rows = payload.get("results", [])
    status_counts = {}
    for row in rows:
        status_counts[row.get("status", "unknown")] = (
            status_counts.get(row.get("status", "unknown"), 0) + 1
        )
    print("\nSaved validation payload:")
    print("  targets:", len(rows))
    print("  status counts:", status_counts)
    print("  wave-medium assumption:", payload.get("wave_medium_assumption"))
    print("  fit range:", payload.get("fit_wave_range_A"))
    print("\n  ID      role                 status        fit Teff  ref Teff")
    print("  ----------------------------------------------------------------")
    for row in rows:
        fit = row.get("fit") or {}
        ref = row.get("reference") or {}
        fit_teff = fit.get("teff")
        ref_teff = ref.get("teff")
        print(
            "  {0:<7} {1:<20} {2:<13} {3:>8} {4:>8}".format(
                str(row.get("xsl_id", ""))[:7],
                str(row.get("validation_role", ""))[:20],
                str(row.get("status", ""))[:13],
                "—" if fit_teff is None else "{0:.0f}".format(float(fit_teff)),
                "—" if ref_teff is None else "{0:.0f}".format(float(ref_teff)),
            )
        )


def _run_validation(args):
    from scripts import xsl_validation

    command = [
        str(MANIFEST),
        "--output",
        str(args.output_json),
        "--resume",
    ]
    if args.phoenix_dir:
        command.extend(["--phoenix-dir", str(args.phoenix_dir)])
    print("Running XSL validation runner...", flush=True)
    status = xsl_validation.main(command)
    if status not in (None, 0):
        raise RuntimeError("XSL validation runner failed with status {0}.".format(status))
    return Path(args.output_json)


def _render_plots(payload, args):
    import matplotlib.pyplot as plt

    plotted = 0
    plot_dir = Path(args.plot_dir) if args.plot_dir else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    for row in payload.get("results", []):
        plot_data = row.get("validation_plot")
        if not plot_data:
            continue
        title = "{0}: {1} [{2}]".format(
            row.get("xsl_id", "XSL target"),
            row.get("spectral_type", ""),
            row.get("validation_role", "unknown"),
        )
        fig, _axes = sp.plot_xsl_validation_payload(
            plot_data,
            scale_mode=args.scale_mode,
            title=title,
        )
        plotted += 1
        if plot_dir is not None:
            filename = "{0}_{1}.png".format(
                str(row.get("xsl_id", "xsl")).replace("/", "_"),
                str(args.scale_mode),
            )
            fig.savefig(plot_dir / filename, dpi=160)
            print("  wrote", plot_dir / filename)
        if args.no_show:
            plt.close(fig)
        if plotted >= max(0, int(args.max_target_plots)):
            break
    if plotted == 0:
        print("No validation_plot payloads were available to render.")
    elif not args.no_show:
        plt.show()


def main(argv=None):
    args = build_parser().parse_args(argv)
    _print_reader_context(args.reader)
    _print_one_spectrum_summary(args.reader)

    payload_path = _run_validation(args) if args.run_validation else Path(args.results_json)
    payload = _load_payload(payload_path)
    _print_validation_table(payload)

    if args.max_target_plots > 0:
        _render_plots(payload, args)

    print(
        "\nInterpretation: XSL DR3 validation is product-aware. Standard targets "
        "are the ordinary recovery sample; stress/unsupported targets document "
        "the current model domain and should not be mixed into ordinary "
        "accuracy statistics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
