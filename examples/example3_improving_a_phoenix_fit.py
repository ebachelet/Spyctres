#!/usr/bin/env python
"""Example 3: improve a PHOENIX fit one assumption at a time.

This example shows how to improve a first pass without blindly expanding the
whole PHOENIX grid.  It builds a reviewed setup, inspects the proposed windows
and mask, then optionally runs assumption-sensitivity fit variants.

What this demonstrates
----------------------
How to make a first-pass fit more deliberate by changing the reviewed setup
and comparing parameter stability, quality flags, and diagnostic plots.

What this does not prove
------------------------
The stronger setup is not automatically "the truth"; it is a better-audited
classification pass that still depends on resolution, masks, continuum choices,
and the PHOENIX model-support regime.  This is still a single-arm UVB example;
multi-arm UVB/VIS/NIR fitting belongs in a later workflow.

Example dry run:

  python examples/example3_improving_a_phoenix_fit.py --no-show

Example opt-in fit comparison:

  python examples/example3_improving_a_phoenix_fit.py \
    --run-fits --R 6200 \
    --output-json /tmp/spyctres_example3.json \
    --output-plot /tmp/spyctres_example3_standard.png \
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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 3: review PHOENIX FitSetup variants, then optionally "
            "run assumption-sensitivity fits."
        ),
        epilog=(
            "Dry run:\n"
            "  python examples/example3_improving_a_phoenix_fit.py --no-show\n\n"
            "Run fits:\n"
            "  python examples/example3_improving_a_phoenix_fit.py --run-fits "
            "--R 6200 --output-json /tmp/spyctres_example3.json "
            "--output-plot /tmp/spyctres_example3_standard.png --no-show\n\n"
            "Next:\n"
            "  python examples/example4_reviewed_balmer_analysis.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectrum", nargs="?", default=str(EXAMPLE_UVB))
    parser.add_argument("--reader", default="xshooter_merge1d")
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--intent",
        default="quicklook_classification",
        help="Readiness intent for reviewed setup suggestions.",
    )
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument("--run-fits", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _progress(event):
    message = event.get("message", str(event)) if isinstance(event, dict) else str(event)
    elapsed = (
        event.get("elapsed_s")
        if isinstance(event, dict)
        else getattr(event, "elapsed_s", None)
    )
    if elapsed is None:
        print(message, flush=True)
    else:
        print("[{0:6.1f}s] {1}".format(float(elapsed), message), flush=True)


def _print_setup_table(setup_rows):
    print("\nReviewed setup variants:", flush=True)
    for label, setup in setup_rows:
        summary = setup.summary()
        print(
            "{0:24s} mode={1:9s} window={2} resolution={3} "
            "continuum={4} ready={5} blockers={6}".format(
                label,
                str(summary.get("mode")),
                summary.get("recommended_window_label"),
                summary.get("resolution_summary"),
                summary.get("continuum_summary"),
                summary.get("ready_for_intent"),
                ", ".join(summary.get("blockers_for_intent") or []) or "none",
            ),
            flush=True,
        )


def _print_fit_interpretation():
    print(
        "\nInterpretation guide:\n"
        "  These variants are assumption-sensitivity checks, not four results "
        "to average.\n"
        "  Similar results across variants support a broad quicklook "
        "classification, while large jumps mean the setup is not stable yet.\n"
        "  Exploratory overrides mean Spyctres computed a diagnostic fit despite "
        "readiness blockers; use these results for diagnosis, not reviewed analysis "
        "inference.\n"
        "  High reduced chi-square means model, continuum, LSF, abundance, "
        "activity, artifact, or error-bar systematics still need review.\n"
        "  For scientific use, continue with Example 4's reviewed-analysis readiness "
        "workflow.",
        flush=True,
    )


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
    selection = sp.select_diagnostic_windows(spec, max_windows=6)
    reviewed_mask = sp.build_mask(
        spec,
        archive="mask",
        tellurics="warn",
        dibs=False,
    )
    selected_region_ids = [item["id"] for item in selection.selected[:3]]
    print("\nSelected diagnostic regions for reviewed fits:", flush=True)
    print(selection.summary_text(max_rows=6), flush=True)
    print(reviewed_mask.summary_text(), flush=True)

    print(
        "\nScope note: these regions are fitted together by fit_stellar_spectrum(); "
        "this is different from Example 2's independent local line fits.",
        flush=True,
    )

    print("Building reviewed setup variants...", flush=True)
    quick_setup = sp.suggest_fit_setup(
        spec,
        mode="quicklook",
        intent=args.intent,
    )
    region_setup = (
        quick_setup.with_regions(selected_region_ids)
        if selected_region_ids
        else quick_setup
    )
    resolution_setup = (
        region_setup.with_resolution(R=args.resolution_R)
        if args.resolution_R is not None
        else region_setup
    )
    continuum_setup = resolution_setup.with_continuum_degree(1)

    standard_setup = sp.suggest_fit_setup(
        spec,
        mode="standard",
        intent=args.intent,
        assumed_resolution=args.resolution_R,
    )
    if selected_region_ids:
        standard_setup = standard_setup.with_regions(selected_region_ids)
    resolution_label = (
        "regions + R" if args.resolution_R is not None else "regions (no explicit R)"
    )
    continuum_label = (
        "regions + R + continuum"
        if args.resolution_R is not None
        else "regions + continuum"
    )
    setup_rows = [
        ("quick", quick_setup),
        ("regions", region_setup),
        (resolution_label, resolution_setup),
        (continuum_label, continuum_setup),
        ("standard search", standard_setup),
    ]
    _print_setup_table(setup_rows)

    payload = {
        "example": "example3_improving_a_phoenix_fit",
        "diagnostic_windows": selection.to_dict(),
        "reviewed_mask": reviewed_mask.to_metadata(),
        "quicklook_setup": quick_setup.to_dict(),
        "variant_setups": {
            "quick": quick_setup.to_dict(),
            "regions": region_setup.to_dict(),
            "regions_plus_R": resolution_setup.to_dict(),
            "regions_plus_R_plus_continuum": continuum_setup.to_dict(),
            "standard_search": standard_setup.to_dict(),
        },
    }

    if args.run_fits:
        variants = [
            ("quick", quick_setup, None),
            ("regions + mask", region_setup, reviewed_mask.valid_mask),
            (continuum_label, continuum_setup, reviewed_mask.valid_mask),
            ("standard search", standard_setup, reviewed_mask.valid_mask),
        ]
        results = []
        labels = []
        for label, setup, valid_mask in variants:
            working_setup = setup
            if not (setup.summary().get("ready_for_intent") is True):
                working_setup = setup.allow_exploratory(
                    reason=(
                        "Example 3 tutorial comparison of one assumption at a "
                        "time; parameters are not interpreted as final science."
                    )
                )
            print("\nRunning variant: {0}".format(label), flush=True)
            result = sp.fit_stellar_spectrum(
                spec,
                model="phoenix",
                setup=working_setup,
                valid_mask=valid_mask,
                phoenix_dir=args.phoenix_dir,
                progress_callback=_progress,
            )
            print(result.summary_text(include_hash=False, max_flags=5), flush=True)
            results.append(result)
            labels.append(label)
        comparison = sp.compare_fits(results, labels=labels)
        print("\nFit progression comparison:", flush=True)
        print(sp.format_fit_comparison_table(comparison), flush=True)
        payload.update(
            {
                "results": {
                    label: result.to_dict(
                        include_arrays=False,
                        include_local_paths=True,
                    )
                    for label, result in zip(labels, results)
                },
                "comparison": comparison,
            }
        )
        print("\nFit progression comparison saved in the JSON payload.", flush=True)
        if args.output_plot:
            fig, _axes = sp.plot_model_line_windows(
                results[-1],
                windows=selection.selected[:3],
                segment=spec,
                savepath=args.output_plot,
                title="Example 3: final variant line-window residual check",
                ncols=2,
                figsize_per_panel=(7.2, 3.7),
            )
            print(
                "Saved final-variant line plot: {0}".format(args.output_plot),
                flush=True,
            )
            if args.no_show:
                import matplotlib.pyplot as plt

                plt.close(fig)
        _print_fit_interpretation()

    if args.output_json:
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)
    if args.run_fits:
        print(
            "\nScope note: Example 3 compares fit setups and fit stability. "
            "Use the quality flags and comparison payload before interpreting "
            "a parameter change as physical.",
            flush=True,
        )
    else:
        print(
            "\nScope note: no PHOENIX fits were run; this reviewed quicklook "
            "and standard setup choices only.",
            flush=True,
        )
    print(
        "Next: run examples/example4_reviewed_balmer_analysis.py for the "
        "audit-first reviewed-analysis scaffold.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
