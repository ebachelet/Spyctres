#!/usr/bin/env python
"""Example 4A: reviewed-analysis Balmer preparation and baseline analysis.

This is the concise recipe-led reviewed-analysis example. It uses the maintained
X-SHOOTER UVB Balmer preparation recipe, prints exactly what the recipe did,
builds a reviewed-analysis setup, and optionally runs one joint PHOENIX fit
plus an individual-line consistency check.

What this demonstrates
----------------------
How a tested Spyctres recipe supports a reproducible, inspectable scientific
workflow without asking every user to reimplement wavelength conversion,
sideband normalization, segment construction, and mask-callable bookkeeping.

What this does not prove
------------------------
A successful baseline fit is not automatically analysis-ready. Final scientific
use still needs residual inspection plus systematic checks
of normalization, continuum degree, core-mask width, resolution/LSF assumptions,
line selection, and external validation where available.

Example 4A review-only run:

  python examples/example4_reviewed_balmer_analysis.py --no-show

Example 4A baseline fit:

  python examples/example4_reviewed_balmer_analysis.py \
    --run-level fit \
    --allow-exploratory-fit \
    --override-reason "tutorial residual review; not final analysis" \
    --output-json /tmp/spyctres_example4.json \
    --output-plot /tmp/spyctres_example4.png \
    --no-show

Example 4A line-consistency run:

  python examples/example4_reviewed_balmer_analysis.py \
    --run-level line_consistency \
    --allow-exploratory-fit \
    --override-reason "tutorial line-consistency review; not final analysis" \
    --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp
from Spyctres._serialization import atomic_write_json


EXAMPLE_UVB = sp.example_data_path("TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 4A: reviewed-analysis X-SHOOTER UVB/Balmer analysis "
            "using a transparent recipe-led workflow."
        ),
        epilog=(
            "Review only:\n"
            "  python examples/example4_reviewed_balmer_analysis.py --no-show\n\n"
            "Baseline fit:\n"
            "  python examples/example4_reviewed_balmer_analysis.py "
            "--run-level fit --allow-exploratory-fit "
            "--override-reason 'tutorial residual review; not final analysis' "
            "--output-json /tmp/spyctres_example4.json "
            "--output-plot /tmp/spyctres_example4.png --no-show\n\n"
            "Line consistency:\n"
            "  python examples/example4_reviewed_balmer_analysis.py "
            "--run-level line_consistency --allow-exploratory-fit "
            "--override-reason 'tutorial line-consistency review; not final analysis' "
            "--no-show\n\n"
            "Next:\n"
            "  python examples/example4b_balmer_stability_checks.py --help\n"
            "  python examples/example6_multiarm_classification.py --help\n"
            "  python examples/reviewed_xshooter_uvb_analysis.py --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectrum", nargs="?", default=str(EXAMPLE_UVB))
    parser.add_argument("--reader", default="xshooter_merge1d")
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--run-level",
        choices=("review", "fit", "line_consistency"),
        default="review",
        help=(
            "review=inspect recipe/setup only; fit=run one joint PHOENIX fit; "
            "line_consistency=also fit prepared Balmer lines individually."
        ),
    )
    parser.add_argument(
        "--run-fit",
        action="store_true",
        help="Compatibility shortcut for --run-level fit.",
    )
    parser.add_argument(
        "--run-baseline-fit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-line-consistency",
        action="store_true",
        help="Compatibility shortcut for --run-level line_consistency.",
    )
    parser.add_argument(
        "--allow-exploratory-fit",
        action="store_true",
        help=(
            "Allow the optional fit when reviewed-analysis readiness is blocked. "
            "Requires --override-reason and records the result as exploratory."
        ),
    )
    parser.add_argument(
        "--override-reason",
        default=None,
        help="Required reason when --allow-exploratory-fit is supplied.",
    )
    parser.add_argument("--balmer-core-mask", type=float, default=3.0)
    parser.add_argument("--continuum-degree", type=int, default=1)
    parser.add_argument("--sideband-width", type=float, default=10.0)
    parser.add_argument("--sideband-order", type=int, default=1)
    parser.add_argument(
        "--extra-artifact-region",
        action="append",
        nargs=2,
        metavar=("WMIN", "WMAX"),
        type=float,
        default=[],
        help=(
            "Optional user-reviewed wavelength interval to exclude in addition "
            "to reader quality masks and recipe masks. Repeat for multiple "
            "intervals; do not use this to duplicate pixels already rejected "
            "by the reader quality mask."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional compact JSON summary of recipe/setup/result metadata.",
    )
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument(
        "--record-input-checksum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When writing a fit report, record a SHA256 checksum of the input "
            "file bytes. Off by default."
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def _progress(event):
    elapsed = getattr(event, "elapsed_s", None)
    if elapsed is None and isinstance(event, dict):
        elapsed = event.get("elapsed_s")
    message = event.get("message", str(event)) if isinstance(event, dict) else str(event)
    if elapsed is None:
        print(message, flush=True)
    else:
        print("[{0:6.1f}s] {1}".format(float(elapsed), message), flush=True)


def _resolve_run_level(args):
    if args.run_line_consistency:
        return "line_consistency"
    if args.run_fit or args.run_baseline_fit:
        return "fit"
    return args.run_level


def _show_or_close(fig, no_show):
    if no_show:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _require_or_apply_exploratory_override(setup, args):
    if setup.summary()["ready_for_intent"] is True:
        return setup
    if not args.allow_exploratory_fit:
        raise ValueError(
            "Reviewed-analysis readiness is blocked. Re-run with "
            "--allow-exploratory-fit --override-reason '...' if you want a "
            "diagnostic tutorial fit, or resolve the blockers first."
        )
    reason = "" if args.override_reason is None else str(args.override_reason).strip()
    if not reason:
        raise ValueError("--allow-exploratory-fit requires a non-empty --override-reason.")
    return setup.allow_exploratory(reason=reason)


def _manual_artifact_masks(regions):
    regions = tuple((float(wmin), float(wmax)) for wmin, wmax in (regions or ()))
    if not regions:
        return ()

    return (
        sp.wavelength_region_exclusion_mask(
            "manually_reviewed_artifacts",
            regions,
            metadata={
                "reason": "Regions identified during visual inspection",
                "action": "masked",
            },
        ),
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instrument is not None:
        if args.reader != "xshooter_merge1d":
            raise ValueError("Pass --reader or --instrument, not both.")
        args.reader = args.instrument
    run_level = _resolve_run_level(args)

    print("Reading X-SHOOTER UVB spectrum...", flush=True)
    spec = sp.read_spectrum(args.spectrum, reader=args.reader)
    print(spec.summary(), flush=True)

    print("\nInspecting candidate windows and mask warnings...", flush=True)
    windows = sp.select_diagnostic_windows(spec, max_windows=6)
    print(windows.summary_text(max_rows=6), flush=True)
    reviewed_mask = sp.build_mask(
        spec,
        archive="mask",
        tellurics="warn",
        dibs=False,
    )
    print(reviewed_mask.summary_text(), flush=True)

    fig, _axes = sp.plot_spectrum_audit(
        spec,
        diagnostic_selection=windows,
        warning_regions=reviewed_mask.warning_regions,
        title="Example 4A: audit view before Balmer fitting",
        figsize=(14.0, 8.0),
    )
    _show_or_close(fig, args.no_show)

    print("\nPreparing transparent Balmer recipe case...", flush=True)
    balmer_case = sp.recipes.prepare_xshooter_balmer_case(
        spec,
        window_mode="notebook",
        norm_mode="sideband",
        sideband_width=args.sideband_width,
        sideband_order=args.sideband_order,
        core_mask=args.balmer_core_mask,
    )
    print(balmer_case.summary_text(), flush=True)
    fig, _axes = balmer_case.plot_preparation(
        title="Example 4A: what the Balmer recipe prepared",
        ncols=2,
        figsize_per_panel=(7.2, 3.6),
    )
    _show_or_close(fig, args.no_show)

    extra_masks = _manual_artifact_masks(args.extra_artifact_region)
    all_exclusion_masks = balmer_case.combined_exclusion_masks(extra_masks)
    fit_valid_masks = balmer_case.valid_masks_for(extra_masks)
    if extra_masks:
        print(
            "\nUsing extra user-reviewed artifact regions: {0}".format(
                args.extra_artifact_region
            ),
            flush=True,
        )
    else:
        print(
            "\nNo extra visual-inspection artifact regions were supplied; "
            "using reader quality masks plus the recipe Balmer-core mask.",
            flush=True,
        )

    print("\nBuilding reviewed-analysis setup...", flush=True)
    setup = balmer_case.suggest_fit_setup(
        mode="standard",
        intent="reviewed_analysis",
        continuum_degree=args.continuum_degree,
        extra_exclusion_masks=extra_masks,
    )
    reviewed_analysis_audit = sp.analysis_readiness_audit(
        balmer_case.collection,
        regions=balmer_case.fit_regions_by_segment,
        exclude_masks=all_exclusion_masks,
    )
    print(setup.summary_text(include_hash=False), flush=True)
    print("\nReviewed-analysis ready:", reviewed_analysis_audit["analysis_ready"], flush=True)
    print("Reviewed-analysis blockers:", reviewed_analysis_audit["blockers"], flush=True)
    print("Reviewed-analysis warnings:", reviewed_analysis_audit["warnings"], flush=True)
    print("\nSegment-level artifact accounting:", flush=True)
    for segment_audit in reviewed_analysis_audit["audit"]["segments"]:
        metrics = segment_audit["artifact_metrics"]
        print("  {0}".format(segment_audit["name"]), flush=True)
        print(
            "    unhandled artifacts inside fit window: {0:.2%}".format(
                float(metrics["unhandled_artifact_fraction_inside_fit_window"])
            ),
            flush=True,
        )
        print(
            "    already rejected by reader/input mask: {0:.2%}".format(
                float(metrics["already_rejected_input_mask_fraction_inside_fit_window"])
            ),
            flush=True,
        )
        print(
            "    flags: {0}".format(segment_audit["interpretation_flags"]),
            flush=True,
        )
    if extra_masks:
        print(
            "Response: extra user-reviewed artifact masks are active; inspect "
            "the residuals after fitting.",
            flush=True,
        )
    elif "artifact_review_required" in reviewed_analysis_audit["blockers"]:
        print(
            "Response: inspect red x pixels and residuals before adding any "
            "new wavelength mask; do not duplicate reader-quality rejections.",
            flush=True,
        )
    else:
        print(
            "Response: no unhandled artifact blocker remains; still check Hδ "
            "line consistency before treating the result as robust.",
            flush=True,
        )

    result = None
    line_results = None
    comparison = None
    if run_level in {"fit", "line_consistency"}:
        run_setup = _require_or_apply_exploratory_override(setup, args)
        print("\nRunning one joint Balmer PHOENIX fit...", flush=True)
        result = sp.fit_stellar_spectrum(
            balmer_case.collection,
            model="phoenix",
            setup=run_setup,
            valid_mask=fit_valid_masks,
            phoenix_dir=args.phoenix_dir,
            progress_callback=_progress,
        )
        print(result.summary_text(include_hash=False, max_flags=8), flush=True)
        print("\n" + result.quality_report_text(), flush=True)

        fig, _axes = sp.plot_model_line_windows(
            result,
            windows=balmer_case.fit_windows,
            title="Example 4A: joint Balmer fit, shown line by line",
            show_residuals=True,
            residual_kind="pull",
            ncols=2,
            figsize_per_panel=(7.2, 5.2),
            savepath=args.output_plot,
        )
        _show_or_close(fig, args.no_show)

        if args.output_report_json:
            result.save_report_json(
                args.output_report_json,
                record_input_checksum=args.record_input_checksum,
            )
            print("Wrote fit report JSON: {0}".format(args.output_report_json), flush=True)

        if run_level == "line_consistency":
            print("\nRunning individual Balmer-line consistency fits...", flush=True)
            line_results = sp.fit_case_lines_individually(
                balmer_case,
                base_setup=run_setup,
                model="phoenix",
                phoenix_dir=args.phoenix_dir,
                progress_callback=_progress,
                extra_exclusion_masks=extra_masks,
            )
            for label, line_result in line_results.items():
                print("\n" + label, flush=True)
                print(line_result.summary_text(include_hash=False, max_flags=5), flush=True)
            comparison = sp.compare_fits(
                list(line_results.values()),
                labels=list(line_results.keys()),
            )
            print("\nLine-consistency comparison:", flush=True)
            print(sp.format_fit_comparison_table(comparison), flush=True)
            print(
                "\nHow to read this comparison: the rows are individual "
                "Balmer-line fits compared with the first fitted line. Large "
                "Teff/logg/[Fe/H]/RV shifts mean the joint Balmer result is "
                "sensitive to line choice and needs the Example 4B stability "
                "checks before interpretation.",
                flush=True,
            )

            # Overlay the joint fit and the individual-line fits in the same
            # Balmer windows.  Each single-line model appears only in the line
            # region that actually constrained it; dashed parts are model
            # values on pixels that were deliberately masked.
            fig, _axes = sp.plot_fit_comparison_line_windows(
                [result, *line_results.values()],
                labels=["joint Balmer fit", *line_results.keys()],
                windows=balmer_case.fit_windows,
                title="Example 4A: joint fit compared with individual-line fits",
                ncols=2,
                figsize_per_panel=(7.2, 3.8),
                footer=(
                    "Solid model traces use fitted pixels. Dashed model spans "
                    "show masked pixels for context only."
                ),
            )
            _show_or_close(fig, args.no_show)
    else:
        print(
            "\nReview-only run complete. Use --run-level fit for one baseline "
            "PHOENIX fit, or --run-level line_consistency for individual "
            "Balmer-line checks.",
            flush=True,
        )

    if args.output_json:
        payload = {
            "example": "example4a_reviewed_balmer_analysis",
            "run_level": run_level,
            "spectrum": str(args.spectrum),
            "reader": args.reader,
            "recipe": balmer_case.to_dict(),
            "setup": setup.to_dict(),
            "analysis_readiness": reviewed_analysis_audit,
            "result": None
            if result is None
            else result.to_dict(include_arrays=False, include_local_paths=True),
            "line_results": None
            if line_results is None
            else {
                label: line_result.to_dict(
                    include_arrays=False,
                    include_local_paths=True,
                )
                for label, line_result in line_results.items()
            },
            "line_comparison": comparison,
        }
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)

    print(
        "\nEvidence gathered here: the spectrum was read with explicit product "
        "metadata, Balmer windows were prepared by a maintained recipe, the "
        "readiness audit was recorded, and the optional joint fit can be "
        "inspected line by line. Evidence still missing: bounded stability "
        "checks, multi-arm consistency, and external validation before treating "
        "the parameters as reviewed-analysis results.",
        flush=True,
    )
    print(
        "Next: run examples/example4b_balmer_stability_checks.py for "
        "bounded follow-up variants, then Example 6 for UVB/VIS/NIR consistency.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
