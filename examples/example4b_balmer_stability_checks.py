#!/usr/bin/env python
"""Example 4B: follow-up stability checks for the Example 4A spectrum.

Example 4A prepares and audits a reviewed-analysis X-SHOOTER UVB Balmer
fit for Gaia21ccu.  This follow-up keeps the same spectrum and demonstrates
the next review layer mentioned at the end of 4A: continuum-degree, Balmer-core
mask, line-selection, and resolution/LSF sensitivity checks.

Gaia21ccu is deliberately a complex teaching case.  It is useful here because
it forces the workflow to separate "we can compute a fit" from "we can interpret
that fit scientifically."  Use Example 1 or the Gaia benchmark validation
helpers for a cleaner first-success benchmark-star spectrum.

What this demonstrates
----------------------
How to treat a baseline fit as the first evidence layer, not as the final
answer.  Each variant changes one reviewed-analysis choice and compares the
result back to the baseline.  Large parameter shifts are warning signs that the
result depends on preprocessing/model choices.

What this does not prove
------------------------
This compact example is not a full reviewed-analysis pipeline.  It does not run a
posterior sampler, empirical LSF fitting, rotational broadening, external
reference-star calibration, or the larger advanced scaffold.  It is meant to
teach the review pattern in a bounded way.

Review-only run, no PHOENIX needed:

  python examples/example4b_balmer_stability_checks.py --no-show

Baseline fit:

  python examples/example4b_balmer_stability_checks.py \
    --run-level baseline \
    --allow-exploratory-fit \
    --override-reason "tutorial stability baseline; not final analysis" \
    --output-json /tmp/spyctres_example4b.json \
    --plot-dir /tmp/spyctres_example4b \
    --no-show

Bounded stability run:

  python examples/example4b_balmer_stability_checks.py \
    --run-level stability \
    --allow-exploratory-fit \
    --override-reason "tutorial stability variants; not final analysis" \
    --output-json /tmp/spyctres_example4b_stability.json \
    --plot-dir /tmp/spyctres_example4b \
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

BASELINE_CORE_MASK_A = 8.0
BASELINE_CONTINUUM_DEGREE = 1
BASELINE_SIDEBAND_ORDER = 1
BASELINE_SIDEBAND_WIDTH_A = 10.0
BASELINE_WINDOW_MODE = "notebook"
BASELINE_NORM_MODE = "sideband"

DEFAULT_VARIANT_IDS = (
    "continuum_degree_0",
    "continuum_degree_2",
    "core_mask_4A",
    "core_mask_10A",
    "without_hdelta",
    "without_hbeta",
    "resolution_R5000",
    "resolution_R6200",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 4B: follow-up stability checks for the same "
            "Gaia21ccu X-SHOOTER UVB Balmer case prepared in Example 4A."
        ),
        epilog=(
            "Review the plan, no PHOENIX needed:\n"
            "  python examples/example4b_balmer_stability_checks.py --no-show\n\n"
            "Run only the baseline fit:\n"
            "  python examples/example4b_balmer_stability_checks.py "
            "--run-level baseline --allow-exploratory-fit "
            "--override-reason 'tutorial stability baseline; not final analysis' "
            "--plot-dir /tmp/spyctres_example4b --no-show\n\n"
            "Run a bounded variant suite:\n"
            "  python examples/example4b_balmer_stability_checks.py "
            "--run-level stability --allow-exploratory-fit "
            "--override-reason 'tutorial stability variants; not final analysis' "
            "--output-json /tmp/spyctres_example4b_stability.json "
            "--plot-dir /tmp/spyctres_example4b --no-show\n\n"
            "Quick subset while editing:\n"
            "  add --max-variants 3\n\n"
            "Next:\n"
            "  python examples/example6_multiarm_classification.py --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectrum", nargs="?", default=str(EXAMPLE_UVB))
    parser.add_argument("--reader", default="xshooter_merge1d")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--run-level",
        choices=("review", "baseline", "stability"),
        default="review",
        help="review=no PHOENIX; baseline=one fit; stability=baseline plus variants.",
    )
    parser.add_argument(
        "--variant-ids",
        default=",".join(DEFAULT_VARIANT_IDS),
        help="Comma-separated stability variants to run with --run-level stability.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help=(
            "Maximum number of requested variants to run. By default all "
            "requested variants are run; use e.g. --max-variants 3 for a quick "
            "editing pass."
        ),
    )
    parser.add_argument(
        "--allow-exploratory-fit",
        action="store_true",
        help=(
            "Allow fitting if the readiness audit is blocked. Requires "
            "--override-reason and records the fit as exploratory."
        ),
    )
    parser.add_argument("--override-reason", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def _split_csv(value):
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _savepath(plot_dir, filename):
    if not plot_dir:
        return None
    path = Path(plot_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _show_or_close(no_show):
    import matplotlib.pyplot as plt

    if no_show:
        plt.close("all")
    else:
        plt.show()


def _progress(event):
    elapsed = getattr(event, "elapsed_s", None)
    message = str(event)
    if elapsed is None:
        print(message, flush=True)
    else:
        print("[{0:6.1f}s] {1}".format(float(elapsed), message), flush=True)


def _prepare_case(spec, *, core_mask=BASELINE_CORE_MASK_A):
    return sp.recipes.prepare_xshooter_balmer_case(
        spec,
        window_mode=BASELINE_WINDOW_MODE,
        norm_mode=BASELINE_NORM_MODE,
        sideband_width=BASELINE_SIDEBAND_WIDTH_A,
        sideband_order=BASELINE_SIDEBAND_ORDER,
        core_mask=core_mask,
    )


def _setup_for_case(case, *, continuum_degree=BASELINE_CONTINUUM_DEGREE, R=None):
    setup = case.suggest_fit_setup(
        mode="standard",
        intent="reviewed_analysis",
        continuum_degree=continuum_degree,
    )
    if R is not None:
        setup = setup.with_resolution(R=float(R))
    return setup


def _line_subset(case, labels):
    wanted = {str(label) for label in labels}
    segments = []
    masks = []
    windows = []
    regions = []
    for segment, valid_mask, window, region in zip(
        case.fit_segments,
        case.valid_masks,
        case.fit_windows,
        case.fit_regions,
    ):
        if str(segment.name) not in wanted:
            continue
        segments.append(segment)
        masks.append(valid_mask)
        windows.append(window)
        regions.append(region)
    if not segments:
        raise ValueError("Line subset retained no segments.")
    return (
        sp.SpectrumCollection(
            segments,
            name="example4b_" + "_".join(str(segment.name) for segment in segments),
            meta={"workflow": "example4b_line_selection_variant"},
        ),
        tuple(masks),
        tuple(windows),
        tuple(regions),
    )


def _fit_variant(
    *,
    label,
    collection,
    setup,
    valid_masks,
    phoenix_dir,
    show_progress,
):
    print("\nRunning {0}...".format(label), flush=True)
    result = sp.fit_stellar_spectrum(
        collection,
        model="phoenix",
        setup=setup,
        valid_mask=valid_masks,
        phoenix_dir=phoenix_dir,
        progress_callback=_progress if show_progress else None,
    )
    print(result.summary_text(include_hash=False, max_flags=6), flush=True)
    return result


def _apply_override_if_needed(setup, audit, args):
    if audit["analysis_ready"] is True:
        return setup
    if not args.allow_exploratory_fit:
        raise ValueError(
            "Reviewed-analysis readiness is blocked. Re-run with "
            "--allow-exploratory-fit --override-reason '...' for a diagnostic "
            "follow-up fit, or resolve the blockers first."
        )
    reason = str(args.override_reason or "").strip()
    if not reason:
        raise ValueError("--allow-exploratory-fit requires a non-empty --override-reason.")
    return setup.allow_exploratory(reason=reason)


def _variant_plan(spec):
    baseline_case = _prepare_case(spec, core_mask=BASELINE_CORE_MASK_A)
    hgamma_hbeta = _line_subset(baseline_case, ("Hγ", "Hβ"))
    hdelta_hgamma = _line_subset(baseline_case, ("Hδ", "Hγ"))

    return {
        "continuum_degree_0": {
            "label": "continuum degree 0",
            "case": baseline_case,
            "collection": baseline_case.collection,
            "valid_masks": baseline_case.valid_masks,
            "windows": baseline_case.fit_windows,
            "setup": _setup_for_case(baseline_case, continuum_degree=0),
            "question": "Does the answer depend strongly on residual continuum flexibility?",
        },
        "continuum_degree_2": {
            "label": "continuum degree 2",
            "case": baseline_case,
            "collection": baseline_case.collection,
            "valid_masks": baseline_case.valid_masks,
            "windows": baseline_case.fit_windows,
            "setup": _setup_for_case(baseline_case, continuum_degree=2),
            "question": "Can a more flexible residual continuum absorb Balmer-wing information?",
        },
        "core_mask_4A": {
            "label": "Balmer core mask ±4 A",
            "case": _prepare_case(spec, core_mask=4.0),
            "question": "Do parameters shift when more of the line core is fitted?",
        },
        "core_mask_10A": {
            "label": "Balmer core mask ±10 A",
            "case": _prepare_case(spec, core_mask=10.0),
            "question": "Do parameters shift when more core/inner-wing data are removed?",
        },
        "without_hdelta": {
            "label": "without Hδ",
            "collection": hgamma_hbeta[0],
            "valid_masks": hgamma_hbeta[1],
            "windows": hgamma_hbeta[2],
            "setup": _setup_for_case(baseline_case).with_regions(hgamma_hbeta[3]),
            "question": "Is the result robust to removing the line with the largest reader-rejected interval?",
        },
        "without_hbeta": {
            "label": "without Hβ",
            "collection": hdelta_hgamma[0],
            "valid_masks": hdelta_hgamma[1],
            "windows": hdelta_hgamma[2],
            "setup": _setup_for_case(baseline_case).with_regions(hdelta_hgamma[3]),
            "question": "Is the result robust to removing the strongest available Balmer line?",
        },
        "resolution_R5000": {
            "label": "assumed R=5000",
            "case": baseline_case,
            "collection": baseline_case.collection,
            "valid_masks": baseline_case.valid_masks,
            "windows": baseline_case.fit_windows,
            "setup": _setup_for_case(baseline_case, R=5000.0),
            "question": "Does a lower constant-R LSF assumption shift the answer?",
        },
        "resolution_R6200": {
            "label": "assumed R=6200",
            "case": baseline_case,
            "collection": baseline_case.collection,
            "valid_masks": baseline_case.valid_masks,
            "windows": baseline_case.fit_windows,
            "setup": _setup_for_case(baseline_case, R=6200.0),
            "question": "Does a higher constant-R LSF assumption shift the answer?",
        },
    }


def _complete_variant_spec(variant):
    if "case" in variant:
        case = variant["case"]
        variant = dict(variant)
        variant.setdefault("collection", case.collection)
        variant.setdefault("valid_masks", case.valid_masks)
        variant.setdefault("windows", case.fit_windows)
        variant.setdefault("setup", _setup_for_case(case))
    return variant


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_variants is not None and args.max_variants < 0:
        raise ValueError("--max-variants must be >= 0.")

    print("Reading the same X-SHOOTER UVB spectrum as Example 4A...", flush=True)
    spec = sp.read_spectrum(args.spectrum, reader=args.reader)
    print(spec.summary(), flush=True)

    print("\nRebuilding the baseline Balmer case from Example 4A...", flush=True)
    baseline_case = _prepare_case(spec, core_mask=BASELINE_CORE_MASK_A)
    baseline_setup = _setup_for_case(baseline_case)
    baseline_audit = sp.analysis_readiness_audit(
        baseline_case.collection,
        regions=baseline_case.fit_regions_by_segment,
        exclude_masks=baseline_case.exclusion_masks,
    )
    print(baseline_case.summary_text(), flush=True)
    print(baseline_setup.summary_text(include_hash=False), flush=True)
    print("Reviewed-analysis ready:", baseline_audit["analysis_ready"], flush=True)
    print("Reviewed-analysis blockers:", baseline_audit["blockers"], flush=True)
    print("Reviewed-analysis warnings:", baseline_audit["warnings"], flush=True)

    print("\nPlanned follow-up checks:", flush=True)
    plan = _variant_plan(spec)
    for variant_id in DEFAULT_VARIANT_IDS:
        variant = _complete_variant_spec(plan[variant_id])
        print(
            "  {0}: {1} — {2}".format(
                variant_id,
                variant["label"],
                variant["question"],
            ),
            flush=True,
        )

    baseline_case.plot_preparation(
        title="Example 4B: baseline Balmer preparation from Example 4A",
        ncols=2,
        figsize_per_panel=(7.2, 3.6),
        savepath=_savepath(args.plot_dir, "example4b_baseline_preparation.png"),
    )
    _show_or_close(args.no_show)

    baseline_result = None
    variant_results = []
    comparison = None
    candidate_feature_matches = []
    known_feature_overlap = None
    known_residual_windows = None
    if args.run_level in {"baseline", "stability"}:
        run_setup = _apply_override_if_needed(baseline_setup, baseline_audit, args)
        baseline_result = _fit_variant(
            label="baseline",
            collection=baseline_case.collection,
            setup=run_setup,
            valid_masks=baseline_case.valid_masks,
            phoenix_dir=args.phoenix_dir,
            show_progress=args.show_progress,
        )
        sp.plot_model_line_windows(
            baseline_result,
            windows=baseline_case.fit_windows,
            title="Example 4B: baseline fit in Balmer windows",
            show_residuals=True,
            residual_kind="pull",
            ncols=2,
            figsize_per_panel=(7.2, 5.2),
            savepath=_savepath(args.plot_dir, "example4b_baseline_windows.png"),
        )
        _show_or_close(args.no_show)

    if args.run_level == "stability":
        requested_all = _split_csv(args.variant_ids)
        requested = (
            requested_all
            if args.max_variants is None
            else requested_all[: args.max_variants]
        )
        omitted = requested_all[len(requested):]
        print(
            "\nRunning stability variants ({0}/{1} requested): {2}".format(
                len(requested),
                len(requested_all),
                ", ".join(requested),
            ),
            flush=True,
        )
        if omitted:
            print(
                "Omitted by --max-variants: {0}".format(", ".join(omitted)),
                flush=True,
            )
        for variant_id in requested:
            if variant_id not in plan:
                known = ", ".join(sorted(plan))
                raise ValueError(
                    "Unknown variant {0!r}. Known: {1}.".format(variant_id, known)
                )
            variant = _complete_variant_spec(plan[variant_id])
            variant_setup = _apply_override_if_needed(
                variant["setup"],
                baseline_audit,
                args,
            )
            result = _fit_variant(
                label=variant["label"],
                collection=variant["collection"],
                setup=variant_setup,
                valid_masks=variant["valid_masks"],
                phoenix_dir=args.phoenix_dir,
                show_progress=args.show_progress,
            )
            variant_results.append(
                {
                    "id": variant_id,
                    "label": variant["label"],
                    "question": variant["question"],
                    "result": result,
                    "windows": variant["windows"],
                }
            )

        if baseline_result is not None and variant_results:
            comparison = sp.compare_fits(
                [baseline_result] + [item["result"] for item in variant_results],
                labels=["baseline"] + [item["label"] for item in variant_results],
            )
            print("\nStability comparison:", flush=True)
            print(sp.format_fit_comparison_table(comparison), flush=True)
            print(
                "\nHow to read this table: each row changes one analysis "
                "choice relative to the baseline. Stable parameters across "
                "rows are encouraging; large shifts identify the assumption "
                "that needs review. The status and grid/bounds columns are "
                "as important as chi-square.",
                flush=True,
            )

            sp.plot_fit_comparison_line_windows(
                [baseline_result] + [item["result"] for item in variant_results],
                labels=["baseline"] + [item["label"] for item in variant_results],
                windows=baseline_case.fit_windows,
                title="Example 4B: baseline and stability variants overplotted",
                ncols=2,
                figsize_per_panel=(7.2, 3.8),
                footer=(
                    "Large separations between traces show sensitivity to "
                    "reasonable analysis choices."
                ),
                savepath=_savepath(
                    args.plot_dir,
                    "example4b_variant_comparison_windows.png",
                ),
            )
            _show_or_close(args.no_show)

            sp.plot_model_line_windows(
                variant_results[-1]["result"],
                windows=variant_results[-1]["windows"],
                title="Example 4B: last stability variant in Balmer windows",
                show_residuals=True,
                residual_kind="pull",
                ncols=2,
                figsize_per_panel=(7.2, 5.2),
                savepath=_savepath(args.plot_dir, "example4b_last_variant_windows.png"),
            )
            _show_or_close(args.no_show)

    if baseline_result is not None:
        print("\nTriage user-noticed residual structures...", flush=True)
        suspicious_residual_regions = [
            {
                "label": "unexplained dip near Hgamma red wing",
                "region_A": (4415.0, 4445.0),
            },
            {
                "label": "unexplained Hbeta red-wing drop",
                "region_A": (4875.0, 4910.0),
            },
        ]
        candidate_feature_matches = sp.find_known_nonstellar_features(
            suspicious_residual_regions,
            padding_A=0.0,
        )
        if candidate_feature_matches:
            print("Candidate catalog overlaps:", flush=True)
            for match in candidate_feature_matches:
                print(
                    "  {0}: {1} ({2}), catalog region={3}, overlap={4:.1f} A".format(
                        match.get("query_label") or "residual region",
                        match["name"],
                        match["id"],
                        match["region_A"],
                        match["overlap_A"],
                    ),
                    flush=True,
                )
        else:
            print(
                "  No known broad non-stellar catalog feature overlaps the "
                "user-noticed intervals.",
                flush=True,
            )
        candidate_feature_ids = tuple(
            dict.fromkeys(match["id"] for match in candidate_feature_matches)
        )
        if candidate_feature_ids:
            known_feature_overlap = sp.annotate_nonstellar_features(
                baseline_case.collection,
                baseline_result,
                feature_names=candidate_feature_ids,
                policy="warn",
                verbose=True,
            )
        known_residual_windows = sp.diagnose_known_residual_windows(
            baseline_case.collection,
            baseline_result,
            threshold_sigma=2.5,
            verbose=True,
        )
        flagged = known_residual_windows.get("flagged_windows", [])
        if flagged:
            for item in flagged:
                print(
                    "  {0}: median={1:.2f} sigma, rms={2:.2f} sigma; {3}".format(
                        item["name"],
                        item["median_sigma"],
                        item["rms_sigma"],
                        item["recommended_action"],
                    ),
                    flush=True,
                )
        else:
            print("  No curated known-feature residual window was flagged.", flush=True)

        feature_windows = []
        for window in baseline_case.fit_windows:
            wmin, wmax = window["limits_A"]
            overlaps_candidate = any(
                max(wmin, match["region_A"][0])
                < min(wmax, match["region_A"][1])
                for match in candidate_feature_matches
            )
            if overlaps_candidate:
                feature_windows.append(window)

        if not feature_windows:
            for window in baseline_case.fit_windows:
                wmin, wmax = window["limits_A"]
                overlaps_user_region = any(
                    max(wmin, region["region_A"][0])
                    < min(wmax, region["region_A"][1])
                    for region in suspicious_residual_regions
                )
                if overlaps_user_region:
                    feature_windows.append(window)
        sp.plot_model_line_windows(
            baseline_result,
            windows=feature_windows,
            annotation_regions=candidate_feature_matches,
            title="Example 4B: residual triage with candidate catalog features",
            show_residuals=True,
            residual_kind="pull",
            ncols=2,
            figsize_per_panel=(7.2, 5.0),
            footer=(
                "Orange = candidate catalog feature overlap, not a mask or "
                "correction."
            ),
            savepath=_savepath(
                args.plot_dir,
                "example4b_residual_triage_windows.png",
            ),
        )
        _show_or_close(args.no_show)

    if args.output_json:
        payload = {
            "example": "example4b_balmer_stability_checks",
            "spectrum": str(args.spectrum),
            "reader": args.reader,
            "run_level": args.run_level,
            "baseline_case": baseline_case.to_dict(),
            "baseline_setup": baseline_setup.to_dict(),
            "baseline_audit": baseline_audit,
            "planned_variants": [
                {
                    "id": variant_id,
                    "label": _complete_variant_spec(plan[variant_id])["label"],
                    "question": _complete_variant_spec(plan[variant_id])["question"],
                }
                for variant_id in DEFAULT_VARIANT_IDS
            ],
            "baseline_result": None
            if baseline_result is None
            else baseline_result.to_dict(include_arrays=False, include_local_paths=True),
            "variant_results": [
                {
                    "id": item["id"],
                    "label": item["label"],
                    "question": item["question"],
                    "result": item["result"].to_dict(
                        include_arrays=False,
                        include_local_paths=True,
                    ),
                }
                for item in variant_results
            ],
            "stability_comparison": comparison,
            "candidate_feature_matches": candidate_feature_matches,
            "known_feature_overlap": known_feature_overlap,
            "known_residual_windows": known_residual_windows,
            "interpretation": (
                "Example 4B follows up the same Gaia21ccu Balmer case as "
                "Example 4A. Variants are bounded sensitivity checks, not a "
                "complete final-analysis uncertainty model."
            ),
        }
        atomic_write_json(args.output_json, payload)
        print("Wrote JSON: {0}".format(args.output_json), flush=True)

    if args.run_level == "review":
        print(
            "\nEvidence gathered here: the baseline Balmer case, reviewed setup, "
            "and planned stability variants were reconstructed without running "
            "PHOENIX. Run --run-level baseline or --run-level stability when "
            "you are ready to inspect model residuals and parameter shifts.",
            flush=True,
        )
    elif args.run_level == "baseline":
        print(
            "\nEvidence gathered here: the baseline Balmer fit and residual "
            "triage were run. Evidence still missing: bounded stability variants "
            "against continuum, core-mask, line-selection, and resolution/LSF "
            "choices.",
            flush=True,
        )
    else:
        print(
            "\nEvidence gathered here: the baseline Balmer fit was compared "
            "against bounded continuum, core-mask, line-selection, and "
            "resolution/LSF variants, and suspicious residual regions were "
            "checked against Spyctres' diagnostic catalogs. This is a stability "
            "audit, not an external systematic-error calibration.",
            flush=True,
        )
    print(
        "Next: use Example 6 for multi-arm consistency, or Example 7 for "
        "reference-star validation discipline.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
