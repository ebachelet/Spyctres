#!/usr/bin/env python
"""Advanced Example 6: multi-arm X-SHOOTER classification and consistency checks.

This example reads the bundled Gaia21ccu UVB, VIS, and NIR X-SHOOTER arms into
one ``SpectrumCollection``.  The default run is PHOENIX-free: it inspects arm
metadata, masks, and diagnostic windows so the user can decide which features
are worth fitting.  A bounded multi-arm PHOENIX fit is available with
``--run-fit`` after that inspection.

This is an advanced optional example, not a mandatory step after the core
Examples 1-5 path.

What this demonstrates
----------------------
How to keep UVB/VIS/NIR arms separate while analysing them together: each arm
keeps its own wavelength coverage, mask, uncertainty, and resolving-power
metadata.  Spyctres then selects diagnostic windows across the combined
coverage and can optionally fit selected windows jointly.

What this does not prove
------------------------
This is a classification/consistency workflow, not an automatic reviewed-analysis
claim.  X-SHOOTER arm flux calibration, telluric residuals, wavelength-dependent
LSF structure, and line-specific residuals still need human review before a
multi-arm result can support reviewed-analysis stellar parameters.

Example inspection run:

  python examples/example6_multiarm_classification.py --no-show

Example optional multi-arm fit:

  python examples/example6_multiarm_classification.py \
    --run-fit \
    --allow-exploratory-fit \
    --override-reason "multi-arm tutorial classification; not final analysis" \
    --plot-dir /tmp/spyctres_example6_multiarm \
    --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_DIR.parent
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp


EXAMPLE_ARM_FILES = (
    "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
    "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_VIS_TELL_CORR.fits",
    "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_NIR_TELL_CORR.fits",
)

DEFAULT_FIT_WINDOW_IDS = (
    "h_delta",
    "h_gamma",
    "h_beta",
    "h_alpha",
)

HOT_DIAGNOSTIC_WINDOW_IDS = (
    "h_delta",
    "h_gamma",
    "h_beta",
    "h_alpha",
    "he_i_4471",
    "mg_ii_4481",
    "he_i_5876",
    "paschen_beta",
    "br_gamma",
)

COOL_DIAGNOSTIC_WINDOW_IDS = (
    "ca_hk_h_epsilon",
    "ch_g_band",
    "mg_i_b",
    "na_i_d",
    "ca_ii_triplet_paschen",
    "tio_7050",
    "na_i_8200",
    "tio_red_bands",
    "na_i_kband",
    "ca_i_kband",
    "co_23um_bandhead",
)

CONTEXT_ONLY_WINDOW_IDS = (
    "paschen_gamma_delta",
    "brackett_h_band",
)

# Extra plot markers used in the wide NIR/cool-star diagnostic windows below.
# They are annotations only: the broad windows remain classification checks,
# not automatic masks or automatic physical identifications.
NIR_HYDROGEN_MARKERS = (
    ("Paδ", 10049.37),
    ("Paγ", 10938.09),
    ("Paβ", 12818.08),
    ("Brγ", 21661.0),
)

COOL_STAR_MARKERS = (
    ("TiO 7050", 7050.0),
    ("Na I 2.21 μm", 22062.0),
    ("Ca I 2.26 μm", 22630.0),
    ("CO 2-0", 22935.0),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 6: inspect and optionally fit bundled X-SHOOTER "
            "UVB/VIS/NIR arms as a multi-arm classification workflow."
        ),
        epilog=(
            "Inspect only, no PHOENIX library needed:\n"
            "  python examples/example6_multiarm_classification.py --no-show\n\n"
            "Optional bounded multi-arm fit:\n"
            "  python examples/example6_multiarm_classification.py --run-fit "
            "--allow-exploratory-fit "
            "--override-reason 'multi-arm tutorial classification; not final analysis' "
            "--plot-dir /tmp/spyctres_example6_multiarm --no-show\n\n"
            "Next:\n"
            "  python examples/example5_batch_fitting.py --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectra",
        nargs="*",
        help=(
            "Optional UVB/VIS/NIR files. If omitted, use the bundled Gaia21ccu "
            "X-SHOOTER arms under examples/data/."
        ),
    )
    parser.add_argument("--reader", default="xshooter_merge1d")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--max-windows", type=int, default=30)
    parser.add_argument(
        "--fit-window-ids",
        default=",".join(DEFAULT_FIT_WINDOW_IDS),
        help=(
            "Comma-separated diagnostic-window ids to include in the optional "
            "fit when they overlap the loaded arms."
        ),
    )
    parser.add_argument(
        "--run-fit",
        action="store_true",
        help="Run the optional multi-arm PHOENIX fit after inspection.",
    )
    parser.add_argument(
        "--allow-exploratory-fit",
        action="store_true",
        help=(
            "Allow the optional fit when readiness is blocked. Requires "
            "--override-reason and records the result as exploratory."
        ),
    )
    parser.add_argument("--override-reason", default=None)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print PHOENIX progress messages during --run-fit.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory for PNG audit/referee plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create/save plots without opening an interactive Matplotlib window.",
    )
    return parser


def _example_paths():
    return [sp.example_data_path(name) for name in EXAMPLE_ARM_FILES]


def _split_csv(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _savepath(plot_dir, filename):
    if not plot_dir:
        return None
    path = Path(plot_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _arm_label_from_path(path):
    text = Path(path).name.upper()
    for label in ("UVB", "VIS", "NIR"):
        if label in text:
            return label
    return None


def _maybe_show(no_show):
    import matplotlib.pyplot as plt

    if no_show:
        plt.close("all")
    else:
        plt.show()


def _print_classification_window_rows(collection, windows):
    """Print hot/cool diagnostic-window quality decisions for this spectrum."""
    groups = (
        ("hot/intermediate evidence", HOT_DIAGNOSTIC_WINDOW_IDS),
        ("cool-star checks", COOL_DIAGNOSTIC_WINDOW_IDS),
        ("NIR context only", CONTEXT_ONLY_WINDOW_IDS),
    )
    edge_margin_A = 150.0
    min_usable_fraction_for_fit = 0.65
    min_contiguous_fraction_for_fit = 0.30
    rows = []
    seen = set()
    for group_label, group_ids in groups:
        for item in windows.select_by_id(
            group_ids,
            require_all=False,
            include_rejected=True,
        ):
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            near_edge = False
            arms = []
            for contribution in item.get("segment_contributions", []):
                if contribution.get("n_pixels", 0) <= 0:
                    continue
                segment = collection.segments[contribution["segment_index"]]
                arms.append(str(segment.meta.get("example_arm_label", segment.name)))
                arm_lo = float(np.nanmin(segment.wave))
                arm_hi = float(np.nanmax(segment.wave))
                op_lo, op_hi = contribution.get(
                    "operational_region_A",
                    item["region_A"],
                )
                if op_lo <= arm_lo + edge_margin_A or op_hi >= arm_hi - edge_margin_A:
                    near_edge = True
            usable = float(item.get("usable_fraction", 0.0) or 0.0)
            contiguous = float(
                item.get("largest_contiguous_usable_fraction", 0.0) or 0.0
            )
            low_usable = usable < min_usable_fraction_for_fit
            fragmented = contiguous < min_contiguous_fraction_for_fit
            if item["id"] in DEFAULT_FIT_WINDOW_IDS and not (
                near_edge or low_usable or fragmented
            ):
                decision = "default fit candidate"
            elif near_edge:
                decision = "diagnostic only: near arm edge/join"
            elif fragmented:
                decision = "diagnostic only: fragmented usable pixels"
            elif low_usable:
                decision = "diagnostic only: low usable fraction"
            elif item["id"] in {"paschen_beta", "ca_ii_triplet_paschen"}:
                decision = "diagnostic only: telluric/blend-sensitive consistency check"
            else:
                decision = "diagnostic only: classification check"
            rows.append((group_label, item, sorted(set(arms)), usable, contiguous, decision))

    print("\nHot-vs-cool diagnostic windows:", flush=True)
    for _group, item, arms, usable, contiguous, decision in rows:
        print(
            "  {0:24s} {1:28s} arms={2:8s} usable={3:.2f} "
            "contiguous={4:.2f} -> {5}".format(
                item["id"],
                item["label"],
                ",".join(arms) or "none",
                usable,
                contiguous,
                decision,
            ),
            flush=True,
        )
        if item.get("risk_tags"):
            print("    risks: {0}".format(", ".join(item["risk_tags"])), flush=True)
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.allow_exploratory_fit and not str(args.override_reason or "").strip():
        raise ValueError("--allow-exploratory-fit requires --override-reason.")

    paths = [Path(path) for path in args.spectra] if args.spectra else _example_paths()

    print("Reading X-SHOOTER arms...", flush=True)
    segments = []
    for path in paths:
        segment = sp.read_spectrum(path, reader=args.reader)
        if isinstance(segment, sp.SpectrumCollection):
            segments.extend(segment.segments)
        else:
            label = _arm_label_from_path(path)
            if label:
                segment.meta["example_arm_label"] = label
            segments.append(segment)

    collection = sp.SpectrumCollection(
        segments,
        name="example6_xshooter_multiarm",
        meta={"workflow": "example6_multiarm_classification"},
    )
    print(collection.summary(), flush=True)

    print("\nPer-arm provenance:", flush=True)
    for segment in collection.segments:
        summary = segment.summary()
        resolution = summary.get("resolution") or {}
        print(
            "  {0}: {1:.1f}-{2:.1f} Å, medium={3}, frame={4}, "
            "stellar_rest={5}, R={6}".format(
                segment.name,
                summary["wavelength_range_A"][0],
                summary["wavelength_range_A"][1],
                summary.get("wave_medium"),
                summary.get("observer_frame"),
                summary.get("stellar_rest_status"),
                resolution.get("value"),
            ),
            flush=True,
        )

    print("\nSelecting diagnostic windows across all arms...", flush=True)
    windows = sp.select_diagnostic_windows(collection, max_windows=args.max_windows)
    print(windows.summary_text(max_rows=args.max_windows), flush=True)

    print("\nBuilding reviewed mask bundle...", flush=True)
    reviewed_mask = sp.build_mask(
        collection,
        archive="mask",
        tellurics="warn",
        dibs=False,
    )
    print(reviewed_mask.summary_text(), flush=True)

    sp.plot_spectrum(
        collection,
        title="Example 6: loaded X-SHOOTER UVB/VIS/NIR arms",
        figsize=(14.0, 4.2),
        savepath=_savepath(args.plot_dir, "example6_multiarm_spectrum.png"),
    )
    sp.plot_spectrum_audit(
        collection,
        diagnostic_selection=windows,
        warning_regions=reviewed_mask.warning_regions,
        title="Example 6: multi-arm audit view",
        figsize=(14.0, 8.0),
        savepath=_savepath(args.plot_dir, "example6_multiarm_audit.png"),
    )

    _print_classification_window_rows(collection, windows)

    display_wave = np.concatenate([segment.wave for segment in collection.segments])
    display_flux = np.concatenate([segment.flux for segment in collection.segments])
    display_valid = np.concatenate(reviewed_mask.valid_masks_by_segment)
    display_order = np.argsort(display_wave)

    hot_windows = windows.select_by_id(
        HOT_DIAGNOSTIC_WINDOW_IDS,
        require_all=False,
        include_rejected=True,
    )
    cool_windows = windows.select_by_id(
        COOL_DIAGNOSTIC_WINDOW_IDS,
        require_all=False,
        include_rejected=True,
    )
    context_windows = windows.select_by_id(
        CONTEXT_ONLY_WINDOW_IDS,
        require_all=False,
        include_rejected=True,
    )
    sp.plot_spectrum_line_windows(
        display_wave[display_order],
        display_flux[display_order],
        hot_windows,
        valid_mask=display_valid[display_order],
        line_groups=("balmer", "hei", "mgii", NIR_HYDROGEN_MARKERS),
        title="Example 6: hot/intermediate-star diagnostic windows",
        ncols=3,
        figsize_per_panel=(5.8, 3.1),
        footer=(
            "Hydrogen, He I, and Mg II windows are positive evidence for a hot "
            "or intermediate classification; inspect NIR/telluric-sensitive "
            "checks before fitting them."
        ),
        savepath=_savepath(args.plot_dir, "example6_hot_diagnostic_windows.png"),
    )
    sp.plot_spectrum_line_windows(
        display_wave[display_order],
        display_flux[display_order],
        cool_windows,
        valid_mask=display_valid[display_order],
        line_groups=("caii", "nai", COOL_STAR_MARKERS),
        title="Example 6: cool-star diagnostic windows",
        ncols=3,
        figsize_per_panel=(5.8, 3.1),
        footer=(
            "Strong Ca/Na/TiO/CO structure would support a cooler or composite "
            "interpretation; weak cool-star features are useful negative "
            "classification evidence."
        ),
        savepath=_savepath(args.plot_dir, "example6_cool_diagnostic_windows.png"),
    )
    if context_windows:
        sp.plot_spectrum_line_windows(
            display_wave[display_order],
            display_flux[display_order],
            context_windows,
            valid_mask=display_valid[display_order],
            line_groups=(NIR_HYDROGEN_MARKERS,),
            title="Example 6: NIR context-only windows",
            ncols=2,
            figsize_per_panel=(6.8, 3.1),
            footer=(
                "These windows are useful context, but are "
                "edge/telluric/fragmentation-sensitive in this dataset."
            ),
            savepath=_savepath(args.plot_dir, "example6_nir_context_windows.png"),
        )

    fit_window_ids = _split_csv(args.fit_window_ids)
    fit_windows = sorted(
        windows.select_by_id(
            fit_window_ids,
            require_all=False,
            include_rejected=True,
        ),
        key=lambda item: float(item["region_A"][0]),
    )
    print("\nWindows proposed for optional multi-arm fit:", flush=True)
    for item in fit_windows:
        print(
            "  {0}: {1} {2}".format(item["id"], item["label"], item["region_A"]),
            flush=True,
        )
    print(
        "  These are conservative fit candidates. Other plotted VIS/NIR "
        "windows remain classification checks unless their masks, tellurics, "
        "edge distance, and model suitability have been reviewed.",
        flush=True,
    )

    fit_selection = sp.build_fit_collection_from_windows(
        collection,
        windows,
        window_ids=fit_window_ids,
        valid_masks=reviewed_mask.valid_masks_by_segment,
        min_usable_fraction=0.65,
        min_contiguous_fraction=0.30,
        name="example6_xshooter_retained_fit_arms",
    )
    print("\n" + fit_selection.summary_text(), flush=True)

    # The fit collection is now aligned with the retained masks and regions.
    # Arms/windows that fail the conservative retention gate are still shown
    # above as diagnostic context; they are simply not used in the optional fit.
    fit_collection = fit_selection.collection
    fit_valid_masks = fit_selection.valid_masks_by_segment
    fit_regions = list(fit_selection.regions)
    fit_windows = sorted(
        windows.select_by_id(
            fit_selection.retained_window_ids,
            require_all=False,
            include_rejected=True,
        ),
        key=lambda item: float(item["region_A"][0]),
    )

    fit_audit = sp.audit_spectrum_for_fit(
        fit_collection,
        regions=fit_regions,
        intent="quicklook_classification",
    )
    print(
        "\nMulti-arm fit audit: ready={0}, fitted_pixels={1}, blockers={2}".format(
            fit_audit.get("fit_ready"),
            fit_audit.get("n_fit_candidate"),
            ", ".join(fit_audit.get("blockers_for_intent") or []) or "none",
        ),
        flush=True,
    )

    setup = sp.suggest_fit_setup(
        fit_collection,
        mode="quicklook",
        intent="quicklook_classification",
    ).with_regions(fit_regions).with_readiness(fit_audit)
    print("\nFit setup for the optional multi-arm fit:", flush=True)
    print(setup.summary_text(include_hash=False), flush=True)

    result = None
    if args.run_fit:
        if fit_audit.get("fit_ready") is not True:
            if not args.allow_exploratory_fit:
                raise ValueError(
                    "The multi-arm audit is blocked. Inspect the plots first, "
                    "then rerun with --allow-exploratory-fit --override-reason "
                    "'...' if this is an explicit tutorial/diagnostic fit."
                )
            setup = setup.allow_exploratory(reason=args.override_reason)

        print("\nRunning optional multi-arm PHOENIX fit...", flush=True)
        result = sp.fit_stellar_spectrum(
            fit_collection,
            model="phoenix",
            setup=setup,
            valid_mask=fit_valid_masks,
            phoenix_dir=args.phoenix_dir,
            progress_callback=(
                (lambda event: print(f"[{event.elapsed_s:6.1f}s] {event}", flush=True))
                if args.show_progress
                else None
            ),
        )
        print(result.summary_text(include_hash=False, max_flags=8), flush=True)
        print("\n" + result.quality_report_text(), flush=True)

        sp.plot_fit_referee(
            result,
            layout="stacked",
            flux_ylim_mode="visible",
            savepath=_savepath(args.plot_dir, "example6_multiarm_fit.png"),
        )
        sp.plot_model_line_windows(
            result,
            windows=fit_windows[:8],
            title="Example 6: multi-arm diagnostic windows",
            show_residuals=True,
            residual_kind="pull",
            ncols=2,
            figsize_per_panel=(7.2, 5.2),
            savepath=_savepath(args.plot_dir, "example6_multiarm_line_windows.png"),
        )
    else:
        print(
            "\nNo PHOENIX fit was run. Rerun with --run-fit after reviewing "
            "the audit plot and selected windows.",
            flush=True,
        )

    _maybe_show(args.no_show)
    if result is None:
        print(
            "\nEvidence gathered here: the arms were ingested together, their "
            "metadata/masks were preserved, and hot/cool/NIR diagnostic windows "
            "were triaged. No PHOENIX model fit was run.",
            flush=True,
        )
    else:
        print(
            "\nEvidence gathered here: the same reviewed windows were fitted "
            "jointly across the retained arms, while fragmented or "
            "telluric/edge-sensitive windows stayed diagnostic-only.",
            flush=True,
        )
    print(
        "Scope note: Example 6 is a multi-arm classification/consistency "
        "workflow. It complements, but does not replace, the stricter "
        "reviewed-analysis checks in Example 4.",
        flush=True,
    )
    return 0 if result is not None or not args.run_fit else 1


if __name__ == "__main__":
    raise SystemExit(main())
