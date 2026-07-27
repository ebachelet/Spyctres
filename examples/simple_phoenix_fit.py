"""Example 1: minimal command-line path for the public Spyctres PHOENIX API.

Purpose
-------
This script demonstrates the recommended public `fit_stellar_spectrum()`
workflow from a reduced spectrum to a structured PHOENIX result and an
interactive diagnostic plot. The plot is a
wide, stacked diagnostic view: observed spectrum and model on top, residuals
underneath. This makes it easier to inspect broad failures, masked regions, and
line-profile mismatches than a small square pop-up.

The script does perform a full-spectrum fit over the usable input pixels, but it
is not configured as a precision line-profile analysis. In particular, it does
not fit rotational or macroturbulent broadening, detailed abundance patterns, or
instrument-specific LSF variations. The reader's nominal resolution (or ``--R``)
is the only instrumental broadening supplied. Individual observed lines can
therefore be wider or narrower than the demonstration model even when the broad
atmospheric classification is useful.

By default, the diagnostic plot focuses on the fitted wavelength range and draws
the PHOENIX/model residuals only on pixels that were actually used by the fit.
Use ``--plot-xlim all`` if you want to inspect the full loaded segment and mask
boundaries.

The example also opens a default-on zoomed line-diagnostic figure for
Balmer/Ca/Mg lines that overlap fitted pixels, adding He I only for hot fitted
solutions. These panels are visual diagnostics, not separate local line fits.

Example
-------
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_result.json \
  --output-plot /tmp/spyctres_fit.png
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt
from Spyctres import fit_stellar_spectrum, prepare_phoenix_fit_kwargs
from Spyctres._spectrum_helpers import spectrum_segments
from Spyctres._workflow_helpers import (
    archive_mask_count as _shared_archive_mask_count,
    archive_masks_by_segment as _shared_archive_masks_by_segment,
    fit_kwargs_with_archive_policy as _shared_fit_kwargs_with_archive_policy,
    resolution_assumption_for_audit as _shared_resolution_assumption_for_audit,
    resolution_override_summary as _shared_resolution_override_summary,
)
from Spyctres.diagnostics import (
    KNOWN_RESIDUAL_WINDOWS,
    annotate_nonstellar_features,
    diagnose_known_residual_windows,
)
from Spyctres.io import read_spectrum
from Spyctres.plotting import (
    COMMON_LINES,
    plot_fit_referee,
    plot_fit_windows,
    save_figure,
)
from Spyctres.preprocessing import (
    OPTICAL_DIB_DIAGNOSTIC_FEATURES,
    audit_spectrum_for_fit,
    nonstellar_feature_mask,
    nonstellar_feature_masks,
    nonstellar_feature_metadata,
)


DIB_FEATURE_NAMES = OPTICAL_DIB_DIAGNOSTIC_FEATURES


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal fit_stellar_spectrum() PHOENIX demonstration. This is "
            "not a precision line-profile fit: rotation, macroturbulence, "
            "abundance variations, and detailed LSF structure are not fitted."
        ),
        epilog=(
            "Example:\n"
            "  python examples/simple_phoenix_fit.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter "
            "--output-json /tmp/spyctres_result.json "
            "--output-plot /tmp/spyctres_fit.png\n\n"
            "Notes for user-supplied external spectra:\n"
            "  SDSS reader metadata intentionally keeps resolution=None. If "
            "you run your own SDSS quicklook, --R 2000 is an explicit "
            "approximation, not precision SDSS LSF modelling.\n"
            "  UVES-POP resolving power, wavelength medium, and any error "
            "column should be verified from the product documentation before "
            "precision work. No SDSS/UVES-POP spectra are bundled with this "
            "example."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    input_group = parser.add_argument_group("minimal input")
    reader_group = parser.add_argument_group("reader metadata and archive options")
    fit_group = parser.add_argument_group("fit search controls")
    mask_group = parser.add_argument_group("mask and residual diagnostics")
    output_group = parser.add_argument_group("plots and outputs")

    input_group.add_argument("spectrum", help="Reduced one-dimensional spectrum file.")
    input_group.add_argument("--instrument", required=True, help="Registered reader name.")
    reader_group.add_argument(
        "--sdss-mask-policy",
        choices=("auto", "ivar_only", "and_mask_conservative", "stellar_strict", "sky_strict"),
        default="auto",
        help=(
            "SDSS reader bitmask policy. 'auto' uses stellar_strict for this "
            "PHOENIX fitting example; other instruments ignore this option."
        ),
    )
    reader_group.add_argument(
        "--wave-medium",
        choices=("keep", "unknown", "air", "vacuum"),
        default="keep",
        help=(
            "Override the reader's wavelength-medium metadata before fitting. "
            "Use this for products whose documentation says the wavelength "
            "grid is air or vacuum but the generic reader cannot know that. "
            "The default 'keep' preserves reader metadata."
        ),
    )
    reader_group.add_argument(
        "--uves-err-column",
        type=int,
        default=None,
        help=(
            "Optional zero-based uncertainty column for UVES-POP ASCII files. "
            "UVES-POP two-column reads keep err=None by default; pass 2 when "
            "the third numeric column is a 1-sigma error for your product."
        ),
    )
    fit_group.add_argument("--phoenix-dir", default=None)
    fit_group.add_argument(
        "--auto-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use spectrum metadata/coverage to choose first-pass fit defaults. "
            "Expert CLI values still override the suggestions."
        ),
    )
    fit_group.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Search-budget mode used by --auto-defaults.",
    )
    fit_group.add_argument("--teff", type=float, default=None)
    fit_group.add_argument("--feh", type=float, default=None)
    fit_group.add_argument("--logg", type=float, default=None)
    fit_group.add_argument("--rv", type=float, default=None)
    fit_group.add_argument("--teff-min", type=float, default=None)
    fit_group.add_argument("--teff-max", type=float, default=None)
    fit_group.add_argument("--feh-min", type=float, default=None)
    fit_group.add_argument("--feh-max", type=float, default=None)
    fit_group.add_argument("--logg-min", type=float, default=None)
    fit_group.add_argument("--logg-max", type=float, default=None)
    fit_group.add_argument("--rv-min", type=float, default=None)
    fit_group.add_argument("--rv-max", type=float, default=None)
    fit_group.add_argument("--wmin", type=float, default=None, help="Override fit-window minimum wavelength in Angstrom.")
    fit_group.add_argument("--wmax", type=float, default=None, help="Override fit-window maximum wavelength in Angstrom.")
    fit_group.add_argument("--R", type=float, default=None, dest="resolution_R")
    mask_group.add_argument(
        "--show-dibs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Annotate known diffuse interstellar bands, currently DIB 4428 "
            "and DIB 4882, on the overview diagnostic plot when they overlap "
            "the spectrum."
        ),
    )
    mask_group.add_argument(
        "--mask-dibs",
        action="store_true",
        help=(
            "Exclude known DIB regions from the stellar fit. This is equivalent "
            "to --nonstellar-feature-policy mask_known."
        ),
    )
    mask_group.add_argument(
        "--nonstellar-feature-policy",
        choices=("warn", "mask_known", "ignore"),
        default="warn",
        help=(
            "How to handle known non-stellar features such as DIB 4428 and "
            "DIB 4882. 'warn' annotates/flags only, 'mask_known' excludes the "
            "named regions, and 'ignore' records overlap but raises no flags. "
            "These correspond to flag/mask/off in the reviewer terminology; "
            "residual diagnostics are controlled separately."
        ),
    )
    mask_group.add_argument(
        "--dib-padding",
        type=float,
        default=0.0,
        help="Extra Angstrom half-width padding for DIB annotations/masks.",
    )
    mask_group.add_argument(
        "--archive-mask-policy",
        choices=("apply", "warn", "ignore"),
        default="apply",
        help=(
            "How to handle recognized archive/product bad-region catalogs. "
            "'apply' excludes them as named fit masks and records provenance; "
            "'warn' leaves them fitted but raises readiness/quality warnings; "
            "'ignore' leaves them fitted and records that the user explicitly "
            "ignored archive mask advice."
        ),
    )
    mask_group.add_argument(
        "--known-residual-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inspect curated diagnostic windows, such as the H-beta red wing, "
            "and flag coherent residuals without automatically masking them."
        ),
    )
    mask_group.add_argument(
        "--known-residual-threshold",
        type=float,
        default=2.5,
        help=(
            "Flag a known diagnostic window when either |median residual| or "
            "RMS residual exceeds this sigma threshold."
        ),
    )
    output_group.add_argument("--output-json", default=None)
    output_group.add_argument("--output-plot", default=None)
    output_group.add_argument(
        "--plot-layout",
        choices=("stacked", "side_by_side"),
        default="stacked",
        help=(
            "Diagnostic plot layout. 'stacked' is the default interactive view: "
            "wide data/model panel over a wide residual panel."
        ),
    )
    output_group.add_argument(
        "--plot-xlim",
        choices=("fit", "all"),
        default="fit",
        help=(
            "Diagnostic plot x-axis. 'fit' focuses on fitted wavelengths and "
            "draws model/residuals only where pixels were used. 'all' shows the "
            "full loaded segment while still hiding model/residuals on unused "
            "pixels."
        ),
    )
    output_group.add_argument(
        "--line-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open/save a companion zoomed-line diagnostic figure for identified "
            "Balmer/Ca/Mg lines, with He I added only for hot fitted solutions. Use "
            "--no-line-diagnostics to disable it."
        ),
    )
    output_group.add_argument(
        "--line-groups",
        default="auto",
        help=(
            "Comma-separated line groups used for the zoomed diagnostics, or "
            "'auto'. Known groups include balmer, caii, nai, mgii, and hei."
        ),
    )
    output_group.add_argument(
        "--hot-line-teff-threshold",
        type=float,
        default=10500.0,
        help=(
            "Fitted Teff threshold in K above which auto line diagnostics include "
            "hot-star He I panels."
        ),
    )
    output_group.add_argument(
        "--line-window-half-width",
        type=float,
        default=30.0,
        help="Half-width in Angstrom for each zoomed diagnostic line panel.",
    )
    output_group.add_argument(
        "--output-line-plot",
        default=None,
        help=(
            "Optional path for the zoomed-line diagnostic figure. If omitted "
            "and --output-plot is supplied, a *_lines companion path is used."
        ),
    )
    output_group.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive fit figure (useful for batch runs).",
    )
    return parser


def _fit_kwargs_from_args(args, spectrum):
    fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
        spectrum,
        auto_defaults=args.auto_defaults,
        defaults_mode=args.defaults_mode,
        science_case="classification",
        p0_overrides=(args.teff, args.feh, args.logg, args.rv),
        lower_bound_overrides=(
            args.teff_min,
            args.feh_min,
            args.logg_min,
            args.rv_min,
        ),
        upper_bound_overrides=(
            args.teff_max,
            args.feh_max,
            args.logg_max,
            args.rv_max,
        ),
        window=(
            args.wmin,
            args.wmax,
        ) if args.wmin is not None or args.wmax is not None else None,
        resolution_R=args.resolution_R,
    )
    if args.mask_dibs or args.nonstellar_feature_policy == "mask_known":
        for mask_spec in nonstellar_feature_masks(
            DIB_FEATURE_NAMES,
            padding_A=args.dib_padding,
        ):
            _append_exclusion_mask(fit_kwargs, mask_spec)
    if args.archive_mask_policy == "apply":
        fit_kwargs = _fit_kwargs_with_archive_policy(
            fit_kwargs,
            _archive_masks_by_segment(spectrum),
            args.archive_mask_policy,
        )
    return fit_kwargs, suggestion


def _resolution_override_summary(args):
    return _shared_resolution_override_summary(args.resolution_R)


def _assumed_resolution_for_audit(args):
    return _shared_resolution_assumption_for_audit(args.resolution_R)


def _display_risk_flags(args, risk_flags):
    flags = [str(flag) for flag in risk_flags]
    if args.resolution_R is not None:
        flags = [flag for flag in flags if flag != "missing_resolution"]
    return flags


def _display_warnings(args, warnings):
    warnings = [str(warning) for warning in warnings]
    if args.resolution_R is None:
        return warnings
    return [
        warning
        for warning in warnings
        if "lacks resolution metadata" not in warning
    ]


def _print_classification_branch_summary(suggestion, max_rows=3):
    branch_plan = suggestion.provenance.get("classification_branches", {})
    branches = list(branch_plan.get("branches") or ())
    recommended = branch_plan.get("recommended_branch") or {}
    if not branches:
        return
    print("Branch-aware first-pass candidates:", flush=True)
    if recommended:
        regions = recommended.get("fit_regions_A") or ()
        print(
            "  - recommended: {0} ({1} fitted diagnostic window(s))".format(
                recommended.get("label", recommended.get("id", "unknown")),
                len(regions),
            ),
            flush=True,
        )
    else:
        print("  - no branch passed the minimum coverage checks", flush=True)
    for branch in branches[: int(max_rows)]:
        status = branch.get("status", "unknown")
        label = branch.get("label", branch.get("id", "unknown"))
        windows = branch.get("fit_window_ids") or branch.get("matched_window_ids") or ()
        print(
            "  - {0}: {1}; score={2:.2f}; windows={3}".format(
                label,
                status,
                float(branch.get("score", 0.0)),
                ", ".join(windows) or "none",
            ),
            flush=True,
        )
    if branch_plan.get("ambiguous_top_branches"):
        print(
            "  - note: top branches are similar; compare branch-specific fits before trusting one.",
            flush=True,
        )


def _append_exclusion_mask(fit_kwargs, mask_spec):
    masks = list(fit_kwargs.get("exclude_masks", []) or [])
    if fit_kwargs.get("exclude_mask") is not None:
        masks.append(fit_kwargs.pop("exclude_mask"))
    masks.append(mask_spec)
    fit_kwargs["exclude_masks"] = masks


def _reader_kwargs_from_args(args):
    instrument = str(args.instrument).strip().lower()
    kwargs = {}
    if instrument in {"sdss", "sdss_spec", "segue"}:
        policy = args.sdss_mask_policy
        if policy == "auto":
            policy = "stellar_strict"
        kwargs["sdss_mask_policy"] = policy
    if instrument in {"uves_pop", "uves-pop", "uvespop"}:
        err_column = getattr(args, "uves_err_column", None)
        if err_column is not None:
            kwargs["err_column"] = int(err_column)
    return kwargs


def _coerce_segments(spectrum):
    return spectrum_segments(spectrum, tuple_is_collection=True, coerce=False)


def _override_segment_wave_medium(segment, wave_medium):
    meta = dict(getattr(segment, "meta", {}) or {})
    meta["wave_medium"] = wave_medium
    meta["wave_medium_source"] = "user_override"
    return segment.copy(meta=meta, wave_medium=wave_medium)


def _override_spectrum_wave_medium(spectrum, wave_medium):
    wave_medium = str(wave_medium).strip().lower()
    if wave_medium == "keep":
        return spectrum
    if wave_medium not in {"unknown", "air", "vacuum"}:
        raise ValueError("--wave-medium must be keep, unknown, air, or vacuum.")
    if hasattr(spectrum, "segments"):
        meta = dict(getattr(spectrum, "meta", {}) or {})
        meta["wave_medium_override"] = wave_medium
        return spectrum.copy(
            segments=[
                _override_segment_wave_medium(segment, wave_medium)
                for segment in spectrum.segments
            ],
            meta=meta,
        )
    if isinstance(spectrum, tuple):
        return tuple(_override_segment_wave_medium(segment, wave_medium) for segment in spectrum)
    if isinstance(spectrum, list):
        return [_override_segment_wave_medium(segment, wave_medium) for segment in spectrum]
    return _override_segment_wave_medium(spectrum, wave_medium)


def _archive_masks_by_segment(spectrum):
    return _shared_archive_masks_by_segment(spectrum)


def _archive_mask_count(archive_masks):
    return _shared_archive_mask_count(archive_masks)


def _fit_kwargs_with_archive_policy(fit_kwargs, archive_masks, policy):
    return _shared_fit_kwargs_with_archive_policy(fit_kwargs, archive_masks, policy)


def _parse_line_groups(value, result=None, hot_teff_threshold=10500.0):
    value = str(value).strip().lower()
    if value == "auto":
        groups = ["balmer", "caii", "mgii"]
        teff = None
        if result is not None:
            try:
                teff = float(result["teff"])
            except (KeyError, TypeError, ValueError):
                teff = None
        if teff is not None and np.isfinite(teff) and teff >= hot_teff_threshold:
            groups.append("hei")
        return groups

    groups = []
    for raw in value.split(","):
        group = raw.strip().lower()
        if not group:
            continue
        if group not in COMMON_LINES:
            known = ", ".join(sorted(COMMON_LINES))
            raise ValueError(
                "Unknown line group '{0}'. Known groups: {1}.".format(group, known)
            )
        groups.append(group)
    if not groups:
        raise ValueError("At least one line group is required for line diagnostics.")
    return groups


def _line_window_quality(segment, used_mask, label, center, wmin, wmax):
    wave = np.asarray(segment.wave, dtype=float)
    flux = np.asarray(segment.flux, dtype=float)
    used = np.asarray(used_mask, dtype=bool)
    in_window = (wave >= float(wmin)) & (wave <= float(wmax)) & np.isfinite(wave)
    finite_flux = np.isfinite(flux)
    total = int(np.count_nonzero(in_window))
    used_good = in_window & used & finite_flux
    n_used = int(np.count_nonzero(used_good))

    zero_like = in_window & finite_flux & np.isclose(flux, 0.0, rtol=0.0, atol=1.0e-14)
    n_zero = int(np.count_nonzero(zero_like))
    zero_fraction = 0.0 if total == 0 else float(n_zero) / float(total)
    usable_fraction = 0.0 if total == 0 else float(n_used) / float(total)

    center_half_width = min(5.0, max(1.0, 0.1 * (float(wmax) - float(wmin))))
    center_band = in_window & (np.abs(wave - float(center)) <= center_half_width)
    center_total = int(np.count_nonzero(center_band))
    center_used = int(np.count_nonzero(center_band & used & finite_flux))
    center_zero = int(np.count_nonzero(center_band & zero_like))
    center_zero_fraction = (
        0.0 if center_total == 0 else float(center_zero) / float(center_total)
    )

    reasons = []
    if total == 0:
        reasons.append("no_pixels_in_window")
    if n_used < 8:
        reasons.append("too_few_fitted_pixels")
    if total > 0 and usable_fraction < 0.35:
        reasons.append("low_usable_fraction")
    if total > 0 and zero_fraction > 0.25:
        reasons.append("zero_flux_block")
    if center_total == 0:
        reasons.append("no_pixels_near_line_center")
    elif center_used == 0:
        reasons.append("line_center_not_fitted")
    elif center_zero_fraction > 0.5:
        reasons.append("line_center_zero_flux_block")

    return {
        "label": label,
        "center_A": float(center),
        "window_A": [float(wmin), float(wmax)],
        "status": "skipped" if reasons else "selected",
        "reasons": reasons,
        "n_pixels": total,
        "n_fitted_pixels": n_used,
        "usable_fraction": usable_fraction,
        "zero_flux_fraction": zero_fraction,
        "center_half_width_A": center_half_width,
        "center_n_pixels": center_total,
        "center_n_fitted_pixels": center_used,
        "center_zero_flux_fraction": center_zero_fraction,
    }


def _line_windows_for_used_pixels(
    segment,
    used_mask,
    groups,
    half_width_A,
    return_diagnostics=False,
):
    if half_width_A <= 0:
        raise ValueError("--line-window-half-width must be positive.")
    wave = np.asarray(segment.wave, dtype=float)
    used = np.asarray(used_mask, dtype=bool)
    finite_used = used & np.isfinite(wave)
    if not np.any(finite_used):
        if return_diagnostics:
            return [], {"selected": [], "skipped": []}
        return []
    wmin = float(np.nanmin(wave[finite_used]))
    wmax = float(np.nanmax(wave[finite_used]))
    raw_windows = []
    selected_records = []
    skipped_records = []
    seen = set()
    for group in groups:
        for label, center in COMMON_LINES[group]:
            center = float(center)
            if center < wmin or center > wmax:
                continue
            key = (round(center, 4), label)
            if key in seen:
                continue
            seen.add(key)
            line_label = "{0} {1:.1f} Å".format(label, center)
            line_wmin = center - half_width_A
            line_wmax = center + half_width_A
            record = _line_window_quality(
                segment,
                used,
                line_label,
                center,
                line_wmin,
                line_wmax,
            )
            if record["status"] == "skipped":
                skipped_records.append(record)
                continue
            selected_records.append(record)
            raw_windows.append((line_label, line_wmin, line_wmax))
    windows = _merge_overlapping_line_windows(sorted(raw_windows, key=lambda item: item[1]))
    if return_diagnostics:
        return windows, {
            "selected": selected_records,
            "skipped": skipped_records,
        }
    return windows


def _merge_overlapping_line_windows(windows):
    if not windows:
        return []
    merged = []
    current_labels = [windows[0][0]]
    current_min = float(windows[0][1])
    current_max = float(windows[0][2])
    current_center = 0.5 * (current_min + current_max)
    current_half_width = 0.5 * (current_max - current_min)
    for label, wmin, wmax in windows[1:]:
        wmin = float(wmin)
        wmax = float(wmax)
        center = 0.5 * (wmin + wmax)
        half_width = 0.5 * (wmax - wmin)
        very_close = abs(center - current_center) <= 0.75 * min(
            current_half_width,
            half_width,
        )
        if wmin <= current_max and very_close:
            current_labels.append(label)
            current_max = max(current_max, wmax)
            current_center = 0.5 * (current_min + current_max)
            current_half_width = 0.5 * (current_max - current_min)
        else:
            merged.append((" + ".join(current_labels), current_min, current_max))
            current_labels = [label]
            current_min = wmin
            current_max = wmax
            current_center = center
            current_half_width = half_width
    merged.append((" + ".join(current_labels), current_min, current_max))
    return merged


def _line_panel_columns(n_windows):
    n_windows = int(n_windows)
    if n_windows <= 1:
        return 1
    if n_windows <= 4:
        return 2
    return 3


def _derive_line_plot_path(output_line_plot, output_plot, segment_index=None):
    path = output_line_plot or output_plot
    if path is None:
        return None
    path = Path(path)
    suffix = path.suffix or ".png"
    if output_line_plot:
        new_path = path
    else:
        new_path = path.with_name("{0}_lines{1}".format(path.stem, suffix))
    if segment_index is not None:
        new_path = new_path.with_name(
            "{0}_segment{1}{2}".format(
                new_path.stem,
                int(segment_index) + 1,
                new_path.suffix or suffix,
            )
        )
    return str(new_path)


def _annotate_nonstellar_features(args, spectrum, result):
    policy = "mask_known" if args.mask_dibs else args.nonstellar_feature_policy
    return annotate_nonstellar_features(
        spectrum,
        result,
        feature_names=DIB_FEATURE_NAMES,
        policy=policy,
        padding_A=args.dib_padding,
        show=args.show_dibs,
        verbose=True,
    )


def _diagnose_known_residual_windows(args, spectrum, result):
    return diagnose_known_residual_windows(
        spectrum,
        result,
        enabled=args.known_residual_diagnostics,
        threshold_sigma=args.known_residual_threshold,
        windows=KNOWN_RESIDUAL_WINDOWS,
        verbose=True,
    )


def _print_interpretation_hints(args, result, fit_kwargs):
    flags = set(getattr(result, "quality_flags", ()))
    bounds = fit_kwargs.get("bounds", None)
    teff = None
    teff_upper = None
    try:
        teff = float(result["teff"])
    except (KeyError, TypeError, ValueError):
        teff = None
    if bounds is not None:
        try:
            teff_upper = float(bounds[1][0])
        except (IndexError, TypeError, ValueError):
            teff_upper = None

    if (
        args.defaults_mode == "quicklook"
        and {"high_chi2", "structured_residuals"} & flags
    ):
        print(
            "\nHint: this quicklook fit is flagged for structured/high residuals. "
            "Try rerunning with --defaults-mode diagnostic to widen the Teff "
            "search before interpreting individual line mismatches.",
            flush=True,
        )
    if (
        args.defaults_mode == "quicklook"
        and teff is not None
        and teff_upper is not None
        and np.isfinite(teff)
        and np.isfinite(teff_upper)
        and teff >= teff_upper - 250.0
    ):
        print(
            "Hint: the fitted Teff is close to the quicklook upper bound "
            "({0:.0f} K). A hotter diagnostic run may be more appropriate.".format(
                teff_upper
            ),
            flush=True,
        )


def _build_line_diagnostic_plots(args, spectrum, result):
    groups = _parse_line_groups(
        args.line_groups,
        result=result,
        hot_teff_threshold=args.hot_line_teff_threshold,
    )
    if str(args.line_groups).strip().lower() == "auto":
        print(
            "Auto line diagnostics selected groups: {0}.".format(
                ", ".join(groups)
            ),
            flush=True,
        )
    segments = _coerce_segments(spectrum)
    models = tuple(getattr(result, "models", ()))
    used_masks = tuple(getattr(result, "used_masks", ()))
    excluded_masks = tuple(getattr(result, "excluded_masks", ()))
    if len(models) != len(segments):
        raise ValueError("Line diagnostics require one reconstructed model per segment.")

    figures = []
    plot_paths = {}
    quality_payload = {
        "schema_version": 1,
        "purpose": (
            "Quality gate for zoomed diagnostic-line panels; skipped windows "
            "are not used for fitting or plotted as trustworthy local diagnostics."
        ),
        "segments": [],
    }
    multiple_segments = len(segments) > 1
    for index, (segment, model) in enumerate(zip(segments, models)):
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        err = None if getattr(segment, "err", None) is None else np.asarray(segment.err)
        model = np.asarray(model, dtype=float)
        if index < len(used_masks):
            used = np.asarray(used_masks[index], dtype=bool)
        else:
            used = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
        if index < len(excluded_masks):
            excluded = np.asarray(excluded_masks[index], dtype=bool)
        else:
            excluded = np.zeros(wave.size, dtype=bool)

        windows, window_quality = _line_windows_for_used_pixels(
            segment,
            used,
            groups,
            args.line_window_half_width,
            return_diagnostics=True,
        )
        quality_payload["segments"].append(
            {
                "segment_index": int(index),
                "segment_name": getattr(segment, "name", None),
                "selected": window_quality["selected"],
                "skipped": window_quality["skipped"],
            }
        )
        if window_quality["skipped"]:
            skipped_labels = [
                "{0} ({1})".format(
                    record["label"],
                    ", ".join(record["reasons"]) or "unspecified",
                )
                for record in window_quality["skipped"]
            ]
            print(
                "Skipping unreliable line diagnostic windows for {0}: {1}.".format(
                    getattr(segment, "name", "segment {0}".format(index + 1)),
                    "; ".join(skipped_labels),
                ),
                flush=True,
            )
        if not windows:
            continue

        model_plot = np.full_like(model, np.nan, dtype=float)
        finite_model = np.isfinite(model)
        model_plot[used & finite_model] = model[used & finite_model]
        ncols = _line_panel_columns(len(windows))
        savepath = _derive_line_plot_path(
            args.output_line_plot,
            args.output_plot,
            segment_index=index if multiple_segments else None,
        )
        print(
            "Building zoomed line diagnostics for {0}: {1} panel(s).".format(
                getattr(segment, "name", "segment {0}".format(index + 1)),
                len(windows),
            ),
            flush=True,
        )
        fig, _axes = plot_fit_windows(
            wave,
            flux,
            err,
            model_plot,
            windows,
            ncols=ncols,
            title="Zoomed line diagnostics: {0}".format(
                getattr(segment, "name", "segment {0}".format(index + 1))
            ),
            line_groups=groups,
            excluded_mask=excluded | ~used,
            figsize_per_panel=(5.2, 3.0),
            data_label="observed",
            model_label="model on fitted pixels",
        )
        if savepath is not None:
            save_figure(fig, savepath)
            key = "line_diagnostics"
            if multiple_segments:
                key = "line_diagnostics_segment_{0}".format(index + 1)
            plot_paths[key] = savepath
        figures.append(fig)

    result.summary["line_diagnostic_window_quality"] = quality_payload
    result.provenance["line_diagnostic_window_quality"] = {
        "schema_version": quality_payload["schema_version"],
        "note": "See result summary for selected/skipped diagnostic-line windows.",
    }
    if not figures:
        print(
            "No requested diagnostic lines overlap fitted pixels; skipping "
            "zoomed line diagnostics.",
            flush=True,
        )
    return figures, plot_paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("Reading spectrum...", flush=True)
    reader_kwargs = _reader_kwargs_from_args(args)
    if reader_kwargs:
        print(
            "Using reader options: {0}".format(
                ", ".join("{0}={1}".format(k, v) for k, v in reader_kwargs.items())
            ),
            flush=True,
        )
    spectrum = read_spectrum(
        args.spectrum,
        instrument=args.instrument,
        warn_unknown=(args.wave_medium == "keep"),
        **reader_kwargs,
    )
    spectrum = _override_spectrum_wave_medium(spectrum, args.wave_medium)
    if args.wave_medium != "keep":
        print(
            "Overriding wavelength-medium metadata: {0}.".format(args.wave_medium),
            flush=True,
        )
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, spectrum)
    archive_masks = _archive_masks_by_segment(spectrum)
    archive_mask_count = _archive_mask_count(archive_masks)
    if archive_masks:
        print(
            "Archive mask policy: {0} ({1} recognized archive mask region(s)).".format(
                args.archive_mask_policy,
                archive_mask_count,
            ),
            flush=True,
        )
    resolution_override = _resolution_override_summary(args)
    readiness = audit_spectrum_for_fit(
        spectrum,
        fit_windows=fit_kwargs.get("regions"),
        exclude_masks=fit_kwargs.get("exclude_masks"),
        intended_use="first_pass_classification",
        assumed_resolution=_assumed_resolution_for_audit(args),
    )
    print(
        "Spectrum readiness: fit_ready={0}, quicklook_only={1}, fitted_pixels={2}, flags={3}".format(
            readiness["fit_ready"],
            readiness["quicklook_only"],
            readiness["n_fit_candidate"],
            ", ".join(readiness["interpretation_flags"]) or "none",
        ),
        flush=True,
    )
    for warning in readiness.get("warnings", []):
        print("Spectrum readiness WARNING: {0}".format(warning), flush=True)
    if suggestion is not None:
        print("Suggested first-pass fit defaults:", flush=True)
        for reason in suggestion.reasons:
            print("  - {0}".format(reason), flush=True)
        _print_classification_branch_summary(suggestion)
        if resolution_override is not None:
            print(
                "  - using user-supplied assumed resolution R={0:g}".format(
                    resolution_override["assumed_resolution_R"]
                ),
                flush=True,
            )
        interpretation = suggestion.provenance.get("interpretation", {})
        if interpretation:
            print("Default interpretation:", flush=True)
            print(
                "  - intended use: {0}".format(
                    interpretation.get("intended_use", "first_pass_classification")
                ),
                flush=True,
            )
            mode_policy = interpretation.get("mode_policy", {})
            if mode_policy:
                print(
                    "  - defaults mode: {0} ({1} budget; {2})".format(
                        mode_policy.get("mode", args.defaults_mode),
                        mode_policy.get("search_budget", "unknown"),
                        mode_policy.get("default_status", "first_pass"),
                    ),
                    flush=True,
                )
            print(
                "  - RV role: {0}".format(
                    interpretation.get("rv_role", "alignment_parameter")
                ),
                flush=True,
            )
            print(
                "  - {0}".format(interpretation.get("rv_note", "")),
                flush=True,
            )
            risk_flags = _display_risk_flags(
                args,
                interpretation.get("risk_flags", []),
            )
            if risk_flags:
                print(
                    "  - assumption/risk flags: {0}".format(
                        ", ".join(str(flag) for flag in risk_flags)
                    ),
                    flush=True,
                )
            next_step = interpretation.get("recommended_next_step")
            if next_step:
                print("  - next step: {0}".format(next_step), flush=True)
        for warning in _display_warnings(args, suggestion.warnings):
            print("  WARNING: {0}".format(warning), flush=True)
    elif resolution_override is not None:
        print(
            "Using user-supplied assumed resolution R={0:g}.".format(
                resolution_override["assumed_resolution_R"]
            ),
            flush=True,
        )
    print("Running public fit_stellar_spectrum() workflow...", flush=True)
    result = fit_stellar_spectrum(
        spectrum,
        model="phoenix",
        phoenix_dir=args.phoenix_dir,
        auto_defaults=False,
        science_case="classification",
        progress_callback=lambda event: print(event, flush=True),
        **fit_kwargs,
    )
    if suggestion is not None:
        result.summary["fit_default_suggestion"] = suggestion.to_dict()
    result.summary["spectrum_readiness"] = readiness
    result.summary["archive_mask_policy"] = {
        "policy": args.archive_mask_policy,
        "recognized_mask_count": archive_mask_count,
        "applied": bool(args.archive_mask_policy == "apply"),
    }
    result.provenance["spectrum_readiness"] = readiness
    result.provenance["archive_mask_policy"] = dict(result.summary["archive_mask_policy"])
    if args.wave_medium != "keep":
        result.summary["wave_medium_override"] = {
            "wave_medium": args.wave_medium,
            "source": "user_override",
        }
        result.provenance["wave_medium_override"] = dict(
            result.summary["wave_medium_override"]
        )
    if resolution_override is not None:
        result.summary.update(resolution_override)
        result.provenance["resolution_override"] = dict(resolution_override)
    nonstellar_payload = _annotate_nonstellar_features(args, spectrum, result)
    _diagnose_known_residual_windows(args, spectrum, result)
    summary = {
        key: result[key]
        for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
    }
    print(json.dumps(summary, indent=2))
    print()
    print(result.quality_report_text())
    _print_interpretation_hints(args, result, fit_kwargs)
    print(
        "\nInterpretation: this example demonstrates ingestion, a native-grid "
        "full-spectrum PHOENIX fit, structured results, and plotting. It is not "
        "a precision line-width fit; line-profile mismatches can reflect stellar "
        "rotation/macroturbulence, abundance differences, or an approximate LSF. "
        "The diagnostic plot focuses on fitted wavelengths by default; gray "
        "points and shaded regions mark pixels excluded from the fit. The "
        "optional zoomed line panels are visual diagnostics rather than "
        "separate local line-profile fits."
    )

    if not result.models:
        if args.output_json:
            result.save_json(args.output_json)
        raise RuntimeError("Fit did not converge, so no model is available to plot.")
    generated_plot_paths = {}
    print("Building diagnostic plot...", flush=True)
    fig, _axes = plot_fit_referee(
        result,
        segment=spectrum,
        savepath=args.output_plot,
        layout=args.plot_layout,
        figsize_per_segment=(
            (16.0, 6.4) if args.plot_layout == "stacked" else (12.0, 3.4)
        ),
        max_points_per_segment=20000,
        xlim_mode=args.plot_xlim,
        feature_regions=(
            nonstellar_feature_metadata(DIB_FEATURE_NAMES, padding_A=args.dib_padding)
            if (
                args.show_dibs
                and nonstellar_payload["features"]
                and nonstellar_payload.get("policy") != "ignore"
            )
            else None
        ),
    )
    generated_plot_paths.update(getattr(fig, "spyctres_generated_files", {}) or {})
    extra_figures = []
    if args.line_diagnostics:
        extra_figures, line_paths = _build_line_diagnostic_plots(args, spectrum, result)
        generated_plot_paths.update(line_paths)
    if args.output_json:
        result.save_json(
            args.output_json,
            plot_paths=generated_plot_paths or None,
        )
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
        for extra_fig in extra_figures:
            plt.close(extra_fig)


if __name__ == "__main__":
    main()
