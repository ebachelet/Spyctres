"""Minimal command-line example for the public Spyctres PHOENIX API.

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
from pathlib import Path

import numpy as np
from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt
from Spyctres import fit_stellar_spectrum, prepare_phoenix_fit_kwargs
from Spyctres.io import read_spectrum
from Spyctres.plotting import COMMON_LINES, plot_fit_referee, plot_fit_windows
from Spyctres.preprocessing import (
    OPTICAL_DIB_DIAGNOSTIC_FEATURES,
    nonstellar_feature_mask,
    nonstellar_feature_masks,
    nonstellar_feature_metadata,
    overlapping_nonstellar_features,
)


DIB_FEATURE_NAMES = OPTICAL_DIB_DIAGNOSTIC_FEATURES


KNOWN_RESIDUAL_WINDOWS = (
    {
        "name": "DIB 4882 / Hβ red wing",
        "region_A": (4876.0, 4908.0),
        "kind": "diffuse_interstellar_band_overlap",
        "linked_feature": "dib_4882",
        "diagnostic_line": "Hbeta",
        "description": (
            "Coherent residuals here can be caused by unmodelled DIB 4882 "
            "absorption, but intrinsic/composite Balmer structure is also "
            "possible for non-ordinary targets."
        ),
        "likely_causes": (
            "DIB 4882 or another non-stellar broad absorption feature",
            "intrinsic or composite Balmer-line structure",
            "Balmer-wing sensitivity to Teff/logg",
            "continuum placement across broad Hβ",
            "rotation/macroturbulence not fitted by this example",
            "approximate or missing wavelength-dependent LSF",
            "PHOENIX LTE/model-domain mismatch for hot or peculiar spectra",
        ),
        "recommended_action": (
            "Compare the default fit with a rerun using --mask-dibs, but do "
            "not treat masking as a final explanation if similar asymmetries "
            "appear in other Balmer lines."
        ),
    },
)


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
            "--output-plot /tmp/spyctres_fit.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spectrum", help="Reduced one-dimensional spectrum file.")
    parser.add_argument("--instrument", required=True, help="Registered reader name.")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--auto-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use spectrum metadata/coverage to choose first-pass fit defaults. "
            "Expert CLI values still override the suggestions."
        ),
    )
    parser.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Search-budget mode used by --auto-defaults.",
    )
    parser.add_argument("--teff", type=float, default=None)
    parser.add_argument("--feh", type=float, default=None)
    parser.add_argument("--logg", type=float, default=None)
    parser.add_argument("--rv", type=float, default=None)
    parser.add_argument("--teff-min", type=float, default=None)
    parser.add_argument("--teff-max", type=float, default=None)
    parser.add_argument("--feh-min", type=float, default=None)
    parser.add_argument("--feh-max", type=float, default=None)
    parser.add_argument("--logg-min", type=float, default=None)
    parser.add_argument("--logg-max", type=float, default=None)
    parser.add_argument("--rv-min", type=float, default=None)
    parser.add_argument("--rv-max", type=float, default=None)
    parser.add_argument("--wmin", type=float, default=None, help="Override fit-window minimum wavelength in Angstrom.")
    parser.add_argument("--wmax", type=float, default=None, help="Override fit-window maximum wavelength in Angstrom.")
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument(
        "--show-dibs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Annotate known diffuse interstellar bands, currently DIB 4428 "
            "and DIB 4882, on the overview diagnostic plot when they overlap "
            "the spectrum."
        ),
    )
    parser.add_argument(
        "--mask-dibs",
        action="store_true",
        help=(
            "Exclude known DIB regions from the stellar fit. This is equivalent "
            "to --nonstellar-feature-policy mask_known."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--dib-padding",
        type=float,
        default=0.0,
        help="Extra Angstrom half-width padding for DIB annotations/masks.",
    )
    parser.add_argument(
        "--known-residual-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inspect curated diagnostic windows, such as the H-beta red wing, "
            "and flag coherent residuals without automatically masking them."
        ),
    )
    parser.add_argument(
        "--known-residual-threshold",
        type=float,
        default=2.5,
        help=(
            "Flag a known diagnostic window when either |median residual| or "
            "RMS residual exceeds this sigma threshold."
        ),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument(
        "--plot-layout",
        choices=("stacked", "side_by_side"),
        default="stacked",
        help=(
            "Diagnostic plot layout. 'stacked' is the default interactive view: "
            "wide data/model panel over a wide residual panel."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--line-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open/save a companion zoomed-line diagnostic figure for identified "
            "Balmer/Ca/Mg lines, with He I added only for hot fitted solutions. Use "
            "--no-line-diagnostics to disable it."
        ),
    )
    parser.add_argument(
        "--line-groups",
        default="auto",
        help=(
            "Comma-separated line groups used for the zoomed diagnostics, or "
            "'auto'. Known groups include balmer, caii, nai, mgii, and hei."
        ),
    )
    parser.add_argument(
        "--hot-line-teff-threshold",
        type=float,
        default=10500.0,
        help=(
            "Fitted Teff threshold in K above which auto line diagnostics include "
            "hot-star He I panels."
        ),
    )
    parser.add_argument(
        "--line-window-half-width",
        type=float,
        default=30.0,
        help="Half-width in Angstrom for each zoomed diagnostic line panel.",
    )
    parser.add_argument(
        "--output-line-plot",
        default=None,
        help=(
            "Optional path for the zoomed-line diagnostic figure. If omitted "
            "and --output-plot is supplied, a *_lines companion path is used."
        ),
    )
    parser.add_argument(
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
    return fit_kwargs, suggestion


def _append_exclusion_mask(fit_kwargs, mask_spec):
    masks = list(fit_kwargs.get("exclude_masks", []) or [])
    if fit_kwargs.get("exclude_mask") is not None:
        masks.append(fit_kwargs.pop("exclude_mask"))
    masks.append(mask_spec)
    fit_kwargs["exclude_masks"] = masks


def _coerce_segments(spectrum):
    if hasattr(spectrum, "segments"):
        return list(spectrum.segments)
    if isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    return [spectrum]


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


def _line_windows_for_used_pixels(segment, used_mask, groups, half_width_A):
    if half_width_A <= 0:
        raise ValueError("--line-window-half-width must be positive.")
    wave = np.asarray(segment.wave, dtype=float)
    used = np.asarray(used_mask, dtype=bool)
    finite_used = used & np.isfinite(wave)
    if not np.any(finite_used):
        return []
    wmin = float(np.nanmin(wave[finite_used]))
    wmax = float(np.nanmax(wave[finite_used]))
    raw_windows = []
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
            raw_windows.append((line_label, center - half_width_A, center + half_width_A))
    return _merge_overlapping_line_windows(sorted(raw_windows, key=lambda item: item[1]))


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


def _add_quality_flag(result, flag):
    flags = list(getattr(result, "quality_flags", ()))
    if flag not in flags:
        flags.append(flag)
    result.summary["quality_flags"] = list(flags)
    object.__setattr__(result, "quality_flags", tuple(flags))


def _annotate_nonstellar_features(args, spectrum, result):
    policy = "mask_known" if args.mask_dibs else args.nonstellar_feature_policy
    overlaps = overlapping_nonstellar_features(
        spectrum,
        names=DIB_FEATURE_NAMES,
        padding_A=args.dib_padding,
    )
    overlap_diagnostics = _nonstellar_overlap_diagnostics(overlaps)
    frame_warnings = _nonstellar_frame_warnings(spectrum, overlaps)
    payload = {
        "show_dibs": bool(args.show_dibs),
        "mask_dibs": bool(policy == "mask_known"),
        "policy": policy,
        "mask_application_frame": "data",
        "frame_note": (
            "Known non-stellar feature regions are interpreted on the current "
            "data wavelength grid; no observer/barycentric/stellar-rest frame "
            "transformation is applied implicitly."
        ),
        "features": overlaps,
        "overlap_diagnostics": overlap_diagnostics,
        "frame_warnings": frame_warnings,
    }
    result.summary["nonstellar_features"] = payload
    if overlaps and policy != "ignore":
        _add_quality_flag(result, "nonstellar_feature_overlap")
        if policy == "mask_known":
            _add_quality_flag(result, "nonstellar_mask_applied")
        if frame_warnings:
            _add_quality_flag(result, "nonstellar_feature_frame_ambiguous")
        if overlap_diagnostics:
            _add_quality_flag(result, "diagnostic_line_contaminated")
            for item in overlap_diagnostics:
                if item.get("flag") == "dib_overlap_balmer_wing":
                    _add_quality_flag(result, "dib_overlap_balmer_wing")
        names = ", ".join(item["name"] for item in overlaps)
        action = "masked" if policy == "mask_known" else "shown but not masked"
        print(
            "\nNote: non-stellar feature(s) overlap this fit: {0}. "
            "They are {1}; PHOENIX is not expected to model DIB absorption.".format(
                names,
                action,
            ),
            flush=True,
        )
        if overlap_diagnostics:
            detail = "; ".join(
                "{0} overlaps {1}".format(
                    item.get("feature", "feature"),
                    item.get("diagnostic_line", "diagnostic line"),
                )
                for item in overlap_diagnostics
            )
            print(
                "Diagnostic contamination warning: {0}. Consider a controlled "
                "rerun with --mask-dibs and compare the fitted parameters, but "
                "also inspect other Balmer lines before treating this as a "
                "settled DIB detection.".format(
                    detail
                ),
                flush=True,
            )
    elif overlaps:
        names = ", ".join(item["name"] for item in overlaps)
        print(
            "\nNote: non-stellar feature overlap was detected but ignored by "
            "policy: {0}. Provenance is still recorded in the result JSON.".format(
                names
            ),
            flush=True,
        )
    return payload


def _nonstellar_overlap_diagnostics(overlaps):
    diagnostics = []
    for feature in overlaps:
        diagnostic_lines = set(feature.get("diagnostic_lines") or [])
        if feature.get("kind") == "diffuse_interstellar_band" and "Hbeta" in diagnostic_lines:
            diagnostics.append(
                {
                    "flag": "dib_overlap_balmer_wing",
                    "feature": feature.get("name"),
                    "feature_id": "dib_4882",
                    "diagnostic_line": "Hbeta",
                    "overlap_region_A": feature.get("region_A"),
                    "origin_hypothesis": "catalog_overlap_only",
                    "recommended_action": (
                        "Rerun with --mask-dibs or "
                        "--nonstellar-feature-policy mask_known and compare "
                        "Teff, logg, [Fe/H], RV, chi2_red, and residual flags; "
                        "inspect other Balmer lines before interpreting this "
                        "as a DIB detection."
                    ),
                }
            )
    return diagnostics


def _nonstellar_frame_warnings(spectrum, overlaps):
    if not overlaps:
        return []
    segments = _coerce_segments(spectrum)
    warnings = []
    for feature in overlaps:
        if feature.get("frame_type") != "ism_velocity":
            continue
        affected_segments = set(feature.get("segments") or [])
        ambiguous_segments = []
        for index, segment in enumerate(segments):
            name = getattr(segment, "name", None) or "segment {0}".format(index + 1)
            if affected_segments and name not in affected_segments:
                continue
            observer_frame = str(getattr(segment, "observer_frame", "unknown")).lower()
            stellar_rest_status = str(
                getattr(segment, "stellar_rest_status", "unknown")
            ).lower()
            wave_frame = str(getattr(segment, "wave_frame", "unknown")).lower()
            if "unknown" in {observer_frame, stellar_rest_status, wave_frame}:
                ambiguous_segments.append(
                    {
                        "segment": name,
                        "observer_frame": observer_frame,
                        "stellar_rest_status": stellar_rest_status,
                        "wave_frame": wave_frame,
                    }
                )
        if ambiguous_segments:
            warnings.append(
                {
                    "feature": feature.get("name"),
                    "feature_id": feature.get("id"),
                    "frame_type": feature.get("frame_type"),
                    "warning": "nonstellar_feature_frame_ambiguous",
                    "affected_segments": ambiguous_segments,
                    "recommended_action": (
                        "Treat fixed DIB intervals as data-frame diagnostics "
                        "until the spectrum and ISM velocity frames are known."
                    ),
                }
            )
    return warnings


def _segment_error_array(segment, flux, model):
    err = getattr(segment, "err", None)
    if err is not None:
        err = np.asarray(err, dtype=float)
        if err.shape == flux.shape:
            good = np.isfinite(err) & (err > 0.0)
            if np.any(good):
                out = np.full(flux.shape, np.nan, dtype=float)
                out[good] = err[good]
                return out

    residual = np.asarray(flux, dtype=float) - np.asarray(model, dtype=float)
    finite = residual[np.isfinite(residual)]
    if finite.size >= 5:
        median = float(np.nanmedian(finite))
        mad = float(np.nanmedian(np.abs(finite - median)))
        robust_sigma = 1.4826 * mad if mad > 0.0 else float(np.nanstd(finite))
    else:
        robust_sigma = float(np.nanstd(finite)) if finite.size else np.nan
    if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
        robust_sigma = 1.0
    return np.full(flux.shape, robust_sigma, dtype=float)


def _diagnose_known_residual_windows(args, spectrum, result):
    payload = {
        "enabled": bool(args.known_residual_diagnostics),
        "threshold_sigma": float(args.known_residual_threshold),
        "windows": [],
        "flagged_windows": [],
    }
    result.summary["known_residual_windows"] = payload
    if not args.known_residual_diagnostics:
        return payload
    threshold = float(args.known_residual_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("--known-residual-threshold must be finite and positive.")

    segments = _coerce_segments(spectrum)
    models = tuple(getattr(result, "models", ()))
    used_masks = tuple(getattr(result, "used_masks", ()))
    if len(models) != len(segments):
        return payload

    for segment_index, (segment, model) in enumerate(zip(segments, models)):
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        model = np.asarray(model, dtype=float)
        if model.shape != wave.shape or flux.shape != wave.shape:
            continue
        if segment_index < len(used_masks):
            used = np.asarray(used_masks[segment_index], dtype=bool)
        else:
            used = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
        err = _segment_error_array(segment, flux, model)
        sigma_resid = (flux - model) / err
        frac_resid = (flux - model) / np.where(np.abs(model) > 0.0, model, np.nan)
        for window in KNOWN_RESIDUAL_WINDOWS:
            wmin, wmax = window["region_A"]
            in_window = (
                used
                & np.isfinite(wave)
                & np.isfinite(sigma_resid)
                & (wave >= float(wmin))
                & (wave <= float(wmax))
            )
            n_used = int(np.count_nonzero(in_window))
            if n_used < 5:
                continue
            values = sigma_resid[in_window]
            frac_values = frac_resid[in_window]
            median_sigma = float(np.nanmedian(values))
            rms_sigma = float(np.sqrt(np.nanmean(values * values)))
            max_abs_sigma = float(np.nanmax(np.abs(values)))
            median_fractional_residual = (
                None
                if not np.any(np.isfinite(frac_values))
                else float(np.nanmedian(frac_values))
            )
            fraction_negative = float(np.mean(values < 0.0))
            fraction_positive = float(np.mean(values > 0.0))
            residual_sign = _coherent_residual_sign(
                median_sigma,
                fraction_negative=fraction_negative,
                fraction_positive=fraction_positive,
                threshold=threshold,
            )
            origin_hypothesis = _origin_hypothesis_for_window(
                residual_sign,
                linked_feature=window.get("linked_feature"),
            )
            recommended_action = _recommended_action_for_origin(
                origin_hypothesis,
                default_action=window.get("recommended_action"),
            )
            flagged = bool(
                abs(median_sigma) >= threshold or rms_sigma >= threshold
            )
            residual_detection = {
                "candidate_detected": flagged,
                "origin_hypothesis": origin_hypothesis,
                "residual_sign": residual_sign,
                "fraction_negative_pixels": fraction_negative,
                "fraction_positive_pixels": fraction_positive,
                "phase": "phase_a_heuristic",
                "cross_line_consistency_checked": False,
                "note": (
                    "This is a single-window heuristic. A future cross-line "
                    "classifier should test whether the pattern repeats at a "
                    "consistent velocity offset across Balmer/Ca/Na diagnostics."
                ),
            }
            item = {
                "name": window["name"],
                "linked_feature": window.get("linked_feature"),
                "diagnostic_line": window.get("diagnostic_line"),
                "segment": getattr(segment, "name", None),
                "segment_index": int(segment_index),
                "region_A": [float(wmin), float(wmax)],
                "kind": window["kind"],
                "description": window["description"],
                "likely_causes": list(window["likely_causes"]),
                "recommended_action": recommended_action,
                "n_used": n_used,
                "median_sigma": median_sigma,
                "rms_sigma": rms_sigma,
                "max_abs_sigma": max_abs_sigma,
                "median_fractional_residual": median_fractional_residual,
                "fraction_negative_pixels": fraction_negative,
                "fraction_positive_pixels": fraction_positive,
                "residual_sign": residual_sign,
                "origin_hypothesis": origin_hypothesis,
                "absorption_like": bool(median_sigma <= -threshold),
                "emission_like": bool(median_sigma >= threshold),
                "residual_detection": residual_detection,
                "flagged": flagged,
            }
            payload["windows"].append(item)
            if flagged:
                payload["flagged_windows"].append(item)

    if payload["flagged_windows"]:
        _attach_residual_detections_to_nonstellar_features(result, payload)
        _add_quality_flag(result, "known_line_region_residual")
        if any(item.get("absorption_like") for item in payload["flagged_windows"]):
            _add_quality_flag(result, "diagnostic_line_contaminated")
            _add_quality_flag(result, "dib_overlap_balmer_wing")
            _add_quality_flag(result, "dib_candidate_detected")
        names = ", ".join(item["name"] for item in payload["flagged_windows"])
        print(
            "\nNote: coherent residuals were detected in known diagnostic "
            "window(s): {0}. These are reported, not masked automatically; "
            "inspect continuum placement, LSF/rotation assumptions, possible "
            "DIB absorption, and intrinsic/composite Balmer structure.".format(
                names
            ),
            flush=True,
        )
    return payload


def _coherent_residual_sign(
    median_sigma,
    *,
    fraction_negative,
    fraction_positive,
    threshold,
):
    if median_sigma <= -threshold and fraction_negative >= 0.6:
        return "absorption_like"
    if median_sigma >= threshold and fraction_positive >= 0.6:
        return "emission_like"
    if abs(median_sigma) >= threshold:
        return "coherent_mixed_sign"
    return "not_coherent"


def _origin_hypothesis_for_window(residual_sign, linked_feature=None):
    if residual_sign == "emission_like":
        return "intrinsic_or_composite_candidate"
    if residual_sign == "absorption_like" and linked_feature:
        return "ambiguous"
    if residual_sign == "absorption_like":
        return "external_contaminant"
    return "ambiguous"


def _recommended_action_for_origin(origin_hypothesis, default_action=None):
    if origin_hypothesis == "intrinsic_or_composite_candidate":
        return (
            "This positive/emission-like pattern is evidence against a DIB-only "
            "explanation. Masking is not recommended as the default response; "
            "flag the target for manual review and inspect other lines."
        )
    if origin_hypothesis == "ambiguous":
        return (
            "Treat this as a candidate external-contaminant or intrinsic-line "
            "signal. Compare a named-mask refit, but inspect other Balmer lines "
            "before adopting masking as the explanation."
        )
    return default_action


def _attach_residual_detections_to_nonstellar_features(result, residual_payload):
    nonstellar = result.summary.get("nonstellar_features")
    if not nonstellar:
        return
    detections = [
        dict(item)
        for item in residual_payload.get("flagged_windows", [])
        if item.get("linked_feature")
    ]
    if not detections:
        return
    by_feature = {item["linked_feature"]: item for item in detections}
    for feature in nonstellar.get("features", []):
        feature_id = str(feature.get("id") or "").lower()
        if feature_id in by_feature:
            feature["residual_detection"] = by_feature[feature_id][
                "residual_detection"
            ]
    nonstellar["residual_detections"] = detections


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

        windows = _line_windows_for_used_pixels(
            segment,
            used,
            groups,
            args.line_window_half_width,
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
            fig.savefig(savepath, bbox_inches="tight")
            key = "line_diagnostics"
            if multiple_segments:
                key = "line_diagnostics_segment_{0}".format(index + 1)
            plot_paths[key] = savepath
        figures.append(fig)

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
    spectrum = read_spectrum(args.spectrum, instrument=args.instrument)
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, spectrum)
    if suggestion is not None:
        print("Suggested first-pass fit defaults:", flush=True)
        for reason in suggestion.reasons:
            print("  - {0}".format(reason), flush=True)
        for warning in suggestion.warnings:
            print("  WARNING: {0}".format(warning), flush=True)
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
