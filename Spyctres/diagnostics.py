"""Lightweight fit diagnostics shared by examples, notebooks, and scripts.

The helpers in this module implement Phase-A diagnostic bookkeeping only:
catalog overlap, frame warnings, optional known-feature mask provenance, and
single-window residual summaries.  They intentionally do not claim a physical
classification for DIB/telluric/intrinsic stellar origins.
"""

import numpy as np

from .preprocessing import overlapping_nonstellar_features


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
            "Compare the default fit with a named-mask rerun, but do not "
            "treat masking as a final explanation if similar asymmetries "
            "appear in other Balmer lines."
        ),
    },
)


def _coerce_segments(spectrum):
    if hasattr(spectrum, "segments"):
        return list(spectrum.segments)
    if isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    return [spectrum]


def add_quality_flag(result, flag):
    """Attach a quality flag to a mutable result-like object."""
    flags = list(getattr(result, "quality_flags", ()))
    if flag not in flags:
        flags.append(flag)
    result.summary["quality_flags"] = list(flags)
    object.__setattr__(result, "quality_flags", tuple(flags))


def annotate_nonstellar_features(
    spectrum,
    result,
    *,
    feature_names=("dib_4428", "dib_4882"),
    policy="warn",
    padding_A=0.0,
    show=True,
    assumed_ism_rv_kms=None,
    verbose=False,
):
    """Record known non-stellar feature overlap and conservative warnings.

    Parameters
    ----------
    spectrum
        ``SpectrumSegment``/``SpectrumCollection``-like input.
    result
        Mutable result object with ``summary`` and ``quality_flags`` fields.
    feature_names
        Known feature IDs from ``Spyctres.preprocessing.NONSTELLAR_FEATURES``.
    policy : {"warn", "mask_known", "ignore"}
        Diagnostic policy.  ``mask_known`` records that a caller applied named
        masks; this helper itself does not modify fitted pixels.
    assumed_ism_rv_kms : float or None
        Reserved for future feature-frame transformations.  If absent, ISM
        masks in stellar-rest spectra are flagged as frame-ambiguous.
    """
    policy = str(policy).strip().lower()
    if policy not in {"warn", "mask_known", "ignore"}:
        raise ValueError("policy must be one of: warn, mask_known, ignore.")

    overlaps = overlapping_nonstellar_features(
        spectrum,
        names=feature_names,
        padding_A=padding_A,
    )
    overlaps, fitted_overlap_available = _attach_fitted_pixel_overlap(
        spectrum,
        result,
        overlaps,
    )
    active_overlaps = [
        item
        for item in overlaps
        if item.get("fitted_pixel_overlap", True)
        or not item.get("fitted_pixel_overlap_available", False)
    ]
    action = (
        "masked"
        if policy == "mask_known"
        else "ignored"
        if policy == "ignore"
        else "flagged"
    )
    for feature in overlaps:
        feature["action"] = action
        feature["mask_applied"] = bool(policy == "mask_known")
        feature.setdefault("residual_detection", None)

    overlap_diagnostics = _nonstellar_overlap_diagnostics(active_overlaps)
    frame_warnings = _nonstellar_frame_warnings(
        spectrum,
        active_overlaps,
        assumed_ism_rv_kms=assumed_ism_rv_kms,
    )
    payload = {
        "show_features": bool(show),
        "show_dibs": bool(show),
        "mask_known": bool(policy == "mask_known"),
        "mask_dibs": bool(policy == "mask_known"),
        "policy": policy,
        "mask_application_frame": "data",
        "assumed_ism_rv_kms": (
            None if assumed_ism_rv_kms is None else float(assumed_ism_rv_kms)
        ),
        "frame_note": (
            "Known non-stellar feature regions are interpreted on the current "
            "data wavelength grid; no observer/barycentric/stellar-rest frame "
            "transformation is applied implicitly."
        ),
        "overlap_basis": (
            "fitted_pixels"
            if fitted_overlap_available
            else "loaded_valid_segment_pixels"
        ),
        "features": overlaps,
        "overlap_diagnostics": overlap_diagnostics,
        "frame_warnings": frame_warnings,
    }
    result.summary["nonstellar_features"] = payload
    if active_overlaps and policy != "ignore":
        add_quality_flag(result, "nonstellar_feature_overlap")
        if policy == "mask_known":
            add_quality_flag(result, "nonstellar_mask_applied")
        if frame_warnings:
            add_quality_flag(result, "nonstellar_feature_frame_ambiguous")
        if overlap_diagnostics:
            add_quality_flag(result, "diagnostic_line_contaminated")
            for item in overlap_diagnostics:
                if item.get("flag") == "dib_overlap_balmer_wing":
                    add_quality_flag(result, "dib_overlap_balmer_wing")
        if verbose:
            names = ", ".join(item["name"] for item in active_overlaps)
            user_action = (
                "masked" if policy == "mask_known" else "shown but not masked"
            )
            print(
                "\nNote: non-stellar feature(s) overlap this fit: {0}. "
                "They are {1}; PHOENIX is not expected to model DIB "
                "absorption.".format(names, user_action),
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
                    "Diagnostic contamination warning: {0}. Consider a "
                    "controlled named-mask rerun and compare the fitted "
                    "parameters, but also inspect other Balmer lines before "
                    "treating this as a settled DIB detection.".format(detail),
                    flush=True,
                )
    elif overlaps and verbose:
        names = ", ".join(item["name"] for item in overlaps)
        print(
            "\nNote: non-stellar feature overlap was detected but ignored by "
            "policy or falls outside fitted pixels: {0}. Provenance is still "
            "recorded in the result JSON.".format(names),
            flush=True,
        )
    return payload


def _attach_fitted_pixel_overlap(spectrum, result, overlaps):
    segments = _coerce_segments(spectrum)
    used_masks = tuple(getattr(result, "used_masks", ()) or ())
    fitted_overlap_available = len(used_masks) == len(segments) and len(segments) > 0
    if not fitted_overlap_available:
        for feature in overlaps:
            feature["fitted_pixel_overlap_available"] = False
        return overlaps, False

    out = []
    for feature in overlaps:
        wmin, wmax = feature.get("region_A", (None, None))
        if wmin is None or wmax is None:
            item = dict(feature)
            item["fitted_pixel_overlap_available"] = True
            item["fitted_pixel_overlap"] = False
            item["fitted_pixel_overlap_pixels"] = 0
            item["fitted_segment_overlaps"] = []
            out.append(item)
            continue
        fitted_pixels = 0
        fitted_segment_overlaps = []
        for index, (segment, used_mask) in enumerate(zip(segments, used_masks)):
            wave = np.asarray(segment.wave, dtype=float)
            used = np.asarray(used_mask, dtype=bool)
            if used.shape != wave.shape:
                fitted_overlap_available = False
                break
            in_feature = (
                used
                & np.isfinite(wave)
                & (wave >= float(wmin))
                & (wave <= float(wmax))
            )
            n = int(np.count_nonzero(in_feature))
            if n:
                fitted_segment_overlaps.append(
                    {
                        "segment": getattr(segment, "name", None),
                        "segment_index": int(index),
                        "fitted_pixel_overlap_pixels": n,
                    }
                )
            fitted_pixels += n
        item = dict(feature)
        item["fitted_pixel_overlap_available"] = fitted_overlap_available
        item["fitted_pixel_overlap"] = bool(
            fitted_overlap_available and fitted_pixels > 0
        )
        item["fitted_pixel_overlap_pixels"] = int(fitted_pixels)
        item["fitted_segment_overlaps"] = fitted_segment_overlaps
        out.append(item)
    if not fitted_overlap_available:
        for item in out:
            item["fitted_pixel_overlap_available"] = False
            item.pop("fitted_pixel_overlap", None)
            item.pop("fitted_pixel_overlap_pixels", None)
            item.pop("fitted_segment_overlaps", None)
    return out, fitted_overlap_available


def _nonstellar_overlap_diagnostics(overlaps):
    diagnostics = []
    for feature in overlaps:
        diagnostic_lines = set(feature.get("diagnostic_lines") or [])
        if (
            feature.get("kind") == "diffuse_interstellar_band"
            and "Hbeta" in diagnostic_lines
        ):
            diagnostics.append(
                {
                    "flag": "dib_overlap_balmer_wing",
                    "feature": feature.get("name"),
                    "feature_id": "dib_4882",
                    "diagnostic_line": "Hbeta",
                    "overlap_region_A": feature.get("region_A"),
                    "origin_hypothesis": "catalog_overlap_only",
                    "recommended_action": (
                        "Rerun with a named DIB mask and compare Teff, logg, "
                        "[Fe/H], RV, chi2_red, and residual flags; inspect "
                        "other Balmer lines before interpreting this as a DIB "
                        "detection."
                    ),
                }
            )
    return diagnostics


def _nonstellar_frame_warnings(spectrum, overlaps, *, assumed_ism_rv_kms=None):
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
            reasons = []
            if "unknown" in {observer_frame, stellar_rest_status, wave_frame}:
                reasons.append("unknown_spectrum_frame_metadata")
            if stellar_rest_status == "corrected" and assumed_ism_rv_kms is None:
                reasons.append("stellar_rest_spectrum_without_ism_velocity")
            if reasons:
                ambiguous_segments.append(
                    {
                        "segment": name,
                        "observer_frame": observer_frame,
                        "stellar_rest_status": stellar_rest_status,
                        "wave_frame": wave_frame,
                        "reasons": reasons,
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
                        "until the spectrum, stellar-rest correction, and ISM "
                        "velocity frames are known."
                    ),
                }
            )
    return warnings


def diagnose_known_residual_windows(
    spectrum,
    result,
    *,
    enabled=True,
    threshold_sigma=2.5,
    windows=KNOWN_RESIDUAL_WINDOWS,
    verbose=False,
):
    """Summarize residuals in curated known-feature diagnostic windows."""
    payload = {
        "enabled": bool(enabled),
        "threshold_sigma": float(threshold_sigma),
        "windows": [],
        "flagged_windows": [],
    }
    result.summary["known_residual_windows"] = payload
    if not enabled:
        return payload
    threshold = float(threshold_sigma)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_sigma must be finite and positive.")

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
        for window in windows:
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
            flagged = bool(abs(median_sigma) >= threshold or rms_sigma >= threshold)
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
        add_quality_flag(result, "known_line_region_residual")
        if any(item.get("absorption_like") for item in payload["flagged_windows"]):
            add_quality_flag(result, "diagnostic_line_contaminated")
            add_quality_flag(result, "dib_overlap_balmer_wing")
            add_quality_flag(result, "dib_candidate_detected")
        if verbose:
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
