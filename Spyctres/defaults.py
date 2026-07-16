"""Conservative first-pass PHOENIX fitting defaults.

These helpers are intentionally rule-based and transparent. They do not try to
classify the star before fitting; they choose a practical first wavelength
window, broad-but-bounded PHOENIX parameter ranges, and a modest search budget
from spectrum metadata and wavelength coverage. Every assumption is returned in
provenance so callers can display or override it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import SpectrumCollection, SpectrumSegment, coerce_spectrum
from .preprocessing import (
    OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES,
    nonstellar_feature_metadata,
)


@dataclass(frozen=True)
class PhoenixFitDefaults:
    """Suggested PHOENIX fit configuration with auditable provenance."""

    fit_kwargs: dict
    provenance: dict
    reasons: tuple = ()
    warnings: tuple = ()

    def to_dict(self):
        return {
            "fit_kwargs": _jsonable(self.fit_kwargs),
            "provenance": _jsonable(self.provenance),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def spectrum_wavelength_range(spectrum):
    """Return the finite, mask-valid wavelength range of a spectrum.

    The mask convention follows the Spyctres container contract:
    ``True`` means a pixel is valid/usable. This helper is intentionally small
    because it is used by command-line examples to fill in a missing window edge
    when the user supplies only ``--wmin`` or ``--wmax``.
    """
    segments = _as_segments(spectrum)
    waves = []
    for segment in segments:
        wave = _segment_valid_wave(segment)
        if wave.size:
            waves.append(wave)
    if not waves:
        raise ValueError("No finite valid wavelengths are available.")
    merged = np.concatenate(waves)
    return float(np.min(merged)), float(np.max(merged))


def clip_grid_to_bounds(values, lower, upper):
    """Clip a sparse initial-search grid to scalar lower/upper bounds."""
    lower = float(lower)
    upper = float(upper)
    if upper <= lower:
        raise ValueError("Grid upper bound must be greater than lower bound.")
    clipped = [float(value) for value in values if lower <= float(value) <= upper]
    if clipped:
        return clipped
    return [0.5 * (lower + upper)]


def prepare_phoenix_fit_kwargs(
    spectrum,
    *,
    auto_defaults=True,
    defaults_mode="quicklook",
    science_case="classification",
    fallback_p0=(5750.0, 0.0, 4.5, 0.0),
    fallback_bounds=((4500.0, -1.5, 2.5, -300.0), (10000.0, 0.5, 5.5, 300.0)),
    p0_overrides=None,
    lower_bound_overrides=None,
    upper_bound_overrides=None,
    window=None,
    resolution_R=None,
    extra_kwargs=None,
):
    """Build PHOENIX fit keyword arguments from defaults plus overrides.

    This is the small reusable layer used by examples and smoke tests. It keeps
    automatic first-pass choices auditable while preserving expert control:
    every value supplied in an override wins over the suggestion.

    Returns
    -------
    fit_kwargs : dict
        Keyword arguments suitable for ``fit_phoenix_spectrum()`` and, after
        optional caller-specific additions such as explicit PHOENIX subgrids,
        ``fit_phoenix_full_spectrum()``.
    suggestion : PhoenixFitDefaults or None
        The underlying suggestion object when ``auto_defaults=True``.
    """
    suggestion = None
    if auto_defaults:
        suggestion = suggest_phoenix_fit_defaults(
            spectrum,
            mode=defaults_mode,
            science_case=science_case,
        )
        fit_kwargs = dict(suggestion.fit_kwargs)
    else:
        fit_kwargs = {
            "p0": tuple(float(value) for value in fallback_p0),
            "bounds": (
                tuple(float(value) for value in fallback_bounds[0]),
                tuple(float(value) for value in fallback_bounds[1]),
            ),
            "forward_model": "native_interp",
            "rv_init": "grid",
            "rv_grid_n": 41,
            "mdeg": 2,
        }

    if extra_kwargs:
        fit_kwargs.update(dict(extra_kwargs))

    p0 = list(fit_kwargs.get("p0", fallback_p0))
    for index, value in enumerate(p0_overrides or (None, None, None, None)):
        if value is not None:
            p0[index] = float(value)
    fit_kwargs["p0"] = tuple(p0)

    bounds = fit_kwargs.get("bounds", fallback_bounds)
    lower = list(bounds[0])
    upper = list(bounds[1])
    for index, value in enumerate(lower_bound_overrides or (None, None, None, None)):
        if value is not None:
            lower[index] = float(value)
    for index, value in enumerate(upper_bound_overrides or (None, None, None, None)):
        if value is not None:
            upper[index] = float(value)
    if any(hi <= lo for lo, hi in zip(lower, upper)):
        raise ValueError("Fit bounds must have min < max for every parameter.")
    fit_kwargs["bounds"] = (tuple(lower), tuple(upper))

    if window is not None:
        requested_lo, requested_hi = window
        data_lo, data_hi = spectrum_wavelength_range(spectrum)
        existing = fit_kwargs.get("regions", [(data_lo, data_hi)])
        base_lo, base_hi = existing[0]
        wmin = float(requested_lo) if requested_lo is not None else float(base_lo)
        wmax = float(requested_hi) if requested_hi is not None else float(base_hi)
        if wmax <= wmin:
            raise ValueError("Fit-window maximum must be greater than minimum.")
        fit_kwargs["regions"] = [(wmin, wmax)]

    if "coarse_teff_grid" in fit_kwargs:
        fit_kwargs["coarse_teff_grid"] = clip_grid_to_bounds(
            fit_kwargs["coarse_teff_grid"], lower[0], upper[0]
        )
    if "coarse_feh_grid" in fit_kwargs:
        fit_kwargs["coarse_feh_grid"] = clip_grid_to_bounds(
            fit_kwargs["coarse_feh_grid"], lower[1], upper[1]
        )
    if "coarse_logg_grid" in fit_kwargs:
        fit_kwargs["coarse_logg_grid"] = clip_grid_to_bounds(
            fit_kwargs["coarse_logg_grid"], lower[2], upper[2]
        )

    if resolution_R is not None:
        fit_kwargs["R"] = float(resolution_R)
    return fit_kwargs, suggestion


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_segments(spectrum):
    if isinstance(spectrum, SpectrumSegment):
        return [spectrum]
    if isinstance(spectrum, SpectrumCollection):
        return list(spectrum.segments)
    coerced = coerce_spectrum(spectrum, warn_unknown=False)
    if isinstance(coerced, SpectrumCollection):
        return list(coerced.segments)
    return [coerced]


def _segment_valid_wave(segment):
    wave = np.asarray(segment.wave, dtype=float)
    mask = np.asarray(segment.mask, dtype=bool)
    good = mask & np.isfinite(wave) & (wave > 0.0)
    return wave[good]


def _coverage_summary(segments):
    ranges = []
    instruments = []
    arms = []
    media = []
    observer_frames = []
    stellar_rest = []
    has_errors = []
    has_resolution = []
    for segment in segments:
        wave = _segment_valid_wave(segment)
        if wave.size:
            ranges.append((float(np.min(wave)), float(np.max(wave))))
        meta = getattr(segment, "meta", {}) or {}
        instrument = meta.get("instrument", None)
        if instrument is not None:
            instruments.append(str(instrument))
        arm = meta.get("arm", None)
        if arm is not None:
            arms.append(str(arm))
        media.append(str(getattr(segment, "wave_medium", "unknown")).lower())
        observer_frames.append(str(getattr(segment, "observer_frame", "unknown")).lower())
        stellar_rest.append(
            str(getattr(segment, "stellar_rest_status", "unknown")).lower()
        )
        has_errors.append(segment.err is not None)
        has_resolution.append(getattr(segment, "resolution", None) is not None)
    if not ranges:
        raise ValueError("No valid wavelengths are available for default selection.")
    return {
        "ranges": ranges,
        "wave_min": float(min(lo for lo, _hi in ranges)),
        "wave_max": float(max(hi for _lo, hi in ranges)),
        "instruments": sorted(set(instruments)),
        "arms": sorted(set(arms)),
        "wave_media": sorted(set(media)),
        "observer_frames": sorted(set(observer_frames)),
        "stellar_rest_status": sorted(set(stellar_rest)),
        "all_have_errors": bool(all(has_errors)),
        "all_have_resolution": bool(all(has_resolution)),
    }


def _overlap_width(ranges, lo, hi):
    width = 0.0
    for seg_lo, seg_hi in ranges:
        width += max(0.0, min(float(hi), seg_hi) - max(float(lo), seg_lo))
    return float(width)


def _choose_window(summary, mode):
    instruments = " ".join(summary["instruments"]).lower()
    arms = " ".join(summary["arms"]).lower()
    ranges = summary["ranges"]

    if "floyds" in instruments:
        candidates = [("floyds_blue_optical", 4000.0, 5200.0, 500.0)]
    elif "pepsi" in instruments:
        # PEPSI products often cover narrow orders. Use the available overlap
        # rather than pretending a broad classification window is present.
        candidates = []
    elif "uvb" in arms or "xshooter" in instruments or "x-shooter" in instruments:
        candidates = [
            ("blue_optical_classification", 3800.0, 5200.0, 500.0),
            ("optical_classification", 4000.0, 7000.0, 800.0),
        ]
    else:
        candidates = [
            ("blue_optical_classification", 3800.0, 5200.0, 500.0),
            ("red_optical_classification", 5200.0, 7000.0, 500.0),
            ("optical_classification", 4000.0, 7000.0, 800.0),
            ("near_ir_diagnostic", 7000.0, 9000.0, 500.0),
        ]

    if mode == "standard":
        candidates = [
            ("standard_optical_classification", 3800.0, 7000.0, 1000.0),
        ] + candidates

    for label, lo, hi, min_width in candidates:
        width = _overlap_width(ranges, lo, hi)
        if width >= min_width:
            return {
                "label": label,
                "regions": [(float(lo), float(hi))],
                "overlap_A": float(width),
                "source": "coverage_rule",
            }

    data_lo = float(summary["wave_min"])
    data_hi = float(summary["wave_max"])
    return {
        "label": "available_wavelength_range",
        "regions": [(data_lo, data_hi)],
        "overlap_A": float(max(0.0, data_hi - data_lo)),
        "source": "fallback_to_available_range",
    }


def _parameter_defaults(window_label, mode):
    if "red" in window_label or "near_ir" in window_label:
        p0 = (5500.0, 0.0, 4.0, 0.0)
        teff_bounds = (3500.0, 8000.0)
        teff_targets = (4000.0, 5000.0, 6000.0, 7000.0)
    else:
        p0 = (6000.0, 0.0, 4.0, 0.0)
        teff_bounds = (4500.0, 10000.0)
        teff_targets = (5000.0, 6000.0, 7500.0, 9000.0)

    if mode == "diagnostic":
        teff_bounds = (max(3000.0, teff_bounds[0] - 500.0), min(12000.0, teff_bounds[1] + 1500.0))
        teff_targets = tuple(sorted(set(teff_targets + (11000.0,))))

    feh_bounds = (-1.5, 0.5)
    logg_bounds = (2.5, 5.5)
    return {
        "p0": p0,
        "bounds": (
            (teff_bounds[0], feh_bounds[0], logg_bounds[0], -300.0),
            (teff_bounds[1], feh_bounds[1], logg_bounds[1], 300.0),
        ),
        "coarse_teff_grid": [
            value for value in teff_targets if teff_bounds[0] <= value <= teff_bounds[1]
        ],
        "coarse_feh_grid": [-1.0, 0.0, 0.5],
        "coarse_logg_grid": [3.0, 4.0, 5.0],
    }


def _telluric_catalog_overlaps(regions):
    """Return broad telluric catalog features overlapping suggested regions."""
    overlaps = []
    for meta in nonstellar_feature_metadata(OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES):
        feature_lo, feature_hi = meta["region_A"]
        overlap_A = 0.0
        for region_lo, region_hi in regions:
            overlap_A += max(
                0.0,
                min(float(feature_hi), float(region_hi))
                - max(float(feature_lo), float(region_lo)),
            )
        if overlap_A > 0.0:
            item = dict(meta)
            item["overlap_A"] = float(overlap_A)
            item["default_action"] = "warn_only"
            item["masking_note"] = (
                "Broad catalog telluric regions are for warning/provenance. "
                "Use telluric_transmission_exclusion_mask() for opt-in "
                "high-resolution transmission-threshold masking."
            )
            overlaps.append(item)
    return overlaps


def _mode_policy(mode):
    """Return the user-facing contract for a first-pass defaults mode."""
    mode = str(mode)
    if mode == "quicklook":
        return {
            "mode": "quicklook",
            "fit_stage": "triage_first_pass",
            "search_budget": "light",
            "default_status": "quicklook_only_until_quality_review",
            "final_science_ready_by_default": False,
            "description": (
                "Fast conservative first-pass settings intended to locate a "
                "reasonable parameter region, not to exhaustively validate the "
                "stellar classification."
            ),
            "recommended_followup": (
                "Inspect the quality report and plots; for important targets, "
                "rerun with standard settings or explicit expert bounds."
            ),
        }
    if mode == "standard":
        return {
            "mode": "standard",
            "fit_stage": "ordinary_first_pass",
            "search_budget": "moderate",
            "default_status": "first_pass_classification",
            "final_science_ready_by_default": False,
            "description": (
                "Moderate first-pass settings for ordinary classification "
                "after metadata, masks, and resolution assumptions have been "
                "reviewed."
            ),
            "recommended_followup": (
                "Inspect quality flags, residuals, and model-domain warnings "
                "before treating parameters as science-ready."
            ),
        }
    return {
        "mode": "diagnostic",
        "fit_stage": "stress_or_debug_run",
        "search_budget": "wider",
        "default_status": "diagnostic_not_for_ordinary_statistics",
        "final_science_ready_by_default": False,
        "description": (
            "Wider diagnostic settings for stress tests or suspicious spectra; "
            "results should be interpreted as debugging evidence."
        ),
        "recommended_followup": (
            "Use this to understand failures or edge cases; do not fold it "
            "into ordinary validation statistics without explicit review."
        ),
    }


def _interpretation_policy(summary, window, mode, science_case, telluric_overlaps):
    """Return a compact, machine-readable interpretation of the defaults.

    This block is intended for examples, notebooks, and future GUI displays.
    It does not change fit behavior; it names how conservative automatic
    defaults should be interpreted before the user inspects the final quality
    report.
    """
    risk_flags = []
    if "unknown" in summary["wave_media"]:
        risk_flags.append("unknown_wave_medium")
    if "unknown" in summary["observer_frames"]:
        risk_flags.append("unknown_observer_frame")
    if "unknown" in summary["stellar_rest_status"]:
        risk_flags.append("stellar_rest_status_unknown")
    if not summary["all_have_errors"]:
        risk_flags.append("missing_uncertainties")
    if not summary["all_have_resolution"]:
        risk_flags.append("missing_resolution")
    if window["overlap_A"] < 300.0:
        risk_flags.append("narrow_wavelength_window")
    if telluric_overlaps:
        risk_flags.append("broad_telluric_catalog_overlap")

    stellar_rest_values = set(summary["stellar_rest_status"])
    observer_values = set(summary["observer_frames"])
    medium_values = set(summary["wave_media"])
    metadata_complete = (
        "unknown" not in medium_values
        and "unknown" not in observer_values
        and "unknown" not in stellar_rest_values
    )
    if stellar_rest_values == {"corrected"}:
        rv_role = "rest_frame_consistency_check"
        rv_note = (
            "The loaded spectrum is labelled stellar-rest corrected, so fitted "
            "rv_kms should be interpreted as a residual alignment/check, not a "
            "new stellar radial-velocity measurement."
        )
    elif metadata_complete and observer_values <= {"barycentric", "heliocentric"}:
        rv_role = "candidate_stellar_rv"
        rv_note = (
            "The wavelength metadata are sufficiently explicit for rv_kms to "
            "be treated as a candidate stellar RV, subject to the final quality "
            "flags and product-specific frame validation."
        )
    else:
        rv_role = "alignment_parameter_until_metadata_verified"
        rv_note = (
            "The fitted rv_kms should be treated as a model/data alignment "
            "parameter until wavelength medium, observer frame, and stellar-rest "
            "semantics are verified."
        )

    mode_policy = _mode_policy(mode)
    if mode == "quicklook":
        recommended_next_step = (
            "Inspect the quality report and diagnostic plots; rerun with "
            "mode='standard' or expert bounds if the target is scientifically "
            "important or the quicklook flags are non-trivial."
        )
    elif mode == "standard":
        recommended_next_step = (
            "Use the result as an ordinary first-pass classification only after "
            "checking quality flags, residual structure, and model-domain flags."
        )
    else:
        recommended_next_step = (
            "Treat this as a diagnostic/stress run. It deliberately widens the "
            "search and should not be folded into ordinary validation statistics."
        )

    return {
        "intended_use": "first_pass_{0}".format(science_case or "classification"),
        "mode": mode,
        "classification_scope": (
            "Line/shape-based PHOENIX classification over the selected window; "
            "not a precision abundance, rotation, macroturbulence, or detailed "
            "instrument-LSF analysis."
        ),
        "rv_role": rv_role,
        "rv_note": rv_note,
        "automatic_choices_are_overridable": True,
        "mode_policy": mode_policy,
        "final_science_ready_by_default": mode_policy[
            "final_science_ready_by_default"
        ],
        "risk_flags": sorted(set(risk_flags)),
        "recommended_next_step": recommended_next_step,
    }


def suggest_phoenix_fit_defaults(
    spectrum,
    mode="quicklook",
    science_case="classification",
):
    """Suggest conservative PHOENIX fit keyword arguments for a loaded spectrum.

    Parameters
    ----------
    spectrum : SpectrumSegment, SpectrumCollection, or coercible spectrum input
        Spectrum already loaded into Spyctres' common format.
    mode : {"quicklook", "standard", "diagnostic"}
        Controls the initial search budget. ``quicklook`` is intentionally
        modest for first-run examples; ``standard`` uses a slightly stronger
        multistart; ``diagnostic`` widens the Teff range.
    science_case : str
        Currently only ``"classification"`` has special handling; the value is
        still recorded in provenance for future extension.

    Returns
    -------
    PhoenixFitDefaults
        Contains ``fit_kwargs`` plus reasons/warnings explaining each
        assumption. Expert callers can edit or override any returned keyword.
    """
    mode = str(mode).strip().lower()
    if mode not in {"quicklook", "standard", "diagnostic"}:
        raise ValueError("mode must be quicklook, standard, or diagnostic.")
    science_case = str(science_case).strip().lower()

    segments = _as_segments(spectrum)
    summary = _coverage_summary(segments)
    window = _choose_window(summary, mode)
    params = _parameter_defaults(window["label"], mode)
    telluric_overlaps = _telluric_catalog_overlaps(window["regions"])
    interpretation = _interpretation_policy(
        summary,
        window,
        mode,
        science_case,
        telluric_overlaps,
    )

    multistart = 1 if mode == "quicklook" else 2
    rv_grid_n = 41 if mode == "quicklook" else 61
    if set(summary["stellar_rest_status"]) == {"corrected"}:
        rv_grid_n = min(rv_grid_n, 21)
    mode_policy = dict(interpretation["mode_policy"])
    mode_policy["multistart"] = int(multistart)
    mode_policy["rv_grid_n"] = int(rv_grid_n)

    reasons = [
        "selected {0} window {1} because it overlaps the loaded wavelength coverage by {2:.1f} A".format(
            window["label"],
            window["regions"],
            window["overlap_A"],
        ),
        "using {0} defaults mode with {1} search budget".format(
            mode,
            mode_policy["search_budget"],
        ),
        "using broad PHOENIX classification bounds rather than the full grid",
        "using native_interp so RV shifting and LSF convolution happen before final resampling",
    ]
    warnings = []
    if "unknown" in summary["wave_media"]:
        warnings.append(
            "wavelength medium is unknown for at least one segment; verify air/vacuum assumptions"
        )
    if "unknown" in summary["observer_frames"]:
        warnings.append(
            "observer-motion frame is unknown for at least one segment; do not apply barycentric corrections blindly"
        )
    if "unknown" in summary["stellar_rest_status"]:
        warnings.append(
            "stellar rest-frame status is unknown; fitted rv_kms needs metadata review"
        )
    if not summary["all_have_errors"]:
        warnings.append(
            "at least one segment lacks formal uncertainties; the fitter will estimate fallback errors"
        )
    if not summary["all_have_resolution"]:
        warnings.append(
            "at least one segment lacks resolution metadata; no instrumental broadening may be applied there"
        )
    if window["overlap_A"] < 300.0:
        warnings.append(
            "selected wavelength span is narrow; treat atmospheric parameters as local diagnostics"
        )
    if telluric_overlaps:
        warnings.append(
            "suggested fit window overlaps broad topocentric telluric catalog "
            "regions; these are warning/provenance regions, not the preferred "
            "actual mask. Use telluric_transmission_exclusion_mask() if "
            "telluric masking is explicitly desired and the wavelength frame "
            "is suitable."
        )

    fit_kwargs = {
        "p0": params["p0"],
        "bounds": params["bounds"],
        "regions": window["regions"],
        "forward_model": "native_interp",
        "physical_init": "coarse",
        "coarse_teff_grid": params["coarse_teff_grid"],
        "coarse_feh_grid": params["coarse_feh_grid"],
        "coarse_logg_grid": params["coarse_logg_grid"],
        "coarse_decimate": 12,
        "multistart": multistart,
        "rv_init": "grid",
        "rv_grid_n": rv_grid_n,
        "mdeg": 2,
    }

    provenance = {
        "operation": "suggest_phoenix_fit_defaults",
        "mode": mode,
        "science_case": science_case,
        "coverage": summary,
        "window": window,
        "telluric_catalog_policy": {
            "default_action": "warn_only",
            "feature_source": "broad_catalog_regions",
            "actual_masking_preference": "transmission_threshold",
            "recommended_helper": "telluric_transmission_exclusion_mask",
            "overlaps": telluric_overlaps,
        },
        "interpretation": interpretation,
        "mode_policy": mode_policy,
        "assumption_policy": "metadata-backed where available; otherwise conservative and provenance-recorded",
    }
    return PhoenixFitDefaults(
        fit_kwargs=fit_kwargs,
        provenance=provenance,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
