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

    multistart = 1 if mode == "quicklook" else 2
    rv_grid_n = 41 if mode == "quicklook" else 61
    if set(summary["stellar_rest_status"]) == {"corrected"}:
        rv_grid_n = min(rv_grid_n, 21)

    reasons = [
        "selected {0} window {1} because it overlaps the loaded wavelength coverage by {2:.1f} A".format(
            window["label"],
            window["regions"],
            window["overlap_A"],
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
        "assumption_policy": "metadata-backed where available; otherwise conservative and provenance-recorded",
    }
    return PhoenixFitDefaults(
        fit_kwargs=fit_kwargs,
        provenance=provenance,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
