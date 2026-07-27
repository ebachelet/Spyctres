"""Conservative first-pass PHOENIX fitting defaults.

These helpers are intentionally rule-based and transparent. They do not try to
classify the star before fitting; they choose practical first wavelength
windows, broad-but-bounded PHOENIX parameter ranges, and a modest search budget
from spectrum metadata and wavelength coverage. The branch-aware suggestions are
feature-coverage triage, not a hidden spectral-type label. Every assumption is
returned in provenance so callers can display or override it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._serialization import json_safe
from ._spectrum_helpers import spectrum_segments
from .diagnostic_windows import (
    build_diagnostic_window_combinations,
    select_diagnostic_windows,
)
from .io import SpectrumCollection
from .preprocessing import (
    OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES,
    audit_spectrum_for_fit,
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


def _stable_json(value):
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_payload(payload):
    payload = dict(_jsonable(payload))
    payload.pop("setup_hash", None)
    payload.pop("configuration_hash", None)
    encoded = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FitSetup(Mapping):
    """Reviewed first-pass fitting setup with mapping compatibility.

    ``FitSetup`` is the compact object returned by :func:`suggest_fit_setup`.
    It behaves like a read-only mapping for compatibility with existing
    notebooks and scripts, while adding a stable ``setup_hash`` plus helper
    methods for user-facing summaries and JSON serialization.
    """

    payload: dict

    def __post_init__(self):
        payload = _jsonable(dict(self.payload))
        setup_hash = payload.get("setup_hash") or _hash_payload(payload)
        payload["setup_hash"] = str(setup_hash)
        payload["configuration_hash"] = str(setup_hash)
        object.__setattr__(self, "payload", payload)

    def __getitem__(self, key):
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self):
        return len(self.payload)

    @property
    def setup_hash(self):
        return self.payload["setup_hash"]

    @property
    def fit_kwargs(self):
        return dict(self.payload.get("fit_kwargs") or {})

    @property
    def readiness(self):
        return self.payload.get("readiness")

    def summary(self, max_actions=3):
        """Return a compact JSON-safe user-facing setup summary."""
        readiness = self.payload.get("readiness") or {}
        window = self.payload.get("recommended_window") or {}
        actions = list(readiness.get("actions_for_intent") or ())[: int(max_actions)]
        warnings = list(readiness.get("warnings_for_intent") or ())[: int(max_actions)]
        return _jsonable(
            {
                "schema_version": 1,
                "setup_hash": self.setup_hash,
                "model": self.payload.get("model"),
                "mode": self.payload.get("mode"),
                "science_case": self.payload.get("science_case"),
                "recommended_window_label": window.get("label"),
                "recommended_regions_A": window.get("regions"),
                "recommended_branch_id": self.payload.get("recommended_branch_id"),
                "readiness_intent": readiness.get("intent"),
                "ready_for_intent": readiness.get("ready_for_intent"),
                "blockers_for_intent": readiness.get("blockers_for_intent") or [],
                "warnings_for_intent": warnings,
                "top_actions": actions,
                "risk_flags": list(self.payload.get("risk_flags") or []),
                "next_steps": list(self.payload.get("next_steps") or [])[
                    : int(max_actions)
                ],
            }
        )

    def summary_text(self, max_actions=3):
        """Return a compact plain-text setup summary for notebooks/CLI use."""
        summary = self.summary(max_actions=max_actions)
        lines = [
            "Spyctres fit setup",
            "  hash: {0}".format(summary.get("setup_hash")),
            "  model/mode: {0}/{1}".format(
                summary.get("model"),
                summary.get("mode"),
            ),
        ]
        if summary.get("recommended_window_label"):
            lines.append(
                "  window: {0} {1}".format(
                    summary.get("recommended_window_label"),
                    summary.get("recommended_regions_A"),
                )
            )
        lines.append(
            "  readiness: intent={0}, ready={1}".format(
                summary.get("readiness_intent"),
                summary.get("ready_for_intent"),
            )
        )
        if summary.get("blockers_for_intent"):
            lines.append(
                "  blockers: {0}".format(
                    ", ".join(summary["blockers_for_intent"])
                )
            )
        if summary.get("warnings_for_intent"):
            lines.append(
                "  warnings: {0}".format(
                    ", ".join(summary["warnings_for_intent"])
                )
            )
        actions = []
        for item in summary.get("top_actions") or []:
            if isinstance(item, Mapping) and item.get("flag") and item.get("action"):
                actions.append("{0}: {1}".format(item["flag"], item["action"]))
        if actions:
            lines.append("  top actions:")
            lines.extend("    - {0}".format(item) for item in actions)
        return "\n".join(lines)

    def to_dict(self):
        return _jsonable(self.payload)

    def to_json(self, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return json.dumps(self.to_dict(), **kwargs)


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
    if isinstance(value, FitSetup):
        return value.to_dict()
    return json_safe(value)


def _as_segments(spectrum):
    return spectrum_segments(
        spectrum,
        tuple_is_collection=False,
        coerce=True,
        warn_unknown=False,
    )


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


CLASSIFICATION_BRANCH_PROFILES = (
    {
        "id": "blue_optical_balmer_metal",
        "label": "Blue optical Balmer/metal classification",
        "description": (
            "A mixed blue-optical branch using Balmer wings plus Ca/Mg/metal "
            "features. This is the safest first-pass branch when the spectrum "
            "covers the classical 3900-5200 A classification region but the "
            "stellar type is not yet known."
        ),
        "window_ids": (
            "ca_hk_h_epsilon",
            "h_delta",
            "ca_i_4227",
            "ch_g_band",
            "h_gamma",
            "h_beta",
            "mg_i_b",
        ),
        "fit_window_ids": (
            "ca_hk_h_epsilon",
            "h_delta",
            "ca_i_4227",
            "h_gamma",
            "h_beta",
            "mg_i_b",
        ),
        "min_windows": 3,
        "min_total_overlap_A": 140.0,
        "max_fit_windows": 6,
        "priority": 1.20,
        "p0": (7000.0, 0.0, 4.0, 0.0),
        "teff_bounds": (4500.0, 12000.0),
        "teff_grid": (5000.0, 6000.0, 7500.0, 9000.0, 11000.0),
        "feh_bounds": (-1.5, 0.5),
        "logg_bounds": (2.5, 5.5),
        "coarse_feh_grid": (-1.0, 0.0, 0.5),
        "coarse_logg_grid": (3.0, 4.0, 5.0),
        "ordinary_default": True,
    },
    {
        "id": "hot_blue_hydrogen_stress",
        "label": "Hot/intermediate blue hydrogen stress check",
        "description": (
            "Balmer plus He/Mg hot-star indicators. Useful as a follow-up "
            "branch, but not the ordinary default because PHOENIX LTE models "
            "near the hot end can miss NLTE-sensitive H/He physics."
        ),
        "window_ids": (
            "ca_hk_h_epsilon",
            "h_delta",
            "h_gamma",
            "he_i_4471",
            "mg_ii_4481",
            "si_ii_4128_4130",
            "h_beta",
        ),
        "fit_window_ids": ("h_delta", "h_gamma", "h_beta"),
        "min_windows": 2,
        "min_total_overlap_A": 90.0,
        "max_fit_windows": 3,
        "priority": 0.78,
        "p0": (9000.0, 0.0, 3.5, 0.0),
        "teff_bounds": (6500.0, 12000.0),
        "teff_grid": (7000.0, 9000.0, 11000.0),
        "feh_bounds": (-1.0, 0.5),
        "logg_bounds": (2.0, 5.0),
        "coarse_feh_grid": (-0.5, 0.0, 0.5),
        "coarse_logg_grid": (2.5, 3.5, 4.5),
        "ordinary_default": False,
        "risk_tags": ("phoenix_hot_star_limit", "nlte_sensitive"),
    },
    {
        "id": "fgk_optical_balmer_metal",
        "label": "F/G/K optical Balmer/metal branch",
        "description": (
            "Optical Balmer and atomic/molecular-metal windows for ordinary "
            "F/G/K-like spectra when the blue or red optical region is present."
        ),
        "window_ids": (
            "ca_hk_h_epsilon",
            "ca_i_4227",
            "ch_g_band",
            "h_gamma",
            "h_beta",
            "mg_i_b",
            "na_i_d",
            "h_alpha",
        ),
        "fit_window_ids": (
            "ca_i_4227",
            "h_gamma",
            "h_beta",
            "mg_i_b",
            "na_i_d",
            "h_alpha",
        ),
        "min_windows": 2,
        "min_total_overlap_A": 90.0,
        "max_fit_windows": 5,
        "priority": 0.92,
        "p0": (5750.0, 0.0, 4.3, 0.0),
        "teff_bounds": (4300.0, 8000.0),
        "teff_grid": (4500.0, 5500.0, 6500.0, 7500.0),
        "feh_bounds": (-1.5, 0.5),
        "logg_bounds": (2.5, 5.5),
        "coarse_feh_grid": (-1.0, 0.0, 0.5),
        "coarse_logg_grid": (3.0, 4.0, 5.0),
        "ordinary_default": True,
    },
    {
        "id": "cool_red_optical_molecular",
        "label": "Cool red-optical molecular/alkali branch",
        "description": (
            "Red-optical TiO/VO/alkali/Ca-triplet windows for late-K/M or "
            "other cool spectra. These are especially useful when the blue "
            "Balmer region is absent or low-S/N."
        ),
        "window_ids": (
            "mg_i_b",
            "na_i_d",
            "h_alpha",
            "tio_7050",
            "vo_7450",
            "k_i_7700",
            "vo_7900",
            "na_i_8200",
            "ca_ii_triplet_paschen",
            "feh_8700",
            "tio_red_bands",
        ),
        "fit_window_ids": (
            "mg_i_b",
            "h_alpha",
            "tio_7050",
            "na_i_8200",
            "ca_ii_triplet_paschen",
            "tio_red_bands",
        ),
        "min_windows": 2,
        "min_total_overlap_A": 140.0,
        "max_fit_windows": 5,
        "priority": 1.05,
        "p0": (4000.0, 0.0, 4.0, 0.0),
        "teff_bounds": (3000.0, 6500.0),
        "teff_grid": (3200.0, 4000.0, 5000.0, 6000.0),
        "feh_bounds": (-1.5, 0.5),
        "logg_bounds": (1.0, 5.5),
        "coarse_feh_grid": (-1.0, 0.0, 0.5),
        "coarse_logg_grid": (1.5, 3.0, 4.5),
        "ordinary_default": True,
        "risk_tags": ("molecular_model_sensitive", "telluric_sensitive"),
    },
    {
        "id": "near_ir_hydrogen",
        "label": "Near-IR Paschen/Brackett hydrogen branch",
        "description": (
            "Paschen and Brackett windows for hot/intermediate spectra with "
            "near-IR coverage. Treat this as a hydrogen-line classification "
            "branch, not an abundance or detailed LSF solution."
        ),
        "window_ids": (
            "ca_ii_triplet_paschen",
            "paschen_gamma_delta",
            "paschen_beta",
            "brackett_h_band",
            "br_gamma",
        ),
        "fit_window_ids": (
            "ca_ii_triplet_paschen",
            "paschen_gamma_delta",
            "paschen_beta",
            "brackett_h_band",
            "br_gamma",
        ),
        "min_windows": 1,
        "min_total_overlap_A": 120.0,
        "max_fit_windows": 4,
        "priority": 0.96,
        "p0": (8500.0, 0.0, 3.5, 0.0),
        "teff_bounds": (6000.0, 12000.0),
        "teff_grid": (6500.0, 8000.0, 10000.0, 11500.0),
        "feh_bounds": (-1.0, 0.5),
        "logg_bounds": (2.0, 5.0),
        "coarse_feh_grid": (-0.5, 0.0, 0.5),
        "coarse_logg_grid": (2.5, 3.5, 4.5),
        "ordinary_default": True,
        "risk_tags": ("telluric_sensitive",),
    },
    {
        "id": "cool_near_ir_molecular",
        "label": "Cool near-IR atomic/CO branch",
        "description": (
            "K-band Na I, Ca I, and CO bandhead windows for cool-star "
            "near-IR classification when those features are covered."
        ),
        "window_ids": (
            "ca_ii_triplet_paschen",
            "feh_8700",
            "tio_red_bands",
            "na_i_kband",
            "ca_i_kband",
            "co_23um_bandhead",
        ),
        "fit_window_ids": (
            "ca_ii_triplet_paschen",
            "tio_red_bands",
            "na_i_kband",
            "ca_i_kband",
            "co_23um_bandhead",
        ),
        "min_windows": 1,
        "min_total_overlap_A": 180.0,
        "max_fit_windows": 4,
        "priority": 1.08,
        "p0": (3800.0, 0.0, 3.0, 0.0),
        "teff_bounds": (2800.0, 6000.0),
        "teff_grid": (3000.0, 3800.0, 4800.0, 5800.0),
        "feh_bounds": (-1.5, 0.5),
        "logg_bounds": (0.5, 5.5),
        "coarse_feh_grid": (-1.0, 0.0, 0.5),
        "coarse_logg_grid": (1.0, 3.0, 5.0),
        "ordinary_default": True,
        "risk_tags": ("molecular_model_sensitive", "telluric_sensitive"),
    },
)


def _diagnostic_records_by_id(selection):
    return {
        str(record.get("id")): record
        for record in selection.get("selected", ())
        if isinstance(record, dict) and record.get("id") is not None
    }


def _record_fit_allowed(record):
    if str(record.get("default_fit_policy", "include")) == "exclude":
        return False
    return str(record.get("model_support", "supported")) not in {
        "unsupported",
        "stress_only",
    }


def _profile_parameters(profile, mode):
    teff_lo, teff_hi = profile["teff_bounds"]
    if mode == "diagnostic":
        teff_lo = max(2500.0, float(teff_lo) - 500.0)
        teff_hi = min(12000.0, float(teff_hi) + 1000.0)
    feh_lo, feh_hi = profile["feh_bounds"]
    logg_lo, logg_hi = profile["logg_bounds"]
    teff_grid = [
        float(value)
        for value in profile["teff_grid"]
        if float(teff_lo) <= float(value) <= float(teff_hi)
    ]
    if not teff_grid:
        teff_grid = [0.5 * (float(teff_lo) + float(teff_hi))]
    return {
        "p0": tuple(float(value) for value in profile["p0"]),
        "bounds": (
            (float(teff_lo), float(feh_lo), float(logg_lo), -300.0),
            (float(teff_hi), float(feh_hi), float(logg_hi), 300.0),
        ),
        "coarse_teff_grid": teff_grid,
        "coarse_feh_grid": [float(value) for value in profile["coarse_feh_grid"]],
        "coarse_logg_grid": [float(value) for value in profile["coarse_logg_grid"]],
    }


def _regions_from_records(records):
    regions = []
    for record in records:
        region = record.get("operational_region_A") or record.get("region_A")
        if region is None or len(region) != 2:
            continue
        lo, hi = float(region[0]), float(region[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            regions.append((lo, hi))
    return regions


def _branch_record(profile, selected_by_id, mode):
    matched = [
        selected_by_id[window_id]
        for window_id in profile["window_ids"]
        if window_id in selected_by_id
    ]
    fit_pool = [
        selected_by_id[window_id]
        for window_id in profile["fit_window_ids"]
        if window_id in selected_by_id and _record_fit_allowed(selected_by_id[window_id])
    ]
    max_fit_windows = int(profile["max_fit_windows"])
    ranked_fit_records = sorted(
        fit_pool,
        key=lambda item: (-float(item.get("score", 0.0)), float(item["region_A"][0])),
    )[:max_fit_windows]
    fit_records = sorted(
        ranked_fit_records,
        key=lambda item: float((item.get("operational_region_A") or item["region_A"])[0]),
    )
    fit_regions = _regions_from_records(fit_records)
    total_overlap = float(sum(float(item.get("overlap_A", 0.0)) for item in matched))
    raw_score = float(sum(float(item.get("score", 0.0)) for item in matched))
    coverage_gate = (
        len(matched) >= int(profile["min_windows"])
        and total_overlap >= float(profile["min_total_overlap_A"])
        and bool(fit_regions)
    )
    score = raw_score * float(profile["priority"])
    if not profile.get("ordinary_default", True):
        score *= 0.65
    status = "candidate" if coverage_gate else "insufficient_coverage"
    if not fit_regions and matched:
        status = "diagnostic_only"
    parameters = _profile_parameters(profile, mode)
    risk_tags = sorted(
        {
            str(tag)
            for item in matched
            for tag in (item.get("risk_tags") or ())
        }
        | {str(tag) for tag in profile.get("risk_tags", ())}
    )
    return {
        "id": profile["id"],
        "label": profile["label"],
        "description": profile["description"],
        "status": status,
        "score": float(score),
        "raw_window_score": float(raw_score),
        "ordinary_default": bool(profile.get("ordinary_default", True)),
        "matched_window_count": int(len(matched)),
        "required_window_count": int(profile["min_windows"]),
        "total_overlap_A": float(total_overlap),
        "required_overlap_A": float(profile["min_total_overlap_A"]),
        "matched_window_ids": [item["id"] for item in matched],
        "fit_window_ids": [item["id"] for item in fit_records],
        "fit_regions_A": [(float(lo), float(hi)) for lo, hi in fit_regions],
        "parameter_defaults": parameters,
        "risk_tags": risk_tags,
        "notes": (
            "Branch suggestions are feature-coverage triage. They do not prove "
            "the stellar type; compare branch stability and quality flags "
            "before promoting a result."
        ),
    }


def suggest_classification_branches(
    spectrum,
    *,
    selection=None,
    mode="quicklook",
    max_branches=None,
):
    """Suggest branch-aware first-pass classification region sets.

    The helper uses the existing diagnostic-window catalog. It does not run a
    fit and it does not infer the final spectral type. Instead, it answers the
    practical setup question: given the wavelength coverage and usable pixels,
    which hot/blue, F/G/K, cool/red, Paschen/Brackett, or CO/Ca/TiO-style
    branches are plausible enough to try first?
    """
    mode = str(mode).strip().lower()
    if mode not in {"quicklook", "standard", "diagnostic"}:
        raise ValueError("mode must be quicklook, standard, or diagnostic.")
    if selection is None:
        selection = select_diagnostic_windows(spectrum)
    selected_by_id = _diagnostic_records_by_id(selection)
    branches = [
        _branch_record(profile, selected_by_id, mode)
        for profile in CLASSIFICATION_BRANCH_PROFILES
    ]
    branches = sorted(
        branches,
        key=lambda item: (
            item["status"] != "candidate",
            -float(item["score"]),
            item["label"],
        ),
    )
    if max_branches is not None:
        max_branches = int(max_branches)
        if max_branches < 1:
            raise ValueError("max_branches must be >= 1 when supplied.")
        branches = branches[:max_branches]
    candidates = [item for item in branches if item["status"] == "candidate"]
    recommended = candidates[0] if candidates else None
    ambiguous = False
    if len(candidates) > 1 and recommended is not None:
        top_score = float(recommended["score"])
        second_score = float(candidates[1]["score"])
        ambiguous = bool(top_score > 0.0 and second_score >= 0.85 * top_score)
    default_action = (
        "use_recommended_branch_regions"
        if recommended is not None
        else "fallback_to_coverage_window"
    )
    return {
        "schema_version": 1,
        "operation": "suggest_classification_branches",
        "mode": mode,
        "selection_source": "diagnostic_window_catalog",
        "policy": {
            "fit_regions_are_diagnostic_windows": True,
            "branches_are_not_final_spectral_types": True,
            "stress_only_windows_do_not_drive_default_fit": True,
            "ambiguous_top_branches_are_reported": True,
            "default_action": default_action,
        },
        "recommended_branch_id": None if recommended is None else recommended["id"],
        "recommended_branch": recommended,
        "ambiguous_top_branches": ambiguous,
        "branches": branches,
    }


def _window_from_branch_plan(branch_plan):
    branch = branch_plan.get("recommended_branch")
    if not branch:
        return None
    regions = branch.get("fit_regions_A") or ()
    if not regions:
        return None
    overlap = sum(float(hi) - float(lo) for lo, hi in regions)
    return {
        "label": "classification_branch:{0}".format(branch["id"]),
        "regions": [(float(lo), float(hi)) for lo, hi in regions],
        "overlap_A": float(overlap),
        "source": "diagnostic_window_branch",
        "branch_id": branch["id"],
        "branch_label": branch["label"],
    }


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


def _parameter_defaults(window_label, mode, branch=None):
    if branch is not None:
        return dict(branch["parameter_defaults"])

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
    coverage_window = _choose_window(summary, mode)
    fallback_params = _parameter_defaults(coverage_window["label"], mode)
    diagnostic_spectrum = SpectrumCollection(segments) if len(segments) > 1 else segments[0]
    rv_padding = max(
        abs(fallback_params["bounds"][0][3]),
        abs(fallback_params["bounds"][1][3]),
    )
    diagnostic_selection = select_diagnostic_windows(
        diagnostic_spectrum,
        rv_kms=fallback_params["p0"][3],
        rv_padding_kms=rv_padding,
    )
    branch_plan = suggest_classification_branches(
        diagnostic_spectrum,
        selection=diagnostic_selection,
        mode=mode,
    )
    branch_window = _window_from_branch_plan(branch_plan)
    if branch_window is not None:
        window = branch_window
        params = _parameter_defaults(
            window["label"],
            mode,
            branch=branch_plan["recommended_branch"],
        )
    else:
        window = coverage_window
        params = fallback_params
    diagnostic_combinations = build_diagnostic_window_combinations(
        diagnostic_selection,
        max_windows=6 if mode == "quicklook" else 8,
        max_single_windows=4 if mode == "quicklook" else 6,
    )
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
        "identified {0} diagnostic feature window(s) from wavelength coverage for optional follow-up checks".format(
            len(diagnostic_selection["selected"])
        ),
    ]
    recommended_branch = branch_plan.get("recommended_branch")
    if recommended_branch is not None:
        reasons.append(
            "recommended branch-aware first-pass region set '{0}' with {1} fitted diagnostic window(s)".format(
                recommended_branch["label"],
                len(recommended_branch.get("fit_regions_A") or ()),
            )
        )
    else:
        reasons.append(
            "no diagnostic branch passed the minimum coverage checks; using the broad coverage fallback window"
        )
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
    if branch_plan.get("ambiguous_top_branches"):
        warnings.append(
            "multiple first-pass classification branches have similar support; compare branch-specific fits before trusting one branch"
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
        "diagnostic_windows": {
            "selection": diagnostic_selection,
            "recommended_combinations": diagnostic_combinations,
            "note": (
                "Diagnostic windows are scored from wavelength coverage, "
                "usable pixels, and a cheap contrast proxy. They are intended "
                "for follow-up sanity checks and are not automatically fitted "
                "as a blind all-combinations grid."
            ),
        },
        "classification_branches": branch_plan,
        "coverage_window_fallback": coverage_window,
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


def _setup_next_steps(suggestion, readiness):
    interpretation = suggestion.provenance.get("interpretation", {})
    next_steps = [
        "Review wavelength-medium, observer-frame, stellar-rest, and resolution metadata before interpreting rv_kms or line widths.",
        "Inspect diagnostic windows and residual plots before treating a first-pass classification as science-ready.",
    ]
    mode_step = interpretation.get("recommended_next_step")
    if mode_step:
        next_steps.append(str(mode_step))
    if readiness is not None:
        intent = readiness.get("intent") or readiness.get("intended_use") or "fit"
        ready_for_intent = readiness.get(
            "ready_for_intent",
            readiness.get("fit_ready", False),
        )
        if not ready_for_intent:
            flags = ", ".join(
                readiness.get("blockers_for_intent")
                or readiness.get("interpretation_flags", [])
                or ["unknown"]
            )
            next_steps.append(
                "Readiness audit is not fit-ready for {0}; resolve or "
                "explicitly accept these blockers before refinement: {1}.".format(
                    intent,
                    flags,
                )
            )
            for item in (
                readiness.get("actions_for_intent")
                or readiness.get("recommended_actions", [])
                or []
            ):
                flag = item.get("flag")
                action = item.get("action")
                if flag and action:
                    next_steps.append("{0}: {1}".format(flag, action))
        elif readiness.get("warnings_for_intent"):
            flags = ", ".join(readiness.get("warnings_for_intent") or ())
            next_steps.append(
                "Intent-specific audit is ready for {0}, but review these "
                "warnings before interpreting the result: {1}.".format(
                    intent,
                    flags,
                )
            )
        elif readiness.get("quicklook_only", False):
            next_steps.append(
                "Treat this setup as quicklook-only until the readiness flags and model residuals are reviewed."
            )
    return list(dict.fromkeys(next_steps))


def _assumed_resolution_fit_kwargs(assumed_resolution):
    if assumed_resolution is None:
        return None
    if isinstance(assumed_resolution, dict):
        quantity = str(assumed_resolution.get("quantity", "R"))
        value = assumed_resolution.get("value", assumed_resolution.get("R"))
    else:
        quantity = "R"
        value = assumed_resolution
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    if quantity == "R":
        return {"R": float(value)}
    if quantity in {"fwhm_kms", "sigma_kms"}:
        return {quantity: float(value)}
    return None


def _warnings_with_assumed_resolution(warnings, assumed_resolution):
    warnings = [str(item) for item in warnings]
    override = _assumed_resolution_fit_kwargs(assumed_resolution)
    if override is None:
        return warnings
    warnings = [
        item for item in warnings if "lacks resolution metadata" not in item
    ]
    key, value = next(iter(override.items()))
    if key == "R":
        message = (
            "using user-supplied assumed resolution R={0:g}; treat this as an "
            "approximate quicklook assumption unless the product documentation "
            "justifies it"
        ).format(value)
    else:
        message = (
            "using user-supplied assumed {0}={1:g} for setup/readiness; treat "
            "this as an approximate quicklook assumption unless documented"
        ).format(key, value)
    return warnings + [message]


def suggest_fit_setup(
    spectrum,
    *,
    model="phoenix",
    mode="quicklook",
    science_case="classification",
    readiness_intent=None,
    include_readiness=True,
    assumed_resolution=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
):
    """Return a compact first-pass setup recommendation for a loaded spectrum.

    This public helper is deliberately PHOENIX-free: it does not load templates,
    build caches, optimize parameters, or mutate the input spectrum.  It answers
    the setup question a user or GUI usually has immediately after ingestion:
    which wavelength windows, broad parameter ranges, diagnostic branches,
    resolution/frame warnings, and readiness flags should be reviewed before a
    fit is launched?

    The returned :class:`FitSetup` behaves like a read-only dictionary for
    compatibility, is JSON-safe via ``to_dict()``/``to_json()``, and is built
    from the canonical
    ``suggest_phoenix_fit_defaults()``, ``select_diagnostic_windows()``,
    ``suggest_classification_branches()``, and optional
    ``audit_spectrum_for_fit()`` layers. Expert users can still override every
    suggested fit keyword.
    """
    model = str(model).strip().lower()
    if model != "phoenix":
        raise ValueError("suggest_fit_setup currently supports model='phoenix'.")

    suggestion = suggest_phoenix_fit_defaults(
        spectrum,
        mode=mode,
        science_case=science_case,
    )
    readiness = None
    if include_readiness:
        readiness = audit_spectrum_for_fit(
            spectrum,
            regions=suggestion.fit_kwargs.get("regions"),
            exclude_mask=exclude_mask,
            exclude_masks=exclude_masks,
            mask_threshold=mask_threshold,
            intended_use=science_case,
            intent=readiness_intent,
            assumed_resolution=assumed_resolution,
        )

    fit_kwargs = dict(suggestion.fit_kwargs)
    assumed_lsf_kwargs = _assumed_resolution_fit_kwargs(assumed_resolution)
    if assumed_lsf_kwargs:
        fit_kwargs.update(assumed_lsf_kwargs)

    interpretation = suggestion.provenance.get("interpretation", {})
    branch_plan = suggestion.provenance.get("classification_branches", {})
    diagnostic_info = suggestion.provenance.get("diagnostic_windows", {})
    window = suggestion.provenance.get("window", {})
    readiness_flags = [] if readiness is None else readiness.get("interpretation_flags", [])
    risk_flags = sorted(
        set(interpretation.get("risk_flags", []) or [])
        | set(readiness_flags or [])
    )
    if assumed_lsf_kwargs:
        risk_flags = [
            flag
            for flag in risk_flags
            if flag not in {"missing_resolution", "resolution_assumption_required"}
        ]
    warnings = _warnings_with_assumed_resolution(
        suggestion.warnings,
        assumed_resolution,
    )
    payload = {
        "schema_version": 1,
        "operation": "suggest_fit_setup",
        "model": model,
        "mode": str(mode).strip().lower(),
        "science_case": str(science_case).strip().lower(),
        "minimal_fit_call": "fit_stellar_spectrum(spec, model='phoenix')",
        "minimal_setup_fit_call": (
            "fit_stellar_spectrum(spec, model='phoenix', setup=setup)"
        ),
        "fit_kwargs": fit_kwargs,
        "recommended_window": window,
        "recommended_branch_id": branch_plan.get("recommended_branch_id"),
        "recommended_branch": branch_plan.get("recommended_branch"),
        "diagnostic_windows": {
            "selected": diagnostic_info.get("selection", {}).get("selected", []),
            "recommended_combinations": diagnostic_info.get(
                "recommended_combinations",
                {},
            ),
        },
        "readiness": readiness,
        "warnings": list(dict.fromkeys(warnings)),
        "risk_flags": risk_flags,
        "reasons": list(suggestion.reasons),
        "next_steps": _setup_next_steps(suggestion, readiness),
        "provenance": {
            "defaults": suggestion.provenance,
            "readiness_included": bool(include_readiness),
            "readiness_intent": None
            if readiness is None
            else readiness.get("intent"),
            "assumed_resolution": _jsonable(assumed_resolution),
            "mask_threshold": float(mask_threshold),
            "expert_overrides": (
                "Pass explicit regions, bounds, p0, resolution_R/R, "
                "exclude_masks, defaults_mode, or mode to build a new reviewed "
                "setup before fitting."
            ),
        },
    }
    return FitSetup(payload)
