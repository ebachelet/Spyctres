"""Diagnostic wavelength-window catalog and selection helpers.

The catalog in this module is intentionally lightweight: it identifies broad
stellar-feature windows that are useful for first-pass classification,
readiness checks, and later expert workflows.  It does not replace the
full-spectrum PHOENIX fit and it does not claim that a covered feature is
physically measured.  Reference provenance is recorded through ``reference_ids``
matching entries in ``references.json``; the most relevant sources are NIST ASD
for atomic wavelengths, Gray & Corbally for MK classification practice,
Cenarro et al. for Ca II triplet/Paschen behaviour, Meyer et al. for H-band
classification, Riddick et al. and McGovern et al. for late-M gravity-sensitive
optical molecular/alkali features, and Rayner et al. for cool-star near-IR
spectra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import SpectrumCollection, SpectrumSegment, coerce_spectrum
from .waveutils import C_KMS, convert_wavelength_medium


DEFAULT_DIAGNOSTIC_ROLES = (
    "temperature",
    "gravity",
    "metallicity",
    "rv",
    "hot_star",
    "cool_star",
    "molecular",
)


@dataclass(frozen=True)
class DiagnosticWindow:
    """A broad diagnostic spectral window.

    ``region_A`` is the canonical vacuum-wavelength, stellar-rest-frame window
    in Angstrom.  Runtime selection converts this canonical region to each
    segment's operational wavelength medium and applies any explicit RV/padding
    supplied by the caller.  The windows are approximate and must remain
    auditable; exact line measurements belong in the local line-fitting layer.
    """

    id: str
    label: str
    region_A: tuple
    roles: tuple
    physical_sensitivity: tuple = ()
    task_suitability: tuple = ()
    applicability_tags: tuple = ()
    feature_family: tuple = ()
    model_support: str = "supported"
    source_wavelength_medium: str = "vacuum"
    canonical_wave_medium: str = "vacuum"
    canonical_reference_frame: str = "stellar_rest"
    default_selection_policy: str = "include"
    default_fit_policy: str = "auto"
    parent_id: str | None = None
    subwindows: tuple = ()
    features: tuple = ()
    priority: float = 1.0
    min_overlap_A: float = 8.0
    min_pixels: int = 6
    teff_range_K: tuple = (None, None)
    spectral_type_hint: str | None = None
    wave_medium_note: str = "canonical catalog coordinates are vacuum Angstrom"
    risk_tags: tuple = ()
    reference_ids: tuple = ()
    notes: str = ""

    def __post_init__(self):
        wmin, wmax = (float(self.region_A[0]), float(self.region_A[1]))
        if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
            raise ValueError("DiagnosticWindow.region_A must have finite wmin < wmax.")
        if not str(self.id).strip():
            raise ValueError("DiagnosticWindow.id must be non-empty.")
        if not self.roles:
            raise ValueError("DiagnosticWindow.roles must be non-empty.")
        priority = float(self.priority)
        if not np.isfinite(priority) or priority <= 0.0:
            raise ValueError("DiagnosticWindow.priority must be finite and > 0.")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "region_A", (wmin, wmax))
        object.__setattr__(self, "roles", tuple(str(role) for role in self.roles))
        split = _split_roles(self.roles, self.id)
        object.__setattr__(
            self,
            "physical_sensitivity",
            _tuple_or_default(self.physical_sensitivity, split["physical_sensitivity"]),
        )
        object.__setattr__(
            self,
            "task_suitability",
            _tuple_or_default(self.task_suitability, split["task_suitability"]),
        )
        object.__setattr__(
            self,
            "applicability_tags",
            _tuple_or_default(self.applicability_tags, split["applicability_tags"]),
        )
        object.__setattr__(
            self,
            "feature_family",
            _tuple_or_default(self.feature_family, split["feature_family"]),
        )
        model_support = str(self.model_support).strip().lower()
        if model_support not in {
            "supported",
            "uncertain",
            "stress_only",
            "unsupported",
        }:
            raise ValueError(
                "model_support must be supported, uncertain, stress_only, or unsupported."
            )
        object.__setattr__(self, "model_support", model_support)
        object.__setattr__(
            self,
            "source_wavelength_medium",
            str(self.source_wavelength_medium).strip().lower(),
        )
        object.__setattr__(
            self,
            "canonical_wave_medium",
            str(self.canonical_wave_medium).strip().lower(),
        )
        object.__setattr__(
            self,
            "canonical_reference_frame",
            str(self.canonical_reference_frame).strip().lower(),
        )
        if self.canonical_wave_medium != "vacuum":
            raise ValueError("DiagnosticWindow canonical_wave_medium must be vacuum.")
        if self.canonical_reference_frame != "stellar_rest":
            raise ValueError(
                "DiagnosticWindow canonical_reference_frame must be stellar_rest."
            )
        object.__setattr__(
            self,
            "default_selection_policy",
            _validate_policy(self.default_selection_policy, "default_selection_policy"),
        )
        fit_policy = str(self.default_fit_policy).strip().lower()
        if fit_policy == "auto":
            fit_policy = (
                "warn"
                if self.risk_tags or model_support in {"uncertain", "stress_only"}
                else "include"
            )
        object.__setattr__(
            self,
            "default_fit_policy",
            _validate_policy(fit_policy, "default_fit_policy"),
        )
        object.__setattr__(
            self,
            "parent_id",
            None if self.parent_id is None else str(self.parent_id),
        )
        object.__setattr__(
            self,
            "subwindows",
            tuple(_normalize_subwindow(item) for item in self.subwindows),
        )
        object.__setattr__(
            self, "features", tuple(str(feature) for feature in self.features)
        )
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "min_overlap_A", float(self.min_overlap_A))
        object.__setattr__(self, "min_pixels", int(self.min_pixels))
        object.__setattr__(self, "risk_tags", tuple(str(tag) for tag in self.risk_tags))
        object.__setattr__(
            self,
            "reference_ids",
            tuple(str(ref) for ref in self.reference_ids),
        )

    @property
    def width_A(self):
        return float(self.region_A[1] - self.region_A[0])

    def applicability_score(self, initial_teff=None):
        """Return a soft Teff applicability score without hard-rejecting windows."""
        if initial_teff is None:
            return 1.0
        teff = float(initial_teff)
        if not np.isfinite(teff):
            return 1.0
        lo, hi = self.teff_range_K
        if lo is None and hi is None:
            return 1.0
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        if lo <= teff <= hi:
            return 1.0
        margin = max(500.0, 0.12 * max(abs(lo), abs(hi), abs(teff), 1.0))
        if teff < lo:
            distance = lo - teff
        else:
            distance = teff - hi
        if distance <= margin:
            return 0.55
        return 0.20

    def to_metadata(self):
        lo, hi = self.region_A
        teff_lo, teff_hi = self.teff_range_K
        return {
            "id": self.id,
            "label": self.label,
            "region_A": [float(lo), float(hi)],
            "region_vacuum_A": [float(lo), float(hi)],
            "width_A": self.width_A,
            "roles": list(self.roles),
            "physical_sensitivity": list(self.physical_sensitivity),
            "task_suitability": list(self.task_suitability),
            "applicability": {
                "tags": list(self.applicability_tags),
                "teff_range_K": [
                    None if teff_lo is None else float(teff_lo),
                    None if teff_hi is None else float(teff_hi),
                ],
                "spectral_type_hint": self.spectral_type_hint,
            },
            "feature_family": list(self.feature_family),
            "model_support": self.model_support,
            "canonical_coordinates": {
                "unit": "Angstrom",
                "wave_medium": self.canonical_wave_medium,
                "reference_frame": self.canonical_reference_frame,
                "region_vacuum_A": [float(lo), float(hi)],
                "source_wavelength_medium": self.source_wavelength_medium,
                "conversion_provenance": (
                    "Catalog entries are stored as broad vacuum-wavelength "
                    "stellar-rest-frame windows. Runtime selection converts "
                    "to each segment's declared wave_medium before checking "
                    "coverage."
                ),
            },
            "features": list(self.features),
            "priority": float(self.priority),
            "min_overlap_A": float(self.min_overlap_A),
            "min_pixels": int(self.min_pixels),
            "teff_range_K": [
                None if teff_lo is None else float(teff_lo),
                None if teff_hi is None else float(teff_hi),
            ],
            "spectral_type_hint": self.spectral_type_hint,
            "wave_medium_note": self.wave_medium_note,
            "risk_tags": list(self.risk_tags),
            "default_selection_policy": self.default_selection_policy,
            "default_fit_policy": self.default_fit_policy,
            "parent_id": self.parent_id,
            "subwindows": [dict(item) for item in self.subwindows],
            "reference_ids": list(self.reference_ids),
            "notes": self.notes,
        }


def _tuple_or_default(value, default):
    value = tuple(str(item) for item in (value or ()))
    return value if value else tuple(default)


def _validate_policy(value, field):
    value = str(value).strip().lower()
    if value not in {"include", "warn", "exclude"}:
        raise ValueError(f"{field} must be include, warn, or exclude.")
    return value


def _normalize_subwindow(item):
    item = dict(item)
    region = item.get("region_A", item.get("region_vacuum_A"))
    if region is None or len(region) != 2:
        raise ValueError("Diagnostic subwindows require region_A.")
    lo = float(region[0])
    hi = float(region[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Diagnostic subwindow region_A must have wmin < wmax.")
    out = {
        "id": str(item.get("id", "subwindow")),
        "label": str(item.get("label", item.get("id", "subwindow"))),
        "region_vacuum_A": [lo, hi],
        "region_A": [lo, hi],
        "roles": [str(role) for role in item.get("roles", ())],
        "notes": str(item.get("notes", "")),
    }
    if "feature_family" in item:
        out["feature_family"] = [str(value) for value in item["feature_family"]]
    return out


def _split_roles(roles, window_id):
    roles = {str(role) for role in roles}
    physical = sorted(roles & {"temperature", "gravity", "metallicity"})
    tasks = ["classification"]
    if "rv" in roles:
        tasks.append("rv")
    applicability = sorted(roles & {"hot_star", "cool_star"})
    family = []
    window_id = str(window_id).lower()
    if any(role in roles for role in ("molecular",)) or any(
        token in window_id for token in ("tio", "vo_", "co_", "cah", "feh", "ch_")
    ):
        family.append("molecular")
    if any(token in window_id for token in ("h_", "paschen", "brackett", "br_")):
        family.append("hydrogen")
    if "he_" in window_id:
        family.append("helium")
    if any(
        token in window_id
        for token in ("ca_", "mg_", "na_", "k_i", "ki_", "li_", "si_")
    ):
        family.append("atomic_metal")
    return {
        "physical_sensitivity": tuple(physical),
        "task_suitability": tuple(dict.fromkeys(tasks)),
        "applicability_tags": tuple(applicability),
        "feature_family": tuple(dict.fromkeys(family)),
    }


DIAGNOSTIC_WINDOW_CATALOG = (
    DiagnosticWindow(
        id="ca_hk_h_epsilon",
        label="Ca II H/K + Hε",
        region_A=(3900.0, 3998.0),
        roles=("temperature", "gravity", "metallicity", "rv"),
        features=("Ca II K 3933.7", "Ca II H 3968.5", "Hε 3970.1"),
        priority=1.15,
        min_overlap_A=45.0,
        min_pixels=20,
        teff_range_K=(3500.0, 12000.0),
        spectral_type_hint="A-M; blended/interstellar-sensitive in many products",
        risk_tags=("interstellar_ca", "balmer_blend", "artifact_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
        notes="Useful sanity window, but Ca II can include interstellar absorption.",
    ),
    DiagnosticWindow(
        id="h_delta",
        label="Hδ",
        region_A=(4070.0, 4135.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Hδ 4101.7",),
        priority=1.05,
        min_overlap_A=35.0,
        min_pixels=18,
        teff_range_K=(5000.0, 12000.0),
        spectral_type_hint="A/F stars and hot-star Balmer-wing checks",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="si_ii_4128_4130",
        label="Si II 4128/4130",
        region_A=(4116.0, 4144.0),
        roles=("temperature", "gravity", "metallicity", "rv", "hot_star"),
        features=("Si II 4128/4130",),
        priority=0.58,
        min_overlap_A=16.0,
        min_pixels=8,
        teff_range_K=(7500.0, 12000.0),
        spectral_type_hint="late-B/A hot-star check near the PHOENIX LTE limit",
        risk_tags=(
            "phoenix_hot_star_limit",
            "nlte_sensitive",
            "metal_abundance_sensitive",
        ),
        model_support="stress_only",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
        notes=(
            "Presence/absence is useful for early-type triage, but this is not "
            "a precision abundance or NLTE-safe window in the current PHOENIX path."
        ),
    ),
    DiagnosticWindow(
        id="ca_i_4227",
        label="Ca I 4227",
        region_A=(4205.0, 4248.0),
        roles=("temperature", "gravity", "metallicity", "cool_star"),
        features=("Ca I 4226.7",),
        priority=0.80,
        min_overlap_A=25.0,
        min_pixels=12,
        teff_range_K=(3200.0, 7500.0),
        spectral_type_hint="G-M cool-star/metal-line check",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="ch_g_band",
        label="CH G band",
        region_A=(4280.0, 4335.0),
        roles=("temperature", "metallicity", "cool_star", "molecular"),
        features=("CH G band near 4300 A",),
        priority=0.78,
        min_overlap_A=30.0,
        min_pixels=14,
        teff_range_K=(4300.0, 6500.0),
        spectral_type_hint="F/G/K carbon- and metallicity-sensitive check",
        risk_tags=(
            "carbon_abundance_sensitive",
            "continuum_sensitive",
            "molecular_model_sensitive",
        ),
        model_support="uncertain",
        reference_ids=("gray_corbally2009_spectral_classification",),
        notes=(
            "Useful for broad classification, especially when paired with Balmer "
            "and metal-line windows; fixed-abundance PHOENIX fits should not "
            "over-interpret CH residuals as only Teff/logg/[Fe/H]."
        ),
    ),
    DiagnosticWindow(
        id="h_gamma",
        label="Hγ",
        region_A=(4310.0, 4372.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Hγ 4340.5",),
        priority=1.10,
        min_overlap_A=35.0,
        min_pixels=18,
        teff_range_K=(5000.0, 12000.0),
        spectral_type_hint="A/F stars and hot-star Balmer-wing checks",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="he_i_4471",
        label="He I 4471",
        region_A=(4448.0, 4495.0),
        roles=("temperature", "rv", "hot_star"),
        features=("He I 4471.5",),
        priority=0.75,
        min_overlap_A=25.0,
        min_pixels=12,
        teff_range_K=(8000.0, 12000.0),
        spectral_type_hint="hot-star/model-domain sanity check",
        risk_tags=("phoenix_hot_star_limit", "nlte_sensitive"),
        model_support="stress_only",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
        notes="Presence/absence is diagnostic; PHOENIX LTE grid limits still apply.",
    ),
    DiagnosticWindow(
        id="mg_ii_4481",
        label="Mg II 4481",
        region_A=(4472.0, 4492.0),
        roles=("temperature", "metallicity", "rv", "hot_star"),
        features=("Mg II 4481",),
        priority=0.55,
        min_overlap_A=12.0,
        min_pixels=7,
        teff_range_K=(7000.0, 12000.0),
        spectral_type_hint="late-B/A hot-star check, often read with He I 4471",
        risk_tags=(
            "he_i_4471_overlap",
            "phoenix_hot_star_limit",
            "metal_abundance_sensitive",
        ),
        model_support="uncertain",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
        notes=(
            "The He I 4471 / Mg II 4481 neighbourhood is useful for hot-star "
            "triage, but the current lightweight catalog keeps the overlapping "
            "features explicit rather than fitting a classification ratio."
        ),
    ),
    DiagnosticWindow(
        id="si_iii_4552",
        label="Si III 4552",
        region_A=(4540.0, 4566.0),
        roles=("temperature", "gravity", "metallicity", "rv", "hot_star"),
        features=("Si III 4552",),
        priority=0.42,
        min_overlap_A=14.0,
        min_pixels=7,
        teff_range_K=(10000.0, 12000.0),
        spectral_type_hint="early-B stress check; outside ordinary PHOENIX comfort zone",
        risk_tags=(
            "phoenix_hot_star_limit",
            "nlte_sensitive",
            "metal_abundance_sensitive",
        ),
        model_support="stress_only",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
        notes=(
            "Included to flag spectra that may be too hot for the present "
            "PHOENIX LTE workflow; it should not drive ordinary fits."
        ),
    ),
    DiagnosticWindow(
        id="h_beta",
        label="Hβ",
        region_A=(4830.0, 4898.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Hβ 4861.3",),
        priority=1.12,
        min_overlap_A=40.0,
        min_pixels=20,
        teff_range_K=(5000.0, 12000.0),
        spectral_type_hint="A/F stars and hot-star Balmer-wing checks",
        risk_tags=("dib_4882_overlap", "broad_line_core_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
            "galazutdinov2020_very_broad_dibs",
        ),
    ),
    DiagnosticWindow(
        id="mg_i_b",
        label="Mg I b",
        region_A=(5150.0, 5208.0),
        roles=("temperature", "gravity", "metallicity", "rv", "cool_star"),
        features=("Mg I b triplet",),
        priority=0.95,
        min_overlap_A=35.0,
        min_pixels=18,
        teff_range_K=(3500.0, 7500.0),
        spectral_type_hint="F-M metal-line/gravity sanity check",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="na_i_d",
        label="Na I D",
        region_A=(5870.0, 5906.0),
        roles=("rv", "metallicity", "cool_star"),
        features=("Na I D 5890/5896",),
        priority=0.65,
        min_overlap_A=22.0,
        min_pixels=10,
        teff_range_K=(3500.0, 7000.0),
        spectral_type_hint="cool-star/ISM sanity check",
        risk_tags=("interstellar_na", "telluric_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="he_i_5876",
        label="He I 5876",
        region_A=(5858.0, 5894.0),
        roles=("temperature", "rv", "hot_star"),
        features=("He I 5875.6",),
        priority=0.60,
        min_overlap_A=20.0,
        min_pixels=10,
        teff_range_K=(8000.0, 12000.0),
        spectral_type_hint="hot-star/model-domain sanity check",
        risk_tags=("na_d_overlap", "nlte_sensitive"),
        model_support="stress_only",
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="h_alpha",
        label="Hα",
        region_A=(6530.0, 6600.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Hα 6562.8",),
        priority=0.95,
        min_overlap_A=38.0,
        min_pixels=20,
        teff_range_K=(4500.0, 12000.0),
        spectral_type_hint="hydrogen-line check; emission/activity-sensitive",
        risk_tags=("emission_sensitive", "activity_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="tio_7050",
        label="TiO 7050 band",
        region_A=(7000.0, 7165.0),
        roles=("temperature", "cool_star", "molecular"),
        features=("TiO band system near 7050 A",),
        priority=0.85,
        min_overlap_A=80.0,
        min_pixels=30,
        teff_range_K=(2500.0, 4500.0),
        spectral_type_hint="late-K/M cool-star molecular check",
        risk_tags=("telluric_sensitive", "molecular_model_sensitive"),
        reference_ids=("gray_corbally2009_spectral_classification",),
    ),
    DiagnosticWindow(
        id="vo_7450",
        label="VO 7450",
        region_A=(7380.0, 7580.0),
        roles=("temperature", "gravity", "cool_star", "molecular"),
        features=("VO band structure near 7450 A",),
        priority=0.66,
        min_overlap_A=85.0,
        min_pixels=30,
        teff_range_K=(2500.0, 4200.0),
        spectral_type_hint="late-M gravity/temperature molecular check",
        risk_tags=(
            "telluric_o2_overlap",
            "molecular_model_sensitive",
            "continuum_sensitive",
        ),
        model_support="uncertain",
        reference_ids=(
            "riddick2007_m_dwarf_optical_indices",
            "mcgovern2004_brown_dwarf_gravity_features",
        ),
    ),
    DiagnosticWindow(
        id="k_i_7700",
        label="K I 7665/7699",
        region_A=(7652.0, 7712.0),
        roles=("gravity", "metallicity", "rv", "cool_star"),
        features=("K I 7665/7699",),
        priority=0.50,
        min_overlap_A=35.0,
        min_pixels=16,
        teff_range_K=(2500.0, 5200.0),
        spectral_type_hint="cool-star gravity/alkali check",
        risk_tags=(
            "telluric_o2_overlap",
            "pressure_broadened",
            "stellar_activity_sensitive",
        ),
        model_support="uncertain",
        reference_ids=(
            "nist_asd_v512",
            "mcgovern2004_brown_dwarf_gravity_features",
        ),
        notes=(
            "The stellar K I doublet lies close to strong topocentric O2 telluric "
            "absorption; use it only when telluric handling and wavelength frame "
            "metadata are trustworthy."
        ),
    ),
    DiagnosticWindow(
        id="vo_7900",
        label="VO 7900/8100",
        region_A=(7860.0, 8155.0),
        roles=("temperature", "gravity", "cool_star", "molecular"),
        features=("VO bands near 7900-8150 A",),
        priority=0.70,
        min_overlap_A=130.0,
        min_pixels=45,
        teff_range_K=(2500.0, 4200.0),
        spectral_type_hint="late-M gravity/temperature molecular check",
        risk_tags=(
            "telluric_h2o_overlap",
            "molecular_model_sensitive",
            "continuum_sensitive",
        ),
        model_support="uncertain",
        reference_ids=(
            "riddick2007_m_dwarf_optical_indices",
            "mcgovern2004_brown_dwarf_gravity_features",
        ),
    ),
    DiagnosticWindow(
        id="na_i_8200",
        label="Na I 8190",
        region_A=(8170.0, 8230.0),
        roles=("gravity", "metallicity", "cool_star"),
        features=("Na I 8183/8195",),
        priority=0.75,
        min_overlap_A=35.0,
        min_pixels=18,
        teff_range_K=(3000.0, 6000.0),
        spectral_type_hint="cool-star gravity/metal-line check",
        risk_tags=("telluric_h2o_overlap", "molecular_model_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
            "riddick2007_m_dwarf_optical_indices",
            "mcgovern2004_brown_dwarf_gravity_features",
        ),
    ),
    DiagnosticWindow(
        id="ca_ii_triplet_paschen",
        label="Ca II triplet + Paschen",
        region_A=(8470.0, 8708.0),
        roles=("gravity", "metallicity", "temperature", "rv", "cool_star", "hot_star"),
        features=("Ca II 8498/8542/8662", "high-order Paschen lines"),
        priority=1.05,
        min_overlap_A=110.0,
        min_pixels=25,
        teff_range_K=(3500.0, 12000.0),
        spectral_type_hint="Ca triplet for cool stars; Paschen blend for hotter stars",
        risk_tags=("paschen_ca_blend", "telluric_sensitive", "chromosphere_sensitive"),
        subwindows=(
            {
                "id": "ca_triplet_lines",
                "label": "Ca II triplet line cores",
                "region_A": [8485.0, 8678.0],
                "roles": ["gravity", "metallicity", "rv", "cool_star"],
                "feature_family": ["atomic_metal"],
                "notes": "Child diagnostic for Ca-dominated triplet checks.",
            },
            {
                "id": "paschen_triplet_blend",
                "label": "Paschen blend checks in CaT region",
                "region_A": [8460.0, 8755.0],
                "roles": ["temperature", "gravity", "hot_star"],
                "feature_family": ["hydrogen"],
                "notes": (
                    "Child diagnostic for Paschen contamination/recovery; "
                    "do not interpret the parent window as pure Ca."
                ),
            },
        ),
        reference_ids=(
            "nist_asd_v512",
            "cenarro2001_ca_triplet_indices",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="feh_8700",
        label="FeH 8700",
        region_A=(8648.0, 8732.0),
        roles=("temperature", "gravity", "cool_star", "molecular"),
        features=("FeH band near 8700 A",),
        priority=0.52,
        min_overlap_A=42.0,
        min_pixels=18,
        teff_range_K=(2500.0, 4200.0),
        spectral_type_hint="late-M gravity/molecular check near Ca triplet",
        risk_tags=(
            "ca_triplet_overlap",
            "molecular_model_sensitive",
            "telluric_sensitive",
        ),
        model_support="uncertain",
        reference_ids=(
            "riddick2007_m_dwarf_optical_indices",
            "mcgovern2004_brown_dwarf_gravity_features",
        ),
    ),
    DiagnosticWindow(
        id="tio_red_bands",
        label="Red TiO bands",
        region_A=(8430.0, 8900.0),
        roles=("temperature", "cool_star", "molecular"),
        features=("TiO red-system bands",),
        priority=0.80,
        min_overlap_A=180.0,
        min_pixels=60,
        teff_range_K=(2500.0, 4500.0),
        spectral_type_hint="late-K/M molecular check",
        risk_tags=("telluric_sensitive", "ca_triplet_overlap", "molecular_model_sensitive"),
        reference_ids=(
            "gray_corbally2009_spectral_classification",
            "rayner2009_irtf_cool_stars",
            "riddick2007_m_dwarf_optical_indices",
        ),
    ),
    DiagnosticWindow(
        id="paschen_gamma_delta",
        label="Paγ / Paδ",
        region_A=(9980.0, 11020.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Paδ 10049", "Paγ 10938"),
        priority=0.75,
        min_overlap_A=180.0,
        min_pixels=45,
        teff_range_K=(6000.0, 12000.0),
        spectral_type_hint="near-IR hydrogen check",
        risk_tags=("telluric_sensitive",),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="paschen_beta",
        label="Paβ",
        region_A=(12700.0, 12940.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Paβ 12818",),
        priority=0.65,
        min_overlap_A=100.0,
        min_pixels=30,
        teff_range_K=(6000.0, 12000.0),
        spectral_type_hint="J-band hydrogen check",
        risk_tags=("telluric_sensitive",),
        reference_ids=(
            "nist_asd_v512",
            "gray_corbally2009_spectral_classification",
        ),
    ),
    DiagnosticWindow(
        id="brackett_h_band",
        label="H-band Brackett series",
        region_A=(15500.0, 17450.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Brackett-series H-band lines",),
        priority=0.70,
        min_overlap_A=250.0,
        min_pixels=60,
        teff_range_K=(7000.0, 12000.0),
        spectral_type_hint="H-band hydrogen classification sanity check",
        risk_tags=("telluric_sensitive", "continuum_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "meyer1998_hband_classification",
        ),
    ),
    DiagnosticWindow(
        id="br_gamma",
        label="Brγ",
        region_A=(21550.0, 21780.0),
        roles=("temperature", "gravity", "rv", "hot_star"),
        features=("Brγ 21661",),
        priority=0.60,
        min_overlap_A=100.0,
        min_pixels=30,
        teff_range_K=(7000.0, 12000.0),
        spectral_type_hint="K-band hydrogen check",
        risk_tags=("telluric_sensitive", "emission_sensitive"),
        reference_ids=(
            "nist_asd_v512",
            "meyer1998_hband_classification",
        ),
    ),
    DiagnosticWindow(
        id="na_i_kband",
        label="K-band Na I",
        region_A=(21950.0, 22130.0),
        roles=("temperature", "gravity", "metallicity", "rv", "cool_star"),
        features=("Na I 2.20/2.21 μm",),
        priority=0.62,
        min_overlap_A=80.0,
        min_pixels=25,
        teff_range_K=(3000.0, 6500.0),
        spectral_type_hint="K-band late-type-star classification check",
        risk_tags=("telluric_sensitive", "continuum_sensitive", "veiling_sensitive"),
        model_support="uncertain",
        reference_ids=(
            "nist_asd_v512",
            "luhman_rieke1998_kband_classification",
            "hubrig2007_kband_companion_classification",
            "kleinmann_hall1986_kband_atlas",
        ),
    ),
    DiagnosticWindow(
        id="ca_i_kband",
        label="K-band Ca I",
        region_A=(22540.0, 22740.0),
        roles=("temperature", "gravity", "metallicity", "rv", "cool_star"),
        features=("Ca I 2.26 μm",),
        priority=0.60,
        min_overlap_A=90.0,
        min_pixels=25,
        teff_range_K=(3000.0, 6500.0),
        spectral_type_hint="K-band late-type-star classification check",
        risk_tags=("telluric_sensitive", "continuum_sensitive", "veiling_sensitive"),
        model_support="uncertain",
        reference_ids=(
            "nist_asd_v512",
            "luhman_rieke1998_kband_classification",
            "hubrig2007_kband_companion_classification",
            "kleinmann_hall1986_kband_atlas",
        ),
    ),
    DiagnosticWindow(
        id="co_23um_bandhead",
        label="CO 2.3 μm bandhead",
        region_A=(22850.0, 23650.0),
        roles=("temperature", "gravity", "cool_star", "molecular"),
        features=("CO first-overtone bandheads near 2.3 μm",),
        priority=0.90,
        min_overlap_A=250.0,
        min_pixels=60,
        teff_range_K=(2500.0, 5000.0),
        spectral_type_hint="cool giant/dwarf molecular check",
        risk_tags=("telluric_sensitive", "molecular_model_sensitive"),
        reference_ids=(
            "kleinmann_hall1986_kband_atlas",
            "rayner2009_irtf_cool_stars",
            "hubrig2007_kband_companion_classification",
        ),
    ),
)


def diagnostic_window_catalog(profile="stellar"):
    """Return the named diagnostic-window catalog.

    Parameters
    ----------
    profile : {"stellar", "optical", "near_ir"}
        ``stellar`` returns all current entries.  The narrower profiles are
        convenience filters; callers can still filter by role or wavelength.
    """
    profile = str(profile).strip().lower()
    if profile in {"stellar", "all", "default"}:
        return tuple(DIAGNOSTIC_WINDOW_CATALOG)
    if profile in {"optical", "blue_optical", "red_optical"}:
        return tuple(w for w in DIAGNOSTIC_WINDOW_CATALOG if w.region_A[0] < 9500.0)
    if profile in {"near_ir", "nir", "infrared"}:
        return tuple(w for w in DIAGNOSTIC_WINDOW_CATALOG if w.region_A[1] > 9000.0)
    raise ValueError("profile must be stellar, optical, or near_ir.")


def select_diagnostic_windows(
    spectrum,
    *,
    windows=None,
    roles=None,
    initial_teff=None,
    rv_kms=None,
    rv_padding_kms=0.0,
    min_overlap_A=0.0,
    min_pixels=1,
    max_windows=None,
):
    """Select useful diagnostic windows from the loaded wavelength coverage.

    The selector is deliberately cheap.  It scores coverage, usable-pixel
    fraction, a robust in-window contrast proxy, and optional Teff
    applicability.  It does not run PHOENIX or fit every possible combination.
    """
    segments = _as_segments(spectrum)
    windows = diagnostic_window_catalog() if windows is None else tuple(windows)
    role_filter = None if roles is None else {str(role) for role in roles}
    min_overlap_A = float(min_overlap_A)
    min_pixels = int(min_pixels)
    if min_overlap_A < 0.0:
        raise ValueError("min_overlap_A must be >= 0.")
    if min_pixels < 0:
        raise ValueError("min_pixels must be >= 0.")
    rv_used = None if rv_kms is None else float(rv_kms)
    rv_padding_kms = float(rv_padding_kms)
    if rv_used is not None and not np.isfinite(rv_used):
        raise ValueError("rv_kms must be finite when supplied.")
    if not np.isfinite(rv_padding_kms) or rv_padding_kms < 0.0:
        raise ValueError("rv_padding_kms must be finite and >= 0.")

    selected = []
    rejected = []
    for window in windows:
        if not isinstance(window, DiagnosticWindow):
            raise TypeError("windows must contain DiagnosticWindow objects.")
        window_terms = (
            set(window.roles)
            | set(window.physical_sensitivity)
            | set(window.task_suitability)
            | set(window.applicability_tags)
            | set(window.feature_family)
        )
        if role_filter is not None and not (window_terms & role_filter):
            item = window.to_metadata()
            item["reject_reason"] = "role_filter"
            rejected.append(item)
            continue

        stats = _window_stats(
            segments,
            window,
            rv_kms=rv_used,
            rv_padding_kms=rv_padding_kms,
        )
        coverage_fraction = (
            0.0
            if window.width_A <= 0.0
            else min(1.0, stats["overlap_A"] / window.width_A)
        )
        usable_fraction = stats["usable_fraction"]
        teff_score = window.applicability_score(initial_teff)
        components = _score_components(window, stats, coverage_fraction, teff_score)
        unconditioned_components = _score_components(
            window,
            stats,
            coverage_fraction,
            1.0,
        )
        score = _score_from_components(components)
        unconditioned_score = _score_from_components(unconditioned_components)

        required_overlap = max(float(window.min_overlap_A), min_overlap_A)
        required_pixels = max(int(window.min_pixels), min_pixels)
        item = window.to_metadata()
        item.update(stats)
        item["coverage_fraction"] = float(coverage_fraction)
        item["teff_applicability_score"] = float(teff_score)
        item["score_components"] = components
        item["unconditioned_score_components"] = unconditioned_components
        item["score"] = float(score)
        item["unconditioned_score"] = float(unconditioned_score)
        item["selected"] = bool(
            stats["n_usable_pixels"] >= required_pixels
            and stats["overlap_A"] >= required_overlap
            and window.default_selection_policy != "exclude"
        )
        item["selection_requirements"] = {
            "min_overlap_A": float(required_overlap),
            "min_pixels": int(required_pixels),
        }
        if item["selected"]:
            selected.append(item)
        else:
            reasons = []
            if stats["n_usable_pixels"] < required_pixels:
                reasons.append("too_few_usable_pixels")
            if stats["overlap_A"] < required_overlap:
                reasons.append("insufficient_wavelength_overlap")
            if window.default_selection_policy == "exclude":
                reasons.append("selection_policy_exclude")
            item["reject_reason"] = ",".join(reasons) or "not_selected"
            rejected.append(item)

    selected = sorted(
        selected,
        key=lambda item: (-float(item["score"]), float(item["region_A"][0])),
    )
    overflow = []
    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows < 1:
            raise ValueError("max_windows must be >= 1 when supplied.")
        overflow = selected[max_windows:]
        for item in overflow:
            item = dict(item)
            item["selected"] = False
            item["reject_reason"] = "max_windows_limit"
            rejected.append(item)
        selected = selected[:max_windows]

    return {
        "schema_version": 1,
        "operation": "select_diagnostic_windows",
        "catalog_profile": "stellar",
        "initial_teff_K": None if initial_teff is None else float(initial_teff),
        "rv_window_mapping": {
            "catalog_reference_frame": "stellar_rest",
            "catalog_wave_medium": "vacuum",
            "rv_used_for_window_kms": rv_used,
            "rv_padding_kms": float(rv_padding_kms),
            "policy": (
                "Catalog windows are converted to each segment wave_medium. "
                "For spectra not labelled stellar-rest corrected, rv_kms "
                "shifts the operational window and rv_padding_kms widens it."
            ),
        },
        "roles_filter": None if role_filter is None else sorted(role_filter),
        "selection_policy": {
            "score_terms": [
                "catalog_priority",
                "wavelength_coverage",
                "contiguous_usable_coverage",
                "usable_pixel_fraction",
                "snr",
                "continuum_detrended_contrast",
                "risk_penalty",
                "resolution_elements",
                "optional_teff_applicability",
            ],
            "max_windows": max_windows,
            "hard_reject_terms": ["coverage", "usable_pixels"],
            "teff_is_soft_weight": True,
            "unconditioned_scores_retained": True,
            "expensive_fits_run": False,
        },
        "input_coverage": _coverage_metadata(segments),
        "selected": _json_native(selected),
        "rejected": _json_native(rejected),
    }


def build_diagnostic_window_combinations(
    selection,
    *,
    max_windows=6,
    max_single_windows=5,
    include_leave_one_out=True,
    include_leave_one_family_out=True,
    include_role_balanced=True,
    include_trusted_baseline=True,
    priority_roles=DEFAULT_DIAGNOSTIC_ROLES,
):
    """Build a bounded set of candidate window combinations.

    This returns design/provenance only.  It avoids the combinatorial explosion
    of trying every possible subset and keeps expensive PHOENIX fits opt-in.
    """
    selected = _selected_records(selection)
    max_windows = int(max_windows)
    max_single_windows = int(max_single_windows)
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1.")
    if max_single_windows < 0:
        raise ValueError("max_single_windows must be >= 0.")

    active = selected[:max_windows]
    combinations = []
    seen = set()

    def add(kind, records, label):
        ids = tuple(record["id"] for record in records)
        if not ids or ids in seen:
            return
        seen.add(ids)
        combinations.append(_combination_record(kind, label, records))

    if include_trusted_baseline:
        trusted = [
            record
            for record in selected
            if record.get("default_fit_policy") == "include"
            and record.get("model_support") == "supported"
        ][:max_windows]
        add("trusted_baseline", trusted, "Trusted low-risk baseline windows")

    add("all_selected_top", active, "Top selected diagnostic windows")

    if include_role_balanced:
        role_records = []
        used_ids = set()
        for role in priority_roles:
            for record in selected:
                if role in record.get("roles", ()) and record["id"] not in used_ids:
                    role_records.append(record)
                    used_ids.add(record["id"])
                    break
            if len(role_records) >= max_windows:
                break
        add("role_balanced", role_records, "Role-balanced diagnostic windows")

    for record in active[:max_single_windows]:
        add(
            "single_window",
            [record],
            "Single-window sanity check: {0}".format(record["label"]),
        )

    if include_leave_one_out and len(active) > 2:
        for record in active[:max_single_windows]:
            keep = [item for item in active if item["id"] != record["id"]]
            add(
                "leave_one_out",
                keep,
                "Leave-one-window-out: omit {0}".format(record["label"]),
            )

    if include_leave_one_family_out and len(active) > 2:
        families = sorted(
            {
                family
                for record in active
                for family in record.get("feature_family", ())
            }
        )
        for family in families[:max_single_windows]:
            keep = [
                item
                for item in active
                if family not in item.get("feature_family", ())
            ]
            add(
                "leave_one_family_out",
                keep,
                "Leave-one-family-out: omit {0}".format(family),
            )

    return {
        "schema_version": 1,
        "operation": "build_diagnostic_window_combinations",
        "strategy": (
            "bounded_trusted_baseline_role_balanced_single_window_leave_one_out"
        ),
        "ranking_note": (
            "These combinations are not ranked by raw chi-square. Future fit "
            "comparisons should use held-out checks, parameter stability, "
            "quality flags, and common-evaluation residuals."
        ),
        "expensive_fits_run": False,
        "combinations": _json_native(combinations),
    }


def fit_regions_from_combination(combination):
    """Return ``[(wmin, wmax), ...]`` regions from a combination record."""
    return [
        (float(region[0]), float(region[1]))
        for region in combination.get("regions_A", ())
    ]


def format_diagnostic_window_table(selection, max_rows=None):
    """Return a compact plain-text table for selected diagnostic windows."""
    selected = _selected_records(selection)
    if max_rows is not None:
        selected = selected[: int(max_rows)]
    if not selected:
        return "No diagnostic windows passed the coverage/quality thresholds."
    header = "id                          region_A        roles                 score  usable"
    lines = [header, "-" * len(header)]
    for item in selected:
        lo, hi = item["region_A"]
        roles = ",".join(item.get("roles", ()))
        if len(roles) > 20:
            roles = roles[:19] + "…"
        lines.append(
            "{id:<27} {lo:7.0f}-{hi:<7.0f} {roles:<21} {score:5.2f}  {usable:5.2f}".format(
                id=item["id"],
                lo=float(lo),
                hi=float(hi),
                roles=roles,
                score=float(item.get("score", 0.0)),
                usable=float(item.get("usable_fraction", 0.0)),
            )
        )
    return "\n".join(lines)


def _as_segments(spectrum):
    if isinstance(spectrum, SpectrumSegment):
        return [spectrum]
    if isinstance(spectrum, SpectrumCollection):
        return list(spectrum.segments)
    if isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    coerced = coerce_spectrum(spectrum, warn_unknown=False)
    if isinstance(coerced, SpectrumCollection):
        return list(coerced.segments)
    return [coerced]


def _coverage_metadata(segments):
    rows = []
    for segment in segments:
        wave = np.asarray(segment.wave, dtype=float)
        finite = np.isfinite(wave)
        rows.append(
            {
                "segment": getattr(segment, "name", None),
                "n_pixels": int(wave.size),
                "n_finite_wave": int(np.count_nonzero(finite)),
                "wmin_A": float(np.nanmin(wave[finite])) if np.any(finite) else None,
                "wmax_A": float(np.nanmax(wave[finite])) if np.any(finite) else None,
                "wave_medium": getattr(segment, "wave_medium", "unknown"),
                "observer_frame": getattr(segment, "observer_frame", "unknown"),
                "stellar_rest_status": getattr(
                    segment, "stellar_rest_status", "unknown"
                ),
            }
        )
    return rows


def _window_stats(segments, window, *, rv_kms=None, rv_padding_kms=0.0):
    n_total = 0
    n_usable = 0
    wave_spans = []
    usable_flux = []
    usable_wave = []
    usable_err = []
    n_runs_total = 0
    largest_run_A = 0.0
    largest_gap_A = 0.0
    resolution_elements = []
    segment_rows = []
    for index, segment in enumerate(segments):
        op = _operational_region_for_segment(
            window,
            segment,
            rv_kms=rv_kms,
            rv_padding_kms=rv_padding_kms,
        )
        wmin, wmax = op["region_A"]
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        base = np.isfinite(wave) & np.isfinite(flux)
        mask = np.asarray(segment.mask, dtype=bool)
        if mask.shape == wave.shape:
            usable = base & mask
        else:
            usable = base
        inside_total = base & (wave >= wmin) & (wave <= wmax)
        inside_usable = usable & (wave >= wmin) & (wave <= wmax)
        n_total_i = int(np.count_nonzero(inside_total))
        n_usable_i = int(np.count_nonzero(inside_usable))
        n_total += n_total_i
        n_usable += n_usable_i
        if n_total_i:
            ww = wave[inside_total]
            wave_spans.append((float(np.nanmin(ww)), float(np.nanmax(ww))))
        if n_usable_i:
            ww = wave[inside_usable]
            ff = flux[inside_usable]
            usable_wave.append(ww)
            usable_flux.append(ff)
            if segment.err is not None:
                err = np.asarray(segment.err, dtype=float)
                if err.shape == wave.shape:
                    usable_err.append(err[inside_usable])
            run_stats = _contiguous_run_stats(wave, inside_usable)
            n_runs_total += run_stats["n_contiguous_runs"]
            largest_run_A = max(largest_run_A, run_stats["largest_run_A"])
            largest_gap_A = max(largest_gap_A, run_stats["largest_gap_A"])
            elems = _resolution_elements_for_span(
                segment,
                run_stats["largest_run_A"],
                0.5 * (wmin + wmax),
                n_usable_i,
            )
            if elems is not None:
                resolution_elements.append(elems)
        segment_rows.append(
            {
                "segment": getattr(segment, "name", None),
                "segment_index": int(index),
                "n_pixels": n_total_i,
                "n_usable_pixels": n_usable_i,
                "operational_region_A": op["region_A"],
                "operational_region_vacuum_rest_A": list(window.region_A),
                "wave_medium": op["wave_medium"],
                "medium_conversion_applied": op["medium_conversion_applied"],
                "rv_used_for_window_kms": op["rv_used_for_window_kms"],
                "rv_padding_kms": op["rv_padding_kms"],
                "coordinate_warnings": op["coordinate_warnings"],
            }
        )

    overlap_A = sum(max(0.0, hi - lo) for lo, hi in wave_spans)
    usable_fraction = 0.0 if n_total == 0 else float(n_usable / n_total)
    contiguous_fraction = 0.0 if overlap_A <= 0.0 else min(1.0, largest_run_A / overlap_A)
    contrast = _detrended_contrast(usable_wave, usable_flux)
    snr = _robust_snr(usable_flux, usable_err)
    n_resolution_elements = (
        None if not resolution_elements else float(np.nansum(resolution_elements))
    )
    return {
        "overlap_A": float(overlap_A),
        "n_pixels": int(n_total),
        "n_usable_pixels": int(n_usable),
        "usable_fraction": float(usable_fraction),
        "largest_contiguous_usable_A": float(largest_run_A),
        "largest_contiguous_usable_fraction": float(contiguous_fraction),
        "n_contiguous_runs": int(n_runs_total),
        "largest_gap_A": float(largest_gap_A),
        "n_resolution_elements": n_resolution_elements,
        "local_snr": snr,
        "feature_contrast": float(contrast),
        "detrended_contrast": float(contrast),
        "segment_contributions": segment_rows,
    }


def _score_components(window, stats, coverage_fraction, teff_score):
    risk_penalty = _risk_penalty(window)
    snr = stats.get("local_snr")
    if snr is None:
        snr_score = 0.80
    else:
        snr_score = float(np.clip((float(snr) - 2.0) / 18.0, 0.05, 1.0))
    n_res = stats.get("n_resolution_elements")
    if n_res is None:
        resolution_score = 0.85
    else:
        resolution_score = float(np.clip(float(n_res) / 6.0, 0.05, 1.0))
    contrast = min(float(stats.get("detrended_contrast", 0.0)), 1.5)
    contrast_score = float(1.0 + 0.15 * contrast)
    return {
        "catalog_priority": float(window.priority),
        "coverage_score": float(np.clip(coverage_fraction, 0.0, 1.0)),
        "contiguous_coverage_score": float(
            np.clip(stats.get("largest_contiguous_usable_fraction", 0.0), 0.0, 1.0)
        ),
        "usable_fraction_score": float(
            np.clip(stats.get("usable_fraction", 0.0), 0.0, 1.0)
        ),
        "snr_score": snr_score,
        "detrended_contrast_score": contrast_score,
        "applicability_score": float(teff_score),
        "risk_penalty": float(risk_penalty),
        "resolution_element_score": resolution_score,
    }


def _score_from_components(components):
    score = 1.0
    for value in components.values():
        score *= float(value)
    return float(score)


def _risk_penalty(window):
    penalty = 1.0
    if window.default_fit_policy == "warn":
        penalty *= 0.90
    elif window.default_fit_policy == "exclude":
        penalty *= 0.25
    if window.model_support == "uncertain":
        penalty *= 0.85
    elif window.model_support == "stress_only":
        penalty *= 0.70
    elif window.model_support == "unsupported":
        penalty *= 0.25
    for tag in window.risk_tags:
        if "telluric" in tag or "sky" in tag:
            penalty *= 0.92
        elif "interstellar" in tag or "dib" in tag:
            penalty *= 0.90
        elif "artifact" in tag:
            penalty *= 0.88
        elif "nlte" in tag or "model" in tag:
            penalty *= 0.88
        else:
            penalty *= 0.96
    return float(np.clip(penalty, 0.05, 1.0))


def _operational_region_for_segment(window, segment, *, rv_kms=None, rv_padding_kms=0.0):
    lo_vac, hi_vac = window.region_A
    medium = str(getattr(segment, "wave_medium", "unknown")).lower()
    warnings = []
    if medium in {"air", "vacuum"}:
        region = convert_wavelength_medium(
            np.array([lo_vac, hi_vac], dtype=float),
            from_medium="vacuum",
            to_medium=medium,
        )
        medium_conversion_applied = medium != "vacuum"
    else:
        region = np.array([lo_vac, hi_vac], dtype=float)
        medium_conversion_applied = False
        warnings.append("segment_wave_medium_unknown_no_conversion")

    stellar_rest = str(getattr(segment, "stellar_rest_status", "unknown")).lower()
    rv_used = 0.0
    if stellar_rest != "corrected" and rv_kms is not None:
        rv_used = float(rv_kms)
        region = region * (1.0 + rv_used / C_KMS)
    elif stellar_rest not in {"corrected", "observed"}:
        warnings.append("stellar_rest_status_unknown_no_rv_shift")

    center = 0.5 * float(region[0] + region[1])
    pad_A = abs(center) * float(rv_padding_kms) / C_KMS
    region = np.array([region[0] - pad_A, region[1] + pad_A], dtype=float)
    return {
        "region_A": [float(np.min(region)), float(np.max(region))],
        "wave_medium": medium,
        "medium_conversion_applied": bool(medium_conversion_applied),
        "rv_used_for_window_kms": float(rv_used),
        "rv_padding_kms": float(rv_padding_kms),
        "coordinate_warnings": warnings,
    }


def _contiguous_run_stats(wave, usable_mask):
    usable_mask = np.asarray(usable_mask, dtype=bool)
    indices = np.where(usable_mask)[0]
    if indices.size == 0:
        return {
            "n_contiguous_runs": 0,
            "largest_run_A": 0.0,
            "largest_gap_A": 0.0,
        }
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    runs = np.split(indices, breaks)
    largest = 0.0
    for run in runs:
        if run.size == 1:
            span = 0.0
        else:
            span = float(abs(wave[run[-1]] - wave[run[0]]))
        largest = max(largest, span)
    ww = np.sort(np.asarray(wave[indices], dtype=float))
    gaps = np.diff(ww)
    largest_gap = float(np.nanmax(gaps)) if gaps.size else 0.0
    return {
        "n_contiguous_runs": int(len(runs)),
        "largest_run_A": float(largest),
        "largest_gap_A": largest_gap,
    }


def _resolution_elements_for_span(segment, span_A, center_A, n_usable):
    span_A = float(span_A)
    if span_A <= 0.0:
        return None
    resolution = getattr(segment, "resolution", None)
    if resolution is None:
        return None
    quantity = getattr(resolution, "quantity", None)
    mode = getattr(resolution, "mode", None)
    try:
        if mode == "constant":
            value = float(resolution.value)
        elif mode == "tabulated":
            value = float(
                np.interp(
                    float(center_A),
                    np.asarray(resolution.wave_A, dtype=float),
                    np.asarray(resolution.values, dtype=float),
                )
            )
        else:
            return None
        if quantity == "R":
            fwhm_A = abs(float(center_A)) / value
        elif quantity == "fwhm_kms":
            fwhm_A = abs(float(center_A)) * value / C_KMS
        elif quantity == "sigma_kms":
            fwhm_A = abs(float(center_A)) * value * 2.354820045 / C_KMS
        else:
            return None
    except Exception:
        return None
    if not np.isfinite(fwhm_A) or fwhm_A <= 0.0:
        return None
    return float(max(0.0, span_A / fwhm_A))


def _detrended_contrast(wave_chunks, flux_chunks):
    wave_chunks = [np.asarray(chunk, dtype=float) for chunk in wave_chunks if len(chunk)]
    flux_chunks = [np.asarray(chunk, dtype=float) for chunk in flux_chunks if len(chunk)]
    if not wave_chunks or not flux_chunks:
        return 0.0
    wave = np.concatenate(wave_chunks)
    flux = np.concatenate(flux_chunks)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    if flux.size < 5:
        return 0.0
    if flux.size >= 8 and np.nanmax(wave) > np.nanmin(wave):
        x = (wave - np.nanmedian(wave)) / max(np.nanmax(wave) - np.nanmin(wave), 1e-30)
        try:
            coeff = np.polyfit(x, flux, deg=1)
            baseline = np.polyval(coeff, x)
            valid = np.isfinite(baseline) & (baseline != 0.0)
            if np.count_nonzero(valid) >= 5:
                flux = flux[valid] / baseline[valid]
        except np.linalg.LinAlgError:
            pass
    p10, p90 = np.nanpercentile(flux, [10.0, 90.0])
    median = np.nanmedian(flux)
    scale = max(abs(float(median)), 1e-30)
    return float(max(0.0, (p90 - p10) / scale))


def _robust_snr(flux_chunks, err_chunks):
    if not flux_chunks or not err_chunks:
        return None
    flux = np.concatenate([np.asarray(chunk, dtype=float) for chunk in flux_chunks])
    err = np.concatenate([np.asarray(chunk, dtype=float) for chunk in err_chunks])
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    if np.count_nonzero(good) < 3:
        return None
    snr = np.nanmedian(np.abs(flux[good]) / err[good])
    if not np.isfinite(snr):
        return None
    return float(snr)


def _selected_records(selection):
    if isinstance(selection, dict):
        records = selection.get("selected", ())
    else:
        records = selection
    return sorted(
        [dict(item) for item in records],
        key=lambda item: (-float(item.get("score", 0.0)), float(item["region_A"][0])),
    )


def _combination_record(kind, label, records):
    roles = sorted({role for item in records for role in item.get("roles", ())})
    families = sorted(
        {family for item in records for family in item.get("feature_family", ())}
    )
    return {
        "id": "{0}:{1}".format(kind, "+".join(item["id"] for item in records)),
        "kind": kind,
        "label": label,
        "window_ids": [item["id"] for item in records],
        "window_labels": [item["label"] for item in records],
        "regions_A": [list(item["region_A"]) for item in records],
        "canonical_regions_vacuum_A": [
            list(item.get("region_vacuum_A", item["region_A"])) for item in records
        ],
        "roles": roles,
        "feature_families": families,
        "default_fit_policies": sorted(
            {item.get("default_fit_policy", "warn") for item in records}
        ),
        "held_out_evaluation_recommended": bool(kind != "all_selected_top"),
        "n_windows": int(len(records)),
        "estimated_usable_pixels": int(
            sum(int(item.get("n_usable_pixels", 0)) for item in records)
        ),
        "score_sum": float(sum(float(item.get("score", 0.0)) for item in records)),
    }


def _json_native(value):
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
