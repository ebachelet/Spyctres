"""Shared spectrum masking and preprocessing provenance helpers."""

from dataclasses import dataclass

import numpy as np

C_KMS = 299792.458


@dataclass(frozen=True)
class NonStellarFeature:
    """Known non-photospheric spectral feature.

    These features are useful for diagnostics and optional masking because a
    stellar-atmosphere model such as PHOENIX is not expected to reproduce them.
    Wavelengths are stored in Angstrom on the observed wavelength scale normally
    used for optical spectra; callers remain responsible for applying any
    explicit frame/medium conversion needed for unusual data products.
    """

    name: str
    center_A: float
    half_width_A: float
    kind: str
    description: str
    reference_ids: tuple = ()
    default_action: str = "warn"
    frame_type: str = "ism_velocity"
    feature_frame: str = "interstellar_rest"
    mask_application_frame: str = "data"
    diagnostic_lines: tuple = ()
    cross_references: tuple = ()
    velocity_margin_kms: float | None = 100.0
    default_region_A: tuple | None = None

    def region(self, padding_A=0.0):
        padding_A = float(padding_A)
        if not np.isfinite(padding_A) or padding_A < 0.0:
            raise ValueError("padding_A must be finite and >= 0.")
        if self.default_region_A is not None:
            wmin, wmax = self.default_region_A
            return (float(wmin) - padding_A, float(wmax) + padding_A)
        half_width = float(self.half_width_A) + padding_A
        return (float(self.center_A) - half_width, float(self.center_A) + half_width)

    def to_metadata(self):
        wmin, wmax = self.region()
        return {
            "name": self.name,
            "center_A": float(self.center_A),
            "half_width_A": float(self.half_width_A),
            "region_A": [float(wmin), float(wmax)],
            "kind": self.kind,
            "feature_type": self.kind,
            "description": self.description,
            "reference_ids": list(self.reference_ids),
            "default_action": self.default_action,
            "frame_type": self.frame_type,
            "feature_frame": self.feature_frame,
            "mask_application_frame": self.mask_application_frame,
            "diagnostic_lines": list(self.diagnostic_lines),
            "cross_references": list(self.cross_references),
            "velocity_margin_kms": (
                None
                if self.velocity_margin_kms is None
                else float(self.velocity_margin_kms)
            ),
        }


NONSTELLAR_FEATURES = {
    "dib_4428": NonStellarFeature(
        name="DIB 4428",
        center_A=4428.8,
        half_width_A=12.0,
        kind="diffuse_interstellar_band",
        description=(
            "Broad diffuse interstellar/circumstellar absorption feature; "
            "not expected to be reproduced by stellar PHOENIX atmospheres."
        ),
        reference_ids=(
            "hobbs2008_dib_catalog_hd204827",
            "garcia_hernandez2013_fullerene_pn_dibs",
        ),
        diagnostic_lines=(),
    ),
    "dib_4882": NonStellarFeature(
        name="DIB 4882",
        center_A=4882.0,
        half_width_A=22.5,
        kind="diffuse_interstellar_band",
        description=(
            "Broad diffuse interstellar absorption feature that can overlap "
            "the red wing of H-beta in hot-star fits."
        ),
        reference_ids=(
            "galazutdinov2020_very_broad_dibs",
            "hobbs2008_dib_catalog_hd204827",
        ),
        diagnostic_lines=("Hbeta",),
        default_region_A=(4870.0, 4915.0),
    ),
    "dib_6284": NonStellarFeature(
        name="DIB 6284",
        center_A=6284.0,
        half_width_A=12.0,
        kind="diffuse_interstellar_band",
        description=(
            "Diffuse interstellar band in a wavelength region that can be "
            "entangled with nearby telluric oxygen absorption."
        ),
        reference_ids=("hobbs2008_dib_catalog_hd204827",),
        cross_references=("telluric_o2_gamma_6280",),
    ),
    "telluric_o2_gamma_6280": NonStellarFeature(
        name="Telluric O2 6280 region",
        center_A=6280.0,
        half_width_A=20.0,
        kind="telluric_band",
        description=(
            "Topocentric oxygen telluric absorption region near DIB 6284; "
            "useful for warning about possible DIB/telluric confusion."
        ),
        reference_ids=(
            "maier2011_h2ccc_dib_candidates",
            "ulmer_moll2019_telluric_correction",
        ),
        frame_type="topocentric_fixed",
        feature_frame="topocentric",
        velocity_margin_kms=None,
        cross_references=("dib_6284",),
    ),
    "telluric_o2_b_6867": NonStellarFeature(
        name="Telluric O2 B band",
        center_A=6867.0,
        half_width_A=28.0,
        kind="telluric_band",
        description=(
            "Topocentric oxygen B-band absorption region; residuals here "
            "should not be interpreted as stellar or interstellar features "
            "without telluric review."
        ),
        reference_ids=("ulmer_moll2019_telluric_correction",),
        frame_type="topocentric_fixed",
        feature_frame="topocentric",
        velocity_margin_kms=None,
    ),
    "telluric_o2_a_7605": NonStellarFeature(
        name="Telluric O2 A band",
        center_A=7605.0,
        half_width_A=55.0,
        kind="telluric_band",
        description=(
            "Strong topocentric oxygen A-band absorption region that often "
            "requires explicit telluric correction or masking."
        ),
        reference_ids=(
            "kimeswenger2020_o2_aband_telluric",
            "ulmer_moll2019_telluric_correction",
        ),
        frame_type="topocentric_fixed",
        feature_frame="topocentric",
        velocity_margin_kms=None,
    ),
    "telluric_h2o_7200": NonStellarFeature(
        name="Telluric H2O 7200",
        center_A=7200.0,
        half_width_A=120.0,
        kind="telluric_band",
        description=(
            "Broad topocentric water-vapour absorption complex; depth is "
            "strongly observing-condition dependent."
        ),
        reference_ids=("ulmer_moll2019_telluric_correction",),
        frame_type="topocentric_fixed",
        feature_frame="topocentric",
        velocity_margin_kms=None,
    ),
    "telluric_h2o_8200": NonStellarFeature(
        name="Telluric H2O 8200",
        center_A=8200.0,
        half_width_A=170.0,
        kind="telluric_band",
        description=(
            "Broad topocentric water-vapour absorption complex; depth is "
            "strongly observing-condition dependent."
        ),
        reference_ids=("ulmer_moll2019_telluric_correction",),
        frame_type="topocentric_fixed",
        feature_frame="topocentric",
        velocity_margin_kms=None,
    ),
    "telluric_h2o_9400": NonStellarFeature(
        name="Telluric H2O 9400",
        center_A=9400.0,
        half_width_A=220.0,
        kind="telluric_band",
        description=(
            "Broad topocentric water-vapour absorption complex; depth is "
            "strongly observing-condition dependent."
        ),
        reference_ids=("ulmer_moll2019_telluric_correction",),
        frame_type="topocentric_fixed",
        feature_frame="topocentric",
        velocity_margin_kms=None,
    ),
}
# Backward-compatible alias for JSON/tests/scripts produced before the 6280 Å
# region was renamed.  The public label avoids the ambiguous "alpha" wording;
# optical O2 A and B conventionally refer to the ~7605 Å and ~6867 Å bands.
NONSTELLAR_FEATURES["telluric_o2_alpha_6280"] = NONSTELLAR_FEATURES[
    "telluric_o2_gamma_6280"
]

OPTICAL_DIB_DIAGNOSTIC_FEATURES = ("dib_4428", "dib_4882")
OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES = (
    "telluric_o2_gamma_6280",
    "telluric_o2_b_6867",
    "telluric_o2_a_7605",
    "telluric_h2o_7200",
    "telluric_h2o_8200",
    "telluric_h2o_9400",
)


def nonstellar_feature_regions(names=("dib_4428",), padding_A=0.0):
    """Return wavelength regions for known non-stellar features.

    Parameters
    ----------
    names : sequence or str
        Feature identifiers from ``NONSTELLAR_FEATURES``.
    padding_A : float, optional
        Extra half-width padding in Angstrom added to every region.
    """
    if isinstance(names, str):
        names = (names,)
    regions = []
    for name in names:
        key = str(name).strip().lower()
        if key not in NONSTELLAR_FEATURES:
            raise KeyError(
                "Unknown non-stellar feature {0!r}. Known features: {1}.".format(
                    name,
                    ", ".join(sorted(NONSTELLAR_FEATURES)),
                )
            )
        regions.append(NONSTELLAR_FEATURES[key].region(padding_A=padding_A))
    return regions


def nonstellar_feature_metadata(names=("dib_4428",), padding_A=0.0):
    """Return JSON-safe metadata for selected non-stellar features."""
    if isinstance(names, str):
        names = (names,)
    out = []
    for name in names:
        key = str(name).strip().lower()
        if key not in NONSTELLAR_FEATURES:
            raise KeyError(
                "Unknown non-stellar feature {0!r}. Known features: {1}.".format(
                    name,
                    ", ".join(sorted(NONSTELLAR_FEATURES)),
                )
            )
        meta = NONSTELLAR_FEATURES[key].to_metadata()
        meta["id"] = key
        if padding_A:
            wmin, wmax = NONSTELLAR_FEATURES[key].region(padding_A=padding_A)
            meta["padding_A"] = float(padding_A)
            meta["region_A"] = [float(wmin), float(wmax)]
        out.append(meta)
    return out


@dataclass(frozen=True)
class ExclusionMaskSpec:
    """Named exclusion-mask callable.

    Exclusion masks use the opposite polarity from ``SpectrumSegment.mask``:
    after numeric thresholding, True means the pixel is rejected/excluded.
    ``ExclusionMaskSpec`` gives that callable a stable provenance name without
    requiring users to remember the accepted tuple/dict shorthand forms.
    Optional metadata is copied into mask provenance when masks are composed.
    """

    name: str
    callable: object
    metadata: dict | None = None

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("ExclusionMaskSpec.name must be a non-empty string.")
        if not callable(self.callable):
            raise TypeError("ExclusionMaskSpec.callable must be callable.")
        object.__setattr__(self, "name", name)
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", metadata)

    def __call__(self, wave):
        return self.callable(wave)


def exclusion_mask(name, fn, metadata=None):
    """Return a named exclusion-mask specification.

    Examples
    --------
    ``exclude_masks=[exclusion_mask("telluric", telluric_fn)]``

    The callable should return boolean-like or numeric values on the supplied
    wavelength grid. Boolean True means reject; numeric values reject where
    ``value > mask_threshold``. Nonfinite numeric outputs are rejected and
    recorded in mask provenance.
    """
    return ExclusionMaskSpec(name=name, callable=fn, metadata=metadata)


def dilate_boolean_mask(mask, n_pix=3):
    """Grow a boolean mask by nearest-neighbour pixels on each side."""
    grown = np.asarray(mask, dtype=bool).copy()
    n_pix = int(max(0, n_pix))
    if n_pix == 0 or grown.size == 0:
        return grown
    for _ in range(n_pix):
        tmp = grown.copy()
        tmp[:-1] |= grown[1:]
        tmp[1:] |= grown[:-1]
        grown = tmp
    return grown


def combine_exclusion_masks(mask_specs, *, name="combined_exclusion_masks", threshold=0.5):
    """Combine named exclusion-mask specs into one compatibility callable.

    Prefer passing ``exclude_masks=[...]`` directly to modern Spyctres fitting
    helpers. This adapter is for older diagnostic paths that still accept only
    one callable. The returned callable preserves component names and metadata
    in the combined spec's provenance.
    """
    if mask_specs is None:
        return None
    specs = list(mask_specs)
    if len(specs) == 0:
        return None
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    normalized = _normalize_mask_specs(exclude_masks=specs)
    component_metadata = {
        item_name: dict(metadata) for item_name, _fn, metadata in normalized
    }

    def _mask(wave):
        wave = np.asarray(wave, dtype=float)
        combined = np.zeros_like(wave, dtype=bool)
        for _item_name, fn, _metadata in normalized:
            cur = np.asarray(fn(wave))
            if cur.dtype == bool:
                combined |= cur
            else:
                combined |= cur > threshold
        return combined

    return exclusion_mask(
        name,
        _mask,
        metadata={
            "method": "combined_exclusion_masks",
            "component_masks": list(component_metadata),
            "component_mask_metadata": component_metadata,
            "numeric_mask_threshold": threshold,
        },
    )


def telluric_transmission_exclusion_mask(
    threshold=0.95,
    *,
    name="telluric:transmission_threshold",
    loader=None,
):
    """Return a named high-resolution telluric transmission exclusion mask.

    This wraps Spyctres' legacy ``load_telluric_lines(threshold)`` mechanism in
    the newer named-mask API.  It is intentionally distinct from the broad
    Phase-A telluric catalog regions: this helper is the preferred opt-in path
    for actual telluric masking when the wavelength frame is suitable.

    The returned callable follows the exclusion-mask convention:
    ``True`` means reject this pixel from the stellar fit.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("telluric transmission threshold must satisfy 0 < threshold < 1.")
    if loader is None:
        from .Spyctres import load_telluric_lines

        loader = load_telluric_lines
    _transmission, telluric_mask = loader(threshold)

    def _mask(wave):
        return np.asarray(telluric_mask(wave)) > 0.5

    metadata = {
        "mask_type": "telluric",
        "method": "transmission_threshold",
        "model_file": "LBL_A10_s0_w050_R0300000_T.fits",
        "threshold": threshold,
        "frame_type": "topocentric_fixed",
        "feature_frame": "topocentric",
        "action": "masked",
        "coarse_mask": False,
        "fallback_broad_regions_used": False,
        "fallback_broad_region_ids": [],
        "preferred_for_actual_telluric_masking": True,
        "note": (
            "High-resolution telluric transmission-threshold mask; distinct "
            "from broad catalog-region telluric warnings."
        ),
    }
    return exclusion_mask(name, _mask, metadata=metadata)


def broad_telluric_catalog_fallback_mask(
    names=OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES,
    padding_A=0.0,
    *,
    name="telluric:broad_catalog_fallback",
    use_case="quicklook/product_masking_fallback",
):
    """Return a coarse broad-catalog telluric fallback mask.

    This helper is intentionally labelled as a fallback/quicklook product mask.
    It should not be used as the preferred telluric fit mask when the
    high-resolution transmission-threshold model is available.
    """
    feature_names = tuple(names) if not isinstance(names, str) else (names,)
    regions = nonstellar_feature_regions(feature_names, padding_A=padding_A)
    metadata = _catalog_feature_mask_metadata(feature_names, padding_A=padding_A)
    metadata.update(
        {
            "method": "broad_catalog_fallback",
            "fallback_broad_regions_used": True,
            "fallback_broad_region_ids": list(metadata["catalog_feature_ids"]),
            "preferred_for_actual_telluric_masking": False,
            "use_case": str(use_case),
            "note": (
                "Coarse catalog-region telluric fallback for quicklook/product "
                "masking; prefer telluric_transmission_exclusion_mask() for "
                "actual telluric masking when the transmission model is available."
            ),
        }
    )
    return exclusion_mask(
        name,
        lambda wave: _inside_regions(np.asarray(wave, dtype=float), regions),
        metadata=metadata,
    )


def nonstellar_feature_mask(names=("dib_4428",), padding_A=0.0):
    """Return a named exclusion mask for non-stellar features.

    The returned mask follows the explicit-exclusion convention:
    ``True`` means reject this pixel from the stellar fit.
    """
    feature_names = tuple(names) if not isinstance(names, str) else (names,)
    regions = nonstellar_feature_regions(feature_names, padding_A=padding_A)
    label = "nonstellar:" + "+".join(str(name).strip().lower() for name in feature_names)
    metadata = _catalog_feature_mask_metadata(feature_names, padding_A=padding_A)
    return exclusion_mask(
        label,
        lambda wave: _inside_regions(np.asarray(wave, dtype=float), regions),
        metadata=metadata,
    )


def nonstellar_feature_masks(names=("dib_4428",), padding_A=0.0):
    """Return one named exclusion mask per selected non-stellar feature.

    This keeps mask provenance separate for each feature, which is preferable
    for audit trails and pixel-count summaries. The masks are still applied in
    the current data wavelength grid; no frame transformation is implicit.
    """
    if isinstance(names, str):
        names = (names,)
    return [
        nonstellar_feature_mask((str(name).strip().lower(),), padding_A=padding_A)
        for name in names
    ]


def known_feature_masks(names=("dib_4428",), padding_A=0.0):
    """Alias for non-stellar feature masks using public diagnostic terminology."""
    return nonstellar_feature_masks(names=names, padding_A=padding_A)


def _catalog_feature_mask_metadata(names, padding_A=0.0):
    metadata = nonstellar_feature_metadata(names, padding_A=padding_A)
    kinds = {item.get("kind") for item in metadata}
    telluric_ids = [
        item["id"] for item in metadata if item.get("kind") == "telluric_band"
    ]
    return {
        "mask_type": "nonstellar_catalog_region",
        "method": "broad_catalog_regions",
        "catalog_feature_ids": [item["id"] for item in metadata],
        "catalog_feature_names": [item["name"] for item in metadata],
        "feature_types": sorted(str(kind) for kind in kinds if kind),
        "regions_A": [item["region_A"] for item in metadata],
        "padding_A": float(padding_A),
        "action": "masked",
        "coarse_mask": bool(telluric_ids),
        "warning": "coarse_telluric_mask" if telluric_ids else None,
        "quality_flags": (
            ["coarse_telluric_mask_applied"] if telluric_ids else []
        ),
        "frame_type": (
            "topocentric_fixed"
            if telluric_ids and len(telluric_ids) == len(metadata)
            else "mixed_or_feature_specific"
        ),
        "feature_frame": (
            "topocentric"
            if telluric_ids and len(telluric_ids) == len(metadata)
            else "mixed_or_feature_specific"
        ),
    }


def overlapping_nonstellar_features(spectrum_or_segments, names=("dib_4428",), padding_A=0.0):
    """Return selected non-stellar features overlapping valid spectrum coverage."""
    if isinstance(names, str):
        names = (names,)
    if hasattr(spectrum_or_segments, "segments"):
        segments = list(spectrum_or_segments.segments)
    elif isinstance(spectrum_or_segments, (list, tuple)):
        segments = list(spectrum_or_segments)
    else:
        segments = [spectrum_or_segments]

    overlaps = []
    for meta in nonstellar_feature_metadata(names, padding_A=padding_A):
        wmin, wmax = meta["region_A"]
        total_overlap = 0.0
        total_overlap_pixels = 0
        total_valid_pixels = 0
        segment_names = []
        segment_overlaps = []
        for segment in segments:
            wave = np.asarray(segment.wave, dtype=float)
            mask = np.asarray(getattr(segment, "mask", np.ones(wave.shape, dtype=bool)), dtype=bool)
            good = mask & np.isfinite(wave)
            if not np.any(good):
                continue
            total_valid_pixels += int(np.count_nonzero(good))
            lo = float(np.nanmin(wave[good]))
            hi = float(np.nanmax(wave[good]))
            overlap = max(0.0, min(wmax, hi) - max(wmin, lo))
            if overlap > 0.0:
                in_feature = good & (wave >= float(wmin)) & (wave <= float(wmax))
                overlap_pixels = int(np.count_nonzero(in_feature))
                total_overlap += overlap
                total_overlap_pixels += overlap_pixels
                segment_names.append(getattr(segment, "name", None))
                segment_overlaps.append(
                    {
                        "segment": getattr(segment, "name", None),
                        "overlap_A": float(overlap),
                        "overlap_pixels": overlap_pixels,
                        "valid_pixels": int(np.count_nonzero(good)),
                        "overlap_fraction_of_valid_pixels": float(
                            overlap_pixels / max(1, int(np.count_nonzero(good)))
                        ),
                    }
                )
        if total_overlap > 0.0:
            item = dict(meta)
            item["overlap_A"] = float(total_overlap)
            item["overlap_pixels"] = int(total_overlap_pixels)
            item["valid_pixels"] = int(total_valid_pixels)
            item["overlap_fraction_of_valid_pixels"] = float(
                total_overlap_pixels / max(1, total_valid_pixels)
            )
            item["segments"] = [name for name in segment_names if name]
            item["segment_overlaps"] = segment_overlaps
            overlaps.append(item)
    return overlaps


@dataclass(frozen=True)
class MaskResult:
    """Result of composing all masks for one spectrum segment."""

    effective_mask: np.ndarray
    excluded_mask: np.ndarray
    rejection_masks: dict
    settings: dict

    @property
    def fit_use_mask(self):
        """Alias for ``effective_mask``; True means the pixel is fitted."""
        return self.effective_mask

    @property
    def explicit_exclusion_mask(self):
        """Alias for ``excluded_mask``; True means explicitly rejected."""
        return self.excluded_mask

    @property
    def data_invalid_mask(self):
        """Pixels rejected by the segment mask or invalid wave/flux/error."""
        out = np.zeros_like(self.effective_mask, dtype=bool)
        for key in ("segment_mask", "invalid_wave", "invalid_flux", "invalid_error"):
            if key in self.rejection_masks:
                out |= np.asarray(self.rejection_masks[key], dtype=bool)
        return out

    @property
    def fit_rejection_mask(self):
        """Pixels rejected for any reason; inverse of ``effective_mask``."""
        return ~np.asarray(self.effective_mask, dtype=bool)

    @property
    def counts(self):
        """Return JSON-safe pixel counts for each rejection reason.

        Per-reason counts are raw reason counts: if a pixel is rejected by two
        reasons, it contributes to both reason counts. The total/union counts
        below are overlap-aware.
        """
        reason_counts = {
            key: int(np.count_nonzero(mask))
            for key, mask in self.rejection_masks.items()
        }
        overlap_reason_masks = [
            np.asarray(mask, dtype=bool)
            for key, mask in self.rejection_masks.items()
            if key not in {"exclude_mask", "nonfinite_mask_output"}
        ]
        reason_stack = (
            np.vstack(
                overlap_reason_masks
            )
            if overlap_reason_masks
            else np.zeros((0, self.effective_mask.size), dtype=bool)
        )
        n_reasons = (
            np.sum(reason_stack, axis=0)
            if reason_stack.size
            else np.zeros_like(self.effective_mask, dtype=int)
        )
        counts = dict(reason_counts)
        counts["reason_counts"] = dict(reason_counts)
        counts["total"] = int(self.effective_mask.size)
        counts["used"] = int(np.count_nonzero(self.effective_mask))
        counts["rejected"] = counts["total"] - counts["used"]
        counts["explicitly_excluded"] = int(np.count_nonzero(self.excluded_mask))
        counts["n_pixels"] = counts["total"]
        counts["n_fit"] = counts["used"]
        counts["n_rejected_total"] = counts["rejected"]
        counts["n_rejected_by_reason"] = dict(reason_counts)
        counts["n_rejected_by_explicit_union"] = int(
            np.count_nonzero(self.explicit_exclusion_mask)
        )
        counts["n_rejected_by_data_invalid"] = int(
            np.count_nonzero(self.data_invalid_mask)
        )
        counts["n_rejected_by_multiple_reasons"] = int(np.count_nonzero(n_reasons > 1))
        outside_regions = np.asarray(
            self.rejection_masks.get(
                "outside_regions",
                np.zeros_like(self.effective_mask, dtype=bool),
            ),
            dtype=bool,
        )
        inside_regions = ~outside_regions
        counts["n_outside_fit_window"] = int(np.count_nonzero(outside_regions))
        counts["n_inside_fit_window"] = int(np.count_nonzero(inside_regions))
        counts["n_rejected_inside_fit_window"] = int(
            np.count_nonzero(inside_regions & self.fit_rejection_mask)
        )
        return counts

    def to_metadata(self, label="fit_mask"):
        """Return a compact, JSON-safe provenance record."""
        return {
            "operation": "mask",
            "label": str(label),
            "settings": dict(self.settings),
            "counts": self.counts,
        }

    def to_summary(self):
        """Return a compact JSON-safe mask summary for notebooks/GUI output."""
        counts = self.counts
        total = max(1, int(counts["n_pixels"]))
        explicit_names = list(self.settings.get("individual_exclusion_masks", []))
        explicit_counts = {
            name: int(counts.get("exclude_mask:{0}".format(name), 0))
            for name in explicit_names
        }
        return {
            "n_pixels": int(counts["n_pixels"]),
            "n_fit": int(counts["n_fit"]),
            "n_rejected_total": int(counts["n_rejected_total"]),
            "fit_fraction": float(counts["n_fit"] / total),
            "rejected_fraction": float(counts["n_rejected_total"] / total),
            "n_outside_fit_window": int(counts["n_outside_fit_window"]),
            "outside_fit_window_fraction": float(
                counts["n_outside_fit_window"] / total
            ),
            "n_inside_fit_window": int(counts["n_inside_fit_window"]),
            "n_rejected_inside_fit_window": int(
                counts["n_rejected_inside_fit_window"]
            ),
            "rejected_inside_fit_window_fraction": float(
                0.0
                if counts["n_inside_fit_window"] <= 0
                else counts["n_rejected_inside_fit_window"]
                / float(counts["n_inside_fit_window"])
            ),
            "n_rejected_by_data_invalid": int(counts["n_rejected_by_data_invalid"]),
            "data_invalid_fraction": float(
                counts["n_rejected_by_data_invalid"] / total
            ),
            "n_rejected_by_explicit_union": int(
                counts["n_rejected_by_explicit_union"]
            ),
            "explicit_exclusion_fraction": float(
                counts["n_rejected_by_explicit_union"] / total
            ),
            "n_rejected_by_multiple_reasons": int(
                counts["n_rejected_by_multiple_reasons"]
            ),
            "multiple_rejection_fraction": float(
                counts["n_rejected_by_multiple_reasons"] / total
            ),
            "explicit_exclusion_counts": explicit_counts,
            "mask_true_means": self.settings.get("mask_true_means", "use"),
            "exclude_mask_true_means": self.settings.get(
                "exclude_mask_true_means",
                "reject",
            ),
        }


def convert_mask_polarity(mask, input_true_means="reject", output_true_means="use"):
    """Convert a mask between common polarity conventions.

    Parameters
    ----------
    mask : array-like
        Boolean-like mask.
    input_true_means, output_true_means : {"use", "reject"}
        Input and desired output polarity. ``"reject"`` corresponds to the
        Astropy/Specutils/NumPy masked-array convention; ``"use"`` corresponds
        to Spyctres' internal fit-use convention.
    """
    input_true_means = str(input_true_means).strip().lower()
    output_true_means = str(output_true_means).strip().lower()
    allowed = {"use", "reject"}
    if input_true_means not in allowed or output_true_means not in allowed:
        raise ValueError("Mask polarity must be 'use' or 'reject'.")
    out = np.asarray(mask, dtype=bool)
    if input_true_means == output_true_means:
        return np.array(out, dtype=bool, copy=True)
    return ~out


def coerce_boolean_mask(
    values,
    shape,
    threshold=0.5,
    name="mask",
    nonfinite_policy="false",
):
    """Convert boolean/numeric mask values to a validated boolean array."""
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    nonfinite_policy = str(nonfinite_policy).strip().lower()
    if nonfinite_policy not in {"false", "true", "raise"}:
        raise ValueError("nonfinite_policy must be 'false', 'true', or 'raise'.")
    array = np.asarray(values)
    if array.shape != tuple(shape):
        raise ValueError(
            "{0} must have shape {1}; got {2}.".format(
                name, tuple(shape), array.shape
            )
        )
    if array.dtype == bool:
        return np.array(array, dtype=bool, copy=True)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("{0} must contain boolean or real numeric values.".format(name))
    finite = np.isfinite(array)
    if nonfinite_policy == "raise" and np.any(~finite):
        raise ValueError("{0} contains nonfinite values.".format(name))
    out = np.asarray(array > threshold, dtype=bool)
    if nonfinite_policy == "true":
        out |= ~finite
    return out


def _nonfinite_numeric_mask(values, shape, name):
    """Return nonfinite positions for validated real numeric mask values."""
    array = np.asarray(values)
    if array.shape != tuple(shape):
        raise ValueError(
            "{0} must have shape {1}; got {2}.".format(
                name, tuple(shape), array.shape
            )
        )
    if array.dtype == bool:
        return np.zeros(tuple(shape), dtype=bool)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("{0} must contain boolean or real numeric values.".format(name))
    return ~np.isfinite(array)


def _mask_callable_name(mask_spec, fallback="exclude_mask"):
    """Return a stable, human-readable label for a mask callable spec."""
    if isinstance(mask_spec, ExclusionMaskSpec):
        return mask_spec.name
    if isinstance(mask_spec, dict):
        if "name" in mask_spec:
            return str(mask_spec["name"])
        mask_spec = mask_spec.get("callable", mask_spec.get("func"))
    if (
        isinstance(mask_spec, tuple)
        and len(mask_spec) == 2
        and isinstance(mask_spec[0], str)
    ):
        return str(mask_spec[0])
    return getattr(mask_spec, "__name__", type(mask_spec).__name__) or fallback


def _mask_callable_metadata(mask_spec):
    """Return JSON-safe metadata attached to a mask callable spec."""
    if isinstance(mask_spec, ExclusionMaskSpec):
        return dict(mask_spec.metadata or {})
    if isinstance(mask_spec, dict):
        metadata = mask_spec.get("metadata", {})
        return dict(metadata or {})
    return {}


def _is_named_mask_tuple(mask_spec):
    return (
        isinstance(mask_spec, tuple)
        and len(mask_spec) == 2
        and isinstance(mask_spec[0], str)
    )


def _mask_callable_function(mask_spec):
    """Extract the callable from a supported mask callable spec."""
    if isinstance(mask_spec, ExclusionMaskSpec):
        return mask_spec.callable
    if isinstance(mask_spec, dict):
        if "callable" in mask_spec:
            mask_spec = mask_spec["callable"]
        elif "func" in mask_spec:
            mask_spec = mask_spec["func"]
        else:
            raise TypeError(
                "Mask spec dictionaries require a 'callable' or 'func' key. "
                "Accepted exclusion-mask forms are: callable, "
                "('name', callable), {'name': 'name', 'callable': callable}, "
                "ExclusionMaskSpec, or a list/tuple of those forms."
            )
    elif _is_named_mask_tuple(mask_spec):
        mask_spec = mask_spec[1]
    if not callable(mask_spec):
        raise TypeError(
            "Mask specs must be callables or named callable specs. Accepted "
            "forms are: callable, ('name', callable), "
            "{'name': 'name', 'callable': callable}, ExclusionMaskSpec, "
            "or a list/tuple of those forms."
        )
    return mask_spec


def _normalize_mask_specs(exclude_mask=None, exclude_masks=None):
    """Normalize mask specs to ``(name, callable, metadata)`` triples."""
    if exclude_mask is not None and exclude_masks is not None:
        raise ValueError(
            "Pass exclusion masks through either exclude_mask or exclude_masks, "
            "not both. Use exclude_masks for the preferred named/multiple-mask API."
        )
    specs = []
    if exclude_mask is not None:
        if (
            not callable(exclude_mask)
            and not isinstance(exclude_mask, dict)
            and not _is_named_mask_tuple(exclude_mask)
        ):
            specs.extend(list(exclude_mask))
        else:
            specs.append(exclude_mask)
    if exclude_masks is not None:
        if (
            callable(exclude_masks)
            or isinstance(exclude_masks, dict)
            or _is_named_mask_tuple(exclude_masks)
        ):
            specs.append(exclude_masks)
        else:
            specs.extend(list(exclude_masks))

    normalized = []
    used_names = set()
    for spec in specs:
        name = _mask_callable_name(spec)
        fn = _mask_callable_function(spec)
        metadata = _mask_callable_metadata(spec)
        if name in used_names:
            raise ValueError(
                "Duplicate exclusion mask name {0!r}; use unique names so "
                "mask provenance and overlap counts remain unambiguous.".format(name)
            )
        used_names.add(name)
        normalized.append((name, fn, metadata))
    return normalized


def _telluric_frame_warning_for_segment(seg, callable_name, metadata):
    """Return a frame warning for topocentric telluric masks, if needed."""
    if metadata.get("frame_type") != "topocentric_fixed":
        return None
    mask_type = str(metadata.get("mask_type", "")).lower()
    method = str(metadata.get("method", "")).lower()
    is_telluric = (
        mask_type == "telluric"
        or "telluric" in mask_type
        or method == "transmission_threshold"
        or any(
            str(feature_type) == "telluric_band"
            for feature_type in metadata.get("feature_types", [])
        )
    )
    if not is_telluric:
        return None
    observer_frame = str(getattr(seg, "observer_frame", "unknown")).lower()
    stellar_rest_status = str(
        getattr(seg, "stellar_rest_status", "unknown")
    ).lower()
    reasons = []
    if observer_frame != "topocentric":
        reasons.append("observer_frame_not_known_topocentric")
    if stellar_rest_status != "raw":
        reasons.append("stellar_rest_status_not_known_raw")
    if not reasons:
        return None
    return {
        "mask": callable_name,
        "warning": "telluric_mask_frame_ambiguous",
        "frame_type": metadata.get("frame_type"),
        "feature_frame": metadata.get("feature_frame"),
        "observer_frame": observer_frame,
        "stellar_rest_status": stellar_rest_status,
        "reasons": reasons,
        "message": (
            "Telluric features are topocentric. This spectrum is not known to "
            "be on a raw topocentric wavelength grid, so fixed telluric mask "
            "intervals may be misaligned."
        ),
    }


def _inside_regions(wave, regions):
    mask = np.zeros_like(wave, dtype=bool)
    for region in regions:
        if len(region) != 2:
            raise ValueError("Each wavelength region must contain (wmin, wmax).")
        wmin, wmax = map(float, region)
        if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax < wmin:
            raise ValueError("Wavelength regions require finite wmin <= wmax.")
        mask |= (wave >= wmin) & (wave <= wmax)
    return mask


def _serialize_regions(regions):
    if regions is None:
        return None
    return [[float(wmin), float(wmax)] for wmin, wmax in regions]


def compose_fit_mask(
    seg,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
):
    """Compose fit masks once and retain a reason-resolved audit trail.

    ``effective_mask`` is True only where a pixel may be fitted. The separate
    reason masks are True where that reason rejects a pixel; reasons may
    overlap. ``excluded_mask`` preserves the historical plotting definition:
    it contains wavelength-rule/callable exclusions, but not invalid data or
    pixels disabled by the segment's existing mask.

    Mask polarity is explicit: input segment masks use ``True == valid/use``;
    exclude-mask callables use ``True == reject/exclude`` after thresholding.
    A callable may also be supplied as ``("name", fn)`` or
    ``{"name": "name", "callable": fn}`` so provenance remains readable.
    """
    regions = None if regions is None else tuple(regions)
    exclude_regions = None if exclude_regions is None else tuple(exclude_regions)
    mask_threshold = float(mask_threshold)
    if not np.isfinite(mask_threshold):
        raise ValueError("mask_threshold must be finite.")

    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    if wave.shape != flux.shape or wave.ndim != 1:
        raise ValueError("Segment wave and flux must be matching 1D arrays.")

    shape = wave.shape
    base_mask = coerce_boolean_mask(seg.mask, shape, name="segment mask")
    reasons = {
        "segment_mask": ~base_mask,
        "invalid_wave": ~np.isfinite(wave),
        "invalid_flux": ~np.isfinite(flux),
    }

    if seg.err is None:
        reasons["invalid_error"] = np.zeros(shape, dtype=bool)
    else:
        err = np.asarray(seg.err, dtype=float)
        if err.shape != shape:
            raise ValueError("Segment err must match wave and flux shape.")
        reasons["invalid_error"] = ~np.isfinite(err) | (err <= 0)

    if regions is None:
        reasons["outside_regions"] = np.zeros(shape, dtype=bool)
    else:
        reasons["outside_regions"] = ~_inside_regions(wave, regions)

    if exclude_regions is None:
        reasons["exclude_regions"] = np.zeros(shape, dtype=bool)
    else:
        reasons["exclude_regions"] = _inside_regions(wave, exclude_regions)

    callable_masks = []
    nonfinite_callable_masks = []
    callable_metadata = {}
    telluric_frame_warnings = []
    callable_quality_flags = []
    for callable_name, callable_fn, metadata in _normalize_mask_specs(
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
    ):
        callable_metadata[callable_name] = dict(metadata)
        callable_quality_flags.extend(str(flag) for flag in metadata.get("quality_flags", []))
        warning = _telluric_frame_warning_for_segment(seg, callable_name, metadata)
        if warning is not None:
            telluric_frame_warnings.append(warning)
            callable_quality_flags.append("telluric_mask_frame_ambiguous")
        raw_values = callable_fn(wave)
        callable_mask = coerce_boolean_mask(
            raw_values,
            shape,
            threshold=mask_threshold,
            name="{0} result".format(callable_name),
            nonfinite_policy="true",
        )
        nonfinite_mask = _nonfinite_numeric_mask(
            raw_values,
            shape,
            name="{0} result".format(callable_name),
        )
        callable_masks.append((callable_name, callable_mask))
        nonfinite_callable_masks.append((callable_name, nonfinite_mask))

    if callable_masks:
        exclude_union = np.zeros(shape, dtype=bool)
        nonfinite_union = np.zeros(shape, dtype=bool)
        for callable_name, callable_mask in callable_masks:
            exclude_union |= callable_mask
            reasons["exclude_mask:{0}".format(callable_name)] = callable_mask
        for callable_name, nonfinite_mask in nonfinite_callable_masks:
            nonfinite_union |= nonfinite_mask
            if np.any(nonfinite_mask):
                reasons[
                    "nonfinite_mask_output:{0}".format(callable_name)
                ] = nonfinite_mask
        reasons["exclude_mask"] = exclude_union
        reasons["nonfinite_mask_output"] = nonfinite_union
        callable_names = [name for name, _mask in callable_masks]
    else:
        reasons["exclude_mask"] = np.zeros(shape, dtype=bool)
        reasons["nonfinite_mask_output"] = np.zeros(shape, dtype=bool)
        callable_names = []

    rejected = np.zeros(shape, dtype=bool)
    for reason_mask in reasons.values():
        rejected |= reason_mask

    explicitly_excluded = (
        reasons["outside_regions"]
        | reasons["exclude_regions"]
        | reasons["exclude_mask"]
    )
    settings = {
        "regions": _serialize_regions(regions),
        "exclude_regions": _serialize_regions(exclude_regions),
        "exclude_mask": callable_names[0] if len(callable_names) == 1 else None,
        "exclude_masks": list(callable_names),
        "exclude_masks_api": (
            "exclude_masks" if exclude_masks is not None else "exclude_mask"
        ),
        "exclude_mask_summary_kind": "union",
        "explicit_exclusion_union_mask_name": "exclude_mask",
        "individual_exclusion_masks": list(callable_names),
        "exclude_mask_union_derived_from": list(callable_names),
        "exclude_mask_metadata": {
            name: callable_metadata.get(name, {}) for name in callable_names
        },
        "telluric_mask_frame_warnings": telluric_frame_warnings,
        "quality_flags": sorted(set(callable_quality_flags)),
        "mask_threshold": mask_threshold,
        "numeric_mask_threshold": mask_threshold,
        "numeric_mask_reject_if": "> threshold",
        "mask_true_means": "use",
        "boolean_mask_true_means": "reject",
        "exclude_mask_true_means": "reject",
        "nonfinite_mask_value_policy": "reject",
    }
    return MaskResult(
        effective_mask=~rejected,
        excluded_mask=explicitly_excluded,
        rejection_masks={key: value.copy() for key, value in reasons.items()},
        settings=settings,
    )


def apply_fit_mask(
    seg,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
    label="fit_mask",
):
    """Return a masked segment copy, its mask result, and recorded provenance."""
    result = compose_fit_mask(
        seg,
        regions=regions,
        exclude_regions=exclude_regions,
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
        mask_threshold=mask_threshold,
    )
    meta = dict(seg.meta)
    history = list(meta.get("preprocessing", []))
    history.append(result.to_metadata(label=label))
    meta["preprocessing"] = history
    return seg.copy(mask=result.effective_mask, meta=meta), result


def _json_safe_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_safe_scalar(value)


def _audit_segments(spectrum):
    if hasattr(spectrum, "segments"):
        return list(spectrum.segments)
    if (
        isinstance(spectrum, (list, tuple))
        and not hasattr(spectrum, "wave")
        and not hasattr(spectrum, "flux")
    ):
        return list(spectrum)
    return [spectrum]


def _regions_mask(wave, regions):
    wave = np.asarray(wave, dtype=float)
    if regions is None:
        return np.ones(wave.shape, dtype=bool)
    return _inside_regions(wave, regions)


def _resolution_metadata(seg, assumed_resolution=None):
    resolution = getattr(seg, "resolution", None)
    if assumed_resolution is not None:
        if isinstance(assumed_resolution, dict):
            quantity = str(assumed_resolution.get("quantity", "R"))
            value = assumed_resolution.get("value", assumed_resolution.get("R"))
            source = assumed_resolution.get("source", "user_override")
            warning = assumed_resolution.get(
                "assumption_warning",
                "approximate quicklook resolution",
            )
        else:
            quantity = "R"
            value = assumed_resolution
            source = "user_override"
            warning = "approximate quicklook resolution"
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        return {
            "present": value is not None and np.isfinite(value) and value > 0,
            "quantity": quantity,
            "mode": "constant",
            "value": None if value is None else float(value),
            "source": str(source),
            "assumption_warning": str(warning),
            "assumed": True,
        }
    if resolution is None:
        return {
            "present": False,
            "quantity": None,
            "mode": None,
            "value": None,
            "source": None,
            "assumed": False,
        }
    if hasattr(resolution, "to_metadata"):
        meta = dict(resolution.to_metadata())
    elif isinstance(resolution, dict):
        meta = dict(resolution)
    else:
        meta = {"quantity": "R", "mode": "constant", "value": resolution}
    return {
        "present": True,
        "quantity": meta.get("quantity"),
        "mode": meta.get("mode", "constant"),
        "value": _json_safe_scalar(meta.get("value")),
        "source": meta.get("source"),
        "assumed": False,
        "metadata": _json_safe(meta),
    }


def _resolution_fwhm_A(wave, resolution_meta):
    if not resolution_meta.get("present"):
        return None
    if str(resolution_meta.get("mode", "constant")).lower() != "constant":
        return None
    value = resolution_meta.get("value")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    quantity = str(resolution_meta.get("quantity"))
    wave = np.asarray(wave, dtype=float)
    if quantity == "R":
        return wave / value
    if quantity == "fwhm_kms":
        return wave * value / C_KMS
    if quantity == "sigma_kms":
        return wave * (2.354820045 * value) / C_KMS
    return None


def _lsf_provenance_metadata(seg, resolution_meta, assumed_resolution=None):
    meta = getattr(seg, "meta", {}) or {}
    sdss_lsf = dict(meta.get("sdss_lsf", {}) or {})
    warnings = []
    if sdss_lsf.get("present"):
        attached = bool(sdss_lsf.get("attach_wdisp_resolution", False))
        if attached:
            lsf_source = sdss_lsf.get(
                "lsf_source",
                "sdss_wdisp_attached_as_tabulated_sigma_kms",
            )
            warnings.append(
                "SDSS tabulated LSF present as provenance; current PHOENIX "
                "fitter does not apply wavelength-dependent LSF convolution."
            )
        else:
            lsf_source = sdss_lsf.get("lsf_source", "sdss_wdisp_not_applied")
            if assumed_resolution is not None or resolution_meta.get("present"):
                warnings.append(
                    "SDSS tabulated LSF present but not applied; using explicit "
                    "constant-R assumption."
                )
            else:
                warnings.append(
                    "SDSS tabulated LSF present but not applied; no active LSF "
                    "broadening assumption supplied."
                )
        return {
            "lsf_source": lsf_source,
            "sdss_wdisp_present": True,
            "sdss_wdisp_attached": attached,
            "active_lsf_convolution": False,
            "warning": warnings[0],
            "warnings": warnings,
        }

    source = resolution_meta.get("source")
    if resolution_meta.get("present"):
        source = source or "segment_or_user_constant_lsf"
    return {
        "lsf_source": source,
        "sdss_wdisp_present": False,
        "sdss_wdisp_attached": False,
        "active_lsf_convolution": bool(resolution_meta.get("present")),
        "warning": None,
        "warnings": [],
    }


def _count_true_blocks(mask, min_length):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    count = 0
    run = 0
    for value in mask:
        if value:
            run += 1
        else:
            if run >= min_length:
                count += 1
            run = 0
    if run >= min_length:
        count += 1
    return int(count)


def _fraction(numerator, denominator):
    denominator = int(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _segment_readiness(
    seg,
    regions=None,
    assumed_resolution=None,
    spike_sigma=10.0,
    flat_block_min=5,
    gap_factor=5.0,
):
    wave = np.asarray(getattr(seg, "wave"), dtype=float)
    flux = np.asarray(getattr(seg, "flux"), dtype=float)
    if wave.shape != flux.shape:
        raise ValueError("readiness audit requires matching wave/flux shapes.")
    n_total = int(wave.size)
    input_mask = np.asarray(
        getattr(seg, "mask", np.ones(wave.shape, dtype=bool)),
        dtype=bool,
    )
    if input_mask.shape != wave.shape:
        raise ValueError("readiness audit requires mask shape to match wave.")
    err = getattr(seg, "err", None)
    if err is None:
        err_good = np.ones(wave.shape, dtype=bool)
        errors_present = False
    else:
        err = np.asarray(err, dtype=float)
        if err.shape != wave.shape:
            raise ValueError("readiness audit requires err shape to match wave.")
        err_good = np.isfinite(err) & (err > 0)
        errors_present = True

    finite_wave = np.isfinite(wave) & (wave > 0)
    finite_flux = np.isfinite(flux)
    finite = finite_wave & finite_flux
    inside = _regions_mask(wave, regions)
    candidate = input_mask & finite & err_good
    fit_candidate = candidate & inside

    finite_good_flux = flux[finite & input_mask]
    if finite_good_flux.size:
        scale = float(np.nanmedian(np.abs(finite_good_flux)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    else:
        scale = 1.0
    near_zero_threshold = max(1.0e-30, 1.0e-6 * scale)
    near_zero = finite_flux & (np.abs(flux) <= near_zero_threshold)
    negative_flux = finite_flux & (flux < 0.0)

    spike = np.zeros(wave.shape, dtype=bool)
    robust_pool = flux[fit_candidate]
    if robust_pool.size >= 10:
        median = float(np.nanmedian(robust_pool))
        mad = float(np.nanmedian(np.abs(robust_pool - median)))
        robust_sigma = 1.4826 * mad
        if np.isfinite(robust_sigma) and robust_sigma > 0:
            spike = (
                fit_candidate
                & np.isfinite(flux)
                & (np.abs(flux - median) > float(spike_sigma) * robust_sigma)
            )

    finite_sorted_wave = np.sort(wave[finite_wave])
    spacing_median = None
    max_gap_A = None
    large_gap_count = 0
    if finite_sorted_wave.size >= 3:
        spacing = np.diff(finite_sorted_wave)
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        if spacing.size:
            spacing_median = float(np.nanmedian(spacing))
            max_gap_A = float(np.nanmax(spacing))
            large_gap_count = int(np.sum(spacing > float(gap_factor) * spacing_median))

    resolution_meta = _resolution_metadata(seg, assumed_resolution=assumed_resolution)
    pixels_per_fwhm_median = None
    lsf_undersampled = None
    fwhm_A = _resolution_fwhm_A(wave, resolution_meta)
    if fwhm_A is not None and spacing_median is not None and spacing_median > 0:
        sampled = fwhm_A[finite_wave & np.isfinite(fwhm_A) & (fwhm_A > 0)]
        if sampled.size:
            pixels_per_fwhm_median = float(np.nanmedian(sampled) / spacing_median)
            lsf_undersampled = bool(pixels_per_fwhm_median < 2.0)

    bad_error = ~err_good if errors_present else np.zeros(wave.shape, dtype=bool)
    artifact_mask = (~finite) | near_zero | negative_flux | spike | (~input_mask) | bad_error
    artifact_inside = artifact_mask & inside
    flat_block_count = _count_true_blocks(near_zero & inside, flat_block_min)
    n_inside = int(np.sum(inside))
    n_fit = int(np.sum(fit_candidate))
    rejected_inside = int(np.sum(inside & ~fit_candidate))
    lsf_provenance = _lsf_provenance_metadata(
        seg,
        resolution_meta,
        assumed_resolution=assumed_resolution,
    )
    metadata = {
        "wave_medium": getattr(seg, "wave_medium", "unknown"),
        "observer_frame": getattr(seg, "observer_frame", "unknown"),
        "stellar_rest_status": getattr(seg, "stellar_rest_status", "unknown"),
        "errors_present": bool(errors_present),
        "resolution": resolution_meta,
        "lsf_provenance": lsf_provenance,
    }
    flags = []
    if not errors_present:
        flags.append("missing_uncertainties")
    for key in ("wave_medium", "observer_frame", "stellar_rest_status"):
        if str(metadata.get(key, "unknown")).lower() in {"", "none", "unknown"}:
            flags.append("{0}_unknown".format(key))
    if not resolution_meta.get("present"):
        flags.append("resolution_assumption_required")
    if lsf_provenance.get("lsf_source") == "sdss_wdisp_not_applied":
        flags.append("sdss_wdisp_lsf_not_applied")
    if lsf_undersampled:
        flags.append("lsf_undersampled")
    if _fraction(np.sum(artifact_inside), max(n_inside, 1)) > 0.02:
        flags.append("artifact_review_required")
    if flat_block_count > 0:
        flags.append("flat_zero_block_detected")
    if large_gap_count > 0:
        flags.append("large_wavelength_gap_detected")
    if n_fit <= 0:
        flags.append("no_fitted_pixels")

    return {
        "name": str(getattr(seg, "name", "segment")),
        "n_total": n_total,
        "n_inside_fit_window": n_inside,
        "n_fit_candidate": n_fit,
        "outside_fit_window_fraction": _fraction(n_total - n_inside, n_total),
        "rejected_inside_fit_window_fraction": _fraction(rejected_inside, n_inside),
        "mask_true_means": "use",
        "metadata": _json_safe(metadata),
        "artifact_metrics": {
            "nonfinite_fraction": _fraction(np.sum(~finite), n_total),
            "near_zero_fraction": _fraction(np.sum(near_zero & inside), n_inside),
            "negative_flux_fraction": _fraction(
                np.sum(negative_flux & inside),
                n_inside,
            ),
            "extreme_spike_fraction": _fraction(np.sum(spike), n_inside),
            "artifact_fraction_inside_fit_window": _fraction(
                np.sum(artifact_inside),
                n_inside,
            ),
            "flat_zero_block_count": flat_block_count,
            "large_gap_count": large_gap_count,
            "max_gap_A": max_gap_A,
        },
        "sampling_metrics": {
            "pixel_spacing_median_A": spacing_median,
            "pixels_per_fwhm_median": pixels_per_fwhm_median,
            "lsf_undersampled": lsf_undersampled,
        },
        "warnings": list(lsf_provenance.get("warnings") or []),
        "interpretation_flags": sorted(set(flags)),
    }


def audit_spectrum_for_fit(
    spectrum,
    regions=None,
    fit_windows=None,
    intended_use="classification",
    assumed_resolution=None,
    spike_sigma=10.0,
    flat_block_min=5,
    gap_factor=5.0,
):
    """Return a JSON-safe pre-fit readiness audit for a spectrum.

    This helper intentionally does not repair or reject a spectrum. Readers are
    allowed to ingest heterogeneous products, while fitting workflows can call
    this lightweight audit to record whether wavelength-frame metadata,
    uncertainties, instrumental resolution, sampling, and obvious artifacts are
    sufficient for the intended PHOENIX use case.

    ``mask`` follows the Spyctres container convention: ``True`` means a pixel is
    valid/usable. The returned ``interpretation_flags`` are warnings for the
    caller or user interface; they are not automatic exclusions.
    """
    if fit_windows is not None:
        if regions is not None:
            raise ValueError("Pass either regions or fit_windows, not both.")
        regions = fit_windows
    segments = _audit_segments(spectrum)
    per_segment = [
        _segment_readiness(
            seg,
            regions=regions,
            assumed_resolution=assumed_resolution,
            spike_sigma=spike_sigma,
            flat_block_min=flat_block_min,
            gap_factor=gap_factor,
        )
        for seg in segments
    ]
    n_total = int(sum(item["n_total"] for item in per_segment))
    n_inside = int(sum(item["n_inside_fit_window"] for item in per_segment))
    n_fit = int(sum(item["n_fit_candidate"] for item in per_segment))
    rejected_inside = int(n_inside - n_fit)
    flags = sorted(
        {
            flag
            for item in per_segment
            for flag in item.get("interpretation_flags", [])
        }
    )
    warnings = sorted(
        {
            warning
            for item in per_segment
            for warning in item.get("warnings", [])
            if warning
        }
    )
    hard_flags = {
        "no_fitted_pixels",
        "resolution_assumption_required",
        "wave_medium_unknown",
        "observer_frame_unknown",
        "stellar_rest_status_unknown",
        "artifact_review_required",
        "lsf_undersampled",
    }
    fit_ready = bool(n_fit > 0 and not (set(flags) & hard_flags))
    return _json_safe(
        {
            "schema_version": 1,
            "intended_use": str(intended_use),
            "fit_ready": fit_ready,
            "quicklook_only": not fit_ready,
            "mask_true_means": "use",
            "n_segments": len(per_segment),
            "n_total": n_total,
            "n_inside_fit_window": n_inside,
            "n_fit_candidate": n_fit,
            "outside_fit_window_fraction": _fraction(n_total - n_inside, n_total),
            "rejected_inside_fit_window_fraction": _fraction(
                rejected_inside,
                n_inside,
            ),
            "warnings": warnings,
            "interpretation_flags": flags,
            "segments": per_segment,
        }
    )


def artifact_exclusion_mask_from_segment(
    seg,
    *,
    name="artifact:fallback_same_grid",
    include_nonfinite=True,
    include_bad_input_mask=True,
    include_bad_errors=True,
    include_near_zero=True,
    include_negative_flux=False,
    include_spikes=False,
    spike_sigma=10.0,
):
    """Return an opt-in fallback bad-pixel mask for one observed segment.

    This helper is intentionally conservative and same-grid only. It is useful
    for quicklook products that lack native data-quality flags, but it should
    not replace instrument-supplied bitmasks when those are available. Boolean
    ``True`` in the returned callable means reject/exclude.
    """
    wave0 = np.asarray(getattr(seg, "wave"), dtype=float)
    flux = np.asarray(getattr(seg, "flux"), dtype=float)
    if wave0.shape != flux.shape:
        raise ValueError("artifact mask requires matching wave/flux shapes.")
    mask = np.zeros(wave0.shape, dtype=bool)
    finite = np.isfinite(wave0) & (wave0 > 0.0) & np.isfinite(flux)
    if include_nonfinite:
        mask |= ~finite
    if include_bad_input_mask:
        input_mask = np.asarray(
            getattr(seg, "mask", np.ones(wave0.shape, dtype=bool)),
            dtype=bool,
        )
        if input_mask.shape != wave0.shape:
            raise ValueError("artifact mask requires segment mask shape to match wave.")
        mask |= ~input_mask
    err = getattr(seg, "err", None)
    if include_bad_errors and err is not None:
        err = np.asarray(err, dtype=float)
        if err.shape != wave0.shape:
            raise ValueError("artifact mask requires err shape to match wave.")
        mask |= ~(np.isfinite(err) & (err > 0.0))
    if include_near_zero and np.any(finite):
        scale = float(np.nanmedian(np.abs(flux[finite])))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        mask |= finite & (np.abs(flux) <= max(1.0e-30, 1.0e-6 * scale))
    if include_negative_flux:
        mask |= np.isfinite(flux) & (flux < 0.0)
    if include_spikes:
        pool = flux[finite & ~mask]
        if pool.size >= 10:
            median = float(np.nanmedian(pool))
            mad = float(np.nanmedian(np.abs(pool - median)))
            robust_sigma = 1.4826 * mad
            if np.isfinite(robust_sigma) and robust_sigma > 0.0:
                mask |= (
                    finite
                    & (np.abs(flux - median) > float(spike_sigma) * robust_sigma)
                )
    mask.setflags(write=False)
    wave0_copy = wave0.copy()
    wave0_copy.setflags(write=False)

    def _mask(wave):
        wave = np.asarray(wave, dtype=float)
        if wave.shape != wave0_copy.shape or not np.allclose(
            wave,
            wave0_copy,
            rtol=0.0,
            atol=1.0e-8,
            equal_nan=True,
        ):
            raise ValueError(
                "artifact_exclusion_mask_from_segment is same-grid only; "
                "build a new mask for this wavelength grid."
            )
        return np.array(mask, dtype=bool)

    return exclusion_mask(
        name,
        _mask,
        metadata={
            "method": "fallback_artifact_same_grid",
            "mask_type": "artifact",
            "preferred_source": "native_reader_data_quality_mask",
            "same_grid_required": True,
            "include_nonfinite": bool(include_nonfinite),
            "include_bad_input_mask": bool(include_bad_input_mask),
            "include_bad_errors": bool(include_bad_errors),
            "include_near_zero": bool(include_near_zero),
            "include_negative_flux": bool(include_negative_flux),
            "include_spikes": bool(include_spikes),
            "spike_sigma": float(spike_sigma),
            "n_rejected": int(np.sum(mask)),
            "fraction_rejected": _fraction(np.sum(mask), mask.size),
        },
    )
