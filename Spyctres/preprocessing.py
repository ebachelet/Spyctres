"""Shared spectrum masking and preprocessing provenance helpers."""

from dataclasses import dataclass

import numpy as np


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

    def region(self, padding_A=0.0):
        padding_A = float(padding_A)
        if not np.isfinite(padding_A) or padding_A < 0.0:
            raise ValueError("padding_A must be finite and >= 0.")
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
            "description": self.description,
            "reference_ids": list(self.reference_ids),
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
    ),
}


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
    """

    name: str
    callable: object

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("ExclusionMaskSpec.name must be a non-empty string.")
        if not callable(self.callable):
            raise TypeError("ExclusionMaskSpec.callable must be callable.")
        object.__setattr__(self, "name", name)

    def __call__(self, wave):
        return self.callable(wave)


def exclusion_mask(name, fn):
    """Return a named exclusion-mask specification.

    Examples
    --------
    ``exclude_masks=[exclusion_mask("telluric", telluric_fn)]``

    The callable should return boolean-like or numeric values on the supplied
    wavelength grid. Boolean True means reject; numeric values reject where
    ``value > mask_threshold``. Nonfinite numeric outputs are rejected and
    recorded in mask provenance.
    """
    return ExclusionMaskSpec(name=name, callable=fn)


def nonstellar_feature_mask(names=("dib_4428",), padding_A=0.0):
    """Return a named exclusion mask for non-stellar features.

    The returned mask follows the explicit-exclusion convention:
    ``True`` means reject this pixel from the stellar fit.
    """
    feature_names = tuple(names) if not isinstance(names, str) else (names,)
    regions = nonstellar_feature_regions(feature_names, padding_A=padding_A)
    label = "nonstellar:" + "+".join(str(name).strip().lower() for name in feature_names)
    return exclusion_mask(label, lambda wave: _inside_regions(np.asarray(wave, dtype=float), regions))


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
        segment_names = []
        for segment in segments:
            wave = np.asarray(segment.wave, dtype=float)
            mask = np.asarray(getattr(segment, "mask", np.ones(wave.shape, dtype=bool)), dtype=bool)
            good = mask & np.isfinite(wave)
            if not np.any(good):
                continue
            lo = float(np.nanmin(wave[good]))
            hi = float(np.nanmax(wave[good]))
            overlap = max(0.0, min(wmax, hi) - max(wmin, lo))
            if overlap > 0.0:
                total_overlap += overlap
                segment_names.append(getattr(segment, "name", None))
        if total_overlap > 0.0:
            item = dict(meta)
            item["overlap_A"] = float(total_overlap)
            item["segments"] = [name for name in segment_names if name]
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
    """Normalize one or more mask callable specs to ``(name, callable)`` pairs."""
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
        if name in used_names:
            raise ValueError(
                "Duplicate exclusion mask name {0!r}; use unique names so "
                "mask provenance and overlap counts remain unambiguous.".format(name)
            )
        used_names.add(name)
        normalized.append((name, fn))
    return normalized


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
    for callable_name, callable_fn in _normalize_mask_specs(
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
    ):
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
