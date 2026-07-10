"""Shared spectrum masking and preprocessing provenance helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskResult:
    """Result of composing all masks for one spectrum segment."""

    effective_mask: np.ndarray
    excluded_mask: np.ndarray
    rejection_masks: dict
    settings: dict

    @property
    def counts(self):
        """Return JSON-safe pixel counts for each rejection reason."""
        counts = {
            key: int(np.count_nonzero(mask))
            for key, mask in self.rejection_masks.items()
        }
        counts["total"] = int(self.effective_mask.size)
        counts["used"] = int(np.count_nonzero(self.effective_mask))
        counts["rejected"] = counts["total"] - counts["used"]
        counts["explicitly_excluded"] = int(np.count_nonzero(self.excluded_mask))
        return counts

    def to_metadata(self, label="fit_mask"):
        """Return a compact, JSON-safe provenance record."""
        return {
            "operation": "mask",
            "label": str(label),
            "settings": dict(self.settings),
            "counts": self.counts,
        }


def coerce_boolean_mask(values, shape, threshold=0.5, name="mask"):
    """Convert boolean/numeric mask values to a validated boolean array."""
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
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
    return np.asarray(array > threshold, dtype=bool)


def _mask_callable_name(mask_spec, fallback="exclude_mask"):
    """Return a stable, human-readable label for a mask callable spec."""
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
    if isinstance(mask_spec, dict):
        if "callable" in mask_spec:
            mask_spec = mask_spec["callable"]
        elif "func" in mask_spec:
            mask_spec = mask_spec["func"]
        else:
            raise TypeError("Mask spec dictionaries require a 'callable' or 'func' key.")
    elif _is_named_mask_tuple(mask_spec):
        mask_spec = mask_spec[1]
    if not callable(mask_spec):
        raise TypeError("Mask specs must be callables or named callable specs.")
    return mask_spec


def _normalize_mask_specs(exclude_mask=None, exclude_masks=None):
    """Normalize one or more mask callable specs to ``(name, callable)`` pairs."""
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
    used_names = {}
    for spec in specs:
        name = _mask_callable_name(spec)
        fn = _mask_callable_function(spec)
        base = name
        count = used_names.get(base, 0)
        used_names[base] = count + 1
        if count:
            name = "{0}_{1}".format(base, count + 1)
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
    for callable_name, callable_fn in _normalize_mask_specs(
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
    ):
        callable_mask = coerce_boolean_mask(
            callable_fn(wave),
            shape,
            threshold=mask_threshold,
            name="{0} result".format(callable_name),
        )
        callable_masks.append((callable_name, callable_mask))

    if callable_masks:
        exclude_union = np.zeros(shape, dtype=bool)
        for callable_name, callable_mask in callable_masks:
            exclude_union |= callable_mask
            reasons["exclude_mask:{0}".format(callable_name)] = callable_mask
        reasons["exclude_mask"] = exclude_union
        callable_names = [name for name, _mask in callable_masks]
    else:
        reasons["exclude_mask"] = np.zeros(shape, dtype=bool)
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
        "mask_threshold": mask_threshold,
        "mask_true_means": "use",
        "exclude_mask_true_means": "reject",
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
