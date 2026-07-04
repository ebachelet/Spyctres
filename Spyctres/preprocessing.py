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
    mask_threshold=0.5,
):
    """Compose fit masks once and retain a reason-resolved audit trail.

    ``effective_mask`` is True only where a pixel may be fitted. The separate
    reason masks are True where that reason rejects a pixel; reasons may
    overlap. ``excluded_mask`` preserves the historical plotting definition:
    it contains wavelength-rule/callable exclusions, but not invalid data or
    pixels disabled by the segment's existing mask.
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

    if exclude_mask is None:
        reasons["exclude_mask"] = np.zeros(shape, dtype=bool)
        callable_name = None
    else:
        reasons["exclude_mask"] = coerce_boolean_mask(
            exclude_mask(wave),
            shape,
            threshold=mask_threshold,
            name="exclude_mask result",
        )
        callable_name = getattr(exclude_mask, "__name__", type(exclude_mask).__name__)

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
        "exclude_mask": callable_name,
        "mask_threshold": mask_threshold,
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
    mask_threshold=0.5,
    label="fit_mask",
):
    """Return a masked segment copy, its mask result, and recorded provenance."""
    result = compose_fit_mask(
        seg,
        regions=regions,
        exclude_regions=exclude_regions,
        exclude_mask=exclude_mask,
        mask_threshold=mask_threshold,
    )
    meta = dict(seg.meta)
    history = list(meta.get("preprocessing", []))
    history.append(result.to_metadata(label=label))
    meta["preprocessing"] = history
    return seg.copy(mask=result.effective_mask, meta=meta), result
