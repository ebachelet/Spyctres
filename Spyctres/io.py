# Spyctres/io.py
import io as _pyio
import os
import re
import warnings
from dataclasses import dataclass
import numpy as np
from astropy.io import fits


COMMON_SPECTRUM_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResolutionDescriptor:
    """Instrumental-resolution description for a spectrum segment.

    ``quantity`` is one of ``R``, ``fwhm_kms``, or ``sigma_kms``. ``mode`` is
    either ``constant`` or ``tabulated``. A collection represents per-order or
    per-arm resolution by assigning a descriptor to each of its segments;
    wavelength-dependent resolution within one segment uses ``tabulated``.
    """

    quantity: str
    mode: str = "constant"
    value: float = None
    wave_A: object = None
    values: object = None
    source: str = None

    def __post_init__(self):
        quantity = str(self.quantity).strip()
        mode = str(self.mode).strip().lower()
        if quantity not in {"R", "fwhm_kms", "sigma_kms"}:
            raise ValueError("resolution quantity must be R, fwhm_kms, or sigma_kms.")
        if mode not in {"constant", "tabulated"}:
            raise ValueError("resolution mode must be constant or tabulated.")

        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "mode", mode)

        if mode == "constant":
            value = float(self.value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("constant resolution value must be finite and > 0.")
            if self.wave_A is not None or self.values is not None:
                raise ValueError("constant resolution must not define wave_A or values.")
            object.__setattr__(self, "value", value)
            return

        if self.value is not None:
            raise ValueError("tabulated resolution must use values, not value.")
        wave = np.asarray(self.wave_A, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if wave.ndim != 1 or values.ndim != 1 or wave.shape != values.shape:
            raise ValueError(
                "tabulated resolution wave_A and values must be matching 1D arrays."
            )
        if wave.size < 2 or not np.all(np.isfinite(wave)) or np.any(wave <= 0):
            raise ValueError("tabulated resolution requires at least two positive wavelengths.")
        if not np.all(np.diff(wave) > 0):
            raise ValueError("tabulated resolution wavelengths must be unique and increasing.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("tabulated resolution values must be finite and > 0.")
        wave = wave.copy()
        values = values.copy()
        wave.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "wave_A", wave)
        object.__setattr__(self, "values", values)

    def to_metadata(self):
        record = {
            "quantity": self.quantity,
            "mode": self.mode,
            "source": self.source,
        }
        if self.mode == "constant":
            record["value"] = float(self.value)
        else:
            record["wave_A"] = self.wave_A.tolist()
            record["values"] = self.values.tolist()
        return record


def _coerce_resolution_descriptor(value):
    if value is None or isinstance(value, ResolutionDescriptor):
        return value
    if isinstance(value, dict):
        return ResolutionDescriptor(**value)
    if np.isscalar(value):
        return ResolutionDescriptor(quantity="R", value=value, source="scalar_R")
    raise TypeError("resolution must be a ResolutionDescriptor, mapping, scalar R, or None.")


class SpectrumSegment(object):
    """
    Minimal internal spectrum container.

    wave: 1D array (Angstrom by convention unless specified)
    flux: 1D array
    err:  1D array (1-sigma), optional
    mask: boolean array (True = valid/use; False = excluded), optional
    meta: dict, optional
    wave_medium: "air", "vacuum", or "unknown"
    observer_frame: "topocentric", "heliocentric", "barycentric", or "unknown"
    stellar_rest_status: "observed", "corrected", or "unknown"
    stellar_rv_applied_kms: RV removed to reach the stellar rest frame, optional
    resolution: optional ResolutionDescriptor for the instrumental LSF

    ``wave_frame`` remains as a compatibility field. New code should use the
    independent observer-frame and stellar-rest fields because a spectrum can,
    for example, be both barycentric and corrected to the stellar rest frame.
    ``err`` is always a 1-sigma standard deviation in the common format.
    """

    def __init__(
        self,
        wave,
        flux,
        err=None,
        mask=None,
        meta=None,
        wave_medium="unknown",
        wave_frame="unknown",
        name=None,
        observer_frame=None,
        stellar_rest_status=None,
        stellar_rv_applied_kms=None,
        resolution=None,
    ):
        self.wave = np.asarray(wave, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

        if self.wave.ndim != 1 or self.flux.ndim != 1:
            raise ValueError("wave and flux must be 1D arrays.")
        if self.wave.shape[0] != self.flux.shape[0]:
            raise ValueError("wave and flux must have the same length.")

        if err is not None:
            self.err = np.asarray(err, dtype=float)
            if self.err.ndim != 1 or self.err.shape[0] != self.wave.shape[0]:
                raise ValueError("err must be 1D and match wave length.")
        else:
            self.err = None

        if mask is None:
            m = np.isfinite(self.wave) & np.isfinite(self.flux)
            if self.err is not None:
                m &= np.isfinite(self.err) & (self.err > 0)
            self.mask = m
        else:
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.ndim != 1 or self.mask.shape[0] != self.wave.shape[0]:
                raise ValueError("mask must be 1D and match wave length.")

        self.meta = {} if meta is None else dict(meta)
        self.wave_medium = str(wave_medium).strip().lower()
        self.wave_frame = str(wave_frame).strip().lower()
        if observer_frame is None:
            if self.wave_frame in {"topocentric", "heliocentric", "barycentric"}:
                observer_frame = self.wave_frame
            else:
                observer_frame = "unknown"
        if stellar_rest_status is None:
            if self.wave_frame == "stellar_rest":
                stellar_rest_status = "corrected"
            elif self.wave_frame in {"topocentric", "heliocentric", "barycentric"}:
                stellar_rest_status = "observed"
            else:
                stellar_rest_status = "unknown"
        self.observer_frame = str(observer_frame).strip().lower()
        self.stellar_rest_status = str(stellar_rest_status).strip().lower()
        if stellar_rv_applied_kms is None:
            self.stellar_rv_applied_kms = None
        else:
            value = float(stellar_rv_applied_kms)
            if not np.isfinite(value):
                raise ValueError("stellar_rv_applied_kms must be finite.")
            self.stellar_rv_applied_kms = value
        self.resolution = _coerce_resolution_descriptor(resolution)
        self.name = name

    def copy(
        self,
        wave=None,
        flux=None,
        err=None,
        mask=None,
        meta=None,
        wave_medium=None,
        wave_frame=None,
        name=None,
        observer_frame=None,
        stellar_rest_status=None,
        stellar_rv_applied_kms=None,
        resolution=None,
    ):
        target_wave_frame = self.wave_frame if wave_frame is None else wave_frame
        if observer_frame is None:
            if wave_frame is None:
                observer_frame = self.observer_frame
            elif str(wave_frame).strip().lower() in {
                "topocentric",
                "heliocentric",
                "barycentric",
            }:
                observer_frame = str(wave_frame).strip().lower()
            elif str(wave_frame).strip().lower() == "stellar_rest":
                observer_frame = self.observer_frame
            else:
                observer_frame = "unknown"
        if stellar_rest_status is None:
            if wave_frame is None:
                stellar_rest_status = self.stellar_rest_status
            elif str(wave_frame).strip().lower() == "stellar_rest":
                stellar_rest_status = "corrected"
            elif str(wave_frame).strip().lower() in {
                "topocentric",
                "heliocentric",
                "barycentric",
            }:
                stellar_rest_status = "observed"
            else:
                stellar_rest_status = "unknown"

        return SpectrumSegment(
            self.wave if wave is None else wave,
            self.flux if flux is None else flux,
            self.err if err is None else err,
            self.mask if mask is None else mask,
            meta=self.meta if meta is None else meta,
            wave_medium=self.wave_medium if wave_medium is None else wave_medium,
            wave_frame=target_wave_frame,
            name=self.name if name is None else name,
            observer_frame=observer_frame,
            stellar_rest_status=stellar_rest_status,
            stellar_rv_applied_kms=(
                self.stellar_rv_applied_kms
                if stellar_rv_applied_kms is None
                else stellar_rv_applied_kms
            ),
            resolution=self.resolution if resolution is None else resolution,
        )

    def sorted(self):
        idx = np.argsort(self.wave)
        return self.copy(
            wave=self.wave[idx],
            flux=self.flux[idx],
            err=None if self.err is None else self.err[idx],
            mask=self.mask[idx],
        )

    def subset(self, selector, name=None, name_suffix=None):
        """
        Return a subsetted copy of the segment.

        selector may be:
        - a boolean mask with the same length as the segment, or
        - an integer/slice indexer understood by NumPy.
        """
        if isinstance(selector, slice):
            idx = selector
        else:
            selector = np.asarray(selector)
            if selector.dtype == bool:
                if selector.ndim != 1 or selector.shape[0] != self.wave.shape[0]:
                    raise ValueError("Boolean selector must be 1D and match wave length.")
                idx = selector
            else:
                idx = selector

        out_name = self.name if name is None else name
        if name_suffix is not None:
            base = "" if out_name is None else str(out_name)
            out_name = "{0}_{1}".format(base, name_suffix) if base else str(name_suffix)

        return self.copy(
            wave=self.wave[idx],
            flux=self.flux[idx],
            err=None if self.err is None else self.err[idx],
            mask=self.mask[idx],
            meta=dict(self.meta),
            name=out_name,
        )

    def window(self, wmin=None, wmax=None, clip_left=0, clip_right=0, name=None, name_suffix=None):
        """
        Return a wavelength-windowed copy of the segment.

        The wavelength cut is inclusive in [wmin, wmax]. After that, optional
        pixel clipping is applied on the left/right edges of the retained block.
        """
        keep = np.ones_like(self.wave, dtype=bool)
        if wmin is not None:
            keep &= (self.wave >= float(wmin))
        if wmax is not None:
            keep &= (self.wave <= float(wmax))

        idx = np.where(keep)[0]
        if idx.size == 0:
            raise ValueError("No points remain after wavelength windowing.")

        i0 = idx[0]
        i1 = idx[-1] + 1

        out = self.subset(slice(i0, i1), name=name, name_suffix=name_suffix)

        if clip_left > 0 or clip_right > 0:
            n = len(out.wave)
            j0 = int(max(0, clip_left))
            j1 = int(n - max(0, clip_right))
            if j1 <= j0:
                raise ValueError("Edge clipping removed all points.")
            out = out.subset(slice(j0, j1), name=out.name)

        return out

    def with_wave(self, wave, wave_medium=None, wave_frame=None, name=None, name_suffix=None):
        """
        Return a copy with a replaced wavelength array and optionally updated
        wavelength metadata. Flux, err, and mask are preserved.
        """
        out_name = self.name if name is None else name
        if name_suffix is not None:
            base = "" if out_name is None else str(out_name)
            out_name = "{0}_{1}".format(base, name_suffix) if base else str(name_suffix)

        return self.copy(
            wave=np.asarray(wave, dtype=float),
            meta=dict(self.meta),
            wave_medium=self.wave_medium if wave_medium is None else wave_medium,
            wave_frame=self.wave_frame if wave_frame is None else wave_frame,
            name=out_name,
        )


class SpectrumCollection(object):
    """
    Thin container for a joint fit over multiple SpectrumSegment objects.

    Parameters
    ----------
    segments : SpectrumSegment or sequence of SpectrumSegment
        Segment objects to be grouped together.
    weights : array-like, optional
        Positive per-segment weights used by joint fitting. If None, all
        segments receive unit weight.
    meta : dict, optional
        Optional collection-level metadata.
    name : str, optional
        Optional collection name.
    """

    def __init__(self, segments, weights=None, meta=None, name=None):
        if isinstance(segments, SpectrumSegment):
            segments = [segments]
        else:
            segments = list(segments)

        if len(segments) == 0:
            raise ValueError("SpectrumCollection requires at least one SpectrumSegment.")

        for i, seg in enumerate(segments):
            if not isinstance(seg, SpectrumSegment):
                raise TypeError(
                    "All entries in SpectrumCollection must be SpectrumSegment objects; "
                    "got type {0} at index {1}.".format(type(seg).__name__, i)
                )

        self.segments = list(segments)

        if weights is None:
            self.weights = np.ones(len(self.segments), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.ndim != 1 or len(w) != len(self.segments):
                raise ValueError("weights must be 1D and match the number of segments.")
            if not np.all(np.isfinite(w)):
                raise ValueError("weights must be finite.")
            if np.any(w <= 0):
                raise ValueError("weights must be > 0.")
            self.weights = np.array(w, copy=True, dtype=float)

        self.meta = {} if meta is None else dict(meta)
        self.name = name

    def __len__(self):
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, item):
        return self.segments[item]

    def copy(self, segments=None, weights=None, meta=None, name=None):
        return SpectrumCollection(
            list(self.segments) if segments is None else segments,
            weights=np.array(self.weights, copy=True) if weights is None else weights,
            meta=self.meta if meta is None else meta,
            name=self.name if name is None else name,
        )

    @property
    def names(self):
        return [seg.name for seg in self.segments]


_VALID_WAVE_MEDIA = {"air", "vacuum", "unknown"}
_VALID_OBSERVER_FRAMES = {"topocentric", "heliocentric", "barycentric", "unknown"}
_VALID_STELLAR_REST_STATUS = {"observed", "corrected", "unknown"}


def _normalize_spectral_label(value, valid, field):
    label = "unknown" if value is None else str(value).strip().lower()
    if label not in valid:
        raise ValueError(
            "Unsupported {0} '{1}'. Valid values are: {2}.".format(
                field,
                value,
                ", ".join(sorted(valid)),
            )
        )
    return label


def canonicalize_segment(
    segment,
    wave_unit="angstrom",
    uncertainty_kind="sigma",
    sort=True,
    duplicate_policy="raise",
    warn_unknown=True,
    source="user",
):
    """Return a validated ``SpectrumSegment`` in Spyctres' common format.

    The common representation uses wavelength in Angstrom, 1D float arrays,
    optional 1-sigma errors, a boolean use-mask (True means use), an explicit
    resolution descriptor, and independent wavelength-medium, observer-frame,
    and stellar-rest labels. Flux values and units are preserved: ingestion
    never normalizes, merges orders, or resamples a spectrum.
    """
    if not isinstance(segment, SpectrumSegment):
        raise TypeError("canonicalize_segment requires a SpectrumSegment.")

    wave = _wave_to_angstrom(segment.wave, wave_unit)
    flux = np.asarray(segment.flux, dtype=float).copy()
    err = None if segment.err is None else np.asarray(segment.err, dtype=float).copy()
    mask = np.asarray(segment.mask, dtype=bool).copy()

    if wave.ndim != 1 or flux.ndim != 1 or wave.shape != flux.shape:
        raise ValueError("wave and flux must be matching 1D arrays.")
    if err is not None and (err.ndim != 1 or err.shape != wave.shape):
        raise ValueError("err must be 1D and match wave length.")
    if mask.ndim != 1 or mask.shape != wave.shape:
        raise ValueError("mask must be 1D and match wave length.")
    if wave.size == 0:
        raise ValueError("A spectrum must contain at least one pixel.")

    uncertainty_kind = str(uncertainty_kind).strip().lower()
    if uncertainty_kind not in {"sigma", "variance", "inverse_variance"}:
        raise ValueError(
            "uncertainty_kind must be sigma, variance, or inverse_variance."
        )
    if err is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if uncertainty_kind == "variance":
                err = np.sqrt(err)
            elif uncertainty_kind == "inverse_variance":
                err = 1.0 / np.sqrt(err)

    duplicate_policy = str(duplicate_policy).strip().lower()
    if duplicate_policy != "raise":
        raise ValueError(
            "duplicate_policy currently supports only 'raise'; use separate "
            "SpectrumCollection segments for overlapping orders."
        )

    medium = _normalize_spectral_label(
        segment.wave_medium,
        _VALID_WAVE_MEDIA,
        "wave_medium",
    )
    observer_frame = _normalize_spectral_label(
        segment.observer_frame,
        _VALID_OBSERVER_FRAMES,
        "observer_frame",
    )
    stellar_rest_status = _normalize_spectral_label(
        segment.stellar_rest_status,
        _VALID_STELLAR_REST_STATUS,
        "stellar_rest_status",
    )
    if warn_unknown and medium == "unknown":
        warnings.warn(
            "Spectrum wavelength medium is unknown; set wave_medium before "
            "medium-sensitive modeling.",
            UserWarning,
            stacklevel=2,
        )
    if warn_unknown and observer_frame == "unknown":
        warnings.warn(
            "Spectrum observer frame is unknown; set observer_frame before "
            "applying observer-to-barycentre corrections.",
            UserWarning,
            stacklevel=2,
        )
    if warn_unknown and stellar_rest_status == "unknown":
        warnings.warn(
            "Spectrum stellar-rest status is unknown; state whether stellar RV "
            "has already been removed before interpreting fitted rv_kms.",
            UserWarning,
            stacklevel=2,
        )

    finite_positive_wave = np.isfinite(wave) & (wave > 0)
    valid = finite_positive_wave & np.isfinite(flux)
    if err is not None:
        valid &= np.isfinite(err) & (err > 0)
    mask &= valid

    finite_wave = np.sort(wave[finite_positive_wave])
    if finite_wave.size > 1 and np.any(np.diff(finite_wave) == 0):
        raise ValueError(
            "Spectrum wavelengths must be unique within a segment; represent "
            "overlapping echelle orders as separate SpectrumCollection segments."
        )

    was_sorted = bool(np.all(np.diff(wave[finite_positive_wave]) > 0))
    if sort:
        order = np.argsort(wave, kind="stable")
        wave = wave[order]
        flux = flux[order]
        mask = mask[order]
        if err is not None:
            err = err[order]

    meta = dict(segment.meta)
    history = list(meta.get("ingestion", []))
    history.append(
        {
            "operation": "canonicalize_spectrum",
            "schema_version": COMMON_SPECTRUM_SCHEMA_VERSION,
            "source": str(source),
            "wave_unit_input": str(wave_unit),
            "wave_unit_output": "angstrom",
            "uncertainty_kind_input": uncertainty_kind,
            "uncertainty_kind_output": "sigma",
            "mask_true_means": "use",
            "sorted": bool(sort and not was_sorted),
            "duplicate_policy": duplicate_policy,
            "n_pixels": int(wave.size),
            "n_valid": int(np.count_nonzero(valid)),
            "n_masked": int(wave.size - np.count_nonzero(mask)),
        }
    )
    meta["ingestion"] = history
    meta["wave_unit"] = "angstrom"
    meta["wave_medium"] = medium
    meta["observer_frame"] = observer_frame
    meta["stellar_rest_status"] = stellar_rest_status
    meta["stellar_rv_applied_kms"] = segment.stellar_rv_applied_kms
    meta["mask_true_means"] = "use"
    meta["spectrum_schema_version"] = COMMON_SPECTRUM_SCHEMA_VERSION
    if err is not None:
        meta["error_kind"] = "sigma"

    resolution = segment.resolution
    if resolution is None:
        resolution_R = meta.get("resolution_R", None)
        if resolution_R is not None:
            resolution = ResolutionDescriptor(
                quantity="R",
                value=resolution_R,
                source="legacy metadata resolution_R",
            )
    meta["resolution"] = None if resolution is None else resolution.to_metadata()

    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium=medium,
        wave_frame=segment.wave_frame,
        name=segment.name,
        observer_frame=observer_frame,
        stellar_rest_status=stellar_rest_status,
        stellar_rv_applied_kms=segment.stellar_rv_applied_kms,
        resolution=resolution,
    )


def spectrum_from_arrays(
    wave,
    flux,
    err=None,
    mask=None,
    meta=None,
    wave_unit="angstrom",
    uncertainty_kind="sigma",
    wave_medium="unknown",
    wave_frame="unknown",
    observer_frame=None,
    stellar_rest_status=None,
    stellar_rv_applied_kms=None,
    resolution=None,
    name=None,
    **canonicalize_kwargs
):
    """Build and canonicalize a spectrum supplied as arrays."""
    segment = SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium=wave_medium,
        wave_frame=wave_frame,
        name=name,
        observer_frame=observer_frame,
        stellar_rest_status=stellar_rest_status,
        stellar_rv_applied_kms=stellar_rv_applied_kms,
        resolution=resolution,
    )
    return canonicalize_segment(
        segment,
        wave_unit=wave_unit,
        uncertainty_kind=uncertainty_kind,
        source="arrays",
        **canonicalize_kwargs
    )


def coerce_spectrum(data, **kwargs):
    """Coerce supported user input into the common spectrum container.

    Accepted inputs are ``SpectrumSegment``, ``SpectrumCollection``, mappings
    with ``wave``/``flux`` keys, or tuple-like ``(wave, flux[, err[, mask]])``.
    Collections are preserved rather than concatenated or resampled.
    """
    if isinstance(data, SpectrumSegment):
        return canonicalize_segment(data, **kwargs)

    if isinstance(data, SpectrumCollection):
        segments = [canonicalize_segment(seg, **kwargs) for seg in data]
        meta = dict(data.meta)
        meta["common_format"] = "SpectrumCollection"
        meta["spectrum_schema_version"] = COMMON_SPECTRUM_SCHEMA_VERSION
        return data.copy(segments=segments, meta=meta)

    if isinstance(data, dict):
        if "wave" not in data or "flux" not in data:
            raise ValueError("Spectrum mappings require 'wave' and 'flux' keys.")
        fields = dict(data)
        fields.update(kwargs)
        return spectrum_from_arrays(**fields)

    if isinstance(data, (tuple, list)) and 2 <= len(data) <= 4:
        fields = {"wave": data[0], "flux": data[1]}
        if len(data) >= 3:
            fields["err"] = data[2]
        if len(data) == 4:
            fields["mask"] = data[3]
        fields.update(kwargs)
        return spectrum_from_arrays(**fields)

    raise TypeError(
        "Spectrum input must be a SpectrumSegment, SpectrumCollection, mapping, "
        "or (wave, flux[, err[, mask]]) tuple."
    )
        
        
def concatenate_segments(segments, sort=True, name=None):
    """
    Concatenate multiple SpectrumSegment objects into a single SpectrumSegment.

    If all input segments share the same wavelength medium and frame, those are
    preserved. Otherwise they are set to "unknown" on the merged segment.
    """
    segments = list(segments)
    if len(segments) == 0:
        raise ValueError("concatenate_segments requires at least one segment.")

    wave = np.concatenate([s.wave for s in segments])
    flux = np.concatenate([s.flux for s in segments])

    if any(s.err is None for s in segments):
        err = None
    else:
        err = np.concatenate([s.err for s in segments])

    mask = np.concatenate([s.mask for s in segments])

    media = {str(s.wave_medium).lower() for s in segments}
    frames = {str(s.wave_frame).lower() for s in segments}
    wave_medium = next(iter(media)) if len(media) == 1 else "unknown"
    wave_frame = next(iter(frames)) if len(frames) == 1 else "unknown"

    meta = {
        "n_segments": len(segments),
        "segment_names": [s.name for s in segments],
        "wave_medium": wave_medium,
        "wave_frame": wave_frame,
    }
    out = SpectrumSegment(
        wave,
        flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium=wave_medium,
        wave_frame=wave_frame,
        name=name,
    )
    return out.sorted() if sort else out


def make_window_segments(seg, windows, pad=0.0, name_prefix=None):
    """
    Build one SpectrumSegment per wavelength window.

    Parameters
    ----------
    seg : SpectrumSegment
        Input segment.
    windows : sequence
        Each entry may be either (wmin, wmax) or (label, wmin, wmax).
    pad : float, optional
        Extra wavelength padding added on both sides of each window.
    name_prefix : str, optional
        Prefix used when a window does not provide its own label.

    Returns
    -------
    list of SpectrumSegment
    """
    out = []
    for i, item in enumerate(windows):
        if len(item) == 2:
            wmin, wmax = item
            label = None
        elif len(item) == 3:
            label, wmin, wmax = item
        else:
            raise ValueError("Each window must be (wmin, wmax) or (label, wmin, wmax).")

        wmin_pad = float(wmin) - float(pad)
        wmax_pad = float(wmax) + float(pad)

        if label is not None:
            name = str(label)
        elif name_prefix is not None:
            name = "{0}_{1}".format(name_prefix, i + 1)
        else:
            name = seg.name

        out.append(seg.window(wmin=wmin_pad, wmax=wmax_pad, name=name))

    return out


def make_padded_window_segments(seg, windows, pad=5.0, name_prefix=None):
    """
    Build one SpectrumSegment per wavelength window with padded support.

    The returned segment covers the support region [wmin-pad, wmax+pad], but
    seg.mask is True only on the inner fit window [wmin, wmax]. This lets the
    fitter evaluate models on a padded support region while computing chi-square
    only on the inner fit pixels.

    Parameters
    ----------
    seg : SpectrumSegment
        Input segment.
    windows : sequence
        Each entry may be either (wmin, wmax) or (label, wmin, wmax).
    pad : float, optional
        Extra wavelength padding added on both sides of each fit window.
    name_prefix : str, optional
        Prefix used when a window does not provide its own label.

    Returns
    -------
    list of SpectrumSegment
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = None if seg.err is None else np.asarray(seg.err, dtype=float)
    base_mask = np.asarray(seg.mask, dtype=bool)

    out = []
    for i, item in enumerate(windows):
        if len(item) == 2:
            wmin, wmax = item
            label = None
        elif len(item) == 3:
            label, wmin, wmax = item
        else:
            raise ValueError("Each window must be (wmin, wmax) or (label, wmin, wmax).")

        support_lo = float(wmin) - float(pad)
        support_hi = float(wmax) + float(pad)

        keep = (wave >= support_lo) & (wave <= support_hi)
        if not np.any(keep):
            continue

        fit_mask = base_mask[keep] & (wave[keep] >= float(wmin)) & (wave[keep] <= float(wmax))

        if label is not None:
            name = str(label)
        elif name_prefix is not None:
            base = "" if seg.name is None else str(seg.name)
            name = "{0}_{1}_win{2}".format(base, name_prefix, i) if base else "{0}_win{1}".format(name_prefix, i)
        else:
            base = "" if seg.name is None else str(seg.name)
            name = "{0}_win{1}".format(base, i) if base else "win{0}".format(i)

        out.append(
            SpectrumSegment(
                wave=wave[keep],
                flux=flux[keep],
                err=None if err is None else err[keep],
                mask=fit_mask,
                meta=dict(seg.meta),
                wave_medium=seg.wave_medium,
                wave_frame=seg.wave_frame,
                name=name,
            )
        )

    if len(out) == 0:
        raise ValueError("No points remain after applying padded windows.")

    return out



def _pepsi_fiber_to_resolution(fiber):
    """
    PEPSI nominal resolving power by science fiber diameter.
    100 um -> 250000
    200 um -> 130000
    300 um -> 50000
    """
    if fiber is None:
        return None

    digits = "".join(ch for ch in str(fiber) if ch.isdigit())
    mapping = {
        "100": 250000.0,
        "200": 130000.0,
        "300": 50000.0,
    }
    return mapping.get(digits, None)


def _normalize_hdu_token(value):
    """
    Normalize FITS extension-like names for robust matching.

    Examples
    --------
    'ORD13_ERRS' -> 'ERRS'
    'ORD16_QUAL' -> 'QUAL'
    'FLUX'       -> 'FLUX'
    """
    if value is None:
        return None

    s = str(value).strip().upper()
    if not s:
        return None

    parts = s.split("_")

    # Common X-SHOOTER merged-1D pattern: ORDxx_ERRS / ORDxx_QUAL
    if len(parts) >= 2 and parts[-1] in ["FLUX", "ERRS", "QUAL"]:
        return parts[-1]

    return s


def _find_hdu_by_name(hdul, extname):
    """
    Return HDU matching EXTNAME, or None if not found.

    Matching is tolerant of logical product names such as ORD13_ERRS
    versus actual EXTNAME='ERRS'.
    """
    target = _normalize_hdu_token(extname)
    if target is None:
        return None

    for hdu in hdul:
        name = _normalize_hdu_token(hdu.header.get("EXTNAME", None))
        if name == target:
            return hdu

    return None


def _resolve_hdu(hdul, ref):
    """
    Resolve an HDU either by integer index or by extension-name-like string.

    For string references, first try EXTNAME matching, then also allow
    matches against SCIDATA / ERRDATA / QUALDATA header pointers after
    normalization.
    """
    if ref is None:
        return None

    if isinstance(ref, (int, np.integer)):
        return hdul[int(ref)]

    target = _normalize_hdu_token(ref)

    hdu = _find_hdu_by_name(hdul, target)
    if hdu is not None:
        return hdu

    for hdu in hdul:
        hdr = hdu.header
        aliases = [
            hdr.get("EXTNAME"),
            hdr.get("SCIDATA"),
            hdr.get("ERRDATA"),
            hdr.get("QUALDATA"),
        ]
        aliases = [_normalize_hdu_token(x) for x in aliases]
        if target in aliases:
            return hdu

    raise ValueError("Could not find FITS extension '{0}'.".format(ref))


def _build_linear_wave_from_header(hdr, n_pix):
    """
    Build a linear wavelength array from CRVAL1/CDELT1/CRPIX1.
    """
    if "CRVAL1" not in hdr or "CDELT1" not in hdr:
        raise ValueError("Header is missing CRVAL1/CDELT1 needed for wavelength solution.")

    crval1 = float(hdr["CRVAL1"])
    cdelt1 = float(hdr["CDELT1"])
    crpix1 = float(hdr.get("CRPIX1", 1.0))

    i = np.arange(int(n_pix), dtype=float)
    return crval1 + (i + 1.0 - crpix1) * cdelt1


def _wave_to_angstrom(wave, unit):
    """
    Convert wavelength array to Angstrom.
    """
    wave = np.asarray(wave, dtype=float)
    u = "" if unit is None else str(unit).strip().lower()

    if u in ["a", "aa", "angstrom", "angstroms", "ang"]:
        return wave

    if u in ["nm", "nanometer", "nanometers"]:
        return wave * 10.0

    if u in ["um", "micron", "microns", "micrometer", "micrometers"]:
        return wave * 1.0e4

    raise ValueError("Unsupported wavelength unit '{0}'.".format(unit))


def _xshooter_slit_keyword_for_arm(arm):
    """
    Map X-SHOOTER arm to the slit keyword used in these headers.
    """
    arm = str(arm).strip().upper()
    mapping = {
        "UVB": "HIERARCH ESO INS OPTI3 NAME",
        "VIS": "HIERARCH ESO INS OPTI4 NAME",
        "NIR": "HIERARCH ESO INS OPTI5 NAME",
    }
    return mapping.get(arm, None)


def _xshooter_parse_slit_width(slit_name):
    """
    Parse slit width in arcsec from strings like '1.0x11', '0.7x11', '0.6x11'.
    Returns float or the string 'IFU', or None.
    """
    if slit_name is None:
        return None

    s = str(slit_name).strip().upper()

    if not s:
        return None

    if "IFU" in s:
        return "IFU"

    # Take the first numeric token, which is the slit width in strings like 1.0x11
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    if m is not None:
        return float(m.group(1))

    return None


def _xshooter_resolution_from_slit(arm, slit_name):
    """
    Nominal X-SHOOTER resolving power as a function of arm and slit width.
    Values follow the X-SHOOTER user manual.
    """
    arm = str(arm).strip().upper()
    slit = _xshooter_parse_slit_width(slit_name)

    table = {
        "UVB": {0.5: 9700.0, 0.8: 6700.0, 1.0: 5400.0, 1.3: 4100.0, 1.6: 3300.0, "IFU": 7900.0},
        "VIS": {0.4: 17400.0, 0.7: 11000.0, 0.9: 8800.0, 1.2: 6700.0, 1.5: 5400.0, "IFU": 12600.0},
        "NIR": {0.4: 11300.0, 0.6: 8100.0, 0.9: 5600.0, 1.2: 4300.0, "IFU": 8100.0},
    }

    if slit is None or arm not in table:
        return None

    if slit == "IFU":
        return table[arm].get("IFU")

    widths = [k for k in table[arm].keys() if k != "IFU"]
    if len(widths) == 0:
        return None

    # Use nearest supported slit width to be robust against header formatting quirks
    nearest = min(widths, key=lambda w: abs(float(w) - float(slit)))

    if abs(float(nearest) - float(slit)) > 0.051:
        return None

    return table[arm][nearest]
    

def _xshooter_telluric_corrected(hdr):
    """
    Heuristic flag for telluric-corrected X-SHOOTER products.
    """
    prodcatg = str(hdr.get("HIERARCH ESO PRO CATG", "")).upper()
    pipefile = str(hdr.get("PIPEFILE", "")).upper()

    return ("TELLURIC" in prodcatg) or ("TELLURIC" in pipefile)


def read_pepsi_nor(
    path,
    ext=1,
    wave_col="Arg",
    flux_col="Fun",
    var_col="Var",
    wave_medium="unknown",
    wave_frame="unknown",
    infer_resolution=True,
    product_profile="generic",
):
    """
    Read PEPSI .nor files (FITS binary table).

    Expected columns: Arg, Fun, Var (variance).

    ``product_profile`` must be selected from release documentation, never from
    the ``.dxt.nor`` suffix. ``generic`` preserves the historical assumption
    that Arg is numerically in Angstrom and leaves its medium/frame unknown.
    ``pets_stellar_rest`` implements the documented
    NASA Exoplanet Archive PETS convention (Arg in microns, air, already in the
    stellar centre-of-mass frame). ``cds_aanda_671_a7`` implements that CDS
    release's documented Angstrom, Solar-System-barycentric scale while leaving
    the unreported air/vacuum medium unknown.

    Product conventions:
    https://exoplanetarchive.ipac.caltech.edu/docs/PEPSIMission.html
    https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/671/A7?format=html&tex=true
    """
    profiles = {
        "generic": {
            "wave_scale": 1.0,
            "wave_unit_input": "angstrom_assumed",
            "wave_medium": "unknown",
            "wave_frame": "unknown",
            "observer_frame": "unknown",
            "stellar_rest_status": "unknown",
            "reference": None,
        },
        "pets_stellar_rest": {
            "wave_scale": 1.0e4,
            "wave_unit_input": "micron",
            "wave_medium": "air",
            "wave_frame": "stellar_rest",
            "observer_frame": "barycentric",
            "stellar_rest_status": "corrected",
            "reference": "https://exoplanetarchive.ipac.caltech.edu/docs/PEPSIMission.html",
        },
        "cds_aanda_671_a7": {
            "wave_scale": 1.0,
            "wave_unit_input": "angstrom",
            "wave_medium": "unknown",
            "wave_frame": "barycentric",
            "observer_frame": "barycentric",
            "stellar_rest_status": "observed",
            "reference": "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/671/A7?format=html&tex=true",
        },
    }
    product_profile = str(product_profile).strip().lower()
    if product_profile not in profiles:
        raise ValueError(
            "Unknown PEPSI product_profile {!r}; choose generic, "
            "pets_stellar_rest, or cds_aanda_671_a7.".format(product_profile)
        )
    profile = profiles[product_profile]
    if product_profile != "generic":
        if wave_medium != "unknown" and wave_medium != profile["wave_medium"]:
            raise ValueError("wave_medium conflicts with the selected PEPSI product profile.")
        if wave_frame != "unknown" and wave_frame != profile["wave_frame"]:
            raise ValueError("wave_frame conflicts with the selected PEPSI product profile.")
        wave_medium = profile["wave_medium"]
        wave_frame = profile["wave_frame"]

    path = os.path.abspath(os.path.expanduser(path))
    with fits.open(path, memmap=False) as hdul:
        data = hdul[ext].data
        cols = set(c.name for c in hdul[ext].columns)

        if wave_col not in cols or flux_col not in cols:
            raise ValueError(
                "Missing required columns in {0}: found {1}".format(path, sorted(cols))
            )

        wave = np.array(data[wave_col], dtype=float) * profile["wave_scale"]
        flux = np.array(data[flux_col], dtype=float)

        err = None
        if var_col in cols:
            var = np.array(data[var_col], dtype=float)
            err = np.sqrt(var)

        phdr = hdul[0].header if len(hdul) > 0 else fits.Header()
        ehdr = hdul[ext].header

        def get_any(key, default=None):
            if key in ehdr:
                return ehdr[key]
            if key in phdr:
                return phdr[key]
            return default

        def header_record(key):
            header = ehdr if key in ehdr else phdr if key in phdr else None
            if header is None:
                return None
            return {"value": header[key], "comment": header.comments[key]}

        corrections_applied = (
            True if product_profile == "pets_stellar_rest" else None
        )
        velocity_corrections = {
            "RADVEL": header_record("RADVEL"),
            "OBSVEL": header_record("OBSVEL"),
            "SSBVEL": header_record("SSBVEL"),
            "pets_radvel_applied": corrections_applied,
            "pets_obsvel_applied": corrections_applied,
        }

        meta = {
            "path": path,
            "ext": ext,
            "columns": sorted(cols),
            "instrument": get_any("INSTRUME"),
            "object": get_any("OBJECT"),
            "arm": get_any("ARM"),
            "fiber": get_any("FIBER"),
            "cross_disperser": get_any("CROSDIS"),
            "date_obs": get_any("DATE-OBS"),
            "time_obs": get_any("TIME-OBS"),
            "exptime": get_any("EXPTIME"),
            "jd_obs": get_any("JD-OBS"),
            "jd_tdb": get_any("JD-TDB"),
            "ra": get_any("RA"),
            "dec": get_any("DEC"),
            "ra2000": get_any("RA2000"),
            "dec2000": get_any("DE2000"),
            "ssbvel_mps": get_any("SSBVEL"),
            "charave_mps": get_any("CHARAVE"),
            "was_file": get_any("WAS"),
            "trace_file": get_any("TRACE"),
            "wasrep": get_any("WASREP"),
            "wasfwhm_pix": get_any("WASFWHM"),
            "wave_medium": wave_medium,
            "wave_frame": wave_frame,
            "pepsi_product_profile": product_profile,
            "pepsi_profile_reference": profile["reference"],
            "pepsi_arg_unit_input": profile["wave_unit_input"],
            "velocity_corrections": velocity_corrections,
        }

        resolution_R = None
        if infer_resolution:
            wasrep = meta.get("wasrep")
            if wasrep is not None:
                try:
                    resolution_R = float(wasrep)
                except Exception:
                    resolution_R = None

            if resolution_R is None:
                resolution_R = _pepsi_fiber_to_resolution(meta.get("fiber"))

        meta["resolution_R"] = resolution_R

        name = os.path.basename(path)
        return SpectrumSegment(
            wave,
            flux,
            err=err,
            meta=meta,
            wave_medium=wave_medium,
            wave_frame=wave_frame,
            name=name,
            observer_frame=(
                profile["observer_frame"]
                if product_profile != "generic"
                else (
                    wave_frame
                    if wave_frame in {"topocentric", "heliocentric", "barycentric"}
                    else "unknown"
                )
            ),
            stellar_rest_status=(
                profile["stellar_rest_status"]
                if product_profile != "generic"
                else (
                    "corrected"
                    if wave_frame == "stellar_rest"
                    else (
                        "observed"
                        if wave_frame in {"topocentric", "heliocentric", "barycentric"}
                        else "unknown"
                    )
                )
            ),
            resolution=(
                None
                if resolution_R is None
                else ResolutionDescriptor(
                    quantity="R",
                    value=resolution_R,
                    source="PEPSI WASREP/fiber metadata",
                )
            ),
        ).sorted()


def pepsi_ssbvel_correction_kms(segment):
    """Return SSBVEL in km/s only when it has not already been applied.

    This deliberately refuses barycentric or stellar-rest products: applying
    SSBVEL to either would double-correct the wavelength scale. For a generic
    product, use remains an explicit caller choice because ``.dxt.nor`` does
    not establish a reference frame.
    """
    if not isinstance(segment, SpectrumSegment):
        raise TypeError("segment must be a SpectrumSegment.")
    if (
        segment.observer_frame == "barycentric"
        or segment.stellar_rest_status == "corrected"
    ):
        raise ValueError(
            "SSBVEL must not be applied: this spectrum is already barycentric "
            "or corrected to the stellar rest frame."
        )
    value = segment.meta.get("ssbvel_mps")
    if value is None:
        return 0.0
    return 1.0e-3 * float(value)


def read_xshooter_1d(
    path,
    flux_ext=0,
    err_ext=None,
    qual_ext=None,
    wave_unit=None,
    wave_medium="air",
    wave_frame="topocentric",
    infer_resolution=True,
):
    """
    Read a merged 1D X-SHOOTER spectrum stored as a linear FITS image product.

    Assumptions for the current implementation:
    - flux is in the primary HDU by default
    - wavelength is reconstructed from CRVAL1/CDELT1/CRPIX1
    - error and quality HDUs are resolved from ERRDATA / QUALDATA when present
    - no air/vacuum or barycentric correction is applied here; those are carried as metadata
    """
    path = os.path.abspath(os.path.expanduser(path))

    with fits.open(path, memmap=False) as hdul:
        flux_hdu = _resolve_hdu(hdul, flux_ext)
        phdr = flux_hdu.header
        flux = np.asarray(flux_hdu.data, dtype=float)

        if flux.ndim != 1:
            raise ValueError("X-SHOOTER reader expects a 1D flux array in the selected HDU.")

        err_ref = err_ext if err_ext is not None else phdr.get("ERRDATA", 1)
        qual_ref = qual_ext if qual_ext is not None else phdr.get("QUALDATA", 2)

        err_hdu = _resolve_hdu(hdul, err_ref)
        qual_hdu = _resolve_hdu(hdul, qual_ref)

        err = None if err_hdu is None else np.asarray(err_hdu.data, dtype=float)
        qual = None if qual_hdu is None else np.asarray(qual_hdu.data)

        if err is not None and err.shape != flux.shape:
            raise ValueError("Error array shape does not match flux array shape.")
        if qual is not None and qual.shape != flux.shape:
            raise ValueError("Quality array shape does not match flux array shape.")

        raw_wave = _build_linear_wave_from_header(phdr, flux.size)

        cunit1 = phdr.get("CUNIT1", None)
        unit_in = wave_unit if wave_unit is not None else cunit1
        if unit_in is None:
            unit_in = "nm"

        wave = _wave_to_angstrom(raw_wave, unit_in)

        arm = str(phdr.get("HIERARCH ESO SEQ ARM", phdr.get("ARM", "unknown"))).strip().upper()
        slit_key = _xshooter_slit_keyword_for_arm(arm)
        slit_name = phdr.get(slit_key, None) if slit_key is not None else None

        resolution_R = None
        if infer_resolution:
            resolution_R = _xshooter_resolution_from_slit(arm, slit_name)

        mask = np.isfinite(wave) & np.isfinite(flux)

        if err is not None:
            mask &= np.isfinite(err) & (err > 0)

        if qual is not None:
            mask &= (np.asarray(qual) == 0)

        barycorr_kms = phdr.get("HIERARCH ESO QC VRAD BARYCOR", None)
        helicorr_kms = phdr.get("HIERARCH ESO QC VRAD HELICOR", None)

        meta = {
            "path": path,
            "instrument": phdr.get("INSTRUME", "XSHOOTER"),
            "object": phdr.get("OBJECT"),
            "arm": arm,
            "mode": phdr.get("HIERARCH ESO INS MODE"),
            "slit_keyword": slit_key,
            "slit_name": slit_name,
            "slit_width_arcsec": _xshooter_parse_slit_width(slit_name),
            "resolution_R": resolution_R,
            "date_obs": phdr.get("DATE-OBS"),
            "mjd_obs": phdr.get("MJD-OBS"),
            "exptime": phdr.get("EXPTIME"),
            "ra": phdr.get("RA"),
            "dec": phdr.get("DEC"),
            "bunit": phdr.get("BUNIT"),
            "cunit1": cunit1,
            "wave_unit_input": unit_in,
            "prodcatg": phdr.get("HIERARCH ESO PRO CATG"),
            "pipefile": phdr.get("PIPEFILE"),
            "err_ext": err_ref,
            "qual_ext": qual_ref,
            "barycorr_kms": barycorr_kms,
            "helicorr_kms": helicorr_kms,
            "telluric_corrected": _xshooter_telluric_corrected(phdr),
            "wave_medium": wave_medium,
            "wave_frame": wave_frame,
        }

        name = os.path.basename(path)

        return SpectrumSegment(
            wave,
            flux,
            err=err,
            mask=mask,
            meta=meta,
            wave_medium=wave_medium,
            wave_frame=wave_frame,
            name=name,
            observer_frame=(
                wave_frame
                if wave_frame in {"topocentric", "heliocentric", "barycentric"}
                else "unknown"
            ),
            stellar_rest_status=(
                "observed"
                if wave_frame in {"topocentric", "heliocentric", "barycentric"}
                else "unknown"
            ),
            resolution=(
                None
                if resolution_R is None
                else ResolutionDescriptor(
                    quantity="R",
                    value=resolution_R,
                    source="X-SHOOTER arm/slit metadata",
                )
            ),
        ).sorted()

def read_floyds_csv(path, name=None):
    """
    Read a reduced 1D FLOYDS spectrum from a simple ASCII/CSV export.

    Expected format:
    - optional comment lines beginning with '#'
    - one header line with column names such as 'wavelength flux'
    - numeric rows thereafter

    Returns
    -------
    SpectrumSegment
    """
    path = os.path.abspath(os.path.expanduser(path))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    comments = {}
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        body = line.lstrip()[1:].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            comments[k.strip().lower()] = v.strip()

    data_text = "".join(
        ln for ln in lines
        if ln.strip() and not ln.lstrip().startswith("#")
    )
    if not data_text.strip():
        raise ValueError("No tabular data found in FLOYDS file: {0}".format(path))

    arr = np.genfromtxt(
        _pyio.StringIO(data_text),
        names=True,
        dtype=float,
        encoding=None,
    )

    if arr.dtype.names is None:
        raise ValueError(
            "Could not parse named columns from FLOYDS file: {0}".format(path)
        )

    names = list(arr.dtype.names)
    names_l = [n.lower() for n in names]

    wave_candidates = ["wavelength", "wave", "lambda", "lam"]
    flux_candidates = ["flux", "f_lambda", "flam", "fnu"]
    err_candidates = ["err", "error", "uncertainty", "sigma", "flux_err", "fluxerror"]

    def _pick(candidates):
        for c in candidates:
            if c in names_l:
                return names[names_l.index(c)]
        return None

    wave_col = _pick(wave_candidates)
    flux_col = _pick(flux_candidates)
    err_col = _pick(err_candidates)

    if wave_col is None or flux_col is None:
        raise ValueError(
            "Need wavelength and flux columns in FLOYDS file: {0}. "
            "Found columns: {1}".format(path, names)
        )

    wave = np.asarray(arr[wave_col], dtype=float)
    flux = np.asarray(arr[flux_col], dtype=float)
    err = None if err_col is None else np.asarray(arr[err_col], dtype=float)

    mask = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)

    meta = {
        "path": path,
        "instrument": "FLOYDS",
        "facility": comments.get("facility"),
        "date_obs": comments.get("date-obs"),
        "resolution_R": 500.0,
        "resolution_note": "Approximate nominal FLOYDS merged-spectrum value; actual R varies with wavelength and slit.",
        "wave_medium": "unknown",
        "wave_frame": "unknown",
    }

    seg_name = (
        name
        or comments.get("object")
        or comments.get("target")
        or os.path.basename(path)
    )

    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium="unknown",
        wave_frame="unknown",
        name=seg_name,
        observer_frame="unknown",
        stellar_rest_status="unknown",
        resolution=ResolutionDescriptor(
            quantity="R",
            value=500.0,
            source="approximate nominal FLOYDS value",
        ),
    ).sorted()
    

def read_gemini_gmos_ascii(path, name=None):
    """
    Read a reduced 1D Gemini/GMOS spectrum from an IRAF wspectext-like ASCII export.

    Expected format:
    - optional FITS-like header cards at the top
    - optional END line terminating the header
    - numeric rows thereafter, typically wavelength flux
    - an optional third numeric column is treated as err

    Returns
    -------
    SpectrumSegment
    """
    path = os.path.abspath(os.path.expanduser(path))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header = {}
    data_lines = []
    in_header = True

    for line in lines:
        s = line.strip()

        if not s:
            continue

        if in_header:
            if s == "END":
                in_header = False
                continue

            # FITS-like header card, e.g. KEYWORD = value / comment
            if "=" in line and not re.match(r"^[+-]?[0-9]", s):
                key, rest = line.split("=", 1)
                key = key.strip()
                value = rest.split("/", 1)[0].strip()

                if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                    value = value[1:-1].strip()

                header[key] = value
                continue

            # No explicit END: if the line begins numerically, data start here.
            if re.match(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?", s):
                in_header = False
                data_lines.append(line)
                continue

            # Otherwise ignore stray non-data lines in the header block.
            continue

        data_lines.append(line)

    if len(data_lines) == 0:
        raise ValueError("No numeric spectral data found in Gemini/GMOS ASCII file: {0}".format(path))

    arr = np.genfromtxt(_pyio.StringIO("".join(data_lines)), dtype=float)

    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("Need at least two numeric columns (wave, flux) in: {0}".format(path))
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            "Could not parse Gemini/GMOS ASCII spectrum with at least two columns: {0}".format(path)
        )

    wave = np.asarray(arr[:, 0], dtype=float)
    flux = np.asarray(arr[:, 1], dtype=float)
    err = None
    if arr.shape[1] >= 3:
        err = np.asarray(arr[:, 2], dtype=float)

    mask = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)

    meta = {
        "path": path,
        "instrument": "GMOS",
        "facility": "Gemini",
        "object": header.get("OBJECT"),
        "filename": header.get("FILENAME"),
        "origin": header.get("ORIGIN"),
        "iraf_type": header.get("IRAFTYPE"),
        "wave_unit_input": "angstrom",
        "resolution_R": None,
        "wave_medium": "unknown",
        "wave_frame": "unknown",
        "header_cards": dict(header),
    }

    seg_name = (
        name
        or header.get("OBJECT")
        or header.get("FILENAME")
        or os.path.basename(path)
    )

    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium="unknown",
        wave_frame="unknown",
        name=seg_name,
        observer_frame="unknown",
        stellar_rest_status="unknown",
        resolution=None,
    ).sorted()
    

READERS = {}


def register_reader(names, func):
    """
    Register one reader function under one or more instrument aliases.
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        key = str(name).strip().lower()
        if not key:
            continue
        READERS[key] = func


register_reader(["pepsi", "pepsi_nor", "pepsi-1d", "pepsi1d"], read_pepsi_nor)
register_reader(["xshooter", "x-shooter", "xsh", "xshooter_1d", "xshooter-1d"], read_xshooter_1d)
register_reader(["floyds", "floyds_csv", "lco_floyds"], read_floyds_csv)
register_reader(["gemini", "gmos", "gemini_gmos", "gmos_ascii", "gemini_ascii"], read_gemini_gmos_ascii)

  
def read_spectrum(
    path,
    instrument=None,
    warn_unknown=True,
    **kwargs
):
    """
    Dispatch to one of the registered 1D spectrum readers.

    Parameters
    ----------
    path : str
        Input file path.
    instrument : str
        Reader alias such as "pepsi", "xshooter", "floyds", or "gemini".
    **kwargs
        Additional reader-specific keyword arguments.

    Returns
    -------
    SpectrumSegment or SpectrumCollection
        Reader output converted to the versioned common spectrum format.
    """
    inst = (instrument or "").strip().lower()
    func = READERS.get(inst, None)

    if func is None:
        raise ValueError(
            "Unknown instrument '{0}'. Supported: pepsi, xshooter, floyds, gemini".format(instrument)
        )

    spectrum = func(path, **kwargs)
    return coerce_spectrum(
        spectrum,
        wave_unit="angstrom",
        warn_unknown=warn_unknown,
        source="reader:{0}".format(inst),
    )
