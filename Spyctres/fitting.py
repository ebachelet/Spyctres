# Spyctres/fitting.py
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import legvander

# Observed-grid PHOENIX RV resampling is implemented in waveutils.py so that
# this fitting layer does not import the legacy monolithic Spyctres.py module.
from .io import SpectrumSegment, SpectrumCollection
from .phoenix import validate_phoenix_teff
from .phoenix_forward import (
    infer_segments_wave_medium,
    fit_bounds_from_segments,
    prepare_phoenix_native_template,
    build_phoenix_native_models_for_segments,
    build_native_interp_wave_grid_for_segments,
    convolve_to_resolution_loglam,
    resolve_gaussian_lsf_fwhm_kms,
)
from .preprocessing import compose_fit_mask
from .results import build_fit_quality_report
# Profiling linear continuum coefficients inside the nonlinear fit follows the
# separable least-squares design used by pPXF; see Cappellari (2023):
# https://doi.org/10.1093/mnras/stad2597

# RV handling:
# Spyctres.velocity_correction is a legacy helper and is left unchanged.
# PHOENIX fitting paths report rv_kms in the standard astronomical convention:
# positive RV redshifts the template/model. Observed-grid PHOENIX paths must
# call _apply_observed_grid_rv_shift(), not velocity_correction() directly.

from .waveutils import C_KMS, resample_flux_with_velocity_shift_observed_grid


def _coerce_segments_input(segments):
    """
    Normalize supported segment inputs to a list of SpectrumSegment objects
    plus a matching positive weight vector.

    Supported inputs
    ----------------
    - SpectrumSegment
    - list/tuple of SpectrumSegment
    - SpectrumCollection
    """
    if isinstance(segments, SpectrumCollection):
        seg_list = list(segments.segments)
        seg_weights = np.asarray(segments.weights, dtype=float)
        collection_name = segments.name
        collection_meta = dict(segments.meta)
    elif isinstance(segments, SpectrumSegment):
        seg_list = [segments]
        seg_weights = np.ones(1, dtype=float)
        collection_name = None
        collection_meta = {}
    elif isinstance(segments, (list, tuple)):
        seg_list = list(segments)
        seg_weights = np.ones(len(seg_list), dtype=float)
        collection_name = None
        collection_meta = {}
    else:
        raise TypeError(
            "segments must be a SpectrumSegment, SpectrumCollection, or "
            "list/tuple of SpectrumSegment objects."
        )

    if len(seg_list) == 0:
        raise ValueError("No segments were provided for fitting.")

    for i, seg in enumerate(seg_list):
        if not isinstance(seg, SpectrumSegment):
            raise TypeError(
                "All segment inputs must be SpectrumSegment objects; "
                "got type {0} at index {1}.".format(type(seg).__name__, i)
            )

    if seg_weights.ndim != 1 or len(seg_weights) != len(seg_list):
        raise ValueError("Segment weights must be 1D and match the number of segments.")
    if not np.all(np.isfinite(seg_weights)):
        raise ValueError("Segment weights must be finite.")
    if np.any(seg_weights <= 0):
        raise ValueError("Segment weights must be > 0.")

    return seg_list, seg_weights, collection_name, collection_meta


def _resolve_broadening_fwhm_kms(R=None, fwhm_kms=None):
    """
    Resolve the effective Gaussian FWHM in km/s.
    Exactly one of R or fwhm_kms may be provided.
    """
    return resolve_gaussian_lsf_fwhm_kms(R=R, fwhm_kms=fwhm_kms)


def _resolve_segment_fwhm_kms(seg, R=None, fwhm_kms=None):
    """
    Resolve the Gaussian LSF FWHM in km/s for one segment.

    Precedence:
    1. seg.resolution
    2. seg.meta["lsf_fwhm_kms"]
    3. seg.meta["fwhm_kms"]
    4. seg.meta["resolution_R"]
    5. global fwhm_kms
    6. global R
    7. None
    """
    descriptor = getattr(seg, "resolution", None)
    if descriptor is not None:
        if descriptor.mode != "constant":
            raise ValueError(
                "Wavelength-dependent resolution is recorded for segment {0}, "
                "but the current fitter supports only a constant LSF per segment.".format(
                    getattr(seg, "name", None)
                )
            )
        if descriptor.quantity == "R":
            return resolve_gaussian_lsf_fwhm_kms(R=descriptor.value)
        if descriptor.quantity == "fwhm_kms":
            return resolve_gaussian_lsf_fwhm_kms(fwhm_kms=descriptor.value)
        if descriptor.quantity == "sigma_kms":
            return resolve_gaussian_lsf_fwhm_kms(
                fwhm_kms=descriptor.value * 2.3548200450309493
            )

    meta = getattr(seg, "meta", {}) or {}

    for key in ("lsf_fwhm_kms", "fwhm_kms"):
        val = meta.get(key, None)
        if val is None:
            continue
        try:
            return resolve_gaussian_lsf_fwhm_kms(fwhm_kms=val)
        except ValueError as exc:
            raise ValueError(
                "Invalid {0} for segment {1}: {2}".format(
                    key,
                    getattr(seg, "name", None),
                    exc,
                )
            ) from exc

    val = meta.get("resolution_R", None)
    if val is not None:
        try:
            return resolve_gaussian_lsf_fwhm_kms(R=val)
        except ValueError as exc:
            raise ValueError(
                "Invalid resolution_R for segment {0}: {1}".format(
                    getattr(seg, "name", None),
                    exc,
                )
            ) from exc

    return _resolve_broadening_fwhm_kms(R=R, fwhm_kms=fwhm_kms)


def _validate_optimizer_loss(loss, loss_f_scale):
    """Validate scipy.optimize.least_squares robust-loss controls."""
    allowed = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    loss = str(loss)
    if loss not in allowed:
        raise ValueError(
            "loss must be one of {0}; got {1!r}.".format(
                ", ".join(sorted(allowed)),
                loss,
            )
        )
    loss_f_scale = float(loss_f_scale)
    if not np.isfinite(loss_f_scale) or loss_f_scale <= 0.0:
        raise ValueError("loss_f_scale must be finite and > 0.")
    return loss, loss_f_scale
    
    
def _gaussian_broaden_velocity(wave, flux, fwhm_kms=None):
    """
    Compatibility wrapper for Gaussian velocity-space broadening.

    The canonical implementation lives in Spyctres.phoenix_forward as
    convolve_to_resolution_loglam(), because instrumental broadening is part of
    the forward model. This wrapper is kept temporarily for older internal calls.
    """
    return convolve_to_resolution_loglam(
        wave_A=wave,
        flux=flux,
        fwhm_kms=fwhm_kms,
    )


def build_effective_fit_mask(
    seg,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
):
    """
    Build the effective boolean fit mask for a single SpectrumSegment.

    This reproduces the point-selection logic used by fit_phoenix_full_spectrum
    for a single segment, except that if seg.err is None it does not invent a
    synthetic error array. In that case only finite wave/flux and the supplied
    mask/region logic are applied.
    """
    return compose_fit_mask(
        seg,
        regions=regions,
        exclude_regions=exclude_regions,
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
        mask_threshold=mask_threshold,
    ).effective_mask


def build_excluded_mask(
    seg,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
):
    """
    Build a boolean mask of pixels explicitly excluded by region/exclude rules.

    This is intended for plotting diagnostics. It does not mark pixels excluded
    only because they are NaN, have non-positive errors, or lie outside seg.mask.
    """
    return compose_fit_mask(
        seg,
        regions=regions,
        exclude_regions=exclude_regions,
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
        mask_threshold=mask_threshold,
    ).excluded_mask


def _estimate_sigma(flux):
    """Rough robust sigma estimate for flux if no errors are provided."""
    f = np.asarray(flux, dtype=float)
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(f)
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return float(sig)


def _segment_support_ok(seg):
    """Return pixels eligible for model support in the fitter/reconstructor."""
    w_full = np.asarray(seg.wave, dtype=float)
    f_full = np.asarray(seg.flux, dtype=float)
    support_ok = np.isfinite(w_full) & np.isfinite(f_full)
    if seg.err is not None:
        e_full = np.asarray(seg.err, dtype=float)
        support_ok &= np.isfinite(e_full) & (e_full > 0)
    return support_ok


def _select_segment_option(option, index, seg):
    """Select an optional per-segment setting by index or segment name."""
    if not isinstance(option, dict):
        return option
    if "callable" in option or "func" in option:
        return option
    if index in option:
        return option[index]
    if seg.name in option:
        return option[seg.name]
    return None


def _is_global_named_mask_spec(option):
    return (
        isinstance(option, dict)
        and ("callable" in option or "func" in option)
    )


def _resolve_per_segment_mask_options(
    segments,
    option,
    label,
    strict=True,
):
    """Resolve global or per-segment mask options for all input segments."""
    if option is None:
        return [None] * len(segments)
    if not isinstance(option, dict) or _is_global_named_mask_spec(option):
        return [option] * len(segments)

    resolved = [None] * len(segments)
    assigned_by = [None] * len(segments)
    names = [seg.name for seg in segments]
    named_keys = [key for key in option if isinstance(key, str)]

    if named_keys:
        seen = {}
        duplicates = set()
        for index, name in enumerate(names):
            if name is None:
                continue
            if name in seen:
                duplicates.add(name)
            seen[name] = index
        if duplicates and strict:
            raise ValueError(
                "{0} uses name-keyed masks, but segment names are not unique: {1}.".format(
                    label, ", ".join(sorted(str(x) for x in duplicates))
                )
            )

    for key, value in option.items():
        if isinstance(key, int):
            index = int(key)
            if index < 0 or index >= len(segments):
                if strict:
                    raise ValueError(
                        "{0} index key {1} is out of range for {2} segments.".format(
                            label, key, len(segments)
                        )
                    )
                continue
            source = "index"
        elif isinstance(key, str):
            matches = [i for i, name in enumerate(names) if name == key]
            if len(matches) == 0:
                if strict:
                    raise ValueError(
                        "{0} segment-name key {1!r} does not match any segment.".format(
                            label, key
                        )
                    )
                continue
            if len(matches) > 1:
                if strict:
                    raise ValueError(
                        "{0} segment-name key {1!r} matches multiple segments.".format(
                            label, key
                        )
                    )
                continue
            index = matches[0]
            source = "name"
        else:
            if strict:
                raise TypeError(
                    "{0} keys must be integer segment indices or string segment names.".format(
                        label
                    )
                )
            continue

        if assigned_by[index] is not None and strict:
            raise ValueError(
                "{0} assigns segment {1} by both {2} and {3}; use one target form.".format(
                    label, index, assigned_by[index], source
                )
            )
        resolved[index] = value
        assigned_by[index] = source

    return resolved


def _build_data_vectors(
    segments,
    segment_weights=None,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
    mask_assignment_strict=True,
):
    """
    Build synchronized support-wave and fit-point data vectors.

    Returns
    -------
    support_wave_all : ndarray
        Concatenated full support wavelength grid across retained segments.
    flux_fit_all, err_fit_all : ndarray
        Concatenated flux/error vectors for fit pixels only.
    support_slices : list[slice]
        Per-segment slices into support_wave_all.
    fit_slices : list[slice]
        Per-segment slices into flux_fit_all / err_fit_all.
    fit_masks : list[ndarray(bool)]
        Boolean masks mapping each segment support grid to its fit pixels.
    fit_weights : ndarray
        Positive per-segment weights aligned with the retained segment list.
    seg_meta : list[dict]
        Per-segment metadata for retained segments.
    """
    if segment_weights is None:
        segment_weights = np.ones(len(segments), dtype=float)
    else:
        segment_weights = np.asarray(segment_weights, dtype=float)
        if segment_weights.ndim != 1 or len(segment_weights) != len(segments):
            raise ValueError("segment_weights must be 1D and match the number of segments.")

    support_wave_all = []
    flux_fit_all = []
    err_fit_all = []
    support_slices = []
    fit_slices = []
    fit_masks = []
    fit_weights = []
    seg_meta = []
    exclude_mask_by_segment = _resolve_per_segment_mask_options(
        segments,
        exclude_mask,
        "exclude_mask",
        strict=mask_assignment_strict,
    )
    exclude_masks_by_segment = _resolve_per_segment_mask_options(
        segments,
        exclude_masks,
        "exclude_masks",
        strict=mask_assignment_strict,
    )

    start_support = 0
    start_fit = 0

    for i, (seg, seg_weight) in enumerate(zip(segments, segment_weights)):
        w_full = np.asarray(seg.wave, dtype=float)
        f_full = np.asarray(seg.flux, dtype=float)

        support_ok = _segment_support_ok(seg)

        if seg.err is None:
            e_full = np.ones_like(f_full) * _estimate_sigma(
                f_full[support_ok] if np.any(support_ok) else f_full
            )
        else:
            e_full = np.asarray(seg.err, dtype=float)

        reg = _select_segment_option(regions, i, seg)
        ex = _select_segment_option(exclude_regions, i, seg)
        ex_mask = exclude_mask_by_segment[i]
        ex_masks = exclude_masks_by_segment[i]

        mask_result = compose_fit_mask(
            seg,
            regions=reg,
            exclude_regions=ex,
            exclude_mask=ex_mask,
            exclude_masks=ex_masks,
            mask_threshold=mask_threshold,
        )
        fit_m = mask_result.effective_mask
        fit_m &= support_ok

        n_support = int(np.sum(support_ok))
        n_fit = int(np.sum(fit_m))

        if n_support == 0 or n_fit == 0:
            continue

        w_support = w_full[support_ok].astype(float)
        f_fit = f_full[fit_m].astype(float)
        e_fit = e_full[fit_m].astype(float)

        support_wave_all.append(w_support)
        flux_fit_all.append(f_fit)
        err_fit_all.append(e_fit)

        support_slices.append(slice(start_support, start_support + n_support))
        fit_slices.append(slice(start_fit, start_fit + n_fit))
        fit_masks.append(fit_m[support_ok])
        fit_weights.append(float(seg_weight))

        seg_meta.append({
            "name": seg.name,
            "index": int(i),
            "weight": float(seg_weight),
            "wave_min": float(w_support.min()),
            "wave_max": float(w_support.max()),
            "n_support": n_support,
            "n_fit": n_fit,
            "mask_provenance": mask_result.to_metadata(label="fit selection"),
            "mask_summary": mask_result.to_summary(),
        })

        start_support += n_support
        start_fit += n_fit

    support_wave_all = np.concatenate(support_wave_all) if support_wave_all else np.array([], dtype=float)
    flux_fit_all = np.concatenate(flux_fit_all) if flux_fit_all else np.array([], dtype=float)
    err_fit_all = np.concatenate(err_fit_all) if err_fit_all else np.array([], dtype=float)
    fit_weights = np.asarray(fit_weights, dtype=float)

    return (
        support_wave_all,
        flux_fit_all,
        err_fit_all,
        support_slices,
        fit_slices,
        fit_masks,
        fit_weights,
        seg_meta,
    )
    
    
def _pick_subgrid(full_grid, center, half_width, n_min=3, n_max=None):
    """
    Pick a small sorted subgrid around 'center' from a known full_grid.
    half_width is in the same units as the grid.
    """
    g = np.asarray(full_grid, dtype=float)
    if g.ndim != 1 or g.size == 0:
        raise ValueError("full_grid must be a non-empty 1D array.")

    lo = center - half_width
    hi = center + half_width
    sub = g[(g >= lo) & (g <= hi)]

    if sub.size < n_min:
        # fall back to nearest points
        n = int(n_min if n_max is None else min(n_min, n_max))
        idx = np.argsort(np.abs(g - center))[:max(1, n)]
        return np.sort(g[idx])

    if n_max is not None and sub.size > n_max:
        idx = np.argsort(np.abs(sub - center))[:int(n_max)]
        sub = sub[idx]

    return np.sort(sub)


def _apply_observed_grid_rv_shift(wave, model_flux, rv_kms):
    """
    Apply RV to a model already sampled on the observed wavelength grid.

    PHOENIX fitter convention
    -------------------------
    The PHOENIX fitting API reports rv_kms using the standard astronomical
    convention:

        positive RV => redshifted template/model features
        lambda_observed = lambda_rest * (1 + RV / c)

    Legacy compatibility
    --------------------
    This wrapper delegates to waveutils.py rather than importing the legacy
    monolithic Spyctres.py module. The legacy Spyctres.velocity_correction()
    public API remains unchanged.
    """
    return resample_flux_with_velocity_shift_observed_grid(
        wave=wave,
        flux=model_flux,
        rv_kms=rv_kms,
    )
    
    
def _chi2_for_params(
    support_wave_all, flux_all, err_all,
    support_slices, fit_slices, fit_masks, segment_weights,
    teff, feh, logg, rv_tot, phoenix_lib, mdeg,
    decimate=1,
    segment_fwhm_kms=None,
):
    """
    Compute weighted chi-square with per-segment multiplicative polynomial solved linearly.
    Used for RV initialization.

    The model is built on the full support wavelength grid, but chi2 is
    evaluated only on the fit pixels inside each segment.
    """
    model0 = phoenix_lib.evaluate(teff, feh, logg)
    shifted = _apply_observed_grid_rv_shift(support_wave_all, model0, rv_tot)
    chi2 = 0.0
    
    if segment_fwhm_kms is None:
        segment_fwhm_kms = [None] * len(support_slices)
        
    for support_sl, fit_sl, fit_mask, seg_weight, seg_fwhm in zip(
        support_slices, fit_slices, fit_masks, segment_weights, segment_fwhm_kms
    ):
        w_support = support_wave_all[support_sl]
        m_support = _gaussian_broaden_velocity(
            w_support,
            shifted[support_sl],
            fwhm_kms=seg_fwhm,
        )

        w = w_support[fit_mask]
        f = flux_all[fit_sl]
        e = err_all[fit_sl]
        m = m_support[fit_mask]

        if decimate and int(decimate) > 1:
            idx = np.arange(len(w))[::int(decimate)]
            w = w[idx]
            f = f[idx]
            e = e[idx]
            m = m[idx]

        m_corr, _ = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
        r = (f - m_corr) / e
        chi2 += float(seg_weight) * float(np.sum(r * r))
    
    return chi2


def _make_forward_segments(segments, support_wave_all, support_slices, fit_masks):
    """
    Build support-grid SpectrumSegment objects for forward-model evaluation.

    These segments live on the cleaned support wavelength grids used internally
    by fitting.py, with seg.mask marking the fit pixels on each support grid.
    """
    if not (len(segments) == len(support_slices) == len(fit_masks)):
        raise ValueError(
            "Forward segments, support slices, and fit masks must remain aligned."
        )
    out = []
    for seg, support_sl, fit_mask in zip(segments, support_slices, fit_masks):
        w = np.asarray(support_wave_all[support_sl], dtype=float)
        out.append(
            SpectrumSegment(
                wave=w,
                flux=np.ones_like(w, dtype=float),
                err=np.ones_like(w, dtype=float),
                mask=np.asarray(fit_mask, dtype=bool),
                meta=dict(getattr(seg, "meta", {})),
                wave_medium=getattr(seg, "wave_medium", None),
                wave_frame=getattr(seg, "wave_frame", None),
                name=getattr(seg, "name", None),
                observer_frame=getattr(seg, "observer_frame", "unknown"),
                stellar_rest_status=getattr(seg, "stellar_rest_status", "unknown"),
                stellar_rv_applied_kms=getattr(
                    seg, "stellar_rv_applied_kms", None
                ),
                resolution=getattr(seg, "resolution", None),
            )
        )
    return out


def _retained_segments_from_meta(segments, seg_meta):
    """Recover the input segments retained by `_build_data_vectors`."""
    retained = []
    for meta in seg_meta:
        index = int(meta["index"])
        if index < 0 or index >= len(segments):
            raise ValueError("Retained segment index is outside the input segment list.")
        retained.append(segments[index])
    return retained


def _full_spectrum_parameter_count(n_retained_segments, mdeg):
    """Count nonlinear parameters plus retained per-segment continua."""
    n_retained_segments = int(n_retained_segments)
    mdeg = int(mdeg)
    if n_retained_segments < 1:
        raise ValueError("n_retained_segments must be >= 1.")
    if mdeg < 0:
        raise ValueError("mdeg must be >= 0.")
    return 4 + n_retained_segments * (mdeg + 1)


def _metadata_values(segments, attribute):
    values = []
    for segment in segments:
        value = getattr(segment, attribute, "unknown")
        values.append("unknown" if value is None else str(value))
    return values


def _metadata_summary(segments):
    fields = (
        "wave_medium",
        "wave_frame",
        "observer_frame",
        "stellar_rest_status",
    )
    summary = {}
    for field in fields:
        values = _metadata_values(segments, field)
        summary[field] = {
            "values": values,
            "unique": sorted(set(values)),
            "unknown_count": int(sum(value.lower() == "unknown" for value in values)),
        }
    return summary


def _parameter_grid_summary(phoenix_lib):
    grid = getattr(phoenix_lib, "_grid", None)
    if grid is None:
        grid = (
            getattr(phoenix_lib, "DEFAULT_TEFF_GRID", ()),
            getattr(phoenix_lib, "DEFAULT_FEH_GRID", ()),
            getattr(phoenix_lib, "DEFAULT_LOGG_GRID", ()),
        )
    labels = ("teff", "feh", "logg")
    summary = {}
    for label, values in zip(labels, grid):
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            summary[label] = {"min": None, "max": None, "n": 0}
        else:
            summary[label] = {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "n": int(finite.size),
            }
    return summary


def _grid_edge_flags(best, grid_summary, rtol=0.0, atol=1e-8):
    """Return JSON-safe PHOENIX grid-edge flags with low/high specificity.

    The legacy per-parameter flags (``teff``, ``feh``, ``logg``) are retained
    for compatibility.  The ``*_low`` and ``*_high`` flags make the failure
    mode auditable when a fit lands on, or is clipped beyond, a grid boundary.
    """
    flags = {}
    any_edge = False
    for label, value in zip(("teff", "feh", "logg"), best[:3]):
        info = grid_summary.get(label, {})
        lo = info.get("min")
        hi = info.get("max")
        if lo is None or hi is None:
            flags[label] = False
            flags["{0}_low".format(label)] = False
            flags["{0}_high".format(label)] = False
            continue
        value = float(value)
        lo = float(lo)
        hi = float(hi)
        low = bool(np.isclose(value, lo, rtol=rtol, atol=atol) or value <= lo)
        high = bool(np.isclose(value, hi, rtol=rtol, atol=atol) or value >= hi)
        edge = bool(low or high)
        flags[label] = edge
        flags["{0}_low".format(label)] = low
        flags["{0}_high".format(label)] = high
        any_edge = bool(any_edge or edge)
    flags["fit_bound_hit"] = any_edge
    return flags


def _unique_metadata_value(segments, attribute):
    values = _metadata_values(segments, attribute)
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _velocity_convention_summary(forward_segments, rv_kms, rv_bary_kms):
    """Return explicit velocity/frame semantics for result JSON and audits."""
    rv_kms = float(rv_kms)
    rv_bary_kms = float(rv_bary_kms)
    total = rv_kms + rv_bary_kms
    return {
        "rv_kms_fit": rv_kms,
        "rv_bary_kms_input": rv_bary_kms,
        "total_model_shift_kms": float(total),
        "rv_sign_convention": (
            "positive velocity redshifts the stellar template/model; "
            "positive stellar RV means the source is receding"
        ),
        "rv_combination_formula": "total_model_shift_kms = rv_kms_fit + rv_bary_kms_input",
        "rv_bary_applied_to_model": bool(not np.isclose(rv_bary_kms, 0.0)),
        "rv_bary_term_in_model_formula": True,
        "rv_bary_applied_to_data": "unknown",
        "wavelength_frame_assumption": {
            "wave_frame": _unique_metadata_value(forward_segments, "wave_frame"),
            "observer_frame": _unique_metadata_value(forward_segments, "observer_frame"),
            "stellar_rest_status": _unique_metadata_value(
                forward_segments, "stellar_rest_status"
            ),
            "wave_medium": _unique_metadata_value(forward_segments, "wave_medium"),
        },
    }


def _residual_shape_diagnostics(residuals):
    residuals = np.asarray(residuals, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return {
            "residual_rms": None,
            "residual_slope": None,
            "durbin_watson": None,
            "residual_autocorrelation_flag": False,
        }
    rms = float(np.sqrt(np.mean(finite * finite)))
    if finite.size > 1:
        x = np.linspace(-0.5, 0.5, finite.size)
        slope = float(np.polyfit(x, finite, 1)[0])
        denom = float(np.sum(finite * finite))
        dw = None if denom <= 0.0 else float(np.sum(np.diff(finite) ** 2) / denom)
        autocorr = bool(dw is not None and (dw < 1.2 or dw > 2.8))
    else:
        slope = None
        dw = None
        autocorr = False
    return {
        "residual_rms": rms,
        "residual_slope": slope,
        "durbin_watson": dw,
        "residual_autocorrelation_flag": autocorr,
    }


def _build_phoenix_fit_diagnostics(
    *,
    residuals,
    chi2,
    chi2_red,
    dof,
    n_parameters,
    input_segments,
    forward_segments,
    seg_meta,
    mdeg,
    best_parameters,
    phoenix_lib,
    segment_fwhm_kms,
    local_solutions,
    coarse_initialization,
    rv_kms=0.0,
    rv_bary_kms=0.0,
):
    """Build a compact, JSON-safe diagnostics block for PHOENIX fits."""
    retained_count = int(len(seg_meta))
    input_count = int(len(input_segments))
    support_points = int(sum(meta.get("n_support", 0) for meta in seg_meta))
    fit_points = int(sum(meta.get("n_fit", 0) for meta in seg_meta))
    mask_fraction = (
        None
        if support_points <= 0
        else float(1.0 - fit_points / float(support_points))
    )

    segment_diagnostics = []
    for index, meta in enumerate(seg_meta):
        n_support = int(meta.get("n_support", 0))
        n_fit = int(meta.get("n_fit", 0))
        mask_summary = dict(meta.get("mask_summary", {}))
        mask_provenance = dict(meta.get("mask_provenance", {}))
        segment_diagnostics.append(
            {
                "name": meta.get("name"),
                "input_index": int(meta.get("index", index)),
                "weight": float(meta.get("weight", 1.0)),
                "n_support": n_support,
                "n_fit": n_fit,
                "mask_fraction": (
                    None if n_support <= 0 else float(1.0 - n_fit / float(n_support))
                ),
                "wave_min": meta.get("wave_min"),
                "wave_max": meta.get("wave_max"),
                "mask_summary": mask_summary,
                "mask_provenance": mask_provenance,
                "lsf_fwhm_kms": (
                    None
                    if segment_fwhm_kms[index] is None
                    else float(segment_fwhm_kms[index])
                ),
                "resolution_R_effective": (
                    None
                    if segment_fwhm_kms[index] is None
                    else float(C_KMS / segment_fwhm_kms[index])
                ),
            }
        )

    grid_summary = _parameter_grid_summary(phoenix_lib)
    edge_flags = _grid_edge_flags(np.asarray(best_parameters, dtype=float), grid_summary)
    residual_shape = _residual_shape_diagnostics(residuals)
    local_solution_summaries = []
    for solution in local_solutions:
        local_solution_summaries.append(
            {
                "start": solution.get("start"),
                "solution": solution.get("solution"),
                "chi2": solution.get("chi2"),
                "success": bool(solution.get("success", False)),
                "status": solution.get("status"),
                "nfev": solution.get("nfev"),
            }
        )

    coarse = coarse_initialization or {}
    coarse_summary = {
        "available": bool(coarse),
        "candidates_evaluated": coarse.get("candidates_evaluated"),
        "candidates_complete": coarse.get("candidates_complete"),
        "best": coarse.get("best"),
        "top_candidates": coarse.get("top_candidates", []),
    }
    velocity_convention = _velocity_convention_summary(
        forward_segments,
        rv_kms=rv_kms,
        rv_bary_kms=rv_bary_kms,
    )

    return {
        "schema_version": 1,
        "n_pixels": int(np.asarray(residuals).size),
        "n_input_segments": input_count,
        "n_retained_segments": retained_count,
        "n_dropped_segments": int(input_count - retained_count),
        "n_parameters": int(n_parameters),
        "degrees_of_freedom": int(dof),
        "chi2": float(chi2),
        "reduced_chi2": float(chi2_red),
        "segment_diagnostics": segment_diagnostics,
        "grid_summary": grid_summary,
        "grid_edge_flags": edge_flags,
        "rv_start_values": [
            float(solution["start"][3])
            for solution in local_solution_summaries
            if solution.get("start") is not None and len(solution["start"]) >= 4
        ],
        "velocity_convention": velocity_convention,
        "total_model_shift_kms": velocity_convention["total_model_shift_kms"],
        "coarse_grid_candidates": coarse_summary,
        "local_solution_summaries": local_solution_summaries,
        "mask_fraction": mask_fraction,
        "continuum_degree": int(mdeg),
        "continuum_warning_flags": [],
        "segment_rv_scatter": None,
        "wavelength_metadata_summary": _metadata_summary(forward_segments),
        "resolution_metadata_summary": {
            "missing_count": int(sum(value is None for value in segment_fwhm_kms)),
            "segment_lsf_fwhm_kms": [
                None if value is None else float(value)
                for value in segment_fwhm_kms
            ],
            "segment_resolution_R_effective": [
                None if value is None else float(C_KMS / value)
                for value in segment_fwhm_kms
            ],
        },
        **residual_shape,
    }


def _phoenix_quality_flags(diagnostics, success=True, high_chi2_threshold=5.0):
    """Return deterministic quality flags; flags never alter the fit."""
    flags = []
    if not bool(success):
        flags.append("optimizer_local_minimum_suspected")
    reduced_chi2 = diagnostics.get("reduced_chi2")
    if reduced_chi2 is not None and float(reduced_chi2) > float(high_chi2_threshold):
        flags.append("high_chi2")
    if diagnostics.get("residual_autocorrelation_flag"):
        flags.append("structured_residuals")
    slope = diagnostics.get("residual_slope")
    if slope is not None and abs(float(slope)) > 1.0:
        flags.append("residual_slope")
    edge_flags = diagnostics.get("grid_edge_flags", {})
    for label in ("teff", "feh", "logg"):
        if edge_flags.get(label):
            flags.append("grid_edge_{0}".format(label))
        for side in ("low", "high"):
            if edge_flags.get("{0}_{1}".format(label, side)):
                flags.append("grid_edge_{0}_{1}".format(label, side))
    if edge_flags.get("fit_bound_hit"):
        flags.append("fit_bound_hit")
    if diagnostics.get("resolution_metadata_summary", {}).get("missing_count", 0) > 0:
        flags.append("resolution_missing")

    wavelength = diagnostics.get("wavelength_metadata_summary", {})
    unknown_fields = [
        field
        for field, info in wavelength.items()
        if int(info.get("unknown_count", 0)) > 0
    ]
    if "wave_frame" in unknown_fields:
        flags.append("wavelength_frame_ambiguous")
    if unknown_fields:
        flags.append("metadata_incomplete")

    if diagnostics.get("mask_fraction") is not None and diagnostics["mask_fraction"] > 0.5:
        flags.append("mask_fraction_high")
    if int(diagnostics.get("n_dropped_segments", 0)) > 0:
        flags.append("segment_no_fit_pixels")

    segment_diagnostics = diagnostics.get("segment_diagnostics", [])
    n_parameters = int(diagnostics.get("n_parameters", 0) or 0)
    min_pixels = max(20, n_parameters + 5)
    if any(int(segment.get("n_fit", 0)) < min_pixels for segment in segment_diagnostics):
        flags.append("too_few_fit_pixels")
    if any(
        segment.get("mask_fraction") is not None
        and float(segment.get("mask_fraction")) > 0.5
        for segment in segment_diagnostics
    ):
        flags.append("segment_mask_fraction_high")
    if any(
        int(segment.get("mask_summary", {}).get("n_rejected_by_explicit_union", 0))
        > int(segment.get("mask_summary", {}).get("n_fit", 0))
        for segment in segment_diagnostics
    ):
        flags.append("explicit_exclusion_dominates")
    if any(
        int(
            segment.get("mask_provenance", {})
            .get("counts", {})
            .get("nonfinite_mask_output", 0)
        )
        > 0
        for segment in segment_diagnostics
    ):
        flags.append("nonfinite_mask_output")

    return ["ok"] if not flags else sorted(set(flags))


def _chi2_for_params_native_interp(
    forward_segments,
    flux_all,
    err_all,
    fit_slices,
    fit_masks,
    segment_weights,
    teff,
    feh,
    logg,
    rv_tot,
    phoenix_lib,
    model_wave_grid,
    model_wave_medium,
    mdeg,
    decimate=1,
    segment_fwhm_kms=None,
    model_margin_A=200.0,
):
    """
    Compute weighted chi-square for the native_interp branch.

    The PHOENIX model is interpolated in parameter space on a dense model-space
    wavelength grid, then shifted, convolved, and resampled to each segment
    support grid before continuum fitting.
    """
    model_dense = phoenix_lib.evaluate(teff, feh, logg)
    return _chi2_for_dense_model_native_interp(
        forward_segments=forward_segments,
        flux_all=flux_all,
        err_all=err_all,
        fit_slices=fit_slices,
        fit_masks=fit_masks,
        segment_weights=segment_weights,
        model_dense=model_dense,
        rv_tot=rv_tot,
        model_wave_grid=model_wave_grid,
        model_wave_medium=model_wave_medium,
        mdeg=mdeg,
        decimate=decimate,
        segment_fwhm_kms=segment_fwhm_kms,
        model_margin_A=model_margin_A,
    )


def _chi2_for_dense_model_native_interp(
    forward_segments,
    flux_all,
    err_all,
    fit_slices,
    fit_masks,
    segment_weights,
    model_dense,
    rv_tot,
    model_wave_grid,
    model_wave_medium,
    mdeg,
    decimate=1,
    segment_fwhm_kms=None,
    model_margin_A=200.0,
):
    """Score a prepared dense PHOENIX model without an interpolator lookup."""

    model_list = build_phoenix_native_models_for_segments(
        segments=forward_segments,
        phoenix_wave_native=model_wave_grid,
        template_flux_native=model_dense,
        rv_kms=rv_tot,
        rv_bary_kms=0.0,
        segment_fwhm_kms=segment_fwhm_kms,
        phoenix_wave_medium=model_wave_medium,
        model_margin_A=model_margin_A,
        bounds_use_fit_mask=True,
        extrapolate=True,
    )

    chi2 = 0.0
    for seg, model_full, fit_sl, fit_mask, seg_weight in zip(
        forward_segments, model_list, fit_slices, fit_masks, segment_weights
    ):
        w = np.asarray(seg.wave, dtype=float)[fit_mask]
        f = flux_all[fit_sl]
        e = err_all[fit_sl]
        m = np.asarray(model_full, dtype=float)[fit_mask]

        if decimate and int(decimate) > 1:
            idx = np.arange(len(w))[::int(decimate)]
            w = w[idx]
            f = f[idx]
            e = e[idx]
            m = m[idx]

        m_corr, _ = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
        r = (f - m_corr) / e
        chi2 += float(seg_weight) * float(np.sum(r * r))

    return chi2


def _default_coarse_grid(full_grid, targets):
    """Map a compact set of physical targets onto installed grid nodes."""
    grid = np.asarray(full_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("full_grid must be a non-empty 1D array.")
    return np.unique([grid[np.argmin(np.abs(grid - value))] for value in targets])


def _axis_between_coarse_neighbors(full_grid, coarse_grid, center, max_points):
    """Return installed nodes bracketed by neighboring sparse-grid values."""
    full_grid = np.unique(np.asarray(full_grid, dtype=float))
    coarse_grid = np.unique(np.asarray(coarse_grid, dtype=float))
    center_index = int(np.argmin(np.abs(coarse_grid - float(center))))
    lo = coarse_grid[max(0, center_index - 1)]
    hi = coarse_grid[min(coarse_grid.size - 1, center_index + 1)]
    candidates = full_grid[(full_grid >= lo) & (full_grid <= hi)]
    max_points = int(max_points)
    if candidates.size <= max_points:
        return candidates
    indices = np.unique(
        np.rint(np.linspace(0, candidates.size - 1, max_points)).astype(int)
    )
    return candidates[indices]


def _coarse_physical_start_native_interp(
    forward_segments,
    flux_all,
    err_all,
    fit_slices,
    fit_masks,
    segment_weights,
    phoenix_lib,
    model_wave_grid,
    model_wave_medium,
    rv_tot,
    mdeg,
    segment_fwhm_kms,
    model_margin_A,
    teff_grid=None,
    feh_grid=None,
    logg_grid=None,
    decimate=12,
):
    """Select a physical starting region by scoring sparse PHOENIX nodes.

    This stage reads only the requested node templates and never constructs a
    full rectangular interpolator. It therefore avoids the memory blow-up of a
    survey-wide dense cache while still testing widely separated regions of
    parameter space. The local interpolator is built only after the best coarse
    node has been identified. The staged strategy is motivated by the dispersed
    initial searches used by ASPCAP/FERRE; see Garcia Perez et al. (2016):
    https://doi.org/10.3847/0004-6256/151/6/144
    """
    if teff_grid is None:
        teff_grid = _default_coarse_grid(
            phoenix_lib.DEFAULT_TEFF_GRID,
            [3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0, 12000.0],
        )
    if feh_grid is None:
        feh_grid = _default_coarse_grid(
            phoenix_lib.DEFAULT_FEH_GRID, [-2.0, -1.0, 0.0]
        )
    if logg_grid is None:
        logg_grid = _default_coarse_grid(
            phoenix_lib.DEFAULT_LOGG_GRID, [0.0, 2.5, 4.5]
        )

    teff_grid = np.unique(np.asarray(teff_grid, dtype=float))
    feh_grid = np.unique(np.asarray(feh_grid, dtype=float))
    logg_grid = np.unique(np.asarray(logg_grid, dtype=float))
    if min(teff_grid.size, feh_grid.size, logg_grid.size) == 0:
        raise ValueError("Coarse physical grids must be non-empty.")
    decimate = int(decimate)
    if decimate < 1:
        raise ValueError("coarse_decimate must be >= 1.")

    scores = []
    evaluated = set()

    def score_node(teff, feh, logg):
        node = (float(teff), float(feh), float(logg))
        if node in evaluated:
            return
        evaluated.add(node)
        validate_phoenix_teff(teff)
        if not phoenix_lib.has_template(teff, logg, feh):
            return
        try:
            _wave, model_dense = phoenix_lib.load_template(
                teff,
                logg,
                feh,
                wave=model_wave_grid,
                wave_medium=model_wave_medium,
            )
            chi2 = _chi2_for_dense_model_native_interp(
                forward_segments=forward_segments,
                flux_all=flux_all,
                err_all=err_all,
                fit_slices=fit_slices,
                fit_masks=fit_masks,
                segment_weights=segment_weights,
                model_dense=model_dense,
                rv_tot=rv_tot,
                model_wave_grid=model_wave_grid,
                model_wave_medium=model_wave_medium,
                mdeg=mdeg,
                decimate=decimate,
                segment_fwhm_kms=segment_fwhm_kms,
                model_margin_A=model_margin_A,
            )
        except (FileNotFoundError, ValueError):
            return
        if np.isfinite(chi2):
            scores.append(
                {
                    "teff": float(teff),
                    "feh": float(feh),
                    "logg": float(logg),
                    "chi2": float(chi2),
                }
            )

    for teff in teff_grid:
        for feh in feh_grid:
            for logg in logg_grid:
                score_node(teff, feh, logg)
    if not scores:
        raise ValueError("No valid PHOENIX templates were available for coarse initialization.")
    scores.sort(key=lambda item: item["chi2"])

    # Refine Teff and log(g) between the neighboring sparse nodes while holding
    # metallicity at the best first-stage value. This prevents an edge node
    # such as 12000 K from creating a one-sided local interpolation grid.
    first_best = scores[0]
    refine_teff = _axis_between_coarse_neighbors(
        phoenix_lib.DEFAULT_TEFF_GRID,
        teff_grid,
        first_best["teff"],
        max_points=11,
    )
    refine_logg = _axis_between_coarse_neighbors(
        phoenix_lib.DEFAULT_LOGG_GRID,
        logg_grid,
        first_best["logg"],
        max_points=7,
    )
    for teff in refine_teff:
        for logg in refine_logg:
            score_node(teff, first_best["feh"], logg)
    scores.sort(key=lambda item: item["chi2"])
    return scores[0], scores


def _start_is_duplicate(candidate, starts):
    """Return whether a candidate local start is already represented."""
    return any(
        np.allclose(candidate, existing, rtol=0.0, atol=1e-10)
        for existing in starts
    )


def _build_local_multistarts(
    center,
    bounds,
    count,
    alternate_rv=None,
    candidate_points=None,
):
    """Return deterministic interior starts sharing one local interpolator.

    ``candidate_points`` may contain physically motivated starts from an
    external coarse search. Each row may be ``(teff, feh, logg)`` or
    ``(teff, feh, logg, rv_kms)``. Candidate starts are accepted only if they
    are finite, inside the current local-interpolator bounds, and not
    duplicates. This keeps physical multistart bounded by the user-requested
    ``count``.
    """
    count = int(count)
    if count < 1:
        raise ValueError("multistart must be >= 1.")
    center = np.asarray(center, dtype=float)
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    if center.shape != (4,) or lower.shape != (4,) or upper.shape != (4,):
        raise ValueError("center and bounds must contain four parameters.")
    eps = np.maximum(1e-10, (upper - lower) * 1e-8)
    center = np.clip(center, lower + eps, upper - eps)
    starts = [center]
    if alternate_rv is not None and len(starts) < count:
        alternate_rv = float(alternate_rv)
        if not lower[3] <= alternate_rv <= upper[3]:
            raise ValueError("alternate_rv must lie inside the RV bounds.")
        alternate = center.copy()
        alternate[3] = alternate_rv
        if not _start_is_duplicate(alternate, starts):
            starts.append(alternate)
    if candidate_points is not None:
        for candidate in candidate_points:
            if len(starts) >= count:
                break
            candidate = np.asarray(candidate, dtype=float)
            if candidate.shape == (3,):
                candidate = np.r_[candidate, center[3]]
            if candidate.shape != (4,) or not np.all(np.isfinite(candidate)):
                continue
            if np.any(candidate <= lower + eps) or np.any(candidate >= upper - eps):
                continue
            if _start_is_duplicate(candidate, starts):
                continue
            starts.append(candidate)
    for axis in range(3):
        for fraction in (0.25, 0.75):
            if len(starts) >= count:
                break
            candidate = center.copy()
            candidate[axis] = lower[axis] + fraction * (upper[axis] - lower[axis])
            if _start_is_duplicate(candidate, starts):
                continue
            starts.append(candidate)
        if len(starts) >= count:
            break
    patterns = (
        (0.25, 0.75, 0.25),
        (0.75, 0.25, 0.75),
        (0.25, 0.25, 0.75),
        (0.75, 0.75, 0.25),
    )
    for pattern in patterns:
        if len(starts) >= count:
            break
        candidate = center.copy()
        candidate[:3] = lower[:3] + np.asarray(pattern) * (upper[:3] - lower[:3])
        if _start_is_duplicate(candidate, starts):
            continue
        starts.append(candidate)
    while len(starts) < count:
        fraction = len(starts) / float(count + 1)
        candidate = center.copy()
        candidate[:3] = lower[:3] + fraction * (upper[:3] - lower[:3])
        if _start_is_duplicate(candidate, starts):
            candidate = center.copy()
            candidate[:3] = lower[:3] + (fraction + 1e-6) * (
                upper[:3] - lower[:3]
            )
        starts.append(candidate)
    return starts

   
def _solve_multiplicative_legendre(wave, flux, err, model_flux, mdeg):
    """
    Solve for multiplicative Legendre polynomial coefficients c such that:
      flux ≈ model_flux * P(x), with P(x) = V(x) @ c

    This solves the weighted least squares problem in flux space:
      minimize || (flux - model_flux*(V@c)) / err ||^2
    """
    if mdeg < 0:
        raise ValueError("mdeg must be >= 0")

    good = np.isfinite(model_flux) & np.isfinite(flux) & np.isfinite(err) & (err > 0) & (model_flux != 0)
    if np.sum(good) < (mdeg + 1):
        return model_flux, np.r_[1.0, np.zeros(mdeg)]

    w = wave[good]
    f = flux[good]
    e = err[good]
    m = model_flux[good]

    # Map wavelength to [-1, 1] for Legendre basis
    denom = (w.max() - w.min())
    if denom == 0:
        return model_flux, np.r_[1.0, np.zeros(mdeg)]
    x = 2.0 * (w - w.min()) / denom - 1.0

    V = legvander(x, mdeg)  # (N, mdeg+1)

    # Weighted linear system: (m/e)*V @ c ≈ (f/e)
    wgt = 1.0 / e
    A = V * (m * wgt)[:, None]
    b = f * wgt

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Apply polynomial to all points
    denom_all = (wave.max() - wave.min())
    if denom_all == 0:
        poly = np.ones_like(wave)
    else:
        x_all = 2.0 * (wave - wave.min()) / denom_all - 1.0
        V_all = legvander(x_all, mdeg)
        poly = V_all @ coeffs

    return model_flux * poly, coeffs
    

def evaluate_legendre_continuum(wave_eval, wave_ref, coeffs):
    """
    Evaluate a Legendre continuum on wave_eval using the same wavelength
    normalization that was defined by wave_ref during the fit.

    Parameters
    ----------
    wave_eval : array-like
        Wavelength grid where the continuum should be evaluated.
    wave_ref : array-like
        Reference wavelength grid that defined the [-1, 1] normalization used
        when fitting coeffs.
    coeffs : array-like
        Legendre coefficients.

    Returns
    -------
    poly : ndarray
        Continuum multiplicative factor on wave_eval.
    """
    wave_eval = np.asarray(wave_eval, dtype=float)
    wave_ref = np.asarray(wave_ref, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)

    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("coeffs must be a non-empty 1D array.")

    if wave_ref.size == 0:
        return np.ones_like(wave_eval, dtype=float)

    wmin = float(np.min(wave_ref))
    wmax = float(np.max(wave_ref))
    denom = wmax - wmin

    if (not np.isfinite(denom)) or (denom <= 0):
        return np.ones_like(wave_eval, dtype=float) * float(coeffs[0])

    x = 2.0 * (wave_eval - wmin) / denom - 1.0
    V = legvander(x, coeffs.size - 1)
    return V @ coeffs
    

def reconstruct_phoenix_legendre_models_for_segments(
    segments,
    phoenix_lib,
    fit_result,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
    mdeg=2,
    rv_bary_kms=None,
    R=None,
    fwhm_kms=None,
    forward_model=None,
    model_margin_A=None,
):
    """
    Reconstruct per-segment fitted PHOENIX model arrays on the full pixel grid
    of each segment using the standard multiplicative Legendre continuum model.

    This is intended for plotting and fit diagnostics. It mirrors the poly-mode
    behavior of fit_phoenix_full_spectrum, but evaluates the final continuum-
    corrected model on each segment's full wavelength grid rather than only on
    the fit pixels.

    Returns
    -------
    model_full_list : list[ndarray]
        Continuum-corrected model on the full grid of each input segment.
    coeffs_list : list[ndarray]
        Legendre coefficients for each segment.
    used_masks : list[ndarray(bool)]
        Effective fit masks actually used for each segment.
    excluded_masks : list[ndarray(bool)]
        Explicit exclusion masks for plotting diagnostics.
    """
    segments, _segment_weights, _collection_name, _collection_meta = _coerce_segments_input(segments)

    teff = float(fit_result["teff"])
    feh = float(fit_result["feh"])
    logg = float(fit_result["logg"])
    rv_kms = float(fit_result["rv_kms"])
    if rv_bary_kms is None:
        rv_bary_kms = float(fit_result.get("rv_bary_kms", 0.0))

    if forward_model is None:
        forward_model = str(fit_result.get("forward_model", "interp_observed"))
    if model_margin_A is None:
        model_margin_A = float(fit_result.get("model_margin_A", 200.0))

    used_masks = []
    excluded_masks = []
    exclude_mask_by_segment = _resolve_per_segment_mask_options(
        segments,
        exclude_mask,
        "exclude_mask",
        strict=True,
    )
    exclude_masks_by_segment = _resolve_per_segment_mask_options(
        segments,
        exclude_masks,
        "exclude_masks",
        strict=True,
    )
    for index, seg in enumerate(segments):
        reg = _select_segment_option(regions, index, seg)
        ex = _select_segment_option(exclude_regions, index, seg)
        ex_mask = exclude_mask_by_segment[index]
        ex_masks = exclude_masks_by_segment[index]
        used_masks.append(
            build_effective_fit_mask(
                seg,
                regions=reg,
                exclude_regions=ex,
                exclude_mask=ex_mask,
                exclude_masks=ex_masks,
                mask_threshold=mask_threshold,
            )
        )
        excluded_masks.append(
            build_excluded_mask(
                seg,
                regions=reg,
                exclude_regions=ex,
                exclude_mask=ex_mask,
                exclude_masks=ex_masks,
                mask_threshold=mask_threshold,
            )
        )
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=fwhm_kms)
        for seg in segments
    ]
    model_full_list = []
    coeffs_list = []

    if forward_model == "interp_observed":
        support_masks = [_segment_support_ok(seg) for seg in segments]
        support_lengths = [int(np.sum(mask)) for mask in support_masks]
        n_support_total = int(sum(support_lengths))

        model_support_all = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
        if len(model_support_all) != n_support_total:
            raise ValueError(
                "Model grid length does not match total support wavelength grid: "
                "{0} vs {1}".format(len(model_support_all), n_support_total)
            )

        i0 = 0
        for seg, used_mask, seg_fwhm, support_ok in zip(
            segments, used_masks, segment_fwhm_kms, support_masks
        ):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)

            if seg.err is None:
                sigma = _estimate_sigma(flux_full[used_mask] if np.any(used_mask) else flux_full)
                err_full = np.ones_like(flux_full, dtype=float) * sigma
            else:
                err_full = np.asarray(seg.err, dtype=float)

            n_support = int(np.sum(support_ok))
            i1 = i0 + n_support

            model_broad_full = np.full_like(wave_full, np.nan, dtype=float)
            if n_support > 0:
                wave_support = wave_full[support_ok]
                model0_support = model_support_all[i0:i1]
                shifted_support = _apply_observed_grid_rv_shift(
                    wave_support,
                    model0_support,
                    rv_bary_kms + rv_kms,
                )
                model_broad_full[support_ok] = _gaussian_broaden_velocity(
                    wave_support,
                    shifted_support,
                    fwhm_kms=seg_fwhm,
                )
            continuum_pixels = used_mask & np.isfinite(model_broad_full)
            if np.any(continuum_pixels):
                w_used = wave_full[continuum_pixels]
                f_used = flux_full[continuum_pixels]
                e_used = err_full[continuum_pixels]
                m_used = model_broad_full[continuum_pixels]

                _model_corr_used, coeffs = _solve_multiplicative_legendre(
                    w_used, f_used, e_used, m_used, mdeg=mdeg
                )
                poly_full = evaluate_legendre_continuum(wave_full, w_used, coeffs)
                model_full = np.asarray(model_broad_full, dtype=float) * poly_full
            else:
                coeffs = np.r_[1.0, np.zeros(int(mdeg), dtype=float)]
                model_full = np.asarray(model_broad_full, dtype=float)

            model_full_list.append(model_full)
            coeffs_list.append(np.asarray(coeffs, dtype=float))
            i0 = i1

    elif forward_model == "native_interp":
        model_dense = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)

        model_wave_medium = infer_segments_wave_medium(
            segments,
            default=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
        )
        
        model_raw_list = build_phoenix_native_models_for_segments(
            segments=segments,
            phoenix_wave_native=np.asarray(phoenix_lib.wave, dtype=float),
            template_flux_native=model_dense,
            rv_kms=rv_kms,
            rv_bary_kms=rv_bary_kms,
            segment_fwhm_kms=segment_fwhm_kms,
            phoenix_wave_medium=model_wave_medium,
            model_margin_A=model_margin_A,
            bounds_use_fit_mask=True,
            extrapolate=True,
        )
        
        for seg, used_mask, model_broad_full in zip(segments, used_masks, model_raw_list):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)

            if seg.err is None:
                sigma = _estimate_sigma(flux_full[used_mask] if np.any(used_mask) else flux_full)
                err_full = np.ones_like(flux_full, dtype=float) * sigma
            else:
                err_full = np.asarray(seg.err, dtype=float)

            if np.any(used_mask):
                w_used = wave_full[used_mask]
                f_used = flux_full[used_mask]
                e_used = err_full[used_mask]
                m_used = np.asarray(model_broad_full, dtype=float)[used_mask]

                _model_corr_used, coeffs = _solve_multiplicative_legendre(
                    w_used, f_used, e_used, m_used, mdeg=mdeg
                )
                poly_full = evaluate_legendre_continuum(wave_full, w_used, coeffs)
                model_full = np.asarray(model_broad_full, dtype=float) * poly_full
            else:
                coeffs = np.r_[1.0, np.zeros(int(mdeg), dtype=float)]
                model_full = np.asarray(model_broad_full, dtype=float)

            model_full_list.append(model_full)
            coeffs_list.append(np.asarray(coeffs, dtype=float))

    else:
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")

    return model_full_list, coeffs_list, used_masks, excluded_masks
    

def diagnose_phoenix_fixed_params(
    segments,
    phoenix_lib,
    params,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
    mdeg=2,
    rv_bary_kms=0.0,
    R=None,
    fwhm_kms=None,
    forward_model="native_interp",
    model_margin_A=200.0,
):
    """
    Evaluate a PHOENIX model at fixed parameters and return per-segment
    residual diagnostics before and after the multiplicative Legendre continuum.

    This function does not optimize. It is intended for debugging structured
    residuals and comparing candidate parameter sets on exactly the same
    data pixels, masks, broadening, wavelength grid, and continuum model.

    Important
    ---------
    This diagnostic assumes that phoenix_lib has already been built on the
    wavelength grid required by the chosen forward_model. The normal use pattern
    is therefore:

        1. run fit_phoenix_full_spectrum(...)
        2. call diagnose_phoenix_fixed_params(...) with the same segments,
           phoenix_lib, forward_model, model_margin_A, R/fwhm settings, and masks.

    Parameters
    ----------
    segments : SpectrumSegment, SpectrumCollection, or sequence of SpectrumSegment
        Spectrum data to diagnose.

    phoenix_lib : PhoenixLibrary
        PHOENIX library whose interpolator has already been built on the
        correct support grid.

    params : sequence or dict
        Either (teff, feh, logg, rv_kms), or a dict containing keys
        'teff', 'feh', 'logg', and 'rv_kms'.

    Returns
    -------
    result : dict
        Contains per-segment wavelength, flux, error, raw model, continuum-
        corrected model, residuals, continuum coefficients, and chi-square
        summaries.
    """
    segments, segment_weights, collection_name, collection_meta = _coerce_segments_input(segments)

    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")

    if isinstance(params, dict):
        teff = float(params["teff"])
        feh = float(params["feh"])
        logg = float(params["logg"])
        if "rv_kms" in params:
            rv_kms = float(params["rv_kms"])
        else:
            rv_kms = float(params["rv"])
    else:
        teff, feh, logg, rv_kms = map(float, params)

    (
        support_wave_all,
        flux_all,
        err_all,
        support_slices,
        fit_slices,
        fit_masks,
        fit_weights,
        seg_meta,
    ) = _build_data_vectors(
        segments,
        segment_weights=segment_weights,
        regions=regions,
        exclude_regions=exclude_regions,
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
        mask_threshold=mask_threshold,
    )

    if support_wave_all.size == 0 or flux_all.size == 0:
        raise ValueError("No data points selected for fixed-parameter diagnostic.")

    retained_segments = _retained_segments_from_meta(segments, seg_meta)
    forward_segments = _make_forward_segments(
        segments=retained_segments,
        support_wave_all=support_wave_all,
        support_slices=support_slices,
        fit_masks=fit_masks,
    )

    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=fwhm_kms)
        for seg in forward_segments
    ]

    if forward_model == "interp_observed":
        model_wave_grid = support_wave_all

        segment_media = sorted(
            set(str(seg.wave_medium).lower() for seg in retained_segments)
        )
        if len(segment_media) == 1:
            model_wave_medium = segment_media[0]
        else:
            model_wave_medium = None

        if phoenix_lib.wave is None:
            raise RuntimeError("PHOENIX interpolator is not built.")

        if (len(phoenix_lib.wave) != len(model_wave_grid)) or (
            not np.allclose(phoenix_lib.wave, model_wave_grid, rtol=0.0, atol=0.0)
        ):
            raise ValueError(
                "PHOENIX interpolator wavelength grid does not match the "
                "diagnostic support grid. Run fit_phoenix_full_spectrum first "
                "with the same segments and forward_model, or rebuild the "
                "interpolator on this grid."
            )

    else:
        model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
            segments=forward_segments,
            phoenix_lib=phoenix_lib,
            model_margin_A=model_margin_A,
        )

        if phoenix_lib.wave is None:
            raise RuntimeError("PHOENIX interpolator is not built.")

        if (len(phoenix_lib.wave) != len(model_wave_grid)) or (
            not np.allclose(phoenix_lib.wave, model_wave_grid, rtol=0.0, atol=0.0)
        ):
            raise ValueError(
                "PHOENIX interpolator wavelength grid does not match the "
                "native diagnostic grid. Run fit_phoenix_full_spectrum first "
                "with the same segments, forward_model, model_margin_A, and "
                "parameter grid, or rebuild the interpolator on this grid."
            )

    model0 = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)

    if forward_model == "interp_observed":
        rv_tot = float(rv_bary_kms) + float(rv_kms)
        shifted = _apply_observed_grid_rv_shift(support_wave_all, model0, rv_tot)

        model_full_list = []
        for support_sl, seg_fwhm in zip(support_slices, segment_fwhm_kms):
            w_support = support_wave_all[support_sl]
            model_full_list.append(
                _gaussian_broaden_velocity(
                    w_support,
                    shifted[support_sl],
                    fwhm_kms=seg_fwhm,
                )
            )

    else:
        model_full_list = build_phoenix_native_models_for_segments(
            segments=forward_segments,
            phoenix_wave_native=model_wave_grid,
            template_flux_native=model0,
            rv_kms=rv_kms,
            rv_bary_kms=rv_bary_kms,
            segment_fwhm_kms=segment_fwhm_kms,
            phoenix_wave_medium=model_wave_medium,
            model_margin_A=model_margin_A,
            bounds_use_fit_mask=True,
            extrapolate=True,
        )

    segment_results = []
    chi2_raw_total = 0.0
    chi2_corr_total = 0.0
    n_total = 0

    for seg, model_raw_full, fit_sl, fit_mask, seg_weight, seg_fwhm, meta in zip(
        forward_segments,
        model_full_list,
        fit_slices,
        fit_masks,
        fit_weights,
        segment_fwhm_kms,
        seg_meta,
    ):
        wave_full = np.asarray(seg.wave, dtype=float)
        model_raw_full = np.asarray(model_raw_full, dtype=float)

        wave_fit = wave_full[fit_mask]
        model_raw = model_raw_full[fit_mask]
        flux = flux_all[fit_sl]
        err = err_all[fit_sl]

        model_corr, coeffs = _solve_multiplicative_legendre(
            wave_fit,
            flux,
            err,
            model_raw,
            mdeg=mdeg,
        )

        resid_raw = (flux - model_raw) / err
        resid_corr = (flux - model_corr) / err

        chi2_raw = float(np.sum(resid_raw * resid_raw))
        chi2_corr = float(np.sum(resid_corr * resid_corr))
        n = int(resid_corr.size)

        chi2_raw_total += float(seg_weight) * chi2_raw
        chi2_corr_total += float(seg_weight) * chi2_corr
        n_total += n

        segment_results.append(
            {
                "name": meta.get("name"),
                "index": meta.get("index"),
                "weight": float(seg_weight),
                "wave": wave_fit.copy(),
                "flux": flux.copy(),
                "err": err.copy(),
                "model_raw": model_raw.copy(),
                "model_corr": model_corr.copy(),
                "coeffs": np.asarray(coeffs, dtype=float),
                "resid_raw": resid_raw.copy(),
                "resid_corr": resid_corr.copy(),
                "chi2_raw": chi2_raw,
                "chi2_corr": chi2_corr,
                "chi2_raw_weighted": float(seg_weight) * chi2_raw,
                "chi2_corr_weighted": float(seg_weight) * chi2_corr,
                "chi2_red_corr": chi2_corr / max(1, n - (int(mdeg) + 1)),
                "n": n,
                "lsf_fwhm_kms": None if seg_fwhm is None else float(seg_fwhm),
                "resolution_R_effective": None if seg_fwhm is None else float(C_KMS / seg_fwhm),
                "wave_min": float(np.min(wave_fit)) if n else np.nan,
                "wave_max": float(np.max(wave_fit)) if n else np.nan,
                "resid_corr_median": float(np.nanmedian(resid_corr)) if n else np.nan,
                "resid_corr_std": float(np.nanstd(resid_corr)) if n else np.nan,
            }
        )

    n_cont = len(segment_results) * (int(mdeg) + 1)
    dof_effective = max(1, int(n_total) - int(n_cont))

    return {
        "params": {
            "teff": teff,
            "feh": feh,
            "logg": logg,
            "rv_kms": rv_kms,
            "rv_bary_kms": float(rv_bary_kms),
        },
        "forward_model": str(forward_model),
        "model_margin_A": float(model_margin_A),
        "mdeg": int(mdeg),
        "collection_name": collection_name,
        "collection_meta": collection_meta,
        "segments": segment_results,
        "segment_names": [s["name"] for s in segment_results],
        "segment_weights": [float(w) for w in fit_weights],
        "segment_lsf_fwhm_kms": [
            None if x is None else float(x) for x in segment_fwhm_kms
        ],
        "segment_resolution_R_effective": [
            None if x is None else float(C_KMS / x) for x in segment_fwhm_kms
        ],
        "chi2_raw_total": float(chi2_raw_total),
        "chi2_corr_total": float(chi2_corr_total),
        "n_total": int(n_total),
        "n_continuum_params": int(n_cont),
        "dof_effective": int(dof_effective),
        "chi2_red_corr_effective": float(chi2_corr_total / dof_effective),
    }
    

def default_telluric_regions_optical_angstrom():
    """
    Very small default set of strong O2 bands in the optical.
    From molecfit documentation: O2 γ (0.628–0.634 µm), O2 B (0.686–0.695 µm), O2 A (0.759–0.772 µm).
    """
    return [
        (6280.0, 6340.0),  # O2 gamma
        (6860.0, 6950.0),  # O2 B
        (7590.0, 7720.0),  # O2 A (not in your current PEPSI red-009 range, but harmless)
    ]

   
def fit_phoenix_full_spectrum(
    segments,
    phoenix_lib,
    p0,
    bounds=None,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
    exclude_masks=None,
    mask_threshold=0.5,
    mdeg=2,
    rv_bary_kms=0.0,
    R=None,
    fwhm_kms=None,
    forward_model="native_interp",
    model_margin_A=200.0,
    teff_grid=None,
    feh_grid=None,
    logg_grid=None,
    cache_path=None,
    allow_missing=False,
    physical_init=None,
    coarse_teff_grid=None,
    coarse_feh_grid=None,
    coarse_logg_grid=None,
    coarse_decimate=12,
    multistart=1,
    rv_init="grid",
    rv_grid_n=81,
    rv_grid_decimate=5,
    x_scale=None,
    verbose=0,
    max_nfev=200,
    loss="linear",
    loss_f_scale=1.0,
    progress_callback=None,
    ):
    """
    Fit PHOENIX templates to one or more SpectrumSegment objects.

    The nonlinear fit parameters are `(teff, feh, logg, rv_kms)`. At each model
    evaluation, the PHOENIX spectrum is interpolated in parameter space and then
    forwarded to the data using one of two wavelength-space model paths:

    - `forward_model="interp_observed"`:
      interpolate directly on the observed support wavelength grid, then apply the
      PHOENIX RV convention through _apply_observed_grid_rv_shift(), and broaden
      there. This is a legacy/fast compatibility path. It is not the recommended
      scientific path when line profiles are important.

    - `forward_model="native_interp"`:
      interpolate on a dense model-space wavelength grid, then apply the
      standard-sign RV shift, convolve in velocity/log-lambda space, and resample
      last to each segment support grid. This is the recommended scientific path
      for PHOENIX line-profile fitting.

    In both cases, the model is multiplied by a per-segment Legendre polynomial
    continuum solved analytically by weighted least squares.

    Parameters
    ----------
    segments : SpectrumSegment, SpectrumCollection, or sequence of SpectrumSegment
        Input spectrum segments to fit. A SpectrumCollection may also carry
        per-segment weights used in the joint objective.

    phoenix_lib : PhoenixLibrary
        PHOENIX template library interface from `Spyctres.phoenix`, pointing to
        a local PHOENIX installation.

    p0 : tuple
        Initial guess `(teff, feh, logg, rv_kms)`.

    bounds : tuple, optional
        Parameter bounds as
        `((teff_min, feh_min, logg_min, rv_min), (teff_max, feh_max, logg_max, rv_max))`.
        If None, defaults to the requested PHOENIX subgrid bounds.

    regions : None, list[tuple], or dict, optional
        Inclusion regions in wavelength. May be:
        - None: use all wavelengths
        - list of `(wmin, wmax)` tuples applied to all segments
        - dict mapping segment index or `seg.name` to a list of `(wmin, wmax)`

    exclude_regions : None, list[tuple], or dict, optional
        Exclusion regions in wavelength, with the same format as `regions`.

    exclude_mask : callable, optional
        Callable applied to each segment wavelength array. Points where the
        returned mask is True are excluded. Non-boolean outputs are converted to
        boolean using a threshold (`> 0.5`), which is useful for Spyctres
        telluric masks. May also be a dict keyed by segment index or name.

    exclude_masks : sequence or dict, optional
        Preferred multi-mask API. A sequence of named masks applies globally,
        for example ``[("telluric", fn), ("line_core", fn)]``. A dict may map
        segment index or segment name to one or more named masks.

    mask_threshold : float, optional
        Numeric threshold used when converting non-boolean exclusion-mask
        outputs. Segment masks always use ``True == valid/use``; exclusion
        callables always use ``True == reject`` after thresholding.

    mdeg : int, optional
        Degree of the multiplicative Legendre polynomial solved independently
        for each segment. `mdeg=0` corresponds to a constant multiplicative
        scale.

    rv_bary_kms : float, optional
        Fixed barycentric velocity term in km/s added to the fitted `rv_kms`.
        It must use the same standard sign convention as `rv_kms`: positive values
        redshift the template/model.

    RV sign convention
    ------------------
    The returned rv_kms follows the standard astronomical convention: positive
    rv_kms redshifts the template/model spectrum. The legacy
    Spyctres.velocity_correction helper is not modified; the observed-grid
    compatibility branch wraps it internally so that PHOENIX fitting results use
    this convention consistently with native_interp.
    
    R : float, optional
        Resolving power of the Gaussian instrumental line-spread function,
        defined as `R = lambda / Delta_lambda_FWHM`. If provided, this is
        converted to a constant velocity FWHM and applied after Doppler shifting
        and before continuum fitting.

    fwhm_kms : float, optional
        Gaussian instrumental FWHM in km/s. Alternative to `R`. Exactly one of
        `R` or `fwhm_kms` may be provided.

    forward_model : {"interp_observed", "native_interp"}, optional
        Choice of wavelength-space forward-model path. `native_interp` is the
        scientific default and uses the validated shift/convolve/resample-last
        sequence. `interp_observed` remains available as an explicit legacy/fast
        compatibility path.

    model_margin_A : float, optional
        Wavelength margin in Angstrom used by `forward_model="native_interp"`
        when preparing the dense model-space wavelength grid.
        
    teff_grid, feh_grid, logg_grid : array-like, optional
        PHOENIX parameter grids to use when building the interpolator. If not
        provided, defaults are chosen by the caller or PHOENIX helper logic.

    cache_path : str, optional
        Path to an `.npz` cache file for the PHOENIX interpolator built on the
        current model wavelength grid. For `interp_observed` this is the
        observed support grid; for `native_interp` it is the dense model-space
        wavelength grid.

    allow_missing : bool, optional
        If True, allow missing PHOENIX templates when building the interpolator.
        Missing grid points are filled with NaNs and may degrade interpolation.

    physical_init : {None, "coarse"}, optional
        If ``"coarse"``, score a sparse set of widely separated installed
        PHOENIX nodes before constructing the local interpolator. This removes
        dependence on the physical part of ``p0`` without allocating a dense
        full-library cache. When ``multistart > 1``, the best compatible
        coarse candidates are promoted into the local optimizer start queue.
        Currently available for ``native_interp`` only.

    coarse_teff_grid, coarse_feh_grid, coarse_logg_grid : array-like, optional
        Sparse PHOENIX node axes used by ``physical_init="coarse"``. Compact
        survey-inspired defaults are used when omitted.

    coarse_decimate : int, optional
        Pixel decimation used only while ranking coarse physical nodes.

    multistart : int, optional
        Number of deterministic local least-squares starts. All starts share
        the same local PHOENIX interpolator; ``1`` preserves historical
        behavior. A stellar-rest input always adds/tests an explicit zero-RV
        start when the selected start has nonzero RV. If coarse physical
        initialization is enabled, ranked coarse candidates are preferred
        before generic deterministic perturbations.

    rv_init : {"grid", None}, optional
        Strategy for initializing the radial velocity:
        - `"grid"`: perform a coarse RV scan and use the best value to seed the fit
        - `None`: use the RV value from `p0` directly

    rv_grid_n : int, optional
        Number of trial RV points in the coarse initialization grid when
        `rv_init="grid"`.

    rv_grid_decimate : int, optional
        Decimation factor used during the coarse RV scan to accelerate the
        initialization step.

    x_scale : array-like or str, optional
        Passed to `scipy.optimize.least_squares` as the parameter scaling.

    verbose : int, optional
        Verbosity level passed to the optimizer.

    max_nfev : int, optional
        Maximum number of function evaluations for the nonlinear optimizer.

    loss : {"linear", "soft_l1", "huber", "cauchy", "arctan"}, optional
        Robust loss passed to ``scipy.optimize.least_squares``. The default
        ``"linear"`` preserves ordinary least squares and historical behavior.
        ``"soft_l1"`` or ``"cauchy"`` can be useful for exploratory fits with
        outliers that are not yet captured by masks.

    loss_f_scale : float, optional
        Positive scale parameter passed to ``least_squares`` as ``f_scale``.
        Residuals are already normalized by uncertainty, so values near 1 are
        the natural starting point.

    progress_callback : callable, optional
        Called with short status strings before long operations such as cache
        load/rebuild, RV grid scan, and local optimizer starts/finishes. For
        command-line use, pass ``lambda msg: print(msg, flush=True)``.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - `p_best`: best-fit parameter array `[teff, feh, logg, rv_kms]`
        - `teff`, `feh`, `logg`, `rv_kms`: best-fit parameters
        - `chi2`, `chi2_red`: chi-square and reduced chi-square
        - `success`, `status`, `message`: optimizer status information
        - `resolution_R`: resolving power used for instrumental broadening, if any
        - `lsf_fwhm_kms`: Gaussian LSF FWHM in km/s, if any
        - `segment_names`, `segment_weights`, `collection_name`, `collection_meta`
    """
    def report(message):
        if progress_callback is not None:
            progress_callback(str(message))

    segments, segment_weights, collection_name, collection_meta = _coerce_segments_input(segments)
        
    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")

    p0 = np.asarray(p0, dtype=float)
    if p0.shape != (4,) or not np.all(np.isfinite(p0)):
        raise ValueError("p0 must contain four finite values: teff, feh, logg, rv_kms.")

    mdeg = int(mdeg)
    if mdeg < 0:
        raise ValueError("mdeg must be >= 0.")
    coarse_decimate = int(coarse_decimate)
    if coarse_decimate < 1:
        raise ValueError("coarse_decimate must be >= 1.")
    multistart = int(multistart)
    if multistart < 1:
        raise ValueError("multistart must be >= 1.")
    rv_grid_n = int(rv_grid_n)
    if rv_grid_n < 2:
        raise ValueError("rv_grid_n must be >= 2.")
    rv_grid_decimate = int(rv_grid_decimate)
    if rv_grid_decimate < 1:
        raise ValueError("rv_grid_decimate must be >= 1.")
    if max_nfev is None:
        max_nfev = 200
    max_nfev = int(max_nfev)
    if max_nfev < 1:
        raise ValueError("max_nfev must be >= 1.")
    loss, loss_f_scale = _validate_optimizer_loss(loss, loss_f_scale)
    rv_bary_kms = float(rv_bary_kms)
    if not np.isfinite(rv_bary_kms):
        raise ValueError("rv_bary_kms must be finite.")
            
    (
        support_wave_all,
        flux_all,
        err_all,
        support_slices,
        fit_slices,
        fit_masks,
        fit_weights,
        seg_meta,
    ) = _build_data_vectors(
        segments,
        segment_weights=segment_weights,
        regions=regions,
        exclude_regions=exclude_regions,
        exclude_mask=exclude_mask,
        exclude_masks=exclude_masks,
        mask_threshold=mask_threshold,
    )
    if support_wave_all.size == 0 or flux_all.size == 0:
        raise ValueError("No data points selected for fitting.")
    report(
        "Prepared fit data: {0} retained segment(s), {1} fitted pixel(s).".format(
            len(seg_meta),
            int(flux_all.size),
        )
    )

    retained_segments = _retained_segments_from_meta(segments, seg_meta)
    forward_segments = _make_forward_segments(
        segments=retained_segments,
        support_wave_all=support_wave_all,
        support_slices=support_slices,
        fit_masks=fit_masks,
    )
    
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=fwhm_kms)
        for seg in forward_segments
    ]
    
    teff0, feh0, logg0, rv0 = map(float, p0)
    if forward_model == "interp_observed":
        model_wave_grid = support_wave_all

        segment_media = sorted(
            set(str(seg.wave_medium).lower() for seg in retained_segments)
        )
        if len(segment_media) == 1:
            model_wave_medium = segment_media[0]
        else:
            model_wave_medium = None
    else:
        report("Building native PHOENIX interpolation wavelength grid.")
        model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
            segments=forward_segments,
            phoenix_lib=phoenix_lib,
            model_margin_A=model_margin_A,
        )

    if physical_init not in (None, "coarse"):
        raise ValueError("physical_init must be 'coarse' or None.")
    coarse_initialization = None
    if physical_init == "coarse":
        if forward_model != "native_interp":
            raise ValueError("physical_init='coarse' requires forward_model='native_interp'.")
        report("Running coarse physical-parameter initialization.")
        coarse_best, coarse_scores = _coarse_physical_start_native_interp(
            forward_segments=forward_segments,
            flux_all=flux_all,
            err_all=err_all,
            fit_slices=fit_slices,
            fit_masks=fit_masks,
            segment_weights=fit_weights,
            phoenix_lib=phoenix_lib,
            model_wave_grid=model_wave_grid,
            model_wave_medium=model_wave_medium,
            rv_tot=rv_bary_kms + rv0,
            mdeg=mdeg,
            segment_fwhm_kms=segment_fwhm_kms,
            model_margin_A=model_margin_A,
            teff_grid=coarse_teff_grid,
            feh_grid=coarse_feh_grid,
            logg_grid=coarse_logg_grid,
            decimate=coarse_decimate,
        )
        teff0 = coarse_best["teff"]
        feh0 = coarse_best["feh"]
        logg0 = coarse_best["logg"]
        coarse_initialization = {
            "best": dict(coarse_best),
            "candidates_evaluated": int(len(coarse_scores)),
            "candidates": [dict(item) for item in coarse_scores],
            "candidates_complete": True,
            "top_candidates": [dict(item) for item in coarse_scores[:5]],
            "decimate": int(coarse_decimate),
        }
        if verbose:
            print("Coarse physical init best:", coarse_best)
        report(
            "Coarse physical initialization selected Teff={0:g}, [Fe/H]={1:g}, logg={2:g}.".format(
                teff0,
                feh0,
                logg0,
            )
        )

    # Materialize one local interpolation grid around either p0 or the best
    # sparse-grid node. Every local multistart below reuses this same grid.
    if teff_grid is None:
        teff_grid_req = _pick_subgrid(
            phoenix_lib.DEFAULT_TEFF_GRID, teff0, half_width=800.0, n_min=5, n_max=9
        )
    else:
        teff_grid_req = np.asarray(teff_grid, dtype=float)

    if feh_grid is None:
        feh_grid_req = _pick_subgrid(
            phoenix_lib.DEFAULT_FEH_GRID, feh0, half_width=0.75, n_min=3, n_max=5
        )
    else:
        feh_grid_req = np.asarray(feh_grid, dtype=float)

    if logg_grid is None:
        logg_grid_req = _pick_subgrid(
            phoenix_lib.DEFAULT_LOGG_GRID, logg0, half_width=0.75, n_min=3, n_max=5
        )
    else:
        logg_grid_req = np.asarray(logg_grid, dtype=float)

    if not phoenix_lib.interpolator_matches(
        model_wave_grid,
        teff_grid_req,
        feh_grid_req,
        logg_grid_req,
        observed_wave_medium=model_wave_medium,
    ):
        report("Preparing PHOENIX interpolator/cache.")
        phoenix_lib.build_interpolator(
            observed_wave=model_wave_grid,
            teff_grid=teff_grid_req,
            feh_grid=feh_grid_req,
            logg_grid=logg_grid_req,
            cache_path=cache_path,
            allow_missing=allow_missing,
            observed_wave_medium=model_wave_medium,
            progress_callback=progress_callback,
        )
    else:
        report("Reusing existing in-memory PHOENIX interpolator.")
    
    # Set default bounds from the interpolator grid if none supplied
    if bounds is None:
        bounds = (
            (
                float(np.min(teff_grid_req)),
                float(np.min(feh_grid_req)),
                float(np.min(logg_grid_req)),
                -300.0,
            ),
            (
                float(np.max(teff_grid_req)),
                float(np.max(feh_grid_req)),
                float(np.max(logg_grid_req)),
                +300.0,
            ),
        )
    lower_bounds = np.asarray(bounds[0], dtype=float)
    upper_bounds = np.asarray(bounds[1], dtype=float)
    if (
        lower_bounds.shape != (4,)
        or upper_bounds.shape != (4,)
        or not np.all(np.isfinite(lower_bounds))
        or not np.all(np.isfinite(upper_bounds))
        or np.any(upper_bounds <= lower_bounds)
    ):
        raise ValueError(
            "bounds must be two finite four-element arrays with lower < upper."
        )
    bounds = (lower_bounds, upper_bounds)
        
    broadening_fwhm_kms = _resolve_broadening_fwhm_kms(R=R, fwhm_kms=fwhm_kms)
    
    invalid_model_evaluations = {"count": 0}

    def residuals(p):
        teff, feh, logg, rv_kms = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        rv_tot = rv_bary_kms + rv_kms

        try:
            model0 = phoenix_lib.evaluate(teff, feh, logg)
        except ValueError:
            invalid_model_evaluations["count"] += 1
            return np.ones_like(flux_all) * 1e6

        out = np.empty_like(flux_all)

        if forward_model == "interp_observed":
            shifted = _apply_observed_grid_rv_shift(support_wave_all, model0, rv_tot)
            
            for support_sl, fit_sl, fit_mask, seg_weight, seg_fwhm in zip(
                support_slices, fit_slices, fit_masks, fit_weights, segment_fwhm_kms
            ):
                w_support = support_wave_all[support_sl]
                m_support = _gaussian_broaden_velocity(
                    w_support,
                    shifted[support_sl],
                    fwhm_kms=seg_fwhm,
                )
                
                w = w_support[fit_mask]
                f = flux_all[fit_sl]
                e = err_all[fit_sl]
                m = m_support[fit_mask]
                
                m_corr, coeffs = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
                out[fit_sl] = np.sqrt(float(seg_weight)) * (f - m_corr) / e
        else:
            model_list = build_phoenix_native_models_for_segments(
                segments=forward_segments,
                phoenix_wave_native=model_wave_grid,
                template_flux_native=model0,
                rv_kms=rv_kms,
                rv_bary_kms=rv_bary_kms,
                segment_fwhm_kms=segment_fwhm_kms,
                phoenix_wave_medium=model_wave_medium,
                model_margin_A=model_margin_A,
                bounds_use_fit_mask=True,
                extrapolate=True,
            )
            
            for seg, model_full, fit_sl, fit_mask, seg_weight in zip(
                forward_segments, model_list, fit_slices, fit_masks, fit_weights
            ):
                w_support = np.asarray(seg.wave, dtype=float)

                w = w_support[fit_mask]
                f = flux_all[fit_sl]
                e = err_all[fit_sl]
                m = np.asarray(model_full, dtype=float)[fit_mask]

                m_corr, coeffs = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
                out[fit_sl] = np.sqrt(float(seg_weight)) * (f - m_corr) / e

        return out
    
    # RV initialization by coarse grid scan (optional)
    if rv_init == "grid":
        rv_lo, rv_hi = float(bounds[0][3]), float(bounds[1][3])
        rv_grid = np.linspace(rv_lo, rv_hi, int(rv_grid_n))
        report(
            "Running coarse RV grid scan with {0} trial velocities.".format(
                int(rv_grid.size)
            )
        )
        
        chi2s = np.empty(rv_grid.size, dtype=float)
        for j, rv in enumerate(rv_grid):
            if forward_model == "interp_observed":
                chi2s[j] = _chi2_for_params(
                    support_wave_all,
                    flux_all,
                    err_all,
                    support_slices,
                    fit_slices,
                    fit_masks,
                    fit_weights,
                    teff0,
                    feh0,
                    logg0,
                    rv_bary_kms + float(rv),
                    phoenix_lib,
                    mdeg=mdeg,
                    decimate=rv_grid_decimate,
                    segment_fwhm_kms=segment_fwhm_kms,
                )
            else:
                chi2s[j] = _chi2_for_params_native_interp(
                    forward_segments=forward_segments,
                    flux_all=flux_all,
                    err_all=err_all,
                    fit_slices=fit_slices,
                    fit_masks=fit_masks,
                    segment_weights=fit_weights,
                    teff=teff0,
                    feh=feh0,
                    logg=logg0,
                    rv_tot=rv_bary_kms + float(rv),
                    phoenix_lib=phoenix_lib,
                    model_wave_grid=model_wave_grid,
                    model_wave_medium=model_wave_medium,
                    mdeg=mdeg,
                    decimate=rv_grid_decimate,
                    segment_fwhm_kms=segment_fwhm_kms,
                    model_margin_A=model_margin_A,
                )
        rv0_best = float(rv_grid[int(np.argmin(chi2s))])
        if verbose:
            print("RV init grid best:", rv0_best)
        report("Coarse RV grid scan selected rv_kms={0:.6g}.".format(rv0_best))
        p0 = (teff0, feh0, logg0, rv0_best)
    elif rv_init is None:
        report("Skipping coarse RV grid scan; using supplied initial rv_kms.")
        p0 = (teff0, feh0, logg0, rv0)
    else:
        raise ValueError("rv_init must be 'grid' or None.")
    
    if x_scale is None:
        x_scale = np.array([100.0, 0.1, 0.1, 10.0], dtype=float)

    stellar_rest_input = bool(forward_segments) and all(
        getattr(segment, "stellar_rest_status", "unknown") == "corrected"
        for segment in forward_segments
    )
    alternate_rv = 0.0 if stellar_rest_input else (rv0 if rv_init == "grid" else None)
    effective_multistart = int(multistart)
    if (
        stellar_rest_input
        and not np.isclose(float(p0[3]), 0.0, rtol=0.0, atol=1e-12)
    ):
        effective_multistart = max(2, effective_multistart)
    coarse_candidate_points = None
    if coarse_initialization is not None and effective_multistart > 1:
        coarse_candidate_points = [
            (
                float(candidate["teff"]),
                float(candidate["feh"]),
                float(candidate["logg"]),
                float(p0[3]),
            )
            for candidate in coarse_initialization.get("candidates", [])[
                : max(1, effective_multistart * 3)
            ]
        ]
        if coarse_candidate_points:
            report(
                "Adding top coarse physical candidates to the local multistart queue."
            )
    starts = _build_local_multistarts(
        p0,
        bounds,
        effective_multistart,
        alternate_rv=alternate_rv,
        candidate_points=coarse_candidate_points,
    )
    local_results = []
    for start_index, start in enumerate(starts):
        report(
            "Starting local optimizer {0}/{1}: p0=({2:g}, {3:g}, {4:g}, {5:g}).".format(
                start_index + 1,
                len(starts),
                float(start[0]),
                float(start[1]),
                float(start[2]),
                float(start[3]),
            )
        )
        candidate_result = least_squares(
            residuals,
            x0=np.asarray(start, dtype=float),
            bounds=bounds,
            method="trf",
            x_scale=x_scale,
            max_nfev=int(max_nfev),
            loss=loss,
            f_scale=loss_f_scale,
            verbose=2 if verbose else 0,
        )
        local_results.append(candidate_result)
        candidate_chi2 = float(np.sum(candidate_result.fun * candidate_result.fun))
        report(
            "Finished local optimizer {0}/{1}: chi2={2:.6g}, success={3}.".format(
                start_index + 1,
                len(starts),
                candidate_chi2,
                bool(candidate_result.success),
            )
        )
        if verbose and len(starts) > 1:
            print(
                "Local start {0}/{1}: chi2={2:.6g}".format(
                    start_index + 1,
                    len(starts),
                    candidate_chi2,
                )
            )
    res = min(local_results, key=lambda item: float(np.sum(item.fun * item.fun)))
    multistart_diagnostics = []
    for start, item in zip(starts, local_results):
        multistart_diagnostics.append(
            {
                "start": np.asarray(start, dtype=float).tolist(),
                "solution": np.asarray(item.x, dtype=float).tolist(),
                "chi2": float(np.sum(item.fun * item.fun)),
                "success": bool(item.success),
                "status": int(item.status),
                "nfev": int(item.nfev),
            }
        )

    # Compute diagnostics. If segment weights are used, chi2 is the weighted
    # sum of squared normalized residuals.
    r = res.fun
    chi2 = float(np.sum(r * r))
    n = int(r.size)
    # Analytically solved continuum coefficients still consume degrees of freedom.
    k = _full_spectrum_parameter_count(len(forward_segments), mdeg)
    dof = max(1, n - k)
    chi2_red = chi2 / dof

    covariance = None
    parameter_errors = None
    try:
        covariance = np.linalg.pinv(res.jac.T @ res.jac) * chi2_red
        parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    except (ValueError, np.linalg.LinAlgError):
        covariance = None
        parameter_errors = None

    diagnostics = _build_phoenix_fit_diagnostics(
        residuals=r,
        chi2=chi2,
        chi2_red=chi2_red,
        dof=dof,
        n_parameters=k,
        input_segments=segments,
        forward_segments=forward_segments,
        seg_meta=seg_meta,
        mdeg=mdeg,
        best_parameters=res.x,
        phoenix_lib=phoenix_lib,
        segment_fwhm_kms=segment_fwhm_kms,
        local_solutions=multistart_diagnostics,
        coarse_initialization=coarse_initialization,
        rv_kms=float(res.x[3]),
        rv_bary_kms=rv_bary_kms,
    )
    quality_flags = _phoenix_quality_flags(diagnostics, success=res.success)
    velocity_convention = diagnostics["velocity_convention"]

    summary = {
        "success": bool(res.success),
        "message": res.message,
        "p_best": res.x,
        "teff": float(res.x[0]),
        "feh": float(res.x[1]),
        "logg": float(res.x[2]),
        "rv_kms": float(res.x[3]),
        "rv_bary_kms": float(rv_bary_kms),
        "rv_kms_fit": velocity_convention["rv_kms_fit"],
        "rv_bary_kms_input": velocity_convention["rv_bary_kms_input"],
        "total_model_shift_kms": velocity_convention["total_model_shift_kms"],
        "rv_sign_convention": velocity_convention["rv_sign_convention"],
        "rv_combination_formula": velocity_convention["rv_combination_formula"],
        "rv_bary_applied_to_model": velocity_convention["rv_bary_applied_to_model"],
        "rv_bary_term_in_model_formula": velocity_convention[
            "rv_bary_term_in_model_formula"
        ],
        "rv_bary_applied_to_data": velocity_convention["rv_bary_applied_to_data"],
        "wavelength_frame_assumption": velocity_convention[
            "wavelength_frame_assumption"
        ],
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2_red,
        "n_points": n,
        "status": int(res.status),
        "nfev": int(res.nfev),
        "optimizer_loss": str(loss),
        "optimizer_loss_f_scale": float(loss_f_scale),
        "invalid_model_evaluations": int(invalid_model_evaluations["count"]),
        "physical_initialization": physical_init,
        "coarse_initialization": coarse_initialization,
        "multistart": int(len(starts)),
        "multistart_requested": int(multistart),
        "stellar_rest_zero_rv_start_tested": bool(
            stellar_rest_input
            and any(np.isclose(start[3], 0.0, rtol=0.0, atol=1e-12) for start in starts)
        ),
        "multistart_diagnostics": multistart_diagnostics,
        "n_free_parameters": int(k),
        "covariance": covariance,
        "parameter_errors": parameter_errors,
        "seg_meta": seg_meta,
        "forward_model": str(forward_model),
        "model_margin_A": float(model_margin_A),
        "n_segments": int(len(seg_meta)),
        "segment_names": [m.get("name") for m in seg_meta],
        "segment_weights": [float(w) for w in fit_weights],
        "collection_name": collection_name,
        "collection_meta": collection_meta,
        "segment_lsf_fwhm_kms": [
            None if x is None else float(x) for x in segment_fwhm_kms
        ],
        "segment_resolution_R_effective": [
            None if x is None else float(C_KMS / x) for x in segment_fwhm_kms
        ],
        # Backward-compatible global broadening metadata.
        # The actual per-segment broadening used in the fit is stored above in
        # segment_lsf_fwhm_kms and segment_resolution_R_effective.
        "resolution_R": None if R is None else float(R),
        "lsf_fwhm_kms": None if broadening_fwhm_kms is None else float(broadening_fwhm_kms),
        "diagnostics": diagnostics,
        "quality_flags": quality_flags,
        # Note: did not store poly coeffs in this minimal version to avoid re-evaluating.
    }
    summary["quality_report"] = build_fit_quality_report(summary)
    report(
        "Selected best fit: Teff={0:.6g}, [Fe/H]={1:.6g}, logg={2:.6g}, rv_kms={3:.6g}, chi2_red={4:.6g}.".format(
            summary["teff"],
            summary["feh"],
            summary["logg"],
            summary["rv_kms"],
            summary["chi2_red"],
        )
    )
    return summary
