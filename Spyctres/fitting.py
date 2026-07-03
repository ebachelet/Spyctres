# Spyctres/fitting.py
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import legvander

# Observed-grid PHOENIX RV resampling is implemented in waveutils.py so that
# this fitting layer does not import the legacy monolithic Spyctres.py module.
from .io import SpectrumSegment, SpectrumCollection
from .phoenix_forward import (
    infer_segments_wave_medium,
    fit_bounds_from_segments,
    prepare_phoenix_native_template,
    build_phoenix_native_models_for_segments,
    build_native_interp_wave_grid_for_segments,
    convolve_to_resolution_loglam,
    resolve_gaussian_lsf_fwhm_kms,
)
# Why multiplicative polynomial: it is a standard way to absorb low-frequency continuum differences and calibration mismatches during full-spectrum fitting.

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
    1. seg.meta["lsf_fwhm_kms"]
    2. seg.meta["fwhm_kms"]
    3. seg.meta["resolution_R"]
    4. global fwhm_kms
    5. global R
    6. None
    """
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


def _to_bool_mask(x, threshold=0.5):
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    return a > threshold
    

def _select_region(wave, regions):
    """Return boolean mask selecting points inside any (wmin,wmax) in regions."""
    if regions is None:
        return np.ones_like(wave, dtype=bool)
    m = np.zeros_like(wave, dtype=bool)
    for (wmin, wmax) in regions:
        m |= (wave >= wmin) & (wave <= wmax)
    return m


def _exclude_region(wave, exclude_regions):
    """Return boolean mask True for points NOT in any excluded interval."""
    if exclude_regions is None:
        return np.ones_like(wave, dtype=bool)
    m = np.ones_like(wave, dtype=bool)
    for (wmin, wmax) in exclude_regions:
        m &= ~((wave >= wmin) & (wave <= wmax))
    return m
    

def build_effective_fit_mask(seg, regions=None, exclude_regions=None, exclude_mask=None):
    """
    Build the effective boolean fit mask for a single SpectrumSegment.

    This reproduces the point-selection logic used by fit_phoenix_full_spectrum
    for a single segment, except that if seg.err is None it does not invent a
    synthetic error array. In that case only finite wave/flux and the supplied
    mask/region logic are applied.
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)

    m = np.asarray(seg.mask, dtype=bool)
    m &= np.isfinite(wave) & np.isfinite(flux)

    if seg.err is not None:
        err = np.asarray(seg.err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    m &= _select_region(wave, regions)
    m &= _exclude_region(wave, exclude_regions)

    if exclude_mask is not None:
        m &= ~_to_bool_mask(exclude_mask(wave))

    return m


def build_excluded_mask(seg, regions=None, exclude_regions=None, exclude_mask=None):
    """
    Build a boolean mask of pixels explicitly excluded by region/exclude rules.

    This is intended for plotting diagnostics. It does not mark pixels excluded
    only because they are NaN, have non-positive errors, or lie outside seg.mask.
    """
    wave = np.asarray(seg.wave, dtype=float)
    m = np.zeros_like(wave, dtype=bool)

    if regions is not None:
        m |= ~_select_region(wave, regions)

    if exclude_regions is not None:
        m |= ~_exclude_region(wave, exclude_regions)

    if exclude_mask is not None:
        m |= _to_bool_mask(exclude_mask(wave))

    return m


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


def _build_data_vectors(
    segments,
    segment_weights=None,
    regions=None,
    exclude_regions=None,
    exclude_mask=None,
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

    start_support = 0
    start_fit = 0

    for i, (seg, seg_weight) in enumerate(zip(segments, segment_weights)):
        w_full = np.asarray(seg.wave, dtype=float)
        f_full = np.asarray(seg.flux, dtype=float)

        support_ok = np.isfinite(w_full) & np.isfinite(f_full)

        if seg.err is None:
            e_full = np.ones_like(f_full) * _estimate_sigma(
                f_full[support_ok] if np.any(support_ok) else f_full
            )
            err_ok = np.isfinite(e_full) & (e_full > 0)
        else:
            e_full = np.asarray(seg.err, dtype=float)
            err_ok = np.isfinite(e_full) & (e_full > 0)

        support_ok &= err_ok

        if isinstance(regions, dict):
            reg = regions.get(i, regions.get(seg.name, None))
        else:
            reg = regions

        if isinstance(exclude_regions, dict):
            ex = exclude_regions.get(i, exclude_regions.get(seg.name, None))
        else:
            ex = exclude_regions

        fit_m = build_effective_fit_mask(
            seg,
            regions=reg,
            exclude_regions=ex,
            exclude_mask=exclude_mask,
        )
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
            )
        )
    return out


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
    exclude_mask=None,
    mdeg=2,
    rv_bary_kms=0.0,
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

    if forward_model is None:
        forward_model = str(fit_result.get("forward_model", "interp_observed"))
    if model_margin_A is None:
        model_margin_A = float(fit_result.get("model_margin_A", 200.0))

    used_masks = [build_effective_fit_mask(seg, exclude_mask=exclude_mask) for seg in segments]
    excluded_masks = [build_excluded_mask(seg, exclude_mask=exclude_mask) for seg in segments]
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=fwhm_kms)
        for seg in segments
    ]
    model_full_list = []
    coeffs_list = []

    if forward_model == "interp_observed":
        support_lengths = [len(seg.wave) for seg in segments]
        n_support_total = int(sum(support_lengths))

        model_support_all = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
        if len(model_support_all) != n_support_total:
            raise ValueError(
                "Model grid length does not match total support wavelength grid: "
                "{0} vs {1}".format(len(model_support_all), n_support_total)
            )

        i0 = 0
        for seg, used_mask, seg_fwhm in zip(segments, used_masks, segment_fwhm_kms):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)

            if seg.err is None:
                sigma = _estimate_sigma(flux_full[used_mask] if np.any(used_mask) else flux_full)
                err_full = np.ones_like(flux_full, dtype=float) * sigma
            else:
                err_full = np.asarray(seg.err, dtype=float)

            n_support = len(wave_full)
            i1 = i0 + n_support

            model0_full = model_support_all[i0:i1]
            shifted_full = _apply_observed_grid_rv_shift(
                wave_full,
                model0_full,
                rv_bary_kms + rv_kms,
            )
            model_broad_full = _gaussian_broaden_velocity(
                wave_full,
                shifted_full,
                fwhm_kms=seg_fwhm,
            )
            if np.any(used_mask):
                w_used = wave_full[used_mask]
                f_used = flux_full[used_mask]
                e_used = err_full[used_mask]
                m_used = model_broad_full[used_mask]

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
    )

    if support_wave_all.size == 0 or flux_all.size == 0:
        raise ValueError("No data points selected for fixed-parameter diagnostic.")

    forward_segments = _make_forward_segments(
        segments=segments,
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

        segment_media = sorted(set(str(seg.wave_medium).lower() for seg in segments))
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
    mdeg=2,
    rv_bary_kms=0.0,
    R=None,
    fwhm_kms=None,
    forward_model="interp_observed",
    model_margin_A=200.0,
    teff_grid=None,
    feh_grid=None,
    logg_grid=None,
    cache_path=None,
    allow_missing=False,
    rv_init="grid",
    rv_grid_n=81,
    rv_grid_decimate=5,
    x_scale=None,
    verbose=0,
    max_nfev=200,
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
        telluric masks.

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
        Choice of wavelength-space forward-model path. The default preserves the
        original observed-grid behavior. The `native_interp` mode keeps the fit
        continuous in `(teff, feh, logg)` but uses the native-grid-inspired
        shift/convolve/resample-last sequence validated against the X-SHOOTER
        PHOENIX notebook reference.

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
    segments, segment_weights, collection_name, collection_meta = _coerce_segments_input(segments)
        
    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")
            
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
    )
    if support_wave_all.size == 0 or flux_all.size == 0:
        raise ValueError("No data points selected for fitting.")
    
    forward_segments = _make_forward_segments(
        segments=segments,
        support_wave_all=support_wave_all,
        support_slices=support_slices,
        fit_masks=fit_masks,
    )
    
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=fwhm_kms)
        for seg in forward_segments
    ]
    
    teff0, feh0, logg0, rv0 = map(float, p0)
    # Materialize the requested PHOENIX subgrid before deciding whether the
    # current interpolator can be reused.
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
    
    if forward_model == "interp_observed":
        model_wave_grid = support_wave_all

        segment_media = sorted(set(str(seg.wave_medium).lower() for seg in segments))
        if len(segment_media) == 1:
            model_wave_medium = segment_media[0]
        else:
            model_wave_medium = None
    else:
        model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
            segments=forward_segments,
            phoenix_lib=phoenix_lib,
            model_margin_A=model_margin_A,
        )

    need_rebuild = False

    if phoenix_lib.wave is None:
        need_rebuild = True
    elif (len(phoenix_lib.wave) != len(model_wave_grid)) or (
        not np.allclose(phoenix_lib.wave, model_wave_grid, rtol=0.0, atol=0.0)
    ):
        need_rebuild = True
    elif phoenix_lib._grid is None:
        need_rebuild = True
    else:
        tg, zg, gg = phoenix_lib._grid
        if (
            (len(tg) != len(teff_grid_req)) or
            (len(zg) != len(feh_grid_req)) or
            (len(gg) != len(logg_grid_req)) or
            (not np.allclose(tg, teff_grid_req, rtol=0.0, atol=0.0)) or
            (not np.allclose(zg, feh_grid_req, rtol=0.0, atol=0.0)) or
            (not np.allclose(gg, logg_grid_req, rtol=0.0, atol=0.0))
        ):
            need_rebuild = True

    if need_rebuild:
        phoenix_lib.build_interpolator(
            observed_wave=model_wave_grid,
            teff_grid=teff_grid_req,
            feh_grid=feh_grid_req,
            logg_grid=logg_grid_req,
            cache_path=cache_path,
            allow_missing=allow_missing,
            observed_wave_medium=model_wave_medium,
        )
    
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
        
    broadening_fwhm_kms = _resolve_broadening_fwhm_kms(R=R, fwhm_kms=fwhm_kms)
    
    def residuals(p):
        teff, feh, logg, rv_kms = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        rv_tot = rv_bary_kms + rv_kms

        try:
            model0 = phoenix_lib.evaluate(teff, feh, logg)
        except Exception:
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
        p0 = (teff0, feh0, logg0, rv0_best)
    elif rv_init is None:
        p0 = (teff0, feh0, logg0, rv0)
    else:
        raise ValueError("rv_init must be 'grid' or None.")
    
    if x_scale is None:
        x_scale = np.array([100.0, 0.1, 0.1, 10.0], dtype=float)
       
    res = least_squares(
        residuals,
        x0=np.array(p0, dtype=float),
        bounds=bounds,
        method="trf",
        x_scale=x_scale,
        max_nfev=int(max_nfev),
        verbose=2 if verbose else 0,
    )

    # Compute diagnostics. If segment weights are used, chi2 is the weighted
    # sum of squared normalized residuals.
    r = res.fun
    chi2 = float(np.sum(r * r))
    n = int(r.size)
    k = 4  # teff, feh, logg, rv
    # Effective dof includes polynomial coefficients, but they were solved analytically.
    # Report dof as N - k for a conservative baseline.
    dof = max(1, n - k)
    chi2_red = chi2 / dof

    return {
        "success": bool(res.success),
        "message": res.message,
        "p_best": res.x,
        "teff": float(res.x[0]),
        "feh": float(res.x[1]),
        "logg": float(res.x[2]),
        "rv_kms": float(res.x[3]),
        "rv_bary_kms": float(rv_bary_kms),
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2_red,
        "n_points": n,
        "status": int(res.status),
        "nfev": int(res.nfev),
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
        # Note: did not store poly coeffs in this minimal version to avoid re-evaluating.
    }
