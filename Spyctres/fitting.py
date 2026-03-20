# Spyctres/fitting.py
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.legendre import legvander

from .Spyctres import velocity_correction
from .io import SpectrumSegment
from .phoenix_forward import (
    infer_segments_wave_medium,
    fit_bounds_from_segments,
    prepare_phoenix_native_template,
    build_phoenix_native_models_for_segments,
)
# Why multiplicative polynomial: it is a standard way to absorb low-frequency continuum differences and calibration mismatches during full-spectrum fitting.

# RV handling: velocity_correction applies a Doppler shift by re-sampling the model spectrum at shifted wavelengths, consistent with the standard approximation Δλ/λ ≈ v/c for small velocities.

C_KMS = 299792.458

def _resolve_broadening_fwhm_kms(R=None, fwhm_kms=None):
    """
    Resolve the effective Gaussian FWHM in km/s.
    Exactly one of R or fwhm_kms may be provided.
    """
    if (R is not None) and (fwhm_kms is not None):
        raise ValueError("Provide only one of R or fwhm_kms, not both.")

    if fwhm_kms is not None:
        fwhm_kms = float(fwhm_kms)
        if fwhm_kms <= 0:
            raise ValueError("fwhm_kms must be > 0.")
        return fwhm_kms

    if R is None:
        return None

    R = float(R)
    if R <= 0:
        raise ValueError("R must be > 0.")
    return C_KMS / R


def _gaussian_broaden_velocity(wave, flux, fwhm_kms=None):
    """
    Convolve a spectrum with a Gaussian LSF of constant FWHM in velocity space.

    Implementation:
    1) resample onto a uniform log-lambda grid
    2) Gaussian filter in pixel space
    3) interpolate back onto the input wave grid
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if fwhm_kms is None:
        return flux.copy()

    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if np.sum(good) < 5:
        return flux.copy()

    w = wave[good]
    f = flux[good]

    lnw = np.log(w)
    dln = np.nanmedian(np.diff(lnw))
    if (not np.isfinite(dln)) or (dln <= 0):
        return flux.copy()

    lnw_uniform = np.arange(lnw[0], lnw[-1] + 0.5 * dln, dln)
    f_uniform = np.interp(lnw_uniform, lnw, f)

    sigma_kms = float(fwhm_kms) / 2.3548200450309493
    dv_per_pix = C_KMS * dln
    sigma_pix = sigma_kms / dv_per_pix

    if (not np.isfinite(sigma_pix)) or (sigma_pix <= 0):
        return flux.copy()

    f_broad = gaussian_filter1d(f_uniform, sigma_pix, mode="nearest")

    out = np.array(flux, copy=True, dtype=float)
    out[good] = np.interp(lnw, lnw_uniform, f_broad)
    return out


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


def _build_data_vectors(segments, regions=None, exclude_regions=None, exclude_mask=None):
    """
    Build two synchronized representations of the data:

    1) support-wave representation used to build / evaluate the PHOENIX model
       on the full wavelength support of each segment;
    2) fit-point representation used for chi-square evaluation, where only
       seg.mask-selected pixels (plus optional region / exclusion logic) enter
       the objective.

    Returns
    -------
    support_wave_all : ndarray
        Concatenated full support wavelength grid across segments.
    flux_fit_all, err_fit_all : ndarray
        Concatenated flux/error vectors for fit pixels only.
    support_slices : list[slice]
        Per-segment slices into support_wave_all.
    fit_slices : list[slice]
        Per-segment slices into flux_fit_all / err_fit_all.
    fit_masks : list[ndarray(bool)]
        Boolean masks mapping each segment support grid to its fit pixels.
    seg_meta : list[dict]
        Per-segment metadata.
    """
    support_wave_all = []
    flux_fit_all = []
    err_fit_all = []
    support_slices = []
    fit_slices = []
    fit_masks = []
    seg_meta = []

    start_support = 0
    start_fit = 0

    for i, seg in enumerate(segments):
        w_full = np.asarray(seg.wave, dtype=float)
        f_full = np.asarray(seg.flux, dtype=float)

        support_ok = np.isfinite(w_full) & np.isfinite(f_full)

        if seg.err is None:
            e_full = np.ones_like(f_full) * _estimate_sigma(f_full[support_ok] if np.any(support_ok) else f_full)
            err_ok = np.isfinite(e_full) & (e_full > 0)
        else:
            e_full = np.asarray(seg.err, dtype=float)
            err_ok = np.isfinite(e_full) & (e_full > 0)

        support_ok &= err_ok

        if isinstance(regions, dict):
            reg = regions.get(i, regions.get(seg.name, None))
        else:
            reg = regions

        fit_m = np.asarray(seg.mask, dtype=bool)
        fit_m &= support_ok
        fit_m &= _select_region(w_full, reg)

        if isinstance(exclude_regions, dict):
            ex = exclude_regions.get(i, exclude_regions.get(seg.name, None))
        else:
            ex = exclude_regions
        fit_m &= _exclude_region(w_full, ex)

        if exclude_mask is not None:
            fit_m &= ~_to_bool_mask(exclude_mask(w_full))

        n_support = int(np.sum(support_ok))
        n_fit = int(np.sum(fit_m))

        if n_support == 0:
            continue
        if n_fit == 0:
            continue

        w_support = w_full[support_ok].astype(float)
        f_fit = f_full[fit_m].astype(float)
        e_fit = e_full[fit_m].astype(float)

        fit_mask_on_support = fit_m[support_ok]

        support_wave_all.append(w_support)
        flux_fit_all.append(f_fit)
        err_fit_all.append(e_fit)

        support_slices.append(slice(start_support, start_support + n_support))
        fit_slices.append(slice(start_fit, start_fit + n_fit))
        fit_masks.append(fit_mask_on_support)

        seg_meta.append({
            "name": seg.name,
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

    return support_wave_all, flux_fit_all, err_fit_all, support_slices, fit_slices, fit_masks, seg_meta
    
    
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


def _chi2_for_params(
    support_wave_all, flux_all, err_all,
    support_slices, fit_slices, fit_masks,
    teff, feh, logg, rv_tot, phoenix_lib, mdeg,
    decimate=1,
    broadening_fwhm_kms=None,
):
    """
    Compute chi2 with per-segment multiplicative polynomial solved linearly.
    Used for RV initialization.

    The model is built on the full support wavelength grid, but chi2 is
    evaluated only on the fit pixels inside each segment.
    """
    model0 = phoenix_lib.evaluate(teff, feh, logg)
    shifted = velocity_correction(np.c_[support_wave_all, model0], rv_tot)[:, 1]

    chi2 = 0.0
    for support_sl, fit_sl, fit_mask in zip(support_slices, fit_slices, fit_masks):
        w_support = support_wave_all[support_sl]
        m_support = _gaussian_broaden_velocity(
            w_support, shifted[support_sl], fwhm_kms=broadening_fwhm_kms
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
        chi2 += float(np.sum(r * r))

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


def _build_native_interp_wave_grid(forward_segments, phoenix_lib, model_margin_A=200.0):
    """
    Build a dense wavelength grid for the native_interp branch.

    The grid is a prepared subset of the native PHOENIX wavelength grid, already
    converted into the segments' wavelength medium and restricted to the fit
    bounds plus margin.
    """
    target_wave_medium = infer_segments_wave_medium(
        forward_segments,
        default=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
    )

    wmin_fit, wmax_fit = fit_bounds_from_segments(
        forward_segments,
        use_fit_mask=True,
    )
    
    if getattr(phoenix_lib, "phoenix_wave", None) is None:
        raise ValueError("phoenix_lib.phoenix_wave is not initialized.")
        
    model_wave_grid, _dummy_flux = prepare_phoenix_native_template(
        phoenix_wave_native=np.asarray(phoenix_lib.phoenix_wave, dtype=float),
        template_flux_native=np.ones_like(np.asarray(phoenix_lib.phoenix_wave, dtype=float)),
        target_wave_medium=target_wave_medium,
        phoenix_wave_medium=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
        wmin=wmin_fit,
        wmax=wmax_fit,
        margin_A=model_margin_A,
    )

    return model_wave_grid, target_wave_medium


def _chi2_for_params_native_interp(
    forward_segments,
    flux_all,
    err_all,
    fit_slices,
    fit_masks,
    teff,
    feh,
    logg,
    rv_tot,
    phoenix_lib,
    model_wave_grid,
    model_wave_medium,
    mdeg,
    decimate=1,
    R=None,
    model_margin_A=200.0,
):
    """
    Compute chi2 for the native_interp branch.

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
        R=R,
        phoenix_wave_medium=model_wave_medium,
        model_margin_A=model_margin_A,
        bounds_use_fit_mask=True,
        extrapolate=True,
    )

    chi2 = 0.0
    for seg, model_full, fit_sl, fit_mask in zip(forward_segments, model_list, fit_slices, fit_masks):
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
        chi2 += float(np.sum(r * r))

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
      interpolate directly on the observed support wavelength grid, then shift
      and broaden there. This is the original fast path.

    - `forward_model="native_interp"`:
      interpolate on a dense model-space wavelength grid, then apply the
      notebook-faithful forward-model order: shift on that dense grid, convolve
      at constant resolving power, and resample last to each segment support grid.

    In both cases, the model is multiplied by a per-segment Legendre polynomial
    continuum solved analytically by weighted least squares.

    Parameters
    ----------
    segments : list[SpectrumSegment]
        Input spectrum segments to fit.

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
    """
    if not isinstance(segments, (list, tuple)):
        segments = [segments]
        
    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")
        
    if forward_model == "native_interp" and fwhm_kms is not None:
        raise NotImplementedError(
            "forward_model='native_interp' currently supports R-based broadening only."
        )
        
    support_wave_all, flux_all, err_all, support_slices, fit_slices, fit_masks, seg_meta = _build_data_vectors(
        segments, regions=regions, exclude_regions=exclude_regions, exclude_mask=exclude_mask
    )
    if support_wave_all.size == 0 or flux_all.size == 0:
        raise ValueError("No data points selected for fitting.")
    
    forward_segments = _make_forward_segments(
        segments=segments,
        support_wave_all=support_wave_all,
        support_slices=support_slices,
        fit_masks=fit_masks,
    )
    
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
        model_wave_grid, model_wave_medium = _build_native_interp_wave_grid(
            forward_segments=forward_segments,
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
            spec = np.c_[support_wave_all, model0]
            shifted = velocity_correction(spec, rv_tot)[:, 1]

            for support_sl, fit_sl, fit_mask, meta in zip(support_slices, fit_slices, fit_masks, seg_meta):
                w_support = support_wave_all[support_sl]
                m_support = _gaussian_broaden_velocity(
                    w_support, shifted[support_sl], fwhm_kms=broadening_fwhm_kms
                )

                w = w_support[fit_mask]
                f = flux_all[fit_sl]
                e = err_all[fit_sl]
                m = m_support[fit_mask]

                m_corr, coeffs = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
                out[fit_sl] = (f - m_corr) / e
        else:
            model_list = build_phoenix_native_models_for_segments(
                segments=forward_segments,
                phoenix_wave_native=model_wave_grid,
                template_flux_native=model0,
                rv_kms=rv_kms,
                rv_bary_kms=rv_bary_kms,
                R=R,
                phoenix_wave_medium=model_wave_medium,
                model_margin_A=model_margin_A,
                bounds_use_fit_mask=True,
                extrapolate=True,
            )

            for seg, model_full, fit_sl, fit_mask in zip(forward_segments, model_list, fit_slices, fit_masks):
                w_support = np.asarray(seg.wave, dtype=float)

                w = w_support[fit_mask]
                f = flux_all[fit_sl]
                e = err_all[fit_sl]
                m = np.asarray(model_full, dtype=float)[fit_mask]

                m_corr, coeffs = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
                out[fit_sl] = (f - m_corr) / e

        return out
    
    # RV initialization by coarse grid scan (optional)
    if rv_init == "grid":
        rv_lo, rv_hi = float(bounds[0][3]), float(bounds[1][3])
        rv_grid = np.linspace(rv_lo, rv_hi, int(rv_grid_n))
        
        chi2s = np.empty(rv_grid.size, dtype=float)
        for j, rv in enumerate(rv_grid):
            if forward_model == "interp_observed":
                chi2s[j] = _chi2_for_params(
                    support_wave_all, flux_all, err_all,
                    support_slices, fit_slices, fit_masks,
                    teff0, feh0, logg0,
                    rv_bary_kms + float(rv),
                    phoenix_lib,
                    mdeg=mdeg,
                    decimate=rv_grid_decimate,
                    broadening_fwhm_kms=broadening_fwhm_kms,
                )
            else:
                chi2s[j] = _chi2_for_params_native_interp(
                    forward_segments=forward_segments,
                    flux_all=flux_all,
                    err_all=err_all,
                    fit_slices=fit_slices,
                    fit_masks=fit_masks,
                    teff=teff0,
                    feh=feh0,
                    logg=logg0,
                    rv_tot=rv_bary_kms + float(rv),
                    phoenix_lib=phoenix_lib,
                    model_wave_grid=model_wave_grid,
                    model_wave_medium=model_wave_medium,
                    mdeg=mdeg,
                    decimate=rv_grid_decimate,
                    R=R,
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

    # Compute diagnostics
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
        "resolution_R": None if R is None else float(R),
        "lsf_fwhm_kms": None if broadening_fwhm_kms is None else float(broadening_fwhm_kms),
        # Note: did not store poly coeffs in this minimal version to avoid re-evaluating.
    }
