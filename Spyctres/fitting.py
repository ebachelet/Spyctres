# Spyctres/fitting.py
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import legvander

from .Spyctres import velocity_correction

# Why multiplicative polynomial: it is a standard way to absorb low-frequency continuum differences and calibration mismatches during full-spectrum fitting.

# RV handling: velocity_correction applies a Doppler shift by re-sampling the model spectrum at shifted wavelengths, consistent with the standard approximation Δλ/λ ≈ v/c for small velocities.

C_KMS = 299792.458


def _select_region(wave, regions):
    """Return boolean mask selecting points inside any (wmin,wmax) in regions."""
    if regions is None:
        return np.ones_like(wave, dtype=bool)
    m = np.zeros_like(wave, dtype=bool)
    for (wmin, wmax) in regions:
        m |= (wave >= wmin) & (wave <= wmax)
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


def _build_data_vectors(segments, regions=None):
    """
    Build concatenated wave/flux/err arrays plus per-segment slices.
    regions can be:
      - None: use full coverage
      - list of (wmin,wmax): applied to all segments
      - dict: key is segment index or segment.name, value is list of (wmin,wmax)
    """
    wave_all = []
    flux_all = []
    err_all = []
    seg_slices = []
    seg_meta = []

    start = 0
    for i, seg in enumerate(segments):
        if isinstance(regions, dict):
            reg = regions.get(i, regions.get(seg.name, None))
        else:
            reg = regions

        m = np.asarray(seg.mask, dtype=bool)
        m &= np.isfinite(seg.wave) & np.isfinite(seg.flux)
        m &= _select_region(seg.wave, reg)

        w = seg.wave[m].astype(float)
        f = seg.flux[m].astype(float)

        if seg.err is None:
            s = _estimate_sigma(f)
            e = np.ones_like(f) * s
        else:
            e = seg.err[m].astype(float)
            m2 = np.isfinite(e) & (e > 0)
            w, f, e = w[m2], f[m2], e[m2]

        n = len(w)
        wave_all.append(w)
        flux_all.append(f)
        err_all.append(e)

        seg_slices.append(slice(start, start + n))
        seg_meta.append({"name": seg.name, "wave_min": float(w.min()), "wave_max": float(w.max())})
        start += n

    wave_all = np.concatenate(wave_all) if wave_all else np.array([], dtype=float)
    flux_all = np.concatenate(flux_all) if flux_all else np.array([], dtype=float)
    err_all = np.concatenate(err_all) if err_all else np.array([], dtype=float)

    return wave_all, flux_all, err_all, seg_slices, seg_meta

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

def _chi2_for_params(wave_all, flux_all, err_all, seg_slices, teff, feh, logg, rv_tot, phoenix_lib, mdeg, decimate=1):
    """
    Compute chi2 with per-segment multiplicative polynomial solved linearly.
    Used for RV initialization.
    """
    model0 = phoenix_lib.evaluate(teff, feh, logg)
    shifted = velocity_correction(np.c_[wave_all, model0], rv_tot)[:, 1]

    chi2 = 0.0
    for sl in seg_slices:
        if decimate and int(decimate) > 1:
            idx = np.arange(sl.start, sl.stop, int(decimate))
            w = wave_all[idx]
            f = flux_all[idx]
            e = err_all[idx]
            m = shifted[idx]
        else:
            w = wave_all[sl]
            f = flux_all[sl]
            e = err_all[sl]
            m = shifted[sl]

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

def fit_phoenix_full_spectrum(
    segments,
    phoenix_lib,
    p0,
    bounds=None,
    regions=None,
    mdeg=2,
    rv_bary_kms=0.0,
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

    Parameters
    ----------
    segments : list[SpectrumSegment]
    phoenix_lib : PhoenixLibrary (from Spyctres.phoenix), pointing to local PHOENIX install
    p0 : (teff, feh, logg, rv_kms)
    bounds : ((teff_min, feh_min, logg_min, rv_min), (teff_max, feh_max, logg_max, rv_max))
    regions : None, list[(wmin,wmax)], or dict mapping seg index or seg.name to list[(wmin,wmax)]
    mdeg : int, degree of multiplicative Legendre polynomial per segment (0 means just a constant scale)
    rv_bary_kms : float, fixed barycentric term added to fitted rv_kms
    teff_grid, feh_grid, logg_grid, cache_path : passed to phoenix_lib.build_interpolator if needed

    Returns
    -------
    dict with best-fit params, covariance approximation, per-segment polynomial coeffs, chi2, dof
    """
    if not isinstance(segments, (list, tuple)):
        segments = [segments]

    wave_all, flux_all, err_all, seg_slices, seg_meta = _build_data_vectors(segments, regions=regions)
    if wave_all.size == 0:
        raise ValueError("No data points selected for fitting.")
    
    teff0, feh0, logg0, rv0 = map(float, p0)
    
    # Build or reuse interpolator on the exact concatenated wavelength grid
    if phoenix_lib.wave is None or (len(phoenix_lib.wave) != len(wave_all)) or (not np.allclose(phoenix_lib.wave, wave_all)):
        # If user did not provide explicit grids, choose a small local subgrid around p0.
        # This avoids accidentally trying to load the full PHOENIX cube.
        if teff_grid is None:
            teff_grid = _pick_subgrid(phoenix_lib.DEFAULT_TEFF_GRID, teff0, half_width=600.0, n_min=5, n_max=9)
        if feh_grid is None:
            feh_grid = _pick_subgrid(phoenix_lib.DEFAULT_FEH_GRID, feh0, half_width=0.75, n_min=3, n_max=5)
        if logg_grid is None:
            logg_grid = _pick_subgrid(phoenix_lib.DEFAULT_LOGG_GRID, logg0, half_width=0.75, n_min=3, n_max=5)
        phoenix_lib.build_interpolator(
            observed_wave=wave_all,
            teff_grid=teff_grid,
            feh_grid=feh_grid,
            logg_grid=logg_grid,
            cache_path=cache_path,
            allow_missing=allow_missing,
        )

    # Set default bounds from the interpolator grid if none supplied
    if bounds is None:
        if phoenix_lib._grid is None:
            raise ValueError("bounds is None but phoenix_lib has no _grid to infer bounds from.")
        tg, zg, gg = phoenix_lib._grid
        bounds = (
            (float(np.min(tg)), float(np.min(zg)), float(np.min(gg)), -300.0),
            (float(np.max(tg)), float(np.max(zg)), float(np.max(gg)), +300.0),
        )

    def residuals(p):
        teff, feh, logg, rv_kms = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        rv_tot = rv_bary_kms + rv_kms

        try:
            model0 = phoenix_lib.evaluate(teff, feh, logg)
        except Exception:
            # Outside grid etc.
            return np.ones_like(flux_all) * 1e6

        spec = np.c_[wave_all, model0]
        shifted = velocity_correction(spec, rv_tot)[:, 1]

        # Apply per-segment multiplicative polynomial and compute residuals
        out = np.empty_like(flux_all)
        #poly_coeffs = []
        for sl, meta in zip(seg_slices, seg_meta):
            w = wave_all[sl]
            f = flux_all[sl]
            e = err_all[sl]
            m = shifted[sl]

            m_corr, coeffs = _solve_multiplicative_legendre(w, f, e, m, mdeg=mdeg)
            #poly_coeffs.append(coeffs)
            out[sl] = (f - m_corr) / e

        return out
    
    # RV initialization by coarse grid scan (optional)
    if rv_init == "grid":
        rv_lo, rv_hi = float(bounds[0][3]), float(bounds[1][3])
        rv_grid = np.linspace(rv_lo, rv_hi, int(rv_grid_n))

        chi2s = []
        for rv in rv_grid:
            chi2s.append(
                _chi2_for_params(
                    wave_all, flux_all, err_all, seg_slices,
                    teff0, feh0, logg0,
                    rv_bary_kms + float(rv),
                    phoenix_lib,
                    mdeg=mdeg,
                    decimate=rv_grid_decimate,
                )
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
        # Note: did not store poly coeffs in this minimal version to avoid re-evaluating.
    }
