"""
Workflow-level fitting recipes built on top of Spyctres core primitives.

This module is intentionally higher-level than Spyctres.fitting. It contains
maintained recipe helpers that are useful for real workflows and examples,
but are more specialized than the generic full-spectrum fitter.

Current scope
-------------
- X-SHOOTER/Balmer-window helper presets
- sideband normalization for line-window workflows
- sideband-aware PHOENIX fitting on top of the native-grid forward model
- reconstruction of fitted models for plotting

These helpers operate on generic SpectrumSegment objects, so the logic is not
strictly tied to one instrument even when some presets are X-SHOOTER-oriented.
"""

import numpy as np
from scipy.optimize import least_squares

from .io import SpectrumSegment
from .waveutils import convert_wavelength_medium
from .Spyctres import velocity_correction
from .fitting import (
    build_effective_fit_mask,
    build_excluded_mask,
    reconstruct_phoenix_legendre_models_for_segments,
    _resolve_broadening_fwhm_kms,
    _gaussian_broaden_velocity,
)
from .phoenix_forward import (
    build_phoenix_native_models_for_segments,
    infer_segments_wave_medium,
    fit_bounds_from_segments,
    prepare_phoenix_native_template,
)

BALMER_CENTERS_VAC = {
    "Hδ": 4101.74,
    "Hγ": 4340.47,
    "Hβ": 4861.33,
}

XSHOOTER_BALMER_WINDOWS = {
    "current": [
        ("Hδ", 4076.0, 4128.0),
        ("Hγ", 4314.0, 4366.0),
        ("Hβ", 4836.0, 4888.0),
    ],
    "notebook": [
        ("Hδ", 3980.0, 4220.0),
        ("Hγ", 4220.0, 4480.0),
        ("Hβ", 4700.0, 5020.0),
    ],
}

XSHOOTER_NOTEBOOK_CONT_WINDOWS = {
    "Hδ": ((-80.0, -30.0), (30.0, 80.0)),
    "Hγ": ((-80.0, -30.0), (30.0, 80.0)),
    "Hβ": ((-120.0, -40.0), (40.0, 120.0)),
}


def xshooter_balmer_windows(window_mode="notebook"):
    """
    Return X-SHOOTER UVB Balmer-window presets.

    Parameters
    ----------
    window_mode : {"current", "notebook"}
        Current narrow windows or broader notebook-style windows.

    Returns
    -------
    list of (label, wmin, wmax)
    """
    mode = str(window_mode).strip().lower()
    if mode not in XSHOOTER_BALMER_WINDOWS:
        raise ValueError("window_mode must be 'current' or 'notebook'.")
    return list(XSHOOTER_BALMER_WINDOWS[mode])


def attach_balmer_metadata(segments, cont_windows=None, centers_vac=None):
    """
    Attach Balmer line metadata to window segments in-place.

    This expects segment names like 'Hδ', 'Hγ', 'Hβ'. For each segment, it stores:
    - line_label
    - line_center_vac
    - line_center_data
    - cont_windows

    Parameters
    ----------
    segments : list[SpectrumSegment]
        Input Balmer-window segments.
    cont_windows : dict, optional
        Mapping from line label to continuum sideband windows.
        Defaults to XSHOOTER_NOTEBOOK_CONT_WINDOWS.
    centers_vac : dict, optional
        Mapping from line label to vacuum line center in Angstrom.
        Defaults to BALMER_CENTERS_VAC.

    Returns
    -------
    list[SpectrumSegment]
        The same list, for convenience.
    """
    if cont_windows is None:
        cont_windows = XSHOOTER_NOTEBOOK_CONT_WINDOWS
    if centers_vac is None:
        centers_vac = BALMER_CENTERS_VAC

    for seg in segments:
        label = str(seg.name)
        if label not in centers_vac:
            raise ValueError(
                "Segment name {0!r} not recognized as a supported Balmer label.".format(label)
            )

        center_vac = float(centers_vac[label])
        seg.meta["line_label"] = label
        seg.meta["line_center_vac"] = center_vac

        seg_medium = str(seg.wave_medium).lower()
        if seg_medium in ("air", "vacuum"):
            center_data = float(
                convert_wavelength_medium(
                    np.array([center_vac], dtype=float),
                    from_medium="vacuum",
                    to_medium=seg_medium,
                )[0]
            )
        else:
            center_data = center_vac

        seg.meta["line_center_data"] = center_data
        seg.meta["cont_windows"] = cont_windows[label]

    return segments


def ensure_phoenix_interpolator_for_segments(
    segments,
    phoenix_lib,
    teff_grid,
    feh_grid,
    logg_grid,
    cache_path=None,
):
    """
    Ensure the PHOENIX interpolator is built on the concatenated support grid
    of the current segments.
    """
    support_wave_all = np.concatenate([np.asarray(seg.wave, dtype=float) for seg in segments])

    segment_media = sorted(set(str(seg.wave_medium).lower() for seg in segments))
    if len(segment_media) == 1:
        observed_wave_medium = segment_media[0]
    else:
        observed_wave_medium = None

    need_rebuild = False
    if phoenix_lib.wave is None:
        need_rebuild = True
    elif (len(phoenix_lib.wave) != len(support_wave_all)) or (
        not np.allclose(phoenix_lib.wave, support_wave_all, rtol=0.0, atol=0.0)
    ):
        need_rebuild = True
    elif phoenix_lib._grid is None:
        need_rebuild = True
    else:
        tg, zg, gg = phoenix_lib._grid
        if (
            (len(tg) != len(teff_grid)) or
            (len(zg) != len(feh_grid)) or
            (len(gg) != len(logg_grid)) or
            (not np.allclose(tg, np.asarray(teff_grid, dtype=float), rtol=0.0, atol=0.0)) or
            (not np.allclose(zg, np.asarray(feh_grid, dtype=float), rtol=0.0, atol=0.0)) or
            (not np.allclose(gg, np.asarray(logg_grid, dtype=float), rtol=0.0, atol=0.0))
        ):
            need_rebuild = True

    if need_rebuild:
        phoenix_lib.build_interpolator(
            observed_wave=support_wave_all,
            teff_grid=np.asarray(teff_grid, dtype=float),
            feh_grid=np.asarray(feh_grid, dtype=float),
            logg_grid=np.asarray(logg_grid, dtype=float),
            cache_path=cache_path,
            observed_wave_medium=observed_wave_medium,
        )

    return support_wave_all


def _build_native_interp_wave_grid_for_segments(segments, phoenix_lib, model_margin_A=200.0):
    """
    Build the dense wavelength grid used by the native_interp sideband fitter.
    """
    if getattr(phoenix_lib, "phoenix_wave", None) is None:
        raise ValueError("phoenix_lib.phoenix_wave is not initialized.")

    target_wave_medium = infer_segments_wave_medium(
        segments,
        default=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
    )

    wmin_fit, wmax_fit = fit_bounds_from_segments(
        segments,
        use_fit_mask=True,
    )

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


def _build_sideband_mask(seg, wave, fit_mask, sideband_width=10.0):
    """
    Build sideband mask for one segment.

    If seg.meta['cont_windows'] is present, use those explicit sidebands
    relative to seg.meta['line_center_data']. Otherwise fall back to
    edge sidebands of the fit window.
    """
    wave = np.asarray(wave, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)

    cont_windows = seg.meta.get("cont_windows", None)
    center = seg.meta.get("line_center_data", None)

    if cont_windows is not None and center is not None:
        sb_mask = np.zeros_like(wave, dtype=bool)
        center = float(center)
        for a, b in cont_windows:
            sb_mask |= fit_mask & (wave > center + float(a)) & (wave < center + float(b))
        sb_mode = "explicit"
        lo = float(np.min(wave[fit_mask])) if np.any(fit_mask) else np.nan
        hi = float(np.max(wave[fit_mask])) if np.any(fit_mask) else np.nan
        return sb_mask, sb_mode, lo, hi

    fit_wave = wave[fit_mask]
    lo = float(np.min(fit_wave))
    hi = float(np.max(fit_wave))
    sb_mask = fit_mask & (
        ((wave >= lo) & (wave <= lo + float(sideband_width))) |
        ((wave >= hi - float(sideband_width)) & (wave <= hi))
    )
    return sb_mask, "edge", lo, hi


def normalize_segment_sidebands(seg, sideband_width=10.0, sideband_order=1):
    """
    Normalize one segment using a weighted polynomial continuum fit to either:
    - explicit per-line sidebands stored in seg.meta['cont_windows'], or
    - fallback edge sidebands of the fit window.

    Returns
    -------
    seg_n : SpectrumSegment
        Sideband-normalized segment.
    info : dict
        Small diagnostic dictionary about the normalization.
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = None if seg.err is None else np.asarray(seg.err, dtype=float)
    fit_mask = np.asarray(seg.mask, dtype=bool)

    if np.sum(fit_mask) < 6:
        return seg, {"mode": "none", "n_sideband": 0}

    sb_mask, sb_mode, lo, hi = _build_sideband_mask(
        seg, wave, fit_mask, sideband_width=sideband_width
    )

    good = sb_mask & np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        good &= np.isfinite(err) & (err > 0)

    order = int(sideband_order)

    if np.sum(good) >= (order + 2):
        if err is None:
            coeffs = np.polyfit(wave[good], flux[good], deg=order)
        else:
            coeffs = np.polyfit(wave[good], flux[good], deg=order, w=1.0 / err[good])
        cont = np.polyval(coeffs, wave)
        mode = "poly"
    else:
        level = float(np.nanmedian(flux[fit_mask]))
        coeffs = np.array([level], dtype=float)
        cont = np.full_like(wave, level, dtype=float)
        mode = "constant"

    pos = np.isfinite(cont) & (cont > 0)
    if not np.any(pos):
        raise ValueError("Sideband normalization produced a non-positive continuum.")

    fallback = float(np.nanmedian(cont[pos]))
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, fallback)

    flux_n = flux / cont
    err_n = None if err is None else err / cont

    seg_n = SpectrumSegment(
        wave=wave,
        flux=flux_n,
        err=err_n,
        mask=fit_mask,
        meta=dict(seg.meta),
        wave_medium=seg.wave_medium,
        wave_frame=seg.wave_frame,
        name=seg.name,
    )

    seg_n.meta["norm_mode"] = "sideband"
    seg_n.meta["sideband_width"] = float(sideband_width)
    seg_n.meta["sideband_order"] = int(sideband_order)
    seg_n.meta["sideband_cont_coeffs"] = np.asarray(coeffs, dtype=float).tolist()

    info = {
        "mode": mode,
        "sideband_mode": sb_mode,
        "n_sideband": int(np.sum(good)),
        "fit_lo": lo,
        "fit_hi": hi,
        "coeffs": coeffs,
    }
    return seg_n, info


def normalize_segments_sidebands(segments, sideband_width=10.0, sideband_order=1):
    """
    Apply sideband normalization to a list of segments.

    Returns
    -------
    segments_n : list[SpectrumSegment]
    info : list[dict]
    """
    out = []
    info = []
    for seg in segments:
        seg_n, seg_info = normalize_segment_sidebands(
            seg,
            sideband_width=sideband_width,
            sideband_order=sideband_order,
        )
        out.append(seg_n)
        info.append(seg_info)
    return out, info


def normalize_model_sidebands(seg, model_flux, sideband_width=10.0, sideband_order=1):
    """
    Normalize a model array on a segment grid using the same sideband logic
    as the data-side normalization.

    Returns
    -------
    model_n : ndarray
    info : dict
    """
    wave = np.asarray(seg.wave, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)
    fit_mask = np.asarray(seg.mask, dtype=bool)

    if np.sum(fit_mask) < 6:
        return model_flux.copy(), {"mode": "none", "n_sideband": 0}

    sb_mask, sb_mode, lo, hi = _build_sideband_mask(
        seg, wave, fit_mask, sideband_width=sideband_width
    )

    good = sb_mask & np.isfinite(wave) & np.isfinite(model_flux)
    order = int(sideband_order)

    if np.sum(good) >= (order + 2):
        coeffs = np.polyfit(wave[good], model_flux[good], deg=order)
        cont = np.polyval(coeffs, wave)
        mode = "poly"
    else:
        level = float(np.nanmedian(model_flux[fit_mask]))
        coeffs = np.array([level], dtype=float)
        cont = np.full_like(wave, level, dtype=float)
        mode = "constant"

    pos = np.isfinite(cont) & (cont > 0)
    if not np.any(pos):
        raise ValueError("Model sideband normalization produced a non-positive continuum.")

    fallback = float(np.nanmedian(cont[pos]))
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, fallback)

    model_n = model_flux / cont
    info = {
        "mode": mode,
        "sideband_mode": sb_mode,
        "n_sideband": int(np.sum(good)),
        "fit_lo": lo,
        "fit_hi": hi,
        "coeffs": coeffs,
    }
    return model_n, info


def _solve_sideband_multiplicative_poly(wave, flux, err, model, used_mask, order=1):
    """
    Solve a weighted multiplicative polynomial after sideband normalization.

    This mirrors the notebook logic:
        flux ~ model * poly(w)
    on the used wing pixels.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    model = np.asarray(model, dtype=float)
    used_mask = np.asarray(used_mask, dtype=bool)
    order = int(order)

    if order <= 0:
        return model.copy(), np.array([1.0], dtype=float)

    good = (
        used_mask &
        np.isfinite(wave) &
        np.isfinite(flux) &
        np.isfinite(err) & (err > 0) &
        np.isfinite(model)
    )

    if np.sum(good) < (order + 2):
        return model.copy(), np.array([1.0], dtype=float)

    x0 = float(np.mean(wave[good]))
    xscale = float(np.ptp(wave[good]))
    if (not np.isfinite(xscale)) or (xscale <= 0):
        xscale = 1.0

    x = (wave[good] - x0) / xscale
    A = np.vander(x, N=order + 1, increasing=True)

    rhs = flux[good] / (model[good] + 1e-30)
    W = 1.0 / err[good]

    Aw = A * W[:, None]
    bw = rhs * W
    coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    x_all = (wave - x0) / xscale
    poly_all = np.vander(x_all, N=order + 1, increasing=True) @ coeffs

    return model * poly_all, coeffs


def make_balmer_core_exclude_mask(core_halfwidth=3.0, wave_medium="vacuum"):
    """
    Build a boolean exclusion mask callable for the Balmer line cores.

    Parameters
    ----------
    core_halfwidth : float
        Half-width in Angstrom around each Balmer line center to exclude.
    wave_medium : {"air", "vacuum", "unknown"}
        Wavelength medium of the observed data.
    """
    centers_vac = np.array([4101.74, 4340.47, 4861.33], dtype=float)

    wave_medium = str(wave_medium).lower()
    if wave_medium in ("air", "vacuum"):
        centers = convert_wavelength_medium(
            centers_vac,
            from_medium="vacuum",
            to_medium=wave_medium,
        )
    else:
        centers = centers_vac.copy()

    def _mask(wave):
        wave = np.asarray(wave, dtype=float)
        m = np.zeros_like(wave, dtype=bool)
        for c in centers:
            m |= np.abs(wave - c) <= float(core_halfwidth)
        return m

    return _mask


def fit_phoenix_sideband_symmetric(
    segments,
    phoenix_lib,
    p0,
    exclude_mask=None,
    rv_bary_kms=0.0,
    R=None,
    forward_model="interp_observed",
    model_margin_A=200.0,
    teff_grid=None,
    feh_grid=None,
    logg_grid=None,
    cache_path=None,
    rv_init="grid",
    rv_grid_n=81,
    verbose=1,
    max_nfev=200,
    sideband_width=10.0,
    sideband_order=1,
    sideband_poly_order=1,
    bounds=None,
):
    """
    Sideband-normalized fitter for Balmer-window workflows.

    Data are already sideband-normalized segment-by-segment, the model is
    sideband-normalized the same way, and a low-order multiplicative polynomial
    is then solved on the used wing pixels before residuals are computed.

    The wavelength-space forward model can follow either:
    - forward_model="interp_observed": interpolate directly on the segment support
      grid, then shift and broaden there.
    - forward_model="native_interp": interpolate on a dense model-space wavelength
      grid, then shift, convolve, and resample last.
    """
    if not isinstance(segments, (list, tuple)):
        segments = [segments]

    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")

    used_masks = [
        build_effective_fit_mask(seg, exclude_mask=exclude_mask)
        for seg in segments
    ]
    if not any(np.any(m) for m in used_masks):
        raise ValueError("No usable points remain after masking.")

    support_lengths = [len(seg.wave) for seg in segments]

    if forward_model == "interp_observed":
        support_wave_all = ensure_phoenix_interpolator_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            teff_grid=teff_grid,
            feh_grid=feh_grid,
            logg_grid=logg_grid,
            cache_path=cache_path,
        )
        model_wave_grid = support_wave_all

        segment_media = sorted(set(str(seg.wave_medium).lower() for seg in segments))
        if len(segment_media) == 1:
            model_wave_medium = segment_media[0]
        else:
            model_wave_medium = None

    else:
        model_wave_grid, model_wave_medium = _build_native_interp_wave_grid_for_segments(
            segments=segments,
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
                (len(tg) != len(teff_grid)) or
                (len(zg) != len(feh_grid)) or
                (len(gg) != len(logg_grid)) or
                (not np.allclose(tg, np.asarray(teff_grid, dtype=float), rtol=0.0, atol=0.0)) or
                (not np.allclose(zg, np.asarray(feh_grid, dtype=float), rtol=0.0, atol=0.0)) or
                (not np.allclose(gg, np.asarray(logg_grid, dtype=float), rtol=0.0, atol=0.0))
            ):
                need_rebuild = True

        if need_rebuild:
            phoenix_lib.build_interpolator(
                observed_wave=model_wave_grid,
                teff_grid=np.asarray(teff_grid, dtype=float),
                feh_grid=np.asarray(feh_grid, dtype=float),
                logg_grid=np.asarray(logg_grid, dtype=float),
                cache_path=cache_path,
                observed_wave_medium=model_wave_medium,
            )

    if bounds is None:
        tg, zg, gg = phoenix_lib._grid
        bounds = (
            (float(np.min(tg)), float(np.min(zg)), float(np.min(gg)), -300.0),
            (float(np.max(tg)), float(np.max(zg)), float(np.max(gg)), +300.0),
        )

    broadening_fwhm_kms = _resolve_broadening_fwhm_kms(R=R, fwhm_kms=None)
    n_points = int(sum(np.sum(m) for m in used_masks))

    def residuals(p):
        teff, feh, logg, rv_kms = map(float, p)

        try:
            model0 = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
        except Exception:
            return np.ones(n_points, dtype=float) * 1e6

        out = []

        if forward_model == "interp_observed":
            rv_tot = float(rv_bary_kms) + float(rv_kms)
            if len(model0) != len(support_wave_all):
                return np.ones(n_points, dtype=float) * 1e6

            shifted_all = velocity_correction(np.c_[support_wave_all, model0], rv_tot)[:, 1]

            i0 = 0
            for seg, used_mask, n_support in zip(segments, used_masks, support_lengths):
                i1 = i0 + n_support

                wave = np.asarray(seg.wave, dtype=float)
                flux = np.asarray(seg.flux, dtype=float)
                err = np.asarray(seg.err, dtype=float)

                model_full = shifted_all[i0:i1]
                model_full = _gaussian_broaden_velocity(
                    wave, model_full, fwhm_kms=broadening_fwhm_kms
                )

                model_norm, _ = normalize_model_sidebands(
                    seg,
                    model_full,
                    sideband_width=sideband_width,
                    sideband_order=sideband_order,
                )

                model_corr, _ = _solve_sideband_multiplicative_poly(
                    wave=wave,
                    flux=flux,
                    err=err,
                    model=model_norm,
                    used_mask=used_mask,
                    order=sideband_poly_order,
                )

                out.append((flux[used_mask] - model_corr[used_mask]) / err[used_mask])
                i0 = i1

        else:
            model_list = build_phoenix_native_models_for_segments(
                segments=segments,
                phoenix_wave_native=np.asarray(phoenix_lib.wave, dtype=float),
                template_flux_native=model0,
                rv_kms=rv_kms,
                rv_bary_kms=rv_bary_kms,
                R=R,
                phoenix_wave_medium=model_wave_medium,
                model_margin_A=model_margin_A,
                bounds_use_fit_mask=True,
                extrapolate=True,
            )

            for seg, used_mask, model_full in zip(segments, used_masks, model_list):
                wave = np.asarray(seg.wave, dtype=float)
                flux = np.asarray(seg.flux, dtype=float)
                err = np.asarray(seg.err, dtype=float)

                model_norm, _ = normalize_model_sidebands(
                    seg,
                    model_full,
                    sideband_width=sideband_width,
                    sideband_order=sideband_order,
                )

                model_corr, _ = _solve_sideband_multiplicative_poly(
                    wave=wave,
                    flux=flux,
                    err=err,
                    model=model_norm,
                    used_mask=used_mask,
                    order=sideband_poly_order,
                )

                out.append((flux[used_mask] - model_corr[used_mask]) / err[used_mask])

        return np.concatenate(out)

    teff0, feh0, logg0, rv0 = map(float, p0)

    if rv_init == "grid":
        rv_lo, rv_hi = float(bounds[0][3]), float(bounds[1][3])
        rv_grid = np.linspace(rv_lo, rv_hi, int(rv_grid_n))
        chi2s = np.array(
            [np.sum(residuals((teff0, feh0, logg0, float(rv))) ** 2) for rv in rv_grid],
            dtype=float,
        )
        rv0_best = float(rv_grid[int(np.argmin(chi2s))])
        if verbose:
            print("RV init grid best:", rv0_best)
        p0_use = (teff0, feh0, logg0, rv0_best)
    else:
        p0_use = (teff0, feh0, logg0, rv0)

    res = least_squares(
        residuals,
        x0=np.array(p0_use, dtype=float),
        bounds=bounds,
        method="trf",
        x_scale=np.array([100.0, 0.1, 0.1, 10.0], dtype=float),
        max_nfev=int(max_nfev),
        verbose=2 if verbose else 0,
    )

    r = res.fun
    chi2 = float(np.sum(r * r))
    n = int(r.size)
    k = 4
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
        "forward_model": str(forward_model),
        "model_margin_A": float(model_margin_A),
        "resolution_R": None if R is None else float(R),
        "lsf_fwhm_kms": None if broadening_fwhm_kms is None else float(broadening_fwhm_kms),
    }


def build_plot_models_for_segments(
    segments,
    phoenix_lib,
    fit_result,
    exclude_mask=None,
    mdeg=2,
    rv_bary_kms=0.0,
    R=None,
    fwhm_kms=None,
    norm_mode="poly",
    sideband_width=10.0,
    sideband_order=1,
    sideband_poly_order=1,
    forward_model=None,
    model_margin_A=None,
):
    """
    Reconstruct per-segment fitted model arrays on the full pixel grid of each segment.
    """
    teff = float(fit_result["teff"])
    feh = float(fit_result["feh"])
    logg = float(fit_result["logg"])
    rv_kms = float(fit_result["rv_kms"])

    if forward_model is None:
        forward_model = str(fit_result.get("forward_model", "interp_observed"))
    if model_margin_A is None:
        model_margin_A = float(fit_result.get("model_margin_A", 200.0))

    if norm_mode == "poly":
        return reconstruct_phoenix_legendre_models_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            fit_result=fit_result,
            exclude_mask=exclude_mask,
            mdeg=mdeg,
            rv_bary_kms=rv_bary_kms,
            R=R,
            fwhm_kms=fwhm_kms,
            forward_model=forward_model,
            model_margin_A=model_margin_A,
        )

    if norm_mode != "sideband":
        raise ValueError("norm_mode must be 'poly' or 'sideband'.")

    used_masks = [
        build_effective_fit_mask(seg, exclude_mask=exclude_mask)
        for seg in segments
    ]
    excluded_masks = [
        build_excluded_mask(seg, exclude_mask=exclude_mask)
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

        broadening_fwhm_kms = _resolve_broadening_fwhm_kms(R=R, fwhm_kms=fwhm_kms)

        i0 = 0
        for seg, used_mask in zip(segments, used_masks):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)
            err_full = np.asarray(seg.err, dtype=float)

            n_support = len(wave_full)
            i1 = i0 + n_support

            model0_full = model_support_all[i0:i1]

            spec = np.c_[wave_full, model0_full]
            shifted_full = velocity_correction(spec, rv_bary_kms + rv_kms)[:, 1]
            model_broad_full = _gaussian_broaden_velocity(
                wave_full, shifted_full, fwhm_kms=broadening_fwhm_kms
            )

            model_norm_full, _norm_info = normalize_model_sidebands(
                seg,
                model_broad_full,
                sideband_width=sideband_width,
                sideband_order=sideband_order,
            )

            model_corr_full, coeffs = _solve_sideband_multiplicative_poly(
                wave=wave_full,
                flux=flux_full,
                err=err_full,
                model=model_norm_full,
                used_mask=used_mask,
                order=sideband_poly_order,
            )

            model_full_list.append(model_corr_full.copy())
            coeffs_list.append(coeffs)
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
            R=R,
            phoenix_wave_medium=model_wave_medium,
            model_margin_A=model_margin_A,
            bounds_use_fit_mask=True,
            extrapolate=True,
        )

        for seg, used_mask, model_broad_full in zip(segments, used_masks, model_raw_list):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)
            err_full = np.asarray(seg.err, dtype=float)

            model_norm_full, _norm_info = normalize_model_sidebands(
                seg,
                model_broad_full,
                sideband_width=sideband_width,
                sideband_order=sideband_order,
            )

            model_corr_full, coeffs = _solve_sideband_multiplicative_poly(
                wave=wave_full,
                flux=flux_full,
                err=err_full,
                model=model_norm_full,
                used_mask=used_mask,
                order=sideband_poly_order,
            )

            model_full_list.append(model_corr_full.copy())
            coeffs_list.append(coeffs)
    else:
        raise ValueError("Unknown forward_model: {0}".format(forward_model))

    return model_full_list, coeffs_list, used_masks, excluded_masks


__all__ = [
    "BALMER_CENTERS_VAC",
    "XSHOOTER_BALMER_WINDOWS",
    "XSHOOTER_NOTEBOOK_CONT_WINDOWS",
    "xshooter_balmer_windows",
    "attach_balmer_metadata",
    "normalize_segment_sidebands",
    "normalize_segments_sidebands",
    "normalize_model_sidebands",
    "make_balmer_core_exclude_mask",
    "fit_phoenix_sideband_symmetric",
    "build_plot_models_for_segments",
]
