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

from .io import SpectrumSegment, make_padded_window_segments
from .waveutils import (
    C_KMS,
    convert_segment_wavelength_medium,
    convert_wavelength_medium,
)
from .fitting import (
    build_effective_fit_mask,
    build_excluded_mask,
    reconstruct_phoenix_legendre_models_for_segments,
    _resolve_broadening_fwhm_kms,
    _resolve_segment_fwhm_kms,
    _gaussian_broaden_velocity,
    _apply_observed_grid_rv_shift,
)
from .phoenix_forward import (
    build_phoenix_native_models_for_segments,
    infer_segments_wave_medium,
    build_native_interp_wave_grid_for_segments,
)

BALMER_CENTERS_VAC = {
    "Hα": 6562.80,
    "Hβ": 4861.33,
    "Hγ": 4340.47,
    "Hδ": 4101.74,
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

XSHOOTER_BALMER_CORE_MASK_DEFAULT_A = 10.0
XSHOOTER_BALMER_CORE_MASK_CONSERVATIVE_A = 12.0

BALMER_LABEL_ALIASES = {
    "hα": "Hα",
    "halpha": "Hα",
    "ha": "Hα",
    "hβ": "Hβ",
    "hbeta": "Hβ",
    "hb": "Hβ",
    "hγ": "Hγ",
    "hgamma": "Hγ",
    "hg": "Hγ",
    "hδ": "Hδ",
    "hdelta": "Hδ",
    "hd": "Hδ",
}


def _sideband_fit_parameter_count(n_segments, sideband_poly_order):
    """Count nonlinear and profiled continuum parameters for fit diagnostics."""
    n_segments = int(n_segments)
    sideband_poly_order = int(sideband_poly_order)
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1.")
    if sideband_poly_order < 0:
        raise ValueError("sideband_poly_order must be >= 0.")
    return 4 + n_segments * (sideband_poly_order + 1)


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
        A fresh list of preset UVB Balmer windows.
    """
    mode = str(window_mode).strip().lower()
    if mode not in XSHOOTER_BALMER_WINDOWS:
        raise ValueError("window_mode must be 'current' or 'notebook'.")
    return list(XSHOOTER_BALMER_WINDOWS[mode])


def _canonical_balmer_label(label):
    """
    Normalize common ASCII and Unicode Balmer-line aliases to a canonical label.

    Examples
    --------
    Halpha -> Hα
    Hbeta  -> Hβ
    Hgamma -> Hγ
    Hdelta -> Hδ
    """
    raw = str(label).strip()
    key = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    return BALMER_LABEL_ALIASES.get(key, raw)


def _canonicalize_balmer_dict_keys(d):
    """
    Return a copy of a dict whose keys are Balmer-line labels normalized to
    the canonical internal form.
    """
    if d is None:
        return {}
    return {_canonical_balmer_label(k): v for k, v in d.items()}


def attach_balmer_metadata(segments, cont_windows=None, centers_vac=None):
    """
    Attach Balmer-line metadata to segments in-place.

    This helper accepts both canonical Unicode labels such as 'Hβ', 'Hγ', and
    common ASCII aliases such as 'Hbeta', 'Hgamma', 'Hdelta', 'Halpha'.

    For each segment it stores:
    - line_label       : canonical internal label
    - line_label_input : original input label
    - line_center_vac  : vacuum line center
    - line_center_data : line center converted into the segment wavelength medium
    - cont_windows     : continuum sideband windows, if available

    Parameters
    ----------
    segments : list[SpectrumSegment]
        Input line-window segments.
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

    cont_windows = _canonicalize_balmer_dict_keys(cont_windows)
    centers_vac = _canonicalize_balmer_dict_keys(centers_vac)

    for seg in segments:
        raw_label = str(seg.name)
        label = _canonical_balmer_label(raw_label)

        if label not in centers_vac:
            raise ValueError(
                "Segment name {0!r} not recognized as a supported Balmer label.".format(raw_label)
            )

        center_vac = float(centers_vac[label])

        seg.meta["line_label_input"] = raw_label
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

        if label in cont_windows:
            seg.meta["cont_windows"] = cont_windows[label]
        else:
            seg.meta.pop("cont_windows", None)

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
    observed_wave_medium = segment_media[0] if len(segment_media) == 1 else None

    phoenix_lib.ensure_interpolator(
        wave=support_wave_all,
        teff_grid=np.asarray(teff_grid, dtype=float),
        feh_grid=np.asarray(feh_grid, dtype=float),
        logg_grid=np.asarray(logg_grid, dtype=float),
        cache_path=cache_path,
        observed_wave_medium=observed_wave_medium,
    )

    return support_wave_all


def _build_sideband_mask(seg, wave, fit_mask, sideband_width=10.0):
    """
    Build a sideband mask for one segment.

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
        observer_frame=seg.observer_frame,
        stellar_rest_status=seg.stellar_rest_status,
        stellar_rv_applied_kms=seg.stellar_rv_applied_kms,
        resolution=seg.resolution,
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
    on the used pixels.
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


def make_balmer_core_exclude_mask(
    core_halfwidth=XSHOOTER_BALMER_CORE_MASK_DEFAULT_A,
    wave_medium="vacuum",
):
    """
    Build a boolean exclusion-mask callable for the UVB Balmer line cores.

    This helper currently targets the three Balmer lines used by the X-SHOOTER
    UVB recipes: Hδ, Hγ, and Hβ.

    Parameters
    ----------
    core_halfwidth : float
        Half-width in Angstrom around each line center to exclude.
        For X-SHOOTER UVB Balmer-wing classification, the 
        default value is 10 Angstrom. Use 6–12 Angstroms as a robustness range.
    wave_medium : {"air", "vacuum", "unknown"}
        Wavelength medium of the observed data.
    """
    centers_vac = np.array(
        [
            BALMER_CENTERS_VAC["Hδ"],
            BALMER_CENTERS_VAC["Hγ"],
            BALMER_CENTERS_VAC["Hβ"],
        ],
        dtype=float,
    )

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
    forward_model="native_interp",
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
    Sideband-normalized fitter for line-window workflows.

    Data are sideband-normalized segment-by-segment, the model is
    sideband-normalized the same way, and a low-order multiplicative polynomial
    is then solved on the used pixels before residuals are computed.

    The wavelength-space forward model can follow either:
    - forward_model="interp_observed": interpolate directly on the segment
      support grid, then apply the PHOENIX RV convention and broaden there.
      This is retained as a fast/legacy compatibility path.
    - forward_model="native_interp": interpolate on a dense model-space
      wavelength grid, then shift, convolve, and resample last. This is the
      recommended path for line-profile work.
      
    RV convention
    -------------
    The returned `rv_kms` follows the PHOENIX fitting convention used by
    Spyctres.fitting: positive RV redshifts the template/model. The observed-grid
    branch uses `_apply_observed_grid_rv_shift()` internally to preserve this
    convention while leaving the legacy Spyctres.velocity_correction API unchanged.
    """
    if isinstance(segments, SpectrumSegment):
        segments = [segments]
    else:
        segments = list(segments)

    for seg in segments:
        if seg.err is None:
            raise ValueError(
                "fit_phoenix_sideband_symmetric requires seg.err for all segments. "
                "Provide uncertainties or use fit_phoenix_full_spectrum(), which "
                "can estimate fallback errors."
            )

    teff0, feh0, logg0, rv0 = map(float, p0)
    
    if teff_grid is None:
        teff_grid_req = phoenix_lib.DEFAULT_TEFF_GRID
    else:
        teff_grid_req = np.asarray(teff_grid, dtype=float)

    if feh_grid is None:
        feh_grid_req = phoenix_lib.DEFAULT_FEH_GRID
    else:
        feh_grid_req = np.asarray(feh_grid, dtype=float)

    if logg_grid is None:
        logg_grid_req = phoenix_lib.DEFAULT_LOGG_GRID
    else:
        logg_grid_req = np.asarray(logg_grid, dtype=float)
                    
    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")

    used_masks = [
        build_effective_fit_mask(seg, exclude_mask=exclude_mask)
        for seg in segments
    ]
    if not any(np.any(m) for m in used_masks):
        raise ValueError("No usable points remain after masking.")

    support_lengths = [len(seg.wave) for seg in segments]
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=None)
        for seg in segments
    ]
    if forward_model == "interp_observed":
        support_wave_all = ensure_phoenix_interpolator_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            teff_grid=teff_grid_req,
            feh_grid=feh_grid_req,
            logg_grid=logg_grid_req,
            cache_path=cache_path,
        )
    else:
        model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            model_margin_A=model_margin_A,
        )

        if not phoenix_lib.interpolator_matches(
            model_wave_grid,
            teff_grid_req,
            feh_grid_req,
            logg_grid_req,
            observed_wave_medium=model_wave_medium,
        ):
            phoenix_lib.build_interpolator(
                observed_wave=model_wave_grid,
                teff_grid=teff_grid_req,
                feh_grid=feh_grid_req,
                logg_grid=logg_grid_req,
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

            shifted_all = _apply_observed_grid_rv_shift(
                support_wave_all,
                model0,
                rv_tot,
            )
            
            i0 = 0
            for seg, used_mask, n_support, seg_fwhm in zip(
                segments, used_masks, support_lengths, segment_fwhm_kms
            ):
                i1 = i0 + n_support

                wave = np.asarray(seg.wave, dtype=float)
                flux = np.asarray(seg.flux, dtype=float)
                err = np.asarray(seg.err, dtype=float)

                model_full = shifted_all[i0:i1]
                model_full = _gaussian_broaden_velocity(
                    wave,
                    model_full,
                    fwhm_kms=seg_fwhm,
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
                segment_fwhm_kms=segment_fwhm_kms,
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
    
    if rv_init == "grid":
        rv_lo, rv_hi = float(bounds[0][3]), float(bounds[1][3])
        rv_grid = np.linspace(rv_lo, rv_hi, int(rv_grid_n))
        chi2s = np.array(
            [np.sum(residuals((teff0, feh0, logg0, float(rv))) ** 2) for rv in rv_grid],
            dtype=float,
        )
        rv0_use = float(rv_grid[np.argmin(chi2s)])
        if verbose:
            print("RV init grid best:", rv0_use)
        p0_use = (teff0, feh0, logg0, rv0_use)
    elif rv_init is None:
        p0_use = (teff0, feh0, logg0, rv0)
    else:
        raise ValueError("rv_init must be 'grid' or None.")

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
    k = _sideband_fit_parameter_count(len(segments), sideband_poly_order)
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
        "segment_lsf_fwhm_kms": [
            None if x is None else float(x) for x in segment_fwhm_kms
        ],
        "segment_resolution_R_effective": [
            None if x is None else float(C_KMS / x) for x in segment_fwhm_kms
        ],
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

    Parameters
    ----------
    norm_mode : {"poly", "sideband"}
        Reconstruction path. "poly" delegates to the generic fitter-side
        polynomial reconstruction. "sideband" rebuilds the sideband-normalized
        model and the local multiplicative polynomial used by the recipe fitter.
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
            model_norm_full, _ = normalize_model_sidebands(
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
            segment_fwhm_kms=segment_fwhm_kms,
            phoenix_wave_medium=model_wave_medium,
            model_margin_A=model_margin_A,
            bounds_use_fit_mask=True,
            extrapolate=True,
        )
        for seg, used_mask, model_broad_full in zip(segments, used_masks, model_raw_list):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)
            err_full = np.asarray(seg.err, dtype=float)

            model_norm_full, _ = normalize_model_sidebands(
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


PEPSI_LEGACY_CENTERS_AIR = [6495.0, 6545.0, 6561.0, 8498.0, 8542.0, 8662.0]


def build_pepsi_legacy_windows(centers_air=None, halfwidth_A=10.0):
    centers = PEPSI_LEGACY_CENTERS_AIR if centers_air is None else centers_air
    out = []
    for c in centers:
        c = float(c)
        out.append(("legacy_{0:.1f}".format(c), c - float(halfwidth_A), c + float(halfwidth_A)))
    return out


def convert_air_windows_to_medium(window_defs_air, to_medium):
    to_medium = str(to_medium).strip().lower()

    if to_medium in ("air", "unknown", ""):
        return list(window_defs_air)

    if to_medium != "vacuum":
        raise ValueError("Unsupported wavelength medium: {0}".format(to_medium))

    out = []
    for label, wmin_air, wmax_air in window_defs_air:
        w_air = np.array([float(wmin_air), float(wmax_air)], dtype=float)
        w_new = convert_wavelength_medium(w_air, from_medium="air", to_medium="vacuum")
        out.append((label, float(w_new[0]), float(w_new[1])))

    return out


def apply_pepsi_wave_hypothesis(seg, hypothesis):
    hypothesis = str(hypothesis).strip().lower()

    if hypothesis == "unknown":
        meta = dict(seg.meta)
        meta["wave_medium"] = "unknown"
        return seg.copy(meta=meta, wave_medium="unknown", name=(seg.name or "seg") + "_unknown")

    if hypothesis == "air":
        meta = dict(seg.meta)
        meta["wave_medium"] = "air"
        return seg.copy(meta=meta, wave_medium="air", name=(seg.name or "seg") + "_air")

    if hypothesis == "vacuum":
        meta = dict(seg.meta)
        meta["wave_medium"] = "vacuum"
        return seg.copy(meta=meta, wave_medium="vacuum", name=(seg.name or "seg") + "_vacuum")

    if hypothesis == "air_to_vac":
        meta = dict(seg.meta)
        meta["wave_medium"] = "air"
        assumed_air = seg.copy(
            meta=meta,
            wave_medium="air",
        )
        converted = convert_segment_wavelength_medium(
            assumed_air,
            to_medium="vacuum",
            method="ciddor1996",
        )
        return converted.copy(
            name=(seg.name or "seg") + "_air2vac",
        ).sorted()

    raise ValueError("Unknown wavelength hypothesis: {0}".format(hypothesis))


def build_pepsi_normalized_mask(seg, flux_min=0.2, flux_max=1.1):
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)

    good = np.asarray(seg.mask, dtype=bool)
    good &= np.isfinite(wave) & np.isfinite(flux)
    good &= (flux > float(flux_min)) & (flux < float(flux_max))

    if seg.err is not None:
        err = np.asarray(seg.err, dtype=float)
        good &= np.isfinite(err) & (err > 0)

    return good
    

def build_pepsi_legacy_segments(
    input_segments,
    wave_hypothesis="air",
    centers_air=None,
    halfwidth_A=10.0,
    flux_min=0.2,
    flux_max=1.1,
    window_pad_A=2.0,
):
    window_defs_air = build_pepsi_legacy_windows(
        centers_air=centers_air,
        halfwidth_A=halfwidth_A,
    )

    working_input_segments = []
    fit_segments = []

    for seg0 in input_segments:
        legacy_mask = build_pepsi_normalized_mask(
            seg0,
            flux_min=flux_min,
            flux_max=flux_max,
        )
        seg0 = seg0.copy(mask=legacy_mask)
        seg = apply_pepsi_wave_hypothesis(seg0, wave_hypothesis)
        working_input_segments.append(seg)

        working_window_defs = convert_air_windows_to_medium(
            window_defs_air,
            to_medium=seg.wave_medium,
        )

        wlo = float(np.nanmin(seg.wave))
        whi = float(np.nanmax(seg.wave))
        present_defs = [
            (label, wmin, wmax)
            for label, wmin, wmax in working_window_defs
            if (wmax >= wlo) and (wmin <= whi)
        ]

        if len(present_defs) == 0:
            continue

        seg_windows = make_padded_window_segments(
            seg,
            [(wmin, wmax) for _label, wmin, wmax in present_defs],
            pad=window_pad_A,
            name_prefix="line",
        )

        for sw, win_def in zip(seg_windows, present_defs):
            sw.name = win_def[0]
            sw.meta["source_file"] = seg.meta.get("source_file")
            sw.meta["legacy_window_air"] = tuple(
                x for x in next(w for w in window_defs_air if w[0] == win_def[0])
            )
            sw.meta["legacy_window_working"] = tuple(win_def)
            sw.meta["legacy_window_medium"] = seg.wave_medium

        fit_segments.extend(seg_windows)

    if len(fit_segments) == 0:
        raise ValueError("No PEPSI legacy line windows overlap the supplied segment(s).")

    return working_input_segments, fit_segments, window_defs_air


def make_pepsi_legacy_cache_support_segments(
    input_segments,
    window_defs_air,
    window_pad_A=2.0,
):
    support_segments = []

    for seg in input_segments:
        working_window_defs = convert_air_windows_to_medium(
            window_defs_air,
            to_medium=seg.wave_medium,
        )

        wlo = float(np.nanmin(seg.wave))
        whi = float(np.nanmax(seg.wave))

        present_defs = [
            (label, wmin, wmax)
            for label, wmin, wmax in working_window_defs
            if (wmax >= wlo) and (wmin <= whi)
        ]

        for label, wmin, wmax in present_defs:
            keep = (
                np.isfinite(seg.wave) &
                (seg.wave >= float(wmin) - float(window_pad_A)) &
                (seg.wave <= float(wmax) + float(window_pad_A))
            )

            if not np.any(keep):
                continue

            n_keep = int(np.sum(keep))
            support_segments.append(
                seg.copy(
                    wave=seg.wave[keep],
                    flux=np.ones(n_keep, dtype=float),
                    err=np.ones(n_keep, dtype=float),
                    mask=np.ones(n_keep, dtype=bool),
                    meta=dict(seg.meta),
                    name="cache_support_{0}".format(label),
                )
            )

    if len(support_segments) == 0:
        raise ValueError("No PEPSI legacy cache support windows overlap the supplied segment(s).")

    return support_segments


def evaluate_pepsi_legacy_max_models(
    phoenix_lib,
    segments,
    model_wave_grid,
    model_wave_medium,
    teff,
    feh,
    logg,
    rv_kms,
    rv_bary_kms,
    R,
    model_margin_A,
):
    template_flux = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
    segment_fwhm_kms = [segment_fwhm_kms_from_R(seg, R=R) for seg in segments]
    return build_phoenix_native_models_for_segments(
        segments=segments,
        phoenix_wave_native=model_wave_grid,
        template_flux_native=template_flux,
        rv_kms=float(rv_kms),
        rv_bary_kms=float(rv_bary_kms),
        segment_fwhm_kms=segment_fwhm_kms,
        phoenix_wave_medium=model_wave_medium,
        model_margin_A=model_margin_A,
        bounds_use_fit_mask=True,
        extrapolate=True,
    )


def pepsi_legacy_max_likelihood_terms(seg, model_full, log_err_scale=0.0):
    """
    Return likelihood terms for one window.

    The model is normalized by its maximum on the used pixels in the window,
    matching the old model/max(model) comparison. The errors are scaled by
    10**log_err_scale and used as variances in a Gaussian negative log-likelihood.
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = np.asarray(seg.err, dtype=float)
    model_full = np.asarray(model_full, dtype=float)
    used = np.asarray(seg.mask, dtype=bool)
    used &= np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    used &= np.isfinite(model_full)

    if np.sum(used) < 4:
        return np.inf, 0, np.full_like(model_full, np.nan, dtype=float), used

    mmax = float(np.nanmax(model_full[used]))
    if (not np.isfinite(mmax)) or mmax == 0.0:
        return np.inf, 0, np.full_like(model_full, np.nan, dtype=float), used

    model_norm = model_full / mmax
    sigma = (10.0 ** float(log_err_scale)) * err[used]
    var = sigma * sigma
    resid = flux[used] - model_norm[used]
    nll_terms = resid * resid / var + np.log(2.0 * np.pi * var)
    return float(np.sum(nll_terms)), int(np.sum(used)), model_norm, used
    
    
def segment_fwhm_kms_from_R(seg, R=None):
    if R is None:
        R = getattr(seg, "meta", {}).get("resolution_R", None)
    if R is None:
        return None
    R = float(R)
    if R <= 0:
        return None
    return C_KMS / R


def ensure_phoenix_native_interpolator_for_segments(
    segments,
    phoenix_lib,
    teff_grid,
    feh_grid,
    logg_grid,
    cache_path=None,
    model_margin_A=20.0,
):
    model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
        segments=segments,
        phoenix_lib=phoenix_lib,
        model_margin_A=model_margin_A,
    )

    teff_grid = np.asarray(teff_grid, dtype=float)
    feh_grid = np.asarray(feh_grid, dtype=float)
    logg_grid = np.asarray(logg_grid, dtype=float)

    phoenix_lib.ensure_interpolator(
        wave=model_wave_grid,
        teff_grid=teff_grid,
        feh_grid=feh_grid,
        logg_grid=logg_grid,
        cache_path=cache_path,
        observed_wave_medium=model_wave_medium,
    )

    return model_wave_grid, model_wave_medium


def pick_grid_range(grid, lo=None, hi=None):
    g = np.asarray(grid, dtype=float)
    m = np.ones_like(g, dtype=bool)
    if lo is not None:
        m &= (g >= float(lo))
    if hi is not None:
        m &= (g <= float(hi))
    out = g[m]
    if out.size == 0:
        raise ValueError("Requested PHOENIX grid range is empty.")
    return out
    
    
__all__ = [
    "BALMER_CENTERS_VAC",
    "XSHOOTER_BALMER_WINDOWS",
    "XSHOOTER_NOTEBOOK_CONT_WINDOWS",
    "XSHOOTER_BALMER_CORE_MASK_DEFAULT_A",
    "XSHOOTER_BALMER_CORE_MASK_CONSERVATIVE_A",
    "xshooter_balmer_windows",
    "attach_balmer_metadata",
    "normalize_segment_sidebands",
    "normalize_segments_sidebands",
    "normalize_model_sidebands",
    "make_balmer_core_exclude_mask",
    "fit_phoenix_sideband_symmetric",
    "build_plot_models_for_segments",
    "PEPSI_LEGACY_CENTERS_AIR",
    "build_pepsi_legacy_windows",
    "convert_air_windows_to_medium",
    "apply_pepsi_wave_hypothesis",
    "build_pepsi_normalized_mask",
    "build_pepsi_legacy_segments",
    "make_pepsi_legacy_cache_support_segments",
    "segment_fwhm_kms_from_R",
    "ensure_phoenix_native_interpolator_for_segments",
    "pick_grid_range",
    "evaluate_pepsi_legacy_max_models",
    "pepsi_legacy_max_likelihood_terms",
]
