# Spyctres/phoenix_forward.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .waveutils import convert_wavelength_medium

C_KMS = 299792.458


def _normalize_wave_medium(wave_medium, default="unknown"):
    if wave_medium is None:
        return str(default).lower()
    s = str(wave_medium).strip().lower()
    if s in ("air", "vacuum", "unknown"):
        return s
    return str(default).lower()


def infer_segments_wave_medium(segments, default="unknown"):
    """
    Infer a common wavelength medium from a list of SpectrumSegment objects.

    Returns:
        - "air" or "vacuum" if all segments agree on a recognized medium
        - default otherwise
    """
    media = sorted(set(_normalize_wave_medium(seg.wave_medium, default=default) for seg in segments))
    if len(media) == 1 and media[0] in ("air", "vacuum"):
        return media[0]
    return _normalize_wave_medium(default, default=default)


def fit_bounds_from_segments(segments, use_fit_mask=True):
    """
    Return the min/max wavelength bounds that should define the model support.

    By default this uses only seg.mask == True pixels, not the full segment
    support. This matches the notebook-faithful X-SHOOTER logic.
    """
    los = []
    his = []

    for seg in segments:
        wave = np.asarray(seg.wave, dtype=float)

        if use_fit_mask:
            m = np.asarray(seg.mask, dtype=bool)
        else:
            m = np.isfinite(wave)

        m &= np.isfinite(wave)
        if np.any(m):
            los.append(float(np.min(wave[m])))
            his.append(float(np.max(wave[m])))

    if len(los) == 0:
        raise ValueError("No valid segment wavelengths found for model bounds.")

    return min(los), max(his)


def prepare_phoenix_native_template(
    phoenix_wave_native,
    template_flux_native,
    target_wave_medium,
    phoenix_wave_medium="vacuum",
    wmin=None,
    wmax=None,
    margin_A=200.0,
):
    """
    Prepare a native PHOENIX template before RV shift and convolution.

    Steps:
      1. Convert PHOENIX wavelengths into the target data wavelength medium.
      2. Subset to [wmin - margin_A, wmax + margin_A] if bounds are given.
      3. Return sorted finite arrays.
    """
    wave = np.asarray(phoenix_wave_native, dtype=np.float64).copy()
    flux = np.asarray(template_flux_native, dtype=np.float64).copy()

    src_medium = _normalize_wave_medium(phoenix_wave_medium, default="vacuum")
    dst_medium = _normalize_wave_medium(target_wave_medium, default=src_medium)

    if src_medium in ("air", "vacuum") and dst_medium in ("air", "vacuum") and src_medium != dst_medium:
        wave = convert_wavelength_medium(
            wave,
            from_medium=src_medium,
            to_medium=dst_medium,
        )

    m = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)

    if wmin is not None:
        m &= (wave >= float(wmin) - float(margin_A))
    if wmax is not None:
        m &= (wave <= float(wmax) + float(margin_A))

    wave = wave[m]
    flux = flux[m]

    if len(wave) < 10:
        raise ValueError("Too few PHOENIX points remain after native-template preparation.")

    if not np.all(np.diff(wave) > 0):
        idx = np.argsort(wave)
        wave = wave[idx]
        flux = flux[idx]

    return wave, flux


def doppler_shift_wave(wave_A, rv_kms):
    """
    Apply a non-relativistic Doppler shift to a wavelength array.

    This matches the Gaia21ccu notebook-faithful reference path currently used for
    X-SHOOTER development.
    """
    wave_A = np.asarray(wave_A, dtype=np.float64)
    return wave_A * (1.0 + float(rv_kms) / C_KMS)


def convolve_to_resolution_loglam(wave_A, flux, R):
    """
    Convolve a spectrum with a Gaussian LSF at constant resolving power R
    on a log-lambda grid.

    This is the native-grid broadening step validated against the Gaia21ccu
    X-SHOOTER notebook reference.
    """
    wave_A = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if R is None:
        return flux.copy()

    if not np.all(np.diff(wave_A) > 0):
        idx = np.argsort(wave_A)
        wave_A = wave_A[idx]
        flux = flux[idx]

    m = np.isfinite(wave_A) & (wave_A > 0) & np.isfinite(flux)
    wave_A = wave_A[m]
    flux = flux[m]

    if len(wave_A) < 5:
        return flux.copy()

    loglam = np.log(wave_A)
    dloglam = np.median(np.diff(loglam))
    if (not np.isfinite(dloglam)) or (dloglam <= 0):
        return flux.copy()

    sigma_v = C_KMS / (float(R) * 2.3548200450309493)
    sigma_pix = (sigma_v / C_KMS) / dloglam

    if sigma_pix < 0.3:
        return flux.copy()

    return gaussian_filter1d(flux, sigma_pix, mode="nearest")


def resample_flux(w_src, f_src, w_tgt, extrapolate=True):
    """
    Resample a model spectrum onto a target wavelength grid.

    The default uses linear extrapolation to match the validated notebook-scan
    reference. In normal use, a sufficient margin_A should make extrapolation
    unnecessary on fitted pixels.
    """
    w_src = np.asarray(w_src, dtype=np.float64)
    f_src = np.asarray(f_src, dtype=np.float64)
    w_tgt = np.asarray(w_tgt, dtype=np.float64)

    m = np.isfinite(w_src) & np.isfinite(f_src)
    w_src = w_src[m]
    f_src = f_src[m]

    if len(w_src) < 4:
        return np.full_like(w_tgt, np.nan, dtype=float)

    if not np.all(np.diff(w_src) > 0):
        idx = np.argsort(w_src)
        w_src = w_src[idx]
        f_src = f_src[idx]

    fill_value = "extrapolate" if extrapolate else np.nan
    f = interp1d(
        w_src,
        f_src,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    return np.asarray(f(w_tgt), dtype=float)


def build_phoenix_native_model_to_wave(
    wave_target,
    phoenix_wave_native,
    template_flux_native,
    rv_kms=0.0,
    rv_bary_kms=0.0,
    R=None,
    target_wave_medium="vacuum",
    phoenix_wave_medium="vacuum",
    wmin=None,
    wmax=None,
    model_margin_A=200.0,
    extrapolate=True,
):
    """
    Build a PHOENIX model on a target observed wavelength grid using the
    native-grid forward-model order:

      convert medium -> subset with margin -> Doppler shift -> convolve ->
      resample to target grid
    """
    w_native, f_native = prepare_phoenix_native_template(
        phoenix_wave_native=phoenix_wave_native,
        template_flux_native=template_flux_native,
        target_wave_medium=target_wave_medium,
        phoenix_wave_medium=phoenix_wave_medium,
        wmin=wmin,
        wmax=wmax,
        margin_A=model_margin_A,
    )

    rv_tot = float(rv_bary_kms) + float(rv_kms)
    w_shift = doppler_shift_wave(w_native, rv_tot)

    m = np.isfinite(w_shift) & (w_shift > 0) & np.isfinite(f_native)
    w_shift = w_shift[m]
    f_native = f_native[m]

    f_conv = convolve_to_resolution_loglam(w_shift, f_native, R)

    return resample_flux(
        w_src=w_shift,
        f_src=f_conv,
        w_tgt=wave_target,
        extrapolate=extrapolate,
    )


def build_phoenix_native_models_for_segments(
    segments,
    phoenix_wave_native,
    template_flux_native,
    rv_kms=0.0,
    rv_bary_kms=0.0,
    R=None,
    phoenix_wave_medium="vacuum",
    model_margin_A=200.0,
    bounds_use_fit_mask=True,
    extrapolate=True,
    return_native=False,
):
    """
    Build one PHOENIX model array per segment using the validated native-grid
    forward-model order.

    This function is intentionally continuum-agnostic. It returns only the
    physical template prediction on each segment grid. Continuum nuisance terms
    should be handled by fitting.py.
    """
    target_wave_medium = infer_segments_wave_medium(
        segments,
        default=phoenix_wave_medium,
    )

    wmin, wmax = fit_bounds_from_segments(
        segments,
        use_fit_mask=bounds_use_fit_mask,
    )

    w_native, f_native = prepare_phoenix_native_template(
        phoenix_wave_native=phoenix_wave_native,
        template_flux_native=template_flux_native,
        target_wave_medium=target_wave_medium,
        phoenix_wave_medium=phoenix_wave_medium,
        wmin=wmin,
        wmax=wmax,
        margin_A=model_margin_A,
    )

    rv_tot = float(rv_bary_kms) + float(rv_kms)
    w_shift = doppler_shift_wave(w_native, rv_tot)

    m = np.isfinite(w_shift) & (w_shift > 0) & np.isfinite(f_native)
    w_shift = w_shift[m]
    f_native = f_native[m]

    f_conv = convolve_to_resolution_loglam(w_shift, f_native, R)

    model_list = [
        resample_flux(
            w_src=w_shift,
            f_src=f_conv,
            w_tgt=np.asarray(seg.wave, dtype=float),
            extrapolate=extrapolate,
        )
        for seg in segments
    ]

    if not return_native:
        return model_list

    return model_list, {
        "target_wave_medium": target_wave_medium,
        "wmin_fit": float(wmin),
        "wmax_fit": float(wmax),
        "wave_native_prepared": w_native,
        "wave_shifted": w_shift,
        "flux_convolved": f_conv,
    }


__all__ = [
    "C_KMS",
    "infer_segments_wave_medium",
    "fit_bounds_from_segments",
    "prepare_phoenix_native_template",
    "doppler_shift_wave",
    "convolve_to_resolution_loglam",
    "resample_flux",
    "build_phoenix_native_model_to_wave",
    "build_phoenix_native_models_for_segments",
]
