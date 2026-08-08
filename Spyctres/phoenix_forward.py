# Spyctres/phoenix_forward.py
import warnings

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.interpolate import CubicSpline, interp1d

from .waveutils import convert_wavelength_medium, C_KMS


GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493

def _coerce_segment_list(segments):
    """
    Normalize a segment-like input to a non-empty plain list.

    Accepted inputs
    ---------------
    - a single segment-like object with a ``wave`` attribute
    - a list/tuple of such objects
    - a collection-like object with a ``segments`` attribute
    """
    if hasattr(segments, "segments"):
        seg_list = list(getattr(segments, "segments"))
    elif isinstance(segments, (list, tuple)):
        seg_list = list(segments)
    else:
        seg_list = [segments]

    if len(seg_list) == 0:
        raise ValueError("No segments were provided.")

    for i, seg in enumerate(seg_list):
        if not hasattr(seg, "wave"):
            raise TypeError("Item {0} does not look like a spectrum segment.".format(i))

    return seg_list

def _normalize_wave_medium(wave_medium, default="unknown"):
    if wave_medium is None:
        return str(default).lower()
    s = str(wave_medium).strip().lower()
    if s in ("air", "vacuum", "unknown"):
        return s
    return str(default).lower()


def infer_segments_wave_medium(segments, default="unknown"):
    """
    Infer a common wavelength medium from a segment list or collection.

    Returns
    -------
    str
        "air" or "vacuum" if all segments agree on a recognized medium,
        otherwise ``default`` normalized through _normalize_wave_medium().
    """
    segments = _coerce_segment_list(segments)
    media = sorted(set(_normalize_wave_medium(seg.wave_medium, default=default) for seg in segments))
    if len(media) == 1 and media[0] in ("air", "vacuum"):
        return media[0]
    return _normalize_wave_medium(default, default=default)


def fit_bounds_from_segments(segments, use_fit_mask=True):
    """
    Return the min/max wavelength bounds that should define the model support.

    Parameters
    ----------
    segments : segment list or collection
    use_fit_mask : bool, default=True
        If True, only seg.mask-selected pixels contribute to the bounds.

    Returns
    -------
    (wmin, wmax) : tuple[float, float]
    """
    segments = _coerce_segment_list(segments)

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


def build_native_interp_wave_grid_for_segments(
    segments,
    phoenix_lib,
    model_margin_A=200.0,
):
    """
    Build the native PHOENIX wavelength grid needed for native_interp modeling.

    The returned grid is a clipped subset of phoenix_lib.phoenix_wave, converted
    into the common wavelength medium of the supplied segments and restricted
    to the fit-mask bounds plus a margin.

    This helper deliberately uses phoenix_lib.phoenix_wave, not phoenix_lib.wave.
    phoenix_lib.wave is the current interpolator grid and may already be an
    observed or cached model grid.
    """
    segments = _coerce_segment_list(segments)

    if getattr(phoenix_lib, "phoenix_wave", None) is None:
        raise ValueError("phoenix_lib.phoenix_wave is not initialized.")

    model_wave_medium = infer_segments_wave_medium(
        segments,
        default=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
    )

    fit_min, fit_max = fit_bounds_from_segments(
        segments,
        use_fit_mask=True,
    )

    phoenix_wave = np.asarray(phoenix_lib.phoenix_wave, dtype=float)

    model_wave_grid, _dummy_flux = prepare_phoenix_native_template(
        phoenix_wave_native=phoenix_wave,
        template_flux_native=np.ones_like(phoenix_wave, dtype=float),
        target_wave_medium=model_wave_medium,
        phoenix_wave_medium=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
        wmin=float(fit_min),
        wmax=float(fit_max),
        margin_A=float(model_margin_A),
    )

    return model_wave_grid, model_wave_medium
    
    
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
    X-SHOOTER development. See ``waveutils.shift_wavelength_velocity`` for the
    full relativistic factor. The difference is about 5e-7 at 300 km/s; this
    classical form remains here only to preserve the validated reference path.
    """
    wave_A = np.asarray(wave_A, dtype=np.float64)
    return wave_A * (1.0 + float(rv_kms) / C_KMS)


def resolve_gaussian_lsf_fwhm_kms(R=None, fwhm_kms=None):
    """Resolve a Gaussian LSF specification to a velocity FWHM in km/s."""
    if (R is not None) and (fwhm_kms is not None):
        raise ValueError("Provide only one of R or fwhm_kms, not both.")

    if R is None and fwhm_kms is None:
        return None

    if fwhm_kms is None:
        R = float(R)
        if not np.isfinite(R) or R <= 0:
            raise ValueError("R must be finite and > 0.")
        return C_KMS / R

    fwhm_kms = float(fwhm_kms)
    if not np.isfinite(fwhm_kms) or fwhm_kms <= 0:
        raise ValueError("fwhm_kms must be finite and > 0.")
    return fwhm_kms


def _pixel_edges_from_centers(x):
    """Return pixel edges for a strictly increasing 1D coordinate array."""
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def _rebin_flux_density_piecewise_constant(old_edges, old_flux, new_edges):
    """Flux-conserving rebin of pixel-averaged samples onto new pixel edges."""
    cumulative = np.concatenate(
        ([0.0], np.cumsum(old_flux * np.diff(old_edges), dtype=float))
    )
    cumulative_new = np.interp(new_edges, old_edges, cumulative)
    return np.diff(cumulative_new) / np.diff(new_edges)


def convolve_to_resolution_loglam(
    wave_A,
    flux,
    R=None,
    fwhm_kms=None,
    padding_sigma=6.0,
):
    """
    Convolve a spectrum with a Gaussian LSF on a uniform log-lambda grid.

    The analytic Fourier treatment follows Cappellari (2017):
    https://doi.org/10.1093/mnras/stw3020

    The log-grid velocity step is set by the target LSF with approximately
    one pixel per Gaussian sigma. Convolution uses the analytic Fourier
    transform of the Gaussian, avoiding a directly sampled kernel near the
    Nyquist limit.

    The returned array always has the same shape and ordering as the input flux.
    Exactly one of ``R`` or ``fwhm_kms`` may be supplied. If both are None,
    the input flux is returned unchanged. ``padding_sigma`` controls the FFT
    edge padding in Gaussian-sigma units and must be at least 3.
    """
    wave_A = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if wave_A.shape != flux.shape:
        raise ValueError("wave_A and flux must have the same shape.")

    if wave_A.ndim != 1:
        raise ValueError("wave_A and flux must be 1D arrays.")

    fwhm_kms = resolve_gaussian_lsf_fwhm_kms(R=R, fwhm_kms=fwhm_kms)
    if fwhm_kms is None:
        return flux.copy()

    padding_sigma = float(padding_sigma)
    if not np.isfinite(padding_sigma) or padding_sigma < 3.0:
        raise ValueError("padding_sigma must be finite and >= 3.")

    good = np.isfinite(wave_A) & (wave_A > 0) & np.isfinite(flux)
    if np.sum(good) < 5:
        return flux.copy()

    w_good = wave_A[good]
    f_good = flux[good]

    order = np.argsort(w_good)
    w_sorted = w_good[order]
    f_sorted = f_good[order]

    loglam = np.log(w_sorted)
    dlog_input = np.diff(loglam)
    if np.any(dlog_input <= 0):
        raise ValueError("Valid wavelengths must be unique.")

    sigma_v = fwhm_kms / GAUSSIAN_FWHM_TO_SIGMA
    median_dv_input = C_KMS * float(np.median(dlog_input))
    if median_dv_input > sigma_v:
        warnings.warn(
            "Input wavelength sampling is coarser than the requested Gaussian "
            "LSF sigma; the broadened result may be undersampled.",
            UserWarning,
            stacklevel=2,
        )

    input_edges = _pixel_edges_from_centers(loglam)
    target_dlog = sigma_v / C_KMS
    log_span = float(input_edges[-1] - input_edges[0])
    n_bins = max(2, int(np.ceil(log_span / target_dlog)))
    uniform_edges = np.linspace(input_edges[0], input_edges[-1], n_bins + 1)
    log_uniform = 0.5 * (uniform_edges[:-1] + uniform_edges[1:])
    dlog_uniform = log_span / n_bins

    # Flux-conserving rebinning averages over a top-hat pixel whose variance is
    # dv**2 / 12. Subtract that known pixel response in quadrature so the total
    # effective broadening remains the user-requested Gaussian sigma.
    dv_uniform = C_KMS * dlog_uniform
    pixel_sigma_v = dv_uniform / np.sqrt(12.0)
    kernel_sigma_v = np.sqrt(max(0.0, sigma_v**2 - pixel_sigma_v**2))
    sigma_pix = (kernel_sigma_v / C_KMS) / dlog_uniform

    f_uniform = _rebin_flux_density_piecewise_constant(
        input_edges,
        f_sorted,
        uniform_edges,
    )

    # Edge-value padding preserves the previous nearest-edge behavior while
    # placing the periodic FFT seam far enough away to avoid wraparound.
    pad_min = max(1, int(np.ceil(padding_sigma * sigma_pix)))
    n_fft = next_fast_len(len(f_uniform) + 2 * pad_min)
    pad_left = pad_min
    pad_right = n_fft - len(f_uniform) - pad_left
    f_padded = np.pad(f_uniform, (pad_left, pad_right), mode="edge")

    frequency = rfftfreq(n_fft)
    gaussian_ft = np.exp(-0.5 * (2.0 * np.pi * sigma_pix * frequency) ** 2)
    f_conv_padded = irfft(rfft(f_padded) * gaussian_ft, n=n_fft)
    f_conv_uniform = f_conv_padded[pad_left:pad_left + len(f_uniform)]

    # Cubic reconstruction is materially more accurate than linear/PCHIP when
    # returning a Nyquist-sampled broadened profile to the original grid. The
    # physical range guard below removes the tiny edge overshoot a cubic can
    # otherwise introduce.
    return_interp = CubicSpline(
        log_uniform,
        f_conv_uniform,
        bc_type="natural",
        extrapolate=True,
    )
    f_conv_sorted = np.asarray(return_interp(loglam), dtype=float)
    f_conv_sorted = np.clip(
        f_conv_sorted,
        float(np.min(f_sorted)),
        float(np.max(f_sorted)),
    )

    f_conv_good = np.empty_like(f_good)
    f_conv_good[order] = f_conv_sorted

    out = flux.copy()
    out[good] = f_conv_good
    return out
    

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
    fwhm_kms=None,
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

    f_conv = convolve_to_resolution_loglam(w_shift, f_native, R=R, fwhm_kms=fwhm_kms)

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
    fwhm_kms=None,
    segment_fwhm_kms=None,
    phoenix_wave_medium="vacuum",
    model_margin_A=200.0,
    bounds_use_fit_mask=True,
    extrapolate=True,
    return_native=False,
):
    """
    Build one PHOENIX model array per segment using the native-grid
    forward-model order.

    This function is continuum-agnostic. It returns only the physical template
    prediction on each segment grid. Continuum nuisance terms should be handled
    by higher-level fitting code.

    Parameters
    ----------
    segments : segment list or collection
    R, fwhm_kms : float, optional
        Global instrumental broadening specification. Exactly one may be
        supplied. Ignored if ``segment_fwhm_kms`` is provided.
    segment_fwhm_kms : sequence, optional
        Per-segment Gaussian LSF FWHM values in km/s. If supplied, must match
        the number of kept segments.
    """
    segments = _coerce_segment_list(segments)

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

    if segment_fwhm_kms is None:
        resolved_fwhm = resolve_gaussian_lsf_fwhm_kms(
            R=R,
            fwhm_kms=fwhm_kms,
        )
        segment_fwhm_kms = [resolved_fwhm] * len(segments)
    else:
        if len(segment_fwhm_kms) != len(segments):
            raise ValueError("segment_fwhm_kms must match the number of segments.")
        segment_fwhm_kms = [
            resolve_gaussian_lsf_fwhm_kms(fwhm_kms=x)
            for x in segment_fwhm_kms
        ]

    conv_cache = {}
    model_list = []

    for seg, seg_fwhm in zip(segments, segment_fwhm_kms):
        key = None if seg_fwhm is None else float(seg_fwhm)
        if key not in conv_cache:
            conv_cache[key] = convolve_to_resolution_loglam(
                w_shift,
                f_native,
                fwhm_kms=key,
            )

        model_list.append(
            resample_flux(
                w_src=w_shift,
                f_src=conv_cache[key],
                w_tgt=np.asarray(seg.wave, dtype=float),
                extrapolate=extrapolate,
            )
        )

    if not return_native:
        return model_list

    return model_list, {
        "target_wave_medium": target_wave_medium,
        "wmin_fit": float(wmin),
        "wmax_fit": float(wmax),
        "wave_native_prepared": w_native,
        "wave_shifted": w_shift,
        "segment_fwhm_kms": segment_fwhm_kms,
    }
