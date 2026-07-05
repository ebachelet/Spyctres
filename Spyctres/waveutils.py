"""
Wavelength and velocity-conversion utilities for spectral fitting.

This module provides:
- relativistic wavelength shifts for radial velocities
- air <-> vacuum wavelength conversion using the Ciddor relation
- explicit wavelength-medium conversion helpers

Notes
-----
PHOENIX HiRes spectra are tabulated on a vacuum wavelength grid. When
comparing PHOENIX templates to observed spectra, the wavelength medium
(air or vacuum) should therefore be treated explicitly rather than assumed.

References
----------
Husser, T.-O., Wende-von Berg, S., Dreizler, S., Homeier, D., Reiners, A.,
Barman, T., & Hauschildt, P. H. (2013), A new extensive library of PHOENIX
stellar atmospheres and synthetic spectra, Astronomy & Astrophysics, 553, A6,
https://doi.org/10.1051/0004-6361/201219058.

Ciddor, P. E. (1996), Refractive index of air: new equations for the visible
and near infrared, Applied Optics, 35, 1566-1573,
https://doi.org/10.1364/AO.35.001566.

Morton, D. C. (2000), Atomic Data for Resonance Absorption Lines. II,
https://doi.org/10.1086/317349; Birch, K. P. & Downs, M. J. (1994),
https://doi.org/10.1088/0026-1394/31/4/006. These support the VALD3
vacuum-to-air convention. VALD3's reversible air-to-vacuum inverse is
documented at https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion.
"""
import numpy as np

C_KMS = 299792.458
AIR_VACUUM_MIN_A = 2000.0

def _as_float_array(x):
    """Return x as a NumPy float array."""
    return np.asarray(x, dtype=float)


def doppler_factor(v_kms):
    """
    Relativistic Doppler factor for wavelength shifts.

    Positive velocity means redshift:
        lambda_shifted = lambda_rest * doppler_factor(v_kms)
    """
    beta = float(v_kms) / C_KMS
    if abs(beta) >= 1.0:
        raise ValueError("Absolute velocity must be smaller than the speed of light.")
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def shift_wavelength_velocity(wave, v_kms):
    """
    Shift wavelengths by a radial velocity using the relativistic Doppler factor.

    ``phoenix_forward.doppler_shift_wave`` intentionally retains the classical
    factor for notebook-reference reproducibility. Do not mix the two factors
    within one RV derivation.

    Parameters
    ----------
    wave : array-like
        Input wavelength array.
    v_kms : float
        Radial velocity in km/s. Positive values redshift the wavelengths.

    Returns
    -------
    ndarray
        Shifted wavelength array.
    """
    wave = _as_float_array(wave)
    return wave * doppler_factor(v_kms)



def resample_flux_with_velocity_shift_observed_grid(wave, flux, rv_kms):
    """
    Apply an RV shift to a model already sampled on the observed wavelength grid.

    Convention
    ----------
    Positive ``rv_kms`` redshifts template/model features. The returned flux is
    evaluated on the original input wavelength grid. This helper is intended for
    observed-grid PHOENIX fitting paths and deliberately does not modify the
    legacy ``Spyctres.velocity_correction`` convention.

    Notes
    -----
    If the rest-frame model is sampled as ``flux(wave_rest)``, then a positive
    radial velocity gives ``wave_obs = wave_rest * doppler_factor(rv_kms)``.
    Therefore, to return model flux on a fixed observed wavelength grid, we
    sample the rest-frame model at ``wave / doppler_factor(rv_kms)``.
    """
    from scipy.interpolate import interp1d

    wave = _as_float_array(wave)
    flux = _as_float_array(flux)

    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have the same shape.")

    if wave.ndim != 1:
        raise ValueError("wave and flux must be 1D arrays.")

    good = np.isfinite(wave) & np.isfinite(flux)
    if np.sum(good) < 2:
        return np.full_like(wave, np.nan, dtype=float)

    w = wave[good]
    f = flux[good]

    if not np.all(np.diff(w) > 0):
        idx = np.argsort(w)
        w = w[idx]
        f = f[idx]

    sample_wave = wave / doppler_factor(float(rv_kms))

    interpolator = interp1d(
        w,
        f,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(interpolator(sample_wave), dtype=float)

def _ciddor_factor_from_vacuum_angstrom(wave_vac):
    """
    Refractive index factor f = lambda_vac / lambda_air
    Uses the Ciddor (1996) relation as quoted by Husser et al. (2013),
    valid for lambda > 2000 Angstrom.
    """
    wave_vac = _as_float_array(wave_vac)
    sigma2 = (1.0e4 / wave_vac) ** 2
    f = (
        1.0
        + 0.05792105 / (238.0185 - sigma2)
        + 0.00167917 / (57.362 - sigma2)
    )
    return f


def vacuum_to_air_ciddor(wave_vac):
    """
    Convert vacuum wavelengths to air wavelengths using the
    Ciddor (1996) formula quoted in Husser et al. (2013).

    Parameters
    ----------
    wave_vac : array-like
        Vacuum wavelengths in Angstrom.

    Returns
    -------
    ndarray
        Air wavelengths in Angstrom.
    """
    wave_vac = _as_float_array(wave_vac)
    converted = wave_vac.copy()
    use = wave_vac > AIR_VACUUM_MIN_A
    converted[use] = wave_vac[use] / _ciddor_factor_from_vacuum_angstrom(
        wave_vac[use]
    )
    return converted


def air_to_vacuum_ciddor(wave_air, n_iter=3):
    """
    Convert air wavelengths to vacuum wavelengths by inverting
    the Ciddor relation iteratively.

    Parameters
    ----------
    wave_air : array-like
        Air wavelengths in Angstrom.
    n_iter : int, optional
        Number of fixed-point iterations. Default is 3.

    Returns
    -------
    ndarray
        Vacuum wavelengths in Angstrom.
    """
    wave_air = _as_float_array(wave_air)
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    wave_vac = wave_air.copy()
    use = wave_air > AIR_VACUUM_MIN_A
    for _ in range(n_iter):
        wave_vac[use] = wave_air[use] * _ciddor_factor_from_vacuum_angstrom(
            wave_vac[use]
        )
    return wave_vac


def vacuum_to_air_vald(wave_vac):
    """Convert vacuum to air wavelengths using the VALD3/Morton convention.

    The Morton (2000) relation follows Birch & Downs (1994). Primary papers:
    https://doi.org/10.1086/317349 and
    https://doi.org/10.1088/0026-1394/31/4/006
    """
    wave_vac = _as_float_array(wave_vac)
    converted = wave_vac.copy()
    use = wave_vac > AIR_VACUUM_MIN_A
    sigma2 = (1.0e4 / wave_vac[use]) ** 2
    refractive_index = (
        1.0
        + 0.0000834254
        + 0.02406147 / (130.0 - sigma2)
        + 0.00015998 / (38.9 - sigma2)
    )
    converted[use] = wave_vac[use] / refractive_index
    return converted


def air_to_vacuum_vald(wave_air):
    """Convert air to vacuum wavelengths with VALD3's reversible inverse.

    These are the coefficients historically embedded in ``get_element_lines``.
    The exact Piskunov inverse and its validity range are documented by VALD3:
    https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    """
    wave_air = _as_float_array(wave_air)
    converted = wave_air.copy()
    use = wave_air > AIR_VACUUM_MIN_A
    sigma2 = (1.0e4 / wave_air[use]) ** 2
    refractive_index = (
        1.0
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - sigma2)
        + 0.0001599740894897 / (38.92568793293 - sigma2)
    )
    converted[use] = wave_air[use] * refractive_index
    return converted


def _normalize_air_vacuum_method(method):
    method = str(method).strip().lower()
    aliases = {
        "ciddor": "ciddor1996",
        "ciddor1996": "ciddor1996",
        "vald": "vald3",
        "vald3": "vald3",
        "morton": "vald3",
        "morton2000": "vald3",
    }
    if method not in aliases:
        raise ValueError(
            "Unknown wavelength conversion method '{0}'. Supported methods are "
            "'ciddor1996' and 'vald3'.".format(method)
        )
    return aliases[method]


def convert_wavelength_medium(
    wave,
    from_medium,
    to_medium,
    method="ciddor1996",
):
    """
    Convert wavelength medium between 'air', 'vacuum', and 'unknown'.
    
    Conversions involving 'unknown' are rejected deliberately, because the
    wavelength medium must be specified explicitly for reliable spectral fitting.
    
    Parameters
    ----------
    wave : array-like
        Input wavelength array in Angstrom.
    from_medium : str
        'air', 'vacuum', or 'unknown'
    to_medium : str
        'air', 'vacuum', or 'unknown'
    method : {'ciddor1996', 'vald3'}
        Explicit refractive-index convention. The default preserves the current
        PHOENIX workflow; ``vald3`` preserves the legacy line-list path.

    Returns
    -------
    ndarray
        Converted wavelength array.

    Raises
    ------
    ValueError
        If the conversion is unsupported or ambiguous.
    """
    wave = _as_float_array(wave)
    from_medium = str(from_medium).lower()
    to_medium = str(to_medium).lower()
    method = _normalize_air_vacuum_method(method)

    if from_medium == to_medium:
        return wave.copy()

    if "unknown" in (from_medium, to_medium):
        raise ValueError(
            "Cannot convert wavelength medium when from/to is 'unknown'."
        )

    if from_medium == "vacuum" and to_medium == "air":
        if method == "ciddor1996":
            return vacuum_to_air_ciddor(wave)
        return vacuum_to_air_vald(wave)

    if from_medium == "air" and to_medium == "vacuum":
        if method == "ciddor1996":
            return air_to_vacuum_ciddor(wave)
        return air_to_vacuum_vald(wave)

    raise ValueError(
        "Unsupported wavelength-medium conversion: {0} -> {1}".format(
            from_medium, to_medium
        )
    )


def convert_segment_wavelength_medium(segment, to_medium, method="ciddor1996"):
    """Return a converted segment copy with wavelength-conversion provenance."""
    from_medium = str(segment.wave_medium).strip().lower()
    to_medium = str(to_medium).strip().lower()
    method = _normalize_air_vacuum_method(method)
    converted_wave = convert_wavelength_medium(
        segment.wave,
        from_medium=from_medium,
        to_medium=to_medium,
        method=method,
    )

    resolution = getattr(segment, "resolution", None)
    if resolution is not None and resolution.mode == "tabulated":
        from .io import ResolutionDescriptor

        resolution = ResolutionDescriptor(
            quantity=resolution.quantity,
            mode="tabulated",
            wave_A=convert_wavelength_medium(
                resolution.wave_A,
                from_medium=from_medium,
                to_medium=to_medium,
                method=method,
            ),
            values=resolution.values,
            source=resolution.source,
        )

    meta = dict(segment.meta)
    history = list(meta.get("wavelength_conversions", []))
    history.append(
        {
            "from_medium": from_medium,
            "to_medium": to_medium,
            "method": method,
            "air_vacuum_boundary_A": AIR_VACUUM_MIN_A,
        }
    )
    meta["wavelength_conversions"] = history
    meta["wave_medium"] = to_medium
    meta["resolution"] = (
        None if resolution is None else resolution.to_metadata()
    )

    return segment.copy(
        wave=converted_wave,
        meta=meta,
        wave_medium=to_medium,
        resolution=resolution,
    )
