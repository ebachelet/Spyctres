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
doi:10.1051/0004-6361/201219058.

Ciddor, P. E. (1996), Refractive index of air: new equations for the visible
and near infrared, Applied Optics, 35, 1566-1573.
"""
import numpy as np

C_KMS = 299792.458

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
    return wave_vac / _ciddor_factor_from_vacuum_angstrom(wave_vac)


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
    wave_vac = wave_air.copy()
    for _ in range(int(n_iter)):
        wave_vac = wave_air * _ciddor_factor_from_vacuum_angstrom(wave_vac)
    return wave_vac


def convert_wavelength_medium(wave, from_medium, to_medium):
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

    if from_medium == to_medium:
        return wave.copy()

    if "unknown" in (from_medium, to_medium):
        raise ValueError(
            "Cannot convert wavelength medium when from/to is 'unknown'."
        )

    if from_medium == "vacuum" and to_medium == "air":
        return vacuum_to_air_ciddor(wave)

    if from_medium == "air" and to_medium == "vacuum":
        return air_to_vacuum_ciddor(wave)

    raise ValueError(
        "Unsupported wavelength-medium conversion: {0} -> {1}".format(
            from_medium, to_medium
        )
    )
