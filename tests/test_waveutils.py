import numpy as np

from Spyctres.phoenix_forward import (
    build_phoenix_native_model_to_wave,
    convolve_to_resolution_loglam,
)
from Spyctres.waveutils import (
    air_to_vacuum_ciddor,
    doppler_factor,
    shift_wavelength_velocity,
    vacuum_to_air_ciddor,
)


def test_positive_velocity_redshifts_wavelengths():
    wave = np.array([5000.0, 6000.0])
    shifted = shift_wavelength_velocity(wave, 25.0)

    assert np.all(shifted > wave)
    assert np.allclose(shifted, wave * doppler_factor(25.0))


def test_air_vacuum_round_trip():
    wave_air = np.array([3000.0, 5000.0, 6500.0, 9000.0])
    wave_vacuum = air_to_vacuum_ciddor(wave_air)
    recovered_air = vacuum_to_air_ciddor(wave_vacuum)

    assert np.all(wave_vacuum > wave_air)
    assert np.allclose(recovered_air, wave_air, rtol=0.0, atol=1.0e-8)


def test_no_broadening_is_exact_noop():
    wave = np.linspace(5000.0, 5010.0, 101)
    flux = 1.0 - 0.4 * np.exp(-0.5 * ((wave - 5005.0) / 0.08) ** 2)

    broadened = convolve_to_resolution_loglam(wave, flux)

    assert np.array_equal(broadened, flux)
    assert broadened is not flux


def test_broadening_rejects_two_width_specifications():
    wave = np.linspace(5000.0, 5010.0, 101)
    flux = np.ones_like(wave)

    with np.testing.assert_raises(ValueError):
        convolve_to_resolution_loglam(wave, flux, R=50_000.0, fwhm_kms=6.0)


def test_stellar_and_barycentric_velocity_terms_add_explicitly():
    wave = np.linspace(4990.0, 5010.0, 4001)
    flux = 1.0 - 0.5 * np.exp(-0.5 * ((wave - 5000.0) / 0.08) ** 2)
    target = np.linspace(4992.0, 5008.0, 1601)

    split_terms = build_phoenix_native_model_to_wave(
        target,
        wave,
        flux,
        rv_kms=12.0,
        rv_bary_kms=-3.0,
        model_margin_A=0.0,
    )
    combined_term = build_phoenix_native_model_to_wave(
        target,
        wave,
        flux,
        rv_kms=9.0,
        rv_bary_kms=0.0,
        model_margin_A=0.0,
    )

    assert np.allclose(split_terms, combined_term, rtol=0.0, atol=1.0e-12)
