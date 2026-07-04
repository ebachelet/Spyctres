import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from Spyctres.phoenix_forward import (
    _rebin_flux_density_piecewise_constant,
    build_phoenix_native_model_to_wave,
    convolve_to_resolution_loglam,
    resolve_gaussian_lsf_fwhm_kms,
)
from Spyctres.waveutils import (
    AIR_VACUUM_MIN_A,
    air_to_vacuum_ciddor,
    air_to_vacuum_vald,
    convert_segment_wavelength_medium,
    convert_wavelength_medium,
    doppler_factor,
    shift_wavelength_velocity,
    vacuum_to_air_ciddor,
    vacuum_to_air_vald,
)
from Spyctres.io import ResolutionDescriptor, SpectrumSegment


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


def test_vald_reference_values_match_documented_coefficients():
    wave_air = np.array([2000.1, 5000.0, 10000.0, 100000.0])
    expected_vacuum = np.array(
        [2000.748103141021, 5001.394848638070, 10002.741686781708, 100027.26415149611]
    )
    wave_vacuum = np.array([2000.1, 5000.0, 10000.0, 100000.0])
    expected_air = np.array(
        [1999.452009425855, 4998.605522013399, 9997.259056168934, 99972.74327898231]
    )

    assert np.allclose(
        air_to_vacuum_vald(wave_air),
        expected_vacuum,
        rtol=0.0,
        atol=5.0e-11,
    )
    assert np.allclose(
        vacuum_to_air_vald(wave_vacuum),
        expected_air,
        rtol=0.0,
        atol=5.0e-11,
    )


@pytest.mark.parametrize("method", ["ciddor1996", "vald3"])
def test_air_vacuum_boundary_and_round_trip(method):
    wave_air = np.array([1500.0, AIR_VACUUM_MIN_A, 2000.1, 5000.0, 10000.0])
    wave_vacuum = convert_wavelength_medium(
        wave_air,
        from_medium="air",
        to_medium="vacuum",
        method=method,
    )
    recovered = convert_wavelength_medium(
        wave_vacuum,
        from_medium="vacuum",
        to_medium="air",
        method=method,
    )

    assert np.array_equal(wave_vacuum[:2], wave_air[:2])
    assert np.all(wave_vacuum[2:] > wave_air[2:])
    assert np.allclose(recovered, wave_air, rtol=0.0, atol=2.0e-8)


def test_conversion_method_is_explicit_and_validated():
    wave = np.array([5000.0])
    ciddor = convert_wavelength_medium(wave, "air", "vacuum")
    explicit_ciddor = convert_wavelength_medium(
        wave, "air", "vacuum", method="ciddor1996"
    )
    vald = convert_wavelength_medium(wave, "air", "vacuum", method="vald3")

    assert np.array_equal(ciddor, explicit_ciddor)
    assert not np.array_equal(ciddor, vald)
    with pytest.raises(ValueError, match="Unknown wavelength conversion method"):
        convert_wavelength_medium(wave, "air", "air", method="unspecified")

    with pytest.raises(ValueError, match="n_iter"):
        air_to_vacuum_ciddor(wave, n_iter=0)


def test_segment_conversion_preserves_data_and_converts_tabulated_lsf_grid():
    resolution = ResolutionDescriptor(
        quantity="sigma_kms",
        mode="tabulated",
        wave_A=[4000.0, 5000.0, 6000.0],
        values=[13.0, 11.0, 12.0],
        source="test",
    )
    segment = SpectrumSegment(
        wave=[4000.0, 5000.0, 6000.0],
        flux=[1.0, 0.8, 1.1],
        err=[0.1, 0.1, 0.2],
        mask=[True, False, True],
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        resolution=resolution,
    )

    converted = convert_segment_wavelength_medium(
        segment,
        to_medium="vacuum",
        method="vald3",
    )

    assert converted is not segment
    assert np.array_equal(segment.wave, [4000.0, 5000.0, 6000.0])
    assert np.all(converted.wave > segment.wave)
    assert np.array_equal(converted.flux, segment.flux)
    assert np.array_equal(converted.err, segment.err)
    assert np.array_equal(converted.mask, segment.mask)
    assert converted.observer_frame == "barycentric"
    assert converted.stellar_rest_status == "observed"
    assert np.all(converted.resolution.wave_A > resolution.wave_A)
    assert converted.meta["resolution"]["wave_A"] == pytest.approx(
        converted.resolution.wave_A.tolist()
    )
    record = converted.meta["wavelength_conversions"][-1]
    assert record["method"] == "vald3"
    assert record["from_medium"] == "air"
    assert record["to_medium"] == "vacuum"


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


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_broadening_rejects_invalid_widths(bad_value):
    with pytest.raises(ValueError):
        resolve_gaussian_lsf_fwhm_kms(fwhm_kms=bad_value)

    with pytest.raises(ValueError):
        resolve_gaussian_lsf_fwhm_kms(R=bad_value)


def _line_grid(center, grid_kind):
    velocity = np.linspace(-60.0, 60.0, 2401)
    if grid_kind == "log":
        wave = center * np.exp(velocity / 299792.458)
    elif grid_kind == "linear":
        endpoints = center * np.exp(np.array([-60.0, 60.0]) / 299792.458)
        wave = np.linspace(endpoints[0], endpoints[1], len(velocity))
    elif grid_kind == "irregular":
        velocity = velocity + 0.015 * np.sin(np.arange(len(velocity)) * 0.37)
        wave = center * np.exp(velocity / 299792.458)
    else:
        raise AssertionError(grid_kind)

    velocity_actual = 299792.458 * np.log(wave / center)
    intrinsic_sigma = 0.6
    flux = 1.0 - 0.4 * np.exp(-0.5 * (velocity_actual / intrinsic_sigma) ** 2)
    return wave, flux, intrinsic_sigma


def _absorption_moments(wave, flux, center):
    velocity = 299792.458 * np.log(wave / center)
    absorption = np.clip(1.0 - flux, 0.0, None)
    area = np.trapz(absorption, velocity)
    mean = np.trapz(absorption * velocity, velocity) / area
    variance = np.trapz(absorption * (velocity - mean) ** 2, velocity) / area
    return area, np.sqrt(variance)


@pytest.mark.parametrize("center", [4500.0, 6500.0, 8500.0])
@pytest.mark.parametrize("grid_kind", ["linear", "log", "irregular"])
def test_fft_lsf_has_constant_velocity_width(center, grid_kind):
    wave, flux, intrinsic_sigma = _line_grid(center, grid_kind)
    fwhm_kms = 10.0
    target_sigma = fwhm_kms / 2.3548200450309493

    broadened = convolve_to_resolution_loglam(
        wave,
        flux,
        fwhm_kms=fwhm_kms,
    )
    _area, measured_sigma = _absorption_moments(wave, broadened, center)
    expected_sigma = np.sqrt(intrinsic_sigma**2 + target_sigma**2)

    assert measured_sigma == pytest.approx(expected_sigma, rel=0.02)


def test_R_and_fwhm_interfaces_are_equivalent():
    wave, flux, _intrinsic_sigma = _line_grid(6500.0, "linear")
    R = 50_000.0

    by_R = convolve_to_resolution_loglam(wave, flux, R=R)
    by_fwhm = convolve_to_resolution_loglam(
        wave,
        flux,
        fwhm_kms=299792.458 / R,
    )

    assert np.allclose(by_R, by_fwhm, rtol=0.0, atol=1.0e-14)


def test_fft_lsf_preserves_continuum_and_equivalent_width():
    wave, flux, _intrinsic_sigma = _line_grid(6500.0, "irregular")
    continuum = np.ones_like(wave)

    broadened_continuum = convolve_to_resolution_loglam(
        wave,
        continuum,
        fwhm_kms=10.0,
    )
    broadened_line = convolve_to_resolution_loglam(
        wave,
        flux,
        fwhm_kms=10.0,
    )
    area_before, _sigma_before = _absorption_moments(wave, flux, 6500.0)
    area_after, _sigma_after = _absorption_moments(wave, broadened_line, 6500.0)

    assert np.allclose(broadened_continuum, 1.0, rtol=0.0, atol=1.0e-12)
    assert area_after == pytest.approx(area_before, rel=0.02)


def test_log_grid_rebinning_conserves_integrated_flux():
    old_edges = np.linspace(0.0, 10.0, 1002)
    old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
    old_flux = 1.0 - 0.8 * np.exp(-0.5 * ((old_centers - 4.973) / 0.017) ** 2)
    new_edges = np.linspace(old_edges[0], old_edges[-1], 38)

    new_flux = _rebin_flux_density_piecewise_constant(
        old_edges,
        old_flux,
        new_edges,
    )

    old_integral = np.sum(old_flux * np.diff(old_edges))
    new_integral = np.sum(new_flux * np.diff(new_edges))
    assert new_integral == pytest.approx(old_integral, rel=0.0, abs=1.0e-12)


def test_fft_lsf_matches_well_sampled_real_space_reference():
    center = 6500.0
    velocity = np.linspace(-80.0, 80.0, 1601)
    dv = velocity[1] - velocity[0]
    wave = center * np.exp(velocity / 299792.458)
    flux = 1.0 - 0.5 * np.exp(-0.5 * (velocity / 2.0) ** 2)
    fwhm_kms = 8.0
    sigma_v = fwhm_kms / 2.3548200450309493

    fft_result = convolve_to_resolution_loglam(
        wave,
        flux,
        fwhm_kms=fwhm_kms,
    )
    direct_result = gaussian_filter1d(
        flux,
        sigma=sigma_v / dv,
        mode="nearest",
        truncate=8.0,
    )

    core = np.abs(velocity) < 40.0
    assert np.max(np.abs(fft_result[core] - direct_result[core])) < 5.0e-3


def test_return_interpolation_has_no_line_ringing():
    wave, flux, _intrinsic_sigma = _line_grid(6500.0, "irregular")
    broadened = convolve_to_resolution_loglam(wave, flux, fwhm_kms=10.0)

    assert np.nanmin(broadened) >= -1.0e-12
    assert np.nanmax(broadened) <= 1.0 + 1.0e-12


def test_padding_width_is_explicit_and_validated():
    wave, flux, _intrinsic_sigma = _line_grid(6500.0, "linear")

    with pytest.raises(ValueError, match="padding_sigma"):
        convolve_to_resolution_loglam(
            wave,
            flux,
            fwhm_kms=10.0,
            padding_sigma=2.0,
        )

    default_result = convolve_to_resolution_loglam(wave, flux, fwhm_kms=10.0)
    explicit_result = convolve_to_resolution_loglam(
        wave,
        flux,
        fwhm_kms=10.0,
        padding_sigma=6.0,
    )
    assert np.array_equal(default_result, explicit_result)


def test_fft_lsf_preserves_order_shape_and_nonfinite_positions():
    wave, flux, _intrinsic_sigma = _line_grid(6500.0, "linear")
    wave = wave[::-1]
    flux = flux[::-1]
    flux[10] = np.nan

    broadened = convolve_to_resolution_loglam(wave, flux, fwhm_kms=10.0)

    assert broadened.shape == flux.shape
    assert np.isnan(broadened[10])
    assert np.all(np.isfinite(broadened[np.arange(len(flux)) != 10]))


def test_fft_lsf_rejects_duplicate_wavelengths():
    wave = np.array([5000.0, 5001.0, 5001.0, 5002.0, 5003.0])
    flux = np.ones_like(wave)

    with pytest.raises(ValueError, match="unique"):
        convolve_to_resolution_loglam(wave, flux, fwhm_kms=10.0)


def test_fft_lsf_warns_for_undersampled_input():
    wave = 5000.0 * np.exp(np.arange(10.0) * 10.0 / 299792.458)
    flux = np.ones_like(wave)

    with pytest.warns(UserWarning, match="coarser"):
        convolve_to_resolution_loglam(wave, flux, fwhm_kms=5.0)


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
