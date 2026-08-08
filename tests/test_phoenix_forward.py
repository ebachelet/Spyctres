import numpy as np

from Spyctres.io import SpectrumSegment
from Spyctres.phoenix_forward import (
    build_native_interp_wave_grid_for_segments,
    build_phoenix_native_models_for_segments,
)
from Spyctres.waveutils import convert_wavelength_medium


class _DummyPhoenixLibrary:
    phoenix_wave_medium = "vacuum"

    def __init__(self, phoenix_wave):
        self.phoenix_wave = np.asarray(phoenix_wave, dtype=float)


def _xsl_like_air_segment_from_vacuum_wave(vacuum_wave):
    air_wave = convert_wavelength_medium(
        vacuum_wave,
        from_medium="vacuum",
        to_medium="air",
    )
    return SpectrumSegment(
        wave=air_wave,
        flux=np.ones_like(air_wave, dtype=float),
        err=np.full_like(air_wave, 0.01, dtype=float),
        mask=np.ones_like(air_wave, dtype=bool),
        wave_medium="air",
        wave_frame="stellar_rest",
        observer_frame="not_applicable",
        stellar_rest_status="corrected",
        name="xsl_like_air_segment",
    )


def test_native_interp_grid_converts_phoenix_vacuum_to_xsl_like_air_medium():
    """XSL DR3-style air data must not be compared on PHOENIX vacuum pixels."""
    vacuum_wave = np.linspace(4995.0, 5005.0, 11)
    expected_air_wave = convert_wavelength_medium(
        vacuum_wave,
        from_medium="vacuum",
        to_medium="air",
    )
    segment = _xsl_like_air_segment_from_vacuum_wave(vacuum_wave)
    phoenix_lib = _DummyPhoenixLibrary(vacuum_wave)

    model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
        [segment],
        phoenix_lib,
        model_margin_A=0.0,
    )

    assert model_wave_medium == "air"
    np.testing.assert_allclose(model_wave_grid, expected_air_wave, rtol=0.0, atol=1e-8)
    assert np.max(np.abs(model_wave_grid - vacuum_wave)) > 1.0


def test_native_forward_records_air_target_medium_for_xsl_like_segments():
    """The end-to-end native helper should prepare PHOENIX in the data medium."""
    vacuum_wave = np.linspace(4995.0, 5005.0, 11)
    template_flux = 1.0 + 0.01 * (vacuum_wave - np.median(vacuum_wave))
    expected_air_wave = convert_wavelength_medium(
        vacuum_wave,
        from_medium="vacuum",
        to_medium="air",
    )
    segment = _xsl_like_air_segment_from_vacuum_wave(vacuum_wave)

    model_list, diagnostics = build_phoenix_native_models_for_segments(
        [segment],
        phoenix_wave_native=vacuum_wave,
        template_flux_native=template_flux,
        phoenix_wave_medium="vacuum",
        rv_kms=0.0,
        fwhm_kms=None,
        model_margin_A=0.0,
        return_native=True,
    )

    assert len(model_list) == 1
    assert diagnostics["target_wave_medium"] == "air"
    np.testing.assert_allclose(
        diagnostics["wave_native_prepared"],
        expected_air_wave,
        rtol=0.0,
        atol=1e-8,
    )
