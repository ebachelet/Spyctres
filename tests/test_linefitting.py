import numpy as np

from Spyctres.io import ResolutionDescriptor, SpectrumCollection, SpectrumSegment
from Spyctres.linefitting import (
    LineFitConfig,
    LineSpec,
    compare_line_fits,
    fit_line,
    fit_lines,
    known_line_spec,
    list_known_lines,
)
from Spyctres.waveutils import C_KMS


def make_line(kind="absorption", rv_kms=24.0, sigma_A=0.18, with_err=True):
    rng = np.random.default_rng(12)
    rest = 6562.8
    center = rest * (1.0 + rv_kms / C_KMS)
    wave = np.linspace(center - 4.0, center + 4.0, 401)
    amplitude = -0.35 if kind == "absorption" else 0.5
    continuum = 1.0 + 0.002 * (wave - center)
    flux = continuum * (1.0 + amplitude * np.exp(-0.5 * ((wave - center) / sigma_A) ** 2))
    err = np.full_like(wave, 0.003)
    flux += rng.normal(0.0, err)
    return SpectrumSegment(
        wave,
        flux,
        err=err if with_err else None,
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="observed",
    ), rest, center


def test_absorption_line_recovers_center_rv_width_and_equivalent_width():
    segment, rest, center = make_line()
    result = fit_line(
        segment,
        LineSpec("Halpha", rest, kind="absorption", window_A=4.0),
        LineFitConfig(rv_guess_kms=20.0),
    )

    assert result.success
    assert abs(result.center_wave - center) < 0.01
    assert abs(result.rv_kms - 24.0) < 0.5
    assert abs(result.sigma_A - 0.18) < 0.01
    assert result.equivalent_width_A > 0
    assert result.line_flux != result.line_flux


def test_beginner_line_alias_and_center_calls_are_supported():
    segment, rest, center = make_line()

    alias_result = fit_line(segment, "Halpha", LineFitConfig(rv_guess_kms=24.0))
    custom_result = fit_line(
        segment,
        center=rest,
        name="custom_halpha",
        window_A=4.0,
        wave_medium="air",
        config=LineFitConfig(rv_guess_kms=24.0),
    )

    assert alias_result.success
    assert custom_result.success
    assert abs(alias_result.center_wave - center) < 0.01
    assert abs(custom_result.center_wave - center) < 0.01
    assert known_line_spec("Hgamma").name == "Hgamma"


def test_list_known_lines_exposes_names_details_aliases_and_filters():
    names = list_known_lines()

    assert "Hgamma" in names
    assert "Mg II 4481" in names

    blue_names = list_known_lines(wmin=4300.0, wmax=4500.0)
    assert "Hgamma" in blue_names
    assert "Mg II 4481" in blue_names
    assert "Halpha" not in blue_names

    aliases = list_known_lines(include_aliases=True, wmin=4300.0, wmax=4500.0)
    assert "hgamma" in aliases
    assert "mg_ii_4481" in aliases

    details = list_known_lines(details=True, wmin=4300.0, wmax=4350.0)
    assert details == [
        {
            "name": "Hgamma",
            "rest_wave_A": 4340.47,
            "wave_medium": "air",
            "default_window_A": 16.0,
            "kind": "absorption",
            "aliases": ["h_gamma", "hgamma", "hγ"],
        }
    ]


def test_line_alias_selects_covering_segment_from_collection():
    segment, _rest, center = make_line()
    blue = SpectrumSegment(
        np.linspace(4300.0, 4355.0, 200),
        np.ones(200),
        err=np.full(200, 0.02),
        wave_medium="air",
    )
    collection = SpectrumCollection([blue, segment])

    result = fit_line(collection, "Halpha", LineFitConfig(rv_guess_kms=24.0))

    assert result.success
    assert abs(result.center_wave - center) < 0.01


def test_emission_line_has_positive_flux_and_negative_equivalent_width():
    segment, rest, _ = make_line(kind="emission")
    result = fit_line(
        segment,
        LineSpec("emission", rest, kind="emission", window_A=4.0),
        LineFitConfig(rv_guess_kms=24.0),
    )

    assert result.success
    assert result.line_flux > 0
    assert result.equivalent_width_A < 0


def test_missing_errors_are_estimated_and_masked_pixels_are_excluded():
    segment, rest, center = make_line(with_err=False)
    mask = np.ones_like(segment.wave, dtype=bool)
    mask[(segment.wave > center + 0.5) & (segment.wave < center + 1.0)] = False
    segment = segment.copy(mask=mask)
    result = fit_line(
        segment,
        LineSpec("Halpha", rest, window_A=4.0),
        LineFitConfig(rv_guess_kms=24.0),
    )

    assert result.success
    assert "err_missing_or_estimated" in result.flags
    assert result.n_points == np.count_nonzero(mask)


def test_fit_line_accepts_public_valid_mask_override():
    segment, rest, center = make_line()
    valid_mask = np.ones_like(segment.wave, dtype=bool)
    valid_mask[(segment.wave > center + 0.5) & (segment.wave < center + 1.0)] = False

    result = fit_line(
        segment,
        LineSpec("Halpha", rest, window_A=4.0),
        LineFitConfig(rv_guess_kms=24.0),
        valid_mask=valid_mask,
    )

    assert result.success
    assert result.n_points == np.count_nonzero(valid_mask)


def test_nearby_independent_lines_are_flagged_as_blend_candidates():
    segment, rest, _ = make_line()
    results = fit_lines(
        segment,
        [
            "Halpha",
            LineSpec("second", rest + 1.0, window_A=4.0),
        ],
        LineFitConfig(rv_guess_kms=24.0),
    )

    assert all("blend_candidate" in result.flags for result in results)


def test_compare_line_fits_returns_compact_table():
    segment, rest, _ = make_line()
    first = fit_line(
        segment,
        LineSpec("Halpha", rest, window_A=4.0),
        LineFitConfig(rv_guess_kms=24.0),
    )
    second = fit_line(
        segment,
        LineSpec("nearby", rest + 1.0, window_A=4.0),
        LineFitConfig(rv_guess_kms=24.0),
    )

    comparison = compare_line_fits([first, second], labels=["Halpha", "nearby"])

    assert comparison["operation"] == "compare_line_fits"
    assert len(comparison.rows) == 2
    assert comparison.summary()["n_results"] == 2
    assert "Spyctres line-fit comparison" in comparison.summary_text()


def test_line_rest_wavelength_is_converted_to_segment_medium():
    segment, rest_air, _ = make_line(rv_kms=0.0)
    from Spyctres.waveutils import air_to_vacuum_ciddor
    rest_vacuum = float(air_to_vacuum_ciddor([rest_air])[0])
    result = fit_line(
        segment,
        LineSpec("Halpha", rest_vacuum, wave_medium="vacuum", window_A=4.0),
    )

    assert abs(result.rv_kms) < 0.5
    assert result.segment_wave_medium == "air"
    assert abs(result.rest_wave_in_segment_medium - rest_air) < 1.0e-5


def test_instrumental_resolution_is_reported_when_available():
    segment, rest, _ = make_line()
    segment = segment.copy(
        resolution=ResolutionDescriptor(quantity="R", value=50_000.0)
    )
    result = fit_line(
        segment,
        LineSpec("Halpha", rest, window_A=4.0),
        LineFitConfig(rv_guess_kms=24.0),
    )

    assert np.isfinite(result.instrumental_fwhm_A)
    assert "lsf_missing" not in result.flags
