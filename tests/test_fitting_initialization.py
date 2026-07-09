import numpy as np
import pytest

from Spyctres.fitting import (
    _build_data_vectors,
    _build_local_multistarts,
    _build_phoenix_fit_diagnostics,
    _coarse_physical_start_native_interp,
    _default_coarse_grid,
    _full_spectrum_parameter_count,
    _grid_edge_flags,
    _make_forward_segments,
    _phoenix_quality_flags,
    _retained_segments_from_meta,
    _resolve_segment_fwhm_kms,
    _velocity_convention_summary,
    fit_phoenix_full_spectrum,
    reconstruct_phoenix_legendre_models_for_segments,
)
from Spyctres.io import ResolutionDescriptor, SpectrumCollection, SpectrumSegment


def test_default_coarse_grid_maps_targets_to_unique_nodes():
    grid = np.array([3000.0, 4000.0, 5000.0, 6000.0])
    selected = _default_coarse_grid(grid, [3100.0, 3900.0, 4100.0, 5900.0])
    assert np.array_equal(selected, [3000.0, 4000.0, 6000.0])


def test_local_multistarts_are_deterministic_and_inside_bounds():
    center = np.array([5000.0, -0.5, 4.0, 12.0])
    bounds = ((4500.0, -1.0, 3.0, -100.0), (5500.0, 0.5, 5.0, 100.0))
    starts = _build_local_multistarts(center, bounds, 4)

    assert len(starts) == 4
    assert np.allclose(starts[0], center)
    assert np.allclose(starts[1], [4750.0, -0.5, 4.0, 12.0])
    assert np.allclose(starts[2], [5250.0, -0.5, 4.0, 12.0])
    assert np.allclose(starts[3], [5000.0, -0.625, 4.0, 12.0])
    for start in starts:
        assert np.all(start > np.asarray(bounds[0]))
        assert np.all(start < np.asarray(bounds[1]))


def test_local_multistarts_can_force_zero_rv_independently():
    center = np.array([5000.0, -0.5, 4.0, 7.5])
    bounds = ((4500.0, -1.0, 3.0, -100.0), (5500.0, 0.5, 5.0, 100.0))
    starts = _build_local_multistarts(center, bounds, 2, alternate_rv=0.0)
    assert starts[0][3] == 7.5
    assert starts[1][3] == 0.0


def test_local_multistarts_reject_an_unavailable_alternate_rv():
    center = np.array([5000.0, -0.5, 4.0, 20.0])
    bounds = ((4500.0, -1.0, 3.0, 10.0), (5500.0, 0.5, 5.0, 100.0))
    with pytest.raises(ValueError, match="alternate_rv"):
        _build_local_multistarts(center, bounds, 2, alternate_rv=0.0)


def test_forward_segments_preserve_wavelength_state_and_resolution():
    resolution = ResolutionDescriptor(quantity="sigma_kms", value=11.0)
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 0.9, 1.0],
        err=[0.01, 0.01, 0.01],
        wave_medium="air",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        stellar_rv_applied_kms=42.0,
        resolution=resolution,
    )

    forwarded = _make_forward_segments(
        [segment],
        support_wave_all=segment.wave,
        support_slices=[slice(0, 3)],
        fit_masks=[np.array([True, True, False])],
    )[0]

    assert forwarded.wave_medium == "air"
    assert forwarded.observer_frame == "barycentric"
    assert forwarded.stellar_rest_status == "corrected"
    assert forwarded.stellar_rv_applied_kms == 42.0
    assert forwarded.resolution is resolution
    assert _resolve_segment_fwhm_kms(forwarded) == pytest.approx(
        11.0 * 2.3548200450309493
    )


def test_data_vectors_align_forward_metadata_after_dropping_a_segment():
    dropped = SpectrumSegment(
        [4000.0, 4001.0],
        [1.0, 1.0],
        err=[0.1, 0.1],
        mask=[False, False],
        name="dropped",
        resolution=ResolutionDescriptor(quantity="R", value=1000.0),
    )
    second = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        err=[0.1, 0.1],
        name="second",
        observer_frame="barycentric",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
    )
    third = SpectrumSegment(
        [6000.0, 6001.0],
        [1.0, 1.0],
        err=[0.1, 0.1],
        name="third",
        observer_frame="heliocentric",
        resolution=ResolutionDescriptor(quantity="R", value=3000.0),
    )
    vectors = _build_data_vectors([dropped, second, third])
    support_wave, _, _, support_slices, _, fit_masks, _, seg_meta = vectors
    retained = _retained_segments_from_meta([dropped, second, third], seg_meta)
    forwarded = _make_forward_segments(
        retained, support_wave, support_slices, fit_masks
    )

    assert [meta["index"] for meta in seg_meta] == [1, 2]
    assert [segment.name for segment in forwarded] == ["second", "third"]
    assert [segment.observer_frame for segment in forwarded] == [
        "barycentric",
        "heliocentric",
    ]
    assert [segment.resolution.value for segment in forwarded] == [2000.0, 3000.0]


def test_forward_segment_alignment_mismatch_is_rejected():
    segment = SpectrumSegment([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="remain aligned"):
        _make_forward_segments(
            [segment],
            support_wave_all=segment.wave,
            support_slices=[],
            fit_masks=[],
        )


def test_parameter_count_uses_only_retained_segments():
    assert _full_spectrum_parameter_count(2, mdeg=2) == 10


def test_multisegment_weighted_chi2_and_dof_accounting():
    class ConstantLibrary:
        DEFAULT_TEFF_GRID = np.array([4999.0, 5000.0, 5001.0])
        DEFAULT_FEH_GRID = np.array([-0.1, 0.0, 0.1])
        DEFAULT_LOGG_GRID = np.array([3.9, 4.0, 4.1])

        def __init__(self):
            self._n_wave = None

        def interpolator_matches(self, observed_wave, *args, **kwargs):
            return self._n_wave == len(observed_wave)

        def build_interpolator(self, observed_wave, *args, **kwargs):
            self._n_wave = len(observed_wave)

        def evaluate(self, teff, feh, logg):
            return np.ones(self._n_wave, dtype=float)

    first = SpectrumSegment(
        wave=np.arange(5000.0, 5005.0),
        flux=[0.9, 1.1, 1.0, 1.0, 1.0],
        err=np.full(5, 0.1),
        name="first",
    )
    second = SpectrumSegment(
        wave=np.arange(6000.0, 6005.0),
        flux=[1.2, 0.8, 1.0, 1.0, 1.0],
        err=np.full(5, 0.2),
        name="second",
    )
    collection = SpectrumCollection(
        [first, second],
        weights=[1.0, 4.0],
        name="weighted-two-segment",
    )

    result = fit_phoenix_full_spectrum(
        collection,
        phoenix_lib=ConstantLibrary(),
        p0=(5000.0, 0.0, 4.0, 0.0),
        bounds=((4999.0, -0.1, 3.9, -1.0), (5001.0, 0.1, 4.1, 1.0)),
        mdeg=0,
        rv_init=None,
        multistart=1,
        max_nfev=1,
        forward_model="interp_observed",
    )

    assert result["n_points"] == 10
    assert result["diagnostics"]["n_parameters"] == 6
    assert result["dof"] == 4
    assert result["chi2"] == pytest.approx(10.0)
    assert result["chi2_red"] == pytest.approx(2.5)
    assert result["diagnostics"]["degrees_of_freedom"] == 4
    assert result["diagnostics"]["segment_diagnostics"][0]["n_fit"] == 5
    assert result["diagnostics"]["segment_diagnostics"][1]["weight"] == 4.0


def test_full_spectrum_fit_rejects_invalid_optimizer_controls_early():
    segment = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        err=[0.1, 0.1],
    )
    with pytest.raises(ValueError, match="max_nfev"):
        fit_phoenix_full_spectrum(
            segment,
            phoenix_lib=object(),
            p0=(5000.0, 0.0, 4.0, 0.0),
            max_nfev=0,
        )
    with pytest.raises(ValueError, match="rv_grid_n"):
        fit_phoenix_full_spectrum(
            segment,
            phoenix_lib=object(),
            p0=(5000.0, 0.0, 4.0, 0.0),
            rv_grid_n=1,
        )
    with pytest.raises(ValueError, match="p0"):
        fit_phoenix_full_spectrum(
            segment,
            phoenix_lib=object(),
            p0=(5000.0, 0.0, 4.0),
        )


def test_interp_observed_reconstruction_preserves_invalid_original_pixels():
    segment = SpectrumSegment(
        [5000.0, 5001.0, 5002.0],
        [1.0, np.nan, 1.0],
        err=[0.1, 0.1, 0.1],
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
    )

    class Library:
        def evaluate(self, teff, feh, logg):
            return np.ones(2, dtype=float)

    models, coeffs, used_masks, excluded_masks = (
        reconstruct_phoenix_legendre_models_for_segments(
            segment,
            phoenix_lib=Library(),
            fit_result={
                "teff": 5000.0,
                "feh": 0.0,
                "logg": 4.0,
                "rv_kms": 0.0,
                "rv_bary_kms": 0.0,
                "forward_model": "interp_observed",
            },
            mdeg=0,
        )
    )

    assert models[0].shape == segment.wave.shape
    assert np.isfinite(models[0][[0, 2]]).all()
    assert np.isnan(models[0][1])
    assert np.array_equal(used_masks[0], [True, False, True])
    assert coeffs[0].shape == (1,)
    assert excluded_masks[0].shape == segment.wave.shape


def test_phoenix_diagnostics_and_quality_flags_are_json_safe():
    class Library:
        DEFAULT_TEFF_GRID = np.array([5000.0, 6000.0])
        DEFAULT_FEH_GRID = np.array([-0.5, 0.0])
        DEFAULT_LOGG_GRID = np.array([3.0, 4.0])

    dropped = SpectrumSegment(
        [4000.0, 4001.0],
        [1.0, 1.0],
        err=[0.1, 0.1],
        mask=[False, False],
        name="dropped",
    )
    retained = SpectrumSegment(
        [5000.0, 5001.0, 5002.0, 5003.0],
        [1.0, 0.9, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.1],
        mask=[True, True, False, True],
        name="retained",
        wave_medium="unknown",
        wave_frame="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )
    vectors = _build_data_vectors([dropped, retained])
    support_wave, _, _, support_slices, _, fit_masks, _, seg_meta = vectors
    forward = _make_forward_segments(
        _retained_segments_from_meta([dropped, retained], seg_meta),
        support_wave,
        support_slices,
        fit_masks,
    )
    diagnostics = _build_phoenix_fit_diagnostics(
        residuals=np.array([3.0, -3.0, 3.0]),
        chi2=27.0,
        chi2_red=9.0,
        dof=3,
        n_parameters=4,
        input_segments=[dropped, retained],
        forward_segments=forward,
        seg_meta=seg_meta,
        mdeg=0,
        best_parameters=np.array([5000.0, -0.25, 3.5, 0.0]),
        phoenix_lib=Library(),
        segment_fwhm_kms=[None],
        local_solutions=[
            {
                "start": [5000.0, -0.25, 3.5, 0.0],
                "solution": [5000.0, -0.25, 3.5, 0.0],
                "chi2": 27.0,
                "success": True,
                "status": 1,
                "nfev": 3,
            }
        ],
        coarse_initialization=None,
        rv_kms=12.0,
        rv_bary_kms=-3.0,
    )
    flags = _phoenix_quality_flags(diagnostics, success=True)

    assert diagnostics["n_input_segments"] == 2
    assert diagnostics["n_retained_segments"] == 1
    assert diagnostics["n_dropped_segments"] == 1
    assert diagnostics["grid_edge_flags"]["teff"] is True
    assert diagnostics["grid_edge_flags"]["teff_low"] is True
    assert diagnostics["grid_edge_flags"]["teff_high"] is False
    assert diagnostics["grid_edge_flags"]["feh"] is False
    assert diagnostics["grid_edge_flags"]["fit_bound_hit"] is True
    assert diagnostics["resolution_metadata_summary"]["missing_count"] == 1
    assert diagnostics["rv_start_values"] == [0.0]
    assert diagnostics["total_model_shift_kms"] == pytest.approx(9.0)
    assert diagnostics["velocity_convention"]["rv_kms_fit"] == pytest.approx(12.0)
    assert diagnostics["velocity_convention"]["rv_bary_kms_input"] == pytest.approx(-3.0)
    assert diagnostics["velocity_convention"]["total_model_shift_kms"] == pytest.approx(9.0)
    assert diagnostics["velocity_convention"]["rv_bary_applied_to_model"] is True
    assert diagnostics["velocity_convention"]["rv_bary_applied_to_data"] == "unknown"
    assert (
        diagnostics["velocity_convention"]["rv_combination_formula"]
        == "total_model_shift_kms = rv_kms_fit + rv_bary_kms_input"
    )
    assert (
        diagnostics["velocity_convention"]["wavelength_frame_assumption"][
            "stellar_rest_status"
        ]
        == "unknown"
    )
    assert "high_chi2" in flags
    assert "grid_edge_teff" in flags
    assert "grid_edge_teff_low" in flags
    assert "fit_bound_hit" in flags
    assert "resolution_missing" in flags
    assert "wavelength_frame_ambiguous" in flags
    assert "metadata_incomplete" in flags


def test_grid_edge_flags_report_low_and_high_boundaries():
    summary = {
        "teff": {"min": 5000.0, "max": 6000.0, "n": 2},
        "feh": {"min": -0.5, "max": 0.0, "n": 2},
        "logg": {"min": 3.0, "max": 4.0, "n": 2},
    }

    low = _grid_edge_flags(np.array([5000.0, -0.25, 3.5, 0.0]), summary)
    high = _grid_edge_flags(np.array([5500.0, 0.0, 4.0, 0.0]), summary)
    interior = _grid_edge_flags(np.array([5500.0, -0.25, 3.5, 0.0]), summary)

    assert low["teff"] is True
    assert low["teff_low"] is True
    assert low["teff_high"] is False
    assert low["fit_bound_hit"] is True

    assert high["feh"] is True
    assert high["feh_high"] is True
    assert high["logg_high"] is True
    assert high["fit_bound_hit"] is True

    assert interior["teff"] is False
    assert interior["feh"] is False
    assert interior["logg"] is False
    assert interior["fit_bound_hit"] is False


def test_velocity_convention_summary_reports_mixed_frame_state():
    first = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        wave_medium="air",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
    )
    second = SpectrumSegment(
        [6000.0, 6001.0],
        [1.0, 1.0],
        wave_medium="vacuum",
        wave_frame="topocentric",
        observer_frame="topocentric",
        stellar_rest_status="observed",
    )

    summary = _velocity_convention_summary(
        [first, second],
        rv_kms=10.0,
        rv_bary_kms=0.0,
    )

    assert summary["total_model_shift_kms"] == pytest.approx(10.0)
    assert summary["rv_bary_applied_to_model"] is False
    assert summary["rv_bary_term_in_model_formula"] is True
    assert summary["wavelength_frame_assumption"]["observer_frame"] == "mixed"
    assert summary["wavelength_frame_assumption"]["stellar_rest_status"] == "mixed"
    assert summary["wavelength_frame_assumption"]["wave_medium"] == "mixed"


class _NodeLibrary:
    DEFAULT_TEFF_GRID = np.array([4000.0, 5000.0, 6000.0])
    DEFAULT_FEH_GRID = np.array([-1.0, 0.0])
    DEFAULT_LOGG_GRID = np.array([2.5, 4.5])

    def has_template(self, teff, logg, feh):
        return True

    def load_template(self, teff, logg, feh, wave=None, wave_medium=None):
        wave = np.asarray(wave, dtype=float)
        flux = _node_flux(wave, teff, feh, logg)
        return wave, flux


def _node_flux(wave, teff, feh, logg):
    teff_depth = 0.05 + (float(teff) - 4000.0) / 20000.0
    feh_depth = 0.05 + (float(feh) + 1.0) * 0.08
    logg_depth = 0.04 + (float(logg) - 2.5) * 0.04
    return (
        1.0
        - teff_depth * np.exp(-0.5 * ((wave - 5020.0) / 0.8) ** 2)
        - feh_depth * np.exp(-0.5 * ((wave - 5050.0) / 0.9) ** 2)
        - logg_depth * np.exp(-0.5 * ((wave - 5080.0) / 1.0) ** 2)
    )


def test_coarse_initializer_selects_best_sparse_physical_node():
    model_wave = np.linspace(4950.0, 5150.0, 4001)
    observed_wave = np.linspace(5000.0, 5100.0, 1001)
    target = _node_flux(observed_wave, 5000.0, -1.0, 4.5)
    segment = SpectrumSegment(
        observed_wave,
        target,
        err=np.full(observed_wave.size, 0.01),
        wave_medium="vacuum",
        name="synthetic",
    )

    best, scores = _coarse_physical_start_native_interp(
        forward_segments=[segment],
        flux_all=target,
        err_all=np.full(target.size, 0.01),
        fit_slices=[slice(0, target.size)],
        fit_masks=[np.ones(target.size, dtype=bool)],
        segment_weights=np.ones(1),
        phoenix_lib=_NodeLibrary(),
        model_wave_grid=model_wave,
        model_wave_medium="vacuum",
        rv_tot=0.0,
        mdeg=0,
        segment_fwhm_kms=[None],
        model_margin_A=20.0,
        teff_grid=[4000.0, 5000.0, 6000.0],
        feh_grid=[-1.0, 0.0],
        logg_grid=[2.5, 4.5],
        decimate=2,
    )

    assert len(scores) == 12
    assert best["teff"] == 5000.0
    assert best["feh"] == -1.0
    assert best["logg"] == 4.5
    assert best["chi2"] < 1e-8
