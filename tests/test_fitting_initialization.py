import inspect

import numpy as np
import pytest

from Spyctres.fitting import (
    FitProgressEvent,
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
from Spyctres.recipes import (
    XshooterBalmerCase,
    fit_phoenix_sideband_symmetric,
    prepare_xshooter_balmer_case,
)


def test_phoenix_fitters_default_to_native_interp():
    assert (
        inspect.signature(fit_phoenix_full_spectrum)
        .parameters["forward_model"]
        .default
        == "native_interp"
    )
    assert (
        inspect.signature(fit_phoenix_sideband_symmetric)
        .parameters["forward_model"]
        .default
        == "native_interp"
    )


def test_prepare_xshooter_balmer_case_centralizes_segments_and_masks():
    wave = np.linspace(3950.0, 5050.0, 1200)
    segment = SpectrumSegment(
        wave=wave,
        flux=1.0 + 1e-4 * (wave - np.nanmedian(wave)),
        err=np.full_like(wave, 0.02),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium="vacuum",
        name="uvb",
    )

    case = prepare_xshooter_balmer_case(
        segment,
        window_mode="notebook",
        window_pad=20.0,
        norm_mode="sideband",
        sideband_width=10.0,
        sideband_order=1,
        core_mask=12.0,
    )

    assert isinstance(case, XshooterBalmerCase)
    assert [seg.name for seg in case.fit_segments] == ["Hδ", "Hγ", "Hβ"]
    assert all(seg.meta["norm_mode"] == "sideband" for seg in case.fit_segments)
    assert all("line_center_data" in seg.meta for seg in case.fit_segments)
    assert [mask.name for mask in case.exclude_masks] == ["balmer_core"]
    assert case.provenance["recipe"] == "prepare_xshooter_balmer_case"
    assert case.provenance["exclude_mask_metadata"]["balmer_core"]["method"] == (
        "balmer_core_halfwidth"
    )


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


def test_local_multistarts_use_ranked_physical_candidates_when_supplied():
    center = np.array([5000.0, -0.5, 4.0, 12.0])
    bounds = ((4500.0, -1.0, 3.0, -100.0), (5500.0, 0.5, 5.0, 100.0))
    starts = _build_local_multistarts(
        center,
        bounds,
        4,
        alternate_rv=0.0,
        candidate_points=[
            center[:3],  # duplicate of selected best coarse node
            [4400.0, -0.5, 4.0],  # outside local Teff bounds
            [5200.0, -0.25, 4.5],
            [4800.0, 0.0, 3.5, 15.0],
        ],
    )

    assert len(starts) == 4
    assert np.allclose(starts[0], center)
    assert np.allclose(starts[1], [5000.0, -0.5, 4.0, 0.0])
    assert np.allclose(starts[2], [5200.0, -0.25, 4.5, 12.0])
    assert np.allclose(starts[3], [4800.0, 0.0, 3.5, 15.0])


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
            self.build_progress_callback_received = None

        def interpolator_matches(self, observed_wave, *args, **kwargs):
            return self._n_wave == len(observed_wave)

        def build_interpolator(self, observed_wave, *args, **kwargs):
            self._n_wave = len(observed_wave)
            self.build_progress_callback_received = kwargs.get("progress_callback")

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

    messages = []
    callback = messages.append
    library = ConstantLibrary()
    result = fit_phoenix_full_spectrum(
        collection,
        phoenix_lib=library,
        p0=(5000.0, 0.0, 4.0, 0.0),
        bounds=((4999.0, -0.1, 3.9, -1.0), (5001.0, 0.1, 4.1, 1.0)),
        mdeg=0,
        rv_init="grid",
        rv_grid_n=3,
        multistart=1,
        max_nfev=1,
        forward_model="interp_observed",
        loss="soft_l1",
        loss_f_scale=2.0,
        progress_callback=callback,
    )

    assert result["n_points"] == 10
    assert result["diagnostics"]["n_parameters"] == 6
    assert result["dof"] == 4
    assert result["chi2"] == pytest.approx(10.0)
    assert result["chi2_red"] == pytest.approx(2.5)
    assert result["diagnostics"]["degrees_of_freedom"] == 4
    assert result["diagnostics"]["segment_diagnostics"][0]["n_fit"] == 5
    assert result["diagnostics"]["segment_diagnostics"][1]["weight"] == 4.0
    assert result["quality_report"]["n_points"] == 10
    assert result["quality_report"]["degrees_of_freedom"] == 4
    assert result["quality_report"]["segments"][0]["n_fit"] == 5
    assert result["quality_report"]["segments"][0]["explicit_exclusion_fraction"] == 0.0
    assert result["optimizer_loss"] == "soft_l1"
    assert result["optimizer_loss_f_scale"] == 2.0
    assert result["optimizer_cost"] is not None
    assert result["optimizer_cost_twice"] == pytest.approx(2.0 * result["optimizer_cost"])
    assert result["effective_chi2"] == pytest.approx(result["chi2"])
    assert result["effective_chi2_red"] == pytest.approx(result["chi2_red"])
    assert result["raw_chi2"] == pytest.approx(result["chi2"])
    assert result["raw_chi2_red"] == pytest.approx(result["chi2_red"])
    assert result["error_model"] == "nominal"
    assert result["error_floor_applied"] is False
    assert "robust_loss_active" in result["quality_flags"]
    assert result["parameter_uncertainty"]["available"] is True
    assert "parameter_errors_local_linearized" in result["quality_flags"]
    assert "parameter_errors_ignore_model_systematics" in result["quality_flags"]
    assert "parameter_errors_unreliable_if_robust_loss" in result["quality_flags"]
    assert "parameter_errors_unreliable_if_segment_weights" in result["quality_flags"]
    assert result["quality_report"]["parameter_uncertainty"]["available"] is True
    assert result["quality_report"]["optimizer_loss"] == "soft_l1"
    assert result["quality_report"]["error_model"] == "nominal"
    assert result["segment_error_floor_fraction"] == [0.0, 0.0]
    assert result["segment_error_floor_abs"] == [0.0, 0.0]
    assert callable(library.build_progress_callback_received)
    assert library.build_progress_callback_received is not callback
    library.build_progress_callback_received("Synthetic cache callback message.")
    library.build_progress_callback_received(
        {
            "stage": "build_flux_cube",
            "message": "Building PHOENIX flux cube: 2/3 templates.",
            "current": 2,
            "total": 3,
            "unit": "templates",
            "elapsed_s": 1.25,
        }
    )
    assert all(isinstance(message, FitProgressEvent) for message in messages)
    assert all(message.elapsed_s is not None for message in messages)
    assert any("Prepared fit data" in message for message in messages)
    assert any(str(message).startswith("Preparing PHOENIX") for message in messages)
    assert any("Running coarse RV grid scan" in message for message in messages)
    assert any("Coarse RV grid scan selected" in message for message in messages)
    assert any("Starting local optimizer" in message for message in messages)
    assert any("Finished local optimizer" in message for message in messages)
    assert any("Selected best fit" in message for message in messages)
    phases = [message.phase for message in messages]
    assert "prepare_data" in phases
    assert "phoenix_cache" in phases
    assert "build_flux_cube" in phases
    assert "rv_scan" in phases
    assert "local_optimize" in phases
    assert "complete" in phases
    structured = [
        message for message in messages
        if message.phase == "build_flux_cube"
    ][-1]
    assert structured.current == 2
    assert structured.total == 3
    assert structured.fraction == pytest.approx(2.0 / 3.0)
    assert structured.payload["unit"] == "templates"
    assert structured.payload["phoenix_elapsed_s"] == pytest.approx(1.25)


def test_full_spectrum_fit_reports_error_floor_error_model():
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

    segment = SpectrumSegment(
        wave=np.arange(5000.0, 5005.0),
        flux=[0.9, 1.1, 1.0, 1.0, 1.0],
        err=np.full(5, 0.01),
        name="floor-test",
    )
    result = fit_phoenix_full_spectrum(
        segment,
        phoenix_lib=ConstantLibrary(),
        p0=(5000.0, 0.0, 4.0, 0.0),
        bounds=((4999.0, -0.1, 3.9, -1.0), (5001.0, 0.1, 4.1, 1.0)),
        mdeg=0,
        rv_init=None,
        max_nfev=1,
        forward_model="interp_observed",
        error_floor_fraction=0.1,
    )

    assert result["error_floor_applied"] is True
    assert result["error_model"] == "floor_inflated"
    assert result["raw_chi2"] > result["effective_chi2"]
    assert result["raw_chi2_red"] > result["effective_chi2_red"]
    assert result["quality_report"]["error_floor_applied"] is True
    assert result["quality_report"]["raw_chi2"] == pytest.approx(result["raw_chi2"])
    assert result["quality_report"]["effective_chi2"] == pytest.approx(result["chi2"])
    assert "error_floor_applied" in result["quality_flags"]
    if result["parameter_uncertainty"]["available"]:
        assert "parameter_errors_unreliable_if_error_floor" in result["quality_flags"]


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
    with pytest.raises(ValueError, match="loss must be"):
        fit_phoenix_full_spectrum(
            segment,
            phoenix_lib=object(),
            p0=(5000.0, 0.0, 4.0, 0.0),
            loss="not_a_loss",
        )
    with pytest.raises(ValueError, match="loss_f_scale"):
        fit_phoenix_full_spectrum(
            segment,
            phoenix_lib=object(),
            p0=(5000.0, 0.0, 4.0, 0.0),
            loss_f_scale=0.0,
        )
    with pytest.raises(ValueError, match="error_floor_fraction"):
        fit_phoenix_full_spectrum(
            segment,
            phoenix_lib=object(),
            p0=(5000.0, 0.0, 4.0, 0.0),
            error_floor_fraction=-0.1,
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
        parameter_errors_available=True,
    )
    flags = _phoenix_quality_flags(diagnostics, success=True)

    assert diagnostics["n_input_segments"] == 2
    assert diagnostics["n_retained_segments"] == 1
    assert diagnostics["n_dropped_segments"] == 1
    assert diagnostics["effective_chi2"] == pytest.approx(27.0)
    assert diagnostics["effective_chi2_red"] == pytest.approx(9.0)
    assert diagnostics["raw_chi2"] == pytest.approx(27.0)
    assert diagnostics["raw_chi2_red"] == pytest.approx(9.0)
    assert diagnostics["error_model"] == "nominal"
    assert diagnostics["error_floor_applied"] is False
    assert diagnostics["parameter_uncertainty"]["available"] is True
    assert (
        diagnostics["parameter_uncertainty"]["method"]
        == "jacobian_pseudoinverse_scaled_by_reduced_chi2"
    )
    assert (
        diagnostics["parameter_uncertainty"][
            "assumes_independent_gaussian_pixel_errors"
        ]
        is True
    )
    assert diagnostics["parameter_uncertainty"]["includes_model_systematics"] is False
    assert diagnostics["optimizer_loss"] == "linear"
    assert diagnostics["grid_edge_flags"]["teff"] is True
    assert diagnostics["grid_edge_flags"]["teff_low"] is True
    assert diagnostics["grid_edge_flags"]["teff_high"] is False
    assert diagnostics["grid_edge_flags"]["feh"] is False
    assert diagnostics["grid_edge_flags"]["fit_bound_hit"] is True
    assert diagnostics["resolution_metadata_summary"]["missing_count"] == 1
    assert diagnostics["resolution_metadata_summary"]["gaussian_lsf_assumed"] is False
    assert diagnostics["resolution_metadata_summary"]["constant_lsf_assumed"] is False
    assert diagnostics["resolution_metadata_summary"]["pixels_per_fwhm_median"] is None
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
    assert diagnostics["segment_diagnostics"][0]["mask_summary"]["n_fit"] == 3
    assert (
        diagnostics["segment_diagnostics"][0]["mask_provenance"]["label"]
        == "fit selection"
    )
    assert "high_chi2" in flags
    assert "parameter_errors_local_linearized" in flags
    assert "parameter_errors_ignore_model_systematics" in flags
    assert "parameter_errors_unreliable_if_high_chi2" in flags
    assert "grid_edge_teff" in flags
    assert "grid_edge_teff_low" in flags
    assert "fit_bound_hit" in flags
    assert "resolution_missing" in flags
    assert "wavelength_frame_ambiguous" in flags
    assert "unknown_wave_medium_used_in_fit" in flags
    assert "unknown_observer_frame_used_in_fit" in flags
    assert "stellar_rest_status_unknown" in flags
    assert "rv_interpretation_ambiguous" in flags
    assert "metadata_incomplete" in flags
    assert "segment_no_fit_pixels" in flags
    assert "too_few_fit_pixels" in flags


def test_quality_flags_include_mask_derived_warnings():
    diagnostics = {
        "reduced_chi2": 1.0,
        "n_parameters": 4,
        "n_dropped_segments": 0,
        "mask_fraction": 0.2,
        "grid_edge_flags": {},
        "wavelength_metadata_summary": {},
        "resolution_metadata_summary": {"missing_count": 0},
        "segment_diagnostics": [
            {
                "n_fit": 12,
                "mask_fraction": 0.6,
                "mask_summary": {
                    "n_fit": 12,
                    "n_rejected_by_explicit_union": 20,
                },
                "mask_provenance": {
                    "counts": {"nonfinite_mask_output": 1},
                    "settings": {
                        "quality_flags": [
                            "telluric_mask_frame_ambiguous",
                            "coarse_telluric_mask_applied",
                        ],
                    },
                },
            }
        ],
    }

    flags = _phoenix_quality_flags(diagnostics, success=True)

    assert "too_few_fit_pixels" in flags
    assert "segment_mask_fraction_high" in flags
    assert "explicit_exclusion_dominates" in flags
    assert "nonfinite_mask_output" in flags
    assert "telluric_mask_frame_ambiguous" in flags
    assert "coarse_telluric_mask_applied" in flags


def test_quality_flags_include_robust_loss_and_error_floor_warnings():
    diagnostics = {
        "reduced_chi2": 1.0,
        "n_parameters": 4,
        "n_dropped_segments": 0,
        "mask_fraction": 0.0,
        "grid_edge_flags": {},
        "wavelength_metadata_summary": {},
        "resolution_metadata_summary": {"missing_count": 0},
        "segment_diagnostics": [{"n_fit": 100, "mask_fraction": 0.0}],
        "optimizer_loss": "soft_l1",
        "error_floor_applied": True,
        "parameter_uncertainty": {
            "caveat_flags": [
                "parameter_errors_local_linearized",
                "parameter_errors_unreliable_if_robust_loss",
                "parameter_errors_unreliable_if_error_floor",
            ],
        },
    }

    flags = _phoenix_quality_flags(diagnostics, success=True)

    assert "robust_loss_active" in flags
    assert "error_floor_applied" in flags
    assert "parameter_errors_local_linearized" in flags
    assert "parameter_errors_unreliable_if_robust_loss" in flags
    assert "parameter_errors_unreliable_if_error_floor" in flags


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


def test_velocity_convention_flags_recorded_barycentric_metadata_risks():
    topocentric = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        wave_frame="topocentric",
        observer_frame="topocentric",
        stellar_rest_status="observed",
        meta={"barycorr_kms": 22.0},
    )
    corrected = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        meta={"barycorr_kms": 22.0},
    )

    recorded = _velocity_convention_summary(
        [topocentric],
        rv_kms=0.0,
        rv_bary_kms=0.0,
    )
    assert (
        recorded["barycentric_velocity_metadata"]["recorded_not_applied"]
        is True
    )
    flags = _phoenix_quality_flags(
        {
            "reduced_chi2": 1.0,
            "grid_edge_flags": {},
            "n_dropped_segments": 0,
            "mask_fraction": 0.0,
            "resolution_metadata_summary": {"missing_count": 0},
            "wavelength_metadata_summary": {},
            "velocity_convention": recorded,
            "segment_diagnostics": [{"n_fit": 100, "mask_fraction": 0.0}],
            "n_parameters": 4,
        }
    )
    assert "barycentric_correction_recorded_not_applied" in flags
    assert "rv_interpretation_ambiguous" in flags

    double = _velocity_convention_summary(
        [corrected],
        rv_kms=0.0,
        rv_bary_kms=22.0,
    )
    assert (
        double["barycentric_velocity_metadata"][
            "possible_double_barycentric_or_rest_correction"
        ]
        is True
    )
    flags = _phoenix_quality_flags(
        {
            "reduced_chi2": 1.0,
            "grid_edge_flags": {},
            "n_dropped_segments": 0,
            "mask_fraction": 0.0,
            "resolution_metadata_summary": {"missing_count": 0},
            "wavelength_metadata_summary": {},
            "velocity_convention": double,
            "segment_diagnostics": [{"n_fit": 100, "mask_fraction": 0.0}],
            "n_parameters": 4,
        }
    )
    assert "possible_double_barycentric_or_rest_correction" in flags
    assert "rv_interpretation_ambiguous" in flags


def test_lsf_sampling_diagnostics_flag_low_sampling_and_tabulated_lsf():
    class Library:
        DEFAULT_TEFF_GRID = np.array([5000.0, 6000.0])
        DEFAULT_FEH_GRID = np.array([-0.5, 0.0])
        DEFAULT_LOGG_GRID = np.array([3.0, 4.0])

    segment = SpectrumSegment(
        [5000.0, 5010.0, 5020.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        mask=[True, True, True],
        name="undersampled",
        wave_medium="vacuum",
        wave_frame="topocentric",
        observer_frame="topocentric",
        stellar_rest_status="observed",
    )
    diagnostics = _build_phoenix_fit_diagnostics(
        residuals=np.array([0.0, 0.0, 0.0]),
        chi2=0.0,
        chi2_red=0.0,
        dof=1,
        n_parameters=4,
        input_segments=[segment],
        forward_segments=[segment],
        seg_meta=[
            {
                "name": "undersampled",
                "index": 0,
                "weight": 1.0,
                "n_support": 3,
                "n_fit": 3,
                "wave_min": 5000.0,
                "wave_max": 5020.0,
                "mask_summary": {"n_fit": 3},
                "mask_provenance": {},
                "error_floor": {},
            }
        ],
        mdeg=0,
        best_parameters=np.array([5500.0, -0.25, 3.5, 0.0]),
        phoenix_lib=Library(),
        segment_fwhm_kms=[5.0],
        local_solutions=[],
        coarse_initialization=None,
    )
    resolution = diagnostics["resolution_metadata_summary"]
    assert resolution["gaussian_lsf_assumed"] is True
    assert resolution["constant_lsf_assumed"] is True
    assert resolution["pixels_per_fwhm_min"] < 2.0
    assert resolution["low_sampling_warning"] is True
    flags = _phoenix_quality_flags(diagnostics)
    assert "low_sampling_warning" in flags

    tabulated = segment.copy(
        resolution=ResolutionDescriptor(
            quantity="R",
            mode="tabulated",
            wave_A=[5000.0, 5020.0],
            values=[8000.0, 9000.0],
            source="unit_test_tabulated_lsf",
        )
    )
    diagnostics = _build_phoenix_fit_diagnostics(
        residuals=np.array([0.0, 0.0, 0.0]),
        chi2=0.0,
        chi2_red=0.0,
        dof=1,
        n_parameters=4,
        input_segments=[tabulated],
        forward_segments=[tabulated],
        seg_meta=[
            {
                "name": "tabulated",
                "index": 0,
                "weight": 1.0,
                "n_support": 3,
                "n_fit": 3,
                "wave_min": 5000.0,
                "wave_max": 5020.0,
                "mask_summary": {"n_fit": 3},
                "mask_provenance": {},
                "error_floor": {},
            }
        ],
        mdeg=0,
        best_parameters=np.array([5500.0, -0.25, 3.5, 0.0]),
        phoenix_lib=Library(),
        segment_fwhm_kms=[None],
        local_solutions=[],
        coarse_initialization=None,
    )
    assert (
        diagnostics["resolution_metadata_summary"][
            "tabulated_lsf_present_but_not_supported_by_fitter"
        ]
        is True
    )
    flags = _phoenix_quality_flags(diagnostics)
    assert "tabulated_lsf_present_but_not_supported_by_fitter" in flags


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
