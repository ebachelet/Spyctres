import numpy as np
import pytest

from Spyctres.fitting import (
    _build_data_vectors,
    build_effective_fit_mask,
    build_excluded_mask,
    reconstruct_phoenix_legendre_models_for_segments,
)
from Spyctres.io import (
    ResolutionDescriptor,
    SpectrumCollection,
    SpectrumSegment,
    make_padded_window_segments,
)
from Spyctres.preprocessing import apply_fit_mask, compose_fit_mask


def test_segment_default_mask_rejects_invalid_data_and_errors():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, np.nan, 5003.0],
        flux=[1.0, np.nan, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.0],
    )

    assert np.array_equal(segment.mask, [True, False, False, False])


def test_float_exclusion_mask_is_thresholded_and_composed():
    segment = SpectrumSegment(
        wave=np.arange(5.0),
        flux=np.ones(5),
        err=np.ones(5),
    )

    def exclude_mask(_wave):
        return np.array([0.0, 0.4, 0.6, 1.0, 0.0])

    effective = build_effective_fit_mask(
        segment,
        regions=[(1.0, 4.0)],
        exclude_regions=[(4.0, 4.0)],
        exclude_mask=exclude_mask,
    )
    excluded = build_excluded_mask(
        segment,
        regions=[(1.0, 4.0)],
        exclude_regions=[(4.0, 4.0)],
        exclude_mask=exclude_mask,
    )

    assert np.array_equal(effective, [False, True, False, False, False])
    assert np.array_equal(excluded, [True, False, True, True, True])


def test_named_multiple_exclusion_masks_are_unionized_and_recorded():
    segment = SpectrumSegment(
        wave=np.arange(5.0),
        flux=np.ones(5),
        err=np.ones(5),
    )

    def telluric(_wave):
        return np.array([0.0, 0.9, 0.0, 0.0, 0.0])

    def line_core(_wave):
        return np.array([False, False, False, True, False])

    result = compose_fit_mask(
        segment,
        exclude_mask=[("telluric", telluric), {"name": "line_core", "callable": line_core}],
    )

    assert np.array_equal(result.effective_mask, [True, False, True, False, True])
    assert np.array_equal(result.excluded_mask, [False, True, False, True, False])
    assert np.array_equal(
        result.rejection_masks["exclude_mask:telluric"],
        [False, True, False, False, False],
    )
    assert np.array_equal(
        result.rejection_masks["exclude_mask:line_core"],
        [False, False, False, True, False],
    )
    assert result.settings["exclude_masks"] == ["telluric", "line_core"]
    assert result.settings["mask_true_means"] == "use"
    assert result.settings["exclude_mask_true_means"] == "reject"


def test_mask_threshold_is_exposed_through_fitting_wrappers():
    segment = SpectrumSegment(
        wave=np.arange(4.0),
        flux=np.ones(4),
        err=np.ones(4),
    )

    def soft_exclusion(_wave):
        return np.array([0.2, 0.5, 0.7, 0.9])

    default = build_effective_fit_mask(segment, exclude_mask=soft_exclusion)
    stricter = build_effective_fit_mask(
        segment,
        exclude_mask=soft_exclusion,
        mask_threshold=0.8,
    )

    assert np.array_equal(default, [True, True, False, False])
    assert np.array_equal(stricter, [True, True, True, False])


def test_collection_preserves_positive_segment_weights():
    first = SpectrumSegment([1.0, 2.0], [1.0, 1.0], name="first")
    second = SpectrumSegment([3.0, 4.0], [1.0, 1.0], name="second")

    collection = SpectrumCollection([first, second], weights=[1.0, 2.5])

    assert collection.names == ["first", "second"]
    assert np.array_equal(collection.weights, [1.0, 2.5])


def test_mask_provenance_retains_overlapping_rejection_reasons():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0, 5003.0, 5004.0],
        flux=[1.0, np.nan, 1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.0, 0.1, 0.1],
        mask=[True, True, True, False, True],
    )

    def telluric_mask(_wave):
        return [0.0, 0.8, 0.0, 0.0, 0.0]

    result = compose_fit_mask(
        segment,
        regions=[(5001.0, 5004.0)],
        exclude_regions=[(5003.0, 5003.0)],
        exclude_mask=telluric_mask,
    )

    assert np.array_equal(result.effective_mask, [False, False, False, False, True])
    assert np.array_equal(result.excluded_mask, [True, True, False, True, False])
    assert np.array_equal(
        result.rejection_masks["invalid_flux"],
        [False, True, False, False, False],
    )
    assert np.array_equal(
        result.rejection_masks["exclude_mask"],
        [False, True, False, False, False],
    )
    assert result.counts["used"] == 1
    assert result.counts["rejected"] == 4


def test_apply_fit_mask_copies_inputs_and_records_json_safe_history():
    segment = SpectrumSegment(
        wave=[1.0, 2.0, 3.0],
        flux=[1.0, 1.0, 1.0],
        meta={"target": "star"},
    )
    original_mask = segment.mask.copy()

    masked, result = apply_fit_mask(
        segment,
        exclude_regions=[(2.0, 2.0)],
        label="science selection",
    )

    assert np.array_equal(segment.mask, original_mask)
    assert "preprocessing" not in segment.meta
    assert np.array_equal(masked.mask, [True, False, True])
    assert masked.meta["target"] == "star"
    record = masked.meta["preprocessing"][0]
    assert record["operation"] == "mask"
    assert record["label"] == "science selection"
    assert record["counts"] == result.counts
    assert record["settings"]["exclude_regions"] == [[2.0, 2.0]]


def test_mask_callable_shape_is_validated_without_broadcasting():
    segment = SpectrumSegment([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="result must have shape"):
        compose_fit_mask(segment, exclude_mask=lambda _wave: [True])


def test_mask_threshold_must_be_finite():
    segment = SpectrumSegment([1.0, 2.0], [1.0, 1.0])

    with pytest.raises(ValueError, match="mask_threshold must be finite"):
        compose_fit_mask(segment, mask_threshold=np.nan)


@pytest.mark.parametrize(
    "regions",
    [[(2.0, 1.0)], [(1.0, np.inf)], [(1.0, 2.0, 3.0)]],
)
def test_invalid_wavelength_regions_are_rejected(regions):
    segment = SpectrumSegment([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        compose_fit_mask(segment, regions=regions)


def test_missing_errors_do_not_reject_otherwise_valid_pixels():
    segment = SpectrumSegment([1.0, 2.0], [1.0, 1.0], err=None)
    result = compose_fit_mask(segment)

    assert np.array_equal(result.effective_mask, [True, True])
    assert not np.any(result.rejection_masks["invalid_error"])


def test_padded_windows_preserve_wavelength_state_and_resolution():
    resolution = ResolutionDescriptor(quantity="sigma_kms", value=11.0)
    segment = SpectrumSegment(
        wave=np.arange(4990.0, 5011.0),
        flux=np.ones(21),
        err=np.full(21, 0.01),
        wave_medium="air",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        stellar_rv_applied_kms=42.0,
        resolution=resolution,
    )

    window = make_padded_window_segments(segment, [(4998.0, 5002.0)], pad=2.0)[0]

    assert window.wave_medium == "air"
    assert window.observer_frame == "barycentric"
    assert window.stellar_rest_status == "corrected"
    assert window.stellar_rv_applied_kms == 42.0
    assert window.resolution is resolution


def test_fit_segment_metadata_contains_mask_provenance():
    segment = SpectrumSegment(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        name="test",
    )
    vectors = _build_data_vectors(
        [segment],
        exclude_regions=[(2.0, 2.0)],
    )
    seg_meta = vectors[-1]

    provenance = seg_meta[0]["mask_provenance"]
    assert provenance["operation"] == "mask"
    assert provenance["counts"]["used"] == 2
    assert provenance["counts"]["exclude_regions"] == 1


def test_build_data_vectors_selects_per_segment_exclusion_masks_by_name_and_index():
    first = SpectrumSegment(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        name="first",
    )
    second = SpectrumSegment(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        name="second",
    )

    def first_mask(wave):
        return wave == 2.0

    def second_mask(wave):
        return wave == 3.0

    vectors = _build_data_vectors(
        [first, second],
        exclude_mask={
            "first": ("first_line", first_mask),
            1: ("second_line", second_mask),
        },
    )
    fit_masks = vectors[5]
    seg_meta = vectors[-1]

    assert np.array_equal(fit_masks[0], [True, False, True])
    assert np.array_equal(fit_masks[1], [True, True, False])
    assert seg_meta[0]["mask_provenance"]["settings"]["exclude_masks"] == ["first_line"]
    assert seg_meta[1]["mask_provenance"]["settings"]["exclude_masks"] == ["second_line"]


def test_build_data_vectors_keeps_global_named_mask_dict_as_callable_spec():
    segment = SpectrumSegment(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        name="science",
    )

    def central_mask(wave):
        return wave == 2.0

    vectors = _build_data_vectors(
        [segment],
        exclude_mask={"name": "central", "callable": central_mask},
    )
    fit_masks = vectors[5]
    seg_meta = vectors[-1]

    assert np.array_equal(fit_masks[0], [True, False, True])
    assert seg_meta[0]["mask_provenance"]["settings"]["exclude_masks"] == ["central"]


def test_reconstruction_masks_match_per_segment_exclusion_dictionary():
    class DummyPhoenixLibrary:
        wave = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=float)
        phoenix_wave_medium = "air"

        def evaluate(self, teff, feh, logg):
            return np.ones(6, dtype=float)

    first = SpectrumSegment(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        name="first",
        wave_medium="air",
    )
    second = SpectrumSegment(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        name="second",
        wave_medium="air",
    )

    def first_mask(wave):
        return wave == 1.0

    def second_mask(wave):
        return wave == 3.0

    _models, _coeffs, used_masks, excluded_masks = (
        reconstruct_phoenix_legendre_models_for_segments(
            [first, second],
            phoenix_lib=DummyPhoenixLibrary(),
            fit_result={
                "teff": 5000.0,
                "feh": 0.0,
                "logg": 4.0,
                "rv_kms": 0.0,
                "rv_bary_kms": 0.0,
                "forward_model": "interp_observed",
            },
            exclude_mask={
                "first": ("blue_edge", first_mask),
                "second": ("red_edge", second_mask),
            },
        )
    )

    assert np.array_equal(used_masks[0], [False, True, True])
    assert np.array_equal(excluded_masks[0], [True, False, False])
    assert np.array_equal(used_masks[1], [True, True, False])
    assert np.array_equal(excluded_masks[1], [False, False, True])
