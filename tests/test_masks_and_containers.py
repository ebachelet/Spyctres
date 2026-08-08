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
from Spyctres.preprocessing import (
    NONSTELLAR_FEATURES,
    ExclusionMaskSpec,
    apply_fit_mask,
    broad_telluric_catalog_fallback_mask,
    build_mask,
    combine_exclusion_masks,
    compose_fit_mask,
    convert_mask_polarity,
    dilate_boolean_mask,
    exclusion_mask,
    find_known_nonstellar_features,
    nonstellar_feature_mask,
    nonstellar_feature_masks,
    nonstellar_feature_regions,
    overlapping_nonstellar_features,
    telluric_transmission_exclusion_mask,
    wavelength_region_exclusion_mask,
)


def test_segment_default_mask_rejects_invalid_data_and_errors():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, np.nan, 5003.0],
        flux=[1.0, np.nan, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.0],
    )

    assert np.array_equal(segment.mask, [True, False, False, False])


def test_segment_valid_and_invalid_mask_aliases_and_constructors():
    invalid_mask = np.array([False, True, False])
    segment = SpectrumSegment.from_invalid_mask(
        wave=[1.0, 2.0, 3.0],
        flux=[1.0, 1.0, 1.0],
        invalid_mask=invalid_mask,
    )

    assert np.array_equal(segment.mask, [True, False, True])
    assert np.array_equal(segment.valid_mask, segment.mask)
    assert np.array_equal(segment.use_mask, segment.mask)
    assert np.array_equal(segment.invalid_mask, invalid_mask)
    assert np.array_equal(
        convert_mask_polarity(invalid_mask, input_true_means="reject", output_true_means="use"),
        segment.mask,
    )

    valid_segment = SpectrumSegment.from_valid_mask(
        wave=[1.0, 2.0, 3.0],
        flux=[1.0, 1.0, 1.0],
        valid_mask=[True, False, True],
    )
    assert np.array_equal(valid_segment.mask, segment.mask)


def test_segment_and_collection_summaries_are_public_display_helpers():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 0.9, np.nan],
        err=[0.1, 0.1, 0.1],
        mask=[True, False, True],
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=5000.0),
        meta={"ingestion": [{"source": "reader:test_reader"}]},
        name="demo",
    )
    collection = SpectrumCollection([segment])

    summary = segment.summary()
    provenance = segment.provenance_summary()
    collection_summary = collection.summary()

    assert summary["type"] == "SpectrumSegment"
    assert summary["reader"] == "test_reader"
    assert summary["n_pixels"] == 3
    assert summary["n_valid_pixels"] == 1
    assert summary["resolution"]["quantity"] == "R"
    assert provenance["reader"] == "test_reader"
    assert collection_summary["type"] == "SpectrumCollection"
    assert collection_summary["readers"] == ["test_reader"]


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
        exclude_masks=[("telluric", telluric), {"name": "line_core", "callable": line_core}],
    )

    assert np.array_equal(result.effective_mask, [True, False, True, False, True])
    assert np.array_equal(result.excluded_mask, [False, True, False, True, False])
    assert np.array_equal(result.fit_use_mask, result.effective_mask)
    assert np.array_equal(result.explicit_exclusion_mask, result.excluded_mask)
    assert np.array_equal(result.fit_rejection_mask, ~result.effective_mask)
    assert np.array_equal(
        result.rejection_masks["exclude_mask:telluric"],
        [False, True, False, False, False],
    )
    assert np.array_equal(
        result.rejection_masks["exclude_mask:line_core"],
        [False, False, False, True, False],
    )
    assert result.settings["exclude_masks"] == ["telluric", "line_core"]
    assert result.settings["exclude_masks_api"] == "exclude_masks"
    assert result.settings["exclude_mask_summary_kind"] == "union"
    assert result.settings["exclude_mask_union_derived_from"] == ["telluric", "line_core"]
    assert result.settings["mask_true_means"] == "use"
    assert result.settings["exclude_mask_true_means"] == "reject"
    assert result.settings["numeric_mask_reject_if"] == "> threshold"
    assert result.counts["n_rejected_by_explicit_union"] == 2
    assert result.counts["n_rejected_by_multiple_reasons"] == 0


def test_build_mask_exposes_single_segment_valid_mask_and_summary():
    segment = SpectrumSegment(
        wave=np.linspace(4410.0, 4450.0, 41),
        flux=np.ones(41),
        err=np.full(41, 0.02),
        wave_medium="air",
    )

    bundle = build_mask(segment, names="dib_4428")

    assert len(bundle) == 1
    assert bundle.valid_mask.shape == segment.wave.shape
    assert not np.all(bundle.valid_mask)
    summary = bundle.summary()
    assert summary["n_exclusion_masks"] == 1
    assert summary["has_valid_mask"] is True
    assert "Spyctres mask bundle" in bundle.summary_text()


def test_exclusion_mask_apis_are_mutually_exclusive():
    segment = SpectrumSegment(
        wave=np.arange(3.0),
        flux=np.ones(3),
        err=np.ones(3),
    )

    def legacy_mask(wave):
        return wave == 1.0

    def named_mask(wave):
        return wave == 2.0

    with pytest.raises(ValueError, match="either exclude_mask or exclude_masks"):
        compose_fit_mask(
            segment,
            exclude_mask=legacy_mask,
            exclude_masks=[("named", named_mask)],
        )

    with pytest.raises(ValueError, match="either exclude_mask or exclude_masks"):
        _build_data_vectors(
            [segment],
            exclude_mask=legacy_mask,
            exclude_masks=[("named", named_mask)],
        )


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


def test_threshold_equality_keeps_pixel_and_nonfinite_numeric_masks_reject():
    segment = SpectrumSegment(
        wave=np.arange(4.0),
        flux=np.ones(4),
        err=np.ones(4),
    )

    def soft_exclusion(_wave):
        return np.array([0.5, np.nan, np.inf, 0.6])

    result = compose_fit_mask(segment, exclude_masks=[("soft", soft_exclusion)])

    assert np.array_equal(result.rejection_masks["exclude_mask:soft"], [False, True, True, True])
    assert np.array_equal(result.rejection_masks["nonfinite_mask_output:soft"], [False, True, True, False])
    assert np.array_equal(result.effective_mask, [True, False, False, False])
    assert result.settings["nonfinite_mask_value_policy"] == "reject"
    assert result.counts["nonfinite_mask_output"] == 2


def test_exclusion_mask_spec_helper_and_duplicate_name_validation():
    segment = SpectrumSegment(
        wave=[1.0, 2.0, 3.0, 4.0],
        flux=[1.0, 1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.1],
    )

    spec = exclusion_mask("central", lambda wave: (wave >= 2.0) & (wave <= 3.0))
    assert isinstance(spec, ExclusionMaskSpec)

    result = compose_fit_mask(segment, exclude_mask=[spec])

    assert np.array_equal(result.effective_mask, [True, False, False, True])
    assert result.settings["exclude_masks"] == ["central"]
    assert result.counts["exclude_mask:central"] == 2

    with pytest.raises(ValueError, match="Duplicate exclusion mask name"):
        compose_fit_mask(
            segment,
            exclude_mask=[
                exclusion_mask("central", lambda wave: wave == 1.0),
                exclusion_mask("central", lambda wave: wave == 4.0),
            ],
        )


def test_wavelength_region_exclusion_mask_rejects_intervals_and_records_metadata():
    segment = SpectrumSegment(
        wave=[3999.0, 4001.0, 4003.0, 4006.0],
        flux=[1.0, 1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.1],
    )

    mask = wavelength_region_exclusion_mask(
        "manual_bad_region",
        [(4000.0, 4004.0)],
        metadata={"reason": "visual inspection"},
    )
    result = compose_fit_mask(segment, exclude_masks=[mask])

    assert np.array_equal(result.effective_mask, [True, False, False, True])
    assert result.settings["exclude_masks"] == ["manual_bad_region"]
    metadata = result.settings["exclude_mask_metadata"]["manual_bad_region"]
    assert metadata["method"] == "wavelength_intervals"
    assert metadata["regions_A"] == [[4000.0, 4004.0]]
    assert metadata["reason"] == "visual inspection"


def test_nonstellar_feature_mask_rejects_dib_region_and_records_name():
    segment = SpectrumSegment(
        wave=[4410.0, 4428.8, 4440.0, 4460.0],
        flux=[1.0, 1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.1],
    )

    mask = nonstellar_feature_mask("dib_4428")
    result = compose_fit_mask(segment, exclude_masks=[mask])

    assert "dib_4428" in NONSTELLAR_FEATURES
    assert nonstellar_feature_regions("dib_4428") == [(4416.8, 4440.8)]
    assert np.array_equal(result.effective_mask, [True, False, False, True])
    assert result.settings["exclude_masks"] == ["nonstellar:dib_4428"]
    assert result.counts["exclude_mask:nonstellar:dib_4428"] == 2


def test_dib_4882_feature_uses_balmer_overlap_region():
    regions = nonstellar_feature_regions("dib_4882")
    masks = nonstellar_feature_masks(["dib_4428", "dib_4882"])

    assert regions == [(4870.0, 4915.0)]
    assert [mask.name for mask in masks] == [
        "nonstellar:dib_4428",
        "nonstellar:dib_4882",
    ]
    assert "Hbeta" in NONSTELLAR_FEATURES["dib_4882"].diagnostic_lines
    assert NONSTELLAR_FEATURES["dib_4882"].frame_type == "ism_velocity"


def test_telluric_features_are_topocentric_fixed_and_cross_reference_dib():
    o2_a = NONSTELLAR_FEATURES["telluric_o2_a_7605"]
    o2_6280 = NONSTELLAR_FEATURES["telluric_o2_gamma_6280"]
    region = nonstellar_feature_regions("telluric_o2_a_7605")

    assert o2_a.kind == "telluric_band"
    assert o2_a.frame_type == "topocentric_fixed"
    assert o2_a.feature_frame == "topocentric"
    assert o2_a.velocity_margin_kms is None
    assert region == [(7550.0, 7660.0)]
    assert "dib_6284" in o2_6280.cross_references
    assert NONSTELLAR_FEATURES["telluric_o2_alpha_6280"] is o2_6280
    assert nonstellar_feature_regions("telluric_o2_alpha_6280") == [
        (6260.0, 6300.0)
    ]


def test_telluric_transmission_exclusion_mask_records_precise_mask_provenance():
    def loader(threshold):
        assert threshold == pytest.approx(0.9)

        def transmission(wave):
            return np.ones_like(np.asarray(wave, dtype=float))

        def mask(wave):
            wave = np.asarray(wave, dtype=float)
            return np.where((wave >= 2.0) & (wave <= 3.0), 1.0, 0.0)

        return transmission, mask

    segment = SpectrumSegment(
        wave=[1.0, 2.0, 3.0, 4.0],
        flux=[1.0, 1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.1],
        observer_frame="topocentric",
        stellar_rest_status="raw",
    )

    mask = telluric_transmission_exclusion_mask(threshold=0.9, loader=loader)
    result = compose_fit_mask(segment, exclude_masks=[mask])

    assert np.array_equal(result.effective_mask, [True, False, False, True])
    metadata = result.settings["exclude_mask_metadata"][mask.name]
    assert metadata["method"] == "transmission_threshold"
    assert metadata["threshold"] == pytest.approx(0.9)
    assert metadata["coarse_mask"] is False
    assert metadata["fallback_broad_regions_used"] is False
    assert metadata["model_file"] == "LBL_A10_s0_w050_R0300000_T.fits"
    assert result.settings["telluric_mask_frame_warnings"] == []


def test_topocentric_telluric_mask_warns_on_stellar_rest_frame():
    def loader(_threshold):
        return (
            lambda wave: np.ones_like(np.asarray(wave, dtype=float)),
            lambda wave: np.asarray(wave, dtype=float) > 1.0,
        )

    segment = SpectrumSegment(
        wave=[1.0, 2.0, 3.0],
        flux=[1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1],
        observer_frame="barycentric",
        stellar_rest_status="corrected",
    )

    mask = telluric_transmission_exclusion_mask(threshold=0.95, loader=loader)
    result = compose_fit_mask(segment, exclude_masks=[mask])
    warning = result.settings["telluric_mask_frame_warnings"][0]

    assert warning["warning"] == "telluric_mask_frame_ambiguous"
    assert warning["observer_frame"] == "barycentric"
    assert "telluric_mask_frame_ambiguous" in result.settings["quality_flags"]


def test_catalog_telluric_mask_is_labelled_coarse_and_warns_on_frame():
    segment = SpectrumSegment(
        wave=np.linspace(7550.0, 7660.0, 12),
        flux=np.ones(12),
        err=np.ones(12),
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )

    mask = nonstellar_feature_mask("telluric_o2_a_7605")
    result = compose_fit_mask(segment, exclude_masks=[mask])
    metadata = result.settings["exclude_mask_metadata"][mask.name]

    assert metadata["method"] == "broad_catalog_regions"
    assert metadata["coarse_mask"] is True
    assert metadata["warning"] == "coarse_telluric_mask"
    assert "coarse_telluric_mask_applied" in result.settings["quality_flags"]
    assert "telluric_mask_frame_ambiguous" in result.settings["quality_flags"]


def test_broad_telluric_catalog_fallback_is_not_preferred_actual_mask():
    segment = SpectrumSegment(
        wave=np.linspace(6840.0, 6900.0, 10),
        flux=np.ones(10),
        err=np.ones(10),
        observer_frame="topocentric",
        stellar_rest_status="raw",
    )

    mask = broad_telluric_catalog_fallback_mask(
        names=["telluric_o2_b_6867"],
        use_case="unit_test_quicklook",
    )
    result = compose_fit_mask(segment, exclude_masks=[mask])
    metadata = result.settings["exclude_mask_metadata"][mask.name]

    assert metadata["method"] == "broad_catalog_fallback"
    assert metadata["fallback_broad_regions_used"] is True
    assert metadata["preferred_for_actual_telluric_masking"] is False
    assert metadata["use_case"] == "unit_test_quicklook"
    assert "telluric_o2_b_6867" in metadata["fallback_broad_region_ids"]
    assert np.any(~result.effective_mask)


def test_find_known_nonstellar_features_searches_user_regions():
    matches = find_known_nonstellar_features(
        [
            {"label": "unknown dip near Hgamma", "region_A": (4415.0, 4445.0)},
            (4875.0, 4910.0),
        ]
    )

    ids = [item["id"] for item in matches]
    assert "dib_4428" in ids
    assert "dib_4882" in ids
    hgamma = next(item for item in matches if item["id"] == "dib_4428")
    assert hgamma["query_label"] == "unknown dip near Hgamma"
    assert hgamma["diagnostic_only"] is True
    assert hgamma["overlap_A"] > 0.0


def test_combine_exclusion_masks_preserves_component_metadata():
    first = exclusion_mask(
        "first",
        lambda wave: np.asarray(wave, dtype=float) > 2.0,
        metadata={"method": "first_method"},
    )
    second = exclusion_mask(
        "second",
        lambda wave: np.asarray(wave, dtype=float) == 1.0,
        metadata={"method": "second_method"},
    )

    combined = combine_exclusion_masks([first, second], name="combo")
    assert combined.name == "combo"
    assert np.array_equal(combined([0.0, 1.0, 2.0, 3.0]), [False, True, False, True])
    assert combined.metadata["method"] == "combined_exclusion_masks"
    assert combined.metadata["component_masks"] == ["first", "second"]
    assert (
        combined.metadata["component_mask_metadata"]["first"]["method"]
        == "first_method"
    )


def test_dilate_boolean_mask_grows_nearest_neighbours():
    assert np.array_equal(
        dilate_boolean_mask([False, False, True, False, False], n_pix=1),
        [False, True, True, True, False],
    )
    assert np.array_equal(
        dilate_boolean_mask([False, True, False], n_pix=0),
        [False, True, False],
    )


def test_overlapping_nonstellar_features_reports_valid_coverage_overlap():
    segment = SpectrumSegment(
        wave=np.linspace(4300.0, 4920.0, 80),
        flux=np.ones(80),
        err=np.ones(80),
        name="demo",
    )

    overlaps = overlapping_nonstellar_features(
        segment,
        names=["dib_4428", "dib_4882"],
    )

    by_name = {item["name"]: item for item in overlaps}
    assert set(by_name) == {"DIB 4428", "DIB 4882"}
    assert by_name["DIB 4428"]["center_A"] == pytest.approx(4428.8)
    assert by_name["DIB 4428"]["overlap_A"] > 0.0
    assert by_name["DIB 4882"]["overlap_A"] == pytest.approx(45.0)
    assert by_name["DIB 4882"]["overlap_pixels"] > 0
    assert by_name["DIB 4882"]["overlap_fraction_of_valid_pixels"] > 0.0
    assert by_name["DIB 4882"]["id"] == "dib_4882"
    assert by_name["DIB 4882"]["frame_type"] == "ism_velocity"
    assert by_name["DIB 4882"]["diagnostic_lines"] == ["Hbeta"]
    assert by_name["DIB 4882"]["segments"] == ["demo"]
    assert by_name["DIB 4882"]["segment_overlaps"][0]["segment"] == "demo"


def test_overlap_aware_counts_do_not_double_count_total_rejections():
    segment = SpectrumSegment(
        wave=np.arange(4.0),
        flux=np.ones(4),
        err=np.ones(4),
    )

    def telluric(wave):
        return (wave == 1.0) | (wave == 2.0)

    def core(wave):
        return wave == 2.0

    result = compose_fit_mask(
        segment,
        exclude_masks=[("telluric", telluric), ("core", core)],
    )

    assert result.counts["exclude_mask:telluric"] == 2
    assert result.counts["exclude_mask:core"] == 1
    assert result.counts["n_rejected_by_explicit_union"] == 2
    assert result.counts["n_rejected_total"] == 2
    assert result.counts["n_rejected_by_multiple_reasons"] == 1

    summary = result.to_summary()
    assert summary["n_pixels"] == 4
    assert summary["n_fit"] == 2
    assert summary["fit_fraction"] == 0.5
    assert summary["explicit_exclusion_fraction"] == 0.5
    assert summary["data_invalid_fraction"] == 0.0
    assert summary["multiple_rejection_fraction"] == 0.25
    assert summary["explicit_exclusion_counts"] == {"telluric": 2, "core": 1}
    assert summary["n_rejected_by_multiple_reasons"] == 1
    assert summary["mask_true_means"] == "use"
    assert summary["exclude_mask_true_means"] == "reject"


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
    assert np.array_equal(
        result.data_invalid_mask,
        [False, True, True, True, False],
    )


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


def test_fit_mask_summary_splits_window_trimming_from_inside_rejections():
    segment = SpectrumSegment(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        err=[0.1, 0.1, 0.1, 0.1, 0.1],
    )

    result = compose_fit_mask(
        segment,
        regions=[(2.0, 4.0)],
        exclude_regions=[(3.0, 3.0)],
    )
    summary = result.to_summary()

    assert summary["n_outside_fit_window"] == 2
    assert summary["outside_fit_window_fraction"] == pytest.approx(2.0 / 5.0)
    assert summary["n_inside_fit_window"] == 3
    assert summary["n_rejected_inside_fit_window"] == 1
    assert summary["rejected_inside_fit_window_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["rejected_fraction"] == pytest.approx(3.0 / 5.0)


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
        exclude_masks={
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


def test_build_data_vectors_applies_per_segment_error_floor_fraction():
    first = SpectrumSegment(
        [1.0, 2.0],
        [1.0, 1.0],
        err=[0.01, 0.01],
        name="first",
    )
    second = SpectrumSegment(
        [1.0, 2.0],
        [2.0, 2.0],
        err=[0.02, 0.02],
        name="second",
    )

    vectors = _build_data_vectors(
        [first, second],
        error_floor_fraction={"first": 0.1, 1: 0.2},
    )
    _support_wave, _flux, err, _support_slices, fit_slices, _fit_masks, _weights, seg_meta = vectors

    assert np.allclose(
        err[fit_slices[0]],
        np.sqrt(0.01**2 + 0.1**2),
    )
    assert np.allclose(
        err[fit_slices[1]],
        np.sqrt(0.02**2 + 0.4**2),
    )
    assert seg_meta[0]["error_floor"]["error_floor_fraction"] == 0.1
    assert seg_meta[0]["error_floor"]["error_floor_abs"] == pytest.approx(0.1)
    assert seg_meta[1]["error_floor"]["error_floor_fraction"] == 0.2
    assert seg_meta[1]["error_floor"]["error_floor_abs"] == pytest.approx(0.4)


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


def test_per_segment_mask_assignment_rejects_ambiguous_or_bad_keys():
    first = SpectrumSegment([1.0, 2.0], [1.0, 1.0], name="dup")
    second = SpectrumSegment([1.0, 2.0], [1.0, 1.0], name="dup")
    unique = SpectrumSegment([1.0, 2.0], [1.0, 1.0], name="unique")

    def mask(_wave):
        return [False, True]

    with pytest.raises(ValueError, match="not unique"):
        _build_data_vectors([first, second], exclude_masks={"dup": ("m", mask)})

    with pytest.raises(ValueError, match="out of range"):
        _build_data_vectors([unique], exclude_masks={2: ("m", mask)})

    with pytest.raises(ValueError, match="does not match"):
        _build_data_vectors([unique], exclude_masks={"missing": ("m", mask)})

    with pytest.raises(ValueError, match="by both"):
        _build_data_vectors([unique], exclude_masks={0: ("i", mask), "unique": ("n", mask)})


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
