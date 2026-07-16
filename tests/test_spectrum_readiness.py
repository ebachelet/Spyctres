import numpy as np

from Spyctres.io import ResolutionDescriptor, SpectrumCollection, SpectrumSegment
from Spyctres.recipes import sdss_quicklook_resolution_assumption
from Spyctres.preprocessing import (
    artifact_exclusion_mask_from_segment,
    audit_spectrum_for_fit,
)


def test_audit_records_mask_polarity_and_window_fractions():
    segment = SpectrumSegment(
        wave=[3900.0, 4000.0, 4100.0, 5300.0],
        flux=[1.0, 1.1, 1.2, 1.3],
        mask=[True, False, True, True],
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
    )

    audit = audit_spectrum_for_fit(segment, fit_windows=[(3950.0, 4200.0)])

    assert audit["mask_true_means"] == "use"
    assert audit["n_total"] == 4
    assert audit["n_inside_fit_window"] == 2
    assert audit["n_fit_candidate"] == 1
    assert audit["outside_fit_window_fraction"] == 0.5
    assert audit["rejected_inside_fit_window_fraction"] == 0.5


def test_audit_unknown_metadata_and_missing_resolution_are_quicklook_only():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 1.1, 1.2],
        wave_medium="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )

    audit = audit_spectrum_for_fit(segment)

    assert audit["fit_ready"] is False
    assert audit["quicklook_only"] is True
    assert "missing_uncertainties" in audit["interpretation_flags"]
    assert "resolution_assumption_required" in audit["interpretation_flags"]
    assert "wave_medium_unknown" in audit["interpretation_flags"]
    assert "observer_frame_unknown" in audit["interpretation_flags"]
    assert "stellar_rest_status_unknown" in audit["interpretation_flags"]


def test_audit_user_resolution_override_removes_resolution_missing_flag():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 1.1, 1.2],
        err=[0.1, 0.1, 0.1],
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
    )

    audit = audit_spectrum_for_fit(
        segment,
        assumed_resolution={
            "quantity": "R",
            "value": 2000.0,
            "source": "user_override",
        },
    )

    assert audit["fit_ready"] is True
    assert "resolution_assumption_required" not in audit["interpretation_flags"]
    resolution = audit["segments"][0]["metadata"]["resolution"]
    assert resolution["source"] == "user_override"
    assert resolution["value"] == 2000.0


def test_audit_flags_obvious_artifacts_inside_fit_window_only():
    segment = SpectrumSegment(
        wave=[3900.0, 4000.0, 4001.0, 4002.0, 4003.0, 4004.0, 5300.0],
        flux=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, np.nan],
        err=[0.1] * 7,
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
    )

    audit = audit_spectrum_for_fit(
        segment,
        fit_windows=[(3999.0, 4010.0)],
        flat_block_min=3,
    )

    flags = set(audit["interpretation_flags"])
    assert "artifact_review_required" in flags
    assert "flat_zero_block_detected" in flags
    assert audit["segments"][0]["artifact_metrics"]["flat_zero_block_count"] == 1


def test_audit_flags_lsf_undersampling_when_pixels_are_too_coarse():
    segment = SpectrumSegment(
        wave=[5000.0, 5010.0, 5020.0, 5030.0],
        flux=[1.0, 1.1, 1.2, 1.3],
        err=[0.1, 0.1, 0.1, 0.1],
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=5000.0),
    )

    audit = audit_spectrum_for_fit(segment)

    assert "lsf_undersampled" in audit["interpretation_flags"]
    assert audit["segments"][0]["sampling_metrics"]["pixels_per_fwhm_median"] < 2.0


def test_audit_aggregates_spectrum_collection():
    first = SpectrumSegment(
        wave=[4000.0, 4001.0],
        flux=[1.0, 1.1],
        err=[0.1, 0.1],
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
        name="first",
    )
    second = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 1.1, 1.2],
        err=[0.1, 0.1, 0.1],
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
        name="second",
    )

    audit = audit_spectrum_for_fit(SpectrumCollection([first, second]))

    assert audit["n_segments"] == 2
    assert audit["n_total"] == 5
    assert audit["n_fit_candidate"] == 5
    assert audit["fit_ready"] is True


def test_artifact_exclusion_mask_is_explicit_same_grid_fallback():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 0.0, np.nan],
        err=[0.1, 0.1, 0.1],
        mask=[True, True, True],
    )

    spec = artifact_exclusion_mask_from_segment(segment)

    assert np.array_equal(spec(segment.wave), [False, True, True])
    assert spec.metadata["method"] == "fallback_artifact_same_grid"
    assert spec.metadata["same_grid_required"] is True


def test_artifact_exclusion_mask_rejects_different_grid():
    segment = SpectrumSegment(wave=[5000.0, 5001.0], flux=[1.0, 0.0])
    spec = artifact_exclusion_mask_from_segment(segment)

    with np.testing.assert_raises_regex(ValueError, "same-grid only"):
        spec([5000.0, 5001.5])


def test_sdss_quicklook_resolution_assumption_is_explicit():
    payload = sdss_quicklook_resolution_assumption()

    assert payload["quantity"] == "R"
    assert payload["value"] == 2000.0
    assert payload["reader_default_resolution"] is None
    assert "quicklook" in payload["assumption_warning"]


def test_audit_warns_when_sdss_wdisp_is_present_but_constant_r_is_used():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0, 5003.0],
        flux=[1.0, 1.1, 1.2, 1.3],
        err=[0.1, 0.1, 0.1, 0.1],
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        meta={
            "sdss_lsf": {
                "present": True,
                "lsf_source": "sdss_wdisp_not_applied",
                "attach_wdisp_resolution": False,
            }
        },
    )

    audit = audit_spectrum_for_fit(
        segment,
        assumed_resolution={
            "quantity": "R",
            "value": 2000.0,
            "source": "user_override",
        },
    )

    assert audit["fit_ready"] is True
    assert "sdss_wdisp_lsf_not_applied" in audit["interpretation_flags"]
    assert audit["warnings"] == [
        "SDSS tabulated LSF present but not applied; using explicit constant-R assumption."
    ]
    lsf = audit["segments"][0]["metadata"]["lsf_provenance"]
    assert lsf["lsf_source"] == "sdss_wdisp_not_applied"
    assert lsf["active_lsf_convolution"] is False
