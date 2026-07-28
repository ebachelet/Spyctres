import numpy as np
import pytest

from Spyctres.io import ResolutionDescriptor, SpectrumCollection, SpectrumSegment
from Spyctres.recipes import sdss_quicklook_resolution_assumption
from Spyctres.preprocessing import (
    archive_exclusion_masks,
    archive_mask_catalog,
    artifact_exclusion_mask_from_segment,
    audit_spectrum_for_fit,
    publication_readiness_audit,
    readiness_flag_actions,
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
    actions = {item["flag"]: item for item in audit["recommended_actions"]}
    assert actions["wave_medium_unknown"]["severity"] == "blocker"
    assert "wave_medium='air'" in actions["wave_medium_unknown"]["action"]
    assert "1-sigma uncertainties" in actions["missing_uncertainties"]["action"]
    assert "recommended_actions" in audit["segments"][0]


def test_audit_intent_specific_readiness_allows_quicklook_metadata_warnings():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 1.1, 1.2],
        err=[0.1, 0.1, 0.1],
        wave_medium="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )

    audit = audit_spectrum_for_fit(
        segment,
        intent="quicklook_classification",
        assumed_resolution={
            "quantity": "R",
            "value": 2000.0,
            "source": "user_override",
        },
    )

    assert audit["fit_ready"] is False
    assert audit["quicklook_only"] is True
    assert audit["intent"] == "quicklook_classification"
    assert audit["ready_for_intent"] is True
    assert audit["blockers_for_intent"] == []
    assert "wave_medium_unknown" in audit["warnings_for_intent"]
    assert "observer_frame_unknown" in audit["warnings_for_intent"]
    assert "stellar_rest_status_unknown" in audit["warnings_for_intent"]
    assert "physical_radial_velocity" in audit["invalid_interpretations_for_intent"]
    actions = {item["flag"]: item for item in audit["actions_for_intent"]}
    assert actions["wave_medium_unknown"]["intent_severity"] == "warning"
    assert (
        audit["readiness_by_intent"]["radial_velocity"]["ready_for_intent"]
        is False
    )


def test_audit_radial_velocity_intent_blocks_frame_metadata():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 1.1, 1.2],
        err=[0.1, 0.1, 0.1],
        wave_medium="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )

    audit = audit_spectrum_for_fit(
        segment,
        intent="radial_velocity",
        assumed_resolution={
            "quantity": "R",
            "value": 2000.0,
            "source": "user_override",
        },
    )

    assert audit["ready_for_intent"] is False
    assert set(audit["blockers_for_intent"]) == {
        "observer_frame_unknown",
        "stellar_rest_status_unknown",
        "wave_medium_unknown",
    }
    assert "physical_radial_velocity" in audit["invalid_interpretations_for_intent"]
    actions = {item["flag"]: item for item in audit["actions_for_intent"]}
    assert actions["observer_frame_unknown"]["intent_severity"] == "blocker"


def test_audit_rejects_unknown_explicit_intent():
    segment = SpectrumSegment(
        wave=[5000.0, 5001.0, 5002.0],
        flux=[1.0, 1.1, 1.2],
        err=[0.1, 0.1, 0.1],
    )

    with pytest.raises(ValueError, match="Unknown readiness intent"):
        audit_spectrum_for_fit(segment, intent="precision_abundance")


def test_readiness_flag_actions_handles_unknown_flags():
    actions = readiness_flag_actions(["custom_future_flag"])

    assert actions == [
        {
            "flag": "custom_future_flag",
            "severity": "review",
            "action": (
                "Inspect this readiness flag in the fit-quality report "
                "before interpreting fitted parameters."
            ),
            "detail": "No specialized action has been registered for this flag.",
        }
    ]


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


def test_audit_does_not_treat_high_resolution_absorption_lines_as_spikes():
    wave = np.linspace(4800.0, 5200.0, 4000)
    flux = np.ones_like(wave)
    for center, depth, sigma in (
        (4861.3, 0.45, 0.45),
        (5167.3, 0.25, 0.12),
        (5172.7, 0.30, 0.12),
        (5183.6, 0.28, 0.12),
    ):
        flux -= depth * np.exp(-0.5 * ((wave - center) / sigma) ** 2)
    segment = SpectrumSegment(
        wave=wave,
        flux=flux,
        err=np.full(wave.size, 0.02),
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        resolution=ResolutionDescriptor(quantity="R", value=42000.0),
    )

    audit = audit_spectrum_for_fit(
        segment,
        fit_windows=[(4830.0, 4898.0), (5150.0, 5208.0)],
    )

    flags = set(audit["interpretation_flags"])
    assert "artifact_review_required" not in flags
    metrics = audit["segments"][0]["artifact_metrics"]
    assert metrics["extreme_spike_fraction"] == 0.0


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


def test_archive_mask_catalog_is_explicit_and_opt_in():
    catalog = archive_mask_catalog("uves_pop")
    masks = archive_exclusion_masks("uves_pop")

    assert any(item["id"] == "uves_pop_flag_flattening_5760_5840" for item in catalog)
    assert len(masks) == len(catalog)
    assert masks[0].metadata["automatic_reader_default"] is False


def test_audit_records_archive_mask_overlap_without_applying_it():
    segment = SpectrumSegment(
        wave=[5750.0, 5775.0, 5800.0, 5850.0],
        flux=[1.0, 1.1, 1.2, 1.3],
        err=[0.1, 0.1, 0.1, 0.1],
        wave_medium="air",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=80000.0),
        meta={"archive_mask_catalog": archive_mask_catalog("uves_pop")},
    )

    audit = audit_spectrum_for_fit(segment, fit_windows=[(5700.0, 5900.0)])

    assert "archive_mask_overlap_inside_fit_window" in audit["interpretation_flags"]
    assert "archive_mask_fraction_high" in audit["interpretation_flags"]
    summary = audit["segments"][0]["archive_mask_summary"]
    assert summary["n_pixels_inside_catalog_regions"] >= 2
    assert summary["masks_applied"] == []
    assert "uves_pop_flag_flattening_5760_5840" in summary["masks_not_applied"]


def test_audit_archive_mask_application_removes_fit_overlap_blocking_flags():
    segment = SpectrumSegment(
        wave=[5750.0, 5775.0, 5800.0, 5850.0],
        flux=[1.0, 1.1, 1.2, 1.3],
        err=[0.1, 0.1, 0.1, 0.1],
        wave_medium="air",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=80000.0),
        meta={"archive_mask_catalog": archive_mask_catalog("uves_pop")},
    )
    masks = archive_exclusion_masks("uves_pop")

    audit = audit_spectrum_for_fit(
        segment,
        fit_windows=[(5700.0, 5900.0)],
        exclude_masks=masks,
    )

    assert "archive_mask_overlap_inside_fit_window" not in audit["interpretation_flags"]
    assert "archive_mask_fraction_high" not in audit["interpretation_flags"]
    assert audit["n_fit_candidate"] < audit["n_inside_fit_window"]
    summary = audit["segments"][0]["archive_mask_summary"]
    assert summary["n_pixels_inside_catalog_regions"] >= 2
    assert summary["n_fit_candidate_pixels_inside_catalog_regions"] == 0
    mask_provenance = audit["segments"][0]["metadata"]["mask_provenance"]
    assert "archive:uves_pop_flag_flattening_5760_5840" in mask_provenance[
        "settings"
    ]["exclude_masks"]


def test_publication_readiness_blocks_quicklook_only_metadata():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 5050.0, 250),
        flux=np.ones(250),
        wave_medium="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )

    readiness = publication_readiness_audit(segment, min_fit_pixels=10)

    assert readiness["publication_ready"] is False
    blockers = set(readiness["blockers"])
    assert "missing_uncertainties" in blockers
    assert "resolution_assumption_required" in blockers
    assert "wave_medium_unknown" in blockers
    assert "observer_frame_unknown" in blockers
    assert "stellar_rest_status_unknown" in blockers
    assert readiness["audit"]["quicklook_only"] is True


def test_publication_readiness_requires_validated_resolution_by_default():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 5050.0, 250),
        flux=np.ones(250),
        err=np.full(250, 0.1),
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
    )

    readiness = publication_readiness_audit(
        segment,
        min_fit_pixels=10,
        assumed_resolution={
            "quantity": "R",
            "value": 2000.0,
            "source": "user_override",
        },
    )

    assert readiness["audit"]["fit_ready"] is True
    assert readiness["publication_ready"] is False
    assert "resolution_is_assumed_not_validated" in readiness["blockers"]

    allowed = publication_readiness_audit(
        segment,
        min_fit_pixels=10,
        assumed_resolution={
            "quantity": "R",
            "value": 2000.0,
            "source": "user_override",
        },
        allow_assumed_resolution=True,
    )
    assert allowed["publication_ready"] is True


def test_publication_readiness_passes_documented_fit_ready_segment():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 5050.0, 250),
        flux=np.ones(250),
        err=np.full(250, 0.1),
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
    )

    readiness = publication_readiness_audit(segment, min_fit_pixels=10)

    assert readiness["publication_ready"] is True
    assert readiness["blockers"] == []


def test_publication_readiness_blocks_unapplied_sdss_wdisp_by_default():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 5050.0, 250),
        flux=np.ones(250),
        err=np.full(250, 0.1),
        wave_medium="vacuum",
        observer_frame="heliocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=2000.0),
        meta={
            "sdss_lsf": {
                "present": True,
                "lsf_source": "sdss_wdisp_not_applied",
                "attach_wdisp_resolution": False,
            }
        },
    )

    readiness = publication_readiness_audit(segment, min_fit_pixels=10)

    assert readiness["publication_ready"] is False
    assert "sdss_wdisp_lsf_not_applied" in readiness["blockers"]

    allowed = publication_readiness_audit(
        segment,
        min_fit_pixels=10,
        allow_sdss_wdisp_not_applied=True,
    )
    assert allowed["publication_ready"] is True
    assert "sdss_wdisp_lsf_not_applied" in allowed["warnings"]
