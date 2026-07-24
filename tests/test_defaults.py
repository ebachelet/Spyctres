import numpy as np
import pytest

from Spyctres import (
    suggest_classification_branches,
    suggest_fit_setup,
    suggest_phoenix_fit_defaults,
)
from Spyctres.defaults import prepare_phoenix_fit_kwargs
from Spyctres.io import SpectrumCollection, SpectrumSegment


def test_suggest_phoenix_fit_defaults_prefers_blue_optical_window():
    segment = SpectrumSegment(
        wave=np.linspace(3000.0, 5600.0, 1000),
        flux=np.ones(1000),
        err=np.full(1000, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        meta={"instrument": "XSHOOTER", "arm": "UVB"},
        resolution=5000.0,
    )

    suggestion = suggest_phoenix_fit_defaults(segment)

    assert suggestion.fit_kwargs["forward_model"] == "native_interp"
    assert suggestion.fit_kwargs["regions"] == [
        (3900.0, 3998.0),
        (4070.0, 4135.0),
        (4205.0, 4248.0),
        (4310.0, 4372.0),
        (4830.0, 4898.0),
        (5150.0, 5208.0),
    ]
    assert suggestion.fit_kwargs["physical_init"] == "coarse"
    assert suggestion.fit_kwargs["rv_grid_n"] == 41
    assert suggestion.fit_kwargs["bounds"][0][0] == pytest.approx(4500.0)
    assert suggestion.fit_kwargs["bounds"][1][0] == pytest.approx(12000.0)
    assert not suggestion.warnings
    assert (
        suggestion.provenance["window"]["label"]
        == "classification_branch:blue_optical_balmer_metal"
    )
    branch_plan = suggestion.provenance["classification_branches"]
    assert branch_plan["recommended_branch_id"] == "blue_optical_balmer_metal"
    assert branch_plan["policy"]["branches_are_not_final_spectral_types"] is True
    assert branch_plan["recommended_branch"]["fit_window_ids"] == [
        "ca_hk_h_epsilon",
        "h_delta",
        "ca_i_4227",
        "h_gamma",
        "h_beta",
        "mg_i_b",
    ]
    diagnostic_windows = suggestion.provenance["diagnostic_windows"]
    selected_ids = {
        item["id"] for item in diagnostic_windows["selection"]["selected"]
    }
    assert {"h_beta", "h_gamma", "h_delta"} <= selected_ids
    assert "co_23um_bandhead" not in selected_ids
    assert diagnostic_windows["recommended_combinations"]["expensive_fits_run"] is False
    interpretation = suggestion.provenance["interpretation"]
    assert interpretation["intended_use"] == "first_pass_classification"
    assert interpretation["rv_role"] == "candidate_stellar_rv"
    assert interpretation["risk_flags"] == []
    assert interpretation["automatic_choices_are_overridable"] is True
    assert interpretation["final_science_ready_by_default"] is False
    assert interpretation["mode_policy"]["fit_stage"] == "triage_first_pass"
    assert interpretation["mode_policy"]["search_budget"] == "light"
    assert suggestion.provenance["mode_policy"]["multistart"] == 1
    assert suggestion.provenance["mode_policy"]["rv_grid_n"] == 41
    assert any("quicklook defaults mode" in item for item in suggestion.reasons)
    assert any("branch-aware first-pass" in item for item in suggestion.reasons)


def test_suggest_fit_setup_wraps_defaults_and_readiness_for_users():
    segment = SpectrumSegment(
        wave=np.linspace(3000.0, 5600.0, 1000),
        flux=np.ones(1000),
        err=np.full(1000, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        meta={"instrument": "XSHOOTER", "arm": "UVB"},
        resolution=500.0,
    )

    setup = suggest_fit_setup(segment)

    assert setup["operation"] == "suggest_fit_setup"
    assert setup["model"] == "phoenix"
    assert setup["minimal_fit_call"] == "fit_stellar_spectrum(spec, model='phoenix')"
    assert setup["recommended_branch_id"] == "blue_optical_balmer_metal"
    assert setup["fit_kwargs"]["forward_model"] == "native_interp"
    assert setup["readiness"]["fit_ready"] is True
    assert setup["readiness"]["n_fit_candidate"] > 0
    assert setup["risk_flags"] == []
    assert any("metadata" in item.lower() for item in setup["next_steps"])
    assert setup["provenance"]["readiness_included"] is True


def test_suggest_fit_setup_reports_unknown_metadata_as_review_flags():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 7100.0, 100),
        flux=np.ones(100),
        err=None,
        wave_medium="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
        meta={"instrument": "GMOS"},
    )

    setup = suggest_fit_setup(segment)

    assert setup["readiness"]["fit_ready"] is False
    assert "unknown_wave_medium" in setup["risk_flags"]
    assert "wave_medium_unknown" in setup["risk_flags"]
    assert "missing_uncertainties" in setup["risk_flags"]
    assert "resolution_assumption_required" in setup["risk_flags"]
    assert any("Readiness audit is not fit-ready" in item for item in setup["next_steps"])
    assert any(
        item.startswith("wave_medium_unknown:")
        and "wave_medium='air'" in item
        for item in setup["next_steps"]
    )


def test_suggest_fit_setup_assumed_resolution_is_not_reported_as_missing():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 7100.0, 100),
        flux=np.ones(100),
        err=np.full(100, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        meta={"instrument": "SDSS"},
        resolution=None,
    )

    setup = suggest_fit_setup(segment, assumed_resolution=2000.0)

    assert setup["fit_kwargs"]["R"] == pytest.approx(2000.0)
    assert "missing_resolution" not in setup["risk_flags"]
    assert "resolution_assumption_required" not in setup["risk_flags"]
    assert any("R=2000" in item for item in setup["warnings"])


def test_suggest_fit_setup_is_explicitly_phoenix_only_for_now():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 5100.0, 20),
        flux=np.ones(20),
        err=np.full(20, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        resolution=1000.0,
    )

    with pytest.raises(ValueError, match="model='phoenix'"):
        suggest_fit_setup(segment, model="other")


def test_suggest_phoenix_fit_defaults_records_unknown_metadata_warnings():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 7100.0, 100),
        flux=np.ones(100),
        err=None,
        wave_medium="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
        meta={"instrument": "GMOS"},
    )

    suggestion = suggest_phoenix_fit_defaults(segment)

    assert suggestion.fit_kwargs["regions"] == [(5200.0, 7000.0)]
    assert suggestion.fit_kwargs["bounds"][0][0] == pytest.approx(3500.0)
    branch_plan = suggestion.provenance["classification_branches"]
    assert branch_plan["recommended_branch_id"] is None
    assert branch_plan["policy"]["default_action"] == "fallback_to_coverage_window"
    assert any("wavelength medium is unknown" in item for item in suggestion.warnings)
    assert any("lacks formal uncertainties" in item for item in suggestion.warnings)
    assert any("lacks resolution metadata" in item for item in suggestion.warnings)
    assert any("telluric catalog" in item for item in suggestion.warnings)
    interpretation = suggestion.provenance["interpretation"]
    assert interpretation["rv_role"] == "alignment_parameter_until_metadata_verified"
    assert "unknown_wave_medium" in interpretation["risk_flags"]
    assert "unknown_observer_frame" in interpretation["risk_flags"]
    assert "stellar_rest_status_unknown" in interpretation["risk_flags"]
    assert "missing_uncertainties" in interpretation["risk_flags"]
    assert "missing_resolution" in interpretation["risk_flags"]
    assert "broad_telluric_catalog_overlap" in interpretation["risk_flags"]
    telluric_policy = suggestion.provenance["telluric_catalog_policy"]
    assert telluric_policy["default_action"] == "warn_only"
    assert telluric_policy["actual_masking_preference"] == "transmission_threshold"
    assert telluric_policy["recommended_helper"] == "telluric_transmission_exclusion_mask"
    assert any(item["id"] == "telluric_o2_b_6867" for item in telluric_policy["overlaps"])


def test_suggest_phoenix_fit_defaults_uses_shorter_rv_scan_for_stellar_rest_collection():
    first = SpectrumSegment(
        wave=np.linspace(3900.0, 5000.0, 50),
        flux=np.ones(50),
        err=np.full(50, 0.1),
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        resolution=7000.0,
    )
    second = SpectrumSegment(
        wave=np.linspace(5200.0, 6500.0, 50),
        flux=np.ones(50),
        err=np.full(50, 0.1),
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        resolution=7000.0,
    )

    suggestion = suggest_phoenix_fit_defaults(
        SpectrumCollection([first, second]),
        mode="standard",
    )

    assert suggestion.fit_kwargs["regions"] == [(3800.0, 7000.0)]
    assert suggestion.fit_kwargs["multistart"] == 2
    assert suggestion.fit_kwargs["rv_grid_n"] == 21
    assert suggestion.provenance["mode_policy"]["mode"] == "standard"
    assert suggestion.provenance["mode_policy"]["search_budget"] == "moderate"
    assert suggestion.provenance["mode_policy"]["rv_grid_n"] == 21
    assert suggestion.provenance["coverage"]["stellar_rest_status"] == ["corrected"]
    assert (
        suggestion.provenance["interpretation"]["rv_role"]
        == "rest_frame_consistency_check"
    )
    assert suggestion.provenance["telluric_catalog_policy"]["overlaps"]
    assert (
        suggestion.provenance["classification_branches"]["recommended_branch_id"]
        is None
    )


def test_suggest_phoenix_fit_defaults_uses_dense_red_cool_branch():
    segment = SpectrumSegment(
        wave=np.linspace(6500.0, 8900.0, 1600),
        flux=np.ones(1600),
        err=np.full(1600, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        resolution=5000.0,
    )

    suggestion = suggest_phoenix_fit_defaults(segment)

    assert (
        suggestion.provenance["classification_branches"]["recommended_branch_id"]
        == "cool_red_optical_molecular"
    )
    selected = {
        window_id
        for window_id in suggestion.provenance["classification_branches"][
            "recommended_branch"
        ]["fit_window_ids"]
    }
    assert {"h_alpha", "tio_7050", "ca_ii_triplet_paschen"} <= selected
    assert suggestion.fit_kwargs["bounds"][0][0] == pytest.approx(3000.0)


def test_suggest_classification_branches_identifies_cool_kband_branch():
    segment = SpectrumSegment(
        wave=np.linspace(21800.0, 23800.0, 700),
        flux=np.ones(700),
        err=np.full(700, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        resolution=4000.0,
    )

    branch_plan = suggest_classification_branches(segment)

    assert branch_plan["operation"] == "suggest_classification_branches"
    assert branch_plan["recommended_branch_id"] == "cool_near_ir_molecular"
    assert branch_plan["recommended_branch"]["fit_window_ids"] == [
        "na_i_kband",
        "ca_i_kband",
        "co_23um_bandhead",
    ]
    assert branch_plan["policy"]["stress_only_windows_do_not_drive_default_fit"] is True


def test_prepare_phoenix_fit_kwargs_applies_expert_overrides_and_clips_grids():
    segment = SpectrumSegment(
        wave=np.linspace(3600.0, 5600.0, 100),
        flux=np.ones(100),
        err=np.full(100, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        meta={"instrument": "XSHOOTER", "arm": "UVB"},
        resolution=5000.0,
    )

    fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
        segment,
        p0_overrides=(6200.0, None, None, 12.0),
        lower_bound_overrides=(6000.0, -0.5, None, None),
        upper_bound_overrides=(7000.0, 0.0, None, None),
        window=(4000.0, 5000.0),
        resolution_R=7000.0,
    )

    assert suggestion is not None
    assert fit_kwargs["p0"] == pytest.approx((6200.0, 0.0, 4.0, 12.0))
    assert fit_kwargs["bounds"][0][:2] == pytest.approx((6000.0, -0.5))
    assert fit_kwargs["bounds"][1][:2] == pytest.approx((7000.0, 0.0))
    assert fit_kwargs["regions"] == [(4000.0, 5000.0)]
    assert fit_kwargs["R"] == pytest.approx(7000.0)
    assert fit_kwargs["coarse_teff_grid"] == [6000.0]
    assert fit_kwargs["coarse_feh_grid"] == [0.0]


def test_prepare_phoenix_fit_kwargs_can_run_without_auto_defaults():
    segment = SpectrumSegment(
        wave=np.linspace(5000.0, 5100.0, 10),
        flux=np.ones(10),
        err=np.full(10, 0.1),
    )

    fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
        segment,
        auto_defaults=False,
        p0_overrides=(5000.0, -0.2, 4.0, 0.0),
        window=(None, 5090.0),
    )

    assert suggestion is None
    assert fit_kwargs["p0"] == pytest.approx((5000.0, -0.2, 4.0, 0.0))
    assert fit_kwargs["regions"] == [(5000.0, 5090.0)]
    assert fit_kwargs["rv_grid_n"] == 41
