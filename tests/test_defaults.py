import numpy as np
import pytest

from Spyctres import suggest_phoenix_fit_defaults
from Spyctres.defaults import prepare_phoenix_fit_kwargs
from Spyctres.io import SpectrumCollection, SpectrumSegment


def test_suggest_phoenix_fit_defaults_prefers_blue_optical_window():
    segment = SpectrumSegment(
        wave=np.linspace(3000.0, 5600.0, 100),
        flux=np.ones(100),
        err=np.full(100, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        meta={"instrument": "XSHOOTER", "arm": "UVB"},
        resolution=5000.0,
    )

    suggestion = suggest_phoenix_fit_defaults(segment)

    assert suggestion.fit_kwargs["forward_model"] == "native_interp"
    assert suggestion.fit_kwargs["regions"] == [(3800.0, 5200.0)]
    assert suggestion.fit_kwargs["physical_init"] == "coarse"
    assert suggestion.fit_kwargs["rv_grid_n"] == 41
    assert suggestion.fit_kwargs["bounds"][0][0] == pytest.approx(4500.0)
    assert suggestion.fit_kwargs["bounds"][1][0] == pytest.approx(10000.0)
    assert not suggestion.warnings
    assert suggestion.provenance["window"]["label"] == "blue_optical_classification"


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
    assert any("wavelength medium is unknown" in item for item in suggestion.warnings)
    assert any("lacks formal uncertainties" in item for item in suggestion.warnings)
    assert any("lacks resolution metadata" in item for item in suggestion.warnings)
    assert any("telluric catalog" in item for item in suggestion.warnings)
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
    assert suggestion.provenance["coverage"]["stellar_rest_status"] == ["corrected"]
    assert suggestion.provenance["telluric_catalog_policy"]["overlaps"]


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
