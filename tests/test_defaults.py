import numpy as np
import pytest

from Spyctres import suggest_phoenix_fit_defaults
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
