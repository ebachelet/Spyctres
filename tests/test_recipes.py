import pytest

import numpy as np

from Spyctres.io import ResolutionDescriptor, SpectrumSegment
from Spyctres.recipes import (
    _sideband_fit_parameter_count,
    normalize_segment_sidebands,
)


def test_sideband_parameter_count_uses_the_sideband_polynomial_order():
    assert _sideband_fit_parameter_count(3, 1) == 10
    assert _sideband_fit_parameter_count(3, 2) == 13


def test_sideband_parameter_count_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_segments"):
        _sideband_fit_parameter_count(0, 1)
    with pytest.raises(ValueError, match="sideband_poly_order"):
        _sideband_fit_parameter_count(1, -1)


def test_sideband_normalization_preserves_scientific_metadata():
    resolution = ResolutionDescriptor(quantity="R", value=5100.0)
    segment = SpectrumSegment(
        wave=np.linspace(4000.0, 4020.0, 21),
        flux=np.linspace(1.0, 1.1, 21),
        err=np.full(21, 0.01),
        wave_medium="air",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        stellar_rv_applied_kms=12.0,
        resolution=resolution,
    )

    normalized, _ = normalize_segment_sidebands(segment)

    assert normalized.wave_medium == "air"
    assert normalized.observer_frame == "barycentric"
    assert normalized.stellar_rest_status == "corrected"
    assert normalized.stellar_rv_applied_kms == 12.0
    assert normalized.resolution is resolution
