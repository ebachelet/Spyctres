import numpy as np

from Spyctres.fitting import build_effective_fit_mask, build_excluded_mask
from Spyctres.io import SpectrumCollection, SpectrumSegment


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


def test_collection_preserves_positive_segment_weights():
    first = SpectrumSegment([1.0, 2.0], [1.0, 1.0], name="first")
    second = SpectrumSegment([3.0, 4.0], [1.0, 1.0], name="second")

    collection = SpectrumCollection([first, second], weights=[1.0, 2.5])

    assert collection.names == ["first", "second"]
    assert np.array_equal(collection.weights, [1.0, 2.5])
