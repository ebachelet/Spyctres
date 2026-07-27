import pytest

from Spyctres._spectrum_helpers import spectrum_segments
from Spyctres._workflow_helpers import (
    archive_mask_count,
    archive_masks_by_segment,
    fit_kwargs_with_archive_policy,
    resolution_assumption_for_audit,
    resolution_override_summary,
    unique_archive_masks,
)
from Spyctres.io import SpectrumCollection, SpectrumSegment
from Spyctres.preprocessing import exclusion_mask


def _segment(name="seg", masks=()):
    meta = {
        "archive_mask_catalog": [
            {
                "id": mask_id,
                "region_A": [5000.0, 5010.0],
                "reason": "synthetic",
            }
            for mask_id in masks
        ],
    }
    return SpectrumSegment([5000.0, 5010.0], [1.0, 1.0], name=name, meta=meta)


def test_spectrum_segments_preserves_collection_and_optional_tuple_collection():
    first = _segment("a")
    second = _segment("b")
    collection = SpectrumCollection([first, second])

    assert spectrum_segments(first) == [first]
    assert spectrum_segments(collection) == [first, second]
    assert spectrum_segments((first, second), tuple_is_collection=True) == [first, second]


def test_spectrum_segments_can_coerce_array_tuple_when_requested():
    wave = [5000.0, 5010.0]
    flux = [1.0, 0.9]

    segments = spectrum_segments((wave, flux), tuple_is_collection=False, coerce=True)

    assert len(segments) == 1
    assert segments[0].wave.tolist() == pytest.approx(wave)
    assert segments[0].flux.tolist() == pytest.approx(flux)


def test_resolution_override_helpers_are_consistent_and_validate_width():
    assert resolution_override_summary(None) is None
    assert resolution_assumption_for_audit(None) is None

    assert resolution_override_summary(2000.0) == {
        "resolution_source": "user_override",
        "assumed_resolution_R": 2000.0,
        "assumption_warning": "approximate quicklook resolution",
    }
    assert resolution_assumption_for_audit(2000.0) == {
        "quantity": "R",
        "value": 2000.0,
        "source": "user_override",
        "assumption_warning": "approximate quicklook resolution",
    }
    with pytest.raises(ValueError, match="finite and > 0"):
        resolution_override_summary(0.0)


def test_archive_masks_by_segment_and_count_use_segment_indices():
    first = _segment("a", masks=("gap_a",))
    second = _segment("b")
    third = _segment("c", masks=("gap_c1", "gap_c2"))

    masks = archive_masks_by_segment(SpectrumCollection([first, second, third]))

    assert sorted(masks) == [0, 2]
    assert [mask.name for mask in masks[0]] == ["archive:gap_a"]
    assert archive_mask_count(masks) == 3


def test_fit_kwargs_with_archive_policy_merges_global_and_indexed_masks():
    archive = archive_masks_by_segment([_segment("a", masks=("gap_a",))])
    global_mask = exclusion_mask(
        "global",
        lambda wave: wave > 0,
        metadata={"reason": "synthetic"},
    )
    indexed_mask = exclusion_mask(
        "indexed",
        lambda wave: wave > 0,
        metadata={"reason": "synthetic"},
    )

    global_merged = fit_kwargs_with_archive_policy(
        {"exclude_masks": [global_mask]},
        archive,
        "apply",
    )
    indexed_merged = fit_kwargs_with_archive_policy(
        {"exclude_masks": {0: [indexed_mask]}},
        archive,
        "apply",
    )

    assert [mask.name for mask in global_merged["exclude_masks"][0]] == [
        "archive:gap_a",
        "global",
    ]
    assert [mask.name for mask in indexed_merged["exclude_masks"][0]] == [
        "archive:gap_a",
        "indexed",
    ]
    assert fit_kwargs_with_archive_policy({"x": 1}, archive, "warn") == {"x": 1}


def test_unique_archive_masks_deduplicates_by_mask_name():
    first = _segment("a", masks=("same_gap",))
    second = _segment("b", masks=("same_gap", "other_gap"))

    masks = unique_archive_masks([first, second])

    assert [mask.name for mask in masks] == [
        "archive:same_gap",
        "archive:other_gap",
    ]
    assert unique_archive_masks([first], policy="warn") == ()
