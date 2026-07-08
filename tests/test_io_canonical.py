import json

import numpy as np
import pytest
from astropy.io import fits

from Spyctres.fitting import _resolve_segment_fwhm_kms
from Spyctres.io import (
    COMMON_SPECTRUM_SCHEMA_VERSION,
    READERS,
    ResolutionDescriptor,
    SpectrumCollection,
    SpectrumSegment,
    canonicalize_segment,
    concatenate_segments,
    coerce_spectrum,
    pepsi_ssbvel_correction_kms,
    read_pepsi_nor,
    read_xsl_dr3,
    read_spectrum,
)


def _write_pepsi_nor(path, wave, **header_values):
    primary = fits.PrimaryHDU()
    for key, value in header_values.items():
        primary.header[key] = value
    columns = [
        fits.Column(name="Arg", format="D", array=np.asarray(wave, dtype=float)),
        fits.Column(name="Fun", format="D", array=np.ones(len(wave))),
        fits.Column(name="Var", format="D", array=np.full(len(wave), 0.04)),
    ]
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)


def test_array_ingestion_converts_units_uncertainty_and_mask_polarity():
    segment = coerce_spectrum(
        {
            "wave": [500.2, 500.1, 500.3],
            "flux": [2.0, 1.0, np.nan],
            "err": [0.04, 0.01, 0.09],
            "mask": [True, False, True],
            "wave_unit": "nm",
            "uncertainty_kind": "variance",
            "wave_medium": "vacuum",
            "observer_frame": "barycentric",
            "stellar_rest_status": "observed",
        },
        warn_unknown=False,
    )

    assert np.array_equal(segment.wave, [5001.0, 5002.0, 5003.0])
    assert np.allclose(segment.err, [0.1, 0.2, 0.3])
    # Canonical polarity is True == valid/use. The original False remains
    # excluded, and the non-finite flux is also excluded.
    assert np.array_equal(segment.mask, [False, True, False])
    assert segment.meta["mask_true_means"] == "use"
    assert segment.meta["error_kind"] == "sigma"
    assert segment.meta["spectrum_schema_version"] == COMMON_SPECTRUM_SCHEMA_VERSION


def test_observer_and_stellar_rest_frames_are_independent():
    segment = coerce_spectrum(
        {
            "wave": [5000.0, 5001.0],
            "flux": [1.0, 1.0],
            "wave_medium": "vacuum",
            "observer_frame": "barycentric",
            "stellar_rest_status": "corrected",
            "stellar_rv_applied_kms": 31.25,
        },
        warn_unknown=False,
    )

    assert segment.observer_frame == "barycentric"
    assert segment.stellar_rest_status == "corrected"
    assert segment.stellar_rv_applied_kms == pytest.approx(31.25)


def test_legacy_wave_frame_copy_updates_new_frame_axes():
    segment = SpectrumSegment([5000.0, 5001.0], [1.0, 1.0])

    barycentric = segment.copy(wave_frame="barycentric")
    rest = barycentric.copy(wave_frame="stellar_rest")

    assert barycentric.observer_frame == "barycentric"
    assert barycentric.stellar_rest_status == "observed"
    assert rest.observer_frame == "barycentric"
    assert rest.stellar_rest_status == "corrected"


def test_resolution_descriptor_supports_constant_and_tabulated_lsf():
    constant = ResolutionDescriptor(quantity="R", value=10_000.0, source="header")
    tabulated = ResolutionDescriptor(
        quantity="sigma_kms",
        mode="tabulated",
        wave_A=[3500.0, 5500.0, 10_000.0],
        values=[13.0, 11.0, 16.0],
        source="arm-dependent",
    )

    segment = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        resolution=constant,
    )
    copied = segment.copy()

    assert copied.resolution is constant
    assert tabulated.to_metadata()["values"] == [13.0, 11.0, 16.0]
    assert _resolve_segment_fwhm_kms(segment) == pytest.approx(29.9792458)


def test_current_fitter_rejects_recorded_variable_lsf_clearly():
    descriptor = ResolutionDescriptor(
        quantity="R",
        mode="tabulated",
        wave_A=[5000.0, 5001.0],
        values=[10_000.0, 11_000.0],
    )
    segment = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.0],
        resolution=descriptor,
    )

    with pytest.raises(ValueError, match="current fitter supports only a constant LSF"):
        _resolve_segment_fwhm_kms(segment)


def test_duplicate_wavelengths_raise_instead_of_merging_orders():
    segment = SpectrumSegment(
        [5000.0, 5001.0, 5001.0],
        [1.0, 1.0, 2.0],
    )

    with pytest.raises(ValueError, match="separate SpectrumCollection segments"):
        canonicalize_segment(segment, warn_unknown=False)


def test_collection_ingestion_preserves_segments_and_weights():
    first = SpectrumSegment([2.0, 1.0], [1.0, 1.0])
    second = SpectrumSegment([4.0, 3.0], [1.0, 1.0])
    collection = SpectrumCollection([first, second], weights=[1.0, 2.0])

    canonical = coerce_spectrum(collection, warn_unknown=False)

    assert isinstance(canonical, SpectrumCollection)
    assert np.array_equal(canonical.weights, [1.0, 2.0])
    assert np.array_equal(canonical[0].wave, [1.0, 2.0])
    assert np.array_equal(canonical[1].wave, [3.0, 4.0])


def test_concatenate_segments_rejects_overlapping_ranges():
    first = SpectrumSegment([5000.0, 5002.0], [1.0, 1.0])
    second = SpectrumSegment([5001.0, 5003.0], [1.0, 1.0])

    with pytest.raises(ValueError, match="SpectrumCollection"):
        concatenate_segments([first, second])


def test_unknown_semantics_are_allowed_but_warned():
    segment = SpectrumSegment([1.0, 2.0], [1.0, 1.0])

    with pytest.warns(UserWarning) as caught:
        canonicalize_segment(segment)

    messages = [str(item.message) for item in caught]
    assert any("wavelength medium is unknown" in message for message in messages)
    assert any("observer frame is unknown" in message for message in messages)
    assert any("stellar-rest status is unknown" in message for message in messages)


def test_read_spectrum_enforces_common_format_for_registered_reader(monkeypatch):
    def reader(_path):
        return SpectrumSegment(
            [5002.0, 5001.0],
            [1.0, 1.0],
            wave_medium="vacuum",
            observer_frame="topocentric",
            stellar_rest_status="observed",
        )

    monkeypatch.setitem(READERS, "test_reader", reader)
    segment = read_spectrum(
        "unused",
        instrument="test_reader",
        warn_unknown=False,
    )

    assert np.array_equal(segment.wave, [5001.0, 5002.0])
    assert segment.meta["ingestion"][-1]["source"] == "reader:test_reader"


def test_references_registry_is_valid_json_with_unique_ids():
    with open("references.json", "r", encoding="utf-8") as handle:
        registry = json.load(handle)

    ids = [entry["id"] for entry in registry["references"]]
    assert registry["schema_version"] == 1
    assert len(ids) == len(set(ids))
    assert all(entry["url"].startswith("https://") for entry in registry["references"])
    assert all(entry["affected_code"] for entry in registry["references"])


def test_pepsi_generic_profile_does_not_infer_semantics_from_suffix(tmp_path):
    path = tmp_path / "example.dxt.nor"
    _write_pepsi_nor(path, [5000.0, 5001.0], SSBVEL=-23000.0)

    segment = read_pepsi_nor(path)

    assert np.array_equal(segment.wave, [5000.0, 5001.0])
    assert segment.wave_medium == "unknown"
    assert segment.observer_frame == "unknown"
    assert segment.stellar_rest_status == "unknown"
    assert segment.meta["pepsi_product_profile"] == "generic"
    assert pepsi_ssbvel_correction_kms(segment) == pytest.approx(-23.0)


def test_pepsi_pets_profile_converts_microns_and_blocks_double_correction(tmp_path):
    path = tmp_path / "pets.nor"
    _write_pepsi_nor(
        path,
        [0.5000, 0.5001],
        RADVEL=(12.3, "stellar radial velocity"),
        OBSVEL=(-23.0, "barycentric radial velocity"),
        SSBVEL=(-23000.0, "SSB velocity of observer m/s"),
    )

    segment = read_pepsi_nor(path, product_profile="pets_stellar_rest")

    assert np.allclose(segment.wave, [5000.0, 5001.0])
    assert segment.wave_medium == "air"
    assert segment.observer_frame == "barycentric"
    assert segment.stellar_rest_status == "corrected"
    assert segment.meta["velocity_corrections"]["pets_radvel_applied"] is True
    assert segment.meta["velocity_corrections"]["pets_obsvel_applied"] is True
    with pytest.raises(ValueError, match="must not be applied"):
        pepsi_ssbvel_correction_kms(segment)


def test_pepsi_cds_profile_is_barycentric_but_medium_remains_unknown(tmp_path):
    path = tmp_path / "cds.nor"
    _write_pepsi_nor(path, [5000.0, 5001.0], SSBVEL=-23000.0)

    segment = read_pepsi_nor(path, product_profile="cds_aanda_671_a7")

    assert np.array_equal(segment.wave, [5000.0, 5001.0])
    assert segment.wave_medium == "unknown"
    assert segment.observer_frame == "barycentric"
    assert segment.stellar_rest_status == "observed"
    with pytest.raises(ValueError, match="must not be applied"):
        pepsi_ssbvel_correction_kms(segment)


def test_xsl_dr3_reader_splits_effective_lsf_regions(tmp_path):
    path = tmp_path / "xsl_spectrum_X0000_merged.fits"
    wave_nm = np.array([500.0, 589.0, 600.0, 993.0, 1000.0, 1200.0])
    columns = [
        fits.Column(name="WAVE", format="D", array=wave_nm),
        fits.Column(name="FLUX", format="D", array=np.ones(wave_nm.size)),
        fits.Column(name="FLUX_DR", format="D", array=np.full(wave_nm.size, 2.0)),
        fits.Column(name="ERR", format="D", array=np.full(wave_nm.size, 0.1)),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)

    collection = read_xsl_dr3(path, flux_variant="dereddened")

    assert isinstance(collection, SpectrumCollection)
    assert [segment.meta["arm"] for segment in collection] == ["UVB", "VIS", "NIR"]
    assert [segment.resolution.value for segment in collection] == [13.0, 11.0, 16.0]
    assert all(segment.resolution.quantity == "sigma_kms" for segment in collection)
    assert all(segment.stellar_rest_status == "corrected" for segment in collection)
    assert all(segment.wave_medium == "air" for segment in collection)
    assert np.array_equal(collection[0].wave, [5000.0, 5890.0])
    assert np.all(collection[0].flux == 2.0)
