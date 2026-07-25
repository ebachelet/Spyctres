import numpy as np
import pytest
import gzip
from astropy.io import fits

from Spyctres.io import (
    read_sdss_spec,
    read_spectrum,
    read_uves_pop_ascii,
    sdss_wdisp_to_resolution_descriptor,
)


def _write_sdss_spec(path, wave_A, flux, ivar, and_mask, wdisp=None, or_mask=None):
    columns = [
        fits.Column(name="loglam", format="D", array=np.log10(wave_A)),
        fits.Column(name="flux", format="D", array=np.asarray(flux, dtype=float)),
        fits.Column(name="ivar", format="D", array=np.asarray(ivar, dtype=float)),
        fits.Column(name="and_mask", format="J", array=np.asarray(and_mask, dtype=np.int32)),
    ]
    if wdisp is not None:
        columns.append(
            fits.Column(name="wdisp", format="D", array=np.asarray(wdisp, dtype=float))
        )
    if or_mask is not None:
        columns.append(
            fits.Column(name="or_mask", format="J", array=np.asarray(or_mask, dtype=np.int32))
        )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(
        path
    )


def test_uves_pop_two_column_nm_converts_to_angstrom(tmp_path):
    path = tmp_path / "uves_nm.dat"
    path.write_text("500.0 1.0\n500.2 1.2\n", encoding="utf-8")

    segment = read_uves_pop_ascii(path)

    assert np.allclose(segment.wave, [5000.0, 5002.0])
    assert np.allclose(segment.flux, [1.0, 1.2])
    assert segment.meta["wave_unit_input"] == "nm"


def test_uves_pop_gzip_two_column_nm_converts_to_angstrom(tmp_path):
    path = tmp_path / "uves_nm.dat.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("500.0 1.0\n500.2 1.2\n")

    segment = read_uves_pop_ascii(path)

    assert np.allclose(segment.wave, [5000.0, 5002.0])
    assert np.allclose(segment.flux, [1.0, 1.2])
    assert segment.meta["wave_unit_input"] == "nm"


def test_uves_pop_two_column_angstrom_is_not_converted(tmp_path):
    path = tmp_path / "uves_angstrom.dat"
    path.write_text("4999.0 0.9\n5001.0 1.1\n", encoding="utf-8")

    segment = read_uves_pop_ascii(path)

    assert np.allclose(segment.wave, [4999.0, 5001.0])
    assert np.allclose(segment.flux, [0.9, 1.1])
    assert segment.meta["wave_unit_input"] == "angstrom"


def test_uves_pop_nonfinite_flux_is_masked_false(tmp_path):
    path = tmp_path / "uves_nonfinite.dat"
    path.write_text("5000.0 1.0\n5001.0 nan\n5002.0 1.2\n", encoding="utf-8")

    segment = read_uves_pop_ascii(path)

    assert np.array_equal(segment.mask, [True, False, True])


def test_uves_pop_third_column_is_not_error_unless_requested(tmp_path):
    path = tmp_path / "uves_three_columns.dat"
    path.write_text("5000.0 1.0 0.01\n5001.0 1.1 0.02\n", encoding="utf-8")

    segment = read_uves_pop_ascii(path, err_column=None)

    assert segment.err is None
    assert segment.meta["uves_pop_reader"]["third_column_assumed_error"] is False


def test_uves_pop_requested_error_column_controls_error_mask(tmp_path):
    path = tmp_path / "uves_errors.dat"
    path.write_text(
        "5000.0 1.0 0.10\n"
        "5001.0 1.1 0.00\n"
        "5002.0 1.2 nan\n",
        encoding="utf-8",
    )

    segment = read_uves_pop_ascii(path, err_column=2)

    assert np.allclose(segment.err[0], 0.10)
    assert segment.err[1] == pytest.approx(0.0)
    assert np.isnan(segment.err[2])
    assert np.array_equal(segment.mask, [True, False, False])


def test_uves_pop_metadata_records_unknown_frames_and_nominal_resolution(tmp_path):
    path = tmp_path / "uves_metadata.dat"
    path.write_text("5000.0 1.0\n5001.0 1.1\n", encoding="utf-8")

    segment = read_uves_pop_ascii(path)

    assert segment.wave_medium == "unknown"
    assert segment.observer_frame == "heliocentric"
    assert segment.stellar_rest_status == "unknown"
    assert segment.meta["wave_medium"] == "unknown"
    assert segment.meta["observer_frame"] == "heliocentric"
    assert segment.meta["stellar_rest_status"] == "unknown"
    assert segment.resolution.quantity == "R"
    assert segment.resolution.value == pytest.approx(80000.0)
    assert "Nominal UVES-POP" in segment.meta["resolution_note"]
    assert segment.meta["fit_readiness_role"] == "quicklook_only_without_formal_errors"
    assert segment.meta["archive_mask_summary"]["masks_available"] is True
    assert segment.meta["archive_mask_summary"]["masks_applied"] == []
    assert "uves_pop_flag_flattening_5760_5840" in segment.meta["archive_mask_summary"]["masks_not_applied"]


def test_sdss_spec_loglam_ivar_and_default_and_mask_policy(tmp_path):
    path = tmp_path / "spec-0001-00000-0001.fits"
    wave = np.array([4000.0, 4001.0, 4002.0, 4003.0])
    _write_sdss_spec(
        path,
        wave_A=wave,
        flux=[1.0, 1.1, 1.2, 1.3],
        ivar=[4.0, 0.0, 9.0, 16.0],
        and_mask=[0, 0, 1, 0],
    )

    segment = read_sdss_spec(path)

    assert np.allclose(segment.wave, wave)
    assert segment.err[0] == pytest.approx(0.5)
    assert np.isnan(segment.err[1])
    assert segment.err[2] == pytest.approx(1.0 / 3.0)
    assert segment.err[3] == pytest.approx(0.25)
    assert np.array_equal(segment.mask, [True, False, False, True])
    assert segment.meta["sdss_mask_policy"]["ivar_positive_required"] is True
    assert segment.meta["sdss_mask_policy"]["and_mask_zero_required"] is True
    assert segment.meta["sdss_mask_policy"]["name"] == "and_mask_conservative"
    assert segment.meta["sdss_mask_policy"]["rejection_counts"]["n_used"] == 2


def test_sdss_spec_no_and_mask_option_keeps_and_masked_pixel(tmp_path):
    path = tmp_path / "spec-no-and-mask.fits"
    wave = np.array([5000.0, 5001.0, 5002.0])
    _write_sdss_spec(
        path,
        wave_A=wave,
        flux=[1.0, 1.1, 1.2],
        ivar=[1.0, 0.0, 4.0],
        and_mask=[0, 1, 1],
    )

    segment = read_sdss_spec(path, use_and_mask=False)

    assert np.array_equal(segment.mask, [True, False, True])
    assert segment.meta["sdss_mask_policy"]["and_mask_zero_required"] is False
    assert segment.meta["sdss_mask_policy"]["name"] == "ivar_only"


def test_sdss_spec_strict_policy_uses_or_mask_when_present(tmp_path):
    path = tmp_path / "spec-strict-mask.fits"
    wave = np.array([5100.0, 5101.0, 5102.0])
    _write_sdss_spec(
        path,
        wave_A=wave,
        flux=[1.0, 1.1, 1.2],
        ivar=[1.0, 1.0, 1.0],
        and_mask=[0, 0, 1],
        or_mask=[0, 1, 0],
    )

    segment = read_sdss_spec(path, sdss_mask_policy="stellar_strict")

    assert np.array_equal(segment.mask, [True, False, False])
    policy = segment.meta["sdss_mask_policy"]
    assert policy["name"] == "stellar_strict"
    assert policy["and_mask_zero_required"] is True
    assert policy["or_mask_zero_required"] is True
    assert policy["rejection_counts"]["n_rejected_and_mask_nonzero"] == 1
    assert policy["rejection_counts"]["n_rejected_or_mask_nonzero"] == 1


def test_sdss_spec_metadata_records_wavelength_and_frame_semantics(tmp_path):
    path = tmp_path / "spec-metadata.fits"
    _write_sdss_spec(
        path,
        wave_A=np.array([6000.0, 6001.0]),
        flux=[1.0, 1.1],
        ivar=[1.0, 1.0],
        and_mask=[0, 0],
    )

    segment = read_sdss_spec(path)

    assert segment.wave_medium == "vacuum"
    assert segment.observer_frame == "heliocentric"
    assert segment.stellar_rest_status == "observed"
    assert segment.meta["wave_medium"] == "vacuum"
    assert segment.meta["observer_frame"] == "heliocentric"
    assert segment.meta["stellar_rest_status"] == "observed"
    assert segment.resolution is None


def test_sdss_wdisp_is_preserved_but_not_attached_by_default(tmp_path):
    path = tmp_path / "spec-wdisp-default.fits"
    _write_sdss_spec(
        path,
        wave_A=np.array([4000.0, 4000.9213, 4001.8428]),
        flux=[1.0, 1.1, 1.2],
        ivar=[1.0, 1.0, 1.0],
        and_mask=[0, 0, 0],
        wdisp=[1.0, 1.1, 1.2],
    )

    segment = read_sdss_spec(path)

    assert segment.resolution is None
    assert segment.meta["sdss_lsf"]["present"] is True
    assert segment.meta["sdss_lsf"]["lsf_source"] == "sdss_wdisp_not_applied"
    assert segment.meta["sdss_lsf"]["attach_wdisp_resolution"] is False
    assert segment.meta["sdss_lsf"]["reader_default_resolution"] is None
    assert segment.meta["sdss_lsf"]["active_lsf_convolution"] is False


def test_sdss_wdisp_opt_in_resolution_descriptor(tmp_path):
    path = tmp_path / "spec-wdisp-attached.fits"
    wave = 10 ** (np.log10(4000.0) + np.arange(4) * 1.0e-4)
    _write_sdss_spec(
        path,
        wave_A=wave,
        flux=[1.0, 1.1, 1.2, 1.3],
        ivar=[1.0, 1.0, 1.0, 1.0],
        and_mask=[0, 0, 0, 0],
        wdisp=[1.0, 1.0, 1.0, 1.0],
    )

    segment = read_sdss_spec(path, attach_wdisp_resolution=True)

    assert segment.resolution.quantity == "sigma_kms"
    assert segment.resolution.mode == "tabulated"
    assert np.allclose(segment.resolution.wave_A, wave)
    expected_sigma = 299792.458 * np.log(10.0) * 1.0e-4
    assert np.allclose(segment.resolution.values, expected_sigma)
    assert segment.meta["sdss_lsf"]["attach_wdisp_resolution"] is True
    assert (
        segment.meta["sdss_lsf"]["lsf_source"]
        == "sdss_wdisp_attached_as_tabulated_sigma_kms"
    )
    assert segment.meta["sdss_lsf"]["active_lsf_convolution"] is False


def test_sdss_wdisp_descriptor_helper_rejects_bad_shapes():
    with pytest.raises(ValueError, match="matching shape"):
        sdss_wdisp_to_resolution_descriptor([4000.0, 4001.0], [1.0])


def test_read_spectrum_external_reader_aliases(tmp_path):
    uves_path = tmp_path / "hd22049.dat"
    uves_path.write_text("500.0 1.0\n500.1 1.1\n", encoding="utf-8")
    sdss_path = tmp_path / "spec-alias.fits"
    _write_sdss_spec(
        sdss_path,
        wave_A=np.array([4100.0, 4101.0]),
        flux=[1.0, 1.1],
        ivar=[1.0, 1.0],
        and_mask=[0, 0],
    )

    for alias in ("uves_pop", "uves-pop", "uvespop"):
        segment = read_spectrum(uves_path, instrument=alias, warn_unknown=False)
        assert np.allclose(segment.wave, [5000.0, 5001.0])
        assert segment.meta["ingestion"][-1]["source"] == "reader:{0}".format(alias)

    for alias in ("sdss", "sdss_spec", "segue"):
        segment = read_spectrum(sdss_path, instrument=alias, warn_unknown=False)
        assert np.allclose(segment.wave, [4100.0, 4101.0])
        assert segment.meta["ingestion"][-1]["source"] == "reader:{0}".format(alias)
