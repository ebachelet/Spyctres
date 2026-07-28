import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from Spyctres.io import (
    ResolutionDescriptor,
    read_sdss_spec,
    read_spectrum,
    read_uves_pop_ascii,
)


def _write_sdss_spec(path, loglam, flux, ivar=None, and_mask=None):
    primary = fits.PrimaryHDU()
    primary.header["PLATEID"] = 1660
    primary.header["MJD"] = 53230
    primary.header["FIBERID"] = 23
    primary.header["OBJECT"] = "synthetic-star"
    primary.header["CLASS"] = "STAR"
    primary.header["Z"] = 0.0123

    columns = [
        fits.Column(name="loglam", format="D", array=np.asarray(loglam, dtype=float)),
        fits.Column(name="flux", format="D", array=np.asarray(flux, dtype=float)),
    ]
    if ivar is not None:
        columns.append(
            fits.Column(name="ivar", format="D", array=np.asarray(ivar, dtype=float))
        )
    if and_mask is not None:
        columns.append(
            fits.Column(name="and_mask", format="J", array=np.asarray(and_mask, dtype=np.int32))
        )
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)


def _write_sdss_spec_with_specobj(path):
    primary = fits.PrimaryHDU()
    primary.header["PLATEID"] = 3298
    primary.header["MJD"] = 54924
    primary.header["FIBERID"] = 159
    coadd = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="loglam", format="D", array=np.log10([4000.0, 4001.0])),
            fits.Column(name="flux", format="D", array=[1.0, 1.1]),
            fits.Column(name="ivar", format="D", array=[1.0, 1.0]),
        ],
        name="COADD",
    )
    specobj = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CLASS", format="16A", array=["STAR"]),
            fits.Column(name="SUBCLASS", format="16A", array=["G2"]),
            fits.Column(name="Z", format="E", array=[-0.0004]),
            fits.Column(name="OBJID", format="5J", array=[np.array([1, 2, 3, 4, 5])]),
        ],
        name="SPECOBJ",
    )
    fits.HDUList([primary, coadd, specobj]).writeto(path)


def test_uves_pop_ascii_auto_detects_nm_and_does_not_assume_error(tmp_path):
    path = tmp_path / "uves_nm.dat"
    path.write_text(
        "# wavelength[nm] flux optional_extra\n"
        "500.2 1.2 0.01\n"
        "500.0 1.0 0.02\n",
        encoding="utf-8",
    )

    segment = read_uves_pop_ascii(path)

    assert np.allclose(segment.wave, [5000.0, 5002.0])
    assert np.allclose(segment.flux, [1.0, 1.2])
    assert segment.err is None
    assert np.array_equal(segment.mask, [True, True])
    assert segment.wave_medium == "unknown"
    assert segment.observer_frame == "unknown"
    assert segment.stellar_rest_status == "unknown"
    assert isinstance(segment.resolution, ResolutionDescriptor)
    assert segment.resolution.quantity == "R"
    assert segment.resolution.value == pytest.approx(80000.0)
    assert segment.meta["uves_pop_reader"]["third_column_assumed_error"] is False
    assert segment.meta["wave_unit_input"] == "nm"
    assert segment.meta["fit_readiness_role"] == "quicklook_only_without_formal_errors"
    assert segment.meta["archive_mask_summary"]["masks_available"] is True


def test_uves_pop_ascii_angstrom_and_requested_error_column(tmp_path):
    path = tmp_path / "uves_A.dat"
    path.write_text(
        "5001.0 0.9 0.03\n"
        "5000.0 1.0 0.02\n",
        encoding="utf-8",
    )

    segment = read_uves_pop_ascii(path, wave_unit="angstrom", err_column=2)

    assert np.allclose(segment.wave, [5000.0, 5001.0])
    assert np.allclose(segment.flux, [1.0, 0.9])
    assert np.allclose(segment.err, [0.02, 0.03])
    assert segment.meta["uves_pop_reader"]["err_column"] == 2
    assert segment.meta["wave_unit_input"] == "angstrom"
    assert (
        segment.meta["fit_readiness_role"]
        == "fit_candidate_if_archive_metadata_verified"
    )


def test_sdss_spec_reads_loglam_flux_ivar_and_and_mask(tmp_path):
    path = tmp_path / "spec-1660-53230-0023.fits"
    wave = np.array([4000.0, 4001.0, 4002.0, 4003.0])
    loglam = np.log10(wave)
    flux = np.array([1.0, 2.0, np.nan, 4.0])
    ivar = np.array([4.0, 0.0, 9.0, 16.0])
    and_mask = np.array([0, 0, 0, 1])
    _write_sdss_spec(path, loglam, flux, ivar=ivar, and_mask=and_mask)

    segment = read_sdss_spec(path)

    assert np.allclose(segment.wave, wave)
    assert np.allclose(segment.flux[:2], [1.0, 2.0])
    assert np.allclose(segment.err[[0, 2, 3]], [0.5, 1.0 / 3.0, 0.25])
    assert np.isnan(segment.err[1])
    assert np.array_equal(segment.mask, [True, False, False, False])
    assert segment.wave_medium == "vacuum"
    assert segment.observer_frame == "heliocentric"
    assert segment.stellar_rest_status == "observed"
    assert segment.resolution is None
    assert segment.meta["plate"] == 1660
    assert segment.meta["mjd"] == 53230
    assert segment.meta["fiber"] == 23
    assert segment.meta["object"] == "synthetic-star"
    assert segment.meta["class"] == "STAR"
    assert segment.meta["redshift"] == pytest.approx(0.0123)
    assert segment.meta["sdss_mask_policy"]["ivar_positive_required"] is True
    assert segment.meta["sdss_mask_policy"]["and_mask_zero_required"] is True
    assert segment.meta["sdss_mask_policy"]["name"] == "and_mask_conservative"


def test_sdss_spec_can_ignore_and_mask_but_keeps_ivar_polarity(tmp_path):
    path = tmp_path / "spec-3298-54924-0159.fits"
    wave = np.array([5000.0, 5001.0, 5002.0])
    _write_sdss_spec(
        path,
        np.log10(wave),
        [1.0, 1.0, 1.0],
        ivar=[1.0, 0.0, 4.0],
        and_mask=[0, 1, 1],
    )

    segment = read_sdss_spec(path, use_and_mask=False)

    assert np.array_equal(segment.mask, [True, False, True])
    assert np.allclose(segment.err[[0, 2]], [1.0, 0.5])
    assert np.isnan(segment.err[1])
    assert segment.meta["sdss_mask_policy"]["and_mask_zero_required"] is False
    assert segment.meta["sdss_mask_policy"]["name"] == "ivar_only"


def test_sdss_spec_preserves_specobj_metadata_when_not_in_primary_header(tmp_path):
    path = tmp_path / "spec-with-specobj.fits"
    _write_sdss_spec_with_specobj(path)

    segment = read_sdss_spec(path)

    assert segment.meta["plate"] == 3298
    assert segment.meta["mjd"] == 54924
    assert segment.meta["fiber"] == 159
    assert segment.meta["class"] == "STAR"
    assert segment.meta["subclass"] == "G2"
    assert segment.meta["redshift"] == pytest.approx(-0.0004)
    assert segment.meta["object"] == [1, 2, 3, 4, 5]


def test_read_spectrum_aliases_for_uves_pop_and_sdss(tmp_path):
    uves_path = tmp_path / "hd22049.dat"
    uves_path.write_text("500.0 1.0\n500.1 1.1\n", encoding="utf-8")
    sdss_path = tmp_path / "spec-1-2-3.fits"
    _write_sdss_spec(sdss_path, np.log10([4000.0, 4001.0]), [1.0, 1.1], ivar=[1.0, 1.0])

    for alias in ("uves_pop_ascii", "uves_pop", "uves-pop", "uvespop"):
        segment = read_spectrum(uves_path, reader=alias, warn_unknown=False)
        assert np.allclose(segment.wave, [5000.0, 5001.0])
        assert segment.meta["ingestion"][-1]["source"] == "reader:{0}".format(alias)

    for alias in ("sdss_spec", "sdss", "segue"):
        segment = read_spectrum(sdss_path, reader=alias)
        assert np.allclose(segment.wave, [4000.0, 4001.0])
        assert segment.meta["ingestion"][-1]["source"] == "reader:{0}".format(alias)


def test_unknown_instrument_error_lists_registered_readers(tmp_path):
    path = tmp_path / "dummy.dat"
    path.write_text("1 1\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        read_spectrum(path, reader="not_a_reader", warn_unknown=False)

    message = str(caught.value)
    assert "sdss_spec" in message
    assert "uves_pop_ascii" in message
    assert "xshooter_merge1d" in message


def test_io_smoketest_no_show_and_plot_dir(tmp_path):
    spectrum_path = tmp_path / "hd115617.dat"
    spectrum_path.write_text("500.0 1.0\n500.1 1.1\n", encoding="utf-8")
    plot_dir = tmp_path / "plots"
    script = Path(__file__).resolve().parents[1] / "scripts" / "io_smoketest.py"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(repo_root)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--instrument",
            "uves_pop",
            str(spectrum_path),
            "--no-show",
            "--plot-dir",
            str(plot_dir),
        ],
        cwd=str(repo_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote quick-look plot" in completed.stdout
    assert len(list(plot_dir.glob("*.png"))) == 1
