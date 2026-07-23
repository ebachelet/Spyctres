import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from scripts import external_spectra_validation


def _write_uves(path, *, nm=True, with_error=True):
    wave0 = 500.0 if nm else 5000.0
    step = 0.02 if nm else 0.2
    lines = ["# synthetic UVES-POP-like spectrum\n"]
    for index in range(20):
        wave = wave0 + index * step
        flux = 1.0 + 0.02 * np.sin(index / 3.0)
        if with_error:
            lines.append("{0:.6f} {1:.8f} 0.02\n".format(wave, flux))
        else:
            lines.append("{0:.6f} {1:.8f}\n".format(wave, flux))
    path.write_text("".join(lines), encoding="utf-8")


def _write_sdss_spec(path, *, bad_masks=False, wdisp=False):
    wave = np.linspace(4000.0, 4019.0, 20)
    flux = 1.0 + 0.01 * np.cos(np.arange(wave.size) / 3.0)
    ivar = np.full(wave.size, 25.0)
    and_mask = np.zeros(wave.size, dtype=np.int32)
    if bad_masks:
        ivar[1:5] = 0.0
        and_mask[8:12] = 1
    columns = [
        fits.Column(name="loglam", format="D", array=np.log10(wave)),
        fits.Column(name="flux", format="D", array=flux),
        fits.Column(name="ivar", format="D", array=ivar),
        fits.Column(name="and_mask", format="J", array=and_mask),
    ]
    if wdisp:
        columns.append(
            fits.Column(name="wdisp", format="D", array=np.full(wave.size, 1.3))
        )
    primary = fits.PrimaryHDU()
    primary.header["PLATEID"] = 1660
    primary.header["MJD"] = 53230
    primary.header["FIBERID"] = 23
    primary.header["CLASS"] = "STAR"
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)


def test_external_validation_manifest_writes_json_and_csv(tmp_path):
    uves = tmp_path / "hd22049.dat"
    sdss = tmp_path / "spec-1660-53230-0023.fits"
    _write_uves(uves, nm=True, with_error=True)
    _write_sdss_spec(sdss, bad_masks=True, wdisp=True)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "instrument",
                "target_id",
                "role",
                "label",
                "wave_unit",
                "err_column",
                "sdss_mask_policy",
                "assumed_resolution_R",
                "fit_wmin",
                "fit_wmax",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": uves.name,
                "instrument": "uves_pop",
                "target_id": "uves_clean",
                "role": "clean",
                "label": "UVES clean",
                "wave_unit": "nm",
                "err_column": "2",
                "fit_wmin": "5000",
                "fit_wmax": "5004",
            }
        )
        writer.writerow(
            {
                "path": sdss.name,
                "instrument": "sdss",
                "target_id": "sdss_dirty",
                "role": "dirty",
                "label": "SDSS dirty",
                "sdss_mask_policy": "and_mask_conservative",
                "assumed_resolution_R": "2000",
                "fit_wmin": "4000",
                "fit_wmax": "4020",
            }
        )
    output_json = tmp_path / "out" / "external.json"
    output_csv = tmp_path / "out" / "external.csv"

    status = external_spectra_validation.main(
        [
            "--manifest",
            str(manifest),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--no-plots",
        ]
    )

    assert status == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["operation"] == "external_spectra_validation"
    assert payload["target_count"] == 2
    assert payload["summary"]["by_instrument"] == {"uves_pop": 1, "sdss": 1}
    assert payload["summary"]["by_role"] == {"clean": 1, "dirty": 1}
    by_id = {row["target_id"]: row for row in payload["results"]}
    assert by_id["uves_clean"]["status"] == "ok"
    assert by_id["uves_clean"]["reader_kwargs"]["err_column"] == 2
    assert by_id["uves_clean"]["segments"][0]["wave_medium"] == "unknown"
    assert by_id["sdss_dirty"]["status"] == "ok"
    assert by_id["sdss_dirty"]["reader_kwargs"]["sdss_mask_policy"] == "and_mask_conservative"
    assert by_id["sdss_dirty"]["readiness"]["n_fit_candidate"] < 20
    assert by_id["sdss_dirty"]["segments"][0]["sdss_lsf"]["present"] is True
    assert by_id["sdss_dirty"]["segments"][0]["sdss_lsf"]["lsf_source"] == "sdss_wdisp_not_applied"
    assert output_csv.exists()
    assert "role_expectation_assessment" in output_csv.read_text(encoding="utf-8")


def test_external_validation_scan_root_and_resume(tmp_path, capsys):
    root = tmp_path / "spectra_test_set"
    sdss_dir = root / "SDSS"
    uves_dir = root / "UVES-POP"
    dirty_sdss_dir = root / "dirty" / "SDSS"
    sdss_dir.mkdir(parents=True)
    uves_dir.mkdir(parents=True)
    dirty_sdss_dir.mkdir(parents=True)
    _write_sdss_spec(sdss_dir / "spec-3298-54924-0159.fits")
    _write_sdss_spec(dirty_sdss_dir / "spec-3298-54924-0159.fits", bad_masks=True)
    _write_uves(uves_dir / "hd115617.dat", nm=False, with_error=False)
    output_json = tmp_path / "scan.json"

    first = external_spectra_validation.main(
        [
            "--scan-root",
            str(root),
            "--output-json",
            str(output_json),
            "--no-plots",
        ]
    )
    second = external_spectra_validation.main(
        [
            "--scan-root",
            str(root),
            "--output-json",
            str(output_json),
            "--resume",
            "--no-plots",
        ]
    )

    assert first == 0
    assert second == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["target_count"] == 3
    assert payload["summary"]["by_instrument"] == {"sdss": 2, "uves_pop": 1}
    assert len(payload["results"]) == 3
    target_ids = [row["target_id"] for row in payload["results"]]
    assert len(target_ids) == len(set(target_ids))
    assert any("dirty_SDSS_spec-3298-54924-0159" in item for item in target_ids)
    stdout = capsys.readouterr().out
    assert "Skipping completed" in stdout
