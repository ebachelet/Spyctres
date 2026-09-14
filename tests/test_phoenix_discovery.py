import os

import numpy as np
import pytest
from astropy.io import fits

from Spyctres.cli import main as cli_main
from Spyctres.phoenix import PhoenixLibrary, phoenix_relpath


MODEL_FAMILY = "PHOENIX-ACES-AGSS-COND-2011"
WAVE_FILENAME = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"


def _make_root(tmp_path):
    wave = np.linspace(5000.0, 5010.0, 11)
    fits.writeto(tmp_path / WAVE_FILENAME, wave, overwrite=True)
    return wave


def _write_template(template_root, teff=5000.0, logg=4.0, feh=0.0):
    relative = phoenix_relpath(teff, logg, feh)
    path = template_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    flux = np.linspace(0.9, 1.1, 11)
    fits.writeto(path, flux, overwrite=True)
    return path, flux


def test_standard_nested_layout_is_used_for_all_template_operations(tmp_path):
    wave = _make_root(tmp_path)
    nested = tmp_path / MODEL_FAMILY
    template_path, flux = _write_template(nested)

    library = PhoenixLibrary(tmp_path, verbose=False)

    assert library.template_dir == str(nested)
    assert library.template_path(5000.0, 4.0, 0.0) == str(template_path)
    assert library.has_template(5000.0, 4.0, 0.0)
    assert library.scan_available_points() == [(5000.0, 0.0, 4.0)]

    loaded_wave, loaded_flux = library.load_template(5000.0, 4.0, 0.0)
    assert np.array_equal(loaded_wave, wave)
    assert np.array_equal(loaded_flux, flux)


def test_legacy_flat_layout_remains_supported(tmp_path):
    _make_root(tmp_path)
    template_path, _flux = _write_template(tmp_path)

    library = PhoenixLibrary(tmp_path, verbose=False)

    assert library.template_dir == str(tmp_path)
    assert library.template_path(5000.0, 4.0, 0.0) == str(template_path)
    assert library.has_template(5000.0, 4.0, 0.0)
    assert library.scan_available_points() == [(5000.0, 0.0, 4.0)]


def test_alpha_enhanced_directories_do_not_form_the_ordinary_grid(tmp_path):
    _make_root(tmp_path)
    nested = tmp_path / MODEL_FAMILY
    ordinary_relative = phoenix_relpath(5000.0, 4.0, 0.0)
    alpha_path = nested / "Z-0.0.Alpha=+0.40" / os.path.basename(ordinary_relative)
    alpha_path.parent.mkdir(parents=True)
    fits.writeto(alpha_path, np.ones(11), overwrite=True)

    library = PhoenixLibrary(tmp_path, verbose=False)

    with pytest.raises(RuntimeError, match="alpha-enhanced"):
        library.scan_available_points()


def test_nested_layout_wins_without_merging_a_valid_flat_tree(tmp_path):
    _make_root(tmp_path)
    nested = tmp_path / MODEL_FAMILY
    nested_path, _flux = _write_template(nested, teff=5000.0)
    _write_template(tmp_path, teff=5100.0)

    library = PhoenixLibrary(tmp_path, verbose=False)

    assert library.template_dir == str(nested)
    assert library.template_path(5000.0, 4.0, 0.0) == str(nested_path)
    assert library.scan_available_points() == [(5000.0, 0.0, 4.0)]
    assert not library.has_template(5100.0, 4.0, 0.0)


def test_discovery_failure_reports_every_searched_location(tmp_path):
    _make_root(tmp_path)
    library = PhoenixLibrary(tmp_path, verbose=False)

    with pytest.raises(RuntimeError) as exc_info:
        library.scan_available_points()

    message = str(exc_info.value)
    assert "base_dir={0}".format(tmp_path) in message
    assert "wavelength_file={0}".format(tmp_path / WAVE_FILENAME) in message
    assert "nested_template_candidate={0}".format(tmp_path / MODEL_FAMILY) in message
    assert "flat_template_candidate={0}".format(tmp_path) in message


def test_doctor_discovers_standard_nested_layout(tmp_path, capsys, monkeypatch):
    from Spyctres import setup_check

    _make_root(tmp_path)
    nested = tmp_path / MODEL_FAMILY
    _write_template(nested)
    monkeypatch.setattr(setup_check, "check_python", lambda: True)

    return_code = cli_main(
        ["doctor", "--phoenix-dir", str(tmp_path), "--require-phoenix"]
    )

    output = capsys.readouterr().out
    assert return_code == 0
    assert "[OK] PHOENIX template discovery" in output
    assert "Spyctres setup check passed." in output


def test_doctor_skip_scan_does_not_require_templates(tmp_path, capsys, monkeypatch):
    from Spyctres import setup_check

    _make_root(tmp_path)
    monkeypatch.setattr(setup_check, "check_python", lambda: True)

    return_code = cli_main(
        [
            "doctor",
            "--phoenix-dir",
            str(tmp_path),
            "--require-phoenix",
            "--skip-phoenix-scan",
        ]
    )

    output = capsys.readouterr().out
    assert return_code == 0
    assert "[WARN] PHOENIX template scan skipped" in output
    assert "PHOENIX template discovery" not in output
