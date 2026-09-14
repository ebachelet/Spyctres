from types import SimpleNamespace
import importlib

import numpy as np
import pytest


def test_legacy_module_import_and_core_entry_points():
    from Spyctres import Spyctres

    expected = (
        "velocity_correction",
        "Barycentric_velocity",
        "get_element_lines",
        "star_spectrum_new",
        "load_telluric_lines",
    )

    for name in expected:
        assert callable(getattr(Spyctres, name))


def test_legacy_element_lines_preserve_vald_air_to_vacuum_values(monkeypatch):
    from Spyctres import Spyctres

    table = {
        "wavel": np.array([2000.0, 2000.1, 5000.0]),
        "INT": np.array([100.0, 100.0, 100.0]),
        "Element": np.array(["FE", "FE", "FE"]),
        "Spectrum": np.array(["I", "I", "I"]),
    }
    monkeypatch.setattr(
        Spyctres.fits,
        "open",
        lambda _path: [None, SimpleNamespace(data=table)],
    )

    lines = Spyctres.get_element_lines(
        wavelength_range=[1900.0, 6000.0],
        require_elements=[],
        intensity_threshold=0.0,
    )
    wavelengths = lines[:, 0].astype(float)

    assert wavelengths[0] == 2000.0
    assert wavelengths[1] == pytest.approx(2000.748103141021, abs=5.0e-11)
    assert wavelengths[2] == pytest.approx(5001.394848638070, abs=5.0e-11)


def test_legacy_bin_spectrum_raises_instead_of_breakpoint(monkeypatch):
    from Spyctres import Spyctres

    def fail_breakpoint():
        raise AssertionError("breakpoint() should not be called")

    monkeypatch.setattr("builtins.breakpoint", fail_breakpoint)
    malformed = np.array([[0.0, 1.0], [2.0, 1.0]])

    with pytest.raises(RuntimeError, match="Legacy bin_spectrum failed"):
        Spyctres.bin_spectrum(malformed, np.array([0.5, 1.5]))


def test_legacy_old_bin_spectrum_raises_instead_of_breakpoint(monkeypatch):
    legacy_old = importlib.import_module("Spyctres.Spyctres_old")

    def fail_breakpoint():
        raise AssertionError("breakpoint() should not be called")

    monkeypatch.setattr("builtins.breakpoint", fail_breakpoint)
    malformed = np.array([[0.0, 1.0], [2.0, 1.0]])

    with pytest.raises(RuntimeError, match="Legacy bin_spectrum failed"):
        legacy_old.bin_spectrum(malformed, np.array([0.5, 1.5]))
