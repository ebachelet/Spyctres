import numpy as np
import pytest
from astropy.io import fits

from Spyctres.phoenix import (
    PHOENIX_MAX_TEFF_K,
    PhoenixLibrary,
    validate_phoenix_teff,
)


def make_library(tmp_path):
    fits.writeto(
        tmp_path / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
        np.linspace(5000.0, 5010.0, 11),
        overwrite=True,
    )
    return PhoenixLibrary(tmp_path, verbose=False)


def test_cache_rejects_changed_wavelength_grid(tmp_path):
    library = make_library(tmp_path)
    library.wave = np.array([5000.0, 5001.0, 5002.0])
    library._grid = (
        np.array([5000.0, 5100.0]),
        np.array([-0.5, 0.0]),
        np.array([4.0, 4.5]),
    )
    library._observed_wave_medium = "vacuum"
    library._flux_grid = np.ones((2, 2, 2, 3))

    cache_path = tmp_path / "cache.npz"
    library.save_cache(cache_path, observed_wave_medium="vacuum")

    other = make_library(tmp_path)
    with pytest.raises(ValueError, match="wavelength grid"):
        other.load_cache(
            cache_path,
            expected_wave=np.array([5000.0, 5001.0, 5003.0]),
        )


def test_phoenix_temperature_guard_accepts_boundary_and_rejects_hotter_models():
    assert validate_phoenix_teff(PHOENIX_MAX_TEFF_K) == PHOENIX_MAX_TEFF_K

    with pytest.raises(ValueError, match="appropriate physics"):
        validate_phoenix_teff(PHOENIX_MAX_TEFF_K + 0.1)


def test_cache_rejects_different_phoenix_source_root(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    library = make_library(first_root)
    library.wave = np.array([5000.0, 5001.0])
    library._grid = (
        np.array([5000.0]),
        np.array([0.0]),
        np.array([4.0]),
    )
    library._flux_grid = np.ones((1, 1, 1, 2))
    cache_path = tmp_path / "cache.npz"
    library.save_cache(cache_path, observed_wave_medium="vacuum")

    other = make_library(second_root)
    with pytest.raises(ValueError, match="source root"):
        other.load_cache(cache_path)


def test_interpolator_match_uses_exact_wave_and_parameter_axes(tmp_path):
    library = make_library(tmp_path)
    library.wave = np.array([5000.0, 5001.0])
    library._observed_wave_medium = "vacuum"
    library._grid = (
        np.array([5000.0]),
        np.array([0.0]),
        np.array([4.0]),
    )

    assert library.interpolator_matches(
        [5000.0, 5001.0], [5000.0], [0.0], [4.0], "vacuum"
    )
    assert not library.interpolator_matches(
        [5000.0, 5001.1], [5000.0], [0.0], [4.0]
    )
    assert not library.interpolator_matches(
        [5000.0, 5001.0], [5000.0], [0.0], [4.0], "air"
    )


def test_interpolator_uses_scaled_axes_but_preserves_physical_grid(tmp_path):
    library = make_library(tmp_path)
    library.wave = np.array([5000.0, 5001.0])
    teff_grid = np.array([5000.0, 7000.0])
    feh_grid = np.array([-1.0, 0.5])
    logg_grid = np.array([3.0, 5.0])

    flux_grid = np.empty((2, 2, 2, 2), dtype=float)
    for it, teff in enumerate(teff_grid):
        for iz, feh in enumerate(feh_grid):
            for ig, logg in enumerate(logg_grid):
                flux_grid[it, iz, ig, 0] = 0.001 * teff + 10.0 * feh + logg
                flux_grid[it, iz, ig, 1] = 0.002 * teff - feh + 2.0 * logg

    library._set_interpolator_from_flux_grid(
        (teff_grid, feh_grid, logg_grid),
        flux_grid,
    )

    assert np.array_equal(library._grid[0], teff_grid)
    assert np.array_equal(library._interp_grid[0], [0.0, 1.0])
    assert np.array_equal(library._interp_grid[1], [0.0, 1.0])
    assert np.array_equal(library._interp_grid[2], [0.0, 1.0])

    evaluated = library.evaluate(6000.0, -0.25, 4.0)

    assert evaluated[0] == pytest.approx(0.001 * 6000.0 + 10.0 * -0.25 + 4.0)
    assert evaluated[1] == pytest.approx(0.002 * 6000.0 - -0.25 + 2.0 * 4.0)

    for it, teff in enumerate(teff_grid):
        for iz, feh in enumerate(feh_grid):
            for ig, logg in enumerate(logg_grid):
                node_eval = library.evaluate(teff, feh, logg)
                assert np.array_equal(node_eval, flux_grid[it, iz, ig])


def test_build_interpolator_emits_structured_flux_cube_and_cache_progress(
    tmp_path, monkeypatch
):
    library = make_library(tmp_path)
    template_path = tmp_path / "template.fits"
    template_path.touch()

    monkeypatch.setattr(
        library,
        "template_path",
        lambda teff, logg, feh: str(template_path),
    )
    monkeypatch.setattr(
        library,
        "_resample_template_fast",
        lambda path, wave_clip, mask, observed_wave: np.ones_like(
            np.asarray(observed_wave, dtype=float)
        ),
    )

    events = []
    cache_path = tmp_path / "cache.npz"
    library.build_interpolator(
        observed_wave=np.linspace(5000.0, 5010.0, 5),
        teff_grid=[5000.0],
        feh_grid=[0.0],
        logg_grid=[4.0],
        observed_wave_medium="vacuum",
        cache_path=cache_path,
        progress_callback=events.append,
    )

    assert all(isinstance(event, dict) for event in events)
    stages = [event["stage"] for event in events]
    assert "build_support_grid" in stages
    assert "build_flux_cube" in stages
    assert "save_cache" in stages
    assert stages[-1] == "done"

    flux_events = [event for event in events if event["stage"] == "build_flux_cube"]
    assert any(event.get("current") == 0 for event in flux_events)
    assert any(
        event.get("current") == event.get("total") == 1 for event in flux_events
    )
    assert all(event.get("unit") == "templates" for event in flux_events)
    assert any("flux_grid_nbytes" in event for event in flux_events)

    save_events = [event for event in events if event["stage"] == "save_cache"]
    assert any("This can take a while" in event["message"] for event in save_events)
    assert any("Finished saving" in event["message"] for event in save_events)
    assert all(event.get("cache_path") == str(cache_path) for event in save_events)
    assert cache_path.exists()
