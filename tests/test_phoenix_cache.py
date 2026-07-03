import numpy as np
import pytest
from astropy.io import fits

from Spyctres.phoenix import PhoenixLibrary


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
    library._flux_grid = np.ones((2, 2, 2, 3))

    cache_path = tmp_path / "cache.npz"
    library.save_cache(cache_path, observed_wave_medium="vacuum")

    other = make_library(tmp_path)
    with pytest.raises(ValueError, match="wavelength grid"):
        other.load_cache(
            cache_path,
            expected_wave=np.array([5000.0, 5001.0, 5003.0]),
        )
