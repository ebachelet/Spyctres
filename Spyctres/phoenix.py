# Spyctres/phoenix.py
import os
import hashlib
import numpy as np
from astropy.io import fits

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RegularGridInterpolator


def _as_float_array(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a 1D wavelength array.")
    return x


def _format_feh_dir(feh):
    # PHOENIX uses Z-0.0 (not Z+0.0). Keep + for other positive metallicities.
    feh = float(feh)
    s = "{:+.1f}".format(feh)
    if abs(feh) < 1e-12:
        s = "-0.0"
    return "Z{0}".format(s)


def _format_feh_file(feh):
    feh = float(feh)
    s = "{:+.1f}".format(feh)
    if abs(feh) < 1e-12:
        s = "-0.0"
    return s


def _format_logg(logg):
    return "{:.2f}".format(float(logg))


def phoenix_relpath(teff, logg, feh, model_tag="PHOENIX-ACES-AGSS-COND-2011-HiRes"):
    """
    Return relative path from the template root to a PHOENIX template.

    Layout supported here (your install):
      <base_dir>/
        WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
        Z-0.0/
          lte05000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
        Z-1.0/
          ...

    If you later want to support the alternative nested layout, we can add it,
    but for now we match your PHOENIXv2 tree.
    """
    teff_i = int(round(float(teff)))
    logg_s = _format_logg(logg)
    feh_s = _format_feh_file(feh)
    zdir = _format_feh_dir(feh)

    fname = "lte{0}-{1}{2}.{3}.fits".format(
        str(teff_i).zfill(5), logg_s, feh_s, model_tag
    )
    return os.path.join(zdir, fname)

class PhoenixLibrary(object):
    """
    Minimal PHOENIX template backend.

    Typical base_dir layout (HiResFITS):
      base_dir/
        WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
        PHOENIX-ACES-AGSS-COND-2011/
          Z-0.0/
            lte05000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
          Z-1.0/
            ...

    This class supports:
      - Loading a single template (and optionally resampling to a target wave grid)
      - Building a cached regular-grid interpolator on (Teff, [Fe/H], logg)
    """

    DEFAULT_TEFF_GRID = np.array([
        2300., 2400., 2500., 2600., 2700., 2800., 2900., 3000.,
        3100., 3200., 3300., 3400., 3500., 3600., 3700., 3800.,
        3900., 4000., 4100., 4200., 4300., 4400., 4500., 4600.,
        4700., 4800., 4900., 5000., 5100., 5200., 5300., 5400.,
        5500., 5600., 5700., 5800., 5900., 6000., 6100., 6200.,
        6300., 6400., 6500., 6600., 6700., 6800., 6900., 7000.,
        7200., 7400., 7600., 7800., 8000., 8200., 8400., 8600.,
        8800., 9000., 9200., 9400., 9600., 9800., 10000., 10200.,
        10400., 10600., 10800., 11000., 11200., 11400., 11600., 11800.,
        12000.
    ], dtype=float)

    DEFAULT_LOGG_GRID = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.], dtype=float)
    DEFAULT_FEH_GRID = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5], dtype=float)

    def __init__(self, base_dir,
                 wave_filename="WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
                 model_tag="PHOENIX-ACES-AGSS-COND-2011-HiRes",
                 verbose=True):
        self._flux_grid = None
        self.base_dir = os.path.abspath(os.path.expanduser(base_dir))
        self.wave_filename = wave_filename
        self.model_tag = model_tag
        self.verbose = bool(verbose)

        wave_path = os.path.join(self.base_dir, self.wave_filename)
        if not os.path.exists(wave_path):
            raise FileNotFoundError("PHOENIX wavelength file not found: {0}".format(wave_path))

        self.phoenix_wave = fits.getdata(wave_path).astype(float)
        self._interp = None
        self._grid = None
        self.wave = None  # the wave grid of the interpolator, usually observed_wave

    def template_path(self, teff, logg, feh):
        rel = phoenix_relpath(teff, logg, feh, model_tag=self.model_tag)
        return os.path.join(self.base_dir, rel)

    def load_template(self, teff, logg, feh, wave=None):
        """
        Load a single PHOENIX template flux array.
        If wave is provided, resample to that wavelength grid.
        """
        path = self.template_path(teff, logg, feh)
        if not os.path.exists(path):
            raise FileNotFoundError("PHOENIX template not found: {0}".format(path))
        
        flux = fits.getdata(path).astype(float)
        
        if wave is None:
            return self.phoenix_wave.copy(), flux

        wave = _as_float_array(wave)
        # Only spline over the region we need
        wmin, wmax = float(wave.min()), float(wave.max())
        mask = (self.phoenix_wave >= wmin) & (self.phoenix_wave <= wmax)
        if mask.sum() < 4:
            raise ValueError("Requested wave range does not overlap PHOENIX wave grid.")

        spline = InterpolatedUnivariateSpline(self.phoenix_wave[mask], flux[mask], k=3, ext=1)
        return wave, spline(wave)

    @staticmethod
    def _default_cache_name(teff_grid, feh_grid, logg_grid, wave):
        h = hashlib.md5()
        h.update(np.asarray(teff_grid, dtype=float).tobytes())
        h.update(np.asarray(feh_grid, dtype=float).tobytes())
        h.update(np.asarray(logg_grid, dtype=float).tobytes())
        h.update(np.asarray(wave, dtype=float).tobytes())
        return "phoenix_cache_{0}.npz".format(h.hexdigest()[:12])

    def build_interpolator(self,
                           observed_wave,
                           teff_grid=None,
                           feh_grid=None,
                           logg_grid=None,
                           cache_path=None,
                           allow_missing=False):
        """
        Build (or load) a RegularGridInterpolator on (Teff, [Fe/H], logg).
        The templates are resampled onto observed_wave and stored in memory (and optionally cached).

        If cache_path exists, it will be loaded.
        If cache_path is None, nothing is written to disk.
        """
        observed_wave = _as_float_array(observed_wave)
        self.wave = observed_wave.copy()

        teff_grid = self.DEFAULT_TEFF_GRID if teff_grid is None else np.asarray(teff_grid, dtype=float)
        feh_grid = self.DEFAULT_FEH_GRID if feh_grid is None else np.asarray(feh_grid, dtype=float)
        logg_grid = self.DEFAULT_LOGG_GRID if logg_grid is None else np.asarray(logg_grid, dtype=float)

        if cache_path is not None:
            cache_path = os.path.abspath(os.path.expanduser(cache_path))
            if os.path.exists(cache_path):
                self.load_cache(cache_path, expected_wave=observed_wave)
                return self._interp

        flux_grid = np.full((len(teff_grid), len(feh_grid), len(logg_grid), len(observed_wave)),
                            np.nan, dtype=float)
        
        
        for it, teff in enumerate(teff_grid):
            for iz, feh in enumerate(feh_grid):
                for ig, logg in enumerate(logg_grid):
                    try:
                        _, f = self.load_template(teff, logg, feh, wave=observed_wave)
                        flux_grid[it, iz, ig, :] = f
                    except Exception as e:
                        if not allow_missing:
                            raise
                        if self.verbose:
                            print("Skipping missing template teff={0} feh={1} logg={2}: {3}".format(teff, feh, logg, str(e)))
        
        self._flux_grid = flux_grid
        
        if np.isnan(flux_grid).any() and not allow_missing:
            raise RuntimeError("NaNs present in flux_grid but allow_missing=False.")

        # RegularGridInterpolator expects axes order matching the data
        self._grid = (teff_grid, feh_grid, logg_grid)
        self._interp = RegularGridInterpolator(self._grid, flux_grid, method="linear", bounds_error=True)

        if cache_path is not None:
            self.save_cache(cache_path)

        return self._interp

    def evaluate(self, teff, feh, logg):
        """
        Evaluate the interpolated spectrum on self.wave.
        build_interpolator() must have been called first.
        """
        if self._interp is None or self.wave is None:
            raise RuntimeError("Interpolator not built. Call build_interpolator() first.")
        p = (float(teff), float(feh), float(logg))
        return self._interp(p)
    
    def save_cache(self, cache_path, dtype=np.float32):
        if self._grid is None or self.wave is None or self._flux_grid is None:
            raise RuntimeError("Nothing to save. Build interpolator first.")

        teff_grid, feh_grid, logg_grid = self._grid

        flux = self._flux_grid
        if dtype is not None:
            flux = flux.astype(dtype, copy=False)

        np.savez_compressed(
            cache_path,
            teff_grid=np.asarray(teff_grid, dtype=float),
            feh_grid=np.asarray(feh_grid, dtype=float),
            logg_grid=np.asarray(logg_grid, dtype=float),
            wave=np.asarray(self.wave, dtype=float),
            flux_grid=flux,
            model_tag=self.model_tag,
            wave_filename=self.wave_filename,
        )
        if self.verbose:
            print("Saved PHOENIX cache to {0}".format(cache_path))
    
    def load_cache(self, cache_path, expected_wave=None):
        d = np.load(cache_path, allow_pickle=False)

        teff_grid = d["teff_grid"].astype(float)
        feh_grid = d["feh_grid"].astype(float)
        logg_grid = d["logg_grid"].astype(float)
        wave = d["wave"].astype(float)
        flux_grid = d["flux_grid"]

        if expected_wave is not None:
            expected_wave = _as_float_array(expected_wave)
            if len(expected_wave) != len(wave) or not np.allclose(expected_wave, wave, rtol=0, atol=0):
                raise ValueError("Cached wavelength grid does not match requested observed_wave.")

        self.wave = wave
        self._grid = (teff_grid, feh_grid, logg_grid)
        self._flux_grid = flux_grid.astype(float, copy=False)

        self._interp = RegularGridInterpolator(self._grid, self._flux_grid, method="linear", bounds_error=True)

        if self.verbose:
            print("Loaded PHOENIX cache from {0}".format(cache_path))

        return self._interp
