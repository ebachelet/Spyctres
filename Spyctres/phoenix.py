# Spyctres/phoenix.py
import os
import hashlib
import numpy as np
from astropy.io import fits

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RegularGridInterpolator
from .waveutils import convert_wavelength_medium


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


def _same_float_array(a, b):
    """
    Exact array equality for 1D float-like grids used in cache validation.
    """
    if a is None or b is None:
        return False

    a = _as_float_array(a)
    b = _as_float_array(b)

    if len(a) != len(b):
        return False

    return np.allclose(a, b, rtol=0.0, atol=0.0)
 
def _normalize_cache_string(x):
    """
    Normalize scalar strings loaded from npz cache entries.
    """
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        if x.shape == ():
            x = x.item()
        else:
            raise ValueError("Expected a scalar cache string entry.")

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)
    

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
             phoenix_wave_medium="vacuum",
             verbose=True):
        self._flux_grid = None
        self.base_dir = os.path.abspath(os.path.expanduser(base_dir))
        self.wave_filename = wave_filename
        self.model_tag = model_tag
        self.phoenix_wave_medium = str(phoenix_wave_medium).lower()
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

    def scan_available_points(self):
        """
        Scan the local PHOENIX directory tree and return the available grid
        points as a sorted list of (teff, feh, logg) tuples.
        """
        import glob
        import re

        pat = re.compile(
            r"lte(?P<teff>\d+)-(?P<logg>\d+\.\d+)(?P<feh>[+-]\d+\.\d+)\.PHOENIX"
        )

        points = []
        for zdir in sorted(glob.glob(os.path.join(self.base_dir, "Z*"))):
            for path in sorted(glob.glob(os.path.join(zdir, "lte*.fits"))):
                name = os.path.basename(path)
                m = pat.search(name)
                if m is None:
                    continue

                teff = float(m.group("teff"))
                logg = float(m.group("logg"))
                feh = float(m.group("feh"))
                points.append((teff, feh, logg))

        if len(points) == 0:
            raise RuntimeError(
                "No PHOENIX templates found under {0}".format(self.base_dir)
            )

        return sorted(set(points))

    def available_axes(self):
        """
        Return the actually installed Teff / [Fe/H] / logg axes.
        """
        pts = self.scan_available_points()
        teff = np.unique([p[0] for p in pts]).astype(float)
        feh = np.unique([p[1] for p in pts]).astype(float)
        logg = np.unique([p[2] for p in pts]).astype(float)
        return teff, feh, logg   
        
    def has_template(self, teff, logg, feh):
        """
        Return True if the requested PHOENIX template exists on disk.
        """
        return os.path.exists(self.template_path(teff, logg, feh))

    def complete_subgrid(self, teff_grid, feh_grid, logg_grid, max_iter=20):
        """
        Trim candidate Teff / [Fe/H] / logg axes to a complete rectangular
        subgrid that actually exists on disk.

        This is conservative: it iteratively removes axis values that do not
        have templates for all combinations with the current remaining axes.
        """
        teff_grid = np.unique(_as_float_array(teff_grid))
        feh_grid = np.unique(_as_float_array(feh_grid))
        logg_grid = np.unique(_as_float_array(logg_grid))

        for _ in range(int(max_iter)):
            changed = False

            teff_new = np.array([
                t for t in teff_grid
                if all(self.has_template(t, g, z) for g in logg_grid for z in feh_grid)
            ], dtype=float)
            if len(teff_new) != len(teff_grid):
                teff_grid = teff_new
                changed = True

            feh_new = np.array([
                z for z in feh_grid
                if all(self.has_template(t, g, z) for t in teff_grid for g in logg_grid)
            ], dtype=float)
            if len(feh_new) != len(feh_grid):
                feh_grid = feh_new
                changed = True

            logg_new = np.array([
                g for g in logg_grid
                if all(self.has_template(t, g, z) for t in teff_grid for z in feh_grid)
            ], dtype=float)
            if len(logg_new) != len(logg_grid):
                logg_grid = logg_new
                changed = True

            if len(teff_grid) == 0 or len(feh_grid) == 0 or len(logg_grid) == 0:
                raise ValueError("No complete PHOENIX subgrid remains after availability filtering.")

            if not changed:
                break
        else:
            raise RuntimeError("complete_subgrid did not converge within max_iter.")

        return teff_grid, feh_grid, logg_grid
    
    def load_template(self, teff, logg, feh, wave=None, wave_medium=None):
        """
        Load a single PHOENIX template flux array.

        If `wave` is provided, resample the template onto that wavelength grid.
        The PHOENIX wavelength grid is first converted from `self.phoenix_wave_medium`
        into `wave_medium` when needed.
        """
        path = self.template_path(teff, logg, feh)
        if not os.path.exists(path):
            raise FileNotFoundError("PHOENIX template not found: {0}".format(path))

        flux = fits.getdata(path).astype(float)

        template_wave = self.phoenix_wave.copy()
        target_medium = self.phoenix_wave_medium if wave_medium is None else str(wave_medium).lower()

        if target_medium != self.phoenix_wave_medium:
            template_wave = convert_wavelength_medium(
                template_wave,
                from_medium=self.phoenix_wave_medium,
                to_medium=target_medium,
            )

        if wave is None:
            return template_wave, flux

        wave = _as_float_array(wave)

        wmin, wmax = float(wave.min()), float(wave.max())
        mask = (template_wave >= wmin) & (template_wave <= wmax)
        if mask.sum() < 4:
            raise ValueError("Requested wave range does not overlap PHOENIX wave grid.")

        spline = InterpolatedUnivariateSpline(template_wave[mask], flux[mask], k=3, ext=1)
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
                       observed_wave_medium=None,
                       cache_path=None,
                       allow_missing=False):
        """
        Build (or load) a RegularGridInterpolator on (Teff, [Fe/H], logg).
        The templates are resampled onto observed_wave and stored in memory (and optionally cached).

        If cache_path exists, it will be loaded.
        If cache_path is None, nothing is written to disk.
        """
        observed_wave = _as_float_array(observed_wave)
        observed_wave_medium = (
            self.phoenix_wave_medium
            if observed_wave_medium is None
            else str(observed_wave_medium).lower()
        )
        self.wave = observed_wave.copy()

        teff_grid = self.DEFAULT_TEFF_GRID if teff_grid is None else np.asarray(teff_grid, dtype=float)
        feh_grid = self.DEFAULT_FEH_GRID if feh_grid is None else np.asarray(feh_grid, dtype=float)
        logg_grid = self.DEFAULT_LOGG_GRID if logg_grid is None else np.asarray(logg_grid, dtype=float)
        
        if cache_path is not None:
            cache_path = os.path.abspath(os.path.expanduser(cache_path))
            if os.path.exists(cache_path):
                try:
                    self.load_cache(
                        cache_path,
                        expected_wave=observed_wave,
                        expected_teff_grid=teff_grid,
                        expected_feh_grid=feh_grid,
                        expected_logg_grid=logg_grid,
                        expected_observed_wave_medium=observed_wave_medium,
                    )
                    return self._interp
                except ValueError as e:
                    if self.verbose:
                        print("Cache mismatch, rebuilding:", cache_path)
                        print(str(e))
                                
        flux_grid = np.full((len(teff_grid), len(feh_grid), len(logg_grid), len(observed_wave)),
                            np.nan, dtype=float)    
        
        for it, teff in enumerate(teff_grid):
            for iz, feh in enumerate(feh_grid):
                for ig, logg in enumerate(logg_grid):
                    try:
                        _, f = self.load_template(
                            teff, logg, feh,
                            wave=observed_wave,
                            wave_medium=observed_wave_medium,
                        )
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
            self.save_cache(cache_path, observed_wave_medium=observed_wave_medium)

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
    
    def save_cache(self, cache_path, observed_wave_medium):
        if self._grid is None or self.wave is None or self._flux_grid is None:
            raise RuntimeError("Nothing to save. Build interpolator first.")

        teff_grid, feh_grid, logg_grid = self._grid
        flux = self._flux_grid.astype(float, copy=False)

        np.savez_compressed(
            cache_path,
            teff_grid=np.asarray(teff_grid, dtype=float),
            feh_grid=np.asarray(feh_grid, dtype=float),
            logg_grid=np.asarray(logg_grid, dtype=float),
            wave=np.asarray(self.wave, dtype=float),
            flux_grid=flux,
            model_tag=self.model_tag,
            wave_filename=self.wave_filename,
            phoenix_wave_medium=np.asarray(self.phoenix_wave_medium),
            observed_wave_medium=np.asarray(observed_wave_medium),
        )
        if self.verbose:
            print("Saved PHOENIX cache to {0}".format(cache_path))
    
    def load_cache(
        self,
        cache_path,
        expected_wave=None,
        expected_teff_grid=None,
        expected_feh_grid=None,
        expected_logg_grid=None,
        expected_observed_wave_medium=None,
    ):
        d = np.load(cache_path, allow_pickle=False)

        teff_grid = d["teff_grid"].astype(float)
        feh_grid = d["feh_grid"].astype(float)
        logg_grid = d["logg_grid"].astype(float)
        wave = d["wave"].astype(float)
        flux_grid = d["flux_grid"]
        cached_model_tag = _normalize_cache_string(d["model_tag"]) if "model_tag" in d.files else None
        cached_wave_filename = _normalize_cache_string(d["wave_filename"]) if "wave_filename" in d.files else None
        cached_phoenix_wave_medium = _normalize_cache_string(d["phoenix_wave_medium"]) if "phoenix_wave_medium" in d.files else None
        cached_observed_wave_medium = _normalize_cache_string(d["observed_wave_medium"]) if "observed_wave_medium" in d.files else None   
            
        if expected_wave is not None:
            if not _same_float_array(expected_wave, wave):
                raise ValueError("Cached wavelength grid does not match requested observed_wave.")

        if expected_teff_grid is not None:
            if not _same_float_array(expected_teff_grid, teff_grid):
                raise ValueError("Cached teff_grid does not match requested teff_grid.")

        if expected_feh_grid is not None:
            if not _same_float_array(expected_feh_grid, feh_grid):
                raise ValueError("Cached feh_grid does not match requested feh_grid.")

        if expected_logg_grid is not None:
            if not _same_float_array(expected_logg_grid, logg_grid):
                raise ValueError("Cached logg_grid does not match requested logg_grid.")
        
        if cached_model_tag is None:
            raise ValueError("Cached model_tag is missing.")

        if cached_model_tag != str(self.model_tag):
            raise ValueError(
                "Cached model_tag does not match current PhoenixLibrary.model_tag: "
                "{0} != {1}".format(cached_model_tag, self.model_tag)
            )
        
        if cached_wave_filename is None:
            raise ValueError("Cached wave_filename is missing.")

        if cached_wave_filename != str(self.wave_filename):
            raise ValueError(
                "Cached wave_filename does not match current PhoenixLibrary.wave_filename: "
                "{0} != {1}".format(cached_wave_filename, self.wave_filename)
            )
            
        if cached_phoenix_wave_medium is None:
            raise ValueError("Cached phoenix_wave_medium is missing.")
        if cached_phoenix_wave_medium != str(self.phoenix_wave_medium):
            raise ValueError(
                "Cached phoenix_wave_medium does not match current PhoenixLibrary.phoenix_wave_medium: "
                "{0} != {1}".format(cached_phoenix_wave_medium, self.phoenix_wave_medium)
            )

        if expected_observed_wave_medium is not None:
            if cached_observed_wave_medium is None:
                raise ValueError("Cached observed_wave_medium is missing.")
            if cached_observed_wave_medium != str(expected_observed_wave_medium):
                raise ValueError(
                    "Cached observed_wave_medium does not match requested observed_wave_medium: "
                    "{0} != {1}".format(cached_observed_wave_medium, expected_observed_wave_medium)
                )     
                   
        self.wave = wave
        self._grid = (teff_grid, feh_grid, logg_grid)
        self._flux_grid = flux_grid.astype(float, copy=False)

        self._interp = RegularGridInterpolator(
            self._grid,
            self._flux_grid,
            method="linear",
            bounds_error=True,
        )

        if self.verbose:
            print("Loaded PHOENIX cache from {0}".format(cache_path))

        return self._interp
