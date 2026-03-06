import os
import numpy as np
from Spyctres.phoenix import PhoenixLibrary

PHOENIX_DIR = os.environ.get("SPYCTRES_PHOENIX_DIR", None)
if PHOENIX_DIR is None:
    raise SystemExit("Set SPYCTRES_PHOENIX_DIR to your HiResFITS directory first.")

lib = PhoenixLibrary(PHOENIX_DIR, verbose=True)

# Pick a small wavelength chunk from the PHOENIX wave grid to keep things fast
w = lib.phoenix_wave
mask = (w > 6500) & (w < 6600)
wave_small = w[mask][::5]  # decimate a bit

# First check: can we load a single template and resample?
# Users can adjust these to a combination they know exists in their local PHOENIX install.
teff0, logg0, feh0 = 5000, 4.0, 0.0
wave_out, flux_out = lib.load_template(teff0, logg0, feh0, wave=wave_small)

print("Loaded template:", teff0, logg0, feh0, "shape=", flux_out.shape, "finite=", np.isfinite(flux_out).all())

# Second check: build a tiny interpolator on 2x2x2 = 8 templates.
# If any are missing locally, this will throw with a file path that tells you what was not found.
teff_grid = np.array([5000, 5100], dtype=float)
feh_grid = np.array([-0.5, 0.0], dtype=float)
logg_grid = np.array([4.0, 4.5], dtype=float)

lib.build_interpolator(
    observed_wave=wave_small,
    teff_grid=teff_grid,
    feh_grid=feh_grid,
    logg_grid=logg_grid,
    cache_path=None,
    allow_missing=False
)

f_mid = lib.evaluate(5050, -0.25, 4.25)
print("Interpolated spectrum shape=", f_mid.shape, "finite=", np.isfinite(f_mid).all(), "min/max=", float(np.min(f_mid)), float(np.max(f_mid)))
