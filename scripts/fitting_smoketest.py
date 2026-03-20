import os
import numpy as np

from Spyctres.phoenix import PhoenixLibrary
from Spyctres.io import SpectrumSegment
from Spyctres.fitting import fit_phoenix_full_spectrum, _solve_multiplicative_legendre, _gaussian_broaden_velocity
from Spyctres.Spyctres import velocity_correction
import warnings
# pysynphot is legacy and emits a pkg_resources deprecation warning.
# Suppress it in smoke-test scripts to keep output readable.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pysynphot.*",
)

PHOENIX_DIR = os.environ.get("SPYCTRES_PHOENIX_DIR", None)
if PHOENIX_DIR is None:
    raise SystemExit("Set SPYCTRES_PHOENIX_DIR to your PHOENIXv2 directory first.")

R_TEST = 50000.0
FWHM_TEST = 299792.458 / R_TEST

# Build a small wavelength chunk for a fast test
lib = PhoenixLibrary(PHOENIX_DIR, verbose=True)
w = lib.phoenix_wave
mask = (w > 6500) & (w < 6600)
wave = w[mask][::5]

# Tiny grid around the truth
teff_grid = np.array([4900, 5000, 5100], dtype=float)
feh_grid  = np.array([-0.5, 0.0], dtype=float)
logg_grid = np.array([4.0, 4.5], dtype=float)

lib.build_interpolator(
    observed_wave=wave,
    teff_grid=teff_grid,
    feh_grid=feh_grid,
    logg_grid=logg_grid,
    cache_path="/tmp/spyctres_fit_cache.npz",
    allow_missing=False
)

# Synthetic truth (must lie inside the grid bounds)
truth = dict(teff=5050.0, feh=-0.25, logg=4.25, rv=12.3)

# Generate synthetic data using the same shift operator used in fitting
model0 = lib.evaluate(truth["teff"], truth["feh"], truth["logg"])
shifted = velocity_correction(np.c_[wave, model0], truth["rv"])[:, 1]
shifted = _gaussian_broaden_velocity(wave, shifted, fwhm_kms=FWHM_TEST)

# Mild multiplicative continuum tilt
x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
cont = 1.0 + 0.02 * x
flux = shifted * cont

# Add noise
sigma = 0.01 * np.median(flux)
rng = np.random.RandomState(0)
flux_n = flux + rng.normal(0.0, sigma, size=flux.size)
err = np.ones_like(flux_n) * sigma

seg = SpectrumSegment(wave, flux_n, err=err, name="synthetic")

def chi2_red_with_poly(teff, feh, logg, rv, mdeg=2):
    m0 = lib.evaluate(teff, feh, logg)
    sh = velocity_correction(np.c_[wave, m0], rv)[:, 1]
    sh = _gaussian_broaden_velocity(wave, sh, fwhm_kms=FWHM_TEST)
    m_corr, coeffs = _solve_multiplicative_legendre(wave, flux_n, err, sh, mdeg=mdeg)
    r = (flux_n - m_corr) / err
    chi2 = float(np.sum(r * r))
    dof = max(1, len(r) - 4)
    return chi2 / dof, coeffs

chi2t, ct = chi2_red_with_poly(truth["teff"], truth["feh"], truth["logg"], truth["rv"], mdeg=2)
print("chi2_red at TRUTH:", chi2t, "poly coeffs:", ct)

# RV shift sanity check (non-zero means RV actually changes the spectrum)
m0 = lib.evaluate(truth["teff"], truth["feh"], truth["logg"])
sh0 = velocity_correction(np.c_[wave, m0], 0.0)[:, 1]
sh1 = velocity_correction(np.c_[wave, m0], truth["rv"])[:, 1]
print("RV effect (max |Δ| / median):", float(np.max(np.abs(sh1 - sh0)) / np.median(np.abs(sh0))))

# Fit
p0 = (5000.0, 0.0, 4.0, 0.0)
bounds = ((4900.0, -0.5, 4.0, -50.0), (5100.0, 0.0, 4.5, 50.0))

out = fit_phoenix_full_spectrum(
    [seg],
    phoenix_lib=lib,
    p0=p0,
    R=R_TEST,
    bounds=bounds,
    regions=None,
    mdeg=2,
    rv_bary_kms=0.0,
    verbose=2,      # show optimizer progress
    max_nfev=500
)

print("Truth:", truth)
print("Best :", dict(teff=out["teff"], feh=out["feh"], logg=out["logg"], rv=out["rv_kms"]))
print("chi2_red (reported):", out["chi2_red"])

chi2b, cb = chi2_red_with_poly(out["teff"], out["feh"], out["logg"], out["rv_kms"], mdeg=2)
print("chi2_red at BEST  :", chi2b, "poly coeffs:", cb)
