import os
import argparse
import warnings
# pysynphot is legacy and emits a pkg_resources deprecation warning.
# Suppress it in smoke-test scripts to keep output readable.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pysynphot.*",
)

import numpy as np

from Spyctres.phoenix import PhoenixLibrary
from Spyctres.io import SpectrumSegment
from Spyctres.fitting import (
    fit_phoenix_full_spectrum,
    reconstruct_phoenix_legendre_models_for_segments,
    _solve_multiplicative_legendre,
    _gaussian_broaden_velocity,
    _apply_observed_grid_rv_shift,
)
from Spyctres.config import resolve_phoenix_dir

def build_parser():
    return argparse.ArgumentParser(
        description=(
            "Synthetic PHOENIX full-spectrum fitter smoke test.\n"
            "This builds a small synthetic spectrum from the PHOENIX library, "
            "adds noise and a mild continuum tilt, then checks whether "
            "fit_phoenix_full_spectrum() recovers the planted parameters."
        ),
        epilog=(
            "Examples:\n"
            "  export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2\n"
            "  python scripts/fitting_smoketest.py\n\n"
            "  python scripts/fitting_smoketest.py \\\n"
            "    --phoenix-dir /path/to/PHOENIXv2 \\\n"
            "    --cache-path /tmp/spyctres_fit_cache.npz \\\n"
            "    --verbose 2\n\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def chi2_red_with_poly(lib, wave, flux_n, err, teff, feh, logg, rv, fwhm_kms, mdeg=2):
    m0 = lib.evaluate(teff, feh, logg)
    sh = _apply_observed_grid_rv_shift(wave, m0, rv)
    sh = _gaussian_broaden_velocity(wave, sh, fwhm_kms=fwhm_kms)
    m_corr, coeffs = _solve_multiplicative_legendre(wave, flux_n, err, sh, mdeg=mdeg)
    r = (flux_n - m_corr) / err
    chi2 = float(np.sum(r * r))
    dof = max(1, len(r) - 4 - (int(mdeg) + 1))
    return chi2 / dof, coeffs


def main():
    parser = build_parser()
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Path to local PHOENIXv2 directory. Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file.",
    )
    parser.add_argument(
        "--cache-path",
        default="/tmp/spyctres_fit_cache.npz",
        help="Cache path for the observed-grid PHOENIX interpolator.",
    )
    parser.add_argument(
        "--wave-min",
        type=float,
        default=6500.0,
        help="Minimum wavelength in Angstrom for the synthetic chunk.",
    )
    parser.add_argument(
        "--wave-max",
        type=float,
        default=6600.0,
        help="Maximum wavelength in Angstrom for the synthetic chunk.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Subsampling stride applied to the native PHOENIX wavelength grid.",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=50000.0,
        help="Resolving power used to broaden the synthetic spectrum.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic noise generation.",
    )
    parser.add_argument(
        "--noise-frac",
        type=float,
        default=0.01,
        help="Noise sigma as a fraction of the median noiseless flux.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity passed to fit_phoenix_full_spectrum().",
    )
    args = parser.parse_args()
    try:
        args.phoenix_dir = resolve_phoenix_dir(args.phoenix_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    
    if args.phoenix_dir is None:
        parser.error(
            "No PHOENIX directory supplied. Set --phoenix-dir, SPYCTRES_PHOENIX_DIR, "
            "or [paths].phoenix_dir in ~/.config/spyctres/config.toml."
        )
        
    if args.wave_max <= args.wave_min:
        parser.error("--wave-max must be greater than --wave-min.")

    if args.stride < 1:
        parser.error("--stride must be >= 1.")

    if args.R <= 0:
        parser.error("--R must be positive.")

    if args.noise_frac <= 0:
        parser.error("--noise-frac must be positive.")

    r_test = float(args.R)
    fwhm_test = 299792.458 / r_test

    # Build a small wavelength chunk for a fast test
    lib = PhoenixLibrary(args.phoenix_dir, verbose=True)
    w = lib.phoenix_wave
    mask = (w > args.wave_min) & (w < args.wave_max)
    wave = w[mask][::args.stride]
    # PHOENIX native wavelengths are vacuum wavelengths.

    if wave.size < 10:
        parser.error(
            "Selected wavelength chunk is too small after masking/stride. "
            "Adjust --wave-min/--wave-max/--stride."
        )

    # Tiny grid around the truth
    teff_grid = np.array([4900, 5000, 5100], dtype=float)
    feh_grid = np.array([-0.5, 0.0], dtype=float)
    logg_grid = np.array([4.0, 4.5], dtype=float)

    lib.build_interpolator(
        observed_wave=wave,
        teff_grid=teff_grid,
        feh_grid=feh_grid,
        logg_grid=logg_grid,
        cache_path=args.cache_path,
        allow_missing=False,
        progress_callback=lambda message: print(message, flush=True),
    )

    # Synthetic truth (must lie inside the grid bounds)
    truth = dict(teff=5050.0, feh=-0.25, logg=4.25, rv=12.3)

    # Generate synthetic data using the same shift operator used in fitting
    model0 = lib.evaluate(truth["teff"], truth["feh"], truth["logg"])
    shifted = _apply_observed_grid_rv_shift(wave, model0, truth["rv"])
    shifted = _gaussian_broaden_velocity(wave, shifted, fwhm_kms=fwhm_test)

    # Mild multiplicative continuum tilt
    x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
    cont = 1.0 + 0.02 * x
    flux = shifted * cont

    # Add noise
    sigma = float(args.noise_frac) * float(np.median(flux))
    rng = np.random.RandomState(args.seed)
    flux_n = flux + rng.normal(0.0, sigma, size=flux.size)
    err = np.ones_like(flux_n) * sigma

    seg = SpectrumSegment(
        wave,
        flux_n,
        err=err,
        wave_medium="vacuum",
        name="synthetic",
    )

    chi2t, ct = chi2_red_with_poly(
        lib, wave, flux_n, err,
        truth["teff"], truth["feh"], truth["logg"], truth["rv"],
        fwhm_test, mdeg=2,
    )
    print("chi2_red at TRUTH:", chi2t, "poly coeffs:", ct)

    # RV shift sanity check
    m0 = lib.evaluate(truth["teff"], truth["feh"], truth["logg"])
    sh0 = _apply_observed_grid_rv_shift(wave, m0, 0.0)
    sh1 = _apply_observed_grid_rv_shift(wave, m0, truth["rv"])
    print("RV effect (max |Δ| / median):", float(np.max(np.abs(sh1 - sh0)) / np.median(np.abs(sh0))))

    # Fit
    p0 = (5000.0, 0.0, 4.0, 0.0)
    bounds = ((4900.0, -0.5, 4.0, -50.0), (5100.0, 0.0, 4.5, 50.0))

    out = fit_phoenix_full_spectrum(
        [seg],
        phoenix_lib=lib,
        p0=p0,
        R=r_test,
        bounds=bounds,
        regions=None,
        mdeg=2,
        rv_bary_kms=0.0,
        verbose=args.verbose,
        max_nfev=500,
        progress_callback=lambda message: print(message, flush=True),
    )

    print("Truth:", truth)
    print("Best :", dict(teff=out["teff"], feh=out["feh"], logg=out["logg"], rv=out["rv_kms"]))
    print("chi2_red (reported):", out["chi2_red"])

    models, coeffs, used_masks, _excluded = (
        reconstruct_phoenix_legendre_models_for_segments(
            [seg], lib, out, mdeg=2, R=r_test
        )
    )
    used = used_masks[0]
    resid = (flux_n[used] - models[0][used]) / err[used]
    chi2b = float(np.sum(resid * resid)) / max(1, np.count_nonzero(used) - 7)
    print("chi2_red at BEST  :", chi2b, "poly coeffs:", coeffs[0])


if __name__ == "__main__":
    main()
