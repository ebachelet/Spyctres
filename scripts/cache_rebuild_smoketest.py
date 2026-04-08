import os
import argparse
import tempfile
import warnings
import numpy as np

# pysynphot is legacy and emits a pkg_resources deprecation warning.
# Suppress it in smoke-test scripts to keep output readable.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pysynphot.*",
)

from Spyctres.phoenix import PhoenixLibrary
from Spyctres.fitting import fit_phoenix_full_spectrum
from Spyctres.io import SpectrumSegment
from Spyctres.config import load_user_config, get_config_value, resolve_setting

def build_parser():
    return argparse.ArgumentParser(
        description=(
            "PHOENIX interpolator cache-rebuild smoke test.\n"
            "This creates a synthetic spectrum, runs the fitter once to write a cache, "
            "then runs it again with the same wavelength grid but a different Teff grid. "
            "The second run should rebuild the cache and still succeed."
        ),
        epilog=(
            "Examples:\n"
            "  export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2\n"
            "  python scripts/cache_rebuild_smoketest.py\n\n"
            "  python scripts/cache_rebuild_smoketest.py \\\n"
            "    --phoenix-dir /path/to/PHOENIXv2 \\\n"
            "    --cache-path /tmp/spyctres_cache_rebuild_test.npz\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def make_synthetic_segment(phoenix_dir):
    truth = dict(teff=5050.0, feh=-0.25, logg=4.25, rv=0.0)

    wave = np.linspace(5000.0, 5300.0, 2000)

    lib_truth = PhoenixLibrary(phoenix_dir, verbose=False)
    lib_truth.build_interpolator(
        observed_wave=wave,
        teff_grid=np.array([4800.0, 5000.0, 5200.0, 5400.0]),
        feh_grid=np.array([-0.5, 0.0]),
        logg_grid=np.array([4.0, 4.5]),
        cache_path=None,
        allow_missing=False,
    )

    flux = np.asarray(lib_truth.evaluate(truth["teff"], truth["feh"], truth["logg"]), dtype=float)
    err = np.full_like(flux, 0.02 * np.median(np.abs(flux)))
    rng = np.random.default_rng(12345)
    noisy_flux = flux + rng.normal(0.0, err, size=flux.size)

    seg = SpectrumSegment(
        wave=wave,
        flux=noisy_flux,
        err=err,
        mask=np.isfinite(noisy_flux) & np.isfinite(err) & (err > 0),
        meta={"instrument": "synthetic"},
        wave_medium="vacuum",
        wave_frame="stellar_rest",
        name="synthetic_cache_rebuild_test",
    )
    return seg


def run_fit(seg, phoenix_dir, cache_path, teff_grid, label):
    print("\n=== {0} ===".format(label))
    lib = PhoenixLibrary(phoenix_dir, verbose=True)

    result = fit_phoenix_full_spectrum(
        segments=[seg],
        phoenix_lib=lib,
        p0=(5050.0, -0.25, 4.25, 0.0),
        teff_grid=np.asarray(teff_grid, dtype=float),
        feh_grid=np.array([-0.5, 0.0]),
        logg_grid=np.array([4.0, 4.5]),
        cache_path=cache_path,
        rv_init="grid",
        mdeg=2,
        verbose=False,
    )

    print(
        "best:",
        {
            "teff": result["teff"],
            "feh": result["feh"],
            "logg": result["logg"],
            "rv": result["rv_kms"],
        },
    )
    print("chi2_red:", result["chi2_red"])


def main():
    parser = build_parser()
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Path to local PHOENIXv2 directory. Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file.",
    )
    parser.add_argument(
        "--cache-path",
        default=os.path.join(tempfile.gettempdir(), "spyctres_cache_rebuild_test.npz"),
        help="Cache path used for the rebuild test.",
    )
    args = parser.parse_args()
    config = load_user_config()
    phoenix_dir_cfg = get_config_value(config, "paths", "phoenix_dir", default=None)
    
    args.phoenix_dir = resolve_setting(
        args.phoenix_dir,
        env_var_name="SPYCTRES_PHOENIX_DIR",
        config_value=phoenix_dir_cfg,
        default=None,
    )
    if args.phoenix_dir is None:
        parser.error(
        "No PHOENIX directory supplied. Set --phoenix-dir, SPYCTRES_PHOENIX_DIR, "
        "or [paths].phoenix_dir in ~/.config/spyctres/config.toml."
    )
    
    if not os.path.isdir(args.phoenix_dir):
        parser.error("PHOENIX directory not found: {0}".format(args.phoenix_dir))

    cache_path = args.cache_path

    if os.path.exists(cache_path):
        os.remove(cache_path)

    seg = make_synthetic_segment(args.phoenix_dir)

    run_fit(
        seg,
        args.phoenix_dir,
        cache_path,
        teff_grid=[4800.0, 5000.0, 5200.0, 5400.0],
        label="FIRST RUN",
    )

    run_fit(
        seg,
        args.phoenix_dir,
        cache_path,
        teff_grid=[4700.0, 4900.0, 5100.0, 5300.0],
        label="SECOND RUN WITH DIFFERENT TEFF GRID",
    )


if __name__ == "__main__":
    main()
