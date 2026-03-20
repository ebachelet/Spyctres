import os
import argparse
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


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "PHOENIX backend smoke test.\n"
            "This checks that Spyctres can load a local PHOENIX template, build a "
            "small interpolator on a tiny parameter grid, and round-trip that "
            "interpolator through the .npz cache."
        ),
        epilog=(
            "Examples:\n"
            "  export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2\n"
            "  python scripts/phoenix_smoketest.py\n\n"
            "  python scripts/phoenix_smoketest.py \\\n"
            "    --phoenix-dir /path/to/PHOENIXv2 \\\n"
            "    --cache-path /tmp/spyctres_phoenix_cache_test.npz \\\n"
            "    --verbose\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main():
    parser = build_parser()
    parser.add_argument(
        "--phoenix-dir",
        default=os.environ.get("SPYCTRES_PHOENIX_DIR", None),
        help="Path to local PHOENIXv2 directory. Defaults to SPYCTRES_PHOENIX_DIR.",
    )
    parser.add_argument(
        "--wave-min",
        type=float,
        default=6500.0,
        help="Minimum wavelength in Angstrom for the small test chunk.",
    )
    parser.add_argument(
        "--wave-max",
        type=float,
        default=6600.0,
        help="Maximum wavelength in Angstrom for the small test chunk.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Subsampling stride applied to the PHOENIX wavelength grid.",
    )
    parser.add_argument(
        "--cache-path",
        default="/tmp/spyctres_phoenix_cache_test.npz",
        help="Cache path used for the interpolator cache round-trip test.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose PHOENIX library output.",
    )
    args = parser.parse_args()

    if args.phoenix_dir is None:
        parser.error("No PHOENIX directory supplied. Set --phoenix-dir or SPYCTRES_PHOENIX_DIR.")

    if not os.path.isdir(args.phoenix_dir):
        parser.error("PHOENIX directory not found: {0}".format(args.phoenix_dir))

    if args.wave_max <= args.wave_min:
        parser.error("--wave-max must be greater than --wave-min.")

    if args.stride < 1:
        parser.error("--stride must be >= 1.")

    lib = PhoenixLibrary(args.phoenix_dir, verbose=args.verbose)

    # Pick a small wavelength chunk from the PHOENIX wave grid to keep things fast.
    # PHOENIX native wavelengths are vacuum wavelengths.
    w = lib.phoenix_wave
    mask = (w > args.wave_min) & (w < args.wave_max)
    wave_small = w[mask][::args.stride]

    if wave_small.size < 10:
        parser.error(
            "Selected wavelength chunk is too small after masking/stride. "
            "Adjust --wave-min/--wave-max/--stride."
        )

    # First check: can we load a single template and resample?
    # Users can adjust these to a combination they know exists in their local PHOENIX install.
    teff0, logg0, feh0 = 5000, 4.0, 0.0
    wave_out, flux_out = lib.load_template(teff0, logg0, feh0, wave=wave_small)

    print(
        "Loaded template:",
        teff0, logg0, feh0,
        "shape=", flux_out.shape,
        "finite=", np.isfinite(flux_out).all(),
    )

    # Second check: build a tiny interpolator on 2x2x2 = 8 templates.
    # If any are missing locally, this will raise with a file path that tells you what was not found.
    teff_grid = np.array([5000, 5100], dtype=float)
    feh_grid = np.array([-0.5, 0.0], dtype=float)
    logg_grid = np.array([4.0, 4.5], dtype=float)

    lib.build_interpolator(
        observed_wave=wave_small,
        teff_grid=teff_grid,
        feh_grid=feh_grid,
        logg_grid=logg_grid,
        cache_path=None,
        allow_missing=False,
    )

    f_mid = lib.evaluate(5050, -0.25, 4.25)
    print(
        "Interpolated spectrum shape=", f_mid.shape,
        "finite=", np.isfinite(f_mid).all(),
        "min/max=", float(np.min(f_mid)), float(np.max(f_mid)),
    )

    cache_path = args.cache_path
    try:
        os.remove(cache_path)
    except OSError:
        pass

    lib.build_interpolator(
        observed_wave=wave_small,
        teff_grid=teff_grid,
        feh_grid=feh_grid,
        logg_grid=logg_grid,
        cache_path=cache_path,
        allow_missing=False,
    )
    f1 = lib.evaluate(5050, -0.25, 4.25)

    lib2 = PhoenixLibrary(args.phoenix_dir, verbose=args.verbose)
    lib2.build_interpolator(
        observed_wave=wave_small,
        teff_grid=teff_grid,
        feh_grid=feh_grid,
        logg_grid=logg_grid,
        cache_path=cache_path,
        allow_missing=False,
    )
    f2 = lib2.evaluate(5050, -0.25, 4.25)

    diff = f1 - f2
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.max(np.abs(f1)))
    max_rel = float(max_abs / denom) if denom > 0 else 0.0

    print("Max |Δ|:", max_abs)
    print("Max rel Δ:", max_rel)
    print("Cache allclose:", np.allclose(f1, f2, rtol=1e-6, atol=0.0))


if __name__ == "__main__":
    main()
