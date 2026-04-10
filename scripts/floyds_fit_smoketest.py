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
import matplotlib.pyplot as plt

from Spyctres.config import load_user_config, get_config_value, resolve_setting
from Spyctres.io import read_spectrum
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.fitting import (
    fit_phoenix_full_spectrum,
    reconstruct_phoenix_legendre_models_for_segments,
)
from Spyctres.plotting import plot_full_spectrum_fit


def pick_grid_range(grid, lo=None, hi=None):
    g = np.asarray(grid, dtype=float)
    m = np.ones_like(g, dtype=bool)
    if lo is not None:
        m &= (g >= float(lo))
    if hi is not None:
        m &= (g <= float(hi))
    out = g[m]
    if out.size == 0:
        raise ValueError("Requested PHOENIX grid range is empty.")
    return out


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "Quick PHOENIX fit smoke test for a reduced 1D FLOYDS spectrum.\n"
            "This is a first-pass quicklook fitter for low-resolution FLOYDS data.\n"
            "It fits a selected wavelength range with a multiplicative polynomial continuum."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/floyds_fit_smoketest.py /path/to/FLOYDS.csv\n\n"
            "  python scripts/floyds_fit_smoketest.py \\\n"
            "    --wave-medium vacuum \\\n"
            "    --forward-model native_interp \\\n"
            "    --wmin 3800 --wmax 5600 \\\n"
            "    /path/to/FLOYDS.csv\n\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main():
    parser = build_parser()
    parser.add_argument("file", help="Input FLOYDS CSV spectrum")
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Path to local PHOENIXv2 directory. Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file.",
    )
    parser.add_argument(
        "--wave-medium",
        choices=["unknown", "air", "vacuum"],
        default="unknown",
        help="Wavelength medium hypothesis for the observed spectrum.",
    )
    parser.add_argument(
        "--forward-model",
        choices=["interp_observed", "native_interp"],
        default="native_interp",
        help="Forward-model path. For unknown wavelength medium, prefer native_interp.",
    )
    parser.add_argument(
        "--model-margin",
        type=float,
        default=200.0,
        help="Margin in Angstrom for native_interp model preparation.",
    )
    parser.add_argument("--wmin", type=float, default=3800.0, help="Minimum wavelength in Angstrom")
    parser.add_argument("--wmax", type=float, default=5600.0, help="Maximum wavelength in Angstrom")
    parser.add_argument("--clip-left", type=int, default=0, help="Clip this many pixels from the left edge")
    parser.add_argument("--clip-right", type=int, default=0, help="Clip this many pixels from the right edge")
    parser.add_argument("--R-override", type=float, default=None, help="Override metadata resolving power R")
    parser.add_argument("--teff-min", type=float, default=7000.0, help="Minimum Teff for explicit PHOENIX grid")
    parser.add_argument("--teff-max", type=float, default=12000.0, help="Maximum Teff for explicit PHOENIX grid")
    parser.add_argument("--feh-min", type=float, default=-1.0, help="Minimum [Fe/H] for explicit PHOENIX grid")
    parser.add_argument("--feh-max", type=float, default=0.5, help="Maximum [Fe/H] for explicit PHOENIX grid")
    parser.add_argument("--logg-min", type=float, default=2.5, help="Minimum logg for explicit PHOENIX grid")
    parser.add_argument("--logg-max", type=float, default=5.5, help="Maximum logg for explicit PHOENIX grid")
    parser.add_argument("--mdeg", type=int, default=3, help="Legendre continuum degree")
    parser.add_argument("--teff0", type=float, default=9500.0, help="Initial Teff")
    parser.add_argument("--feh0", type=float, default=-0.5, help="Initial [Fe/H]")
    parser.add_argument("--logg0", type=float, default=4.0, help="Initial logg")
    parser.add_argument("--rv0", type=float, default=0.0, help="Initial stellar RV in km/s")
    parser.add_argument("--rv-init", choices=["grid", "none"], default="grid", help="RV initialization strategy")
    parser.add_argument("--rv-grid-n", type=int, default=161, help="Number of trial RV points in coarse RV scan")
    parser.add_argument("--cache-path", default="/tmp/spyctres_floyds_fit_cache.npz")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    config = load_user_config()
    phoenix_dir_cfg = get_config_value(config, "paths", "phoenix_dir", default=None)

    args.phoenix_dir = resolve_setting(
        args.phoenix_dir,
        env_var_name="SPYCTRES_PHOENIX_DIR",
        config_value=phoenix_dir_cfg,
        default=None,
    )

    if not os.path.isfile(args.file):
        parser.error("Input file not found: {0}".format(args.file))

    if args.phoenix_dir is None:
        parser.error(
            "No PHOENIX directory supplied. Set --phoenix-dir, SPYCTRES_PHOENIX_DIR, "
            "or [paths].phoenix_dir in ~/.config/spyctres/config.toml."
        )

    if not os.path.isdir(args.phoenix_dir):
        parser.error("PHOENIX directory not found: {0}".format(args.phoenix_dir))

    if args.wmax <= args.wmin:
        parser.error("--wmax must be greater than --wmin.")

    if args.forward_model == "interp_observed" and args.wave_medium == "unknown":
        parser.error(
            "--forward-model interp_observed requires a known --wave-medium. "
            "Use --wave-medium air or vacuum, or use --forward-model native_interp."
        )

    seg0 = read_spectrum(args.file, instrument="floyds")

    if args.wave_medium != "unknown":
        meta = dict(seg0.meta)
        meta["wave_medium"] = args.wave_medium
        seg0 = seg0.copy(meta=meta, wave_medium=args.wave_medium)

    seg = seg0.window(
        wmin=args.wmin,
        wmax=args.wmax,
        clip_left=args.clip_left,
        clip_right=args.clip_right,
        name_suffix="fitwin",
    )
    
    if seg.err is None:
        print("WARNING: no uncertainty column found; chi2_red is only heuristic in this quicklook fit.")
        
    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))

    teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
    teff_grid_req = pick_grid_range(teff_avail, args.teff_min, args.teff_max)
    feh_grid_req = pick_grid_range(feh_avail, args.feh_min, args.feh_max)
    logg_grid_req = pick_grid_range(logg_avail, args.logg_min, args.logg_max)

    teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
        teff_grid_req, feh_grid_req, logg_grid_req
    )

    R = args.R_override if args.R_override is not None else seg.meta.get("resolution_R", None)

    out = fit_phoenix_full_spectrum(
        [seg],
        phoenix_lib=phoenix_lib,
        p0=(args.teff0, args.feh0, args.logg0, args.rv0),
        exclude_mask=None,
        mdeg=args.mdeg,
        rv_bary_kms=0.0,
        R=R,
        forward_model=args.forward_model,
        model_margin_A=args.model_margin,
        teff_grid=teff_grid_fit,
        feh_grid=feh_grid_fit,
        logg_grid=logg_grid_fit,
        cache_path=args.cache_path,
        rv_init=None if args.rv_init == "none" else "grid",
        rv_grid_n=args.rv_grid_n,
        verbose=args.verbose,
        max_nfev=300,
    )

    model_list, coeffs_list, used_masks, excluded_masks = reconstruct_phoenix_legendre_models_for_segments(
        segments=[seg],
        phoenix_lib=phoenix_lib,
        fit_result=out,
        exclude_mask=None,
        mdeg=args.mdeg,
        rv_bary_kms=0.0,
        R=R,
        fwhm_kms=None,
        forward_model=args.forward_model,
        model_margin_A=args.model_margin,
    )

    print("File:", args.file)
    print("Object:", seg.name)
    print("Instrument:", seg.meta.get("instrument"))
    print("Facility:", seg.meta.get("facility"))
    print("Date obs:", seg.meta.get("date_obs"))
    print("Wave medium:", seg.wave_medium)
    print("Wave frame:", seg.wave_frame)
    print("Pixels used:", int(np.sum(used_masks[0])), "/", len(seg.wave))
    print("Window [A]:", (args.wmin, args.wmax))
    print("R used:", R)
    print("Teff grid used:", teff_grid_fit)
    print("FeH  grid used:", feh_grid_fit)
    print("logg grid used:", logg_grid_fit)
    print("Best-fit:")
    print("  Teff   =", out["teff"])
    print("  [Fe/H] =", out["feh"])
    print("  logg   =", out["logg"])
    print("  RV     =", out["rv_kms"])
    print("  chi2   =", out["chi2"])
    print("  dof    =", out["dof"])
    print("  chi2_red =", out["chi2_red"])
    print("  success  =", out["success"])
    print("  message  =", out["message"])
    print("Continuum coeffs:", coeffs_list[0])
    
    title = (
        "{0}  {1:.0f}-{2:.0f} A  Teff={3:.0f}  [Fe/H]={4:.2f}  "
        "logg={5:.2f}  RV={6:.1f}  chi2_red={7:.2f}".format(
            os.path.basename(args.file),
            args.wmin,
            args.wmax,
            out["teff"],
            out["feh"],
            out["logg"],
            out["rv_kms"],
            out["chi2_red"],
        )
    )

    fig, axes = plot_full_spectrum_fit(
        wave=seg.wave,
        flux=seg.flux,
        err=seg.err,
        model=model_list[0],
        used_mask=used_masks[0],
        excluded_mask=excluded_masks[0],
        title=title,
        line_groups=["balmer", "caii", "hei"],
    )
    plt.show()


if __name__ == "__main__":
    main()
