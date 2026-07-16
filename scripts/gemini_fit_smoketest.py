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
from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt

from Spyctres.config import load_user_config, get_config_value, resolve_setting
from Spyctres.defaults import prepare_phoenix_fit_kwargs
from Spyctres.io import read_spectrum
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.results import format_fit_quality_report
from Spyctres.fitting import (
    fit_phoenix_full_spectrum,
    reconstruct_phoenix_legendre_models_for_segments,
)
from Spyctres.preprocessing import (
    broad_telluric_catalog_fallback_mask,
    dilate_boolean_mask,
    telluric_transmission_exclusion_mask,
)
from Spyctres.plotting import plot_full_spectrum_fit
from Spyctres.recipes import pick_grid_range

def build_gemini_fit_mask(
    seg,
    gap_grow_pix=8,
    min_flux_frac=0.25,
    broad_telluric_fallback=None,
):
    """
    Build a stricter fit mask for Gemini/GMOS ASCII spectra.

    This excludes:
    - non-finite points
    - deep near-zero dropout regions, treated as instrumental gaps/bad regions
    - optional broad catalog telluric fallback regions, only when explicitly
      requested for quicklook/product masking

    The bad mask is then dilated to avoid fitting the edges of discontinuities.
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux)

    if seg.err is not None:
        err = np.asarray(seg.err, dtype=float)
        good &= np.isfinite(err) & (err > 0)

    pos = flux[np.isfinite(flux) & (flux > 0)]
    if pos.size > 0:
        median_flux = float(np.nanmedian(pos))
        deep_gap = flux <= float(min_flux_frac) * median_flux
    else:
        deep_gap = np.zeros_like(flux, dtype=bool)

    if broad_telluric_fallback is None:
        telluric = np.zeros_like(wave, dtype=bool)
    else:
        telluric = np.asarray(broad_telluric_fallback(wave)) > 0.5

    bad = (~good) | deep_gap | telluric
    bad = dilate_boolean_mask(bad, n_pix=gap_grow_pix)

    good &= ~bad
    return good


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "Developer/regression quicklook smoke test for a reduced 1D Gemini/GMOS ASCII spectrum.\n"
            "This is a first-pass quicklook fitter for GMOS data.\n"
            "It masks obvious zero-flux gap regions, then fits a selected wavelength "
            "range with a multiplicative polynomial continuum."
        ),
        epilog=(
            "No Gemini spectrum is currently bundled under examples/data/. "
            "Use this smoke test with an explicitly supplied GMOS ASCII file, "
            "for example with --wave-medium air --forward-model native_interp "
            "--wmin 4700 --wmax 8800.\n\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main():
    parser = build_parser()
    parser.add_argument("file", help="Input Gemini/GMOS ASCII spectrum")
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
    parser.add_argument(
        "--auto-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use spectrum metadata/coverage to choose first-pass fit defaults. "
            "Expert CLI values still override the suggestions."
        ),
    )
    parser.add_argument(
        "--defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Search-budget mode used by --auto-defaults.",
    )
    parser.add_argument("--wmin", type=float, default=None, help="Minimum wavelength in Angstrom")
    parser.add_argument("--wmax", type=float, default=None, help="Maximum wavelength in Angstrom")
    parser.add_argument("--clip-left", type=int, default=0, help="Clip this many pixels from the left edge")
    parser.add_argument("--clip-right", type=int, default=0, help="Clip this many pixels from the right edge")
    parser.add_argument("--gap-grow-pix", type=int, default=3, help="Grow bad gap masks by this many pixels")
    parser.add_argument(
        "--min-flux-frac",
        type=float,
        default=0.25,
        help="Reject points with flux below this fraction of the positive-flux median.",
    )
    parser.add_argument(
        "--use-telluric-mask",
        action="store_true",
        help=(
            "Apply the high-resolution telluric transmission-threshold mask as "
            "an explicit fit exclusion."
        ),
    )
    parser.add_argument(
        "--telluric-threshold",
        type=float,
        default=0.90,
        help="Transmission threshold used by --use-telluric-mask.",
    )
    parser.add_argument(
        "--use-broad-telluric-fallback-mask",
        action="store_true",
        help=(
            "Apply coarse broad catalog telluric regions as a Gemini quicklook/"
            "product bad-region mask. This is a fallback, not the preferred "
            "telluric fit mask."
        ),
    )
    parser.add_argument("--R-override", type=float, default=500.0, help="Override resolving power R")
    parser.add_argument("--teff-min", type=float, default=None, help="Minimum Teff for explicit PHOENIX grid")
    parser.add_argument("--teff-max", type=float, default=None, help="Maximum Teff for explicit PHOENIX grid")
    parser.add_argument("--feh-min", type=float, default=None, help="Minimum [Fe/H] for explicit PHOENIX grid")
    parser.add_argument("--feh-max", type=float, default=None, help="Maximum [Fe/H] for explicit PHOENIX grid")
    parser.add_argument("--logg-min", type=float, default=None, help="Minimum logg for explicit PHOENIX grid")
    parser.add_argument("--logg-max", type=float, default=None, help="Maximum logg for explicit PHOENIX grid")
    parser.add_argument("--mdeg", type=int, default=None, help="Legendre continuum degree")
    parser.add_argument("--teff0", type=float, default=None, help="Initial Teff")
    parser.add_argument("--feh0", type=float, default=None, help="Initial [Fe/H]")
    parser.add_argument("--logg0", type=float, default=None, help="Initial logg")
    parser.add_argument("--rv0", type=float, default=None, help="Initial stellar RV in km/s")
    parser.add_argument("--rv-init", choices=["grid", "none"], default="grid", help="RV initialization strategy")
    parser.add_argument("--rv-grid-n", type=int, default=None, help="Number of trial RV points in coarse RV scan")
    parser.add_argument("--cache-path", default="/tmp/spyctres_gemini_fit_cache.npz")
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

    if args.forward_model == "interp_observed" and args.wave_medium == "unknown":
        parser.error(
            "--forward-model interp_observed requires a known --wave-medium. "
            "Use --wave-medium air or vacuum, or use --forward-model native_interp."
        )
    if args.use_telluric_mask and args.use_broad_telluric_fallback_mask:
        parser.error(
            "Use either --use-telluric-mask or --use-broad-telluric-fallback-mask, "
            "not both. The broad catalog mask is only a fallback/quicklook path."
        )

    print("Reading Gemini spectrum...", flush=True)
    seg0 = read_spectrum(args.file, instrument="gemini")

    broad_telluric_fallback = None
    if args.use_broad_telluric_fallback_mask:
        print("Applying broad catalog telluric fallback as Gemini quicklook/product mask...", flush=True)
        broad_telluric_fallback = broad_telluric_catalog_fallback_mask(
            use_case="gemini_quicklook_product_masking",
        )
        meta = broad_telluric_fallback.metadata
        print(
            "Telluric fallback method={0}, frame_type={1}, "
            "fallback_broad_regions_used={2}".format(
                meta.get("method"),
                meta.get("frame_type"),
                meta.get("fallback_broad_regions_used"),
            ),
            flush=True,
        )

    strict_mask = build_gemini_fit_mask(
        seg0,
        gap_grow_pix=args.gap_grow_pix,
        min_flux_frac=args.min_flux_frac,
        broad_telluric_fallback=broad_telluric_fallback,
    )
    seg0_meta = dict(seg0.meta)
    if broad_telluric_fallback is not None:
        seg0_meta["quicklook_telluric_fallback_mask"] = dict(
            broad_telluric_fallback.metadata
        )
    seg0 = seg0.copy(mask=strict_mask, meta=seg0_meta)

    exclude_masks = None
    if args.use_telluric_mask:
        print("Loading high-resolution telluric transmission mask...", flush=True)
        telluric_mask = telluric_transmission_exclusion_mask(
            threshold=args.telluric_threshold
        )
        exclude_masks = [telluric_mask]
        meta = telluric_mask.metadata
        print(
            "Telluric mask method={0}, threshold={1}, model={2}, frame_type={3}, "
            "fallback_broad_regions_used={4}".format(
                meta.get("method"),
                meta.get("threshold"),
                meta.get("model_file"),
                meta.get("frame_type"),
                meta.get("fallback_broad_regions_used"),
            ),
            flush=True,
        )

    if args.wave_medium != "unknown":
        meta = dict(seg0.meta)
        meta["wave_medium"] = args.wave_medium
        seg0 = seg0.copy(meta=meta, wave_medium=args.wave_medium)

    try:
        fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
            seg0,
            auto_defaults=args.auto_defaults,
            defaults_mode=args.defaults_mode,
            science_case="classification",
            fallback_p0=(9500.0, -0.5, 4.0, 0.0),
            fallback_bounds=(
                (7000.0, -1.0, 2.5, -300.0),
                (12000.0, 0.5, 5.5, 300.0),
            ),
            p0_overrides=(args.teff0, args.feh0, args.logg0, args.rv0),
            lower_bound_overrides=(
                args.teff_min,
                args.feh_min,
                args.logg_min,
                None,
            ),
            upper_bound_overrides=(
                args.teff_max,
                args.feh_max,
                args.logg_max,
                None,
            ),
            window=(
                args.wmin,
                args.wmax,
            ) if args.wmin is not None or args.wmax is not None else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mdeg is not None:
        fit_kwargs["mdeg"] = int(args.mdeg)
    if args.rv_grid_n is not None:
        fit_kwargs["rv_grid_n"] = int(args.rv_grid_n)
    fit_kwargs["forward_model"] = args.forward_model
    if args.rv_init == "none":
        fit_kwargs["rv_init"] = None

    if suggestion is not None:
        print("Suggested first-pass fit defaults:", flush=True)
        for reason in suggestion.reasons:
            print("  - {0}".format(reason), flush=True)
        for warning in suggestion.warnings:
            print("  WARNING: {0}".format(warning), flush=True)

    fit_regions = fit_kwargs.get("regions", [(None, None)])
    if len(fit_regions) != 1:
        parser.error("Gemini smoke test expects a single fit window.")
    fit_wmin, fit_wmax = fit_regions[0]

    print("Preparing fit window...", flush=True)
    seg = seg0.window(
        wmin=fit_wmin,
        wmax=fit_wmax,
        clip_left=args.clip_left,
        clip_right=args.clip_right,
        name_suffix="fitwin",
    )

    if not np.any(seg.mask):
        raise ValueError("No usable points remain after Gemini gap masking and windowing.")

    if seg.err is None:
        print("WARNING: no uncertainty column found; chi2_red is only heuristic in this quicklook fit.")

    print("Loading PHOENIX library...", flush=True)
    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))

    print("Selecting PHOENIX grid...", flush=True)
    teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
    bounds = fit_kwargs["bounds"]
    teff_grid_req = pick_grid_range(teff_avail, bounds[0][0], bounds[1][0])
    feh_grid_req = pick_grid_range(feh_avail, bounds[0][1], bounds[1][1])
    logg_grid_req = pick_grid_range(logg_avail, bounds[0][2], bounds[1][2])

    teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
        teff_grid_req, feh_grid_req, logg_grid_req
    )

    R = args.R_override if args.R_override is not None else seg.meta.get("resolution_R", None)
    
    if R is None:
        print("WARNING: no resolving power supplied; model is not instrumentally broadened.")
        
    print("Running PHOENIX fit...", flush=True)
    out = fit_phoenix_full_spectrum(
        [seg],
        phoenix_lib=phoenix_lib,
        p0=fit_kwargs["p0"],
        bounds=fit_kwargs["bounds"],
        exclude_masks=exclude_masks,
        mdeg=fit_kwargs["mdeg"],
        rv_bary_kms=0.0,
        R=R,
        forward_model=fit_kwargs["forward_model"],
        model_margin_A=args.model_margin,
        teff_grid=teff_grid_fit,
        feh_grid=feh_grid_fit,
        logg_grid=logg_grid_fit,
        cache_path=args.cache_path,
        physical_init=fit_kwargs.get("physical_init"),
        coarse_teff_grid=fit_kwargs.get("coarse_teff_grid"),
        coarse_feh_grid=fit_kwargs.get("coarse_feh_grid"),
        coarse_logg_grid=fit_kwargs.get("coarse_logg_grid"),
        coarse_decimate=fit_kwargs.get("coarse_decimate", 12),
        multistart=fit_kwargs.get("multistart", 1),
        rv_init=fit_kwargs.get("rv_init"),
        rv_grid_n=fit_kwargs["rv_grid_n"],
        verbose=args.verbose,
        max_nfev=300,
        progress_callback=lambda message: print(message, flush=True),
    )

    print("Reconstructing best-fit model for plotting...", flush=True)
    model_list, coeffs_list, used_masks, excluded_masks = reconstruct_phoenix_legendre_models_for_segments(
        segments=[seg],
        phoenix_lib=phoenix_lib,
        fit_result=out,
        exclude_masks=exclude_masks,
        mdeg=fit_kwargs["mdeg"],
        rv_bary_kms=0.0,
        R=R,
        fwhm_kms=None,
        forward_model=fit_kwargs["forward_model"],
        model_margin_A=args.model_margin,
    )

    print("File:", args.file)
    print("Object:", seg.name)
    print("Instrument:", seg.meta.get("instrument"))
    print("Facility:", seg.meta.get("facility"))
    print("Origin:", seg.meta.get("origin"))
    print("Wave medium:", seg.wave_medium)
    print("Wave frame:", seg.wave_frame)
    print("Pixels used:", int(np.sum(used_masks[0])), "/", len(seg.wave))
    print("Window [A]:", (fit_wmin, fit_wmax))
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
    print(format_fit_quality_report(out))
    print("Continuum coeffs:", coeffs_list[0])

    title = (
        "{0}  {1:.0f}-{2:.0f} A  Teff={3:.0f}  [Fe/H]={4:.2f}  "
        "logg={5:.2f}  RV={6:.1f}  chi2_red={7:.2f}".format(
            os.path.basename(args.file),
            fit_wmin,
            fit_wmax,
            out["teff"],
            out["feh"],
            out["logg"],
            out["rv_kms"],
            out["chi2_red"],
        )
    )

    print("Building diagnostic plot...", flush=True)
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
