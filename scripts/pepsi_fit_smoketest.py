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

from Spyctres import Spyctres
from Spyctres.config import load_user_config, get_config_value, resolve_setting
from Spyctres.io import read_spectrum, make_padded_window_segments
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.fitting import (
    fit_phoenix_full_spectrum,
    build_effective_fit_mask,
    reconstruct_phoenix_legendre_models_for_segments,
)
from Spyctres.waveutils import convert_wavelength_medium
from Spyctres.plotting import plot_full_spectrum_fit


WINDOW_PRESETS = {
    "blue_balmer": [
        ("blue_classification", 4285.0, 4490.0),
    ],
    "red_halpha": [
        ("6495 blend", 6485.0, 6505.0),
        ("6545 region", 6535.0, 6555.0),
        ("6561 region", 6551.0, 6571.0),
    ],
    "red_metals": [
        ("Ca I 6439", 6432.0, 6447.0),
        ("6495 blend", 6488.0, 6502.0),
        ("6591 blend", 6585.0, 6597.0),
        ("Li I 6708", 6702.0, 6714.0),
    ],
    "caii_triplet": [
        ("Ca II 8498", 8492.0, 8504.0),
        ("Ca II 8542", 8536.0, 8548.0),
        ("Ca II 8662", 8656.0, 8668.0),
        ("Mg I 8807", 8802.0, 8812.0),
    ],
}


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
            "Quick PHOENIX fit smoke test for a reduced 1D PEPSI .dxt.nor spectrum.\n"
            "This is a generic PEPSI quicklook fitter. It fits selected narrow windows\n"
            "for one PEPSI segment, with optional telluric masking, optional use of\n"
            "header SSBVEL as a barycentric correction, and configurable wavelength-medium\n"
            "hypotheses."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/pepsi_fit_smoketest.py /path/to/pepsib.20230603.014.dxt.nor\n\n"
            "  python scripts/pepsi_fit_smoketest.py \\\n"
            "    --window-preset blue_balmer \\\n"
            "    --wave-hypothesis air \\\n"
            "    --use-telluric-mask \\\n"
            "    /path/to/pepsib.20230603.014.dxt.nor\n\n"
            "  python scripts/pepsi_fit_smoketest.py \\\n"
            "    --window 'LineA' 6432 6447 \\\n"
            "    --window 'LineB' 6488 6502 \\\n"
            "    /path/to/pepsir.20230603.009.dxt.nor\n\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def apply_wave_hypothesis(seg, hypothesis):
    """
    Build a PEPSI segment variant for a wavelength-medium hypothesis.

    hypothesis:
      - unknown    : leave wavelengths unchanged, wave_medium='unknown'
      - air        : leave wavelengths unchanged, wave_medium='air'
      - vacuum     : leave wavelengths unchanged, wave_medium='vacuum'
      - air_to_vac : convert stored wavelengths from air to vacuum
    """
    if hypothesis == "unknown":
        meta = dict(seg.meta)
        meta["wave_medium"] = "unknown"
        return seg.copy(meta=meta, wave_medium="unknown", name=(seg.name or "seg") + "_unknown")

    if hypothesis == "air":
        meta = dict(seg.meta)
        meta["wave_medium"] = "air"
        return seg.copy(meta=meta, wave_medium="air", name=(seg.name or "seg") + "_air")

    if hypothesis == "vacuum":
        meta = dict(seg.meta)
        meta["wave_medium"] = "vacuum"
        return seg.copy(meta=meta, wave_medium="vacuum", name=(seg.name or "seg") + "_vacuum")

    if hypothesis == "air_to_vac":
        wave_new = convert_wavelength_medium(
            np.asarray(seg.wave, dtype=float),
            from_medium="air",
            to_medium="vacuum",
        )
        meta = dict(seg.meta)
        meta["wave_medium"] = "vacuum"
        return seg.copy(
            wave=wave_new,
            meta=meta,
            wave_medium="vacuum",
            name=(seg.name or "seg") + "_air2vac",
        ).sorted()

    raise ValueError("Unknown wave hypothesis: {0}".format(hypothesis))


def choose_auto_window_preset(seg):
    wmin = float(np.nanmin(seg.wave))
    wmax = float(np.nanmax(seg.wave))

    if wmax < 5000.0:
        return "blue_balmer"

    if 6200.0 < wmin < 7000.0 and wmax < 7500.0:
        return "red_halpha"

    if wmin > 7300.0:
        return "caii_triplet"

    raise ValueError(
        "Could not infer an automatic PEPSI window preset from wavelength range "
        "[{0:.1f}, {1:.1f}] A.".format(wmin, wmax)
    )


def build_pepsi_normalized_mask(seg, flux_min=0.2, flux_max=1.1):
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)

    good = np.asarray(seg.mask, dtype=bool)
    good &= np.isfinite(wave) & np.isfinite(flux)
    good &= (flux > float(flux_min)) & (flux < float(flux_max))

    if seg.err is not None:
        err = np.asarray(seg.err, dtype=float)
        good &= np.isfinite(err) & (err > 0)

    return good
    
    
def parse_custom_windows(window_args):
    """
    Parse repeated --window LABEL WMIN WMAX arguments.
    """
    out = []
    for item in window_args:
        if len(item) != 3:
            raise ValueError("Each --window must be: LABEL WMIN WMAX")
        label, wmin, wmax = item
        out.append((str(label), float(wmin), float(wmax)))
    return out


def build_window_segments(seg, window_defs, pad=2.0):
    segments = make_padded_window_segments(
        seg,
        [(wmin, wmax) for _, wmin, wmax in window_defs],
        pad=pad,
        name_prefix="line",
    )
    for seg_i, (label, _wmin, _wmax) in zip(segments, window_defs):
        seg_i.name = label
    return segments


def concat_with_gap(arrays, gap_value=np.nan, dtype=float):
    """
    Concatenate arrays with a single separator element between them.
    """
    arrays = [np.asarray(a, dtype=dtype) for a in arrays]
    if len(arrays) == 0:
        return np.array([], dtype=dtype)

    out = []
    for i, arr in enumerate(arrays):
        out.append(arr)
        if i < len(arrays) - 1:
            out.append(np.array([gap_value], dtype=dtype))
    return np.concatenate(out)


def concat_bool_with_gap(arrays):
    """
    Concatenate boolean arrays with a False separator between windows.
    """
    arrays = [np.asarray(a, dtype=bool) for a in arrays]
    if len(arrays) == 0:
        return np.array([], dtype=bool)

    out = []
    for i, arr in enumerate(arrays):
        out.append(arr)
        if i < len(arrays) - 1:
            out.append(np.array([False], dtype=bool))
    return np.concatenate(out)


def main():
    parser = build_parser()
    parser.add_argument("file", help="Input PEPSI .dxt.nor file")
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Path to local PHOENIXv2 directory. Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file.",
    )
    parser.add_argument(
        "--wave-hypothesis",
        choices=["unknown", "air", "vacuum", "air_to_vac"],
        default="air",
        help="Observed wavelength-medium hypothesis.",
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
        default=20.0,
        help="Margin in Angstrom for native_interp model preparation.",
    )
    parser.add_argument(
        "--window-pad",
        type=float,
        default=2.0,
        help="Padding in Angstrom added around each PEPSI fit window.",
    )
    parser.add_argument(
        "--window-preset",
        choices=["auto", "blue_balmer", "red_halpha", "red_metals", "caii_triplet"],
        default="auto",
        help="Window preset to use for PEPSI quicklook fitting.",
    )
    parser.add_argument(
        "--window",
        nargs=3,
        action="append",
        metavar=("LABEL", "WMIN", "WMAX"),
        help="Custom fit window. Can be given multiple times.",
    )
    parser.add_argument(
        "--use-telluric-mask",
        action="store_true",
        help="Apply built-in telluric mask.",
    )
    parser.add_argument(
        "--telluric-threshold",
        type=float,
        default=0.90,
        help="Telluric mask threshold.",
    )
    parser.add_argument(
        "--use-ssbvel",
        action="store_true",
        help="Use header SSBVEL as a barycentric correction term in km/s.",
    )
    parser.add_argument("--R-override", type=float, default=None, help="Override metadata resolving power R")
    parser.add_argument("--teff-min", type=float, default=4500.0, help="Minimum Teff for explicit PHOENIX grid")
    parser.add_argument("--teff-max", type=float, default=12000.0, help="Maximum Teff for explicit PHOENIX grid")
    parser.add_argument("--feh-min", type=float, default=-1.5, help="Minimum [Fe/H] for explicit PHOENIX grid")
    parser.add_argument("--feh-max", type=float, default=0.5, help="Maximum [Fe/H] for explicit PHOENIX grid")
    parser.add_argument("--logg-min", type=float, default=2.5, help="Minimum logg for explicit PHOENIX grid")
    parser.add_argument("--logg-max", type=float, default=5.5, help="Maximum logg for explicit PHOENIX grid")
    parser.add_argument("--mdeg", type=int, default=1, help="Legendre continuum degree")
    parser.add_argument("--teff0", type=float, default=6500.0, help="Initial Teff")
    parser.add_argument("--feh0", type=float, default=-0.5, help="Initial [Fe/H]")
    parser.add_argument("--logg0", type=float, default=4.0, help="Initial logg")
    parser.add_argument("--rv0", type=float, default=0.0, help="Initial stellar RV in km/s")
    parser.add_argument("--rv-init", choices=["grid", "none"], default="grid", help="RV initialization strategy")
    parser.add_argument("--rv-grid-n", type=int, default=161, help="Number of trial RV points in coarse RV scan")
    parser.add_argument("--cache-path", default=None, help="Interpolator cache path")
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

    if args.forward_model == "interp_observed" and args.wave_hypothesis == "unknown":
        parser.error(
            "--forward-model interp_observed requires a known wavelength medium. "
            "Use --wave-hypothesis air, vacuum, or air_to_vac, or use native_interp."
        )

    seg0 = read_spectrum(args.file, instrument="pepsi")
    seg0 = seg0.copy(mask=build_pepsi_normalized_mask(seg0))
    seg = apply_wave_hypothesis(seg0, args.wave_hypothesis)

    if args.window is not None:
        window_defs = parse_custom_windows(args.window)
        window_preset_used = "custom"
    else:
        if args.window_preset == "auto":
            window_preset_used = choose_auto_window_preset(seg)
        else:
            window_preset_used = args.window_preset
        window_defs = WINDOW_PRESETS[window_preset_used]

    segments = build_window_segments(seg, window_defs, pad=args.window_pad)

    exclude_mask = None
    if args.use_telluric_mask:
        _, telluric_mask = Spyctres.load_telluric_lines(args.telluric_threshold)

        def exclude_mask(wave):
            return np.asarray(telluric_mask(wave)) > 0.5

    used_masks_plot = [
        build_effective_fit_mask(seg_i, exclude_mask=exclude_mask)
        for seg_i in segments
    ]
    if not any(np.any(m) for m in used_masks_plot):
        raise ValueError("No usable points remain after masking.")

    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))

    teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
    teff_grid_req = pick_grid_range(teff_avail, args.teff_min, args.teff_max)
    feh_grid_req = pick_grid_range(feh_avail, args.feh_min, args.feh_max)
    logg_grid_req = pick_grid_range(logg_avail, args.logg_min, args.logg_max)
    teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
        teff_grid_req, feh_grid_req, logg_grid_req
    )

    rv_bary_kms = 0.0
    ssbvel_mps = seg.meta.get("ssbvel_mps")
    if args.use_ssbvel and ssbvel_mps is not None:
        rv_bary_kms = 1.0e-3 * float(ssbvel_mps)

    R = args.R_override if args.R_override is not None else seg.meta.get("resolution_R", None)

    if args.cache_path is None:
        tag = os.path.basename(args.file).replace(".", "_")
        args.cache_path = "/tmp/spyctres_{0}_{1}_cache.npz".format(tag, args.wave_hypothesis)

    out = fit_phoenix_full_spectrum(
        segments,
        phoenix_lib=phoenix_lib,
        p0=(args.teff0, args.feh0, args.logg0, args.rv0),
        exclude_mask=exclude_mask,
        mdeg=args.mdeg,
        rv_bary_kms=rv_bary_kms,
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
        segments=segments,
        phoenix_lib=phoenix_lib,
        fit_result=out,
        exclude_mask=exclude_mask,
        mdeg=args.mdeg,
        rv_bary_kms=rv_bary_kms,
        R=R,
        fwhm_kms=None,
        forward_model=args.forward_model,
        model_margin_A=args.model_margin,
    )

    print("File:", args.file)
    print("Object:", seg.meta.get("object"))
    print("Instrument:", seg.meta.get("instrument"))
    print("Fiber:", seg.meta.get("fiber"))
    print("Cross disperser:", seg.meta.get("cross_disperser"))
    print("Window preset used:", window_preset_used)
    print("Wave medium hypothesis:", args.wave_hypothesis)
    print("Wave medium used:", seg.wave_medium)
    print("Wave frame:", seg.wave_frame)
    print("SSBVEL m/s:", ssbvel_mps)
    print("Barycorr used [km/s]:", rv_bary_kms)
    print("Telluric mask:", bool(args.use_telluric_mask))
    print("R used:", R)
    print("Teff grid used:", teff_grid_fit)
    print("FeH  grid used:", feh_grid_fit)
    print("logg grid used:", logg_grid_fit)
    print("Windows:")
    for label, wmin, wmax in window_defs:
        print(" ", label, (wmin, wmax))
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
    print("Continuum coeffs per window:")
    for seg_i, coeffs_i in zip(segments, coeffs_list):
        print(" ", seg_i.name, coeffs_i)

    wave_plot = concat_with_gap([seg_i.wave for seg_i in segments], gap_value=np.nan, dtype=float)
    flux_plot = concat_with_gap([seg_i.flux for seg_i in segments], gap_value=np.nan, dtype=float)
    err_plot = concat_with_gap([seg_i.err for seg_i in segments], gap_value=np.nan, dtype=float)
    model_plot = concat_with_gap(model_list, gap_value=np.nan, dtype=float)
    used_plot = concat_bool_with_gap(used_masks)
    excl_plot = concat_bool_with_gap(excluded_masks)

    title = (
        "{0}  {1}  Teff={2:.0f}  [Fe/H]={3:.2f}  "
        "logg={4:.2f}  RV={5:.1f}  chi2_red={6:.2f}".format(
            os.path.basename(args.file),
            args.wave_hypothesis,
            out["teff"],
            out["feh"],
            out["logg"],
            out["rv_kms"],
            out["chi2_red"],
        )
    )

    fig, axes = plot_full_spectrum_fit(
        wave=wave_plot,
        flux=flux_plot,
        err=err_plot,
        model=model_plot,
        used_mask=used_plot,
        excluded_mask=excl_plot,
        title=title,
        line_groups=None,
    )
    plt.show()


if __name__ == "__main__":
    main()
