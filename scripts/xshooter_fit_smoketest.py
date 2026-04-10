import os
import sys
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

from Spyctres.io import read_spectrum, SpectrumSegment, make_padded_window_segments
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.waveutils import convert_wavelength_medium
from Spyctres.fitting import (
    fit_phoenix_full_spectrum,
    build_effective_fit_mask,
)
from Spyctres.plotting import plot_full_spectrum_fit
from Spyctres.Spyctres import load_telluric_lines
from Spyctres.config import load_user_config, get_config_value, resolve_setting
from Spyctres.recipes import (
    xshooter_balmer_windows,
    attach_balmer_metadata,
    normalize_segments_sidebands,
    make_balmer_core_exclude_mask,
    fit_phoenix_sideband_symmetric,
    build_plot_models_for_segments,
)


def _durbin_watson(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    denom = np.sum(x * x)
    if denom <= 0:
        return np.nan
    return np.sum(np.diff(x) ** 2) / denom


def _grid_step(grid):
    grid = np.unique(np.asarray(grid, dtype=float))
    if grid.size < 2:
        return np.nan
    d = np.diff(grid)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return np.nan
    return float(np.min(d))


def _edge_distance_in_steps(value, grid):
    grid = np.unique(np.asarray(grid, dtype=float))
    if grid.size == 0:
        return np.nan

    step = _grid_step(grid)
    dmin = abs(float(value) - float(np.min(grid)))
    dmax = abs(float(value) - float(np.max(grid)))
    d = min(dmin, dmax)

    if not np.isfinite(step) or step <= 0:
        return 0.0 if d == 0.0 else np.nan

    return d / step


def _is_on_grid_edge(value, grid, max_edge_distance_steps=0.25):
    dsteps = _edge_distance_in_steps(value, grid)
    return np.isfinite(dsteps) and (dsteps <= float(max_edge_distance_steps))


def _window_rms(wave, z, center, halfwidth=12.0):
    m = np.abs(wave - center) <= float(halfwidth)
    if np.sum(m) < 5:
        return np.nan
    return float(np.sqrt(np.mean(z[m] ** 2)))


def _window_annulus_rms(wave, z, center, inner_halfwidth=12.0, outer_halfwidth=30.0):
    d = np.abs(wave - center)
    m = (d >= float(inner_halfwidth)) & (d <= float(outer_halfwidth))
    if np.sum(m) < 5:
        return np.nan
    return float(np.sqrt(np.mean(z[m] ** 2)))


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


def _dedupe_p0_list(starts):
    out = []
    seen = set()
    for s in starts:
        s = tuple(float(x) for x in s)
        key = tuple(round(x, 6) for x in s)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _clamp_start_to_grids(p0, teff_grid=None, feh_grid=None, logg_grid=None):
    teff, feh, logg, rv = map(float, p0)

    if teff_grid is not None and len(teff_grid) > 0:
        teff = float(np.clip(teff, np.min(teff_grid), np.max(teff_grid)))
    if feh_grid is not None and len(feh_grid) > 0:
        feh = float(np.clip(feh, np.min(feh_grid), np.max(feh_grid)))
    if logg_grid is not None and len(logg_grid) > 0:
        logg = float(np.clip(logg, np.min(logg_grid), np.max(logg_grid)))

    return (teff, feh, logg, rv)


def build_start_list(args, teff_grid=None, feh_grid=None, logg_grid=None):
    starts = [(args.teff0, args.feh0, args.logg0, args.rv0)]

    if args.multistart:
        if args.balmer_only:
            starts.extend([
                (9000.0,  -1.0, 4.0, args.rv0),
                (9800.0,  -1.0, 3.5, args.rv0),
                (10600.0, -1.0, 3.0, args.rv0),
                (11600.0, -1.0, 3.0, args.rv0),
                (11600.0, -1.5, 3.5, args.rv0),
            ])
        else:
            starts.extend([
                (4500.0, -0.5, 4.5, args.rv0),
                (6500.0, -0.5, 4.0, args.rv0),
                (8500.0, -1.0, 3.5, args.rv0),
            ])

    starts = [
        _clamp_start_to_grids(s, teff_grid=teff_grid, feh_grid=feh_grid, logg_grid=logg_grid)
        for s in starts
    ]
    return _dedupe_p0_list(starts)


def _flag_present(argv, dest):
    """
    Return True if the corresponding CLI flag was explicitly given.

    This lets preset values behave like defaults: explicit CLI flags win.
    """
    flag = "--" + str(dest).replace("_", "-")
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def apply_named_preset(args, argv):
    """
    Apply a named benchmark preset, while keeping explicit CLI flags in control.
    """
    preset = getattr(args, "preset", None)
    if preset is None:
        return args

    if preset == "xshooter_uvb_notebook":
        preset_values = {
            "balmer_only": True,
            "norm_mode": "sideband",
            "forward_model": "native_interp",
            "window_mode": "notebook",
            "core_mask": 12.0,
            "window_pad": 20.0,
            "R_override": 5100.0,
            "teff_min": 9500.0,
            "teff_max": 12500.0,
            "feh_min": -0.5,
            "feh_max": 0.5,
            "logg_min": 3.0,
            "logg_max": 6.0,
            "teff0": 9800.0,
            "feh0": -0.5,
            "logg0": 3.0,
            "rv0": -5.0,
            "multistart": True,
        }
    else:
        raise ValueError("Unknown preset: {0}".format(preset))

    for dest, value in preset_values.items():
        if not _flag_present(argv, dest):
            setattr(args, dest, value)

    return args
    
    
def _edge_count(edge_flags):
    return int(sum(bool(v) for v in edge_flags.values()))


def _installed_axis_limit(value, grid, max_edge_distance_steps=0.25):
    """
    Diagnose whether a fitted value is effectively on the minimum or maximum
    of the actually installed model axis.
    """
    grid = np.unique(np.asarray(grid, dtype=float))
    if grid.size == 0:
        return {
            "is_edge": False,
            "side": None,
            "limit_value": np.nan,
            "distance_steps": np.nan,
        }

    lo = float(np.min(grid))
    hi = float(np.max(grid))
    dsteps = _edge_distance_in_steps(value, grid)

    if not np.isfinite(dsteps) or dsteps > float(max_edge_distance_steps):
        return {
            "is_edge": False,
            "side": None,
            "limit_value": np.nan,
            "distance_steps": dsteps,
        }

    if abs(float(value) - hi) <= abs(float(value) - lo):
        side = "upper"
        limit_value = hi
    else:
        side = "lower"
        limit_value = lo

    return {
        "is_edge": True,
        "side": side,
        "limit_value": limit_value,
        "distance_steps": dsteps,
    }


def evaluate_installed_limits(result, teff_avail=None, feh_avail=None, logg_avail=None):
    out = {}

    if teff_avail is not None:
        out["teff"] = _installed_axis_limit(result["teff"], teff_avail)
    else:
        out["teff"] = None

    if feh_avail is not None:
        out["feh"] = _installed_axis_limit(result["feh"], feh_avail)
    else:
        out["feh"] = None

    if logg_avail is not None:
        out["logg"] = _installed_axis_limit(result["logg"], logg_avail)
    else:
        out["logg"] = None

    summary = []
    if out["teff"] is not None and out["teff"]["is_edge"]:
        if out["teff"]["side"] == "upper":
            summary.append("Teff >= {0:g} K (installed-grid ceiling)".format(out["teff"]["limit_value"]))
        else:
            summary.append("Teff <= {0:g} K (installed-grid floor)".format(out["teff"]["limit_value"]))

    if out["feh"] is not None and out["feh"]["is_edge"]:
        if out["feh"]["side"] == "upper":
            summary.append("[Fe/H] >= {0:g} (installed-grid ceiling)".format(out["feh"]["limit_value"]))
        else:
            summary.append("[Fe/H] <= {0:g} (installed-grid floor)".format(out["feh"]["limit_value"]))

    if out["logg"] is not None and out["logg"]["is_edge"]:
        if out["logg"]["side"] == "upper":
            summary.append("logg >= {0:g} (installed-grid ceiling)".format(out["logg"]["limit_value"]))
        else:
            summary.append("logg <= {0:g} (installed-grid floor)".format(out["logg"]["limit_value"]))

    out["summary"] = summary
    return out


def select_best_record(records):
    return min(
        records,
        key=lambda rec: (
            float(rec["result"]["chi2_red"]),
            _edge_count(rec["quality"]["edge_flags"]),
            abs(float(rec["quality"]["z_median"])),
        ),
    )


def evaluate_fit_quality(seg, model_corr_full, used_mask, result, phoenix_lib):
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = np.asarray(seg.err, dtype=float)
    used_mask = np.asarray(used_mask, dtype=bool)

    w = wave[used_mask]
    f = flux[used_mask]
    e = err[used_mask]
    m = np.asarray(model_corr_full, dtype=float)[used_mask]

    z = (f - m) / e

    dw = _durbin_watson(z)
    z_med = float(np.nanmedian(z))
    z_std = float(np.nanstd(z))

    # Residual slope in sigma per 1000 Angstrom
    xk = (w - np.nanmean(w)) / 1000.0
    if np.sum(np.isfinite(xk) & np.isfinite(z)) >= 2:
        slope = float(np.polyfit(xk, z, 1)[0])
    else:
        slope = np.nan

    centers_vac = {
        "Hdelta": 4101.74,
        "Hgamma": 4340.47,
        "Hbeta": 4861.33,
    }

    wave_medium = str(seg.wave_medium).lower()
    balmer_rms = {}

    for label, center_vac in centers_vac.items():
        if wave_medium in ("air", "vacuum"):
            center_use = float(
                convert_wavelength_medium(
                    np.array([center_vac], dtype=float),
                    from_medium="vacuum",
                    to_medium=wave_medium,
                )[0]
            )
        else:
            center_use = float(center_vac)

        balmer_rms[label] = _window_annulus_rms(
            w, z, center_use,
            inner_halfwidth=12.0,
            outer_halfwidth=30.0,
        )

    tg, zg, gg = phoenix_lib._grid
    edge_flags = {
        "teff_edge": _is_on_grid_edge(result["teff"], tg),
        "feh_edge": _is_on_grid_edge(result["feh"], zg),
        "logg_edge": _is_on_grid_edge(result["logg"], gg),
    }

    edge_distance_steps = {
        "teff_edge_steps": _edge_distance_in_steps(result["teff"], tg),
        "feh_edge_steps": _edge_distance_in_steps(result["feh"], zg),
        "logg_edge_steps": _edge_distance_in_steps(result["logg"], gg),
    }

    verdict = "PASS"
    reasons = []

    # Heuristic thresholds for smoke-test diagnostics
    if result["chi2_red"] > 15.0:
        verdict = "FAIL"
        reasons.append("chi2_red>15")
    elif result["chi2_red"] > 5.0:
        if verdict != "FAIL":
            verdict = "WARN"
        reasons.append("chi2_red>5")

    if any(edge_flags.values()):
        if verdict != "FAIL":
            verdict = "WARN"
        reasons.append("parameter_on_grid_edge")

    if np.isfinite(dw) and (dw < 1.2 or dw > 2.8):
        if verdict != "FAIL":
            verdict = "WARN"
        reasons.append("residual_autocorrelation")

    if np.isfinite(slope) and abs(slope) > 1.0:
        if verdict != "FAIL":
            verdict = "WARN"
        reasons.append("residual_slope")

    finite_balmer = [v for v in balmer_rms.values() if np.isfinite(v)]
    if len(finite_balmer) > 0 and max(finite_balmer) > 4.0:
        if verdict != "FAIL":
            verdict = "WARN"
        reasons.append("balmer_window_rms>4")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "edge_distance_steps": edge_distance_steps,
        "chi2_red": float(result["chi2_red"]),
        "dw": float(dw) if np.isfinite(dw) else np.nan,
        "z_median": z_med,
        "z_std": z_std,
        "residual_slope_per_1000A": slope,
        "balmer_rms": balmer_rms,
        "edge_flags": edge_flags,
    }


def evaluate_region_quality(segments, model_corr_list, used_masks):
    """
    Per-window reduced chi2 diagnostics for Balmer-only mode.

    For each segment, compute reduced chi2 on the actually used pixels.
    This is more meaningful than comparing against stale hard-coded
    continuum regions from an earlier windowing scheme.
    """
    per_window = {}

    labels = ["Hδ", "Hγ", "Hβ"]
    for i, (seg, model_corr_full, used_mask) in enumerate(zip(segments, model_corr_list, used_masks)):
        wave = np.asarray(seg.wave, dtype=float)
        flux = np.asarray(seg.flux, dtype=float)
        err = np.asarray(seg.err, dtype=float)
        model = np.asarray(model_corr_full, dtype=float)
        used_mask = np.asarray(used_mask, dtype=bool)

        m = used_mask & np.isfinite(model)
        n = int(np.sum(m))

        if n <= 4:
            chi2_red = np.nan
        else:
            chi2 = float(np.sum(((flux[m] - model[m]) / err[m]) ** 2))
            chi2_red = chi2 / (n - 4)

        label = labels[i] if i < len(labels) else "win{0}".format(i)
        per_window[label] = {
            "chi2_red": chi2_red,
            "n": n,
            "wave_min": float(np.min(wave[m])) if np.any(m) else np.nan,
            "wave_max": float(np.max(wave[m])) if np.any(m) else np.nan,
        }

    finite_vals = [v["chi2_red"] for v in per_window.values() if np.isfinite(v["chi2_red"])]

    verdict = "PASS"
    reasons = []
    if len(finite_vals) > 0 and max(finite_vals) > 5.0:
        verdict = "WARN"
        reasons.append("line_windows_still_poor")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "per_window": per_window,
    }


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "PHOENIX full-spectrum fitting smoke test for reduced X-SHOOTER 1D spectra.\n"
            "Use this to exercise the generic package fitter on a real X-SHOOTER file.\n"
            "For the validated Balmer-wing benchmark, use --balmer-only --forward-model native_interp "
            "--window-mode notebook --core-mask 12."
        ),
        epilog=(
            "Examples:\n"
            "  export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2\n\n"
            "  Validated Balmer-wing benchmark:\n"
            "    python scripts/xshooter_fit_smoketest.py \\\n"
            "      --balmer-only \\\n"
            "      --forward-model native_interp \\\n"
            "      --window-mode notebook \\\n"
            "      --core-mask 12 \\\n"
            "      path/to/xshooter_uvb.fits\n\n"
            "  Sideband-normalized Balmer fit:\n"
            "    python scripts/xshooter_fit_smoketest.py \\\n"
            "      --balmer-only \\\n"
            "      --norm-mode sideband \\\n"
            "      --forward-model native_interp \\\n"
            "      --window-mode notebook \\\n"
            "      --core-mask 12 \\\n"
            "      path/to/xshooter_uvb.fits\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main():
    parser = build_parser()
    parser.add_argument("file", help="Input X-SHOOTER FITS file")
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Path to local PHOENIXv2 directory. Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file.",
    )
    parser.add_argument(
        "--multistart",
        action="store_true",
        help="Try several starting guesses and keep the best fit",
    )
    parser.add_argument(
        "--norm-mode",
        choices=["poly", "sideband"],
        default="poly",
        help="Continuum handling for Balmer-only mode: polynomial or sideband normalization",
    )
    parser.add_argument(
        "--sideband-width",
        type=float,
        default=10.0,
        help="Width in Angstrom of each fallback edge sideband when explicit per-line sidebands are not defined",
    )
    parser.add_argument(
        "--sideband-order",
        type=int,
        default=1,
        help="Polynomial order for sideband continuum normalization",
    )
    parser.add_argument(
        "--sideband-poly-order",
        type=int,
        default=1,
        help="Polynomial order for multiplicative wing correction after sideband normalization",
    )
    parser.add_argument(
        "--R-override",
        type=float,
        default=None,
        help="Override metadata resolving power R for the fit",
    )
    parser.add_argument(
        "--preset",
        choices=["xshooter_uvb_notebook"],
        default=None,
        help=(
            "Apply a named benchmark preset. "
            "'xshooter_uvb_notebook' reproduces the notebook-faithful UVB "
            "sideband benchmark settings. Explicit CLI flags override preset values."
        ),
    )
    parser.add_argument(
        "--forward-model",
        choices=["interp_observed", "native_interp"],
        default="interp_observed",
        help="Forward-model path. 'native_interp' is the validated wavelength-space reference; 'interp_observed' is the legacy observed-grid path.",
    )
    parser.add_argument(
        "--model-margin",
        type=float,
        default=200.0,
        help="Margin in Angstrom for native_interp model preparation",
    )
    parser.add_argument("--wmin", type=float, default=3980.0, help="Minimum wavelength in Angstrom")
    parser.add_argument("--wmax", type=float, default=5500.0, help="Maximum wavelength in Angstrom")
    parser.add_argument("--teff-min", type=float, default=9000.0,
                        help="Minimum Teff for explicit PHOENIX grid in Balmer-only mode")
    parser.add_argument("--teff-max", type=float, default=11600.0,
                        help="Maximum Teff for explicit PHOENIX grid in Balmer-only mode")
    parser.add_argument("--feh-min", type=float, default=-2.0,
                        help="Minimum [Fe/H] for explicit PHOENIX grid in Balmer-only mode")
    parser.add_argument("--feh-max", type=float, default=0.0,
                        help="Maximum [Fe/H] for explicit PHOENIX grid in Balmer-only mode")
    parser.add_argument("--logg-min", type=float, default=2.0,
                        help="Minimum logg for explicit PHOENIX grid in Balmer-only mode")
    parser.add_argument("--logg-max", type=float, default=4.5,
                        help="Maximum logg for explicit PHOENIX grid in Balmer-only mode")
    parser.add_argument("--clip-left", type=int, default=0, help="Clip this many pixels from the left edge")
    parser.add_argument("--clip-right", type=int, default=0, help="Clip this many pixels from the right edge")
    parser.add_argument("--balmer-only", action="store_true",
                        help="Fit only Balmer windows instead of the full selected wavelength range")
    parser.add_argument("--core-mask", type=float, default=3.0,
                        help="Half-width in Angstrom to mask around Balmer line centers in --balmer-only mode. The validated notebook-style benchmark uses 12.")
    parser.add_argument("--window-pad", type=float, default=5.0,
                        help="Padding in Angstrom added on each side of Balmer fit windows")
    parser.add_argument(
        "--window-mode",
        choices=["current", "notebook"],
        default="current",
        help="Use the current narrow Balmer windows or the broader notebook-style windows. The validated benchmark uses 'notebook'.",
    )
    parser.add_argument("--mdeg", type=int, default=2, help="Legendre continuum degree")
    parser.add_argument("--teff0", type=float, default=5000.0, help="Initial Teff")
    parser.add_argument("--feh0", type=float, default=-0.5, help="Initial [Fe/H]")
    parser.add_argument("--logg0", type=float, default=4.0, help="Initial logg")
    parser.add_argument("--rv0", type=float, default=0.0, help="Initial stellar RV in km/s")
    parser.add_argument("--rv-init", choices=["grid", "none"], default="grid", help="RV initialization strategy")
    parser.add_argument("--rv-grid-n", type=int, default=81, help="Number of trial RV points in coarse RV scan")
    parser.add_argument("--telluric-threshold", type=float, default=0.90, help="Telluric mask threshold")
    parser.add_argument("--use-telluric-mask", action="store_true", help="Apply built-in telluric mask")
    parser.add_argument("--use-barycorr", action="store_true", help="Pass header barycentric correction into fit")
    parser.add_argument("--cache-path", default="/tmp/spyctres_xshooter_fit_cache.npz")
    parser.add_argument("--verbose", type=int, default=1)
    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    args = apply_named_preset(args, raw_argv)
    
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

    if args.norm_mode == "sideband" and not args.balmer_only:
        parser.error("--norm-mode sideband currently requires --balmer-only.")

    if args.phoenix_dir is None:
        parser.error(
            "No PHOENIX directory supplied. Set --phoenix-dir, SPYCTRES_PHOENIX_DIR, "
            "or [paths].phoenix_dir in ~/.config/spyctres/config.toml."
        )

    if not os.path.isdir(args.phoenix_dir):
        parser.error("PHOENIX directory not found: {0}".format(args.phoenix_dir))

    if args.wmax <= args.wmin:
        parser.error("--wmax must be greater than --wmin.")

    seg0 = read_spectrum(args.file, instrument="xshooter")
    arm = str(seg0.meta.get("arm", "")).strip().upper()
    if arm and arm != "UVB":
        parser.error(
            "This script is currently configured for X-SHOOTER UVB fitting, but the reader reported arm={0!r}.".format(arm)
        )

    seg_clip = seg0.window(
        wmin=args.wmin,
        wmax=args.wmax,
        clip_left=args.clip_left,
        clip_right=args.clip_right,
        name_suffix="fitwin",
    )

    balmer_windows = xshooter_balmer_windows(args.window_mode)

    if args.balmer_only:
        segments = make_padded_window_segments(
            seg_clip,
            [(wmin, wmax) for _, wmin, wmax in balmer_windows],
            pad=args.window_pad,
            name_prefix="balmer",
        )

        # Rename the generic padded-window segments to the physical Balmer labels
        # expected by the recipe metadata helper.
        for seg_i, (label, _wmin, _wmax) in zip(segments, balmer_windows):
            seg_i.name = label
    else:
        segments = [seg_clip]

    if args.balmer_only:
        if args.window_mode == "notebook":
            attach_balmer_metadata(segments)
        else:
            current_cont_windows = {label: None for label, _, _ in balmer_windows}
            attach_balmer_metadata(segments, cont_windows=current_cont_windows)

    norm_info = None
    fit_mdeg = args.mdeg

    if args.balmer_only and args.norm_mode == "sideband":
        segments, norm_info = normalize_segments_sidebands(
            segments,
            sideband_width=args.sideband_width,
            sideband_order=args.sideband_order,
        )

    seg_ref = segments[0]
    exclude_mask_list = []

    if args.use_telluric_mask:
        _, telluric_mask = load_telluric_lines(args.telluric_threshold)
        exclude_mask_list.append(telluric_mask)

    if args.balmer_only:
        core_mask = make_balmer_core_exclude_mask(
            core_halfwidth=args.core_mask,
            wave_medium=seg_ref.wave_medium,
        )
        exclude_mask_list.append(core_mask)

    if len(exclude_mask_list) == 0:
        exclude_mask = None
    else:
        def exclude_mask(wave):
            wave = np.asarray(wave, dtype=float)
            m = np.zeros_like(wave, dtype=bool)
            for fn in exclude_mask_list:
                m |= (np.asarray(fn(wave)) > 0.5)
            return m

    used_masks_plot = [
        build_effective_fit_mask(seg, exclude_mask=exclude_mask)
        for seg in segments
    ]

    if not any(np.any(m) for m in used_masks_plot):
        raise ValueError("No usable points remain after masking.")

    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))

    if args.balmer_only:
        teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
        print("Installed Teff range:", float(np.min(teff_avail)), float(np.max(teff_avail)))
        print("Installed FeH  range:", float(np.min(feh_avail)), float(np.max(feh_avail)))
        print("Installed logg range:", float(np.min(logg_avail)), float(np.max(logg_avail)))

        teff_grid_req = pick_grid_range(teff_avail, args.teff_min, args.teff_max)
        feh_grid_req = pick_grid_range(feh_avail, args.feh_min, args.feh_max)
        logg_grid_req = pick_grid_range(logg_avail, args.logg_min, args.logg_max)

        teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
            teff_grid_req, feh_grid_req, logg_grid_req
        )
    else:
        teff_avail = None
        feh_avail = None
        logg_avail = None
        teff_grid_req = None
        feh_grid_req = None
        logg_grid_req = None
        teff_grid_fit = None
        feh_grid_fit = None
        logg_grid_fit = None

    start_list = build_start_list(
        args,
        teff_grid=teff_grid_fit,
        feh_grid=feh_grid_fit,
        logg_grid=logg_grid_fit,
    )

    rv_bary_kms = 0.0
    if args.use_barycorr:
        rv_bary_kms = float(seg_ref.meta.get("barycorr_kms") or 0.0)

    R = args.R_override if args.R_override is not None else seg_ref.meta.get("resolution_R", None)

    records = []

    for i, p0_i in enumerate(start_list, start=1):
        print("\n=== START {0}/{1}: p0={2} ===".format(i, len(start_list), p0_i))

        if args.norm_mode == "sideband":
            result_i = fit_phoenix_sideband_symmetric(
                segments=segments,
                phoenix_lib=phoenix_lib,
                p0=p0_i,
                exclude_mask=exclude_mask,
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
                max_nfev=200,
                sideband_width=args.sideband_width,
                sideband_order=args.sideband_order,
                sideband_poly_order=args.sideband_poly_order,
            )
        else:
            result_i = fit_phoenix_full_spectrum(
                segments=segments,
                phoenix_lib=phoenix_lib,
                p0=p0_i,
                exclude_mask=exclude_mask,
                mdeg=fit_mdeg,
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
                max_nfev=200,
            )

        model_corr_list_i, coeffs_list_i, used_masks_i, excluded_masks_i = build_plot_models_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            fit_result=result_i,
            exclude_mask=exclude_mask,
            mdeg=fit_mdeg,
            rv_bary_kms=rv_bary_kms,
            R=R,
            fwhm_kms=None,
            norm_mode=args.norm_mode,
            sideband_width=args.sideband_width,
            sideband_order=args.sideband_order,
            sideband_poly_order=args.sideband_poly_order,
            forward_model=args.forward_model,
            model_margin_A=args.model_margin,
        )

        wave_cat = np.concatenate([np.asarray(s.wave, dtype=float) for s in segments])
        flux_cat = np.concatenate([np.asarray(s.flux, dtype=float) for s in segments])
        err_cat = np.concatenate([np.asarray(s.err, dtype=float) for s in segments])
        model_cat = np.concatenate(model_corr_list_i)
        used_cat = np.concatenate(used_masks_i)

        seg_metrics = SpectrumSegment(
            wave=wave_cat,
            flux=flux_cat,
            err=err_cat,
            mask=used_cat,
            meta=dict(seg_ref.meta),
            wave_medium=seg_ref.wave_medium,
            wave_frame=seg_ref.wave_frame,
            name="metrics_concat",
        )

        quality_i = evaluate_fit_quality(
            seg=seg_metrics,
            model_corr_full=model_cat,
            used_mask=used_cat,
            result=result_i,
            phoenix_lib=phoenix_lib,
        )

        region_quality_i = evaluate_region_quality(
            segments=segments,
            model_corr_list=model_corr_list_i,
            used_masks=used_masks_i,
        )

        records.append({
            "p0": p0_i,
            "result": result_i,
            "model_corr_list": model_corr_list_i,
            "coeffs_list": coeffs_list_i,
            "used_masks": used_masks_i,
            "excluded_masks": excluded_masks_i,
            "quality": quality_i,
            "region_quality": region_quality_i,
        })

        print(
            "  -> chi2_red={0:.6f}  teff={1:.1f}  feh={2:.3f}  logg={3:.3f}  rv={4:.2f}  edge_flags={5}".format(
                result_i["chi2_red"],
                result_i["teff"],
                result_i["feh"],
                result_i["logg"],
                result_i["rv_kms"],
                quality_i["edge_flags"],
            )
        )

    best_rec = select_best_record(records)

    selected_p0 = best_rec["p0"]
    result = best_rec["result"]
    model_corr_list = best_rec["model_corr_list"]
    coeffs_list = best_rec["coeffs_list"]
    used_masks_plot = best_rec["used_masks"]
    excluded_masks_plot = best_rec["excluded_masks"]
    quality = best_rec["quality"]
    region_quality = best_rec["region_quality"]
    installed_limits = evaluate_installed_limits(
        result=result,
        teff_avail=teff_avail,
        feh_avail=feh_avail,
        logg_avail=logg_avail,
    )

    print("File:", args.file)
    print("Preset:", args.preset)
    print("Object:", seg_ref.meta.get("object"))
    print("Arm:", seg_ref.meta.get("arm"))
    print("Pixels used:", int(sum(np.sum(m) for m in used_masks_plot)), "/", int(sum(len(s.wave) for s in segments)))
    print("Window [A]:", (args.wmin, args.wmax))
    print("Telluric mask:", bool(args.use_telluric_mask))
    print("Barycorr used [km/s]:", rv_bary_kms)
    print("R used:", R)
    print("R override:", args.R_override)
    print("Teff grid requested:", teff_grid_req)
    print("FeH  grid requested:", feh_grid_req)
    print("logg grid requested:", logg_grid_req)
    print("Teff grid used:", teff_grid_fit)
    print("FeH  grid used:", feh_grid_fit)
    print("logg grid used:", logg_grid_fit)
    print("Normalization mode:", args.norm_mode)

    if norm_info is not None:
        for label, info in zip([x[0] for x in balmer_windows], norm_info):
            print(
                "  ", label,
                "sidebands N =", info["n_sideband"],
                "fit=({0:.1f},{1:.1f})".format(info["fit_lo"], info["fit_hi"]),
                "mode =", info["mode"],
            )

    if args.norm_mode == "sideband":
        print("  sideband_order:", args.sideband_order)
        print("  sideband_poly_order:", args.sideband_poly_order)

    print("Best-fit:")
    print("  Teff   =", result["teff"])
    print("  [Fe/H] =", result["feh"])
    print("  logg   =", result["logg"])
    print("  RV     =", result["rv_kms"])
    print("  chi2   =", result["chi2"])
    print("  dof    =", result["dof"])
    print("  chi2_red =", result["chi2_red"])
    print("  success  =", result["success"])
    print("  message  =", result["message"])
    print("Continuum coeffs per segment:", coeffs_list)
    print("Quality verdict:", quality["verdict"])
    print("Quality reasons:", quality["reasons"])
    print("Region quality verdict:", region_quality["verdict"])
    print("Region quality reasons:", region_quality["reasons"])
    print("  Per-window chi2_red:")
    for k, v in region_quality["per_window"].items():
        print("   ", k, "chi2_red =", v["chi2_red"], "N =", v["n"],
              "range=({0:.1f},{1:.1f})".format(v["wave_min"], v["wave_max"]))
    print("Installed-grid limits:", installed_limits["summary"])
    print("  Durbin-Watson:", quality["dw"])
    print("  Residual median:", quality["z_median"])
    print("  Residual std:", quality["z_std"])
    print("  Residual slope [sigma / 1000 A]:", quality["residual_slope_per_1000A"])
    print("  Edge flags:", quality["edge_flags"])
    print("  Edge distance [grid steps]:", quality["edge_distance_steps"])
    print("Start list tried:", start_list)
    print("Selected start:", selected_p0)
    print("Window mode:", args.window_mode)
    print("  Balmer-window RMS:", quality["balmer_rms"])

    title = "{0}  {1:.0f}-{2:.0f} A  Teff={3:.0f}  [Fe/H]={4:.2f}  logg={5:.2f}  RV={6:.1f}  chi2_red={7:.2f}".format(
        os.path.basename(args.file),
        args.wmin,
        args.wmax,
        result["teff"],
        result["feh"],
        result["logg"],
        result["rv_kms"],
        result["chi2_red"],
    )

    if args.balmer_only:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()

        for ax, (label, _, _), seg_i, model_i, excl_i in zip(
            axes, balmer_windows, segments, model_corr_list, excluded_masks_plot
        ):
            ax.plot(seg_i.wave, seg_i.flux, lw=1.0, label="Data")
            ax.plot(seg_i.wave, model_i, lw=1.0, label="Model")

            if np.any(excl_i):
                y0, y1 = ax.get_ylim()
                ax.fill_between(
                    seg_i.wave, y0, y1,
                    where=excl_i,
                    alpha=0.15,
                    step=None,
                )
                ax.set_ylim(y0, y1)

            ax.set_title(label)
            ax.set_xlabel("Wavelength (Å)")
            ax.set_ylabel("Flux")
            ax.legend(loc="best")

        if len(axes) > len(segments):
            for ax in axes[len(segments):]:
                ax.axis("off")

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    else:
        fig, axes = plot_full_spectrum_fit(
            wave=segments[0].wave,
            flux=segments[0].flux,
            err=segments[0].err,
            model=model_corr_list[0],
            used_mask=used_masks_plot[0],
            excluded_mask=excluded_masks_plot[0],
            title=title,
            line_groups=["balmer", "caii", "hei"],
        )
        plt.show()


if __name__ == "__main__":
    main()
