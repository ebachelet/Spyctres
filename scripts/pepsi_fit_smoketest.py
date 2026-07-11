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
from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from Spyctres import Spyctres
from Spyctres.results import format_fit_quality_report
from Spyctres.config import resolve_phoenix_dir
from Spyctres.io import (
    read_spectrum,
    make_padded_window_segments,
    pepsi_ssbvel_correction_kms,
    SpectrumCollection,
)
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.fitting import (
    fit_phoenix_full_spectrum,
    build_effective_fit_mask,
    reconstruct_phoenix_legendre_models_for_segments,
)
from Spyctres.plotting import plot_full_spectrum_fit
from Spyctres.recipes import (
    apply_pepsi_wave_hypothesis,
    build_pepsi_normalized_mask,
    build_pepsi_legacy_segments,
    make_pepsi_legacy_cache_support_segments,
    ensure_phoenix_native_interpolator_for_segments,
    pick_grid_range,
    evaluate_pepsi_legacy_max_models,
    pepsi_legacy_max_likelihood_terms,
)


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


def build_parser():
    return argparse.ArgumentParser(
        description=(
            "PEPSI PHOENIX fitting smoke test.\n"
            "\n"
            "This script has two roles:\n"
            "  1. quicklook: a generic one-file PEPSI full-spectrum/window fit;\n"
            "  2. legacy_max: a PEPSI line-window comparison mode that mimics the\n"
            "     local model/max(model) normalization used in PEPSI line-window tests.\n"
            "\n"
            "For normal development, prefer one of the named presets. Expert flags\n"
            "remain available for diagnostics and sensitivity tests."
        ),
        epilog=(
            "Recommended examples:\n"
            "\n"
            "  Fast PEPSI red-window regression test:\n"
            "    python scripts/pepsi_fit_smoketest.py \\\n"
            "      --preset pepsi_legacy_red_fast \\\n"
            "      examples/data/pepsir.20230603.009.dxt.nor \\\n"
            "      examples/data/pepsir.20230603.010.dxt.nor\n"
            "\n"
            "  Slower full-grid PEPSI red-window diagnostic:\n"
            "    python scripts/pepsi_fit_smoketest.py \\\n"
            "      --preset pepsi_legacy_red_full \\\n"
            "      examples/data/pepsir.20230603.009.dxt.nor \\\n"
            "      examples/data/pepsir.20230603.010.dxt.nor\n"
            "\n"
            "  One-file quicklook fit:\n"
            "    python scripts/pepsi_fit_smoketest.py \\\n"
            "      --preset pepsi_quicklook \\\n"
            "      examples/data/pepsir.20230603.010.dxt.nor\n"
            "\n"
            "Configuration:\n"
            "  export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2\n"
            "\n"
            "or use ~/.config/spyctres/config.toml:\n"
            "  [paths]\n"
            "  phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


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


def parse_custom_windows(window_args):
    """Parse repeated --window LABEL WMIN WMAX arguments."""
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


def _flag_present(argv, dest):
    """
    Return True if the corresponding flag was explicitly given.

    This lets preset values behave like defaults: explicit flags win.
    """
    flag = "--" + str(dest).replace("_", "-")
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def apply_named_preset(args, argv):
    """Apply a named PEPSI fitting preset, while keeping explicit flags in control."""
    preset = getattr(args, "preset", None)
    if preset is None:
        return args

    if preset == "pepsi_legacy_red_fast":
        preset_values = {
            "mode": "legacy_max",
            "fast": True,
            "wave_hypothesis": "air",
            "forward_model": "native_interp",
            "use_ssbvel": True,
            "use_telluric_mask": True,
            "teff0": 5500.0,
            "feh0": -1.0,
            "logg0": 3.0,
            "rv0": -50.0,
        }
    elif preset == "pepsi_legacy_red_full":
        preset_values = {
            "mode": "legacy_max",
            "fast": False,
            "wave_hypothesis": "air",
            "forward_model": "native_interp",
            "use_ssbvel": True,
            "use_telluric_mask": True,
            "teff0": 5500.0,
            "feh0": -1.0,
            "logg0": 3.0,
            "rv0": -50.0,
        }
    elif preset == "pepsi_quicklook":
        preset_values = {
            "mode": "quicklook",
            "wave_hypothesis": "air",
            "forward_model": "native_interp",
            "use_ssbvel": True,
            "use_telluric_mask": True,
        }
    else:
        raise ValueError("Unknown PEPSI preset: {0}".format(preset))

    for dest, value in preset_values.items():
        if not _flag_present(argv, dest):
            setattr(args, dest, value)

    return args


def concat_with_gap(arrays, gap_value=np.nan, dtype=float):
    """Concatenate arrays with a single separator element between them."""
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
    """Concatenate boolean arrays with a False separator between windows."""
    arrays = [np.asarray(a, dtype=bool) for a in arrays]
    if len(arrays) == 0:
        return np.array([], dtype=bool)

    out = []
    for i, arr in enumerate(arrays):
        out.append(arr)
        if i < len(arrays) - 1:
            out.append(np.array([False], dtype=bool))
    return np.concatenate(out)


def _make_unit_collection(segments, window_defs, args):
    """
    Wrap PEPSI legacy line-window segments in a SpectrumCollection.

    The current PEPSI legacy fit uses unit weights, preserving the previous
    list-based behavior. The collection wrapper gives us the same abstraction
    used by the generic fitter and leaves a clean place for future per-window or
    per-arm weights.
    """
    weights = np.ones(len(segments), dtype=float)
    return SpectrumCollection(
        segments=segments,
        weights=weights,
        meta={
            "instrument": "PEPSI",
            "mode": "legacy_max",
            "wave_hypothesis": args.wave_hypothesis,
            "legacy_windows_air": list(window_defs),
            "legacy_flux_range": (float(args.legacy_flux_min), float(args.legacy_flux_max)),
        },
        name="pepsi_legacy_windows",
    )


def run_legacy_pepsi_fit(args, parser):
    """
    Run the PEPSI legacy-max comparison fit.

    This remains a script-level driver. The reusable PEPSI-specific
    segment/window/cache-support construction and legacy likelihood pieces live
    in Spyctres.recipes. The fitted line windows are wrapped in a
    SpectrumCollection so the joint fit has the same container abstraction as the
    generic full-spectrum fitter.
    """
    if args.forward_model != "native_interp":
        parser.error("--mode legacy_max currently requires --forward-model native_interp.")

    files = list(args.files)
    for path in files:
        if not os.path.isfile(path):
            parser.error("Input file not found: {0}".format(path))

    if args.phoenix_dir is None:
        parser.error(
            "No PHOENIX directory supplied. Set --phoenix-dir, SPYCTRES_PHOENIX_DIR, "
            "or [paths].phoenix_dir in ~/.config/spyctres/config.toml."
        )

    if not os.path.isdir(args.phoenix_dir):
        parser.error("PHOENIX directory not found: {0}".format(args.phoenix_dir))

    raw_segments = []
    for path in files:
        print("Reading PEPSI spectrum: {0}".format(path), flush=True)
        seg = read_spectrum(
            path,
            instrument="pepsi",
            product_profile=args.product_profile,
        )
        meta = dict(seg.meta)
        meta["source_file"] = path
        raw_segments.append(seg.copy(meta=meta))

    print("Preparing PEPSI legacy windows...", flush=True)
    input_segments, segments, window_defs = build_pepsi_legacy_segments(
        raw_segments,
        wave_hypothesis=args.wave_hypothesis,
        centers_air=args.legacy_centers,
        halfwidth_A=args.legacy_halfwidth,
        flux_min=args.legacy_flux_min,
        flux_max=args.legacy_flux_max,
        window_pad_A=args.window_pad,
    )

    if args.use_telluric_mask:
        print("Loading telluric mask...", flush=True)
        _, telluric_mask = Spyctres.load_telluric_lines(args.telluric_threshold)

        def exclude_mask(wave):
            return np.asarray(telluric_mask(wave)) > 0.5

        segments = [
            seg.copy(mask=np.asarray(seg.mask, dtype=bool) & ~exclude_mask(seg.wave))
            for seg in segments
        ]

    collection = _make_unit_collection(segments, window_defs, args)
    segments = list(collection.segments)
    segment_weights = np.asarray(collection.weights, dtype=float)

    print("Loading PHOENIX library...", flush=True)
    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))

    print("Selecting PHOENIX grid...", flush=True)
    teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
    teff_grid_req = pick_grid_range(teff_avail, args.teff_min, args.teff_max)
    feh_grid_req = pick_grid_range(feh_avail, args.feh_min, args.feh_max)
    logg_grid_req = pick_grid_range(logg_avail, args.logg_min, args.logg_max)

    teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
        teff_grid_req,
        feh_grid_req,
        logg_grid_req,
    )

    rv_bary_values = []
    for seg in input_segments:
        if args.use_ssbvel:
            try:
                rv_bary_values.append(pepsi_ssbvel_correction_kms(seg))
            except ValueError as exc:
                parser.error(str(exc))

    rv_bary_kms = float(np.nanmedian(rv_bary_values)) if rv_bary_values else 0.0

    R = args.R_override
    if R is None:
        r_vals = [seg.meta.get("resolution_R", None) for seg in input_segments]
        r_vals = [float(x) for x in r_vals if x is not None]
        R = float(np.nanmedian(r_vals)) if r_vals else None

    if args.cache_path is None:
        names = "_".join(os.path.basename(p).replace(".", "_") for p in files)
        if len(names) > 80:
            names = "legacy_pepsi_joint"

        speed_tag = "fast" if args.fast else "full"
        args.cache_path = "/tmp/spyctres_{0}_{1}_{2}_legacy_cache.npz".format(
            names,
            args.wave_hypothesis,
            speed_tag,
        )

    cache_support_segments = make_pepsi_legacy_cache_support_segments(
        input_segments=input_segments,
        window_defs_air=window_defs,
        window_pad_A=args.window_pad,
    )

    print("Preparing PHOENIX native interpolation cache...", flush=True)
    model_wave_grid, model_wave_medium = ensure_phoenix_native_interpolator_for_segments(
        segments=cache_support_segments,
        phoenix_lib=phoenix_lib,
        teff_grid=teff_grid_fit,
        feh_grid=feh_grid_fit,
        logg_grid=logg_grid_fit,
        cache_path=args.cache_path,
        model_margin_A=args.model_margin,
        progress_callback=lambda message: print(message, flush=True),
    )

    lo = np.array(
        [
            np.min(teff_grid_fit),
            np.min(feh_grid_fit),
            np.min(logg_grid_fit),
            float(args.rv_min),
            float(args.legacy_log_scale_min),
        ],
        dtype=float,
    )

    hi = np.array(
        [
            np.max(teff_grid_fit),
            np.max(feh_grid_fit),
            np.max(logg_grid_fit),
            float(args.rv_max),
            float(args.legacy_log_scale_max),
        ],
        dtype=float,
    )

    x0 = np.array(
        [args.teff0, args.feh0, args.logg0, args.rv0, 0.0],
        dtype=float,
    )
    x0 = np.minimum(np.maximum(x0, lo), hi)

    def objective(x):
        teff, feh, logg, rv, log_scale = map(float, x)

        try:
            models = evaluate_pepsi_legacy_max_models(
                phoenix_lib=phoenix_lib,
                segments=segments,
                model_wave_grid=model_wave_grid,
                model_wave_medium=model_wave_medium,
                teff=teff,
                feh=feh,
                logg=logg,
                rv_kms=rv,
                rv_bary_kms=rv_bary_kms,
                R=R,
                model_margin_A=args.model_margin,
            )
        except Exception:
            return 1.0e100

        total = 0.0
        n_total = 0

        for seg, weight, model in zip(segments, segment_weights, models):
            val, n, _model_norm, _used = pepsi_legacy_max_likelihood_terms(
                seg,
                model,
                log_err_scale=log_scale,
            )

            if not np.isfinite(val):
                return 1.0e100

            total += float(weight) * float(val)
            n_total += int(n)

        if n_total == 0:
            return 1.0e100

        return total

    if args.rv_init != "none":
        rv_grid = np.linspace(float(args.rv_min), float(args.rv_max), int(args.rv_grid_n))
        vals = []

        for rv in rv_grid:
            xt = x0.copy()
            xt[3] = rv
            vals.append(objective(xt))

        ibest = int(np.nanargmin(vals))
        x0[3] = float(rv_grid[ibest])
        print("Legacy RV init grid best:", x0[3])

    print("Running PEPSI legacy optimization...", flush=True)
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi)),
        options={
            "maxiter": int(args.legacy_maxiter),
            "ftol": 1e-8,
        },
    )

    teff, feh, logg, rv, log_scale = map(float, res.x)

    best_models_raw = evaluate_pepsi_legacy_max_models(
        phoenix_lib=phoenix_lib,
        segments=segments,
        model_wave_grid=model_wave_grid,
        model_wave_medium=model_wave_medium,
        teff=teff,
        feh=feh,
        logg=logg,
        rv_kms=rv,
        rv_bary_kms=rv_bary_kms,
        R=R,
        model_margin_A=args.model_margin,
    )

    model_norm_list = []
    used_masks = []
    chi2 = 0.0
    npts = 0

    for seg, weight, model in zip(segments, segment_weights, best_models_raw):
        _nll, n, model_norm, used = pepsi_legacy_max_likelihood_terms(
            seg,
            model,
            log_err_scale=log_scale,
        )

        e = np.asarray(seg.err, dtype=float)[used] * (10.0 ** log_scale)
        r = (np.asarray(seg.flux, dtype=float)[used] - model_norm[used]) / e

        chi2 += float(weight) * float(np.sum(r * r))
        npts += int(n)
        model_norm_list.append(model_norm)
        used_masks.append(used)

    dof = max(1, npts - 5)
    chi2_red = chi2 / dof

    print("Mode: legacy_max")
    print("Collection:", collection.name)
    print("Collection segments:", len(collection.segments))
    print("Files:")
    for path in files:
        print(" ", path)

    print("Wave medium hypothesis:", args.wave_hypothesis)
    print("Barycorr used [km/s]:", rv_bary_kms)
    print("Telluric mask:", bool(args.use_telluric_mask))
    print("R used:", R)
    print("Legacy flux range:", (args.legacy_flux_min, args.legacy_flux_max))

    print("Legacy windows defined in air:")
    for label, wmin, wmax in window_defs:
        print(" ", label, (wmin, wmax))

    print("Windows actually fitted:")
    for seg, weight in zip(segments, segment_weights):
        working = seg.meta.get("legacy_window_working", None)
        medium = seg.meta.get("legacy_window_medium", seg.wave_medium)

        extra = ""
        if working is not None:
            extra = " working_window_{0}={1}".format(medium, working[1:])

        print(
            " ",
            seg.name,
            "from",
            os.path.basename(str(seg.meta.get("source_file", ""))),
            "N=",
            int(np.sum(seg.mask)),
            "weight=",
            float(weight),
            extra,
        )

    print("Best-fit:")
    print("  Teff   =", teff)
    print("  [Fe/H] =", feh)
    print("  logg   =", logg)
    print("  RV     =", rv)
    print("  log10 error scale =", log_scale)
    print("  error scale =", 10.0 ** log_scale)
    print("  nll    =", float(res.fun))
    print("  chi2   =", chi2)
    print("  dof    =", dof)
    print("  chi2_red =", chi2_red)
    print("  success  =", bool(res.success))
    print("  message  =", res.message)

    wave_plot = concat_with_gap([seg.wave for seg in segments], gap_value=np.nan, dtype=float)
    flux_plot = concat_with_gap([seg.flux for seg in segments], gap_value=np.nan, dtype=float)
    err_plot = concat_with_gap([seg.err * (10.0 ** log_scale) for seg in segments], gap_value=np.nan, dtype=float)
    model_plot = concat_with_gap(model_norm_list, gap_value=np.nan, dtype=float)
    used_plot = concat_bool_with_gap(used_masks)
    excl_plot = np.zeros_like(used_plot, dtype=bool)

    title = (
        "PEPSI legacy_max {0}  Teff={1:.0f}  [Fe/H]={2:.2f}  "
        "logg={3:.2f}  RV={4:.1f}  chi2_red={5:.2f}".format(
            args.wave_hypothesis,
            teff,
            feh,
            logg,
            rv,
            chi2_red,
        )
    )

    print("Building diagnostic plot...", flush=True)
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


def main():
    parser = build_parser()

    standard = parser.add_argument_group("standard options")
    expert = parser.add_argument_group("expert fitting controls")
    legacy = parser.add_argument_group("legacy_max controls")
    cache = parser.add_argument_group("cache and verbosity")

    standard.add_argument("files", nargs="+", help="Input PEPSI .dxt.nor file(s)")
    standard.add_argument(
        "--preset",
        choices=["pepsi_quicklook", "pepsi_legacy_red_fast", "pepsi_legacy_red_full"],
        default=None,
        help=(
            "Apply a named PEPSI preset. "
            "Recommended: pepsi_legacy_red_fast for development, "
            "pepsi_legacy_red_full for a slower red-window diagnostic, "
            "or pepsi_quicklook for a single-file generic fit. "
            "Explicit flags override preset values."
        ),
    )
    standard.add_argument(
        "--phoenix-dir",
        default=None,
        help=(
            "Path to local PHOENIXv2 directory. "
            "Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file."
        ),
    )
    standard.add_argument(
        "--mode",
        choices=["quicklook", "legacy_max"],
        default="quicklook",
        help=(
            "Fit mode. Use quicklook for a generic one-file PEPSI fit, "
            "or legacy_max for the local line-window PEPSI comparison mode."
        ),
    )
    standard.add_argument(
        "--window-preset",
        choices=["auto", "blue_balmer", "red_halpha", "red_metals", "caii_triplet"],
        default="auto",
        help="Window preset for quicklook mode.",
    )
    standard.add_argument(
        "--window",
        nargs=3,
        action="append",
        metavar=("LABEL", "WMIN", "WMAX"),
        help="Custom quicklook fit window. Can be given multiple times.",
    )
    standard.add_argument(
        "--use-telluric-mask",
        action="store_true",
        help="Apply Spyctres' built-in telluric mask.",
    )
    standard.add_argument(
        "--use-ssbvel",
        action="store_true",
        help="Use header SSBVEL as a barycentric correction term.",
    )
    standard.add_argument(
        "--product-profile",
        choices=["generic", "pets_stellar_rest", "cds_aanda_671_a7"],
        default="generic",
        help=(
            "Documented PEPSI release convention. The .dxt.nor suffix is never "
            "used to infer wavelength units or frame."
        ),
    )
    standard.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use a smaller PEPSI development grid and shorter optimization. "
            "Intended for code testing, not final science results."
        ),
    )

    expert.add_argument(
        "--wave-hypothesis",
        choices=["unknown", "air", "vacuum", "air_to_vac"],
        default="air",
        help="Observed wavelength-medium hypothesis.",
    )
    expert.add_argument(
        "--forward-model",
        choices=["interp_observed", "native_interp"],
        default="native_interp",
        help=(
            "Forward-model path. native_interp is the recommended path for "
            "line-profile work."
        ),
    )
    expert.add_argument(
        "--model-margin",
        type=float,
        default=20.0,
        help="Margin in Angstrom for native_interp model preparation.",
    )
    expert.add_argument(
        "--window-pad",
        type=float,
        default=2.0,
        help="Padding in Angstrom added around each PEPSI fit window.",
    )
    expert.add_argument(
        "--R-override",
        type=float,
        default=None,
        help="Override metadata resolving power R.",
    )
    expert.add_argument(
        "--teff-min",
        type=float,
        default=4500.0,
        help="Minimum Teff for explicit PHOENIX grid.",
    )
    expert.add_argument(
        "--teff-max",
        type=float,
        default=12000.0,
        help="Maximum Teff for explicit PHOENIX grid.",
    )
    expert.add_argument(
        "--feh-min",
        type=float,
        default=-1.5,
        help="Minimum [Fe/H] for explicit PHOENIX grid.",
    )
    expert.add_argument(
        "--feh-max",
        type=float,
        default=0.5,
        help="Maximum [Fe/H] for explicit PHOENIX grid.",
    )
    expert.add_argument(
        "--logg-min",
        type=float,
        default=2.5,
        help="Minimum logg for explicit PHOENIX grid.",
    )
    expert.add_argument(
        "--logg-max",
        type=float,
        default=5.5,
        help="Maximum logg for explicit PHOENIX grid.",
    )
    expert.add_argument(
        "--mdeg",
        type=int,
        default=1,
        help="Legendre continuum degree for quicklook mode.",
    )
    expert.add_argument("--teff0", type=float, default=6500.0, help="Initial Teff.")
    expert.add_argument("--feh0", type=float, default=-0.5, help="Initial [Fe/H].")
    expert.add_argument("--logg0", type=float, default=4.0, help="Initial logg.")
    expert.add_argument("--rv0", type=float, default=0.0, help="Initial stellar RV in km/s.")
    expert.add_argument(
        "--rv-init",
        choices=["grid", "none"],
        default="grid",
        help="RV initialization strategy.",
    )
    expert.add_argument(
        "--rv-grid-n",
        type=int,
        default=161,
        help="Number of trial RV points in coarse RV scan.",
    )
    expert.add_argument("--rv-min", type=float, default=-300.0, help="Minimum RV in km/s.")
    expert.add_argument("--rv-max", type=float, default=300.0, help="Maximum RV in km/s.")
    expert.add_argument(
        "--telluric-threshold",
        type=float,
        default=0.90,
        help="Telluric mask threshold.",
    )

    legacy.add_argument(
        "--legacy-centers",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Line centers in Angstrom for --mode legacy_max. "
            "Default: 6495 6545 6561 8498 8542 8662."
        ),
    )
    legacy.add_argument(
        "--legacy-halfwidth",
        type=float,
        default=10.0,
        help="Half-width in Angstrom for legacy windows.",
    )
    legacy.add_argument(
        "--legacy-flux-min",
        type=float,
        default=0.2,
        help="Minimum normalized flux retained in legacy mode.",
    )
    legacy.add_argument(
        "--legacy-flux-max",
        type=float,
        default=1.1,
        help="Maximum normalized flux retained in legacy mode.",
    )
    legacy.add_argument(
        "--legacy-log-scale-min",
        type=float,
        default=-2.0,
        help="Minimum log10 error-scale in legacy mode.",
    )
    legacy.add_argument(
        "--legacy-log-scale-max",
        type=float,
        default=2.0,
        help="Maximum log10 error-scale in legacy mode.",
    )
    legacy.add_argument(
        "--legacy-maxiter",
        type=int,
        default=120,
        help="Maximum optimizer iterations in legacy mode.",
    )

    cache.add_argument("--cache-path", default=None, help="Interpolator cache path.")
    cache.add_argument("--verbose", type=int, default=1, help="Verbosity level.")

    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    args = apply_named_preset(args, raw_argv)

    if args.fast:
        args.teff_min = 5000.0
        args.teff_max = 6000.0
        args.feh_min = -1.5
        args.feh_max = 0.5
        args.logg_min = 2.5
        args.logg_max = 4.0

        if args.rv0 != 0.0:
            args.rv_min = max(float(args.rv_min), float(args.rv0) - 75.0)
            args.rv_max = min(float(args.rv_max), float(args.rv0) + 75.0)

        args.rv_grid_n = min(int(args.rv_grid_n), 41)
        args.legacy_maxiter = min(int(args.legacy_maxiter), 50)
        args.legacy_log_scale_max = min(float(args.legacy_log_scale_max), 1.0)
        args.verbose = min(int(args.verbose), 1)

    try:
        args.phoenix_dir = resolve_phoenix_dir(args.phoenix_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if args.mode == "legacy_max":
        run_legacy_pepsi_fit(args, parser)
        return

    if len(args.files) != 1:
        parser.error("quicklook mode accepts exactly one PEPSI file. Use --mode legacy_max for joint fits.")

    args.file = args.files[0]

    if not os.path.isfile(args.file):
        parser.error("Input file not found: {0}".format(args.file))

    if args.phoenix_dir is None:
        parser.error(
            "No PHOENIX directory supplied. Set --phoenix-dir, SPYCTRES_PHOENIX_DIR, "
            "or [paths].phoenix_dir in ~/.config/spyctres/config.toml."
        )

    if args.forward_model == "interp_observed" and args.wave_hypothesis == "unknown":
        parser.error(
            "--forward-model interp_observed requires a known wavelength medium. "
            "Use --wave-hypothesis air, vacuum, or air_to_vac, or use native_interp."
        )

    print("Reading PEPSI spectrum...", flush=True)
    seg0 = read_spectrum(
        args.file,
        instrument="pepsi",
        product_profile=args.product_profile,
    )
    seg0 = seg0.copy(mask=build_pepsi_normalized_mask(seg0))
    seg = apply_pepsi_wave_hypothesis(seg0, args.wave_hypothesis)

    if args.window is not None:
        window_defs = parse_custom_windows(args.window)
        window_preset_used = "custom"
    else:
        if args.window_preset == "auto":
            window_preset_used = choose_auto_window_preset(seg)
        else:
            window_preset_used = args.window_preset
        window_defs = WINDOW_PRESETS[window_preset_used]

    print("Preparing PEPSI fit windows...", flush=True)
    segments = build_window_segments(seg, window_defs, pad=args.window_pad)

    exclude_mask = None
    if args.use_telluric_mask:
        print("Loading telluric mask...", flush=True)
        _, telluric_mask = Spyctres.load_telluric_lines(args.telluric_threshold)

        def exclude_mask(wave):
            return np.asarray(telluric_mask(wave)) > 0.5

    used_masks_plot = [
        build_effective_fit_mask(seg_i, exclude_mask=exclude_mask)
        for seg_i in segments
    ]
    if not any(np.any(m) for m in used_masks_plot):
        raise ValueError("No usable points remain after masking.")

    print("Loading PHOENIX library...", flush=True)
    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))

    print("Selecting PHOENIX grid...", flush=True)
    teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
    teff_grid_req = pick_grid_range(teff_avail, args.teff_min, args.teff_max)
    feh_grid_req = pick_grid_range(feh_avail, args.feh_min, args.feh_max)
    logg_grid_req = pick_grid_range(logg_avail, args.logg_min, args.logg_max)
    teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
        teff_grid_req, feh_grid_req, logg_grid_req
    )

    rv_bary_kms = 0.0
    ssbvel_mps = seg.meta.get("ssbvel_mps")
    if args.use_ssbvel:
        try:
            rv_bary_kms = pepsi_ssbvel_correction_kms(seg)
        except ValueError as exc:
            parser.error(str(exc))

    R = args.R_override if args.R_override is not None else seg.meta.get("resolution_R", None)

    if args.cache_path is None:
        tag = os.path.basename(args.file).replace(".", "_")
        args.cache_path = "/tmp/spyctres_{0}_{1}_cache.npz".format(tag, args.wave_hypothesis)

    print("Running PHOENIX fit...", flush=True)
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
        progress_callback=lambda message: print(message, flush=True),
    )

    print("Reconstructing best-fit model for plotting...", flush=True)
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
    print("PEPSI product profile:", seg.meta.get("pepsi_product_profile"))
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
    print(format_fit_quality_report(out))
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
