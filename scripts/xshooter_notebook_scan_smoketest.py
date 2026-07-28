import os
import argparse
import warnings

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

from Spyctres.io import read_spectrum, SpectrumSegment
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.waveutils import convert_wavelength_medium
from Spyctres.preprocessing import (
    combine_exclusion_masks,
)
from Spyctres.config import load_user_config, get_config_value, resolve_setting
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from Spyctres.recipes import (
    normalize_model_sidebands,
    pick_grid_range,
    prepare_xshooter_balmer_case,
    solve_sideband_multiplicative_poly,
)

C_KMS = 299792.458


def doppler_shift_wave(wave_A, rv_kms):
    wave_A = np.asarray(wave_A, dtype=np.float64)
    return wave_A * (1.0 + float(rv_kms) / C_KMS)


def convolve_to_resolution_loglam(wave_A, flux, R):
    """
    Convolve with a Gaussian in log-lambda, matching the notebook cell-54 path.
    """
    wave_A = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if R is None:
        return flux.copy()

    if not np.all(np.diff(wave_A) > 0):
        idx = np.argsort(wave_A)
        wave_A = wave_A[idx]
        flux = flux[idx]

    m = np.isfinite(wave_A) & (wave_A > 0) & np.isfinite(flux)
    wave_A = wave_A[m]
    flux = flux[m]

    if len(wave_A) < 5:
        return flux.copy()

    loglam = np.log(wave_A)
    dloglam = np.median(np.diff(loglam))

    sigma_v = C_KMS / (float(R) * 2.355)
    sigma_pix = (sigma_v / C_KMS) / dloglam

    if sigma_pix < 0.3:
        return flux.copy()

    return gaussian_filter1d(flux, sigma_pix, mode="nearest")


def resample_flux(w_src, f_src, w_tgt):
    w_src = np.asarray(w_src, dtype=np.float64)
    f_src = np.asarray(f_src, dtype=np.float64)
    w_tgt = np.asarray(w_tgt, dtype=np.float64)

    m = np.isfinite(w_src) & np.isfinite(f_src)
    w_src = w_src[m]
    f_src = f_src[m]

    if len(w_src) < 4:
        return np.full_like(w_tgt, np.nan, dtype=float)

    if not np.all(np.diff(w_src) > 0):
        idx = np.argsort(w_src)
        w_src = w_src[idx]
        f_src = f_src[idx]

    f = interp1d(
        w_src,
        f_src,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(f(w_tgt), dtype=float)
    

def build_used_mask(seg, exclude_mask=None):
    m = np.asarray(seg.mask, dtype=bool)
    m &= np.isfinite(seg.wave) & np.isfinite(seg.flux)

    if seg.err is not None:
        m &= np.isfinite(seg.err) & (seg.err > 0)

    if exclude_mask is not None:
        m &= ~(np.asarray(exclude_mask(seg.wave)) > 0.5)

    return m


def build_excluded_mask(seg, exclude_mask=None):
    m = np.zeros_like(seg.wave, dtype=bool)
    if exclude_mask is not None:
        m |= (np.asarray(exclude_mask(seg.wave)) > 0.5)
    return m


def evaluate_template_on_segments(
    segments,
    phoenix_wave_native,
    template_flux_native,
    rv_kms,
    rv_bary_kms,
    R,
    exclude_mask,
    sideband_width,
    sideband_order,
    sideband_poly_order,
    phoenix_wave_medium="vacuum",
    model_margin_A=200.0,
    return_models=False,
):
    """
    Notebook-faithful template evaluation:

    1. Convert native PHOENIX wavelengths into the data wavelength convention.
    2. Subset a wide region before any model operations.
    3. Doppler shift on the native/high-res grid.
    4. Convolve in log-lambda at resolving power R.
    5. Resample to each observed segment grid.
    6. Apply the same sideband normalization and wing polynomial as the data path.
    """
    used_masks = [build_used_mask(seg, exclude_mask=exclude_mask) for seg in segments]
    excluded_masks = [build_excluded_mask(seg, exclude_mask=exclude_mask) for seg in segments]

    fit_los = []
    fit_his = []
    for seg in segments:
        m_fit = np.asarray(seg.mask, dtype=bool)
        if np.any(m_fit):
            fit_los.append(float(np.min(np.asarray(seg.wave, dtype=float)[m_fit])))
            fit_his.append(float(np.max(np.asarray(seg.wave, dtype=float)[m_fit])))

    if len(fit_los) == 0:
        raise ValueError("No fit pixels found in segments.")

    wmin_need = min(fit_los) - float(model_margin_A)
    wmax_need = max(fit_his) + float(model_margin_A)

    segment_media = sorted(set(str(seg.wave_medium).lower() for seg in segments))
    if len(segment_media) == 1 and segment_media[0] in ("air", "vacuum"):
        data_wave_medium = segment_media[0]
    else:
        data_wave_medium = str(phoenix_wave_medium).lower()

    if data_wave_medium != str(phoenix_wave_medium).lower():
        w_ph_all = convert_wavelength_medium(
            np.asarray(phoenix_wave_native, dtype=float),
            from_medium=str(phoenix_wave_medium).lower(),
            to_medium=data_wave_medium,
        )
    else:
        w_ph_all = np.asarray(phoenix_wave_native, dtype=float).copy()

    f_ph_all = np.asarray(template_flux_native, dtype=float)

    m_ph = (
        np.isfinite(w_ph_all) &
        np.isfinite(f_ph_all) &
        (w_ph_all > 0) &
        (w_ph_all >= wmin_need) &
        (w_ph_all <= wmax_need)
    )

    w_ph = w_ph_all[m_ph]
    f_ph = f_ph_all[m_ph]

    if len(w_ph) < 10:
        raise ValueError("Too few PHOENIX points remain after notebook-style subsetting.")

    rv_tot = float(rv_bary_kms) + float(rv_kms)
    w_shift = doppler_shift_wave(w_ph, rv_tot)

    m_ok = np.isfinite(w_shift) & (w_shift > 0) & np.isfinite(f_ph)
    w_shift = w_shift[m_ok]
    f_ph = f_ph[m_ok]

    f_conv = convolve_to_resolution_loglam(w_shift, f_ph, R)

    chi2 = 0.0
    npts = 0
    model_corr_list = []
    coeffs_list = []
    per_window = {}

    for i, (seg, used_mask, excl_mask) in enumerate(zip(segments, used_masks, excluded_masks)):
        wave = np.asarray(seg.wave, dtype=float)
        flux = np.asarray(seg.flux, dtype=float)
        err = np.asarray(seg.err, dtype=float)
        used_mask = np.asarray(used_mask, dtype=bool)
        excl_mask = np.asarray(excl_mask, dtype=bool)

        idx = np.argsort(wave)
        wave = wave[idx]
        flux = flux[idx]
        err = err[idx]
        used_mask = used_mask[idx]
        excl_mask = excl_mask[idx]

        model0 = resample_flux(w_shift, f_conv, wave)
        model_norm, _model_norm_info = normalize_model_sidebands(
            seg=SpectrumSegment(
                wave=wave,
                flux=flux,
                err=err,
                mask=np.asarray(seg.mask, dtype=bool)[idx],
                meta=dict(seg.meta),
                wave_medium=seg.wave_medium,
                wave_frame=seg.wave_frame,
                name=seg.name,
            ),
            model_flux=model0,
            sideband_width=sideband_width,
            sideband_order=sideband_order,
        )

        model_corr, coeffs = solve_sideband_multiplicative_poly(
            wave=wave,
            flux=flux,
            err=err,
            model=model_norm,
            used_mask=used_mask,
            order=sideband_poly_order,
        )

        z = (flux[used_mask] - model_corr[used_mask]) / err[used_mask]
        chi2_i = float(np.sum(z * z))
        n_i = int(np.sum(used_mask))

        chi2 += chi2_i
        npts += n_i
        model_corr_list.append(model_corr)
        coeffs_list.append(coeffs)

        label = str(seg.meta.get("line_label", f"win{i}"))
        per_window[label] = {
            "chi2_red": np.nan if n_i <= 4 else chi2_i / (n_i - 4),
            "n": n_i,
            "wave_min": float(np.min(wave[used_mask])) if np.any(used_mask) else np.nan,
            "wave_max": float(np.max(wave[used_mask])) if np.any(used_mask) else np.nan,
        }

        excluded_masks[i] = excl_mask

    out = {
        "chi2": chi2,
        "npts": npts,
        "chi2_red": np.nan if npts <= 4 else chi2 / (npts - 4),
        "used_masks": used_masks,
        "excluded_masks": excluded_masks,
        "per_window": per_window,
    }

    if return_models:
        out["model_corr_list"] = model_corr_list
        out["coeffs_list"] = coeffs_list

    return out
    
def build_parser():
    return argparse.ArgumentParser(
        description=(
            "Developer/regression notebook-faithful discrete PHOENIX grid scan for X-SHOOTER UVB Balmer windows.\n"
            "This is a validation reference script, not the generic package fitter or a public example.\n"
            "It expects a reduced X-SHOOTER UVB FITS file and a local PHOENIXv2 installation."
        ),
        epilog=(
            "Examples:\n"
            "  export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2\n"
            "  python scripts/xshooter_notebook_scan_smoketest.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits\n\n"
            "  python scripts/xshooter_notebook_scan_smoketest.py \\\n"
            "    --phoenix-dir /path/to/PHOENIXv2 \\\n"
            "    --use-telluric-mask --use-barycorr \\\n"
            "    examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits\n"
            "  ~/.config/spyctres/config.toml:\n"
            "    [paths]\n"
            "    phoenix_dir = \"/path/to/PHOENIXv2\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    
    
def main():
    parser = build_parser()
    parser.add_argument("file", help="Input X-SHOOTER UVB FITS file")
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Path to local PHOENIXv2 directory. Precedence: CLI > SPYCTRES_PHOENIX_DIR > config file.",
    )
    parser.add_argument("--wmin", type=float, default=3980.0)
    parser.add_argument("--wmax", type=float, default=5500.0)
    parser.add_argument("--clip-left", type=int, default=0)
    parser.add_argument("--clip-right", type=int, default=0)
    parser.add_argument("--core-mask", type=float, default=12.0)
    parser.add_argument("--window-pad", type=float, default=20.0)
    parser.add_argument("--sideband-width", type=float, default=10.0)
    parser.add_argument("--sideband-order", type=int, default=1)
    parser.add_argument("--sideband-poly-order", type=int, default=1)
    parser.add_argument("--R-override", type=float, default=5100.0)
    parser.add_argument(
        "--model-margin",
        type=float,
        default=200.0,
        help="Margin in Angstrom in data convention for native PHOENIX subset before RV shift and convolution",
    )
    parser.add_argument("--teff-min", type=float, default=9500.0)
    parser.add_argument("--teff-max", type=float, default=12500.0)
    parser.add_argument("--feh-min", type=float, default=-0.5)
    parser.add_argument("--feh-max", type=float, default=0.5)
    parser.add_argument("--logg-min", type=float, default=3.0)
    parser.add_argument("--logg-max", type=float, default=6.0)
    parser.add_argument("--rv-coarse-step", type=float, default=30.0)
    parser.add_argument("--rv-fine-step", type=float, default=5.0)
    parser.add_argument("--rv-fine-halfspan", type=float, default=30.0)
    parser.add_argument("--use-barycorr", action="store_true")
    parser.add_argument("--use-telluric-mask", action="store_true")
    parser.add_argument("--telluric-threshold", type=float, default=0.90)
    parser.add_argument("--top-n", type=int, default=10)
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
    
    seg0 = read_spectrum(args.file, reader="xshooter_merge1d")
    arm = str(seg0.meta.get("arm", "")).strip().upper()
    if arm and arm != "UVB":
        parser.error(
            "This script expects an X-SHOOTER UVB file, but the reader reported arm={0!r}.".format(arm)
        )
    balmer_case = prepare_xshooter_balmer_case(
        seg0,
        wmin=args.wmin,
        wmax=args.wmax,
        clip_left=args.clip_left,
        clip_right=args.clip_right,
        window_mode="notebook",
        window_pad=args.window_pad,
        norm_mode="sideband",
        sideband_width=args.sideband_width,
        sideband_order=args.sideband_order,
        core_mask=args.core_mask,
        use_telluric_mask=args.use_telluric_mask,
        telluric_threshold=args.telluric_threshold,
    )
    segments = list(balmer_case.fit_segments)
    balmer_windows = list(balmer_case.balmer_windows)
    seg_ref = segments[0]

    for mask in balmer_case.exclude_masks:
        meta = getattr(mask, "metadata", {}) or {}
        if meta.get("method") == "transmission_threshold":
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

    exclude_mask = combine_exclusion_masks(
        balmer_case.exclude_masks,
        name="xshooter_notebook_scan_exclusions",
    )

    phoenix_lib = PhoenixLibrary(args.phoenix_dir, verbose=bool(args.verbose))
    teff_avail, feh_avail, logg_avail = phoenix_lib.available_axes()
    teff_grid_req = pick_grid_range(teff_avail, args.teff_min, args.teff_max)
    feh_grid_req = pick_grid_range(feh_avail, args.feh_min, args.feh_max)
    logg_grid_req = pick_grid_range(logg_avail, args.logg_min, args.logg_max)
    teff_grid_fit, feh_grid_fit, logg_grid_fit = phoenix_lib.complete_subgrid(
        teff_grid_req, feh_grid_req, logg_grid_req
    )

    rv_bary_kms = 0.0
    if args.use_barycorr:
        rv_bary_kms = float(seg_ref.meta.get("barycorr_kms") or 0.0)

    R = args.R_override if args.R_override is not None else seg_ref.meta.get("resolution_R", None)
    
    rv_coarse = np.arange(-300.0, 300.0 + 1e-9, args.rv_coarse_step)
    template_cache = {}
    rows = []

    n_total = len(teff_grid_fit) * len(feh_grid_fit) * len(logg_grid_fit)
    count = 0

    for teff in teff_grid_fit:
        for feh in feh_grid_fit:
            for logg in logg_grid_fit:
                count += 1
                if args.verbose:
                    print(f"[{count}/{n_total}] Teff={teff:.0f} [Fe/H]={feh:+.1f} logg={logg:.1f}")
                
                key = (float(teff), float(feh), float(logg))
                if key not in template_cache:
                    _wave_native, tpl_flux = phoenix_lib.load_template(
                        teff=float(teff),
                        logg=float(logg),
                        feh=float(feh),
                        wave=None,
                        wave_medium=None,
                    )
                    template_cache[key] = np.asarray(tpl_flux, dtype=float)

                tpl_flux = template_cache[key]
                
                best = None
                for rv in rv_coarse:
                    rec = evaluate_template_on_segments(
                        segments=segments,
                        phoenix_wave_native=phoenix_lib.phoenix_wave,
                        template_flux_native=tpl_flux,
                        rv_kms=float(rv),
                        rv_bary_kms=rv_bary_kms,
                        R=R,
                        exclude_mask=exclude_mask,
                        sideband_width=args.sideband_width,
                        sideband_order=args.sideband_order,
                        sideband_poly_order=args.sideband_poly_order,
                        phoenix_wave_medium=phoenix_lib.phoenix_wave_medium,
                        model_margin_A=args.model_margin,
                        return_models=False,
                    )
                    if rec["npts"] <= 0:
                        continue
                    if (best is None) or (rec["chi2"] < best["chi2"]):
                        best = {"chi2": rec["chi2"], "npts": rec["npts"], "rv": float(rv)}

                if best is None:
                    continue

                rv0 = best["rv"]
                rv_fine = np.arange(
                    rv0 - args.rv_fine_halfspan,
                    rv0 + args.rv_fine_halfspan + 1e-9,
                    args.rv_fine_step,
                )
                for rv in rv_fine:
                    rec = evaluate_template_on_segments(
                        segments=segments,
                        phoenix_wave_native=phoenix_lib.phoenix_wave,
                        template_flux_native=tpl_flux,
                        rv_kms=float(rv),
                        rv_bary_kms=rv_bary_kms,
                        R=R,
                        exclude_mask=exclude_mask,
                        sideband_width=args.sideband_width,
                        sideband_order=args.sideband_order,
                        sideband_poly_order=args.sideband_poly_order,
                        phoenix_wave_medium=phoenix_lib.phoenix_wave_medium,
                        model_margin_A=args.model_margin,
                        return_models=False,
                    )
                    if rec["npts"] <= 0:
                        continue
                    if rec["chi2"] < best["chi2"]:
                        best = {"chi2": rec["chi2"], "npts": rec["npts"], "rv": float(rv)}

                rows.append({
                    "teff": float(teff),
                    "feh": float(feh),
                    "logg": float(logg),
                    "rv_kms": float(best["rv"]),
                    "chi2": float(best["chi2"]),
                    "npts": int(best["npts"]),
                    "chi2_red": np.nan if best["npts"] <= 4 else float(best["chi2"]) / (best["npts"] - 4),
                })

    if len(rows) == 0:
        raise RuntimeError("No valid template evaluations were produced.")

    rows = sorted(rows, key=lambda r: r["chi2"])
    best = rows[0]
    ref_point = (9800.0, -0.5, 3.0)
    ref_match = None
    for j, row in enumerate(rows, start=1):
        if (
            abs(row["teff"] - ref_point[0]) < 1e-6 and
            abs(row["feh"] - ref_point[1]) < 1e-6 and
            abs(row["logg"] - ref_point[2]) < 1e-6
        ):
            ref_match = (j, row)
            break
    
    best_key = (best["teff"], best["feh"], best["logg"])
    best_tpl_flux = template_cache[best_key]

    best_eval = evaluate_template_on_segments(
        segments=segments,
        phoenix_wave_native=phoenix_lib.phoenix_wave,
        template_flux_native=best_tpl_flux,
        rv_kms=best["rv_kms"],
        rv_bary_kms=rv_bary_kms,
        R=R,
        exclude_mask=exclude_mask,
        sideband_width=args.sideband_width,
        sideband_order=args.sideband_order,
        sideband_poly_order=args.sideband_poly_order,
        phoenix_wave_medium=phoenix_lib.phoenix_wave_medium,
        model_margin_A=args.model_margin,
        return_models=True,
    )

    print("File:", args.file)
    print("Object:", seg_ref.meta.get("object"))
    print("Arm:", seg_ref.meta.get("arm"))
    print("R used:", R)
    print("Barycorr used [km/s]:", rv_bary_kms)
    print("Best discrete-template fit:")
    print("  Teff   =", best["teff"])
    print("  [Fe/H] =", best["feh"])
    print("  logg   =", best["logg"])
    print("  RV     =", best["rv_kms"])
    print("  chi2   =", best["chi2"])
    print("  npts   =", best["npts"])
    print("  chi2_red approx =", best["chi2_red"])
    print("Top results:")
    for row in rows[:args.top_n]:
        print(
            "  Teff={0:.0f} [Fe/H]={1:+.1f} logg={2:.1f} RV={3:+.1f} chi2={4:.1f} chi2_red~{5:.3f}".format(
                row["teff"], row["feh"], row["logg"], row["rv_kms"], row["chi2"], row["chi2_red"]
            )
        )
    if ref_match is not None:
        j, row = ref_match
        print("Notebook-like reference point:")
        print(
            "  rank={0}  Teff={1:.0f} [Fe/H]={2:+.1f} logg={3:.1f} RV={4:+.1f} chi2={5:.1f} delta_chi2={6:.1f}".format(
                j,
                row["teff"],
                row["feh"],
                row["logg"],
                row["rv_kms"],
                row["chi2"],
                row["chi2"] - best["chi2"],
            )
        )
    print("Per-window chi2_red:")
    for k, v in best_eval["per_window"].items():
        print(
            "  {0}: chi2_red={1} N={2} range=({3:.1f},{4:.1f})".format(
                k, v["chi2_red"], v["n"], v["wave_min"], v["wave_max"]
            )
        )

    title = (
        f"{os.path.basename(args.file)}  discrete PHOENIX scan  "
        f"Teff={best['teff']:.0f}  [Fe/H]={best['feh']:.2f}  "
        f"logg={best['logg']:.2f}  RV={best['rv_kms']:.1f}  "
        f"chi2_red~={best['chi2_red']:.2f}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (label, _, _), seg_i, model_i, excl_i in zip(
        axes, balmer_windows, segments, best_eval["model_corr_list"], best_eval["excluded_masks"]
    ):
        ax.plot(seg_i.wave, seg_i.flux, lw=1.0, label="Data")
        ax.plot(seg_i.wave, model_i, lw=1.0, label="Model")

        if np.any(excl_i):
            y0, y1 = ax.get_ylim()
            ax.fill_between(seg_i.wave, y0, y1, where=excl_i, alpha=0.15)
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


if __name__ == "__main__":
    main()
