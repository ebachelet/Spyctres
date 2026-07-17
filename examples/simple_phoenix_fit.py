"""Example 1: minimal command-line path for the public Spyctres PHOENIX API.

Purpose
-------
This script demonstrates the recommended public `fit_stellar_spectrum()`
workflow from a reduced spectrum to a structured PHOENIX result and an
interactive diagnostic plot. The plot is a
wide, stacked diagnostic view: observed spectrum and model on top, residuals
underneath. This makes it easier to inspect broad failures, masked regions, and
line-profile mismatches than a small square pop-up.

The script does perform a full-spectrum fit over the usable input pixels, but it
is not configured as a precision line-profile analysis. In particular, it does
not fit rotational or macroturbulent broadening, detailed abundance patterns, or
instrument-specific LSF variations. The reader's nominal resolution (or ``--R``)
is the only instrumental broadening supplied. Individual observed lines can
therefore be wider or narrower than the demonstration model even when the broad
atmospheric classification is useful.

By default, the diagnostic plot focuses on the fitted wavelength range and draws
the PHOENIX/model residuals only on pixels that were actually used by the fit.
Use ``--plot-xlim all`` if you want to inspect the full loaded segment and mask
boundaries.

The example also opens a default-on zoomed line-diagnostic figure for
Balmer/Ca/Mg lines that overlap fitted pixels, adding He I only for hot fitted
solutions. These panels are visual diagnostics, not separate local line fits.

Example
-------
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_result.json \
  --output-plot /tmp/spyctres_fit.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
from Spyctres import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt
from Spyctres import fit_stellar_spectrum, prepare_phoenix_fit_kwargs
from Spyctres.diagnostics import (
    KNOWN_RESIDUAL_WINDOWS,
    annotate_nonstellar_features,
    diagnose_known_residual_windows,
)
from Spyctres.io import read_spectrum
from Spyctres.plotting import COMMON_LINES, plot_fit_referee, plot_fit_windows
from Spyctres.preprocessing import (
    OPTICAL_DIB_DIAGNOSTIC_FEATURES,
    archive_exclusion_masks_for_segment,
    audit_spectrum_for_fit,
    nonstellar_feature_mask,
    nonstellar_feature_masks,
    nonstellar_feature_metadata,
)


DIB_FEATURE_NAMES = OPTICAL_DIB_DIAGNOSTIC_FEATURES


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal fit_stellar_spectrum() PHOENIX demonstration. This is "
            "not a precision line-profile fit: rotation, macroturbulence, "
            "abundance variations, and detailed LSF structure are not fitted."
        ),
        epilog=(
            "Example:\n"
            "  python examples/simple_phoenix_fit.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter "
            "--output-json /tmp/spyctres_result.json "
            "--output-plot /tmp/spyctres_fit.png\n\n"
            "Approximate SDSS quicklook example:\n"
            "  python examples/simple_phoenix_fit.py spec-PLATE-MJD-FIBER.fits "
            "--instrument sdss --R 2000 --sdss-mask-policy stellar_strict\n"
            "  SDSS reader metadata intentionally keeps resolution=None; "
            "--R 2000 is an explicit quicklook approximation, not precision "
            "SDSS LSF modelling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spectrum", help="Reduced one-dimensional spectrum file.")
    parser.add_argument("--instrument", required=True, help="Registered reader name.")
    parser.add_argument(
        "--sdss-mask-policy",
        choices=("auto", "ivar_only", "and_mask_conservative", "stellar_strict", "sky_strict"),
        default="auto",
        help=(
            "SDSS reader bitmask policy. 'auto' uses stellar_strict for this "
            "PHOENIX fitting example; other instruments ignore this option."
        ),
    )
    parser.add_argument("--phoenix-dir", default=None)
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
    parser.add_argument("--teff", type=float, default=None)
    parser.add_argument("--feh", type=float, default=None)
    parser.add_argument("--logg", type=float, default=None)
    parser.add_argument("--rv", type=float, default=None)
    parser.add_argument("--teff-min", type=float, default=None)
    parser.add_argument("--teff-max", type=float, default=None)
    parser.add_argument("--feh-min", type=float, default=None)
    parser.add_argument("--feh-max", type=float, default=None)
    parser.add_argument("--logg-min", type=float, default=None)
    parser.add_argument("--logg-max", type=float, default=None)
    parser.add_argument("--rv-min", type=float, default=None)
    parser.add_argument("--rv-max", type=float, default=None)
    parser.add_argument("--wmin", type=float, default=None, help="Override fit-window minimum wavelength in Angstrom.")
    parser.add_argument("--wmax", type=float, default=None, help="Override fit-window maximum wavelength in Angstrom.")
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument(
        "--show-dibs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Annotate known diffuse interstellar bands, currently DIB 4428 "
            "and DIB 4882, on the overview diagnostic plot when they overlap "
            "the spectrum."
        ),
    )
    parser.add_argument(
        "--mask-dibs",
        action="store_true",
        help=(
            "Exclude known DIB regions from the stellar fit. This is equivalent "
            "to --nonstellar-feature-policy mask_known."
        ),
    )
    parser.add_argument(
        "--nonstellar-feature-policy",
        choices=("warn", "mask_known", "ignore"),
        default="warn",
        help=(
            "How to handle known non-stellar features such as DIB 4428 and "
            "DIB 4882. 'warn' annotates/flags only, 'mask_known' excludes the "
            "named regions, and 'ignore' records overlap but raises no flags. "
            "These correspond to flag/mask/off in the reviewer terminology; "
            "residual diagnostics are controlled separately."
        ),
    )
    parser.add_argument(
        "--dib-padding",
        type=float,
        default=0.0,
        help="Extra Angstrom half-width padding for DIB annotations/masks.",
    )
    parser.add_argument(
        "--archive-mask-policy",
        choices=("apply", "warn", "ignore"),
        default="apply",
        help=(
            "How to handle recognized archive/product bad-region catalogs. "
            "'apply' excludes them as named fit masks and records provenance; "
            "'warn' leaves them fitted but raises readiness/quality warnings; "
            "'ignore' leaves them fitted and records that the user explicitly "
            "ignored archive mask advice."
        ),
    )
    parser.add_argument(
        "--known-residual-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inspect curated diagnostic windows, such as the H-beta red wing, "
            "and flag coherent residuals without automatically masking them."
        ),
    )
    parser.add_argument(
        "--known-residual-threshold",
        type=float,
        default=2.5,
        help=(
            "Flag a known diagnostic window when either |median residual| or "
            "RMS residual exceeds this sigma threshold."
        ),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument(
        "--plot-layout",
        choices=("stacked", "side_by_side"),
        default="stacked",
        help=(
            "Diagnostic plot layout. 'stacked' is the default interactive view: "
            "wide data/model panel over a wide residual panel."
        ),
    )
    parser.add_argument(
        "--plot-xlim",
        choices=("fit", "all"),
        default="fit",
        help=(
            "Diagnostic plot x-axis. 'fit' focuses on fitted wavelengths and "
            "draws model/residuals only where pixels were used. 'all' shows the "
            "full loaded segment while still hiding model/residuals on unused "
            "pixels."
        ),
    )
    parser.add_argument(
        "--line-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open/save a companion zoomed-line diagnostic figure for identified "
            "Balmer/Ca/Mg lines, with He I added only for hot fitted solutions. Use "
            "--no-line-diagnostics to disable it."
        ),
    )
    parser.add_argument(
        "--line-groups",
        default="auto",
        help=(
            "Comma-separated line groups used for the zoomed diagnostics, or "
            "'auto'. Known groups include balmer, caii, nai, mgii, and hei."
        ),
    )
    parser.add_argument(
        "--hot-line-teff-threshold",
        type=float,
        default=10500.0,
        help=(
            "Fitted Teff threshold in K above which auto line diagnostics include "
            "hot-star He I panels."
        ),
    )
    parser.add_argument(
        "--line-window-half-width",
        type=float,
        default=30.0,
        help="Half-width in Angstrom for each zoomed diagnostic line panel.",
    )
    parser.add_argument(
        "--output-line-plot",
        default=None,
        help=(
            "Optional path for the zoomed-line diagnostic figure. If omitted "
            "and --output-plot is supplied, a *_lines companion path is used."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive fit figure (useful for batch runs).",
    )
    return parser


def _fit_kwargs_from_args(args, spectrum):
    fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
        spectrum,
        auto_defaults=args.auto_defaults,
        defaults_mode=args.defaults_mode,
        science_case="classification",
        p0_overrides=(args.teff, args.feh, args.logg, args.rv),
        lower_bound_overrides=(
            args.teff_min,
            args.feh_min,
            args.logg_min,
            args.rv_min,
        ),
        upper_bound_overrides=(
            args.teff_max,
            args.feh_max,
            args.logg_max,
            args.rv_max,
        ),
        window=(
            args.wmin,
            args.wmax,
        ) if args.wmin is not None or args.wmax is not None else None,
        resolution_R=args.resolution_R,
    )
    if args.mask_dibs or args.nonstellar_feature_policy == "mask_known":
        for mask_spec in nonstellar_feature_masks(
            DIB_FEATURE_NAMES,
            padding_A=args.dib_padding,
        ):
            _append_exclusion_mask(fit_kwargs, mask_spec)
    if args.archive_mask_policy == "apply":
        fit_kwargs = _fit_kwargs_with_archive_policy(
            fit_kwargs,
            _archive_masks_by_segment(spectrum),
            args.archive_mask_policy,
        )
    return fit_kwargs, suggestion


def _resolution_override_summary(args):
    if args.resolution_R is None:
        return None
    return {
        "resolution_source": "user_override",
        "assumed_resolution_R": float(args.resolution_R),
        "assumption_warning": "approximate quicklook resolution",
    }


def _assumed_resolution_for_audit(args):
    if args.resolution_R is None:
        return None
    return {
        "quantity": "R",
        "value": float(args.resolution_R),
        "source": "user_override",
        "assumption_warning": "approximate quicklook resolution",
    }


def _display_risk_flags(args, risk_flags):
    flags = [str(flag) for flag in risk_flags]
    if args.resolution_R is not None:
        flags = [flag for flag in flags if flag != "missing_resolution"]
    return flags


def _display_warnings(args, warnings):
    warnings = [str(warning) for warning in warnings]
    if args.resolution_R is None:
        return warnings
    return [
        warning
        for warning in warnings
        if "lacks resolution metadata" not in warning
    ]


def _append_exclusion_mask(fit_kwargs, mask_spec):
    masks = list(fit_kwargs.get("exclude_masks", []) or [])
    if fit_kwargs.get("exclude_mask") is not None:
        masks.append(fit_kwargs.pop("exclude_mask"))
    masks.append(mask_spec)
    fit_kwargs["exclude_masks"] = masks


def _reader_kwargs_from_args(args):
    instrument = str(args.instrument).strip().lower()
    if instrument not in {"sdss", "sdss_spec", "segue"}:
        return {}
    policy = args.sdss_mask_policy
    if policy == "auto":
        policy = "stellar_strict"
    return {"sdss_mask_policy": policy}


def _coerce_segments(spectrum):
    if hasattr(spectrum, "segments"):
        return list(spectrum.segments)
    if isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    return [spectrum]


def _archive_masks_by_segment(spectrum):
    out = {}
    for index, segment in enumerate(_coerce_segments(spectrum)):
        masks = archive_exclusion_masks_for_segment(segment)
        if masks:
            out[index] = masks
    return out


def _archive_mask_count(archive_masks):
    return int(sum(len(value) for value in archive_masks.values()))


def _fit_kwargs_with_archive_policy(fit_kwargs, archive_masks, policy):
    fit_kwargs = dict(fit_kwargs)
    if policy != "apply" or not archive_masks:
        return fit_kwargs
    existing = fit_kwargs.get("exclude_masks")
    if existing is None:
        fit_kwargs["exclude_masks"] = dict(archive_masks)
        return fit_kwargs
    merged = dict(archive_masks)
    if isinstance(existing, dict):
        for key, value in existing.items():
            current = list(merged.get(key, []) or [])
            current.extend(list(value if isinstance(value, (list, tuple)) else [value]))
            merged[key] = current
    else:
        for key in list(merged):
            current = list(merged[key])
            current.extend(list(existing if isinstance(existing, (list, tuple)) else [existing]))
            merged[key] = current
    fit_kwargs["exclude_masks"] = merged
    return fit_kwargs


def _parse_line_groups(value, result=None, hot_teff_threshold=10500.0):
    value = str(value).strip().lower()
    if value == "auto":
        groups = ["balmer", "caii", "mgii"]
        teff = None
        if result is not None:
            try:
                teff = float(result["teff"])
            except (KeyError, TypeError, ValueError):
                teff = None
        if teff is not None and np.isfinite(teff) and teff >= hot_teff_threshold:
            groups.append("hei")
        return groups

    groups = []
    for raw in value.split(","):
        group = raw.strip().lower()
        if not group:
            continue
        if group not in COMMON_LINES:
            known = ", ".join(sorted(COMMON_LINES))
            raise ValueError(
                "Unknown line group '{0}'. Known groups: {1}.".format(group, known)
            )
        groups.append(group)
    if not groups:
        raise ValueError("At least one line group is required for line diagnostics.")
    return groups


def _line_windows_for_used_pixels(segment, used_mask, groups, half_width_A):
    if half_width_A <= 0:
        raise ValueError("--line-window-half-width must be positive.")
    wave = np.asarray(segment.wave, dtype=float)
    used = np.asarray(used_mask, dtype=bool)
    finite_used = used & np.isfinite(wave)
    if not np.any(finite_used):
        return []
    wmin = float(np.nanmin(wave[finite_used]))
    wmax = float(np.nanmax(wave[finite_used]))
    raw_windows = []
    seen = set()
    for group in groups:
        for label, center in COMMON_LINES[group]:
            center = float(center)
            if center < wmin or center > wmax:
                continue
            key = (round(center, 4), label)
            if key in seen:
                continue
            seen.add(key)
            line_label = "{0} {1:.1f} Å".format(label, center)
            raw_windows.append((line_label, center - half_width_A, center + half_width_A))
    return _merge_overlapping_line_windows(sorted(raw_windows, key=lambda item: item[1]))


def _merge_overlapping_line_windows(windows):
    if not windows:
        return []
    merged = []
    current_labels = [windows[0][0]]
    current_min = float(windows[0][1])
    current_max = float(windows[0][2])
    current_center = 0.5 * (current_min + current_max)
    current_half_width = 0.5 * (current_max - current_min)
    for label, wmin, wmax in windows[1:]:
        wmin = float(wmin)
        wmax = float(wmax)
        center = 0.5 * (wmin + wmax)
        half_width = 0.5 * (wmax - wmin)
        very_close = abs(center - current_center) <= 0.75 * min(
            current_half_width,
            half_width,
        )
        if wmin <= current_max and very_close:
            current_labels.append(label)
            current_max = max(current_max, wmax)
            current_center = 0.5 * (current_min + current_max)
            current_half_width = 0.5 * (current_max - current_min)
        else:
            merged.append((" + ".join(current_labels), current_min, current_max))
            current_labels = [label]
            current_min = wmin
            current_max = wmax
            current_center = center
            current_half_width = half_width
    merged.append((" + ".join(current_labels), current_min, current_max))
    return merged


def _line_panel_columns(n_windows):
    n_windows = int(n_windows)
    if n_windows <= 1:
        return 1
    if n_windows <= 4:
        return 2
    return 3


def _derive_line_plot_path(output_line_plot, output_plot, segment_index=None):
    path = output_line_plot or output_plot
    if path is None:
        return None
    path = Path(path)
    suffix = path.suffix or ".png"
    if output_line_plot:
        new_path = path
    else:
        new_path = path.with_name("{0}_lines{1}".format(path.stem, suffix))
    if segment_index is not None:
        new_path = new_path.with_name(
            "{0}_segment{1}{2}".format(
                new_path.stem,
                int(segment_index) + 1,
                new_path.suffix or suffix,
            )
        )
    return str(new_path)


def _annotate_nonstellar_features(args, spectrum, result):
    policy = "mask_known" if args.mask_dibs else args.nonstellar_feature_policy
    return annotate_nonstellar_features(
        spectrum,
        result,
        feature_names=DIB_FEATURE_NAMES,
        policy=policy,
        padding_A=args.dib_padding,
        show=args.show_dibs,
        verbose=True,
    )


def _diagnose_known_residual_windows(args, spectrum, result):
    return diagnose_known_residual_windows(
        spectrum,
        result,
        enabled=args.known_residual_diagnostics,
        threshold_sigma=args.known_residual_threshold,
        windows=KNOWN_RESIDUAL_WINDOWS,
        verbose=True,
    )


def _print_interpretation_hints(args, result, fit_kwargs):
    flags = set(getattr(result, "quality_flags", ()))
    bounds = fit_kwargs.get("bounds", None)
    teff = None
    teff_upper = None
    try:
        teff = float(result["teff"])
    except (KeyError, TypeError, ValueError):
        teff = None
    if bounds is not None:
        try:
            teff_upper = float(bounds[1][0])
        except (IndexError, TypeError, ValueError):
            teff_upper = None

    if (
        args.defaults_mode == "quicklook"
        and {"high_chi2", "structured_residuals"} & flags
    ):
        print(
            "\nHint: this quicklook fit is flagged for structured/high residuals. "
            "Try rerunning with --defaults-mode diagnostic to widen the Teff "
            "search before interpreting individual line mismatches.",
            flush=True,
        )
    if (
        args.defaults_mode == "quicklook"
        and teff is not None
        and teff_upper is not None
        and np.isfinite(teff)
        and np.isfinite(teff_upper)
        and teff >= teff_upper - 250.0
    ):
        print(
            "Hint: the fitted Teff is close to the quicklook upper bound "
            "({0:.0f} K). A hotter diagnostic run may be more appropriate.".format(
                teff_upper
            ),
            flush=True,
        )


def _build_line_diagnostic_plots(args, spectrum, result):
    groups = _parse_line_groups(
        args.line_groups,
        result=result,
        hot_teff_threshold=args.hot_line_teff_threshold,
    )
    if str(args.line_groups).strip().lower() == "auto":
        print(
            "Auto line diagnostics selected groups: {0}.".format(
                ", ".join(groups)
            ),
            flush=True,
        )
    segments = _coerce_segments(spectrum)
    models = tuple(getattr(result, "models", ()))
    used_masks = tuple(getattr(result, "used_masks", ()))
    excluded_masks = tuple(getattr(result, "excluded_masks", ()))
    if len(models) != len(segments):
        raise ValueError("Line diagnostics require one reconstructed model per segment.")

    figures = []
    plot_paths = {}
    multiple_segments = len(segments) > 1
    for index, (segment, model) in enumerate(zip(segments, models)):
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        err = None if getattr(segment, "err", None) is None else np.asarray(segment.err)
        model = np.asarray(model, dtype=float)
        if index < len(used_masks):
            used = np.asarray(used_masks[index], dtype=bool)
        else:
            used = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
        if index < len(excluded_masks):
            excluded = np.asarray(excluded_masks[index], dtype=bool)
        else:
            excluded = np.zeros(wave.size, dtype=bool)

        windows = _line_windows_for_used_pixels(
            segment,
            used,
            groups,
            args.line_window_half_width,
        )
        if not windows:
            continue

        model_plot = np.full_like(model, np.nan, dtype=float)
        finite_model = np.isfinite(model)
        model_plot[used & finite_model] = model[used & finite_model]
        ncols = _line_panel_columns(len(windows))
        savepath = _derive_line_plot_path(
            args.output_line_plot,
            args.output_plot,
            segment_index=index if multiple_segments else None,
        )
        print(
            "Building zoomed line diagnostics for {0}: {1} panel(s).".format(
                getattr(segment, "name", "segment {0}".format(index + 1)),
                len(windows),
            ),
            flush=True,
        )
        fig, _axes = plot_fit_windows(
            wave,
            flux,
            err,
            model_plot,
            windows,
            ncols=ncols,
            title="Zoomed line diagnostics: {0}".format(
                getattr(segment, "name", "segment {0}".format(index + 1))
            ),
            line_groups=groups,
            excluded_mask=excluded | ~used,
            figsize_per_panel=(5.2, 3.0),
            data_label="observed",
            model_label="model on fitted pixels",
        )
        if savepath is not None:
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, bbox_inches="tight")
            key = "line_diagnostics"
            if multiple_segments:
                key = "line_diagnostics_segment_{0}".format(index + 1)
            plot_paths[key] = savepath
        figures.append(fig)

    if not figures:
        print(
            "No requested diagnostic lines overlap fitted pixels; skipping "
            "zoomed line diagnostics.",
            flush=True,
        )
    return figures, plot_paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("Reading spectrum...", flush=True)
    reader_kwargs = _reader_kwargs_from_args(args)
    if reader_kwargs:
        print(
            "Using reader options: {0}".format(
                ", ".join("{0}={1}".format(k, v) for k, v in reader_kwargs.items())
            ),
            flush=True,
        )
    spectrum = read_spectrum(
        args.spectrum,
        instrument=args.instrument,
        **reader_kwargs,
    )
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, spectrum)
    archive_masks = _archive_masks_by_segment(spectrum)
    archive_mask_count = _archive_mask_count(archive_masks)
    if archive_masks:
        print(
            "Archive mask policy: {0} ({1} recognized archive mask region(s)).".format(
                args.archive_mask_policy,
                archive_mask_count,
            ),
            flush=True,
        )
    resolution_override = _resolution_override_summary(args)
    readiness = audit_spectrum_for_fit(
        spectrum,
        fit_windows=fit_kwargs.get("regions"),
        exclude_masks=fit_kwargs.get("exclude_masks"),
        intended_use="first_pass_classification",
        assumed_resolution=_assumed_resolution_for_audit(args),
    )
    print(
        "Spectrum readiness: fit_ready={0}, quicklook_only={1}, fitted_pixels={2}, flags={3}".format(
            readiness["fit_ready"],
            readiness["quicklook_only"],
            readiness["n_fit_candidate"],
            ", ".join(readiness["interpretation_flags"]) or "none",
        ),
        flush=True,
    )
    for warning in readiness.get("warnings", []):
        print("Spectrum readiness WARNING: {0}".format(warning), flush=True)
    if suggestion is not None:
        print("Suggested first-pass fit defaults:", flush=True)
        for reason in suggestion.reasons:
            print("  - {0}".format(reason), flush=True)
        if resolution_override is not None:
            print(
                "  - using user-supplied assumed resolution R={0:g}".format(
                    resolution_override["assumed_resolution_R"]
                ),
                flush=True,
            )
        interpretation = suggestion.provenance.get("interpretation", {})
        if interpretation:
            print("Default interpretation:", flush=True)
            print(
                "  - intended use: {0}".format(
                    interpretation.get("intended_use", "first_pass_classification")
                ),
                flush=True,
            )
            mode_policy = interpretation.get("mode_policy", {})
            if mode_policy:
                print(
                    "  - defaults mode: {0} ({1} budget; {2})".format(
                        mode_policy.get("mode", args.defaults_mode),
                        mode_policy.get("search_budget", "unknown"),
                        mode_policy.get("default_status", "first_pass"),
                    ),
                    flush=True,
                )
            print(
                "  - RV role: {0}".format(
                    interpretation.get("rv_role", "alignment_parameter")
                ),
                flush=True,
            )
            print(
                "  - {0}".format(interpretation.get("rv_note", "")),
                flush=True,
            )
            risk_flags = _display_risk_flags(
                args,
                interpretation.get("risk_flags", []),
            )
            if risk_flags:
                print(
                    "  - assumption/risk flags: {0}".format(
                        ", ".join(str(flag) for flag in risk_flags)
                    ),
                    flush=True,
                )
            next_step = interpretation.get("recommended_next_step")
            if next_step:
                print("  - next step: {0}".format(next_step), flush=True)
        for warning in _display_warnings(args, suggestion.warnings):
            print("  WARNING: {0}".format(warning), flush=True)
    elif resolution_override is not None:
        print(
            "Using user-supplied assumed resolution R={0:g}.".format(
                resolution_override["assumed_resolution_R"]
            ),
            flush=True,
        )
    print("Running public fit_stellar_spectrum() workflow...", flush=True)
    result = fit_stellar_spectrum(
        spectrum,
        model="phoenix",
        phoenix_dir=args.phoenix_dir,
        auto_defaults=False,
        science_case="classification",
        progress_callback=lambda event: print(event, flush=True),
        **fit_kwargs,
    )
    if suggestion is not None:
        result.summary["fit_default_suggestion"] = suggestion.to_dict()
    result.summary["spectrum_readiness"] = readiness
    result.summary["archive_mask_policy"] = {
        "policy": args.archive_mask_policy,
        "recognized_mask_count": archive_mask_count,
        "applied": bool(args.archive_mask_policy == "apply"),
    }
    result.provenance["spectrum_readiness"] = readiness
    result.provenance["archive_mask_policy"] = dict(result.summary["archive_mask_policy"])
    if resolution_override is not None:
        result.summary.update(resolution_override)
        result.provenance["resolution_override"] = dict(resolution_override)
    nonstellar_payload = _annotate_nonstellar_features(args, spectrum, result)
    _diagnose_known_residual_windows(args, spectrum, result)
    summary = {
        key: result[key]
        for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
    }
    print(json.dumps(summary, indent=2))
    print()
    print(result.quality_report_text())
    _print_interpretation_hints(args, result, fit_kwargs)
    print(
        "\nInterpretation: this example demonstrates ingestion, a native-grid "
        "full-spectrum PHOENIX fit, structured results, and plotting. It is not "
        "a precision line-width fit; line-profile mismatches can reflect stellar "
        "rotation/macroturbulence, abundance differences, or an approximate LSF. "
        "The diagnostic plot focuses on fitted wavelengths by default; gray "
        "points and shaded regions mark pixels excluded from the fit. The "
        "optional zoomed line panels are visual diagnostics rather than "
        "separate local line-profile fits."
    )

    if not result.models:
        if args.output_json:
            result.save_json(args.output_json)
        raise RuntimeError("Fit did not converge, so no model is available to plot.")
    generated_plot_paths = {}
    print("Building diagnostic plot...", flush=True)
    fig, _axes = plot_fit_referee(
        result,
        segment=spectrum,
        savepath=args.output_plot,
        layout=args.plot_layout,
        figsize_per_segment=(
            (16.0, 6.4) if args.plot_layout == "stacked" else (12.0, 3.4)
        ),
        max_points_per_segment=20000,
        xlim_mode=args.plot_xlim,
        feature_regions=(
            nonstellar_feature_metadata(DIB_FEATURE_NAMES, padding_A=args.dib_padding)
            if (
                args.show_dibs
                and nonstellar_payload["features"]
                and nonstellar_payload.get("policy") != "ignore"
            )
            else None
        ),
    )
    generated_plot_paths.update(getattr(fig, "spyctres_generated_files", {}) or {})
    extra_figures = []
    if args.line_diagnostics:
        extra_figures, line_paths = _build_line_diagnostic_plots(args, spectrum, result)
        generated_plot_paths.update(line_paths)
    if args.output_json:
        result.save_json(
            args.output_json,
            plot_paths=generated_plot_paths or None,
        )
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
        for extra_fig in extra_figures:
            plt.close(extra_fig)


if __name__ == "__main__":
    main()
