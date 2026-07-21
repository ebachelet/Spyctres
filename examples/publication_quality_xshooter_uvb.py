"""Example 7: publication-oriented X-SHOOTER UVB workflow scaffold.

This script is deliberately stricter than ``simple_phoenix_fit.py``. It is not
the one-command path for new users; it is a reproducible scaffold for expert
work where fitted stellar parameters should eventually be defensible in a paper
or technical report.

The default run is audit-only. It reads the bundled X-SHOOTER UVB spectrum,
constructs explicit Balmer-window segments with documented masks, runs both the
ordinary fit-readiness audit and the stricter publication-readiness audit, and
writes a JSON checkpoint. The expensive PHOENIX baseline fit is opt-in via
``--run-baseline-fit``.

Example audit-only run
----------------------
python examples/publication_quality_xshooter_uvb.py \
  --output-json /tmp/spyctres_publication_xshooter_uvb.json

Optional review artifacts can also be written as CSV/PNG tables, including
core-mask sensitivity, generic diagnostic windows, systematic-variant plans,
per-line Balmer observed-profile diagnostics, and baseline-fit Balmer residual
diagnostics when ``--run-baseline-fit`` is enabled.

Example baseline fit after PHOENIX is configured
------------------------------------------------
python examples/publication_quality_xshooter_uvb.py \
  --run-baseline-fit \
  --output-json /tmp/spyctres_publication_xshooter_uvb_fit.json \
  --output-plot /tmp/spyctres_publication_xshooter_uvb_fit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Let the example run directly from a source checkout, even if Spyctres has not
# been installed into the active Python environment yet.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres import (
    build_diagnostic_window_combinations,
    fit_stellar_spectrum,
    prepare_phoenix_fit_kwargs,
    publication_readiness_audit,
    select_diagnostic_windows,
)
from Spyctres.io import SpectrumCollection, SpectrumSegment, read_spectrum
from Spyctres.plotting import plot_fit_referee
from Spyctres.preprocessing import (
    archive_exclusion_masks_for_segment,
    audit_spectrum_for_fit,
    overlapping_nonstellar_features,
)
from Spyctres.recipes import prepare_xshooter_balmer_case


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


METAL_RV_WINDOWS = (
    {
        "label": "Ca II H/K sanity window",
        "region_A": [3900.0, 3995.0],
        "purpose": (
            "Independent wavelength/RV and line-depth sanity check; can be "
            "affected by interstellar Ca II, H epsilon, and local artifacts."
        ),
        "used_in_baseline_fit": False,
    },
    {
        "label": "Mg I b / blue-metal sanity window",
        "region_A": [5150.0, 5205.0],
        "purpose": (
            "Potential metal-line/RV consistency check for cooler spectra. "
            "For hot spectra it may carry little classification leverage."
        ),
        "used_in_baseline_fit": False,
    },
)


PUBLICATION_FOLLOWUP_STEPS = (
    "Inspect formal uncertainties, bad-pixel/product masks, and wavelength "
    "medium/frame provenance before interpreting fitted parameters.",
    "Compare joint Balmer-window and per-line Balmer fits; use leave-one-line-out "
    "checks to identify line-specific model or mask failures.",
    "Run systematic variants for continuum mode, Balmer-core mask width, fit "
    "windows, resolution assumptions, and residual/outlier policy.",
    "Run target-matched synthetic injection/recovery before claiming calibrated "
    "uncertainties for this data quality and spectral type.",
    "Only after these checks pass should an optional posterior/profile scan be "
    "used for publication uncertainty tables.",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Publication-oriented X-SHOOTER UVB scaffold. Default is audit-only; "
            "use --run-baseline-fit to run the expensive PHOENIX baseline fit."
        ),
        epilog=(
            "Examples:\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--output-json /tmp/spyctres_publication_xshooter_uvb.json\n\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--output-json /tmp/spyctres_publication_xshooter_uvb.json "
            "--output-balmer-line-csv /tmp/spyctres_balmer_lines.csv "
            "--output-systematic-plan-csv /tmp/spyctres_systematics.csv\n\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--run-baseline-fit "
            "--output-json /tmp/spyctres_publication_xshooter_uvb_fit.json "
            "--output-plot /tmp/spyctres_publication_xshooter_uvb_fit.png "
            "--output-balmer-residual-csv /tmp/spyctres_balmer_residuals.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=str(EXAMPLE_UVB),
        help=(
            "X-SHOOTER UVB-like reduced 1D spectrum. Defaults to the bundled "
            "UVB example in examples/data/."
        ),
    )
    parser.add_argument(
        "--instrument",
        default="xshooter",
        help="Registered Spyctres reader name. Default: xshooter.",
    )
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_publication_xshooter_uvb.json",
        help="Atomic JSON checkpoint/output path.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional baseline-fit referee plot path. Requires --run-baseline-fit.",
    )
    parser.add_argument(
        "--output-comparison-csv",
        default=None,
        help=(
            "Optional compact CSV table for the Balmer-core mask sensitivity "
            "grid. If --run-core-mask-fit-grid is set, fitted parameters are "
            "included."
        ),
    )
    parser.add_argument(
        "--output-comparison-plot",
        default=None,
        help=(
            "Optional compact PNG plot for the Balmer-core mask sensitivity "
            "grid. Does not run extra fits by itself."
        ),
    )
    parser.add_argument(
        "--output-diagnostic-window-csv",
        default=None,
        help=(
            "Optional CSV table of generic diagnostic windows available in "
            "the loaded spectrum."
        ),
    )
    parser.add_argument(
        "--output-systematic-plan-csv",
        default=None,
        help=(
            "Optional CSV table listing the planned fit-level systematic "
            "variants. This is a plan/report only; expensive variant fits are "
            "not run by default."
        ),
    )
    parser.add_argument(
        "--output-balmer-line-csv",
        default=None,
        help=(
            "Optional CSV table of cheap per-line Balmer observed-profile "
            "diagnostics: sideband coverage, line-depth proxies, wing "
            "asymmetry, mask fractions, and known DIB overlaps."
        ),
    )
    parser.add_argument(
        "--output-balmer-residual-csv",
        default=None,
        help=(
            "Optional CSV table of per-line Balmer model-residual diagnostics "
            "after --run-baseline-fit. Requires reconstructed model arrays."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the output JSON already exists, print its status and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when --resume finds an existing output JSON.",
    )
    parser.add_argument(
        "--run-baseline-fit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a baseline native-grid PHOENIX fit on the Balmer-window "
            "segments. Default: False, so the scaffold can run without PHOENIX."
        ),
    )
    parser.add_argument(
        "--balmer-window-mode",
        choices=("current", "notebook"),
        default="notebook",
        help="Balmer-window preset from Spyctres.recipes.",
    )
    parser.add_argument("--window-pad", type=float, default=5.0)
    parser.add_argument(
        "--norm-mode",
        choices=("poly", "sideband"),
        default="poly",
        help=(
            "Preparation normalization mode for the Balmer segments. 'poly' "
            "leaves continuum handling to the PHOENIX fit; 'sideband' applies "
            "explicit local sideband normalization first."
        ),
    )
    parser.add_argument("--sideband-width", type=float, default=10.0)
    parser.add_argument("--sideband-order", type=int, default=1)
    parser.add_argument(
        "--balmer-core-mask",
        type=float,
        default=10.0,
        help=(
            "Half-width in Angstrom for the opt-in Balmer-core exclusion mask. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--core-mask-grid",
        default="0,4,6,8,10,12",
        help=(
            "Comma-separated Balmer-core half-widths in Angstrom to audit as "
            "a sensitivity grid. Use an empty string to disable. Width 0 means "
            "no Balmer-core mask."
        ),
    )
    parser.add_argument(
        "--min-core-mask-retained-fraction",
        type=float,
        default=0.70,
        help=(
            "Diagnostic threshold for flagging overly large Balmer-core masks. "
            "A mask-grid point retaining less than this fraction of the fitted "
            "pixels available in the no-core-mask reference is flagged as "
            "excessive. This is provenance only; it does not alter the fit "
            "objective. Default: 0.70."
        ),
    )
    parser.add_argument(
        "--run-core-mask-fit-grid",
        action="store_true",
        help=(
            "Also run a PHOENIX fit for every --core-mask-grid value. This can "
            "be expensive and is off by default; the default grid records "
            "readiness/pixel-count sensitivity only."
        ),
    )
    parser.add_argument(
        "--archive-mask-policy",
        choices=("apply", "warn", "ignore"),
        default="apply",
        help=(
            "How to handle recognized archive/product bad-region catalogs. "
            "'apply' adds them as named exclusion masks; 'warn' leaves them "
            "unmasked but visible to readiness audits; 'ignore' records an "
            "explicit expert override."
        ),
    )
    parser.add_argument(
        "--R",
        type=float,
        default=None,
        dest="resolution_R",
        help=(
            "Optional user-supplied constant resolving power. Publication "
            "readiness treats this as assumed unless --allow-assumed-resolution "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--allow-assumed-resolution",
        action="store_true",
        help=(
            "Allow a user-supplied --R to pass the publication gate. Use only "
            "after separately validating that approximation."
        ),
    )
    parser.add_argument("--min-fit-pixels", type=int, default=200)
    parser.add_argument(
        "--max-rejected-inside-fit-window-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--baseline-defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="standard",
    )
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--rv-grid-n", type=int, default=41)
    parser.add_argument("--multistart", type=int, default=3)
    parser.add_argument("--mdeg", type=int, default=2)
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
    parser.add_argument(
        "--systematic-mdeg-grid",
        default="1,2,3",
        help=(
            "Comma-separated continuum polynomial degrees to include in the "
            "systematic-variant plan. Default: 1,2,3."
        ),
    )
    parser.add_argument(
        "--systematic-norm-modes",
        default="poly,sideband",
        help=(
            "Comma-separated preparation normalization modes to include in "
            "the systematic-variant plan. Allowed: poly,sideband."
        ),
    )
    parser.add_argument(
        "--systematic-resolution-scales",
        default="0.9,1.0,1.1",
        help=(
            "Comma-separated multiplicative scales around --R for the "
            "resolution systematic plan. Used only when --R is supplied."
        ),
    )
    parser.add_argument(
        "--max-systematic-variants",
        type=int,
        default=12,
        help=(
            "Maximum planned fit-level systematic variants to list. This "
            "keeps publication scaffolds bounded. Default: 12."
        ),
    )
    return parser


def _json_native(value):
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_native(payload)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _read_existing(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_core_mask_grid(value):
    value = "" if value is None else str(value).strip()
    if not value:
        return []
    widths = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        width = float(item)
        if not np.isfinite(width) or width < 0.0:
            raise ValueError("--core-mask-grid values must be finite and >= 0.")
        widths.append(float(width))
    return list(dict.fromkeys(widths))


def _parse_int_grid(value, *, option_name):
    value = "" if value is None else str(value).strip()
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise ValueError("{0} values must be >= 0.".format(option_name))
        out.append(parsed)
    return list(dict.fromkeys(out))


def _parse_positive_float_grid(value, *, option_name):
    value = "" if value is None else str(value).strip()
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = float(item)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("{0} values must be finite and > 0.".format(option_name))
        out.append(float(parsed))
    return list(dict.fromkeys(out))


def _parse_norm_modes(value):
    value = "" if value is None else str(value).strip()
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"poly", "sideband"}
    bad = [item for item in modes if item not in allowed]
    if bad:
        raise ValueError(
            "--systematic-norm-modes values must be drawn from poly,sideband; "
            "got {0}.".format(",".join(bad))
        )
    return list(dict.fromkeys(modes))


def _unique_float_values(values):
    out = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        if not any(abs(value - existing) <= 1e-9 for existing in out):
            out.append(value)
    return out


def _safe_id_token(value):
    text = str(value)
    replacements = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "Å": "A",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    keep = []
    for char in text.lower():
        if char.isalnum():
            keep.append(char)
        else:
            keep.append("_")
    return "_".join(part for part in "".join(keep).split("_") if part)


def _single_segment(spectrum):
    if isinstance(spectrum, SpectrumSegment):
        return spectrum
    if isinstance(spectrum, SpectrumCollection) and len(spectrum) == 1:
        return spectrum[0]
    raise ValueError(
        "This scaffold expects one UVB segment. Use the multi-arm notebook or "
        "build an explicit expert workflow for multi-segment products."
    )


def _resolution_assumption(args):
    if args.resolution_R is None:
        return None
    return {
        "quantity": "R",
        "value": float(args.resolution_R),
        "source": "user_override",
        "resolution_source": "user_override",
        "assumed_resolution_R": float(args.resolution_R),
        "assumption_warning": (
            "user-supplied constant resolution for publication scaffold; "
            "validate before using for final parameters"
        ),
    }


def _archive_masks_by_segment(segments, policy):
    if policy != "apply":
        return ()
    masks = []
    seen = set()
    for segment in segments:
        for mask in archive_exclusion_masks_for_segment(segment):
            if mask.name in seen:
                continue
            masks.append(mask)
            seen.add(mask.name)
    return tuple(masks)


def _prepare_balmer_collection(args, segment, *, core_mask_halfwidth):
    case = prepare_xshooter_balmer_case(
        segment,
        window_mode=args.balmer_window_mode,
        window_pad=args.window_pad,
        norm_mode=args.norm_mode,
        sideband_width=args.sideband_width,
        sideband_order=args.sideband_order,
        core_mask=core_mask_halfwidth if core_mask_halfwidth > 0 else None,
    )
    collection = SpectrumCollection(
        case.fit_segments,
        meta={
            "workflow": "publication_quality_xshooter_uvb_scaffold",
            "source_spectrum": str(args.spectrum),
            "balmer_case": case.provenance,
        },
        name="xshooter_uvb_publication_balmer_windows",
    )
    archive_masks = _archive_masks_by_segment(
        collection.segments,
        args.archive_mask_policy,
    )
    exclude_masks = tuple(case.exclude_masks) + tuple(archive_masks)
    return case, collection, exclude_masks


def _fit_window_summary(segments):
    rows = []
    for segment in segments:
        wave = np.asarray(segment.wave, dtype=float)
        good = np.isfinite(wave)
        rows.append(
            {
                "name": segment.name,
                "n_pixels": int(wave.size),
                "wmin_A": float(np.nanmin(wave[good])) if np.any(good) else None,
                "wmax_A": float(np.nanmax(wave[good])) if np.any(good) else None,
                "wave_medium": segment.wave_medium,
                "observer_frame": segment.observer_frame,
                "stellar_rest_status": segment.stellar_rest_status,
            }
        )
    return rows


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _median_or_none(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.nanmedian(values))


def _percentile_or_none(values, percentile):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.nanpercentile(values, float(percentile)))


def _integrate_trapezoid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return None
    x = x[good]
    y = y[good]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    if dx.size == 0:
        return None
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def _exclusion_union_for_wave(wave, exclude_masks):
    wave = np.asarray(wave, dtype=float)
    union = np.zeros(wave.shape, dtype=bool)
    rows = []
    for mask in exclude_masks or ():
        name = getattr(mask, "name", "exclude_mask")
        try:
            raw = np.asarray(mask(wave))
        except Exception as exc:  # pragma: no cover - defensive provenance path
            rows.append(
                {
                    "name": str(name),
                    "status": "error",
                    "error": "{0}: {1}".format(type(exc).__name__, exc),
                    "n_pixels": 0,
                }
            )
            continue
        if raw.shape != wave.shape:
            rows.append(
                {
                    "name": str(name),
                    "status": "shape_mismatch",
                    "n_pixels": 0,
                }
            )
            continue
        if raw.dtype == bool:
            cur = raw
        else:
            cur = np.asarray(raw, dtype=float) > 0.5
            cur |= ~np.isfinite(np.asarray(raw, dtype=float))
        union |= cur
        rows.append(
            {
                "name": str(name),
                "status": "ok",
                "n_pixels": int(np.count_nonzero(cur)),
            }
        )
    return union, rows


def _sideband_masks_for_segment(segment, wave, base_valid):
    center = _finite_or_none(segment.meta.get("line_center_data"))
    cont_windows = segment.meta.get("cont_windows")
    if center is None or cont_windows is None:
        return {
            "mode": "fallback_percentile",
            "blue": np.zeros(wave.shape, dtype=bool),
            "red": np.zeros(wave.shape, dtype=bool),
            "combined": np.zeros(wave.shape, dtype=bool),
            "windows_A": [],
        }
    blue = np.zeros(wave.shape, dtype=bool)
    red = np.zeros(wave.shape, dtype=bool)
    windows = []
    for lo_offset, hi_offset in cont_windows:
        lo = center + float(lo_offset)
        hi = center + float(hi_offset)
        cur = base_valid & (wave >= min(lo, hi)) & (wave <= max(lo, hi))
        if hi <= center:
            blue |= cur
        elif lo >= center:
            red |= cur
        else:
            blue |= cur & (wave < center)
            red |= cur & (wave > center)
        windows.append([float(min(lo, hi)), float(max(lo, hi))])
    return {
        "mode": "explicit_sidebands",
        "blue": blue,
        "red": red,
        "combined": blue | red,
        "windows_A": windows,
    }


def _line_feature_overlap_notes(segment):
    return overlapping_nonstellar_features(
        segment,
        names=("dib_4428", "dib_4882"),
        padding_A=0.0,
    )


def _balmer_line_diagnostics(collection, exclude_masks, *, core_mask_halfwidth):
    """Return cheap observed-profile diagnostics for each Balmer fit segment.

    These summaries intentionally avoid PHOENIX models. They are meant to flag
    line-specific data/model-risk before expensive publication fits: sideband
    coverage, masked-pixel fractions, line-depth proxies, wing asymmetry, and
    known DIB overlaps.
    """
    rows = []
    core_mask_halfwidth = float(core_mask_halfwidth)
    if not np.isfinite(core_mask_halfwidth) or core_mask_halfwidth < 0.0:
        core_mask_halfwidth = 0.0

    for index, segment in enumerate(collection.segments):
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        err = None if segment.err is None else np.asarray(segment.err, dtype=float)
        segment_mask = np.asarray(
            getattr(segment, "mask", np.ones(wave.shape, dtype=bool)),
            dtype=bool,
        )
        base_valid = (
            segment_mask
            & np.isfinite(wave)
            & np.isfinite(flux)
        )
        if err is not None and err.shape == wave.shape:
            base_valid &= np.isfinite(err) & (err > 0)
        excluded, exclusion_rows = _exclusion_union_for_wave(wave, exclude_masks)
        fit_candidate = base_valid & ~excluded

        center = _finite_or_none(segment.meta.get("line_center_data"))
        sidebands = _sideband_masks_for_segment(segment, wave, base_valid)
        sideband_flux = flux[sidebands["combined"]]
        continuum = _median_or_none(sideband_flux)
        continuum_source = sidebands["mode"]
        if continuum is None or continuum <= 0.0:
            continuum = _percentile_or_none(flux[base_valid], 90.0)
            continuum_source = "fallback_90th_percentile"
        if continuum is None or continuum <= 0.0:
            continuum = _median_or_none(flux[base_valid])
            continuum_source = "fallback_median"

        normalized = None
        if continuum is not None and continuum > 0.0:
            normalized = flux / float(continuum)

        if center is None:
            core_mask = np.zeros(wave.shape, dtype=bool)
            blue_wing = np.zeros(wave.shape, dtype=bool)
            red_wing = np.zeros(wave.shape, dtype=bool)
        else:
            core_mask = base_valid & (np.abs(wave - center) <= core_mask_halfwidth)
            blue_wing = fit_candidate & (wave < center - core_mask_halfwidth)
            red_wing = fit_candidate & (wave > center + core_mask_halfwidth)

        line_mask = base_valid
        fit_line_mask = fit_candidate
        core_norm = None if normalized is None else normalized[core_mask]
        fit_norm = None if normalized is None else normalized[fit_line_mask]
        blue_norm = None if normalized is None else normalized[blue_wing]
        red_norm = None if normalized is None else normalized[red_wing]

        core_min = _percentile_or_none(core_norm, 5.0) if core_norm is not None else None
        absorption_depth = (
            None if core_min is None else float(1.0 - float(core_min))
        )
        ew_proxy = (
            None
            if normalized is None
            else _integrate_trapezoid(wave[line_mask], 1.0 - normalized[line_mask])
        )
        ew_fit_proxy = (
            None
            if normalized is None
            else _integrate_trapezoid(
                wave[fit_line_mask],
                1.0 - normalized[fit_line_mask],
            )
        )
        blue_median = _median_or_none(blue_norm) if blue_norm is not None else None
        red_median = _median_or_none(red_norm) if red_norm is not None else None
        wing_asymmetry = None
        if blue_median is not None and red_median is not None:
            denom = max(abs(0.5 * (blue_median + red_median)), 1e-30)
            wing_asymmetry = float((red_median - blue_median) / denom)

        n_base = int(np.count_nonzero(base_valid))
        n_fit = int(np.count_nonzero(fit_candidate))
        n_core = int(np.count_nonzero(core_mask))
        n_core_excluded = int(np.count_nonzero(core_mask & excluded))
        n_sideband = int(np.count_nonzero(sidebands["combined"]))
        n_blue = int(np.count_nonzero(sidebands["blue"]))
        n_red = int(np.count_nonzero(sidebands["red"]))
        feature_overlaps = _line_feature_overlap_notes(segment)

        flags = []
        if n_fit < 50:
            flags.append("low_fit_pixels")
        if n_base > 0 and n_fit / max(1, n_base) < 0.5:
            flags.append("high_line_mask_fraction")
        if n_sideband < 10 or n_blue < 3 or n_red < 3:
            flags.append("weak_or_missing_sideband_coverage")
        if absorption_depth is None:
            flags.append("line_depth_unmeasured")
        elif absorption_depth < 0.02:
            flags.append("weak_or_emission_like_core")
        elif absorption_depth > 0.85:
            flags.append("very_deep_core_or_artifact")
        if wing_asymmetry is not None and abs(wing_asymmetry) > 0.08:
            flags.append("wing_asymmetry_review")
        if feature_overlaps:
            flags.append("known_nonstellar_overlap")

        rows.append(
            {
                "segment_index": int(index),
                "segment_name": segment.name,
                "line_label": segment.meta.get("line_label", segment.name),
                "line_center_vac_A": segment.meta.get("line_center_vac"),
                "line_center_data_A": center,
                "wave_medium": segment.wave_medium,
                "diagnostic_type": "observed_profile_proxy",
                "n_pixels": int(wave.size),
                "n_base_valid": n_base,
                "n_fit_candidate": n_fit,
                "fit_candidate_fraction": float(n_fit / max(1, n_base)),
                "n_core_pixels": n_core,
                "n_core_excluded_pixels": n_core_excluded,
                "core_excluded_fraction": float(n_core_excluded / max(1, n_core)),
                "core_mask_halfwidth_A": float(core_mask_halfwidth),
                "sideband_mode": sidebands["mode"],
                "sideband_windows_A": sidebands["windows_A"],
                "n_sideband_pixels": n_sideband,
                "n_blue_sideband_pixels": n_blue,
                "n_red_sideband_pixels": n_red,
                "continuum_proxy": continuum,
                "continuum_proxy_source": continuum_source,
                "core_5th_percentile_normalized_flux": core_min,
                "absorption_depth_proxy": absorption_depth,
                "equivalent_width_proxy_A": ew_proxy,
                "fit_candidate_equivalent_width_proxy_A": ew_fit_proxy,
                "blue_wing_median_normalized_flux": blue_median,
                "red_wing_median_normalized_flux": red_median,
                "wing_asymmetry_fraction": wing_asymmetry,
                "known_nonstellar_overlaps": feature_overlaps,
                "exclusion_masks": exclusion_rows,
                "quality_flags": flags,
                "interpretation": (
                    "Observed-profile proxy only. Use this to decide which "
                    "Balmer lines need visual/fitted follow-up; do not treat "
                    "these values as calibrated atmospheric parameters."
                ),
            }
        )

    flagged = [
        row["line_label"]
        for row in rows
        if row.get("quality_flags")
    ]
    depths = [
        row["absorption_depth_proxy"]
        for row in rows
        if row.get("absorption_depth_proxy") is not None
    ]
    asym = [
        abs(row["wing_asymmetry_fraction"])
        for row in rows
        if row.get("wing_asymmetry_fraction") is not None
    ]
    return {
        "schema_version": 1,
        "status": "computed" if rows else "no_lines",
        "method": "observed_balmer_profile_proxy_diagnostics",
        "core_mask_halfwidth_A": float(core_mask_halfwidth),
        "summary": {
            "n_lines": int(len(rows)),
            "n_flagged_lines": int(len(flagged)),
            "flagged_lines": flagged,
            "median_absorption_depth_proxy": (
                None if not depths else float(np.nanmedian(depths))
            ),
            "max_abs_wing_asymmetry_fraction": (
                None if not asym else float(np.nanmax(asym))
            ),
        },
        "lines": rows,
        "interpretation": (
            "These diagnostics are cheap observed-spectrum checks for line-by-"
            "line review. They help identify Balmer lines affected by masks, "
            "sideband weakness, DIB overlap, artifacts, or asymmetric wings "
            "before expensive PHOENIX systematic variants are run."
        ),
    }


def _write_balmer_line_diagnostic_csv(path, diagnostics):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "line_label",
        "segment_name",
        "line_center_data_A",
        "n_base_valid",
        "n_fit_candidate",
        "fit_candidate_fraction",
        "n_core_pixels",
        "n_core_excluded_pixels",
        "core_excluded_fraction",
        "n_sideband_pixels",
        "continuum_proxy_source",
        "absorption_depth_proxy",
        "equivalent_width_proxy_A",
        "fit_candidate_equivalent_width_proxy_A",
        "wing_asymmetry_fraction",
        "known_nonstellar_overlaps",
        "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in diagnostics.get("lines", ()):
            writer.writerow(
                {
                    "line_label": row.get("line_label"),
                    "segment_name": row.get("segment_name"),
                    "line_center_data_A": row.get("line_center_data_A"),
                    "n_base_valid": row.get("n_base_valid"),
                    "n_fit_candidate": row.get("n_fit_candidate"),
                    "fit_candidate_fraction": row.get("fit_candidate_fraction"),
                    "n_core_pixels": row.get("n_core_pixels"),
                    "n_core_excluded_pixels": row.get("n_core_excluded_pixels"),
                    "core_excluded_fraction": row.get("core_excluded_fraction"),
                    "n_sideband_pixels": row.get("n_sideband_pixels"),
                    "continuum_proxy_source": row.get("continuum_proxy_source"),
                    "absorption_depth_proxy": row.get("absorption_depth_proxy"),
                    "equivalent_width_proxy_A": row.get("equivalent_width_proxy_A"),
                    "fit_candidate_equivalent_width_proxy_A": row.get(
                        "fit_candidate_equivalent_width_proxy_A"
                    ),
                    "wing_asymmetry_fraction": row.get("wing_asymmetry_fraction"),
                    "known_nonstellar_overlaps": ";".join(
                        item.get("id", item.get("name", "feature"))
                        for item in row.get("known_nonstellar_overlaps", ())
                    ),
                    "quality_flags": ";".join(row.get("quality_flags", ())),
                }
            )


def _optional_bool_array_tuple(value, n_segments):
    if value is None:
        return tuple(None for _index in range(n_segments))
    items = tuple(value)
    out = []
    for index in range(n_segments):
        if index >= len(items) or items[index] is None:
            out.append(None)
        else:
            out.append(np.asarray(items[index], dtype=bool))
    return tuple(out)


def _result_model_mask_arrays(result, *, n_segments):
    models = getattr(result, "models", None)
    used_masks = getattr(result, "used_masks", None)
    excluded_masks = getattr(result, "excluded_masks", None)
    if isinstance(result, dict):
        models = result.get("models", models)
        used_masks = result.get("used_masks", used_masks)
        excluded_masks = result.get("excluded_masks", excluded_masks)
    if models is None:
        return None, None, None
    model_items = tuple(models)
    if len(model_items) < n_segments:
        return None, None, None
    return (
        tuple(np.asarray(item, dtype=float) for item in model_items[:n_segments]),
        _optional_bool_array_tuple(used_masks, n_segments),
        _optional_bool_array_tuple(excluded_masks, n_segments),
    )


def _residual_metric_block(residual, flux, model, err=None):
    residual = np.asarray(residual, dtype=float)
    flux = np.asarray(flux, dtype=float)
    model = np.asarray(model, dtype=float)
    good = np.isfinite(residual) & np.isfinite(flux) & np.isfinite(model)
    if err is not None:
        err = np.asarray(err, dtype=float)
        good &= np.isfinite(err)
    if np.count_nonzero(good) == 0:
        return {
            "n_pixels": 0,
            "median_fractional_residual": None,
            "rms_fractional_residual": None,
            "median_abs_sigma": None,
            "chi2_red_proxy": None,
            "max_abs_sigma": None,
        }
    residual = residual[good]
    flux = flux[good]
    model = model[good]
    scale = max(
        abs(float(np.nanmedian(flux))),
        abs(float(np.nanmedian(model))),
        1e-30,
    )
    block = {
        "n_pixels": int(residual.size),
        "median_fractional_residual": float(np.nanmedian(residual) / scale),
        "rms_fractional_residual": float(np.sqrt(np.nanmean(residual**2)) / scale),
        "median_abs_sigma": None,
        "chi2_red_proxy": None,
        "max_abs_sigma": None,
    }
    if err is not None:
        err = err[good]
        good_err = np.isfinite(err) & (err > 0.0)
        if np.count_nonzero(good_err) > 0:
            sigma = residual[good_err] / err[good_err]
            block["median_abs_sigma"] = float(np.nanmedian(np.abs(sigma)))
            block["chi2_red_proxy"] = float(np.nanmean(sigma**2))
            block["max_abs_sigma"] = float(np.nanmax(np.abs(sigma)))
    return block


def _balmer_model_residual_diagnostics(
    collection,
    result,
    *,
    core_mask_halfwidth,
    min_pixels=3,
):
    """Return per-line residual summaries from a reconstructed PHOENIX fit.

    This is the model-facing companion to ``_balmer_line_diagnostics``.  It
    intentionally distinguishes pixels used by the fit from masked/core pixels:
    the former summarize the likelihood region, while the latter show where
    the overplotted model should be read as an extrapolated diagnostic only.
    """
    segments = list(collection.segments)
    models, used_masks, excluded_masks = _result_model_mask_arrays(
        result,
        n_segments=len(segments),
    )
    if models is None:
        return {
            "schema_version": 1,
            "status": "skipped_no_reconstructed_model",
            "method": "balmer_model_residual_diagnostics",
            "lines": [],
            "summary": None,
            "reason": (
                "Per-line model residual diagnostics require reconstructed "
                "model arrays from fit_stellar_spectrum(..., reconstruct=True)."
            ),
        }

    core_mask_halfwidth = float(core_mask_halfwidth)
    if not np.isfinite(core_mask_halfwidth) or core_mask_halfwidth < 0.0:
        core_mask_halfwidth = 0.0
    min_pixels = int(min_pixels)
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1.")

    rows = []
    for index, segment in enumerate(segments):
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        model = np.asarray(models[index], dtype=float)
        if wave.shape != flux.shape or wave.shape != model.shape:
            rows.append(
                {
                    "segment_index": int(index),
                    "segment_name": segment.name,
                    "line_label": segment.meta.get("line_label", segment.name),
                    "status": "shape_mismatch",
                    "quality_flags": ["model_shape_mismatch"],
                }
            )
            continue
        err = None if segment.err is None else np.asarray(segment.err, dtype=float)
        segment_mask = np.asarray(
            getattr(segment, "mask", np.ones(wave.shape, dtype=bool)),
            dtype=bool,
        )
        valid = segment_mask & np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
        if err is not None and err.shape == wave.shape:
            valid &= np.isfinite(err) & (err > 0.0)
        else:
            err = None

        used = used_masks[index]
        if used is None or used.shape != wave.shape:
            used = valid.copy()
        else:
            used = valid & used
        excluded = excluded_masks[index]
        if excluded is None or excluded.shape != wave.shape:
            excluded = np.zeros(wave.shape, dtype=bool)
        else:
            excluded = valid & excluded

        center = _finite_or_none(segment.meta.get("line_center_data"))
        if center is None:
            core = np.zeros(wave.shape, dtype=bool)
            blue_wing = np.zeros(wave.shape, dtype=bool)
            red_wing = np.zeros(wave.shape, dtype=bool)
        else:
            core = valid & (np.abs(wave - center) <= core_mask_halfwidth)
            blue_wing = used & (wave < center - core_mask_halfwidth)
            red_wing = used & (wave > center + core_mask_halfwidth)

        residual = flux - model
        used_block = _residual_metric_block(
            residual[used],
            flux[used],
            model[used],
            None if err is None else err[used],
        )
        core_block = _residual_metric_block(
            residual[core],
            flux[core],
            model[core],
            None if err is None else err[core],
        )
        excluded_block = _residual_metric_block(
            residual[excluded],
            flux[excluded],
            model[excluded],
            None if err is None else err[excluded],
        )
        blue_block = _residual_metric_block(
            residual[blue_wing],
            flux[blue_wing],
            model[blue_wing],
            None if err is None else err[blue_wing],
        )
        red_block = _residual_metric_block(
            residual[red_wing],
            flux[red_wing],
            model[red_wing],
            None if err is None else err[red_wing],
        )
        wing_residual_asymmetry = None
        if (
            blue_block["median_fractional_residual"] is not None
            and red_block["median_fractional_residual"] is not None
        ):
            wing_residual_asymmetry = float(
                red_block["median_fractional_residual"]
                - blue_block["median_fractional_residual"]
            )

        flags = []
        if used_block["n_pixels"] < min_pixels:
            flags.append("insufficient_used_pixels")
        if (
            used_block["chi2_red_proxy"] is not None
            and used_block["chi2_red_proxy"] > 9.0
        ):
            flags.append("line_high_chi2_proxy")
        if (
            used_block["median_abs_sigma"] is not None
            and used_block["median_abs_sigma"] > 3.0
        ):
            flags.append("line_large_median_abs_sigma")
        if (
            used_block["rms_fractional_residual"] is not None
            and used_block["rms_fractional_residual"] > 0.10
        ):
            flags.append("line_large_fractional_rms")
        if wing_residual_asymmetry is not None and abs(wing_residual_asymmetry) > 0.05:
            flags.append("line_wing_residual_asymmetry")
        if int(np.count_nonzero(core & excluded)) > 0:
            flags.append("core_model_only_not_fitted")

        rows.append(
            {
                "segment_index": int(index),
                "segment_name": segment.name,
                "line_label": segment.meta.get("line_label", segment.name),
                "line_center_data_A": center,
                "status": "ok",
                "diagnostic_type": "model_residuals",
                "n_valid_pixels": int(np.count_nonzero(valid)),
                "n_used_pixels": int(np.count_nonzero(used)),
                "n_excluded_pixels": int(np.count_nonzero(excluded)),
                "n_core_pixels": int(np.count_nonzero(core)),
                "n_core_excluded_pixels": int(np.count_nonzero(core & excluded)),
                "used_residuals": used_block,
                "masked_or_excluded_residuals": excluded_block,
                "core_residuals_model_only": core_block,
                "blue_wing_used_residuals": blue_block,
                "red_wing_used_residuals": red_block,
                "wing_residual_asymmetry_fraction": wing_residual_asymmetry,
                "quality_flags": flags,
                "interpretation": (
                    "Model-residual diagnostic for this line. Used-pixel "
                    "metrics summarize fitted pixels; core/excluded metrics "
                    "are model-overplot checks only when those pixels were "
                    "masked from the likelihood."
                ),
            }
        )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    chi2_values = [
        row["used_residuals"]["chi2_red_proxy"]
        for row in ok_rows
        if row["used_residuals"].get("chi2_red_proxy") is not None
    ]
    rms_values = [
        row["used_residuals"]["rms_fractional_residual"]
        for row in ok_rows
        if row["used_residuals"].get("rms_fractional_residual") is not None
    ]
    flags = sorted(
        {
            flag
            for row in rows
            for flag in row.get("quality_flags", ())
        }
    )
    return {
        "schema_version": 1,
        "status": "computed" if ok_rows else "no_evaluable_lines",
        "method": "balmer_model_residual_diagnostics",
        "core_mask_halfwidth_A": float(core_mask_halfwidth),
        "summary": {
            "n_lines": int(len(rows)),
            "n_evaluated_lines": int(len(ok_rows)),
            "mean_used_chi2_red_proxy": (
                None if not chi2_values else float(np.nanmean(chi2_values))
            ),
            "max_used_rms_fractional_residual": (
                None if not rms_values else float(np.nanmax(rms_values))
            ),
            "quality_flags": flags,
        },
        "lines": rows,
        "interpretation": (
            "Per-line residual diagnostics from the reconstructed baseline "
            "PHOENIX model. They identify line-specific failures and masked "
            "regions requiring visual review; they are not final uncertainty "
            "estimates."
        ),
    }


def _write_balmer_model_residual_csv(path, diagnostics):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "line_label",
        "segment_name",
        "status",
        "n_valid_pixels",
        "n_used_pixels",
        "n_excluded_pixels",
        "n_core_pixels",
        "n_core_excluded_pixels",
        "used_chi2_red_proxy",
        "used_median_abs_sigma",
        "used_rms_fractional_residual",
        "core_rms_fractional_residual",
        "masked_rms_fractional_residual",
        "wing_residual_asymmetry_fraction",
        "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in diagnostics.get("lines", ()):
            used = row.get("used_residuals") or {}
            core = row.get("core_residuals_model_only") or {}
            masked = row.get("masked_or_excluded_residuals") or {}
            writer.writerow(
                {
                    "line_label": row.get("line_label"),
                    "segment_name": row.get("segment_name"),
                    "status": row.get("status"),
                    "n_valid_pixels": row.get("n_valid_pixels"),
                    "n_used_pixels": row.get("n_used_pixels"),
                    "n_excluded_pixels": row.get("n_excluded_pixels"),
                    "n_core_pixels": row.get("n_core_pixels"),
                    "n_core_excluded_pixels": row.get("n_core_excluded_pixels"),
                    "used_chi2_red_proxy": used.get("chi2_red_proxy"),
                    "used_median_abs_sigma": used.get("median_abs_sigma"),
                    "used_rms_fractional_residual": used.get(
                        "rms_fractional_residual"
                    ),
                    "core_rms_fractional_residual": core.get(
                        "rms_fractional_residual"
                    ),
                    "masked_rms_fractional_residual": masked.get(
                        "rms_fractional_residual"
                    ),
                    "wing_residual_asymmetry_fraction": row.get(
                        "wing_residual_asymmetry_fraction"
                    ),
                    "quality_flags": ";".join(row.get("quality_flags", ())),
                }
            )


def _diagnostic_window_payload(segment):
    selection = select_diagnostic_windows(segment, max_windows=12)
    combinations = build_diagnostic_window_combinations(
        selection,
        max_windows=8,
        max_single_windows=6,
    )
    return {
        "selection": selection,
        "recommended_combinations": combinations,
        "note": (
            "These generic windows are selected from the uploaded spectral "
            "range before any expensive PHOENIX window-combination fits. They "
            "include optical, Paschen/Brackett, Ca, TiO, and CO diagnostics "
            "when the wavelength coverage supports them."
        ),
    }


def _systematic_window_variants(case):
    windows = list(case.provenance.get("balmer_windows", ()))
    labels = [str(item[0]) for item in windows]
    variants = [
        {
            "window_set": "joint_balmer",
            "window_labels": labels,
            "rationale": (
                "Baseline joint fit keeps all selected Balmer wings in one "
                "likelihood, which is the preferred first publication check."
            ),
        }
    ]
    for label in labels:
        variants.append(
            {
                "window_set": "single_{0}".format(_safe_id_token(label)),
                "window_labels": [label],
                "rationale": (
                    "Single-line fit for line-specific model, mask, or "
                    "continuum failures; compare against the joint Balmer fit."
                ),
            }
        )
    for label in labels:
        retained = [item for item in labels if item != label]
        if retained:
            variants.append(
                {
                    "window_set": "leave_out_{0}".format(_safe_id_token(label)),
                    "window_labels": retained,
                    "rationale": (
                        "Leave-one-line-out stability check; large parameter "
                        "shifts identify a line that should be inspected "
                        "before publication use."
                    ),
                }
            )
    return variants


def _variant_record(
    *,
    variant_id,
    category,
    label,
    baseline_args,
    preparation_overrides=None,
    fit_overrides=None,
    window_labels=None,
    rationale,
    executable=True,
    skip_reasons=None,
):
    preparation_overrides = dict(preparation_overrides or {})
    fit_overrides = dict(fit_overrides or {})
    skip_reasons = list(skip_reasons or ())
    if not executable and not skip_reasons:
        skip_reasons = ["not_executable_without_additional_validation"]
    return {
        "id": variant_id,
        "category": category,
        "label": label,
        "status": "planned_not_run" if executable else "not_planned_for_execution",
        "requires_phoenix": bool(executable),
        "run_by_default": False,
        "executable_now": bool(executable),
        "skip_reasons": skip_reasons,
        "baseline_context": {
            "norm_mode": baseline_args["norm_mode"],
            "mdeg": int(baseline_args["mdeg"]),
            "balmer_core_mask_A": baseline_args["balmer_core_mask_A"],
            "resolution_R": baseline_args["resolution_R"],
            "window_set": "joint_balmer",
        },
        "preparation_overrides": preparation_overrides,
        "fit_overrides": fit_overrides,
        "window_labels": list(window_labels or ()),
        "rationale": rationale,
        "interpretation": (
            "Run only after the baseline fit is understood. Compare parameter "
            "shifts, residual flags, and common-window diagnostics; do not "
            "treat lower chi-square alone as proof that the variant is better."
        ),
    }


def _build_systematic_variant_plan(args, case, *, core_mask_recommendation=None):
    """Return a bounded, auditable fit-systematics plan.

    This is intentionally a plan by default, not another hidden expensive fit
    loop.  The listed variants are the first checks a reviewer would expect
    before publication-style parameter claims: continuum degree, preparation
    normalization, core-mask choice, resolution assumption, and line-window
    sensitivity.
    """
    max_variants = int(args.max_systematic_variants)
    if max_variants < 1:
        raise ValueError("--max-systematic-variants must be >= 1.")

    mdeg_values = _parse_int_grid(
        args.systematic_mdeg_grid,
        option_name="--systematic-mdeg-grid",
    )
    norm_modes = _parse_norm_modes(args.systematic_norm_modes)
    resolution_scales = _parse_positive_float_grid(
        args.systematic_resolution_scales,
        option_name="--systematic-resolution-scales",
    )

    recommended_core = None
    if isinstance(core_mask_recommendation, dict):
        recommended_core = core_mask_recommendation.get(
            "recommended_core_mask_halfwidth_A"
        )
    core_values = _unique_float_values(
        [
            float(args.balmer_core_mask),
            recommended_core,
            0.0,
        ]
    )

    baseline = {
        "norm_mode": str(args.norm_mode),
        "mdeg": int(args.mdeg),
        "balmer_core_mask_A": float(args.balmer_core_mask),
        "resolution_R": None if args.resolution_R is None else float(args.resolution_R),
    }

    variants = [
        _variant_record(
            variant_id="baseline",
            category="baseline",
            label="Current baseline configuration",
            baseline_args=baseline,
            preparation_overrides={
                "norm_mode": baseline["norm_mode"],
                "balmer_core_mask_A": baseline["balmer_core_mask_A"],
            },
            fit_overrides={
                "mdeg": baseline["mdeg"],
                "resolution_R": baseline["resolution_R"],
            },
            window_labels=[item[0] for item in case.provenance.get("balmer_windows", ())],
            rationale=(
                "Reference point for all systematic comparisons. This is the "
                "configuration used by --run-baseline-fit."
            ),
        )
    ]

    for mdeg in mdeg_values:
        if int(mdeg) == baseline["mdeg"]:
            continue
        variants.append(
            _variant_record(
                variant_id="continuum_mdeg_{0}".format(mdeg),
                category="continuum_degree",
                label="Continuum degree mdeg={0}".format(mdeg),
                baseline_args=baseline,
                fit_overrides={"mdeg": int(mdeg)},
                window_labels=[
                    item[0] for item in case.provenance.get("balmer_windows", ())
                ],
                rationale=(
                    "Tests whether the stellar parameters are being driven by "
                    "the multiplicative continuum flexibility rather than "
                    "line physics."
                ),
            )
        )

    for mode in norm_modes:
        if mode == baseline["norm_mode"]:
            continue
        variants.append(
            _variant_record(
                variant_id="normalization_{0}".format(mode),
                category="preprocessing_normalization",
                label="Preparation normalization: {0}".format(mode),
                baseline_args=baseline,
                preparation_overrides={"norm_mode": mode},
                window_labels=[
                    item[0] for item in case.provenance.get("balmer_windows", ())
                ],
                rationale=(
                    "Checks sensitivity to leaving continuum structure for "
                    "the global multiplicative polynomial versus applying "
                    "explicit local sideband normalization first."
                ),
            )
        )

    for width in core_values:
        if abs(width - baseline["balmer_core_mask_A"]) <= 1e-9:
            continue
        variants.append(
            _variant_record(
                variant_id="balmer_core_mask_{0:g}A".format(width),
                category="balmer_core_mask",
                label="Balmer-core mask half-width {0:g} A".format(width),
                baseline_args=baseline,
                preparation_overrides={"balmer_core_mask_A": float(width)},
                window_labels=[
                    item[0] for item in case.provenance.get("balmer_windows", ())
                ],
                rationale=(
                    "Checks whether the inferred parameters are sensitive to "
                    "how aggressively NLTE/model-sensitive Balmer cores are "
                    "excluded while preserving wing information."
                ),
            )
        )

    if args.resolution_R is None:
        variants.append(
            _variant_record(
                variant_id="resolution_scale_unavailable",
                category="resolution_assumption",
                label="Resolution/LSF variants unavailable",
                baseline_args=baseline,
                rationale=(
                    "Resolution systematics require an explicit baseline --R "
                    "or validated segment LSF metadata. The reader must not "
                    "invent publication-quality resolution assumptions."
                ),
                executable=False,
                skip_reasons=["missing_explicit_or_validated_resolution"],
            )
        )
    else:
        R0 = float(args.resolution_R)
        for scale in resolution_scales:
            if abs(float(scale) - 1.0) <= 1e-9:
                continue
            variants.append(
                _variant_record(
                    variant_id="resolution_scale_{0:g}".format(scale),
                    category="resolution_assumption",
                    label="Resolution R={0:g} ({1:g}x baseline)".format(
                        R0 * float(scale),
                        float(scale),
                    ),
                    baseline_args=baseline,
                    fit_overrides={"resolution_R": float(R0 * float(scale))},
                    window_labels=[
                        item[0] for item in case.provenance.get("balmer_windows", ())
                    ],
                    rationale=(
                        "Tests sensitivity to the assumed constant resolving "
                        "power. This is a bounded approximation, not "
                        "wavelength-dependent LSF modelling."
                    ),
                )
            )

    for window_variant in _systematic_window_variants(case):
        if window_variant["window_set"] == "joint_balmer":
            continue
        variants.append(
            _variant_record(
                variant_id="window_set_{0}".format(window_variant["window_set"]),
                category="fit_windows",
                label="Window set: {0}".format(window_variant["window_set"]),
                baseline_args=baseline,
                preparation_overrides={"window_set": window_variant["window_set"]},
                window_labels=window_variant["window_labels"],
                rationale=window_variant["rationale"],
            )
        )

    truncated = len(variants) > max_variants
    variants = variants[:max_variants]
    return {
        "schema_version": 1,
        "status": "planned",
        "default_execution": "not_run",
        "max_variants": int(max_variants),
        "truncated": bool(truncated),
        "variant_policy": {
            "purpose": (
                "Bounded fit-level systematic plan for publication-oriented "
                "work. It records what should be varied before treating "
                "stellar parameters as defensible."
            ),
            "not_raw_chi2_ranked": True,
            "expensive_fits_are_opt_in": True,
            "compare_using": [
                "parameter_shifts",
                "quality_flags",
                "referee_plots",
                "held_out_residuals",
                "common_window_residuals",
            ],
        },
        "dimensions": {
            "continuum_degrees": mdeg_values,
            "preparation_norm_modes": norm_modes,
            "core_mask_halfwidths_A": core_values,
            "resolution_scales": (
                [] if args.resolution_R is None else resolution_scales
            ),
            "window_sets": _systematic_window_variants(case),
        },
        "baseline": baseline,
        "variants": variants,
        "questions_before_execution": [
            "Is the assumed constant R validated for this spectrum and slit/setup?",
            "Should sideband normalization be trusted for this target's continuum shape?",
            "Do individual Balmer lines show artifacts, emission, DIB/telluric overlap, or NLTE-sensitive cores?",
            "Which parameter shift threshold should trigger manual review for this observing program?",
        ],
    }


def _write_systematic_variant_plan_csv(path, plan):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "category",
        "label",
        "status",
        "executable_now",
        "run_by_default",
        "window_labels",
        "preparation_overrides",
        "fit_overrides",
        "skip_reasons",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in plan.get("variants", ()):
            writer.writerow(
                {
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "label": row.get("label"),
                    "status": row.get("status"),
                    "executable_now": row.get("executable_now"),
                    "run_by_default": row.get("run_by_default"),
                    "window_labels": ";".join(row.get("window_labels", ())),
                    "preparation_overrides": json.dumps(
                        row.get("preparation_overrides") or {},
                        sort_keys=True,
                    ),
                    "fit_overrides": json.dumps(
                        row.get("fit_overrides") or {},
                        sort_keys=True,
                    ),
                    "skip_reasons": ";".join(row.get("skip_reasons", ())),
                    "rationale": row.get("rationale"),
                }
            )


def _base_payload(args, spectrum_path, segment, case, collection, exclude_masks):
    generic_windows = _diagnostic_window_payload(segment)
    return {
        "schema_version": 1,
        "workflow": "publication_quality_xshooter_uvb_scaffold",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "audit_started",
        "input": {
            "spectrum": str(spectrum_path),
            "instrument": str(args.instrument),
        },
        "analysis_design": {
            "phase": "scaffold",
            "default_claim_status": "exploratory_until_publication_checks_pass",
            "balmer_windows": case.provenance["balmer_windows"],
            "metal_rv_windows": list(METAL_RV_WINDOWS),
            "generic_diagnostic_windows": generic_windows,
            "core_mask_grid_A": _parse_core_mask_grid(args.core_mask_grid),
            "core_mask_information_policy": {
                "method": "explicit_sensitivity_grid_with_information_penalty",
                "minimum_retained_fraction": float(
                    args.min_core_mask_retained_fraction
                ),
                "penalty_scope": "diagnostic_model_selection_not_chi2_objective",
                "preferred_width_rule": (
                    "Treat width=0 as a no-core-mask reference, then prefer "
                    "the smallest nonzero width that is not flagged excessive. "
                    "Do not select a larger mask merely because it lowers "
                    "in-window chi-square after discarding useful wing pixels."
                ),
                "literature_basis": [
                    "cappellari2023_ppxf_optimization",
                    "garcia_perez2016_aspcap",
                    "mashonkina2008_balmer_core_exclusion",
                ],
            },
            "core_mask_evaluation_note": (
                "Balmer-core masking is a sensitivity axis, not a hard-coded "
                "truth. Audit the grid first; run fit-grid variants only when "
                "the extra PHOENIX cost is justified."
            ),
            "followup_steps_before_publication_claims": list(PUBLICATION_FOLLOWUP_STEPS),
        },
        "balmer_case": case.provenance,
        "fit_segments": _fit_window_summary(collection.segments),
        "exclude_masks": {
            "names": [mask.name for mask in exclude_masks],
            "metadata": {mask.name: dict(mask.metadata) for mask in exclude_masks},
            "archive_mask_policy": args.archive_mask_policy,
        },
        "per_line_balmer_diagnostics": _balmer_line_diagnostics(
            collection,
            exclude_masks,
            core_mask_halfwidth=float(args.balmer_core_mask),
        ),
        "ordinary_readiness": None,
        "publication_readiness": None,
        "core_mask_sensitivity": None,
        "core_mask_sensitivity_recommendation": None,
        "systematic_variant_plan": _build_systematic_variant_plan(args, case),
        "baseline_fit": None,
        "baseline_line_residual_diagnostics": None,
    }


def _write_diagnostic_window_csv(path, diagnostic_payload):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = diagnostic_payload["selection"]["selected"]
    columns = [
        "id",
        "label",
        "region_A",
        "roles",
        "score",
        "n_usable_pixels",
        "usable_fraction",
        "feature_contrast",
        "risk_tags",
        "spectral_type_hint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "region_A": "{0:.2f}-{1:.2f}".format(*row["region_A"]),
                    "roles": ";".join(row.get("roles", ())),
                    "score": "{0:.5g}".format(float(row.get("score", 0.0))),
                    "n_usable_pixels": int(row.get("n_usable_pixels", 0)),
                    "usable_fraction": "{0:.5g}".format(
                        float(row.get("usable_fraction", 0.0))
                    ),
                    "feature_contrast": "{0:.5g}".format(
                        float(row.get("feature_contrast", 0.0))
                    ),
                    "risk_tags": ";".join(row.get("risk_tags", ())),
                    "spectral_type_hint": row.get("spectral_type_hint", ""),
                }
            )


def _core_mask_fit_value(record, key):
    fit = record.get("fit") or {}
    value = fit.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _write_core_mask_comparison_csv(path, records):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "core_mask_halfwidth_A",
        "n_fit_candidate",
        "information_retention_fraction",
        "additional_information_loss_fraction",
        "core_mask_information_penalty",
        "excessive_core_mask",
        "recommended_core_mask",
        "rejected_inside_fit_window_fraction",
        "publication_ready",
        "publication_blockers",
        "teff",
        "logg",
        "feh",
        "rv_kms",
        "chi2_red",
        "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            fit = record.get("fit") or {}
            writer.writerow(
                {
                    "core_mask_halfwidth_A": record["core_mask_halfwidth_A"],
                    "n_fit_candidate": record["n_fit_candidate"],
                    "information_retention_fraction": record.get(
                        "information_retention_fraction"
                    ),
                    "additional_information_loss_fraction": record.get(
                        "additional_information_loss_fraction"
                    ),
                    "core_mask_information_penalty": record.get(
                        "core_mask_information_penalty"
                    ),
                    "excessive_core_mask": record.get("excessive_core_mask"),
                    "recommended_core_mask": record.get("recommended_core_mask"),
                    "rejected_inside_fit_window_fraction": record[
                        "rejected_inside_fit_window_fraction"
                    ],
                    "publication_ready": record["publication_ready"],
                    "publication_blockers": ";".join(record["publication_blockers"]),
                    "teff": _core_mask_fit_value(record, "teff"),
                    "logg": _core_mask_fit_value(record, "logg"),
                    "feh": _core_mask_fit_value(record, "feh"),
                    "rv_kms": _core_mask_fit_value(record, "rv_kms"),
                    "chi2_red": _core_mask_fit_value(record, "chi2_red"),
                    "quality_flags": ";".join(fit.get("quality_flags", ())),
                }
            )


def _write_core_mask_comparison_plot(path, records):
    if path is None:
        return
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    widths = np.asarray([item["core_mask_halfwidth_A"] for item in records], dtype=float)
    nfit = np.asarray([item["n_fit_candidate"] for item in records], dtype=float)
    rejected = np.asarray(
        [item["rejected_inside_fit_window_fraction"] for item in records],
        dtype=float,
    )
    teff = np.asarray([_core_mask_fit_value(item, "teff") for item in records], dtype=float)
    logg = np.asarray([_core_mask_fit_value(item, "logg") for item in records], dtype=float)
    chi2 = np.asarray(
        [_core_mask_fit_value(item, "chi2_red") for item in records],
        dtype=float,
    )
    retained = np.asarray(
        [item.get("information_retention_fraction") for item in records],
        dtype=float,
    )
    penalty = np.asarray(
        [item.get("core_mask_information_penalty") for item in records],
        dtype=float,
    )
    recommended_width = _recommended_core_mask_width(records)
    has_fit = np.any(np.isfinite(teff)) or np.any(np.isfinite(chi2))
    nrows = 4 if has_fit else 3
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, 3.0 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    axes[0].plot(widths, nfit, marker="o")
    axes[0].set_ylabel("fitted pixels")
    axes[0].grid(alpha=0.25)
    axes[1].plot(widths, rejected, marker="o", color="tab:orange")
    axes[1].set_ylabel("rejected inside")
    axes[1].grid(alpha=0.25)
    axes[2].plot(widths, retained, marker="o", color="tab:green", label="retained")
    axes[2].plot(widths, penalty, marker="s", color="tab:red", label="penalty")
    axes[2].set_ylabel("information")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)
    if recommended_width is not None:
        for ax in axes:
            ax.axvline(recommended_width, color="tab:green", ls=":", alpha=0.45)
    if has_fit:
        if np.any(np.isfinite(teff)):
            axes[3].plot(widths, teff, marker="o", label="Teff")
            axes[3].set_ylabel("Teff [K]")
            ax_r = axes[3].twinx()
            if np.any(np.isfinite(logg)):
                ax_r.plot(widths, logg, marker="s", color="tab:green", label="logg")
                ax_r.set_ylabel("logg")
        if np.any(np.isfinite(chi2)):
            ax_chi = axes[3].twinx()
            ax_chi.spines.right.set_position(("outward", 54))
            ax_chi.plot(widths, chi2, marker="^", color="tab:red", label="χ²ν")
            ax_chi.set_ylabel("χ²ν")
        axes[3].grid(alpha=0.25)
    axes[-1].set_xlabel("Balmer-core mask half-width [Å]")
    fig.suptitle(
        "Balmer-core mask sensitivity "
        "(penalty is diagnostic; it is not added to χ²)"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _recommended_core_mask_width(records):
    for record in records:
        if record.get("recommended_core_mask"):
            return float(record["core_mask_halfwidth_A"])
    return None


def _run_readiness(args, collection, exclude_masks):
    resolution = _resolution_assumption(args)
    ordinary = audit_spectrum_for_fit(
        collection,
        exclude_masks=exclude_masks,
        intended_use="publication_scaffold_baseline",
        assumed_resolution=resolution,
    )
    publication = publication_readiness_audit(
        collection,
        exclude_masks=exclude_masks,
        intended_use="publication_quality_stellar_parameters",
        assumed_resolution=resolution,
        min_fit_pixels=args.min_fit_pixels,
        max_rejected_inside_fit_window_fraction=(
            args.max_rejected_inside_fit_window_fraction
        ),
        allow_assumed_resolution=args.allow_assumed_resolution,
    )
    return ordinary, publication


def _fit_kwargs_from_args(args, collection, exclude_masks):
    extra = {
        "exclude_masks": tuple(exclude_masks),
        "forward_model": "native_interp",
        "rv_init": "grid",
        "rv_grid_n": int(args.rv_grid_n),
        "multistart": int(args.multistart),
        "mdeg": int(args.mdeg),
        "max_nfev": int(args.max_nfev),
    }
    return prepare_phoenix_fit_kwargs(
        collection,
        auto_defaults=True,
        defaults_mode=args.baseline_defaults_mode,
        science_case="publication_scaffold",
        resolution_R=args.resolution_R,
        p0_overrides=(args.teff, args.feh, args.logg, args.rv),
        lower_bound_overrides=(args.teff_min, args.feh_min, args.logg_min, args.rv_min),
        upper_bound_overrides=(args.teff_max, args.feh_max, args.logg_max, args.rv_max),
        extra_kwargs=extra,
    )


def _run_baseline_fit(
    args,
    collection,
    exclude_masks,
    *,
    output_plot=None,
    return_result=False,
):
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, collection, exclude_masks)
    print("Running baseline native-grid PHOENIX fit...", flush=True)
    result = fit_stellar_spectrum(
        collection,
        model="phoenix",
        phoenix_dir=args.phoenix_dir,
        auto_defaults=False,
        science_case="publication_scaffold",
        progress_callback=lambda event: print(event, flush=True),
        **fit_kwargs,
    )
    if suggestion is not None:
        result.summary["fit_default_suggestion"] = suggestion.to_dict()
    plot_paths = {}
    if output_plot:
        print("Writing baseline referee plot...", flush=True)
        fig, _axes = plot_fit_referee(
            result,
            segment=collection,
            savepath=output_plot,
            layout="stacked",
            xlim_mode="fit",
            flux_ylim_mode="visible",
            show_raw_model=False,
            figsize_per_segment=(15.0, 5.4),
        )
        plot_paths.update(getattr(fig, "spyctres_generated_files", {}) or {})
    payload = result.to_dict(
        include_arrays=False,
        plot_paths=plot_paths or None,
        relative_to=Path(args.output_json).parent,
        include_local_paths=False,
    )
    if return_result:
        return payload, result
    return payload


def _summarize_core_mask_variant(width, ordinary, publication, fit_payload=None):
    return {
        "core_mask_halfwidth_A": float(width),
        "mask_enabled": bool(width > 0.0),
        "n_inside_fit_window": int(ordinary["n_inside_fit_window"]),
        "n_fit_candidate": int(ordinary["n_fit_candidate"]),
        "outside_fit_window_fraction": ordinary["outside_fit_window_fraction"],
        "rejected_inside_fit_window_fraction": (
            ordinary["rejected_inside_fit_window_fraction"]
        ),
        "interpretation_flags": list(ordinary["interpretation_flags"]),
        "publication_ready": bool(publication["publication_ready"]),
        "publication_blockers": list(publication["blockers"]),
        "segment_fit_pixels": [
            {
                "name": item["name"],
                "n_inside_fit_window": item["n_inside_fit_window"],
                "n_fit_candidate": item["n_fit_candidate"],
                "rejected_inside_fit_window_fraction": (
                    item["rejected_inside_fit_window_fraction"]
                ),
            }
            for item in ordinary["segments"]
        ],
        "fit": fit_payload,
    }


def _annotate_core_mask_information(records, *, min_retained_fraction):
    # Literature basis: Balmer-wing analyses commonly exclude only the
    # model-sensitive core while preserving the wings that carry Teff/logg
    # leverage; see Mashonkina et al. 2008,
    # https://doi.org/10.1051/0004-6361:20078060.
    if not records:
        return {
            "status": "no_core_mask_grid",
            "recommended_core_mask_halfwidth_A": None,
        }
    min_retained_fraction = float(min_retained_fraction)
    if not np.isfinite(min_retained_fraction) or not (0.0 < min_retained_fraction <= 1.0):
        raise ValueError("--min-core-mask-retained-fraction must be in (0, 1].")
    reference = next(
        (record for record in records if float(record["core_mask_halfwidth_A"]) == 0.0),
        None,
    )
    if reference is None:
        reference = max(records, key=lambda item: int(item["n_fit_candidate"]))
    reference_nfit = max(int(reference["n_fit_candidate"]), 1)
    reference_rejected = float(reference["rejected_inside_fit_window_fraction"])
    for record in records:
        nfit = int(record["n_fit_candidate"])
        retained = float(np.clip(nfit / reference_nfit, 0.0, 1.0))
        added_loss = float(max(0.0, 1.0 - retained))
        added_rejected = float(
            max(
                0.0,
                float(record["rejected_inside_fit_window_fraction"]) - reference_rejected,
            )
        )
        penalty = float(max(added_loss, added_rejected))
        record["core_mask_reference_halfwidth_A"] = float(
            reference["core_mask_halfwidth_A"]
        )
        record["reference_n_fit_candidate"] = int(reference_nfit)
        record["information_retention_fraction"] = retained
        record["additional_information_loss_fraction"] = added_loss
        record["additional_rejected_inside_fit_window_fraction"] = added_rejected
        record["core_mask_information_penalty"] = penalty
        record["excessive_core_mask"] = bool(retained < min_retained_fraction)
        record["core_mask_penalty_note"] = (
            "Diagnostic only: this penalty is not added to the optimizer's "
            "spectral chi-square. It helps compare mask-width variants so "
            "large masks do not look artificially attractive just because they "
            "discard difficult Balmer-wing pixels."
        )
        record["recommended_core_mask"] = False

    nonzero_eligible = [
        record
        for record in records
        if float(record["core_mask_halfwidth_A"]) > 0.0
        and not record["excessive_core_mask"]
        and int(record["n_fit_candidate"]) > 0
    ]
    if nonzero_eligible:
        recommended = min(
            nonzero_eligible,
            key=lambda item: (
                int(len(item.get("publication_blockers", ()))),
                float(item["core_mask_information_penalty"]),
                float(item["core_mask_halfwidth_A"]),
            ),
        )
        recommended["recommended_core_mask"] = True
        width = float(recommended["core_mask_halfwidth_A"])
        reason = (
            "smallest/best-retained nonzero core mask that is not flagged "
            "as excessive under the information-retention threshold"
        )
    else:
        width = None
        reason = "no nonzero core-mask width passed the information-retention threshold"
    return {
        "status": "evaluated",
        "reference_core_mask_halfwidth_A": float(reference["core_mask_halfwidth_A"]),
        "reference_n_fit_candidate": int(reference_nfit),
        "minimum_retained_fraction": float(min_retained_fraction),
        "recommended_core_mask_halfwidth_A": width,
        "recommendation_reason": reason,
        "selection_warning": (
            "This recommendation is for mask-width triage only. Final choice "
            "still requires visual residual checks, line-by-line stability, "
            "LSF validation, and synthetic recovery tests."
        ),
    }


def _run_core_mask_sensitivity(args, segment):
    widths = _parse_core_mask_grid(args.core_mask_grid)
    if not widths:
        return {
            "records": [],
            "recommendation": {
                "status": "no_core_mask_grid",
                "recommended_core_mask_halfwidth_A": None,
            },
        }
    print(
        "Auditing Balmer-core mask sensitivity for widths: {0}".format(
            ", ".join("{0:g}".format(width) for width in widths)
        ),
        flush=True,
    )
    records = []
    for width in widths:
        _case_i, collection_i, exclude_masks_i = _prepare_balmer_collection(
            args,
            segment,
            core_mask_halfwidth=width,
        )
        ordinary_i, publication_i = _run_readiness(
            args,
            collection_i,
            exclude_masks_i,
        )
        fit_payload = None
        if args.run_core_mask_fit_grid:
            print(
                "Running core-mask fit variant: halfwidth={0:g} A".format(width),
                flush=True,
            )
            fit_payload = _run_baseline_fit(
                args,
                collection_i,
                exclude_masks_i,
                output_plot=None,
            )
        records.append(
            _summarize_core_mask_variant(
                width,
                ordinary_i,
                publication_i,
                fit_payload=fit_payload,
            )
        )
    recommendation = _annotate_core_mask_information(
        records,
        min_retained_fraction=args.min_core_mask_retained_fraction,
    )
    print(
        "Core-mask recommendation: {0} A ({1})".format(
            recommendation["recommended_core_mask_halfwidth_A"],
            recommendation["recommendation_reason"],
        ),
        flush=True,
    )
    return {
        "records": records,
        "recommendation": recommendation,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_path = Path(args.output_json)
    if args.resume and output_path.exists() and not args.force:
        existing = _read_existing(output_path)
        print(
            "Existing checkpoint: status={0}, publication_ready={1}".format(
                existing.get("status"),
                existing.get("publication_readiness", {}).get("publication_ready"),
            ),
            flush=True,
        )
        return 0

    print("Reading X-SHOOTER UVB spectrum...", flush=True)
    spectrum = read_spectrum(args.spectrum, instrument=args.instrument)
    segment = _single_segment(spectrum)

    print("Preparing explicit Balmer-window scaffold...", flush=True)
    case, collection, exclude_masks = _prepare_balmer_collection(
        args,
        segment,
        core_mask_halfwidth=float(args.balmer_core_mask),
    )
    payload = _base_payload(args, args.spectrum, segment, case, collection, exclude_masks)
    _write_diagnostic_window_csv(
        args.output_diagnostic_window_csv,
        payload["analysis_design"]["generic_diagnostic_windows"],
    )
    _write_balmer_line_diagnostic_csv(
        args.output_balmer_line_csv,
        payload["per_line_balmer_diagnostics"],
    )
    _atomic_write_json(output_path, payload)
    print("Wrote scaffold checkpoint: {0}".format(output_path), flush=True)

    print("Running ordinary and publication-readiness audits...", flush=True)
    ordinary, publication = _run_readiness(args, collection, exclude_masks)
    payload["ordinary_readiness"] = ordinary
    payload["publication_readiness"] = publication
    payload["status"] = (
        "publication_gate_passed_needs_systematics"
        if publication["publication_ready"]
        else "exploratory_publication_gate_blocked"
    )
    _atomic_write_json(output_path, payload)
    print(
        "Publication readiness: ready={0}, blockers={1}".format(
            publication["publication_ready"],
            ", ".join(publication["blockers"]) or "none",
        ),
        flush=True,
    )

    core_mask_result = _run_core_mask_sensitivity(args, segment)
    payload["core_mask_sensitivity"] = core_mask_result["records"]
    payload["core_mask_sensitivity_recommendation"] = core_mask_result[
        "recommendation"
    ]
    payload["systematic_variant_plan"] = _build_systematic_variant_plan(
        args,
        case,
        core_mask_recommendation=payload["core_mask_sensitivity_recommendation"],
    )
    _write_core_mask_comparison_csv(
        args.output_comparison_csv,
        payload["core_mask_sensitivity"],
    )
    _write_core_mask_comparison_plot(
        args.output_comparison_plot,
        payload["core_mask_sensitivity"],
    )
    _write_systematic_variant_plan_csv(
        args.output_systematic_plan_csv,
        payload["systematic_variant_plan"],
    )
    _atomic_write_json(output_path, payload)

    if args.run_baseline_fit:
        baseline_payload, baseline_result = _run_baseline_fit(
            args,
            collection,
            exclude_masks,
            output_plot=args.output_plot,
            return_result=True,
        )
        payload["baseline_fit"] = baseline_payload
        payload["baseline_line_residual_diagnostics"] = (
            _balmer_model_residual_diagnostics(
                collection,
                baseline_result,
                core_mask_halfwidth=float(args.balmer_core_mask),
            )
        )
        _write_balmer_model_residual_csv(
            args.output_balmer_residual_csv,
            payload["baseline_line_residual_diagnostics"],
        )
        payload["status"] = (
            "baseline_fit_completed_needs_publication_systematics"
            if payload["baseline_fit"].get("success")
            else "baseline_fit_failed"
        )
        _atomic_write_json(output_path, payload)
        summary = {
            key: payload["baseline_fit"].get(key)
            for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
        }
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(
            "Audit-only run complete. Add --run-baseline-fit after PHOENIX "
            "configuration is ready.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
