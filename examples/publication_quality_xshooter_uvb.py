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
  --output-report-json /tmp/spyctres_publication_xshooter_uvb_report.json \
  --output-plot /tmp/spyctres_publication_xshooter_uvb_fit.png

Optional bounded systematic-variant run
---------------------------------------
python examples/publication_quality_xshooter_uvb.py \
  --run-baseline-fit \
  --run-systematic-variants \
  --max-systematic-run-variants 2 \
  --output-json /tmp/spyctres_publication_xshooter_uvb_fit.json \
  --output-systematic-results-csv /tmp/spyctres_systematic_results.csv

Optional same-model synthetic recovery check
--------------------------------------------
python examples/publication_quality_xshooter_uvb.py \
  --run-baseline-fit \
  --run-injection-recovery \
  --injection-recovery-trials 3 \
  --output-json /tmp/spyctres_publication_xshooter_uvb_fit.json \
  --output-injection-recovery-csv /tmp/spyctres_injection_recovery.csv

Optional compact reviewer summary
---------------------------------
python examples/publication_quality_xshooter_uvb.py \
  --run-baseline-fit \
  --output-json /tmp/spyctres_publication_xshooter_uvb_fit.json \
  --output-publication-summary-md /tmp/spyctres_publication_summary.md \
  --output-publication-summary-csv /tmp/spyctres_publication_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
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
    input_checksum_provenance,
    prepare_phoenix_fit_kwargs,
    publication_readiness_audit,
    select_diagnostic_windows,
)
from Spyctres._serialization import (
    atomic_write_csv_rows,
    atomic_write_json,
    save_figure,
)
from Spyctres._spectrum_helpers import spectrum_segments
from Spyctres._workflow_helpers import (
    resolution_assumption_for_audit as _shared_resolution_assumption_for_audit,
    unique_archive_masks,
)
from Spyctres.io import SpectrumCollection, SpectrumSegment, read_spectrum
from Spyctres.plotting import plot_fit_referee
from Spyctres.preprocessing import (
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


LINE_RESIDUAL_INFORMATIONAL_FLAGS = frozenset(
    {
        "core_model_only_not_fitted",
    }
)


CALIBRATION_PARAMETER_THRESHOLDS = {
    "teff": {
        "label": "Teff",
        "unit": "K",
        "acceptable_abs_delta": 100.0,
        "blocking_abs_delta": 250.0,
    },
    "feh": {
        "label": "[Fe/H]",
        "unit": "dex",
        "acceptable_abs_delta": 0.10,
        "blocking_abs_delta": 0.25,
    },
    "logg": {
        "label": "logg",
        "unit": "dex",
        "acceptable_abs_delta": 0.15,
        "blocking_abs_delta": 0.35,
    },
    "rv_kms": {
        "label": "RV",
        "unit": "km/s",
        "acceptable_abs_delta": 3.0,
        "blocking_abs_delta": 10.0,
    },
}


CALIBRATION_CHI2_THRESHOLDS = {
    "label": "chi2_red",
    "unit": "",
    "acceptable_positive_delta": 0.25,
    "blocking_positive_delta": 1.0,
}


CALIBRATION_ASSESSMENT_ORDER = {
    "acceptable": 0,
    "borderline": 1,
    "blocking": 2,
    "not_evaluated": -1,
}


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
            "--output-report-json /tmp/spyctres_publication_xshooter_uvb_report.json "
            "--output-plot /tmp/spyctres_publication_xshooter_uvb_fit.png "
            "--output-balmer-residual-csv /tmp/spyctres_balmer_residuals.csv\n\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--run-baseline-fit --run-systematic-variants "
            "--max-systematic-run-variants 2 "
            "--output-json /tmp/spyctres_publication_xshooter_uvb_fit.json "
            "--output-systematic-results-csv /tmp/spyctres_systematic_results.csv\n\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--run-baseline-fit --run-injection-recovery "
            "--injection-recovery-trials 3 "
            "--output-json /tmp/spyctres_publication_xshooter_uvb_fit.json "
            "--output-injection-recovery-csv /tmp/spyctres_injection_recovery.csv\n\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--run-baseline-fit "
            "--output-json /tmp/spyctres_publication_xshooter_uvb_fit.json "
            "--output-publication-summary-md /tmp/spyctres_publication_summary.md "
            "--output-publication-summary-csv /tmp/spyctres_publication_summary.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
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
        "--output-report-json",
        default=None,
        help=(
            "Optional versioned PhoenixFitResult report for the baseline fit. "
            "Requires --run-baseline-fit. The existing --output-json remains "
            "the full scaffold checkpoint."
        ),
    )
    parser.add_argument(
        "--record-input-checksum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to recording a SHA256 checksum of the original spectrum "
            "file bytes in baseline-fit provenance/report JSON. Disabled by "
            "default for speed and privacy."
        ),
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
        "--output-systematic-results-csv",
        default=None,
        help=(
            "Optional CSV table summarizing executed fit-level systematic "
            "variants. Requires --run-systematic-variants."
        ),
    )
    parser.add_argument(
        "--output-injection-recovery-csv",
        default=None,
        help=(
            "Optional CSV table summarizing opt-in synthetic injection/recovery "
            "trials. Requires --run-injection-recovery."
        ),
    )
    parser.add_argument(
        "--output-publication-summary-md",
        default=None,
        help=(
            "Optional compact Markdown reviewer summary built from the JSON "
            "payload. This does not run extra fits."
        ),
    )
    parser.add_argument(
        "--output-publication-summary-csv",
        default=None,
        help=(
            "Optional compact CSV comparison of baseline, systematic variants, "
            "and injection/recovery trials. This does not run extra fits."
        ),
    )
    parser.add_argument(
        "--output-publication-summary-plot",
        default=None,
        help=(
            "Optional compact PNG plot of recovered-minus-baseline parameter "
            "deltas for completed systematic and injection/recovery checks."
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
        "--run-systematic-variants",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After a successful --run-baseline-fit, execute a bounded subset "
            "of the planned systematic variants. Default: false."
        ),
    )
    parser.add_argument(
        "--run-injection-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After a successful --run-baseline-fit, run bounded same-model "
            "synthetic injection/recovery trials. Default: false."
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
        default=4.0,
        help=(
            "Half-width in Angstrom for the opt-in Balmer-core exclusion mask. "
            "Default: 4 A, matching the conservative information-retention "
            "recommendation for the bundled UVB scaffold. Set <=0 to disable."
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
    parser.add_argument(
        "--systematic-variant-ids",
        default=None,
        help=(
            "Comma-separated variant IDs to execute when "
            "--run-systematic-variants is set. Default: choose a small "
            "priority subset from the plan."
        ),
    )
    parser.add_argument(
        "--max-systematic-run-variants",
        type=int,
        default=4,
        help=(
            "Maximum number of systematic variants to execute in one run when "
            "--systematic-variant-ids is not supplied. Default: 4."
        ),
    )
    parser.add_argument(
        "--skip-existing-systematic-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip already completed systematic variants in the current JSON "
            "checkpoint. Default: true."
        ),
    )
    parser.add_argument(
        "--injection-recovery-trials",
        type=int,
        default=3,
        help=(
            "Number of synthetic recovery trials when --run-injection-recovery "
            "is set. Must be between 1 and 20. Default: 3."
        ),
    )
    parser.add_argument(
        "--injection-noise-scale",
        type=float,
        default=1.0,
        help=(
            "Gaussian noise scale for synthetic recovery, in units of the "
            "segment 1-sigma errors. Use 0 for a deterministic no-noise "
            "optimizer sanity check. Default: 1."
        ),
    )
    parser.add_argument(
        "--injection-default-error-fraction",
        type=float,
        default=0.02,
        help=(
            "Fallback fractional 1-sigma error used only if a segment lacks "
            "usable uncertainties. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--injection-seed",
        type=int,
        default=20260721,
        help="Random seed for synthetic injection/recovery trials.",
    )
    return parser


def _atomic_write_json(path, payload):
    """Backward-compatible wrapper around Spyctres' shared atomic JSON writer."""
    atomic_write_json(path, payload, sort_keys=True)


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
    segments = spectrum_segments(spectrum, tuple_is_collection=False, coerce=False)
    if len(segments) == 1 and isinstance(segments[0], SpectrumSegment):
        return segments[0]
    raise ValueError(
        "This scaffold expects one UVB segment. Use the multi-arm notebook or "
        "build an explicit expert workflow for multi-segment products."
    )


def _resolution_assumption(args):
    assumption = _shared_resolution_assumption_for_audit(
        args.resolution_R,
        assumption_warning=(
            "user-supplied constant resolution for publication scaffold; "
            "validate before using for final parameters"
        ),
    )
    if assumption is None:
        return None
    assumption["resolution_source"] = assumption["source"]
    assumption["assumed_resolution_R"] = assumption["value"]
    return assumption


def _archive_masks_by_segment(segments, policy):
    return unique_archive_masks(segments, policy=policy)


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
    rows = []
    for row in diagnostics.get("lines", ()):
        rows.append(
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
    atomic_write_csv_rows(path, columns, rows)


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
    rows = []
    for row in diagnostics.get("lines", ()):
        used = row.get("used_residuals") or {}
        core = row.get("core_residuals_model_only") or {}
        masked = row.get("masked_or_excluded_residuals") or {}
        rows.append(
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
    atomic_write_csv_rows(path, columns, rows)


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
    rows = []
    for row in plan.get("variants", ()):
        rows.append(
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
    atomic_write_csv_rows(path, columns, rows)


def _parse_id_list(value):
    value = "" if value is None else str(value).strip()
    if not value:
        return []
    return list(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


def _variant_priority_key(variant):
    """Return a stable priority for default bounded systematic execution.

    The default execution set is intentionally small.  It first tests one
    preprocessing/continuum choice, then core-mask sensitivity, then the most
    informative single-line Balmer checks.  Resolution variants enter the
    default queue only when the user supplied a baseline ``--R``.
    """
    variant_id = str(variant.get("id", ""))
    category = str(variant.get("category", ""))
    category_priority = {
        "preprocessing_normalization": 0,
        "balmer_core_mask": 1,
        "continuum_degree": 2,
        "resolution_assumption": 3,
        "fit_windows": 4,
    }.get(category, 99)
    id_priority = {
        "normalization_sideband": 0,
        "balmer_core_mask_4A": 0,
        "continuum_mdeg_1": 0,
        "continuum_mdeg_3": 1,
        "resolution_scale_0.9": 0,
        "resolution_scale_1.1": 1,
        "window_set_single_hgamma": 0,
        "window_set_single_hbeta": 1,
        "window_set_single_hdelta": 2,
    }.get(variant_id, 20)
    if variant_id.startswith("window_set_single_"):
        id_priority = min(id_priority, 10)
    elif variant_id.startswith("window_set_leave_out_"):
        id_priority = min(id_priority, 15)
    return (category_priority, id_priority, variant_id)


def _selected_systematic_variants(plan, args):
    variants = list(plan.get("variants", ()))
    by_id = {str(item.get("id")): item for item in variants}
    requested_ids = _parse_id_list(args.systematic_variant_ids)
    if requested_ids:
        missing = [variant_id for variant_id in requested_ids if variant_id not in by_id]
        if missing:
            raise ValueError(
                "--systematic-variant-ids contains unknown IDs: {0}. "
                "Check --output-systematic-plan-csv or the JSON plan first.".format(
                    ", ".join(missing)
                )
            )
        return [by_id[variant_id] for variant_id in requested_ids]

    max_variants = int(args.max_systematic_run_variants)
    if max_variants < 1:
        raise ValueError("--max-systematic-run-variants must be >= 1.")
    candidates = [
        item
        for item in variants
        if item.get("id") != "baseline" and bool(item.get("executable_now"))
    ]
    candidates.sort(key=_variant_priority_key)
    return candidates[:max_variants]


def _namespace_with_overrides(args, variant):
    data = vars(args).copy()
    preparation = dict(variant.get("preparation_overrides") or {})
    fit = dict(variant.get("fit_overrides") or {})
    if "norm_mode" in preparation:
        data["norm_mode"] = preparation["norm_mode"]
    if "balmer_core_mask_A" in preparation:
        data["balmer_core_mask"] = preparation["balmer_core_mask_A"]
    if "mdeg" in fit:
        data["mdeg"] = fit["mdeg"]
    if "resolution_R" in fit:
        data["resolution_R"] = fit["resolution_R"]
    return argparse.Namespace(**data)


def _segment_label(segment):
    return str(segment.meta.get("line_label", segment.name))


def _filter_collection_by_labels(collection, exclude_masks, labels):
    labels = [str(label) for label in labels or ()]
    if not labels:
        return collection, tuple(exclude_masks)
    wanted = set(labels)
    selected = [
        (index, segment)
        for index, segment in enumerate(collection.segments)
        if _segment_label(segment) in wanted
    ]
    if not selected:
        raise ValueError(
            "Systematic window variant requested labels with no matching "
            "segments: {0}".format(", ".join(labels))
        )
    label_order = {label: index for index, label in enumerate(labels)}
    selected.sort(
        key=lambda item: label_order.get(_segment_label(item[1]), 10**6)
    )
    indices = [index for index, _segment in selected]
    segments = [segment for _index, segment in selected]
    weights = np.asarray(collection.weights, dtype=float)[indices]
    collection_i = collection.copy(
        segments=segments,
        weights=weights,
        meta={
            **dict(collection.meta),
            "systematic_window_subset_labels": labels,
        },
        name="{0}_{1}".format(
            collection.name or "xshooter_uvb_publication_balmer_windows",
            "_".join(_safe_id_token(label) for label in labels),
        ),
    )
    return collection_i, tuple(exclude_masks)


def _prepare_systematic_variant_inputs(args, source_segment, variant):
    variant_args = _namespace_with_overrides(args, variant)
    core_mask = float(variant_args.balmer_core_mask)
    _case_i, collection_i, exclude_masks_i = _prepare_balmer_collection(
        variant_args,
        source_segment,
        core_mask_halfwidth=core_mask,
    )
    return (variant_args,) + _filter_collection_by_labels(
        collection_i,
        exclude_masks_i,
        variant.get("window_labels"),
    )


def _fit_summary_for_systematic_record(fit_payload):
    if not isinstance(fit_payload, dict):
        return {
            "success": None,
            "teff": None,
            "feh": None,
            "logg": None,
            "rv_kms": None,
            "chi2_red": None,
            "quality_flags": [],
        }
    return {
        "success": fit_payload.get("success"),
        "teff": _finite_or_none(fit_payload.get("teff")),
        "feh": _finite_or_none(fit_payload.get("feh")),
        "logg": _finite_or_none(fit_payload.get("logg")),
        "rv_kms": _finite_or_none(fit_payload.get("rv_kms")),
        "chi2_red": _finite_or_none(fit_payload.get("chi2_red")),
        "quality_flags": list(fit_payload.get("quality_flags") or ()),
    }


def _parameter_spread(records):
    summary = {}
    for key in ("teff", "feh", "logg", "rv_kms", "chi2_red"):
        values = [
            _finite_or_none((record.get("fit_summary") or {}).get(key))
            for record in records
            if record.get("status") == "ok"
        ]
        values = [value for value in values if value is not None]
        if values:
            arr = np.asarray(values, dtype=float)
            summary[key] = {
                "n": int(arr.size),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "range": float(np.nanmax(arr) - np.nanmin(arr)),
                "std": float(np.nanstd(arr)),
            }
        else:
            summary[key] = {
                "n": 0,
                "min": None,
                "max": None,
                "range": None,
                "std": None,
            }
    return summary


def _systematic_results_summary(records, *, requested_ids):
    counts = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if not records:
        status = "no_variants_selected"
    elif counts.get("error"):
        status = "completed_with_errors"
    elif counts.get("fit_failed"):
        status = "completed_with_fit_failures"
    elif counts.get("ok"):
        status = "completed"
    else:
        status = "completed_no_successful_fits"
    flags = sorted(
        {
            flag
            for record in records
            for flag in (record.get("fit_summary") or {}).get("quality_flags", ())
        }
    )
    return {
        "schema_version": 1,
        "status": status,
        "n_requested": int(len(requested_ids)),
        "n_records": int(len(records)),
        "status_counts": counts,
        "parameter_spread_ok_variants": _parameter_spread(records),
        "quality_flags_seen": flags,
        "interpretation": (
            "Fit-level systematic variants are sensitivity checks. Compare "
            "parameter shifts, per-line residual diagnostics, and quality "
            "flags; do not rank configurations by in-fit chi-square alone."
        ),
    }


def _run_one_systematic_variant(
    args,
    source_segment,
    variant,
    *,
    fit_runner=None,
):
    if fit_runner is None:
        fit_runner = _run_baseline_fit
    started = time.monotonic()
    variant_id = str(variant.get("id"))
    if variant_id == "baseline":
        return {
            "variant_id": variant_id,
            "category": variant.get("category"),
            "status": "skipped",
            "skip_reasons": ["baseline_fit_recorded_separately"],
            "elapsed_s": 0.0,
        }
    if not bool(variant.get("executable_now")):
        return {
            "variant_id": variant_id,
            "category": variant.get("category"),
            "status": "skipped",
            "skip_reasons": list(variant.get("skip_reasons") or ()),
            "elapsed_s": 0.0,
        }

    try:
        variant_args, collection_i, exclude_masks_i = _prepare_systematic_variant_inputs(
            args,
            source_segment,
            variant,
        )
        ordinary_i, publication_i = _run_readiness(
            variant_args,
            collection_i,
            exclude_masks_i,
        )
        fit_payload, fit_result = fit_runner(
            variant_args,
            collection_i,
            exclude_masks_i,
            output_plot=None,
            return_result=True,
            fit_label="systematic {0}".format(variant_id),
        )
        residuals = _balmer_model_residual_diagnostics(
            collection_i,
            fit_result,
            core_mask_halfwidth=float(variant_args.balmer_core_mask),
        )
        fit_summary = _fit_summary_for_systematic_record(fit_payload)
        status = "ok" if fit_summary.get("success") else "fit_failed"
        return {
            "variant_id": variant_id,
            "category": variant.get("category"),
            "label": variant.get("label"),
            "status": status,
            "elapsed_s": float(time.monotonic() - started),
            "preparation_overrides": dict(variant.get("preparation_overrides") or {}),
            "fit_overrides": dict(variant.get("fit_overrides") or {}),
            "window_labels": list(variant.get("window_labels") or ()),
            "ordinary_readiness": ordinary_i,
            "publication_readiness": publication_i,
            "fit_summary": fit_summary,
            "fit": fit_payload,
            "line_residual_diagnostics": residuals,
        }
    except Exception as exc:  # pragma: no cover - defensive checkpoint path.
        return {
            "variant_id": variant_id,
            "category": variant.get("category"),
            "label": variant.get("label"),
            "status": "error",
            "error": "{0}: {1}".format(type(exc).__name__, exc),
            "elapsed_s": float(time.monotonic() - started),
            "preparation_overrides": dict(variant.get("preparation_overrides") or {}),
            "fit_overrides": dict(variant.get("fit_overrides") or {}),
            "window_labels": list(variant.get("window_labels") or ()),
        }


def _write_systematic_variant_results_csv(path, results):
    if path is None:
        return
    columns = [
        "variant_id",
        "category",
        "status",
        "success",
        "teff",
        "feh",
        "logg",
        "rv_kms",
        "chi2_red",
        "n_residual_lines",
        "residual_quality_flags",
        "quality_flags",
        "elapsed_s",
        "error",
    ]
    rows = []
    for record in results.get("records", ()):
        fit_summary = record.get("fit_summary") or {}
        residuals = record.get("line_residual_diagnostics") or {}
        residual_summary = residuals.get("summary") or {}
        rows.append(
            {
                "variant_id": record.get("variant_id"),
                "category": record.get("category"),
                "status": record.get("status"),
                "success": fit_summary.get("success"),
                "teff": fit_summary.get("teff"),
                "feh": fit_summary.get("feh"),
                "logg": fit_summary.get("logg"),
                "rv_kms": fit_summary.get("rv_kms"),
                "chi2_red": fit_summary.get("chi2_red"),
                "n_residual_lines": residual_summary.get("n_evaluated_lines"),
                "residual_quality_flags": ";".join(
                    residual_summary.get("quality_flags") or ()
                ),
                "quality_flags": ";".join(fit_summary.get("quality_flags") or ()),
                "elapsed_s": record.get("elapsed_s"),
                "error": record.get("error"),
            }
        )
    atomic_write_csv_rows(path, columns, rows)


def _skipped_systematic_results(status, reason):
    return {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "n_requested": 0,
        "n_records": 0,
        "status_counts": {},
        "requested_variant_ids": [],
        "parameter_spread_ok_variants": _parameter_spread([]),
        "quality_flags_seen": [],
        "records": [],
    }


def _run_selected_systematic_variants(
    args,
    source_segment,
    plan,
    output_path,
    payload,
    *,
    fit_runner=None,
):
    if fit_runner is None:
        fit_runner = _run_baseline_fit
    selected = _selected_systematic_variants(plan, args)
    requested_ids = [str(item.get("id")) for item in selected]
    existing_results = payload.get("systematic_variant_results") or {}
    records = list(existing_results.get("records") or [])
    completed_ids = {
        str(record.get("variant_id"))
        for record in records
        if record.get("status") in {"ok", "fit_failed", "skipped"}
    }

    for index, variant in enumerate(selected, start=1):
        variant_id = str(variant.get("id"))
        if args.skip_existing_systematic_variants and variant_id in completed_ids:
            print(
                "Skipping completed systematic variant {0}/{1}: {2}".format(
                    index,
                    len(selected),
                    variant_id,
                ),
                flush=True,
            )
            continue
        print(
            "Running systematic variant {0}/{1}: {2}".format(
                index,
                len(selected),
                variant_id,
            ),
            flush=True,
        )
        record = _run_one_systematic_variant(
            args,
            source_segment,
            variant,
            fit_runner=fit_runner,
        )
        records = [
            old
            for old in records
            if str(old.get("variant_id")) != variant_id
        ]
        records.append(record)
        payload["systematic_variant_results"] = {
            **_systematic_results_summary(records, requested_ids=requested_ids),
            "requested_variant_ids": requested_ids,
            "records": records,
        }
        _atomic_write_json(output_path, payload)
        print(
            "Systematic variant {0}: status={1}".format(
                variant_id,
                record.get("status"),
            ),
            flush=True,
        )

    results = {
        **_systematic_results_summary(records, requested_ids=requested_ids),
        "requested_variant_ids": requested_ids,
        "records": records,
    }
    payload["systematic_variant_results"] = results
    return results


def _baseline_truth_from_payload(fit_payload):
    return {
        "teff": _finite_or_none((fit_payload or {}).get("teff")),
        "feh": _finite_or_none((fit_payload or {}).get("feh")),
        "logg": _finite_or_none((fit_payload or {}).get("logg")),
        "rv_kms": _finite_or_none((fit_payload or {}).get("rv_kms")),
    }


def _injection_recovery_tolerances():
    return {
        "teff": 250.0,
        "feh": 0.25,
        "logg": 0.35,
        "rv_kms": 10.0,
    }


def _make_synthetic_collection_from_baseline(
    args,
    collection,
    baseline_result,
    *,
    trial_index,
    rng,
):
    """Return a synthetic collection generated from the fitted baseline model.

    This is a same-model injection/recovery check: it asks whether the fitting
    machinery can recover its own injected PHOENIX+continuum solution under the
    existing wavelength grid, masks, and error model.  It is deliberately not a
    validation of PHOENIX physics against real stars.
    """
    segments = list(collection.segments)
    models, _used_masks, _excluded_masks = _result_model_mask_arrays(
        baseline_result,
        n_segments=len(segments),
    )
    if models is None:
        raise ValueError(
            "Synthetic injection/recovery requires reconstructed baseline "
            "model arrays. Run the baseline fit through this scaffold."
        )

    noise_scale = float(args.injection_noise_scale)
    if not np.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError("--injection-noise-scale must be finite and >= 0.")
    default_error_fraction = float(args.injection_default_error_fraction)
    if (
        not np.isfinite(default_error_fraction)
        or default_error_fraction <= 0.0
    ):
        raise ValueError("--injection-default-error-fraction must be finite and > 0.")

    out_segments = []
    error_sources = []
    for segment_index, (segment, model) in enumerate(zip(segments, models)):
        wave = np.asarray(segment.wave, dtype=float)
        model = np.asarray(model, dtype=float)
        if model.shape != wave.shape:
            raise ValueError(
                "Baseline model shape mismatch for segment {0}: {1} vs {2}.".format(
                    segment_index,
                    model.shape,
                    wave.shape,
                )
            )
        if segment.err is None:
            finite_model = model[np.isfinite(model)]
            scale = (
                max(abs(float(np.nanmedian(finite_model))), 1e-30)
                if finite_model.size
                else 1.0
            )
            err = np.full(wave.shape, scale * default_error_fraction, dtype=float)
            error_source = "fallback_fractional_error"
        else:
            err = np.asarray(segment.err, dtype=float).copy()
            good_err = np.isfinite(err) & (err > 0.0)
            if not np.any(good_err):
                finite_model = model[np.isfinite(model)]
                scale = (
                    max(abs(float(np.nanmedian(finite_model))), 1e-30)
                    if finite_model.size
                    else 1.0
                )
                err = np.full(wave.shape, scale * default_error_fraction, dtype=float)
                error_source = "fallback_fractional_error"
            else:
                fallback = float(np.nanmedian(err[good_err]))
                err[~good_err] = fallback
                error_source = "segment_sigma_errors"

        noise = np.zeros(wave.shape, dtype=float)
        if noise_scale > 0.0:
            noise = rng.normal(loc=0.0, scale=err * noise_scale)
        flux = model + noise
        mask = np.asarray(segment.mask, dtype=bool) & np.isfinite(wave) & np.isfinite(flux)
        mask &= np.isfinite(err) & (err > 0.0)
        meta = dict(segment.meta)
        meta.update(
            {
                "synthetic_injection_recovery": True,
                "synthetic_trial_index": int(trial_index),
                "synthetic_source": "baseline_continuum_adjusted_model",
                "synthetic_noise_scale": float(noise_scale),
                "synthetic_error_source": error_source,
            }
        )
        out_segments.append(
            segment.copy(
                flux=flux,
                err=err,
                mask=mask,
                meta=meta,
                name="{0}_synthetic_trial_{1}".format(
                    segment.name,
                    int(trial_index),
                ),
            )
        )
        error_sources.append(error_source)

    return collection.copy(
        segments=out_segments,
        meta={
            **dict(collection.meta),
            "synthetic_injection_recovery": True,
            "synthetic_trial_index": int(trial_index),
            "synthetic_noise_scale": float(noise_scale),
            "synthetic_error_sources": list(dict.fromkeys(error_sources)),
        },
        name="{0}_synthetic_trial_{1}".format(
            collection.name or "publication_balmer",
            int(trial_index),
        ),
    )


def _injection_trial_summary(fit_payload, truth):
    recovered = _baseline_truth_from_payload(fit_payload)
    deltas = {}
    tolerances = _injection_recovery_tolerances()
    pass_flags = {}
    for key, truth_value in truth.items():
        recovered_value = recovered.get(key)
        if truth_value is None or recovered_value is None:
            deltas[key] = None
            pass_flags[key] = False
        else:
            delta = float(recovered_value - truth_value)
            deltas[key] = delta
            pass_flags[key] = bool(abs(delta) <= tolerances[key])
    return {
        "success": fit_payload.get("success") if isinstance(fit_payload, dict) else None,
        "truth": dict(truth),
        "recovered": recovered,
        "delta": deltas,
        "tolerances": tolerances,
        "passed_tolerances": pass_flags,
        "all_passed": bool(pass_flags and all(pass_flags.values())),
        "chi2_red": (
            _finite_or_none(fit_payload.get("chi2_red"))
            if isinstance(fit_payload, dict)
            else None
        ),
        "quality_flags": (
            list(fit_payload.get("quality_flags") or ())
            if isinstance(fit_payload, dict)
            else []
        ),
    }


def _injection_recovery_summary(records, truth):
    counts = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    ok_records = [record for record in records if record.get("status") == "ok"]
    pass_count = sum(
        1
        for record in ok_records
        if (record.get("fit_summary") or {}).get("all_passed")
    )
    delta_stats = {}
    for key in ("teff", "feh", "logg", "rv_kms"):
        values = [
            (record.get("fit_summary") or {}).get("delta", {}).get(key)
            for record in ok_records
        ]
        values = [float(value) for value in values if value is not None]
        if values:
            arr = np.asarray(values, dtype=float)
            delta_stats[key] = {
                "n": int(arr.size),
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "std": float(np.nanstd(arr)),
                "max_abs": float(np.nanmax(np.abs(arr))),
            }
        else:
            delta_stats[key] = {
                "n": 0,
                "mean": None,
                "median": None,
                "std": None,
                "max_abs": None,
            }
    if not records:
        status = "no_trials_selected"
    elif counts.get("error"):
        status = "completed_with_errors"
    elif ok_records and pass_count == len(ok_records):
        status = "completed_all_recovered"
    elif ok_records:
        status = "completed_with_recovery_failures"
    else:
        status = "completed_no_successful_trials"
    flags = sorted(
        {
            flag
            for record in records
            for flag in (record.get("fit_summary") or {}).get("quality_flags", ())
        }
    )
    return {
        "schema_version": 1,
        "status": status,
        "n_records": int(len(records)),
        "n_ok": int(len(ok_records)),
        "n_passed_all_tolerances": int(pass_count),
        "status_counts": counts,
        "truth": dict(truth),
        "tolerances": _injection_recovery_tolerances(),
        "delta_statistics": delta_stats,
        "quality_flags_seen": flags,
        "interpretation": (
            "Same-model synthetic recovery checks optimizer, masks, continuum "
            "handling, and noise response around the baseline solution. Passing "
            "this test is necessary but not sufficient for publication use, "
            "because it does not test PHOENIX model physics against real stars."
        ),
    }


def _run_one_injection_recovery_trial(
    args,
    collection,
    exclude_masks,
    baseline_result,
    truth,
    *,
    trial_index,
    rng,
    fit_runner=None,
):
    if fit_runner is None:
        fit_runner = _run_baseline_fit
    started = time.monotonic()
    try:
        synthetic_collection = _make_synthetic_collection_from_baseline(
            args,
            collection,
            baseline_result,
            trial_index=trial_index,
            rng=rng,
        )
        ordinary_i, publication_i = _run_readiness(
            args,
            synthetic_collection,
            exclude_masks,
        )
        fit_payload, fit_result = fit_runner(
            args,
            synthetic_collection,
            exclude_masks,
            output_plot=None,
            return_result=True,
            fit_label="synthetic recovery trial {0}".format(trial_index),
        )
        residuals = _balmer_model_residual_diagnostics(
            synthetic_collection,
            fit_result,
            core_mask_halfwidth=float(args.balmer_core_mask),
        )
        return {
            "trial_index": int(trial_index),
            "status": "ok" if fit_payload.get("success") else "fit_failed",
            "elapsed_s": float(time.monotonic() - started),
            "ordinary_readiness": ordinary_i,
            "publication_readiness": publication_i,
            "fit_summary": _injection_trial_summary(fit_payload, truth),
            "fit": fit_payload,
            "line_residual_diagnostics": residuals,
        }
    except Exception as exc:  # pragma: no cover - defensive checkpoint path.
        return {
            "trial_index": int(trial_index),
            "status": "error",
            "error": "{0}: {1}".format(type(exc).__name__, exc),
            "elapsed_s": float(time.monotonic() - started),
        }


def _write_injection_recovery_csv(path, results):
    if path is None:
        return
    columns = [
        "trial_index",
        "status",
        "success",
        "truth_teff",
        "truth_feh",
        "truth_logg",
        "truth_rv_kms",
        "recovered_teff",
        "recovered_feh",
        "recovered_logg",
        "recovered_rv_kms",
        "delta_teff",
        "delta_feh",
        "delta_logg",
        "delta_rv_kms",
        "all_passed",
        "chi2_red",
        "quality_flags",
        "elapsed_s",
        "error",
    ]
    rows = []
    for record in results.get("records", ()):
        summary = record.get("fit_summary") or {}
        truth = summary.get("truth") or {}
        recovered = summary.get("recovered") or {}
        delta = summary.get("delta") or {}
        rows.append(
            {
                "trial_index": record.get("trial_index"),
                "status": record.get("status"),
                "success": summary.get("success"),
                "truth_teff": truth.get("teff"),
                "truth_feh": truth.get("feh"),
                "truth_logg": truth.get("logg"),
                "truth_rv_kms": truth.get("rv_kms"),
                "recovered_teff": recovered.get("teff"),
                "recovered_feh": recovered.get("feh"),
                "recovered_logg": recovered.get("logg"),
                "recovered_rv_kms": recovered.get("rv_kms"),
                "delta_teff": delta.get("teff"),
                "delta_feh": delta.get("feh"),
                "delta_logg": delta.get("logg"),
                "delta_rv_kms": delta.get("rv_kms"),
                "all_passed": summary.get("all_passed"),
                "chi2_red": summary.get("chi2_red"),
                "quality_flags": ";".join(summary.get("quality_flags") or ()),
                "elapsed_s": record.get("elapsed_s"),
                "error": record.get("error"),
            }
        )
    atomic_write_csv_rows(path, columns, rows)


def _skipped_injection_recovery(status, reason, truth=None):
    truth = {} if truth is None else dict(truth)
    return {
        **_injection_recovery_summary([], truth),
        "status": status,
        "reason": reason,
        "records": [],
    }


def _run_injection_recovery(
    args,
    collection,
    exclude_masks,
    baseline_result,
    baseline_payload,
    output_path,
    payload,
    *,
    fit_runner=None,
):
    n_trials = int(args.injection_recovery_trials)
    if n_trials < 1 or n_trials > 20:
        raise ValueError("--injection-recovery-trials must be between 1 and 20.")
    truth = _baseline_truth_from_payload(baseline_payload)
    missing_truth = [key for key, value in truth.items() if value is None]
    if missing_truth:
        return _skipped_injection_recovery(
            "skipped_missing_baseline_truth",
            "Baseline fit did not report all injected truth parameters: {0}.".format(
                ", ".join(missing_truth)
            ),
            truth=truth,
        )

    existing = payload.get("injection_recovery") or {}
    records = list(existing.get("records") or [])
    completed = {
        int(record.get("trial_index"))
        for record in records
        if record.get("status") in {"ok", "fit_failed"}
        and record.get("trial_index") is not None
    }
    for trial_index in range(1, n_trials + 1):
        if trial_index in completed:
            print(
                "Skipping completed injection/recovery trial {0}/{1}".format(
                    trial_index,
                    n_trials,
                ),
                flush=True,
            )
            continue
        print(
            "Running injection/recovery trial {0}/{1}".format(
                trial_index,
                n_trials,
            ),
            flush=True,
        )
        trial_rng = np.random.default_rng(int(args.injection_seed) + int(trial_index))
        record = _run_one_injection_recovery_trial(
            args,
            collection,
            exclude_masks,
            baseline_result,
            truth,
            trial_index=trial_index,
            rng=trial_rng,
            fit_runner=fit_runner,
        )
        records = [
            old
            for old in records
            if int(old.get("trial_index", -1)) != trial_index
        ]
        records.append(record)
        records.sort(key=lambda item: int(item.get("trial_index", 0)))
        payload["injection_recovery"] = {
            **_injection_recovery_summary(records, truth),
            "method": "same_model_baseline_noise_injection",
            "config": {
                "n_trials_requested": int(n_trials),
                "noise_scale": float(args.injection_noise_scale),
                "default_error_fraction": float(
                    args.injection_default_error_fraction
                ),
                "seed": int(args.injection_seed),
            },
            "records": records,
        }
        _atomic_write_json(output_path, payload)
        print(
            "Injection/recovery trial {0}: status={1}, passed={2}".format(
                trial_index,
                record.get("status"),
                (record.get("fit_summary") or {}).get("all_passed"),
            ),
            flush=True,
        )

    results = {
        **_injection_recovery_summary(records, truth),
        "method": "same_model_baseline_noise_injection",
        "config": {
            "n_trials_requested": int(n_trials),
            "noise_scale": float(args.injection_noise_scale),
            "default_error_fraction": float(args.injection_default_error_fraction),
            "seed": int(args.injection_seed),
        },
        "records": records,
    }
    payload["injection_recovery"] = results
    return results


def _summary_fit_value(fit_payload, key):
    if not isinstance(fit_payload, dict):
        return None
    return _finite_or_none(fit_payload.get(key))


def _summary_delta(value, baseline_value):
    if value is None or baseline_value is None:
        return None
    return float(value - baseline_value)


def _line_residual_summary_for_report(diagnostics):
    if not isinstance(diagnostics, dict):
        return {
            "status": "not_available",
            "n_lines": 0,
            "max_chi2_red_proxy": None,
            "max_rms_fractional_residual": None,
            "problem_lines": [],
            "quality_flags": [],
            "informational_flags": [],
        }
    lines = list(diagnostics.get("lines") or ())
    ok_lines = [line for line in lines if line.get("status") == "ok"]
    chi2_values = []
    rms_values = []
    problem_lines = []
    flags = set()
    info_flags = set()
    for flag in diagnostics.get("summary", {}).get("quality_flags") or ():
        if flag in LINE_RESIDUAL_INFORMATIONAL_FLAGS:
            info_flags.add(flag)
        else:
            flags.add(flag)
    for line in ok_lines:
        used = line.get("used_residuals") or {}
        chi2 = _finite_or_none(used.get("chi2_red_proxy"))
        rms = _finite_or_none(used.get("rms_fractional_residual"))
        if chi2 is not None:
            chi2_values.append(chi2)
        if rms is not None:
            rms_values.append(rms)
        line_flags = list(line.get("quality_flags") or ())
        problem_flags = []
        for flag in line_flags:
            if flag in LINE_RESIDUAL_INFORMATIONAL_FLAGS:
                info_flags.add(flag)
            else:
                flags.add(flag)
                problem_flags.append(flag)
        if problem_flags:
            problem_lines.append(str(line.get("line_label", line.get("segment_name"))))
    return {
        "status": diagnostics.get("status", "unknown"),
        "n_lines": int(len(ok_lines)),
        "max_chi2_red_proxy": (
            float(np.nanmax(chi2_values)) if chi2_values else None
        ),
        "max_rms_fractional_residual": (
            float(np.nanmax(rms_values)) if rms_values else None
        ),
        "problem_lines": sorted(set(problem_lines)),
        "quality_flags": sorted(flags),
        "informational_flags": sorted(info_flags),
    }


def _comparison_row_from_fit(
    *,
    source_kind,
    source_id,
    label,
    status,
    fit_payload,
    baseline_fit,
    variant_category=None,
    window_labels=None,
    line_residual_diagnostics=None,
    all_passed=None,
    error=None,
):
    baseline = {
        key: _summary_fit_value(baseline_fit, key)
        for key in ("teff", "feh", "logg", "rv_kms", "chi2_red")
    }
    values = {
        key: _summary_fit_value(fit_payload, key)
        for key in ("teff", "feh", "logg", "rv_kms", "chi2_red")
    }
    quality_flags = (
        list(fit_payload.get("quality_flags") or ())
        if isinstance(fit_payload, dict)
        else []
    )
    line_summary = _line_residual_summary_for_report(line_residual_diagnostics)
    return {
        "source_kind": source_kind,
        "source_id": source_id,
        "label": label,
        "variant_category": variant_category,
        "window_labels": list(window_labels or ()),
        "status": status,
        "success": (
            fit_payload.get("success") if isinstance(fit_payload, dict) else None
        ),
        "teff": values["teff"],
        "feh": values["feh"],
        "logg": values["logg"],
        "rv_kms": values["rv_kms"],
        "chi2_red": values["chi2_red"],
        "delta_teff": _summary_delta(values["teff"], baseline["teff"]),
        "delta_feh": _summary_delta(values["feh"], baseline["feh"]),
        "delta_logg": _summary_delta(values["logg"], baseline["logg"]),
        "delta_rv_kms": _summary_delta(values["rv_kms"], baseline["rv_kms"]),
        "delta_chi2_red": _summary_delta(values["chi2_red"], baseline["chi2_red"]),
        "all_passed": all_passed,
        "quality_flags": quality_flags,
        "line_residual_status": line_summary["status"],
        "n_line_residuals": line_summary["n_lines"],
        "max_line_chi2_red_proxy": line_summary["max_chi2_red_proxy"],
        "max_line_rms_fractional_residual": (
            line_summary["max_rms_fractional_residual"]
        ),
        "problem_lines": line_summary["problem_lines"],
        "line_quality_flags": line_summary["quality_flags"],
        "line_info_flags": line_summary["informational_flags"],
        "error": error,
    }


def _comparison_row_from_injection_record(record, baseline_fit):
    summary = record.get("fit_summary") or {}
    recovered = summary.get("recovered") or {}
    delta = summary.get("delta") or {}
    fit_payload = {
        "success": summary.get("success"),
        "teff": recovered.get("teff"),
        "feh": recovered.get("feh"),
        "logg": recovered.get("logg"),
        "rv_kms": recovered.get("rv_kms"),
        "chi2_red": summary.get("chi2_red"),
        "quality_flags": summary.get("quality_flags") or (),
    }
    row = _comparison_row_from_fit(
        source_kind="injection_recovery_trial",
        source_id="trial_{0}".format(record.get("trial_index")),
        label="Injection/recovery trial {0}".format(record.get("trial_index")),
        status=record.get("status"),
        fit_payload=fit_payload,
        baseline_fit=baseline_fit,
        variant_category="injection_recovery",
        line_residual_diagnostics=record.get("line_residual_diagnostics"),
        all_passed=summary.get("all_passed"),
        error=record.get("error"),
    )
    for key in ("teff", "feh", "logg", "rv_kms"):
        row["delta_{0}".format(key)] = _finite_or_none(delta.get(key))
    return row


def _max_abs_shift_by_kind(rows):
    out = {}
    for source_kind in sorted(
        {row.get("source_kind") for row in rows if row.get("source_kind") != "baseline"}
    ):
        kind_rows = [
            row
            for row in rows
            if row.get("source_kind") == source_kind and row.get("status") == "ok"
        ]
        out[source_kind] = {}
        for key in ("teff", "feh", "logg", "rv_kms", "chi2_red"):
            delta_key = "delta_{0}".format(key)
            values = [
                _finite_or_none(row.get(delta_key))
                for row in kind_rows
            ]
            values = [value for value in values if value is not None]
            out[source_kind][delta_key] = (
                float(np.nanmax(np.abs(values))) if values else None
            )
    return out


def _assessment_from_abs_delta(delta, thresholds):
    delta = _finite_or_none(delta)
    if delta is None:
        return "not_evaluated", None
    abs_delta = abs(float(delta))
    if abs_delta <= float(thresholds["acceptable_abs_delta"]):
        return "acceptable", abs_delta
    if abs_delta <= float(thresholds["blocking_abs_delta"]):
        return "borderline", abs_delta
    return "blocking", abs_delta


def _assessment_from_positive_delta(delta, thresholds):
    delta = _finite_or_none(delta)
    if delta is None:
        return "not_evaluated", None
    positive_delta = max(0.0, float(delta))
    if positive_delta <= float(thresholds["acceptable_positive_delta"]):
        return "acceptable", positive_delta
    if positive_delta <= float(thresholds["blocking_positive_delta"]):
        return "borderline", positive_delta
    return "blocking", positive_delta


def _worst_assessment(assessments, *, missing_as=None):
    values = [item for item in assessments if item]
    if missing_as is not None:
        values = [missing_as if item == "not_evaluated" else item for item in values]
    ranked = [
        item
        for item in values
        if item in CALIBRATION_ASSESSMENT_ORDER
        and CALIBRATION_ASSESSMENT_ORDER[item] >= 0
    ]
    if not ranked:
        return "not_evaluated"
    return max(ranked, key=lambda item: CALIBRATION_ASSESSMENT_ORDER[item])


def _format_delta_reason(label, value, unit, assessment, threshold):
    suffix = "" if not unit else " {0}".format(unit)
    return (
        "{0} shift {1:.5g}{2} is {3} relative to threshold {4:.5g}{2}".format(
            label,
            value,
            suffix,
            assessment,
            threshold,
        )
    )


def _sensitivity_scope_for_row(row):
    category = row.get("variant_category")
    source_kind = row.get("source_kind")
    source_id = str(row.get("source_id", ""))
    if source_kind == "baseline":
        return "baseline_reference"
    if source_kind == "injection_recovery_trial":
        return "same_model_recovery"
    if category == "balmer_core_mask" or source_id.startswith("balmer_core_mask"):
        return "core_mask_fit_sensitivity"
    if category == "fit_windows" or source_id.startswith("window_set"):
        return "window_set_fit_sensitivity"
    if source_kind == "systematic_variant":
        return "fit_level_sensitivity"
    return "unknown"


def _assess_comparison_row_sensitivity(row):
    source_kind = row.get("source_kind")
    status = row.get("status")
    scope = _sensitivity_scope_for_row(row)
    if source_kind == "baseline":
        return {
            "sensitivity_scope": scope,
            "sensitivity_assessment": "not_evaluated",
            "sensitivity_reasons": ["baseline reference row"],
        }
    if status not in {"ok", None}:
        return {
            "sensitivity_scope": scope,
            "sensitivity_assessment": "blocking",
            "sensitivity_reasons": [
                "sensitivity check did not complete cleanly: status={0}".format(status)
            ],
        }

    assessments = []
    reasons = []
    evaluated = 0
    for key, thresholds in CALIBRATION_PARAMETER_THRESHOLDS.items():
        assessment, value = _assessment_from_abs_delta(
            row.get("delta_{0}".format(key)),
            thresholds,
        )
        if assessment == "not_evaluated":
            continue
        evaluated += 1
        assessments.append(assessment)
        if assessment != "acceptable":
            limit_key = (
                "acceptable_abs_delta"
                if assessment == "borderline"
                else "blocking_abs_delta"
            )
            reasons.append(
                _format_delta_reason(
                    thresholds["label"],
                    value,
                    thresholds["unit"],
                    assessment,
                    thresholds[limit_key],
                )
            )

    chi2_assessment, chi2_value = _assessment_from_positive_delta(
        row.get("delta_chi2_red"),
        CALIBRATION_CHI2_THRESHOLDS,
    )
    if chi2_assessment != "not_evaluated":
        evaluated += 1
        assessments.append(chi2_assessment)
        if chi2_assessment != "acceptable":
            limit_key = (
                "acceptable_positive_delta"
                if chi2_assessment == "borderline"
                else "blocking_positive_delta"
            )
            reasons.append(
                _format_delta_reason(
                    CALIBRATION_CHI2_THRESHOLDS["label"],
                    chi2_value,
                    CALIBRATION_CHI2_THRESHOLDS["unit"],
                    chi2_assessment,
                    CALIBRATION_CHI2_THRESHOLDS[limit_key],
                )
            )

    if evaluated == 0:
        assessment = "not_evaluated"
        reasons.append("no finite parameter deltas are available")
    else:
        assessment = _worst_assessment(assessments)
        if assessment == "acceptable":
            reasons.append("all finite parameter shifts are within acceptable thresholds")

    if source_kind == "injection_recovery_trial" and row.get("all_passed") is False:
        assessment = "blocking"
        reasons.append("same-model recovery did not pass all configured tolerances")

    return {
        "sensitivity_scope": scope,
        "sensitivity_assessment": assessment,
        "sensitivity_reasons": reasons,
    }


def _core_mask_audit_interpretation(payload):
    records = list(payload.get("core_mask_sensitivity") or ())
    recommendation = payload.get("core_mask_sensitivity_recommendation") or {}
    if not records:
        return {
            "scope": "core_mask_audit",
            "item": "Balmer-core mask audit grid",
            "assessment": "not_evaluated",
            "detail": "core-mask audit grid has not been evaluated",
        }

    diagnostics = payload.get("per_line_balmer_diagnostics") or {}
    default_width = _finite_or_none(diagnostics.get("core_mask_halfwidth_A"))
    default_record = None
    if default_width is not None:
        default_record = next(
            (
                record
                for record in records
                if abs(float(record.get("core_mask_halfwidth_A", np.nan)) - default_width)
                <= 1e-8
            ),
            None,
        )
    recommended_width = _finite_or_none(
        recommendation.get("recommended_core_mask_halfwidth_A")
    )
    excessive_widths = [
        float(record["core_mask_halfwidth_A"])
        for record in records
        if record.get("excessive_core_mask")
    ]

    if default_record is None:
        assessment = "not_evaluated"
        detail = "baseline core-mask width is not present in the audit grid"
    elif default_record.get("excessive_core_mask"):
        assessment = "blocking"
        detail = (
            "baseline core-mask width {0:g} A retains only {1:.3g} of the "
            "no-core-mask fitted pixels".format(
                default_width,
                float(default_record.get("information_retention_fraction", np.nan)),
            )
        )
    elif (
        recommended_width is not None
        and default_width is not None
        and abs(default_width - recommended_width) > 1e-8
    ):
        assessment = "borderline"
        detail = (
            "baseline core-mask width {0:g} A differs from audit recommendation "
            "{1:g} A; compare fitted variants before publication use".format(
                default_width,
                recommended_width,
            )
        )
    else:
        assessment = "acceptable"
        detail = "baseline core-mask width matches the audit recommendation"

    if excessive_widths:
        detail += "; excessive grid widths: {0}".format(
            ", ".join("{0:g} A".format(width) for width in excessive_widths)
        )
    return {
        "scope": "core_mask_audit",
        "item": "Balmer-core mask audit grid",
        "assessment": assessment,
        "detail": detail,
        "baseline_core_mask_halfwidth_A": default_width,
        "recommended_core_mask_halfwidth_A": recommended_width,
        "excessive_core_mask_halfwidths_A": excessive_widths,
    }


def _row_group_interpretation(rows, *, scope, item, missing_detail):
    group = [row for row in rows if row.get("sensitivity_scope") == scope]
    if not group:
        return {
            "scope": scope,
            "item": item,
            "assessment": "not_evaluated",
            "detail": missing_detail,
        }
    assessment = _worst_assessment(
        [row.get("sensitivity_assessment") for row in group],
        missing_as="borderline",
    )
    if assessment == "acceptable":
        detail = "{0} completed check(s); all finite shifts are acceptable".format(
            len(group)
        )
    else:
        flagged = [
            "{0}: {1}".format(
                row.get("source_id"),
                "; ".join(row.get("sensitivity_reasons") or ()),
            )
            for row in group
            if row.get("sensitivity_assessment") == assessment
        ]
        detail = "{0} completed check(s); {1}".format(
            len(group),
            " | ".join(flagged) if flagged else "review parameter shifts",
        )
    return {
        "scope": scope,
        "item": item,
        "assessment": assessment,
        "detail": detail,
        "n_checks": int(len(group)),
    }


def _injection_recovery_interpretation(payload, rows):
    injection = payload.get("injection_recovery") or {}
    recovery_rows = [
        row
        for row in rows
        if row.get("sensitivity_scope") == "same_model_recovery"
    ]
    status = injection.get("status")
    if not recovery_rows:
        return {
            "scope": "same_model_recovery",
            "item": "Synthetic same-model recovery",
            "assessment": "not_evaluated",
            "detail": "synthetic injection/recovery has not been run",
        }
    if status == "completed_all_recovered":
        assessment = "acceptable"
        detail = "{0}/{0} completed trial(s) passed configured recovery tolerances".format(
            len(recovery_rows)
        )
    else:
        assessment = _worst_assessment(
            [row.get("sensitivity_assessment") for row in recovery_rows],
            missing_as="borderline",
        )
        if assessment == "acceptable":
            assessment = "borderline"
        detail = "injection/recovery status={0}; inspect recovery deltas".format(status)
    return {
        "scope": "same_model_recovery",
        "item": "Synthetic same-model recovery",
        "assessment": assessment,
        "detail": detail,
        "n_checks": int(len(recovery_rows)),
        "status": status,
    }


def _publication_readiness_interpretation(payload):
    publication = payload.get("publication_readiness") or {}
    blockers = list(publication.get("blockers") or ())
    if not publication:
        return {
            "scope": "publication_readiness",
            "item": "Publication gate",
            "assessment": "not_evaluated",
            "detail": "publication-readiness audit has not been run",
        }
    if blockers:
        return {
            "scope": "publication_readiness",
            "item": "Publication gate",
            "assessment": "blocking",
            "detail": "readiness blockers: {0}".format(", ".join(blockers)),
        }
    return {
        "scope": "publication_readiness",
        "item": "Publication gate",
        "assessment": "acceptable",
        "detail": "publication-readiness audit has no blockers",
    }


def _build_calibration_interpretation(payload, rows):
    for row in rows:
        row.update(_assess_comparison_row_sensitivity(row))

    checks = [
        _publication_readiness_interpretation(payload),
        _core_mask_audit_interpretation(payload),
        _row_group_interpretation(
            rows,
            scope="core_mask_fit_sensitivity",
            item="Fitted Balmer-core mask sensitivity",
            missing_detail=(
                "no fitted core-mask variants have been run; run an explicit "
                "balmer_core_mask variant before publication use"
            ),
        ),
        _row_group_interpretation(
            rows,
            scope="window_set_fit_sensitivity",
            item="Fitted window-set sensitivity",
            missing_detail=(
                "no fitted window-set variants have been run; run at least one "
                "single-line or leave-one-line-out variant before publication use"
            ),
        ),
        _injection_recovery_interpretation(payload, rows),
    ]
    assessed = [item["assessment"] for item in checks]
    if all(item == "not_evaluated" for item in assessed):
        overall = "not_evaluated"
    elif any(item == "blocking" for item in assessed):
        overall = "blocking"
    elif any(item in {"borderline", "not_evaluated"} for item in assessed):
        overall = "borderline"
    else:
        overall = "acceptable"

    recommendations = []
    for check in checks:
        if check["assessment"] == "blocking":
            recommendations.append(
                "Resolve blocking calibration check: {0} ({1}).".format(
                    check["item"],
                    check["detail"],
                )
            )
        elif check["assessment"] == "borderline":
            recommendations.append(
                "Review borderline calibration check: {0} ({1}).".format(
                    check["item"],
                    check["detail"],
                )
            )
        elif check["assessment"] == "not_evaluated":
            recommendations.append(
                "Complete calibration check: {0} ({1}).".format(
                    check["item"],
                    check["detail"],
                )
            )

    headline_flags = []
    if overall == "blocking":
        headline_flags.append("calibration_interpretation_blocking")
    elif overall == "borderline":
        headline_flags.append("calibration_interpretation_borderline_or_incomplete")
    elif overall == "not_evaluated":
        headline_flags.append("calibration_interpretation_not_evaluated")

    return {
        "schema_version": 1,
        "overall_assessment": overall,
        "thresholds": {
            "parameter_abs_delta": CALIBRATION_PARAMETER_THRESHOLDS,
            "chi2_positive_delta": CALIBRATION_CHI2_THRESHOLDS,
            "policy": (
                "Thresholds are conservative triage values for review. They are "
                "not calibrated final uncertainties and should be revisited after "
                "real reference-star validation."
            ),
        },
        "checks": checks,
        "headline_flags": headline_flags,
        "recommendations": recommendations,
        "interpretation": (
            "Calibration interpretation converts already-run readiness, mask, "
            "systematic-variant, and same-model recovery checks into review "
            "labels. It does not change the fit objective or select a model."
        ),
    }


def _build_publication_stability_interpretation(payload, rows, calibration):
    """Return a plain-language verdict for the current publication scaffold.

    This intentionally interprets already-computed checks only.  It does not
    alter the fit, choose a model, or rescue an unstable result.
    """
    baseline_fit = payload.get("baseline_fit")
    baseline_available = isinstance(baseline_fit, dict)
    checks = list(calibration.get("checks") or ())
    blocking_checks = [
        dict(check) for check in checks if check.get("assessment") == "blocking"
    ]
    borderline_checks = [
        dict(check) for check in checks if check.get("assessment") == "borderline"
    ]
    missing_checks = [
        dict(check) for check in checks if check.get("assessment") == "not_evaluated"
    ]
    unstable_rows = [
        {
            "source_id": row.get("source_id"),
            "label": row.get("label"),
            "sensitivity_scope": row.get("sensitivity_scope"),
            "sensitivity_assessment": row.get("sensitivity_assessment"),
            "sensitivity_reasons": list(row.get("sensitivity_reasons") or ()),
            "delta_teff": row.get("delta_teff"),
            "delta_feh": row.get("delta_feh"),
            "delta_logg": row.get("delta_logg"),
            "delta_rv_kms": row.get("delta_rv_kms"),
            "delta_chi2_red": row.get("delta_chi2_red"),
        }
        for row in rows
        if row.get("source_kind") != "baseline"
        and row.get("sensitivity_assessment") in {"blocking", "borderline"}
    ]

    if not baseline_available:
        claim_status = "not_evaluated_baseline_missing"
        plain = (
            "No baseline PHOENIX fit is present yet. Use the audit products to "
            "inspect masks and metadata, then run the baseline before judging "
            "parameter stability."
        )
        guidance = (
            "Do not quote stellar parameters from this checkpoint; it is an "
            "audit scaffold only."
        )
    elif blocking_checks:
        claim_status = "exploratory_not_publication_stable"
        plain = (
            "The scaffold is behaving as designed: it found one or more "
            "blocking publication-stability checks. Treat the current fitted "
            "parameters as diagnostic/exploratory rather than publication-grade."
        )
        guidance = (
            "Inspect the limiting checks and referee plots before changing "
            "model assumptions. A lower chi-square in one window is not enough "
            "to promote the result."
        )
    elif borderline_checks or missing_checks:
        claim_status = "exploratory_needs_additional_checks"
        plain = (
            "No blocking instability is summarized, but at least one check is "
            "borderline or not yet evaluated. The result is still a candidate "
            "classification/diagnostic fit, not a calibrated publication fit."
        )
        guidance = (
            "Run the suggested bounded follow-up checks, then regenerate the "
            "summary before using parameters scientifically."
        )
    elif calibration.get("overall_assessment") == "acceptable":
        claim_status = "current_scaffold_checks_passed"
        plain = (
            "The currently summarized scaffold checks are acceptable. This is "
            "a readiness signal for reviewer inspection, not an external "
            "validation against reference-star literature."
        )
        guidance = (
            "Share the summary and proceed to real reference-star validation "
            "before broader claims."
        )
    else:
        claim_status = "not_evaluated"
        plain = (
            "Publication stability cannot be interpreted from the current "
            "checkpoint because the required calibration summary is incomplete."
        )
        guidance = "Run or regenerate the publication summary after the needed checks."

    if unstable_rows:
        worst_scopes = sorted(
            {
                str(row.get("sensitivity_scope"))
                for row in unstable_rows
                if row.get("sensitivity_scope")
            }
        )
    else:
        worst_scopes = []

    return {
        "schema_version": 1,
        "claim_status": claim_status,
        "plain_language_summary": plain,
        "user_guidance": guidance,
        "limiting_checks": blocking_checks + borderline_checks,
        "missing_checks": missing_checks,
        "unstable_fit_rows": unstable_rows,
        "dominant_instability_scopes": worst_scopes,
        "interpretation": (
            "Plain-language interpretation of already-computed publication "
            "readiness, mask, window-set, and recovery checks. This is a "
            "reporting layer only and does not alter the likelihood."
        ),
    }


def _path_with_suffix(path, suffix, extension=None):
    """Return a sibling path with an added suffix and optional new extension."""
    path = Path(path)
    new_extension = path.suffix if extension is None else str(extension)
    if new_extension and not new_extension.startswith("."):
        new_extension = ".{0}".format(new_extension)
    return str(path.with_name("{0}{1}{2}".format(path.stem, suffix, new_extension)))


def _shell_join_multiline(parts):
    if len(parts) <= 3:
        return " ".join(shlex.quote(str(part)) for part in parts)

    def quote(part):
        return shlex.quote(str(part))

    lines = ["{0} {1}".format(quote(parts[0]), quote(parts[1]))]
    index = 2
    while index < len(parts):
        token = str(parts[index])
        if (
            token.startswith("--")
            and index + 1 < len(parts)
            and not str(parts[index + 1]).startswith("--")
        ):
            lines.append("  {0} {1}".format(quote(parts[index]), quote(parts[index + 1])))
            index += 2
        else:
            lines.append("  {0}".format(quote(parts[index])))
            index += 1
    return " \\\n".join(lines)


def _append_option(parts, flag, value):
    if value is None:
        return
    parts.extend([flag, str(value)])


def _publication_replay_parts(args):
    """Return the scientific settings worth carrying into suggested commands."""
    if args is None:
        return [str(EXAMPLE_UVB), "--instrument", "xshooter"]

    defaults = build_parser().parse_args([])
    parts = [str(args.spectrum)]
    _append_option(parts, "--instrument", getattr(args, "instrument", "xshooter"))
    _append_option(parts, "--phoenix-dir", getattr(args, "phoenix_dir", None))
    if getattr(args, "allow_assumed_resolution", False):
        parts.append("--allow-assumed-resolution")

    replay_if_changed = (
        ("balmer_window_mode", "--balmer-window-mode"),
        ("norm_mode", "--norm-mode"),
        ("balmer_core_mask", "--balmer-core-mask"),
        ("archive_mask_policy", "--archive-mask-policy"),
        ("resolution_R", "--R"),
        ("baseline_defaults_mode", "--baseline-defaults-mode"),
        ("max_nfev", "--max-nfev"),
        ("rv_grid_n", "--rv-grid-n"),
        ("multistart", "--multistart"),
        ("mdeg", "--mdeg"),
        ("teff", "--teff"),
        ("feh", "--feh"),
        ("logg", "--logg"),
        ("rv", "--rv"),
        ("teff_min", "--teff-min"),
        ("teff_max", "--teff-max"),
        ("feh_min", "--feh-min"),
        ("feh_max", "--feh-max"),
        ("logg_min", "--logg-min"),
        ("logg_max", "--logg-max"),
        ("rv_min", "--rv-min"),
        ("rv_max", "--rv-max"),
    )
    for attr, flag in replay_if_changed:
        value = getattr(args, attr, None)
        default = getattr(defaults, attr, None)
        if value != default:
            _append_option(parts, flag, value)
    return parts


def _summary_artifact_paths(output_json, suffix):
    return {
        "md": _path_with_suffix(output_json, "{0}_summary".format(suffix), ".md"),
        "csv": _path_with_suffix(output_json, "{0}_summary".format(suffix), ".csv"),
        "plot": _path_with_suffix(output_json, "{0}_summary".format(suffix), ".png"),
    }


def _publication_command(
    args,
    output_json,
    *,
    extra_parts=(),
    output_plot=None,
    systematic_csv=None,
    injection_csv=None,
    include_summary_outputs=True,
):
    parts = ["python", "examples/publication_quality_xshooter_uvb.py"]
    parts.extend(_publication_replay_parts(args))
    parts.extend(extra_parts)
    parts.extend(["--output-json", str(output_json)])
    _append_option(parts, "--output-plot", output_plot)
    _append_option(parts, "--output-systematic-results-csv", systematic_csv)
    _append_option(parts, "--output-injection-recovery-csv", injection_csv)
    if include_summary_outputs:
        paths = _summary_artifact_paths(output_json, "")
        parts.extend(
            [
                "--output-publication-summary-md",
                paths["md"],
                "--output-publication-summary-csv",
                paths["csv"],
                "--output-publication-summary-plot",
                paths["plot"],
            ]
        )
    return _shell_join_multiline(parts)


def _planned_variant_ids(payload, *, category, limit=2):
    plan = payload.get("systematic_variant_plan") or {}
    variants = [
        item
        for item in plan.get("variants") or ()
        if item.get("category") == category
        and item.get("id") != "baseline"
        and bool(item.get("executable_now", True))
    ]
    variants.sort(key=_variant_priority_key)
    return [str(item.get("id")) for item in variants[: int(limit)] if item.get("id")]


def _check_by_scope(calibration, scope):
    for item in calibration.get("checks") or ():
        if item.get("scope") == scope:
            return item
    return None


def _recommended_next_actions(args, payload, summary):
    """Translate the current checkpoint state into bounded next commands.

    The commands deliberately write new JSON checkpoints derived from
    ``--output-json``.  At present the scaffold reruns the baseline when adding
    expensive follow-up checks; fresh paths prevent accidental overwrites of a
    reviewed checkpoint.
    """
    base_json = (
        str(getattr(args, "output_json"))
        if args is not None and getattr(args, "output_json", None) is not None
        else "/tmp/spyctres_publication_xshooter_uvb.json"
    )
    calibration = summary.get("calibration_interpretation") or {}
    actions = []

    def add(action, priority, reason, *, command=None, status="recommended", expensive=False):
        record = {
            "priority": int(priority),
            "action": action,
            "status": status,
            "reason": reason,
            "requires_phoenix": bool(expensive),
            "expensive": bool(expensive),
            "writes_new_checkpoint": bool(command),
            "checkpoint_policy": (
                "Suggested commands write a fresh checkpoint and rerun from the "
                "input spectrum; they do not mutate the current reviewed JSON."
                if command
                else None
            ),
            "command": command,
        }
        actions.append(record)

    if not summary.get("baseline_available", False):
        output_json = _path_with_suffix(base_json, "_baseline", ".json")
        add(
            "run_baseline_fit",
            1,
            "No baseline PHOENIX fit is present, so parameter-stability checks are not interpretable yet.",
            command=_publication_command(
                args,
                output_json,
                extra_parts=("--run-baseline-fit",),
                output_plot=_path_with_suffix(output_json, "_referee", ".png"),
            ),
            expensive=True,
        )
        return actions

    publication = payload.get("publication_readiness") or {}
    blockers = list(publication.get("blockers") or ())
    if blockers:
        add(
            "review_publication_readiness_blockers",
            1,
            "Publication-readiness blockers remain: {0}.".format(", ".join(blockers)),
            status="manual_review_required",
            expensive=False,
        )

    core_check = _check_by_scope(calibration, "core_mask_fit_sensitivity")
    if core_check is None or core_check.get("assessment") == "not_evaluated":
        variant_ids = _planned_variant_ids(
            payload,
            category="balmer_core_mask",
            limit=2,
        )
        if variant_ids:
            output_json = _path_with_suffix(base_json, "_coremask_variants", ".json")
            add(
                "run_core_mask_fit_sensitivity",
                2,
                "The audit grid is useful, but at least one fitted Balmer-core mask variant is still needed before publication-style claims.",
                command=_publication_command(
                    args,
                    output_json,
                    extra_parts=(
                        "--run-baseline-fit",
                        "--run-systematic-variants",
                        "--systematic-variant-ids",
                        ",".join(variant_ids),
                    ),
                    systematic_csv=_path_with_suffix(
                        output_json,
                        "_systematics",
                        ".csv",
                    ),
                ),
                expensive=True,
            )
        else:
            add(
                "inspect_core_mask_variant_plan",
                2,
                "No executable Balmer-core mask variants are present in the current systematic plan.",
                status="manual_review_required",
                expensive=False,
            )
    elif core_check.get("assessment") in {"blocking", "borderline"}:
        add(
            "review_core_mask_sensitivity",
            2,
            core_check.get("detail") or "Core-mask fit sensitivity is not yet acceptable.",
            status="manual_review_required",
            expensive=False,
        )

    window_check = _check_by_scope(calibration, "window_set_fit_sensitivity")
    if window_check is None or window_check.get("assessment") == "not_evaluated":
        variant_ids = _planned_variant_ids(payload, category="fit_windows", limit=2)
        if variant_ids:
            output_json = _path_with_suffix(base_json, "_windowset_variants", ".json")
            add(
                "run_window_set_sensitivity",
                3,
                "Run at least one single-line or leave-one-line-out Balmer fit to see whether the joint solution is dominated by one window.",
                command=_publication_command(
                    args,
                    output_json,
                    extra_parts=(
                        "--run-baseline-fit",
                        "--run-systematic-variants",
                        "--systematic-variant-ids",
                        ",".join(variant_ids),
                    ),
                    systematic_csv=_path_with_suffix(
                        output_json,
                        "_systematics",
                        ".csv",
                    ),
                ),
                expensive=True,
            )
        else:
            add(
                "inspect_window_variant_plan",
                3,
                "No executable window-set variants are present in the current systematic plan.",
                status="manual_review_required",
                expensive=False,
            )
    elif window_check.get("assessment") in {"blocking", "borderline"}:
        add(
            "review_window_set_sensitivity",
            3,
            window_check.get("detail") or "Window-set fit sensitivity is not yet acceptable.",
            status="manual_review_required",
            expensive=False,
        )

    recovery_check = _check_by_scope(calibration, "same_model_recovery")
    if recovery_check is None or recovery_check.get("assessment") == "not_evaluated":
        output_json = _path_with_suffix(base_json, "_injection_recovery", ".json")
        add(
            "run_same_model_injection_recovery",
            4,
            "Same-model synthetic recovery has not been run; use a small bounded trial set before treating random-noise recovery as understood.",
            command=_publication_command(
                args,
                output_json,
                extra_parts=(
                    "--run-baseline-fit",
                    "--run-injection-recovery",
                    "--injection-recovery-trials",
                    "3",
                ),
                injection_csv=_path_with_suffix(
                    output_json,
                    "_injection_recovery",
                    ".csv",
                ),
            ),
            expensive=True,
        )
    elif recovery_check.get("assessment") in {"blocking", "borderline"}:
        add(
            "review_same_model_recovery",
            4,
            recovery_check.get("detail") or "Same-model recovery is not yet acceptable.",
            status="manual_review_required",
            expensive=False,
        )

    if not actions:
        add(
            "share_summary_for_review",
            5,
            "All currently summarized publication checks are acceptable; share the Markdown/CSV/PNG summary with the reviewer and move to real reference-star validation.",
            status="ready_for_review",
            expensive=False,
        )

    actions.sort(key=lambda item: (int(item["priority"]), str(item["action"])))
    return actions


def _build_publication_comparison_summary(payload, args=None):
    baseline_fit = payload.get("baseline_fit")
    baseline_available = isinstance(baseline_fit, dict)
    rows = []
    headline_flags = set()

    publication = payload.get("publication_readiness") or {}
    if publication and not publication.get("publication_ready", False):
        headline_flags.add("publication_gate_blocked")
    for blocker in publication.get("blockers") or ():
        headline_flags.add("blocker:{0}".format(blocker))

    if baseline_available:
        rows.append(
            _comparison_row_from_fit(
                source_kind="baseline",
                source_id="baseline",
                label="Baseline fit",
                status="ok" if baseline_fit.get("success") else "fit_failed",
                fit_payload=baseline_fit,
                baseline_fit=baseline_fit,
                variant_category="baseline",
                line_residual_diagnostics=payload.get(
                    "baseline_line_residual_diagnostics"
                ),
                all_passed=None,
            )
        )
        for flag in baseline_fit.get("quality_flags") or ():
            headline_flags.add("baseline:{0}".format(flag))
    else:
        headline_flags.add("baseline_not_run")

    systematic = payload.get("systematic_variant_results") or {}
    systematic_records = list(systematic.get("records") or ())
    if systematic.get("status") in {
        "completed_with_errors",
        "completed_with_fit_failures",
    }:
        headline_flags.add("systematic_variants_need_review")
    elif not systematic_records:
        headline_flags.add("systematic_variants_not_run")
    for record in systematic_records:
        fit_payload = record.get("fit") or {}
        row = _comparison_row_from_fit(
            source_kind="systematic_variant",
            source_id=str(record.get("variant_id")),
            label=record.get("label") or str(record.get("variant_id")),
            status=record.get("status"),
            fit_payload=fit_payload,
            baseline_fit=baseline_fit,
            variant_category=record.get("category"),
            window_labels=record.get("window_labels"),
            line_residual_diagnostics=record.get("line_residual_diagnostics"),
            all_passed=None,
            error=record.get("error"),
        )
        rows.append(row)
        if record.get("status") not in {"ok", "skipped"}:
            headline_flags.add(
                "systematic:{0}:{1}".format(
                    record.get("variant_id"),
                    record.get("status"),
                )
            )

    injection = payload.get("injection_recovery") or {}
    injection_records = list(injection.get("records") or ())
    if injection.get("status") in {
        "completed_with_recovery_failures",
        "completed_with_errors",
        "completed_no_successful_trials",
    }:
        headline_flags.add("injection_recovery_needs_review")
    elif not injection_records:
        headline_flags.add("injection_recovery_not_run")
    for record in injection_records:
        row = _comparison_row_from_injection_record(record, baseline_fit)
        rows.append(row)
        if not row.get("all_passed", False):
            headline_flags.add(
                "injection:{0}:failed_tolerance".format(record.get("trial_index"))
            )

    for row in rows:
        if row.get("line_quality_flags"):
            headline_flags.add("line_residual_flags_present")

    calibration = _build_calibration_interpretation(payload, rows)
    for flag in calibration.get("headline_flags") or ():
        headline_flags.add(flag)
    stability = _build_publication_stability_interpretation(
        payload,
        rows,
        calibration,
    )

    if not baseline_available:
        status = "summary_ready_no_baseline_fit"
    elif "publication_gate_blocked" in headline_flags:
        status = "summary_ready_publication_blocked"
    elif (
        calibration.get("overall_assessment") in {"blocking", "borderline"}
        or any("failed" in flag or "error" in flag for flag in headline_flags)
    ):
        status = "summary_ready_needs_review"
    else:
        status = "summary_ready_for_reviewer"

    recommendations = []
    if "baseline_not_run" in headline_flags:
        recommendations.append(
            "Run --run-baseline-fit before interpreting parameter stability."
        )
    if "systematic_variants_not_run" in headline_flags:
        recommendations.append(
            "Run a bounded --run-systematic-variants subset after inspecting the baseline."
        )
    if "injection_recovery_not_run" in headline_flags:
        recommendations.append(
            "Run --run-injection-recovery to test same-model noise/mask recovery."
        )
    if "line_residual_flags_present" in headline_flags:
        recommendations.append(
            "Inspect line-specific residual flags before treating the fit as publication-grade."
        )
    if publication.get("blockers"):
        recommendations.append(
            "Resolve publication-readiness blockers: {0}.".format(
                ", ".join(publication.get("blockers"))
            )
        )
    recommendations.extend(calibration.get("recommendations") or ())

    summary = {
        "schema_version": 1,
        "status": status,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_available": bool(baseline_available),
        "publication_ready": publication.get("publication_ready"),
        "publication_blockers": list(publication.get("blockers") or ()),
        "headline_flags": sorted(headline_flags),
        "comparison_rows": rows,
        "max_abs_parameter_shifts": _max_abs_shift_by_kind(rows),
        "calibration_interpretation": calibration,
        "publication_stability_interpretation": stability,
        "systematic_variant_status": systematic.get("status"),
        "injection_recovery_status": injection.get("status"),
        "recommendations": recommendations,
        "interpretation": (
            "Compact reviewer summary built from existing JSON products. It "
            "does not run additional fits. Parameter shifts summarize "
            "sensitivity, not model preference; line flags and readiness "
            "blockers should be reviewed before scientific claims."
        ),
    }
    summary["recommended_next_actions"] = _recommended_next_actions(
        args,
        payload,
        summary,
    )
    return summary


def _write_publication_summary_csv(path, summary):
    if path is None:
        return
    columns = [
        "source_kind",
        "source_id",
        "label",
        "variant_category",
        "window_labels",
        "status",
        "success",
        "teff",
        "feh",
        "logg",
        "rv_kms",
        "chi2_red",
        "delta_teff",
        "delta_feh",
        "delta_logg",
        "delta_rv_kms",
        "delta_chi2_red",
        "all_passed",
        "sensitivity_scope",
        "sensitivity_assessment",
        "sensitivity_reasons",
        "quality_flags",
        "line_residual_status",
        "n_line_residuals",
        "max_line_chi2_red_proxy",
        "max_line_rms_fractional_residual",
        "problem_lines",
        "line_quality_flags",
        "line_info_flags",
        "error",
    ]
    rows = []
    for row in summary.get("comparison_rows", ()):
        rows.append(
            {
                **{
                    key: row.get(key)
                    for key in columns
                    if key
                    not in {
                        "quality_flags",
                        "window_labels",
                        "sensitivity_reasons",
                        "problem_lines",
                        "line_quality_flags",
                        "line_info_flags",
                    }
                },
                "window_labels": ";".join(row.get("window_labels") or ()),
                "sensitivity_reasons": " | ".join(
                    row.get("sensitivity_reasons") or ()
                ),
                "quality_flags": ";".join(row.get("quality_flags") or ()),
                "problem_lines": ";".join(row.get("problem_lines") or ()),
                "line_quality_flags": ";".join(row.get("line_quality_flags") or ()),
                "line_info_flags": ";".join(row.get("line_info_flags") or ()),
            }
        )
    atomic_write_csv_rows(path, columns, rows)


def _format_md_value(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        if abs(value) >= 100:
            return "{0:.1f}".format(value)
        if abs(value) >= 10:
            return "{0:.2f}".format(value)
        return "{0:.4g}".format(value)
    return str(value)


def _markdown_table(columns, rows):
    if not rows:
        return "_No rows available._\n"
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "| " + " | ".join("---" for _key, _label in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(item) for item in value)
            values.append(_format_md_value(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_publication_summary_markdown(path, summary):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_rows = [
        row
        for row in summary.get("comparison_rows", ())
        if row.get("source_kind") == "baseline"
    ]
    systematic_rows = [
        row
        for row in summary.get("comparison_rows", ())
        if row.get("source_kind") == "systematic_variant"
    ]
    injection_rows = [
        row
        for row in summary.get("comparison_rows", ())
        if row.get("source_kind") == "injection_recovery_trial"
    ]
    calibration = summary.get("calibration_interpretation") or {}
    stability = summary.get("publication_stability_interpretation") or {}
    param_columns = [
        ("source_id", "id"),
        ("status", "status"),
        ("variant_category", "category"),
        ("sensitivity_assessment", "assessment"),
        ("teff", "Teff"),
        ("delta_teff", "ΔTeff"),
        ("feh", "[Fe/H]"),
        ("delta_feh", "Δ[Fe/H]"),
        ("logg", "logg"),
        ("delta_logg", "Δlogg"),
        ("rv_kms", "RV"),
        ("delta_rv_kms", "ΔRV"),
        ("chi2_red", "χ²ν"),
        ("all_passed", "passed"),
    ]
    calibration_columns = [
        ("scope", "scope"),
        ("item", "item"),
        ("assessment", "assessment"),
        ("detail", "detail"),
    ]
    stability_row_columns = [
        ("source_id", "id"),
        ("sensitivity_scope", "scope"),
        ("sensitivity_assessment", "assessment"),
        ("delta_teff", "ΔTeff"),
        ("delta_feh", "Δ[Fe/H]"),
        ("delta_logg", "Δlogg"),
        ("delta_rv_kms", "ΔRV"),
        ("delta_chi2_red", "Δχ²ν"),
        ("sensitivity_reasons", "reason"),
    ]
    line_columns = [
        ("source_id", "id"),
        ("line_residual_status", "line status"),
        ("n_line_residuals", "n lines"),
        ("max_line_chi2_red_proxy", "max line χ²ν proxy"),
        ("problem_lines", "problem lines"),
        ("line_quality_flags", "line flags"),
        ("line_info_flags", "line info"),
    ]
    content = [
        "# Spyctres publication workflow summary",
        "",
        "Status: `{0}`".format(summary.get("status")),
        "",
        "Interpretation: {0}".format(summary.get("interpretation")),
        "",
        "## Headline flags",
        "",
    ]
    flags = summary.get("headline_flags") or ()
    if flags:
        content.extend("- `{0}`".format(flag) for flag in flags)
    else:
        content.append("- none")
    content.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    recommendations = summary.get("recommendations") or ()
    if recommendations:
        content.extend("- {0}".format(item) for item in recommendations)
    else:
        content.append("- No immediate summary-level recommendation.")
    content.extend(
        [
            "",
            "## Publication-stability interpretation",
            "",
            "Claim status: `{0}`".format(
                stability.get("claim_status", "not_evaluated")
            ),
            "",
            stability.get(
                "plain_language_summary",
                "No publication-stability interpretation is available.",
            ),
            "",
            "Guidance: {0}".format(
                stability.get("user_guidance", "Regenerate the summary.")
            ),
            "",
            "### Limiting fit rows",
            "",
            _markdown_table(
                stability_row_columns,
                stability.get("unstable_fit_rows") or (),
            ),
            "",
        ]
    )
    content.extend(
        [
            "",
            "## Suggested next commands",
            "",
            (
                "These are bounded follow-up suggestions generated from the "
                "current checkpoint state. Commands write fresh checkpoints "
                "derived from the current `--output-json` path."
            ),
            "",
        ]
    )
    next_actions = summary.get("recommended_next_actions") or ()
    if next_actions:
        for action in next_actions:
            content.append(
                "- `{0}` ({1}): {2}".format(
                    action.get("action"),
                    action.get("status"),
                    action.get("reason"),
                )
            )
            command = action.get("command")
            if command:
                content.extend(["", "```bash", command, "```", ""])
    else:
        content.append("- No suggested next command.")
    content.extend(
        [
            "",
            "## Calibration interpretation",
            "",
            "Overall assessment: `{0}`".format(
                calibration.get("overall_assessment", "not_evaluated")
            ),
            "",
            _markdown_table(calibration_columns, calibration.get("checks") or ()),
            "",
            "## Baseline",
            "",
            _markdown_table(param_columns, baseline_rows),
            "",
            "## Systematic variants",
            "",
            _markdown_table(param_columns, systematic_rows),
            "",
            "## Injection/recovery",
            "",
            _markdown_table(param_columns, injection_rows),
            "",
            "## Line residual overview",
            "",
            _markdown_table(
                line_columns,
                baseline_rows + systematic_rows + injection_rows,
            ),
            "",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def _write_publication_summary_plot(path, summary):
    if path is None:
        return
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in summary.get("comparison_rows", ())
        if row.get("source_kind") != "baseline" and row.get("status") == "ok"
    ]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.5, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    keys = [
        ("delta_teff", "ΔTeff [K]"),
        ("delta_feh", "Δ[Fe/H] [dex]"),
        ("delta_logg", "Δlogg [dex]"),
        ("delta_rv_kms", "ΔRV [km/s]"),
    ]
    if not rows:
        for ax in axes:
            ax.axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No completed systematic/injection rows to plot.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    else:
        labels = [str(row.get("source_id")) for row in rows]
        x = np.arange(len(rows))
        severity_colors = {
            "blocking": "tab:red",
            "borderline": "tab:orange",
            "acceptable": "tab:green",
            "not_evaluated": "0.6",
        }
        colors = []
        for row in rows:
            assessment = row.get("sensitivity_assessment")
            if assessment in severity_colors:
                colors.append(severity_colors[assessment])
            elif row.get("source_kind") == "systematic_variant":
                colors.append("tab:blue")
            else:
                colors.append("tab:green")
        for ax, (key, ylabel) in zip(axes, keys):
            values = [
                _finite_or_none(row.get(key))
                for row in rows
            ]
            numeric = np.asarray(
                [np.nan if value is None else value for value in values],
                dtype=float,
            )
            ax.axhline(0.0, color="0.3", lw=0.8)
            ax.bar(x, numeric, color=colors, alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("Publication workflow parameter-shift summary")
    save_figure(fig, path, dpi=160, bbox_inches=None)
    plt.close(fig)


def _write_publication_summary_outputs(args, payload):
    summary = _build_publication_comparison_summary(payload, args=args)
    payload["publication_summary"] = summary
    _write_publication_summary_csv(args.output_publication_summary_csv, summary)
    _write_publication_summary_markdown(args.output_publication_summary_md, summary)
    _write_publication_summary_plot(args.output_publication_summary_plot, summary)
    return summary


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
        "systematic_variant_results": None,
        "injection_recovery": None,
        "publication_summary": None,
        "baseline_fit": None,
        "baseline_line_residual_diagnostics": None,
    }


def _write_diagnostic_window_csv(path, diagnostic_payload):
    if path is None:
        return
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
    csv_rows = []
    for row in rows:
        csv_rows.append(
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
    atomic_write_csv_rows(path, columns, csv_rows)


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
    rows = []
    for record in records:
        fit = record.get("fit") or {}
        rows.append(
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
    atomic_write_csv_rows(path, columns, rows)


def _write_core_mask_comparison_plot(path, records):
    if path is None:
        return
    if not records:
        return
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
    save_figure(fig, path, dpi=160, bbox_inches=None)
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
    fit_label="baseline",
    ordinary_readiness=None,
    publication_readiness=None,
):
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, collection, exclude_masks)
    print("Running {0} native-grid PHOENIX fit...".format(fit_label), flush=True)
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
    if ordinary_readiness is not None:
        result.summary["spectrum_readiness"] = ordinary_readiness
        result.provenance["spectrum_readiness"] = ordinary_readiness
    if publication_readiness is not None:
        result.summary["publication_readiness"] = publication_readiness
        result.provenance["publication_readiness"] = publication_readiness
    result.summary["archive_mask_policy"] = {
        "policy": str(args.archive_mask_policy),
        "applied": bool(str(args.archive_mask_policy).lower() == "apply"),
    }
    result.provenance["archive_mask_policy"] = dict(
        result.summary["archive_mask_policy"]
    )
    if args.record_input_checksum:
        checksum_policy, input_checksum = input_checksum_provenance(args.spectrum)
        result.provenance["input_checksum_policy"] = checksum_policy
        result.provenance["input_checksum"] = input_checksum
    resolution = _resolution_assumption(args)
    if resolution is not None:
        result.summary["resolution_override"] = resolution
        result.provenance["resolution_override"] = dict(resolution)
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


def _write_baseline_report_json(args, result):
    if not args.output_report_json:
        return
    plot_paths = {"referee_plot": args.output_plot} if args.output_plot else None
    result.save_report_json(
        args.output_report_json,
        plot_paths=plot_paths,
        relative_to=Path(args.output_json).expanduser().resolve().parent,
        report_context={
            "workflow": "examples/publication_quality_xshooter_uvb.py",
            "report_scope": "baseline_fit_only",
            "scaffold_checkpoint_json": args.output_json,
            "note": (
                "The full publication scaffold checkpoint remains in "
                "--output-json; this report envelope contains the baseline "
                "PhoenixFitResult payload and key provenance for hand-off."
            ),
        },
    )
    print("Wrote baseline fit report: {0}".format(args.output_report_json), flush=True)


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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_report_json and not args.run_baseline_fit:
        parser.error("--output-report-json requires --run-baseline-fit.")
    output_path = Path(args.output_json)
    if args.resume and output_path.exists() and not args.force:
        existing = _read_existing(output_path)
        if (
            args.output_publication_summary_md is not None
            or args.output_publication_summary_csv is not None
            or args.output_publication_summary_plot is not None
        ):
            summary = _write_publication_summary_outputs(args, existing)
            _atomic_write_json(output_path, existing)
            print(
                "Publication summary: status={0}, rows={1}".format(
                    summary["status"],
                    len(summary["comparison_rows"]),
                ),
                flush=True,
            )
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
            ordinary_readiness=ordinary,
            publication_readiness=publication,
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
        _write_baseline_report_json(args, baseline_result)
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
        if args.run_systematic_variants:
            if not payload["baseline_fit"].get("success"):
                print(
                    "Skipping systematic variants because the baseline fit failed.",
                    flush=True,
                )
                payload["systematic_variant_results"] = _skipped_systematic_results(
                    "skipped_baseline_failed",
                    "Systematic variants require a successful baseline fit.",
                )
            else:
                payload["systematic_variant_results"] = (
                    _run_selected_systematic_variants(
                        args,
                        segment,
                        payload["systematic_variant_plan"],
                        output_path,
                        payload,
                    )
                )
            _write_systematic_variant_results_csv(
                args.output_systematic_results_csv,
                payload["systematic_variant_results"],
            )
            _atomic_write_json(output_path, payload)
        if args.run_injection_recovery:
            if not payload["baseline_fit"].get("success"):
                print(
                    "Skipping injection/recovery because the baseline fit failed.",
                    flush=True,
                )
                payload["injection_recovery"] = _skipped_injection_recovery(
                    "skipped_baseline_failed",
                    "Synthetic recovery requires a successful baseline fit.",
                    truth=_baseline_truth_from_payload(payload["baseline_fit"]),
                )
            else:
                payload["injection_recovery"] = _run_injection_recovery(
                    args,
                    collection,
                    exclude_masks,
                    baseline_result,
                    payload["baseline_fit"],
                    output_path,
                    payload,
                )
            _write_injection_recovery_csv(
                args.output_injection_recovery_csv,
                payload["injection_recovery"],
            )
            _atomic_write_json(output_path, payload)
    else:
        if args.run_systematic_variants:
            print(
                "Skipping systematic variants: add --run-baseline-fit first.",
                flush=True,
            )
            payload["systematic_variant_results"] = _skipped_systematic_results(
                "skipped_requires_run_baseline_fit",
                "Add --run-baseline-fit before executing fit-level systematic variants.",
            )
            _write_systematic_variant_results_csv(
                args.output_systematic_results_csv,
                payload["systematic_variant_results"],
            )
            _atomic_write_json(output_path, payload)
        if args.run_injection_recovery:
            print(
                "Skipping injection/recovery: add --run-baseline-fit first.",
                flush=True,
            )
            payload["injection_recovery"] = _skipped_injection_recovery(
                "skipped_requires_run_baseline_fit",
                "Add --run-baseline-fit before synthetic injection/recovery.",
            )
            _write_injection_recovery_csv(
                args.output_injection_recovery_csv,
                payload["injection_recovery"],
            )
            _atomic_write_json(output_path, payload)
        print(
            "Audit-only run complete. Add --run-baseline-fit after PHOENIX "
            "configuration is ready.",
            flush=True,
        )
    summary = _write_publication_summary_outputs(args, payload)
    _atomic_write_json(output_path, payload)
    print(
        "Publication summary: status={0}, rows={1}".format(
            summary["status"],
            len(summary["comparison_rows"]),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
