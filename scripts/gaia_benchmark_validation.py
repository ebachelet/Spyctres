#!/usr/bin/env python
"""Validate Spyctres against bundled Gaia FGK Benchmark Stars spectra.

The default mode is intentionally lightweight: read the bundled benchmark
spectra, build the same reviewed setup object a user would inspect before a
fit, and write auditable JSON/CSV/plots. Expensive PHOENIX recovery fits require
``--run-fits``.

Example audit-only run
----------------------
python scripts/gaia_benchmark_validation.py \
  --output-json /tmp/spyctres_gaia_benchmark_audit.json \
  --output-csv /tmp/spyctres_gaia_benchmark_audit.csv \
  --output-summary-plot /tmp/spyctres_gaia_benchmark_audit.png

Example recovery run
--------------------
python scripts/gaia_benchmark_validation.py \
  --run-fits \
  --output-json /tmp/spyctres_gaia_benchmark_fits.json \
  --output-csv /tmp/spyctres_gaia_benchmark_fits.csv \
  --output-summary-plot /tmp/spyctres_gaia_benchmark_fits.png \
  --fit-plot-dir /tmp/spyctres_gaia_benchmark_fit_plots \
  --resume
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "Spyctres").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Spyctres import ensure_matplotlib_config_dir

ensure_matplotlib_config_dir()

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from Spyctres import (
    fit_stellar_spectrum,
    plot_fit_referee,
    plot_model_line_windows,
    read_spectrum,
    suggest_fit_setup,
)
from Spyctres.config import resolve_phoenix_dir
from Spyctres._serialization import (
    atomic_write_csv_rows,
    atomic_write_json,
    json_safe,
    safe_filename,
    save_figure,
)
from Spyctres.fitting import reconstruct_phoenix_legendre_models_for_segments
from Spyctres.phoenix import PhoenixLibrary


DEFAULT_MANIFEST = REPO_ROOT / "examples" / "data" / "gaia_benchmark" / "manifest.json"
ORDINARY_ROLES = {"standard"}
STRESS_ROLE_MARKERS = ("stress", "peculiar", "diagnostic", "unsupported")

RECOVERY_THRESHOLDS = {
    "teff": {
        "label": "Teff",
        "unit": "K",
        "acceptable_abs_delta": 300.0,
        "review_abs_delta": 600.0,
    },
    "logg": {
        "label": "logg",
        "unit": "dex",
        "acceptable_abs_delta": 0.6,
        "review_abs_delta": 1.2,
    },
    "feh": {
        "label": "[Fe/H]",
        "unit": "dex",
        "acceptable_abs_delta": 0.5,
        "review_abs_delta": 1.0,
    },
    "rv_kms": {
        "label": "RV",
        "unit": "km/s",
        "acceptable_abs_delta": 5.0,
        "review_abs_delta": 10.0,
    },
}

AGGREGATE_RECOVERY_TARGETS = {
    "teff": {
        "median_abs_bias_target": 100.0,
        "robust_scatter_target": 150.0,
        "unit": "K",
    },
    "logg": {
        "median_abs_bias_target": 0.15,
        "robust_scatter_target": 0.25,
        "unit": "dex",
    },
    "feh": {
        "median_abs_bias_target": 0.10,
        "robust_scatter_target": 0.20,
        "unit": "dex",
    },
    "rv_kms": {
        "median_abs_bias_target": 1.0,
        "robust_scatter_target": None,
        "unit": "km/s",
    },
}

BENCHMARK_WINDOW_SETS = {
    "default": {
        "label": "default diagnostic branch windows",
        "regions": None,
        "diagnostic_only": False,
        "description": (
            "Use the reviewed fit setup's recommended Gaia benchmark windows. "
            "For the bundled HARPS subset this is normally H-beta, Mg I b, "
            "Na I D, and H-alpha."
        ),
    },
    "hydrogen_only": {
        "label": "hydrogen-only diagnostic windows",
        "regions": ((4830.0, 4898.0), (6530.0, 6600.0)),
        "diagnostic_only": True,
        "description": (
            "Fit only H-beta and H-alpha to test whether Balmer features are "
            "driving a benchmark recovery failure."
        ),
    },
    "metal_only": {
        "label": "metal-line diagnostic windows",
        "regions": ((5150.0, 5208.0), (5870.0, 5906.0)),
        "diagnostic_only": True,
        "description": (
            "Fit Mg I b and Na I D to test cool-star/metal-line sensitivity. "
            "Na I D can include interstellar absorption, so this is diagnostic."
        ),
    },
    "no_hbeta": {
        "label": "default windows excluding H-beta",
        "regions": ((5150.0, 5208.0), (5870.0, 5906.0), (6530.0, 6600.0)),
        "diagnostic_only": True,
        "description": (
            "Fit Mg I b, Na I D, and H-alpha to test whether H-beta/DIB "
            "or Balmer-wing mismatch is dominating."
        ),
    },
    "broad_metal_forest": {
        "label": "broad cool-giant optical metal-forest diagnostic",
        "regions": ((5150.0, 5450.0), (6000.0, 6500.0)),
        "diagnostic_only": True,
        "description": (
            "Fit broader optical metal-line forests inside the 480-680 nm "
            "GBSv3 HARPS coverage. This is useful for K-giant diagnostics, "
            "but is more continuum/model-systematics sensitive than the "
            "default branch windows."
        ),
    },
}

GAIA_BENCHMARK_LINE_PLOT_WINDOWS = (
    {
        "id": "h_beta",
        "label": "H-beta",
        "limits_A": (4830.0, 4898.0),
        "markers_A": (4861.33,),
    },
    {
        "id": "mg_b_triplet",
        "label": "Mg I b triplet",
        "limits_A": (5150.0, 5208.0),
        "markers_A": (5167.32, 5172.68, 5183.60),
    },
    {
        "id": "na_d",
        "label": "Na I D",
        "limits_A": (5870.0, 5906.0),
        "markers_A": (5889.95, 5895.92),
    },
    {
        "id": "ca_i_6162",
        "label": "Ca I 6162",
        "limits_A": (6138.0, 6186.0),
        "markers_A": (6162.17,),
    },
    {
        "id": "ca_i_6439",
        "label": "Ca I 6439",
        "limits_A": (6418.0, 6462.0),
        "markers_A": (6439.08,),
    },
    {
        "id": "h_alpha",
        "label": "H-alpha",
        "limits_A": (6530.0, 6600.0),
        "markers_A": (6562.80,),
    },
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Gaia FGK Benchmark Stars validation runner. By default this reads "
            "the bundled benchmark spectra, records setup/readiness metadata, "
            "and writes summary artifacts. PHOENIX recovery fits require "
            "--run-fits."
        ),
        epilog=(
            "Audit-only:\n"
            "  python scripts/gaia_benchmark_validation.py "
            "--output-json /tmp/gbs_audit.json "
            "--output-csv /tmp/gbs_audit.csv "
            "--output-summary-plot /tmp/gbs_audit.png\n\n"
            "Recovery fits:\n"
            "  python scripts/gaia_benchmark_validation.py --run-fits "
            "--output-json /tmp/gbs_fits.json "
            "--output-csv /tmp/gbs_fits.csv "
            "--output-summary-plot /tmp/gbs_fits.png "
            "--fit-plot-dir /tmp/gbs_fit_plots --resume"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Gaia benchmark manifest JSON. Defaults to bundled examples/data subset.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_gaia_benchmark_validation.json",
        help="Atomic JSON checkpoint/output path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional compact CSV summary path.",
    )
    parser.add_argument(
        "--output-summary-plot",
        default=None,
        help="Optional compact PNG/SVG/PDF validation summary plot.",
    )
    parser.add_argument(
        "--fit-plot-dir",
        default=None,
        help="Optional directory for per-target referee fit plots when --run-fits is set.",
    )
    parser.add_argument(
        "--line-plot-dir",
        default=None,
        help=(
            "Optional directory for per-target benchmark line-window plots when "
            "--run-fits is set. These plots zoom in on fixed diagnostic windows "
            "such as H-beta, Mg I b, Na I D, Ca I, and H-alpha."
        ),
    )
    parser.add_argument(
        "--line-plot-reference-model",
        action="store_true",
        help=(
            "When --line-plot-dir is used, also overlay a diagnostic PHOENIX "
            "model evaluated at the manifest reference Teff/logg/[Fe/H]. The "
            "reference values are not used as fit priors; the overlay uses the "
            "best-fit RV because the manifest does not provide an independent RV."
        ),
    )
    parser.add_argument(
        "--run-fits",
        action="store_true",
        help="Run PHOENIX fits. Without this flag, only ingestion/setup/audit metadata are written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing JSON output, skipping completed records.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected targets even when --resume finds existing records.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Select a target by HIP, HD, common name, or filename stem. Repeat for several targets.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional cap on selected targets, useful for quick manual tests.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="standard",
        help="Setup/defaults mode used to choose diagnostic windows and search budget.",
    )
    parser.add_argument(
        "--bounds-policy",
        choices=("benchmark_fgk", "public_defaults"),
        default="benchmark_fgk",
        help=(
            "benchmark_fgk uses one broad, fixed FGK/metal-poor search box for "
            "all targets. public_defaults uses the normal branch defaults and "
            "may not cover very metal-poor stars such as HD140283."
        ),
    )
    parser.add_argument(
        "--window-set",
        choices=tuple(BENCHMARK_WINDOW_SETS),
        default="default",
        help=(
            "Benchmark validation window set. The default uses the reviewed "
            "diagnostic-branch setup. Other choices are diagnostic-only "
            "sensitivity checks for failures such as cool-giant recovery."
        ),
    )
    parser.add_argument(
        "--error-floor-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional fractional error floor added in quadrature during fits. "
            "Keep 0 for ordinary validation; non-zero values are diagnostic "
            "checks for model/continuum/line-list systematics dominating tiny "
            "formal errors."
        ),
    )
    parser.add_argument(
        "--wave-medium",
        choices=("reader", "unknown", "air", "vacuum"),
        default="reader",
        help=(
            "Wavelength-medium policy for the benchmark spectrum. 'reader' uses "
            "the Gaia benchmark reader profile, which treats bundled R42KNorm "
            "wavelengths as air based on optical line-center validation. Use "
            "'unknown' or 'vacuum' only for explicit sensitivity checks."
        ),
    )
    parser.add_argument(
        "--phoenix-dir",
        default=None,
        help="Optional PHOENIX model directory. If omitted, Spyctres config is used.",
    )
    parser.add_argument("--multistart", type=int, default=2)
    parser.add_argument("--rv-grid-n", type=int, default=21)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--mdeg", type=int, default=2)
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def _atomic_write_csv(path, payload):
    if path is None:
        return
    columns = [
        "target_id",
        "name",
        "hd",
        "spectral_type",
        "validation_role",
        "status",
        "teff_ref",
        "teff_fit",
        "delta_teff",
        "teff_assessment",
        "logg_ref",
        "logg_fit",
        "delta_logg",
        "logg_assessment",
        "feh_ref",
        "feh_fit",
        "delta_feh",
        "feh_assessment",
        "rv_ref",
        "rv_fit",
        "delta_rv_kms",
        "rv_assessment",
        "overall_assessment",
        "chi2_red",
        "quality_flags",
        "fit_quality_assessment",
        "window_set",
        "error_floor_fraction",
        "fit_window_A",
        "n_pixels",
        "n_used",
        "warnings",
    ]
    rows = [_csv_row(record) for record in payload.get("results", [])]
    atomic_write_csv_rows(path, columns, rows)


def _load_existing(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("results", [])
    return {
        str(record.get("target_id")): record
        for record in records
        if isinstance(record, dict) and record.get("target_id")
    }


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    spectra = payload.get("spectra")
    if not isinstance(spectra, list) or not spectra:
        raise ValueError("Gaia benchmark manifest requires a non-empty 'spectra' list.")
    return path, payload, [dict(row) for row in spectra]


def _target_id(row):
    return str(row.get("hip") or Path(str(row.get("file", "target"))).stem)


def _target_aliases(row):
    aliases = {
        _target_id(row),
        str(row.get("hd", "")),
        str(row.get("name", "")),
        Path(str(row.get("file", ""))).stem,
        Path(str(row.get("file", ""))).name,
    }
    return {alias.strip().lower() for alias in aliases if alias and alias.strip()}


def select_manifest_rows(rows, selected=None, max_targets=None):
    if selected:
        requested = {str(item).strip().lower() for item in selected if str(item).strip()}
        rows = [row for row in rows if _target_aliases(row) & requested]
        if not rows:
            raise ValueError("No Gaia benchmark manifest rows matched --target.")
    if max_targets is not None:
        rows = rows[: int(max_targets)]
    return rows


def _maybe_override_wave_medium(segment, wave_medium):
    wave_medium = str(wave_medium).strip().lower()
    if wave_medium == "reader":
        return segment
    meta = dict(segment.meta)
    meta["wave_medium"] = wave_medium
    meta["wave_medium_source"] = "user_override"
    meta["wave_medium_warning"] = (
        "This value was supplied by the validation runner user and overrides "
        "the Gaia benchmark reader wavelength-medium profile."
    )
    return segment.copy(wave_medium=wave_medium, meta=meta)


def _segment_summary(segment):
    mask = np.asarray(segment.mask, dtype=bool)
    finite = np.isfinite(segment.wave) & np.isfinite(segment.flux)
    good_wave = segment.wave[mask & np.isfinite(segment.wave)]
    return {
        "name": segment.name,
        "n_pixels": int(segment.wave.size),
        "n_finite": int(np.sum(finite)),
        "n_used": int(np.sum(mask)),
        "wave_min_A": None if good_wave.size == 0 else float(np.nanmin(good_wave)),
        "wave_max_A": None if good_wave.size == 0 else float(np.nanmax(good_wave)),
        "wave_medium": segment.wave_medium,
        "observer_frame": segment.observer_frame,
        "stellar_rest_status": segment.stellar_rest_status,
        "resolution": None
        if segment.resolution is None
        else segment.resolution.to_metadata(),
        "flux_state": segment.meta.get("flux_state"),
    }


def _reference(row):
    return {
        "teff": _float_or_none(row.get("teff_ref")),
        "logg": _float_or_none(row.get("logg_ref")),
        "feh": _float_or_none(row.get("feh_ref")),
    }


def _float_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _is_stress_role(role):
    role = str(role or "").strip().lower()
    return role not in ORDINARY_ROLES or any(marker in role for marker in STRESS_ROLE_MARKERS)


def _benchmark_fit_kwargs(setup_payload, args, row=None):
    fit_kwargs = dict(setup_payload.get("fit_kwargs") or {})
    window_set = str(getattr(args, "window_set", "default")).strip().lower()
    if window_set not in BENCHMARK_WINDOW_SETS:
        raise ValueError(
            "window_set must be one of {0}.".format(
                ", ".join(sorted(BENCHMARK_WINDOW_SETS))
            )
        )
    window_definition = BENCHMARK_WINDOW_SETS[window_set]
    if window_definition["regions"] is not None:
        fit_kwargs["regions"] = [
            (float(lo), float(hi)) for lo, hi in window_definition["regions"]
        ]
    error_floor_fraction = float(getattr(args, "error_floor_fraction", 0.0))
    if not math.isfinite(error_floor_fraction) or error_floor_fraction < 0.0:
        raise ValueError("error_floor_fraction must be finite and >= 0.")
    if args.bounds_policy == "benchmark_fgk":
        role = None if row is None else row.get("validation_role")
        if _is_stress_role(role):
            fit_kwargs.update(
                {
                    "p0": (5500.0, -1.0, 3.5, 0.0),
                    "bounds": (
                        (4000.0, -2.5, 1.5, -150.0),
                        (7000.0, 0.5, 5.5, 150.0),
                    ),
                    "coarse_teff_grid": [4500.0, 5500.0, 6500.0],
                    "coarse_feh_grid": [-2.0, -1.0, 0.0],
                    "coarse_logg_grid": [2.5, 3.5, 4.5],
                }
            )
        else:
            fit_kwargs.update(
                {
                    "p0": (5500.0, 0.0, 4.0, 0.0),
                    "bounds": (
                        (4000.0, -1.0, 1.0, -150.0),
                        (7000.0, 0.5, 5.5, 150.0),
                    ),
                    "coarse_teff_grid": [4500.0, 5500.0, 6500.0],
                    "coarse_feh_grid": [-1.0, 0.0, 0.5],
                    "coarse_logg_grid": [1.5, 3.0, 4.5],
                }
            )
    fit_kwargs.update(
        {
            "multistart": int(args.multistart),
            "rv_grid_n": int(args.rv_grid_n),
            "max_nfev": int(args.max_nfev),
            "mdeg": int(args.mdeg),
            "forward_model": "native_interp",
            "physical_init": "coarse",
            "rv_init": "grid",
            "error_floor_fraction": error_floor_fraction,
        }
    )
    return fit_kwargs


def _fit_policy_record(args, fit_kwargs):
    window_set = str(getattr(args, "window_set", "default")).strip().lower()
    window_definition = dict(BENCHMARK_WINDOW_SETS.get(window_set, {}))
    if window_definition.get("regions") is not None:
        window_definition["regions"] = [
            [float(lo), float(hi)] for lo, hi in window_definition["regions"]
        ]
    return {
        "run_fits": bool(args.run_fits),
        "fit_mode": args.fit_mode,
        "bounds_policy": args.bounds_policy,
        "window_set": window_set,
        "window_set_definition": window_definition,
        "error_floor_fraction": float(getattr(args, "error_floor_fraction", 0.0)),
        "error_floor_interpretation": (
            "Non-zero error floors are diagnostic checks for systematics in "
            "model, continuum, LSF, line lists, or formal uncertainties. They "
            "must be reported separately from ordinary no-floor recovery."
        ),
        "reference_parameters_used_as_priors": False,
        "reference_parameters_used_for_postfit_deltas_only": True,
        "wave_medium_override": args.wave_medium,
        "fit_kwargs": json_safe(fit_kwargs),
    }


def _assessment(delta, param):
    if delta is None:
        return "missing"
    abs_delta = abs(float(delta))
    limits = RECOVERY_THRESHOLDS[param]
    if abs_delta <= limits["acceptable_abs_delta"]:
        return "within_first_pass_tolerance"
    if abs_delta <= limits["review_abs_delta"]:
        return "review"
    return "outside_review_tolerance"


def _fit_deltas(record):
    fit = record.get("fit") or {}
    reference = record.get("reference") or {}
    deltas = {}
    assessments = {}
    for param in ("teff", "logg", "feh", "rv_kms"):
        fitted = _float_or_none(fit.get(param))
        ref = _float_or_none(reference.get(param))
        delta = None if fitted is None or ref is None else fitted - ref
        deltas[param] = delta
        assessments[param] = _assessment(delta, param)
    return deltas, assessments


def _overall_assessment(record):
    if record.get("status") != "ok":
        return record.get("status", "not_fit")
    _deltas, assessments = _fit_deltas(record)
    values = [assessments[param] for param in ("teff", "logg", "feh")]
    if assessments.get("rv_kms") not in {None, "missing"}:
        values.append(assessments["rv_kms"])
    if any(value == "outside_review_tolerance" for value in values):
        return "outside_review_tolerance"
    if any(value in {"review", "missing"} for value in values):
        return "review"
    return "within_first_pass_tolerance"


def _fit_quality_assessment(record):
    """Classify formal fit quality separately from benchmark parameter recovery."""
    if record.get("status") != "ok":
        return record.get("status", "not_fit")
    fit = record.get("fit") or {}
    chi2_red = _float_or_none(fit.get("chi2_red"))
    if chi2_red is None:
        return "missing_formal_fit_quality"
    flags = {str(flag) for flag in record.get("quality_flags", [])}
    if chi2_red <= 5.0 and not flags.intersection(
        {"high_chi2", "structured_residuals", "residual_slope"}
    ):
        return "nominal_formal_fit_quality"
    if chi2_red <= 100.0:
        return "flagged_formal_fit_quality"
    return "systematics_dominated_formal_chi2"


def _robust_scatter(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    return 1.4826 * mad


def _aggregate_recovery_statistics(records):
    stats = {}
    for param in ("teff", "logg", "feh", "rv_kms"):
        deltas = []
        for record in records:
            record_deltas, _assessments = _fit_deltas(record)
            delta = _float_or_none(record_deltas.get(param))
            if delta is not None:
                deltas.append(delta)
        targets = dict(AGGREGATE_RECOVERY_TARGETS[param])
        if not deltas:
            stats[param] = {
                "n": 0,
                "median_bias": None,
                "median_abs_bias": None,
                "robust_scatter": None,
                "targets": targets,
                "target_status": "not_evaluated_no_reference_deltas",
            }
            continue
        arr = np.asarray(deltas, dtype=float)
        median_bias = float(np.nanmedian(arr))
        median_abs_bias = float(abs(median_bias))
        robust_scatter = _robust_scatter(arr)
        bias_ok = median_abs_bias <= float(targets["median_abs_bias_target"])
        scatter_target = targets.get("robust_scatter_target")
        scatter_ok = True if scatter_target is None else (
            robust_scatter is not None and robust_scatter <= float(scatter_target)
        )
        stats[param] = {
            "n": int(arr.size),
            "median_bias": median_bias,
            "median_abs_bias": median_abs_bias,
            "robust_scatter": robust_scatter,
            "targets": targets,
            "target_status": (
                "meets_reporting_target" if bias_ok and scatter_ok else "outside_reporting_target"
            ),
        }
    return stats


def _csv_row(record):
    deltas, assessments = _fit_deltas(record)
    reference = record.get("reference") or {}
    fit = record.get("fit") or {}
    segment = record.get("segment") or {}
    setup = record.get("setup") or {}
    fit_policy = record.get("fit_policy") or {}
    fit_kwargs = ((record.get("fit_policy") or {}).get("fit_kwargs") or {})
    regions = fit_kwargs.get("regions") or ((setup.get("fit_kwargs") or {}).get("regions") or [])
    width = 0.0
    for region in regions:
        if len(region) >= 2:
            width += max(0.0, float(region[1]) - float(region[0]))
    return {
        "target_id": record.get("target_id"),
        "name": record.get("name"),
        "hd": record.get("hd"),
        "spectral_type": record.get("spectral_type"),
        "validation_role": record.get("validation_role"),
        "status": record.get("status"),
        "teff_ref": reference.get("teff"),
        "teff_fit": fit.get("teff"),
        "delta_teff": deltas.get("teff"),
        "teff_assessment": assessments.get("teff"),
        "logg_ref": reference.get("logg"),
        "logg_fit": fit.get("logg"),
        "delta_logg": deltas.get("logg"),
        "logg_assessment": assessments.get("logg"),
        "feh_ref": reference.get("feh"),
        "feh_fit": fit.get("feh"),
        "delta_feh": deltas.get("feh"),
        "feh_assessment": assessments.get("feh"),
        "rv_ref": reference.get("rv_kms"),
        "rv_fit": fit.get("rv_kms"),
        "delta_rv_kms": deltas.get("rv_kms"),
        "rv_assessment": assessments.get("rv_kms"),
        "overall_assessment": _overall_assessment(record),
        "chi2_red": fit.get("chi2_red"),
        "quality_flags": ";".join(str(item) for item in record.get("quality_flags", [])),
        "fit_quality_assessment": _fit_quality_assessment(record),
        "window_set": fit_policy.get("window_set"),
        "error_floor_fraction": fit_policy.get("error_floor_fraction"),
        "fit_window_A": width,
        "n_pixels": segment.get("n_pixels"),
        "n_used": segment.get("n_used"),
        "warnings": ";".join(str(item) for item in record.get("warnings", [])),
    }


def summarize_payload(records):
    statuses = Counter(str(record.get("status", "unknown")) for record in records)
    role_counts = Counter(str(record.get("validation_role", "unknown")) for record in records)
    ordinary = [
        record
        for record in records
        if str(record.get("validation_role", "")).lower() in ORDINARY_ROLES
        and record.get("status") == "ok"
    ]
    ordinary_assessments = Counter(_overall_assessment(record) for record in ordinary)
    ordinary_fit_quality = Counter(_fit_quality_assessment(record) for record in ordinary)
    return {
        "n_records": int(len(records)),
        "by_status": dict(sorted(statuses.items())),
        "by_validation_role": dict(sorted(role_counts.items())),
        "ordinary_roles": sorted(ORDINARY_ROLES),
        "ordinary_recovery_n": int(len(ordinary)),
        "ordinary_recovery_assessments": dict(sorted(ordinary_assessments.items())),
        "ordinary_fit_quality_assessments": dict(sorted(ordinary_fit_quality.items())),
        "thresholds": RECOVERY_THRESHOLDS,
        "per_star_engineering_gates": RECOVERY_THRESHOLDS,
        "ordinary_aggregate_recovery_targets": AGGREGATE_RECOVERY_TARGETS,
        "ordinary_aggregate_recovery": _aggregate_recovery_statistics(ordinary),
        "notes": [
            "Reference parameters are used only for post-fit deltas.",
            "Stress/diagnostic targets are reported separately from ordinary recovery statistics.",
            "Aggregate recovery targets are reporting goals for ordinary FGK samples, not per-star tuning criteria.",
            "Formal fit quality is reported separately from parameter recovery because high-S/N benchmark spectra can be dominated by model/systematic residuals.",
            "Local covariance errors, when present in fit outputs, do not include external systematic uncertainty.",
        ],
    }


def _write_summary_plot(path, payload):
    if path is None:
        return
    records = payload.get("results", [])
    fitted = [record for record in records if record.get("status") == "ok"]

    if fitted:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
        colors = {
            "within_first_pass_tolerance": "tab:green",
            "review": "tab:orange",
            "outside_review_tolerance": "tab:red",
            "missing": "tab:gray",
        }
        for ax, param in zip(axes, ("teff", "logg", "feh")):
            limits = RECOVERY_THRESHOLDS[param]
            labels = []
            values = []
            assessments = []
            ordinary_markers = []
            for record in fitted:
                deltas, record_assessments = _fit_deltas(record)
                if deltas[param] is None:
                    continue
                labels.append(record.get("target_id"))
                values.append(float(deltas[param]))
                assessments.append(record_assessments[param])
                ordinary_markers.append(
                    str(record.get("validation_role", "")).lower() in ORDINARY_ROLES
                )
            x = np.arange(len(values))
            ax.axhline(0.0, color="0.2", lw=0.8)
            ax.axhspan(
                -limits["acceptable_abs_delta"],
                limits["acceptable_abs_delta"],
                color="tab:green",
                alpha=0.10,
                label="first-pass tolerance",
            )
            ax.axhspan(
                -limits["review_abs_delta"],
                -limits["acceptable_abs_delta"],
                color="tab:orange",
                alpha=0.08,
            )
            ax.axhspan(
                limits["acceptable_abs_delta"],
                limits["review_abs_delta"],
                color="tab:orange",
                alpha=0.08,
                label="review zone",
            )
            for xi, yi, assessment, is_ordinary in zip(
                x,
                values,
                assessments,
                ordinary_markers,
            ):
                color = colors.get(assessment, "tab:gray")
                if is_ordinary:
                    ax.scatter(xi, yi, c=color, s=55, zorder=3)
                else:
                    ax.scatter(
                        xi,
                        yi,
                        marker="D",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.5,
                        s=65,
                        zorder=3,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title("{0} fit - reference".format(limits["label"]))
            ax.set_ylabel("{0} ({1})".format(limits["label"], limits["unit"]))
            ax.grid(alpha=0.25)
        axes[0].scatter([], [], c="tab:green", s=55, label="ordinary target")
        axes[0].scatter(
            [],
            [],
            marker="D",
            facecolors="none",
            edgecolors="tab:gray",
            linewidths=1.5,
            s=65,
            label="stress/diagnostic target",
        )
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Gaia FGK Benchmark Stars recovery summary", y=1.03)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        teff = []
        logg = []
        labels = []
        for record in records:
            reference = record.get("reference") or {}
            if reference.get("teff") is None or reference.get("logg") is None:
                continue
            teff.append(float(reference["teff"]))
            logg.append(float(reference["logg"]))
            labels.append(record.get("target_id"))
        if teff:
            ax.scatter(teff, logg, s=60, color="tab:blue")
            for x, y, label in zip(teff, logg, labels):
                ax.annotate(str(label), (x, y), xytext=(4, 4), textcoords="offset points")
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xlabel("Reference Teff (K)")
            ax.set_ylabel("Reference logg")
            ax.set_title("Gaia benchmark subset coverage (audit-only)")
        else:
            ax.text(0.5, 0.5, "No fitted or reference records", ha="center", va="center")
            ax.set_axis_off()
    save_figure(fig, path, dpi=160)
    plt.close(fig)


def _variant_label_for_path(args):
    pieces = [safe_filename(getattr(args, "window_set", "default"), "default")]
    error_floor = float(getattr(args, "error_floor_fraction", 0.0) or 0.0)
    if error_floor > 0.0:
        pieces.append("efloor_{0:g}".format(error_floor).replace(".", "p"))
    return "_".join(pieces)


def _reference_model_overlay(segment, row, result_payload, fit_kwargs, phoenix_lib):
    reference = _reference(row)
    if any(reference.get(key) is None for key in ("teff", "logg", "feh")):
        return {
            "status": "skipped",
            "reason": "missing_reference_parameters",
            "summary": {
                "status": "skipped",
                "reason": "missing_reference_parameters",
            },
        }

    rv_kms = _float_or_none(result_payload.get("rv_kms"))
    rv_bary_kms = _float_or_none(result_payload.get("rv_bary_kms"))
    if rv_kms is None:
        rv_kms = 0.0
    if rv_bary_kms is None:
        rv_bary_kms = 0.0

    diagnostic_fit_result = dict(result_payload)
    diagnostic_fit_result.update(
        {
            "teff": float(reference["teff"]),
            "feh": float(reference["feh"]),
            "logg": float(reference["logg"]),
            "rv_kms": float(rv_kms),
            "rv_bary_kms": float(rv_bary_kms),
            "forward_model": fit_kwargs.get(
                "forward_model", result_payload.get("forward_model", "native_interp")
            ),
            "model_margin_A": float(
                fit_kwargs.get(
                    "model_margin_A",
                    result_payload.get("model_margin_A", 200.0),
                )
            ),
        }
    )

    models, coeffs, used_masks, excluded_masks = (
        reconstruct_phoenix_legendre_models_for_segments(
            segment,
            phoenix_lib,
            diagnostic_fit_result,
            regions=fit_kwargs.get("regions"),
            exclude_regions=fit_kwargs.get("exclude_regions"),
            exclude_mask=fit_kwargs.get("exclude_mask"),
            exclude_masks=fit_kwargs.get("exclude_masks"),
            mask_threshold=fit_kwargs.get("mask_threshold", 0.5),
            error_floor_fraction=fit_kwargs.get("error_floor_fraction", 0.0),
            mdeg=fit_kwargs.get("mdeg", 2),
            rv_bary_kms=rv_bary_kms,
            R=fit_kwargs.get("R"),
            fwhm_kms=fit_kwargs.get("fwhm_kms"),
            forward_model=fit_kwargs.get("forward_model", "native_interp"),
            model_margin_A=float(fit_kwargs.get("model_margin_A", 200.0)),
        )
    )

    summary = {
        "status": "ok",
        "params": {
            "teff": float(reference["teff"]),
            "logg": float(reference["logg"]),
            "feh": float(reference["feh"]),
            "rv_kms": float(rv_kms),
            "rv_bary_kms": float(rv_bary_kms),
        },
        "rv_source": "best_fit_rv",
        "reference_parameters_used_as_priors": False,
        "interpretation": (
            "Diagnostic overlay only: PHOENIX is evaluated at manifest "
            "Teff/logg/[Fe/H] and continuum-adjusted on the same fit pixels. "
            "The fitted RV is reused because the manifest has no independent RV."
        ),
    }
    return {
        "status": "ok",
        "models": tuple(np.asarray(model, dtype=float) for model in models),
        "used_masks": tuple(np.asarray(mask, dtype=bool) for mask in used_masks),
        "excluded_masks": tuple(np.asarray(mask, dtype=bool) for mask in excluded_masks),
        "coeffs": tuple(np.asarray(coeff, dtype=float) for coeff in coeffs),
        "summary": summary,
    }


def _write_line_window_plot(path, segment, result, row, args, reference_overlay=None):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    windows = GAIA_BENCHMARK_LINE_PLOT_WINDOWS

    models = tuple(getattr(result, "models", ()) or ())
    used_masks = tuple(getattr(result, "used_masks", ()) or ())
    if not models:
        raise ValueError(
            "Line-window plots require reconstructed model arrays; run the fit "
            "with reconstruct=True."
        )

    wave = np.asarray(segment.wave, dtype=float)
    flux = np.asarray(segment.flux, dtype=float)
    model = np.asarray(models[0], dtype=float)
    if used_masks:
        used = np.asarray(used_masks[0], dtype=bool)
    else:
        used = np.asarray(segment.mask, dtype=bool)

    reference_models = ()
    reference_used_masks = ()
    reference_summary = None
    if reference_overlay and reference_overlay.get("status") == "ok":
        reference_models = tuple(reference_overlay.get("models") or ())
        reference_used_masks = tuple(reference_overlay.get("used_masks") or ())
        reference_summary = reference_overlay.get("summary")
    reference_model = (
        np.asarray(reference_models[0], dtype=float) if reference_models else None
    )
    reference_used = (
        np.asarray(reference_used_masks[0], dtype=bool)
        if reference_used_masks
        else used
    )

    title = "{0}: {1}".format(
        _target_id(row),
        row.get("name") or Path(str(row.get("file", "spectrum"))).name,
    )
    footer = (
        "Solid model traces are plotted only on pixels used by the fit; dashed "
        "traces mark masked pixels inside a fitted line span. "
        "Reference overlays are diagnostic only and are not fit priors."
    )
    if reference_summary and reference_summary.get("rv_source"):
        footer += " Reference overlay RV source: {0}.".format(
            reference_summary["rv_source"]
        )

    plot_models = [
        {
            "flux": model,
            "label": "best-fit PHOENIX",
            "color": "tab:red",
            "masked_label": "best-fit PHOENIX (masked span)",
        }
    ]
    model_used_masks = [used]
    if reference_model is not None:
        plot_models.append(
            {
                "flux": reference_model,
                "label": "reference Teff/logg/[Fe/H]",
                "color": "tab:blue",
                "masked_label": "reference model (masked span)",
            }
        )
        model_used_masks.append(reference_used)

    fig, _axes = plot_model_line_windows(
        wave,
        flux,
        windows,
        models=plot_models,
        used_mask=used,
        model_used_masks=model_used_masks,
        savepath=path,
        title="Gaia benchmark line-window diagnostics\n{0}".format(title),
        footer=footer,
        ncols=2 if len(windows) > 1 else 1,
    )
    plt.close(fig)
    return str(path)


def _progress_printer(target_id, verbose):
    def callback(event):
        phase = getattr(event, "phase", "")
        if verbose or phase in {"cache", "rv_grid", "local_optimize", "finish"}:
            print("  [{0}] {1}".format(target_id, event), flush=True)

    return callback


def _fit_record(segment, row, setup_payload, fit_kwargs, args, phoenix_lib):
    target_id = _target_id(row)
    line_plot_dir = getattr(args, "line_plot_dir", None)
    fit_plot_dir = getattr(args, "fit_plot_dir", None)
    need_reconstructed_models = bool(fit_plot_dir or line_plot_dir)
    result = fit_stellar_spectrum(
        segment,
        model="phoenix",
        phoenix_lib=phoenix_lib,
        auto_defaults=False,
        reconstruct=need_reconstructed_models,
        warn_unknown=False,
        progress_callback=_progress_printer(target_id, args.verbose),
        **fit_kwargs,
    )
    plot_paths = {}
    if fit_plot_dir:
        plot_dir = Path(fit_plot_dir).expanduser().resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "{0}_fit.png".format(safe_filename(target_id))
        fig, _axes = plot_fit_referee(result, segment=segment, savepath=str(plot_path))
        plt.close(fig)
        plot_paths["referee_fit"] = str(plot_path)

    reference_model_diagnostic = None
    result_preview = result.to_dict(
        include_arrays=False,
        include_local_paths=True,
        plot_paths=None,
    )
    if line_plot_dir:
        line_dir = Path(line_plot_dir).expanduser().resolve()
        line_dir.mkdir(parents=True, exist_ok=True)
        line_path = line_dir / "{0}_{1}_line_windows.png".format(
            safe_filename(target_id),
            _variant_label_for_path(args),
        )
        reference_overlay = None
        if getattr(args, "line_plot_reference_model", False):
            try:
                reference_overlay = _reference_model_overlay(
                    segment,
                    row,
                    result_preview,
                    fit_kwargs,
                    phoenix_lib,
                )
            except Exception as exc:
                reference_overlay = {
                    "status": "error",
                    "summary": {
                        "status": "error",
                        "error": "{0}: {1}".format(type(exc).__name__, exc),
                    },
                }
            reference_model_diagnostic = reference_overlay.get("summary")
        line_plot_path = _write_line_window_plot(
            line_path,
            segment,
            result,
            row,
            args,
            reference_overlay=reference_overlay,
        )
        plot_paths["line_windows"] = str(line_plot_path)

    result_payload = result.to_dict(
        include_arrays=False,
        include_local_paths=True,
        plot_paths=plot_paths or None,
    )
    return {
        "fit": {
            "success": bool(result_payload.get("success")),
            "teff": result_payload.get("teff"),
            "logg": result_payload.get("logg"),
            "feh": result_payload.get("feh"),
            "rv_kms": result_payload.get("rv_kms"),
            "chi2_red": result_payload.get("chi2_red"),
        },
        "fit_result": result_payload,
        "quality_flags": list(result_payload.get("quality_flags") or []),
        "generated_files": result_payload.get("generated_files"),
        "reference_model_diagnostic": reference_model_diagnostic,
    }


def _base_record(row, segment, setup_payload, fit_kwargs, args):
    return {
        "target_id": _target_id(row),
        "hd": row.get("hd"),
        "name": row.get("name"),
        "file": row.get("file"),
        "spectral_type": row.get("spectral_type"),
        "validation_role": row.get("validation_role", "standard"),
        "notes": row.get("notes"),
        "reference": _reference(row),
        "source": {
            "source_url": row.get("source_url"),
            "source_instrument": row.get("source_instrument"),
            "source_quality_flag": row.get("source_quality_flag"),
            "snr": _float_or_none(row.get("snr")),
        },
        "segment": _segment_summary(segment),
        "setup": setup_payload,
        "fit_policy": _fit_policy_record(args, fit_kwargs),
        "warnings": list(setup_payload.get("warnings") or []),
    }


def _is_complete(record, run_fits):
    if not isinstance(record, dict):
        return False
    if run_fits:
        return record.get("status") == "ok"
    return record.get("status") in {"audit_only", "ok"}


def build_payload(manifest_path, manifest_payload, records, args):
    ordered = list(records)
    payload = {
        "schema_version": 1,
        "runner": "scripts/gaia_benchmark_validation.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "source_page": manifest_payload.get("source_page"),
            "source_release": manifest_payload.get("source_release"),
            "citations": manifest_payload.get("citations", []),
        },
        "run_policy": {
            "run_fits": bool(args.run_fits),
            "fit_mode": args.fit_mode,
            "bounds_policy": args.bounds_policy,
            "window_set": args.window_set,
            "error_floor_fraction": float(args.error_floor_fraction),
            "line_plot_requested": bool(args.line_plot_dir),
            "line_plot_reference_model": bool(args.line_plot_reference_model),
            "line_plot_reference_rv_source": "best_fit_rv"
            if args.line_plot_reference_model
            else None,
            "wave_medium": args.wave_medium,
            "reference_parameters_used_as_priors": False,
            "reference_parameters_used_for_postfit_deltas_only": True,
        },
        "results": ordered,
    }
    payload["summary"] = summarize_payload(ordered)
    return payload


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.line_plot_dir or args.line_plot_reference_model) and not args.run_fits:
        parser.error("--line-plot-dir and --line-plot-reference-model require --run-fits.")
    if args.line_plot_reference_model and not args.line_plot_dir:
        parser.error("--line-plot-reference-model requires --line-plot-dir.")
    manifest_path, manifest_payload, rows = load_manifest(args.manifest)
    rows = select_manifest_rows(rows, selected=args.target, max_targets=args.max_targets)
    existing = _load_existing(args.output_json) if args.resume and not args.force else {}
    records_by_id = dict(existing)

    phoenix_lib = None
    if args.run_fits:
        resolved = resolve_phoenix_dir(args.phoenix_dir)
        if resolved is None:
            parser.error("No PHOENIX directory configured; pass --phoenix-dir or configure Spyctres.")
        print("Loading PHOENIX library once for Gaia benchmark validation...", flush=True)
        phoenix_lib = PhoenixLibrary(resolved, verbose=bool(args.verbose))

    completed = []
    for index, row in enumerate(rows, start=1):
        target_id = _target_id(row)
        if args.resume and not args.force and _is_complete(records_by_id.get(target_id), args.run_fits):
            print("Skipping completed {0}/{1}: {2}".format(index, len(rows), target_id), flush=True)
            completed.append(target_id)
            continue

        print("Target {0}/{1}: {2}".format(index, len(rows), target_id), flush=True)
        try:
            path = Path(row["file"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            print("  Reading benchmark spectrum...", flush=True)
            segment = read_spectrum(path, reader="gbs_v3_ascii", warn_unknown=False)
            segment = _maybe_override_wave_medium(segment, args.wave_medium)
            print("  Building reviewed fit setup...", flush=True)
            setup = suggest_fit_setup(
                segment,
                mode=args.fit_mode,
                science_case="benchmark_validation",
                include_readiness=True,
            )
            setup_payload = setup.to_dict()
            fit_kwargs = _benchmark_fit_kwargs(setup_payload, args, row=row)
            record = _base_record(row, segment, setup_payload, fit_kwargs, args)
            if args.run_fits:
                print("  Running PHOENIX benchmark recovery fit...", flush=True)
                record.update(_fit_record(segment, row, setup_payload, fit_kwargs, args, phoenix_lib))
                record["status"] = "ok" if record.get("fit", {}).get("success") else "fit_failed"
            else:
                record["status"] = "audit_only"
            record["recovery"] = {
                "deltas": _fit_deltas(record)[0],
                "assessments": _fit_deltas(record)[1],
                "overall_assessment": _overall_assessment(record),
                "fit_quality_assessment": _fit_quality_assessment(record),
            }
        except Exception as exc:
            record = {
                "target_id": target_id,
                "hd": row.get("hd"),
                "name": row.get("name"),
                "file": row.get("file"),
                "spectral_type": row.get("spectral_type"),
                "validation_role": row.get("validation_role", "standard"),
                "reference": _reference(row),
                "status": "error",
                "error": "{0}: {1}".format(type(exc).__name__, exc),
            }
        records_by_id[target_id] = json_safe(record)
        completed.append(target_id)
        ordered_records = [records_by_id[_target_id(item)] for item in rows if _target_id(item) in records_by_id]
        extras = [
            record
            for key, record in records_by_id.items()
            if key not in {_target_id(item) for item in rows}
        ]
        payload = build_payload(manifest_path, manifest_payload, ordered_records + extras, args)
        atomic_write_json(args.output_json, payload)
        _atomic_write_csv(args.output_csv, payload)
        print("  Wrote checkpoint: {0}".format(args.output_json), flush=True)

    ordered_records = [records_by_id[_target_id(item)] for item in rows if _target_id(item) in records_by_id]
    extras = [
        record
        for key, record in records_by_id.items()
        if key not in {_target_id(item) for item in rows}
    ]
    payload = build_payload(manifest_path, manifest_payload, ordered_records + extras, args)
    atomic_write_json(args.output_json, payload)
    _atomic_write_csv(args.output_csv, payload)
    _write_summary_plot(args.output_summary_plot, payload)
    print(
        "Done. records={0}, statuses={1}, output={2}".format(
            len(payload["results"]),
            payload["summary"]["by_status"],
            args.output_json,
        ),
        flush=True,
    )
    if args.output_csv:
        print("Summary CSV: {0}".format(args.output_csv), flush=True)
    if args.output_summary_plot:
        print("Summary plot: {0}".format(args.output_summary_plot), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
