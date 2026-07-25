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
import csv
import json
import math
import os
import re
import sys
import tempfile
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
    read_spectrum,
    suggest_fit_setup,
)
from Spyctres.config import resolve_phoenix_dir
from Spyctres.phoenix import PhoenixLibrary


DEFAULT_MANIFEST = REPO_ROOT / "examples" / "data" / "gaia_benchmark" / "manifest.json"
ORDINARY_ROLES = {"standard"}

RECOVERY_THRESHOLDS = {
    "teff": {
        "label": "Teff",
        "unit": "K",
        "acceptable_abs_delta": 250.0,
        "review_abs_delta": 500.0,
    },
    "logg": {
        "label": "logg",
        "unit": "dex",
        "acceptable_abs_delta": 0.5,
        "review_abs_delta": 1.0,
    },
    "feh": {
        "label": "[Fe/H]",
        "unit": "dex",
        "acceptable_abs_delta": 0.3,
        "review_abs_delta": 0.6,
    },
}


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
        "--wave-medium",
        choices=("unknown", "air", "vacuum"),
        default="unknown",
        help=(
            "Optional user override for the benchmark wavelength medium. The "
            "bundled manifest leaves this unknown because the source page does "
            "not state air/vacuum explicitly."
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


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(path, payload):
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "overall_assessment",
        "chi2_red",
        "quality_flags",
        "fit_window_A",
        "n_pixels",
        "n_used",
        "warnings",
    ]
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in payload.get("results", []):
                row = _csv_row(record)
                writer.writerow({key: row.get(key) for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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
    if wave_medium == "unknown":
        return segment
    meta = dict(segment.meta)
    meta["wave_medium"] = wave_medium
    meta["wave_medium_source"] = "user_override"
    meta["wave_medium_warning"] = (
        "Gaia benchmark source page did not state air/vacuum explicitly; "
        "this value was supplied by the validation runner user."
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


def _benchmark_fit_kwargs(setup_payload, args):
    fit_kwargs = dict(setup_payload.get("fit_kwargs") or {})
    if args.bounds_policy == "benchmark_fgk":
        fit_kwargs.update(
            {
                "p0": (5500.0, -0.5, 4.0, 0.0),
                "bounds": (
                    (3000.0, -2.5, 0.5, -150.0),
                    (8000.0, 0.5, 5.5, 150.0),
                ),
                "coarse_teff_grid": [3500.0, 4500.0, 5500.0, 6500.0, 7500.0],
                "coarse_feh_grid": [-2.5, -1.5, -0.5, 0.0, 0.5],
                "coarse_logg_grid": [1.0, 3.0, 4.5, 5.0],
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
        }
    )
    return fit_kwargs


def _fit_policy_record(args, fit_kwargs):
    return {
        "run_fits": bool(args.run_fits),
        "fit_mode": args.fit_mode,
        "bounds_policy": args.bounds_policy,
        "reference_parameters_used_as_priors": False,
        "reference_parameters_used_for_postfit_deltas_only": True,
        "wave_medium_override": args.wave_medium,
        "fit_kwargs": _json_safe(fit_kwargs),
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
    for param in ("teff", "logg", "feh"):
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
    values = list(assessments.values())
    if any(value == "outside_review_tolerance" for value in values):
        return "outside_review_tolerance"
    if any(value in {"review", "missing"} for value in values):
        return "review"
    return "within_first_pass_tolerance"


def _csv_row(record):
    deltas, assessments = _fit_deltas(record)
    reference = record.get("reference") or {}
    fit = record.get("fit") or {}
    segment = record.get("segment") or {}
    setup = record.get("setup") or {}
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
        "overall_assessment": _overall_assessment(record),
        "chi2_red": fit.get("chi2_red"),
        "quality_flags": ";".join(str(item) for item in record.get("quality_flags", [])),
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
    return {
        "n_records": int(len(records)),
        "by_status": dict(sorted(statuses.items())),
        "by_validation_role": dict(sorted(role_counts.items())),
        "ordinary_roles": sorted(ORDINARY_ROLES),
        "ordinary_recovery_n": int(len(ordinary)),
        "ordinary_recovery_assessments": dict(sorted(ordinary_assessments.items())),
        "thresholds": RECOVERY_THRESHOLDS,
        "notes": [
            "Reference parameters are used only for post-fit deltas.",
            "Stress/diagnostic targets are reported separately from ordinary recovery statistics.",
            "Local covariance errors, when present in fit outputs, do not include external systematic uncertainty.",
        ],
    }


def _safe_name(value, fallback="target"):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = text.strip("._")
    return text or fallback


def _write_summary_plot(path, payload):
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
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
            for record in fitted:
                deltas, record_assessments = _fit_deltas(record)
                if deltas[param] is None:
                    continue
                labels.append(record.get("target_id"))
                values.append(float(deltas[param]))
                assessments.append(record_assessments[param])
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
            ax.scatter(
                x,
                values,
                c=[colors.get(item, "tab:gray") for item in assessments],
                s=55,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title("{0} fit - reference".format(limits["label"]))
            ax.set_ylabel("{0} ({1})".format(limits["label"], limits["unit"]))
            ax.grid(alpha=0.25)
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
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _progress_printer(target_id, verbose):
    def callback(event):
        phase = getattr(event, "phase", "")
        if verbose or phase in {"cache", "rv_grid", "local_optimize", "finish"}:
            print("  [{0}] {1}".format(target_id, event), flush=True)

    return callback


def _fit_record(segment, row, setup_payload, fit_kwargs, args, phoenix_lib):
    target_id = _target_id(row)
    result = fit_stellar_spectrum(
        segment,
        model="phoenix",
        phoenix_lib=phoenix_lib,
        auto_defaults=False,
        reconstruct=bool(args.fit_plot_dir),
        warn_unknown=False,
        progress_callback=_progress_printer(target_id, args.verbose),
        **fit_kwargs,
    )
    plot_paths = {}
    if args.fit_plot_dir:
        plot_dir = Path(args.fit_plot_dir).expanduser().resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "{0}_fit.png".format(_safe_name(target_id))
        fig, _axes = plot_fit_referee(result, savepath=str(plot_path))
        plt.close(fig)
        plot_paths["referee_fit"] = str(plot_path)
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
            segment = read_spectrum(path, instrument="gaia_benchmark", warn_unknown=False)
            segment = _maybe_override_wave_medium(segment, args.wave_medium)
            print("  Building reviewed fit setup...", flush=True)
            setup = suggest_fit_setup(
                segment,
                mode=args.fit_mode,
                science_case="benchmark_validation",
                include_readiness=True,
            )
            setup_payload = setup.to_dict()
            fit_kwargs = _benchmark_fit_kwargs(setup_payload, args)
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
        records_by_id[target_id] = _json_safe(record)
        completed.append(target_id)
        ordered_records = [records_by_id[_target_id(item)] for item in rows if _target_id(item) in records_by_id]
        extras = [
            record
            for key, record in records_by_id.items()
            if key not in {_target_id(item) for item in rows}
        ]
        payload = build_payload(manifest_path, manifest_payload, ordered_records + extras, args)
        _atomic_write_json(args.output_json, payload)
        _atomic_write_csv(args.output_csv, payload)
        print("  Wrote checkpoint: {0}".format(args.output_json), flush=True)

    ordered_records = [records_by_id[_target_id(item)] for item in rows if _target_id(item) in records_by_id]
    extras = [
        record
        for key, record in records_by_id.items()
        if key not in {_target_id(item) for item in rows}
    ]
    payload = build_payload(manifest_path, manifest_payload, ordered_records + extras, args)
    _atomic_write_json(args.output_json, payload)
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
