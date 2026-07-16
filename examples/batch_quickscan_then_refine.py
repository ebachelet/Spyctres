"""Batch quick-scan then focused-refine PHOENIX example.

This is the practical workflow to show someone who needs to fit many spectra:

1. read each reduced spectrum;
2. run a cheap, conservative PHOENIX quicklook fit;
3. use that result to define a narrower local parameter box;
4. rerun a more focused fit inside that box;
5. checkpoint results after every spectrum so interrupted batches can resume.

The default example uses the bundled X-SHOOTER UVB spectrum because it is fast
and line-rich enough to demonstrate the idea. Multi-arm X-SHOOTER fitting is
valuable, but it brings extra arm/telluric/flux-calibration choices; this
example intentionally keeps the first batch pattern simple.

Example
-------
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output /tmp/spyctres_batch_xshooter_uvb.json \
  --resume

For a multi-file demonstration, pass bundled spectra explicitly:

python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_VIS_TELL_CORR.fits \
  --instrument xshooter --output /tmp/spyctres_batch.json --resume

For heterogeneous batches, use --manifest with a CSV containing at least a
path column and optional target_id, instrument, and R/resolution_R columns.
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
from types import SimpleNamespace

import numpy as np

# Let the example run directly from a source checkout, even if Spyctres has not
# been installed into the active Python environment yet.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres import fit_stellar_spectrum, prepare_phoenix_fit_kwargs
from Spyctres.config import resolve_phoenix_dir
from Spyctres.io import read_spectrum
from Spyctres.phoenix import PhoenixLibrary
from Spyctres.preprocessing import audit_spectrum_for_fit


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


def build_parser():
    # Keep the interface operational: files in, reader name, JSON checkpoint
    # out, plus a few knobs that control how much effort each fit stage spends.
    parser = argparse.ArgumentParser(
        description=(
            "Example batch workflow: cheap PHOENIX quick scan, then focused "
            "local refinement for each spectrum. The default demonstration "
            "uses the bundled X-SHOOTER UVB file."
        ),
        epilog=(
            "Example:\n"
            "  python examples/batch_quickscan_then_refine.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter "
            "--output-json /tmp/spyctres_batch_xshooter_uvb.json "
            "--summary-csv /tmp/spyctres_batch_xshooter_uvb.csv --resume\n\n"
            "Manifest example after saving a CSV with examples/data paths:\n"
            "  python examples/batch_quickscan_then_refine.py "
            "--manifest examples/my_batch_manifest.csv "
            "--refine-quality-policy skip-risky "
            "--output-json /tmp/spyctres_batch_manifest.json --resume\n\n"
            "Manifest columns: path or spectrum; optional target_id, "
            "instrument, R/resolution_R."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "spectra",
        nargs="*",
        help=(
            "Input spectrum files. If omitted, the bundled X-SHOOTER UVB "
            "example spectrum is used. Ignored when --manifest is supplied."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional CSV manifest for heterogeneous batches. Required column: "
            "path or spectrum. Optional columns: target_id, instrument, R or "
            "resolution_R. Relative paths are resolved beside the manifest."
        ),
    )
    parser.add_argument(
        "--instrument",
        default="xshooter",
        help="Registered Spyctres reader name. Default: xshooter.",
    )
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--output",
        "--output-json",
        dest="output",
        default="/tmp/spyctres_batch_quickscan_then_refine.json",
        help="Checkpoint/output JSON path.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=(
            "Optional compact CSV summary path for spreadsheet/pandas review. "
            "The full auditable record remains in --output-json."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip spectra already marked complete in the output checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun spectra even when --resume finds completed records.",
    )
    parser.add_argument(
        "--quick-only",
        "--quicklook",
        dest="quick_only",
        action="store_true",
        help=(
            "Run only the cheap quick-scan stage. --quicklook is a friendly "
            "alias for this mode."
        ),
    )
    parser.add_argument(
        "--quick-defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help="Defaults budget for the first pass.",
    )
    parser.add_argument(
        "--refine-defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="standard",
        help="Defaults budget used as the base for the focused second pass.",
    )
    parser.add_argument(
        "--refine-window",
        choices=("quick", "refine-default"),
        default="quick",
        help=(
            "Use the quick-scan wavelength window for refinement, or keep the "
            "window suggested by --refine-defaults-mode."
        ),
    )
    parser.add_argument(
        "--refine-quality-policy",
        choices=("always", "skip-risky"),
        default="always",
        help=(
            "Refinement gate after the quick scan. 'always' preserves the old "
            "throughput example behavior. 'skip-risky' records the quicklook "
            "result but skips the expensive refinement when spectrum readiness "
            "flags indicate missing frame/resolution assumptions, obvious "
            "artifacts, no fitted pixels, or undersampled LSF."
        ),
    )
    parser.add_argument("--quick-max-nfev", type=int, default=45)
    parser.add_argument("--refine-max-nfev", type=int, default=120)
    parser.add_argument("--quick-rv-grid-n", type=int, default=31)
    parser.add_argument("--refine-rv-grid-n", type=int, default=41)
    parser.add_argument("--refine-multistart", type=int, default=2)
    parser.add_argument(
        "--R",
        type=float,
        default=None,
        dest="resolution_R",
        help=(
            "Explicit assumed resolving power used for both quicklook and "
            "refinement. Treat this as a user-supplied approximation unless "
            "the product metadata justify it."
        ),
    )
    parser.add_argument("--teff-margin", type=float, default=1000.0)
    parser.add_argument("--feh-margin", type=float, default=0.5)
    parser.add_argument("--logg-margin", type=float, default=0.8)
    parser.add_argument("--rv-margin", type=float, default=50.0)
    parser.add_argument(
        "--reconstruct-final",
        action="store_true",
        help=(
            "Reconstruct final models. Off by default because this example is "
            "about throughput and tabular batch results rather than plotting."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose PHOENIX backend output and fit progress events.",
    )
    return parser


def _json_native(value):
    # Fit payloads contain NumPy scalars/arrays. Convert them to strict JSON so
    # batch outputs are easy to inspect with json.tool, pandas, or notebooks.
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


def _resolution_override_payload(args):
    if getattr(args, "resolution_R", None) is None:
        return None
    return {
        "resolution_source": "user_override",
        "assumed_resolution_R": float(args.resolution_R),
        "assumption_warning": "approximate quicklook resolution",
    }


def _resolution_fit_kwargs(args):
    payload = _resolution_override_payload(args)
    if payload is None:
        return {}
    return {"R": payload["assumed_resolution_R"]}


def _assumed_resolution_for_audit(args):
    payload = _resolution_override_payload(args)
    if payload is None:
        return None
    return {
        "quantity": "R",
        "value": payload["assumed_resolution_R"],
        "source": payload["resolution_source"],
        "assumption_warning": payload["assumption_warning"],
    }


def _atomic_write_json(path, payload):
    # Checkpoint through a temporary file and atomic replace so interrupted runs
    # do not leave behind half-written JSON.
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(os.path.basename(path)),
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_native(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_write_summary_csv(path, payload):
    """Write a compact per-target CSV beside the detailed JSON checkpoint."""
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(os.path.basename(path)),
        suffix=".tmp",
        dir=directory,
    )
    fieldnames = [
        "target_id",
        "path",
        "status",
        "fit_ready",
        "readiness_flags",
        "readiness_fit_pixels",
        "outside_fit_window_fraction",
        "rejected_inside_fit_window_fraction",
        "resolution_source",
        "assumed_resolution_R",
        "teff",
        "feh",
        "logg",
        "rv_kms",
        "chi2_red",
        "quality_flags",
        "quick_teff",
        "quick_feh",
        "quick_logg",
        "quick_rv_kms",
        "quick_chi2_red",
        "quick_seconds",
        "refine_seconds",
        "total_seconds",
    ]
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in payload.get("results", []):
                refined = item.get("refined_result") or {}
                quick = item.get("quick_result") or {}
                primary = refined if refined else quick
                readiness = item.get("spectrum_readiness") or {}
                resolution_assumption = item.get("resolution_assumption") or {}
                writer.writerow(
                    {
                        "target_id": item.get("target_id"),
                        "path": item.get("path"),
                        "status": item.get("status"),
                        "fit_ready": readiness.get("fit_ready"),
                        "readiness_flags": ",".join(
                            str(flag)
                            for flag in (readiness.get("interpretation_flags") or [])
                        ),
                        "readiness_fit_pixels": readiness.get("n_fit_candidate"),
                        "outside_fit_window_fraction": readiness.get(
                            "outside_fit_window_fraction"
                        ),
                        "rejected_inside_fit_window_fraction": readiness.get(
                            "rejected_inside_fit_window_fraction"
                        ),
                        "resolution_source": resolution_assumption.get(
                            "resolution_source"
                        ),
                        "assumed_resolution_R": resolution_assumption.get(
                            "assumed_resolution_R"
                        ),
                        "teff": primary.get("teff"),
                        "feh": primary.get("feh"),
                        "logg": primary.get("logg"),
                        "rv_kms": primary.get("rv_kms"),
                        "chi2_red": primary.get("chi2_red"),
                        "quality_flags": ",".join(
                            str(flag)
                            for flag in (primary.get("quality_flags") or [])
                        ),
                        "quick_teff": quick.get("teff"),
                        "quick_feh": quick.get("feh"),
                        "quick_logg": quick.get("logg"),
                        "quick_rv_kms": quick.get("rv_kms"),
                        "quick_chi2_red": quick.get("chi2_red"),
                        "quick_seconds": item.get("quick_seconds"),
                        "refine_seconds": item.get("refine_seconds"),
                        "total_seconds": item.get("total_seconds"),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_checkpoint(path):
    # Resume mode is deliberately simple: load the previous JSON checkpoint and
    # skip spectra already marked with a terminal status.
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _input_paths(paths):
    # With no arguments, keep the example immediately runnable from the repo.
    # Real batches pass explicit files or shell-expanded patterns such as *.fits.
    if paths:
        return [os.path.abspath(os.path.expanduser(os.fspath(path))) for path in paths]
    return [str(EXAMPLE_UVB)]


def _manifest_value(row, *names, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _resolve_manifest_path(value, base_dir):
    path = Path(os.path.expanduser(os.fspath(value)))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def _load_manifest_records(path, default_instrument="xshooter"):
    """Load a lightweight heterogeneous-batch CSV manifest."""
    manifest_path = Path(os.path.expanduser(os.fspath(path))).resolve()
    records = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Batch manifest must be a CSV file with a header row.")
        for index, raw_row in enumerate(reader, start=1):
            row = {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in raw_row.items()
                if key is not None
            }
            path_value = _manifest_value(row, "path", "spectrum", "file")
            if path_value is None:
                raise ValueError(
                    "Batch manifest row {0} is missing a path/spectrum/file column.".format(
                        index
                    )
                )
            resolution_value = _manifest_value(row, "resolution_R", "R")
            if resolution_value in (None, ""):
                resolution_R = None
            else:
                resolution_R = float(resolution_value)
            record = {
                "path": _resolve_manifest_path(path_value, manifest_path.parent),
                "target_id": _manifest_value(row, "target_id", "id", default=None),
                "instrument": _manifest_value(
                    row,
                    "instrument",
                    default=default_instrument,
                ),
                "resolution_R": resolution_R,
                "manifest_row": index,
                "manifest_path": str(manifest_path),
            }
            if not record["target_id"]:
                record["target_id"] = record["path"]
            records.append(record)
    if not records:
        raise ValueError("Batch manifest did not contain any spectra.")
    return records


def _input_records(args):
    if args.manifest:
        return _load_manifest_records(args.manifest, default_instrument=args.instrument)
    return [
        {
            "path": path,
            "target_id": _target_id(path),
            "instrument": args.instrument,
            "resolution_R": args.resolution_R,
        }
        for path in _input_paths(args.spectra)
    ]


def _args_for_record(args, record):
    values = vars(args).copy()
    values["instrument"] = record.get("instrument") or args.instrument
    record_resolution = record.get("resolution_R", None)
    values["resolution_R"] = args.resolution_R if record_resolution is None else record_resolution
    return SimpleNamespace(**values)


def _record_path(record):
    return os.path.abspath(os.path.expanduser(os.fspath(record["path"])))


def _record_target_id(record):
    return str(record.get("target_id") or _target_id(_record_path(record)))


def _target_id(path):
    """Stable checkpoint identifier for local batch processing."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _result_payload(result):
    # This example is about throughput and tabular summaries, so it omits model
    # arrays by default. Use --reconstruct-final only when plots/models matter.
    return result.to_dict(include_arrays=False, include_local_paths=False)


def _brief_result(result):
    # Store a small, scan-friendly summary beside the fuller compact result.
    payload = _result_payload(result)
    return {
        "success": payload.get("success"),
        "teff": payload.get("teff"),
        "feh": payload.get("feh"),
        "logg": payload.get("logg"),
        "rv_kms": payload.get("rv_kms"),
        "chi2_red": payload.get("chi2_red"),
        "quality_flags": list(payload.get("quality_flags") or []),
    }


def _required_quick_parameters(quick_result):
    # The focused second pass only makes sense if the quick pass found finite
    # physical parameters. Fail clearly instead of constructing nonsense bounds.
    values = []
    for key in ("teff", "feh", "logg", "rv_kms"):
        value = quick_result[key]
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Quick-scan result has non-finite {0}.".format(key))
        values.append(value)
    return tuple(values)


def focused_bounds_from_quick_result(quick_result, base_bounds, margins):
    """Return p0 and clipped local bounds around a quick-scan fit.

    This helper is intentionally separate and tested because it is the core
    lesson of the example: do not refit the entire broad classification box for
    every spectrum once a cheap pass has identified the local region.
    """
    p0 = _required_quick_parameters(quick_result)
    lower_base = tuple(float(value) for value in base_bounds[0])
    upper_base = tuple(float(value) for value in base_bounds[1])
    margins = tuple(float(value) for value in margins)
    lower = []
    upper = []
    for value, lo_base, hi_base, margin in zip(p0, lower_base, upper_base, margins):
        # Clip each local interval to the broad defaults/model-domain interval.
        # This is the speed trick: refine locally, not across the whole grid.
        if margin <= 0:
            raise ValueError("Focus margins must be positive.")
        lo = max(lo_base, value - margin)
        hi = min(hi_base, value + margin)
        if hi <= lo:
            # Defensive fallback for pathological edge cases.
            lo, hi = lo_base, hi_base
        lower.append(float(lo))
        upper.append(float(hi))
    return p0, (tuple(lower), tuple(upper))


def make_refine_fit_kwargs(spectrum, quick_result, quick_fit_kwargs, args):
    # Start from the same audited defaults helper as the public API, then replace
    # the broad starting point with a local box centred on the quick result.
    fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
        spectrum,
        auto_defaults=True,
        defaults_mode=args.refine_defaults_mode,
        science_case="classification",
    )
    if args.refine_window == "quick" and quick_fit_kwargs.get("regions") is not None:
        # Usually the quick stage chooses where the spectrum is useful; the
        # refinement then spends extra optimizer effort on the same pixels.
        fit_kwargs["regions"] = list(quick_fit_kwargs["regions"])
    fit_kwargs.update(_resolution_fit_kwargs(args))
    p0, bounds = focused_bounds_from_quick_result(
        quick_result,
        fit_kwargs["bounds"],
        (
            args.teff_margin,
            args.feh_margin,
            args.logg_margin,
            args.rv_margin,
        ),
    )
    fit_kwargs.update(
        {
            "p0": p0,
            "bounds": bounds,
            "rv_grid_n": int(args.refine_rv_grid_n),
            "multistart": int(args.refine_multistart),
            "max_nfev": int(args.refine_max_nfev),
        }
    )
    focus = {
        "source": "quick_scan_result",
        "p0": p0,
        "bounds": bounds,
        "margins": {
            "teff": float(args.teff_margin),
            "feh": float(args.feh_margin),
            "logg": float(args.logg_margin),
            "rv_kms": float(args.rv_margin),
        },
        "refine_window_policy": args.refine_window,
    }
    if _resolution_override_payload(args) is not None:
        focus["resolution_assumption"] = _resolution_override_payload(args)
    return fit_kwargs, suggestion, focus


def _run_configuration(args, records):
    # Record the strategy knobs in the JSON checkpoint. This makes resumed or
    # shared batch products auditable without requiring a separate lab notebook.
    return {
        "example": "examples/batch_quickscan_then_refine.py",
        "instrument": args.instrument,
        "manifest": None if args.manifest is None else os.path.abspath(args.manifest),
        "n_input_spectra": len(records),
        "quick_defaults_mode": args.quick_defaults_mode,
        "refine_defaults_mode": args.refine_defaults_mode,
        "refine_window": args.refine_window,
        "refine_quality_policy": args.refine_quality_policy,
        "quick_max_nfev": int(args.quick_max_nfev),
        "refine_max_nfev": int(args.refine_max_nfev),
        "quick_rv_grid_n": int(args.quick_rv_grid_n),
        "refine_rv_grid_n": int(args.refine_rv_grid_n),
        "refine_multistart": int(args.refine_multistart),
        "quick_only": bool(args.quick_only),
        "reconstruct_final": bool(args.reconstruct_final),
        "resolution_assumption": _resolution_override_payload(args),
        "focus_margins": {
            "teff": float(args.teff_margin),
            "feh": float(args.feh_margin),
            "logg": float(args.logg_margin),
            "rv_kms": float(args.rv_margin),
        },
    }


def _progress_callback(enabled, label):
    # Verbose mode forwards the fitter's progress events with a stage label.
    # Normal mode stays quiet enough for hundred-spectrum batches.
    if not enabled:
        return None

    def callback(event):
        print("    [{0}] {1}".format(label, event), flush=True)

    return callback


REFINE_READINESS_BLOCK_FLAGS = {
    "no_fitted_pixels",
    "resolution_assumption_required",
    "wave_medium_unknown",
    "observer_frame_unknown",
    "stellar_rest_status_unknown",
    "artifact_review_required",
    "lsf_undersampled",
}


def should_refine_after_quicklook(readiness, quick_result, policy="always"):
    """Return ``(should_refine, reasons)`` for the optional refinement gate."""
    policy = str(policy).strip().lower()
    if policy == "always":
        return True, []
    if policy != "skip-risky":
        raise ValueError("policy must be 'always' or 'skip-risky'.")
    flags = set(readiness.get("interpretation_flags") or [])
    reasons = sorted(flags & REFINE_READINESS_BLOCK_FLAGS)
    if not readiness.get("fit_ready", False):
        reasons.append("readiness_fit_ready_false")
    if not quick_result.get("success", False):
        reasons.append("quick_fit_not_successful")
    return len(reasons) == 0, sorted(set(reasons))


def _fit_one(record, args, phoenix_lib):
    started = time.perf_counter()
    input_record = dict(record)
    path = _record_path(record)
    local_args = _args_for_record(args, record)

    # Ingestion is intentionally separate from fitting: each reader normalizes
    # an instrument product into Spyctres' common spectrum container.
    print("Reading spectrum: {0}".format(path), flush=True)
    spectrum = read_spectrum(path, instrument=local_args.instrument)
    quick_fit_kwargs, quick_suggestion = prepare_phoenix_fit_kwargs(
        spectrum,
        auto_defaults=True,
        defaults_mode=local_args.quick_defaults_mode,
        science_case="classification",
        extra_kwargs={
            "max_nfev": int(local_args.quick_max_nfev),
            "rv_grid_n": int(local_args.quick_rv_grid_n),
        },
        resolution_R=local_args.resolution_R,
    )
    readiness = audit_spectrum_for_fit(
        spectrum,
        fit_windows=quick_fit_kwargs.get("regions"),
        intended_use="batch_quick_scan_classification",
        assumed_resolution=_assumed_resolution_for_audit(local_args),
    )
    print(
        "  Readiness: fit_ready={0}, fitted_pixels={1}, flags={2}".format(
            readiness["fit_ready"],
            readiness["n_fit_candidate"],
            ", ".join(readiness["interpretation_flags"]) or "none",
        ),
        flush=True,
    )
    for warning in readiness.get("warnings", []):
        print("  Readiness WARNING: {0}".format(warning), flush=True)
    quick_mode_policy = (
        quick_suggestion.provenance.get("mode_policy", {})
        if quick_suggestion is not None
        else {}
    )
    if quick_mode_policy:
        print(
            "  Defaults mode: {0} ({1} budget; {2})".format(
                quick_mode_policy.get("mode", local_args.quick_defaults_mode),
                quick_mode_policy.get("search_budget", "unknown"),
                quick_mode_policy.get("default_status", "first_pass"),
            ),
            flush=True,
        )

    # Stage 1: a cheap classification-style search. Its job is to locate a
    # sensible region of parameter space, not to be the final scientific answer.
    print("  Quick scan...", flush=True)
    quick_start = time.perf_counter()
    quick_result = fit_stellar_spectrum(
        spectrum,
        model="phoenix",
        phoenix_lib=phoenix_lib,
        auto_defaults=False,
        reconstruct=False,
        progress_callback=_progress_callback(local_args.verbose, "quick"),
        **quick_fit_kwargs,
    )
    if quick_suggestion is not None:
        quick_result.summary["fit_default_suggestion"] = quick_suggestion.to_dict()
    quick_result.summary["spectrum_readiness"] = readiness
    quick_result.provenance["spectrum_readiness"] = readiness
    quick_seconds = time.perf_counter() - quick_start
    record = {
        "target_id": _record_target_id(input_record),
        "path": path,
        "instrument": local_args.instrument,
        "status": "quick_ok" if args.quick_only else "quick_complete",
        "spectrum_readiness": readiness,
        "quick_seconds": float(quick_seconds),
        "quick_result": _brief_result(quick_result),
        "quick_result_full": _result_payload(quick_result),
    }
    if record["target_id"] != path:
        record["target_label"] = record["target_id"]
    if _resolution_override_payload(local_args) is not None:
        record["resolution_assumption"] = _resolution_override_payload(local_args)
    if args.manifest:
        record["manifest_row"] = int(input_record.get("manifest_row", 0) or 0)
        record["manifest_path"] = input_record.get("manifest_path")
    print(
        "  Quick result: Teff={teff} logg={logg} [Fe/H]={feh} RV={rv_kms} chi2={chi2_red}".format(
            **record["quick_result"]
        ),
        flush=True,
    )

    if args.quick_only:
        # Useful when surveying a directory before deciding which spectra merit
        # a more expensive second pass.
        record["total_seconds"] = float(time.perf_counter() - started)
        return record

    should_refine, skip_reasons = should_refine_after_quicklook(
        readiness,
        record["quick_result"],
        policy=args.refine_quality_policy,
    )
    if not should_refine:
        record.update(
            {
                "status": "refine_skipped",
                "refinement_skipped_reason": skip_reasons,
                "total_seconds": float(time.perf_counter() - started),
            }
        )
        print(
            "  Skipping focused refinement because: {0}".format(
                ", ".join(skip_reasons)
            ),
            flush=True,
        )
        return record

    # Stage 2: reuse the quick result to build a focused local fit.
    print("  Focused refinement...", flush=True)
    refine_fit_kwargs, refine_suggestion, focus = make_refine_fit_kwargs(
        spectrum,
        quick_result,
        quick_fit_kwargs,
        local_args,
    )
    refine_start = time.perf_counter()
    refined_result = fit_stellar_spectrum(
        spectrum,
        model="phoenix",
        phoenix_lib=phoenix_lib,
        auto_defaults=False,
        reconstruct=bool(local_args.reconstruct_final),
        progress_callback=_progress_callback(local_args.verbose, "refine"),
        **refine_fit_kwargs,
    )
    refine_seconds = time.perf_counter() - refine_start
    record.update(
        {
            "status": "ok",
            "refine_seconds": float(refine_seconds),
            "refinement_focus": focus,
            "refine_default_suggestion": (
                None if refine_suggestion is None else refine_suggestion.to_dict()
            ),
            "refined_result": _brief_result(refined_result),
            "refined_result_full": _result_payload(refined_result),
            "total_seconds": float(time.perf_counter() - started),
        }
    )
    print(
        "  Refined result: Teff={teff} logg={logg} [Fe/H]={feh} RV={rv_kms} chi2={chi2_red}".format(
            **record["refined_result"]
        ),
        flush=True,
    )
    return record


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = _input_records(args)
    run_configuration = _run_configuration(args, records)
    if _resolution_override_payload(args) is not None:
        print(
            "Using user-supplied assumed resolution R={0:g} for this batch.".format(
                float(args.resolution_R)
            ),
            flush=True,
        )

    # Load any existing checkpoint before touching PHOENIX. On a resumed batch,
    # this lets completed targets be skipped quickly.
    previous = _load_checkpoint(args.output) if args.resume else None
    records_by_id = {}
    if previous is not None:
        for item in previous.get("results", []):
            target_id = item.get("target_id")
            if target_id:
                records_by_id[target_id] = item

    def checkpoint():
        # Preserve input order in the output JSON so batch tables are stable
        # across resumed runs.
        ordered = [
            records_by_id[_record_target_id(record)]
            for record in records
            if _record_target_id(record) in records_by_id
        ]
        payload = {
            "schema_version": 1,
            "purpose": "batch quick-scan then focused-refine PHOENIX example",
            "run_configuration": run_configuration,
            "results": ordered,
        }
        _atomic_write_json(args.output, payload)
        if args.summary_csv:
            _atomic_write_summary_csv(args.summary_csv, payload)
        return payload

    resolved_phoenix_dir = resolve_phoenix_dir(args.phoenix_dir)
    if resolved_phoenix_dir is None:
        raise ValueError(
            "No PHOENIX directory configured. Pass --phoenix-dir or set "
            "SPYCTRES_PHOENIX_DIR / the Spyctres config path."
        )
    print("Loading PHOENIX library once for the whole batch...", flush=True)
    # This is the main speed lesson: initialize the PHOENIX backend once and
    # pass the same object into every fit instead of reloading it per spectrum.
    phoenix_lib = PhoenixLibrary(resolved_phoenix_dir, verbose=bool(args.verbose))

    completed_statuses = {"ok", "quick_ok", "refine_skipped", "fit_failed", "error"}
    for index, record in enumerate(records, start=1):
        target_id = _record_target_id(record)
        path = _record_path(record)
        # Resume skips terminal records unless --force is requested.
        if (
            args.resume
            and not args.force
            and target_id in records_by_id
            and records_by_id[target_id].get("status") in completed_statuses
        ):
            print("Skipping completed {0}/{1}: {2}".format(index, len(records), path), flush=True)
            continue
        print("\nTarget {0}/{1}".format(index, len(records)), flush=True)
        try:
            records_by_id[target_id] = _fit_one(record, args, phoenix_lib)
        except Exception as exc:
            # Do not let one bad file kill an entire batch. Record the failure,
            # checkpoint it, and continue with the next spectrum.
            records_by_id[target_id] = {
                "target_id": target_id,
                "path": path,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print("  ERROR: {0}: {1}".format(type(exc).__name__, exc), flush=True)
        checkpoint()
        print("  Wrote checkpoint: {0}".format(args.output), flush=True)
        if args.summary_csv:
            print("  Wrote summary CSV: {0}".format(args.summary_csv), flush=True)

    payload = checkpoint()
    # Detailed quality flags live in JSON; the terminal gets a compact summary.
    n_ok = sum(1 for item in payload["results"] if item.get("status") == "ok")
    n_quick = sum(1 for item in payload["results"] if item.get("status") == "quick_ok")
    n_skipped = sum(
        1 for item in payload["results"] if item.get("status") == "refine_skipped"
    )
    n_error = sum(1 for item in payload["results"] if item.get("status") == "error")
    print(
        "\nDone. ok={0}, quick_only={1}, refine_skipped={2}, errors={3}, output={4}".format(
            n_ok,
            n_quick,
            n_skipped,
            n_error,
            args.output,
        ),
        flush=True,
    )
    if args.summary_csv:
        print("Summary CSV: {0}".format(args.summary_csv), flush=True)


if __name__ == "__main__":
    main()
