"""Validate user-supplied external spectra through Spyctres ingestion/readiness.

This is a developer/validation runner, not a PHOENIX fitting preset.  It reads
SDSS/SEGUE and UVES-POP spectra through the public Spyctres I/O layer, runs the
lightweight fit-readiness audit, and writes resumable JSON/CSV summaries.  Use
it before adding external spectra to more expensive classification workflows.

Example
-------
python scripts/external_spectra_validation.py \
  --scan-root /path/to/spectra_test_set \
  --output-json /tmp/spyctres_external_validation.json \
  --output-csv /tmp/spyctres_external_validation.csv \
  --resume
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from Spyctres import ensure_matplotlib_config_dir

ensure_matplotlib_config_dir()
import matplotlib.pyplot as plt

from Spyctres import audit_spectrum_for_fit, select_diagnostic_windows, suggest_fit_setup
from Spyctres.io import get_instrument_info, list_instruments, read_spectrum
from Spyctres.plotting import plot_spectrum_audit, plot_spectrum_quicklook


SUPPORTED_EXTERNAL_INSTRUMENTS = ("sdss", "uves_pop")
COMPLETED_STATUSES = frozenset({"ok", "error"})


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Developer/validation runner for user-supplied external SDSS and "
            "UVES-POP spectra. It checks ingestion, metadata, masks, and "
            "fit-readiness; it does not run PHOENIX fits or assign stellar "
            "classes."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/external_spectra_validation.py "
            "--scan-root /path/to/spectra_test_set "
            "--output-json /tmp/spyctres_external_validation.json "
            "--output-csv /tmp/spyctres_external_validation.csv --resume\n\n"
            "  python scripts/external_spectra_validation.py "
            "--manifest external_manifest.csv "
            "--output-json /tmp/spyctres_external_validation.json "
            "--output-csv /tmp/spyctres_external_validation.csv\n\n"
            "Manifest columns: path, instrument, optional target_id, role "
            "(clean/dirty/unknown), label, notes, wave_unit, err_column, "
            "sdss_mask_policy, use_and_mask, attach_wdisp_resolution, "
            "assumed_resolution_R, fit_wmin, fit_wmax."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Optional explicit files. Use with --instrument. For multi-instrument "
            "sets, prefer --manifest or --scan-root."
        ),
    )
    parser.add_argument(
        "--instrument",
        default=None,
        choices=SUPPORTED_EXTERNAL_INSTRUMENTS,
        help="Reader for explicit positional files.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest describing external spectra and optional roles.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Base directory for relative paths in --manifest.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=None,
        help=(
            "Directory to scan for SDSS and UVES-POP spectra. May be repeated. "
            "Recognized layouts include SDSS/spec-*.fits and UVES-POP/*.dat."
        ),
    )
    parser.add_argument(
        "--role",
        default="unknown",
        help="Role label for positional files or scan-root discoveries.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Resumable JSON checkpoint/output path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional compact CSV summary path.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory for diagnostic plots.",
    )
    parser.add_argument(
        "--plot-style",
        default="audit",
        choices=("audit", "quicklook", "both"),
        help=(
            "Plot style written to --plot-dir. 'audit' writes a three-panel "
            "raw/normalized/mask plot with generic metadata-warning and "
            "diagnostic-window overlays; 'quicklook' preserves the old "
            "single-panel view."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write diagnostic plots even if --plot-dir is supplied.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed targets in an existing JSON checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun targets even when --resume finds completed records.",
    )
    parser.add_argument(
        "--fit-window",
        nargs=2,
        action="append",
        type=float,
        metavar=("WMIN_A", "WMAX_A"),
        help=(
            "Readiness fit window in Angstrom. Repeat for multiple windows. "
            "Manifest fit_wmin/fit_wmax override this per target."
        ),
    )
    parser.add_argument(
        "--assumed-resolution",
        type=float,
        default=None,
        help=(
            "Optional global assumed constant resolving power for readiness "
            "audits only. For SDSS quicklook checks, R~2000 is a common rough "
            "choice but is not precision LSF modeling."
        ),
    )
    parser.add_argument(
        "--sdss-mask-policy",
        default="and_mask_conservative",
        choices=("ivar_only", "and_mask_conservative", "stellar_strict", "sky_strict"),
        help="Default SDSS bitmask policy when not specified by manifest.",
    )
    parser.add_argument(
        "--sdss-attach-wdisp-resolution",
        action="store_true",
        help=(
            "Attach SDSS wdisp as tabulated LSF provenance. The current fitter "
            "does not use it for active wavelength-dependent convolution."
        ),
    )
    parser.add_argument(
        "--uves-wave-unit",
        default="auto",
        choices=("auto", "angstrom", "nm"),
        help="Default UVES-POP wavelength unit interpretation.",
    )
    parser.add_argument(
        "--uves-err-column",
        type=int,
        default=None,
        help="Optional zero-based UVES-POP error column. Omit to keep err=None.",
    )
    parser.add_argument(
        "--diagnostic-window-count",
        type=int,
        default=8,
        help="Maximum number of suggested diagnostic windows overlaid on audit plots.",
    )
    parser.add_argument(
        "--setup-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="quicklook",
        help=(
            "PHOENIX-free defaults/setup recommendation mode recorded for "
            "each target. This does not run a fit."
        ),
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=8000,
        help="Maximum plotted samples per segment in each audit-plot trace.",
    )
    return parser


def _json_safe_scalar(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe_scalar(value.item())
    return str(value)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_safe_scalar(value)


def _atomic_write_json(path, payload):
    path = os.path.abspath(os.path.expanduser(str(path)))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(os.path.basename(path)),
        suffix=".tmp",
        dir=directory,
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
    path = os.path.abspath(os.path.expanduser(str(path)))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    columns = [
        "target_id",
        "label",
        "role",
        "instrument",
        "status",
        "role_expectation_assessment",
        "fit_ready",
        "quicklook_only",
        "n_segments",
        "n_total",
        "n_fit_candidate",
        "outside_fit_window_fraction",
        "rejected_inside_fit_window_fraction",
        "interpretation_flags",
        "warnings",
        "wave_min_A",
        "wave_max_A",
        "err_present",
        "resolution_present",
        "setup_mode",
        "recommended_branch_id",
        "recommended_window_label",
        "path",
    ]
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(os.path.basename(path)),
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in payload.get("results", []):
                readiness = row.get("readiness") or {}
                coverage = row.get("coverage") or {}
                setup = row.get("fit_setup_recommendation") or {}
                window = setup.get("recommended_window") or {}
                writer.writerow(
                    {
                        "target_id": row.get("target_id"),
                        "label": row.get("label"),
                        "role": row.get("role"),
                        "instrument": row.get("instrument"),
                        "status": row.get("status"),
                        "role_expectation_assessment": row.get(
                            "role_expectation_assessment"
                        ),
                        "fit_ready": readiness.get("fit_ready"),
                        "quicklook_only": readiness.get("quicklook_only"),
                        "n_segments": readiness.get("n_segments"),
                        "n_total": readiness.get("n_total"),
                        "n_fit_candidate": readiness.get("n_fit_candidate"),
                        "outside_fit_window_fraction": readiness.get(
                            "outside_fit_window_fraction"
                        ),
                        "rejected_inside_fit_window_fraction": readiness.get(
                            "rejected_inside_fit_window_fraction"
                        ),
                        "interpretation_flags": ";".join(
                            readiness.get("interpretation_flags") or ()
                        ),
                        "warnings": ";".join(readiness.get("warnings") or ()),
                        "wave_min_A": coverage.get("wave_min_A"),
                        "wave_max_A": coverage.get("wave_max_A"),
                        "err_present": row.get("err_present"),
                        "resolution_present": row.get("resolution_present"),
                        "setup_mode": setup.get("mode"),
                        "recommended_branch_id": setup.get("recommended_branch_id"),
                        "recommended_window_label": window.get("label"),
                        "path": row.get("path"),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _as_bool(value, default=False):
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Cannot parse boolean value {0!r}.".format(value))


def _optional_float(value, default=None):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _optional_int(value, default=None):
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _safe_target_id(instrument, path, label=None):
    seed = label or os.path.splitext(os.path.basename(str(path)))[0]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in seed)
    return "{0}:{1}".format(instrument, safe or "target")


def _safe_scan_target_id(instrument, root, path):
    relative = Path(path).resolve().relative_to(Path(root).resolve()).with_suffix("")
    seed = "_".join(relative.parts)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in seed)
    return "{0}:{1}".format(instrument, safe or "target")


def _resolve_path(path, base=None):
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    if base:
        return os.path.abspath(os.path.join(os.path.expanduser(str(base)), path))
    return os.path.abspath(path)


def _normalize_instrument(value):
    key = str(value or "").strip().lower()
    if key in {"uves-pop", "uvespop"}:
        key = "uves_pop"
    if key in {"sdss_spec", "segue"}:
        key = "sdss"
    if key not in SUPPORTED_EXTERNAL_INSTRUMENTS:
        raise ValueError(
            "External validation currently supports {0}; got {1!r}. "
            "Registered Spyctres instruments are: {2}.".format(
                ", ".join(SUPPORTED_EXTERNAL_INSTRUMENTS),
                value,
                ", ".join(list_instruments()),
            )
        )
    return key


def _infer_role_from_path(path, default="unknown"):
    parts = [part.lower() for part in Path(path).parts]
    if any(part in {"clean", "good"} for part in parts):
        return "clean"
    if any(part in {"dirty", "bad", "problematic"} for part in parts):
        return "dirty"
    return str(default or "unknown")


def _infer_instrument_from_path(path):
    path_obj = Path(path)
    lower_parts = [part.lower() for part in path_obj.parts]
    basename = path_obj.name.lower()
    if any(part in {"sdss", "segue"} for part in lower_parts) or basename.startswith(
        "spec-"
    ):
        return "sdss"
    if any(part in {"uves-pop", "uves_pop", "uvespop"} for part in lower_parts):
        return "uves_pop"
    if basename.endswith(".dat") and "hd" in basename:
        return "uves_pop"
    return None


def _scan_root(root, role="unknown"):
    root = Path(os.path.expanduser(str(root))).resolve()
    if not root.exists():
        raise ValueError("Scan root does not exist: {0}".format(root))
    candidates = []
    patterns = (
        ("sdss", ("SDSS/spec-*.fits", "SDSS/*.fits", "**/spec-*.fits")),
        ("uves_pop", ("UVES-POP/*.dat", "UVES_POP/*.dat", "**/*.dat")),
    )
    seen = set()
    for instrument, glob_patterns in patterns:
        for pattern in glob_patterns:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                inferred = _infer_instrument_from_path(resolved)
                if inferred is not None and inferred != instrument:
                    continue
                seen.add(resolved)
                candidates.append(
                    {
                        "path": resolved,
                        "instrument": instrument,
                        "role": _infer_role_from_path(resolved, default=role),
                        "label": path.stem,
                        "target_id": _safe_scan_target_id(instrument, root, path),
                        "source": "scan_root",
                    }
                )
    return candidates


def _read_manifest(path, root=None):
    base = root or os.path.dirname(os.path.abspath(os.path.expanduser(str(path))))
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("External spectra manifest contains no rows.")
    required = {"path", "instrument"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            "External spectra manifest requires columns: {0}.".format(
                ", ".join(sorted(required))
            )
        )
    targets = []
    for index, row in enumerate(rows, start=1):
        instrument = _normalize_instrument(row.get("instrument"))
        label = row.get("label") or row.get("target_id") or Path(row["path"]).stem
        target = {
            "path": _resolve_path(row["path"], base=base),
            "path_input": row["path"],
            "instrument": instrument,
            "role": row.get("role") or "unknown",
            "label": label,
            "target_id": row.get("target_id") or _safe_target_id(instrument, row["path"], label),
            "notes": row.get("notes") or "",
            "source": "manifest",
            "row_number": index,
            "reader_options": {
                "wave_unit": row.get("wave_unit"),
                "err_column": row.get("err_column"),
                "sdss_mask_policy": row.get("sdss_mask_policy"),
                "use_and_mask": row.get("use_and_mask"),
                "attach_wdisp_resolution": row.get("attach_wdisp_resolution"),
            },
            "audit_options": {
                "assumed_resolution_R": row.get("assumed_resolution_R"),
                "fit_wmin": row.get("fit_wmin"),
                "fit_wmax": row.get("fit_wmax"),
            },
        }
        targets.append(target)
    return targets


def discover_targets(args):
    targets = []
    if args.manifest:
        targets.extend(_read_manifest(args.manifest, root=args.root))
    if args.scan_root:
        for root in args.scan_root:
            targets.extend(_scan_root(root, role=args.role))
    if args.files:
        if args.instrument is None:
            raise ValueError("Positional files require --instrument.")
        instrument = _normalize_instrument(args.instrument)
        for path in args.files:
            resolved = _resolve_path(path)
            targets.append(
                {
                    "path": resolved,
                    "path_input": path,
                    "instrument": instrument,
                    "role": args.role,
                    "label": Path(path).stem,
                    "target_id": _safe_target_id(instrument, path),
                    "notes": "",
                    "source": "files",
                }
            )
    if not targets:
        raise ValueError("Provide --manifest, --scan-root, or files with --instrument.")

    seen = {}
    duplicates = []
    for target in targets:
        target.setdefault(
            "target_id",
            _safe_target_id(target["instrument"], target["path"], target.get("label")),
        )
        key = str(target["target_id"])
        if key in seen:
            duplicates.append(key)
        seen[key] = target
    if duplicates:
        raise ValueError(
            "External validation target_id values must be unique; duplicates: {0}".format(
                ", ".join(sorted(set(duplicates)))
            )
        )
    return targets


def _reader_kwargs(target, args):
    instrument = _normalize_instrument(target["instrument"])
    options = target.get("reader_options") or {}
    kwargs = {}
    if instrument == "sdss":
        if options.get("sdss_mask_policy") not in (None, ""):
            kwargs["sdss_mask_policy"] = str(options["sdss_mask_policy"]).strip().lower()
        elif options.get("use_and_mask") not in (None, ""):
            kwargs["use_and_mask"] = _as_bool(options.get("use_and_mask"))
        else:
            kwargs["sdss_mask_policy"] = str(args.sdss_mask_policy).strip().lower()
        attach = (
            _as_bool(options.get("attach_wdisp_resolution"))
            if options.get("attach_wdisp_resolution") not in (None, "")
            else bool(args.sdss_attach_wdisp_resolution)
        )
        kwargs["attach_wdisp_resolution"] = attach
    elif instrument == "uves_pop":
        wave_unit = options.get("wave_unit") or args.uves_wave_unit
        kwargs["wave_unit"] = str(wave_unit).strip().lower()
        err_column = _optional_int(options.get("err_column"), args.uves_err_column)
        if err_column is not None:
            kwargs["err_column"] = err_column
    return kwargs


def _fit_windows(target, args):
    options = target.get("audit_options") or {}
    wmin = _optional_float(options.get("fit_wmin"))
    wmax = _optional_float(options.get("fit_wmax"))
    if wmin is not None or wmax is not None:
        if wmin is None or wmax is None:
            raise ValueError(
                "Manifest target {0!r} must provide both fit_wmin and fit_wmax.".format(
                    target.get("target_id")
                )
            )
        return [(float(wmin), float(wmax))]
    if args.fit_window:
        return [(float(pair[0]), float(pair[1])) for pair in args.fit_window]
    return None


def _assumed_resolution(target, args):
    options = target.get("audit_options") or {}
    return _optional_float(options.get("assumed_resolution_R"), args.assumed_resolution)


def _iter_segments(spectrum):
    if hasattr(spectrum, "segments"):
        return list(spectrum.segments)
    return [spectrum]


def _coverage_summary(spectrum):
    waves = []
    for segment in _iter_segments(spectrum):
        wave = np.asarray(segment.wave, dtype=float)
        mask = np.asarray(segment.mask, dtype=bool)
        good = mask & np.isfinite(wave)
        if np.any(good):
            waves.append(wave[good])
    if not waves:
        return {"wave_min_A": None, "wave_max_A": None}
    joined = np.concatenate(waves)
    return {
        "wave_min_A": float(np.nanmin(joined)),
        "wave_max_A": float(np.nanmax(joined)),
    }


def _segment_summary(segment):
    wave = np.asarray(segment.wave, dtype=float)
    flux = np.asarray(segment.flux, dtype=float)
    mask = np.asarray(segment.mask, dtype=bool)
    good = mask & np.isfinite(wave)
    resolution = segment.resolution
    return {
        "name": getattr(segment, "name", None),
        "n_pixels": int(wave.size),
        "n_valid": int(np.count_nonzero(mask)),
        "valid_fraction": (
            None if wave.size == 0 else float(np.count_nonzero(mask)) / float(wave.size)
        ),
        "finite_flux_fraction": (
            None
            if flux.size == 0
            else float(np.count_nonzero(np.isfinite(flux))) / float(flux.size)
        ),
        "wave_min_A": None if not np.any(good) else float(np.nanmin(wave[good])),
        "wave_max_A": None if not np.any(good) else float(np.nanmax(wave[good])),
        "err_present": bool(getattr(segment, "err", None) is not None),
        "wave_medium": getattr(segment, "wave_medium", None),
        "wave_frame": getattr(segment, "wave_frame", None),
        "observer_frame": getattr(segment, "observer_frame", None),
        "stellar_rest_status": getattr(segment, "stellar_rest_status", None),
        "resolution": None if resolution is None else resolution.to_metadata(),
        "archive_mask_summary": (segment.meta or {}).get("archive_mask_summary"),
        "fit_readiness_role": (segment.meta or {}).get("fit_readiness_role"),
        "sdss_lsf": (segment.meta or {}).get("sdss_lsf"),
    }


def _diagnostic_window_selection(spectrum, max_windows):
    max_windows = int(max_windows)
    if max_windows < 1:
        return {
            "schema_version": 1,
            "operation": "select_diagnostic_windows",
            "selected": [],
            "rejected": [],
            "selection_policy": {"max_windows": 0},
        }
    try:
        return select_diagnostic_windows(
            spectrum,
            max_windows=max_windows,
            min_overlap_A=8.0,
            min_pixels=8,
        )
    except Exception as exc:
        return {
            "schema_version": 1,
            "operation": "select_diagnostic_windows",
            "status": "selection_failed",
            "error": exc.__class__.__name__,
            "message": str(exc),
            "selected": [],
        }


def _fit_setup_recommendation(spectrum, args, assumed_resolution):
    try:
        setup = suggest_fit_setup(
            spectrum,
            mode=args.setup_mode,
            science_case="classification",
            include_readiness=False,
            assumed_resolution=assumed_resolution,
        )
        setup["status"] = "ok"
        setup["readiness_source"] = "external_validation.readiness"
        return setup
    except Exception as exc:
        return {
            "schema_version": 1,
            "operation": "suggest_fit_setup",
            "status": "setup_recommendation_failed",
            "error": exc.__class__.__name__,
            "message": str(exc),
        }


def _role_expectation_assessment(role, readiness):
    role = str(role or "unknown").strip().lower()
    fit_ready = bool((readiness or {}).get("fit_ready"))
    n_fit = int((readiness or {}).get("n_fit_candidate") or 0)
    if role == "clean":
        return "clean_passed_readiness" if fit_ready else "clean_needs_review"
    if role == "dirty":
        if fit_ready:
            return "dirty_not_caught_by_readiness"
        if n_fit > 0:
            return "dirty_flagged_quicklook_only"
        return "dirty_not_fit_usable"
    if fit_ready:
        return "fit_ready_unlabelled"
    if n_fit > 0:
        return "quicklook_only_unlabelled"
    return "not_fit_usable_unlabelled"


def _safe_plot_name(target):
    seed = str(target.get("target_id") or target.get("label") or "target")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in seed)
    return "{0}.png".format(safe or "target")


def validate_target(target, args):
    instrument = _normalize_instrument(target["instrument"])
    path = target["path"]
    reader_kwargs = _reader_kwargs(target, args)
    record = {
        "target_id": target.get("target_id"),
        "label": target.get("label"),
        "role": target.get("role", "unknown"),
        "instrument": instrument,
        "path": path,
        "path_input": target.get("path_input", path),
        "notes": target.get("notes", ""),
        "source": target.get("source", "unknown"),
        "reader_kwargs": dict(reader_kwargs),
    }
    if not os.path.exists(path):
        record.update(
            {
                "status": "error",
                "error": "file_not_found",
                "message": "Input spectrum does not exist: {0}".format(path),
            }
        )
        return record

    print(
        "Reading {0} target {1}: {2}".format(
            instrument, target.get("target_id"), path
        ),
        flush=True,
    )
    try:
        spectrum = read_spectrum(path, instrument=instrument, **reader_kwargs)
        fit_windows = _fit_windows(target, args)
        assumed_resolution = _assumed_resolution(target, args)
        readiness = audit_spectrum_for_fit(
            spectrum,
            fit_windows=fit_windows,
            assumed_resolution=assumed_resolution,
            intended_use="external_reader_validation",
        )
        coverage = _coverage_summary(spectrum)
        segments = [_segment_summary(segment) for segment in _iter_segments(spectrum)]
        err_present = any(item["err_present"] for item in segments)
        resolution_present = any(item["resolution"] is not None for item in segments)
        diagnostic_selection = _diagnostic_window_selection(
            spectrum,
            args.diagnostic_window_count,
        )
        fit_setup = _fit_setup_recommendation(
            spectrum,
            args,
            assumed_resolution,
        )
        record.update(
            {
                "status": "ok",
                "instrument_info": get_instrument_info(instrument).to_metadata(),
                "fit_windows_A": fit_windows,
                "assumed_resolution_R": assumed_resolution,
                "coverage": coverage,
                "n_segments": len(segments),
                "err_present": bool(err_present),
                "resolution_present": bool(resolution_present),
                "segments": segments,
                "readiness": readiness,
                "fit_setup_recommendation": fit_setup,
                "diagnostic_window_selection": diagnostic_selection,
                "role_expectation_assessment": _role_expectation_assessment(
                    target.get("role"),
                    readiness,
                ),
            }
        )
        print(
            "  Readiness: fit_ready={0}, Nfit={1}, flags={2}".format(
                readiness.get("fit_ready"),
                readiness.get("n_fit_candidate"),
                ", ".join(readiness.get("interpretation_flags") or ()) or "none",
            ),
            flush=True,
        )
        if fit_setup.get("status") == "ok":
            window = fit_setup.get("recommended_window") or {}
            print(
                "  Setup suggestion: mode={0}, branch={1}, window={2}".format(
                    fit_setup.get("mode"),
                    fit_setup.get("recommended_branch_id") or "none",
                    window.get("label") or "none",
                ),
                flush=True,
            )
        if args.plot_dir and not args.no_plots:
            Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
            plot_paths = {}
            if args.plot_style in {"quicklook", "both"}:
                fig, _ax = plot_spectrum_quicklook(
                    spectrum,
                    use_mask=True,
                    show_error=False,
                )
                suffix = "_quicklook" if args.plot_style == "both" else ""
                plot_path = Path(args.plot_dir) / _safe_plot_name(
                    {**target, "target_id": "{0}{1}".format(target["target_id"], suffix)}
                )
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                plot_paths["quicklook"] = str(plot_path)
                print("  Wrote quicklook plot: {0}".format(plot_path), flush=True)
            if args.plot_style in {"audit", "both"}:
                fig, _axes = plot_spectrum_audit(
                    spectrum,
                    title="{0} ({1})".format(target.get("label"), instrument),
                    diagnostic_selection=diagnostic_selection,
                    max_plot_points=args.max_plot_points,
                )
                suffix = "_audit" if args.plot_style == "both" else ""
                plot_path = Path(args.plot_dir) / _safe_plot_name(
                    {**target, "target_id": "{0}{1}".format(target["target_id"], suffix)}
                )
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                plot_paths["audit"] = str(plot_path)
                print("  Wrote audit plot: {0}".format(plot_path), flush=True)
            record["plots"] = plot_paths
    except Exception as exc:  # validation runner should continue across bad files
        record.update(
            {
                "status": "error",
                "error": exc.__class__.__name__,
                "message": str(exc),
            }
        )
        print(
            "  ERROR: {0}: {1}".format(exc.__class__.__name__, exc),
            flush=True,
        )
    return record


def _summarize(records):
    summary = {
        "total": int(len(records)),
        "by_status": {},
        "by_instrument": {},
        "by_role": {},
        "by_role_expectation_assessment": {},
        "fit_ready_count": 0,
        "quicklook_only_count": 0,
        "error_count": 0,
    }
    for record in records:
        status = str(record.get("status", "unknown"))
        instrument = str(record.get("instrument", "unknown"))
        role = str(record.get("role", "unknown"))
        assessment = str(record.get("role_expectation_assessment", "not_evaluated"))
        readiness = record.get("readiness") or {}
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        summary["by_instrument"][instrument] = (
            summary["by_instrument"].get(instrument, 0) + 1
        )
        summary["by_role"][role] = summary["by_role"].get(role, 0) + 1
        summary["by_role_expectation_assessment"][assessment] = (
            summary["by_role_expectation_assessment"].get(assessment, 0) + 1
        )
        if status == "error":
            summary["error_count"] += 1
        if readiness.get("fit_ready"):
            summary["fit_ready_count"] += 1
        elif status == "ok":
            summary["quicklook_only_count"] += 1
    if summary["error_count"]:
        summary["status"] = "external_validation_has_errors"
    elif records and summary["fit_ready_count"] == 0:
        summary["status"] = "external_validation_no_fit_ready_targets"
    elif summary["by_role_expectation_assessment"].get("clean_needs_review"):
        summary["status"] = "external_validation_clean_targets_need_review"
    elif summary["by_role_expectation_assessment"].get("dirty_not_caught_by_readiness"):
        summary["status"] = "external_validation_dirty_targets_not_caught"
    else:
        summary["status"] = "external_validation_summary_ready"
    return summary


def _payload(records, args, targets):
    return {
        "schema_version": 1,
        "operation": "external_spectra_validation",
        "run_configuration": {
            "supported_external_instruments": list(SUPPORTED_EXTERNAL_INSTRUMENTS),
            "manifest": args.manifest,
            "scan_root": args.scan_root or [],
            "fit_window_A": args.fit_window,
            "assumed_resolution_R": args.assumed_resolution,
            "sdss_mask_policy": args.sdss_mask_policy,
            "sdss_attach_wdisp_resolution": bool(args.sdss_attach_wdisp_resolution),
            "uves_wave_unit": args.uves_wave_unit,
            "uves_err_column": args.uves_err_column,
            "plot_style": args.plot_style,
            "setup_mode": args.setup_mode,
            "diagnostic_window_count": int(args.diagnostic_window_count),
            "max_plot_points": int(args.max_plot_points),
            "note": (
                "This runner audits ingestion/readiness only. It does not run "
                "PHOENIX fits, apply SDSS wdisp convolution, or classify stars."
            ),
        },
        "target_count": int(len(targets)),
        "summary": _summarize(records),
        "results": list(records),
    }


def _load_existing_records(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out = {}
    for record in payload.get("results", []):
        target_id = record.get("target_id")
        if target_id:
            out[str(target_id)] = record
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.force and not args.resume:
        args.resume = True
    targets = discover_targets(args)
    records_by_id = _load_existing_records(args.output_json) if args.resume else {}

    print(
        "External spectra validation: {0} target(s).".format(len(targets)),
        flush=True,
    )

    def checkpoint():
        ordered = [
            records_by_id[target["target_id"]]
            for target in targets
            if target["target_id"] in records_by_id
        ]
        payload = _payload(ordered, args, targets)
        _atomic_write_json(args.output_json, payload)
        _atomic_write_csv(args.output_csv, payload)
        return payload

    for index, target in enumerate(targets, start=1):
        target_id = str(target["target_id"])
        existing = records_by_id.get(target_id)
        if (
            args.resume
            and not args.force
            and existing is not None
            and existing.get("status") in COMPLETED_STATUSES
        ):
            print(
                "Skipping completed {0}/{1}: {2}".format(
                    index, len(targets), target_id
                ),
                flush=True,
            )
            continue
        print(
            "\nTarget {0}/{1}".format(index, len(targets)),
            flush=True,
        )
        records_by_id[target_id] = validate_target(target, args)
        payload = checkpoint()
        print(
            "  Wrote checkpoint: {0} ({1})".format(
                args.output_json, payload["summary"]["status"]
            ),
            flush=True,
        )

    payload = checkpoint()
    print(
        "\nDone. status={0}, ok={1}, errors={2}, fit_ready={3}, quicklook_only={4}".format(
            payload["summary"]["status"],
            payload["summary"]["by_status"].get("ok", 0),
            payload["summary"]["error_count"],
            payload["summary"]["fit_ready_count"],
            payload["summary"]["quicklook_only_count"],
        ),
        flush=True,
    )
    print("JSON: {0}".format(os.path.abspath(args.output_json)), flush=True)
    if args.output_csv:
        print("CSV: {0}".format(os.path.abspath(args.output_csv)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
