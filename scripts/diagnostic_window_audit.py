#!/usr/bin/env python
"""Audit diagnostic-window selection across a manifest of real spectra.

This is a validation/diagnostic runner, not a PHOENIX fitting script.  It reads
spectra, runs the lightweight diagnostic-window selector, and writes compact
JSON/CSV summaries so catalog scoring changes can be checked against bundled
real spectra before they affect fitting workflows.

Example
-------
python scripts/diagnostic_window_audit.py \
  examples/xsl_validation_manifest.csv \
  --output-json /tmp/spyctres_diagnostic_window_audit.json \
  --output-csv /tmp/spyctres_diagnostic_window_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np

from Spyctres.diagnostic_windows import (
    diagnostic_window_catalog,
    select_diagnostic_windows,
)
from Spyctres.io import SpectrumCollection, SpectrumSegment, read_spectrum


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "examples" / "xsl_validation_manifest.csv"
)

ORDINARY_VALIDATION_ROLES = {"standard"}

EXPECTED_GROUPS = (
    {
        "id": "balmer_hydrogen",
        "label": "Balmer hydrogen windows",
        "any_of": ("h_delta", "h_gamma", "h_beta", "h_alpha"),
        "teff_min": 4500.0,
        "teff_max": 12000.0,
        "notes": "Hot/intermediate stars should expose at least one Balmer diagnostic when covered.",
    },
    {
        "id": "hot_star_stress_features",
        "label": "He/Mg/Si hot-star stress windows",
        "any_of": ("he_i_4471", "mg_ii_4481", "si_ii_4128_4130", "si_iii_4552"),
        "teff_min": 7500.0,
        "teff_max": 12000.0,
        "notes": (
            "These are useful early-type triage windows, but He/Si entries are "
            "stress-only or model-sensitive in the current PHOENIX workflow."
        ),
    },
    {
        "id": "ch_g_band",
        "label": "CH G-band intermediate/cool check",
        "any_of": ("ch_g_band",),
        "teff_min": 4300.0,
        "teff_max": 6500.0,
        "notes": "CH helps flag F/G/K behaviour but remains abundance/model sensitive.",
    },
    {
        "id": "cool_red_molecular",
        "label": "Cool-star red molecular windows",
        "any_of": ("tio_7050", "tio_red_bands", "vo_7450", "vo_7900", "feh_8700"),
        "teff_min": 0.0,
        "teff_max": 4500.0,
        "notes": "Late-type spectra should expose TiO/VO/FeH-style diagnostics when covered.",
    },
    {
        "id": "cool_alkali_gravity",
        "label": "Cool-star alkali/gravity windows",
        "any_of": ("na_i_8200", "k_i_7700"),
        "teff_min": 0.0,
        "teff_max": 5200.0,
        "notes": "Na I/K I are useful but telluric and pressure-broadening sensitive.",
    },
    {
        "id": "kband_late_type",
        "label": "K-band late-type windows",
        "any_of": ("na_i_kband", "ca_i_kband", "co_23um_bandhead"),
        "teff_min": 0.0,
        "teff_max": 6500.0,
        "notes": "K-band Na/Ca/CO features are useful only when K-band coverage is present.",
    },
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Developer/validation diagnostic-window audit. Reads a manifest, "
            "runs window selection, and records role-aware expectation checks "
            "without fitting PHOENIX models."
        ),
        epilog=(
            "Default XSL Figure 1 audit:\n"
            "  python scripts/diagnostic_window_audit.py "
            "examples/xsl_validation_manifest.csv "
            "--output-json /tmp/spyctres_diagnostic_window_audit.json "
            "--output-csv /tmp/spyctres_diagnostic_window_audit.csv\n\n"
            "Generic manifest rows may include: path, target_id or xsl_id, "
            "instrument, spectral_type, teff_ref, validation_role."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(DEFAULT_MANIFEST),
        help="CSV manifest. Defaults to the bundled XSL Figure 1 manifest.",
    )
    parser.add_argument(
        "--instrument",
        default="xsl",
        help="Default reader when a manifest row lacks an instrument column.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_diagnostic_window_audit.json",
        help="Atomic JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional compact per-target CSV path.",
    )
    parser.add_argument(
        "--roles",
        default=None,
        help="Optional comma-separated selector role filter.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=40,
        help="Maximum selected windows retained per target. Default: 40.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of top selected windows copied into compact summaries.",
    )
    parser.add_argument(
        "--rv",
        type=float,
        default=None,
        help="Optional preliminary RV used only for observed-frame window mapping.",
    )
    parser.add_argument(
        "--rv-padding-kms",
        type=float,
        default=0.0,
        help="Velocity padding for operational diagnostic windows. Default: 0.",
    )
    parser.add_argument(
        "--xsl-id",
        action="append",
        default=None,
        help="Process only this XSL/target ID. Repeat to select multiple targets.",
    )
    parser.add_argument(
        "--fail-on-standard-missing",
        action="store_true",
        help=(
            "Return non-zero if an ordinary standard target is missing an "
            "applicable expected diagnostic group."
        ),
    )
    return parser


def load_manifest_rows(path, *, selected_ids=None):
    path = Path(path)
    selected = None if selected_ids is None else {str(item).upper() for item in selected_ids}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target_id = _target_id(row)
            if selected is not None and target_id.upper() not in selected:
                continue
            rows.append(dict(row))
    if not rows:
        raise ValueError("No manifest rows selected.")
    return rows


def audit_manifest(
    manifest,
    *,
    instrument="xsl",
    roles=None,
    max_windows=40,
    top_n=12,
    rv_kms=None,
    rv_padding_kms=0.0,
    selected_ids=None,
    progress_callback=None,
):
    """Return a JSON-safe diagnostic-window audit payload for ``manifest``."""
    manifest = Path(manifest)
    rows = load_manifest_rows(manifest, selected_ids=selected_ids)
    target_records = []
    for index, row in enumerate(rows, start=1):
        target_id = _target_id(row)
        _emit_progress(
            progress_callback,
            "Auditing diagnostic windows {0}/{1}: {2}".format(
                index,
                len(rows),
                target_id,
            ),
        )
        target_records.append(
            audit_target(
                row,
                manifest=manifest,
                default_instrument=instrument,
                roles=roles,
                max_windows=max_windows,
                top_n=top_n,
                rv_kms=rv_kms,
                rv_padding_kms=rv_padding_kms,
            )
        )
    return _json_native(
        {
            "schema_version": 1,
            "operation": "diagnostic_window_manifest_audit",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "run_policy": {
                "fits_run": False,
                "phoenix_loaded": False,
                "role_aware_statistics": True,
                "ordinary_validation_roles": sorted(ORDINARY_VALIDATION_ROLES),
                "stress_roles_excluded_from_ordinary_recovery": True,
                "max_windows": int(max_windows),
                "top_n": int(top_n),
                "roles_filter": None if roles is None else list(roles),
                "rv_kms": None if rv_kms is None else float(rv_kms),
                "rv_padding_kms": float(rv_padding_kms),
            },
            "summary": summarize_audit_records(target_records),
            "targets": target_records,
        }
    )


def audit_target(
    row,
    *,
    manifest,
    default_instrument,
    roles=None,
    max_windows=40,
    top_n=12,
    rv_kms=None,
    rv_padding_kms=0.0,
):
    """Audit one manifest row and return a compact JSON-safe record."""
    target_id = _target_id(row)
    path = _resolve_manifest_path(manifest, row["path"])
    instrument = str(row.get("instrument") or default_instrument)
    teff_ref = _optional_float(row, "teff_ref")
    validation_role = str(row.get("validation_role") or "standard").strip() or "standard"
    spectrum = read_spectrum(path, instrument=instrument)
    coverage = spectrum_coverage(spectrum)
    selection = select_diagnostic_windows(
        spectrum,
        roles=roles,
        initial_teff=teff_ref,
        rv_kms=rv_kms,
        rv_padding_kms=rv_padding_kms,
        max_windows=max_windows,
    )
    selected = list(selection.get("selected", ()))
    selected_ids = [item["id"] for item in selected]
    top = selected[: int(top_n)]
    expectation_checks = check_expected_groups(
        selected_ids,
        teff_ref=teff_ref,
        validation_role=validation_role,
        coverage_spans=coverage["spans_A"],
    )
    ordinary = validation_role.strip().lower() in ORDINARY_VALIDATION_ROLES
    missing = [
        item["group_id"]
        for item in expectation_checks
        if item.get("status") == "missing_expected_window"
    ]
    return _json_native(
        {
            "target_id": target_id,
            "xsl_id": row.get("xsl_id") or None,
            "path": str(path),
            "instrument": instrument,
            "star_name": row.get("star_name") or None,
            "spectral_type": row.get("spectral_type") or None,
            "teff_ref": teff_ref,
            "validation_role": validation_role,
            "ordinary_recovery_target": bool(ordinary),
            "status": "ok",
            "coverage": coverage,
            "selection_summary": {
                "n_selected": int(len(selected)),
                "n_rejected": int(len(selection.get("rejected", ()))),
                "top_window_ids": [item["id"] for item in top],
                "top_window_labels": [item["label"] for item in top],
                "selected_window_ids": selected_ids,
                "selected_feature_families": sorted(
                    {
                        family
                        for item in selected
                        for family in item.get("feature_family", ())
                    }
                ),
                "selected_model_support": sorted(
                    {item.get("model_support", "unknown") for item in selected}
                ),
                "selected_risk_tags": sorted(
                    {tag for item in selected for tag in item.get("risk_tags", ())}
                ),
            },
            "expectation_checks": expectation_checks,
            "missing_expected_groups": missing,
            "selection": selection,
        }
    )


def spectrum_coverage(spectrum):
    """Return compact wavelength coverage metadata for a segment/collection."""
    segments = _as_segments(spectrum)
    spans = []
    total_pixels = 0
    valid_pixels = 0
    for index, segment in enumerate(segments):
        wave = np.asarray(segment.wave, dtype=float)
        finite = np.isfinite(wave)
        if not np.any(finite):
            continue
        mask = np.asarray(
            getattr(segment, "mask", np.ones(wave.size, dtype=bool)),
            dtype=bool,
        )
        if mask.shape != wave.shape:
            mask = np.ones(wave.size, dtype=bool)
        valid = finite & mask
        total_pixels += int(wave.size)
        valid_pixels += int(np.count_nonzero(valid))
        spans.append(
            {
                "segment_index": int(index),
                "name": getattr(segment, "name", None),
                "wave_min_A": float(np.nanmin(wave[finite])),
                "wave_max_A": float(np.nanmax(wave[finite])),
                "n_pixels": int(wave.size),
                "n_valid_pixels": int(np.count_nonzero(valid)),
                "wave_medium": getattr(segment, "wave_medium", "unknown"),
                "observer_frame": getattr(segment, "observer_frame", "unknown"),
                "stellar_rest_status": getattr(
                    segment,
                    "stellar_rest_status",
                    "unknown",
                ),
            }
        )
    finite_bounds = [
        (item["wave_min_A"], item["wave_max_A"])
        for item in spans
        if np.isfinite(item["wave_min_A"]) and np.isfinite(item["wave_max_A"])
    ]
    if finite_bounds:
        wave_min = min(lo for lo, _hi in finite_bounds)
        wave_max = max(hi for _lo, hi in finite_bounds)
    else:
        wave_min = None
        wave_max = None
    return {
        "n_segments": int(len(spans)),
        "n_pixels": int(total_pixels),
        "n_valid_pixels": int(valid_pixels),
        "wave_min_A": wave_min,
        "wave_max_A": wave_max,
        "spans_A": [[item["wave_min_A"], item["wave_max_A"]] for item in spans],
        "segments": spans,
    }


def check_expected_groups(
    selected_window_ids,
    *,
    teff_ref,
    validation_role,
    coverage_spans,
):
    """Check broad, role-aware expectations against selected window IDs."""
    selected = set(selected_window_ids)
    ordinary = str(validation_role).strip().lower() in ORDINARY_VALIDATION_ROLES
    checks = []
    for group in EXPECTED_GROUPS:
        applicable, reason = _group_applicability(
            group,
            teff_ref=teff_ref,
            coverage_spans=coverage_spans,
        )
        matched = sorted(selected & set(group["any_of"]))
        if not applicable:
            status = reason
        elif matched:
            status = "ok"
        else:
            status = "missing_expected_window"
        checks.append(
            {
                "group_id": group["id"],
                "label": group["label"],
                "status": status,
                "ordinary_recovery_check": bool(ordinary and applicable),
                "severity": "warning" if ordinary and status == "missing_expected_window" else "info",
                "any_of": list(group["any_of"]),
                "matched_window_ids": matched,
                "notes": group["notes"],
            }
        )
    return checks


def summarize_audit_records(records):
    """Summarize per-target audit records with stress roles kept separate."""
    role_counts = Counter(str(row.get("validation_role", "standard")) for row in records)
    window_counts_all = Counter()
    window_counts_ordinary = Counter()
    top_window_counts_all = Counter()
    top_window_counts_ordinary = Counter()
    family_counts_all = Counter()
    family_counts_ordinary = Counter()
    top_family_counts_all = Counter()
    top_family_counts_ordinary = Counter()
    standard_missing = []
    for row in records:
        summary = row.get("selection_summary", {})
        ordinary = bool(row.get("ordinary_recovery_target"))
        top_ids = set(summary.get("top_window_ids", ()))
        for window_id in summary.get("selected_window_ids", ()):
            window_counts_all[window_id] += 1
            if ordinary:
                window_counts_ordinary[window_id] += 1
            if window_id in top_ids:
                top_window_counts_all[window_id] += 1
                if ordinary:
                    top_window_counts_ordinary[window_id] += 1
        for family in summary.get("selected_feature_families", ()):
            family_counts_all[family] += 1
            if ordinary:
                family_counts_ordinary[family] += 1
        top_families = {
            family
            for item in row.get("selection", {}).get("selected", ())
            if item.get("id") in top_ids
            for family in item.get("feature_family", ())
        }
        for family in top_families:
            top_family_counts_all[family] += 1
            if ordinary:
                top_family_counts_ordinary[family] += 1
        if ordinary:
            for check in row.get("expectation_checks", ()):
                if check.get("status") == "missing_expected_window":
                    standard_missing.append(
                        {
                            "target_id": row.get("target_id"),
                            "spectral_type": row.get("spectral_type"),
                            "teff_ref": row.get("teff_ref"),
                            "group_id": check.get("group_id"),
                            "any_of": list(check.get("any_of", ())),
                        }
                    )
    return {
        "n_targets": int(len(records)),
        "n_ordinary_recovery_targets": int(
            sum(bool(row.get("ordinary_recovery_target")) for row in records)
        ),
        "validation_role_counts": dict(sorted(role_counts.items())),
        "ordinary_missing_expected_groups": standard_missing,
        "n_ordinary_missing_expected_groups": int(len(standard_missing)),
        "top_window_frequency_all": dict(top_window_counts_all.most_common()),
        "top_window_frequency_ordinary": dict(
            top_window_counts_ordinary.most_common()
        ),
        "selected_window_frequency_all": dict(window_counts_all.most_common()),
        "selected_window_frequency_ordinary": dict(window_counts_ordinary.most_common()),
        "top_feature_family_frequency_all": dict(top_family_counts_all.most_common()),
        "top_feature_family_frequency_ordinary": dict(
            top_family_counts_ordinary.most_common()
        ),
        "selected_feature_family_frequency_all": dict(family_counts_all.most_common()),
        "selected_feature_family_frequency_ordinary": dict(
            family_counts_ordinary.most_common()
        ),
        "interpretation": (
            "This summary audits whether broad diagnostic families are selected "
            "when they are applicable and covered. It does not validate fitted "
            "stellar parameters and does not rank PHOENIX models."
        ),
    }


def write_audit_json(path, payload):
    """Write audit payload atomically as JSON."""
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


def write_audit_csv(path, payload):
    """Write a compact per-target audit CSV."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_id",
        "instrument",
        "validation_role",
        "ordinary_recovery_target",
        "spectral_type",
        "teff_ref",
        "wave_min_A",
        "wave_max_A",
        "n_segments",
        "n_selected",
        "top_window_ids",
        "selected_feature_families",
        "missing_expected_groups",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in payload.get("targets", ()):
            coverage = row.get("coverage", {})
            summary = row.get("selection_summary", {})
            writer.writerow(
                {
                    "target_id": row.get("target_id"),
                    "instrument": row.get("instrument"),
                    "validation_role": row.get("validation_role"),
                    "ordinary_recovery_target": row.get("ordinary_recovery_target"),
                    "spectral_type": row.get("spectral_type"),
                    "teff_ref": row.get("teff_ref"),
                    "wave_min_A": coverage.get("wave_min_A"),
                    "wave_max_A": coverage.get("wave_max_A"),
                    "n_segments": coverage.get("n_segments"),
                    "n_selected": summary.get("n_selected"),
                    "top_window_ids": ";".join(summary.get("top_window_ids", ())),
                    "selected_feature_families": ";".join(
                        summary.get("selected_feature_families", ())
                    ),
                    "missing_expected_groups": ";".join(
                        row.get("missing_expected_groups", ())
                    ),
                    "status": row.get("status"),
                }
            )


def main(argv=None):
    args = build_parser().parse_args(argv)
    roles = _parse_roles(args.roles)

    def progress(message):
        print(message, flush=True)

    print("Running diagnostic-window audit...", flush=True)
    payload = audit_manifest(
        args.manifest,
        instrument=args.instrument,
        roles=roles,
        max_windows=args.max_windows,
        top_n=args.top_n,
        rv_kms=args.rv,
        rv_padding_kms=args.rv_padding_kms,
        selected_ids=args.xsl_id,
        progress_callback=progress,
    )
    print("Writing diagnostic-window audit outputs...", flush=True)
    write_audit_json(args.output_json, payload)
    write_audit_csv(args.output_csv, payload)
    missing = payload["summary"]["n_ordinary_missing_expected_groups"]
    print(
        "Done. targets={0}, ordinary_missing_expected_groups={1}, output={2}".format(
            payload["summary"]["n_targets"],
            missing,
            args.output_json,
        ),
        flush=True,
    )
    if args.fail_on_standard_missing and missing:
        return 1
    return 0


def _parse_roles(value):
    if value is None:
        return None
    roles = [item.strip() for item in str(value).split(",") if item.strip()]
    return roles or None


def _target_id(row):
    for key in ("target_id", "xsl_id", "name", "star_name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError("Manifest rows require target_id, xsl_id, name, or star_name.")


def _optional_float(row, key):
    value = str(row.get(key, "")).strip()
    return None if not value else float(value)


def _resolve_manifest_path(manifest, value):
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return value
    return str(Path(manifest).resolve().parent / value)


def _as_segments(spectrum):
    if isinstance(spectrum, SpectrumSegment):
        return [spectrum]
    if isinstance(spectrum, SpectrumCollection):
        return list(spectrum.segments)
    if isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    raise TypeError("Expected SpectrumSegment or SpectrumCollection.")


def _group_applicability(group, *, teff_ref, coverage_spans):
    if teff_ref is None:
        return False, "not_checked_no_teff_ref"
    teff = float(teff_ref)
    if not (float(group["teff_min"]) <= teff <= float(group["teff_max"])):
        return False, "not_applicable_teff"
    if not _group_has_coverage(group["any_of"], coverage_spans):
        return False, "not_applicable_no_coverage"
    return True, "applicable"


def _group_has_coverage(window_ids, coverage_spans):
    catalog = {window.id: window for window in diagnostic_window_catalog()}
    for window_id in window_ids:
        window = catalog.get(window_id)
        if window is None:
            continue
        lo, hi = window.region_A
        for span in coverage_spans or ():
            if len(span) != 2:
                continue
            span_lo, span_hi = sorted((float(span[0]), float(span[1])))
            if max(lo, span_lo) < min(hi, span_hi):
                return True
    return False


def _emit_progress(callback, message):
    if callback is not None:
        callback(message)


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


if __name__ == "__main__":
    raise SystemExit(main())
