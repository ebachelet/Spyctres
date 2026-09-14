"""Serializable result containers for public Spyctres workflows."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import os
import subprocess

import numpy as np

from ._serialization import atomic_write_json, json_safe as _jsonable
from ._version import __version__ as PACKAGE_VERSION


FIT_REPORT_SCHEMA_VERSION = 1
FIT_RESULT_PAYLOAD_SCHEMA_VERSION = 1
FIT_REPORT_TYPE = "spyctres.fit_result_report"
FIT_REPORT_SCHEMA_NAME = "spyctres.fit_result_report"
FIT_REPORT_SCHEMA_STATUS = "experimental"


def _spyctres_version():
    if PACKAGE_VERSION:
        return str(PACKAGE_VERSION)
    try:
        return importlib_metadata.version("Spyctres")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _spyctres_git_commit():
    """Return the current git commit if this is a source checkout."""
    env_value = os.environ.get("SPYCTRES_GIT_COMMIT")
    if env_value:
        return str(env_value)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return None


def _looks_like_local_path(value):
    if not isinstance(value, str):
        return False
    if value.startswith("~"):
        return True
    if os.path.isabs(value):
        return True
    # Windows drive path, for JSON produced on non-Windows systems.
    if len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        return True
    return False


def _without_local_paths(value):
    if isinstance(value, Mapping):
        return {str(key): _without_local_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_local_paths(item) for item in value]
    if _looks_like_local_path(value):
        return None
    return value


def _relative_path_for_json(value, relative_to=None, include_local_paths=False):
    value = os.fspath(value)
    if include_local_paths:
        return value
    if _looks_like_local_path(value):
        if relative_to is None:
            raise ValueError(
                "Absolute/local paths are not included in web-ready JSON by "
                "default; pass a relative plot path, set relative_to, or use "
                "include_local_paths=True."
            )
        base = os.path.abspath(os.path.expanduser(os.fspath(relative_to)))
        path = os.path.abspath(os.path.expanduser(value))
        if os.path.commonpath([base, path]) != base:
            raise ValueError(
                "Generated-file paths must live inside the JSON product "
                "directory when include_local_paths=False."
            )
        return os.path.relpath(path, base)
    normalized = os.path.normpath(value)
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError(
            "Generated-file paths must not traverse outside the JSON product "
            "directory when include_local_paths=False."
        )
    return value


def _normalize_plot_paths(plot_paths, relative_to=None, include_local_paths=False):
    if plot_paths is None:
        return None
    if isinstance(plot_paths, Mapping):
        return {
            str(key): _normalize_plot_paths(
                item,
                relative_to=relative_to,
                include_local_paths=include_local_paths,
            )
            for key, item in plot_paths.items()
        }
    if isinstance(plot_paths, (list, tuple)):
        return [
            _normalize_plot_paths(
                item,
                relative_to=relative_to,
                include_local_paths=include_local_paths,
            )
            for item in plot_paths
        ]
    return _relative_path_for_json(
        plot_paths,
        relative_to=relative_to,
        include_local_paths=include_local_paths,
    )


def _mapping_get(mapping, key, default=None):
    if mapping is None:
        return default
    try:
        return mapping.get(key, default)
    except AttributeError:
        try:
            return mapping[key]
        except (KeyError, TypeError):
            return default


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _format_optional(value, precision=5):
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return "{0:.{1}g}".format(number, int(precision))


def _as_result_payload(result):
    if isinstance(result, PhoenixFitResult):
        return result.to_dict(include_arrays=False)
    if hasattr(result, "to_dict"):
        try:
            return result.to_dict(include_arrays=False)
        except TypeError:
            return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError("fit result must be a mapping or PhoenixFitResult-like object.")


def _extract_fit_setup_hash(payload, provenance):
    for mapping, key in (
        (payload, "fit_setup_hash"),
        (provenance, "fit_setup_hash"),
        (
            payload.get("fit_setup") if isinstance(payload, Mapping) else None,
            "setup_hash",
        ),
    ):
        if isinstance(mapping, Mapping):
            value = mapping.get(key)
            if value:
                return str(value)
    fit_setup = payload.get("fit_setup") if isinstance(payload, Mapping) else None
    if isinstance(fit_setup, Mapping):
        value = fit_setup.get("configuration_hash")
        if value:
            return str(value)
    return None


def _fit_report_provenance_summary(payload):
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else {}
    if not isinstance(provenance, Mapping):
        provenance = {}
    fit_setup = payload.get("fit_setup") if isinstance(payload, Mapping) else {}
    if not isinstance(fit_setup, Mapping):
        fit_setup = {}
    setup_provenance = fit_setup.get("provenance")
    if not isinstance(setup_provenance, Mapping):
        setup_provenance = {}
    readiness = _first_present(
        provenance.get("spectrum_readiness"),
        payload.get("spectrum_readiness"),
        fit_setup.get("readiness"),
    )
    if not isinstance(readiness, Mapping):
        readiness = {}
    resolution_override = _first_present(
        provenance.get("resolution_override"),
        payload.get("resolution_override"),
    )
    if not isinstance(resolution_override, Mapping):
        resolution_override = {}
    assumed_resolution = _first_present(
        setup_provenance.get("assumed_resolution"),
        readiness.get("assumed_resolution"),
        resolution_override,
    )
    if not isinstance(assumed_resolution, Mapping):
        assumed_resolution = {}
    archive_mask_policy = _first_present(
        provenance.get("archive_mask_policy"),
        payload.get("archive_mask_policy"),
    )
    if isinstance(archive_mask_policy, Mapping):
        mask_policy = archive_mask_policy.get("policy")
    else:
        mask_policy = archive_mask_policy
    setup_hash = _extract_fit_setup_hash(payload, provenance)
    model_backend = (
        provenance.get("workflow_model")
        or provenance.get("model_backend")
        or payload.get("model")
        or "phoenix"
    )
    return _jsonable(
        {
            "workflow_api": provenance.get("workflow_api") or provenance.get("api"),
            "model_backend": model_backend,
            "fit_setup_source": provenance.get("fit_setup_source"),
            "fit_setup_hash": setup_hash,
            "reader": provenance.get("reader"),
            "instrument": provenance.get("instrument"),
            "input_was_path": provenance.get("input_was_path"),
            "input_checksum_policy": provenance.get("input_checksum_policy"),
            "input_checksum": provenance.get("input_checksum"),
            "readiness_intent": _first_present(
                readiness.get("intent"),
                setup_provenance.get("readiness_intent"),
            ),
            "ready_for_intent": readiness.get("ready_for_intent"),
            "fit_ready": readiness.get("fit_ready"),
            "quality_flags": list(payload.get("quality_flags") or []),
            "mask_policy": mask_policy,
            "archive_mask_policy": archive_mask_policy,
            "resolution_source": _first_present(
                provenance.get("resolution_source"),
                payload.get("resolution_source"),
                resolution_override.get("resolution_source"),
                resolution_override.get("source"),
                assumed_resolution.get("source"),
            ),
            "assumed_resolution_R": _first_present(
                provenance.get("assumed_resolution_R"),
                payload.get("assumed_resolution_R"),
                resolution_override.get("assumed_resolution_R"),
                resolution_override.get("value")
                if assumed_resolution.get("quantity", "R") == "R"
                else None,
                assumed_resolution.get("value")
                if assumed_resolution.get("quantity", "R") == "R"
                else None,
                assumed_resolution.get("R"),
            ),
            "resolution_R": payload.get("resolution_R"),
            "lsf_fwhm_kms": payload.get("lsf_fwhm_kms"),
            "wavelength_medium_override": _first_present(
                provenance.get("wave_medium_override"),
                payload.get("wave_medium_override"),
            ),
            "wavelength_frame_assumption": payload.get("wavelength_frame_assumption"),
            "rv_convention": provenance.get("rv_convention"),
            "rv_bary_explicit": provenance.get("rv_bary_explicit"),
            "phoenix_source_root": provenance.get("phoenix_source_root"),
            "cache_schema_version": provenance.get("cache_schema_version"),
            "cache_path": provenance.get("cache_path"),
            "phoenix_model_tag": provenance.get("phoenix_model_tag"),
            "phoenix_wave_filename": provenance.get("phoenix_wave_filename"),
            "phoenix_wave_medium": provenance.get("phoenix_wave_medium"),
            "phoenix_template_axis_counts": provenance.get(
                "phoenix_template_axis_counts"
            ),
            "phoenix_template_axis_ranges": provenance.get(
                "phoenix_template_axis_ranges"
            ),
            "phoenix_interpolator_manifest_hash": provenance.get(
                "phoenix_interpolator_manifest_hash"
            ),
            "phoenix_interpolator_manifest": provenance.get(
                "phoenix_interpolator_manifest"
            ),
            "phoenix_composition_note": provenance.get("phoenix_composition_note"),
            "parameter_uncertainty": payload.get("parameter_uncertainty"),
        }
    )


def _extract_result_value(payload, key):
    if key in payload:
        return payload.get(key)
    report = payload.get("quality_report") or {}
    if isinstance(report, Mapping) and key in report:
        return report.get(key)
    diagnostics = payload.get("diagnostics") or {}
    if isinstance(diagnostics, Mapping) and key in diagnostics:
        return diagnostics.get(key)
    return None


def _numeric_delta(reference, comparison):
    try:
        ref = float(reference)
        cmp = float(comparison)
    except (TypeError, ValueError):
        return {
            "reference": reference,
            "comparison": comparison,
            "delta": None,
            "abs_delta": None,
            "fractional_delta": None,
        }
    if not np.isfinite(ref) or not np.isfinite(cmp):
        return {
            "reference": reference,
            "comparison": comparison,
            "delta": None,
            "abs_delta": None,
            "fractional_delta": None,
        }
    delta = cmp - ref
    return {
        "reference": ref,
        "comparison": cmp,
        "delta": float(delta),
        "abs_delta": float(abs(delta)),
        "fractional_delta": (
            None if np.isclose(ref, 0.0) else float(delta / ref)
        ),
    }


def _flag_set(payload):
    flags = payload.get("quality_flags")
    if flags is None:
        report = payload.get("quality_report") or {}
        flags = report.get("quality_flags") if isinstance(report, Mapping) else None
    return {str(flag) for flag in list(flags or [])}


def _feature_names_from_payload(payload):
    nonstellar = payload.get("nonstellar_features")
    if nonstellar is None:
        report = payload.get("quality_report") or {}
        nonstellar = (
            report.get("nonstellar_features")
            if isinstance(report, Mapping)
            else None
        )
    features = []
    if isinstance(nonstellar, Mapping):
        features = list(nonstellar.get("features") or [])
    return {
        str(item.get("name", item.get("id", "")))
        for item in features
        if isinstance(item, Mapping) and (item.get("name") or item.get("id"))
    }


def _residual_window_names_from_payload(payload):
    windows = payload.get("known_residual_windows")
    if windows is None:
        report = payload.get("quality_report") or {}
        windows = (
            report.get("known_residual_windows")
            if isinstance(report, Mapping)
            else None
        )
    flagged = []
    if isinstance(windows, Mapping):
        flagged = list(windows.get("flagged_windows") or [])
    return {
        str(item.get("name", "window"))
        for item in flagged
        if isinstance(item, Mapping)
    }


def _payload_quality_summary(payload):
    report = payload.get("quality_report") or {}
    if isinstance(report, Mapping):
        summary = report.get("quality_flag_summary")
        if isinstance(summary, Mapping):
            return dict(summary)
    return summarize_quality_flags(_flag_set(payload))


def _payload_result_status(payload):
    compact = None
    if hasattr(payload, "compact_summary"):
        compact = payload.compact_summary()
    if compact is None and isinstance(payload, Mapping):
        compact = payload.get("compact_summary")
    if isinstance(compact, Mapping) and compact.get("interpretation_status"):
        return str(compact["interpretation_status"])
    fit_setup = payload.get("fit_setup") if isinstance(payload, Mapping) else {}
    if isinstance(fit_setup, Mapping) and fit_setup.get("exploratory_override"):
        return "exploratory_review_only"
    summary = _payload_quality_summary(payload)
    return str(summary.get("headline_status") or "unknown")


def _payload_boundary_flags(payload):
    flags = {
        flag
        for flag in _flag_set(payload)
        if flag == "fit_bound_hit" or flag.startswith("grid_edge_")
    }
    diagnostics = payload.get("diagnostics") if isinstance(payload, Mapping) else {}
    if isinstance(diagnostics, Mapping):
        edge_flags = diagnostics.get("grid_edge_flags") or {}
        if isinstance(edge_flags, Mapping):
            for key, value in edge_flags.items():
                if not value:
                    continue
                key = str(key)
                if key == "fit_bound_hit":
                    flags.add("fit_bound_hit")
                else:
                    flags.add("grid_edge_{0}".format(key))
    return sorted(flags)


def _payload_comparison_summary(payload, label):
    flags = sorted(_flag_set(payload))
    quality_summary = _payload_quality_summary(payload)
    return _jsonable(
        {
            "label": str(label),
            "success": bool(payload.get("success", False)),
            "teff": _extract_result_value(payload, "teff"),
            "feh": _extract_result_value(payload, "feh"),
            "logg": _extract_result_value(payload, "logg"),
            "rv_kms": _extract_result_value(payload, "rv_kms"),
            "chi2_red": _extract_result_value(payload, "chi2_red"),
            "n_points": _extract_result_value(payload, "n_points"),
            "quality_flags": flags,
            "quality_flag_summary": quality_summary,
            "quality_status": quality_summary.get("headline_status"),
            "interpretation_status": _payload_result_status(payload),
            "boundary_flags": _payload_boundary_flags(payload),
        }
    )


def compare_fit_results(
    reference,
    comparison,
    *,
    labels=("reference", "comparison"),
    parameter_keys=("teff", "feh", "logg", "rv_kms"),
    metric_keys=("chi2_red", "mask_fraction", "n_points", "degrees_of_freedom"),
    thresholds=None,
):
    """Return a JSON-safe comparison of two structured fit results.

    The helper is intentionally model-agnostic. It compares values already
    reported by the two fits and does not decide whether a change is
    scientifically acceptable. If ``thresholds`` is supplied as a mapping from
    key to absolute-delta threshold, the corresponding entries include an
    ``exceeds_threshold`` boolean for caller-side sensitivity checks.
    """
    ref_payload = _as_result_payload(reference)
    cmp_payload = _as_result_payload(comparison)
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries.")
    label_ref, label_cmp = [str(label) for label in labels]
    thresholds = {} if thresholds is None else dict(thresholds)

    parameters = {}
    for key in parameter_keys:
        key = str(key)
        entry = _numeric_delta(
            _extract_result_value(ref_payload, key),
            _extract_result_value(cmp_payload, key),
        )
        if key in thresholds and entry["abs_delta"] is not None:
            entry["threshold"] = float(thresholds[key])
            entry["exceeds_threshold"] = bool(
                entry["abs_delta"] > float(thresholds[key])
            )
        parameters[key] = entry

    metrics = {}
    for key in metric_keys:
        key = str(key)
        entry = _numeric_delta(
            _extract_result_value(ref_payload, key),
            _extract_result_value(cmp_payload, key),
        )
        if key in thresholds and entry["abs_delta"] is not None:
            entry["threshold"] = float(thresholds[key])
            entry["exceeds_threshold"] = bool(
                entry["abs_delta"] > float(thresholds[key])
            )
        metrics[key] = entry

    ref_flags = _flag_set(ref_payload)
    cmp_flags = _flag_set(cmp_payload)
    ref_features = _feature_names_from_payload(ref_payload)
    cmp_features = _feature_names_from_payload(cmp_payload)
    ref_windows = _residual_window_names_from_payload(ref_payload)
    cmp_windows = _residual_window_names_from_payload(cmp_payload)

    out = {
        "schema_version": 1,
        "labels": [label_ref, label_cmp],
        "result_summaries": {
            "reference": _payload_comparison_summary(ref_payload, label_ref),
            "comparison": _payload_comparison_summary(cmp_payload, label_cmp),
        },
        "parameters": parameters,
        "metrics": metrics,
        "quality_flags": {
            label_ref: sorted(ref_flags),
            label_cmp: sorted(cmp_flags),
            "common": sorted(ref_flags & cmp_flags),
            "only_" + label_ref: sorted(ref_flags - cmp_flags),
            "only_" + label_cmp: sorted(cmp_flags - ref_flags),
            "changed": bool(ref_flags != cmp_flags),
        },
        "known_features": {
            label_ref: sorted(ref_features),
            label_cmp: sorted(cmp_features),
            "common": sorted(ref_features & cmp_features),
            "only_" + label_ref: sorted(ref_features - cmp_features),
            "only_" + label_cmp: sorted(cmp_features - ref_features),
            "changed": bool(ref_features != cmp_features),
        },
        "known_residual_windows": {
            label_ref: sorted(ref_windows),
            label_cmp: sorted(cmp_windows),
            "common": sorted(ref_windows & cmp_windows),
            "only_" + label_ref: sorted(ref_windows - cmp_windows),
            "only_" + label_cmp: sorted(cmp_windows - ref_windows),
            "changed": bool(ref_windows != cmp_windows),
        },
    }
    return _jsonable(out)


def compare_fits(
    *results,
    labels=None,
    baseline_index=0,
    **kwargs,
):
    """Compare two or more fit results against a baseline.

    ``compare_fit_results(a, b)`` remains the precise two-result helper.
    ``compare_fits(a, b, c, labels=(...))`` is the notebook-friendly facade:
    it compares every non-baseline result to the chosen baseline and returns a
    compact JSON-safe bundle.
    """
    if len(results) == 1 and isinstance(results[0], (list, tuple)):
        results = tuple(results[0])
    if len(results) < 2:
        raise ValueError("compare_fits requires at least two fit results.")
    baseline_index = int(baseline_index)
    if baseline_index < 0 or baseline_index >= len(results):
        raise ValueError("baseline_index is out of range.")
    if labels is None:
        labels = tuple("fit_{0}".format(index) for index in range(len(results)))
    labels = tuple(str(label) for label in labels)
    if len(labels) != len(results):
        raise ValueError("labels must match the number of fit results.")
    if len(results) == 2 and baseline_index == 0:
        return compare_fit_results(results[0], results[1], labels=labels, **kwargs)

    baseline = results[baseline_index]
    baseline_label = labels[baseline_index]
    comparisons = []
    for index, result in enumerate(results):
        if index == baseline_index:
            continue
        comparisons.append(
            {
                "comparison_index": int(index),
                "label": labels[index],
                "relative_to": baseline_label,
                "comparison": compare_fit_results(
                    baseline,
                    result,
                    labels=(baseline_label, labels[index]),
                    **kwargs,
                ),
            }
        )
    return _jsonable(
        {
            "schema_version": 1,
            "operation": "compare_fits",
            "baseline_index": int(baseline_index),
            "baseline_label": baseline_label,
            "labels": list(labels),
            "comparisons": comparisons,
            "interpretation": (
                "Compare parameter changes, metrics, quality flags, known "
                "feature flags, and residual-window changes. This helper does "
                "not decide which fit is scientifically correct."
            ),
        }
    )


def _format_delta(value, digits=4):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return "{0:+.{1}g}".format(value, int(digits))


def _format_table_value(value, digits=5):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return "{0:.{1}g}".format(value, int(digits))


def _comparison_delta(payload, section, key):
    entry = (payload.get(section) or {}).get(key) or {}
    return entry.get("delta")


def _comparison_value(payload, key):
    summaries = payload.get("result_summaries") or {}
    comparison = summaries.get("comparison")
    if isinstance(comparison, Mapping):
        return comparison.get(key)
    entry = (payload.get("parameters") or {}).get(key)
    if isinstance(entry, Mapping) and entry.get("comparison") is not None:
        return entry.get("comparison")
    entry = (payload.get("metrics") or {}).get(key)
    if isinstance(entry, Mapping):
        return entry.get("comparison")
    return None


def _comparison_status(payload):
    summaries = payload.get("result_summaries") or {}
    comparison = summaries.get("comparison")
    if isinstance(comparison, Mapping):
        return str(
            comparison.get("interpretation_status")
            or comparison.get("quality_status")
            or "unknown"
        )
    return "unknown"


def _comparison_boundary_text(payload):
    summaries = payload.get("result_summaries") or {}
    comparison = summaries.get("comparison")
    flags = []
    if isinstance(comparison, Mapping):
        flags = list(comparison.get("boundary_flags") or [])
    if not flags:
        return "none"
    return ",".join(str(flag).replace("grid_edge_", "edge:") for flag in flags)


def format_fit_comparison_table(comparison, *, max_flags=3):
    """Return a compact plain-text table from ``compare_fits`` output.

    The helper is intentionally display-only. It does not decide which fit is
    scientifically preferred; it simply makes parameter/metric deltas easier to
    inspect in notebooks, scripts, GUI logs, and reviewer handoffs.
    """
    comparison = dict(comparison or {})
    rows = []
    if comparison.get("operation") == "compare_fits":
        baseline = str(comparison.get("baseline_label", "baseline"))
        for item in comparison.get("comparisons") or []:
            payload = dict(item.get("comparison") or {})
            rows.append(
                {
                    "label": str(item.get("label", "comparison")),
                    "relative_to": baseline,
                    "payload": payload,
                }
            )
    elif "parameters" in comparison and "metrics" in comparison:
        labels = comparison.get("labels") or ["reference", "comparison"]
        rows.append(
            {
                "label": str(labels[1] if len(labels) > 1 else "comparison"),
                "relative_to": str(labels[0] if labels else "reference"),
                "payload": comparison,
            }
        )
    else:
        raise ValueError(
            "format_fit_comparison_table() expects output from compare_fits() "
            "or compare_fit_results()."
        )

    header = (
        "label                 relative_to      Teff[K]  ΔTeff  logg  Δlogg "
        "[Fe/H]  ΔFe  RV[km/s]   ΔRV  χ²ν   Δχ²ν  status       grid/bounds  flags"
    )
    lines = [
        "Spyctres fit-comparison table",
        header,
        "-" * len(header),
    ]
    max_flags = int(max_flags)
    for row in rows:
        payload = row["payload"]
        flags = payload.get("quality_flags") or {}
        changed = []
        for key in flags:
            if key.startswith("only_"):
                values = flags.get(key) or []
                changed.extend(str(value) for value in values)
        changed = sorted(set(changed))
        changed_non_ok = [
            flag for flag in changed if str(flag).strip().lower() != "ok"
        ]
        if changed_non_ok:
            changed = changed_non_ok
        if not changed:
            flag_text = "unchanged"
        else:
            shown = changed[:max_flags]
            flag_text = ", ".join(shown)
            if len(changed) > len(shown):
                flag_text += ", ...+{0}".format(len(changed) - len(shown))

        lines.append(
            "{label:<21} {relative:<14} {teff:>7} {dteff:>6} "
            "{logg:>5} {dlogg:>6} {feh:>6} {dfeh:>5} "
            "{rv:>8} {drv:>6} {chi2:>5} {dchi2:>6} "
            "{status:<12} {grid:<18} {flags}".format(
                label=str(row["label"])[:21],
                relative=str(row["relative_to"])[:14],
                teff=_format_table_value(_comparison_value(payload, "teff"), digits=5),
                dteff=_format_delta(_comparison_delta(payload, "parameters", "teff")),
                logg=_format_table_value(_comparison_value(payload, "logg"), digits=4),
                dlogg=_format_delta(_comparison_delta(payload, "parameters", "logg")),
                feh=_format_table_value(_comparison_value(payload, "feh"), digits=4),
                dfeh=_format_delta(_comparison_delta(payload, "parameters", "feh")),
                rv=_format_table_value(_comparison_value(payload, "rv_kms"), digits=5),
                drv=_format_delta(_comparison_delta(payload, "parameters", "rv_kms")),
                chi2=_format_table_value(_comparison_value(payload, "chi2_red"), digits=4),
                dchi2=_format_delta(_comparison_delta(payload, "metrics", "chi2_red")),
                status=_comparison_status(payload)[:12],
                grid=_comparison_boundary_text(payload)[:18],
                flags=flag_text,
            )
        )
    lines.append(
        "Interpretation: small deltas indicate stability against the tested "
        "analysis choice; this table is not an uncertainty model."
    )
    return "\n".join(lines)


QUALITY_FLAG_DESCRIPTIONS = {
    "ok": "No fit-quality warning flags were raised.",
    "optimizer_local_minimum_suspected": (
        "The optimizer did not report a clean successful termination; inspect "
        "local starts and residuals before trusting the result."
    ),
    "high_chi2": (
        "The reduced chi-square is above the configured warning threshold; this "
        "can indicate model mismatch, underestimated errors, or preprocessing "
        "problems."
    ),
    "structured_residuals": (
        "Residuals show significant autocorrelation, suggesting unresolved "
        "spectral structure, continuum errors, or correlated noise."
    ),
    "residual_slope": (
        "Residuals have a large wavelength-dependent slope; inspect continuum "
        "normalization, arm scaling, and flux calibration."
    ),
    "fit_bound_hit": (
        "At least one fitted physical parameter lies on an allowed grid or fit "
        "boundary."
    ),
    "resolution_missing": (
        "At least one fitted segment lacks explicit resolution/LSF metadata."
    ),
    "gaussian_lsf_assumed": (
        "The fit used a Gaussian instrumental line-spread function assumption "
        "for at least one segment."
    ),
    "constant_lsf_assumed": (
        "The fit used a constant-velocity/constant-R LSF approximation for at "
        "least one segment; wavelength-dependent or tabulated LSF effects were "
        "not modelled."
    ),
    "low_sampling_warning": (
        "At least one segment is sampled with fewer than about two pixels per "
        "assumed Gaussian-LSF FWHM at its coarsest fitted spacing."
    ),
    "tabulated_lsf_present_but_not_supported_by_fitter": (
        "A tabulated/wavelength-dependent LSF descriptor was present, but the "
        "current PHOENIX fitter only supports constant Gaussian broadening."
    ),
    "wavelength_frame_ambiguous": (
        "At least one fitted segment has an unknown wavelength frame; RV and "
        "barycentric assumptions need review."
    ),
    "unknown_wave_medium_used_in_fit": (
        "At least one fitted segment has unknown air/vacuum wavelength medium; "
        "the fitted RV may absorb a wavelength-medium mismatch."
    ),
    "unknown_observer_frame_used_in_fit": (
        "At least one fitted segment has unknown observer-motion frame; the "
        "reported RV should be treated as an alignment parameter."
    ),
    "stellar_rest_status_unknown": (
        "At least one fitted segment has unknown stellar-rest correction status; "
        "the fitted RV may not be a physical stellar radial velocity."
    ),
    "rv_interpretation_ambiguous": (
        "Wavelength-frame, observer-frame, stellar-rest, or barycentric metadata "
        "are incomplete or internally risky, so interpret RV as a model alignment "
        "term unless the product semantics are verified."
    ),
    "barycentric_correction_recorded_not_applied": (
        "A barycentric/heliocentric correction value is present in metadata but "
        "was not applied by the fitter; verify whether the wavelength array "
        "already includes it."
    ),
    "possible_double_barycentric_or_rest_correction": (
        "A nonzero barycentric term was supplied for data whose metadata suggest "
        "a barycentric or stellar-rest wavelength scale; this may double-correct "
        "the model/data frame."
    ),
    "metadata_incomplete": (
        "One or more wavelength/metadata fields are unknown or incomplete."
    ),
    "mask_fraction_high": (
        "More than half of the pixels inside the selected fit window were "
        "excluded from the fit."
    ),
    "segment_no_fit_pixels": (
        "At least one input segment had no usable fit pixels and was dropped."
    ),
    "too_few_fit_pixels": (
        "At least one retained segment has very few fit pixels compared with "
        "the model complexity."
    ),
    "segment_mask_fraction_high": (
        "At least one retained segment had more than half of its selected "
        "fit-window pixels rejected."
    ),
    "explicit_exclusion_dominates": (
        "Explicit user/recipe exclusions removed more pixels than remained in "
        "at least one segment."
    ),
    "nonfinite_mask_output": (
        "A mask callable produced nonfinite values; those pixels were rejected "
        "under the mask-output policy."
    ),
    "robust_loss_active": (
        "The optimizer used a robust least-squares loss, so the optimizer cost "
        "is not an ordinary Gaussian chi-square likelihood; compare raw/effective "
        "linear chi-square diagnostics separately."
    ),
    "error_floor_applied": (
        "A fractional uncertainty floor was added in quadrature for at least "
        "one segment; inspect raw and effective chi-square diagnostics."
    ),
    "fallback_errors_used": (
        "At least one fitted segment lacked a per-pixel uncertainty array, so "
        "Spyctres used a robust fallback sigma to scale residuals."
    ),
    "chi2_effective_not_calibrated": (
        "The reported reduced chi-square is an effective relative diagnostic, "
        "not a calibrated Gaussian-likelihood goodness-of-fit statistic."
    ),
    "parameter_errors_local_linearized": (
        "Reported parameter errors are local linearized diagnostics derived "
        "from the optimizer Jacobian, not global posterior uncertainties."
    ),
    "parameter_errors_ignore_model_systematics": (
        "Reported parameter errors do not include PHOENIX grid limitations, "
        "continuum placement, LSF, wavelength-frame, or reduction systematics."
    ),
    "parameter_errors_unreliable_if_high_chi2": (
        "The fit has high reduced chi-square, so local Jacobian-based parameter "
        "errors are likely over-optimistic."
    ),
    "parameter_errors_unreliable_if_robust_loss": (
        "A robust optimizer loss was used; local covariance/error estimates "
        "should not be interpreted as ordinary Gaussian likelihood errors."
    ),
    "parameter_errors_unreliable_if_error_floor": (
        "An uncertainty floor was applied, so reported parameter errors depend "
        "on the adopted error-floor model."
    ),
    "parameter_errors_unreliable_if_fallback_errors": (
        "One or more segments used fallback robust-sigma uncertainties, so "
        "reported parameter errors inherit that approximate error model."
    ),
    "parameter_errors_unreliable_if_segment_weights": (
        "Non-unity segment weights were used, so reported parameter errors "
        "depend on the adopted weighting scheme."
    ),
    "nonstellar_feature_overlap": (
        "A known non-stellar feature, such as a diffuse interstellar band, "
        "overlaps the fitted wavelength range; PHOENIX is not expected to "
        "reproduce this absorption."
    ),
    "known_line_region_residual": (
        "A curated diagnostic line/window shows coherent residuals. This is "
        "not masked automatically; inspect stellar parameters, continuum "
        "placement, LSF/rotation assumptions, and model-domain limitations."
    ),
    "diagnostic_line_contaminated": (
        "A known non-stellar feature overlaps a stellar diagnostic line/window; "
        "compare a run with an explicit named exclusion mask before interpreting "
        "the affected stellar parameter constraints."
    ),
    "dib_overlap_balmer_wing": (
        "A diffuse interstellar band overlaps a Balmer-line wing. PHOENIX does "
        "not model DIB absorption, so inspect or mask that interval explicitly "
        "before tuning stellar parameters to absorb the residual."
    ),
    "dib_candidate_detected": (
        "Residuals inside a known diffuse-interstellar-band window are coherent "
        "enough to be reported as a candidate, not a confirmed ISM measurement."
    ),
    "nonstellar_feature_frame_ambiguous": (
        "A known non-stellar feature with velocity-frame sensitivity overlaps "
        "the fit, but the spectrum frame metadata are incomplete or unknown."
    ),
    "nonstellar_mask_applied": (
        "At least one known non-stellar feature mask was explicitly applied. "
        "Inspect the mask provenance before comparing this fit with unmasked "
        "runs."
    ),
    "telluric_mask_frame_ambiguous": (
        "A topocentric telluric mask was applied to a spectrum whose wavelength "
        "grid is not known to be raw/topocentric; the masked telluric intervals "
        "may be misaligned."
    ),
    "coarse_telluric_mask_applied": (
        "A broad catalog-region telluric mask was applied. This is a coarse "
        "fallback, not the high-resolution transmission-threshold mask."
    ),
}


QUALITY_FLAG_SEVERITY_ORDER = ("blocker", "review", "advisory", "info")
QUALITY_FLAG_SEVERITY_RANK = {
    severity: index for index, severity in enumerate(QUALITY_FLAG_SEVERITY_ORDER)
}


QUALITY_FLAG_CLASSIFICATION = {
    "ok": {
        "severity": "info",
        "category": "success",
        "action": "No action required beyond ordinary plot/residual inspection.",
    },
    "optimizer_local_minimum_suspected": {
        "severity": "blocker",
        "category": "optimizer",
        "action": (
            "Inspect local starts and residuals; rerun with broader bounds or "
            "additional starts before interpreting parameters."
        ),
    },
    "high_chi2": {
        "severity": "review",
        "category": "fit_quality",
        "action": (
            "Inspect residual plots and the error model; test continuum, LSF, "
            "mask, abundance, activity, and model-domain assumptions."
        ),
    },
    "structured_residuals": {
        "severity": "review",
        "category": "fit_quality",
        "action": (
            "Inspect line-window residuals and compare controlled variants; do "
            "not tune stellar parameters to absorb coherent non-stellar or "
            "reduction residuals."
        ),
    },
    "residual_slope": {
        "severity": "review",
        "category": "continuum",
        "action": (
            "Review continuum normalization, segment scaling, and flux "
            "calibration; compare a controlled continuum-degree variant."
        ),
    },
    "fit_bound_hit": {
        "severity": "review",
        "category": "model_grid",
        "action": (
            "Check which parameter hit a bound and rerun with a wider or more "
            "appropriate grid if the model domain supports it."
        ),
    },
    "resolution_missing": {
        "severity": "review",
        "category": "lsf",
        "action": (
            "Provide or validate resolution/LSF metadata before interpreting "
            "line widths or gravity-sensitive profiles."
        ),
    },
    "gaussian_lsf_assumed": {
        "severity": "advisory",
        "category": "lsf",
        "action": (
            "Record that a Gaussian LSF was used; verify this is adequate for "
            "precision line-profile work."
        ),
    },
    "constant_lsf_assumed": {
        "severity": "advisory",
        "category": "lsf",
        "action": (
            "Treat line-width results as constant-LSF approximations; compare "
            "wavelength-dependent LSF support when it becomes available."
        ),
    },
    "low_sampling_warning": {
        "severity": "review",
        "category": "lsf",
        "action": (
            "Avoid precision line-width interpretation; use coarser windows or "
            "validated higher-resolution data/LSF assumptions."
        ),
    },
    "tabulated_lsf_present_but_not_supported_by_fitter": {
        "severity": "review",
        "category": "lsf",
        "action": (
            "Keep tabulated LSF as provenance for now; use an explicit constant "
            "R only for quicklook fits until wavelength-dependent LSF fitting "
            "is validated."
        ),
    },
    "wavelength_frame_ambiguous": {
        "severity": "review",
        "category": "wavelength_frame",
        "action": (
            "Verify observer-frame and rest-frame semantics before interpreting "
            "RV or comparing line centres."
        ),
    },
    "unknown_wave_medium_used_in_fit": {
        "severity": "review",
        "category": "wavelength_medium",
        "action": "Confirm air/vacuum convention and rerun if line centres matter.",
    },
    "unknown_observer_frame_used_in_fit": {
        "severity": "review",
        "category": "wavelength_frame",
        "action": (
            "Confirm whether wavelengths are topocentric, heliocentric, or "
            "barycentric before treating RV as physical."
        ),
    },
    "stellar_rest_status_unknown": {
        "severity": "review",
        "category": "wavelength_frame",
        "action": (
            "Confirm whether the spectrum is already stellar-rest corrected "
            "before applying or interpreting stellar RV shifts."
        ),
    },
    "rv_interpretation_ambiguous": {
        "severity": "review",
        "category": "velocity",
        "action": (
            "Treat RV as a model-alignment parameter until wavelength-frame, "
            "stellar-rest, and barycentric semantics are verified."
        ),
    },
    "barycentric_correction_recorded_not_applied": {
        "severity": "advisory",
        "category": "velocity",
        "action": (
            "Check product documentation/header comments before applying any "
            "barycentric correction manually."
        ),
    },
    "possible_double_barycentric_or_rest_correction": {
        "severity": "blocker",
        "category": "velocity",
        "action": (
            "Stop and verify product frame semantics; remove the extra velocity "
            "term if the wavelength array is already corrected."
        ),
    },
    "metadata_incomplete": {
        "severity": "review",
        "category": "metadata",
        "action": (
            "Fill or explicitly acknowledge missing wavelength, frame, "
            "uncertainty, or resolution metadata before quantitative use."
        ),
    },
    "mask_fraction_high": {
        "severity": "review",
        "category": "mask",
        "action": (
            "Inspect mask provenance and fit windows; reduce unnecessary "
            "exclusions or choose windows with enough retained pixels."
        ),
    },
    "segment_no_fit_pixels": {
        "severity": "blocker",
        "category": "mask",
        "action": (
            "Inspect dropped segments; choose overlapping windows or fix masks "
            "before claiming a multi-segment fit."
        ),
    },
    "too_few_fit_pixels": {
        "severity": "blocker",
        "category": "mask",
        "action": (
            "Use broader/cleaner windows or reduce model complexity; there are "
            "too few fitted pixels for a stable interpretation."
        ),
    },
    "segment_mask_fraction_high": {
        "severity": "review",
        "category": "mask",
        "action": (
            "Inspect per-segment mask split; most masked pixels may be outside "
            "the window, but high in-window rejection needs review."
        ),
    },
    "explicit_exclusion_dominates": {
        "severity": "review",
        "category": "mask",
        "action": (
            "Check whether explicit exclusions are too broad; compare a "
            "controlled narrower-mask variant."
        ),
    },
    "nonfinite_mask_output": {
        "severity": "blocker",
        "category": "mask",
        "action": (
            "Fix the mask callable so it returns finite values that can be "
            "converted cleanly to boolean mask semantics."
        ),
    },
    "robust_loss_active": {
        "severity": "advisory",
        "category": "optimizer",
        "action": (
            "Report robust-loss use and compare raw/effective chi-square rather "
            "than interpreting optimizer cost as a Gaussian likelihood."
        ),
    },
    "error_floor_applied": {
        "severity": "advisory",
        "category": "uncertainty",
        "action": (
            "Record the adopted uncertainty floor and compare raw/effective "
            "diagnostics when judging fit quality."
        ),
    },
    "fallback_errors_used": {
        "severity": "review",
        "category": "uncertainty",
        "action": (
            "Avoid calibrated chi-square/uncertainty claims until formal "
            "per-pixel errors are supplied or externally validated."
        ),
    },
    "chi2_effective_not_calibrated": {
        "severity": "advisory",
        "category": "uncertainty",
        "action": (
            "Use chi-square as a relative diagnostic only; do not quote it as a "
            "calibrated goodness-of-fit probability."
        ),
    },
    "parameter_errors_local_linearized": {
        "severity": "info",
        "category": "uncertainty",
        "action": (
            "Report parameter errors as local optimizer diagnostics, not global "
            "posterior intervals."
        ),
    },
    "parameter_errors_ignore_model_systematics": {
        "severity": "advisory",
        "category": "uncertainty",
        "action": (
            "Add external/systematic uncertainty terms before quoting final "
            "stellar-parameter uncertainties."
        ),
    },
    "parameter_errors_unreliable_if_high_chi2": {
        "severity": "review",
        "category": "uncertainty",
        "action": "Do not rely on local covariance errors until high chi-square is understood.",
    },
    "parameter_errors_unreliable_if_robust_loss": {
        "severity": "advisory",
        "category": "uncertainty",
        "action": "Treat covariance errors from robust-loss fits as approximate diagnostics.",
    },
    "parameter_errors_unreliable_if_error_floor": {
        "severity": "advisory",
        "category": "uncertainty",
        "action": "State the adopted error floor when reporting parameter errors.",
    },
    "parameter_errors_unreliable_if_fallback_errors": {
        "severity": "review",
        "category": "uncertainty",
        "action": "Supply formal errors before treating parameter uncertainties as calibrated.",
    },
    "parameter_errors_unreliable_if_segment_weights": {
        "severity": "advisory",
        "category": "uncertainty",
        "action": "State segment weights and test sensitivity to them.",
    },
    "nonstellar_feature_overlap": {
        "severity": "review",
        "category": "nonstellar_feature",
        "action": (
            "Annotate and inspect the overlapping feature; run a named-mask "
            "sensitivity fit only if the residuals support it."
        ),
    },
    "known_line_region_residual": {
        "severity": "review",
        "category": "residual_window",
        "action": (
            "Inspect the flagged residual window and compare other diagnostics "
            "before assigning a physical cause."
        ),
    },
    "diagnostic_line_contaminated": {
        "severity": "review",
        "category": "nonstellar_feature",
        "action": (
            "Compare fits with and without an explicit named mask, and check "
            "other diagnostic lines before interpreting the affected line."
        ),
    },
    "dib_overlap_balmer_wing": {
        "severity": "review",
        "category": "nonstellar_feature",
        "action": (
            "Treat the DIB as a candidate contaminant; inspect/mask explicitly "
            "rather than letting the stellar model absorb it."
        ),
    },
    "dib_candidate_detected": {
        "severity": "advisory",
        "category": "nonstellar_feature",
        "action": (
            "Report as a candidate DIB residual only; confirmation requires "
            "independent checks and line-of-sight context."
        ),
    },
    "nonstellar_feature_frame_ambiguous": {
        "severity": "review",
        "category": "nonstellar_feature",
        "action": (
            "Verify the spectrum frame or assumed ISM velocity before applying "
            "a fixed feature mask."
        ),
    },
    "nonstellar_mask_applied": {
        "severity": "advisory",
        "category": "mask",
        "action": (
            "Record which named non-stellar masks were applied and compare with "
            "the unmasked baseline."
        ),
    },
    "telluric_mask_frame_ambiguous": {
        "severity": "review",
        "category": "telluric",
        "action": (
            "Verify the mask frame against the spectrum frame; topocentric "
            "telluric masks can be wrong on shifted wavelength grids."
        ),
    },
    "coarse_telluric_mask_applied": {
        "severity": "advisory",
        "category": "telluric",
        "action": (
            "Prefer a validated transmission-threshold mask for quantitative "
            "fits; broad catalog masks are quicklook fallbacks."
        ),
    },
}


QUALITY_FLAG_BLOCKING_FLAGS = tuple(
    flag
    for flag, record in QUALITY_FLAG_CLASSIFICATION.items()
    if record["severity"] == "blocker"
)
QUALITY_FLAG_REVIEW_FLAGS = tuple(
    flag
    for flag, record in QUALITY_FLAG_CLASSIFICATION.items()
    if record["severity"] == "review"
)
QUALITY_FLAG_ADVISORY_FLAGS = tuple(
    flag
    for flag, record in QUALITY_FLAG_CLASSIFICATION.items()
    if record["severity"] == "advisory"
)
QUALITY_FLAG_INFO_FLAGS = tuple(
    flag
    for flag, record in QUALITY_FLAG_CLASSIFICATION.items()
    if record["severity"] == "info"
)


def _describe_dynamic_quality_flag(flag):
    flag = str(flag)
    if flag.startswith("grid_edge_"):
        tail = flag[len("grid_edge_"):]
        parts = tail.split("_")
        if len(parts) == 1:
            return (
                "The fitted {0} value lies on or very near the available model "
                "grid boundary.".format(parts[0])
            )
        if len(parts) == 2 and parts[1] in {"low", "high"}:
            return (
                "The fitted {0} value lies on or very near the {1} edge of the "
                "available model grid.".format(parts[0], parts[1])
            )
    return "No description is registered for this quality flag yet."


def _dynamic_quality_flag_classification(flag):
    flag = str(flag)
    if flag.startswith("grid_edge_"):
        return {
            "severity": "review",
            "category": "model_grid",
            "action": (
                "Inspect the fitted parameter against the available grid and "
                "rerun with a more appropriate bounded range if needed."
            ),
        }
    if flag.endswith("_high_chi2_proxy"):
        return {
            "severity": "review",
            "category": "comparison_proxy",
            "action": (
                "Inspect the corresponding comparison residuals; proxy "
                "chi-square is a diagnostic, not a likelihood ranking."
            ),
        }
    if flag.endswith("_large_median_abs_sigma") or flag.endswith(
        "_large_fractional_rms"
    ):
        return {
            "severity": "review",
            "category": "comparison_proxy",
            "action": (
                "Inspect the held-out/common-window residuals before treating "
                "this window set as stable."
            ),
        }
    if flag.endswith("_warning"):
        return {
            "severity": "review",
            "category": "warning",
            "action": "Inspect this warning before interpreting the affected result.",
        }
    return {
        "severity": "review",
        "category": "unclassified",
        "action": (
            "Inspect this unclassified quality flag in the full report before "
            "using the fitted parameters."
        ),
    }


def classify_quality_flag(flag):
    """Return severity/category/action metadata for one quality flag.

    The returned mapping is JSON-safe and intentionally user-facing.  Static
    flags are covered by ``QUALITY_FLAG_CLASSIFICATION``; dynamic families such
    as ``grid_edge_teff_high`` and comparison-proxy flags are classified by
    pattern.
    """
    flag = str(flag)
    metadata = dict(
        QUALITY_FLAG_CLASSIFICATION.get(
            flag,
            _dynamic_quality_flag_classification(flag),
        )
    )
    severity = str(metadata.get("severity", "review")).lower()
    if severity not in QUALITY_FLAG_SEVERITY_RANK:
        severity = "review"
    metadata["flag"] = flag
    metadata["severity"] = severity
    metadata.setdefault("category", "unclassified")
    metadata.setdefault("action", "Inspect this flag before interpreting the result.")
    metadata["description"] = QUALITY_FLAG_DESCRIPTIONS.get(
        flag,
        _describe_dynamic_quality_flag(flag),
    )
    metadata["blocks_interpretation"] = bool(severity == "blocker")
    metadata["needs_review"] = bool(severity in {"blocker", "review"})
    metadata["priority"] = int(
        metadata.get("priority", QUALITY_FLAG_SEVERITY_RANK[severity])
    )
    return _jsonable(metadata)


def quality_flag_actions(flags, *, max_actions=None, include_ok=False):
    """Return sorted user actions for quality flags.

    ``ok`` is omitted when other flags are present unless ``include_ok`` is
    True.  This keeps notebook output focused on the few things a user should
    actually do next.
    """
    unique_flags = sorted({str(flag) for flag in (flags or [])})
    if not include_ok and len(unique_flags) > 1:
        unique_flags = [flag for flag in unique_flags if flag.lower() != "ok"]
    actions = [classify_quality_flag(flag) for flag in unique_flags]
    actions.sort(
        key=lambda item: (
            QUALITY_FLAG_SEVERITY_RANK.get(item["severity"], 99),
            int(item.get("priority", 99)),
            str(item.get("flag")),
        )
    )
    if max_actions is not None:
        max_actions = int(max_actions)
        if max_actions < 0:
            raise ValueError("max_actions must be >= 0 when supplied.")
        actions = actions[:max_actions]
    return _jsonable(actions)


def summarize_quality_flags(flags, *, max_actions=3):
    """Return a compact severity/action summary for quality flags."""
    actions = quality_flag_actions(flags, include_ok=True)
    counts = {severity: 0 for severity in QUALITY_FLAG_SEVERITY_ORDER}
    for action in actions:
        severity = action.get("severity", "review")
        counts[severity] = counts.get(severity, 0) + 1
    if counts.get("blocker", 0):
        headline = "blocked_for_interpretation"
    elif counts.get("review", 0):
        headline = "needs_review"
    elif counts.get("advisory", 0):
        headline = "usable_with_caveats"
    else:
        headline = "ok"
    return _jsonable(
        {
            "schema_version": 1,
            "headline_status": headline,
            "counts_by_severity": counts,
            "n_flags": len(actions),
            "blocking_flags": [
                item["flag"] for item in actions if item["severity"] == "blocker"
            ],
            "review_flags": [
                item["flag"] for item in actions if item["severity"] == "review"
            ],
            "advisory_flags": [
                item["flag"] for item in actions if item["severity"] == "advisory"
            ],
            "info_flags": [
                item["flag"] for item in actions if item["severity"] == "info"
            ],
            "top_actions": quality_flag_actions(
                flags,
                max_actions=max_actions,
                include_ok=False,
            ),
        }
    )


def describe_quality_flags(flags):
    """Return human-readable descriptions for quality-flag strings."""
    return {
        str(flag): QUALITY_FLAG_DESCRIPTIONS.get(
            str(flag), _describe_dynamic_quality_flag(flag)
        )
        for flag in list(flags or [])
    }


def build_fit_quality_report(summary, diagnostics=None, quality_flags=None):
    """Return a compact, JSON-safe summary of fit quality diagnostics.

    Parameters
    ----------
    summary : mapping
        Fit result dictionary or summary payload.
    diagnostics : mapping, optional
        Diagnostics block. If omitted, ``summary["diagnostics"]`` is used.
    quality_flags : sequence, optional
        Quality flags. If omitted, ``summary["quality_flags"]`` is used.

    Notes
    -----
    This report is intentionally redundant with the full diagnostics block. The
    diagnostics preserve the detailed machine-readable record, while this report
    gives notebooks, scripts, and reviewers a stable headline view of the main
    fit-quality concerns.
    """
    summary = {} if summary is None else dict(summary)
    if diagnostics is None:
        diagnostics = summary.get("diagnostics", {})
    if hasattr(diagnostics, "to_dict"):
        diagnostics = diagnostics.to_dict()
    diagnostics = {} if diagnostics is None else dict(diagnostics)
    if quality_flags is None:
        quality_flags = summary.get("quality_flags", ())
    flags = list(quality_flags or ())
    flag_summary = summarize_quality_flags(flags)

    chi2_red = summary.get("chi2_red", diagnostics.get("reduced_chi2"))
    segments = []
    for segment in diagnostics.get("segment_diagnostics", []):
        mask_summary = dict(segment.get("mask_summary", {}))
        segments.append(
            {
                "name": segment.get("name"),
                "input_index": segment.get("input_index"),
                "n_fit": segment.get("n_fit"),
                "n_support": segment.get("n_support"),
                "mask_fraction": segment.get("mask_fraction"),
                "outside_fit_window_fraction": segment.get(
                    "outside_fit_window_fraction",
                    mask_summary.get("outside_fit_window_fraction"),
                ),
                "rejected_inside_fit_window_fraction": segment.get(
                    "rejected_inside_fit_window_fraction",
                    mask_summary.get("rejected_inside_fit_window_fraction"),
                ),
                "explicit_exclusion_count": mask_summary.get(
                    "n_rejected_by_explicit_union"
                ),
                "explicit_exclusion_fraction": mask_summary.get(
                    "explicit_exclusion_fraction"
                ),
                "data_invalid_count": mask_summary.get(
                    "n_rejected_by_data_invalid"
                ),
                "data_invalid_fraction": mask_summary.get("data_invalid_fraction"),
                "multiple_rejection_count": mask_summary.get(
                    "n_rejected_by_multiple_reasons"
                ),
                "multiple_rejection_fraction": mask_summary.get(
                    "multiple_rejection_fraction"
                ),
                "input_error_model": segment.get("input_error_model"),
                "lsf_fwhm_kms": segment.get("lsf_fwhm_kms"),
                "resolution_R_effective": segment.get("resolution_R_effective"),
            }
        )
    report = {
        "success": summary.get("success"),
        "quality_flags": flags,
        "quality_flag_descriptions": describe_quality_flags(flags),
        "quality_flag_actions": flag_summary["top_actions"],
        "quality_flag_summary": flag_summary,
        "reduced_chi2": chi2_red,
        "effective_chi2": summary.get(
            "effective_chi2", diagnostics.get("effective_chi2")
        ),
        "effective_chi2_red": summary.get(
            "effective_chi2_red", diagnostics.get("effective_chi2_red")
        ),
        "raw_chi2": summary.get("raw_chi2", diagnostics.get("raw_chi2")),
        "raw_chi2_red": summary.get(
            "raw_chi2_red", diagnostics.get("raw_chi2_red")
        ),
        "error_model": summary.get("error_model", diagnostics.get("error_model")),
        "error_floor_applied": summary.get(
            "error_floor_applied", diagnostics.get("error_floor_applied")
        ),
        "fallback_errors_used": summary.get(
            "fallback_errors_used", diagnostics.get("fallback_errors_used")
        ),
        "fallback_error_segments": summary.get(
            "fallback_error_segments", diagnostics.get("fallback_error_segments")
        ),
        "chi2_calibrated": summary.get(
            "chi2_calibrated", diagnostics.get("chi2_calibrated")
        ),
        "chi2_interpretation": summary.get(
            "chi2_interpretation", diagnostics.get("chi2_interpretation")
        ),
        "optimizer_loss": summary.get(
            "optimizer_loss", diagnostics.get("optimizer_loss")
        ),
        "optimizer_loss_f_scale": summary.get(
            "optimizer_loss_f_scale", diagnostics.get("optimizer_loss_f_scale")
        ),
        "optimizer_cost": summary.get(
            "optimizer_cost", diagnostics.get("optimizer_cost")
        ),
        "optimizer_cost_twice": summary.get(
            "optimizer_cost_twice", diagnostics.get("optimizer_cost_twice")
        ),
        "parameter_uncertainty": summary.get(
            "parameter_uncertainty", diagnostics.get("parameter_uncertainty")
        ),
        "n_points": summary.get("n_points", diagnostics.get("n_pixels")),
        "n_parameters": diagnostics.get("n_parameters"),
        "degrees_of_freedom": diagnostics.get("degrees_of_freedom"),
        "mask_fraction": diagnostics.get("mask_fraction"),
        "outside_fit_window_fraction": diagnostics.get(
            "outside_fit_window_fraction"
        ),
        "rejected_inside_fit_window_fraction": diagnostics.get(
            "rejected_inside_fit_window_fraction"
        ),
        "n_input_segments": diagnostics.get("n_input_segments"),
        "n_retained_segments": diagnostics.get("n_retained_segments"),
        "n_dropped_segments": diagnostics.get("n_dropped_segments"),
        "segments": segments,
    }
    if summary.get("nonstellar_features") is not None:
        report["nonstellar_features"] = summary.get("nonstellar_features")
    if summary.get("known_residual_windows") is not None:
        report["known_residual_windows"] = summary.get("known_residual_windows")
    return _jsonable(report)


def format_fit_quality_report(result_or_report):
    """Return a compact human-readable fit-quality summary.

    Accepts a ``PhoenixFitResult``, a low-level fit-result dictionary, or an
    already-built quality-report dictionary.
    """
    if hasattr(result_or_report, "quality_report"):
        report = result_or_report.quality_report()
    elif isinstance(result_or_report, Mapping):
        if "quality_report" in result_or_report:
            report = result_or_report["quality_report"]
        elif "segments" in result_or_report and "quality_flags" in result_or_report:
            report = result_or_report
        else:
            report = build_fit_quality_report(result_or_report)
    else:
        report = {}

    report = {} if report is None else dict(report)
    lines = ["Quality report:"]
    flags = report.get("quality_flags") or ["unknown"]
    lines.append("  flags: {0}".format(", ".join(str(flag) for flag in flags)))
    flag_summary = report.get("quality_flag_summary")
    if not isinstance(flag_summary, Mapping):
        flag_summary = summarize_quality_flags(flags)
    status = flag_summary.get("headline_status")
    counts = flag_summary.get("counts_by_severity") or {}
    if status and status != "ok":
        lines.append(
            "  flag status: {0} (blockers={1}, review={2}, advisory={3})".format(
                status,
                int(counts.get("blocker", 0) or 0),
                int(counts.get("review", 0) or 0),
                int(counts.get("advisory", 0) or 0),
            )
        )
    top_actions = list(flag_summary.get("top_actions") or [])
    if top_actions:
        lines.append("  top actions:")
        for item in top_actions[:3]:
            lines.append(
                "    - {0} [{1}]: {2}".format(
                    item.get("flag"),
                    item.get("severity"),
                    item.get("action"),
                )
            )
    if report.get("reduced_chi2") is not None:
        lines.append("  chi2_red: {0:.4g}".format(float(report["reduced_chi2"])))
    if report.get("optimizer_loss") and report.get("optimizer_loss") != "linear":
        lines.append(
            "  optimizer loss: {0} (f_scale={1})".format(
                report.get("optimizer_loss"),
                report.get("optimizer_loss_f_scale"),
            )
        )
    if report.get("error_model") and report.get("error_model") != "nominal":
        lines.append("  error model: {0}".format(report.get("error_model")))
    if (
        report.get("raw_chi2_red") is not None
        and report.get("effective_chi2_red") is not None
        and report.get("error_model") != "nominal"
    ):
        lines.append(
            "  chi2_red raw/effective: {0:.4g}/{1:.4g}".format(
                float(report["raw_chi2_red"]),
                float(report["effective_chi2_red"]),
            )
        )
    point_parts = []
    if report.get("n_points") is not None:
        point_parts.append("N={0}".format(int(report["n_points"])))
    if report.get("degrees_of_freedom") is not None:
        point_parts.append("dof={0}".format(int(report["degrees_of_freedom"])))
    if report.get("n_parameters") is not None:
        point_parts.append("parameters={0}".format(int(report["n_parameters"])))
    if point_parts:
        lines.append("  fit size: {0}".format(", ".join(point_parts)))
    if report.get("mask_fraction") is not None:
        lines.append("  masked fraction: {0:.1%}".format(float(report["mask_fraction"])))
    mask_split = []
    if report.get("outside_fit_window_fraction") is not None:
        mask_split.append(
            "outside fit window={0:.1%}".format(
                float(report["outside_fit_window_fraction"])
            )
        )
    if report.get("rejected_inside_fit_window_fraction") is not None:
        mask_split.append(
            "rejected inside fit window={0:.1%}".format(
                float(report["rejected_inside_fit_window_fraction"])
            )
        )
    if mask_split:
        lines.append("  mask split: {0}".format(", ".join(mask_split)))
    if report.get("n_dropped_segments"):
        lines.append("  dropped segments: {0}".format(int(report["n_dropped_segments"])))
    nonstellar = report.get("nonstellar_features") or {}
    nonstellar_features = list(nonstellar.get("features") or [])
    if nonstellar_features:
        names = ", ".join(str(item.get("name", "feature")) for item in nonstellar_features)
        policy = nonstellar.get("policy")
        action = "masked" if nonstellar.get("mask_dibs") else "shown/not masked"
        if policy:
            action = "{0}; policy={1}".format(action, policy)
        lines.append("  non-stellar features: {0} ({1})".format(names, action))
    overlap_diagnostics = list(nonstellar.get("overlap_diagnostics") or [])
    if overlap_diagnostics:
        pieces = []
        for item in overlap_diagnostics:
            hypothesis = item.get("origin_hypothesis")
            suffix = "" if not hypothesis else " ({0})".format(hypothesis)
            pieces.append(
                "{0} -> {1}{2}".format(
                    item.get("feature", "feature"),
                    item.get("diagnostic_line", "diagnostic line"),
                    suffix,
                )
            )
        lines.append("  contaminated diagnostics: {0}".format("; ".join(pieces)))
    residual_windows = report.get("known_residual_windows") or {}
    flagged_windows = list(residual_windows.get("flagged_windows") or [])
    if flagged_windows:
        parts = []
        for item in flagged_windows:
            name = str(item.get("name", "window"))
            median_sigma = item.get("median_sigma")
            rms_sigma = item.get("rms_sigma")
            origin = item.get("origin_hypothesis")
            if median_sigma is not None and rms_sigma is not None:
                text = "{0} median={1:.2g}σ rms={2:.2g}σ".format(
                    name,
                    float(median_sigma),
                    float(rms_sigma),
                )
                if origin:
                    text = "{0} origin={1}".format(text, origin)
                parts.append(text)
            else:
                parts.append(name)
        lines.append("  known residual windows: {0}".format("; ".join(parts)))
    segment_lines = []
    for segment in report.get("segments", []):
        name = segment.get("name")
        label = str(name) if name else "segment {0}".format(
            segment.get("input_index", "?")
        )
        pieces = [label]
        if segment.get("n_fit") is not None and segment.get("n_support") is not None:
            pieces.append(
                "Nfit={0}/{1}".format(
                    int(segment["n_fit"]),
                    int(segment["n_support"]),
                )
            )
        if segment.get("mask_fraction") is not None:
            pieces.append("masked={0:.1%}".format(float(segment["mask_fraction"])))
        if segment.get("outside_fit_window_fraction") is not None:
            pieces.append(
                "outside_window={0:.1%}".format(
                    float(segment["outside_fit_window_fraction"])
                )
            )
        if segment.get("rejected_inside_fit_window_fraction") is not None:
            pieces.append(
                "rejected_inside={0:.1%}".format(
                    float(segment["rejected_inside_fit_window_fraction"])
                )
            )
        if segment.get("explicit_exclusion_count") is not None:
            pieces.append(
                "explicit rejects={0}".format(
                    int(segment["explicit_exclusion_count"])
                )
            )
        segment_lines.append("; ".join(pieces))
    if segment_lines:
        lines.append("  segments:")
        lines.extend("    - {0}".format(line) for line in segment_lines)
    return "\n".join(lines)


@dataclass(frozen=True)
class PhoenixFitDiagnostics(Mapping):
    """JSON-safe diagnostic summary for a deterministic PHOENIX fit."""

    payload: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self):
        return len(self.payload)

    def to_dict(self):
        return _jsonable(self.payload)


@dataclass(frozen=True)
class PhoenixFitResult(Mapping):
    """Structured PHOENIX fit result with dictionary compatibility."""

    summary: dict
    models: tuple = field(default_factory=tuple)
    continuum_coefficients: tuple = field(default_factory=tuple)
    used_masks: tuple = field(default_factory=tuple)
    excluded_masks: tuple = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)
    diagnostics: PhoenixFitDiagnostics | dict = field(default_factory=dict)
    quality_flags: tuple = field(default_factory=tuple)
    input_spectrum: object = None

    def __post_init__(self):
        diagnostics = self.diagnostics or self.summary.get("diagnostics", {})
        if not isinstance(diagnostics, PhoenixFitDiagnostics):
            diagnostics = PhoenixFitDiagnostics(dict(diagnostics))
        quality_flags = self.quality_flags or self.summary.get("quality_flags", ())
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "quality_flags", tuple(quality_flags))

    def _mapping_keys(self):
        keys = list(self.summary)
        for key in ("diagnostics", "quality_flags", "provenance"):
            if key not in self.summary:
                keys.append(key)
        return tuple(keys)

    def __getitem__(self, key):
        if key == "diagnostics" and key not in self.summary:
            return self.diagnostics.to_dict()
        if key == "quality_flags" and key not in self.summary:
            return list(self.quality_flags)
        if key == "provenance" and key not in self.summary:
            return self.provenance
        return self.summary[key]

    def __iter__(self):
        return iter(self._mapping_keys())

    def __len__(self):
        return len(self._mapping_keys())

    def to_dict(
        self,
        include_arrays=True,
        include_local_paths=False,
        plot_paths=None,
        relative_to=None,
    ):
        payload = dict(self.summary)
        payload["provenance"] = self.provenance
        payload["diagnostics"] = self.diagnostics.to_dict()
        payload["quality_flags"] = list(self.quality_flags)
        payload["quality_report"] = self.quality_report()
        normalized_plot_paths = _normalize_plot_paths(
            plot_paths,
            relative_to=relative_to,
            include_local_paths=include_local_paths,
        )
        if normalized_plot_paths is not None:
            payload["generated_files"] = {"plots": normalized_plot_paths}
        if include_arrays:
            payload.update(
                models=self.models,
                continuum_coefficients=self.continuum_coefficients,
                used_masks=self.used_masks,
                excluded_masks=self.excluded_masks,
            )
        payload = _jsonable(payload)
        if not include_local_paths:
            payload = _without_local_paths(payload)
        return payload

    def to_report_dict(
        self,
        include_arrays=False,
        include_local_paths=False,
        plot_paths=None,
        relative_to=None,
        report_context=None,
    ):
        """Return a versioned, provenance-rich fit report envelope.

        ``to_dict`` remains the compact direct result payload used by older
        scripts.  The report envelope is meant for reviewer-facing products,
        web/Django hand-off, and longer-lived archives where schema version,
        Spyctres version, path policy, and a small provenance summary should be
        explicit.
        """
        result_payload = self.to_dict(
            include_arrays=include_arrays,
            include_local_paths=include_local_paths,
            plot_paths=plot_paths,
            relative_to=relative_to,
        )
        report = {
            "schema_name": FIT_REPORT_SCHEMA_NAME,
            "schema_status": FIT_REPORT_SCHEMA_STATUS,
            "schema_version": FIT_REPORT_SCHEMA_VERSION,
            "report_type": FIT_REPORT_TYPE,
            "result_payload_schema_version": FIT_RESULT_PAYLOAD_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).replace(
                microsecond=0
            ).isoformat().replace("+00:00", "Z"),
            "spyctres": {
                "version": _spyctres_version(),
                "git_commit": _spyctres_git_commit(),
            },
            "path_policy": {
                "include_local_paths": bool(include_local_paths),
                "local_paths_sanitized": not bool(include_local_paths),
                "plot_paths_relative_to": (
                    None if relative_to is None else "provided_relative_base"
                ),
            },
            "provenance_summary": _fit_report_provenance_summary(result_payload),
            "result": result_payload,
        }
        if report_context is not None:
            report["report_context"] = _jsonable(report_context)
        return _jsonable(report)

    def quality_report(self):
        """Return a compact, JSON-safe summary of fit quality diagnostics."""
        return build_fit_quality_report(
            self.summary,
            diagnostics=self.diagnostics,
            quality_flags=self.quality_flags,
        )

    def quality_report_text(self):
        """Return a compact human-readable fit-quality summary."""
        return format_fit_quality_report(self)

    def compact_summary(self):
        """Return a notebook-friendly, JSON-safe fit summary.

        ``PhoenixFitResult.summary`` is the historic dictionary payload, so the
        callable notebook helper uses a distinct name rather than replacing
        that public attribute.
        """
        payload = self.to_dict(include_arrays=False)
        provenance_summary = _fit_report_provenance_summary(payload)
        fit_setup = payload.get("fit_setup") or {}
        readiness = fit_setup.get("readiness") if isinstance(fit_setup, Mapping) else {}
        if not isinstance(readiness, Mapping):
            readiness = {}
        exploratory = (
            fit_setup.get("exploratory_override")
            if isinstance(fit_setup, Mapping)
            else None
        )
        quality = self.quality_report()
        flags = list(quality.get("quality_flags") or [])
        flag_summary = quality.get("quality_flag_summary")
        if not isinstance(flag_summary, Mapping):
            flag_summary = summarize_quality_flags(flags)
        review_flags = [
            str(flag) for flag in flags if str(flag).lower() not in {"ok", "none"}
        ]
        uncertainty = quality.get("parameter_uncertainty")
        if isinstance(uncertainty, Mapping):
            uncertainty_status = uncertainty.get("status") or uncertainty.get("method")
        elif uncertainty is None:
            uncertainty_status = "not_reported"
        else:
            uncertainty_status = str(uncertainty)

        if exploratory:
            interpretation = "exploratory_review_only"
            sentence = (
                "Fit completed under an explicit exploratory override; use it "
                "for diagnosis and review, not as a final analysis result."
            )
        elif readiness.get("ready_for_intent") is False:
            interpretation = "computed_but_interpretation_blocked"
            sentence = (
                "Fit completed, but setup/readiness blockers must be reviewed "
                "before interpreting the parameters."
            )
        elif flag_summary.get("headline_status") == "blocked_for_interpretation":
            interpretation = "quality_blocked"
            sentence = (
                "Fit completed, but quality blockers prevent quantitative "
                "interpretation until they are resolved."
            )
        elif review_flags:
            interpretation = "review_quality_flags"
            sentence = (
                "Fit completed, but quality flags indicate the result needs "
                "human review before scientific use."
            )
        else:
            interpretation = "reviewed_first_pass"
            sentence = (
                "Fit completed as a reviewed first-pass estimate; inspect "
                "diagnostic plots before using it quantitatively."
            )

        return _jsonable(
            {
                "success": payload.get("success"),
                "teff": payload.get("teff"),
                "logg": payload.get("logg"),
                "feh": payload.get("feh"),
                "rv_kms": payload.get("rv_kms"),
                "chi2_red": quality.get("reduced_chi2"),
                "fit_intent": readiness.get("intent")
                or provenance_summary.get("readiness_intent"),
                "interpretation_status": interpretation,
                "interpretation": sentence,
                "uncertainty_status": uncertainty_status,
                "reader": provenance_summary.get("reader")
                or provenance_summary.get("instrument"),
                "resolution_source": provenance_summary.get("resolution_source"),
                "assumed_resolution_R": provenance_summary.get("assumed_resolution_R"),
                "fitted_pixels": quality.get("n_points"),
                "quality_flags": flags,
                "quality_flag_summary": flag_summary,
                "setup_hash": provenance_summary.get("fit_setup_hash"),
            }
        )

    def summary_text(self, *, include_hash=True, max_flags=None, include_notes=True):
        """Return a compact plain-language result summary for notebooks."""
        summary = self.compact_summary()
        quality_flags = list(summary.get("quality_flags") or ["none"])
        if max_flags is not None:
            max_flags = int(max_flags)
            if max_flags < 0:
                raise ValueError("max_flags must be >= 0 when supplied.")
            n_extra = max(0, len(quality_flags) - max_flags)
            quality_flags = quality_flags[:max_flags]
            if n_extra:
                quality_flags.append("...plus {0} more".format(n_extra))
        pieces = [
            "Spyctres PHOENIX fit",
            "  Teff={0} K, logg={1}, [Fe/H]={2}, RV={3} km/s".format(
                _format_optional(summary.get("teff")),
                _format_optional(summary.get("logg")),
                _format_optional(summary.get("feh")),
                _format_optional(summary.get("rv_kms")),
            ),
        ]
        if summary.get("chi2_red") is not None:
            pieces.append("  chi2_red={0}".format(_format_optional(summary["chi2_red"])))
        pieces.append(
            "  intent={0}; status={1}".format(
                summary.get("fit_intent"),
                summary.get("interpretation_status"),
            )
        )
        reader = summary.get("reader") or "unknown"
        resolution_source = summary.get("resolution_source") or "unknown"
        if reader != "unknown" or resolution_source != "unknown":
            pieces.append(
                "  reader={0}; resolution_source={1}".format(
                    reader,
                    resolution_source,
                )
            )
        if summary.get("assumed_resolution_R") is not None:
            pieces.append(
                "  assumed R={0}".format(
                    _format_optional(summary.get("assumed_resolution_R"))
                )
            )
        flag_summary = summary.get("quality_flag_summary") or {}
        if isinstance(flag_summary, Mapping) and flag_summary.get("headline_status"):
            counts = flag_summary.get("counts_by_severity") or {}
            pieces.append(
                "  quality_status={0}; blockers={1}; review={2}; advisory={3}".format(
                    flag_summary.get("headline_status"),
                    int(counts.get("blocker", 0) or 0),
                    int(counts.get("review", 0) or 0),
                    int(counts.get("advisory", 0) or 0),
                )
            )
        pieces.append(
            "  fitted_pixels={0}; uncertainty={1}; flags={2}".format(
                summary.get("fitted_pixels"),
                summary.get("uncertainty_status"),
                ", ".join(quality_flags),
            )
        )
        if include_hash and summary.get("setup_hash"):
            pieces.append("  setup_hash={0}".format(str(summary["setup_hash"])[:12]))
        pieces.append("  {0}".format(summary.get("interpretation")))
        if isinstance(flag_summary, Mapping):
            top_actions = list(flag_summary.get("top_actions") or [])
            if top_actions:
                item = top_actions[0]
                pieces.append(
                    "  top action: {0} [{1}] — {2}".format(
                        item.get("flag"),
                        item.get("severity"),
                        item.get("action"),
                    )
                )
        if include_notes and "high_chi2" in (summary.get("quality_flags") or []):
            pieces.append(
                "  note: high chi2 can reflect small formal errors plus model, "
                "continuum, LSF, abundance, activity, or missing-physics systematics."
            )
        return "\n".join(pieces)

    def __repr__(self):
        summary = self.compact_summary()
        return (
            "PhoenixFitResult(success={success!r}, teff={teff}, logg={logg}, "
            "feh={feh}, rv_kms={rv}, chi2_red={chi2}, status={status!r})"
        ).format(
            success=summary.get("success"),
            teff=_format_optional(summary.get("teff")),
            logg=_format_optional(summary.get("logg")),
            feh=_format_optional(summary.get("feh")),
            rv=_format_optional(summary.get("rv_kms")),
            chi2=_format_optional(summary.get("chi2_red")),
            status=summary.get("interpretation_status"),
        )

    def to_json(self, **kwargs):
        kwargs.setdefault("allow_nan", False)
        to_dict_keys = {
            "include_arrays",
            "include_local_paths",
            "plot_paths",
            "relative_to",
        }
        to_dict_kwargs = {
            key: kwargs.pop(key) for key in list(kwargs) if key in to_dict_keys
        }
        return json.dumps(self.to_dict(**to_dict_kwargs), **kwargs)

    def to_report_json(self, **kwargs):
        """Serialize the versioned fit-report envelope as JSON."""
        kwargs.setdefault("allow_nan", False)
        to_report_keys = {
            "include_arrays",
            "include_local_paths",
            "plot_paths",
            "relative_to",
            "report_context",
        }
        to_report_kwargs = {
            key: kwargs.pop(key) for key in list(kwargs) if key in to_report_keys
        }
        return json.dumps(self.to_report_dict(**to_report_kwargs), **kwargs)

    def save_json(
        self,
        path,
        include_arrays=False,
        include_local_paths=False,
        plot_paths=None,
        relative_to=None,
        **kwargs,
    ):
        """Write a JSON representation suitable for downstream tools."""
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("allow_nan", False)
        if relative_to is None:
            relative_to = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        atomic_write_json(
            path,
            self.to_dict(
                include_arrays=include_arrays,
                include_local_paths=include_local_paths,
                plot_paths=plot_paths,
                relative_to=relative_to,
            ),
            **kwargs,
        )

    def save_report_json(
        self,
        path,
        include_arrays=False,
        include_local_paths=False,
        plot_paths=None,
        relative_to=None,
        report_context=None,
        **kwargs,
    ):
        """Write the versioned fit-report envelope for archival/review use."""
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("allow_nan", False)
        if relative_to is None:
            relative_to = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        atomic_write_json(
            path,
            self.to_report_dict(
                include_arrays=include_arrays,
                include_local_paths=include_local_paths,
                plot_paths=plot_paths,
                relative_to=relative_to,
                report_context=report_context,
            ),
            **kwargs,
        )
