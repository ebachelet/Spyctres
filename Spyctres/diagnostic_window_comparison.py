"""Bounded diagnostic-window comparison planning and optional fit execution.

This module keeps the diagnostic-window idea deliberately lightweight.  It
builds a small, auditable set of wavelength-window combinations and can run
those combinations through a caller-supplied fitting function.  The comparison
is meant to expose parameter stability and feature sensitivity; it is not a
blind model-selection engine and does not rank solutions by raw chi-square.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .diagnostic_windows import (
    build_diagnostic_window_combinations,
    select_diagnostic_windows,
)
from .results import PhoenixFitResult


DEFAULT_MAX_COMPARISONS = 8


def build_diagnostic_window_comparison_plan(
    spectrum,
    *,
    selection=None,
    combinations=None,
    roles=None,
    initial_teff=None,
    rv_kms=None,
    rv_padding_kms=0.0,
    max_windows=8,
    max_single_windows=5,
    max_comparisons=DEFAULT_MAX_COMPARISONS,
    include_warn_windows=True,
    include_stress_windows=False,
):
    """Return a bounded plan for diagnostic-window comparison fits.

    Parameters
    ----------
    spectrum : SpectrumSegment, SpectrumCollection, or compatible input
        Spectrum used only for cheap window selection if ``selection`` is not
        supplied.
    selection, combinations : mapping, optional
        Precomputed outputs from :func:`select_diagnostic_windows` and
        :func:`build_diagnostic_window_combinations`.
    roles, initial_teff, rv_kms, rv_padding_kms
        Forwarded to ``select_diagnostic_windows`` when ``selection`` is not
        supplied.
    max_windows, max_single_windows, max_comparisons : int
        Bounds that prevent accidental all-subset searches.
    include_warn_windows : bool, optional
        If False, planned fit combinations containing ``default_fit_policy =
        "warn"`` windows are skipped.  The default keeps such windows because
        many useful diagnostics, such as Hβ with possible DIB contamination,
        are still valuable when explicitly labelled.
    include_stress_windows : bool, optional
        If False, combinations containing ``model_support = "stress_only"``
        windows are skipped by default.  This keeps He/NLTE-sensitive stress
        checks from entering ordinary comparison fits unless the user opts in.

    Returns
    -------
    dict
        JSON-serializable plan containing selected windows, planned
        combinations, skipped combinations, operational fit regions, and the
        comparison policy.
    """
    max_comparisons = int(max_comparisons)
    if max_comparisons < 1:
        raise ValueError("max_comparisons must be >= 1.")

    if selection is None:
        selection = select_diagnostic_windows(
            spectrum,
            roles=roles,
            initial_teff=initial_teff,
            rv_kms=rv_kms,
            rv_padding_kms=rv_padding_kms,
        )
    else:
        selection = dict(selection)

    if combinations is None:
        combinations = build_diagnostic_window_combinations(
            selection,
            max_windows=max_windows,
            max_single_windows=max_single_windows,
        )
    else:
        combinations = dict(combinations)

    selected_by_id = {record["id"]: dict(record) for record in selection["selected"]}
    planned = []
    skipped = []
    for combo_index, combo in enumerate(combinations.get("combinations", ())):
        combo = dict(combo)
        records = [
            selected_by_id[window_id]
            for window_id in combo.get("window_ids", ())
            if window_id in selected_by_id
        ]
        skip_reasons = _combination_skip_reasons(
            records,
            include_warn_windows=include_warn_windows,
            include_stress_windows=include_stress_windows,
        )
        comparison = _comparison_record(
            combo,
            records,
            all_selected=selection["selected"],
            candidate_index=combo_index,
        )
        if skip_reasons:
            comparison["skip_reasons"] = skip_reasons
            skipped.append(comparison)
            continue
        if len(planned) >= max_comparisons:
            comparison["skip_reasons"] = ["max_comparisons_limit"]
            skipped.append(comparison)
            continue
        comparison["comparison_index"] = len(planned)
        planned.append(comparison)

    return _json_native(
        {
            "schema_version": 1,
            "operation": "build_diagnostic_window_comparison_plan",
            "created_utc": _utc_now(),
            "status": "planned" if planned else "no_comparisons_planned",
            "selection": selection,
            "combination_candidates": combinations,
            "comparison_policy": {
                "max_comparisons": max_comparisons,
                "max_windows_per_comparison": int(max_windows),
                "max_single_window_comparisons": int(max_single_windows),
                "include_warn_windows": bool(include_warn_windows),
                "include_stress_windows": bool(include_stress_windows),
                "bounded_search": True,
                "not_raw_chi2_ranked": True,
                "interpretation": (
                    "Use this plan to check feature sensitivity and parameter "
                    "stability across a small number of scientifically labelled "
                    "window sets. Do not choose the 'best' physics from raw "
                    "chi-square alone; compare quality flags, held-out windows, "
                    "and common-evaluation residuals."
                ),
            },
            "planned_comparisons": planned,
            "skipped_comparisons": skipped,
        }
    )


def run_diagnostic_window_comparison(
    spectrum,
    *,
    run_fits=False,
    fit_callable=None,
    fit_call_kwargs=None,
    base_fit_kwargs=None,
    progress_callback=None,
    **plan_kwargs,
):
    """Build a diagnostic-window comparison plan and optionally run the fits.

    The default ``run_fits=False`` returns a dry-run plan only.  Set
    ``run_fits=True`` to execute the bounded comparisons.  When ``fit_callable``
    is omitted, Spyctres uses :func:`Spyctres.fit_stellar_spectrum`; callers
    normally pass PHOENIX configuration through ``fit_call_kwargs`` and generic
    fit options through ``base_fit_kwargs``.
    """
    plan = build_diagnostic_window_comparison_plan(spectrum, **plan_kwargs)
    payload = {
        "schema_version": 1,
        "operation": "run_diagnostic_window_comparison",
        "created_utc": _utc_now(),
        "status": "planned_no_fits_run",
        "selection": plan["selection"],
        "comparison_policy": plan["comparison_policy"],
        "planned_comparisons": plan["planned_comparisons"],
        "skipped_comparisons": plan["skipped_comparisons"],
        "run_policy": {
            "run_fits": bool(run_fits),
            "fit_callable": _callable_name(fit_callable),
            "regions_overridden_per_comparison": True,
            "held_out_evaluation": (
                "metadata_only_unless_common_evaluation_residuals_are_added"
            ),
        },
        "fit_records": [],
    }

    if not run_fits:
        payload["fit_records"] = [
            _planned_fit_record(comparison, fit_status="planned_not_run")
            for comparison in plan["planned_comparisons"]
        ]
        return _json_native(payload)

    if fit_callable is None:
        from .api import fit_stellar_spectrum

        fit_callable = fit_stellar_spectrum
        payload["run_policy"]["fit_callable"] = "Spyctres.fit_stellar_spectrum"

    fit_call_kwargs = {} if fit_call_kwargs is None else dict(fit_call_kwargs)
    base_fit_kwargs = {} if base_fit_kwargs is None else dict(base_fit_kwargs)
    records = []
    for comparison in plan["planned_comparisons"]:
        _emit_progress(
            progress_callback,
            comparison,
            phase="start",
            message="Running diagnostic-window comparison {0}/{1}: {2}".format(
                int(comparison["comparison_index"]) + 1,
                len(plan["planned_comparisons"]),
                comparison["label"],
            ),
        )
        record = _planned_fit_record(comparison, fit_status="started")
        call_kwargs = dict(fit_call_kwargs)
        fit_kwargs = dict(base_fit_kwargs)
        fit_kwargs["regions"] = _regions_for_fit(comparison)
        call_kwargs.update(fit_kwargs)
        try:
            result = fit_callable(spectrum, **call_kwargs)
            result_payload = _as_result_payload(result)
            record["fit_status"] = (
                "ok" if bool(result_payload.get("success", True)) else "fit_failed"
            )
            record["result_summary"] = _result_summary(result_payload)
            record["quality_flags"] = list(result_payload.get("quality_flags", ()))
            record["result"] = result_payload
        except Exception as exc:  # pragma: no cover - exercised by tests indirectly
            record["fit_status"] = "error"
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        records.append(record)
        _emit_progress(
            progress_callback,
            comparison,
            phase="end",
            message="Finished diagnostic-window comparison {0}/{1}: {2}".format(
                int(comparison["comparison_index"]) + 1,
                len(plan["planned_comparisons"]),
                record["fit_status"],
            ),
            fit_status=record["fit_status"],
        )

    payload["fit_records"] = records
    if any(record["fit_status"] == "error" for record in records):
        payload["status"] = "completed_with_errors"
    else:
        payload["status"] = "fits_completed"
    payload["run_policy"]["fit_callable"] = _callable_name(fit_callable)
    return _json_native(payload)


def write_diagnostic_window_comparison_json(path, payload):
    """Write comparison payload as atomic JSON, creating the parent directory."""
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


def write_diagnostic_window_comparison_csv(path, payload):
    """Write a compact CSV summary of planned or completed comparisons."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("fit_records") or [
        _planned_fit_record(item, fit_status="planned_not_run")
        for item in payload.get("planned_comparisons", ())
    ]
    columns = [
        "comparison_index",
        "comparison_id",
        "kind",
        "n_windows",
        "window_ids",
        "feature_families",
        "regions_A",
        "estimated_usable_pixels",
        "held_out_window_ids",
        "fit_status",
        "success",
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
        for row in rows:
            comparison = row.get("comparison", row)
            summary = row.get("result_summary") or {}
            writer.writerow(
                {
                    "comparison_index": comparison.get("comparison_index"),
                    "comparison_id": comparison.get("id"),
                    "kind": comparison.get("kind"),
                    "n_windows": comparison.get("n_windows"),
                    "window_ids": ";".join(comparison.get("window_ids", ())),
                    "feature_families": ";".join(
                        comparison.get("feature_families", ())
                    ),
                    "regions_A": _format_regions(comparison.get("regions_for_fit_A", ())),
                    "estimated_usable_pixels": comparison.get(
                        "estimated_usable_pixels"
                    ),
                    "held_out_window_ids": ";".join(
                        comparison.get("held_out_window_ids", ())
                    ),
                    "fit_status": row.get("fit_status"),
                    "success": summary.get("success"),
                    "teff": summary.get("teff"),
                    "logg": summary.get("logg"),
                    "feh": summary.get("feh"),
                    "rv_kms": summary.get("rv_kms"),
                    "chi2_red": summary.get("chi2_red"),
                    "quality_flags": ";".join(row.get("quality_flags", ())),
                }
            )


def plot_diagnostic_window_comparison(payload, savepath=None):
    """Plot a compact comparison summary and optionally save it.

    The figure deliberately shows fit outcomes, stability, and window
    membership rather than presenting a raw chi-square ranking as a scientific
    decision rule.
    """
    import matplotlib.pyplot as plt

    records = payload.get("fit_records") or [
        _planned_fit_record(item, fit_status="planned_not_run")
        for item in payload.get("planned_comparisons", ())
    ]
    if not records:
        raise ValueError("No diagnostic-window comparison records to plot.")

    labels = [
        "{0}: {1}".format(
            int(record.get("comparison", record).get("comparison_index", index)),
            record.get("comparison", record).get("kind", "comparison"),
        )
        for index, record in enumerate(records)
    ]
    x = np.arange(len(records), dtype=float)
    nfit = np.asarray(
        [
            float(
                record.get("comparison", record).get("estimated_usable_pixels", np.nan)
            )
            for record in records
        ],
        dtype=float,
    )
    teff = _summary_array(records, "teff")
    logg = _summary_array(records, "logg")
    feh = _summary_array(records, "feh")
    rv = _summary_array(records, "rv_kms")
    chi2 = _summary_array(records, "chi2_red")
    has_fit_values = any(
        np.any(np.isfinite(values)) for values in (teff, logg, feh, rv, chi2)
    )

    all_window_ids = _ordered_window_ids(records)
    nrows = 3 if has_fit_values else 2
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(max(9.0, 1.25 * len(records)), 2.8 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    if np.any(np.isfinite(chi2)):
        axes[0].plot(x, chi2, marker="o", color="tab:red")
        axes[0].set_ylabel("χ²ν")
    else:
        axes[0].bar(x, nfit, color="0.55")
        axes[0].set_ylabel("usable pixels")
    axes[0].set_title("Diagnostic-window comparison")
    axes[0].grid(alpha=0.25)

    if has_fit_values:
        if np.any(np.isfinite(teff)):
            axes[1].plot(x, teff, marker="o", label="Teff [K]")
            axes[1].set_ylabel("Teff [K]")
        ax_secondary = axes[1].twinx()
        if np.any(np.isfinite(logg)):
            ax_secondary.plot(x, logg, marker="s", color="tab:green", label="logg")
        if np.any(np.isfinite(feh)):
            ax_secondary.plot(x, feh, marker="^", color="tab:purple", label="[Fe/H]")
        if np.any(np.isfinite(rv)):
            ax_secondary.plot(x, rv, marker="d", color="tab:orange", label="RV")
        ax_secondary.set_ylabel("logg / [Fe/H] / RV")
        axes[1].grid(alpha=0.25)
        matrix_ax = axes[2]
    else:
        matrix_ax = axes[1]

    matrix = np.zeros((len(all_window_ids), len(records)), dtype=float)
    for col, record in enumerate(records):
        window_ids = set(record.get("comparison", record).get("window_ids", ()))
        for row, window_id in enumerate(all_window_ids):
            matrix[row, col] = 1.0 if window_id in window_ids else 0.0
    matrix_ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues")
    matrix_ax.set_yticks(np.arange(len(all_window_ids)))
    matrix_ax.set_yticklabels(all_window_ids, fontsize=8)
    matrix_ax.set_xticks(x)
    matrix_ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    matrix_ax.set_ylabel("windows")
    matrix_ax.set_xlabel("comparison")

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=160)
    return fig, axes


def _comparison_record(combo, records, *, all_selected, candidate_index):
    held_out = [
        record
        for record in all_selected
        if record.get("id") not in set(combo.get("window_ids", ()))
    ]
    fit_policy_warnings = _fit_policy_warnings(records)
    return {
        "candidate_index": int(candidate_index),
        "id": combo.get("id"),
        "kind": combo.get("kind"),
        "label": combo.get("label"),
        "window_ids": list(combo.get("window_ids", ())),
        "window_labels": list(combo.get("window_labels", ())),
        "n_windows": int(combo.get("n_windows", len(records))),
        "roles": sorted({role for record in records for role in record.get("roles", ())}),
        "feature_families": sorted(
            {
                family
                for record in records
                for family in record.get("feature_family", ())
            }
        ),
        "canonical_regions_vacuum_A": [
            list(record.get("region_vacuum_A", record.get("region_A", ())))
            for record in records
        ],
        "regions_for_fit_A": _operational_regions_for_records(records),
        "coordinate_policy": (
            "regions_for_fit_A are operational regions in each segment's "
            "declared wavelength medium/frame from the diagnostic-window "
            "selection step; canonical_regions_vacuum_A remain the catalog "
            "stellar-rest vacuum regions."
        ),
        "default_fit_policies": sorted(
            {record.get("default_fit_policy", "warn") for record in records}
        ),
        "model_support": sorted(
            {record.get("model_support", "uncertain") for record in records}
        ),
        "risk_tags": sorted({tag for record in records for tag in record.get("risk_tags", ())}),
        "fit_policy_warnings": fit_policy_warnings,
        "estimated_usable_pixels": int(
            sum(int(record.get("n_usable_pixels", 0)) for record in records)
        ),
        "score_sum": float(sum(float(record.get("score", 0.0)) for record in records)),
        "held_out_window_ids": [record.get("id") for record in held_out],
        "held_out_feature_families": sorted(
            {
                family
                for record in held_out
                for family in record.get("feature_family", ())
            }
        ),
        "held_out_evaluation": {
            "status": "not_computed",
            "reason": (
                "This initial runner records held-out windows as metadata. "
                "Common-evaluation residual checks are planned as a later "
                "publication-quality layer."
            ),
        },
    }


def _combination_skip_reasons(
    records,
    *,
    include_warn_windows,
    include_stress_windows,
):
    if not records:
        return ["no_selected_window_records"]
    reasons = []
    if any(record.get("default_fit_policy") == "exclude" for record in records):
        reasons.append("default_fit_policy_exclude")
    if any(record.get("model_support") == "unsupported" for record in records):
        reasons.append("model_support_unsupported")
    if not include_warn_windows and any(
        record.get("default_fit_policy") == "warn" for record in records
    ):
        reasons.append("warn_windows_disabled")
    if not include_stress_windows and any(
        record.get("model_support") == "stress_only" for record in records
    ):
        reasons.append("stress_windows_disabled")
    return reasons


def _operational_regions_for_records(records):
    regions = []
    for record in records:
        added = False
        for contribution in record.get("segment_contributions", ()):
            if int(contribution.get("n_usable_pixels", 0)) <= 0:
                continue
            region = contribution.get("operational_region_A") or record.get("region_A")
            if _valid_region(region):
                regions.append((float(region[0]), float(region[1])))
                added = True
        if not added and _valid_region(record.get("region_A")):
            regions.append((float(record["region_A"][0]), float(record["region_A"][1])))
    return [[float(lo), float(hi)] for lo, hi in _merge_regions(regions)]


def _merge_regions(regions, *, gap_tolerance_A=0.0):
    normalized = []
    for region in regions:
        if not _valid_region(region):
            continue
        lo, hi = sorted((float(region[0]), float(region[1])))
        normalized.append((lo, hi))
    if not normalized:
        return []
    normalized.sort()
    merged = [normalized[0]]
    for lo, hi in normalized[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + float(gap_tolerance_A):
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _valid_region(region):
    if region is None or len(region) != 2:
        return False
    try:
        lo = float(region[0])
        hi = float(region[1])
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(lo) and np.isfinite(hi) and hi != lo)


def _fit_policy_warnings(records):
    warnings = []
    for record in records:
        if record.get("default_fit_policy") == "warn":
            warnings.append(
                {
                    "window_id": record.get("id"),
                    "reason": "default_fit_policy_warn",
                    "risk_tags": list(record.get("risk_tags", ())),
                }
            )
        if record.get("model_support") == "stress_only":
            warnings.append(
                {
                    "window_id": record.get("id"),
                    "reason": "model_support_stress_only",
                    "risk_tags": list(record.get("risk_tags", ())),
                }
            )
    return warnings


def _planned_fit_record(comparison, *, fit_status):
    return {
        "comparison": dict(comparison),
        "fit_status": fit_status,
        "result_summary": None,
        "quality_flags": [],
        "result": None,
    }


def _regions_for_fit(comparison):
    return [
        (float(region[0]), float(region[1]))
        for region in comparison.get("regions_for_fit_A", ())
    ]


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


def _result_summary(payload):
    return {
        "success": payload.get("success"),
        "teff": payload.get("teff"),
        "logg": payload.get("logg"),
        "feh": payload.get("feh"),
        "rv_kms": payload.get("rv_kms"),
        "chi2_red": payload.get("chi2_red"),
    }


def _summary_array(records, key):
    values = []
    for record in records:
        summary = record.get("result_summary") or {}
        try:
            value = float(summary.get(key))
        except (TypeError, ValueError):
            value = np.nan
        if not np.isfinite(value):
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def _ordered_window_ids(records):
    window_ids = []
    for record in records:
        for window_id in record.get("comparison", record).get("window_ids", ()):
            if window_id not in window_ids:
                window_ids.append(window_id)
    return window_ids


def _format_regions(regions):
    return ";".join(
        "{0:.2f}-{1:.2f}".format(float(region[0]), float(region[1]))
        for region in regions
    )


def _callable_name(func):
    if func is None:
        return "Spyctres.fit_stellar_spectrum"
    module = getattr(func, "__module__", None)
    name = getattr(func, "__name__", None)
    if module and name:
        return "{0}.{1}".format(module, name)
    return str(func)


def _emit_progress(callback, comparison, **payload):
    if callback is None:
        return
    event = {
        "operation": "diagnostic_window_comparison",
        "comparison_id": comparison.get("id"),
        "comparison_index": comparison.get("comparison_index"),
        "time_utc": _utc_now(),
    }
    event.update(payload)
    callback(_json_native(event))


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_native(value):
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
