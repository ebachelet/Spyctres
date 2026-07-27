"""Bounded diagnostic-window comparison planning and optional fit execution.

This module keeps the diagnostic-window idea deliberately lightweight.  It
builds a small, auditable set of wavelength-window combinations and can run
those combinations through a caller-supplied fitting function.  The comparison
is meant to expose parameter stability and feature sensitivity; it is not a
blind model-selection engine and does not rank solutions by raw chi-square.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time

import numpy as np

from ._serialization import (
    atomic_write_csv_rows,
    atomic_write_json,
    json_safe as _json_native,
)
from .diagnostic_windows import (
    build_diagnostic_window_combinations,
    select_diagnostic_windows,
)
from .io import SpectrumCollection, SpectrumSegment, coerce_spectrum
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
    evaluate_heldout=True,
    evaluate_common=True,
    holdout_min_pixels=3,
    common_min_pixels=None,
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
    common_min_pixels = (
        int(holdout_min_pixels) if common_min_pixels is None else int(common_min_pixels)
    )
    common_records = _common_evaluation_records(plan)
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
            "evaluate_heldout": bool(evaluate_heldout),
            "evaluate_common": bool(evaluate_common),
            "holdout_min_pixels": int(holdout_min_pixels),
            "common_min_pixels": int(common_min_pixels),
            "fit_callable": _callable_name(fit_callable),
            "regions_overridden_per_comparison": True,
            "held_out_evaluation": (
                "computed_on_valid_unfitted_pixels_when_reconstructed_models_exist"
            ),
            "common_evaluation": (
                "computed_on_the_same_union_of_planned_comparison_windows_when_"
                "reconstructed_models_exist"
            ),
        },
        "common_evaluation_definition": _common_evaluation_definition(
            common_records,
            min_pixels=common_min_pixels,
        ),
        "common_evaluation_summary": None,
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
            if evaluate_heldout:
                record["held_out_evaluation"] = evaluate_diagnostic_window_holdout(
                    spectrum,
                    result,
                    comparison,
                    selected_windows=plan["selection"]["selected"],
                    min_pixels=holdout_min_pixels,
                )
            else:
                record["held_out_evaluation"] = {
                    "status": "skipped",
                    "reason": "evaluate_heldout_false",
                }
            if evaluate_common:
                record["common_evaluation"] = (
                    evaluate_diagnostic_window_common_evaluation(
                        spectrum,
                        result,
                        common_windows=common_records,
                        min_pixels=common_min_pixels,
                    )
                )
            else:
                record["common_evaluation"] = {
                    "status": "skipped",
                    "reason": "evaluate_common_false",
                }
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
    payload["common_evaluation_summary"] = _summarize_common_evaluations(records)
    return _json_native(payload)


def evaluate_diagnostic_window_holdout(
    spectrum,
    result,
    comparison,
    *,
    selected_windows,
    min_pixels=3,
):
    """Score selected-but-held-out windows using reconstructed model arrays.

    The metric is intentionally modest: it evaluates valid pixels inside
    windows that were selected by the diagnostic catalog but not included in
    the current fit combination.  Pixels already used by the fit or explicitly
    excluded remain out of the evaluation.  This is a predictive sanity check,
    not a replacement for final publication validation.
    """
    held_out_ids = list(comparison.get("held_out_window_ids", ()))
    if not held_out_ids:
        return {
            "status": "no_held_out_windows",
            "method": "valid_unfitted_window_residuals",
            "windows": [],
            "overall": None,
        }

    segments = _as_segments(spectrum)
    models, used_masks, excluded_masks = _model_mask_arrays_from_result(
        result,
        n_segments=len(segments),
    )
    if models is None:
        return {
            "status": "skipped_no_reconstructed_model",
            "reason": (
                "Held-out residual scoring requires reconstructed model arrays. "
                "Run comparison fits with reconstruct=True."
            ),
            "method": "valid_unfitted_window_residuals",
            "windows": [],
            "overall": None,
        }

    selected_by_id = {record["id"]: dict(record) for record in selected_windows}
    min_pixels = int(min_pixels)
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1.")

    rows = []
    for window_id in held_out_ids:
        record = selected_by_id.get(window_id)
        if record is None:
            rows.append(
                {
                    "window_id": window_id,
                    "status": "missing_selection_record",
                    "n_pixels": 0,
                }
            )
            continue
        rows.append(
            _evaluate_one_heldout_window(
                segments,
                models,
                used_masks,
                excluded_masks,
                record,
                min_pixels=min_pixels,
            )
        )

    evaluated = [row for row in rows if row.get("status") == "ok"]
    overall = _summarize_heldout_rows(evaluated)
    if evaluated:
        status = "ok"
    elif rows:
        status = "no_evaluable_held_out_pixels"
    else:
        status = "no_held_out_windows"
    return _json_native(
        {
            "status": status,
            "method": "valid_unfitted_window_residuals",
            "coordinate_policy": (
                "Uses each diagnostic window's operational region from the "
                "selection step. The model is the reconstructed continuum-"
                "adjusted model from the fit; the continuum is not refit on "
                "held-out windows."
            ),
            "min_pixels": int(min_pixels),
            "n_held_out_windows": int(len(held_out_ids)),
            "n_evaluated_windows": int(len(evaluated)),
            "overall": overall,
            "windows": rows,
        }
    )


def evaluate_diagnostic_window_common_evaluation(
    spectrum,
    result,
    *,
    common_windows,
    min_pixels=3,
):
    """Score a fit over the same diagnostic windows used for every comparison.

    Unlike :func:`evaluate_diagnostic_window_holdout`, this metric deliberately
    does *not* remove pixels that were used by the fit.  The goal is a
    common-pixel residual summary for comparing different window combinations,
    not an independent held-out validation.  Explicitly excluded pixels and
    invalid spectrum masks are still respected.
    """
    common_windows = [dict(record) for record in common_windows or ()]
    if not common_windows:
        return {
            "status": "no_common_windows",
            "method": "common_valid_window_residuals",
            "windows": [],
            "overall": None,
        }

    segments = _as_segments(spectrum)
    models, used_masks, excluded_masks = _model_mask_arrays_from_result(
        result,
        n_segments=len(segments),
    )
    if models is None:
        return {
            "status": "skipped_no_reconstructed_model",
            "reason": (
                "Common-evaluation residual scoring requires reconstructed "
                "model arrays. Run comparison fits with reconstruct=True."
            ),
            "method": "common_valid_window_residuals",
            "windows": [],
            "overall": None,
        }

    min_pixels = int(min_pixels)
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1.")

    rows = [
        _evaluate_one_window_residuals(
            segments,
            models,
            used_masks,
            excluded_masks,
            record,
            min_pixels=min_pixels,
            exclude_used_pixels=False,
            quality_flag_prefix="common",
        )
        for record in common_windows
    ]
    evaluated = [row for row in rows if row.get("status") == "ok"]
    overall = _summarize_common_rows(evaluated)
    if evaluated:
        status = "ok"
    elif rows:
        status = "no_evaluable_common_pixels"
    else:
        status = "no_common_windows"
    return _json_native(
        {
            "status": status,
            "method": "common_valid_window_residuals",
            "coordinate_policy": (
                "Uses a fixed union of planned comparison windows for every "
                "fit record. Valid spectrum pixels and explicit exclusion masks "
                "are respected, but fit-used pixels are not removed; this keeps "
                "the evaluated pixels common across comparisons."
            ),
            "min_pixels": int(min_pixels),
            "n_common_windows": int(len(common_windows)),
            "n_evaluated_windows": int(len(evaluated)),
            "common_window_ids": [record.get("id") for record in common_windows],
            "overall": overall,
            "windows": rows,
        }
    )


def write_diagnostic_window_comparison_json(path, payload):
    """Write comparison payload as atomic JSON, creating the parent directory."""
    atomic_write_json(path, payload, sort_keys=True)


def write_diagnostic_window_comparison_csv(path, payload):
    """Write a compact CSV summary of planned or completed comparisons."""
    if path is None:
        return
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
        "heldout_status",
        "heldout_n_evaluated_windows",
        "heldout_n_pixels",
        "heldout_mean_chi2_red_proxy",
        "heldout_median_abs_sigma",
        "common_status",
        "common_n_evaluated_windows",
        "common_n_pixels",
        "common_mean_chi2_red_proxy",
        "common_median_abs_sigma",
        "common_max_rms_fraction",
        "fit_status",
        "success",
        "teff",
        "logg",
        "feh",
        "rv_kms",
        "chi2_red",
        "quality_flags",
    ]
    csv_rows = []
    for row in rows:
        comparison = row.get("comparison", row)
        summary = row.get("result_summary") or {}
        heldout = row.get("held_out_evaluation") or {}
        heldout_overall = heldout.get("overall") or {}
        common = row.get("common_evaluation") or {}
        common_overall = common.get("overall") or {}
        csv_rows.append(
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
                "estimated_usable_pixels": comparison.get("estimated_usable_pixels"),
                "held_out_window_ids": ";".join(
                    comparison.get("held_out_window_ids", ())
                ),
                "heldout_status": heldout.get("status"),
                "heldout_n_evaluated_windows": heldout.get("n_evaluated_windows"),
                "heldout_n_pixels": heldout_overall.get("n_pixels"),
                "heldout_mean_chi2_red_proxy": heldout_overall.get(
                    "mean_chi2_red_proxy"
                ),
                "heldout_median_abs_sigma": heldout_overall.get(
                    "median_abs_sigma"
                ),
                "common_status": common.get("status"),
                "common_n_evaluated_windows": common.get("n_evaluated_windows"),
                "common_n_pixels": common_overall.get("n_pixels"),
                "common_mean_chi2_red_proxy": common_overall.get(
                    "mean_chi2_red_proxy"
                ),
                "common_median_abs_sigma": common_overall.get("median_abs_sigma"),
                "common_max_rms_fraction": common_overall.get("max_rms_fraction"),
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
    atomic_write_csv_rows(path, columns, csv_rows)


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
    heldout_chi2 = _heldout_overall_array(records, "mean_chi2_red_proxy")
    common_chi2 = _common_overall_array(records, "mean_chi2_red_proxy")
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

    if (
        np.any(np.isfinite(chi2))
        or np.any(np.isfinite(heldout_chi2))
        or np.any(np.isfinite(common_chi2))
    ):
        if np.any(np.isfinite(chi2)):
            axes[0].plot(x, chi2, marker="o", color="tab:red", label="fit χ²ν")
        if np.any(np.isfinite(heldout_chi2)):
            axes[0].plot(
                x,
                heldout_chi2,
                marker="s",
                ls="--",
                color="tab:purple",
                label="held-out χ² proxy",
            )
        if np.any(np.isfinite(common_chi2)):
            axes[0].plot(
                x,
                common_chi2,
                marker="^",
                ls=":",
                color="tab:blue",
                label="common-window χ² proxy",
            )
        axes[0].legend(frameon=False, fontsize=8)
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
                "Residual checks require a completed fit with reconstructed "
                "model arrays. Use run_fits=True with reconstruct=True to "
                "compute held-out and common-window diagnostics."
            ),
        },
    }


def _evaluate_one_heldout_window(
    segments,
    models,
    used_masks,
    excluded_masks,
    record,
    *,
    min_pixels,
):
    return _evaluate_one_window_residuals(
        segments,
        models,
        used_masks,
        excluded_masks,
        record,
        min_pixels=min_pixels,
        exclude_used_pixels=True,
        quality_flag_prefix="heldout",
    )


def _evaluate_one_window_residuals(
    segments,
    models,
    used_masks,
    excluded_masks,
    record,
    *,
    min_pixels,
    exclude_used_pixels,
    quality_flag_prefix,
):
    chunks = []
    segment_rows = []
    for contribution in record.get("segment_contributions", ()):
        index = int(contribution.get("segment_index", -1))
        if index < 0 or index >= len(segments):
            continue
        region = contribution.get("operational_region_A")
        if not _valid_region(region):
            continue
        seg = segments[index]
        wave = np.asarray(seg.wave, dtype=float)
        flux = np.asarray(seg.flux, dtype=float)
        model = np.asarray(models[index], dtype=float)
        if wave.shape != flux.shape or wave.shape != model.shape:
            continue
        valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
        mask = np.asarray(getattr(seg, "mask", np.ones(wave.size, dtype=bool)), dtype=bool)
        if mask.shape == wave.shape:
            valid &= mask
        used = used_masks[index]
        if used is not None and exclude_used_pixels:
            valid &= ~used
        excluded = excluded_masks[index]
        if excluded is not None:
            valid &= ~excluded
        lo, hi = sorted((float(region[0]), float(region[1])))
        inside = valid & (wave >= lo) & (wave <= hi)
        n_pixels = int(np.count_nonzero(inside))
        segment_rows.append(
            {
                "segment": getattr(seg, "name", None),
                "segment_index": int(index),
                "region_A": [lo, hi],
                "n_pixels": n_pixels,
            }
        )
        if n_pixels <= 0:
            continue
        err = None if getattr(seg, "err", None) is None else np.asarray(seg.err, dtype=float)
        if err is not None and err.shape == wave.shape:
            err_i = err[inside]
        else:
            err_i = None
        chunks.append(
            {
                "flux": flux[inside],
                "model": model[inside],
                "err": err_i,
            }
        )

    n_total = int(sum(len(chunk["flux"]) for chunk in chunks))
    if n_total < min_pixels:
        return {
            "window_id": record.get("id"),
            "label": record.get("label"),
            "status": "insufficient_pixels",
            "n_pixels": n_total,
            "min_pixels": int(min_pixels),
            "segments": segment_rows,
        }

    flux = np.concatenate([chunk["flux"] for chunk in chunks])
    model = np.concatenate([chunk["model"] for chunk in chunks])
    residual = flux - model
    scale = max(abs(float(np.nanmedian(flux))), 1e-30)
    rms_fraction = float(np.sqrt(np.nanmean(residual**2)) / scale)
    median_fraction = float(np.nanmedian(residual) / scale)
    mad_fraction = float(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual))) / scale)

    err_chunks = [
        chunk["err"]
        for chunk in chunks
        if chunk["err"] is not None and len(chunk["err"]) == len(chunk["flux"])
    ]
    if err_chunks:
        err = np.concatenate(err_chunks)
        good_err = np.isfinite(err) & (err > 0.0)
        if np.count_nonzero(good_err) >= min_pixels and err.size == residual.size:
            sigma = residual[good_err] / err[good_err]
        else:
            sigma = np.array([], dtype=float)
    else:
        sigma = np.array([], dtype=float)
    if sigma.size:
        chi2_proxy = float(np.nanmean(sigma**2))
        median_abs_sigma = float(np.nanmedian(np.abs(sigma)))
        mad_sigma = float(1.4826 * np.nanmedian(np.abs(sigma - np.nanmedian(sigma))))
    else:
        chi2_proxy = None
        median_abs_sigma = None
        mad_sigma = None

    flags = []
    if chi2_proxy is not None and chi2_proxy > 9.0:
        flags.append("{0}_high_chi2_proxy".format(quality_flag_prefix))
    if median_abs_sigma is not None and median_abs_sigma > 3.0:
        flags.append("{0}_large_median_abs_sigma".format(quality_flag_prefix))
    if rms_fraction > 0.10:
        flags.append("{0}_large_fractional_rms".format(quality_flag_prefix))
    return {
        "window_id": record.get("id"),
        "label": record.get("label"),
        "status": "ok",
        "n_pixels": n_total,
        "segments": segment_rows,
        "chi2_red_proxy": chi2_proxy,
        "median_abs_sigma": median_abs_sigma,
        "mad_sigma": mad_sigma,
        "rms_fraction": rms_fraction,
        "median_fractional_residual": median_fraction,
        "mad_fractional_residual": mad_fraction,
        "feature_families": list(record.get("feature_family", ())),
        "risk_tags": list(record.get("risk_tags", ())),
        "quality_flags": flags,
    }


def _summarize_heldout_rows(rows):
    if not rows:
        return None
    return _summarize_residual_rows(
        rows,
        interpretation=(
            "Held-out metrics are residual summaries on valid pixels that were "
            "not used by this comparison fit. They are useful for stability "
            "triage but are not a calibrated publication likelihood."
        ),
    )


def _summarize_common_rows(rows):
    if not rows:
        return None
    return _summarize_residual_rows(
        rows,
        interpretation=(
            "Common-evaluation metrics use the same planned comparison windows "
            "for every fit and keep fit-used pixels in the evaluation. They are "
            "a fairer cross-comparison residual summary than raw in-fit chi-"
            "square, but they are still diagnostic rather than a calibrated "
            "publication likelihood."
        ),
    )


def _summarize_residual_rows(rows, *, interpretation):
    n_pixels = int(sum(int(row.get("n_pixels", 0)) for row in rows))
    chi2 = _finite_values(row.get("chi2_red_proxy") for row in rows)
    med_abs = _finite_values(row.get("median_abs_sigma") for row in rows)
    rms_frac = _finite_values(row.get("rms_fraction") for row in rows)
    flags = sorted(
        {
            flag
            for row in rows
            for flag in row.get("quality_flags", ())
        }
    )
    return {
        "n_pixels": n_pixels,
        "n_windows": int(len(rows)),
        "mean_chi2_red_proxy": None if not chi2 else float(np.nanmean(chi2)),
        "max_chi2_red_proxy": None if not chi2 else float(np.nanmax(chi2)),
        "median_abs_sigma": None if not med_abs else float(np.nanmedian(med_abs)),
        "max_rms_fraction": None if not rms_frac else float(np.nanmax(rms_frac)),
        "quality_flags": flags,
        "interpretation": interpretation,
    }


def _finite_values(values):
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def _common_evaluation_records(plan):
    selected_by_id = {
        record.get("id"): dict(record)
        for record in plan.get("selection", {}).get("selected", ())
    }
    common_ids = []
    for comparison in plan.get("planned_comparisons", ()):
        for window_id in comparison.get("window_ids", ()):
            if window_id not in common_ids and window_id in selected_by_id:
                common_ids.append(window_id)
    return [selected_by_id[window_id] for window_id in common_ids]


def _common_evaluation_definition(records, *, min_pixels):
    records = [dict(record) for record in records or ()]
    if not records:
        status = "no_common_windows"
    else:
        status = "defined"
    return _json_native(
        {
            "status": status,
            "method": "common_valid_window_residuals",
            "window_policy": (
                "Union of window IDs present in planned, non-skipped comparison "
                "records. Stress-only windows enter only when the comparison "
                "plan itself includes them."
            ),
            "pixel_policy": (
                "Every completed fit is evaluated on valid pixels inside the "
                "same common windows. Fit-used pixels are retained so the pixel "
                "set is comparable across combinations; explicit excluded "
                "pixels remain excluded."
            ),
            "min_pixels": int(min_pixels),
            "n_windows": int(len(records)),
            "window_ids": [record.get("id") for record in records],
            "window_labels": [record.get("label") for record in records],
            "feature_families": sorted(
                {
                    family
                    for record in records
                    for family in record.get("feature_family", ())
                }
            ),
            "risk_tags": sorted(
                {tag for record in records for tag in record.get("risk_tags", ())}
            ),
        }
    )


def _summarize_common_evaluations(records):
    common_rows = [
        {
            "comparison_index": record.get("comparison", {}).get("comparison_index"),
            "comparison_id": record.get("comparison", {}).get("id"),
            "kind": record.get("comparison", {}).get("kind"),
            "fit_status": record.get("fit_status"),
            "result_summary": record.get("result_summary") or {},
            "common_overall": (record.get("common_evaluation") or {}).get("overall"),
            "common_status": (record.get("common_evaluation") or {}).get("status"),
        }
        for record in records
    ]
    evaluable = [
        row for row in common_rows if isinstance(row.get("common_overall"), Mapping)
    ]
    if not common_rows:
        status = "no_fit_records"
    elif evaluable:
        status = "ok"
    else:
        status = "no_evaluable_common_residuals"

    best = None
    chi2_values = []
    for row in evaluable:
        chi2 = row["common_overall"].get("mean_chi2_red_proxy")
        try:
            chi2 = float(chi2)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(chi2):
            continue
        chi2_values.append(chi2)
        if best is None or chi2 < best["common_mean_chi2_red_proxy"]:
            best = {
                "comparison_index": row.get("comparison_index"),
                "comparison_id": row.get("comparison_id"),
                "kind": row.get("kind"),
                "common_mean_chi2_red_proxy": chi2,
            }

    return _json_native(
        {
            "status": status,
            "n_fit_records": int(len(common_rows)),
            "n_evaluable_records": int(len(evaluable)),
            "best_by_common_mean_chi2_proxy": best,
            "mean_common_chi2_red_proxy": (
                None if not chi2_values else float(np.nanmean(chi2_values))
            ),
            "parameter_spread": _parameter_spread_for_records(evaluable),
            "records": [
                {
                    "comparison_index": row.get("comparison_index"),
                    "comparison_id": row.get("comparison_id"),
                    "kind": row.get("kind"),
                    "fit_status": row.get("fit_status"),
                    "common_status": row.get("common_status"),
                    "common_mean_chi2_red_proxy": (
                        None
                        if not isinstance(row.get("common_overall"), Mapping)
                        else row["common_overall"].get("mean_chi2_red_proxy")
                    ),
                    "common_n_pixels": (
                        None
                        if not isinstance(row.get("common_overall"), Mapping)
                        else row["common_overall"].get("n_pixels")
                    ),
                    "common_quality_flags": (
                        []
                        if not isinstance(row.get("common_overall"), Mapping)
                        else list(row["common_overall"].get("quality_flags", ()))
                    ),
                }
                for row in common_rows
            ],
            "interpretation": (
                "Common-evaluation summaries compare completed fits on the same "
                "diagnostic-window set. They are useful for feature-sensitivity "
                "triage and parameter-stability checks, not as an automatic "
                "model-selection rule."
            ),
        }
    )


def _parameter_spread_for_records(rows):
    out = {}
    for key in ("teff", "logg", "feh", "rv_kms", "chi2_red"):
        values = _finite_values(
            (row.get("result_summary") or {}).get(key) for row in rows
        )
        if not values:
            out[key] = {
                "n": 0,
                "min": None,
                "max": None,
                "range": None,
                "std": None,
            }
            continue
        out[key] = {
            "n": int(len(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "range": float(np.nanmax(values) - np.nanmin(values)),
            "std": float(np.nanstd(values)),
        }
    return out


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


def _as_segments(spectrum):
    if isinstance(spectrum, SpectrumSegment):
        return [spectrum]
    if isinstance(spectrum, SpectrumCollection):
        return list(spectrum.segments)
    if isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    coerced = coerce_spectrum(spectrum, warn_unknown=False)
    if isinstance(coerced, SpectrumCollection):
        return list(coerced.segments)
    return [coerced]


def _model_mask_arrays_from_result(result, *, n_segments):
    models = getattr(result, "models", None)
    used_masks = getattr(result, "used_masks", None)
    excluded_masks = getattr(result, "excluded_masks", None)
    if isinstance(result, Mapping):
        models = result.get("models", models)
        used_masks = result.get("used_masks", used_masks)
        excluded_masks = result.get("excluded_masks", excluded_masks)
    if models is None:
        return None, None, None
    model_items = tuple(models)
    if len(model_items) < n_segments:
        return None, None, None
    models = tuple(np.asarray(item, dtype=float) for item in model_items[:n_segments])

    def _optional_masks(value):
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

    return models, _optional_masks(used_masks), _optional_masks(excluded_masks)


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


def _heldout_overall_array(records, key):
    values = []
    for record in records:
        heldout = record.get("held_out_evaluation") or {}
        overall = heldout.get("overall") or {}
        try:
            value = float(overall.get(key))
        except (TypeError, ValueError):
            value = np.nan
        if not np.isfinite(value):
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def _common_overall_array(records, key):
    values = []
    for record in records:
        common = record.get("common_evaluation") or {}
        overall = common.get("overall") or {}
        try:
            value = float(overall.get(key))
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
