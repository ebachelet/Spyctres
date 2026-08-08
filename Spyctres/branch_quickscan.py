"""Branch-aware first-pass classification comparison helpers.

The helpers in this module sit between the lightweight default suggestion layer
and the more detailed diagnostic-window comparison machinery.  They answer a
practical user-facing question: given this spectrum, which broad stellar
classification branches are plausible enough to try first, and do those
branches lead to stable first-pass PHOENIX results?

The default is intentionally a dry run.  PHOENIX fits happen only when
``run_fits=True`` is supplied by the caller.
"""

from __future__ import annotations

import numpy as np

from ._serialization import (
    atomic_write_csv_rows,
    atomic_write_json,
    json_safe as _json_native,
    save_figure,
)
from .defaults import suggest_phoenix_fit_defaults
from .diagnostic_window_comparison import (
    _as_result_payload,
    _callable_name,
    _format_regions,
    _result_summary,
    _utc_now,
)
from .io import coerce_spectrum


DEFAULT_MAX_BRANCHES = 3


def build_branch_quickscan_plan(
    spectrum,
    *,
    mode="quicklook",
    max_branches=DEFAULT_MAX_BRANCHES,
):
    """Return an auditable branch-comparison plan without running fits.

    Parameters
    ----------
    spectrum : SpectrumSegment, SpectrumCollection, or compatible input
        Loaded spectrum or object accepted by :func:`Spyctres.io.coerce_spectrum`.
    mode : {"quicklook", "standard", "diagnostic"}
        Passed through to :func:`suggest_phoenix_fit_defaults`; it controls the
        branch parameter budgets.
    max_branches : int
        Maximum number of candidate branches to promote into the comparison
        queue.  This keeps the workflow bounded for large batches.

    Returns
    -------
    dict
        JSON-native branch plan with selected candidate branches, skipped
        branches, the underlying default-suggestion provenance, and the policy
        explaining how to interpret the output.
    """
    max_branches = int(max_branches)
    if max_branches < 1:
        raise ValueError("max_branches must be >= 1.")

    canonical = coerce_spectrum(
        spectrum,
        warn_unknown=False,
        source="branch_quickscan",
    )
    suggestion = suggest_phoenix_fit_defaults(
        canonical,
        mode=mode,
        science_case="branch_quickscan",
    )
    branch_plan = suggestion.provenance["classification_branches"]
    candidates = [
        dict(branch)
        for branch in branch_plan.get("branches", ())
        if branch.get("status") == "candidate"
    ]

    planned = []
    skipped = []
    for candidate_index, branch in enumerate(candidates):
        record = _branch_record(branch, candidate_index=candidate_index)
        if len(planned) < max_branches:
            record["branch_index"] = int(len(planned))
            planned.append(record)
        else:
            record["skip_reasons"] = ["max_branches_limit"]
            skipped.append(record)

    for branch in branch_plan.get("branches", ()):
        if branch.get("status") == "candidate":
            continue
        record = _branch_record(branch, candidate_index=None)
        record["skip_reasons"] = [str(branch.get("status", "not_candidate"))]
        skipped.append(record)

    return _json_native(
        {
            "schema_version": 1,
            "operation": "build_branch_quickscan_plan",
            "created_utc": _utc_now(),
            "status": "planned" if planned else "no_candidate_branches",
            "mode": str(mode),
            "default_suggestion": suggestion.to_dict(),
            "classification_branch_plan": branch_plan,
            "comparison_policy": {
                "max_branches": int(max_branches),
                "bounded_search": True,
                "dry_run_by_default": True,
                "branches_are_not_final_spectral_types": True,
                "raw_chi2_is_diagnostic_not_model_selection": True,
                "interpretation": (
                    "Compare branch coverage, fit quality flags, residual "
                    "diagnostics, and parameter stability. A lower raw chi2 for "
                    "one branch is not by itself a calibrated spectral "
                    "classification."
                ),
            },
            "planned_branches": planned,
            "skipped_branches": skipped,
        }
    )


def run_branch_quickscan(
    spectrum,
    *,
    run_fits=False,
    fit_callable=None,
    fit_call_kwargs=None,
    base_fit_kwargs=None,
    mode="quicklook",
    max_branches=DEFAULT_MAX_BRANCHES,
    progress_callback=None,
):
    """Build a branch plan and optionally run bounded branch-specific fits."""
    plan = build_branch_quickscan_plan(
        spectrum,
        mode=mode,
        max_branches=max_branches,
    )
    payload = {
        "schema_version": 1,
        "operation": "run_branch_quickscan",
        "created_utc": _utc_now(),
        "status": "planned_no_fits_run",
        "mode": plan["mode"],
        "default_suggestion": plan["default_suggestion"],
        "classification_branch_plan": plan["classification_branch_plan"],
        "comparison_policy": plan["comparison_policy"],
        "planned_branches": plan["planned_branches"],
        "skipped_branches": plan["skipped_branches"],
        "run_policy": {
            "run_fits": bool(run_fits),
            "fit_callable": _callable_name(fit_callable),
            "auto_defaults_disabled_for_branch_fits": True,
            "regions_overridden_per_branch": True,
            "default_reconstruct": False,
        },
        "fit_records": [],
        "stability_summary": {
            "status": "not_computed",
            "reason": "fits_not_run",
        },
    }

    if not run_fits:
        payload["fit_records"] = [
            _planned_fit_record(branch, fit_status="planned_not_run")
            for branch in plan["planned_branches"]
        ]
        return _json_native(payload)

    if fit_callable is None:
        from .api import fit_stellar_spectrum

        fit_callable = fit_stellar_spectrum
        payload["run_policy"]["fit_callable"] = "Spyctres.fit_stellar_spectrum"

    fit_call_kwargs = {} if fit_call_kwargs is None else dict(fit_call_kwargs)
    base_fit_kwargs = {} if base_fit_kwargs is None else dict(base_fit_kwargs)
    mode_policy = (
        plan.get("default_suggestion", {})
        .get("provenance", {})
        .get("mode_policy", {})
    )

    records = []
    for branch in plan["planned_branches"]:
        _emit_progress(
            progress_callback,
            branch,
            phase="start",
            message="Running branch quickscan {0}/{1}: {2}".format(
                int(branch["branch_index"]) + 1,
                len(plan["planned_branches"]),
                branch["label"],
            ),
        )
        record = _planned_fit_record(branch, fit_status="started")
        call_kwargs = dict(fit_call_kwargs)
        call_kwargs.setdefault("model", "phoenix")
        call_kwargs.setdefault("auto_defaults", False)
        call_kwargs.setdefault("defaults_mode", str(mode))
        call_kwargs.setdefault("science_case", "branch_quickscan")
        call_kwargs.setdefault("reconstruct", False)

        fit_kwargs = _fit_kwargs_for_branch(
            branch,
            mode_policy=mode_policy,
            base_fit_kwargs=base_fit_kwargs,
        )
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
        except Exception as exc:  # pragma: no cover - defensive CLI path
            record["fit_status"] = "error"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        records.append(record)
        _emit_progress(
            progress_callback,
            branch,
            phase="end",
            message="Finished branch quickscan {0}/{1}: {2}".format(
                int(branch["branch_index"]) + 1,
                len(plan["planned_branches"]),
                record["fit_status"],
            ),
            fit_status=record["fit_status"],
        )

    payload["fit_records"] = records
    payload["status"] = (
        "completed_with_errors"
        if any(record["fit_status"] == "error" for record in records)
        else "fits_completed"
    )
    payload["run_policy"]["fit_callable"] = _callable_name(fit_callable)
    payload["stability_summary"] = summarize_branch_fit_stability(records)
    return _json_native(payload)


def summarize_branch_fit_stability(records):
    """Summarize scalar parameter spread across completed branch fits."""
    completed = [
        record
        for record in records
        if record.get("fit_status") == "ok" and record.get("result_summary")
    ]
    if not completed:
        return {
            "status": "no_completed_fits",
            "n_completed": 0,
            "parameter_spread": {},
        }

    spread = {}
    for key in ("teff", "logg", "feh", "rv_kms", "chi2_red"):
        values = _finite_result_values(completed, key)
        spread[key] = _spread_summary(values)

    best = None
    chi2_values = _finite_result_values(completed, "chi2_red")
    if chi2_values.size:
        best_index = int(np.nanargmin(chi2_values))
        best_record = completed[best_index]
        best = {
            "branch_id": best_record["branch"]["id"],
            "label": best_record["branch"]["label"],
            "chi2_red": float(chi2_values[best_index]),
        }

    return _json_native(
        {
            "status": "ok",
            "n_completed": int(len(completed)),
            "best_by_chi2_red": best,
            "parameter_spread": spread,
            "interpretation": (
                "Use this as a stability diagnostic. Branch-to-branch scatter "
                "should be interpreted together with masks, residual plots, "
                "resolution assumptions, and model-domain flags."
            ),
        }
    )


def write_branch_quickscan_json(path, payload):
    """Write branch quickscan payload as atomic JSON."""
    atomic_write_json(path, payload, sort_keys=True)


def write_branch_quickscan_csv(path, payload):
    """Write a compact CSV summary of branch quickscan rows."""
    if path is None:
        return
    rows = payload.get("fit_records") or [
        _planned_fit_record(branch, fit_status="planned_not_run")
        for branch in payload.get("planned_branches", ())
    ]
    columns = [
        "branch_index",
        "branch_id",
        "label",
        "status",
        "score",
        "recommended",
        "ordinary_default",
        "matched_window_count",
        "total_overlap_A",
        "fit_window_ids",
        "fit_regions_A",
        "risk_tags",
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
        branch = row.get("branch", row)
        summary = row.get("result_summary") or {}
        csv_rows.append(
            {
                "branch_index": branch.get("branch_index"),
                "branch_id": branch.get("id"),
                "label": branch.get("label"),
                "status": branch.get("status"),
                "score": branch.get("score"),
                "recommended": branch.get("recommended"),
                "ordinary_default": branch.get("ordinary_default"),
                "matched_window_count": branch.get("matched_window_count"),
                "total_overlap_A": branch.get("total_overlap_A"),
                "fit_window_ids": ";".join(branch.get("fit_window_ids", ())),
                "fit_regions_A": _format_regions(branch.get("fit_regions_A", ())),
                "risk_tags": ";".join(branch.get("risk_tags", ())),
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


def plot_branch_quickscan(payload, savepath=None):
    """Plot branch support and optional fit summaries."""
    import matplotlib.pyplot as plt

    records = payload.get("fit_records") or [
        _planned_fit_record(branch, fit_status="planned_not_run")
        for branch in payload.get("planned_branches", ())
    ]
    if not records:
        raise ValueError("No branch quickscan records to plot.")

    labels = [
        "{0}: {1}".format(
            record.get("branch", record).get("branch_index", index),
            record.get("branch", record).get("id", "branch"),
        )
        for index, record in enumerate(records)
    ]
    x = np.arange(len(records), dtype=float)
    scores = np.asarray(
        [float(record.get("branch", record).get("score", np.nan)) for record in records],
        dtype=float,
    )
    chi2 = _summary_array(records, "chi2_red")
    teff = _summary_array(records, "teff")
    has_fit_values = np.any(np.isfinite(chi2)) or np.any(np.isfinite(teff))
    nrows = 3 if has_fit_values else 2

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(max(9.0, 1.8 * len(records)), 2.8 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    colors = [
        "tab:blue" if record.get("branch", record).get("recommended") else "0.55"
        for record in records
    ]
    axes[0].bar(x, scores, color=colors)
    axes[0].set_ylabel("branch score")
    axes[0].set_title("Branch quickscan: feature-coverage support")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    matrix_ax = axes[-1]
    if has_fit_values:
        if np.any(np.isfinite(chi2)):
            axes[1].plot(x, chi2, marker="o", color="tab:red", label="fit χ²ν")
            axes[1].set_ylabel("χ²ν")
        if np.any(np.isfinite(teff)):
            ax2 = axes[1].twinx()
            ax2.plot(x, teff, marker="s", color="tab:green", label="Teff")
            ax2.set_ylabel("Teff [K]")
        axes[1].grid(alpha=0.25)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    window_ids = _ordered_fit_window_ids(records)
    matrix = np.zeros((len(window_ids), len(records)), dtype=float)
    for col, record in enumerate(records):
        branch_window_ids = set(record.get("branch", record).get("fit_window_ids", ()))
        for row, window_id in enumerate(window_ids):
            matrix[row, col] = 1.0 if window_id in branch_window_ids else 0.0
    matrix_ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues")
    matrix_ax.set_yticks(np.arange(len(window_ids)))
    matrix_ax.set_yticklabels(window_ids, fontsize=8)
    matrix_ax.set_xticks(x)
    matrix_ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    matrix_ax.set_ylabel("fitted windows")
    matrix_ax.set_xlabel("branch")

    if not bool(payload.get("run_policy", {}).get("run_fits", False)):
        axes[0].text(
            0.01,
            0.94,
            "dry run: add --run-fits for bounded PHOENIX fits",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="0.25",
        )

    if savepath is not None:
        save_figure(fig, savepath, dpi=160, bbox_inches=None)
    return fig, axes


def _branch_record(branch, *, candidate_index):
    record = {
        "candidate_index": None if candidate_index is None else int(candidate_index),
        "id": branch.get("id"),
        "label": branch.get("label"),
        "description": branch.get("description"),
        "status": branch.get("status"),
        "score": float(branch.get("score", 0.0)),
        "raw_window_score": float(branch.get("raw_window_score", 0.0)),
        "ordinary_default": bool(branch.get("ordinary_default", True)),
        "recommended": False,
        "matched_window_count": int(branch.get("matched_window_count", 0)),
        "required_window_count": int(branch.get("required_window_count", 0)),
        "total_overlap_A": float(branch.get("total_overlap_A", 0.0)),
        "required_overlap_A": float(branch.get("required_overlap_A", 0.0)),
        "matched_window_ids": list(branch.get("matched_window_ids", ())),
        "fit_window_ids": list(branch.get("fit_window_ids", ())),
        "fit_regions_A": [
            (float(region[0]), float(region[1]))
            for region in branch.get("fit_regions_A", ())
        ],
        "parameter_defaults": branch.get("parameter_defaults", {}),
        "risk_tags": list(branch.get("risk_tags", ())),
        "notes": branch.get("notes"),
    }
    if candidate_index == 0:
        record["recommended"] = True
    return record


def _fit_kwargs_for_branch(branch, *, mode_policy=None, base_fit_kwargs=None):
    mode_policy = {} if mode_policy is None else dict(mode_policy)
    base_fit_kwargs = {} if base_fit_kwargs is None else dict(base_fit_kwargs)
    parameters = branch.get("parameter_defaults") or {}
    fit_kwargs = {
        "p0": tuple(float(value) for value in parameters["p0"]),
        "bounds": (
            tuple(float(value) for value in parameters["bounds"][0]),
            tuple(float(value) for value in parameters["bounds"][1]),
        ),
        "regions": [
            (float(region[0]), float(region[1]))
            for region in branch.get("fit_regions_A", ())
        ],
        "forward_model": "native_interp",
        "physical_init": "coarse",
        "coarse_teff_grid": [
            float(value) for value in parameters.get("coarse_teff_grid", ())
        ],
        "coarse_feh_grid": [
            float(value) for value in parameters.get("coarse_feh_grid", ())
        ],
        "coarse_logg_grid": [
            float(value) for value in parameters.get("coarse_logg_grid", ())
        ],
        "coarse_decimate": 12,
        "multistart": int(mode_policy.get("multistart", 1)),
        "rv_init": "grid",
        "rv_grid_n": int(mode_policy.get("rv_grid_n", 41)),
        "mdeg": 2,
    }
    fit_kwargs.update(base_fit_kwargs)
    return fit_kwargs


def _planned_fit_record(branch, *, fit_status):
    return {
        "branch": dict(branch),
        "fit_status": str(fit_status),
        "result_summary": None,
        "quality_flags": [],
        "result": None,
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


def _finite_result_values(records, key):
    values = _summary_array(records, key)
    return values[np.isfinite(values)]


def _spread_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "min": None, "max": None, "median": None, "ptp": None}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "ptp": float(np.max(values) - np.min(values)),
    }


def _ordered_fit_window_ids(records):
    window_ids = []
    for record in records:
        for window_id in record.get("branch", record).get("fit_window_ids", ()):
            if window_id not in window_ids:
                window_ids.append(window_id)
    return window_ids or ["no_fit_windows"]


def _emit_progress(callback, branch, **payload):
    if callback is None:
        return
    event = {
        "operation": "branch_quickscan",
        "branch_id": branch.get("id"),
        "branch_index": branch.get("branch_index"),
        "time_utc": _utc_now(),
    }
    event.update(payload)
    callback(_json_native(event))
