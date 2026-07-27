"""Render saved XSL validation JSON payloads into classification plots.

Example
-------
python scripts/xsl_validation_plots.py /tmp/xsl_validation_results.json \
  --output-dir /tmp/xsl_validation_plots \
  --output-pdf /tmp/xsl_validation_plots.pdf
"""

import argparse
import json
import math
import os
from pathlib import Path

from Spyctres import ensure_matplotlib_config_dir

ensure_matplotlib_config_dir()

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Spyctres import plot_xsl_validation_payload
from Spyctres._serialization import (
    atomic_write_csv_rows,
    atomic_write_json,
    safe_filename as _safe_filename,
)


REFERENCE_RECOVERY_THRESHOLDS = {
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
            "Validation plotting runner for validation_plot payloads saved by scripts/xsl_validation.py "
            "as per-target PNGs and/or a multi-page PDF."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/xsl_validation_plots.py "
            "examples/data/xsl_figure1_validation_coarse_results.json "
            "--output-dir /tmp/xsl_plots --output-pdf /tmp/xsl_plots.pdf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("results_json", help="JSON output from scripts/xsl_validation.py.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for per-target image files. Defaults to a sibling "
            "<results_json_stem>_plots directory."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Optional multi-page PDF containing all rendered target plots.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("auto", "global", "per_segment", "none"),
        default="auto",
        help=(
            "Display normalization. 'auto' uses the payload default, normally "
            "global for XSL full-spectrum displays."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Per-target image format used with --output-dir.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--xsl-id",
        action="append",
        default=None,
        help="Render only this XSL ID; repeat to select multiple targets.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help=(
            "Render only rows with this status. Repeat to include several "
            "statuses. Defaults to ok."
        ),
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional cap on the number of rendered targets.",
    )
    parser.add_argument(
        "--no-target-plots",
        action="store_true",
        help=(
            "Skip per-target observed/model plots. Useful when only compact "
            "reference-recovery summary artifacts are requested."
        ),
    )
    parser.add_argument(
        "--output-summary-md",
        default=None,
        help=(
            "Optional Markdown reference-recovery summary comparing fitted "
            "parameters to manifest/literature values."
        ),
    )
    parser.add_argument(
        "--output-summary-csv",
        default=None,
        help="Optional CSV table of per-target reference-recovery deltas.",
    )
    parser.add_argument(
        "--output-summary-plot",
        default=None,
        help="Optional compact PNG/SVG/PDF plot of reference-recovery deltas.",
    )
    parser.add_argument(
        "--output-summary-json",
        default=None,
        help=(
            "Optional JSON copy of the compact reference-recovery and "
            "publication-style stability summary."
        ),
    )
    return parser


def load_validation_results(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("Expected a validation JSON object with a 'results' list.")
    if not isinstance(payload["results"], list):
        raise ValueError("Validation JSON field 'results' must be a list.")
    return payload


def _plot_title(row):
    fit = row.get("fit") or {}
    reference = row.get("reference") or {}
    parts = [
        str(row.get("xsl_id", "XSL target")),
        str(row.get("spectral_type", "")).strip(),
        "[{0}]".format(row.get("validation_role", "unknown")),
    ]
    if fit:
        parts.append(
            "fit: Teff={0:g} K logg={1:g} [Fe/H]={2:g}".format(
                _finite_float(fit.get("teff"), 0.0),
                _finite_float(fit.get("logg"), 0.0),
                _finite_float(fit.get("feh"), 0.0),
            )
        )
    if reference:
        parts.append(
            "ref: Teff={0:g} K logg={1:g} [Fe/H]={2:g}".format(
                _finite_float(reference.get("teff"), 0.0),
                _finite_float(reference.get("logg"), 0.0),
                _finite_float(reference.get("feh"), 0.0),
            )
        )
    return "  |  ".join(part for part in parts if part)


def _finite_float(value, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _finite_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _format_value(value):
    value = _finite_or_none(value)
    if value is None:
        return "—"
    if abs(value) >= 100:
        return "{0:.1f}".format(value)
    if abs(value) >= 10:
        return "{0:.2f}".format(value)
    return "{0:.4g}".format(value)


def _assessment_from_delta(delta, thresholds):
    value = _finite_or_none(delta)
    if value is None:
        return "not_evaluated"
    abs_delta = abs(value)
    if abs_delta <= float(thresholds["acceptable_abs_delta"]):
        return "acceptable"
    if abs_delta <= float(thresholds["review_abs_delta"]):
        return "borderline"
    return "needs_review"


def _worst_assessment(values):
    rank = {
        "diagnostic_only": 0,
        "acceptable": 1,
        "borderline": 2,
        "not_evaluated": 3,
        "needs_review": 4,
    }
    ranked = [value for value in values if value in rank]
    if not ranked:
        return "not_evaluated"
    return max(ranked, key=lambda value: rank[value])


def _median(values):
    values = sorted(value for value in values if value is not None)
    n_values = len(values)
    if not n_values:
        return None
    mid = n_values // 2
    if n_values % 2:
        return float(values[mid])
    return float(0.5 * (values[mid - 1] + values[mid]))


def _quality_flags(row):
    report = row.get("quality_report") or {}
    if isinstance(report, dict):
        flags = report.get("quality_flags") or report.get("flags") or ()
    else:
        flags = ()
    return [str(flag) for flag in flags]


def _row_stability_language(row):
    """Return reviewer-facing stability language for one recovery row."""
    status = str(row.get("status", "unknown") or "unknown")
    statistics_group = str(row.get("statistics_group", "diagnostic_only") or "")
    assessment = str(row.get("recovery_assessment", "not_evaluated") or "")
    enters = bool(row.get("enters_ordinary_recovery_statistics", False))

    if status == "unsupported_physics":
        return {
            "stability_label": "unsupported_physics_excluded",
            "reviewer_priority": "stress_context",
            "reviewer_interpretation": (
                "Target lies outside the supported PHOENIX validation scope "
                "or was explicitly configured as unsupported; do not include "
                "it in ordinary recovery claims."
            ),
        }
    if statistics_group != "ordinary_recovery":
        return {
            "stability_label": "quality_gated_diagnostic_only",
            "reviewer_priority": "context_only",
            "reviewer_interpretation": (
                "Diagnostic/stress target. Display residuals and failure modes, "
                "but exclude it from ordinary reference-star recovery statistics."
            ),
        }
    if status != "ok":
        return {
            "stability_label": "not_recovered",
            "reviewer_priority": "blocking",
            "reviewer_interpretation": (
                "Ordinary reference target did not complete with status='ok'; "
                "inspect input data, masks, PHOENIX coverage, and logs."
            ),
        }
    if not enters:
        return {
            "stability_label": "diagnostic_only_excluded",
            "reviewer_priority": "review",
            "reviewer_interpretation": (
                "The target is not entering ordinary statistics despite a "
                "standard role; inspect validation metadata."
            ),
        }
    if assessment == "acceptable":
        return {
            "stability_label": "classification_stable",
            "reviewer_priority": "low",
            "reviewer_interpretation": (
                "Ordinary target recovered within provisional thresholds. "
                "Use plots to confirm no hidden residual/mask pathology."
            ),
        }
    if assessment == "borderline":
        return {
            "stability_label": "branch_stable_parameter_approximate",
            "reviewer_priority": "medium",
            "reviewer_interpretation": (
                "Broad classification may be useful, but one or more fitted "
                "parameters are close to the provisional review threshold."
            ),
        }
    if assessment == "needs_review":
        return {
            "stability_label": "needs_review_not_stable",
            "reviewer_priority": "high",
            "reviewer_interpretation": (
                "At least one ordinary reference-parameter delta exceeds the "
                "review threshold; do not claim this target as recovered yet."
            ),
        }
    return {
        "stability_label": "metadata_limited_not_evaluated",
        "reviewer_priority": "high",
        "reviewer_interpretation": (
            "Reference or fitted parameters are missing, so recovery stability "
            "cannot be evaluated."
        ),
    }


def _reference_recovery_row(row):
    reference = row.get("reference") or {}
    fit = row.get("fit") or {}
    delta_payload = row.get("delta") or {}
    status = str(row.get("status", "unknown") or "unknown")
    statistics_group = str(
        row.get("statistics_group", "diagnostic_only") or "diagnostic_only"
    )
    ordinary = statistics_group == "ordinary_recovery"
    out = {
        "xsl_id": row.get("xsl_id"),
        "star_name": row.get("star_name"),
        "spectral_type": row.get("spectral_type"),
        "validation_role": row.get("validation_role", "unknown"),
        "statistics_group": statistics_group,
        "status": status,
        "enters_ordinary_recovery_statistics": bool(ordinary and status == "ok"),
        "quality_flags": _quality_flags(row),
    }
    assessments = []
    for key in ("teff", "logg", "feh"):
        ref = _finite_or_none(reference.get(key))
        value = _finite_or_none(fit.get(key))
        delta = _finite_or_none(delta_payload.get(key))
        if delta is None and ref is not None and value is not None:
            delta = value - ref
        assessment = (
            _assessment_from_delta(delta, REFERENCE_RECOVERY_THRESHOLDS[key])
            if ordinary and status == "ok"
            else "diagnostic_only"
        )
        out["{0}_ref".format(key)] = ref
        out["{0}_fit".format(key)] = value
        out["delta_{0}".format(key)] = delta
        out["{0}_assessment".format(key)] = assessment
        assessments.append(assessment)
    out["recovery_assessment"] = (
        _worst_assessment(assessments)
        if ordinary and status == "ok"
        else "diagnostic_only"
    )
    if status == "unsupported_physics":
        out["recovery_assessment"] = "diagnostic_only"
    out.update(_row_stability_language(out))
    return out


def _count_by_key(rows, key):
    counts = {}
    for row in rows:
        value = str(row.get(key, "unknown") or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _build_publication_style_summary(summary):
    """Return publication-scaffold-style stability interpretation.

    This is intentionally scoped to XSL/reference-star validation.  It does
    not claim that Spyctres is publication-ready as a whole.
    """
    rows = list(summary.get("rows") or ())
    ordinary = list(summary.get("ordinary_rows") or ())
    diagnostic = list(summary.get("diagnostic_or_stress_rows") or ())
    status = summary.get("status")
    ordinary_problem_rows = [
        row
        for row in ordinary
        if row.get("stability_label") != "classification_stable"
    ]
    failed_standard_rows = [
        row
        for row in rows
        if row.get("statistics_group") == "ordinary_recovery"
        and row.get("status") != "ok"
    ]
    limiting_rows = list(ordinary_problem_rows) + list(failed_standard_rows)

    if failed_standard_rows:
        claim_status = "not_ready_standard_targets_failed"
        plain = (
            "At least one ordinary standard-reference target did not complete "
            "cleanly. Treat the current output as diagnostic until those rows "
            "are resolved or deliberately reclassified."
        )
    elif not ordinary:
        claim_status = "not_ready_no_standard_reference_recovery"
        plain = (
            "No successful ordinary standard-reference XSL targets are available, "
            "so the validation set is not yet a recovery demonstration."
        )
    elif any(row.get("recovery_assessment") == "needs_review" for row in ordinary):
        claim_status = "exploratory_not_reference_stable"
        plain = (
            "At least one ordinary XSL target exceeds the provisional recovery "
            "thresholds. The current run is useful for debugging, not for a "
            "clean classification-stability claim."
        )
    elif any(row.get("recovery_assessment") == "not_evaluated" for row in ordinary):
        claim_status = "not_ready_reference_metadata_incomplete"
        plain = (
            "Ordinary targets ran, but reference/fitted parameter deltas are "
            "missing for at least one row. Complete the manifest/reference "
            "metadata before interpreting stability."
        )
    elif any(row.get("recovery_assessment") == "borderline" for row in ordinary):
        claim_status = "exploratory_borderline_reference_stability"
        plain = (
            "Ordinary reference-star recovery is borderline. The spectral branch "
            "may be stable, but atmospheric parameters should be described as "
            "approximate until more targets and sensitivity checks pass."
        )
    else:
        claim_status = "classification_stable_for_current_standard_targets"
        plain = (
            "All successful ordinary XSL standard-reference targets are within "
            "the current provisional thresholds. This is a reviewer-ready "
            "classification-stability check, but not a final package-wide "
            "publication claim."
        )

    review_questions = [
        "Are the provisional Teff/logg/[Fe/H] thresholds appropriate for this XSL validation phase?",
        "Should any validation_role values be promoted to ordinary recovery or kept as stress/peculiar diagnostics?",
        "Which residual or mask patterns should block a 'classification_stable' label even when parameter deltas pass?",
        "Should recovery thresholds become spectral-type dependent once the branch-aware first-pass classifier is implemented?",
    ]
    if diagnostic:
        review_questions.append(
            "Do the diagnostic/stress targets expose failure modes that should become automated quality flags?"
        )

    return {
        "schema_version": 1,
        "status": "summary_ready_for_reviewer",
        "input_summary_status": status,
        "claim_status": claim_status,
        "claim_scope": (
            "XSL/reference-star validation only; final software publication "
            "claims still require broader real-spectrum validation, sensitivity "
            "checks, and reviewer inspection."
        ),
        "plain_language_summary": plain,
        "ready_for_referee_review": bool(rows),
        "ready_for_publication_claim": False,
        "ordinary_stability_label_counts": _count_by_key(
            ordinary,
            "stability_label",
        ),
        "all_stability_label_counts": _count_by_key(rows, "stability_label"),
        "diagnostic_or_stress_count": int(len(diagnostic)),
        "limiting_rows": limiting_rows,
        "limiting_checks": [
            {
                "xsl_id": row.get("xsl_id"),
                "stability_label": row.get("stability_label"),
                "recovery_assessment": row.get("recovery_assessment"),
                "reviewer_priority": row.get("reviewer_priority"),
                "interpretation": row.get("reviewer_interpretation"),
            }
            for row in limiting_rows
        ],
        "review_questions": review_questions,
    }


def build_reference_recovery_summary(payload):
    """Build a compact, role-aware XSL/reference-star recovery summary."""
    rows = [
        _reference_recovery_row(row)
        for row in payload.get("results", [])
        if isinstance(row, dict)
    ]
    ordinary = [
        row
        for row in rows
        if row["enters_ordinary_recovery_statistics"]
    ]
    ordinary_assessment = _worst_assessment(
        row.get("recovery_assessment") for row in ordinary
    )
    if not ordinary:
        status = "summary_ready_no_ordinary_reference_fits"
        claim_status = "no_standard_reference_recovery"
        plain = (
            "No successful ordinary standard-reference targets are available. "
            "Stress, peculiar, diagnostic, and unsupported targets are listed "
            "but excluded from recovery statistics."
        )
    elif ordinary_assessment == "not_evaluated":
        status = "summary_ready_reference_recovery_not_evaluated"
        claim_status = "standard_reference_recovery_not_evaluated"
        plain = (
            "Successful ordinary standard-reference targets are present, but "
            "their reference-parameter deltas could not be evaluated. Check "
            "the validation manifest reference values before interpreting "
            "this as a recovery pass."
        )
    elif ordinary_assessment == "needs_review":
        status = "summary_ready_reference_recovery_needs_review"
        claim_status = "reference_recovery_needs_review"
        plain = (
            "At least one ordinary XSL/reference-star recovery exceeds the "
            "provisional review thresholds. Inspect the per-target plots, "
            "metadata, masks, and PHOENIX applicability before using the "
            "configuration as a clean reference demonstration."
        )
    elif ordinary_assessment == "borderline":
        status = "summary_ready_reference_recovery_borderline"
        claim_status = "reference_recovery_borderline"
        plain = (
            "Ordinary reference-star recovery is borderline under the "
            "provisional thresholds. This is useful validation evidence, but "
            "not yet a clean pass."
        )
    else:
        status = "summary_ready_reference_recovery_acceptable"
        claim_status = "reference_recovery_acceptable_for_current_thresholds"
        plain = (
            "Ordinary reference-star recovery is acceptable under the current "
            "provisional thresholds. This supports moving to additional "
            "reference stars and cross-instrument validation."
        )

    stats = {"count": int(len(ordinary))}
    for key in ("teff", "logg", "feh"):
        deltas = [row.get("delta_{0}".format(key)) for row in ordinary]
        stats[key] = {
            "median_delta": _median(deltas),
            "median_absolute_delta": _median(
                [None if value is None else abs(value) for value in deltas]
            ),
            "max_absolute_delta": (
                None
                if not [value for value in deltas if value is not None]
                else float(max(abs(value) for value in deltas if value is not None))
            ),
        }

    diagnostic_rows = [
        row for row in rows if not row["enters_ordinary_recovery_statistics"]
    ]
    recommendations = []
    if status == "summary_ready_no_ordinary_reference_fits":
        recommendations.append(
            "Run at least one validation_role=standard XSL reference target successfully."
        )
    if status == "summary_ready_reference_recovery_not_evaluated":
        recommendations.append(
            "Populate reference Teff/logg/[Fe/H] values for ordinary standard targets before claiming parameter recovery."
        )
    if status == "summary_ready_reference_recovery_needs_review":
        recommendations.append(
            "Inspect ordinary targets with recovery_assessment=needs_review before promoting this workflow."
        )
    if diagnostic_rows:
        recommendations.append(
            "Keep diagnostic/stress/unsupported targets separate from ordinary recovery statistics."
        )
    recommendations.append(
        "Next cross-instrument validation should include the user's clean and dirty SDSS and UVES-POP spectra."
    )

    out = {
        "schema_version": 1,
        "status": status,
        "claim_status": claim_status,
        "plain_language_summary": plain,
        "thresholds": REFERENCE_RECOVERY_THRESHOLDS,
        "statistics_policy": (
            "Only status='ok' rows with statistics_group='ordinary_recovery' "
            "enter ordinary reference-recovery statistics. Stress, peculiar, "
            "diagnostic, and unsupported targets are displayed but excluded."
        ),
        "ordinary_recovery_statistics": stats,
        "validation_role_summary": payload.get("validation_role_summary") or {},
        "rows": rows,
        "ordinary_rows": ordinary,
        "diagnostic_or_stress_rows": diagnostic_rows,
        "recommendations": recommendations,
    }
    out["publication_summary"] = _build_publication_style_summary(out)
    return out


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
            elif isinstance(value, (int, float)):
                value = _format_value(value)
            elif value is None or value == "":
                value = "—"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_reference_recovery_summary_csv(path, summary):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "xsl_id",
        "star_name",
        "spectral_type",
        "validation_role",
        "statistics_group",
        "status",
        "enters_ordinary_recovery_statistics",
        "recovery_assessment",
        "stability_label",
        "reviewer_priority",
        "reviewer_interpretation",
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
        "quality_flags",
    ]
    rows = []
    for row in summary.get("rows", ()):
        rows.append(
            {
                **{key: row.get(key) for key in columns if key != "quality_flags"},
                "quality_flags": ";".join(row.get("quality_flags") or ()),
            }
        )
    atomic_write_csv_rows(path, columns, rows)


def write_reference_recovery_summary_markdown(path, summary):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = summary.get("ordinary_recovery_statistics") or {}
    columns = [
        ("xsl_id", "id"),
        ("spectral_type", "type"),
        ("validation_role", "role"),
        ("status", "status"),
        ("recovery_assessment", "assessment"),
        ("stability_label", "stability"),
        ("reviewer_priority", "priority"),
        ("teff_ref", "Teff ref"),
        ("teff_fit", "Teff fit"),
        ("delta_teff", "ΔTeff"),
        ("logg_ref", "logg ref"),
        ("logg_fit", "logg fit"),
        ("delta_logg", "Δlogg"),
        ("feh_ref", "[Fe/H] ref"),
        ("feh_fit", "[Fe/H] fit"),
        ("delta_feh", "Δ[Fe/H]"),
    ]
    stat_rows = []
    for key, label in (("teff", "Teff"), ("logg", "logg"), ("feh", "[Fe/H]")):
        entry = stats.get(key) or {}
        stat_rows.append(
            {
                "parameter": label,
                "median_delta": entry.get("median_delta"),
                "median_absolute_delta": entry.get("median_absolute_delta"),
                "max_absolute_delta": entry.get("max_absolute_delta"),
            }
        )
    stat_columns = [
        ("parameter", "parameter"),
        ("median_delta", "median Δ"),
        ("median_absolute_delta", "median abs Δ"),
        ("max_absolute_delta", "max abs Δ"),
    ]
    content = [
        "# Spyctres XSL reference-recovery summary",
        "",
        "Status: `{0}`".format(summary.get("status")),
        "",
        "Claim status: `{0}`".format(summary.get("claim_status")),
        "",
        summary.get("plain_language_summary", ""),
        "",
        "## Publication-style stability interpretation",
        "",
        "Claim status: `{0}`".format(
            (summary.get("publication_summary") or {}).get("claim_status")
        ),
        "",
        (summary.get("publication_summary") or {}).get(
            "plain_language_summary",
            "No publication-style interpretation is available.",
        ),
        "",
        "Claim scope: {0}".format(
            (summary.get("publication_summary") or {}).get(
                "claim_scope",
                "XSL/reference-star validation only.",
            )
        ),
        "",
        "Statistics policy: {0}".format(summary.get("statistics_policy")),
        "",
        "## Recommendations",
        "",
    ]
    recommendations = summary.get("recommendations") or ()
    if recommendations:
        content.extend("- {0}".format(item) for item in recommendations)
    else:
        content.append("- No summary-level recommendations.")
    content.extend(
        [
            "",
            "## Ordinary recovery statistics",
            "",
            "Ordinary standard-reference count: `{0}`".format(stats.get("count", 0)),
            "",
            _markdown_table(stat_columns, stat_rows),
            "",
            "## Ordinary standard-reference rows",
            "",
            _markdown_table(columns, summary.get("ordinary_rows") or ()),
            "",
            "## Diagnostic/stress/unsupported rows",
            "",
            _markdown_table(columns, summary.get("diagnostic_or_stress_rows") or ()),
            "",
            "## Questions for reviewer",
            "",
        ]
    )
    review_questions = (summary.get("publication_summary") or {}).get(
        "review_questions",
        (),
    )
    if review_questions:
        content.extend("- {0}".format(item) for item in review_questions)
    else:
        content.append("- No reviewer questions recorded.")
    content.append("")
    path.write_text("\n".join(content), encoding="utf-8")


def write_reference_recovery_summary_json(path, summary):
    if path is None:
        return
    atomic_write_json(path, summary)


def plot_reference_recovery_summary(summary, savepath=None):
    """Plot compact reference-recovery deltas for ordinary validation targets."""
    rows = list(summary.get("ordinary_rows") or ())
    if not rows:
        rows = [
            row for row in summary.get("rows", ()) if row.get("status") == "ok"
        ]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.0, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    axes = list(axes)
    if not rows:
        for ax in axes:
            ax.axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No successful XSL reference-recovery rows to plot.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    else:
        labels = [str(row.get("xsl_id") or index) for index, row in enumerate(rows)]
        x = list(range(len(rows)))
        colors = {
            "acceptable": "tab:green",
            "borderline": "tab:orange",
            "needs_review": "tab:red",
            "diagnostic_only": "0.6",
            "not_evaluated": "0.7",
        }
        for ax, key, ylabel in zip(
            axes,
            ("teff", "logg", "feh"),
            ("ΔTeff [K]", "Δlogg [dex]", "Δ[Fe/H] [dex]"),
        ):
            values = [
                _finite_or_none(row.get("delta_{0}".format(key))) for row in rows
            ]
            heights = [0.0 if value is None else value for value in values]
            bar_colors = [
                colors.get(row.get("{0}_assessment".format(key)), "0.5")
                for row in rows
            ]
            ax.axhline(0.0, color="0.25", lw=0.8)
            threshold = REFERENCE_RECOVERY_THRESHOLDS[key]
            ax.axhline(
                threshold["acceptable_abs_delta"],
                color="tab:green",
                ls=":",
                lw=0.8,
                alpha=0.75,
            )
            ax.axhline(
                -threshold["acceptable_abs_delta"],
                color="tab:green",
                ls=":",
                lw=0.8,
                alpha=0.75,
            )
            ax.axhline(
                threshold["review_abs_delta"],
                color="tab:red",
                ls="--",
                lw=0.8,
                alpha=0.75,
            )
            ax.axhline(
                -threshold["review_abs_delta"],
                color="tab:red",
                ls="--",
                lw=0.8,
                alpha=0.75,
            )
            ax.bar(x, heights, color=bar_colors, alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    publication = summary.get("publication_summary") or {}
    fig.suptitle(
        "XSL reference-star recovery summary — {0}".format(
            publication.get("claim_status", summary.get("claim_status", "unknown"))
        )
    )
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=160)
    return fig, axes


def write_reference_recovery_summary_outputs(
    summary,
    *,
    output_md=None,
    output_csv=None,
    output_plot=None,
    output_json=None,
):
    write_reference_recovery_summary_markdown(output_md, summary)
    write_reference_recovery_summary_csv(output_csv, summary)
    write_reference_recovery_summary_json(output_json, summary)
    if output_plot is not None:
        fig, _axes = plot_reference_recovery_summary(summary, savepath=output_plot)
        plt.close(fig)


def iter_validation_plot_rows(
    payload,
    *,
    xsl_ids=None,
    statuses=("ok",),
    max_targets=None,
):
    """Yield result rows that contain saved validation-plot payloads."""
    xsl_filter = None
    if xsl_ids:
        xsl_filter = {str(value).strip().upper() for value in xsl_ids}
    status_filter = None if statuses is None else {str(value) for value in statuses}
    count = 0
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        if (
            xsl_filter is not None
            and str(row.get("xsl_id", "")).upper() not in xsl_filter
        ):
            continue
        if (
            status_filter is not None
            and str(row.get("status", "")) not in status_filter
        ):
            continue
        plot_data = row.get("validation_plot")
        if not isinstance(plot_data, dict):
            continue
        yield row
        count += 1
        if max_targets is not None and count >= int(max_targets):
            break


def render_validation_plots(
    payload,
    *,
    output_dir=None,
    output_pdf=None,
    scale_mode="auto",
    image_format="png",
    dpi=160,
    xsl_ids=None,
    statuses=("ok",),
    max_targets=None,
):
    """Render saved XSL validation plot payloads.

    Returns a list of generated per-target image paths.  The optional PDF path
    is not included in that list because it contains all rendered targets.
    """
    rows = list(
        iter_validation_plot_rows(
            payload,
            xsl_ids=xsl_ids,
            statuses=statuses,
            max_targets=max_targets,
        )
    )
    if not rows:
        raise ValueError("No matching rows contain validation_plot payloads.")

    image_paths = []
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    pdf = PdfPages(output_pdf) if output_pdf is not None else None
    try:
        for row in rows:
            mode = None if scale_mode == "auto" else scale_mode
            fig, _axes = plot_xsl_validation_payload(
                row["validation_plot"],
                scale_mode=mode,
                title=_plot_title(row),
            )
            if output_path is not None:
                stem = "_".join(
                    item
                    for item in (
                        _safe_filename(row.get("xsl_id"), fallback="target"),
                        _safe_filename(row.get("spectral_type"), fallback=""),
                    )
                    if item
                )
                image_path = output_path / "{0}.{1}".format(stem, image_format)
                fig.savefig(image_path, dpi=dpi)
                image_paths.append(str(image_path))
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return image_paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_targets is not None and args.max_targets < 1:
        raise ValueError("--max-targets must be >= 1.")
    results_path = Path(args.results_json)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(results_path.with_name(results_path.stem + "_plots"))

    payload = load_validation_results(results_path)
    if (
        args.output_summary_md is not None
        or args.output_summary_csv is not None
        or args.output_summary_plot is not None
        or args.output_summary_json is not None
    ):
        summary = build_reference_recovery_summary(payload)
        write_reference_recovery_summary_outputs(
            summary,
            output_md=args.output_summary_md,
            output_csv=args.output_summary_csv,
            output_plot=args.output_summary_plot,
            output_json=args.output_summary_json,
        )
        print(
            "Reference-recovery summary: {0}".format(summary["status"]),
            flush=True,
        )
        if args.output_summary_md:
            print(
                "Summary Markdown: {0}".format(
                    os.path.abspath(args.output_summary_md)
                ),
                flush=True,
            )
        if args.output_summary_csv:
            print(
                "Summary CSV: {0}".format(os.path.abspath(args.output_summary_csv)),
                flush=True,
            )
        if args.output_summary_plot:
            print(
                "Summary plot: {0}".format(
                    os.path.abspath(args.output_summary_plot)
                ),
                flush=True,
            )
        if args.output_summary_json:
            print(
                "Summary JSON: {0}".format(
                    os.path.abspath(args.output_summary_json)
                ),
                flush=True,
            )

    if not args.no_target_plots:
        images = render_validation_plots(
            payload,
            output_dir=output_dir,
            output_pdf=args.output_pdf,
            scale_mode=args.scale_mode,
            image_format=args.format,
            dpi=args.dpi,
            xsl_ids=args.xsl_id,
            statuses=args.status or ("ok",),
            max_targets=args.max_targets,
        )
        print("Rendered {0} XSL validation plot(s).".format(len(images)), flush=True)
        print("Image directory: {0}".format(os.path.abspath(output_dir)), flush=True)
        if args.output_pdf:
            print("PDF: {0}".format(os.path.abspath(args.output_pdf)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
