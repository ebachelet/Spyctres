"""Shared helpers for example/workflow scripts.

These are deliberately kept out of the top-level public API.  They centralize
small but easy-to-diverge pieces of workflow bookkeeping: user-supplied
resolution provenance and optional archive/product mask merging.
"""

from __future__ import annotations

import math

from ._spectrum_helpers import spectrum_segments
from .preprocessing import archive_exclusion_masks_for_segment


DEFAULT_RESOLUTION_ASSUMPTION_WARNING = "approximate quicklook resolution"


def resolution_override_summary(
    resolution_R,
    *,
    source="user_override",
    assumption_warning=DEFAULT_RESOLUTION_ASSUMPTION_WARNING,
):
    """Return JSON-friendly provenance for an explicit constant-R override."""
    if resolution_R is None:
        return None
    value = float(resolution_R)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("resolution_R must be finite and > 0.")
    return {
        "resolution_source": str(source),
        "assumed_resolution_R": value,
        "assumption_warning": str(assumption_warning),
    }


def resolution_assumption_for_audit(
    resolution_R,
    *,
    source="user_override",
    assumption_warning=DEFAULT_RESOLUTION_ASSUMPTION_WARNING,
):
    """Return the fit-readiness ``assumed_resolution`` record for ``R``."""
    payload = resolution_override_summary(
        resolution_R,
        source=source,
        assumption_warning=assumption_warning,
    )
    if payload is None:
        return None
    return {
        "quantity": "R",
        "value": payload["assumed_resolution_R"],
        "source": payload["resolution_source"],
        "assumption_warning": payload["assumption_warning"],
    }


def archive_masks_by_segment(spectrum, *, only_ids=None):
    """Return opt-in archive/product masks keyed by segment index."""
    out = {}
    for index, segment in enumerate(
        spectrum_segments(spectrum, tuple_is_collection=True, coerce=False)
    ):
        masks = archive_exclusion_masks_for_segment(segment, only_ids=only_ids)
        if masks:
            out[index] = masks
    return out


def archive_mask_count(archive_masks):
    """Return the number of masks in an index-keyed archive-mask mapping."""
    return int(sum(len(value) for value in (archive_masks or {}).values()))


def comma_join_or_none(values):
    """Return a compact comma-separated string for user-facing summaries."""
    items = [str(item) for item in (values or ()) if str(item)]
    return ", ".join(items) if items else "none"


def readiness_summary_line(readiness, *, label="Readiness"):
    """Return a one-line, intent-aware readiness summary.

    ``fit_ready`` is kept as the older strict/backward-compatible gate. The
    newer ``ready_for_intent`` field answers the user-facing question: "is this
    spectrum ready for the specific task I asked to do?"
    """
    readiness = readiness or {}
    return (
        "{label}: intent={intent}, ready_for_intent={ready}, "
        "strict_fit_ready={fit_ready}, fitted_pixels={n_fit}, flags={flags}"
    ).format(
        label=str(label),
        intent=readiness.get("intent") or readiness.get("readiness_intent"),
        ready=readiness.get("ready_for_intent"),
        fit_ready=readiness.get("fit_ready"),
        n_fit=readiness.get("n_fit_candidate"),
        flags=comma_join_or_none(readiness.get("interpretation_flags") or ()),
    )


def readiness_intent_detail_lines(readiness, *, prefix="  "):
    """Return short blocker/warning lines for the active readiness intent."""
    readiness = readiness or {}
    lines = []
    blockers = readiness.get("blockers_for_intent") or ()
    warnings = readiness.get("warnings_for_intent") or ()
    invalid = readiness.get("invalid_interpretations_for_intent") or ()
    if blockers:
        lines.append(
            "{0}blockers_for_intent: {1}".format(
                prefix,
                comma_join_or_none(blockers),
            )
        )
    if warnings:
        lines.append(
            "{0}warnings_for_intent: {1}".format(
                prefix,
                comma_join_or_none(warnings),
            )
        )
    if invalid:
        lines.append(
            "{0}invalid_interpretations_for_intent: {1}".format(
                prefix,
                comma_join_or_none(invalid),
            )
        )
    return lines


def unique_archive_masks(spectrum, *, policy="apply", only_ids=None):
    """Return unique archive masks for workflows that build derived segments.

    This preserves the reviewed-analysis scaffold's existing behaviour: when several
    derived line-window segments share the same original archive/product mask,
    the mask is recorded once by name rather than repeated for every window.
    """
    if policy != "apply":
        return ()
    masks = []
    seen = set()
    for segment in spectrum_segments(spectrum, tuple_is_collection=True, coerce=False):
        for mask in archive_exclusion_masks_for_segment(segment, only_ids=only_ids):
            if mask.name in seen:
                continue
            masks.append(mask)
            seen.add(mask.name)
    return tuple(masks)


def _mask_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def fit_kwargs_with_archive_policy(fit_kwargs, archive_masks, policy):
    """Merge archive masks into PHOENIX fit kwargs when requested.

    ``exclude_masks`` accepts either the global list form or the segment-indexed
    dictionary form used by multi-segment fits.  Archive masks are indexed by
    segment and are opt-in; non-``apply`` policies leave the fit kwargs
    unchanged.
    """
    fit_kwargs = dict(fit_kwargs)
    if policy != "apply" or not archive_masks:
        return fit_kwargs

    existing = fit_kwargs.get("exclude_masks")
    if existing is None:
        fit_kwargs["exclude_masks"] = dict(archive_masks)
        return fit_kwargs

    merged = {
        key: _mask_list(value)
        for key, value in dict(archive_masks).items()
    }
    if isinstance(existing, dict):
        for key, value in existing.items():
            current = list(merged.get(key, []) or [])
            current.extend(_mask_list(value))
            merged[key] = current
    else:
        extra = _mask_list(existing)
        for key in list(merged):
            merged[key] = list(merged[key]) + extra
    fit_kwargs["exclude_masks"] = merged
    return fit_kwargs
