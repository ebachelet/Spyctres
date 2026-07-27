"""Serializable result containers for public Spyctres workflows."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
import json
import os
import subprocess

import numpy as np

from ._serialization import atomic_write_json, json_safe as _jsonable


FIT_REPORT_SCHEMA_VERSION = 1
FIT_RESULT_PAYLOAD_SCHEMA_VERSION = 1
FIT_REPORT_TYPE = "spyctres.fit_result_report"


def _spyctres_version():
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
            "instrument": provenance.get("instrument"),
            "input_was_path": provenance.get("input_was_path"),
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
            "phoenix_composition_note": provenance.get("phoenix_composition_note"),
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
            "schema_version": FIT_REPORT_SCHEMA_VERSION,
            "report_type": FIT_REPORT_TYPE,
            "result_payload_schema_version": FIT_RESULT_PAYLOAD_SCHEMA_VERSION,
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
