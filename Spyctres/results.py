"""Serializable result containers for public Spyctres workflows."""

from dataclasses import dataclass, field
from collections.abc import Mapping
import json
import os
import tempfile

import numpy as np


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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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
    "wavelength_frame_ambiguous": (
        "At least one fitted segment has an unknown wavelength frame; RV and "
        "barycentric assumptions need review."
    ),
    "metadata_incomplete": (
        "One or more wavelength/metadata fields are unknown or incomplete."
    ),
    "mask_fraction_high": (
        "More than half of the global support pixels were excluded from the fit."
    ),
    "segment_no_fit_pixels": (
        "At least one input segment had no usable fit pixels and was dropped."
    ),
    "too_few_fit_pixels": (
        "At least one retained segment has very few fit pixels compared with "
        "the model complexity."
    ),
    "segment_mask_fraction_high": (
        "At least one retained segment had more than half of its support pixels "
        "masked out."
    ),
    "explicit_exclusion_dominates": (
        "Explicit user/recipe exclusions removed more pixels than remained in "
        "at least one segment."
    ),
    "nonfinite_mask_output": (
        "A mask callable produced nonfinite values; those pixels were rejected "
        "under the mask-output policy."
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
                "lsf_fwhm_kms": segment.get("lsf_fwhm_kms"),
                "resolution_R_effective": segment.get("resolution_R_effective"),
            }
        )
    report = {
        "success": summary.get("success"),
        "quality_flags": flags,
        "quality_flag_descriptions": describe_quality_flags(flags),
        "reduced_chi2": chi2_red,
        "n_points": summary.get("n_points", diagnostics.get("n_pixels")),
        "n_parameters": diagnostics.get("n_parameters"),
        "degrees_of_freedom": diagnostics.get("degrees_of_freedom"),
        "mask_fraction": diagnostics.get("mask_fraction"),
        "n_input_segments": diagnostics.get("n_input_segments"),
        "n_retained_segments": diagnostics.get("n_retained_segments"),
        "n_dropped_segments": diagnostics.get("n_dropped_segments"),
        "segments": segments,
    }
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
    if report.get("n_dropped_segments"):
        lines.append("  dropped segments: {0}".format(int(report["n_dropped_segments"])))
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
        directory = os.path.dirname(path) or "."
        descriptor, temporary = tempfile.mkstemp(
            prefix=".{0}.".format(os.path.basename(path)),
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self.to_dict(
                        include_arrays=include_arrays,
                        include_local_paths=include_local_paths,
                        plot_paths=plot_paths,
                        relative_to=relative_to,
                    ),
                    handle,
                    **kwargs,
                )
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
