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
        """Return a compact, JSON-safe summary of fit quality diagnostics.

        This is intentionally redundant with the full diagnostics block.  The
        diagnostics preserve the detailed machine-readable record, while this
        report gives notebooks, scripts, and reviewers a stable headline view
        of the main fit-quality concerns.
        """
        diagnostics = self.diagnostics.to_dict()
        flags = list(self.quality_flags)
        chi2_red = self.summary.get("chi2_red", diagnostics.get("reduced_chi2"))
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
                    "multiple_rejection_count": mask_summary.get(
                        "n_rejected_by_multiple_reasons"
                    ),
                    "lsf_fwhm_kms": segment.get("lsf_fwhm_kms"),
                    "resolution_R_effective": segment.get(
                        "resolution_R_effective"
                    ),
                }
            )
        report = {
            "success": self.summary.get("success"),
            "quality_flags": flags,
            "reduced_chi2": chi2_red,
            "n_points": self.summary.get("n_points", diagnostics.get("n_pixels")),
            "n_parameters": diagnostics.get("n_parameters"),
            "degrees_of_freedom": diagnostics.get("degrees_of_freedom"),
            "mask_fraction": diagnostics.get("mask_fraction"),
            "n_input_segments": diagnostics.get("n_input_segments"),
            "n_retained_segments": diagnostics.get("n_retained_segments"),
            "n_dropped_segments": diagnostics.get("n_dropped_segments"),
            "segments": segments,
        }
        return _jsonable(report)

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
