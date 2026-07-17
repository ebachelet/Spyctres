from .matplotlib_setup import ensure_matplotlib_config_dir
ensure_matplotlib_config_dir()

from .api import classify_spectrum, fit_phoenix_spectrum, fit_stellar_spectrum
from .defaults import (
    PhoenixFitDefaults,
    clip_grid_to_bounds,
    prepare_phoenix_fit_kwargs,
    spectrum_wavelength_range,
    suggest_phoenix_fit_defaults,
)
from .results import (
    PhoenixFitDiagnostics,
    PhoenixFitResult,
    build_fit_quality_report,
    compare_fit_results,
    describe_quality_flags,
    format_fit_quality_report,
)
from .linefitting import LineSpec, LineFitConfig, LineFitResult, fit_line, fit_lines
from .plotting import plot_fit_referee, plot_line_fit, plot_xsl_validation_payload
from .diagnostics import (
    KNOWN_RESIDUAL_WINDOWS,
    annotate_nonstellar_features,
    diagnose_known_residual_windows,
)
from .recipes import (
    normalize_segment_sidebands,
    normalize_segments_sidebands,
    sdss_quicklook_resolution_assumption,
)
from .preprocessing import (
    NONSTELLAR_FEATURES,
    OPTICAL_DIB_DIAGNOSTIC_FEATURES,
    OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES,
    ExclusionMaskSpec,
    archive_exclusion_masks,
    archive_exclusion_masks_for_segment,
    archive_mask_catalog,
    archive_mask_summary_for_segment,
    artifact_exclusion_mask_from_segment,
    audit_spectrum_for_fit,
    combine_exclusion_masks,
    dilate_boolean_mask,
    exclusion_mask,
    known_feature_masks,
    nonstellar_feature_mask,
    nonstellar_feature_masks,
    nonstellar_feature_metadata,
    nonstellar_feature_regions,
    overlapping_nonstellar_features,
    broad_telluric_catalog_fallback_mask,
    telluric_transmission_exclusion_mask,
)
from .waveutils import convert_segment_wavelength_medium, convert_wavelength_medium
from .fitting import FitProgressEvent
from .io import (
    SpectrumCollection,
    SpectrumSegment,
    read_sdss_spec,
    read_spectrum,
    read_uves_pop_ascii,
    sdss_wdisp_to_resolution_descriptor,
)

__all__ = [
    "fit_phoenix_spectrum", "fit_stellar_spectrum", "classify_spectrum",
    "read_spectrum", "read_uves_pop_ascii", "read_sdss_spec",
    "sdss_wdisp_to_resolution_descriptor",
    "SpectrumSegment", "SpectrumCollection",
    "ensure_matplotlib_config_dir",
    "PhoenixFitDiagnostics", "PhoenixFitResult",
    "PhoenixFitDefaults", "clip_grid_to_bounds",
    "prepare_phoenix_fit_kwargs", "spectrum_wavelength_range",
    "suggest_phoenix_fit_defaults",
    "build_fit_quality_report", "compare_fit_results", "describe_quality_flags",
    "format_fit_quality_report",
    "LineSpec", "LineFitConfig", "LineFitResult", "fit_line", "fit_lines",
    "plot_fit_referee", "plot_line_fit", "plot_xsl_validation_payload",
    "KNOWN_RESIDUAL_WINDOWS", "annotate_nonstellar_features",
    "diagnose_known_residual_windows",
    "normalize_segment_sidebands", "normalize_segments_sidebands",
    "sdss_quicklook_resolution_assumption",
    "ExclusionMaskSpec", "archive_exclusion_masks",
    "archive_exclusion_masks_for_segment", "archive_mask_catalog",
    "archive_mask_summary_for_segment", "artifact_exclusion_mask_from_segment",
    "audit_spectrum_for_fit",
    "combine_exclusion_masks", "dilate_boolean_mask",
    "exclusion_mask",
    "NONSTELLAR_FEATURES", "OPTICAL_DIB_DIAGNOSTIC_FEATURES",
    "OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES",
    "known_feature_masks", "nonstellar_feature_mask", "nonstellar_feature_masks",
    "nonstellar_feature_metadata", "nonstellar_feature_regions",
    "overlapping_nonstellar_features", "broad_telluric_catalog_fallback_mask",
    "telluric_transmission_exclusion_mask",
    "convert_wavelength_medium", "convert_segment_wavelength_medium",
    "FitProgressEvent",
]
