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
    describe_quality_flags,
    format_fit_quality_report,
)
from .linefitting import LineSpec, LineFitConfig, LineFitResult, fit_line, fit_lines
from .plotting import plot_fit_referee, plot_line_fit, plot_xsl_validation_payload
from .preprocessing import (
    NONSTELLAR_FEATURES,
    ExclusionMaskSpec,
    exclusion_mask,
    nonstellar_feature_mask,
    nonstellar_feature_metadata,
    nonstellar_feature_regions,
    overlapping_nonstellar_features,
)
from .waveutils import convert_segment_wavelength_medium, convert_wavelength_medium
from .fitting import FitProgressEvent
from .io import SpectrumCollection, SpectrumSegment, read_spectrum

__all__ = [
    "fit_phoenix_spectrum", "fit_stellar_spectrum", "classify_spectrum",
    "read_spectrum", "SpectrumSegment", "SpectrumCollection",
    "ensure_matplotlib_config_dir",
    "PhoenixFitDiagnostics", "PhoenixFitResult",
    "PhoenixFitDefaults", "clip_grid_to_bounds",
    "prepare_phoenix_fit_kwargs", "spectrum_wavelength_range",
    "suggest_phoenix_fit_defaults",
    "build_fit_quality_report", "describe_quality_flags",
    "format_fit_quality_report",
    "LineSpec", "LineFitConfig", "LineFitResult", "fit_line", "fit_lines",
    "plot_fit_referee", "plot_line_fit", "plot_xsl_validation_payload",
    "ExclusionMaskSpec", "exclusion_mask",
    "NONSTELLAR_FEATURES", "nonstellar_feature_mask",
    "nonstellar_feature_metadata", "nonstellar_feature_regions",
    "overlapping_nonstellar_features",
    "convert_wavelength_medium", "convert_segment_wavelength_medium",
    "FitProgressEvent",
]
