"""High-level, instrument-independent Spyctres fitting API."""

from .config import resolve_phoenix_dir
from .fitting import (
    fit_phoenix_full_spectrum,
    reconstruct_phoenix_legendre_models_for_segments,
)
from .io import coerce_spectrum
from .phoenix import PhoenixLibrary
from .results import PhoenixFitResult


def fit_phoenix_spectrum(
    spectrum,
    phoenix_lib=None,
    phoenix_dir=None,
    reconstruct=True,
    warn_unknown=True,
    **fit_kwargs,
):
    """Fit any canonicalizable spectrum and return a structured result."""
    canonical = coerce_spectrum(spectrum, warn_unknown=warn_unknown, source="public_api")
    if phoenix_lib is None:
        resolved = resolve_phoenix_dir(phoenix_dir)
        if resolved is None:
            raise ValueError(
                "No PHOENIX directory configured; pass phoenix_dir or phoenix_lib."
            )
        phoenix_lib = PhoenixLibrary(
            resolved, verbose=bool(fit_kwargs.get("verbose", 0))
        )
    elif phoenix_dir is not None:
        raise ValueError("Pass phoenix_lib or phoenix_dir, not both.")

    summary = fit_phoenix_full_spectrum(canonical, phoenix_lib=phoenix_lib, **fit_kwargs)
    models = coefficients = used_masks = excluded_masks = ()
    if reconstruct:
        reconstruction_keys = {
            key: fit_kwargs[key]
            for key in (
                "regions",
                "exclude_regions",
                "exclude_mask",
                "mdeg",
                "rv_bary_kms",
                "R",
                "fwhm_kms",
                "forward_model",
                "model_margin_A",
            )
            if key in fit_kwargs
        }
        models, coefficients, used_masks, excluded_masks = (
            reconstruct_phoenix_legendre_models_for_segments(
                canonical,
                phoenix_lib=phoenix_lib,
                fit_result=summary,
                **reconstruction_keys,
            )
        )

    provenance = {
        "api": "fit_phoenix_spectrum",
        "rv_convention": "positive rv_kms redshifts a receding stellar spectrum",
        "rv_bary_explicit": True,
        "spectrum_schema_version": 1,
        "cache_path": fit_kwargs.get("cache_path"),
    }
    return PhoenixFitResult(
        summary=dict(summary),
        models=tuple(models),
        continuum_coefficients=tuple(coefficients),
        used_masks=tuple(used_masks),
        excluded_masks=tuple(excluded_masks),
        provenance=provenance,
    )
