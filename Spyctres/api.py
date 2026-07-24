"""High-level, instrument-independent Spyctres fitting API."""

import inspect
import os
from collections.abc import Mapping

from .config import resolve_phoenix_dir
from .defaults import FitSetup, prepare_phoenix_fit_kwargs
from .help import missing_call_error
from .fitting import (
    fit_phoenix_full_spectrum,
    reconstruct_phoenix_legendre_models_for_segments,
)
from .io import coerce_spectrum, read_spectrum
from .phoenix import PhoenixLibrary
from .results import PhoenixFitResult


_RECONSTRUCTION_KEYS = (
    set(inspect.signature(fit_phoenix_full_spectrum).parameters)
    & set(
        inspect.signature(
            reconstruct_phoenix_legendre_models_for_segments
        ).parameters
    )
)


def _setup_payload(setup):
    if setup is None:
        return None
    if isinstance(setup, FitSetup):
        return setup.to_dict()
    if hasattr(setup, "to_dict"):
        payload = setup.to_dict()
    elif isinstance(setup, Mapping):
        payload = dict(setup)
    else:
        raise TypeError("setup must be a FitSetup, mapping, or to_dict()-compatible object.")
    if not isinstance(payload, Mapping):
        raise TypeError("setup.to_dict() must return a mapping.")
    if str(payload.get("operation", "")) != "suggest_fit_setup":
        raise ValueError("setup must come from suggest_fit_setup().")
    fit_kwargs = payload.get("fit_kwargs")
    if not isinstance(fit_kwargs, Mapping):
        raise ValueError("setup payload must contain a fit_kwargs mapping.")
    if "setup_hash" not in payload:
        payload = FitSetup(dict(payload)).to_dict()
    return dict(payload)


def fit_phoenix_spectrum(
    spectrum=None,
    phoenix_lib=None,
    phoenix_dir=None,
    reconstruct=True,
    warn_unknown=True,
    **fit_kwargs,
):
    """Fit any canonicalizable spectrum and return a structured result."""
    if spectrum is None:
        raise ValueError(missing_call_error("fit_phoenix_spectrum"))
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
    if reconstruct and summary.get("success", False):
        reconstruction_keys = {
            key: fit_kwargs[key]
            for key in _RECONSTRUCTION_KEYS
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
        "phoenix_source_root": getattr(phoenix_lib, "base_dir", None),
        "cache_schema_version": getattr(phoenix_lib, "CACHE_SCHEMA_VERSION", None),
        "reconstruction_requested": bool(reconstruct),
        "reconstruction_performed": bool(reconstruct and summary.get("success", False)),
    }
    return PhoenixFitResult(
        summary=dict(summary),
        models=tuple(models),
        continuum_coefficients=tuple(coefficients),
        used_masks=tuple(used_masks),
        excluded_masks=tuple(excluded_masks),
        provenance=provenance,
    )


def fit_stellar_spectrum(
    spectrum=None,
    instrument=None,
    model="phoenix",
    phoenix_lib=None,
    phoenix_dir=None,
    auto_defaults=True,
    defaults_mode="quicklook",
    mode=None,
    science_case="classification",
    setup=None,
    mask=None,
    resolution_R=None,
    continuum_degree=None,
    reader_kwargs=None,
    reconstruct=True,
    warn_unknown=True,
    progress_callback=None,
    **fit_kwargs,
):
    """Fit a reduced stellar spectrum with the recommended public workflow.

    This is the "out of the box" entry point for users who should not need to
    know the internal module layout. It accepts either an already-loaded
    ``SpectrumSegment``/``SpectrumCollection``/array-like spectrum, or a path
    plus an ``instrument`` reader name. For the current alpha workflow,
    ``model="phoenix"`` is supported.

    The function applies Spyctres' conservative first-pass PHOENIX defaults by
    default, records the reasons/warnings in the returned result, and lets any
    explicit fit keyword override the suggestion. For example, pass ``regions``,
    ``p0``, ``bounds``, ``rv_grid_n``, or ``mdeg`` to take expert control.
    To fit exactly a reviewed plan, pass ``setup=suggest_fit_setup(spec)`` and
    do not pass additional fit-control overrides.
    """
    if spectrum is None:
        raise ValueError(missing_call_error("fit_stellar_spectrum"))

    model_name = str(model).strip().lower()
    if model_name not in {"phoenix", "phoenix_hires", "phoenix-hires"}:
        raise ValueError(
            "fit_stellar_spectrum currently supports model='phoenix' only."
        )

    setup_payload = _setup_payload(setup)
    if setup_payload is not None:
        if not auto_defaults:
            raise ValueError("Pass setup or auto_defaults=False, not both.")
        setup_mode = str(setup_payload.get("mode", "")).strip().lower()
        requested_mode = str(mode if mode is not None else defaults_mode).strip().lower()
        if (mode is not None or requested_mode != "quicklook") and setup_mode:
            if requested_mode != setup_mode:
                raise ValueError(
                    "setup mode is {0!r}, but requested mode/defaults_mode is "
                    "{1!r}; build a new setup instead.".format(
                        setup_mode,
                        requested_mode,
                    )
                )
        if setup_mode:
            defaults_mode = setup_mode
        setup_science_case = str(
            setup_payload.get("science_case", "")
        ).strip().lower()
        requested_science_case = str(science_case).strip().lower()
        if (
            requested_science_case != "classification"
            and setup_science_case
            and requested_science_case != setup_science_case
        ):
            raise ValueError(
                "setup science_case is {0!r}, but requested science_case is "
                "{1!r}; build a new setup instead.".format(
                    setup_science_case,
                    requested_science_case,
                )
            )
        if setup_science_case:
            science_case = setup_science_case
        explicit_controls = []
        if mask is not None:
            explicit_controls.append("mask")
        if resolution_R is not None:
            explicit_controls.append("resolution_R")
        if continuum_degree is not None:
            explicit_controls.append("continuum_degree")
        explicit_controls.extend(sorted(fit_kwargs))
        if explicit_controls:
            raise ValueError(
                "Pass setup or explicit fit-control overrides, not both. "
                "Build a new setup if you want to change: {0}.".format(
                    ", ".join(explicit_controls)
                )
            )

    if mode is not None:
        if (
            str(defaults_mode).strip().lower() != "quicklook"
            and str(defaults_mode).strip().lower() != str(mode).strip().lower()
        ):
            raise ValueError("Pass mode or defaults_mode, not conflicting values.")
        defaults_mode = str(mode).strip().lower()

    if mask is not None:
        if "exclude_mask" in fit_kwargs or "exclude_masks" in fit_kwargs:
            raise ValueError(
                "Pass mask or exclude_mask(s), not both. mask is the "
                "beginner-facing alias for exclude_masks."
            )
        fit_kwargs["exclude_masks"] = mask
    if resolution_R is not None:
        if "R" in fit_kwargs or "fwhm_kms" in fit_kwargs:
            raise ValueError(
                "Pass resolution_R or R/fwhm_kms, not multiple resolution aliases."
            )
        fit_kwargs["R"] = float(resolution_R)
    if continuum_degree is not None:
        if "mdeg" in fit_kwargs:
            raise ValueError("Pass continuum_degree or mdeg, not both.")
        fit_kwargs["mdeg"] = int(continuum_degree)

    reader_kwargs = {} if reader_kwargs is None else dict(reader_kwargs)
    input_was_path = isinstance(spectrum, (str, os.PathLike))
    if input_was_path:
        if instrument is None:
            raise ValueError(
                missing_call_error(
                    "fit_stellar_spectrum",
                    "A spectrum path was supplied, but no instrument reader was "
                    "specified.",
                )
            )
        canonical = read_spectrum(
            spectrum,
            instrument=instrument,
            warn_unknown=warn_unknown,
            **reader_kwargs,
        )
    else:
        canonical = coerce_spectrum(
            spectrum,
            warn_unknown=warn_unknown,
            source="fit_stellar_spectrum",
        )

    if setup_payload is not None:
        setup_model = str(setup_payload.get("model", "phoenix")).strip().lower()
        if setup_model != "phoenix":
            raise ValueError("fit_stellar_spectrum currently supports PHOENIX setups only.")
        resolved_fit_kwargs = dict(setup_payload["fit_kwargs"])
        suggestion = None
    else:
        resolved_fit_kwargs, suggestion = prepare_phoenix_fit_kwargs(
            canonical,
            auto_defaults=auto_defaults,
            defaults_mode=defaults_mode,
            science_case=science_case,
            extra_kwargs=fit_kwargs,
        )

    if progress_callback is not None:
        if "progress_callback" in resolved_fit_kwargs:
            raise ValueError(
                "Pass progress_callback either as a named argument or in "
                "fit_kwargs/setup, not both."
            )
        resolved_fit_kwargs["progress_callback"] = progress_callback

    result = fit_phoenix_spectrum(
        canonical,
        phoenix_lib=phoenix_lib,
        phoenix_dir=phoenix_dir,
        reconstruct=reconstruct,
        warn_unknown=warn_unknown,
        **resolved_fit_kwargs,
    )
    if setup_payload is not None:
        result.summary["fit_setup"] = setup_payload
        result.summary["fit_setup_hash"] = setup_payload.get("setup_hash")
    if suggestion is not None:
        result.summary["fit_default_suggestion"] = suggestion.to_dict()
    result.provenance.update(
        {
            "workflow_api": "fit_stellar_spectrum",
            "workflow_model": "phoenix",
            "input_was_path": bool(input_was_path),
            "instrument": None if instrument is None else str(instrument),
            "auto_defaults": bool(auto_defaults),
            "defaults_mode": str(defaults_mode),
            "science_case": str(science_case),
            "fit_setup_source": (
                "explicit_setup" if setup_payload is not None else "auto_defaults"
            ),
            "fit_setup_hash": None
            if setup_payload is None
            else setup_payload.get("setup_hash"),
        }
    )
    return result


def classify_spectrum(*args, **kwargs):
    """Alias for :func:`fit_stellar_spectrum` for classification workflows."""
    if not args and "spectrum" not in kwargs:
        raise ValueError(missing_call_error("classify_spectrum"))
    return fit_stellar_spectrum(*args, **kwargs)
