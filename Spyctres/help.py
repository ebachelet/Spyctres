"""Small user-facing help registry for public Spyctres entry points."""

from __future__ import annotations


PUBLIC_FUNCTION_HELP = {
    "read_spectrum": {
        "name": "read_spectrum",
        "purpose": "Read a reduced one-dimensional spectrum into the common Spyctres spectrum container.",
        "minimal_call": 'read_spectrum("spectrum.fits", instrument="xshooter")',
        "required": [
            {
                "name": "path",
                "description": "Path to the input spectrum file.",
            },
            {
                "name": "instrument",
                "description": (
                    "Registered reader name or alias, e.g. xshooter, pepsi, "
                    "floyds, gemini, sdss, uves_pop, or xsl."
                ),
            },
        ],
        "optional": [
            {
                "name": "warn_unknown",
                "default": "True",
                "description": "Warn when common-format metadata are unknown.",
            },
            {
                "name": "reader-specific kwargs",
                "default": "varies",
                "description": (
                    "Extra options passed to the selected reader, such as "
                    "SDSS mask policy or PEPSI product profile."
                ),
            },
        ],
        "advice": "Choose a reader with list_instruments() or inspect details with get_instrument_info('xshooter').",
    },
    "fit_stellar_spectrum": {
        "name": "fit_stellar_spectrum",
        "purpose": "Run the recommended public PHOENIX stellar-spectrum fitting workflow.",
        "minimal_call": (
            'fit_stellar_spectrum("spectrum.fits", instrument="xshooter", '
            'phoenix_dir="/path/to/PHOENIX")'
        ),
        "required": [
            {
                "name": "spectrum",
                "description": (
                    "A path, SpectrumSegment, SpectrumCollection, or compatible "
                    "array-like spectrum."
                ),
            },
            {
                "name": "instrument",
                "description": (
                    "Required only when spectrum is a path; tells Spyctres "
                    "which reader to use."
                ),
            },
            {
                "name": "PHOENIX source",
                "description": (
                    "Pass phoenix_dir, pass phoenix_lib, or configure "
                    "SPYCTRES_PHOENIX_DIR / the Spyctres config file."
                ),
            },
        ],
        "optional": [
            {
                "name": "model",
                "default": "phoenix",
                "description": "Model backend. The alpha public workflow supports PHOENIX.",
            },
            {
                "name": "auto_defaults",
                "default": "True",
                "description": "Let Spyctres suggest first-pass bounds, windows, and RV scan.",
            },
            {
                "name": "defaults_mode",
                "default": "quicklook",
                "description": "Search-budget mode: quicklook, standard, or diagnostic.",
            },
            {
                "name": "reader_kwargs",
                "default": "None",
                "description": "Dictionary of reader-specific options when spectrum is a path.",
            },
            {
                "name": "reconstruct",
                "default": "True",
                "description": "Reconstruct best-fit model arrays for plotting.",
            },
            {
                "name": "progress_callback",
                "default": "None",
                "description": "Callable receiving progress events during long PHOENIX operations.",
            },
            {
                "name": "regions",
                "default": "auto",
                "description": "List of wavelength windows to fit, e.g. [(3800, 5200)].",
            },
            {
                "name": "exclude_masks / exclude_regions",
                "default": "None",
                "description": "Explicit masks or wavelength ranges to exclude from the fit.",
            },
            {
                "name": "R or fwhm_kms",
                "default": "metadata/None",
                "description": "Constant Gaussian instrumental broadening assumption.",
            },
            {
                "name": "p0, bounds, rv_grid_n, multistart, mdeg, max_nfev",
                "default": "auto/default",
                "description": "Expert controls for initialization, search bounds, continuum degree, and optimizer budget.",
            },
        ],
        "advice": (
            "For a first run, use examples/simple_phoenix_fit.py; for many spectra, "
            "use examples/batch_quickscan_then_refine.py."
        ),
    },
    "classify_spectrum": {
        "name": "classify_spectrum",
        "purpose": "Alias for fit_stellar_spectrum() for first-pass classification workflows.",
        "minimal_call": (
            'classify_spectrum("spectrum.fits", instrument="xshooter", '
            'phoenix_dir="/path/to/PHOENIX")'
        ),
        "required": [
            {
                "name": "spectrum",
                "description": "Same as fit_stellar_spectrum().",
            },
        ],
        "optional": [
            {
                "name": "all fit_stellar_spectrum options",
                "default": "same",
                "description": "classify_spectrum forwards arguments to fit_stellar_spectrum.",
            },
        ],
        "advice": "Use describe_public_function('fit_stellar_spectrum') for the full option list.",
    },
    "fit_phoenix_spectrum": {
        "name": "fit_phoenix_spectrum",
        "purpose": "Lower-level PHOENIX fit wrapper for an already canonicalized spectrum.",
        "minimal_call": (
            'fit_phoenix_spectrum(spec, phoenix_dir="/path/to/PHOENIX", '
            'p0=(6000, 0, 4, 0))'
        ),
        "required": [
            {
                "name": "spectrum",
                "description": "SpectrumSegment, SpectrumCollection, or compatible spectrum.",
            },
            {
                "name": "PHOENIX source",
                "description": "Pass phoenix_dir, pass phoenix_lib, or configure PHOENIX.",
            },
            {
                "name": "fit kwargs",
                "description": "At minimum p0/bounds or caller-prepared PHOENIX fit kwargs.",
            },
        ],
        "optional": [
            {
                "name": "reconstruct",
                "default": "True",
                "description": "Reconstruct model arrays after a successful fit.",
            },
            {
                "name": "warn_unknown",
                "default": "True",
                "description": "Warn about unknown common-format metadata.",
            },
            {
                "name": "fit_phoenix_full_spectrum kwargs",
                "default": "varies",
                "description": "Expert fitting controls such as regions, masks, R, p0, bounds, and max_nfev.",
            },
        ],
        "advice": "Most users should call fit_stellar_spectrum() instead.",
    },
}


def list_public_functions():
    """Return public functions with Spyctres call-help metadata."""
    return sorted(PUBLIC_FUNCTION_HELP)


def describe_public_function(name):
    """Return a JSON-safe help record for a public Spyctres function."""
    key = str(name).strip()
    if key not in PUBLIC_FUNCTION_HELP:
        known = ", ".join(list_public_functions())
        raise ValueError(
            "Unknown public function '{0}'. Known help topics: {1}.".format(
                name,
                known,
            )
        )
    record = PUBLIC_FUNCTION_HELP[key]
    return {
        "name": record["name"],
        "purpose": record["purpose"],
        "minimal_call": record["minimal_call"],
        "required": [dict(item) for item in record["required"]],
        "optional": [dict(item) for item in record["optional"]],
        "advice": record["advice"],
    }


def format_public_function_help(name):
    """Return a compact human-readable help string for a public function."""
    record = describe_public_function(name)
    lines = [
        "{0}: {1}".format(record["name"], record["purpose"]),
        "Minimal call:",
        "  {0}".format(record["minimal_call"]),
        "Required:",
    ]
    for item in record["required"]:
        lines.append("  - {0}: {1}".format(item["name"], item["description"]))
    lines.append("Optional extras:")
    for item in record["optional"]:
        default = item.get("default", None)
        suffix = "" if default is None else " [default: {0}]".format(default)
        lines.append(
            "  - {0}{1}: {2}".format(
                item["name"],
                suffix,
                item["description"],
            )
        )
    lines.append("Advice:")
    lines.append("  {0}".format(record["advice"]))
    return "\n".join(lines)


def missing_call_error(name, reason=None):
    """Return a helpful error message for an incomplete public-function call."""
    record = describe_public_function(name)
    parts = []
    if reason:
        parts.append(str(reason))
    else:
        parts.append("Not enough information was supplied.")
    parts.append("Minimal call: {0}".format(record["minimal_call"]))
    parts.append("Advice: {0}".format(record["advice"]))
    parts.append(
        "List options with: describe_public_function('{0}') "
        "or format_public_function_help('{0}').".format(record["name"])
    )
    return "\n".join(parts)
