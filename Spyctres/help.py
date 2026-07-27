"""Small user-facing help registry for public Spyctres entry points.

The package intentionally keeps a broad top-level namespace during alpha so
older workflows and expert notebooks remain usable.  This module provides a
curated map over that namespace: ordinary users see the one-import path first,
while advanced diagnostics remain discoverable without being presented as the
minimal workflow.
"""

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
            'fit_stellar_spectrum(spec, model="phoenix", setup=setup)'
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
                "description": "Let Spyctres suggest first-pass bounds, windows, and RV scan. Use suggest_fit_setup(spec) to inspect those assumptions before fitting.",
            },
            {
                "name": "setup",
                "default": "None",
                "description": (
                    "Reviewed FitSetup from suggest_fit_setup(spec). When "
                    "supplied, the fitter uses those exact fit_kwargs and "
                    "embeds the setup hash in the result."
                ),
            },
            {
                "name": "defaults_mode",
                "default": "quicklook",
                "description": "Search-budget mode: quicklook, standard, or diagnostic. The alias mode=... is also accepted.",
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
                "name": "mask / exclude_masks / exclude_regions",
                "default": "None",
                "description": "Explicit masks or wavelength ranges to exclude from the fit; mask=... is the beginner-facing alias for exclude_masks.",
            },
            {
                "name": "resolution_R / R or fwhm_kms",
                "default": "metadata/None",
                "description": "Constant Gaussian instrumental broadening assumption; resolution_R=... is the beginner-facing alias for R.",
            },
            {
                "name": "continuum_degree / p0, bounds, rv_grid_n, multistart, mdeg, max_nfev",
                "default": "auto/default",
                "description": "Expert controls for initialization, search bounds, continuum degree, and optimizer budget; continuum_degree=... aliases mdeg.",
            },
        ],
        "advice": (
            "For an auditable path, run setup = suggest_fit_setup(spec), inspect "
            "setup.summary(), then call fit_stellar_spectrum(spec, setup=setup)."
        ),
    },
    "suggest_fit_setup": {
        "name": "suggest_fit_setup",
        "purpose": "Inspect Spyctres' first-pass fitting assumptions without running PHOENIX.",
        "minimal_call": "suggest_fit_setup(spec)",
        "required": [
            {
                "name": "spectrum",
                "description": "A loaded SpectrumSegment or SpectrumCollection.",
            },
        ],
        "optional": [
            {
                "name": "model",
                "default": "phoenix",
                "description": "Model backend. The current public setup helper supports PHOENIX.",
            },
            {
                "name": "mode",
                "default": "quicklook",
                "description": "Search-budget mode: quicklook, standard, or diagnostic.",
            },
            {
                "name": "include_readiness",
                "default": "True",
                "description": "Include a pre-fit readiness audit over the suggested fit regions.",
            },
            {
                "name": "readiness_intent",
                "default": "classification/quicklook",
                "description": (
                    "Task-specific readiness policy: inspect, "
                    "quicklook_classification, atmospheric_parameters, "
                    "radial_velocity, or publication."
                ),
            },
            {
                "name": "assumed_resolution / exclude_masks",
                "default": "None",
                "description": "Optional assumptions passed into the readiness audit, without mutating the spectrum.",
            },
        ],
        "advice": "Use this immediately after read_spectrum(); inspect setup.summary(), then pass setup=setup to fit_stellar_spectrum() when ready.",
    },
    "readiness_flag_actions": {
        "name": "readiness_flag_actions",
        "purpose": "Translate readiness-audit flags into short next actions for users.",
        "minimal_call": 'readiness_flag_actions(["wave_medium_unknown"])',
        "required": [
            {
                "name": "flags",
                "description": (
                    "Iterable of readiness flag strings, normally from "
                    "audit_spectrum_for_fit(...)[\"interpretation_flags\"]."
                ),
            },
        ],
        "optional": [],
        "advice": (
            "Most users see these actions through suggest_fit_setup() or the "
            "external validation CSV; call this directly when building custom "
            "notebooks or GUIs."
        ),
    },
    "audit_spectrum_for_fit": {
        "name": "audit_spectrum_for_fit",
        "purpose": "Audit whether a loaded spectrum is ready for a specific fitting or inspection intent.",
        "minimal_call": 'audit_spectrum_for_fit(spec, intent="quicklook_classification")',
        "required": [
            {
                "name": "spectrum",
                "description": "A loaded SpectrumSegment or SpectrumCollection.",
            },
        ],
        "optional": [
            {
                "name": "intent",
                "default": "quicklook_classification",
                "description": (
                    "Task-specific policy: inspect, quicklook_classification, "
                    "atmospheric_parameters, radial_velocity, or publication."
                ),
            },
            {
                "name": "fit_windows / regions",
                "default": "full coverage",
                "description": "Wavelength windows over which readiness should be judged.",
            },
            {
                "name": "assumed_resolution",
                "default": "None",
                "description": "Explicit resolution assumption for the audit, without mutating the spectrum.",
            },
            {
                "name": "exclude_masks",
                "default": "None",
                "description": "Explicit masks to apply when counting fitted pixels.",
            },
        ],
        "advice": (
            "Use ready_for_intent and blockers_for_intent for the task at hand; "
            "strict fit_ready remains a conservative/backward-compatible gate."
        ),
    },
    "classify_spectrum": {
        "name": "classify_spectrum",
        "purpose": (
            "Friendly alias for fit_stellar_spectrum() in exploratory PHOENIX "
            "classification workflows; it is not a formal MK classifier."
        ),
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
        "advice": (
            "Use this when the mental model is first-pass classification. Use "
            "fit_stellar_spectrum() plus a reviewed setup when writing auditable "
            "or publication-oriented code."
        ),
    },
    "plot_spectrum": {
        "name": "plot_spectrum",
        "purpose": "Make a quick first-look plot of a loaded spectrum.",
        "minimal_call": "plot_spectrum(spec)",
        "required": [
            {
                "name": "spectrum",
                "description": "SpectrumSegment, SpectrumCollection, or compatible spectrum.",
            },
        ],
        "optional": [
            {
                "name": "show_masks",
                "default": "False",
                "description": "Switch to the three-panel audit plot showing raw flux, normalized flux, and mask status.",
            },
            {
                "name": "mask",
                "default": "None",
                "description": "Optional MaskBundle or exclusion-mask callable to overlay explicitly rejected regions.",
            },
            {
                "name": "show_tellurics / show_nonstellar",
                "default": "False",
                "description": "Overlay broad warning regions for telluric or non-stellar features.",
            },
        ],
        "advice": "Call plot_spectrum(spec) immediately after read_spectrum(); use plot_fit_referee(result) after fitting.",
    },
    "plot_diagnostic_windows": {
        "name": "plot_diagnostic_windows",
        "purpose": "Plot advisory diagnostic-window coverage on a spectrum.",
        "minimal_call": "plot_diagnostic_windows(spec)",
        "required": [
            {
                "name": "spectrum",
                "description": "Loaded spectrum whose wavelength coverage should be inspected.",
            },
        ],
        "optional": [
            {
                "name": "selection",
                "default": "auto",
                "description": "Precomputed select_diagnostic_windows() output.",
            },
            {
                "name": "plot_spectrum options",
                "default": "varies",
                "description": "Options such as mask, show_tellurics, show_nonstellar, title, and max_plot_points.",
            },
        ],
        "advice": "Diagnostic windows are suggestions for inspection and controlled fits; they are not spectral-type labels.",
    },
    "select_diagnostic_windows": {
        "name": "select_diagnostic_windows",
        "purpose": "Select advisory diagnostic windows from wavelength coverage and metadata.",
        "minimal_call": "select_diagnostic_windows(spec)",
        "required": [
            {
                "name": "spectrum",
                "description": "Loaded spectrum to inspect.",
            },
        ],
        "optional": [
            {
                "name": "roles / branches",
                "default": "auto",
                "description": "Advanced filters for diagnostic-window families and branch candidates.",
            },
        ],
        "advice": (
            "Use the returned windows for inspection and controlled fit plans; "
            "they are suggestions, not hidden preprocessing corrections."
        ),
    },
    "build_mask": {
        "name": "build_mask",
        "purpose": "Create an explicit bundle of named masks and warning regions.",
        "minimal_call": 'build_mask(spec, archive=True, tellurics="warn")',
        "required": [
            {
                "name": "spectrum",
                "description": "Optional but recommended; reader metadata can provide archive/product bad-region catalogs.",
            },
        ],
        "optional": [
            {
                "name": "archive",
                "default": "False",
                "description": "True/'mask' applies archive masks; 'warn' records archive warning regions only.",
            },
            {
                "name": "tellurics",
                "default": "warn",
                "description": "Use 'warn', 'mask' for the transmission model, 'fallback' for broad catalog masks, or 'none'.",
            },
            {
                "name": "dibs / names",
                "default": "False / None",
                "description": "Explicit known non-stellar feature masks, such as DIB regions.",
            },
        ],
        "advice": "Warning regions are not applied. Pass the returned bundle explicitly as exclude_masks=mask.",
    },
    "fit_line": {
        "name": "fit_line",
        "purpose": "Fit one local spectral line as a fast diagnostic.",
        "minimal_call": 'fit_line(spec, "Hgamma")',
        "required": [
            {
                "name": "spectrum",
                "description": "SpectrumSegment or SpectrumCollection containing the line.",
            },
            {
                "name": "line or center",
                "description": "Known line name such as Hgamma, or center=4340.47 for a custom wavelength.",
            },
        ],
        "optional": [
            {
                "name": "config",
                "default": "LineFitConfig()",
                "description": "Expert local-continuum/retry/bounds settings.",
            },
            {
                "name": "kind, window_A, wave_medium",
                "default": "line default",
                "description": "Override the line type, fitting half-window, or line wavelength medium.",
            },
        ],
        "advice": "Use fit_line for local diagnostics and seeding; use fit_stellar_spectrum for atmospheric parameters.",
    },
    "plot_line_fit": {
        "name": "plot_line_fit",
        "purpose": "Plot the result of one local line fit.",
        "minimal_call": "plot_line_fit(line_result)",
        "required": [
            {
                "name": "line_result",
                "description": "LineFitResult returned by fit_line() or fit_lines().",
            },
        ],
        "optional": [
            {
                "name": "matplotlib options",
                "default": "varies",
                "description": "Optional title, axis, and display/save controls.",
            },
        ],
        "advice": "Use line plots to diagnose individual features; do not promote them to global stellar parameters by themselves.",
    },
    "plot_fit_referee": {
        "name": "plot_fit_referee",
        "purpose": "Create the standard observed/model/residual plot for a PHOENIX fit result.",
        "minimal_call": "plot_fit_referee(result)",
        "required": [
            {
                "name": "result",
                "description": "PhoenixFitResult returned by fit_stellar_spectrum() or fit_phoenix_spectrum().",
            },
        ],
        "optional": [
            {
                "name": "savepath",
                "default": "None",
                "description": "Optional output path; parent directories are created defensively.",
            },
            {
                "name": "plot_layout / plot_xlim",
                "default": "stacked / fit",
                "description": "User-facing controls for wide stacked plots and fit-window/full-range display.",
            },
        ],
        "advice": "Use this as the first residual check after every fit; quality flags still need to be read.",
    },
    "compare_fits": {
        "name": "compare_fits",
        "purpose": "Compare two or more Spyctres fit results for stability and quality review.",
        "minimal_call": 'compare_fits(result_a, result_b, labels=("baseline", "variant"))',
        "required": [
            {
                "name": "fit results",
                "description": "Two or more PhoenixFitResult objects or result-like mappings.",
            },
        ],
        "optional": [
            {
                "name": "labels",
                "default": "auto",
                "description": "Short labels used in returned comparison tables.",
            },
        ],
        "advice": (
            "Use comparisons to check sensitivity to masks, windows, resolution, "
            "or setup mode; the lowest raw chi-square alone is not a scientific winner."
        ),
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


PUBLIC_FUNCTION_GROUPS = {
    "beginner": {
        "title": "Beginner one-import path",
        "description": (
            "The smallest practical path for reading a spectrum, reviewing the "
            "setup, running a first-pass PHOENIX fit, and plotting diagnostics."
        ),
        "functions": (
            "read_spectrum",
            "plot_spectrum",
            "suggest_fit_setup",
            "fit_stellar_spectrum",
            "plot_fit_referee",
        ),
    },
    "readiness_and_masks": {
        "title": "Readiness, masks, and diagnostic windows",
        "description": (
            "Tools for making wavelength/frame, uncertainty, LSF, mask, and "
            "window assumptions visible before or beside a fit."
        ),
        "functions": (
            "audit_spectrum_for_fit",
            "readiness_flag_actions",
            "select_diagnostic_windows",
            "plot_diagnostic_windows",
            "build_mask",
        ),
    },
    "line_diagnostics": {
        "title": "Local line diagnostics",
        "description": (
            "Fast local line checks used for feature inspection, RV clues, and "
            "quality diagnostics; not replacements for atmospheric fitting."
        ),
        "functions": (
            "fit_line",
            "plot_line_fit",
        ),
    },
    "fit_review": {
        "title": "Fit review and comparison",
        "description": (
            "Result plotting and stability checks after one or more fits have "
            "been run."
        ),
        "functions": (
            "plot_fit_referee",
            "compare_fits",
        ),
    },
    "advanced": {
        "title": "Advanced and compatibility entry points",
        "description": (
            "Lower-level or compatibility calls for expert scripts and legacy "
            "migration; beginners should usually start elsewhere."
        ),
        "functions": (
            "fit_phoenix_spectrum",
            "classify_spectrum",
        ),
    },
}


PUBLIC_FUNCTION_GROUP_ALIASES = {
    "first_steps": "beginner",
    "quickstart": "beginner",
    "one_import": "beginner",
    "masks": "readiness_and_masks",
    "readiness": "readiness_and_masks",
    "diagnostic_windows": "readiness_and_masks",
    "lines": "line_diagnostics",
    "line_fitting": "line_diagnostics",
    "review": "fit_review",
    "comparison": "fit_review",
    "expert": "advanced",
    "compatibility": "advanced",
}


def _normalize_group_name(group):
    key = str(group).strip().lower().replace("-", "_").replace(" ", "_")
    key = PUBLIC_FUNCTION_GROUP_ALIASES.get(key, key)
    if key not in PUBLIC_FUNCTION_GROUPS:
        known = ", ".join(list_public_function_groups())
        raise ValueError(
            "Unknown public function group '{0}'. Known groups: {1}.".format(
                group,
                known,
            )
        )
    return key


def list_public_function_groups():
    """Return names of curated public-function groups."""
    return list(PUBLIC_FUNCTION_GROUPS)


def describe_public_function_group(group):
    """Return a JSON-safe help record for a curated public-function group."""
    key = _normalize_group_name(group)
    record = PUBLIC_FUNCTION_GROUPS[key]
    return {
        "name": key,
        "title": record["title"],
        "description": record["description"],
        "functions": list(record["functions"]),
    }


def list_public_functions(group=None):
    """Return public functions with Spyctres call-help metadata.

    With no group, this returns the full sorted help-topic list.  With a group
    such as ``"beginner"`` or ``"readiness"``, it returns the curated order for
    that user-facing workflow.
    """
    if group is not None:
        return describe_public_function_group(group)["functions"]
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


def format_public_api_guide(group=None):
    """Return a compact guide to the curated Spyctres public API."""
    lines = [
        "Spyctres public API guide",
        "",
        "Recommended one-import path:",
        "  import Spyctres as sp",
        '  spec = sp.read_spectrum("my_spectrum.fits", instrument="xshooter")',
        "  setup = sp.suggest_fit_setup(spec)",
        '  result = sp.fit_stellar_spectrum(spec, model="phoenix", setup=setup)',
        "  sp.plot_fit_referee(result)",
        "",
    ]
    groups = [describe_public_function_group(group)] if group else [
        describe_public_function_group(name)
        for name in list_public_function_groups()
    ]
    lines.append("Function groups:")
    for item in groups:
        lines.append("  {0}: {1}".format(item["name"], item["title"]))
        lines.append("    {0}".format(item["description"]))
        lines.append("    functions: {0}".format(", ".join(item["functions"])))
    lines.extend(
        [
            "",
            "For details on one function:",
            "  sp.format_public_function_help('fit_stellar_spectrum')",
        ]
    )
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
