"""
Plotting utilities for Spyctres spectral diagnostics.

This module provides lightweight, reusable plotting helpers for:
- marking common spectral lines
- plotting full-spectrum data/model comparisons with residuals
- plotting zoomed windows around selected line centers
- plotting binned spectrum overviews for one or more segments

Design principles
-----------------
- use the Matplotlib object-oriented API
- do not call plt.show() internally
- return (fig, axes) so scripts and notebooks control display/saving
- accept plain arrays or simple spectrum containers, not instrument-specific logic
"""

import textwrap

import numpy as np
import matplotlib.pyplot as plt

from .fitting import evaluate_legendre_continuum


COMMON_LINES = {
    "balmer": [
        ("Hδ", 4101.74),
        ("Hγ", 4340.47),
        ("Hβ", 4861.33),
        ("Hα", 6562.80),
    ],
    "paschen": [
        ("Paγ", 10938.09),
        ("Paβ", 12818.08),
    ],
    "brackett": [
        ("Brγ", 21661.0),
    ],
    "caii": [
        ("Ca II K", 3933.66),
        ("Ca II H", 3968.47),
        ("Ca II", 8498.02),
        ("Ca II", 8542.09),
        ("Ca II", 8662.14),
    ],
    "nai": [
        ("Na I D1", 5895.92),
        ("Na I D2", 5889.95),
    ],
    "hei": [
        ("He I", 4471.48),
        ("He I", 5875.62),
        ("He I", 6678.15),
    ],
}


def plot_line_fit(result, figsize=(7.0, 5.0)):
    """Plot a ``LineFitResult`` and residuals without calling ``show``."""
    wave = np.asarray(result.wave, dtype=float)
    flux = np.asarray(result.flux, dtype=float)
    model = np.asarray(result.model_flux, dtype=float)
    continuum = np.asarray(result.continuum, dtype=float)
    residuals = np.asarray(result.residuals, dtype=float)
    if wave.size == 0:
        raise ValueError("LineFitResult has no fitted samples to plot.")
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(wave, flux, color="black", lw=1.0, label="data")
    axes[0].plot(wave, model, color="tab:red", lw=1.4, label="fit")
    axes[0].plot(wave, continuum, color="tab:blue", ls="--", lw=1.0, label="continuum")
    axes[0].axvline(result.center_wave, color="tab:red", alpha=0.5)
    axes[0].set_ylabel("Flux")
    axes[0].legend(loc="best")
    axes[0].set_title("{0}: {1}".format(result.line_name, ", ".join(result.flags)))
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].plot(wave, residuals, color="black", lw=0.9)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Wavelength [Angstrom]")
    return fig, axes


def _as_float_array(x):
    """Return x as a NumPy float array."""
    return np.asarray(x, dtype=float)


def _as_bool_array(x, n_expected=None):
    """Return x as a NumPy boolean array."""
    arr = np.asarray(x, dtype=bool)
    if n_expected is not None and arr.shape != (n_expected,):
        raise ValueError("Boolean mask has wrong shape.")
    return arr


def _compute_robust_ylim(y, lower=1.0, upper=99.0, pad_frac=0.05):
    """
    Compute robust y-limits from percentiles.

    Parameters
    ----------
    y : array-like
        Input data.
    lower, upper : float, optional
        Percentiles used for the robust interval.
    pad_frac : float, optional
        Fractional padding applied to the interval.

    Returns
    -------
    (ymin, ymax) : tuple of float
    """
    y = _as_float_array(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0

    lo, hi = np.nanpercentile(y, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(y)
        hi = np.nanmax(y)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0

    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _mask_to_spans(wave, mask):
    """
    Convert a boolean mask into contiguous wavelength spans.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    mask : array-like of bool
        Boolean mask on the same grid.

    Returns
    -------
    spans : list of tuple
        List of (wmin, wmax) spans for contiguous True regions.
    """
    wave = _as_float_array(wave)
    mask = _as_bool_array(mask, n_expected=wave.size)

    if wave.size == 0 or not mask.any():
        return []

    idx = np.flatnonzero(mask)
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

    spans = []
    for g in groups:
        if g.size == 0:
            continue
        spans.append((wave[g[0]], wave[g[-1]]))
    return spans


def _resolve_line_group(group):
    """
    Resolve a line-group specification into a list of (label, wavelength_A).

    Parameters
    ----------
    group : str or list
        Either a key in COMMON_LINES or a list of (label, wavelength_A) tuples.

    Returns
    -------
    list
        List of (label, wavelength_A) tuples.
    """
    if isinstance(group, str):
        return COMMON_LINES.get(group.lower(), [])
    return group


def _extract_spectrum_arrays(spec):
    """
    Extract wave/flux/mask/label from a simple spectrum container.

    Supported inputs:
    - SpectrumSegment-like object with .wave, .flux, optional .mask, optional .name
    - dict with keys 'wave', 'flux', optional 'mask', optional 'label' or 'name'

    Parameters
    ----------
    spec : object
        Spectrum container.

    Returns
    -------
    wave, flux, mask, label : tuple
    """
    if isinstance(spec, dict):
        wave = _as_float_array(spec["wave"])
        flux = _as_float_array(spec["flux"])
        mask = spec.get("mask", None)
        label = spec.get("label", spec.get("name", None))
    else:
        wave = _as_float_array(spec.wave)
        flux = _as_float_array(spec.flux)
        mask = getattr(spec, "mask", None)
        label = getattr(spec, "name", None)

    if mask is None:
        mask = np.isfinite(wave) & np.isfinite(flux)
    else:
        mask = _as_bool_array(mask, n_expected=wave.size)

    return wave, flux, mask, label


def mark_lines(ax, lines, xlim=None, ymin=0.0, ymax=0.98, alpha=0.25, fontsize=7):
    """
    Draw vertical markers for known spectral lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    lines : list of (label, wavelength_A)
        Spectral-line labels and wavelengths in Angstrom.
    xlim : tuple, optional
        Plot only lines within this wavelength range. If None, uses current x-limits.
    ymin, ymax : float, optional
        Vertical span of the markers in axes coordinates.
    alpha : float, optional
        Line/text transparency.
    fontsize : float, optional
        Text size for labels.
    """
    if xlim is None:
        xlim = ax.get_xlim()

    for label, lam in lines:
        if xlim[0] <= lam <= xlim[1]:
            ax.axvline(lam, ls="--", lw=1.0, alpha=alpha)
            ax.text(
                lam,
                ymax,
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=fontsize,
                alpha=alpha,
                transform=ax.get_xaxis_transform(),
            )


def bin_median_spectrum(wave, flux, good=None, nbins=600):
    """
    Bin a spectrum using the median flux in each wavelength bin.

    Parameters
    ----------
    wave, flux : array-like
        Input wavelength and flux arrays.
    good : array-like of bool, optional
        Valid-data mask. If None, finite wave/flux points are used.
    nbins : int, optional
        Number of wavelength bins.

    Returns
    -------
    wmid, fb : ndarray
        Bin-center wavelengths and binned median fluxes.
    """
    wave = _as_float_array(wave)
    flux = _as_float_array(flux)

    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have the same shape.")

    if good is None:
        good = np.isfinite(wave) & np.isfinite(flux)
    else:
        good = _as_bool_array(good, n_expected=wave.size)

    ww = wave[good]
    ff = flux[good]

    if ww.size == 0:
        return np.array([]), np.array([])

    idx = np.argsort(ww)
    ww = ww[idx]
    ff = ff[idx]

    bins = np.linspace(ww.min(), ww.max(), int(nbins) + 1)
    wmid = 0.5 * (bins[:-1] + bins[1:])
    fb = np.full(int(nbins), np.nan)

    for i in range(int(nbins)):
        m = (ww >= bins[i]) & (ww < bins[i + 1])
        if np.any(m):
            fb[i] = np.nanmedian(ff[m])

    ok = np.isfinite(fb)
    return wmid[ok], fb[ok]


def plot_binned_spectra(
    spectra,
    nbins=600,
    title=None,
    figsize=(7, 3),
    xlabel="Wavelength (Å)",
    ylabel="Flux (binned)",
    highlight_regions=None,
):
    """
    Plot one or more spectra after median binning.

    Parameters
    ----------
    spectra : list
        List of spectrum containers. Each element may be:
        - a SpectrumSegment-like object with .wave, .flux, optional .mask, optional .name
        - a dict with keys 'wave', 'flux', optional 'mask', optional 'label'
    nbins : int, optional
        Number of bins per spectrum.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size in inches.
    xlabel, ylabel : str, optional
        Axis labels.
    highlight_regions : list of (wmin, wmax), optional
        Wavelength spans to highlight with shaded regions.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    for spec in spectra:
        wave, flux, mask, label = _extract_spectrum_arrays(spec)
        wb, fb = bin_median_spectrum(wave, flux, good=mask, nbins=nbins)
        if wb.size == 0:
            continue
        ax.plot(wb, fb, lw=1.0, label=label)

    if highlight_regions is not None:
        for wmin, wmax in highlight_regions:
            ax.axvspan(wmin, wmax, alpha=0.15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    return fig, ax


def plot_full_spectrum_fit(
    wave,
    flux,
    err=None,
    model=None,
    used_mask=None,
    excluded_mask=None,
    title=None,
    line_groups=None,
    figsize=(10, 6),
    flux_label="Flux",
    data_label="Data",
    model_label="Model",
    residual_ylim=(-6, 6),
    shade_excluded=True,
):
    """
    Plot a full-spectrum fit with residuals.

    Parameters
    ----------
    wave, flux : array-like
        Observed wavelength and flux arrays.
    err : array-like or None, optional
        1-sigma uncertainties. If provided with `model`, normalized residuals
        `(flux - model) / err` are plotted. Otherwise raw residuals are shown.
    model : array-like, optional
        Model flux on the same wavelength grid.
    used_mask : array-like of bool, optional
        Mask of points used in the fit. If provided, points not used can be
        highlighted in the top panel.
    excluded_mask : array-like of bool, optional
        Additional exclusion mask. Typically this marks tellurics or
        user-excluded regions.
    title : str, optional
        Figure title.
    line_groups : list, optional
        List of line groups to mark. Each element may be:
        - a key in COMMON_LINES, e.g. "balmer"
        - a list of (label, wavelength_A)
    figsize : tuple, optional
        Figure size in inches.
    flux_label : str, optional
        Y-axis label for the top panel.
    data_label, model_label : str, optional
        Legend labels for data and model.
    residual_ylim : tuple, optional
        Y-limits for normalized residuals.
    shade_excluded : bool, optional
        If True, excluded regions are shaded. Otherwise excluded points are
        overplotted as markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple
        (ax_flux, ax_resid)
    """
    wave = _as_float_array(wave)
    flux = _as_float_array(flux)

    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have the same shape.")

    if err is not None:
        err = _as_float_array(err)
        if err.shape != wave.shape:
            raise ValueError("err must match wave shape.")

    if model is not None:
        model = _as_float_array(model)
        if model.shape != wave.shape:
            raise ValueError("model must match wave shape.")

    if used_mask is None:
        used_mask = np.ones_like(wave, dtype=bool)
    else:
        used_mask = _as_bool_array(used_mask, n_expected=wave.size)

    if excluded_mask is not None:
        excluded_mask = _as_bool_array(excluded_mask, n_expected=wave.size)

    fig, (ax_flux, ax_resid) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_flux.plot(wave, flux, lw=0.8, label=data_label)
    if model is not None:
        ax_flux.plot(wave, model, lw=0.8, label=model_label)

    if (~used_mask).any():
        ax_flux.plot(
            wave[~used_mask],
            flux[~used_mask],
            ".",
            ms=2,
            alpha=0.5,
            label="Unused",
        )

    if excluded_mask is not None and excluded_mask.any():
        if shade_excluded:
            for wmin, wmax in _mask_to_spans(wave, excluded_mask):
                ax_flux.axvspan(wmin, wmax, alpha=0.12)
                ax_resid.axvspan(wmin, wmax, alpha=0.12)
        else:
            ax_flux.plot(
                wave[excluded_mask],
                flux[excluded_mask],
                ".",
                ms=2,
                alpha=0.5,
                label="Excluded",
            )

    ylo, yhi = _compute_robust_ylim(flux[used_mask] if used_mask.any() else flux)
    ax_flux.set_ylim(ylo, yhi)
    ax_flux.set_ylabel(flux_label)

    handles, labels = ax_flux.get_legend_handles_labels()
    if labels:
        ax_flux.legend(frameon=False, loc="best")

    if title is not None:
        ax_flux.set_title(title)

    if model is not None:
        resid = flux - model
        ax_resid.axhline(0.0, lw=1.0, alpha=0.6)

        if err is not None:
            good = np.isfinite(err) & (err > 0)
            resid_plot = np.full_like(resid, np.nan, dtype=float)
            resid_plot[good] = resid[good] / err[good]
            ax_resid.plot(wave, resid_plot, lw=0.6)
            ax_resid.set_ylabel("(D-M)/σ")
            ax_resid.set_ylim(*residual_ylim)
        else:
            ax_resid.plot(wave, resid, lw=0.6)
            ax_resid.set_ylabel("D-M")
    else:
        ax_resid.axis("off")

    if line_groups is not None:
        xlim = (wave.min(), wave.max())
        for group in line_groups:
            lines = _resolve_line_group(group)
            mark_lines(ax_flux, lines, xlim=xlim, ymax=0.98)

    ax_resid.set_xlabel("Wavelength (Å)")
    fig.tight_layout()
    return fig, (ax_flux, ax_resid)


def _coerce_segment_sequence(segment):
    """Return a list of SpectrumSegment-like objects."""
    if segment is None:
        raise ValueError(
            "plot_fit_referee requires segment=... because fit results do not "
            "store observed flux arrays by default."
        )
    if hasattr(segment, "segments"):
        return list(segment.segments)
    if isinstance(segment, (list, tuple)):
        return list(segment)
    return [segment]


def _segment_plot_label(segment, index):
    name = getattr(segment, "name", None)
    return str(name) if name else "segment {0}".format(index + 1)


def _continuum_and_raw_model(wave, used_mask, corrected_model, coeffs):
    """Recover continuum and pre-continuum model when coefficients are available."""
    if coeffs is None:
        return None, None
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.size == 0:
        return None, None
    used = np.asarray(used_mask, dtype=bool)
    if used.shape != wave.shape or not np.any(used):
        return None, None
    continuum = evaluate_legendre_continuum(wave, wave[used], coeffs)
    finite = np.isfinite(continuum) & (continuum != 0.0)
    raw = np.full_like(corrected_model, np.nan, dtype=float)
    raw[finite] = corrected_model[finite] / continuum[finite]
    return continuum, raw


def _quality_text(result):
    flags = list(getattr(result, "quality_flags", result.get("quality_flags", [])))
    if not flags:
        flags = ["unknown"]
    report = result.quality_report() if hasattr(result, "quality_report") else {}
    summary = []
    for key, label in (
        ("teff", "Teff"),
        ("logg", "logg"),
        ("feh", "[Fe/H]"),
        ("rv_kms", "RV"),
        ("chi2_red", "χ²ν"),
    ):
        try:
            value = result[key]
        except (KeyError, TypeError, AttributeError):
            continue
        if value is None:
            continue
        if key == "teff":
            summary.append("{0}={1:.0f} K".format(label, float(value)))
        elif key == "rv_kms":
            summary.append("{0}={1:.2f} km/s".format(label, float(value)))
        else:
            summary.append("{0}={1:.3g}".format(label, float(value)))
    mask_fraction = report.get("mask_fraction")
    if mask_fraction is not None:
        summary.append("mask={0:.0%}".format(float(mask_fraction)))
    dropped = report.get("n_dropped_segments")
    if dropped:
        summary.append("dropped segments={0}".format(int(dropped)))
    return " | ".join(summary) + "\nflags: " + ", ".join(flags)


def _safe_median_scale(values):
    """Return a finite non-zero median scale, falling back to unity."""
    values = _as_float_array(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    scale = float(np.nanmedian(values))
    if not np.isfinite(scale) or np.isclose(scale, 0.0, rtol=0.0, atol=1e-300):
        return 1.0
    return scale


def _validation_plot_segments(plot_data):
    if not isinstance(plot_data, dict):
        raise TypeError("plot_data must be a validation_plot dictionary.")
    segments = list(plot_data.get("segments", []))
    if not segments:
        raise ValueError("validation plot payload contains no segments.")
    return segments


def _validation_segment_arrays(segment):
    wave = _as_float_array(segment["wave_A"])
    observed = _as_float_array(segment["observed_flux"])
    model = _as_float_array(segment["model_flux"])
    if wave.shape != observed.shape or wave.shape != model.shape:
        raise ValueError("validation plot segment arrays must have matching shapes.")
    used = segment.get("used", np.ones(wave.size, dtype=bool))
    used = _as_bool_array(used, n_expected=wave.size)
    return wave, observed, model, used


def _validation_plot_scales(plot_data, scale_mode):
    """Return per-segment display scales for an XSL validation payload."""
    scale_mode = str(scale_mode).strip().lower()
    if scale_mode not in {"global", "per_segment", "none"}:
        raise ValueError("scale_mode must be 'global', 'per_segment', or 'none'.")

    segments = _validation_plot_segments(plot_data)
    if scale_mode == "none":
        return [1.0 for _segment in segments]

    if scale_mode == "global":
        chunks = []
        fallback = []
        for segment in segments:
            _wave, observed, _model, used = _validation_segment_arrays(segment)
            good = used & np.isfinite(observed)
            if np.any(good):
                chunks.append(observed[good])
            finite = np.isfinite(observed)
            if np.any(finite):
                fallback.append(observed[finite])
        if chunks:
            scale = _safe_median_scale(np.concatenate(chunks))
        elif fallback:
            scale = _safe_median_scale(np.concatenate(fallback))
        else:
            scale = 1.0
        return [scale for _segment in segments]

    scales = []
    for segment in segments:
        _wave, observed, _model, used = _validation_segment_arrays(segment)
        good = used & np.isfinite(observed)
        values = observed[good] if np.any(good) else observed[np.isfinite(observed)]
        scales.append(_safe_median_scale(values))
    return scales


def plot_xsl_validation_payload(
    plot_data,
    *,
    scale_mode=None,
    title=None,
    annotate=True,
    figsize=(11.0, 5.2),
):
    """
    Plot an XSL validation JSON payload with explicit display normalization.

    ``scale_mode="global"`` uses one median scale per star and is the default
    for full-spectrum XSL displays. ``scale_mode="per_segment"`` is retained as
    a diagnostic line-shape view and is labelled as such. ``scale_mode="none"``
    plots the saved flux samples without display normalization.

    The function is display-only: it does not align arms, alter wavelengths, or
    apply any radial-velocity correction.
    """
    if scale_mode is None:
        scale_mode = (
            plot_data.get("display_defaults", {}).get("scale_mode", "global")
            if isinstance(plot_data, dict)
            else "global"
        )
    scale_mode = str(scale_mode).strip().lower()
    segments = _validation_plot_segments(plot_data)
    scales = _validation_plot_scales(plot_data, scale_mode)

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_flux, ax_resid = axes

    for index, (segment, scale) in enumerate(zip(segments, scales)):
        wave, observed, model, used = _validation_segment_arrays(segment)
        finite = np.isfinite(wave) & np.isfinite(observed) & np.isfinite(model)
        if not np.any(finite):
            continue
        label_obs = "observed" if index == 0 else None
        label_model = "model" if index == 0 else None
        ax_flux.plot(
            wave[finite],
            observed[finite] / scale,
            color="0.25",
            lw=0.7,
            label=label_obs,
        )
        ax_flux.plot(
            wave[finite],
            model[finite] / scale,
            color="tab:red",
            lw=0.8,
            alpha=0.9,
            label=label_model,
        )
        resid_good = finite & used
        if np.any(resid_good):
            ax_resid.plot(
                wave[resid_good],
                (observed[resid_good] - model[resid_good]) / scale,
                color="black",
                lw=0.6,
            )

    ylabel = {
        "global": "Flux / global median",
        "per_segment": "Flux / segment median",
        "none": "Flux",
    }[scale_mode]
    ax_flux.set_ylabel(ylabel)
    ax_resid.axhline(0.0, color="0.5", lw=0.8)
    ax_resid.set_ylabel("Data - model")
    ax_resid.set_xlabel("Wavelength (Å)")
    ax_flux.legend(frameon=False, loc="best", fontsize=8)
    if title is None:
        title = plot_data.get("title", "XSL validation display")
    ax_flux.set_title(title, loc="left")

    if annotate:
        if scale_mode == "per_segment":
            note = (
                "Per-segment median normalization: diagnostic line-shape view only; "
                "do not interpret arm-to-arm continuum levels."
            )
        elif scale_mode == "global":
            note = (
                "One display scale per target. Segments preserve UVB/VIS/NIR "
                "resolution metadata; no arm scaling or RV correction is applied here."
            )
        else:
            note = (
                "No display normalization. Segments preserve metadata; no arm "
                "scaling or RV correction is applied here."
            )
        ax_flux.text(
            0.01,
            0.98,
            note,
            ha="left",
            va="top",
            transform=ax_flux.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

    fig.tight_layout()
    return fig, axes


def plot_fit_referee(
    result,
    *,
    segment=None,
    rest_frame=False,
    savepath=None,
    max_points_per_segment=6000,
    figsize_per_segment=(11.0, 3.2),
    residual_ylim=(-6.0, 6.0),
    layout="side_by_side",
):
    """
    Plot a deterministic referee view of a PHOENIX fit.

    Parameters
    ----------
    result : PhoenixFitResult-like
        Fit result containing reconstructed ``models``, ``used_masks``, and
        ``excluded_masks``. The function does not mutate this object.
    segment : SpectrumSegment, SpectrumCollection, or sequence, optional
        Observed spectrum used for the fit. Required because the result object
        intentionally avoids storing observed flux arrays by default.
    rest_frame : bool, optional
        Reserved for future explicit rest-frame display. The current
        implementation leaves wavelengths in the supplied segment frame and
        annotates the fitted RV instead of silently shifting axes.
    savepath : str or path-like, optional
        If provided, save the figure to this path. Matplotlib infers PNG/SVG/etc.
        from the extension.
    max_points_per_segment : int, optional
        Plot at most this many finite samples per segment to keep large spectra
        responsive.
    layout : {"side_by_side", "stacked"}, optional
        ``"side_by_side"`` keeps the compact report layout with flux and
        residual panels in two columns. ``"stacked"`` uses one wide flux panel
        above one wide residual panel per segment, which is usually clearer in
        an interactive notebook or pop-up window.

    Returns
    -------
    fig, axes
        Matplotlib figure and a ``(n_segments, 2)`` array of axes.
    """
    if rest_frame:
        raise NotImplementedError(
            "rest_frame=True is not implemented yet; wavelengths are plotted in "
            "the segment metadata frame to avoid hidden RV/frame assumptions."
        )
    max_points_per_segment = int(max_points_per_segment)
    if max_points_per_segment < 1:
        raise ValueError("max_points_per_segment must be >= 1.")

    segments = _coerce_segment_sequence(segment)
    models = tuple(getattr(result, "models", ()))
    used_masks = tuple(getattr(result, "used_masks", ()))
    excluded_masks = tuple(getattr(result, "excluded_masks", ()))
    coeffs = tuple(getattr(result, "continuum_coefficients", ()))
    if len(models) != len(segments):
        raise ValueError(
            "Result model count ({0}) does not match segment count ({1}).".format(
                len(models), len(segments)
            )
        )

    n_segments = len(segments)
    layout = str(layout).strip().lower()
    if layout not in {"side_by_side", "stacked"}:
        raise ValueError("layout must be 'side_by_side' or 'stacked'.")
    if layout == "side_by_side":
        fig, axes = plt.subplots(
            n_segments,
            2,
            figsize=(figsize_per_segment[0], figsize_per_segment[1] * n_segments),
            squeeze=False,
            gridspec_kw={"width_ratios": [3, 1]},
        )
    else:
        height_ratios = []
        for _index in range(n_segments):
            height_ratios.extend([3, 1])
        fig, axes_flat = plt.subplots(
            2 * n_segments,
            1,
            figsize=(figsize_per_segment[0], figsize_per_segment[1] * n_segments),
            squeeze=False,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes = axes_flat.reshape(n_segments, 2)
    title_text = _quality_text(result)
    if layout == "stacked":
        parts = title_text.split("\nflags: ", 1)
        if len(parts) == 2:
            title_text = "{0}\nflags: {1}".format(
                parts[0],
                textwrap.fill(parts[1], width=135),
            )
    fig.suptitle(title_text, fontsize=9 if layout == "stacked" else 10)

    for index, (seg, model_corr) in enumerate(zip(segments, models)):
        ax_flux, ax_resid = axes[index]
        wave = _as_float_array(seg.wave)
        flux = _as_float_array(seg.flux)
        err = None if getattr(seg, "err", None) is None else _as_float_array(seg.err)
        model_corr = _as_float_array(model_corr)
        if wave.shape != flux.shape or wave.shape != model_corr.shape:
            raise ValueError("Segment wave/flux/model arrays must have matching shapes.")

        if index < len(used_masks):
            used = _as_bool_array(used_masks[index], n_expected=wave.size)
        else:
            used = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model_corr)
        if index < len(excluded_masks):
            excluded = _as_bool_array(excluded_masks[index], n_expected=wave.size)
        else:
            excluded = np.zeros(wave.size, dtype=bool)
        coeff = coeffs[index] if index < len(coeffs) else None
        continuum, model_raw = _continuum_and_raw_model(wave, used, model_corr, coeff)

        valid = np.flatnonzero(
            np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model_corr)
        )
        if valid.size > max_points_per_segment:
            positions = np.unique(
                np.rint(
                    np.linspace(0, valid.size - 1, max_points_per_segment)
                ).astype(int)
            )
            sel = valid[positions]
        else:
            sel = valid

        ax_flux.plot(wave[sel], flux[sel], color="0.25", lw=0.7, label="observed")
        if model_raw is not None:
            ax_flux.plot(
                wave[sel],
                model_raw[sel],
                color="tab:orange",
                lw=0.7,
                alpha=0.8,
                label="PHOENIX+LSF",
            )
        ax_flux.plot(
            wave[sel],
            model_corr[sel],
            color="tab:red",
            lw=0.9,
            label="continuum-adjusted model",
        )
        if continuum is not None:
            finite_cont = np.isfinite(continuum)
            continuum_range = (
                None
                if not np.any(finite_cont)
                else (
                    float(np.nanmin(continuum[finite_cont])),
                    float(np.nanmax(continuum[finite_cont])),
                )
            )
        else:
            continuum_range = None

        for wmin, wmax in _mask_to_spans(wave, excluded):
            ax_flux.axvspan(wmin, wmax, color="tab:purple", alpha=0.08)
            ax_resid.axvspan(wmin, wmax, color="tab:purple", alpha=0.08)
        if (~used).any():
            unused = np.intersect1d(sel, np.flatnonzero(~used), assume_unique=False)
            if unused.size:
                ax_flux.plot(
                    wave[unused],
                    flux[unused],
                    ".",
                    color="0.5",
                    ms=2,
                    alpha=0.5,
                    label="unused",
                )

        ylo, yhi = _compute_robust_ylim(flux[used] if np.any(used) else flux)
        ax_flux.set_ylim(ylo, yhi)
        ax_flux.set_ylabel("Flux")
        ax_flux.set_title(_segment_plot_label(seg, index), loc="left", fontsize=9)
        if layout == "stacked":
            ax_flux.tick_params(labelbottom=False)
        handles, labels = ax_flux.get_legend_handles_labels()
        if index == 0 and labels:
            ax_flux.legend(frameon=False, loc="best", fontsize=8)

        residual = flux - model_corr
        if err is not None:
            if err.shape != wave.shape:
                raise ValueError("Segment err array must match wave shape.")
            good_err = np.isfinite(err) & (err > 0)
            residual_sigma = np.full_like(residual, np.nan, dtype=float)
            residual_sigma[good_err] = residual[good_err] / err[good_err]
            ax_resid.plot(wave[sel], residual_sigma[sel], color="black", lw=0.6)
            ax_resid.set_ylabel("(D-M)/σ")
            ax_resid.set_ylim(*residual_ylim)
        else:
            ax_resid.plot(wave[sel], residual[sel], color="black", lw=0.6)
            ax_resid.set_ylabel("D-M")
        ax_resid.axhline(0.0, color="0.5", lw=0.8)
        ax_resid.set_xlabel("Wavelength (Å)")

        diag = getattr(result, "diagnostics", {})
        if hasattr(diag, "to_dict"):
            diag = diag.to_dict()
        seg_diag = diag.get("segment_diagnostics", [])
        if index < len(seg_diag):
            info = seg_diag[index]
            text_parts = ["Nfit={0}".format(info.get("n_fit", "?"))]
            lsf = info.get("lsf_fwhm_kms")
            if lsf is not None:
                text_parts.append("LSF FWHM={0:.2f} km/s".format(float(lsf)))
            if continuum_range is not None:
                text_parts.append(
                    "P(λ)={0:.3g}–{1:.3g}".format(*continuum_range)
                )
            if layout == "side_by_side":
                flags = ", ".join(getattr(result, "quality_flags", ())) or "none"
                text_parts.append("flags={0}".format(flags))
            text = "\n".join(text_parts)
            ax_resid.text(
                0.01,
                0.98,
                text,
                ha="left",
                va="top",
                transform=ax_resid.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
        fig.spyctres_generated_files = {"referee_plot": str(savepath)}
    return fig, axes


def plot_fit_windows(
    wave,
    flux,
    err,
    model,
    windows,
    ncols=2,
    title=None,
    line_groups=None,
    excluded_mask=None,
    figsize_per_panel=(5, 3),
    data_label="Data",
    model_label="Model",
):
    """
    Plot zoomed windows comparing data and model.

    Parameters
    ----------
    wave, flux : array-like
        Observed wavelength and flux arrays.
    err : array-like or None
        Uncertainty array. Currently retained for interface consistency.
    model : array-like
        Model flux on the same wavelength grid.
    windows : list
        Window specification. Each element may be:
        - (label, wmin, wmax)
        - (wmin, wmax)
    ncols : int, optional
        Number of columns in the panel grid.
    title : str, optional
        Figure title.
    line_groups : list, optional
        Optional spectral-line groups to mark.
    excluded_mask : array-like of bool, optional
        Regions to shade in the zoomed panels.
    figsize_per_panel : tuple, optional
        Figure size per panel in inches.
    data_label, model_label : str, optional
        Legend labels for the first panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """
    wave = _as_float_array(wave)
    flux = _as_float_array(flux)
    model = _as_float_array(model)

    if wave.shape != flux.shape or wave.shape != model.shape:
        raise ValueError("wave, flux, and model must have the same shape.")

    if err is not None:
        err = _as_float_array(err)
        if err.shape != wave.shape:
            raise ValueError("err must match wave shape.")

    if excluded_mask is not None:
        excluded_mask = _as_bool_array(excluded_mask, n_expected=wave.size)

    nwin = len(windows)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nwin / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    first_visible = True

    for ax, win in zip(axes.ravel(), windows):
        if len(win) == 3:
            label, wmin, wmax = win
        elif len(win) == 2:
            wmin, wmax = win
            label = None
        else:
            raise ValueError("Each window must be (label, wmin, wmax) or (wmin, wmax).")

        m = (wave >= wmin) & (wave <= wmax)
        if not m.any():
            ax.set_visible(False)
            continue

        if first_visible:
            ax.plot(wave[m], flux[m], lw=0.9, label=data_label)
            ax.plot(wave[m], model[m], lw=0.9, label=model_label)
            first_visible = False
        else:
            ax.plot(wave[m], flux[m], lw=0.9)
            ax.plot(wave[m], model[m], lw=0.9)

        if excluded_mask is not None and excluded_mask.any():
            for swmin, swmax in _mask_to_spans(wave, excluded_mask & m):
                ax.axvspan(swmin, swmax, alpha=0.12)

        ylo, yhi = _compute_robust_ylim(flux[m], lower=2.0, upper=98.0, pad_frac=0.08)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(wmin, wmax)

        if label is not None:
            ax.set_title(label)

        if line_groups is not None:
            for group in line_groups:
                lines = _resolve_line_group(group)
                mark_lines(ax, lines, xlim=(wmin, wmax), ymax=0.98)

    for ax in axes.ravel()[nwin:]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if labels:
        axes.ravel()[0].legend(frameon=False, loc="best")

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    return fig, axes


def plot_spectrum_quicklook(
    spec,
    use_mask=True,
    show_error=False,
    ax=None,
    title=None,
):
    """
    Very basic quick-look plot for a spectrum-like object.

    Parameters
    ----------
    spec : SpectrumSegment or tuple-like
        Either an object with .wave, .flux, .err, .mask attributes,
        or a tuple/list of (wave, flux[, err[, mask]]).
    use_mask : bool, optional
        If True, plot only pixels where mask is True.
    show_error : bool, optional
        If True and errors are available, overplot the error array.
    ax : matplotlib Axes, optional
        Existing axis to draw on. If None, create a new figure/axis.
    title : str, optional
        Plot title.

    Returns
    -------
    fig, ax
    """
    if hasattr(spec, "wave") and hasattr(spec, "flux"):
        wave = _as_float_array(spec.wave)
        flux = _as_float_array(spec.flux)
        err = None if getattr(spec, "err", None) is None else _as_float_array(spec.err)
        mask = None if getattr(spec, "mask", None) is None else _as_bool_array(spec.mask)
        name = getattr(spec, "name", None)
    else:
        wave = _as_float_array(spec[0])
        flux = _as_float_array(spec[1])
        err = None
        mask = None
        name = None
        if len(spec) > 2 and spec[2] is not None:
            err = _as_float_array(spec[2])
        if len(spec) > 3 and spec[3] is not None:
            mask = _as_bool_array(spec[3])

    good = np.isfinite(wave) & np.isfinite(flux)
    if use_mask and mask is not None:
        good &= mask
    if err is not None and show_error:
        good &= np.isfinite(err)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(wave[good], flux[good], lw=0.8)

    if show_error and err is not None:
        ax.plot(wave[good], err[good], lw=0.6, alpha=0.8)

    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Flux")

    if title is not None:
        ax.set_title(title)
    elif name is not None:
        ax.set_title(str(name))

    return fig, ax
