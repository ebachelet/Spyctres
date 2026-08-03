"""
Workflow-level fitting recipes built on top of Spyctres core primitives.

This module is intentionally higher-level than Spyctres.fitting. It contains
maintained recipe helpers that are useful for real workflows and examples,
but are more specialized than the generic full-spectrum fitter.

Current scope
-------------
- X-SHOOTER/Balmer-window helper presets
- sideband normalization for line-window workflows
- sideband-aware PHOENIX fitting on top of the native-grid forward model
- reconstruction of fitted models for plotting

These helpers operate on generic SpectrumSegment objects, so the logic is not
strictly tied to one instrument even when some presets are X-SHOOTER-oriented.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ._serialization import json_safe, save_figure
from .io import SpectrumCollection, SpectrumSegment, make_padded_window_segments
from .preprocessing import (
    compose_fit_mask,
    exclusion_mask,
    telluric_transmission_exclusion_mask,
)
from .waveutils import (
    C_KMS,
    convert_segment_wavelength_medium,
    convert_wavelength_medium,
)
from .fitting import (
    build_effective_fit_mask,
    build_excluded_mask,
    reconstruct_phoenix_legendre_models_for_segments,
    _resolve_broadening_fwhm_kms,
    _resolve_segment_fwhm_kms,
    _gaussian_broaden_velocity,
    _apply_observed_grid_rv_shift,
    _validate_optimizer_loss,
    _resolve_per_segment_numeric_options,
    _apply_error_floor_to_fit_errors,
    _make_progress_reporter,
)
from .phoenix_forward import (
    build_phoenix_native_models_for_segments,
    infer_segments_wave_medium,
    build_native_interp_wave_grid_for_segments,
)

BALMER_CENTERS_AIR = {
    "Hα": 6562.80,
    "Hβ": 4861.33,
    "Hγ": 4340.47,
    "Hδ": 4101.74,
}

BALMER_CENTERS_VAC = {
    label: float(
        convert_wavelength_medium(
            np.array([center_air], dtype=float),
            from_medium="air",
            to_medium="vacuum",
        )[0]
    )
    for label, center_air in BALMER_CENTERS_AIR.items()
}

XSHOOTER_BALMER_WINDOWS = {
    "current": [
        ("Hδ", 4076.0, 4128.0),
        ("Hγ", 4314.0, 4366.0),
        ("Hβ", 4836.0, 4888.0),
    ],
    "notebook": [
        ("Hδ", 3980.0, 4220.0),
        ("Hγ", 4220.0, 4480.0),
        ("Hβ", 4700.0, 5020.0),
    ],
}

XSHOOTER_NOTEBOOK_CONT_WINDOWS = {
    "Hδ": ((-80.0, -30.0), (30.0, 80.0)),
    "Hγ": ((-80.0, -30.0), (30.0, 80.0)),
    "Hβ": ((-120.0, -40.0), (40.0, 120.0)),
}

XSHOOTER_BALMER_CORE_MASK_DEFAULT_A = 10.0
XSHOOTER_BALMER_CORE_MASK_CONSERVATIVE_A = 12.0

BALMER_LABEL_ALIASES = {
    "hα": "Hα",
    "halpha": "Hα",
    "ha": "Hα",
    "hβ": "Hβ",
    "hbeta": "Hβ",
    "hb": "Hβ",
    "hγ": "Hγ",
    "hgamma": "Hγ",
    "hg": "Hγ",
    "hδ": "Hδ",
    "hdelta": "Hδ",
    "hd": "Hδ",
}


def sdss_quicklook_resolution_assumption(R=2000.0):
    """Return an explicit SDSS quicklook resolving-power assumption.

    SDSS spectra are ingested with ``resolution=None`` because precision use
    should rely on validated wavelength-dependent LSF information. This helper
    packages the common ``R≈2000`` quicklook approximation as provenance rather
    than making it a silent reader default.
    """
    R = float(R)
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be finite and > 0.")
    return {
        "quantity": "R",
        "value": R,
        "resolution_source": "user_override",
        "assumed_resolution_R": R,
        "assumption_warning": (
            "approximate SDSS quicklook resolution; not precision SDSS LSF modelling"
        ),
        "reader_default_resolution": None,
        "intended_use": "quicklook_classification",
    }


@dataclass(frozen=True)
class XshooterBalmerCase:
    """Prepared X-SHOOTER/Balmer fitting inputs with provenance."""

    input_segment: SpectrumSegment
    clipped_segment: SpectrumSegment
    fit_segments: tuple
    balmer_windows: tuple
    exclude_masks: tuple
    norm_info: object
    provenance: dict

    @property
    def collection(self):
        """Return the prepared Balmer segments as a ``SpectrumCollection``."""
        return SpectrumCollection(
            self.fit_segments,
            name="xshooter_uvb_balmer_case",
            meta={
                "workflow": "xshooter_balmer_case",
                "recipe": self.provenance,
            },
        )

    @property
    def fit_regions(self):
        """Return the non-padded Balmer fitting regions in Angstrom."""
        return tuple(
            (float(wmin), float(wmax))
            for _label, wmin, wmax in self.balmer_windows
        )

    @property
    def fit_regions_by_segment(self):
        """Return one non-padded fit-region list per prepared segment.

        Use this form for readiness/audit calls on ``self.collection``.  Each
        prepared line segment carries a padded support region, but its own
        science fit window is only the corresponding Balmer interval.
        """
        return tuple(
            ((float(wmin), float(wmax)),)
            for _label, wmin, wmax in self.balmer_windows
        )

    @property
    def fit_windows(self):
        """Return plotting-window dictionaries for the prepared Balmer lines."""
        rows = []
        for segment, (label, wmin, wmax) in zip(
            self.fit_segments,
            self.balmer_windows,
        ):
            center = segment.meta.get("line_center_data")
            rows.append(
                {
                    "label": str(label),
                    "limits_A": [float(wmin), float(wmax)],
                    "markers_A": [] if center is None else [float(center)],
                }
            )
        return tuple(rows)

    @property
    def exclusion_masks(self):
        """Alias for recipe exclusion masks; True means reject/exclude."""
        return self.exclude_masks

    @property
    def valid_masks(self):
        """Return per-segment usable-pixel masks after recipe exclusions.

        The polarity follows the public Spyctres convention:
        ``True`` means the pixel is usable for fitting.
        """
        return self.valid_masks_for()

    def combined_exclusion_masks(self, extra_exclusion_masks=None):
        """Return recipe exclusions plus optional user-reviewed exclusions."""
        if extra_exclusion_masks is None:
            extra = ()
        elif isinstance(extra_exclusion_masks, (list, tuple)):
            extra = tuple(extra_exclusion_masks)
        else:
            extra = (extra_exclusion_masks,)
        return tuple(self.exclude_masks) + extra

    def valid_masks_for(self, extra_exclusion_masks=None):
        """Return usable-pixel masks after recipe and extra exclusions.

        ``extra_exclusion_masks`` is intended for genuinely new regions found
        by visual inspection. Do not use it to duplicate pixels that the reader
        has already rejected through the segment's input quality mask.
        """
        masks = self.combined_exclusion_masks(extra_exclusion_masks)
        return tuple(
            compose_fit_mask(segment, exclude_masks=masks).fit_use_mask
            for segment in self.fit_segments
        )

    @property
    def normalization_info(self):
        """Return JSON-safe sideband-normalization diagnostics."""
        return json_safe(self.norm_info)

    def summary(self):
        """Return a JSON-safe summary of the prepared Balmer case."""
        valid_masks = self.valid_masks
        segment_rows = []
        for segment, valid, (label, wmin, wmax) in zip(
            self.fit_segments,
            valid_masks,
            self.balmer_windows,
        ):
            wave = np.asarray(segment.wave, dtype=float)
            base_valid = np.asarray(segment.mask, dtype=bool)
            finite_wave = wave[np.isfinite(wave)]
            center = segment.meta.get("line_center_data")
            cont_windows = segment.meta.get("cont_windows")
            sidebands = []
            if cont_windows is not None and center is not None:
                center = float(center)
                for lo, hi in cont_windows:
                    sidebands.append([center + float(lo), center + float(hi)])
            resolution = (
                None
                if segment.resolution is None
                else segment.resolution.to_metadata()
            )
            segment_rows.append(
                {
                    "label": str(label),
                    "segment_name": segment.name,
                    "fit_region_A": [float(wmin), float(wmax)],
                    "segment_range_A": (
                        None
                        if finite_wave.size == 0
                        else [float(np.nanmin(finite_wave)), float(np.nanmax(finite_wave))]
                    ),
                    "line_center_air_A": json_safe(segment.meta.get("line_center_air")),
                    "line_center_vac_A": json_safe(segment.meta.get("line_center_vac")),
                    "line_center_data_A": json_safe(segment.meta.get("line_center_data")),
                    "continuum_sidebands_A": sidebands,
                    "n_pixels": int(wave.size),
                    "n_reader_valid_pixels": int(np.count_nonzero(base_valid)),
                    "n_recipe_valid_pixels": int(np.count_nonzero(valid)),
                    "recipe_valid_fraction": (
                        0.0 if wave.size == 0 else float(np.count_nonzero(valid) / wave.size)
                    ),
                    "wave_medium": segment.wave_medium,
                    "observer_frame": segment.observer_frame,
                    "stellar_rest_status": segment.stellar_rest_status,
                    "resolution": resolution,
                }
            )

        total_pixels = int(sum(item["n_pixels"] for item in segment_rows))
        total_valid = int(sum(item["n_recipe_valid_pixels"] for item in segment_rows))
        media = sorted({segment.wave_medium for segment in self.fit_segments})
        observer_frames = sorted({segment.observer_frame for segment in self.fit_segments})
        rest_statuses = sorted(
            {segment.stellar_rest_status for segment in self.fit_segments}
        )
        resolutions = [
            item["resolution"]
            for item in segment_rows
            if item.get("resolution") is not None
        ]
        if not resolutions:
            resolution_summary = None
        elif all(item == resolutions[0] for item in resolutions):
            resolution_summary = resolutions[0]
        else:
            resolution_summary = resolutions
        return json_safe(
            {
                "recipe": self.provenance.get("recipe", "xshooter_balmer_case"),
                "recipe_version": self.provenance.get("recipe_version"),
                "n_segments": len(self.fit_segments),
                "lines": [row["label"] for row in segment_rows],
                "norm_mode": self.provenance.get("norm_mode"),
                "sideband_width_A": self.provenance.get("sideband_width_A"),
                "sideband_order": self.provenance.get("sideband_order"),
                "core_mask_halfwidth_A": self.provenance.get(
                    "core_mask_halfwidth_A"
                ),
                "exclude_masks": list(self.provenance.get("exclude_masks") or []),
                "mask_true_means": "valid_masks_true_means_usable; "
                "exclusion_masks_true_means_reject",
                "wave_medium": media[0] if len(media) == 1 else media,
                "observer_frame": (
                    observer_frames[0] if len(observer_frames) == 1 else observer_frames
                ),
                "stellar_rest_status": (
                    rest_statuses[0] if len(rest_statuses) == 1 else rest_statuses
                ),
                "resolution": resolution_summary,
                "total_pixels": total_pixels,
                "total_valid_pixels": total_valid,
                "total_valid_fraction": (
                    0.0 if total_pixels == 0 else float(total_valid / total_pixels)
                ),
                "segments": segment_rows,
                "normalization_info": self.normalization_info,
                "provenance": self.provenance,
            }
        )

    def summary_text(self):
        """Return a compact human-readable summary of the preparation."""
        summary = self.summary()
        resolution = summary.get("resolution")
        if isinstance(resolution, dict) and resolution.get("value") is not None:
            res_text = "{0}={1:g} from {2}".format(
                resolution.get("quantity", "resolution"),
                float(resolution.get("value")),
                resolution.get("source", "metadata"),
            )
        elif resolution:
            res_text = "mixed per-segment resolution metadata"
        else:
            res_text = "not available"

        core_width = summary.get("core_mask_halfwidth_A")
        core_text = (
            "none" if core_width is None else "±{0:g} Å".format(float(core_width))
        )
        lines = [
            "Prepared X-SHOOTER UVB Balmer case",
            "  lines: {0}".format(", ".join(summary.get("lines") or [])),
            "  normalization: {0}, sideband_width={1:g} Å, order={2}".format(
                summary.get("norm_mode"),
                float(summary.get("sideband_width_A") or 0.0),
                summary.get("sideband_order"),
            ),
            "  core exclusion: {0}".format(core_text),
            "  wavelength medium/frame: {0}; {1}; stellar_rest={2}".format(
                summary.get("wave_medium"),
                summary.get("observer_frame"),
                summary.get("stellar_rest_status"),
            ),
            "  resolution: {0}".format(res_text),
            "  valid pixels after recipe exclusions: {0}/{1} ({2:.1%})".format(
                int(summary.get("total_valid_pixels") or 0),
                int(summary.get("total_pixels") or 0),
                float(summary.get("total_valid_fraction") or 0.0),
            ),
        ]
        if summary.get("exclude_masks"):
            lines.append(
                "  exclusion masks: {0}".format(
                    ", ".join(summary.get("exclude_masks") or [])
                )
            )
        else:
            lines.append("  exclusion masks: none")
        lines.append("  per-line windows:")
        for row in summary.get("segments") or []:
            sidebands = row.get("continuum_sidebands_A") or []
            if sidebands:
                sb_text = "; ".join(
                    "{0:.1f}-{1:.1f}".format(float(lo), float(hi))
                    for lo, hi in sidebands
                )
            else:
                sb_text = "not used"
            region = row.get("fit_region_A") or [np.nan, np.nan]
            center = row.get("line_center_data_A")
            center_text = "unknown" if center is None else "{0:.2f} Å".format(float(center))
            lines.append(
                "    - {0}: fit {1:.1f}-{2:.1f} Å; center={3}; "
                "sidebands={4}; valid={5}/{6}".format(
                    row.get("label"),
                    float(region[0]),
                    float(region[1]),
                    center_text,
                    sb_text,
                    int(row.get("n_recipe_valid_pixels") or 0),
                    int(row.get("n_pixels") or 0),
                )
            )
        return "\n".join(lines)

    def to_dict(self):
        """Return JSON-safe recipe preparation metadata."""
        return self.summary()

    def suggest_fit_setup(
        self,
        *,
        mode="standard",
        intent="reviewed_analysis",
        continuum_degree=1,
        extra_exclusion_masks=None,
        **kwargs,
    ):
        """Build a ``FitSetup`` that uses this prepared Balmer case."""
        from .defaults import suggest_fit_setup

        exclude_masks = self.combined_exclusion_masks(extra_exclusion_masks)
        setup = suggest_fit_setup(
            self.collection,
            mode=mode,
            intent=intent,
            exclude_masks=exclude_masks,
            **kwargs,
        )
        return setup.with_regions(self.fit_regions).with_continuum_degree(
            continuum_degree
        )

    def plot_preparation(
        self,
        *,
        title="X-SHOOTER UVB Balmer preparation",
        ncols=2,
        figsize_per_panel=(7.2, 3.4),
        savepath=None,
    ):
        """Plot the prepared windows, sidebands, line centres, and core mask.

        The plot is diagnostic only and does not call ``show``. Orange spans
        mark continuum sidebands; pale red spans mark the Balmer-core pixels
        rejected by the recipe; red ``x`` markers show pixels not used after
        reader and recipe exclusions.
        """
        import matplotlib.pyplot as plt

        n = len(self.fit_segments)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            squeeze=False,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        )
        axes_flat = axes.ravel()
        valid_masks = self.valid_masks
        core_width = self.provenance.get("core_mask_halfwidth_A")
        for ax, segment, valid, (label, wmin, wmax) in zip(
            axes_flat,
            self.fit_segments,
            valid_masks,
            self.balmer_windows,
        ):
            wave = np.asarray(segment.wave, dtype=float)
            flux = np.asarray(segment.flux, dtype=float)
            valid = np.asarray(valid, dtype=bool)
            ax.plot(wave, flux, color="0.15", lw=0.9, label="spectrum")
            if np.any(~valid):
                ax.scatter(
                    wave[~valid],
                    flux[~valid],
                    s=24,
                    marker="x",
                    color="tab:red",
                    alpha=0.8,
                    label="not used",
                )
            center = segment.meta.get("line_center_data")
            if center is not None:
                center = float(center)
                ax.axvline(
                    center,
                    color="tab:blue",
                    ls="--",
                    lw=1.0,
                    alpha=0.75,
                    label="line centre",
                )
                cont_windows = segment.meta.get("cont_windows") or ()
                for lo, hi in cont_windows:
                    ax.axvspan(
                        center + float(lo),
                        center + float(hi),
                        color="tab:orange",
                        alpha=0.18,
                        label="continuum sideband",
                    )
                if core_width is not None and float(core_width) > 0.0:
                    ax.axvspan(
                        center - float(core_width),
                        center + float(core_width),
                        color="tab:red",
                        alpha=0.12,
                        label="excluded core",
                    )
            ax.axvspan(
                float(wmin),
                float(wmax),
                color="tab:orange",
                alpha=0.05,
                label="fit window",
            )
            ax.set_title(str(label))
            ax.set_xlabel("Wavelength [Å]")
            ax.set_ylabel("Flux")
            handles, labels = ax.get_legend_handles_labels()
            unique = {}
            for handle, label_text in zip(handles, labels):
                unique.setdefault(label_text, handle)
            ax.legend(unique.values(), unique.keys(), loc="best", fontsize="small")
            ax.grid(alpha=0.15)
        for ax in axes_flat[n:]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.tight_layout()
        if savepath is not None:
            save_figure(fig, savepath)
        return fig, axes


def _sideband_fit_parameter_count(n_segments, sideband_poly_order):
    """Count nonlinear and profiled continuum parameters for fit diagnostics."""
    n_segments = int(n_segments)
    sideband_poly_order = int(sideband_poly_order)
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1.")
    if sideband_poly_order < 0:
        raise ValueError("sideband_poly_order must be >= 0.")
    return 4 + n_segments * (sideband_poly_order + 1)


def xshooter_balmer_windows(window_mode="notebook"):
    """
    Return X-SHOOTER UVB Balmer-window presets.

    Parameters
    ----------
    window_mode : {"current", "notebook"}
        Current narrow windows or broader notebook-style windows.

    Returns
    -------
    list of (label, wmin, wmax)
        A fresh list of preset UVB Balmer windows.
    """
    mode = str(window_mode).strip().lower()
    if mode not in XSHOOTER_BALMER_WINDOWS:
        raise ValueError("window_mode must be 'current' or 'notebook'.")
    return list(XSHOOTER_BALMER_WINDOWS[mode])


def _canonical_balmer_label(label):
    """
    Normalize common ASCII and Unicode Balmer-line aliases to a canonical label.

    Examples
    --------
    Halpha -> Hα
    Hbeta  -> Hβ
    Hgamma -> Hγ
    Hdelta -> Hδ
    """
    raw = str(label).strip()
    key = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    return BALMER_LABEL_ALIASES.get(key, raw)


def _canonicalize_balmer_dict_keys(d):
    """
    Return a copy of a dict whose keys are Balmer-line labels normalized to
    the canonical internal form.
    """
    if d is None:
        return {}
    return {_canonical_balmer_label(k): v for k, v in d.items()}


def attach_balmer_metadata(segments, cont_windows=None, centers_vac=None):
    """
    Attach Balmer-line metadata to segments in-place.

    This helper accepts both canonical Unicode labels such as 'Hβ', 'Hγ', and
    common ASCII aliases such as 'Hbeta', 'Hgamma', 'Hdelta', 'Halpha'.

    For each segment it stores:
    - line_label       : canonical internal label
    - line_label_input : original input label
    - line_center_air  : standard air line center
    - line_center_vac  : vacuum line center
    - line_center_data : line center converted into the segment wavelength medium
    - cont_windows     : continuum sideband windows, if available

    Parameters
    ----------
    segments : list[SpectrumSegment]
        Input line-window segments.
    cont_windows : dict, optional
        Mapping from line label to continuum sideband windows.
        Defaults to XSHOOTER_NOTEBOOK_CONT_WINDOWS.
    centers_vac : dict, optional
        Mapping from line label to vacuum line center in Angstrom.
        Defaults to BALMER_CENTERS_VAC.

    Returns
    -------
    list[SpectrumSegment]
        The same list, for convenience.
    """
    if cont_windows is None:
        cont_windows = XSHOOTER_NOTEBOOK_CONT_WINDOWS
    if centers_vac is None:
        centers_vac = BALMER_CENTERS_VAC

    cont_windows = _canonicalize_balmer_dict_keys(cont_windows)
    centers_vac = _canonicalize_balmer_dict_keys(centers_vac)

    for seg in segments:
        raw_label = str(seg.name)
        label = _canonical_balmer_label(raw_label)

        if label not in centers_vac:
            raise ValueError(
                "Segment name {0!r} not recognized as a supported Balmer label.".format(raw_label)
            )

        center_vac = float(centers_vac[label])
        center_air = (
            float(BALMER_CENTERS_AIR[label])
            if label in BALMER_CENTERS_AIR
            else float(
                convert_wavelength_medium(
                    np.array([center_vac], dtype=float),
                    from_medium="vacuum",
                    to_medium="air",
                )[0]
            )
        )

        seg.meta["line_label_input"] = raw_label
        seg.meta["line_label"] = label
        seg.meta["line_center_air"] = center_air
        seg.meta["line_center_vac"] = center_vac

        seg_medium = str(seg.wave_medium).lower()
        if seg_medium in ("air", "vacuum"):
            center_data = float(
                convert_wavelength_medium(
                    np.array([center_vac], dtype=float),
                    from_medium="vacuum",
                    to_medium=seg_medium,
                )[0]
            )
        else:
            center_data = center_vac

        seg.meta["line_center_data"] = center_data

        if label in cont_windows:
            seg.meta["cont_windows"] = cont_windows[label]
        else:
            seg.meta.pop("cont_windows", None)

    return segments


def ensure_phoenix_interpolator_for_segments(
    segments,
    phoenix_lib,
    teff_grid,
    feh_grid,
    logg_grid,
    cache_path=None,
    progress_callback=None,
):
    """
    Ensure the PHOENIX interpolator is built on the concatenated support grid
    of the current segments.
    """
    support_wave_all = np.concatenate([np.asarray(seg.wave, dtype=float) for seg in segments])

    segment_media = sorted(set(str(seg.wave_medium).lower() for seg in segments))
    observed_wave_medium = segment_media[0] if len(segment_media) == 1 else None

    phoenix_lib.ensure_interpolator(
        wave=support_wave_all,
        teff_grid=np.asarray(teff_grid, dtype=float),
        feh_grid=np.asarray(feh_grid, dtype=float),
        logg_grid=np.asarray(logg_grid, dtype=float),
        cache_path=cache_path,
        observed_wave_medium=observed_wave_medium,
        progress_callback=progress_callback,
    )

    return support_wave_all


def _build_sideband_mask(seg, wave, fit_mask, sideband_width=10.0):
    """
    Build a sideband mask for one segment.

    If seg.meta['cont_windows'] is present, use those explicit sidebands
    relative to seg.meta['line_center_data']. Otherwise fall back to
    edge sidebands of the fit window.
    """
    wave = np.asarray(wave, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)

    cont_windows = seg.meta.get("cont_windows", None)
    center = seg.meta.get("line_center_data", None)

    if cont_windows is not None and center is not None:
        sb_mask = np.zeros_like(wave, dtype=bool)
        center = float(center)
        for a, b in cont_windows:
            sb_mask |= fit_mask & (wave > center + float(a)) & (wave < center + float(b))
        sb_mode = "explicit"
        lo = float(np.min(wave[fit_mask])) if np.any(fit_mask) else np.nan
        hi = float(np.max(wave[fit_mask])) if np.any(fit_mask) else np.nan
        return sb_mask, sb_mode, lo, hi

    fit_wave = wave[fit_mask]
    lo = float(np.min(fit_wave))
    hi = float(np.max(fit_wave))
    sb_mask = fit_mask & (
        ((wave >= lo) & (wave <= lo + float(sideband_width))) |
        ((wave >= hi - float(sideband_width)) & (wave <= hi))
    )
    return sb_mask, "edge", lo, hi


def normalize_segment_sidebands(seg, sideband_width=10.0, sideband_order=1):
    """
    Normalize one segment using a weighted polynomial continuum fit to either:
    - explicit per-line sidebands stored in seg.meta['cont_windows'], or
    - fallback edge sidebands of the fit window.

    Returns
    -------
    seg_n : SpectrumSegment
        Sideband-normalized segment.
    info : dict
        Small diagnostic dictionary about the normalization.
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = None if seg.err is None else np.asarray(seg.err, dtype=float)
    fit_mask = np.asarray(seg.mask, dtype=bool)

    if np.sum(fit_mask) < 6:
        return seg, {"mode": "none", "n_sideband": 0}

    sb_mask, sb_mode, lo, hi = _build_sideband_mask(
        seg, wave, fit_mask, sideband_width=sideband_width
    )

    good = sb_mask & np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        good &= np.isfinite(err) & (err > 0)

    order = int(sideband_order)

    if np.sum(good) >= (order + 2):
        if err is None:
            coeffs = np.polyfit(wave[good], flux[good], deg=order)
        else:
            coeffs = np.polyfit(wave[good], flux[good], deg=order, w=1.0 / err[good])
        cont = np.polyval(coeffs, wave)
        mode = "poly"
    else:
        level = float(np.nanmedian(flux[fit_mask]))
        coeffs = np.array([level], dtype=float)
        cont = np.full_like(wave, level, dtype=float)
        mode = "constant"

    pos = np.isfinite(cont) & (cont > 0)
    if not np.any(pos):
        raise ValueError("Sideband normalization produced a non-positive continuum.")

    fallback = float(np.nanmedian(cont[pos]))
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, fallback)

    flux_n = flux / cont
    err_n = None if err is None else err / cont

    seg_n = SpectrumSegment(
        wave=wave,
        flux=flux_n,
        err=err_n,
        mask=fit_mask,
        meta=dict(seg.meta),
        wave_medium=seg.wave_medium,
        wave_frame=seg.wave_frame,
        name=seg.name,
        observer_frame=seg.observer_frame,
        stellar_rest_status=seg.stellar_rest_status,
        stellar_rv_applied_kms=seg.stellar_rv_applied_kms,
        resolution=seg.resolution,
    )

    seg_n.meta["norm_mode"] = "sideband"
    seg_n.meta["sideband_width"] = float(sideband_width)
    seg_n.meta["sideband_order"] = int(sideband_order)
    seg_n.meta["sideband_cont_coeffs"] = np.asarray(coeffs, dtype=float).tolist()

    info = {
        "mode": mode,
        "sideband_mode": sb_mode,
        "n_sideband": int(np.sum(good)),
        "fit_lo": lo,
        "fit_hi": hi,
        "coeffs": coeffs,
    }
    return seg_n, info


def normalize_segments_sidebands(segments, sideband_width=10.0, sideband_order=1):
    """
    Apply sideband normalization to a list of segments.

    Returns
    -------
    segments_n : list[SpectrumSegment]
    info : list[dict]
    """
    out = []
    info = []
    for seg in segments:
        seg_n, seg_info = normalize_segment_sidebands(
            seg,
            sideband_width=sideband_width,
            sideband_order=sideband_order,
        )
        out.append(seg_n)
        info.append(seg_info)
    return out, info


def normalize_model_sidebands(seg, model_flux, sideband_width=10.0, sideband_order=1):
    """
    Normalize a model array on a segment grid using the same sideband logic
    as the data-side normalization.

    Returns
    -------
    model_n : ndarray
    info : dict
    """
    wave = np.asarray(seg.wave, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)
    fit_mask = np.asarray(seg.mask, dtype=bool)

    if np.sum(fit_mask) < 6:
        return model_flux.copy(), {"mode": "none", "n_sideband": 0}

    sb_mask, sb_mode, lo, hi = _build_sideband_mask(
        seg, wave, fit_mask, sideband_width=sideband_width
    )

    good = sb_mask & np.isfinite(wave) & np.isfinite(model_flux)
    order = int(sideband_order)

    if np.sum(good) >= (order + 2):
        coeffs = np.polyfit(wave[good], model_flux[good], deg=order)
        cont = np.polyval(coeffs, wave)
        mode = "poly"
    else:
        level = float(np.nanmedian(model_flux[fit_mask]))
        coeffs = np.array([level], dtype=float)
        cont = np.full_like(wave, level, dtype=float)
        mode = "constant"

    pos = np.isfinite(cont) & (cont > 0)
    if not np.any(pos):
        raise ValueError("Model sideband normalization produced a non-positive continuum.")

    fallback = float(np.nanmedian(cont[pos]))
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, fallback)

    model_n = model_flux / cont
    info = {
        "mode": mode,
        "sideband_mode": sb_mode,
        "n_sideband": int(np.sum(good)),
        "fit_lo": lo,
        "fit_hi": hi,
        "coeffs": coeffs,
    }
    return model_n, info


def _solve_sideband_multiplicative_poly(wave, flux, err, model, used_mask, order=1):
    """
    Solve a weighted multiplicative polynomial after sideband normalization.

    This mirrors the notebook logic:
        flux ~ model * poly(w)
    on the used pixels.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    model = np.asarray(model, dtype=float)
    used_mask = np.asarray(used_mask, dtype=bool)
    order = int(order)

    if order <= 0:
        return model.copy(), np.array([1.0], dtype=float)

    good = (
        used_mask &
        np.isfinite(wave) &
        np.isfinite(flux) &
        np.isfinite(err) & (err > 0) &
        np.isfinite(model)
    )

    if np.sum(good) < (order + 2):
        return model.copy(), np.array([1.0], dtype=float)

    x0 = float(np.mean(wave[good]))
    xscale = float(np.ptp(wave[good]))
    if (not np.isfinite(xscale)) or (xscale <= 0):
        xscale = 1.0

    x = (wave[good] - x0) / xscale
    A = np.vander(x, N=order + 1, increasing=True)

    rhs = flux[good] / (model[good] + 1e-30)
    W = 1.0 / err[good]

    Aw = A * W[:, None]
    bw = rhs * W
    coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    x_all = (wave - x0) / xscale
    poly_all = np.vander(x_all, N=order + 1, increasing=True) @ coeffs

    return model * poly_all, coeffs


def solve_sideband_multiplicative_poly(wave, flux, err, model, used_mask, order=1):
    """Public wrapper for the sideband multiplicative-polynomial solve."""
    return _solve_sideband_multiplicative_poly(
        wave=wave,
        flux=flux,
        err=err,
        model=model,
        used_mask=used_mask,
        order=order,
    )


def make_balmer_core_exclude_mask(
    core_halfwidth=XSHOOTER_BALMER_CORE_MASK_DEFAULT_A,
    wave_medium="vacuum",
):
    """
    Build a boolean exclusion-mask callable for the UVB Balmer line cores.

    This helper currently targets the three Balmer lines used by the X-SHOOTER
    UVB recipes: Hδ, Hγ, and Hβ.

    Parameters
    ----------
    core_halfwidth : float
        Half-width in Angstrom around each line center to exclude.
        For X-SHOOTER UVB Balmer-wing classification, the 
        default value is 10 Angstrom. Use 6–12 Angstroms as a robustness range.
    wave_medium : {"air", "vacuum", "unknown"}
        Wavelength medium of the observed data.
    """
    centers_vac = np.array(
        [
            BALMER_CENTERS_VAC["Hδ"],
            BALMER_CENTERS_VAC["Hγ"],
            BALMER_CENTERS_VAC["Hβ"],
        ],
        dtype=float,
    )

    wave_medium = str(wave_medium).lower()
    if wave_medium in ("air", "vacuum"):
        centers = convert_wavelength_medium(
            centers_vac,
            from_medium="vacuum",
            to_medium=wave_medium,
        )
    else:
        centers = centers_vac.copy()

    def _mask(wave):
        wave = np.asarray(wave, dtype=float)
        m = np.zeros_like(wave, dtype=bool)
        for c in centers:
            m |= np.abs(wave - c) <= float(core_halfwidth)
        return m

    return _mask


def prepare_xshooter_balmer_case(
    segment,
    *,
    wmin=None,
    wmax=None,
    clip_left=0,
    clip_right=0,
    window_mode="notebook",
    window_pad=5.0,
    norm_mode="poly",
    sideband_width=10.0,
    sideband_order=1,
    core_mask=XSHOOTER_BALMER_CORE_MASK_DEFAULT_A,
    use_telluric_mask=False,
    telluric_threshold=0.90,
):
    """Prepare X-SHOOTER UVB Balmer-window fit segments and masks.

    This centralizes the workflow assembly used by scripts and notebooks:
    clipping/selection, padded Balmer windows, Balmer metadata, optional
    sideband normalization, optional Balmer-core masking, and optional
    high-resolution telluric transmission masking.

    The helper does not read files and does not choose PHOENIX parameter
    bounds. Those remain caller responsibilities.
    """
    if norm_mode not in ("poly", "sideband"):
        raise ValueError("norm_mode must be 'poly' or 'sideband'.")

    if not isinstance(segment, SpectrumSegment):
        raise TypeError("segment must be a SpectrumSegment.")

    clipped = segment.window(
        wmin=wmin,
        wmax=wmax,
        clip_left=clip_left,
        clip_right=clip_right,
        name_suffix="fitwin",
    )

    requested_balmer_windows = tuple(xshooter_balmer_windows(window_mode))
    fit_segments = make_padded_window_segments(
        clipped,
        requested_balmer_windows,
        pad=window_pad,
        name_prefix="balmer",
    )

    window_by_label = {
        str(label): (str(label), float(wmin_i), float(wmax_i))
        for label, wmin_i, wmax_i in requested_balmer_windows
    }
    balmer_windows = []
    for seg_i in fit_segments:
        label, wmin_i, wmax_i = window_by_label[str(seg_i.name)]
        balmer_windows.append((label, wmin_i, wmax_i))
        wave_i = np.asarray(seg_i.wave, dtype=float)
        finite_wave_i = wave_i[np.isfinite(wave_i)]
        seg_i.meta["fit_region_A"] = [float(wmin_i), float(wmax_i)]
        seg_i.meta["support_region_A"] = (
            None
            if finite_wave_i.size == 0
            else [
                float(np.nanmin(finite_wave_i)),
                float(np.nanmax(finite_wave_i)),
            ]
        )
    balmer_windows = tuple(balmer_windows)

    if str(window_mode).strip().lower() == "notebook":
        attach_balmer_metadata(fit_segments)
    else:
        attach_balmer_metadata(
            fit_segments,
            cont_windows={label: None for label, _wmin_i, _wmax_i in balmer_windows},
        )

    norm_info = None
    if norm_mode == "sideband":
        fit_segments, norm_info = normalize_segments_sidebands(
            fit_segments,
            sideband_width=sideband_width,
            sideband_order=sideband_order,
        )

    exclude_masks = []
    if use_telluric_mask:
        exclude_masks.append(
            telluric_transmission_exclusion_mask(threshold=telluric_threshold)
        )

    if core_mask is not None and float(core_mask) > 0.0:
        exclude_masks.append(
            exclusion_mask(
                "balmer_core",
                make_balmer_core_exclude_mask(
                    core_halfwidth=float(core_mask),
                    wave_medium=fit_segments[0].wave_medium,
                ),
                metadata={
                    "mask_type": "stellar_line_core",
                    "method": "balmer_core_halfwidth",
                    "core_halfwidth_A": float(core_mask),
                    "feature_frame": "stellar_rest_or_data_line_center",
                    "action": "masked",
                },
            )
        )

    line_records = []
    resolution_records = []
    for seg_i, (label, wmin_i, wmax_i) in zip(fit_segments, balmer_windows):
        wave_i = np.asarray(seg_i.wave, dtype=float)
        finite_wave_i = wave_i[np.isfinite(wave_i)]
        resolution_records.append(
            None if seg_i.resolution is None else seg_i.resolution.to_metadata()
        )
        line_records.append(
            {
                "label": str(label),
                "fit_region_A": [float(wmin_i), float(wmax_i)],
                "segment_range_A": None
                if finite_wave_i.size == 0
                else [
                    float(np.nanmin(finite_wave_i)),
                    float(np.nanmax(finite_wave_i)),
                ],
                "line_center_air_A": seg_i.meta.get("line_center_air"),
                "line_center_vac_A": seg_i.meta.get("line_center_vac"),
                "line_center_data_A": seg_i.meta.get("line_center_data"),
                "continuum_sidebands_relative_A": seg_i.meta.get("cont_windows"),
                "wave_medium": seg_i.wave_medium,
            }
        )

    non_null_resolutions = [item for item in resolution_records if item is not None]
    if not non_null_resolutions:
        resolution_summary = None
    elif all(item == non_null_resolutions[0] for item in non_null_resolutions):
        resolution_summary = non_null_resolutions[0]
    else:
        resolution_summary = non_null_resolutions

    provenance = {
        "recipe_version": 1,
        "recipe": "prepare_xshooter_balmer_case",
        "input_segment_name": segment.name,
        "input_reader": (segment.meta or {}).get("reader"),
        "input_wave_medium": segment.wave_medium,
        "input_observer_frame": segment.observer_frame,
        "input_stellar_rest_status": segment.stellar_rest_status,
        "resolution": resolution_summary,
        "window_mode": str(window_mode),
        "window_pad_A": float(window_pad),
        "norm_mode": str(norm_mode),
        "sideband_width_A": float(sideband_width),
        "sideband_order": int(sideband_order),
        "core_mask_halfwidth_A": None if core_mask is None else float(core_mask),
        "mask_true_means": {
            "segment_mask": "usable_pixel",
            "valid_masks": "usable_pixel",
            "exclude_masks": "reject_pixel",
        },
        "telluric_mask_requested": bool(use_telluric_mask),
        "telluric_threshold": (
            float(telluric_threshold) if use_telluric_mask else None
        ),
        "balmer_windows": [
            (str(label), float(wmin_i), float(wmax_i))
            for label, wmin_i, wmax_i in balmer_windows
        ],
        "line_records": json_safe(line_records),
        "exclude_masks": [mask.name for mask in exclude_masks],
        "exclude_mask_metadata": {
            mask.name: dict(mask.metadata) for mask in exclude_masks
        },
    }

    return XshooterBalmerCase(
        input_segment=segment,
        clipped_segment=clipped,
        fit_segments=tuple(fit_segments),
        balmer_windows=balmer_windows,
        exclude_masks=tuple(exclude_masks),
        norm_info=norm_info,
        provenance=provenance,
    )


def fit_case_lines_individually(
    case,
    *,
    base_setup=None,
    extra_exclusion_masks=None,
    model="phoenix",
    phoenix_lib=None,
    phoenix_dir=None,
    progress_callback=None,
    **fit_kwargs,
):
    """Fit each segment in a prepared recipe case as a line-consistency check.

    This helper is intentionally small: it does not choose new science settings
    and it does not average the answers. It runs the same public
    ``fit_stellar_spectrum`` path one prepared line at a time so users can see
    whether Hδ, Hγ, and Hβ pull the atmospheric solution in compatible
    directions.
    """
    if not isinstance(case, XshooterBalmerCase):
        raise TypeError("case must be an XshooterBalmerCase.")

    from .api import fit_stellar_spectrum

    results = {}
    valid_masks = case.valid_masks_for(extra_exclusion_masks)
    for segment, valid_mask, (label, wmin, wmax) in zip(
        case.fit_segments,
        valid_masks,
        case.balmer_windows,
    ):
        collection = SpectrumCollection(
            [segment],
            name="xshooter_uvb_balmer_{0}".format(str(label)),
            meta={
                "workflow": "xshooter_balmer_case_individual_line",
                "recipe": case.provenance,
                "line_label": str(label),
            },
        )
        if base_setup is None:
            setup = case.suggest_fit_setup().with_regions([(float(wmin), float(wmax))])
        else:
            setup = base_setup.with_regions([(float(wmin), float(wmax))])
        results[str(label)] = fit_stellar_spectrum(
            collection,
            model=model,
            setup=setup,
            valid_mask=(valid_mask,),
            phoenix_lib=phoenix_lib,
            phoenix_dir=phoenix_dir,
            progress_callback=progress_callback,
            **fit_kwargs,
        )
    return results


def fit_phoenix_sideband_symmetric(
    segments,
    phoenix_lib,
    p0,
    exclude_mask=None,
    exclude_masks=None,
    rv_bary_kms=0.0,
    R=None,
    forward_model="native_interp",
    model_margin_A=200.0,
    teff_grid=None,
    feh_grid=None,
    logg_grid=None,
    cache_path=None,
    rv_init="grid",
    rv_grid_n=81,
    verbose=1,
    max_nfev=200,
    error_floor_fraction=0.0,
    sideband_width=10.0,
    sideband_order=1,
    sideband_poly_order=1,
    bounds=None,
    loss="linear",
    loss_f_scale=1.0,
    progress_callback=None,
):
    """
    Sideband-normalized fitter for line-window workflows.

    Data are sideband-normalized segment-by-segment, the model is
    sideband-normalized the same way, and a low-order multiplicative polynomial
    is then solved on the used pixels before residuals are computed.

    The wavelength-space forward model can follow either:
    - forward_model="interp_observed": interpolate directly on the segment
      support grid, then apply the PHOENIX RV convention and broaden there.
      This is retained as a fast/legacy compatibility path.
    - forward_model="native_interp": interpolate on a dense model-space
      wavelength grid, then shift, convolve, and resample last. This is the
      recommended path for line-profile work.
      
    RV convention
    -------------
    The returned `rv_kms` follows the PHOENIX fitting convention used by
    Spyctres.fitting: positive RV redshifts the template/model. The observed-grid
    branch uses `_apply_observed_grid_rv_shift()` internally to preserve this
    convention while leaving the legacy Spyctres.velocity_correction API unchanged.

    progress_callback : callable, optional
        Called with ``FitProgressEvent`` objects before cache load/rebuild, RV
        grid scanning, and local optimizer start/finish. ``str(event)`` is the
        human-readable status message.

    loss : {"linear", "soft_l1", "huber", "cauchy", "arctan"}, optional
        Robust loss passed to ``scipy.optimize.least_squares``. The default
        preserves ordinary least squares.

    loss_f_scale : float, optional
        Positive robust-loss scale passed to ``least_squares`` as ``f_scale``.

    error_floor_fraction : float, sequence, or dict, optional
        Optional per-segment fractional uncertainty floor. A value ``f`` adds
        ``f * median(abs(flux_fit))`` in quadrature to each fitted pixel's
        uncertainty for that segment. The default ``0.0`` preserves historical
        behavior.
    """
    report = _make_progress_reporter(progress_callback)

    loss, loss_f_scale = _validate_optimizer_loss(loss, loss_f_scale)

    if isinstance(segments, SpectrumSegment):
        segments = [segments]
    else:
        segments = list(segments)
    error_floor_by_segment = _resolve_per_segment_numeric_options(
        segments,
        error_floor_fraction,
        "error_floor_fraction",
        default=0.0,
        strict=True,
    )

    for seg in segments:
        if seg.err is None:
            raise ValueError(
                "fit_phoenix_sideband_symmetric requires seg.err for all segments. "
                "Provide uncertainties or use fit_phoenix_full_spectrum(), which "
                "can estimate fallback errors."
            )

    teff0, feh0, logg0, rv0 = map(float, p0)
    
    if teff_grid is None:
        teff_grid_req = phoenix_lib.DEFAULT_TEFF_GRID
    else:
        teff_grid_req = np.asarray(teff_grid, dtype=float)

    if feh_grid is None:
        feh_grid_req = phoenix_lib.DEFAULT_FEH_GRID
    else:
        feh_grid_req = np.asarray(feh_grid, dtype=float)

    if logg_grid is None:
        logg_grid_req = phoenix_lib.DEFAULT_LOGG_GRID
    else:
        logg_grid_req = np.asarray(logg_grid, dtype=float)
                    
    if forward_model not in ("interp_observed", "native_interp"):
        raise ValueError("forward_model must be 'interp_observed' or 'native_interp'.")

    used_masks = [
        build_effective_fit_mask(
            seg,
            exclude_mask=exclude_mask,
            exclude_masks=exclude_masks,
        )
        for seg in segments
    ]
    if not any(np.any(m) for m in used_masks):
        raise ValueError("No usable points remain after masking.")

    support_lengths = [len(seg.wave) for seg in segments]
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=None)
        for seg in segments
    ]
    if forward_model == "interp_observed":
        report(
            "Preparing observed-grid PHOENIX interpolator/cache.",
            phase="phoenix_cache",
        )
        support_wave_all = ensure_phoenix_interpolator_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            teff_grid=teff_grid_req,
            feh_grid=feh_grid_req,
            logg_grid=logg_grid_req,
            cache_path=cache_path,
            progress_callback=lambda message: report(message, phase="phoenix_cache"),
        )
    else:
        report(
            "Building native PHOENIX interpolation wavelength grid.",
            phase="wavelength_grid",
        )
        model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            model_margin_A=model_margin_A,
        )

        if not phoenix_lib.interpolator_matches(
            model_wave_grid,
            teff_grid_req,
            feh_grid_req,
            logg_grid_req,
            observed_wave_medium=model_wave_medium,
        ):
            report("Preparing PHOENIX interpolator/cache.", phase="phoenix_cache")
            phoenix_lib.build_interpolator(
                observed_wave=model_wave_grid,
                teff_grid=teff_grid_req,
                feh_grid=feh_grid_req,
                logg_grid=logg_grid_req,
                cache_path=cache_path,
                observed_wave_medium=model_wave_medium,
                progress_callback=lambda message: report(message, phase="phoenix_cache"),
            )
        else:
            report(
                "Reusing existing in-memory PHOENIX interpolator.",
                phase="phoenix_cache",
            )

    if bounds is None:
        tg, zg, gg = phoenix_lib._grid
        bounds = (
            (float(np.min(tg)), float(np.min(zg)), float(np.min(gg)), -300.0),
            (float(np.max(tg)), float(np.max(zg)), float(np.max(gg)), +300.0),
        )

    broadening_fwhm_kms = _resolve_broadening_fwhm_kms(R=R, fwhm_kms=None)
    n_points = int(sum(np.sum(m) for m in used_masks))

    def residuals(p):
        teff, feh, logg, rv_kms = map(float, p)

        try:
            model0 = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
        except ValueError:
            return np.ones(n_points, dtype=float) * 1e6

        out = []

        if forward_model == "interp_observed":
            rv_tot = float(rv_bary_kms) + float(rv_kms)
            if len(model0) != len(support_wave_all):
                return np.ones(n_points, dtype=float) * 1e6

            shifted_all = _apply_observed_grid_rv_shift(
                support_wave_all,
                model0,
                rv_tot,
            )
            
            i0 = 0
            for seg, used_mask, n_support, seg_fwhm, seg_error_floor_fraction in zip(
                segments,
                used_masks,
                support_lengths,
                segment_fwhm_kms,
                error_floor_by_segment,
            ):
                i1 = i0 + n_support

                wave = np.asarray(seg.wave, dtype=float)
                flux = np.asarray(seg.flux, dtype=float)
                err = np.asarray(seg.err, dtype=float)

                model_full = shifted_all[i0:i1]
                model_full = _gaussian_broaden_velocity(
                    wave,
                    model_full,
                    fwhm_kms=seg_fwhm,
                )
                model_norm, _ = normalize_model_sidebands(
                    seg,
                    model_full,
                    sideband_width=sideband_width,
                    sideband_order=sideband_order,
                )

                model_corr, _ = _solve_sideband_multiplicative_poly(
                    wave=wave,
                    flux=flux,
                    err=err,
                    model=model_norm,
                    used_mask=used_mask,
                    order=sideband_poly_order,
                )

                err_used, _error_floor_meta = _apply_error_floor_to_fit_errors(
                    flux[used_mask],
                    err[used_mask],
                    seg_error_floor_fraction,
                )
                out.append((flux[used_mask] - model_corr[used_mask]) / err_used)
                i0 = i1

        else:
            model_list = build_phoenix_native_models_for_segments(
                segments=segments,
                phoenix_wave_native=np.asarray(phoenix_lib.wave, dtype=float),
                template_flux_native=model0,
                rv_kms=rv_kms,
                rv_bary_kms=rv_bary_kms,
                segment_fwhm_kms=segment_fwhm_kms,
                phoenix_wave_medium=model_wave_medium,
                model_margin_A=model_margin_A,
                bounds_use_fit_mask=True,
                extrapolate=True,
            )
            for seg, used_mask, model_full, seg_error_floor_fraction in zip(
                segments,
                used_masks,
                model_list,
                error_floor_by_segment,
            ):
                wave = np.asarray(seg.wave, dtype=float)
                flux = np.asarray(seg.flux, dtype=float)
                err = np.asarray(seg.err, dtype=float)

                model_norm, _ = normalize_model_sidebands(
                    seg,
                    model_full,
                    sideband_width=sideband_width,
                    sideband_order=sideband_order,
                )

                model_corr, _ = _solve_sideband_multiplicative_poly(
                    wave=wave,
                    flux=flux,
                    err=err,
                    model=model_norm,
                    used_mask=used_mask,
                    order=sideband_poly_order,
                )

                err_used, _error_floor_meta = _apply_error_floor_to_fit_errors(
                    flux[used_mask],
                    err[used_mask],
                    seg_error_floor_fraction,
                )
                out.append((flux[used_mask] - model_corr[used_mask]) / err_used)

        return np.concatenate(out)
    
    if rv_init == "grid":
        rv_lo, rv_hi = float(bounds[0][3]), float(bounds[1][3])
        rv_grid = np.linspace(rv_lo, rv_hi, int(rv_grid_n))
        report(
            "Running sideband coarse RV grid scan with {0} trial velocities.".format(
                int(rv_grid.size)
            )
        )
        chi2s = np.array(
            [np.sum(residuals((teff0, feh0, logg0, float(rv))) ** 2) for rv in rv_grid],
            dtype=float,
        )
        rv0_use = float(rv_grid[np.argmin(chi2s)])
        if verbose:
            print("RV init grid best:", rv0_use)
        report(
            "Sideband coarse RV grid scan selected rv_kms={0:.6g}.".format(rv0_use),
            phase="rv_scan",
            payload={"rv_kms": float(rv0_use)},
        )
        p0_use = (teff0, feh0, logg0, rv0_use)
    elif rv_init is None:
        report(
            "Skipping sideband coarse RV grid scan; using supplied initial rv_kms.",
            phase="rv_scan",
        )
        p0_use = (teff0, feh0, logg0, rv0)
    else:
        raise ValueError("rv_init must be 'grid' or None.")

    report(
        "Starting sideband local optimizer: p0=({0:g}, {1:g}, {2:g}, {3:g}).".format(
            float(p0_use[0]),
            float(p0_use[1]),
            float(p0_use[2]),
            float(p0_use[3]),
        ),
        phase="local_optimize",
        payload={"start": [float(value) for value in p0_use]},
    )
    res = least_squares(
        residuals,
        x0=np.array(p0_use, dtype=float),
        bounds=bounds,
        method="trf",
        x_scale=np.array([100.0, 0.1, 0.1, 10.0], dtype=float),
        max_nfev=int(max_nfev),
        loss=loss,
        f_scale=loss_f_scale,
        verbose=2 if verbose else 0,
    )
    report(
        "Finished sideband local optimizer: chi2={0:.6g}, success={1}.".format(
            float(np.sum(res.fun * res.fun)),
            bool(res.success),
        ),
        phase="local_optimize",
        fraction=1.0,
        payload={
            "chi2": float(np.sum(res.fun * res.fun)),
            "success": bool(res.success),
        },
    )

    r = res.fun
    chi2 = float(np.sum(r * r))
    n = int(r.size)
    k = _sideband_fit_parameter_count(len(segments), sideband_poly_order)
    dof = max(1, n - k)
    chi2_red = chi2 / dof
    error_floor_applied = bool(any(float(value) > 0.0 for value in error_floor_by_segment))
    error_model = "floor_inflated" if error_floor_applied else "nominal"

    return {
        "success": bool(res.success),
        "message": res.message,
        "p_best": res.x,
        "teff": float(res.x[0]),
        "feh": float(res.x[1]),
        "logg": float(res.x[2]),
        "rv_kms": float(res.x[3]),
        "rv_bary_kms": float(rv_bary_kms),
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2_red,
        "effective_chi2": chi2,
        "effective_chi2_red": chi2_red,
        "raw_chi2": None if error_floor_applied else chi2,
        "raw_chi2_red": None if error_floor_applied else chi2_red,
        "error_model": error_model,
        "error_floor_applied": error_floor_applied,
        "n_points": n,
        "status": int(res.status),
        "nfev": int(res.nfev),
        "optimizer_loss": str(loss),
        "optimizer_loss_f_scale": float(loss_f_scale),
        "optimizer_cost": float(res.cost),
        "optimizer_cost_twice": float(2.0 * res.cost),
        "segment_error_floor_fraction": [
            float(value) for value in error_floor_by_segment
        ],
        "forward_model": str(forward_model),
        "model_margin_A": float(model_margin_A),
        "segment_lsf_fwhm_kms": [
            None if x is None else float(x) for x in segment_fwhm_kms
        ],
        "segment_resolution_R_effective": [
            None if x is None else float(C_KMS / x) for x in segment_fwhm_kms
        ],
        "resolution_R": None if R is None else float(R),
        "lsf_fwhm_kms": None if broadening_fwhm_kms is None else float(broadening_fwhm_kms),
    }


def build_plot_models_for_segments(
    segments,
    phoenix_lib,
    fit_result,
    exclude_mask=None,
    exclude_masks=None,
    mdeg=2,
    rv_bary_kms=0.0,
    R=None,
    fwhm_kms=None,
    norm_mode="poly",
    sideband_width=10.0,
    sideband_order=1,
    sideband_poly_order=1,
    forward_model=None,
    model_margin_A=None,
):
    """
    Reconstruct per-segment fitted model arrays on the full pixel grid of each segment.

    Parameters
    ----------
    norm_mode : {"poly", "sideband"}
        Reconstruction path. "poly" delegates to the generic fitter-side
        polynomial reconstruction. "sideband" rebuilds the sideband-normalized
        model and the local multiplicative polynomial used by the recipe fitter.
    """
    teff = float(fit_result["teff"])
    feh = float(fit_result["feh"])
    logg = float(fit_result["logg"])
    rv_kms = float(fit_result["rv_kms"])

    if forward_model is None:
        forward_model = str(fit_result.get("forward_model", "interp_observed"))
    if model_margin_A is None:
        model_margin_A = float(fit_result.get("model_margin_A", 200.0))

    if norm_mode == "poly":
        return reconstruct_phoenix_legendre_models_for_segments(
            segments=segments,
            phoenix_lib=phoenix_lib,
            fit_result=fit_result,
            exclude_mask=exclude_mask,
            exclude_masks=exclude_masks,
            mdeg=mdeg,
            rv_bary_kms=rv_bary_kms,
            R=R,
            fwhm_kms=fwhm_kms,
            forward_model=forward_model,
            model_margin_A=model_margin_A,
        )

    if norm_mode != "sideband":
        raise ValueError("norm_mode must be 'poly' or 'sideband'.")

    used_masks = [
        build_effective_fit_mask(
            seg,
            exclude_mask=exclude_mask,
            exclude_masks=exclude_masks,
        )
        for seg in segments
    ]
    excluded_masks = [
        build_excluded_mask(
            seg,
            exclude_mask=exclude_mask,
            exclude_masks=exclude_masks,
        )
        for seg in segments
    ]
    segment_fwhm_kms = [
        _resolve_segment_fwhm_kms(seg, R=R, fwhm_kms=fwhm_kms)
        for seg in segments
    ]
    model_full_list = []
    coeffs_list = []

    if forward_model == "interp_observed":
        support_lengths = [len(seg.wave) for seg in segments]
        n_support_total = int(sum(support_lengths))

        model_support_all = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
        if len(model_support_all) != n_support_total:
            raise ValueError(
                "Model grid length does not match total support wavelength grid: "
                "{0} vs {1}".format(len(model_support_all), n_support_total)
            )

        i0 = 0
        for seg, used_mask, seg_fwhm in zip(segments, used_masks, segment_fwhm_kms):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)
            err_full = np.asarray(seg.err, dtype=float)

            n_support = len(wave_full)
            i1 = i0 + n_support

            model0_full = model_support_all[i0:i1]

            shifted_full = _apply_observed_grid_rv_shift(
                wave_full,
                model0_full,
                rv_bary_kms + rv_kms,
            )
            model_broad_full = _gaussian_broaden_velocity(
                wave_full,
                shifted_full,
                fwhm_kms=seg_fwhm,
            )
            model_norm_full, _ = normalize_model_sidebands(
                seg,
                model_broad_full,
                sideband_width=sideband_width,
                sideband_order=sideband_order,
            )

            model_corr_full, coeffs = _solve_sideband_multiplicative_poly(
                wave=wave_full,
                flux=flux_full,
                err=err_full,
                model=model_norm_full,
                used_mask=used_mask,
                order=sideband_poly_order,
            )

            model_full_list.append(model_corr_full.copy())
            coeffs_list.append(coeffs)
            i0 = i1

    elif forward_model == "native_interp":
        model_dense = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)

        model_wave_medium = infer_segments_wave_medium(
            segments,
            default=getattr(phoenix_lib, "phoenix_wave_medium", "vacuum"),
        )

        model_raw_list = build_phoenix_native_models_for_segments(
            segments=segments,
            phoenix_wave_native=np.asarray(phoenix_lib.wave, dtype=float),
            template_flux_native=model_dense,
            rv_kms=rv_kms,
            rv_bary_kms=rv_bary_kms,
            segment_fwhm_kms=segment_fwhm_kms,
            phoenix_wave_medium=model_wave_medium,
            model_margin_A=model_margin_A,
            bounds_use_fit_mask=True,
            extrapolate=True,
        )
        for seg, used_mask, model_broad_full in zip(segments, used_masks, model_raw_list):
            wave_full = np.asarray(seg.wave, dtype=float)
            flux_full = np.asarray(seg.flux, dtype=float)
            err_full = np.asarray(seg.err, dtype=float)

            model_norm_full, _ = normalize_model_sidebands(
                seg,
                model_broad_full,
                sideband_width=sideband_width,
                sideband_order=sideband_order,
            )

            model_corr_full, coeffs = _solve_sideband_multiplicative_poly(
                wave=wave_full,
                flux=flux_full,
                err=err_full,
                model=model_norm_full,
                used_mask=used_mask,
                order=sideband_poly_order,
            )

            model_full_list.append(model_corr_full.copy())
            coeffs_list.append(coeffs)
    else:
        raise ValueError("Unknown forward_model: {0}".format(forward_model))

    return model_full_list, coeffs_list, used_masks, excluded_masks


PEPSI_LEGACY_CENTERS_AIR = [6495.0, 6545.0, 6561.0, 8498.0, 8542.0, 8662.0]


def build_pepsi_legacy_windows(centers_air=None, halfwidth_A=10.0):
    centers = PEPSI_LEGACY_CENTERS_AIR if centers_air is None else centers_air
    out = []
    for c in centers:
        c = float(c)
        out.append(("legacy_{0:.1f}".format(c), c - float(halfwidth_A), c + float(halfwidth_A)))
    return out


def convert_air_windows_to_medium(window_defs_air, to_medium):
    to_medium = str(to_medium).strip().lower()

    if to_medium in ("air", "unknown", ""):
        return list(window_defs_air)

    if to_medium != "vacuum":
        raise ValueError("Unsupported wavelength medium: {0}".format(to_medium))

    out = []
    for label, wmin_air, wmax_air in window_defs_air:
        w_air = np.array([float(wmin_air), float(wmax_air)], dtype=float)
        w_new = convert_wavelength_medium(w_air, from_medium="air", to_medium="vacuum")
        out.append((label, float(w_new[0]), float(w_new[1])))

    return out


def apply_pepsi_wave_hypothesis(seg, hypothesis):
    hypothesis = str(hypothesis).strip().lower()

    if hypothesis == "unknown":
        meta = dict(seg.meta)
        meta["wave_medium"] = "unknown"
        return seg.copy(meta=meta, wave_medium="unknown", name=(seg.name or "seg") + "_unknown")

    if hypothesis == "air":
        meta = dict(seg.meta)
        meta["wave_medium"] = "air"
        return seg.copy(meta=meta, wave_medium="air", name=(seg.name or "seg") + "_air")

    if hypothesis == "vacuum":
        meta = dict(seg.meta)
        meta["wave_medium"] = "vacuum"
        return seg.copy(meta=meta, wave_medium="vacuum", name=(seg.name or "seg") + "_vacuum")

    if hypothesis == "air_to_vac":
        meta = dict(seg.meta)
        meta["wave_medium"] = "air"
        assumed_air = seg.copy(
            meta=meta,
            wave_medium="air",
        )
        converted = convert_segment_wavelength_medium(
            assumed_air,
            to_medium="vacuum",
            method="ciddor1996",
        )
        return converted.copy(
            name=(seg.name or "seg") + "_air2vac",
        ).sorted()

    raise ValueError("Unknown wavelength hypothesis: {0}".format(hypothesis))


def build_pepsi_normalized_mask(seg, flux_min=0.2, flux_max=1.1):
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)

    good = np.asarray(seg.mask, dtype=bool)
    good &= np.isfinite(wave) & np.isfinite(flux)
    good &= (flux > float(flux_min)) & (flux < float(flux_max))

    if seg.err is not None:
        err = np.asarray(seg.err, dtype=float)
        good &= np.isfinite(err) & (err > 0)

    return good
    

def build_pepsi_legacy_segments(
    input_segments,
    wave_hypothesis="air",
    centers_air=None,
    halfwidth_A=10.0,
    flux_min=0.2,
    flux_max=1.1,
    window_pad_A=2.0,
):
    window_defs_air = build_pepsi_legacy_windows(
        centers_air=centers_air,
        halfwidth_A=halfwidth_A,
    )

    working_input_segments = []
    fit_segments = []

    for seg0 in input_segments:
        legacy_mask = build_pepsi_normalized_mask(
            seg0,
            flux_min=flux_min,
            flux_max=flux_max,
        )
        seg0 = seg0.copy(mask=legacy_mask)
        seg = apply_pepsi_wave_hypothesis(seg0, wave_hypothesis)
        working_input_segments.append(seg)

        working_window_defs = convert_air_windows_to_medium(
            window_defs_air,
            to_medium=seg.wave_medium,
        )

        wlo = float(np.nanmin(seg.wave))
        whi = float(np.nanmax(seg.wave))
        present_defs = [
            (label, wmin, wmax)
            for label, wmin, wmax in working_window_defs
            if (wmax >= wlo) and (wmin <= whi)
        ]

        if len(present_defs) == 0:
            continue

        seg_windows = make_padded_window_segments(
            seg,
            [(wmin, wmax) for _label, wmin, wmax in present_defs],
            pad=window_pad_A,
            name_prefix="line",
        )

        for sw, win_def in zip(seg_windows, present_defs):
            sw.name = win_def[0]
            sw.meta["source_file"] = seg.meta.get("source_file")
            sw.meta["legacy_window_air"] = tuple(
                x for x in next(w for w in window_defs_air if w[0] == win_def[0])
            )
            sw.meta["legacy_window_working"] = tuple(win_def)
            sw.meta["legacy_window_medium"] = seg.wave_medium

        fit_segments.extend(seg_windows)

    if len(fit_segments) == 0:
        raise ValueError("No PEPSI legacy line windows overlap the supplied segment(s).")

    return working_input_segments, fit_segments, window_defs_air


def make_pepsi_legacy_cache_support_segments(
    input_segments,
    window_defs_air,
    window_pad_A=2.0,
):
    support_segments = []

    for seg in input_segments:
        working_window_defs = convert_air_windows_to_medium(
            window_defs_air,
            to_medium=seg.wave_medium,
        )

        wlo = float(np.nanmin(seg.wave))
        whi = float(np.nanmax(seg.wave))

        present_defs = [
            (label, wmin, wmax)
            for label, wmin, wmax in working_window_defs
            if (wmax >= wlo) and (wmin <= whi)
        ]

        for label, wmin, wmax in present_defs:
            keep = (
                np.isfinite(seg.wave) &
                (seg.wave >= float(wmin) - float(window_pad_A)) &
                (seg.wave <= float(wmax) + float(window_pad_A))
            )

            if not np.any(keep):
                continue

            n_keep = int(np.sum(keep))
            support_segments.append(
                seg.copy(
                    wave=seg.wave[keep],
                    flux=np.ones(n_keep, dtype=float),
                    err=np.ones(n_keep, dtype=float),
                    mask=np.ones(n_keep, dtype=bool),
                    meta=dict(seg.meta),
                    name="cache_support_{0}".format(label),
                )
            )

    if len(support_segments) == 0:
        raise ValueError("No PEPSI legacy cache support windows overlap the supplied segment(s).")

    return support_segments


def evaluate_pepsi_legacy_max_models(
    phoenix_lib,
    segments,
    model_wave_grid,
    model_wave_medium,
    teff,
    feh,
    logg,
    rv_kms,
    rv_bary_kms,
    R,
    model_margin_A,
):
    template_flux = np.asarray(phoenix_lib.evaluate(teff, feh, logg), dtype=float)
    segment_fwhm_kms = [segment_fwhm_kms_from_R(seg, R=R) for seg in segments]
    return build_phoenix_native_models_for_segments(
        segments=segments,
        phoenix_wave_native=model_wave_grid,
        template_flux_native=template_flux,
        rv_kms=float(rv_kms),
        rv_bary_kms=float(rv_bary_kms),
        segment_fwhm_kms=segment_fwhm_kms,
        phoenix_wave_medium=model_wave_medium,
        model_margin_A=model_margin_A,
        bounds_use_fit_mask=True,
        extrapolate=True,
    )


def pepsi_legacy_max_likelihood_terms(seg, model_full, log_err_scale=0.0):
    """
    Return likelihood terms for one window.

    The model is normalized by its maximum on the used pixels in the window,
    matching the old model/max(model) comparison. The errors are scaled by
    10**log_err_scale and used as variances in a Gaussian negative log-likelihood.
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = np.asarray(seg.err, dtype=float)
    model_full = np.asarray(model_full, dtype=float)
    used = np.asarray(seg.mask, dtype=bool)
    used &= np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    used &= np.isfinite(model_full)

    if np.sum(used) < 4:
        return np.inf, 0, np.full_like(model_full, np.nan, dtype=float), used

    mmax = float(np.nanmax(model_full[used]))
    if (not np.isfinite(mmax)) or mmax == 0.0:
        return np.inf, 0, np.full_like(model_full, np.nan, dtype=float), used

    model_norm = model_full / mmax
    sigma = (10.0 ** float(log_err_scale)) * err[used]
    var = sigma * sigma
    resid = flux[used] - model_norm[used]
    nll_terms = resid * resid / var + np.log(2.0 * np.pi * var)
    return float(np.sum(nll_terms)), int(np.sum(used)), model_norm, used
    
    
def segment_fwhm_kms_from_R(seg, R=None):
    if R is None:
        R = getattr(seg, "meta", {}).get("resolution_R", None)
    if R is None:
        return None
    R = float(R)
    if R <= 0:
        return None
    return C_KMS / R


def ensure_phoenix_native_interpolator_for_segments(
    segments,
    phoenix_lib,
    teff_grid,
    feh_grid,
    logg_grid,
    cache_path=None,
    model_margin_A=20.0,
    progress_callback=None,
):
    model_wave_grid, model_wave_medium = build_native_interp_wave_grid_for_segments(
        segments=segments,
        phoenix_lib=phoenix_lib,
        model_margin_A=model_margin_A,
    )

    teff_grid = np.asarray(teff_grid, dtype=float)
    feh_grid = np.asarray(feh_grid, dtype=float)
    logg_grid = np.asarray(logg_grid, dtype=float)

    phoenix_lib.ensure_interpolator(
        wave=model_wave_grid,
        teff_grid=teff_grid,
        feh_grid=feh_grid,
        logg_grid=logg_grid,
        cache_path=cache_path,
        observed_wave_medium=model_wave_medium,
        progress_callback=progress_callback,
    )

    return model_wave_grid, model_wave_medium


def pick_grid_range(grid, lo=None, hi=None):
    g = np.asarray(grid, dtype=float)
    m = np.ones_like(g, dtype=bool)
    if lo is not None:
        m &= (g >= float(lo))
    if hi is not None:
        m &= (g <= float(hi))
    out = g[m]
    if out.size == 0:
        raise ValueError("Requested PHOENIX grid range is empty.")
    return out
    
    
__all__ = [
    "BALMER_CENTERS_AIR",
    "BALMER_CENTERS_VAC",
    "XSHOOTER_BALMER_WINDOWS",
    "XSHOOTER_NOTEBOOK_CONT_WINDOWS",
    "XSHOOTER_BALMER_CORE_MASK_DEFAULT_A",
    "XSHOOTER_BALMER_CORE_MASK_CONSERVATIVE_A",
    "XshooterBalmerCase",
    "xshooter_balmer_windows",
    "attach_balmer_metadata",
    "normalize_segment_sidebands",
    "normalize_segments_sidebands",
    "normalize_model_sidebands",
    "solve_sideband_multiplicative_poly",
    "make_balmer_core_exclude_mask",
    "prepare_xshooter_balmer_case",
    "fit_case_lines_individually",
    "fit_phoenix_sideband_symmetric",
    "build_plot_models_for_segments",
    "PEPSI_LEGACY_CENTERS_AIR",
    "build_pepsi_legacy_windows",
    "convert_air_windows_to_medium",
    "apply_pepsi_wave_hypothesis",
    "build_pepsi_normalized_mask",
    "build_pepsi_legacy_segments",
    "make_pepsi_legacy_cache_support_segments",
    "segment_fwhm_kms_from_R",
    "ensure_phoenix_native_interpolator_for_segments",
    "pick_grid_range",
    "evaluate_pepsi_legacy_max_models",
    "pepsi_legacy_max_likelihood_terms",
]
