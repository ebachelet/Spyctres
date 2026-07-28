import json

import matplotlib
matplotlib.use("Agg")
import numpy as np

from Spyctres.io import SpectrumCollection, SpectrumSegment
from Spyctres.plotting import (
    plot_diagnostic_windows,
    plot_fit_referee,
    plot_model_line_windows,
    plot_spectrum,
    plot_spectrum_audit,
    plot_xsl_validation_payload,
)
from Spyctres.preprocessing import build_mask
from Spyctres.results import PhoenixFitResult


def _fit_result_for_segments(segments):
    models = tuple(np.asarray(seg.flux, dtype=float) * 0.98 for seg in segments)
    used_masks = tuple(np.asarray(seg.mask, dtype=bool) for seg in segments)
    excluded_masks = tuple(np.zeros(seg.wave.size, dtype=bool) for seg in segments)
    coeffs = tuple(np.array([1.0, 0.0]) for _seg in segments)
    return PhoenixFitResult(
        summary={
            "success": True,
            "teff": 5750.0,
            "feh": 0.0,
            "logg": 4.4,
            "rv_kms": 0.0,
            "chi2_red": 1.1,
            "diagnostics": {
                "mask_fraction": 0.25,
                "n_dropped_segments": 0,
                "segment_diagnostics": [
                    {"name": seg.name, "n_fit": int(np.sum(seg.mask))}
                    for seg in segments
                ],
            },
            "quality_flags": ["ok"],
        },
        models=models,
        continuum_coefficients=coeffs,
        used_masks=used_masks,
        excluded_masks=excluded_masks,
    )


def test_plot_fit_referee_uses_result_input_spectrum_when_available():
    wave = np.linspace(5000.0, 5010.0, 25)
    segment = SpectrumSegment(
        wave,
        1.0 - 0.1 * np.exp(-0.5 * ((wave - 5005.0) / 1.0) ** 2),
        err=np.full(wave.size, 0.02),
        name="synthetic",
    )
    result = _fit_result_for_segments([segment])
    result = PhoenixFitResult(
        summary=result.summary,
        models=result.models,
        continuum_coefficients=result.continuum_coefficients,
        used_masks=result.used_masks,
        excluded_masks=result.excluded_masks,
        input_spectrum=segment,
    )

    fig, axes = plot_fit_referee(result)

    assert axes.shape == (1, 2)
    assert "synthetic" in axes[0, 0].get_title(loc="left")
    fig.clf()


def test_plot_fit_referee_saves_without_mutating_result(tmp_path):
    wave = np.linspace(5000.0, 5010.0, 25)
    segment = SpectrumSegment(
        wave,
        1.0 - 0.1 * np.exp(-0.5 * ((wave - 5005.0) / 1.0) ** 2),
        err=np.full(wave.size, 0.02),
        name="synthetic",
    )
    result = _fit_result_for_segments([segment])
    before = json.dumps(result.to_dict(include_arrays=False), sort_keys=True)
    path = tmp_path / "nested" / "plots" / "referee.png"

    fig, axes = plot_fit_referee(result, segment=segment, savepath=path)

    after = json.dumps(result.to_dict(include_arrays=False), sort_keys=True)
    assert path.exists()
    assert fig.spyctres_generated_files == {"referee_plot": str(path)}
    assert axes.shape == (1, 2)
    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "continuum shape" not in labels
    assert "mask=25%" in fig._suptitle.get_text()
    assert before == after
    fig.clf()


def test_plot_fit_referee_handles_multisegment_collection():
    first = SpectrumSegment(
        np.linspace(4000.0, 4010.0, 12),
        np.ones(12),
        err=np.full(12, 0.05),
        name="blue",
    )
    second = SpectrumSegment(
        np.linspace(6500.0, 6510.0, 15),
        np.ones(15),
        err=np.full(15, 0.05),
        name="red",
    )
    collection = SpectrumCollection([first, second])
    result = _fit_result_for_segments(collection.segments)

    fig, axes = plot_fit_referee(result, segment=collection)

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title(loc="left") == "blue"
    assert axes[1, 0].get_title(loc="left") == "red"
    fig.clf()


def test_plot_fit_referee_stacked_layout_uses_full_width_rows():
    wave = np.linspace(5000.0, 5010.0, 25)
    segment = SpectrumSegment(
        wave,
        1.0 - 0.1 * np.exp(-0.5 * ((wave - 5005.0) / 1.0) ** 2),
        err=np.full(wave.size, 0.02),
        name="synthetic",
    )
    result = _fit_result_for_segments([segment])

    fig, axes = plot_fit_referee(
        result,
        segment=segment,
        layout="stacked",
        figsize_per_segment=(16.0, 6.4),
    )

    assert axes.shape == (1, 2)
    assert fig.get_size_inches()[0] == 16.0
    assert axes[0, 0].get_position().width == axes[0, 1].get_position().width
    assert axes[0, 0].get_position().y0 > axes[0, 1].get_position().y0
    assert axes[0, 1].get_xlabel() == "Wavelength (Å)"
    fig.clf()


def test_plot_fit_referee_defaults_to_fitted_wavelength_span():
    wave = np.linspace(3000.0, 5600.0, 80)
    flux = np.ones(wave.size)
    used = (wave >= 3800.0) & (wave <= 5200.0)
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.05),
        mask=used,
        name="wide_segment",
    )
    result = _fit_result_for_segments([segment])

    fig, axes = plot_fit_referee(result, segment=segment, layout="stacked")

    xlo, xhi = axes[0, 0].get_xlim()
    assert xlo > wave.min()
    assert xhi < wave.max()
    assert xlo < wave[used].min()
    assert xhi > wave[used].max()
    fig.clf()


def test_plot_fit_referee_does_not_draw_model_on_unused_pixels():
    wave = np.linspace(5000.0, 5010.0, 11)
    flux = np.ones(wave.size)
    used = np.ones(wave.size, dtype=bool)
    used[:2] = False
    used[-2:] = False
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.05),
        mask=used,
        name="partly_used",
    )
    result = _fit_result_for_segments([segment])

    fig, axes = plot_fit_referee(result, segment=segment, xlim_mode="all")

    model_lines = [
        line for line in axes[0, 0].lines
        if line.get_label().startswith("continuum-adjusted model")
    ]
    assert len(model_lines) == 1
    y_model = model_lines[0].get_ydata()
    assert np.isnan(y_model[:2]).all()
    assert np.isnan(y_model[-2:]).all()
    assert np.isfinite(y_model[2:-2]).all()

    residual_lines = axes[0, 1].lines
    y_resid = residual_lines[0].get_ydata()
    assert np.isnan(y_resid[:2]).all()
    assert np.isnan(y_resid[-2:]).all()
    assert np.isfinite(y_resid[2:-2]).all()
    fig.clf()


def test_plot_model_line_windows_is_generic_and_shows_masked_model_span(tmp_path):
    wave = np.linspace(4830.0, 4898.0, 21)
    flux = 1.0 - 0.20 * np.exp(-0.5 * ((wave - 4861.33) / 5.0) ** 2)
    model = 1.0 - 0.18 * np.exp(-0.5 * ((wave - 4861.33) / 5.5) ** 2)
    reference = 1.0 - 0.15 * np.exp(-0.5 * ((wave - 4861.33) / 5.5) ** 2)
    used = np.ones_like(wave, dtype=bool)
    used[8:12] = False
    path = tmp_path / "nested" / "line_windows.png"

    fig, axes = plot_model_line_windows(
        wave,
        flux,
        [{"label": "generic H-beta-like window", "limits_A": (4830.0, 4898.0), "markers_A": [4861.33]}],
        models=[
            {
                "flux": model,
                "label": "best model",
                "color": "tab:red",
                "masked_label": "best model masked span",
            },
            {
                "flux": reference,
                "label": "comparison model",
                "color": "tab:blue",
                "masked_label": "comparison masked span",
            },
        ],
        used_mask=used,
        model_used_masks=[used, used],
        savepath=path,
        footer="generic diagnostic",
    )

    assert path.exists()
    assert fig.spyctres_generated_files == {"line_window_plot": str(path)}
    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "best model" in labels
    assert "comparison model" in labels
    assert "best model masked span" in labels
    assert "comparison masked span" in labels
    masked_lines = [
        line for line in axes[0, 0].lines if line.get_label().endswith("masked span")
    ]
    assert masked_lines
    assert all(line.get_linestyle() == "--" for line in masked_lines)
    fig.clf()


def test_plot_fit_referee_visible_ylim_includes_excluded_line_core():
    wave = np.linspace(4840.0, 4882.0, 121)
    core = np.exp(-0.5 * ((wave - 4861.3) / 2.0) ** 2)
    flux = 1.0 - 0.85 * core
    model = 1.0 - 0.80 * core
    used = np.abs(wave - 4861.3) > 6.0
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.03),
        mask=used,
        name="hbeta_core_masked",
    )
    result = PhoenixFitResult(
        summary={
            "success": True,
            "teff": 9000.0,
            "feh": 0.0,
            "logg": 3.0,
            "rv_kms": 0.0,
            "chi2_red": 1.0,
            "diagnostics": {
                "segment_diagnostics": [{"name": segment.name, "n_fit": int(np.sum(used))}],
            },
        },
        models=(model,),
        continuum_coefficients=(np.array([1.0, 0.0]),),
        used_masks=(used,),
        excluded_masks=(~used,),
    )

    fig, axes = plot_fit_referee(
        result,
        segment=segment,
        layout="stacked",
        flux_ylim_mode="visible",
    )

    ylo, yhi = axes[0, 0].get_ylim()
    assert ylo < float(np.nanmin(flux))
    assert yhi > float(np.nanmax(flux))
    excluded_model_lines = [
        line for line in axes[0, 0].lines
        if line.get_label() == "continuum-adjusted model (excluded pixels)"
    ]
    assert len(excluded_model_lines) == 1
    assert excluded_model_lines[0].get_linestyle() == "--"
    y_excluded = excluded_model_lines[0].get_ydata()
    assert np.isfinite(y_excluded[~used]).any()
    fig.clf()


def test_plot_fit_referee_can_hide_model_on_excluded_pixels():
    wave = np.linspace(4840.0, 4882.0, 121)
    flux = np.ones_like(wave)
    model = np.ones_like(wave) * 0.98
    used = np.abs(wave - 4861.3) > 5.0
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.03),
        mask=used,
        name="masked_core",
    )
    result = PhoenixFitResult(
        summary={"success": True, "chi2_red": 1.0},
        models=(model,),
        continuum_coefficients=(np.array([1.0, 0.0]),),
        used_masks=(used,),
        excluded_masks=(~used,),
    )

    fig, axes = plot_fit_referee(
        result,
        segment=segment,
        show_model_on_excluded=False,
    )

    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "continuum-adjusted model (excluded pixels)" not in labels
    fig.clf()


def test_plot_fit_referee_can_hide_raw_model_curve():
    wave = np.linspace(5000.0, 5010.0, 25)
    segment = SpectrumSegment(
        wave,
        np.ones(wave.size),
        err=np.full(wave.size, 0.03),
        name="scaled_only",
    )
    result = _fit_result_for_segments([segment])

    fig, axes = plot_fit_referee(
        result,
        segment=segment,
        show_raw_model=False,
    )

    labels = [line.get_label() for line in axes[0, 0].lines]
    assert not any(label.startswith("PHOENIX+LSF") for label in labels)
    assert any(label.startswith("continuum-adjusted model") for label in labels)
    fig.clf()


def test_plot_fit_referee_can_annotate_feature_regions():
    wave = np.linspace(4400.0, 4460.0, 40)
    segment = SpectrumSegment(
        wave,
        np.ones(wave.size),
        err=np.full(wave.size, 0.05),
        name="dib_demo",
    )
    result = _fit_result_for_segments([segment])

    fig, axes = plot_fit_referee(
        result,
        segment=segment,
        feature_regions=[{"name": "DIB 4428", "region_A": [4416.8, 4440.8]}],
    )

    assert any(text.get_text() == "DIB 4428" for text in axes[0, 0].texts)
    fig.clf()


def test_plot_spectrum_audit_is_generic_and_labelled(tmp_path):
    wave = np.linspace(4100.0, 4000.0, 80)
    flux = 1.0 + 0.02 * np.sin(np.arange(wave.size) / 4.0)
    flux[10:14] = 0.0
    mask = np.ones(wave.size, dtype=bool)
    mask[30:36] = False
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.02),
        mask=mask,
        meta={
            "warning_regions": [
                {"id": "generic_warning", "region_A": [4050.0, 4060.0]},
            ],
        },
        name="generic_uploaded_spectrum",
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
    )
    diagnostic_selection = {
        "selected": [
            {
                "id": "generic_window",
                "label": "Generic window",
                "region_A": [4020.0, 4040.0],
            }
        ]
    }

    fig, axes = plot_spectrum_audit(
        SpectrumCollection([segment]),
        diagnostic_selection=diagnostic_selection,
        warning_regions=[{"id": "manual_warning", "region_A": [4070.0, 4080.0]}],
        max_plot_points=25,
    )

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Raw flux\n(robust y-scale)"
    assert axes[1].get_ylabel().startswith("Flux / local")
    assert axes[2].get_ylabel() == "Mask\nTrue=use"
    legend_labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert "suggested diagnostic window" in legend_labels
    assert "metadata warning region" in legend_labels
    assert "near-zero block" in legend_labels
    assert "diagnostic only" in axes[2].texts[0].get_text()
    fig.clf()


def test_plot_spectrum_facade_draws_single_panel_and_overlays():
    wave = np.linspace(4300.0, 4450.0, 240)
    flux = 1.0 - 0.2 * np.exp(-0.5 * ((wave - 4340.47) / 4.0) ** 2)
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.02),
        mask=np.ones(wave.size, dtype=bool),
        name="facade",
        wave_medium="air",
    )
    mask = build_mask(
        names="dib_4428",
        tellurics="warn",
    )
    selection = {
        "selected": [
            {
                "id": "hgamma",
                "label": "Hgamma",
                "region_A": [4310.0, 4372.0],
            }
        ]
    }

    fig, ax = plot_spectrum(
        segment,
        mask=mask,
        diagnostic_selection=selection,
        show_nonstellar=True,
    )

    assert ax.get_xlabel() == "Wavelength [Å]"
    assert ax.lines
    assert ax.patches
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "suggested diagnostic window" in labels
    assert "explicit exclusion mask" in labels
    assert "warning region" in labels
    fig.clf()


def test_plot_spectrum_facade_show_masks_uses_audit_view():
    segment = SpectrumSegment(
        np.linspace(4000.0, 4100.0, 80),
        np.ones(80),
        mask=np.ones(80, dtype=bool),
    )

    fig, axes = plot_spectrum(segment, show_masks=True)

    assert len(axes) == 3
    assert axes[2].get_ylabel() == "Mask\nTrue=use"
    fig.clf()


def test_plot_diagnostic_windows_selects_when_needed():
    segment = SpectrumSegment(
        np.linspace(4300.0, 4380.0, 200),
        np.ones(200),
        err=np.full(200, 0.02),
        mask=np.ones(200, dtype=bool),
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="observed",
    )

    fig, ax = plot_diagnostic_windows(segment)

    assert ax.patches
    fig.clf()


def test_xsl_validation_payload_defaults_to_global_display_scaling():
    payload = {
        "display_defaults": {"scale_mode": "global"},
        "segments": [
            {
                "name": "UVB",
                "wave_A": [4000.0, 4001.0],
                "observed_flux": [10.0, 10.0],
                "model_flux": [9.0, 11.0],
                "used": [True, True],
            },
            {
                "name": "VIS",
                "wave_A": [6000.0, 6001.0],
                "observed_flux": [20.0, 20.0],
                "model_flux": [18.0, 22.0],
                "used": [True, True],
            },
        ],
    }

    fig, axes = plot_xsl_validation_payload(payload)

    observed_lines = axes[0].lines[0], axes[0].lines[2]
    assert np.allclose(observed_lines[0].get_ydata(), [10.0 / 15.0, 10.0 / 15.0])
    assert np.allclose(observed_lines[1].get_ydata(), [20.0 / 15.0, 20.0 / 15.0])
    assert axes[0].get_ylabel() == "Flux / global median"
    assert "One display scale per target" in axes[0].texts[0].get_text()
    fig.clf()


def test_xsl_validation_payload_per_segment_scaling_is_labelled_diagnostic():
    payload = {
        "segments": [
            {
                "wave_A": [4000.0, 4001.0],
                "observed_flux": [10.0, 10.0],
                "model_flux": [10.0, 10.0],
                "used": [True, True],
            },
            {
                "wave_A": [6000.0, 6001.0],
                "observed_flux": [20.0, 20.0],
                "model_flux": [20.0, 20.0],
                "used": [True, True],
            },
        ],
    }

    fig, axes = plot_xsl_validation_payload(payload, scale_mode="per_segment")

    observed_lines = axes[0].lines[0], axes[0].lines[2]
    assert np.allclose(observed_lines[0].get_ydata(), [1.0, 1.0])
    assert np.allclose(observed_lines[1].get_ydata(), [1.0, 1.0])
    assert axes[0].get_ylabel() == "Flux / segment median"
    assert "diagnostic line-shape view only" in axes[0].texts[0].get_text()
    fig.clf()
