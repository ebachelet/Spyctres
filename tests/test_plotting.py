import json

import matplotlib
matplotlib.use("Agg")
import numpy as np

from Spyctres.io import SpectrumCollection, SpectrumSegment
from Spyctres.plotting import (
    plot_diagnostic_windows,
    plot_fit_comparison_line_windows,
    plot_fit_referee,
    plot_line_fit_comparison,
    plot_model_line_windows,
    plot_prepared_line_window_diagnostics,
    plot_spectrum,
    plot_spectrum_audit,
    plot_spectrum_line_windows,
    plot_xsl_validation_payload,
)
from Spyctres.preprocessing import build_mask
from Spyctres.linefitting import LineFitResult
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


def test_plot_model_line_windows_footer_does_not_overlap_axes():
    wave = np.linspace(4410.0, 4450.0, 51)
    flux = np.ones_like(wave)
    model = np.ones_like(wave) * 0.99

    fig, axes = plot_model_line_windows(
        wave,
        flux,
        [{"label": "feature window", "limits_A": (4410.0, 4450.0)}],
        models=[{"flux": model, "label": "model"}],
        show_residuals=True,
        residual_kind="fractional",
        footer=(
            "Orange = candidate catalog feature overlap, not a mask or correction. "
            "Inspect before deciding whether to run a controlled sensitivity test."
        ),
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_bbox = fig.texts[-1].get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    bottom_axes_y = min(
        ax.get_position().y0 for ax in axes.ravel() if ax.get_visible()
    )

    assert footer_bbox.y1 < bottom_axes_y
    fig.clf()


def test_plot_model_line_windows_shades_annotation_regions_without_masking():
    wave = np.linspace(4410.0, 4450.0, 51)
    flux = np.ones_like(wave)
    model = np.ones_like(wave) * 0.99

    fig, axes = plot_model_line_windows(
        wave,
        flux,
        [{"label": "user-noticed residual", "limits_A": (4410.0, 4450.0)}],
        models=[{"flux": model, "label": "model"}],
        annotation_regions=[
            {
                "label": "candidate broad feature",
                "region_A": (4416.8, 4440.8),
                "center_A": 4428.8,
            }
        ],
    )

    labels = [item.get_label() for item in axes[0, 0].patches]
    assert "candidate: candidate broad feature" in labels
    line_labels = [line.get_label() for line in axes[0, 0].lines]
    assert "model" in line_labels
    fig.clf()


def test_plot_model_line_windows_accepts_fit_result_directly():
    wave = np.linspace(4830.0, 4898.0, 41)
    segment = SpectrumSegment(
        wave,
        1.0 - 0.2 * np.exp(-0.5 * ((wave - 4861.33) / 5.0) ** 2),
        err=np.full(wave.size, 0.03),
        name="hbeta_demo",
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

    fig, axes = plot_model_line_windows(
        result,
        windows=[
            {
                "label": "Hbeta",
                "limits_A": (4830.0, 4898.0),
                "markers_A": [4861.33],
            }
        ],
    )

    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "continuum-adjusted model" in labels
    fig.clf()


def test_plot_prepared_line_window_diagnostics_saves_model_grid(tmp_path):
    first_wave = np.linspace(6490.0, 6500.0, 25)
    second_wave = np.linspace(6556.0, 6566.0, 31)
    first = SpectrumSegment(
        first_wave,
        1.0 - 0.10 * np.exp(-0.5 * ((first_wave - 6495.0) / 0.35) ** 2),
        err=np.full(first_wave.size, 0.02),
        mask=np.ones(first_wave.size, dtype=bool),
        name="legacy_6495.0",
        meta={"legacy_window_air": ("legacy_6495.0", 6485.0, 6505.0)},
    )
    second_mask = np.ones(second_wave.size, dtype=bool)
    second_mask[:4] = False
    second_flux = 1.0 - 0.30 * np.exp(-0.5 * ((second_wave - 6561.0) / 1.5) ** 2)
    second_flux[0] = 9.0
    second = SpectrumSegment(
        second_wave,
        second_flux,
        err=np.full(second_wave.size, 0.03),
        mask=second_mask,
        name="legacy_6561.0",
        meta={"legacy_window_air": ("legacy_6561.0", 6551.0, 6571.0)},
    )
    models = [first.flux * 0.99, second.flux * 1.01]
    path = tmp_path / "prepared" / "pepsi_grid.png"

    fig, axes = plot_prepared_line_window_diagnostics(
        SpectrumCollection([first, second]),
        models=models,
        used_masks=[first.mask, second.mask],
        footer="PEPSI legacy line-window validation: synthetic test.",
        savepath=path,
        ncols=2,
    )

    assert path.exists()
    assert fig.spyctres_generated_files == {
        "prepared_line_window_plot": str(path)
    }
    assert axes.shape == (1, 2)
    assert axes[0, 0].get_title() == "6495.0"
    assert axes[0, 1].get_title() == "6561.0"
    assert axes[0, 0].get_ylim() == axes[0, 1].get_ylim()
    assert axes[0, 0].get_ylim()[1] < 2.0
    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "Model" in labels
    legend = axes[0, 0].get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Data" in legend_labels
    fig.clf()


def test_plot_model_line_windows_can_show_fit_result_residual_panels():
    wave = np.linspace(4830.0, 4898.0, 41)
    flux = 1.0 - 0.2 * np.exp(-0.5 * ((wave - 4861.33) / 5.0) ** 2)
    segment = SpectrumSegment(
        wave,
        flux,
        err=np.full(wave.size, 0.03),
        name="hbeta_demo",
    )
    result = _fit_result_for_segments([segment])
    result = PhoenixFitResult(
        summary=result.summary,
        models=(flux * 0.98,),
        continuum_coefficients=result.continuum_coefficients,
        used_masks=result.used_masks,
        excluded_masks=result.excluded_masks,
        input_spectrum=segment,
    )

    fig, axes = plot_model_line_windows(
        result,
        windows=[
            {
                "label": "Hbeta",
                "limits_A": (4830.0, 4898.0),
                "markers_A": [4861.33],
            }
        ],
        show_residuals=True,
        residual_kind="pull",
    )

    assert axes.shape == (2, 1)
    assert axes[1, 0].get_ylabel() == "(D-M)/σ"
    assert any(line.get_label().endswith("residual") for line in axes[1, 0].lines)
    fig.clf()


def test_plot_model_line_windows_accepts_multi_segment_fit_result():
    hgamma = SpectrumSegment(
        np.linspace(4310.0, 4372.0, 41),
        np.ones(41),
        err=np.full(41, 0.03),
        name="Hgamma",
    )
    hbeta = SpectrumSegment(
        np.linspace(4830.0, 4898.0, 45),
        np.ones(45),
        err=np.full(45, 0.03),
        name="Hbeta",
    )
    collection = SpectrumCollection([hgamma, hbeta])
    result = _fit_result_for_segments(collection.segments)
    result = PhoenixFitResult(
        summary=result.summary,
        models=result.models,
        continuum_coefficients=result.continuum_coefficients,
        used_masks=result.used_masks,
        excluded_masks=result.excluded_masks,
        input_spectrum=collection,
    )

    fig, axes = plot_model_line_windows(
        result,
        windows=[
            {"label": "Hgamma", "limits_A": (4310.0, 4372.0)},
            {"label": "Hbeta", "limits_A": (4830.0, 4898.0)},
        ],
        show_residuals=True,
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "Hgamma"
    assert axes[0, 1].get_title() == "Hbeta"
    assert axes[1, 0].get_ylabel() == "(D-M)/σ"
    fig.clf()


def test_plot_fit_comparison_line_windows_overplots_joint_and_line_results(tmp_path):
    hgamma_wave = np.linspace(4310.0, 4372.0, 41)
    hbeta_wave = np.linspace(4830.0, 4898.0, 45)
    hgamma_flux = 1.0 - 0.2 * np.exp(-0.5 * ((hgamma_wave - 4340.47) / 5.0) ** 2)
    hbeta_flux = 1.0 - 0.2 * np.exp(-0.5 * ((hbeta_wave - 4861.33) / 5.5) ** 2)
    hgamma = SpectrumSegment(
        hgamma_wave,
        hgamma_flux,
        err=np.full(hgamma_wave.size, 0.03),
        name="Hgamma",
    )
    hbeta = SpectrumSegment(
        hbeta_wave,
        hbeta_flux,
        err=np.full(hbeta_wave.size, 0.03),
        name="Hbeta",
    )
    collection = SpectrumCollection([hgamma, hbeta])
    joint_base = _fit_result_for_segments(collection.segments)
    joint = PhoenixFitResult(
        summary=joint_base.summary,
        models=joint_base.models,
        continuum_coefficients=joint_base.continuum_coefficients,
        used_masks=joint_base.used_masks,
        excluded_masks=joint_base.excluded_masks,
        input_spectrum=collection,
    )
    hgamma_base = _fit_result_for_segments([hgamma])
    hgamma_only = PhoenixFitResult(
        summary=hgamma_base.summary,
        models=(hgamma_flux * 1.01,),
        continuum_coefficients=hgamma_base.continuum_coefficients,
        used_masks=hgamma_base.used_masks,
        excluded_masks=hgamma_base.excluded_masks,
        input_spectrum=hgamma,
    )
    hbeta_base = _fit_result_for_segments([hbeta])
    hbeta_only = PhoenixFitResult(
        summary=hbeta_base.summary,
        models=(hbeta_flux * 0.99,),
        continuum_coefficients=hbeta_base.continuum_coefficients,
        used_masks=hbeta_base.used_masks,
        excluded_masks=hbeta_base.excluded_masks,
        input_spectrum=hbeta,
    )
    path = tmp_path / "comparison" / "line_windows.png"

    fig, axes = plot_fit_comparison_line_windows(
        [joint, hgamma_only, hbeta_only],
        labels=["joint Balmer", "Hγ only", "Hβ only"],
        windows=[
            {"label": "Hgamma", "limits_A": (4310.0, 4372.0)},
            {"label": "Hbeta", "limits_A": (4830.0, 4898.0)},
        ],
        savepath=path,
        footer="Compare variants in the same line windows before interpreting stability.",
    )

    assert path.exists()
    assert fig.spyctres_generated_files == {
        "fit_comparison_line_window_plot": str(path)
    }
    assert axes.shape == (1, 2)
    hgamma_labels = [line.get_label() for line in axes[0, 0].lines]
    hbeta_labels = [line.get_label() for line in axes[0, 1].lines]
    assert "joint Balmer" in hgamma_labels
    assert "Hγ only" in hgamma_labels
    assert "Hβ only" not in hgamma_labels
    assert "joint Balmer" in hbeta_labels
    assert "Hβ only" in hbeta_labels
    assert "Hγ only" not in hbeta_labels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_bbox = fig.texts[-1].get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    bottom_axes_y = min(
        ax.get_position().y0 for ax in axes.ravel() if ax.get_visible()
    )
    assert footer_bbox.y1 < bottom_axes_y
    fig.clf()


def test_plot_spectrum_line_windows_observed_only(tmp_path):
    wave = np.linspace(5150.0, 5208.0, 31)
    flux = 1.0 - 0.2 * np.exp(-0.5 * ((wave - 5172.7) / 3.0) ** 2)
    valid = np.ones_like(wave, dtype=bool)
    valid[:3] = False
    path = tmp_path / "observed" / "line_windows.png"

    fig, axes = plot_spectrum_line_windows(
        wave,
        flux,
        [{"label": "Mg-like window", "limits_A": (5150.0, 5208.0)}],
        valid_mask=valid,
        savepath=path,
    )

    assert path.exists()
    assert fig.spyctres_generated_files == {"line_window_plot": str(path)}
    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "observed" in labels
    assert not axes[0, 0].texts
    fig.clf()


def test_plot_spectrum_line_windows_distinguishes_molecular_bands():
    wave = np.linspace(7000.0, 7165.0, 80)
    flux = 1.0 - 0.1 * np.sin((wave - wave[0]) / 20.0)

    fig, axes = plot_spectrum_line_windows(
        wave,
        flux,
        [
            {
                "id": "tio_7050",
                "label": "TiO 7050 band",
                "region_A": (7000.0, 7165.0),
                "feature_family": ["molecular"],
            }
        ],
    )

    assert axes[0, 0].get_title().startswith("molecular band:")
    assert any(
        patch.get_label() == "molecular-band window"
        for patch in axes[0, 0].patches
    )
    fig.clf()


def test_plot_line_fit_comparison_saves_metric_summary(tmp_path):
    wave = tuple(np.linspace(4845.0, 4875.0, 20))
    flux = tuple(np.ones(20))
    first = LineFitResult(
        line_name="Hbeta",
        rest_wave=4861.33,
        kind="absorption",
        success=True,
        rv_kms=5.0,
        equivalent_width_A=0.8,
        fwhm_A=7.5,
        chi2_red=20.0,
        n_points=20,
        mask_fraction=0.1,
        flags=("high_chi2_red",),
        wave=wave,
        flux=flux,
        model_flux=flux,
        continuum=flux,
        residuals=tuple(np.zeros(20)),
    )
    second = LineFitResult(
        line_name="Mg II 4481",
        rest_wave=4481.13,
        kind="absorption",
        success=True,
        rv_kms=4.0,
        equivalent_width_A=0.12,
        fwhm_A=1.2,
        chi2_red=1.4,
        n_points=18,
        mask_fraction=0.0,
        flags=("ok",),
        wave=wave,
        flux=flux,
        model_flux=flux,
        continuum=flux,
        residuals=tuple(np.zeros(20)),
    )
    path = tmp_path / "line" / "comparison.png"

    fig, axes = plot_line_fit_comparison(
        [first, second],
        labels=["broad Balmer", "narrow Mg"],
        savepath=path,
    )

    assert path.exists()
    assert fig.spyctres_generated_files == {"line_fit_comparison_plot": str(path)}
    assert axes.shape == (4,)
    assert axes[-1].get_xticklabels()[0].get_text() == "broad Balmer"
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


def test_plot_spectrum_facade_saves_single_and_audit_views(tmp_path):
    segment = SpectrumSegment(
        np.linspace(4000.0, 4100.0, 80),
        np.ones(80),
        mask=np.ones(80, dtype=bool),
        name="save_demo",
    )
    single_path = tmp_path / "nested" / "single.png"
    audit_path = tmp_path / "nested" / "audit.png"

    fig, _ax = plot_spectrum(segment, savepath=single_path)

    assert single_path.exists()
    assert fig.spyctres_generated_files == {"spectrum_plot": str(single_path)}
    fig.clf()

    fig, axes = plot_spectrum(segment, show_masks=True, savepath=audit_path)

    assert audit_path.exists()
    assert fig.spyctres_generated_files == {"spectrum_audit_plot": str(audit_path)}
    assert len(axes) == 3
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
