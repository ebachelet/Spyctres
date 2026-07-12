import json

import matplotlib
matplotlib.use("Agg")
import numpy as np

from Spyctres.io import SpectrumCollection, SpectrumSegment
from Spyctres.plotting import plot_fit_referee, plot_xsl_validation_payload
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
    path = tmp_path / "referee.png"

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
