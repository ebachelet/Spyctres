import pytest

import json
import numpy as np
import matplotlib.pyplot as plt

from Spyctres.io import ResolutionDescriptor, SpectrumCollection, SpectrumSegment
from Spyctres.recipes import (
    BALMER_CENTERS_AIR,
    BALMER_CENTERS_VAC,
    XshooterBalmerCase,
    _sideband_fit_parameter_count,
    attach_balmer_metadata,
    fit_case_lines_individually,
    make_balmer_core_exclude_mask,
    normalize_segment_sidebands,
    prepare_xshooter_balmer_case,
)
from Spyctres.preprocessing import exclusion_mask
from Spyctres.waveutils import convert_wavelength_medium


def test_sideband_parameter_count_uses_the_sideband_polynomial_order():
    assert _sideband_fit_parameter_count(3, 1) == 10
    assert _sideband_fit_parameter_count(3, 2) == 13


def test_sideband_parameter_count_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_segments"):
        _sideband_fit_parameter_count(0, 1)
    with pytest.raises(ValueError, match="sideband_poly_order"):
        _sideband_fit_parameter_count(1, -1)


def test_balmer_center_constants_distinguish_air_and_vacuum():
    assert BALMER_CENTERS_AIR["Hβ"] == pytest.approx(4861.33)

    expected_vac = convert_wavelength_medium(
        np.array([BALMER_CENTERS_AIR["Hβ"]], dtype=float),
        from_medium="air",
        to_medium="vacuum",
    )[0]

    assert BALMER_CENTERS_VAC["Hβ"] == pytest.approx(expected_vac)
    assert BALMER_CENTERS_VAC["Hβ"] > BALMER_CENTERS_AIR["Hβ"] + 1.0


def test_balmer_core_mask_uses_air_centers_for_air_spectra():
    mask_air = make_balmer_core_exclude_mask(
        core_halfwidth=0.05,
        wave_medium="air",
    )

    for label in ("Hδ", "Hγ", "Hβ"):
        wave = np.array(
            [
                BALMER_CENTERS_AIR[label],
                BALMER_CENTERS_VAC[label],
            ],
            dtype=float,
        )
        masked = mask_air(wave)

        assert masked[0]
        assert not masked[1]


def test_attach_balmer_metadata_records_air_vacuum_and_data_centers():
    air_segment = SpectrumSegment(
        wave=np.linspace(4320.0, 4360.0, 21),
        flux=np.ones(21),
        wave_medium="air",
        name="Hγ",
    )
    vac_segment = SpectrumSegment(
        wave=np.linspace(4320.0, 4360.0, 21),
        flux=np.ones(21),
        wave_medium="vacuum",
        name="Hγ",
    )

    attach_balmer_metadata([air_segment, vac_segment])

    assert air_segment.meta["line_center_air"] == pytest.approx(
        BALMER_CENTERS_AIR["Hγ"]
    )
    assert air_segment.meta["line_center_vac"] == pytest.approx(
        BALMER_CENTERS_VAC["Hγ"]
    )
    assert air_segment.meta["line_center_data"] == pytest.approx(
        BALMER_CENTERS_AIR["Hγ"]
    )
    assert vac_segment.meta["line_center_data"] == pytest.approx(
        BALMER_CENTERS_VAC["Hγ"]
    )


def test_sideband_normalization_preserves_scientific_metadata():
    resolution = ResolutionDescriptor(quantity="R", value=5100.0)
    segment = SpectrumSegment(
        wave=np.linspace(4000.0, 4020.0, 21),
        flux=np.linspace(1.0, 1.1, 21),
        err=np.full(21, 0.01),
        wave_medium="air",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        stellar_rv_applied_kms=12.0,
        resolution=resolution,
    )

    normalized, _ = normalize_segment_sidebands(segment)

    assert normalized.wave_medium == "air"
    assert normalized.observer_frame == "barycentric"
    assert normalized.stellar_rest_status == "corrected"
    assert normalized.stellar_rv_applied_kms == 12.0
    assert normalized.resolution is resolution


def test_xshooter_balmer_case_exposes_summary_collection_and_masks():
    wave = np.linspace(3950.0, 5050.0, 1200)
    segment = SpectrumSegment(
        wave=wave,
        flux=1.0 + 1e-4 * (wave - np.nanmedian(wave)),
        err=np.full_like(wave, 0.02),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(
            quantity="R",
            value=5400.0,
            source="test metadata",
        ),
        name="uvb",
    )

    case = prepare_xshooter_balmer_case(
        segment,
        window_mode="notebook",
        window_pad=5.0,
        norm_mode="sideband",
        sideband_width=10.0,
        sideband_order=1,
        core_mask=6.0,
    )

    assert isinstance(case, XshooterBalmerCase)
    assert isinstance(case.collection, SpectrumCollection)
    assert len(case.collection.segments) == 3
    assert case.fit_regions == (
        (3980.0, 4220.0),
        (4220.0, 4480.0),
        (4700.0, 5020.0),
    )
    assert case.fit_regions_by_segment == (
        ((3980.0, 4220.0),),
        ((4220.0, 4480.0),),
        ((4700.0, 5020.0),),
    )
    assert [item["label"] for item in case.fit_windows] == ["Hδ", "Hγ", "Hβ"]
    assert [mask.name for mask in case.exclusion_masks] == ["balmer_core"]
    assert len(case.valid_masks) == 3
    assert all(mask.dtype == bool for mask in case.valid_masks)
    assert all(np.count_nonzero(mask) < mask.size for mask in case.valid_masks)
    extra_mask = exclusion_mask(
        "manual_test_region",
        lambda w: (np.asarray(w) >= 4300.0) & (np.asarray(w) <= 4310.0),
    )
    combined = case.combined_exclusion_masks([extra_mask])
    assert [mask.name for mask in combined] == ["balmer_core", "manual_test_region"]
    assert np.count_nonzero(case.valid_masks_for([extra_mask])[1]) < np.count_nonzero(
        case.valid_masks[1]
    )

    summary = case.summary()
    assert summary["recipe"] == "prepare_xshooter_balmer_case"
    assert summary["mask_true_means"].startswith("valid_masks_true_means_usable")
    assert summary["norm_mode"] == "sideband"
    assert summary["core_mask_halfwidth_A"] == 6.0
    assert summary["resolution"]["value"] == 5400.0
    assert summary["total_valid_pixels"] < summary["total_pixels"]
    assert summary["segments"][0]["fit_region_A"] == [3980.0, 4220.0]
    assert summary["segments"][0]["continuum_sidebands_A"]
    assert case.fit_segments[0].meta["fit_region_A"] == [3980.0, 4220.0]
    assert case.fit_segments[0].meta["support_region_A"][0] < 3980.0
    assert "Hδ" in case.summary_text()
    json.dumps(case.to_dict(), allow_nan=False)


def test_xshooter_balmer_case_keeps_windows_aligned_when_coverage_is_partial():
    wave = np.linspace(4240.0, 4500.0, 400)
    segment = SpectrumSegment(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.02),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
        resolution=ResolutionDescriptor(quantity="R", value=5400.0),
        name="uvb_partial",
    )

    case = prepare_xshooter_balmer_case(
        segment,
        window_mode="notebook",
        norm_mode="poly",
        core_mask=None,
    )

    assert [segment.name for segment in case.fit_segments] == ["Hγ"]
    assert case.fit_regions == ((4220.0, 4480.0),)
    assert case.fit_regions_by_segment == (((4220.0, 4480.0),),)
    assert case.fit_segments[0].meta["fit_region_A"] == [4220.0, 4480.0]


def test_xshooter_balmer_case_plot_preparation(tmp_path):
    wave = np.linspace(3950.0, 5050.0, 1200)
    segment = SpectrumSegment(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.02),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium="air",
        name="uvb",
    )
    case = prepare_xshooter_balmer_case(
        segment,
        window_mode="notebook",
        norm_mode="sideband",
        core_mask=3.0,
    )

    fig, axes = case.plot_preparation(savepath=tmp_path / "nested" / "case.png")

    assert axes.shape == (2, 2)
    assert (tmp_path / "nested" / "case.png").exists()
    plt.close(fig)


def test_xshooter_balmer_case_suggest_fit_setup_records_regions_and_continuum():
    wave = np.linspace(3950.0, 5050.0, 1200)
    segment = SpectrumSegment(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.02),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium="air",
        name="uvb",
    )
    case = prepare_xshooter_balmer_case(
        segment,
        window_mode="notebook",
        norm_mode="sideband",
        core_mask=3.0,
    )

    setup = case.suggest_fit_setup(
        mode="standard",
        intent="reviewed_analysis",
        continuum_degree=1,
    )
    summary = setup.summary()

    assert summary["mode"] == "standard"
    assert summary["fit_regions_A"] == [
        [3980.0, 4220.0],
        [4220.0, 4480.0],
        [4700.0, 5020.0],
    ]
    assert summary["continuum_degree"] == 1


def test_fit_case_lines_individually_uses_public_fit_path(monkeypatch):
    wave = np.linspace(3950.0, 5050.0, 1200)
    segment = SpectrumSegment(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.02),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium="air",
        name="uvb",
    )
    case = prepare_xshooter_balmer_case(
        segment,
        window_mode="notebook",
        norm_mode="sideband",
        core_mask=3.0,
    )
    base_setup = case.suggest_fit_setup(
        mode="standard",
        intent="reviewed_analysis",
        continuum_degree=1,
    )
    calls = []

    def fake_fit_stellar_spectrum(
        spectrum,
        *,
        model,
        setup,
        valid_mask,
        phoenix_dir,
        progress_callback,
        **fit_kwargs,
    ):
        calls.append(
            {
                "n_segments": len(spectrum.segments),
                "model": model,
                "setup_regions": setup.summary()["fit_regions_A"],
                "valid_mask_lengths": [int(mask.size) for mask in valid_mask],
                "phoenix_dir": phoenix_dir,
                "progress_callback": progress_callback,
                "fit_kwargs": fit_kwargs,
            }
        )
        return {"label": spectrum.meta["line_label"]}

    monkeypatch.setattr(
        "Spyctres.api.fit_stellar_spectrum",
        fake_fit_stellar_spectrum,
    )

    out = fit_case_lines_individually(
        case,
        base_setup=base_setup,
        model="phoenix",
        phoenix_dir="/tmp/phoenix",
        progress_callback="progress",
        max_nfev=7,
    )

    assert list(out) == ["Hδ", "Hγ", "Hβ"]
    assert [item["n_segments"] for item in calls] == [1, 1, 1]
    assert [item["setup_regions"] for item in calls] == [
        [[3980.0, 4220.0]],
        [[4220.0, 4480.0]],
        [[4700.0, 5020.0]],
    ]
    assert all(len(item["valid_mask_lengths"]) == 1 for item in calls)
    assert all(item["model"] == "phoenix" for item in calls)
    assert all(item["phoenix_dir"] == "/tmp/phoenix" for item in calls)
    assert all(item["progress_callback"] == "progress" for item in calls)
    assert all(item["fit_kwargs"]["max_nfev"] == 7 for item in calls)
