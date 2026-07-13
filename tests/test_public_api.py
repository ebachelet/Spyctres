import json

import numpy as np
import pytest

from Spyctres.api import classify_spectrum, fit_phoenix_spectrum, fit_stellar_spectrum
from Spyctres.io import SpectrumSegment
from Spyctres.results import (
    PhoenixFitDiagnostics,
    PhoenixFitResult,
    compare_fit_results,
    describe_quality_flags,
    format_fit_quality_report,
)


def test_structured_result_is_mapping_and_json_serializable():
    result = PhoenixFitResult(
        summary={
            "teff": 5772.0,
            "p_best": np.array([5772.0, 0.0, 4.44, 0.0]),
            "diagnostics": {"reduced_chi2": np.float64(1.0)},
            "quality_flags": ["ok"],
        },
        models=(np.array([1.0, 0.9]),),
        used_masks=(np.array([True, False]),),
        provenance={"api": "test"},
    )

    assert result["teff"] == 5772.0
    assert "diagnostics" in list(result)
    assert "quality_flags" in list(result)
    assert "provenance" in list(result)
    assert isinstance(result.diagnostics, PhoenixFitDiagnostics)
    assert result.quality_flags == ("ok",)
    payload = json.loads(result.to_json())
    assert payload["p_best"] == [5772.0, 0.0, 4.44, 0.0]
    assert payload["models"] == [[1.0, 0.9]]
    assert payload["diagnostics"]["reduced_chi2"] == 1.0
    assert payload["quality_flags"] == ["ok"]
    assert payload["quality_report"]["quality_flags"] == ["ok"]
    assert payload["quality_report"]["reduced_chi2"] == 1.0


def test_structured_result_quality_report_summarizes_fit_selection():
    result = PhoenixFitResult(
        summary={
            "success": True,
            "chi2_red": np.float64(2.0),
            "n_points": np.int64(42),
            "quality_flags": ["segment_mask_fraction_high"],
        },
        diagnostics={
            "n_parameters": np.int64(6),
            "degrees_of_freedom": np.int64(36),
            "mask_fraction": np.float64(0.25),
            "n_input_segments": 2,
            "n_retained_segments": 1,
            "n_dropped_segments": 1,
            "segment_diagnostics": [
                {
                    "name": "VIS",
                    "input_index": 1,
                    "n_fit": np.int64(42),
                    "n_support": np.int64(80),
                    "mask_fraction": np.float64(0.475),
                    "mask_summary": {
                        "n_rejected_by_explicit_union": np.int64(10),
                        "explicit_exclusion_fraction": np.float64(0.125),
                        "n_rejected_by_data_invalid": np.int64(4),
                        "data_invalid_fraction": np.float64(0.05),
                        "n_rejected_by_multiple_reasons": np.int64(2),
                        "multiple_rejection_fraction": np.float64(0.025),
                    },
                    "lsf_fwhm_kms": np.float64(5.0),
                    "resolution_R_effective": np.float64(59958.0),
                }
            ],
        },
    )

    report = result.quality_report()

    assert report["success"] is True
    assert report["quality_flags"] == ["segment_mask_fraction_high"]
    assert (
        "more than half"
        in report["quality_flag_descriptions"]["segment_mask_fraction_high"]
    )
    assert report["reduced_chi2"] == 2.0
    assert report["mask_fraction"] == 0.25
    assert report["n_dropped_segments"] == 1
    assert report["segments"][0]["name"] == "VIS"
    assert report["segments"][0]["explicit_exclusion_count"] == 10
    assert report["segments"][0]["explicit_exclusion_fraction"] == 0.125
    assert report["segments"][0]["data_invalid_count"] == 4
    assert report["segments"][0]["data_invalid_fraction"] == 0.05
    assert report["segments"][0]["multiple_rejection_count"] == 2
    assert report["segments"][0]["multiple_rejection_fraction"] == 0.025

    text = result.quality_report_text()
    assert "Quality report:" in text
    assert "flags: segment_mask_fraction_high" in text
    assert "chi2_red: 2" in text
    assert "masked fraction: 25.0%" in text
    assert "dropped segments: 1" in text
    assert "VIS; Nfit=42/80" in text

    text_from_dict = format_fit_quality_report(result.to_dict(include_arrays=False))
    assert "Quality report:" in text_from_dict
    assert "VIS; Nfit=42/80" in text_from_dict


def test_quality_report_includes_known_feature_and_residual_windows():
    text = format_fit_quality_report(
        {
            "success": True,
            "quality_flags": [
                "nonstellar_feature_overlap",
                "known_line_region_residual",
                "dib_overlap_balmer_wing",
            ],
            "chi2_red": 4.2,
            "nonstellar_features": {
                "show_dibs": True,
                "mask_dibs": False,
                "policy": "warn",
                "features": [{"name": "DIB 4428"}, {"name": "DIB 4882"}],
                "overlap_diagnostics": [
                    {
                        "feature": "DIB 4882",
                        "diagnostic_line": "Hbeta",
                        "origin_hypothesis": "catalog_overlap_only",
                    }
                ],
            },
            "known_residual_windows": {
                "flagged_windows": [
                    {
                        "name": "DIB 4882 / Hβ red wing",
                        "median_sigma": -3.1,
                        "rms_sigma": 4.2,
                        "origin_hypothesis": "ambiguous",
                    }
                ]
            },
        }
    )

    assert "non-stellar features: DIB 4428, DIB 4882" in text
    assert "policy=warn" in text
    assert (
        "contaminated diagnostics: DIB 4882 -> Hbeta (catalog_overlap_only)"
        in text
    )
    assert "DIB 4882 / Hβ red wing median=-3.1σ rms=4.2σ origin=ambiguous" in text


def test_compare_fit_results_reports_parameter_and_quality_deltas():
    reference = PhoenixFitResult(
        summary={
            "success": True,
            "teff": np.float64(9800.0),
            "feh": np.float64(-0.1),
            "logg": np.float64(2.7),
            "rv_kms": np.float64(-5.0),
            "chi2_red": np.float64(12.0),
            "quality_flags": ["high_chi2", "dib_candidate_detected"],
            "nonstellar_features": {
                "features": [{"name": "DIB 4882"}],
            },
            "known_residual_windows": {
                "flagged_windows": [{"name": "DIB 4882 / Hβ red wing"}],
            },
        },
        diagnostics={"mask_fraction": np.float64(0.45), "n_pixels": np.int64(100)},
    )
    masked = PhoenixFitResult(
        summary={
            "success": True,
            "teff": np.float64(9600.0),
            "feh": np.float64(-0.05),
            "logg": np.float64(2.9),
            "rv_kms": np.float64(-4.0),
            "chi2_red": np.float64(9.0),
            "quality_flags": ["high_chi2", "nonstellar_mask_applied"],
            "nonstellar_features": {
                "features": [{"name": "DIB 4882"}, {"name": "DIB 4428"}],
            },
            "known_residual_windows": {"flagged_windows": []},
        },
        diagnostics={"mask_fraction": np.float64(0.5), "n_pixels": np.int64(90)},
    )

    comparison = compare_fit_results(
        reference,
        masked,
        labels=("unmasked", "masked"),
        thresholds={"teff": 100.0, "chi2_red": 1.0},
    )

    assert comparison["labels"] == ["unmasked", "masked"]
    assert comparison["parameters"]["teff"]["delta"] == -200.0
    assert comparison["parameters"]["teff"]["exceeds_threshold"] is True
    assert comparison["metrics"]["chi2_red"]["delta"] == -3.0
    assert comparison["metrics"]["mask_fraction"]["delta"] == pytest.approx(0.05)
    assert comparison["metrics"]["n_points"]["reference"] == 100.0
    assert comparison["metrics"]["n_points"]["comparison"] == 90.0
    assert comparison["quality_flags"]["changed"] is True
    assert comparison["quality_flags"]["only_unmasked"] == ["dib_candidate_detected"]
    assert comparison["quality_flags"]["only_masked"] == ["nonstellar_mask_applied"]
    assert comparison["known_features"]["only_masked"] == ["DIB 4428"]
    assert comparison["known_residual_windows"]["only_unmasked"] == [
        "DIB 4882 / Hβ red wing"
    ]
    json.dumps(comparison)


def test_compare_fit_results_is_top_level_public_api():
    import Spyctres

    reference = {"teff": 6000.0, "quality_flags": ["ok"]}
    comparison = {"teff": 6100.0, "quality_flags": ["high_chi2"]}

    out = Spyctres.compare_fit_results(reference, comparison)

    assert out["parameters"]["teff"]["delta"] == 100.0
    assert out["quality_flags"]["only_reference"] == ["ok"]
    assert out["quality_flags"]["only_comparison"] == ["high_chi2"]


def test_quality_flag_descriptions_cover_static_and_grid_flags():
    descriptions = describe_quality_flags(
        ["metadata_incomplete", "grid_edge_teff_high", "surprise_flag"]
    )

    assert "metadata" in descriptions["metadata_incomplete"]
    assert "high edge" in descriptions["grid_edge_teff_high"]
    assert "No description" in descriptions["surprise_flag"]


def test_wavelength_medium_helpers_are_top_level_public_api():
    import Spyctres

    wave_air = np.array([5000.0])
    converted = Spyctres.convert_wavelength_medium(
        wave_air,
        from_medium="air",
        to_medium="vacuum",
        method="vald3",
    )

    assert converted[0] > wave_air[0]
    assert callable(Spyctres.convert_segment_wavelength_medium)


def test_exclusion_mask_helper_is_top_level_public_api():
    import Spyctres

    spec = Spyctres.exclusion_mask("demo", lambda wave: wave == wave)

    assert isinstance(spec, Spyctres.ExclusionMaskSpec)
    assert spec.name == "demo"


def test_nonstellar_feature_helpers_are_top_level_public_api():
    import Spyctres

    regions = Spyctres.nonstellar_feature_regions("dib_4428")
    dib_4882 = Spyctres.nonstellar_feature_regions("dib_4882")
    masks = Spyctres.known_feature_masks(["dib_4428", "dib_4882"])

    assert "dib_4428" in Spyctres.NONSTELLAR_FEATURES
    assert "dib_4882" in Spyctres.NONSTELLAR_FEATURES
    assert "telluric_o2_a_7605" in Spyctres.OPTICAL_TELLURIC_DIAGNOSTIC_FEATURES
    assert "dib_4882" in Spyctres.OPTICAL_DIB_DIAGNOSTIC_FEATURES
    assert regions == [(4416.8, 4440.8)]
    assert dib_4882 == [(4870.0, 4915.0)]
    assert [mask.name for mask in masks] == [
        "nonstellar:dib_4428",
        "nonstellar:dib_4882",
    ]


def test_structured_result_can_save_compact_json(tmp_path):
    result = PhoenixFitResult(
        summary={"teff": np.float64(5772.0)},
        models=(np.array([np.nan]),),
        diagnostics={"residual_rms": np.float64(np.nan)},
        quality_flags=("metadata_incomplete",),
    )
    path = tmp_path / "result.json"

    result.save_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "models" not in payload
    assert payload["diagnostics"]["residual_rms"] is None
    assert payload["quality_flags"] == ["metadata_incomplete"]


def test_compact_json_sanitizes_local_paths_and_records_relative_plot(tmp_path):
    result = PhoenixFitResult(
        summary={"teff": 5772.0},
        provenance={
            "phoenix_source_root": "/home/someone/PHOENIX",
            "cache_path": "/tmp/spyctres_cache.npz",
            "reader_path": "examples/data/spectrum.fits",
        },
    )
    json_path = tmp_path / "products" / "result.json"
    json_path.parent.mkdir()
    plot_path = json_path.parent / "fit.png"

    result.save_json(json_path, plot_paths={"referee_plot": plot_path})

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = json.dumps(payload)
    assert "/home/" not in text
    assert "/tmp/" not in text
    assert payload["provenance"]["phoenix_source_root"] is None
    assert payload["provenance"]["cache_path"] is None
    assert payload["provenance"]["reader_path"] == "examples/data/spectrum.fits"
    assert payload["generated_files"]["plots"]["referee_plot"] == "fit.png"


def test_to_dict_rejects_absolute_plot_paths_without_relative_base():
    result = PhoenixFitResult(summary={"teff": 5772.0})
    with pytest.raises(ValueError, match="Absolute/local paths"):
        result.to_dict(plot_paths={"referee_plot": "/tmp/fit.png"})


def test_compact_json_rejects_plot_paths_outside_product_directory(tmp_path):
    result = PhoenixFitResult(summary={"teff": 5772.0})
    product_dir = tmp_path / "products"
    product_dir.mkdir()
    with pytest.raises(ValueError, match="inside the JSON product directory"):
        result.to_dict(
            include_arrays=False,
            plot_paths={"referee_plot": tmp_path / "fit.png"},
            relative_to=product_dir,
        )
    with pytest.raises(ValueError, match="must not traverse"):
        result.to_dict(
            include_arrays=False,
            plot_paths={"referee_plot": "../fit.png"},
            relative_to=product_dir,
        )


def test_local_paths_can_be_included_explicitly():
    result = PhoenixFitResult(
        summary={"teff": 5772.0},
        provenance={"phoenix_source_root": "/home/someone/PHOENIX"},
    )
    payload = result.to_dict(include_local_paths=True)
    assert payload["provenance"]["phoenix_source_root"] == "/home/someone/PHOENIX"


def test_public_api_canonicalizes_and_reconstructs(monkeypatch):
    captured = {}

    def fake_fit(spectrum, phoenix_lib, **kwargs):
        captured["spectrum"] = spectrum
        return {
            "success": True,
            "teff": 5000.0,
            "feh": 0.0,
            "logg": 4.5,
            "rv_kms": 10.0,
            "forward_model": "native_interp",
            "model_margin_A": 20.0,
        }

    def fake_reconstruct(spectrum, phoenix_lib, fit_result, **kwargs):
        captured["reconstruction_kwargs"] = kwargs
        return [np.ones(2)], [np.array([1.0])], [np.ones(2, bool)], [np.zeros(2, bool)]

    monkeypatch.setattr("Spyctres.api.fit_phoenix_full_spectrum", fake_fit)
    monkeypatch.setattr(
        "Spyctres.api.reconstruct_phoenix_legendre_models_for_segments",
        fake_reconstruct,
    )
    segment = SpectrumSegment(
        [5001.0, 5000.0],
        [1.0, 1.0],
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
    )

    result = fit_phoenix_spectrum(
        segment,
        phoenix_lib=object(),
        regions=[(5000.0, 5000.5)],
        exclude_regions=[(5000.2, 5000.3)],
    )

    assert isinstance(result, PhoenixFitResult)
    assert np.array_equal(captured["spectrum"].wave, [5000.0, 5001.0])
    assert len(result.models) == 1
    assert captured["reconstruction_kwargs"]["regions"] == [(5000.0, 5000.5)]
    assert captured["reconstruction_kwargs"]["exclude_regions"] == [
        (5000.2, 5000.3)
    ]


def test_public_api_rejects_two_library_sources():
    segment = SpectrumSegment([5000.0], [1.0])
    with pytest.raises(ValueError, match="not both"):
        fit_phoenix_spectrum(
            segment,
            phoenix_lib=object(),
            phoenix_dir="unused",
            reconstruct=False,
            warn_unknown=False,
        )


def test_public_api_skips_reconstruction_after_failed_fit(monkeypatch):
    monkeypatch.setattr(
        "Spyctres.api.fit_phoenix_full_spectrum",
        lambda *args, **kwargs: {"success": False, "teff": 5000.0},
    )
    segment = SpectrumSegment([5000.0], [1.0])
    result = fit_phoenix_spectrum(
        segment, phoenix_lib=object(), warn_unknown=False
    )

    assert result.models == ()
    assert result.provenance["reconstruction_performed"] is False


def test_fit_stellar_spectrum_reads_path_and_applies_defaults(monkeypatch):
    captured = {}
    segment = SpectrumSegment(
        np.linspace(3900.0, 5300.0, 20),
        np.ones(20),
        err=np.full(20, 0.1),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        meta={"instrument": "XSHOOTER", "arm": "UVB"},
        resolution=5000.0,
    )

    def fake_read(path, instrument, warn_unknown=True, **kwargs):
        captured["read"] = {
            "path": path,
            "instrument": instrument,
            "warn_unknown": warn_unknown,
            "kwargs": kwargs,
        }
        return segment

    def fake_fit(spectrum, **kwargs):
        captured["fit_spectrum"] = spectrum
        captured["fit_kwargs"] = kwargs
        return PhoenixFitResult(
            summary={
                "success": True,
                "teff": 6000.0,
                "feh": 0.0,
                "logg": 4.0,
                "rv_kms": 0.0,
                "chi2_red": 1.0,
            },
            provenance={"api": "fit_phoenix_spectrum"},
        )

    monkeypatch.setattr("Spyctres.api.read_spectrum", fake_read)
    monkeypatch.setattr("Spyctres.api.fit_phoenix_spectrum", fake_fit)

    result = fit_stellar_spectrum(
        "example.fits",
        instrument="xshooter",
        phoenix_lib=object(),
        reader_kwargs={"product_profile": "demo"},
        regions=[(4000.0, 5000.0)],
        rv_grid_n=11,
    )

    assert captured["read"]["instrument"] == "xshooter"
    assert captured["read"]["kwargs"] == {"product_profile": "demo"}
    assert captured["fit_spectrum"] is segment
    assert captured["fit_kwargs"]["regions"] == [(4000.0, 5000.0)]
    assert captured["fit_kwargs"]["rv_grid_n"] == 11
    assert captured["fit_kwargs"]["forward_model"] == "native_interp"
    assert "fit_default_suggestion" in result.summary
    assert result.provenance["workflow_api"] == "fit_stellar_spectrum"
    assert result.provenance["input_was_path"] is True
    assert result.provenance["instrument"] == "xshooter"


def test_classify_spectrum_alias_and_manual_defaults(monkeypatch):
    captured = {}
    segment = SpectrumSegment([5000.0, 5010.0], [1.0, 1.0])

    def fake_fit(spectrum, **kwargs):
        captured["kwargs"] = kwargs
        return PhoenixFitResult(
            summary={"success": True, "teff": 5750.0},
            provenance={"api": "fit_phoenix_spectrum"},
        )

    monkeypatch.setattr("Spyctres.api.fit_phoenix_spectrum", fake_fit)

    result = classify_spectrum(
        segment,
        phoenix_lib=object(),
        auto_defaults=False,
        p0=(5100.0, -0.2, 4.0, 3.0),
        reconstruct=False,
        warn_unknown=False,
    )

    assert captured["kwargs"]["p0"] == pytest.approx((5100.0, -0.2, 4.0, 3.0))
    assert captured["kwargs"]["reconstruct"] is False
    assert "fit_default_suggestion" not in result.summary
    assert result.provenance["workflow_api"] == "fit_stellar_spectrum"
    assert result.provenance["auto_defaults"] is False


def test_fit_stellar_spectrum_rejects_missing_instrument_for_paths():
    with pytest.raises(ValueError, match="Pass instrument"):
        fit_stellar_spectrum("example.fits", phoenix_lib=object())


def test_fit_stellar_spectrum_rejects_non_phoenix_model():
    segment = SpectrumSegment([5000.0], [1.0])
    with pytest.raises(ValueError, match="model='phoenix'"):
        fit_stellar_spectrum(segment, model="kurucz", phoenix_lib=object())
