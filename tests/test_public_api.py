import json

import numpy as np
import pytest

from Spyctres.api import fit_phoenix_spectrum
from Spyctres.io import SpectrumSegment
from Spyctres.results import PhoenixFitDiagnostics, PhoenixFitResult


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
    assert isinstance(result.diagnostics, PhoenixFitDiagnostics)
    assert result.quality_flags == ("ok",)
    payload = json.loads(result.to_json())
    assert payload["p_best"] == [5772.0, 0.0, 4.44, 0.0]
    assert payload["models"] == [[1.0, 0.9]]
    assert payload["diagnostics"]["reduced_chi2"] == 1.0
    assert payload["quality_flags"] == ["ok"]


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
