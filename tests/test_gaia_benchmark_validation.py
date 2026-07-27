import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from Spyctres.io import SpectrumSegment
from scripts import gaia_benchmark_validation as gbs


def _write_gbs_spectrum(path):
    wave_nm = np.linspace(480.0, 680.0, 401)
    flux = np.ones_like(wave_nm)
    flux -= 0.18 * np.exp(-0.5 * ((wave_nm - 486.13) / 0.45) ** 2)
    flux -= 0.08 * np.exp(-0.5 * ((wave_nm - 517.0) / 0.30) ** 2)
    err = np.full_like(wave_nm, 0.02)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("waveobs flux err\n")
        for wave, value, sigma in zip(wave_nm, flux, err):
            handle.write(f"{wave:.6f} {value:.8f} {sigma:.6f}\n")


def _write_manifest(path, spectrum_name):
    payload = {
        "schema_version": 1,
        "source_page": "https://example.invalid/gbs",
        "source_release": "synthetic-test",
        "citations": ["synthetic_test"],
        "spectra": [
            {
                "file": spectrum_name,
                "hip": "HIP79672",
                "hd": "HD146233",
                "name": "18 Sco synthetic",
                "spectral_type": "G Dwarf (V)",
                "source_instrument": "HARPS",
                "teff_ref": 5824.0,
                "logg_ref": 4.42,
                "feh_ref": 0.06,
                "snr": 1000.0,
                "source_quality_flag": "-",
                "validation_role": "standard",
                "notes": "Synthetic compact fixture for the validation runner.",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gaia_benchmark_validation_audit_cli_writes_outputs(tmp_path):
    spectrum = tmp_path / "HIP79672_HARPS_1_R42KNorm.txt.gz"
    manifest = tmp_path / "manifest.json"
    output_json = tmp_path / "nested" / "gbs_validation.json"
    output_csv = tmp_path / "nested" / "gbs_validation.csv"
    summary_plot = tmp_path / "nested" / "gbs_validation.png"
    _write_gbs_spectrum(spectrum)
    _write_manifest(manifest, spectrum.name)

    exit_code = gbs.main(
        [
            "--manifest",
            str(manifest),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-summary-plot",
            str(summary_plot),
            "--force",
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
    assert output_csv.exists()
    assert summary_plot.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    record = payload["results"][0]

    assert payload["runner"] == "scripts/gaia_benchmark_validation.py"
    assert payload["run_policy"]["run_fits"] is False
    assert payload["run_policy"]["wave_medium"] == "reader"
    assert payload["run_policy"]["window_set"] == "default"
    assert payload["run_policy"]["error_floor_fraction"] == pytest.approx(0.0)
    assert payload["run_policy"]["reference_parameters_used_as_priors"] is False
    assert record["status"] == "audit_only"
    assert record["target_id"] == "HIP79672"
    assert record["validation_role"] == "standard"
    assert record["reference"] == {"teff": 5824.0, "logg": 4.42, "feh": 0.06}
    assert record["fit_policy"]["bounds_policy"] == "benchmark_fgk"
    assert record["fit_policy"]["wave_medium_override"] == "reader"
    assert record["fit_policy"]["window_set"] == "default"
    assert record["fit_policy"]["error_floor_fraction"] == pytest.approx(0.0)
    assert record["fit_policy"]["reference_parameters_used_as_priors"] is False
    assert record["segment"]["wave_medium"] == "air"
    assert record["segment"]["wave_min_A"] == pytest.approx(4800.0)
    assert record["segment"]["wave_max_A"] == pytest.approx(6800.0)
    assert record["segment"]["resolution"]["quantity"] == "R"
    assert record["segment"]["resolution"]["value"] == pytest.approx(42000.0)
    assert record["setup"]["operation"] == "suggest_fit_setup"
    assert record["recovery"]["overall_assessment"] == "audit_only"
    assert record["recovery"]["fit_quality_assessment"] == "audit_only"
    assert payload["summary"]["by_status"] == {"audit_only": 1}
    assert payload["summary"]["ordinary_recovery_n"] == 0


def test_gaia_benchmark_fit_kwargs_use_reference_independent_search_box():
    args = SimpleNamespace(
        bounds_policy="benchmark_fgk",
        multistart=3,
        rv_grid_n=17,
        max_nfev=123,
        mdeg=2,
    )
    setup_payload = {
        "fit_kwargs": {
            "p0": (5824.0, 0.06, 4.42, 0.0),
            "bounds": ((5000.0, -0.5, 3.5, -50.0), (6500.0, 0.5, 5.0, 50.0)),
        }
    }

    fit_kwargs = gbs._benchmark_fit_kwargs(
        setup_payload,
        args,
        row={"validation_role": "standard"},
    )

    assert fit_kwargs["p0"] == (5500.0, 0.0, 4.0, 0.0)
    assert fit_kwargs["bounds"] == (
        (4000.0, -1.0, 1.0, -150.0),
        (7000.0, 0.5, 5.5, 150.0),
    )
    assert fit_kwargs["coarse_feh_grid"] == [-1.0, 0.0, 0.5]
    assert fit_kwargs["multistart"] == 3
    assert fit_kwargs["rv_grid_n"] == 17
    assert fit_kwargs["forward_model"] == "native_interp"
    assert fit_kwargs["error_floor_fraction"] == pytest.approx(0.0)


def test_gaia_benchmark_diagnostic_window_set_and_error_floor_are_explicit():
    args = SimpleNamespace(
        bounds_policy="benchmark_fgk",
        multistart=2,
        rv_grid_n=21,
        max_nfev=200,
        mdeg=3,
        window_set="broad_metal_forest",
        error_floor_fraction=0.01,
        run_fits=True,
        fit_mode="standard",
        wave_medium="reader",
    )

    fit_kwargs = gbs._benchmark_fit_kwargs(
        {"fit_kwargs": {"regions": [(4830.0, 4898.0)]}},
        args,
        row={"validation_role": "standard"},
    )
    policy = gbs._fit_policy_record(args, fit_kwargs)

    assert fit_kwargs["regions"] == [(5150.0, 5450.0), (6000.0, 6500.0)]
    assert fit_kwargs["mdeg"] == 3
    assert fit_kwargs["error_floor_fraction"] == pytest.approx(0.01)
    assert policy["window_set"] == "broad_metal_forest"
    assert policy["window_set_definition"]["diagnostic_only"] is True
    assert policy["window_set_definition"]["regions"] == [
        [5150.0, 5450.0],
        [6000.0, 6500.0],
    ]
    assert "Non-zero error floors are diagnostic checks" in policy[
        "error_floor_interpretation"
    ]


def test_gaia_benchmark_stress_fit_kwargs_keep_bounded_metal_poor_search():
    args = SimpleNamespace(
        bounds_policy="benchmark_fgk",
        multistart=2,
        rv_grid_n=21,
        max_nfev=200,
        mdeg=2,
    )

    fit_kwargs = gbs._benchmark_fit_kwargs(
        {"fit_kwargs": {}},
        args,
        row={"validation_role": "metal_poor_stress"},
    )

    assert fit_kwargs["p0"] == (5500.0, -1.0, 3.5, 0.0)
    assert fit_kwargs["bounds"][0][1] == pytest.approx(-2.5)
    assert fit_kwargs["coarse_feh_grid"] == [-2.0, -1.0, 0.0]
    assert fit_kwargs["coarse_teff_grid"] == [4500.0, 5500.0, 6500.0]


def test_gaia_benchmark_fit_plot_receives_observed_segment(tmp_path, monkeypatch):
    segment = SpectrumSegment(
        wave=np.linspace(4800.0, 4810.0, 5),
        flux=np.ones(5),
        err=np.full(5, 0.02),
        wave_medium="vacuum",
        name="synthetic",
    )
    called = {}

    class FakeResult:
        def to_dict(self, **kwargs):
            called["to_dict_kwargs"] = kwargs
            return {
                "success": True,
                "teff": 5800.0,
                "logg": 4.4,
                "feh": 0.0,
                "rv_kms": 0.0,
                "chi2_red": 1.0,
                "quality_flags": [],
                "generated_files": kwargs.get("plot_paths"),
            }

    def fake_fit_stellar_spectrum(*args, **kwargs):
        return FakeResult()

    def fake_plot_fit_referee(result, *, segment=None, savepath=None, **kwargs):
        called["segment"] = segment
        called["savepath"] = savepath
        fig, ax = gbs.plt.subplots()
        return fig, ax

    monkeypatch.setattr(gbs, "fit_stellar_spectrum", fake_fit_stellar_spectrum)
    monkeypatch.setattr(gbs, "plot_fit_referee", fake_plot_fit_referee)
    args = SimpleNamespace(fit_plot_dir=str(tmp_path), verbose=0)

    record = gbs._fit_record(
        segment,
        {"hip": "HIP79672"},
        {},
        {},
        args,
        phoenix_lib=object(),
    )

    assert record["fit"]["success"] is True
    assert called["segment"] is segment
    assert called["savepath"].endswith("HIP79672_fit.png")
    assert called["to_dict_kwargs"]["include_local_paths"] is True


def test_gaia_benchmark_line_plot_reconstructs_and_writes_variant_plot(
    tmp_path, monkeypatch
):
    wave = np.linspace(4825.0, 4905.0, 81)
    flux = 1.0 - 0.20 * np.exp(-0.5 * ((wave - 4861.33) / 6.0) ** 2)
    model = 1.0 - 0.18 * np.exp(-0.5 * ((wave - 4861.33) / 6.5) ** 2)
    segment = SpectrumSegment(
        wave=wave,
        flux=flux,
        err=np.full_like(wave, 0.02),
        wave_medium="air",
        name="synthetic",
    )
    used = (wave >= 4830.0) & (wave <= 4898.0)
    used &= np.abs(wave - 4861.33) > 1.5
    called = {}

    class FakeResult:
        models = (model,)
        used_masks = (used,)
        excluded_masks = (~used,)

        def to_dict(self, **kwargs):
            return {
                "success": True,
                "teff": 5800.0,
                "logg": 4.4,
                "feh": 0.0,
                "rv_kms": 0.0,
                "rv_bary_kms": 0.0,
                "chi2_red": 1.0,
                "quality_flags": [],
                "generated_files": {"plots": kwargs.get("plot_paths") or {}},
            }

    def fake_fit_stellar_spectrum(*args, **kwargs):
        called["reconstruct"] = kwargs.get("reconstruct")
        return FakeResult()

    monkeypatch.setattr(gbs, "fit_stellar_spectrum", fake_fit_stellar_spectrum)
    args = SimpleNamespace(
        fit_plot_dir=None,
        line_plot_dir=str(tmp_path),
        line_plot_reference_model=False,
        window_set="default",
        error_floor_fraction=0.0,
        verbose=0,
    )

    record = gbs._fit_record(
        segment,
        {"hip": "HIP79672", "name": "18 Sco synthetic"},
        {},
        {"regions": [(4830.0, 4898.0)], "mdeg": 2},
        args,
        phoenix_lib=object(),
    )

    assert called["reconstruct"] is True
    line_plot = record["generated_files"]["plots"]["line_windows"]
    assert line_plot.endswith("HIP79672_default_line_windows.png")
    assert (tmp_path / "HIP79672_default_line_windows.png").exists()
    assert record["reference_model_diagnostic"] is None


def test_gaia_benchmark_reference_overlay_uses_manifest_params_and_fit_rv(
    monkeypatch,
):
    wave = np.linspace(4830.0, 4898.0, 9)
    segment = SpectrumSegment(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.02),
        wave_medium="air",
        name="synthetic",
    )
    used = np.ones_like(wave, dtype=bool)
    called = {}

    def fake_reconstruct(segments, phoenix_lib, fit_result, **kwargs):
        called["fit_result"] = dict(fit_result)
        called["regions"] = kwargs.get("regions")
        return (
            [np.full_like(wave, 0.99)],
            [np.array([1.0, 0.0, 0.0])],
            [used],
            [np.zeros_like(used, dtype=bool)],
        )

    monkeypatch.setattr(
        gbs,
        "reconstruct_phoenix_legendre_models_for_segments",
        fake_reconstruct,
    )

    overlay = gbs._reference_model_overlay(
        segment,
        {"teff_ref": 5824.0, "logg_ref": 4.42, "feh_ref": 0.06},
        {
            "teff": 5900.0,
            "logg": 4.3,
            "feh": 0.1,
            "rv_kms": 1.25,
            "rv_bary_kms": 0.5,
            "forward_model": "native_interp",
            "model_margin_A": 200.0,
        },
        {"regions": [(4830.0, 4898.0)], "mdeg": 2},
        phoenix_lib=object(),
    )

    assert overlay["status"] == "ok"
    assert overlay["summary"]["params"]["teff"] == pytest.approx(5824.0)
    assert overlay["summary"]["params"]["logg"] == pytest.approx(4.42)
    assert overlay["summary"]["params"]["feh"] == pytest.approx(0.06)
    assert overlay["summary"]["params"]["rv_kms"] == pytest.approx(1.25)
    assert overlay["summary"]["rv_source"] == "best_fit_rv"
    assert called["fit_result"]["teff"] == pytest.approx(5824.0)
    assert called["fit_result"]["rv_kms"] == pytest.approx(1.25)
    assert called["regions"] == [(4830.0, 4898.0)]


def test_gaia_benchmark_recovery_summary_excludes_stress_roles():
    records = [
        {
            "status": "ok",
            "validation_role": "standard",
            "reference": {"teff": 5800.0, "logg": 4.4, "feh": 0.0},
            "fit": {"teff": 5900.0, "logg": 4.3, "feh": 0.1},
        },
        {
            "status": "ok",
            "validation_role": "metal_poor_stress",
            "reference": {"teff": 5800.0, "logg": 3.8, "feh": -2.4},
            "fit": {"teff": 7000.0, "logg": 5.5, "feh": 0.0},
        },
    ]

    summary = gbs.summarize_payload(records)

    assert summary["ordinary_recovery_n"] == 1
    assert summary["ordinary_recovery_assessments"] == {
        "within_first_pass_tolerance": 1
    }
    assert summary["ordinary_fit_quality_assessments"] == {
        "missing_formal_fit_quality": 1
    }
    assert summary["by_validation_role"] == {
        "metal_poor_stress": 1,
        "standard": 1,
    }
