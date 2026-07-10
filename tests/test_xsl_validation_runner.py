import csv
import json
from types import SimpleNamespace

import numpy as np

from Spyctres.io import SpectrumCollection, SpectrumSegment
from scripts import xsl_validation


def _arguments(**overrides):
    values = {
        "coarse_init": None,
        "multistart": None,
        "max_nfev": None,
        "coarse_decimate": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_role_budgets_keep_stress_runs_bounded():
    standard_role, standard = xsl_validation._role_budget(
        {"validation_role": "standard"}, _arguments()
    )
    stress_role, stress = xsl_validation._role_budget(
        {"validation_role": "carbon_star"}, _arguments()
    )

    assert standard_role == "standard"
    assert stress_role == "carbon_star"
    assert standard["multistart"] > stress["multistart"]
    assert standard["max_nfev"] > stress["max_nfev"]
    assert stress["statistics_group"] == "diagnostic_only"


def test_manifest_validation_requires_unique_ids():
    with np.testing.assert_raises_regex(ValueError, "non-empty xsl_id"):
        xsl_validation._validate_manifest_rows([{"xsl_id": ""}])
    with np.testing.assert_raises_regex(ValueError, "Duplicate xsl_id"):
        xsl_validation._validate_manifest_rows(
            [{"xsl_id": "X0001"}, {"xsl_id": "x0001"}]
        )


def test_json_native_rejects_nonfinite_json_extensions():
    converted = xsl_validation._json_native(
        {"array": np.array([1.0, np.nan]), "scalar": np.float64(2.0)}
    )
    assert converted == {"array": [1.0, None], "scalar": 2.0}


def test_ordinary_statistics_exclude_stress_targets():
    results = [
        {
            "statistics_group": "ordinary_recovery",
            "status": "ok",
            "delta": {"teff": 100.0, "logg": 0.2, "feh": -0.1},
        },
        {
            "statistics_group": "diagnostic_only",
            "status": "ok",
            "delta": {"teff": 5000.0, "logg": 4.0, "feh": -3.0},
        },
    ]
    statistics = xsl_validation._ordinary_recovery_statistics(results)
    assert statistics["count"] == 1
    assert statistics["teff"]["median_delta"] == 100.0


def test_validation_plot_payload_is_bounded_and_uses_native_arrays():
    wave = np.linspace(4000.0, 5000.0, 11)
    segment = SpectrumSegment(
        wave,
        np.ones(11),
        err=np.ones(11),
        name="UVB",
        wave_frame="stellar_rest",
    )
    collection = SpectrumCollection([segment])
    fit_result = SimpleNamespace(
        models=(np.linspace(0.9, 1.1, 11),),
        used_masks=(np.array([True] * 10 + [False]),),
    )

    payload = xsl_validation._validation_plot_payload(collection, fit_result, 4)

    assert payload["max_points_per_segment"] == 4
    assert payload["display_defaults"]["scale_mode"] == "global"
    assert payload["display_defaults"]["arm_scaling_applied_by_spyctres"] is False
    assert payload["display_defaults"]["rv_correction_applied_by_spyctres"] is False
    saved = payload["segments"][0]
    assert saved["original_points"] == 11
    assert saved["saved_points"] == 4
    assert saved["wave_A"].shape == (4,)
    assert saved["stellar_rest_status"] == "corrected"
    native = xsl_validation._json_native(payload)
    assert isinstance(native["segments"][0]["wave_A"], list)


def test_runner_checkpoints_each_target_and_resume_skips_completed(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.csv"
    fieldnames = [
        "path",
        "xsl_id",
        "star_name",
        "spectral_type",
        "teff_ref",
        "logg_ref",
        "feh_ref",
        "validation_role",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "path": "one.fits",
                "xsl_id": "X0001",
                "star_name": "one",
                "spectral_type": "G",
                "teff_ref": "5000",
                "logg_ref": "4.0",
                "feh_ref": "0.0",
                "validation_role": "standard",
            }
        )
        writer.writerow(
            {
                "path": "two.fits",
                "xsl_id": "X0002",
                "star_name": "two",
                "spectral_type": "M",
                "teff_ref": "3500",
                "logg_ref": "4.5",
                "feh_ref": "-0.5",
                "validation_role": "cool_stress",
            }
        )

    wave = np.linspace(4000.0, 9000.0, 20)
    collection = SpectrumCollection(
        [
            SpectrumSegment(
                wave,
                np.ones(wave.size),
                err=np.ones(wave.size),
                wave_medium="air",
                wave_frame="stellar_rest",
            )
        ]
    )
    monkeypatch.setattr(xsl_validation, "read_spectrum", lambda *args, **kwargs: collection)

    fit_calls = []

    def fake_fit(*args, **kwargs):
        fit_calls.append(kwargs)
        return {
            "success": True,
            "teff": 5100.0,
            "feh": -0.1,
            "logg": 4.1,
            "rv_kms": 0.0,
            "chi2_red": 1.2,
            "physical_initialization": "coarse",
            "coarse_initialization": {
                "candidates_evaluated": 2,
                "candidates": [
                    {"chi2": np.float64(1.0)},
                    {"chi2": np.float64(2.0)},
                ],
                "candidates_complete": True,
                "top_candidates": [{"chi2": np.float64(1.0)}],
            },
            "multistart": kwargs["multistart"],
            "multistart_requested": kwargs["multistart"],
            "stellar_rest_zero_rv_start_tested": True,
            "multistart_diagnostics": [
                {"start": np.array([5000.0, 0.0, 4.0, 0.0])}
            ],
            "quality_report": {
                "quality_flags": ["ok"],
                "mask_fraction": np.float64(0.1),
            },
        }

    monkeypatch.setattr(xsl_validation, "fit_phoenix_spectrum", fake_fit)
    output = tmp_path / "results.json"
    writes = []
    original_atomic_write = xsl_validation._atomic_write_json

    def recording_write(path, payload):
        writes.append(len(payload["results"]))
        original_atomic_write(path, payload)

    monkeypatch.setattr(xsl_validation, "_atomic_write_json", recording_write)
    xsl_validation.main([str(manifest), "--output", str(output)])

    assert writes == [1, 2, 2]
    assert [call["multistart"] for call in fit_calls] == [4, 2]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_configuration"]["mdeg"] == 2
    assert payload["ordinary_recovery_statistics"]["count"] == 1
    assert payload["results"][1]["statistics_group"] == "diagnostic_only"
    assert payload["results"][0]["quality_report"]["mask_fraction"] == 0.1
    assert payload["results"][0]["initialization"]["local_solutions"][0][
        "start"
    ] == [5000.0, 0.0, 4.0, 0.0]
    coarse = payload["results"][0]["initialization"]["coarse"]
    assert len(coarse["candidates"]) == coarse["candidates_evaluated"] == 2
    assert coarse["candidates_complete"] is True

    fit_calls.clear()
    writes.clear()
    xsl_validation.main(
        [str(manifest), "--output", str(output), "--resume"]
    )
    assert fit_calls == []
    assert writes == [2]

    with np.testing.assert_raises_regex(ValueError, "different validation settings"):
        xsl_validation.main(
            [
                str(manifest),
                "--output",
                str(output),
                "--resume",
                "--mdeg",
                "3",
            ]
        )


def test_runner_records_unsupported_zero_budget_without_fit(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.csv"
    fieldnames = [
        "path",
        "xsl_id",
        "star_name",
        "spectral_type",
        "teff_ref",
        "logg_ref",
        "feh_ref",
        "validation_role",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "path": "one.fits",
                "xsl_id": "XHOT",
                "star_name": "hot",
                "spectral_type": "B",
                "teff_ref": "10000",
                "logg_ref": "4.0",
                "feh_ref": "0.0",
                "validation_role": "unsupported_hot",
            }
        )

    def fail_fit(*args, **kwargs):
        raise AssertionError("unsupported zero-budget target should not run a fit")

    monkeypatch.setattr(xsl_validation, "fit_phoenix_spectrum", fail_fit)
    output = tmp_path / "results.json"
    xsl_validation.main([str(manifest), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"][0]["status"] == "unsupported_physics"
