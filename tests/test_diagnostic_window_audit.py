import csv
import json

import numpy as np
import pytest

from scripts import diagnostic_window_audit


def _write_ascii_spectrum(path, wmin=3800.0, wmax=5220.0, n=900):
    wave = np.linspace(wmin, wmax, n)
    flux = np.ones_like(wave)
    flux -= 0.15 * np.exp(-0.5 * ((wave - 4861.3) / 7.0) ** 2)
    flux -= 0.10 * np.exp(-0.5 * ((wave - 4300.0) / 12.0) ** 2)
    path.write_text(
        "\n".join("{0:.4f} {1:.8f}".format(w, f) for w, f in zip(wave, flux)),
        encoding="utf-8",
    )


def _write_manifest(path, rows):
    fieldnames = [
        "path",
        "target_id",
        "instrument",
        "spectral_type",
        "teff_ref",
        "validation_role",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_diagnostic_window_audit_cli_writes_json_and_csv(tmp_path):
    spectrum = tmp_path / "synthetic_fgk.dat"
    manifest = tmp_path / "manifest.csv"
    output_json = tmp_path / "audit" / "windows.json"
    output_csv = tmp_path / "audit" / "windows.csv"
    output_plot = tmp_path / "audit" / "windows.png"
    _write_ascii_spectrum(spectrum)
    _write_manifest(
        manifest,
        [
            {
                "path": spectrum.name,
                "target_id": "synthetic_g",
                "instrument": "uves_pop",
                "spectral_type": "G2V",
                "teff_ref": "5500",
                "validation_role": "standard",
            }
        ],
    )

    status = diagnostic_window_audit.main(
        [
            str(manifest),
            "--instrument",
            "uves_pop",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-plot",
            str(output_plot),
            "--fail-on-standard-missing",
        ]
    )

    assert status == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["operation"] == "diagnostic_window_manifest_audit"
    assert payload["summary"]["n_targets"] == 1
    assert payload["summary"]["n_ordinary_missing_expected_groups"] == 0
    assert "h_beta" in payload["summary"]["top_window_frequency_ordinary"]
    target = payload["targets"][0]
    assert target["target_id"] == "synthetic_g"
    assert "ch_g_band" in target["selection_summary"]["selected_window_ids"]
    assert "h_beta" in target["selection_summary"]["selected_window_ids"]
    assert output_csv.exists()
    assert output_plot.exists()


def test_stress_missing_expectations_do_not_count_as_ordinary_failures():
    records = [
        {
            "target_id": "ordinary",
            "validation_role": "standard",
            "ordinary_recovery_target": True,
            "spectral_type": "G",
            "teff_ref": 5500.0,
            "selection_summary": {
                "selected_window_ids": ["h_beta"],
                "top_window_ids": ["h_beta"],
                "selected_feature_families": ["hydrogen"],
            },
            "selection": {
                "selected": [{"id": "h_beta", "feature_family": ["hydrogen"]}]
            },
            "expectation_checks": [
                {
                    "group_id": "ch_g_band",
                    "status": "missing_expected_window",
                    "any_of": ["ch_g_band"],
                }
            ],
        },
        {
            "target_id": "stress",
            "validation_role": "cool_stress",
            "ordinary_recovery_target": False,
            "spectral_type": "M",
            "teff_ref": 3400.0,
            "selection_summary": {
                "selected_window_ids": ["h_beta"],
                "top_window_ids": ["h_beta"],
                "selected_feature_families": ["hydrogen"],
            },
            "selection": {
                "selected": [{"id": "h_beta", "feature_family": ["hydrogen"]}]
            },
            "expectation_checks": [
                {
                    "group_id": "cool_red_molecular",
                    "status": "missing_expected_window",
                    "any_of": ["tio_7050"],
                }
            ],
        },
    ]

    summary = diagnostic_window_audit.summarize_audit_records(records)

    assert summary["n_ordinary_recovery_targets"] == 1
    assert summary["n_ordinary_missing_expected_groups"] == 1
    assert summary["ordinary_missing_expected_groups"][0]["target_id"] == "ordinary"
    assert summary["top_window_frequency_ordinary"] == {"h_beta": 1}
    assert summary["top_window_frequency_all"] == {"h_beta": 2}


def test_expected_group_checks_are_coverage_and_teff_aware():
    checks = diagnostic_window_audit.check_expected_groups(
        ["h_beta", "ch_g_band"],
        teff_ref=5500.0,
        validation_role="standard",
        coverage_spans=[[3800.0, 5220.0]],
    )
    by_id = {item["group_id"]: item for item in checks}

    assert by_id["balmer_hydrogen"]["status"] == "ok"
    assert by_id["ch_g_band"]["status"] == "ok"
    assert by_id["kband_late_type"]["status"] == "not_applicable_no_coverage"
    assert by_id["cool_red_molecular"]["status"] == "not_applicable_teff"


def test_plot_audit_heatmap_writes_file_and_rejects_empty_payload(tmp_path):
    payload = {
        "targets": [
            {
                "target_id": "hot",
                "spectral_type": "A0",
                "validation_role": "standard",
                "selection_summary": {"top_window_ids": ["h_beta", "h_gamma"]},
            },
            {
                "target_id": "cool",
                "spectral_type": "M2",
                "validation_role": "cool_stress",
                "selection_summary": {"top_window_ids": ["tio_7050", "h_beta"]},
            },
        ]
    }
    plot_path = tmp_path / "plots" / "audit.png"

    fig, _ax = diagnostic_window_audit.plot_audit_heatmap(
        payload,
        savepath=plot_path,
    )

    import matplotlib.pyplot as plt

    plt.close(fig)
    assert plot_path.exists()

    with pytest.raises(ValueError, match="No diagnostic-window audit targets"):
        diagnostic_window_audit.plot_audit_heatmap({"targets": []})
