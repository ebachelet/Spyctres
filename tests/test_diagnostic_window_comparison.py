import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from Spyctres.diagnostic_window_comparison import (
    build_diagnostic_window_comparison_plan,
    plot_diagnostic_window_comparison,
    run_diagnostic_window_comparison,
    write_diagnostic_window_comparison_csv,
    write_diagnostic_window_comparison_json,
)
from Spyctres.io import SpectrumSegment


def _segment(wave_medium="air"):
    wave = np.linspace(3800.0, 5220.0, 1800)
    flux = np.ones_like(wave)
    for center, depth, sigma in (
        (4101.7, 0.20, 8.0),
        (4340.5, 0.24, 9.0),
        (4861.3, 0.28, 11.0),
        (5175.0, 0.08, 3.0),
    ):
        flux -= depth * np.exp(-0.5 * ((wave - center) / sigma) ** 2)
    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=np.full_like(wave, 0.03),
        mask=np.ones_like(wave, dtype=bool),
        wave_medium=wave_medium,
        observer_frame="barycentric",
        stellar_rest_status="observed",
        name="synthetic-blue",
        resolution=6000.0,
    )


def test_comparison_plan_is_bounded_and_skips_stress_windows_by_default():
    plan = build_diagnostic_window_comparison_plan(
        _segment(),
        max_windows=6,
        max_single_windows=4,
        max_comparisons=5,
    )

    assert plan["status"] == "planned"
    assert len(plan["planned_comparisons"]) <= 5
    assert plan["comparison_policy"]["bounded_search"] is True
    assert plan["comparison_policy"]["not_raw_chi2_ranked"] is True
    assert not any(
        "he_i_4471" in item["window_ids"] for item in plan["planned_comparisons"]
    )
    assert any(
        "stress_windows_disabled" in item.get("skip_reasons", ())
        for item in plan["skipped_comparisons"]
    )


def test_comparison_plan_records_operational_regions_and_held_out_windows():
    plan = build_diagnostic_window_comparison_plan(
        _segment(wave_medium="air"),
        rv_kms=120.0,
        rv_padding_kms=50.0,
        max_comparisons=3,
    )

    comparison = plan["planned_comparisons"][0]
    assert comparison["regions_for_fit_A"]
    assert comparison["canonical_regions_vacuum_A"]
    assert comparison["coordinate_policy"].startswith("regions_for_fit_A")
    assert "held_out_window_ids" in comparison
    assert comparison["held_out_evaluation"]["status"] == "not_computed"


def test_run_comparison_dry_run_has_no_fit_results():
    payload = run_diagnostic_window_comparison(_segment(), max_comparisons=3)

    assert payload["status"] == "planned_no_fits_run"
    assert len(payload["fit_records"]) == len(payload["planned_comparisons"])
    assert all(item["fit_status"] == "planned_not_run" for item in payload["fit_records"])
    assert all(item["result"] is None for item in payload["fit_records"])


def test_run_comparison_executes_fake_fit_with_regions():
    calls = []

    def fake_fit(spectrum, **kwargs):
        regions = kwargs["regions"]
        calls.append(regions)
        return {
            "success": True,
            "teff": 7000.0 + 10.0 * len(regions),
            "logg": 4.0,
            "feh": -0.1,
            "rv_kms": 2.0,
            "chi2_red": 3.0 / max(len(regions), 1),
            "quality_flags": ["synthetic"],
        }

    payload = run_diagnostic_window_comparison(
        _segment(),
        run_fits=True,
        fit_callable=fake_fit,
        base_fit_kwargs={"mdeg": 2},
        max_comparisons=2,
    )

    assert payload["status"] == "fits_completed"
    assert len(calls) == 2
    assert all(call for call in calls)
    assert all(item["fit_status"] == "ok" for item in payload["fit_records"])
    assert payload["fit_records"][0]["result_summary"]["success"] is True
    assert payload["fit_records"][0]["quality_flags"] == ["synthetic"]


def test_comparison_outputs_are_json_csv_and_plot_friendly(tmp_path):
    payload = run_diagnostic_window_comparison(_segment(), max_comparisons=3)
    json_path = tmp_path / "nested" / "comparison.json"
    csv_path = tmp_path / "nested" / "comparison.csv"
    plot_path = tmp_path / "nested" / "comparison.png"

    write_diagnostic_window_comparison_json(json_path, payload)
    write_diagnostic_window_comparison_csv(csv_path, payload)
    fig, _axes = plot_diagnostic_window_comparison(payload, savepath=plot_path)

    import matplotlib.pyplot as plt

    plt.close(fig)
    loaded = json.loads(json_path.read_text())
    assert loaded["operation"] == "run_diagnostic_window_comparison"
    assert csv_path.read_text().splitlines()[0].startswith("comparison_index")
    assert plot_path.exists()


def test_diagnostic_window_comparison_example_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "example" / "windows.json"
    output_csv = tmp_path / "example" / "windows.csv"
    output_plot = tmp_path / "example" / "windows.png"
    cmd = [
        sys.executable,
        "examples/diagnostic_window_comparison.py",
        "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
        "--instrument",
        "xshooter",
        "--max-comparisons",
        "3",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--output-plot",
        str(output_plot),
    ]

    completed = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output_json.read_text())
    assert payload["status"] == "planned_no_fits_run"
    assert len(payload["planned_comparisons"]) <= 3
    assert output_csv.exists()
    assert output_plot.exists()
