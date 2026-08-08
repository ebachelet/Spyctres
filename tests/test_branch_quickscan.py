import csv
import json

import numpy as np

from Spyctres.branch_quickscan import (
    build_branch_quickscan_plan,
    plot_branch_quickscan,
    run_branch_quickscan,
    summarize_branch_fit_stability,
    write_branch_quickscan_csv,
    write_branch_quickscan_json,
)
from Spyctres.io import SpectrumSegment


def _blue_segment():
    wave = np.linspace(3800.0, 5220.0, 1800)
    flux = np.ones_like(wave)
    for center, depth, sigma in (
        (3933.7, 0.10, 2.5),
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
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        name="synthetic-blue",
        resolution=6200.0,
    )


def test_branch_plan_is_bounded_and_records_policy():
    plan = build_branch_quickscan_plan(_blue_segment(), max_branches=2)

    assert plan["operation"] == "build_branch_quickscan_plan"
    assert plan["status"] == "planned"
    assert len(plan["planned_branches"]) <= 2
    assert plan["comparison_policy"]["bounded_search"] is True
    assert plan["comparison_policy"]["branches_are_not_final_spectral_types"] is True
    assert plan["planned_branches"][0]["recommended"] is True
    assert plan["planned_branches"][0]["fit_regions_A"]
    assert (
        plan["classification_branch_plan"]["recommended_branch_id"]
        == plan["planned_branches"][0]["id"]
    )


def test_branch_quickscan_dry_run_does_not_fit():
    payload = run_branch_quickscan(_blue_segment(), max_branches=2)

    assert payload["status"] == "planned_no_fits_run"
    assert payload["run_policy"]["run_fits"] is False
    assert len(payload["fit_records"]) == len(payload["planned_branches"])
    assert all(
        record["fit_status"] == "planned_not_run"
        for record in payload["fit_records"]
    )
    assert payload["stability_summary"]["status"] == "not_computed"


def test_branch_quickscan_runs_fake_fits_with_branch_regions():
    calls = []

    def fake_fit(spectrum, **kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "teff": 7000.0 + 100.0 * len(calls),
            "logg": 4.0,
            "feh": -0.1,
            "rv_kms": 2.0,
            "chi2_red": 3.0 / len(calls),
            "quality_flags": ["synthetic"],
        }

    payload = run_branch_quickscan(
        _blue_segment(),
        run_fits=True,
        fit_callable=fake_fit,
        base_fit_kwargs={"R": 6200.0, "rv_grid_n": 9, "max_nfev": 5},
        max_branches=2,
    )

    assert payload["status"] == "fits_completed"
    assert len(calls) == len(payload["planned_branches"])
    assert all(call["auto_defaults"] is False for call in calls)
    assert all(call["R"] == 6200.0 for call in calls)
    assert all(call["rv_grid_n"] == 9 for call in calls)
    assert all(call["regions"] for call in calls)
    assert all(record["fit_status"] == "ok" for record in payload["fit_records"])
    assert payload["fit_records"][0]["quality_flags"] == ["synthetic"]
    assert payload["stability_summary"]["status"] == "ok"
    assert payload["stability_summary"]["parameter_spread"]["teff"]["n"] >= 1


def test_branch_stability_summary_handles_no_completed_fits():
    summary = summarize_branch_fit_stability(
        [{"fit_status": "error", "result_summary": None}]
    )

    assert summary["status"] == "no_completed_fits"
    assert summary["n_completed"] == 0


def test_branch_quickscan_outputs_are_json_csv_and_plot_friendly(tmp_path):
    payload = run_branch_quickscan(_blue_segment(), max_branches=2)
    json_path = tmp_path / "nested" / "branches.json"
    csv_path = tmp_path / "nested" / "branches.csv"
    plot_path = tmp_path / "nested" / "branches.png"

    write_branch_quickscan_json(json_path, payload)
    write_branch_quickscan_csv(csv_path, payload)
    fig, _axes = plot_branch_quickscan(payload, savepath=plot_path)

    import matplotlib.pyplot as plt

    plt.close(fig)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["operation"] == "run_branch_quickscan"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["branch_id"]
    assert plot_path.exists()

