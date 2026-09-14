import json

import numpy as np

from Spyctres.diagnostic_window_comparison import (
    build_diagnostic_window_comparison_plan,
    evaluate_diagnostic_window_common_evaluation,
    evaluate_diagnostic_window_holdout,
    plot_diagnostic_window_comparison,
    run_diagnostic_window_comparison,
    write_diagnostic_window_comparison_csv,
    write_diagnostic_window_comparison_json,
)
from Spyctres.io import SpectrumSegment
from Spyctres.results import PhoenixFitResult


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
    assert payload["fit_records"][0]["held_out_evaluation"]["status"] == (
        "skipped_no_reconstructed_model"
    )
    assert payload["fit_records"][0]["common_evaluation"]["status"] == (
        "skipped_no_reconstructed_model"
    )
    assert payload["common_evaluation_summary"]["status"] == (
        "no_evaluable_common_residuals"
    )


def test_run_comparison_scores_held_out_windows_with_reconstructed_models():
    segment = _segment()

    def fake_fit(spectrum, **kwargs):
        used = np.zeros(spectrum.wave.size, dtype=bool)
        for lo, hi in kwargs["regions"]:
            used |= (spectrum.wave >= lo) & (spectrum.wave <= hi)
        model = np.asarray(spectrum.flux, dtype=float) * 0.99
        return PhoenixFitResult(
            summary={
                "success": True,
                "teff": 7100.0,
                "logg": 4.1,
                "feh": -0.1,
                "rv_kms": 2.0,
                "chi2_red": 1.5,
                "quality_flags": [],
            },
            models=(model,),
            used_masks=(used,),
            excluded_masks=(np.zeros(spectrum.wave.size, dtype=bool),),
            continuum_coefficients=(np.array([1.0, 0.0]),),
        )

    payload = run_diagnostic_window_comparison(
        segment,
        run_fits=True,
        fit_callable=fake_fit,
        max_comparisons=3,
        holdout_min_pixels=3,
    )

    evaluations = [
        item["held_out_evaluation"]
        for item in payload["fit_records"]
        if item["held_out_evaluation"]["status"] == "ok"
    ]
    common_evaluations = [
        item["common_evaluation"]
        for item in payload["fit_records"]
        if item["common_evaluation"]["status"] == "ok"
    ]
    assert evaluations
    assert evaluations[0]["overall"]["n_pixels"] > 0
    assert evaluations[0]["overall"]["mean_chi2_red_proxy"] is not None
    assert any(row["status"] == "ok" for row in evaluations[0]["windows"])
    assert common_evaluations
    assert common_evaluations[0]["overall"]["n_pixels"] > 0
    assert common_evaluations[0]["overall"]["mean_chi2_red_proxy"] is not None
    assert payload["common_evaluation_summary"]["status"] == "ok"
    assert payload["common_evaluation_summary"]["n_evaluable_records"] >= 1
    assert payload["common_evaluation_summary"]["parameter_spread"]["teff"]["n"] >= 1


def test_common_evaluation_keeps_fit_used_pixels():
    segment = _segment()
    plan = build_diagnostic_window_comparison_plan(
        segment,
        max_comparisons=2,
    )
    common_windows = plan["selection"]["selected"][:2]
    result = PhoenixFitResult(
        summary={"success": True, "chi2_red": 1.0},
        models=(segment.flux * 0.997,),
        used_masks=(np.ones(segment.wave.size, dtype=bool),),
        excluded_masks=(np.zeros(segment.wave.size, dtype=bool),),
    )

    common = evaluate_diagnostic_window_common_evaluation(
        segment,
        result,
        common_windows=common_windows,
        min_pixels=3,
    )

    assert common["status"] == "ok"
    assert common["n_evaluated_windows"] >= 1
    assert common["overall"]["n_pixels"] > 0
    assert common["coordinate_policy"].startswith("Uses a fixed union")


def test_evaluate_diagnostic_window_holdout_can_be_called_directly():
    segment = _segment()
    plan = build_diagnostic_window_comparison_plan(
        segment,
        max_comparisons=2,
    )
    comparison = next(
        item for item in plan["planned_comparisons"] if item["held_out_window_ids"]
    )
    used = np.zeros(segment.wave.size, dtype=bool)
    for lo, hi in comparison["regions_for_fit_A"]:
        used |= (segment.wave >= lo) & (segment.wave <= hi)
    result = PhoenixFitResult(
        summary={"success": True, "chi2_red": 1.0},
        models=(segment.flux * 0.995,),
        used_masks=(used,),
        excluded_masks=(np.zeros(segment.wave.size, dtype=bool),),
    )

    heldout = evaluate_diagnostic_window_holdout(
        segment,
        result,
        comparison,
        selected_windows=plan["selection"]["selected"],
    )

    assert heldout["status"] == "ok"
    assert heldout["n_evaluated_windows"] >= 1
    assert heldout["overall"]["n_pixels"] > 0


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
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("comparison_index")
    assert "heldout_mean_chi2_red_proxy" in header
    assert "common_mean_chi2_red_proxy" in header
    assert plot_path.exists()
