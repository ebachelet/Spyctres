import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from Spyctres.io import SpectrumCollection, SpectrumSegment
from Spyctres.results import PhoenixFitResult


def _load_publication_example_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "publication_quality_xshooter_uvb.py"
    spec = importlib.util.spec_from_file_location(
        "publication_quality_xshooter_uvb_test_module",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_quality_xshooter_uvb_audit_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "publication_scaffold.json"
    comparison_csv = tmp_path / "core_mask_summary.csv"
    comparison_plot = tmp_path / "core_mask_summary.png"
    window_csv = tmp_path / "diagnostic_windows.csv"
    systematic_csv = tmp_path / "systematic_plan.csv"
    line_csv = tmp_path / "balmer_lines.csv"
    cmd = [
        sys.executable,
        "examples/publication_quality_xshooter_uvb.py",
        "--output-json",
        str(output_json),
        "--output-comparison-csv",
        str(comparison_csv),
        "--output-comparison-plot",
        str(comparison_plot),
        "--output-diagnostic-window-csv",
        str(window_csv),
        "--output-systematic-plan-csv",
        str(systematic_csv),
        "--output-balmer-line-csv",
        str(line_csv),
        "--force",
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
    assert payload["workflow"] == "publication_quality_xshooter_uvb_scaffold"
    assert payload["baseline_fit"] is None
    assert payload["baseline_line_residual_diagnostics"] is None
    assert payload["systematic_variant_results"] is None
    assert payload["injection_recovery"] is None
    assert payload["ordinary_readiness"]["n_fit_candidate"] > 0
    assert "publication_readiness" in payload
    assert "balmer_windows" in payload["analysis_design"]
    assert payload["analysis_design"]["metal_rv_windows"]
    generic = payload["analysis_design"]["generic_diagnostic_windows"]
    generic_ids = {item["id"] for item in generic["selection"]["selected"]}
    assert {"h_beta", "h_gamma", "h_delta", "ca_hk_h_epsilon"} <= generic_ids
    assert generic["recommended_combinations"]["combinations"]
    assert payload["analysis_design"]["core_mask_grid_A"] == [0.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    sensitivity = payload["core_mask_sensitivity"]
    recommendation = payload["core_mask_sensitivity_recommendation"]
    assert [item["core_mask_halfwidth_A"] for item in sensitivity] == [
        0.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
    ]
    assert sensitivity[0]["n_fit_candidate"] > sensitivity[-1]["n_fit_candidate"]
    assert sensitivity[0]["information_retention_fraction"] == 1.0
    assert "core_mask_information_penalty" in sensitivity[-1]
    assert "excessive_core_mask" in sensitivity[-1]
    assert recommendation["status"] == "evaluated"
    assert recommendation["recommended_core_mask_halfwidth_A"] is not None
    assert all(item["fit"] is None for item in sensitivity)
    plan = payload["systematic_variant_plan"]
    assert plan["status"] == "planned"
    assert plan["variant_policy"]["expensive_fits_are_opt_in"] is True
    variant_categories = {item["category"] for item in plan["variants"]}
    assert {
        "baseline",
        "continuum_degree",
        "preprocessing_normalization",
        "balmer_core_mask",
        "resolution_assumption",
        "fit_windows",
    } <= variant_categories
    assert any(
        item["id"] == "resolution_scale_unavailable"
        and "missing_explicit_or_validated_resolution" in item["skip_reasons"]
        for item in plan["variants"]
    )
    assert any(item["id"].startswith("window_set_single_") for item in plan["variants"])
    line_diagnostics = payload["per_line_balmer_diagnostics"]
    assert line_diagnostics["status"] == "computed"
    assert line_diagnostics["summary"]["n_lines"] == 3
    line_labels = {item["line_label"] for item in line_diagnostics["lines"]}
    assert {"Hδ", "Hγ", "Hβ"} == line_labels
    assert all(item["diagnostic_type"] == "observed_profile_proxy" for item in line_diagnostics["lines"])
    assert all("absorption_depth_proxy" in item for item in line_diagnostics["lines"])
    assert any(
        overlap.get("id") in {"dib_4428", "dib_4882"}
        for item in line_diagnostics["lines"]
        for overlap in item["known_nonstellar_overlaps"]
    )
    assert comparison_csv.exists()
    assert comparison_plot.exists()
    assert window_csv.exists()
    assert systematic_csv.exists()
    assert line_csv.exists()


def test_balmer_model_residual_diagnostics_from_reconstructed_result(tmp_path):
    module = _load_publication_example_module()
    wave = np.linspace(4310.0, 4370.0, 301)
    center = 4340.5
    flux = 1.0 - 0.25 * np.exp(-0.5 * ((wave - center) / 7.0) ** 2)
    err = np.full_like(wave, 0.02)
    model = flux.copy()
    model += 0.01
    model[wave > center + 8.0] += 0.02
    used = np.abs(wave - center) > 5.0
    excluded = ~used
    segment = SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=np.ones_like(wave, dtype=bool),
        name="Hγ",
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
        meta={
            "line_label": "Hγ",
            "line_center_data": center,
            "line_center_vac": 4340.47,
        },
    )
    collection = SpectrumCollection([segment])
    result = PhoenixFitResult(
        summary={"success": True, "chi2_red": 1.0},
        models=(model,),
        used_masks=(used,),
        excluded_masks=(excluded,),
    )

    diagnostics = module._balmer_model_residual_diagnostics(
        collection,
        result,
        core_mask_halfwidth=5.0,
    )

    assert diagnostics["status"] == "computed"
    assert diagnostics["summary"]["n_evaluated_lines"] == 1
    row = diagnostics["lines"][0]
    assert row["diagnostic_type"] == "model_residuals"
    assert row["used_residuals"]["n_pixels"] == int(np.count_nonzero(used))
    assert row["used_residuals"]["chi2_red_proxy"] is not None
    assert row["wing_residual_asymmetry_fraction"] is not None
    assert "core_model_only_not_fitted" in row["quality_flags"]

    csv_path = tmp_path / "balmer_residuals.csv"
    module._write_balmer_model_residual_csv(csv_path, diagnostics)
    header = csv_path.read_text().splitlines()[0]
    assert "used_chi2_red_proxy" in header
    assert "wing_residual_asymmetry_fraction" in header


def test_systematic_variant_execution_is_bounded_and_checkpointed(tmp_path):
    module = _load_publication_example_module()
    args = module.build_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "publication_with_systematics.json"),
            "--output-systematic-results-csv",
            str(tmp_path / "systematic_results.csv"),
            "--max-systematic-run-variants",
            "2",
            "--force",
        ]
    )

    wave = np.linspace(3900.0, 5050.0, 1600)
    flux = (
        1.0
        - 0.18 * np.exp(-0.5 * ((wave - 4101.7) / 8.0) ** 2)
        - 0.16 * np.exp(-0.5 * ((wave - 4340.5) / 9.0) ** 2)
        - 0.14 * np.exp(-0.5 * ((wave - 4861.3) / 10.0) ** 2)
    )
    err = np.full_like(wave, 0.03)
    segment = SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=np.ones_like(wave, dtype=bool),
        name="synthetic_xshooter_uvb",
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
    )
    case, collection, exclude_masks = module._prepare_balmer_collection(
        args,
        segment,
        core_mask_halfwidth=float(args.balmer_core_mask),
    )
    payload = module._base_payload(
        args,
        args.spectrum,
        segment,
        case,
        collection,
        exclude_masks,
    )

    calls = []

    def fake_runner(
        args_i,
        collection_i,
        exclude_masks_i,
        *,
        output_plot=None,
        return_result=False,
        fit_label="baseline",
    ):
        calls.append(
            {
                "fit_label": fit_label,
                "mdeg": int(args_i.mdeg),
                "norm_mode": str(args_i.norm_mode),
                "n_segments": len(collection_i.segments),
                "exclude_masks": [mask.name for mask in exclude_masks_i],
            }
        )
        models = tuple(np.asarray(seg.flux, dtype=float) * 0.99 for seg in collection_i.segments)
        used_masks = tuple(np.asarray(seg.mask, dtype=bool) for seg in collection_i.segments)
        excluded_masks = tuple(
            np.zeros(np.asarray(seg.wave).shape, dtype=bool)
            for seg in collection_i.segments
        )
        result = PhoenixFitResult(
            summary={
                "success": True,
                "teff": 9000.0 + len(calls),
                "feh": -0.1,
                "logg": 3.0,
                "rv_kms": 0.0,
                "chi2_red": 1.0 + 0.1 * len(calls),
            },
            models=models,
            used_masks=used_masks,
            excluded_masks=excluded_masks,
            quality_flags=("synthetic_test_flag",),
        )
        fit_payload = result.to_dict(include_arrays=False)
        if return_result:
            return fit_payload, result
        return fit_payload

    results = module._run_selected_systematic_variants(
        args,
        segment,
        payload["systematic_variant_plan"],
        args.output_json,
        payload,
        fit_runner=fake_runner,
    )
    module._write_systematic_variant_results_csv(
        args.output_systematic_results_csv,
        results,
    )

    assert results["status"] == "completed"
    assert results["n_requested"] == 2
    assert len(results["records"]) == 2
    assert len(calls) == 2
    assert {record["status"] for record in results["records"]} == {"ok"}
    assert "synthetic_test_flag" in results["quality_flags_seen"]
    assert results["parameter_spread_ok_variants"]["teff"]["n"] == 2
    assert Path(args.output_json).exists()
    checkpoint = json.loads(Path(args.output_json).read_text())
    assert checkpoint["systematic_variant_results"]["n_records"] == 2
    assert Path(args.output_systematic_results_csv).exists()
    csv_header = Path(args.output_systematic_results_csv).read_text().splitlines()[0]
    assert "variant_id" in csv_header
    assert "chi2_red" in csv_header


def test_injection_recovery_is_bounded_and_checkpointed(tmp_path):
    module = _load_publication_example_module()
    args = module.build_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "publication_with_injection.json"),
            "--output-injection-recovery-csv",
            str(tmp_path / "injection_recovery.csv"),
            "--injection-recovery-trials",
            "2",
            "--injection-noise-scale",
            "0",
            "--force",
        ]
    )

    wave = np.linspace(3900.0, 5050.0, 1600)
    flux = (
        1.0
        - 0.18 * np.exp(-0.5 * ((wave - 4101.7) / 8.0) ** 2)
        - 0.16 * np.exp(-0.5 * ((wave - 4340.5) / 9.0) ** 2)
        - 0.14 * np.exp(-0.5 * ((wave - 4861.3) / 10.0) ** 2)
    )
    err = np.full_like(wave, 0.03)
    segment = SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=np.ones_like(wave, dtype=bool),
        name="synthetic_xshooter_uvb",
        wave_medium="air",
        observer_frame="topocentric",
        stellar_rest_status="observed",
    )
    case, collection, exclude_masks = module._prepare_balmer_collection(
        args,
        segment,
        core_mask_halfwidth=float(args.balmer_core_mask),
    )
    payload = module._base_payload(
        args,
        args.spectrum,
        segment,
        case,
        collection,
        exclude_masks,
    )
    baseline_payload = {
        "success": True,
        "teff": 9000.0,
        "feh": 0.0,
        "logg": 3.0,
        "rv_kms": 0.0,
        "chi2_red": 1.0,
    }
    baseline_result = PhoenixFitResult(
        summary=baseline_payload,
        models=tuple(np.asarray(seg.flux, dtype=float) for seg in collection.segments),
        used_masks=tuple(np.asarray(seg.mask, dtype=bool) for seg in collection.segments),
        excluded_masks=tuple(
            np.zeros(np.asarray(seg.wave).shape, dtype=bool)
            for seg in collection.segments
        ),
    )
    calls = []

    def fake_runner(
        args_i,
        collection_i,
        exclude_masks_i,
        *,
        output_plot=None,
        return_result=False,
        fit_label="baseline",
    ):
        calls.append(fit_label)
        result = PhoenixFitResult(
            summary={
                "success": True,
                "teff": 9000.0 + len(calls),
                "feh": 0.01,
                "logg": 3.02,
                "rv_kms": -0.4,
                "chi2_red": 1.05,
            },
            models=tuple(np.asarray(seg.flux, dtype=float) for seg in collection_i.segments),
            used_masks=tuple(
                np.asarray(seg.mask, dtype=bool) for seg in collection_i.segments
            ),
            excluded_masks=tuple(
                np.zeros(np.asarray(seg.wave).shape, dtype=bool)
                for seg in collection_i.segments
            ),
            quality_flags=("synthetic_recovery_test_flag",),
        )
        fit_payload = result.to_dict(include_arrays=False)
        if return_result:
            return fit_payload, result
        return fit_payload

    results = module._run_injection_recovery(
        args,
        collection,
        exclude_masks,
        baseline_result,
        baseline_payload,
        args.output_json,
        payload,
        fit_runner=fake_runner,
    )
    module._write_injection_recovery_csv(
        args.output_injection_recovery_csv,
        results,
    )

    assert results["method"] == "same_model_baseline_noise_injection"
    assert results["status"] == "completed_all_recovered"
    assert results["n_records"] == 2
    assert results["n_passed_all_tolerances"] == 2
    assert len(calls) == 2
    assert all(
        record["fit_summary"]["all_passed"]
        for record in results["records"]
    )
    assert "synthetic_recovery_test_flag" in results["quality_flags_seen"]
    checkpoint = json.loads(Path(args.output_json).read_text())
    assert checkpoint["injection_recovery"]["n_records"] == 2
    csv_header = Path(args.output_injection_recovery_csv).read_text().splitlines()[0]
    assert "delta_teff" in csv_header
    assert "all_passed" in csv_header
