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
    summary_md = tmp_path / "nested" / "publication_summary.md"
    summary_csv = tmp_path / "nested" / "publication_summary.csv"
    summary_plot = tmp_path / "nested" / "publication_summary.png"
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
        "--output-publication-summary-md",
        str(summary_md),
        "--output-publication-summary-csv",
        str(summary_csv),
        "--output-publication-summary-plot",
        str(summary_plot),
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
    assert payload["publication_summary"]["status"] == "summary_ready_no_baseline_fit"
    assert payload["publication_summary"]["comparison_rows"] == []
    assert "baseline_not_run" in payload["publication_summary"]["headline_flags"]
    next_actions = payload["publication_summary"]["recommended_next_actions"]
    assert next_actions[0]["action"] == "run_baseline_fit"
    assert "--run-baseline-fit" in next_actions[0]["command"]
    assert "_baseline.json" in next_actions[0]["command"]
    stability = payload["publication_summary"][
        "publication_stability_interpretation"
    ]
    assert stability["claim_status"] == "not_evaluated_baseline_missing"
    assert "audit scaffold only" in stability["user_guidance"]
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
    assert summary_md.exists()
    assert summary_csv.exists()
    assert summary_plot.exists()
    assert "Spyctres publication workflow summary" in summary_md.read_text()
    assert "Publication-stability interpretation" in summary_md.read_text()
    assert "Suggested next commands" in summary_md.read_text()
    assert "delta_teff" in summary_csv.read_text().splitlines()[0]


def test_publication_summary_compares_baseline_systematics_and_recovery(tmp_path):
    module = _load_publication_example_module()
    line_diagnostics = {
        "status": "computed",
        "summary": {"quality_flags": ["line_high_chi2_proxy"]},
        "lines": [
            {
                "status": "ok",
                "line_label": "Hβ",
                "used_residuals": {
                    "chi2_red_proxy": 12.0,
                    "rms_fractional_residual": 0.08,
                },
                "quality_flags": ["line_high_chi2_proxy"],
            }
        ],
    }
    payload = {
        "publication_readiness": {
            "publication_ready": False,
            "blockers": ["artifact_review_required"],
        },
        "baseline_fit": {
            "success": True,
            "teff": 9000.0,
            "feh": 0.0,
            "logg": 3.0,
            "rv_kms": 0.0,
            "chi2_red": 2.0,
            "quality_flags": ["high_chi2"],
        },
        "baseline_line_residual_diagnostics": line_diagnostics,
        "systematic_variant_results": {
            "status": "completed",
            "records": [
                {
                    "variant_id": "continuum_mdeg_1",
                    "label": "continuum degree 1",
                    "status": "ok",
                    "fit": {
                        "success": True,
                        "teff": 9125.0,
                        "feh": 0.1,
                        "logg": 3.2,
                        "rv_kms": 1.5,
                        "chi2_red": 2.4,
                        "quality_flags": [],
                    },
                    "line_residual_diagnostics": line_diagnostics,
                }
            ],
        },
        "injection_recovery": {
            "status": "completed_with_recovery_failures",
            "records": [
                {
                    "trial_index": 1,
                    "status": "ok",
                    "fit_summary": {
                        "success": True,
                        "recovered": {
                            "teff": 9075.0,
                            "feh": -0.02,
                            "logg": 3.04,
                            "rv_kms": -0.4,
                        },
                        "delta": {
                            "teff": 75.0,
                            "feh": -0.02,
                            "logg": 0.04,
                            "rv_kms": -0.4,
                        },
                        "chi2_red": 1.2,
                        "quality_flags": ["recovery_shift_warning"],
                        "all_passed": False,
                    },
                }
            ],
        },
    }

    summary = module._build_publication_comparison_summary(payload)

    assert summary["status"] == "summary_ready_publication_blocked"
    assert len(summary["comparison_rows"]) == 3
    assert "publication_gate_blocked" in summary["headline_flags"]
    assert "injection_recovery_needs_review" in summary["headline_flags"]
    assert "line_residual_flags_present" in summary["headline_flags"]
    assert summary["max_abs_parameter_shifts"]["systematic_variant"]["delta_teff"] == 125.0
    assert (
        summary["max_abs_parameter_shifts"]["injection_recovery_trial"]["delta_rv_kms"]
        == 0.4
    )
    calibration = summary["calibration_interpretation"]
    assert calibration["overall_assessment"] == "blocking"
    assert "calibration_interpretation_blocking" in summary["headline_flags"]
    assert any(
        item["scope"] == "publication_readiness"
        and item["assessment"] == "blocking"
        for item in calibration["checks"]
    )
    stability = summary["publication_stability_interpretation"]
    assert stability["claim_status"] == "exploratory_not_publication_stable"
    assert "diagnostic/exploratory" in stability["plain_language_summary"]
    assert stability["limiting_checks"]

    args = module.build_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "publication.json"),
            "--output-publication-summary-md",
            str(tmp_path / "summary" / "publication.md"),
            "--output-publication-summary-csv",
            str(tmp_path / "summary" / "publication.csv"),
            "--output-publication-summary-plot",
            str(tmp_path / "summary" / "publication.png"),
        ]
    )
    written_summary = module._write_publication_summary_outputs(args, payload)

    assert payload["publication_summary"]["status"] == written_summary["status"]
    assert (tmp_path / "summary" / "publication.md").exists()
    assert (tmp_path / "summary" / "publication.csv").exists()
    assert (tmp_path / "summary" / "publication.png").exists()
    md_text = (tmp_path / "summary" / "publication.md").read_text()
    assert "Calibration interpretation" in md_text
    assert "Publication-stability interpretation" in md_text
    assert "Claim status: `exploratory_not_publication_stable`" in md_text
    assert "Suggested next commands" in md_text
    assert "Systematic variants" in md_text
    assert "Injection/recovery" in md_text
    csv_text = (tmp_path / "summary" / "publication.csv").read_text()
    assert "sensitivity_assessment" in csv_text.splitlines()[0]
    assert "continuum_mdeg_1" in csv_text
    assert "trial_1" in csv_text


def test_publication_summary_recommends_baseline_command(tmp_path):
    module = _load_publication_example_module()
    args = module.build_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "publication.json"),
        ]
    )
    payload = {
        "publication_readiness": {
            "publication_ready": False,
            "blockers": ["base_fit_readiness_failed"],
        },
        "baseline_fit": None,
        "systematic_variant_results": None,
        "injection_recovery": None,
    }

    summary = module._build_publication_comparison_summary(payload, args=args)
    actions = summary["recommended_next_actions"]

    assert [item["action"] for item in actions] == ["run_baseline_fit"]
    command = actions[0]["command"]
    assert "--run-baseline-fit" in command
    assert str(tmp_path / "publication_baseline.json") in command
    assert str(tmp_path / "publication.json") not in command
    assert actions[0]["writes_new_checkpoint"] is True


def test_publication_baseline_report_writer_saves_versioned_envelope(tmp_path):
    module = _load_publication_example_module()
    args = module.build_parser().parse_args(
        [
            "--run-baseline-fit",
            "--output-json",
            str(tmp_path / "publication_fit.json"),
            "--output-report-json",
            str(tmp_path / "publication_report.json"),
            "--output-plot",
            str(tmp_path / "publication_fit.png"),
        ]
    )
    result = PhoenixFitResult(
        summary={
            "success": True,
            "teff": 9000.0,
            "fit_setup_hash": "baseline-setup",
        },
        provenance={
            "workflow_api": "fit_stellar_spectrum",
            "workflow_model": "phoenix",
        },
        quality_flags=("ok",),
    )

    module._write_baseline_report_json(args, result)

    report = json.loads((tmp_path / "publication_report.json").read_text())
    assert report["report_type"] == "spyctres.fit_result_report"
    assert report["result"]["generated_files"]["plots"]["referee_plot"] == (
        "publication_fit.png"
    )
    assert report["provenance_summary"]["fit_setup_hash"] == "baseline-setup"
    assert report["provenance_summary"]["quality_flags"] == ["ok"]
    assert report["report_context"]["report_scope"] == "baseline_fit_only"
    assert report["report_context"]["scaffold_checkpoint_json"].endswith(
        "publication_fit.json"
    )


def test_publication_summary_recommends_window_set_command(tmp_path):
    module = _load_publication_example_module()
    args = module.build_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "publication_fit.json"),
        ]
    )
    payload = {
        "publication_readiness": {"publication_ready": True, "blockers": []},
        "per_line_balmer_diagnostics": {"core_mask_halfwidth_A": 4.0},
        "core_mask_sensitivity": [
            {
                "core_mask_halfwidth_A": 0.0,
                "n_fit_candidate": 1000,
                "information_retention_fraction": 1.0,
                "excessive_core_mask": False,
            },
            {
                "core_mask_halfwidth_A": 4.0,
                "n_fit_candidate": 930,
                "information_retention_fraction": 0.93,
                "excessive_core_mask": False,
                "recommended_core_mask": True,
            },
        ],
        "core_mask_sensitivity_recommendation": {
            "recommended_core_mask_halfwidth_A": 4.0,
        },
        "baseline_fit": {
            "success": True,
            "teff": 9000.0,
            "feh": 0.0,
            "logg": 3.0,
            "rv_kms": 0.0,
            "chi2_red": 1.0,
            "quality_flags": [],
        },
        "baseline_line_residual_diagnostics": None,
        "systematic_variant_plan": {
            "variants": [
                {
                    "id": "balmer_core_mask_4A",
                    "category": "balmer_core_mask",
                    "executable_now": True,
                },
                {
                    "id": "window_set_single_hgamma",
                    "category": "fit_windows",
                    "executable_now": True,
                },
                {
                    "id": "window_set_single_hbeta",
                    "category": "fit_windows",
                    "executable_now": True,
                },
            ],
        },
        "systematic_variant_results": {
            "status": "completed",
            "records": [
                {
                    "variant_id": "balmer_core_mask_4A",
                    "category": "balmer_core_mask",
                    "label": "Balmer-core mask half-width 4 A",
                    "status": "ok",
                    "fit": {
                        "success": True,
                        "teff": 9040.0,
                        "feh": 0.0,
                        "logg": 3.01,
                        "rv_kms": 0.2,
                        "chi2_red": 1.0,
                        "quality_flags": [],
                    },
                }
            ],
        },
        "injection_recovery": {
            "status": "completed_all_recovered",
            "records": [
                {
                    "trial_index": 1,
                    "status": "ok",
                    "fit_summary": {
                        "success": True,
                        "recovered": {
                            "teff": 9001.0,
                            "feh": 0.0,
                            "logg": 3.0,
                            "rv_kms": 0.0,
                        },
                        "delta": {
                            "teff": 1.0,
                            "feh": 0.0,
                            "logg": 0.0,
                            "rv_kms": 0.0,
                        },
                        "chi2_red": 1.0,
                        "quality_flags": [],
                        "all_passed": True,
                    },
                }
            ],
        },
    }

    summary = module._build_publication_comparison_summary(payload, args=args)
    actions = {
        item["action"]: item
        for item in summary["recommended_next_actions"]
    }

    assert "run_window_set_sensitivity" in actions
    command = actions["run_window_set_sensitivity"]["command"]
    assert "--run-systematic-variants" in command
    assert "--systematic-variant-ids" in command
    assert "window_set_single_hgamma,window_set_single_hbeta" in command
    assert str(tmp_path / "publication_fit_windowset_variants.json") in command
    assert str(tmp_path / "publication_fit.json") not in command


def test_calibration_interpretation_flags_core_mask_and_window_sensitivity():
    module = _load_publication_example_module()
    payload = {
        "publication_readiness": {"publication_ready": True, "blockers": []},
        "per_line_balmer_diagnostics": {"core_mask_halfwidth_A": 10.0},
        "core_mask_sensitivity": [
            {
                "core_mask_halfwidth_A": 0.0,
                "n_fit_candidate": 1000,
                "rejected_inside_fit_window_fraction": 0.02,
                "information_retention_fraction": 1.0,
                "excessive_core_mask": False,
            },
            {
                "core_mask_halfwidth_A": 4.0,
                "n_fit_candidate": 930,
                "rejected_inside_fit_window_fraction": 0.08,
                "information_retention_fraction": 0.93,
                "excessive_core_mask": False,
                "recommended_core_mask": True,
            },
            {
                "core_mask_halfwidth_A": 10.0,
                "n_fit_candidate": 760,
                "rejected_inside_fit_window_fraction": 0.24,
                "information_retention_fraction": 0.76,
                "excessive_core_mask": False,
            },
        ],
        "core_mask_sensitivity_recommendation": {
            "recommended_core_mask_halfwidth_A": 4.0,
        },
        "baseline_fit": {
            "success": True,
            "teff": 9000.0,
            "feh": 0.0,
            "logg": 3.0,
            "rv_kms": 0.0,
            "chi2_red": 1.0,
            "quality_flags": [],
        },
        "baseline_line_residual_diagnostics": None,
        "systematic_variant_results": {
            "status": "completed",
            "records": [
                {
                    "variant_id": "balmer_core_mask_4A",
                    "category": "balmer_core_mask",
                    "label": "Balmer-core mask half-width 4 A",
                    "status": "ok",
                    "fit": {
                        "success": True,
                        "teff": 9300.0,
                        "feh": 0.03,
                        "logg": 3.02,
                        "rv_kms": 12.0,
                        "chi2_red": 1.4,
                        "quality_flags": [],
                    },
                },
                {
                    "variant_id": "window_set_leave_out_hbeta",
                    "category": "fit_windows",
                    "label": "Leave out Hβ",
                    "status": "ok",
                    "window_labels": ["Hδ", "Hγ"],
                    "fit": {
                        "success": True,
                        "teff": 9150.0,
                        "feh": 0.01,
                        "logg": 3.03,
                        "rv_kms": 2.0,
                        "chi2_red": 1.05,
                        "quality_flags": [],
                    },
                },
            ],
        },
        "injection_recovery": {
            "status": "completed_all_recovered",
            "records": [
                {
                    "trial_index": 1,
                    "status": "ok",
                    "fit_summary": {
                        "success": True,
                        "recovered": {
                            "teff": 9010.0,
                            "feh": 0.01,
                            "logg": 3.01,
                            "rv_kms": 0.2,
                        },
                        "delta": {
                            "teff": 10.0,
                            "feh": 0.01,
                            "logg": 0.01,
                            "rv_kms": 0.2,
                        },
                        "chi2_red": 1.0,
                        "quality_flags": [],
                        "all_passed": True,
                    },
                }
            ],
        },
    }

    summary = module._build_publication_comparison_summary(payload)
    calibration = summary["calibration_interpretation"]
    checks = {item["scope"]: item for item in calibration["checks"]}
    rows = {item["source_id"]: item for item in summary["comparison_rows"]}

    assert calibration["overall_assessment"] == "blocking"
    assert checks["core_mask_audit"]["assessment"] == "borderline"
    assert checks["core_mask_fit_sensitivity"]["assessment"] == "blocking"
    assert checks["window_set_fit_sensitivity"]["assessment"] == "borderline"
    assert checks["same_model_recovery"]["assessment"] == "acceptable"
    assert rows["balmer_core_mask_4A"]["sensitivity_assessment"] == "blocking"
    assert rows["window_set_leave_out_hbeta"]["sensitivity_assessment"] == "borderline"
    assert "calibration_interpretation_blocking" in summary["headline_flags"]


def test_publication_summary_treats_masked_core_note_as_informational():
    module = _load_publication_example_module()
    diagnostics = {
        "status": "computed",
        "summary": {"quality_flags": ["core_model_only_not_fitted"]},
        "lines": [
            {
                "status": "ok",
                "line_label": "Hγ",
                "used_residuals": {
                    "chi2_red_proxy": 1.1,
                    "rms_fractional_residual": 0.01,
                },
                "quality_flags": ["core_model_only_not_fitted"],
            }
        ],
    }
    payload = {
        "publication_readiness": {"publication_ready": True, "blockers": []},
        "baseline_fit": {
            "success": True,
            "teff": 9000.0,
            "feh": 0.0,
            "logg": 3.0,
            "rv_kms": 0.0,
            "chi2_red": 1.2,
            "quality_flags": [],
        },
        "baseline_line_residual_diagnostics": diagnostics,
        "systematic_variant_results": {"status": "planned", "records": []},
        "injection_recovery": {"status": "planned", "records": []},
    }

    summary = module._build_publication_comparison_summary(payload)
    row = summary["comparison_rows"][0]

    assert "line_residual_flags_present" not in summary["headline_flags"]
    assert row["line_quality_flags"] == []
    assert row["line_info_flags"] == ["core_model_only_not_fitted"]
    assert row["problem_lines"] == []


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
