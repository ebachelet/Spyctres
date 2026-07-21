import json
import subprocess
import sys
from pathlib import Path


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
