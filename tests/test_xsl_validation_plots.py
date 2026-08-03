import json

import pytest

from scripts import xsl_validation_plots


def _payload():
    return {
        "results": [
            {
                "xsl_id": "X0001",
                "spectral_type": "G2V",
                "validation_role": "standard",
                "status": "ok",
                "fit": {"teff": 5800.0, "logg": 4.4, "feh": 0.0},
                "reference": {"teff": 5770.0, "logg": 4.4, "feh": 0.0},
                "validation_plot": {
                    "display_defaults": {"scale_mode": "global"},
                    "segments": [
                        {
                            "name": "UVB",
                            "wave_A": [4000.0, 4100.0, 4200.0],
                            "observed_flux": [1.0, 1.1, 1.0],
                            "model_flux": [1.0, 1.0, 1.0],
                            "used": [True, True, True],
                        }
                    ],
                },
            },
            {
                "xsl_id": "XHOT",
                "spectral_type": "O",
                "validation_role": "unsupported_hot",
                "status": "unsupported_physics",
            },
        ]
    }


def _summary_payload():
    return {
        "validation_role_summary": {
            "ordinary_recovery_count": 1,
            "diagnostic_or_stress_count": 1,
        },
        "results": [
            {
                "xsl_id": "XSTD",
                "star_name": "standard",
                "spectral_type": "G2V",
                "validation_role": "standard",
                "statistics_group": "ordinary_recovery",
                "status": "ok",
                "reference": {"teff": 5770.0, "logg": 4.4, "feh": 0.0},
                "fit": {"teff": 5800.0, "logg": 4.45, "feh": 0.05},
                "delta": {"teff": 30.0, "logg": 0.05, "feh": 0.05},
                "quality_report": {"quality_flags": ["minor_flag"]},
            },
            {
                "xsl_id": "XSTRESS",
                "star_name": "stress",
                "spectral_type": "C",
                "validation_role": "carbon_star",
                "statistics_group": "diagnostic_only",
                "status": "ok",
                "reference": {"teff": 3500.0, "logg": 1.0, "feh": -0.5},
                "fit": {"teff": 7000.0, "logg": 4.0, "feh": 0.5},
                "delta": {"teff": 3500.0, "logg": 3.0, "feh": 1.0},
            },
        ],
    }


def test_render_validation_plots_writes_png_and_pdf(tmp_path):
    output_dir = tmp_path / "plots"
    output_pdf = tmp_path / "plots.pdf"

    images = xsl_validation_plots.render_validation_plots(
        _payload(),
        output_dir=output_dir,
        output_pdf=output_pdf,
    )

    assert len(images) == 1
    assert images[0].endswith("X0001_G2V.png")
    assert (output_dir / "X0001_G2V.png").exists()
    assert output_pdf.exists()


def test_render_validation_plots_filters_ids_and_statuses(tmp_path):
    with pytest.raises(ValueError, match="No matching rows"):
        xsl_validation_plots.render_validation_plots(
            _payload(),
            output_dir=tmp_path,
            xsl_ids=("XHOT",),
            statuses=("ok",),
        )


def test_main_defaults_to_sibling_plot_directory(tmp_path):
    results = tmp_path / "xsl_results.json"
    results.write_text(json.dumps(_payload()), encoding="utf-8")

    status = xsl_validation_plots.main([str(results)])

    assert status == 0
    assert (tmp_path / "xsl_results_plots" / "X0001_G2V.png").exists()


def test_reference_recovery_summary_excludes_stress_from_ordinary_stats():
    summary = xsl_validation_plots.build_reference_recovery_summary(_summary_payload())

    assert summary["status"] == "summary_ready_reference_recovery_acceptable"
    assert summary["claim_status"] == "reference_recovery_acceptable_for_current_thresholds"
    assert summary["ordinary_recovery_statistics"]["count"] == 1
    assert summary["ordinary_recovery_statistics"]["teff"]["median_delta"] == 30.0
    assert len(summary["ordinary_rows"]) == 1
    assert len(summary["diagnostic_or_stress_rows"]) == 1
    assert summary["ordinary_rows"][0]["stability_label"] == "classification_stable"
    assert summary["diagnostic_or_stress_rows"][0]["recovery_assessment"] == "diagnostic_only"
    assert (
        summary["diagnostic_or_stress_rows"][0]["stability_label"]
        == "quality_gated_diagnostic_only"
    )
    reviewed_analysis = summary["review_summary"]
    assert (
        reviewed_analysis["claim_status"]
        == "classification_stable_for_current_standard_targets"
    )
    assert reviewed_analysis["ready_for_referee_review"] is True
    assert reviewed_analysis["ready_for_reviewed_analysis_claim"] is False
    assert reviewed_analysis["ordinary_stability_label_counts"] == {
        "classification_stable": 1
    }
    assert reviewed_analysis["review_questions"]
    assert any("SDSS and UVES-POP" in item for item in summary["recommendations"])


def test_reference_recovery_summary_flags_standard_outlier():
    payload = _summary_payload()
    payload["results"][0]["fit"]["teff"] = 6600.0
    payload["results"][0]["delta"]["teff"] = 830.0

    summary = xsl_validation_plots.build_reference_recovery_summary(payload)

    assert summary["status"] == "summary_ready_reference_recovery_needs_review"
    assert summary["claim_status"] == "reference_recovery_needs_review"
    assert summary["ordinary_rows"][0]["recovery_assessment"] == "needs_review"
    assert summary["ordinary_rows"][0]["stability_label"] == "needs_review_not_stable"
    assert (
        summary["review_summary"]["claim_status"]
        == "exploratory_not_reference_stable"
    )


def test_reference_recovery_summary_missing_reference_is_not_acceptable():
    payload = _summary_payload()
    payload["results"][0]["reference"] = {}
    payload["results"][0]["delta"] = {}

    summary = xsl_validation_plots.build_reference_recovery_summary(payload)

    assert summary["status"] == "summary_ready_reference_recovery_not_evaluated"
    assert summary["claim_status"] == "standard_reference_recovery_not_evaluated"
    assert summary["ordinary_rows"][0]["recovery_assessment"] == "not_evaluated"
    assert (
        summary["ordinary_rows"][0]["stability_label"]
        == "metadata_limited_not_evaluated"
    )


def test_review_summary_flags_failed_standard_before_empty_recovery():
    payload = _summary_payload()
    payload["results"][0]["status"] = "error"
    payload["results"][0]["fit"] = {}
    payload["results"][0]["delta"] = {}

    summary = xsl_validation_plots.build_reference_recovery_summary(payload)

    assert summary["status"] == "summary_ready_no_ordinary_reference_fits"
    assert summary["rows"][0]["stability_label"] == "not_recovered"
    assert (
        summary["review_summary"]["claim_status"]
        == "not_ready_standard_targets_failed"
    )
    assert summary["review_summary"]["limiting_checks"][0]["xsl_id"] == "XSTD"


def test_reference_recovery_summary_outputs_and_main_no_target_plots(tmp_path):
    results = tmp_path / "xsl_results.json"
    md = tmp_path / "summary" / "xsl_summary.md"
    csv_path = tmp_path / "summary" / "xsl_summary.csv"
    plot = tmp_path / "summary" / "xsl_summary.png"
    json_path = tmp_path / "summary" / "xsl_summary.json"
    results.write_text(json.dumps(_summary_payload()), encoding="utf-8")

    status = xsl_validation_plots.main(
        [
            str(results),
            "--no-target-plots",
            "--output-summary-md",
            str(md),
            "--output-summary-csv",
            str(csv_path),
            "--output-summary-plot",
            str(plot),
            "--output-summary-json",
            str(json_path),
        ]
    )

    assert status == 0
    assert md.exists()
    assert csv_path.exists()
    assert plot.exists()
    assert json_path.exists()
    md_text = md.read_text(encoding="utf-8")
    assert "Spyctres XSL reference-recovery summary" in md_text
    assert "Reviewed-analysis stability interpretation" in md_text
    assert "Questions for reviewer" in md_text
    assert "Diagnostic/stress/unsupported rows" in md_text
    csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "recovery_assessment" in csv_header
    assert "stability_label" in csv_header
    summary_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert "review_summary" in summary_json
