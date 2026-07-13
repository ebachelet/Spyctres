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
