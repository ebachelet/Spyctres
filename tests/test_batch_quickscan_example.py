import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "batch_quickscan_then_refine.py"
    )
    spec = importlib.util.spec_from_file_location("batch_quickscan_then_refine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_input_is_bundled_xshooter_uvb_file():
    module = _load_example_module()

    paths = module._input_paths([])

    assert len(paths) == 1
    assert paths[0].endswith("TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits")
    assert Path(paths[0]).is_file()


def test_parser_accepts_quicklook_output_json_and_resolution_aliases():
    module = _load_example_module()

    args = module.build_parser().parse_args(
        [
            "spectrum.fits",
            "--quicklook",
            "--output-json",
            "/tmp/out.json",
            "--summary-csv",
            "/tmp/out.csv",
            "--R",
            "2000",
        ]
    )

    assert args.quick_only is True
    assert args.output == "/tmp/out.json"
    assert args.summary_csv == "/tmp/out.csv"
    assert args.resolution_R == pytest.approx(2000.0)
    assert module._resolution_override_payload(args) == {
        "resolution_source": "user_override",
        "assumed_resolution_R": 2000.0,
        "assumption_warning": "approximate quicklook resolution",
    }


def test_manifest_records_support_per_target_instrument_and_resolution(tmp_path):
    module = _load_example_module()
    spec_path = tmp_path / "spec.fits"
    spec_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "batch.csv"
    manifest.write_text(
        "target_id,path,instrument,R\n"
        "star_a,spec.fits,sdss,2000\n",
        encoding="utf-8",
    )
    args = module.build_parser().parse_args(
        ["--manifest", str(manifest), "--instrument", "xshooter"]
    )

    records = module._input_records(args)
    local_args = module._args_for_record(args, records[0])

    assert len(records) == 1
    assert records[0]["target_id"] == "star_a"
    assert records[0]["path"] == str(spec_path.resolve())
    assert records[0]["instrument"] == "sdss"
    assert records[0]["resolution_R"] == pytest.approx(2000.0)
    assert local_args.instrument == "sdss"
    assert local_args.resolution_R == pytest.approx(2000.0)


def test_manifest_record_uses_global_resolution_when_row_is_blank(tmp_path):
    module = _load_example_module()
    spec_path = tmp_path / "spec.fits"
    spec_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "batch.csv"
    manifest.write_text("path,instrument\nspec.fits,sdss\n", encoding="utf-8")
    args = module.build_parser().parse_args(
        ["--manifest", str(manifest), "--R", "1800"]
    )

    records = module._input_records(args)
    local_args = module._args_for_record(args, records[0])

    assert local_args.resolution_R == pytest.approx(1800.0)


def test_skip_risky_refinement_gate_uses_readiness_flags():
    module = _load_example_module()
    readiness = {
        "fit_ready": False,
        "interpretation_flags": ["resolution_assumption_required"],
    }
    quick = {"success": True}

    should_refine, reasons = module.should_refine_after_quicklook(
        readiness,
        quick,
        policy="skip-risky",
    )

    assert should_refine is False
    assert "resolution_assumption_required" in reasons
    assert "readiness_fit_ready_false" in reasons


def test_always_refinement_gate_preserves_legacy_behavior():
    module = _load_example_module()

    should_refine, reasons = module.should_refine_after_quicklook(
        {"fit_ready": False, "interpretation_flags": ["no_fitted_pixels"]},
        {"success": False},
        policy="always",
    )

    assert should_refine is True
    assert reasons == []


def test_focused_bounds_from_quick_result_clip_to_base_bounds():
    module = _load_example_module()
    quick = {
        "teff": 9800.0,
        "feh": -0.2,
        "logg": 2.7,
        "rv_kms": -5.0,
    }
    base_bounds = (
        (4500.0, -1.5, 2.5, -300.0),
        (10000.0, 0.5, 5.5, 300.0),
    )

    p0, bounds = module.focused_bounds_from_quick_result(
        quick,
        base_bounds,
        margins=(1000.0, 0.5, 0.8, 50.0),
    )

    assert p0 == pytest.approx((9800.0, -0.2, 2.7, -5.0))
    assert bounds[0] == pytest.approx((8800.0, -0.7, 2.5, -55.0))
    assert bounds[1] == pytest.approx((10000.0, 0.3, 3.5, 45.0))


def test_focused_bounds_reject_nonpositive_margins():
    module = _load_example_module()

    with pytest.raises(ValueError, match="positive"):
        module.focused_bounds_from_quick_result(
            {"teff": 6000.0, "feh": 0.0, "logg": 4.0, "rv_kms": 0.0},
            ((4500.0, -1.5, 2.5, -300.0), (10000.0, 0.5, 5.5, 300.0)),
            margins=(1000.0, 0.0, 0.8, 50.0),
        )


def test_make_refine_fit_kwargs_can_keep_quick_window(monkeypatch):
    module = _load_example_module()

    def fake_prepare(_spectrum, **_kwargs):
        return {
            "p0": (6000.0, 0.0, 4.0, 0.0),
            "bounds": (
                (4500.0, -1.5, 2.5, -300.0),
                (10000.0, 0.5, 5.5, 300.0),
            ),
            "regions": [(3800.0, 7000.0)],
        }, SimpleNamespace(to_dict=lambda: {"fake": True})

    monkeypatch.setattr(module, "prepare_phoenix_fit_kwargs", fake_prepare)
    args = SimpleNamespace(
        refine_defaults_mode="standard",
        refine_window="quick",
        resolution_R=2000.0,
        teff_margin=500.0,
        feh_margin=0.25,
        logg_margin=0.5,
        rv_margin=20.0,
        refine_rv_grid_n=21,
        refine_multistart=2,
        refine_max_nfev=99,
    )
    quick = {
        "teff": 8000.0,
        "feh": 0.0,
        "logg": 4.0,
        "rv_kms": 10.0,
    }

    fit_kwargs, suggestion, focus = module.make_refine_fit_kwargs(
        spectrum=object(),
        quick_result=quick,
        quick_fit_kwargs={"regions": [(3800.0, 5200.0)]},
        args=args,
    )

    assert suggestion.to_dict() == {"fake": True}
    assert fit_kwargs["regions"] == [(3800.0, 5200.0)]
    assert fit_kwargs["p0"] == pytest.approx((8000.0, 0.0, 4.0, 10.0))
    assert fit_kwargs["bounds"][0] == pytest.approx((7500.0, -0.25, 3.5, -10.0))
    assert fit_kwargs["bounds"][1] == pytest.approx((8500.0, 0.25, 4.5, 30.0))
    assert fit_kwargs["rv_grid_n"] == 21
    assert fit_kwargs["multistart"] == 2
    assert fit_kwargs["max_nfev"] == 99
    assert fit_kwargs["R"] == pytest.approx(2000.0)
    assert focus["refine_window_policy"] == "quick"
    assert focus["resolution_assumption"]["resolution_source"] == "user_override"


def test_atomic_outputs_create_parent_directories(tmp_path):
    module = _load_example_module()
    payload = {
        "results": [
            {
                "target_id": "target",
                "path": "spectrum.fits",
                "status": "quick_ok",
                "quick_result": {
                    "teff": 6000.0,
                    "feh": 0.0,
                    "logg": 4.0,
                    "rv_kms": 5.0,
                    "chi2_red": 1.2,
                    "quality_flags": ["ok"],
                },
                "quick_seconds": 0.5,
                "total_seconds": 0.6,
            }
        ]
    }
    json_path = tmp_path / "nested" / "products" / "batch.json"
    csv_path = tmp_path / "nested" / "tables" / "batch.csv"

    module._atomic_write_json(json_path, payload)
    module._atomic_write_summary_csv(csv_path, payload)

    assert json.loads(json_path.read_text(encoding="utf-8"))["results"][0][
        "status"
    ] == "quick_ok"
    text = csv_path.read_text(encoding="utf-8")
    assert "target_id,path,status" in text
    assert "target,spectrum.fits,quick_ok" in text
