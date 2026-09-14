import importlib.util
import csv
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
            "--plot-dir",
            "/tmp/plots",
            "--max-plots",
            "2",
        ]
    )

    assert args.quick_only is True
    assert args.output == "/tmp/out.json"
    assert args.summary_csv == "/tmp/out.csv"
    assert args.resolution_R == pytest.approx(2000.0)
    assert args.plot_dir == "/tmp/plots"
    assert args.max_plots == 2
    assert args.refine_quality_policy == "skip-risky"
    assert module._resolution_override_payload(args) == {
        "resolution_source": "user_override",
        "assumed_resolution_R": 2000.0,
        "assumption_warning": "approximate quicklook resolution",
    }


def test_representative_plot_helper_writes_bounded_plot_records(tmp_path, monkeypatch):
    module = _load_example_module()
    calls = []

    def fake_plot_model_line_windows(*_args, **kwargs):
        savepath = Path(kwargs["savepath"])
        savepath.write_text("plot placeholder", encoding="utf-8")
        calls.append(savepath)
        return None, None

    monkeypatch.setattr(module, "plot_model_line_windows", fake_plot_model_line_windows)
    args = SimpleNamespace(plot_dir=str(tmp_path / "plots"), max_plots=1)
    record = {"path": "target one.fits", "target_id": "target one"}

    first = module._maybe_write_representative_plot(
        object(),
        [(4000.0, 4100.0)],
        record,
        args,
        stage="quick",
    )
    second = module._maybe_write_representative_plot(
        object(),
        [(4000.0, 4100.0)],
        record,
        args,
        stage="refined",
    )

    assert first["status"] == "written"
    assert first["purpose"] == "representative_batch_fit_inspection"
    assert Path(first["path"]).is_file()
    assert "target_one" in Path(first["path"]).name
    assert second is None
    assert len(calls) == 1


def test_manifest_records_support_per_target_reader_and_resolution(tmp_path):
    module = _load_example_module()
    spec_path = tmp_path / "spec.fits"
    spec_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "batch.csv"
    manifest.write_text(
        "target_id,path,reader,R\n"
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


def test_manifest_records_keep_instrument_column_as_compatibility_alias(tmp_path):
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
        ["--manifest", str(manifest), "--reader", "xshooter_merge1d"]
    )

    records = module._input_records(args)
    local_args = module._args_for_record(args, records[0])

    assert records[0]["path"] == str(spec_path.resolve())
    assert records[0]["instrument"] == "sdss"
    assert local_args.instrument == "sdss"


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


def test_skip_risky_refinement_gate_records_intent_specific_readiness():
    module = _load_example_module()
    readiness = {
        "intent": "radial_velocity",
        "ready_for_intent": False,
        "fit_ready": False,
        "interpretation_flags": ["wave_medium_unknown"],
    }
    quick = {"success": True}

    should_refine, reasons = module.should_refine_after_quicklook(
        readiness,
        quick,
        policy="skip-risky",
    )

    assert should_refine is False
    assert "readiness_not_ready_for_radial_velocity" in reasons
    assert "readiness_fit_ready_false" in reasons


def test_skip_risky_refinement_gate_can_ignore_archive_only_flags():
    module = _load_example_module()
    readiness = {
        "fit_ready": True,
        "interpretation_flags": ["archive_mask_overlap_inside_fit_window"],
    }
    quick = {"success": True, "quality_flags": []}

    should_refine_warn, reasons_warn = module.should_refine_after_quicklook(
        readiness,
        quick,
        policy="skip-risky",
        archive_mask_policy="warn",
    )
    should_refine_ignore, reasons_ignore = module.should_refine_after_quicklook(
        readiness,
        quick,
        policy="skip-risky",
        archive_mask_policy="ignore",
    )

    assert should_refine_warn is False
    assert reasons_warn == ["archive_mask_overlap_inside_fit_window"]
    assert should_refine_ignore is True
    assert reasons_ignore == []


def test_skip_risky_refinement_gate_uses_quick_quality_flags():
    module = _load_example_module()
    readiness = {"fit_ready": True, "interpretation_flags": []}
    quick = {"success": True, "quality_flags": ["high_chi2", "ok_to_ignore"]}

    should_refine, reasons = module.should_refine_after_quicklook(
        readiness,
        quick,
        policy="skip-risky",
    )

    assert should_refine is False
    assert reasons == ["quick_result:high_chi2"]


def test_always_refinement_gate_preserves_legacy_behavior():
    module = _load_example_module()

    should_refine, reasons = module.should_refine_after_quicklook(
        {"fit_ready": False, "interpretation_flags": ["no_fitted_pixels"]},
        {"success": False},
        policy="always",
    )

    assert should_refine is True
    assert reasons == []


def test_summary_csv_records_intent_readiness_columns(tmp_path):
    module = _load_example_module()
    output = tmp_path / "summary.csv"
    payload = {
        "results": [
            {
                "target_id": "star",
                "path": "spectrum.fits",
                "status": "quick_complete",
                "spectrum_readiness": {
                    "fit_ready": False,
                    "quicklook_only": True,
                    "intent": "quicklook_classification",
                    "ready_for_intent": True,
                    "blockers_for_intent": [],
                    "warnings_for_intent": ["wave_medium_unknown"],
                    "interpretation_flags": ["wave_medium_unknown"],
                    "n_fit_candidate": 20,
                },
                "quick_result": {"success": True, "teff": 6000.0},
            }
        ]
    }

    module._atomic_write_summary_csv(output, payload)

    with output.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["readiness_intent"] == "quicklook_classification"
    assert row["ready_for_intent"] == "True"
    assert row["warnings_for_intent"] == "wave_medium_unknown"


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
