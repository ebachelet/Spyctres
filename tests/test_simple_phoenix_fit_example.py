import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
from Spyctres.io import SpectrumCollection, SpectrumSegment


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "simple_phoenix_fit.py"
    spec = importlib.util.spec_from_file_location("simple_phoenix_fit_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Segment:
    def __init__(self, wave, flux=None, err=None, name="segment"):
        self.wave = np.asarray(wave, dtype=float)
        self.flux = (
            np.ones_like(self.wave, dtype=float)
            if flux is None
            else np.asarray(flux, dtype=float)
        )
        self.err = (
            np.ones_like(self.wave, dtype=float)
            if err is None
            else np.asarray(err, dtype=float)
        )
        self.name = name


def test_line_windows_are_selected_from_fitted_pixels_only():
    module = _load_example_module()
    segment = _Segment(np.linspace(3000.0, 5200.0, 2000))
    used = (segment.wave >= 3900.0) & (segment.wave <= 4890.0)

    windows = module._line_windows_for_used_pixels(
        segment,
        used,
        groups=["balmer", "caii", "mgii"],
        half_width_A=20.0,
    )

    labels = [window[0] for window in windows]
    assert labels == [
        "Ca II K 3933.7 Å",
        "Ca II H + Hε 3968.5 Å",
        "Hδ 4101.7 Å",
        "Hγ 4340.5 Å",
        "Mg II 4481.1 Å",
        "Hβ 4861.3 Å",
    ]


def test_auto_line_groups_add_hot_lines_only_for_hot_fits():
    module = _load_example_module()

    assert module._parse_line_groups("auto", result={"teff": 9800.0}) == [
        "balmer",
        "caii",
        "mgii",
    ]
    assert module._parse_line_groups("auto", result={"teff": 11000.0}) == [
        "balmer",
        "caii",
        "mgii",
        "hei",
    ]


def test_overlapping_hot_line_windows_are_merged():
    module = _load_example_module()
    segment = _Segment(np.linspace(4400.0, 4520.0, 500))
    used = np.ones(segment.wave.size, dtype=bool)

    windows = module._line_windows_for_used_pixels(
        segment,
        used,
        groups=["mgii", "hei"],
        half_width_A=20.0,
    )

    assert len(windows) == 1
    assert windows[0][0] == "He I 4471.5 Å + Mg II 4481.1 Å"


def test_line_panel_columns_follow_identified_line_count():
    module = _load_example_module()

    assert module._line_panel_columns(1) == 1
    assert module._line_panel_columns(4) == 2
    assert module._line_panel_columns(5) == 3


def test_line_plot_path_defaults_to_companion_file():
    module = _load_example_module()

    assert (
        module._derive_line_plot_path(None, "/tmp/fit.png")
        == "/tmp/fit_lines.png"
    )
    assert (
        module._derive_line_plot_path("/tmp/custom.png", "/tmp/fit.png")
        == "/tmp/custom.png"
    )
    assert (
        module._derive_line_plot_path(None, "/tmp/fit.png", segment_index=1)
        == "/tmp/fit_lines_segment2.png"
    )


def test_append_exclusion_mask_preserves_existing_masks():
    module = _load_example_module()
    fit_kwargs = {"exclude_masks": [("existing", lambda wave: wave == wave)]}
    mask = module.nonstellar_feature_mask("dib_4428")

    module._append_exclusion_mask(fit_kwargs, mask)

    names = [
        item[0] if isinstance(item, tuple) else item.name
        for item in fit_kwargs["exclude_masks"]
    ]
    assert names == ["existing", "nonstellar:dib_4428"]


def test_reader_kwargs_use_stellar_strict_for_sdss_auto_policy():
    module = _load_example_module()
    args = SimpleNamespace(
        instrument="sdss",
        sdss_mask_policy="auto",
        uves_err_column=None,
    )

    assert module._reader_kwargs_from_args(args) == {
        "sdss_mask_policy": "stellar_strict"
    }

    args.instrument = "xshooter"
    assert module._reader_kwargs_from_args(args) == {}

    args.instrument = "uves-pop"
    args.uves_err_column = 2
    assert module._reader_kwargs_from_args(args) == {"err_column": 2}


def test_wave_medium_override_preserves_segment_metadata():
    module = _load_example_module()
    segment = SpectrumSegment(
        [5000.0, 5001.0],
        [1.0, 1.1],
        err=[0.1, 0.1],
        name="demo",
        wave_medium="unknown",
        meta={"origin": "synthetic"},
    )

    out = module._override_spectrum_wave_medium(segment, "air")

    assert out.wave_medium == "air"
    assert out.meta["origin"] == "synthetic"
    assert out.meta["wave_medium"] == "air"
    assert out.meta["wave_medium_source"] == "user_override"
    assert segment.wave_medium == "unknown"


def test_wave_medium_override_handles_collections():
    module = _load_example_module()
    collection = SpectrumCollection(
        [
            SpectrumSegment([5000.0], [1.0], wave_medium="unknown", name="a"),
            SpectrumSegment([6000.0], [1.0], wave_medium="unknown", name="b"),
        ],
        name="two_segments",
    )

    out = module._override_spectrum_wave_medium(collection, "vacuum")

    assert isinstance(out, SpectrumCollection)
    assert out.meta["wave_medium_override"] == "vacuum"
    assert [segment.wave_medium for segment in out] == ["vacuum", "vacuum"]


def test_resolution_override_filters_missing_resolution_display():
    module = _load_example_module()
    args = SimpleNamespace(resolution_R=2000.0)

    payload = module._resolution_override_summary(args)

    assert payload == {
        "resolution_source": "user_override",
        "assumed_resolution_R": 2000.0,
        "assumption_warning": "approximate quicklook resolution",
    }
    assert module._display_risk_flags(
        args,
        ["missing_resolution", "unknown_wave_medium"],
    ) == ["unknown_wave_medium"]
    assert module._display_warnings(
        args,
        [
            "segment lacks resolution metadata",
            "wavelength medium is unknown",
        ],
    ) == ["wavelength medium is unknown"]


def test_parser_accepts_optional_versioned_report_output(tmp_path):
    module = _load_example_module()

    args = module.build_parser().parse_args(
        [
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
            "--instrument",
            "xshooter",
            "--output-json",
            str(tmp_path / "compact.json"),
            "--output-report-json",
            str(tmp_path / "report.json"),
            "--record-input-checksum",
        ]
    )

    assert args.output_json.endswith("compact.json")
    assert args.output_report_json.endswith("report.json")
    assert args.record_input_checksum is True


def test_line_diagnostics_skip_zero_flux_blocks():
    module = _load_example_module()
    wave = np.linspace(8460.0, 8690.0, 1200)
    flux = np.ones_like(wave)
    flux[(wave >= 8538.0) & (wave <= 8660.0)] = 0.0
    segment = _Segment(wave, flux=flux, err=np.ones_like(wave), name="red_gap")
    used = np.ones(wave.size, dtype=bool)

    windows, diagnostics = module._line_windows_for_used_pixels(
        segment,
        used,
        groups=["caii"],
        half_width_A=30.0,
        return_diagnostics=True,
    )

    labels = [window[0] for window in windows]
    skipped = {record["label"]: record["reasons"] for record in diagnostics["skipped"]}
    assert labels == ["Ca II 8498.0 Å"]
    assert "Ca II 8542.1 Å" in skipped
    assert "Ca II 8662.1 Å" in skipped
    assert "zero_flux_block" in skipped["Ca II 8542.1 Å"]


def test_known_residual_window_diagnostic_flags_hbeta_red_wing():
    module = _load_example_module()
    wave = np.linspace(4800.0, 4930.0, 400)
    model = np.ones_like(wave)
    flux = np.ones_like(wave)
    hbeta_red = (wave >= 4876.0) & (wave <= 4908.0)
    flux[hbeta_red] -= 3.0
    segment = _Segment(wave, flux=flux, err=np.ones_like(wave), name="uvb")
    result = SimpleNamespace(
        summary={},
        models=(model,),
        used_masks=(np.ones_like(wave, dtype=bool),),
        quality_flags=(),
    )
    args = SimpleNamespace(
        known_residual_diagnostics=True,
        known_residual_threshold=2.5,
    )

    payload = module._diagnose_known_residual_windows(args, segment, result)

    assert result.summary["known_residual_windows"] is payload
    assert payload["flagged_windows"][0]["name"] == "DIB 4882 / Hβ red wing"
    assert payload["flagged_windows"][0]["linked_feature"] == "dib_4882"
    assert payload["flagged_windows"][0]["absorption_like"] is True
    assert payload["flagged_windows"][0]["origin_hypothesis"] == "ambiguous"
    assert (
        payload["flagged_windows"][0]["residual_detection"]["candidate_detected"]
        is True
    )
    assert payload["flagged_windows"][0]["median_sigma"] < -2.5
    assert "known_line_region_residual" in result.quality_flags
    assert "dib_overlap_balmer_wing" in result.quality_flags
    assert "dib_candidate_detected" in result.quality_flags


def test_positive_known_residual_points_to_intrinsic_or_composite_candidate():
    module = _load_example_module()
    wave = np.linspace(4800.0, 4930.0, 400)
    model = np.ones_like(wave)
    flux = np.ones_like(wave)
    hbeta_red = (wave >= 4876.0) & (wave <= 4908.0)
    flux[hbeta_red] += 3.0
    segment = _Segment(wave, flux=flux, err=np.ones_like(wave), name="uvb")
    result = SimpleNamespace(
        summary={},
        models=(model,),
        used_masks=(np.ones_like(wave, dtype=bool),),
        quality_flags=(),
    )
    args = SimpleNamespace(
        known_residual_diagnostics=True,
        known_residual_threshold=2.5,
    )

    payload = module._diagnose_known_residual_windows(args, segment, result)

    flagged = payload["flagged_windows"][0]
    assert flagged["emission_like"] is True
    assert flagged["residual_sign"] == "emission_like"
    assert flagged["origin_hypothesis"] == "intrinsic_or_composite_candidate"
    assert "Masking is not recommended" in flagged["recommended_action"]
    assert "known_line_region_residual" in result.quality_flags
    assert "dib_candidate_detected" not in result.quality_flags


def test_known_residual_window_diagnostic_can_be_disabled():
    module = _load_example_module()
    wave = np.linspace(4800.0, 4930.0, 400)
    segment = _Segment(wave)
    result = SimpleNamespace(
        summary={},
        models=(np.ones_like(wave),),
        used_masks=(np.ones_like(wave, dtype=bool),),
        quality_flags=(),
    )
    args = SimpleNamespace(
        known_residual_diagnostics=False,
        known_residual_threshold=2.5,
    )

    payload = module._diagnose_known_residual_windows(args, segment, result)

    assert payload["enabled"] is False
    assert payload["flagged_windows"] == []
    assert result.quality_flags == ()


def test_nonstellar_feature_annotation_flags_dib_balmer_overlap():
    module = _load_example_module()
    wave = np.linspace(4400.0, 4920.0, 400)
    segment = _Segment(wave)
    result = SimpleNamespace(summary={}, quality_flags=())
    args = SimpleNamespace(
        show_dibs=True,
        mask_dibs=False,
        nonstellar_feature_policy="warn",
        dib_padding=0.0,
    )

    payload = module._annotate_nonstellar_features(args, segment, result)

    assert payload["policy"] == "warn"
    assert payload["mask_application_frame"] == "data"
    assert [item["name"] for item in payload["features"]] == ["DIB 4428", "DIB 4882"]
    assert payload["features"][0]["action"] == "flagged"
    assert payload["features"][0]["mask_applied"] is False
    assert payload["features"][0]["residual_detection"] is None
    assert payload["overlap_diagnostics"][0]["flag"] == "dib_overlap_balmer_wing"
    assert payload["overlap_diagnostics"][0]["origin_hypothesis"] == "catalog_overlap_only"
    assert payload["frame_warnings"][0]["warning"] == "nonstellar_feature_frame_ambiguous"
    assert "nonstellar_feature_overlap" in result.quality_flags
    assert "diagnostic_line_contaminated" in result.quality_flags
    assert "dib_overlap_balmer_wing" in result.quality_flags
    assert "nonstellar_feature_frame_ambiguous" in result.quality_flags


def test_nonstellar_feature_annotation_mask_policy_records_mask_flag():
    module = _load_example_module()
    wave = np.linspace(4870.0, 4915.0, 100)
    segment = _Segment(wave)
    result = SimpleNamespace(summary={}, quality_flags=())
    args = SimpleNamespace(
        show_dibs=True,
        mask_dibs=True,
        nonstellar_feature_policy="warn",
        dib_padding=0.0,
    )

    payload = module._annotate_nonstellar_features(args, segment, result)

    assert payload["policy"] == "mask_known"
    assert payload["mask_dibs"] is True
    assert payload["features"][0]["action"] == "masked"
    assert payload["features"][0]["mask_applied"] is True
    assert "nonstellar_mask_applied" in result.quality_flags


def test_nonstellar_feature_annotation_ignore_policy_records_without_flags():
    module = _load_example_module()
    segment = _Segment(np.linspace(4870.0, 4915.0, 100))
    result = SimpleNamespace(summary={}, quality_flags=())
    args = SimpleNamespace(
        show_dibs=True,
        mask_dibs=False,
        nonstellar_feature_policy="ignore",
        dib_padding=0.0,
    )

    payload = module._annotate_nonstellar_features(args, segment, result)

    assert payload["policy"] == "ignore"
    assert payload["features"][0]["name"] == "DIB 4882"
    assert payload["features"][0]["action"] == "ignored"
    assert result.quality_flags == ()
