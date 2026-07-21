import numpy as np

from Spyctres.diagnostic_windows import (
    DiagnosticWindow,
    build_diagnostic_window_combinations,
    fit_regions_from_combination,
    format_diagnostic_window_table,
    select_diagnostic_windows,
)
from Spyctres.io import SpectrumCollection, SpectrumSegment
from Spyctres.waveutils import convert_wavelength_medium


def _segment(
    wmin,
    wmax,
    n=1000,
    *,
    mask=None,
    flux=None,
    name="segment",
    wave_medium="air",
):
    wave = np.linspace(wmin, wmax, n)
    if flux is None:
        flux = np.ones(n)
        flux -= 0.15 * np.exp(-0.5 * ((wave - 4861.3) / 7.0) ** 2)
        flux -= 0.10 * np.exp(-0.5 * ((wave - 4340.5) / 6.0) ** 2)
        flux *= 1.0 + 0.02 * np.sin(wave / 70.0)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=np.full(n, 0.02),
        mask=mask,
        wave_medium=wave_medium,
        observer_frame="barycentric",
        stellar_rest_status="observed",
        name=name,
        resolution=5000.0,
    )


def test_blue_optical_selection_includes_balmer_ca_he_and_mg_windows():
    segment = _segment(3800.0, 5220.0)

    selection = select_diagnostic_windows(segment)
    ids = {item["id"] for item in selection["selected"]}

    assert {"h_beta", "h_gamma", "h_delta", "ca_hk_h_epsilon"} <= ids
    assert "he_i_4471" in ids
    assert "mg_ii_4481" in ids
    assert "si_ii_4128_4130" in ids
    assert "mg_i_b" in ids
    assert "co_23um_bandhead" not in ids
    assert selection["selection_policy"]["expensive_fits_run"] is False
    h_beta = next(item for item in selection["selected"] if item["id"] == "h_beta")
    assert h_beta["canonical_coordinates"]["wave_medium"] == "vacuum"
    assert h_beta["canonical_coordinates"]["reference_frame"] == "stellar_rest"
    assert "score_components" in h_beta
    assert "detrended_contrast_score" in h_beta["score_components"]
    assert h_beta["feature_family"] == ["hydrogen"]


def test_blue_optical_selection_includes_ch_g_band_for_intermediate_stars():
    segment = _segment(3800.0, 5220.0)

    selection = select_diagnostic_windows(segment, initial_teff=5500.0)
    ids = {item["id"] for item in selection["selected"]}

    assert "ch_g_band" in ids
    ch = next(item for item in selection["selected"] if item["id"] == "ch_g_band")
    assert "molecular" in ch["feature_family"]
    assert ch["model_support"] == "uncertain"
    assert "carbon_abundance_sensitive" in ch["risk_tags"]


def test_red_optical_selection_includes_cool_molecular_and_alkali_windows():
    segment = _segment(6500.0, 9000.0, n=2500, name="red")

    selection = select_diagnostic_windows(segment, initial_teff=3200.0)
    ids = {item["id"] for item in selection["selected"]}

    assert {"vo_7450", "vo_7900", "feh_8700", "k_i_7700"} <= ids
    ki = next(item for item in selection["selected"] if item["id"] == "k_i_7700")
    assert "atomic_metal" in ki["feature_family"]
    assert "telluric_o2_overlap" in ki["risk_tags"]
    vo = next(item for item in selection["selected"] if item["id"] == "vo_7900")
    assert "molecular" in vo["feature_family"]
    assert vo["model_support"] == "uncertain"


def test_near_ir_selection_has_hot_and_cool_diagnostics():
    wave = np.linspace(8200.0, 24000.0, 2200)
    flux = np.ones_like(wave)
    flux -= 0.12 * (wave > 22935.0) * np.exp(-(wave - 22935.0) / 260.0)
    flux -= 0.05 * np.exp(-0.5 * ((wave - 21661.0) / 20.0) ** 2)
    segment = _segment(8200.0, 24000.0, n=wave.size, flux=flux, name="nir")

    cool_selection = select_diagnostic_windows(segment, initial_teff=3600.0)
    hot_selection = select_diagnostic_windows(segment, initial_teff=10000.0)
    cool_ids = {item["id"] for item in cool_selection["selected"]}
    hot_ids = {item["id"] for item in hot_selection["selected"]}

    assert "ca_ii_triplet_paschen" in cool_ids
    assert "co_23um_bandhead" in cool_ids
    assert "na_i_kband" in cool_ids
    assert "ca_i_kband" in cool_ids
    assert "tio_red_bands" in cool_ids
    assert "paschen_beta" in hot_ids
    assert "brackett_h_band" in hot_ids
    assert "br_gamma" in hot_ids

    cool_scores = {item["id"]: item["score"] for item in cool_selection["selected"]}
    hot_scores = {item["id"]: item["score"] for item in hot_selection["selected"]}
    assert cool_scores["co_23um_bandhead"] > hot_scores["co_23um_bandhead"]
    assert hot_scores["br_gamma"] > cool_scores["br_gamma"]
    assert cool_scores["na_i_kband"] > hot_scores["na_i_kband"]
    co_hot = next(
        item for item in hot_selection["selected"] if item["id"] == "co_23um_bandhead"
    )
    assert co_hot["unconditioned_score"] > co_hot["score"]
    assert hot_selection["selection_policy"]["unconditioned_scores_retained"] is True
    cat = next(
        item for item in cool_selection["selected"] if item["id"] == "ca_ii_triplet_paschen"
    )
    assert cat["subwindows"]
    assert {item["id"] for item in cat["subwindows"]} == {
        "ca_triplet_lines",
        "paschen_triplet_blend",
    }


def test_masked_window_can_be_rejected_for_too_few_usable_pixels():
    mask = np.ones(400, dtype=bool)
    wave = np.linspace(4800.0, 4920.0, mask.size)
    mask[(wave >= 4830.0) & (wave <= 4898.0)] = False
    segment = _segment(4800.0, 4920.0, n=mask.size, mask=mask)

    selection = select_diagnostic_windows(segment)
    selected_ids = {item["id"] for item in selection["selected"]}
    rejected = {item["id"]: item for item in selection["rejected"]}

    assert "h_beta" not in selected_ids
    assert rejected["h_beta"]["reject_reason"] == "too_few_usable_pixels"


def test_collection_selection_combines_segment_coverage():
    blue = _segment(3900.0, 5200.0, n=500, name="blue")
    nir = _segment(21000.0, 23600.0, n=500, name="nir")

    selection = select_diagnostic_windows(SpectrumCollection([blue, nir]))
    ids = {item["id"] for item in selection["selected"]}

    assert "h_beta" in ids
    assert "br_gamma" in ids
    assert "co_23um_bandhead" in ids


def test_role_filter_matches_split_metadata_fields():
    selection = select_diagnostic_windows(
        SpectrumCollection(
            [
                _segment(7000.0, 9000.0, n=1600, name="red"),
                _segment(21000.0, 24000.0, n=1200, name="nir"),
            ]
        ),
        roles=["molecular"],
    )
    ids = {item["id"] for item in selection["selected"]}

    assert "co_23um_bandhead" in ids
    assert "vo_7900" in ids
    assert "feh_8700" in ids
    assert "tio_red_bands" in ids
    assert "br_gamma" not in ids


def test_combinations_are_bounded_and_json_safe():
    selection = select_diagnostic_windows(_segment(3800.0, 9000.0, n=1600))
    combinations = build_diagnostic_window_combinations(
        selection,
        max_windows=4,
        max_single_windows=2,
    )

    kinds = {item["kind"] for item in combinations["combinations"]}
    assert "all_selected_top" in kinds
    assert "role_balanced" in combinations["strategy"]
    assert "single_window" in kinds
    assert "leave_one_out" in kinds
    assert all(item["n_windows"] <= 4 for item in combinations["combinations"])
    assert isinstance(combinations["combinations"][0]["estimated_usable_pixels"], int)
    assert any(
        item["kind"] == "trusted_baseline" for item in combinations["combinations"]
    )
    assert any(
        item["kind"] == "leave_one_family_out"
        for item in combinations["combinations"]
    )
    assert fit_regions_from_combination(combinations["combinations"][0])


def test_format_diagnostic_window_table_is_compact():
    selection = select_diagnostic_windows(_segment(3800.0, 5220.0))

    table = format_diagnostic_window_table(selection, max_rows=3)

    assert "region_A" in table
    assert "h_beta" in table or "h_gamma" in table


def test_air_and_vacuum_segments_select_same_physical_feature():
    window = DiagnosticWindow(
        id="narrow_test_line",
        label="narrow test",
        region_A=(5000.0, 5000.8),
        roles=("rv",),
        min_overlap_A=0.2,
        min_pixels=3,
    )
    vac = _segment(4999.7, 5001.1, n=80, name="vac", wave_medium="vacuum")
    air_wave = convert_wavelength_medium(
        vac.wave,
        from_medium="vacuum",
        to_medium="air",
    )
    air = SpectrumSegment(
        wave=air_wave,
        flux=vac.flux,
        err=vac.err,
        wave_medium="air",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        name="air",
        resolution=80000.0,
    )

    vac_selection = select_diagnostic_windows(vac, windows=[window])
    air_selection = select_diagnostic_windows(air, windows=[window])

    assert vac_selection["selected"][0]["id"] == "narrow_test_line"
    assert air_selection["selected"][0]["id"] == "narrow_test_line"
    op = air_selection["selected"][0]["segment_contributions"][0]
    assert op["wave_medium"] == "air"
    assert op["medium_conversion_applied"] is True
    assert op["operational_region_A"][0] < 5000.0


def test_observed_frame_window_can_be_shifted_by_preliminary_rv():
    window = DiagnosticWindow(
        id="rv_shifted_test_line",
        label="RV shifted test",
        region_A=(5000.0, 5001.0),
        roles=("rv",),
        min_overlap_A=0.2,
        min_pixels=3,
    )
    shifted_center = 5000.5 * (1.0 + 300.0 / 299792.458)
    segment = SpectrumSegment(
        wave=np.linspace(shifted_center - 0.6, shifted_center + 0.6, 80),
        flux=np.ones(80),
        err=np.full(80, 0.02),
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="observed",
        resolution=80000.0,
    )

    without_rv = select_diagnostic_windows(segment, windows=[window])
    with_rv = select_diagnostic_windows(segment, windows=[window], rv_kms=300.0)

    assert not without_rv["selected"]
    assert with_rv["selected"][0]["id"] == "rv_shifted_test_line"
    assert with_rv["selected"][0]["segment_contributions"][0][
        "rv_used_for_window_kms"
    ] == 300.0


def test_gap_and_resolution_element_metrics_are_recorded():
    wave = np.linspace(4800.0, 4920.0, 600)
    mask = np.ones_like(wave, dtype=bool)
    mask[(wave > 4850.0) & (wave < 4875.0)] = False
    segment = SpectrumSegment(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.02),
        mask=mask,
        wave_medium="vacuum",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        resolution=5000.0,
    )

    selection = select_diagnostic_windows(segment)
    h_beta = next(item for item in selection["selected"] if item["id"] == "h_beta")

    assert h_beta["n_contiguous_runs"] >= 2
    assert h_beta["largest_gap_A"] > 20.0
    assert h_beta["n_resolution_elements"] > 1.0
    assert 0.0 < h_beta["largest_contiguous_usable_fraction"] < 1.0
