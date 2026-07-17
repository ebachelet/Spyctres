import json
from types import SimpleNamespace

import numpy as np

from Spyctres.diagnostics import (
    annotate_nonstellar_features,
    diagnose_known_residual_windows,
)
from Spyctres.io import SpectrumSegment


def test_annotate_nonstellar_features_records_json_safe_policy_modes():
    segment = SpectrumSegment(
        wave=np.linspace(4870.0, 4915.0, 50),
        flux=np.ones(50),
        err=np.ones(50),
        name="uvb",
    )
    result = SimpleNamespace(summary={}, quality_flags=())

    payload = annotate_nonstellar_features(
        segment,
        result,
        feature_names=("dib_4882",),
        policy="mask_known",
    )

    assert payload["policy"] == "mask_known"
    assert payload["features"][0]["action"] == "masked"
    assert payload["features"][0]["mask_applied"] is True
    assert payload["features"][0]["overlap_pixels"] == 50
    assert "nonstellar_mask_applied" in result.quality_flags
    json.dumps(payload, allow_nan=False)


def test_annotate_nonstellar_features_flags_stellar_rest_ism_ambiguity():
    segment = SpectrumSegment(
        wave=np.linspace(4410.0, 4445.0, 50),
        flux=np.ones(50),
        err=np.ones(50),
        name="rest",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
    )
    result = SimpleNamespace(summary={}, quality_flags=())

    payload = annotate_nonstellar_features(
        segment,
        result,
        feature_names=("dib_4428",),
        policy="warn",
    )

    warning = payload["frame_warnings"][0]
    reasons = warning["affected_segments"][0]["reasons"]
    assert "stellar_rest_spectrum_without_ism_velocity" in reasons
    assert "nonstellar_feature_frame_ambiguous" in result.quality_flags


def test_annotate_nonstellar_features_accepts_assumed_ism_velocity():
    segment = SpectrumSegment(
        wave=np.linspace(4410.0, 4445.0, 50),
        flux=np.ones(50),
        err=np.ones(50),
        name="rest",
        wave_frame="stellar_rest",
        observer_frame="barycentric",
        stellar_rest_status="corrected",
    )
    result = SimpleNamespace(summary={}, quality_flags=())

    payload = annotate_nonstellar_features(
        segment,
        result,
        feature_names=("dib_4428",),
        policy="warn",
        assumed_ism_rv_kms=0.0,
    )

    assert payload["frame_warnings"] == []
    assert "nonstellar_feature_frame_ambiguous" not in result.quality_flags


def test_annotate_nonstellar_features_uses_fitted_pixel_overlap_when_available():
    wave = np.linspace(4400.0, 4450.0, 100)
    segment = SpectrumSegment(wave, np.ones_like(wave), err=np.ones_like(wave))
    used = np.ones_like(wave, dtype=bool)
    used[(wave >= 4416.8) & (wave <= 4440.8)] = False
    result = SimpleNamespace(summary={}, used_masks=(used,), quality_flags=())

    payload = annotate_nonstellar_features(
        segment,
        result,
        feature_names=("dib_4428",),
        policy="warn",
    )

    assert payload["overlap_basis"] == "fitted_pixels"
    assert payload["features"][0]["overlap_pixels"] > 0
    assert payload["features"][0]["fitted_pixel_overlap"] is False
    assert payload["features"][0]["fitted_pixel_overlap_pixels"] == 0
    assert "nonstellar_feature_overlap" not in result.quality_flags


def test_diagnose_known_residual_windows_is_json_safe_and_conservative():
    wave = np.linspace(4800.0, 4930.0, 400)
    model = np.ones_like(wave)
    flux = np.ones_like(wave)
    flux[(wave >= 4876.0) & (wave <= 4908.0)] -= 3.0
    segment = SpectrumSegment(wave, flux, err=np.ones_like(wave), name="uvb")
    result = SimpleNamespace(
        summary={},
        models=(model,),
        used_masks=(np.ones_like(wave, dtype=bool),),
        quality_flags=(),
    )

    payload = diagnose_known_residual_windows(
        segment,
        result,
        enabled=True,
        threshold_sigma=2.5,
    )

    flagged = payload["flagged_windows"][0]
    assert flagged["origin_hypothesis"] == "ambiguous"
    assert flagged["residual_detection"]["cross_line_consistency_checked"] is False
    assert "dib_candidate_detected" in result.quality_flags
    json.dumps(payload, allow_nan=False)
