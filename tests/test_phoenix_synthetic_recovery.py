import numpy as np
import pytest

from Spyctres.fitting import fit_phoenix_full_spectrum
from Spyctres.io import SpectrumSegment
from Spyctres.preprocessing import exclusion_mask
from Spyctres.waveutils import convert_wavelength_medium


class AnalyticPhoenixLikeLibrary:
    """Small deterministic PHOENIX-like model grid for recovery tests.

    The line depths vary independently with Teff, [Fe/H], and logg.  This keeps
    the synthetic tests fast while still exercising Spyctres' real fitter,
    continuum solve, RV initialization, metadata diagnostics, and masks.
    """

    DEFAULT_TEFF_GRID = np.array([5600.0, 5900.0, 6200.0, 6500.0])
    DEFAULT_FEH_GRID = np.array([-0.5, 0.0, 0.5])
    DEFAULT_LOGG_GRID = np.array([3.6, 4.0, 4.4])

    def __init__(self):
        self.wave = None
        self._grid = (
            self.DEFAULT_TEFF_GRID,
            self.DEFAULT_FEH_GRID,
            self.DEFAULT_LOGG_GRID,
        )

    def interpolator_matches(self, observed_wave, *args, **kwargs):
        return (
            self.wave is not None
            and np.array_equal(np.asarray(observed_wave, dtype=float), self.wave)
        )

    def build_interpolator(self, observed_wave, *args, **kwargs):
        self.wave = np.asarray(observed_wave, dtype=float).copy()

    def evaluate(self, teff, feh, logg):
        if self.wave is None:
            raise ValueError("interpolator has not been built")
        teff = float(teff)
        feh = float(feh)
        logg = float(logg)
        if not (5400.0 <= teff <= 6700.0 and -0.75 <= feh <= 0.75 and 3.4 <= logg <= 4.6):
            raise ValueError("outside synthetic grid")

        wave = self.wave
        t = (teff - 6000.0) / 1000.0
        z = feh
        g = logg - 4.0
        line_teff = (0.19 + 0.08 * t) * _gaussian(wave, 5020.0, 0.65)
        line_feh = (0.18 + 0.10 * z) * _gaussian(wave, 5060.0, 0.75)
        line_logg = (0.17 + 0.11 * g) * _gaussian(wave, 5105.0, 0.85)
        cross_term = (0.08 + 0.02 * t - 0.02 * z + 0.02 * g) * _gaussian(
            wave, 5140.0, 1.1
        )
        return 1.0 - line_teff - line_feh - line_logg - cross_term


def _gaussian(wave, center, sigma):
    return np.exp(-0.5 * ((np.asarray(wave, dtype=float) - center) / sigma) ** 2)


def _shift_model_on_grid(wave, flux, rv_kms):
    # Same standard-sign convention used by the observed-grid PHOENIX branch:
    # positive RV redshifts template features.
    source_wave = np.asarray(wave, dtype=float) / (1.0 + float(rv_kms) / 299792.458)
    return np.interp(source_wave, wave, flux, left=flux[0], right=flux[-1])


def _make_synthetic_segment(
    *,
    true_params=(6200.0, 0.0, 4.0, 12.0),
    err=0.01,
    continuum=(1.0, 0.0),
    wave=None,
    wave_medium="vacuum",
    observer_frame="barycentric",
    stellar_rest_status="observed",
    meta=None,
    mask=None,
    resolution=None,
):
    wave = np.linspace(4990.0, 5165.0, 1200) if wave is None else np.asarray(wave, dtype=float)
    library = AnalyticPhoenixLikeLibrary()
    library.build_interpolator(wave)
    rest = library.evaluate(*true_params[:3])
    shifted = _shift_model_on_grid(wave, rest, true_params[3])
    x = (wave - np.nanmedian(wave)) / np.ptp(wave)
    flux = shifted * (float(continuum[0]) + float(continuum[1]) * x)
    sigma = np.full_like(wave, float(err), dtype=float)
    if mask is None:
        mask = np.ones_like(wave, dtype=bool)
    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=sigma,
        mask=mask,
        wave_medium=wave_medium,
        wave_frame=observer_frame,
        observer_frame=observer_frame,
        stellar_rest_status=stellar_rest_status,
        meta={} if meta is None else dict(meta),
        resolution=resolution,
        name="synthetic_recovery",
    )


def _fit_synthetic(segment, **kwargs):
    library = AnalyticPhoenixLikeLibrary()
    return fit_phoenix_full_spectrum(
        segment,
        phoenix_lib=library,
        p0=kwargs.pop("p0", (5900.0, -0.2, 3.8, 0.0)),
        bounds=kwargs.pop(
            "bounds",
            ((5600.0, -0.5, 3.6, -40.0), (6500.0, 0.5, 4.4, 40.0)),
        ),
        teff_grid=AnalyticPhoenixLikeLibrary.DEFAULT_TEFF_GRID,
        feh_grid=AnalyticPhoenixLikeLibrary.DEFAULT_FEH_GRID,
        logg_grid=AnalyticPhoenixLikeLibrary.DEFAULT_LOGG_GRID,
        forward_model="interp_observed",
        mdeg=kwargs.pop("mdeg", 1),
        rv_grid_n=kwargs.pop("rv_grid_n", 81),
        rv_grid_decimate=kwargs.pop("rv_grid_decimate", 2),
        max_nfev=kwargs.pop("max_nfev", 120),
        **kwargs,
    )


def test_synthetic_clean_recovery_with_rv_and_continuum_tilt():
    truth = (6200.0, 0.0, 4.0, 12.0)
    segment = _make_synthetic_segment(
        true_params=truth,
        continuum=(1.03, 0.08),
        err=0.01,
    )

    result = _fit_synthetic(segment)

    assert result["success"] is True
    assert result["teff"] == pytest.approx(truth[0], abs=90.0)
    assert result["feh"] == pytest.approx(truth[1], abs=0.08)
    assert result["logg"] == pytest.approx(truth[2], abs=0.08)
    assert result["rv_kms"] == pytest.approx(truth[3], abs=0.8)
    assert result["chi2_red"] < 1e-6


def test_synthetic_masks_do_not_bias_recovery_when_reported():
    truth = (5900.0, -0.5, 4.4, -18.0)
    segment = _make_synthetic_segment(true_params=truth, err=0.02)
    bad_region = exclusion_mask(
        "synthetic_bad_pixels",
        lambda wave: (np.asarray(wave) > 5055.0) & (np.asarray(wave) < 5065.0),
        metadata={"method": "synthetic_recovery_region"},
    )

    result = _fit_synthetic(segment, exclude_masks=[bad_region], mdeg=0)

    assert result["success"] is True
    assert result["teff"] == pytest.approx(truth[0], abs=100.0)
    assert result["logg"] == pytest.approx(truth[2], abs=0.08)
    assert result["rv_kms"] == pytest.approx(truth[3], abs=1.0)
    assert result["diagnostics"]["segment_diagnostics"][0]["mask_summary"][
        "n_rejected_by_explicit_union"
    ] > 0


def test_synthetic_air_vacuum_mismatch_is_not_silent():
    truth = (6200.0, 0.0, 4.0, 0.0)
    vacuum_segment = _make_synthetic_segment(true_params=truth, err=0.01)
    air_values = convert_wavelength_medium(
        vacuum_segment.wave,
        from_medium="vacuum",
        to_medium="air",
    )
    mislabeled = vacuum_segment.copy(
        wave=air_values,
        wave_medium="unknown",
        wave_frame="unknown",
        observer_frame="unknown",
        stellar_rest_status="unknown",
    )

    result = _fit_synthetic(mislabeled, mdeg=0)

    assert "unknown_wave_medium_used_in_fit" in result["quality_flags"]
    assert "unknown_observer_frame_used_in_fit" in result["quality_flags"]
    assert "stellar_rest_status_unknown" in result["quality_flags"]
    assert "rv_interpretation_ambiguous" in result["quality_flags"]


def test_synthetic_barycentric_metadata_risks_are_flagged():
    topocentric = _make_synthetic_segment(
        true_params=(6200.0, 0.0, 4.0, 0.0),
        observer_frame="topocentric",
        stellar_rest_status="observed",
        meta={"barycorr_kms": 24.0},
    )
    recorded_not_applied = _fit_synthetic(topocentric, rv_bary_kms=0.0, mdeg=0)
    assert (
        "barycentric_correction_recorded_not_applied"
        in recorded_not_applied["quality_flags"]
    )

    already_corrected = _make_synthetic_segment(
        true_params=(6200.0, 0.0, 4.0, 0.0),
        observer_frame="barycentric",
        stellar_rest_status="corrected",
        meta={"barycorr_kms": 24.0},
    )
    double_risk = _fit_synthetic(already_corrected, rv_bary_kms=24.0, mdeg=0)
    assert (
        "possible_double_barycentric_or_rest_correction"
        in double_risk["quality_flags"]
    )


def test_synthetic_lsf_sampling_guardrails_are_reported():
    coarse_wave = np.linspace(4990.0, 5165.0, 45)
    segment = _make_synthetic_segment(
        true_params=(6200.0, 0.0, 4.0, 0.0),
        wave=coarse_wave,
        err=0.01,
    )

    result = _fit_synthetic(segment, fwhm_kms=10.0, mdeg=0, max_nfev=20)

    resolution = result["diagnostics"]["resolution_metadata_summary"]
    assert resolution["pixels_per_fwhm_min"] < 2.0
    assert resolution["low_sampling_warning"] is True
    assert "low_sampling_warning" in result["quality_flags"]
