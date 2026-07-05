"""Small, LSF-aware local spectral-line diagnostic layer."""

from dataclasses import dataclass, field, replace
import json

import numpy as np
from scipy.optimize import least_squares

from .io import SpectrumSegment
from .waveutils import C_KMS, convert_wavelength_medium


@dataclass(frozen=True)
class LineSpec:
    name: str
    rest_wave: float
    kind: str = "absorption"
    window_A: float = 10.0
    search_kms: float = 150.0
    max_shift_kms: float = 450.0
    blend_group: str | None = None
    wave_medium: str = "unknown"

    def __post_init__(self):
        if self.kind not in {"absorption", "emission", "auto"}:
            raise ValueError("line kind must be absorption, emission, or auto.")
        if self.wave_medium not in {"air", "vacuum", "unknown"}:
            raise ValueError("line wave_medium must be air, vacuum, or unknown.")
        for name in ("rest_wave", "window_A", "search_kms", "max_shift_kms"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("{0} must be finite and > 0.".format(name))


@dataclass(frozen=True)
class LineFitConfig:
    continuum_order: int = 1
    n_retries: int = 8
    retry_scale: float = 0.3
    rv_guess_kms: float = 0.0
    min_points: int = 12
    high_chi2_threshold: float = 5.0
    masked_fraction_threshold: float = 0.4
    random_seed: int | None = 0

    def __post_init__(self):
        if self.continuum_order not in {0, 1, 2, 3}:
            raise ValueError("continuum_order must be between 0 and 3.")
        if self.n_retries < 1 or self.min_points < 6:
            raise ValueError("n_retries must be >= 1 and min_points >= 6.")


@dataclass(frozen=True)
class LineFitResult:
    line_name: str
    rest_wave: float
    kind: str
    success: bool
    center_wave: float = np.nan
    center_err: float = np.nan
    rv_kms: float = np.nan
    rv_err_kms: float = np.nan
    amplitude: float = np.nan
    sigma_A: float = np.nan
    fwhm_A: float = np.nan
    equivalent_width_A: float = np.nan
    line_flux: float = np.nan
    chi2: float = np.nan
    chi2_red: float = np.nan
    n_points: int = 0
    mask_fraction: float = np.nan
    continuum_coefficients: tuple = field(default_factory=tuple)
    flags: tuple = field(default_factory=tuple)
    wave: tuple = field(default_factory=tuple)
    flux: tuple = field(default_factory=tuple)
    model_flux: tuple = field(default_factory=tuple)
    continuum: tuple = field(default_factory=tuple)
    residuals: tuple = field(default_factory=tuple)
    line_wave_medium: str = "unknown"
    segment_wave_medium: str = "unknown"
    rest_wave_in_segment_medium: float = np.nan
    instrumental_fwhm_A: float = np.nan

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)


def _poly_design(x, order):
    return np.polynomial.legendre.legvander(x, int(order))


def _parabolic_seed(wave, flux, kind):
    index = int(np.nanargmin(flux) if kind == "absorption" else np.nanargmax(flux))
    if index == 0 or index == len(wave) - 1:
        return float(wave[index])
    x = wave[index - 1:index + 2]
    y = flux[index - 1:index + 2]
    coeff = np.polyfit(x - x[1], y, 2)
    if coeff[0] == 0:
        return float(wave[index])
    offset = -coeff[1] / (2.0 * coeff[0])
    if abs(offset) > max(abs(x[1] - x[0]), abs(x[2] - x[1])):
        return float(wave[index])
    return float(x[1] + offset)


def fit_line(segment, line, config=None):
    """Fit a Gaussian local line plus multiplicative Legendre continuum."""
    if not isinstance(segment, SpectrumSegment):
        raise TypeError("segment must be a SpectrumSegment.")
    if not isinstance(line, LineSpec):
        line = LineSpec(**line) if isinstance(line, dict) else LineSpec(*line)
    config = LineFitConfig() if config is None else config

    segment_medium = str(segment.wave_medium).lower()
    rest_wave_data = float(line.rest_wave)
    if (
        line.wave_medium in {"air", "vacuum"}
        and segment_medium in {"air", "vacuum"}
        and line.wave_medium != segment_medium
    ):
        rest_wave_data = float(
            convert_wavelength_medium(
                [rest_wave_data],
                from_medium=line.wave_medium,
                to_medium=segment_medium,
            )[0]
        )
    expected = rest_wave_data * (1.0 + float(config.rv_guess_kms) / C_KMS)
    half_window = float(line.window_A)
    in_window = np.isfinite(segment.wave) & (np.abs(segment.wave - expected) <= half_window)
    valid = in_window & np.asarray(segment.mask, dtype=bool) & np.isfinite(segment.flux)
    if segment.err is not None:
        valid &= np.isfinite(segment.err) & (segment.err > 0)
    n_total = int(np.count_nonzero(in_window))
    n_valid = int(np.count_nonzero(valid))
    masked_fraction = 1.0 - n_valid / max(1, n_total)
    if n_valid < config.min_points:
        return LineFitResult(
            line.name, line.rest_wave, line.kind, False,
            n_points=n_valid, mask_fraction=masked_fraction,
            flags=("failed_fit", "insufficient_points"),
            line_wave_medium=line.wave_medium,
            segment_wave_medium=segment_medium,
            rest_wave_in_segment_medium=rest_wave_data,
        )

    wave = np.asarray(segment.wave[valid], dtype=float)
    flux = np.asarray(segment.flux[valid], dtype=float)
    x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
    design = _poly_design(x, config.continuum_order)
    edge = np.abs(wave - expected) > 0.55 * half_window
    if np.count_nonzero(edge) < config.continuum_order + 2:
        edge = np.ones_like(wave, dtype=bool)
    coeff0 = np.linalg.lstsq(design[edge], flux[edge], rcond=None)[0]
    continuum0 = design @ coeff0
    norm = flux / np.where(continuum0 != 0, continuum0, 1.0)
    kind = line.kind
    if kind == "auto":
        kind = "absorption" if 1.0 - np.nanmin(norm) >= np.nanmax(norm) - 1.0 else "emission"
    center0 = _parabolic_seed(wave, norm, kind)
    amplitude0 = float(np.nanmin(norm) - 1.0 if kind == "absorption" else np.nanmax(norm) - 1.0)
    spacing = float(np.median(np.diff(wave)))
    sigma0 = max(2.0 * spacing, expected * 10.0 / C_KMS)

    err_estimated = segment.err is None
    if err_estimated:
        scatter = 1.4826 * np.nanmedian(np.abs(flux[edge] - np.nanmedian(flux[edge])))
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = max(np.nanstd(flux[edge]), 1.0e-6)
        err = np.full_like(flux, scatter)
    else:
        err = np.asarray(segment.err[valid], dtype=float)

    search_A = expected * float(line.search_kms) / C_KMS
    hard_A = expected * float(line.max_shift_kms) / C_KMS
    center_lo, center_hi = expected - hard_A, expected + hard_A
    sigma_lo = max(0.5 * spacing, np.finfo(float).eps)
    sigma_hi = half_window / 2.0
    amp_bounds = (-2.0, -1.0e-8) if kind == "absorption" else (1.0e-8, 5.0)
    lower = np.r_[center_lo, sigma_lo, amp_bounds[0], np.full(len(coeff0), -np.inf)]
    upper = np.r_[center_hi, sigma_hi, amp_bounds[1], np.full(len(coeff0), np.inf)]

    def evaluate(params):
        center, sigma, amplitude = params[:3]
        continuum = design @ params[3:]
        profile = np.exp(-0.5 * ((wave - center) / sigma) ** 2)
        return continuum * (1.0 + amplitude * profile), continuum

    def residuals(params):
        return (flux - evaluate(params)[0]) / err

    rng = np.random.default_rng(config.random_seed)
    base = np.r_[np.clip(center0, center_lo, center_hi), sigma0, amplitude0, coeff0]
    base = np.clip(base, lower + 1.0e-12, upper - 1.0e-12)
    best = None
    for attempt in range(config.n_retries):
        start = base.copy()
        if attempt:
            start[0] += rng.normal(0.0, config.retry_scale * search_A)
            start[1] *= np.exp(rng.normal(0.0, config.retry_scale))
            start[2] *= np.exp(rng.normal(0.0, config.retry_scale))
            start = np.clip(start, lower + 1.0e-12, upper - 1.0e-12)
        candidate = least_squares(residuals, start, bounds=(lower, upper))
        if best is None or np.sum(candidate.fun ** 2) < np.sum(best.fun ** 2):
            best = candidate

    center, sigma, amplitude = map(float, best.x[:3])
    model, continuum = evaluate(best.x)
    chi2 = float(np.sum(best.fun ** 2))
    dof = max(1, n_valid - len(best.x))
    chi2_red = chi2 / dof
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * chi2_red
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    area = abs(amplitude) * np.sqrt(2.0 * np.pi) * sigma
    continuum_at_center = float(
        np.interp(center, wave, continuum)
    )
    integrated_line_flux = area * continuum_at_center
    flags = []
    instrumental_fwhm_A = np.nan
    resolution = getattr(segment, "resolution", None)
    if resolution is not None and resolution.mode == "constant":
        if resolution.quantity == "R":
            instrumental_fwhm_A = center / float(resolution.value)
        elif resolution.quantity == "fwhm_kms":
            instrumental_fwhm_A = center * float(resolution.value) / C_KMS
        elif resolution.quantity == "sigma_kms":
            instrumental_fwhm_A = (
                center * 2.3548200450309493 * float(resolution.value) / C_KMS
            )
    elif segment.meta.get("resolution_R") is not None:
        instrumental_fwhm_A = center / float(segment.meta["resolution_R"])
    if not np.isfinite(instrumental_fwhm_A):
        flags.append("lsf_missing")
    if err_estimated:
        flags.append("err_missing_or_estimated")
    if masked_fraction > config.masked_fraction_threshold:
        flags.append("masked_fraction_high")
    if chi2_red > config.high_chi2_threshold:
        flags.append("high_chi2_red")
    tolerance = 1.0e-4
    if min(center - center_lo, center_hi - center) < tolerance * (center_hi - center_lo):
        flags.append("center_at_bound")
    if sigma - sigma_lo < tolerance * (sigma_hi - sigma_lo):
        flags.append("width_at_lower_bound")
    if sigma_hi - sigma < tolerance * (sigma_hi - sigma_lo):
        flags.append("width_at_upper_bound")
    if abs(center - expected) > search_A:
        flags.append("center_outside_search")
    if not flags:
        flags.append("ok")
    rv = C_KMS * (center / rest_wave_data - 1.0)
    rv_err = C_KMS * float(errors[0]) / rest_wave_data
    return LineFitResult(
        line.name, line.rest_wave, kind, bool(best.success), center, float(errors[0]),
        rv, rv_err, amplitude, sigma, 2.3548200450309493 * sigma,
        area if kind == "absorption" else -area,
        integrated_line_flux if kind == "emission" else np.nan,
        chi2, chi2_red, n_valid, masked_fraction, tuple(best.x[3:]), tuple(flags),
        tuple(wave), tuple(flux), tuple(model), tuple(continuum), tuple(flux - model),
        line.wave_medium, segment_medium, rest_wave_data, instrumental_fwhm_A,
    )


def fit_lines(segment, lines, config=None):
    """Fit multiple lines and flag independently fitted close neighbours."""
    specs = [item if isinstance(item, LineSpec) else LineSpec(**item) for item in lines]
    results = [fit_line(segment, item, config=config) for item in specs]
    for i, spec in enumerate(specs):
        close = any(
            j != i and abs(other.rest_wave - spec.rest_wave) < spec.window_A
            for j, other in enumerate(specs)
        )
        if close and "blend_candidate" not in results[i].flags:
            flags = tuple(flag for flag in results[i].flags if flag != "ok") + ("blend_candidate",)
            results[i] = replace(results[i], flags=flags)
    return results
