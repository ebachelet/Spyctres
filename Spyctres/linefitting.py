"""Small, LSF-aware local spectral-line diagnostic layer."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json

import numpy as np
from scipy.optimize import least_squares

from .io import SpectrumSegment
from .waveutils import C_KMS, convert_wavelength_medium


_LINE_ALIASES = {
    "halpha": ("Halpha", 6562.80, "air", 22.0),
    "h_alpha": ("Halpha", 6562.80, "air", 22.0),
    "hα": ("Halpha", 6562.80, "air", 22.0),
    "hbeta": ("Hbeta", 4861.33, "air", 18.0),
    "h_beta": ("Hbeta", 4861.33, "air", 18.0),
    "hβ": ("Hbeta", 4861.33, "air", 18.0),
    "hgamma": ("Hgamma", 4340.47, "air", 16.0),
    "h_gamma": ("Hgamma", 4340.47, "air", 16.0),
    "hγ": ("Hgamma", 4340.47, "air", 16.0),
    "hdelta": ("Hdelta", 4101.74, "air", 14.0),
    "h_delta": ("Hdelta", 4101.74, "air", 14.0),
    "hδ": ("Hdelta", 4101.74, "air", 14.0),
    "caiik": ("Ca II K", 3933.66, "air", 8.0),
    "ca_ii_k": ("Ca II K", 3933.66, "air", 8.0),
    "ca ii k": ("Ca II K", 3933.66, "air", 8.0),
    "caiih": ("Ca II H", 3968.47, "air", 8.0),
    "ca_ii_h": ("Ca II H", 3968.47, "air", 8.0),
    "ca ii h": ("Ca II H", 3968.47, "air", 8.0),
    "hei4471": ("He I 4471", 4471.48, "air", 8.0),
    "he_i_4471": ("He I 4471", 4471.48, "air", 8.0),
    "he i 4471": ("He I 4471", 4471.48, "air", 8.0),
    "mgii4481": ("Mg II 4481", 4481.13, "air", 8.0),
    "mg_ii_4481": ("Mg II 4481", 4481.13, "air", 8.0),
    "mg ii 4481": ("Mg II 4481", 4481.13, "air", 8.0),
}


def _normalize_line_alias(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _known_line_records():
    records = {}
    for alias, values in _LINE_ALIASES.items():
        label, rest_wave, medium, window_A = values
        record = records.setdefault(
            label,
            {
                "name": label,
                "rest_wave_A": float(rest_wave),
                "wave_medium": medium,
                "default_window_A": float(window_A),
                "kind": "absorption",
                "aliases": set(),
            },
        )
        record["aliases"].add(str(alias))
    out = []
    for record in records.values():
        item = dict(record)
        item["aliases"] = sorted(item["aliases"])
        out.append(item)
    return sorted(out, key=lambda item: (item["rest_wave_A"], item["name"]))


def list_known_lines(*, details=False, include_aliases=False, wmin=None, wmax=None):
    """List built-in local-line names known by :func:`fit_line`.

    By default this returns the compact canonical names accepted by
    ``fit_line(spec, "name")``.  Use ``details=True`` to inspect the rest
    wavelength, catalog wavelength medium, default fitting half-window, and
    accepted aliases.

    Parameters
    ----------
    details : bool, optional
        Return dictionaries with line metadata instead of just names.
    include_aliases : bool, optional
        When ``details=False``, return all accepted alias strings instead of
        canonical display names.  When ``details=True``, aliases are always
        included in each record.
    wmin, wmax : float, optional
        Optional wavelength filter in Angstrom, evaluated against the catalog
        rest wavelength in its listed wavelength medium.
    """
    if wmin is not None:
        wmin = float(wmin)
        if not np.isfinite(wmin):
            raise ValueError("wmin must be finite when supplied.")
    if wmax is not None:
        wmax = float(wmax)
        if not np.isfinite(wmax):
            raise ValueError("wmax must be finite when supplied.")
    if wmin is not None and wmax is not None and wmax < wmin:
        raise ValueError("wmax must be >= wmin.")

    records = []
    for record in _known_line_records():
        rest = float(record["rest_wave_A"])
        if wmin is not None and rest < wmin:
            continue
        if wmax is not None and rest > wmax:
            continue
        records.append(record)

    if details:
        return [dict(record) for record in records]
    if include_aliases:
        aliases = []
        for record in records:
            aliases.extend(record["aliases"])
        return sorted(set(aliases))
    return [record["name"] for record in records]


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

    def summary(self):
        """Return a compact JSON-safe line-fit summary."""
        return {
            "line_name": self.line_name,
            "success": bool(self.success),
            "rest_wave_A": float(self.rest_wave),
            "center_wave_A": float(self.center_wave),
            "rv_kms": float(self.rv_kms),
            "rv_err_kms": float(self.rv_err_kms),
            "equivalent_width_A": float(self.equivalent_width_A),
            "fwhm_A": float(self.fwhm_A),
            "chi2_red": float(self.chi2_red),
            "n_points": int(self.n_points),
            "mask_fraction": float(self.mask_fraction),
            "flags": list(self.flags),
            "interpretation": (
                "local diagnostic line fit; broad lines are not precision-RV anchors"
            ),
        }

    def summary_text(self):
        """Return a compact plain-text summary for notebooks/scripts."""
        flags = ", ".join(self.flags) or "none"
        return (
            "{0}: success={1}, RV={2:.3g} km/s, EW={3:.3g} A, "
            "chi2_red={4:.3g}, N={5}, flags={6}"
        ).format(
            self.line_name,
            bool(self.success),
            float(self.rv_kms),
            float(self.equivalent_width_A),
            float(self.chi2_red),
            int(self.n_points),
            flags,
        )


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


def known_line_spec(name, **overrides):
    """Return a :class:`LineSpec` for a small built-in line alias.

    The alias catalog is intentionally modest and aimed at beginner diagnostics:
    Balmer Hα/Hβ/Hγ/Hδ, Ca II H/K, He I 4471, and Mg II 4481.  Expert users can
    pass a full :class:`LineSpec` or ``center=...`` to :func:`fit_line`.
    """
    key = _normalize_line_alias(name)
    if key not in _LINE_ALIASES:
        known = ", ".join(sorted(_LINE_ALIASES))
        raise ValueError(
            "Unknown line name {0!r}. Pass center=... for a custom line or use "
            "one of: {1}.".format(name, known)
        )
    label, rest_wave, medium, window_A = _LINE_ALIASES[key]
    values = {
        "name": label,
        "rest_wave": float(rest_wave),
        "kind": "absorption",
        "window_A": float(window_A),
        "wave_medium": medium,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return LineSpec(**values)


def _line_spec_from_input(
    line=None,
    *,
    center=None,
    rest_wave=None,
    name=None,
    kind=None,
    window_A=None,
    wave_medium=None,
):
    if isinstance(line, LineSpec):
        spec = line
        updates = {}
        if kind is not None:
            updates["kind"] = str(kind)
        if window_A is not None:
            updates["window_A"] = float(window_A)
        if wave_medium is not None:
            updates["wave_medium"] = str(wave_medium)
        if updates:
            spec = replace(spec, **updates)
    elif isinstance(line, str):
        spec = known_line_spec(
            line,
            kind=kind,
            window_A=window_A,
            wave_medium=wave_medium,
        )
    elif isinstance(line, dict):
        values = dict(line)
        if kind is not None:
            values["kind"] = kind
        if window_A is not None:
            values["window_A"] = window_A
        if wave_medium is not None:
            values["wave_medium"] = wave_medium
        spec = LineSpec(**values)
    elif line is None:
        custom_center = center if center is not None else rest_wave
        if custom_center is None:
            raise ValueError(
                "fit_line requires a LineSpec, a known line name such as "
                "'Hgamma', or center=... for a custom wavelength."
            )
        spec = LineSpec(
            name=str(name or "{0:.3f} A".format(float(custom_center))),
            rest_wave=float(custom_center),
            kind="absorption" if kind is None else str(kind),
            window_A=10.0 if window_A is None else float(window_A),
            wave_medium="unknown" if wave_medium is None else str(wave_medium),
        )
    elif np.isscalar(line):
        spec = LineSpec(
            name=str(name or "{0:.3f} A".format(float(line))),
            rest_wave=float(line),
            kind="absorption" if kind is None else str(kind),
            window_A=10.0 if window_A is None else float(window_A),
            wave_medium="unknown" if wave_medium is None else str(wave_medium),
        )
    else:
        spec = LineSpec(*line)
        updates = {}
        if kind is not None:
            updates["kind"] = str(kind)
        if window_A is not None:
            updates["window_A"] = float(window_A)
        if wave_medium is not None:
            updates["wave_medium"] = str(wave_medium)
        if updates:
            spec = replace(spec, **updates)
    return spec


def _line_center_for_segment(line, segment, config):
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
    return rest_wave_data * (1.0 + float(config.rv_guess_kms) / C_KMS)


def _choose_line_segment(spectrum, line, config):
    if isinstance(spectrum, SpectrumSegment):
        return spectrum
    if hasattr(spectrum, "segments"):
        segments = list(spectrum.segments)
    elif isinstance(spectrum, (list, tuple)) and spectrum and hasattr(spectrum[0], "wave"):
        segments = list(spectrum)
    else:
        raise TypeError("segment must be a SpectrumSegment or SpectrumCollection.")
    for segment in segments:
        wave = np.asarray(segment.wave, dtype=float)
        good = np.isfinite(wave)
        if not np.any(good):
            continue
        expected = _line_center_for_segment(line, segment, config)
        margin = float(line.window_A)
        if np.nanmin(wave[good]) - margin <= expected <= np.nanmax(wave[good]) + margin:
            return segment
    raise ValueError(
        "No spectrum segment covers line {0!r} near {1:.3f} A.".format(
            line.name,
            float(line.rest_wave),
        )
    )


def _coerce_line_valid_mask(spectrum, segment, valid_mask):
    if valid_mask is None:
        return None
    mask = None
    if isinstance(spectrum, SpectrumSegment):
        mask = valid_mask
    elif hasattr(spectrum, "segments"):
        segments = list(spectrum.segments)
        index = segments.index(segment)
        if isinstance(valid_mask, Mapping):
            key = segment.name if segment.name in valid_mask else index
            if key not in valid_mask:
                raise ValueError(
                    "valid_mask mapping must contain the selected segment name or index."
                )
            mask = valid_mask[key]
        else:
            array = np.asarray(valid_mask)
            if array.shape == np.asarray(segment.wave).shape:
                mask = array
            else:
                masks = list(valid_mask)
                if len(masks) != len(segments):
                    raise ValueError(
                        "valid_mask for a SpectrumCollection must contain one "
                        "mask per segment, or one mask matching the selected segment."
                    )
                mask = masks[index]
    else:
        mask = valid_mask
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.asarray(segment.wave).shape:
        raise ValueError("valid_mask must match the selected segment wavelength shape.")
    return mask


def fit_line(
    segment,
    line=None,
    config=None,
    *,
    center=None,
    rest_wave=None,
    name=None,
    kind=None,
    window_A=None,
    wave_medium=None,
    valid_mask=None,
):
    """Fit a Gaussian local line plus multiplicative Legendre continuum.

    Beginner calls may use a built-in line name or a custom wavelength:

    ``fit_line(spec, "Hgamma")`` or ``fit_line(spec, center=4340.47)``.

    Expert calls using :class:`LineSpec`, dictionaries, or tuple-style line
    specifications remain supported.
    """
    config = LineFitConfig() if config is None else config
    line = _line_spec_from_input(
        line,
        center=center,
        rest_wave=rest_wave,
        name=name,
        kind=kind,
        window_A=window_A,
        wave_medium=wave_medium,
    )
    input_spectrum = segment
    segment = _choose_line_segment(segment, line, config)
    user_valid_mask = _coerce_line_valid_mask(input_spectrum, segment, valid_mask)
    if user_valid_mask is not None:
        segment = segment.copy(mask=np.asarray(segment.mask, dtype=bool) & user_valid_mask)

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


def fit_lines(segment, lines, config=None, *, valid_mask=None):
    """Fit multiple lines and flag independently fitted close neighbours."""
    specs = [_line_spec_from_input(item) for item in lines]
    results = [
        fit_line(segment, item, config=config, valid_mask=valid_mask)
        for item in specs
    ]
    for i, spec in enumerate(specs):
        close = any(
            j != i and abs(other.rest_wave - spec.rest_wave) < spec.window_A
            for j, other in enumerate(specs)
        )
        if close and "blend_candidate" not in results[i].flags:
            flags = tuple(flag for flag in results[i].flags if flag != "ok") + ("blend_candidate",)
            results[i] = replace(results[i], flags=flags)
    return results


class LineFitComparison(dict):
    """Dict-compatible line-fit comparison with display helpers."""

    @property
    def rows(self):
        return list(self.get("rows", ()))

    def to_dict(self):
        return dict(self)

    def summary(self):
        return {
            "schema_version": 1,
            "n_results": len(self.rows),
            "n_success": int(sum(1 for row in self.rows if row.get("success"))),
            "rv_median_kms": self.get("rv_median_kms"),
            "rv_scatter_kms": self.get("rv_scatter_kms"),
            "rows": self.rows,
        }

    def plot(self, **kwargs):
        """Plot compact diagnostic metrics for this line-fit comparison."""
        from .plotting import plot_line_fit_comparison

        return plot_line_fit_comparison(self, **kwargs)

    def summary_text(self):
        header = (
            "label                 line           RV[km/s]   EW[A]  "
            "chi2red  Nfit  flags"
        )
        lines = [
            "Spyctres line-fit comparison",
            "  successful={0}/{1}, median RV={2}".format(
                self.summary()["n_success"],
                len(self.rows),
                "n/a"
                if self.get("rv_median_kms") is None
                else "{0:.3g} km/s".format(float(self["rv_median_kms"])),
            ),
            header,
            "-" * len(header),
        ]
        for row in self.rows:
            rv = row.get("rv_kms")
            ew = row.get("equivalent_width_A")
            chi2 = row.get("chi2_red")
            lines.append(
                "{label:<21} {line:<12} {rv:>8} {ew:>7} {chi2:>7} {n:>5}  {flags}".format(
                    label=str(row.get("label", ""))[:21],
                    line=str(row.get("line_name", ""))[:12],
                    rv="nan" if rv is None or not np.isfinite(rv) else "{0:.3g}".format(rv),
                    ew="nan" if ew is None or not np.isfinite(ew) else "{0:.3g}".format(ew),
                    chi2=(
                        "nan"
                        if chi2 is None or not np.isfinite(chi2)
                        else "{0:.3g}".format(chi2)
                    ),
                    n=int(row.get("n_points") or 0),
                    flags=",".join(row.get("flags") or []) or "none",
                )
            )
        return "\n".join(lines)


def compare_line_fits(results, labels=None):
    """Return a compact comparison of local line-fit results."""
    results = list(results)
    if labels is None:
        labels = [getattr(result, "line_name", "line {0}".format(index + 1)) for index, result in enumerate(results)]
    labels = list(labels)
    if len(labels) != len(results):
        raise ValueError("labels length must match line-fit results length.")
    rows = []
    rvs = []
    for label, result in zip(labels, results):
        row = result.summary() if hasattr(result, "summary") else dict(result)
        row["label"] = str(label)
        rows.append(row)
        rv = row.get("rv_kms")
        if rv is not None and np.isfinite(rv) and row.get("success"):
            rvs.append(float(rv))
    if rvs:
        rv_median = float(np.nanmedian(rvs))
        rv_scatter = float(1.4826 * np.nanmedian(np.abs(np.asarray(rvs) - rv_median)))
    else:
        rv_median = None
        rv_scatter = None
    return LineFitComparison(
        {
            "schema_version": 1,
            "operation": "compare_line_fits",
            "rows": rows,
            "rv_median_kms": rv_median,
            "rv_scatter_kms": rv_scatter,
            "interpretation": (
                "Line agreement is a diagnostic. Broad Balmer-line centers "
                "should not be treated as precision radial velocities."
            ),
        }
    )
