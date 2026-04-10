# Spyctres/io.py
import io as _pyio
import os
import re
import numpy as np
from astropy.io import fits


class SpectrumSegment(object):
    """
    Minimal internal spectrum container.

    wave: 1D array (Angstrom by convention unless specified)
    flux: 1D array
    err:  1D array (1-sigma), optional
    mask: boolean array (True = use), optional
    meta: dict, optional
    wave_medium: "air", "vacuum", or "unknown"
    wave_frame: "topocentric", "heliocentric", "barycentric", "stellar_rest", or "unknown"
    """

    def __init__(
        self,
        wave,
        flux,
        err=None,
        mask=None,
        meta=None,
        wave_medium="unknown",
        wave_frame="unknown",
        name=None,
    ):
        self.wave = np.asarray(wave, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

        if self.wave.ndim != 1 or self.flux.ndim != 1:
            raise ValueError("wave and flux must be 1D arrays.")
        if self.wave.shape[0] != self.flux.shape[0]:
            raise ValueError("wave and flux must have the same length.")

        if err is not None:
            self.err = np.asarray(err, dtype=float)
            if self.err.ndim != 1 or self.err.shape[0] != self.wave.shape[0]:
                raise ValueError("err must be 1D and match wave length.")
        else:
            self.err = None

        if mask is None:
            m = np.isfinite(self.wave) & np.isfinite(self.flux)
            if self.err is not None:
                m &= np.isfinite(self.err) & (self.err > 0)
            self.mask = m
        else:
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.ndim != 1 or self.mask.shape[0] != self.wave.shape[0]:
                raise ValueError("mask must be 1D and match wave length.")

        self.meta = {} if meta is None else dict(meta)
        self.wave_medium = str(wave_medium).strip().lower()
        self.wave_frame = str(wave_frame).strip().lower()
        self.name = name

    def copy(
        self,
        wave=None,
        flux=None,
        err=None,
        mask=None,
        meta=None,
        wave_medium=None,
        wave_frame=None,
        name=None,
    ):
        return SpectrumSegment(
            self.wave if wave is None else wave,
            self.flux if flux is None else flux,
            self.err if err is None else err,
            self.mask if mask is None else mask,
            meta=self.meta if meta is None else meta,
            wave_medium=self.wave_medium if wave_medium is None else wave_medium,
            wave_frame=self.wave_frame if wave_frame is None else wave_frame,
            name=self.name if name is None else name,
        )

    def sorted(self):
        idx = np.argsort(self.wave)
        return self.copy(
            wave=self.wave[idx],
            flux=self.flux[idx],
            err=None if self.err is None else self.err[idx],
            mask=self.mask[idx],
        )

    def subset(self, selector, name=None, name_suffix=None):
        """
        Return a subsetted copy of the segment.

        selector may be:
        - a boolean mask with the same length as the segment, or
        - an integer/slice indexer understood by NumPy.
        """
        if isinstance(selector, slice):
            idx = selector
        else:
            selector = np.asarray(selector)
            if selector.dtype == bool:
                if selector.ndim != 1 or selector.shape[0] != self.wave.shape[0]:
                    raise ValueError("Boolean selector must be 1D and match wave length.")
                idx = selector
            else:
                idx = selector

        out_name = self.name if name is None else name
        if name_suffix is not None:
            base = "" if out_name is None else str(out_name)
            out_name = "{0}_{1}".format(base, name_suffix) if base else str(name_suffix)

        return self.copy(
            wave=self.wave[idx],
            flux=self.flux[idx],
            err=None if self.err is None else self.err[idx],
            mask=self.mask[idx],
            meta=dict(self.meta),
            name=out_name,
        )

    def window(self, wmin=None, wmax=None, clip_left=0, clip_right=0, name=None, name_suffix=None):
        """
        Return a wavelength-windowed copy of the segment.

        The wavelength cut is inclusive in [wmin, wmax]. After that, optional
        pixel clipping is applied on the left/right edges of the retained block.
        """
        keep = np.ones_like(self.wave, dtype=bool)
        if wmin is not None:
            keep &= (self.wave >= float(wmin))
        if wmax is not None:
            keep &= (self.wave <= float(wmax))

        idx = np.where(keep)[0]
        if idx.size == 0:
            raise ValueError("No points remain after wavelength windowing.")

        i0 = idx[0]
        i1 = idx[-1] + 1

        out = self.subset(slice(i0, i1), name=name, name_suffix=name_suffix)

        if clip_left > 0 or clip_right > 0:
            n = len(out.wave)
            j0 = int(max(0, clip_left))
            j1 = int(n - max(0, clip_right))
            if j1 <= j0:
                raise ValueError("Edge clipping removed all points.")
            out = out.subset(slice(j0, j1), name=out.name)

        return out

    def with_wave(self, wave, wave_medium=None, wave_frame=None, name=None, name_suffix=None):
        """
        Return a copy with a replaced wavelength array and optionally updated
        wavelength metadata. Flux, err, and mask are preserved.
        """
        out_name = self.name if name is None else name
        if name_suffix is not None:
            base = "" if out_name is None else str(out_name)
            out_name = "{0}_{1}".format(base, name_suffix) if base else str(name_suffix)

        return self.copy(
            wave=np.asarray(wave, dtype=float),
            meta=dict(self.meta),
            wave_medium=self.wave_medium if wave_medium is None else wave_medium,
            wave_frame=self.wave_frame if wave_frame is None else wave_frame,
            name=out_name,
        )
        
        
def concatenate_segments(segments, sort=True, name=None):
    """Concatenate multiple SpectrumSegment objects into one."""
    wave = np.concatenate([s.wave for s in segments])
    flux = np.concatenate([s.flux for s in segments])

    if any(s.err is None for s in segments):
        err = None
    else:
        err = np.concatenate([s.err for s in segments])

    mask = np.concatenate([s.mask for s in segments])

    meta = {"n_segments": len(segments), "segment_names": [s.name for s in segments]}
    out = SpectrumSegment(wave, flux, err=err, mask=mask, meta=meta, name=name)
    return out.sorted() if sort else out


def make_window_segments(seg, windows, pad=0.0, name_prefix=None):
    """
    Build one SpectrumSegment per wavelength window.

    Parameters
    ----------
    seg : SpectrumSegment
        Input segment.
    windows : sequence
        Each entry may be either (wmin, wmax) or (label, wmin, wmax).
    pad : float, optional
        Extra wavelength padding added on both sides of each window.
    name_prefix : str, optional
        Prefix used when a window does not provide its own label.

    Returns
    -------
    list of SpectrumSegment
    """
    out = []
    for i, item in enumerate(windows):
        if len(item) == 2:
            wmin, wmax = item
            label = None
        elif len(item) == 3:
            label, wmin, wmax = item
        else:
            raise ValueError("Each window must be (wmin, wmax) or (label, wmin, wmax).")

        wmin_pad = float(wmin) - float(pad)
        wmax_pad = float(wmax) + float(pad)

        if label is not None:
            name = str(label)
        elif name_prefix is not None:
            name = "{0}_{1}".format(name_prefix, i + 1)
        else:
            name = seg.name

        out.append(seg.window(wmin=wmin_pad, wmax=wmax_pad, name=name))

    return out


def make_padded_window_segments(seg, windows, pad=5.0, name_prefix=None):
    """
    Build one SpectrumSegment per wavelength window with padded support.

    The returned segment covers the support region [wmin-pad, wmax+pad], but
    seg.mask is True only on the inner fit window [wmin, wmax]. This lets the
    fitter evaluate models on a padded support region while computing chi-square
    only on the inner fit pixels.

    Parameters
    ----------
    seg : SpectrumSegment
        Input segment.
    windows : sequence
        Each entry may be either (wmin, wmax) or (label, wmin, wmax).
    pad : float, optional
        Extra wavelength padding added on both sides of each fit window.
    name_prefix : str, optional
        Prefix used when a window does not provide its own label.

    Returns
    -------
    list of SpectrumSegment
    """
    wave = np.asarray(seg.wave, dtype=float)
    flux = np.asarray(seg.flux, dtype=float)
    err = None if seg.err is None else np.asarray(seg.err, dtype=float)
    base_mask = np.asarray(seg.mask, dtype=bool)

    out = []
    for i, item in enumerate(windows):
        if len(item) == 2:
            wmin, wmax = item
            label = None
        elif len(item) == 3:
            label, wmin, wmax = item
        else:
            raise ValueError("Each window must be (wmin, wmax) or (label, wmin, wmax).")

        support_lo = float(wmin) - float(pad)
        support_hi = float(wmax) + float(pad)

        keep = (wave >= support_lo) & (wave <= support_hi)
        if not np.any(keep):
            continue

        fit_mask = base_mask[keep] & (wave[keep] >= float(wmin)) & (wave[keep] <= float(wmax))

        if label is not None:
            name = str(label)
        elif name_prefix is not None:
            base = "" if seg.name is None else str(seg.name)
            name = "{0}_{1}_win{2}".format(base, name_prefix, i) if base else "{0}_win{1}".format(name_prefix, i)
        else:
            base = "" if seg.name is None else str(seg.name)
            name = "{0}_win{1}".format(base, i) if base else "win{0}".format(i)

        out.append(
            SpectrumSegment(
                wave=wave[keep],
                flux=flux[keep],
                err=None if err is None else err[keep],
                mask=fit_mask,
                meta=dict(seg.meta),
                wave_medium=seg.wave_medium,
                wave_frame=seg.wave_frame,
                name=name,
            )
        )

    if len(out) == 0:
        raise ValueError("No points remain after applying padded windows.")

    return out


def _header_get(hdr, key, default=None):
    return hdr[key] if key in hdr else default


def _pepsi_fiber_to_resolution(fiber):
    """
    PEPSI nominal resolving power by science fiber diameter.
    100 um -> 250000
    200 um -> 130000
    300 um -> 50000
    """
    if fiber is None:
        return None

    digits = "".join(ch for ch in str(fiber) if ch.isdigit())
    mapping = {
        "100": 250000.0,
        "200": 130000.0,
        "300": 50000.0,
    }
    return mapping.get(digits, None)


def _normalize_hdu_token(value):
    """
    Normalize FITS extension-like names for robust matching.

    Examples
    --------
    'ORD13_ERRS' -> 'ERRS'
    'ORD16_QUAL' -> 'QUAL'
    'FLUX'       -> 'FLUX'
    """
    if value is None:
        return None

    s = str(value).strip().upper()
    if not s:
        return None

    parts = s.split("_")

    # Common X-SHOOTER merged-1D pattern: ORDxx_ERRS / ORDxx_QUAL
    if len(parts) >= 2 and parts[-1] in ["FLUX", "ERRS", "QUAL"]:
        return parts[-1]

    return s


def _find_hdu_by_name(hdul, extname):
    """
    Return HDU matching EXTNAME, or None if not found.

    Matching is tolerant of logical product names such as ORD13_ERRS
    versus actual EXTNAME='ERRS'.
    """
    target = _normalize_hdu_token(extname)
    if target is None:
        return None

    for hdu in hdul:
        name = _normalize_hdu_token(hdu.header.get("EXTNAME", None))
        if name == target:
            return hdu

    return None


def _resolve_hdu(hdul, ref):
    """
    Resolve an HDU either by integer index or by extension-name-like string.

    For string references, first try EXTNAME matching, then also allow
    matches against SCIDATA / ERRDATA / QUALDATA header pointers after
    normalization.
    """
    if ref is None:
        return None

    if isinstance(ref, (int, np.integer)):
        return hdul[int(ref)]

    target = _normalize_hdu_token(ref)

    hdu = _find_hdu_by_name(hdul, target)
    if hdu is not None:
        return hdu

    for hdu in hdul:
        hdr = hdu.header
        aliases = [
            hdr.get("EXTNAME"),
            hdr.get("SCIDATA"),
            hdr.get("ERRDATA"),
            hdr.get("QUALDATA"),
        ]
        aliases = [_normalize_hdu_token(x) for x in aliases]
        if target in aliases:
            return hdu

    raise ValueError("Could not find FITS extension '{0}'.".format(ref))


def _build_linear_wave_from_header(hdr, n_pix):
    """
    Build a linear wavelength array from CRVAL1/CDELT1/CRPIX1.
    """
    if "CRVAL1" not in hdr or "CDELT1" not in hdr:
        raise ValueError("Header is missing CRVAL1/CDELT1 needed for wavelength solution.")

    crval1 = float(hdr["CRVAL1"])
    cdelt1 = float(hdr["CDELT1"])
    crpix1 = float(hdr.get("CRPIX1", 1.0))

    i = np.arange(int(n_pix), dtype=float)
    return crval1 + (i + 1.0 - crpix1) * cdelt1


def _wave_to_angstrom(wave, unit):
    """
    Convert wavelength array to Angstrom.
    """
    wave = np.asarray(wave, dtype=float)
    u = "" if unit is None else str(unit).strip().lower()

    if u in ["a", "aa", "angstrom", "angstroms", "ang"]:
        return wave

    if u in ["nm", "nanometer", "nanometers"]:
        return wave * 10.0

    if u in ["um", "micron", "microns", "micrometer", "micrometers"]:
        return wave * 1.0e4

    raise ValueError("Unsupported wavelength unit '{0}'.".format(unit))


def _xshooter_slit_keyword_for_arm(arm):
    """
    Map X-SHOOTER arm to the slit keyword used in these headers.
    """
    arm = str(arm).strip().upper()
    mapping = {
        "UVB": "HIERARCH ESO INS OPTI3 NAME",
        "VIS": "HIERARCH ESO INS OPTI4 NAME",
        "NIR": "HIERARCH ESO INS OPTI5 NAME",
    }
    return mapping.get(arm, None)


def _xshooter_parse_slit_width(slit_name):
    """
    Parse slit width in arcsec from strings like '1.0x11', '0.7x11', '0.6x11'.
    Returns float or the string 'IFU', or None.
    """
    if slit_name is None:
        return None

    s = str(slit_name).strip().upper()

    if not s:
        return None

    if "IFU" in s:
        return "IFU"

    # Take the first numeric token, which is the slit width in strings like 1.0x11
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    if m is not None:
        return float(m.group(1))

    return None


def _xshooter_resolution_from_slit(arm, slit_name):
    """
    Nominal X-SHOOTER resolving power as a function of arm and slit width.
    Values follow the X-SHOOTER user manual.
    """
    arm = str(arm).strip().upper()
    slit = _xshooter_parse_slit_width(slit_name)

    table = {
        "UVB": {0.5: 9700.0, 0.8: 6700.0, 1.0: 5400.0, 1.3: 4100.0, 1.6: 3300.0, "IFU": 7900.0},
        "VIS": {0.4: 17400.0, 0.7: 11000.0, 0.9: 8800.0, 1.2: 6700.0, 1.5: 5400.0, "IFU": 12600.0},
        "NIR": {0.4: 11300.0, 0.6: 8100.0, 0.9: 5600.0, 1.2: 4300.0, "IFU": 8100.0},
    }

    if slit is None or arm not in table:
        return None

    if slit == "IFU":
        return table[arm].get("IFU")

    widths = [k for k in table[arm].keys() if k != "IFU"]
    if len(widths) == 0:
        return None

    # Use nearest supported slit width to be robust against header formatting quirks
    nearest = min(widths, key=lambda w: abs(float(w) - float(slit)))

    if abs(float(nearest) - float(slit)) > 0.051:
        return None

    return table[arm][nearest]
    

def _xshooter_telluric_corrected(hdr):
    """
    Heuristic flag for telluric-corrected X-SHOOTER products.
    """
    prodcatg = str(hdr.get("HIERARCH ESO PRO CATG", "")).upper()
    pipefile = str(hdr.get("PIPEFILE", "")).upper()

    return ("TELLURIC" in prodcatg) or ("TELLURIC" in pipefile)


def read_pepsi_nor(
    path,
    ext=1,
    wave_col="Arg",
    flux_col="Fun",
    var_col="Var",
    wave_medium="unknown",
    wave_frame="unknown",
    infer_resolution=True,
):
    """
    Read PEPSI .nor files (FITS binary table).

    Expected columns: Arg, Fun, Var (variance).
    No wavelength-medium or reference-frame conversion is applied here.
    Those are carried as metadata for later handling in the fitter or utilities.
    """
    path = os.path.abspath(os.path.expanduser(path))
    with fits.open(path, memmap=False) as hdul:
        data = hdul[ext].data
        cols = set(c.name for c in hdul[ext].columns)

        if wave_col not in cols or flux_col not in cols:
            raise ValueError(
                "Missing required columns in {0}: found {1}".format(path, sorted(cols))
            )

        wave = np.array(data[wave_col], dtype=float)
        flux = np.array(data[flux_col], dtype=float)

        err = None
        if var_col in cols:
            var = np.array(data[var_col], dtype=float)
            err = np.sqrt(var)

        phdr = hdul[0].header if len(hdul) > 0 else fits.Header()
        ehdr = hdul[ext].header

        def get_any(key, default=None):
            if key in ehdr:
                return ehdr[key]
            if key in phdr:
                return phdr[key]
            return default

        meta = {
            "path": path,
            "ext": ext,
            "columns": sorted(cols),
            "instrument": get_any("INSTRUME"),
            "object": get_any("OBJECT"),
            "arm": get_any("ARM"),
            "fiber": get_any("FIBER"),
            "cross_disperser": get_any("CROSDIS"),
            "date_obs": get_any("DATE-OBS"),
            "time_obs": get_any("TIME-OBS"),
            "exptime": get_any("EXPTIME"),
            "jd_obs": get_any("JD-OBS"),
            "jd_tdb": get_any("JD-TDB"),
            "ra": get_any("RA"),
            "dec": get_any("DEC"),
            "ra2000": get_any("RA2000"),
            "dec2000": get_any("DE2000"),
            "ssbvel_mps": get_any("SSBVEL"),
            "charave_mps": get_any("CHARAVE"),
            "was_file": get_any("WAS"),
            "trace_file": get_any("TRACE"),
            "wasrep": get_any("WASREP"),
            "wasfwhm_pix": get_any("WASFWHM"),
            "wave_medium": wave_medium,
            "wave_frame": wave_frame,
        }

        resolution_R = None
        if infer_resolution:
            wasrep = meta.get("wasrep")
            if wasrep is not None:
                try:
                    resolution_R = float(wasrep)
                except Exception:
                    resolution_R = None

            if resolution_R is None:
                resolution_R = _pepsi_fiber_to_resolution(meta.get("fiber"))

        meta["resolution_R"] = resolution_R

        name = os.path.basename(path)
        return SpectrumSegment(
            wave,
            flux,
            err=err,
            meta=meta,
            wave_medium=wave_medium,
            wave_frame=wave_frame,
            name=name,
        ).sorted()


def read_xshooter_1d(
    path,
    flux_ext=0,
    err_ext=None,
    qual_ext=None,
    wave_unit=None,
    wave_medium="air",
    wave_frame="topocentric",
    infer_resolution=True,
):
    """
    Read a merged 1D X-SHOOTER spectrum stored as a linear FITS image product.

    Assumptions for the current implementation:
    - flux is in the primary HDU by default
    - wavelength is reconstructed from CRVAL1/CDELT1/CRPIX1
    - error and quality HDUs are resolved from ERRDATA / QUALDATA when present
    - no air/vacuum or barycentric correction is applied here; those are carried as metadata
    """
    path = os.path.abspath(os.path.expanduser(path))

    with fits.open(path, memmap=False) as hdul:
        flux_hdu = _resolve_hdu(hdul, flux_ext)
        phdr = flux_hdu.header
        flux = np.asarray(flux_hdu.data, dtype=float)

        if flux.ndim != 1:
            raise ValueError("X-SHOOTER reader expects a 1D flux array in the selected HDU.")

        err_ref = err_ext if err_ext is not None else phdr.get("ERRDATA", 1)
        qual_ref = qual_ext if qual_ext is not None else phdr.get("QUALDATA", 2)

        err_hdu = _resolve_hdu(hdul, err_ref)
        qual_hdu = _resolve_hdu(hdul, qual_ref)

        err = None if err_hdu is None else np.asarray(err_hdu.data, dtype=float)
        qual = None if qual_hdu is None else np.asarray(qual_hdu.data)

        if err is not None and err.shape != flux.shape:
            raise ValueError("Error array shape does not match flux array shape.")
        if qual is not None and qual.shape != flux.shape:
            raise ValueError("Quality array shape does not match flux array shape.")

        raw_wave = _build_linear_wave_from_header(phdr, flux.size)

        cunit1 = phdr.get("CUNIT1", None)
        unit_in = wave_unit if wave_unit is not None else cunit1
        if unit_in is None:
            unit_in = "nm"

        wave = _wave_to_angstrom(raw_wave, unit_in)

        arm = str(phdr.get("HIERARCH ESO SEQ ARM", phdr.get("ARM", "unknown"))).strip().upper()
        slit_key = _xshooter_slit_keyword_for_arm(arm)
        slit_name = phdr.get(slit_key, None) if slit_key is not None else None

        resolution_R = None
        if infer_resolution:
            resolution_R = _xshooter_resolution_from_slit(arm, slit_name)

        mask = np.isfinite(wave) & np.isfinite(flux)

        if err is not None:
            mask &= np.isfinite(err) & (err > 0)

        if qual is not None:
            mask &= (np.asarray(qual) == 0)

        barycorr_kms = phdr.get("HIERARCH ESO QC VRAD BARYCOR", None)
        helicorr_kms = phdr.get("HIERARCH ESO QC VRAD HELICOR", None)

        meta = {
            "path": path,
            "instrument": phdr.get("INSTRUME", "XSHOOTER"),
            "object": phdr.get("OBJECT"),
            "arm": arm,
            "mode": phdr.get("HIERARCH ESO INS MODE"),
            "slit_keyword": slit_key,
            "slit_name": slit_name,
            "slit_width_arcsec": _xshooter_parse_slit_width(slit_name),
            "resolution_R": resolution_R,
            "date_obs": phdr.get("DATE-OBS"),
            "mjd_obs": phdr.get("MJD-OBS"),
            "exptime": phdr.get("EXPTIME"),
            "ra": phdr.get("RA"),
            "dec": phdr.get("DEC"),
            "bunit": phdr.get("BUNIT"),
            "cunit1": cunit1,
            "wave_unit_input": unit_in,
            "prodcatg": phdr.get("HIERARCH ESO PRO CATG"),
            "pipefile": phdr.get("PIPEFILE"),
            "err_ext": err_ref,
            "qual_ext": qual_ref,
            "barycorr_kms": barycorr_kms,
            "helicorr_kms": helicorr_kms,
            "telluric_corrected": _xshooter_telluric_corrected(phdr),
            "wave_medium": wave_medium,
            "wave_frame": wave_frame,
        }

        name = os.path.basename(path)

        return SpectrumSegment(
            wave,
            flux,
            err=err,
            mask=mask,
            meta=meta,
            wave_medium=wave_medium,
            wave_frame=wave_frame,
            name=name,
        ).sorted()

def read_floyds_csv(path, name=None):
    """
    Read a reduced 1D FLOYDS spectrum from a simple ASCII/CSV export.

    Expected format:
    - optional comment lines beginning with '#'
    - one header line with column names such as 'wavelength flux'
    - numeric rows thereafter

    Returns
    -------
    SpectrumSegment
    """
    path = os.path.abspath(os.path.expanduser(path))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    comments = {}
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        body = line.lstrip()[1:].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            comments[k.strip().lower()] = v.strip()

    data_text = "".join(
        ln for ln in lines
        if ln.strip() and not ln.lstrip().startswith("#")
    )
    if not data_text.strip():
        raise ValueError("No tabular data found in FLOYDS file: {0}".format(path))

    arr = np.genfromtxt(
        _pyio.StringIO(data_text),
        names=True,
        dtype=float,
        encoding=None,
    )

    if arr.dtype.names is None:
        raise ValueError(
            "Could not parse named columns from FLOYDS file: {0}".format(path)
        )

    names = list(arr.dtype.names)
    names_l = [n.lower() for n in names]

    wave_candidates = ["wavelength", "wave", "lambda", "lam"]
    flux_candidates = ["flux", "f_lambda", "flam", "fnu"]
    err_candidates = ["err", "error", "uncertainty", "sigma", "flux_err", "fluxerror"]

    def _pick(candidates):
        for c in candidates:
            if c in names_l:
                return names[names_l.index(c)]
        return None

    wave_col = _pick(wave_candidates)
    flux_col = _pick(flux_candidates)
    err_col = _pick(err_candidates)

    if wave_col is None or flux_col is None:
        raise ValueError(
            "Need wavelength and flux columns in FLOYDS file: {0}. "
            "Found columns: {1}".format(path, names)
        )

    wave = np.asarray(arr[wave_col], dtype=float)
    flux = np.asarray(arr[flux_col], dtype=float)
    err = None if err_col is None else np.asarray(arr[err_col], dtype=float)

    mask = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)

    meta = {
        "path": path,
        "instrument": "FLOYDS",
        "facility": comments.get("facility"),
        "date_obs": comments.get("date-obs"),
        "resolution_R": 500.0,
        "resolution_note": "Approximate nominal FLOYDS merged-spectrum value; actual R varies with wavelength and slit.",
        "wave_medium": "unknown",
        "wave_frame": "unknown",
    }

    seg_name = (
        name
        or comments.get("object")
        or comments.get("target")
        or os.path.basename(path)
    )

    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium="unknown",
        wave_frame="unknown",
        name=seg_name,
    ).sorted()
    

def read_gemini_gmos_ascii(path, name=None):
    """
    Read a reduced 1D Gemini/GMOS spectrum from an IRAF wspectext-like ASCII export.

    Expected format:
    - optional FITS-like header cards at the top
    - optional END line terminating the header
    - numeric rows thereafter, typically wavelength flux
    - an optional third numeric column is treated as err

    Returns
    -------
    SpectrumSegment
    """
    path = os.path.abspath(os.path.expanduser(path))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header = {}
    data_lines = []
    in_header = True

    for line in lines:
        s = line.strip()

        if not s:
            continue

        if in_header:
            if s == "END":
                in_header = False
                continue

            # FITS-like header card, e.g. KEYWORD = value / comment
            if "=" in line and not re.match(r"^[+-]?[0-9]", s):
                key, rest = line.split("=", 1)
                key = key.strip()
                value = rest.split("/", 1)[0].strip()

                if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                    value = value[1:-1].strip()

                header[key] = value
                continue

            # No explicit END: if the line begins numerically, data start here.
            if re.match(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?", s):
                in_header = False
                data_lines.append(line)
                continue

            # Otherwise ignore stray non-data lines in the header block.
            continue

        data_lines.append(line)

    if len(data_lines) == 0:
        raise ValueError("No numeric spectral data found in Gemini/GMOS ASCII file: {0}".format(path))

    arr = np.genfromtxt(_pyio.StringIO("".join(data_lines)), dtype=float)

    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("Need at least two numeric columns (wave, flux) in: {0}".format(path))
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            "Could not parse Gemini/GMOS ASCII spectrum with at least two columns: {0}".format(path)
        )

    wave = np.asarray(arr[:, 0], dtype=float)
    flux = np.asarray(arr[:, 1], dtype=float)
    err = None
    if arr.shape[1] >= 3:
        err = np.asarray(arr[:, 2], dtype=float)

    mask = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)

    meta = {
        "path": path,
        "instrument": "GMOS",
        "facility": "Gemini",
        "object": header.get("OBJECT"),
        "filename": header.get("FILENAME"),
        "origin": header.get("ORIGIN"),
        "iraf_type": header.get("IRAFTYPE"),
        "wave_unit_input": "angstrom",
        "resolution_R": None,
        "wave_medium": "unknown",
        "wave_frame": "unknown",
        "header_cards": dict(header),
    }

    seg_name = (
        name
        or header.get("OBJECT")
        or header.get("FILENAME")
        or os.path.basename(path)
    )

    return SpectrumSegment(
        wave=wave,
        flux=flux,
        err=err,
        mask=mask,
        meta=meta,
        wave_medium="unknown",
        wave_frame="unknown",
        name=seg_name,
    ).sorted()
    

READERS = {}


def register_reader(names, func):
    """
    Register one reader function under one or more instrument aliases.
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        key = str(name).strip().lower()
        if not key:
            continue
        READERS[key] = func


register_reader(["pepsi", "pepsi_nor", "pepsi-1d", "pepsi1d"], read_pepsi_nor)
register_reader(["xshooter", "x-shooter", "xsh", "xshooter_1d", "xshooter-1d"], read_xshooter_1d)
register_reader(["floyds", "floyds_csv", "lco_floyds"], read_floyds_csv)
register_reader(["gemini", "gmos", "gemini_gmos", "gmos_ascii", "gemini_ascii"], read_gemini_gmos_ascii)

  
def read_spectrum(path, instrument=None, **kwargs):
    """
    Dispatcher for supported 1D spectrum readers.
    """
    inst = (instrument or "").strip().lower()
    func = READERS.get(inst, None)

    if func is None:
        raise ValueError(
            "Unknown instrument '{0}'. Supported: pepsi, xshooter, floyds, gemini".format(instrument)
        )

    return func(path, **kwargs)
