# Spyctres/io.py
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
        self.wave_medium = str(wave_medium)
        self.wave_frame = str(wave_frame)
        self.name = name

    def sorted(self):
        idx = np.argsort(self.wave)
        return SpectrumSegment(
            self.wave[idx],
            self.flux[idx],
            None if self.err is None else self.err[idx],
            self.mask[idx],
            meta=self.meta,
            wave_medium=self.wave_medium,
            wave_frame=self.wave_frame,
            name=self.name,
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

def read_spectrum(path, instrument=None, **kwargs):
    """
    Dispatcher for supported 1D spectrum readers.
    """
    inst = (instrument or "").strip().lower()

    if inst in ["pepsi", "pepsi_nor", "pepsi-1d", "pepsi1d"]:
        return read_pepsi_nor(path, **kwargs)

    if inst in ["xshooter", "x-shooter", "xsh", "xshooter_1d", "xshooter-1d"]:
        return read_xshooter_1d(path, **kwargs)

    raise ValueError(
        "Unknown instrument '{0}'. Supported: pepsi, xshooter".format(instrument)
    )
