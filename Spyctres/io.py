# Spyctres/io.py
import os
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
    """

    def __init__(self, wave, flux, err=None, mask=None, meta=None, wave_frame="unknown", name=None):
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
        self.wave_frame = wave_frame
        self.name = name

    def sorted(self):
        idx = np.argsort(self.wave)
        return SpectrumSegment(
            self.wave[idx],
            self.flux[idx],
            None if self.err is None else self.err[idx],
            self.mask[idx],
            meta=self.meta,
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


def read_pepsi_nor(path, ext=1, wave_col="Arg", flux_col="Fun", var_col="Var", wave_frame="unknown"):
    """
    Read PEPSI .nor files (FITS binary table) used in your workflow.
    Expected columns: Arg, Fun, Var (variance).
    """
    path = os.path.abspath(os.path.expanduser(path))
    with fits.open(path, memmap=False) as hdul:
        data = hdul[ext].data
        cols = set([c.name for c in hdul[ext].columns])

        if wave_col not in cols or flux_col not in cols:
            raise ValueError("Missing required columns in {0}: found {1}".format(path, sorted(cols)))

        wave = np.array(data[wave_col], dtype=float)
        flux = np.array(data[flux_col], dtype=float)

        err = None
        if var_col in cols:
            var = np.array(data[var_col], dtype=float)
            # Your script treats Var as variance -> sigma = sqrt(Var). :contentReference[oaicite:1]{index=1}
            err = np.sqrt(var)

        meta = {"path": path, "ext": ext, "columns": sorted(cols)}
        name = os.path.basename(path)
        return SpectrumSegment(wave, flux, err=err, meta=meta, wave_frame=wave_frame, name=name).sorted()


def read_spectrum(path, instrument=None, **kwargs):
    """
    Dispatcher. For now: only PEPSI is implemented.
    """
    inst = (instrument or "").strip().lower()
    if inst in ["pepsi", "pepsi_nor", "pepsi-1d", "pepsi1d"]:
        return read_pepsi_nor(path, **kwargs)

    raise ValueError("Unknown instrument '{0}'. Supported: pepsi".format(instrument))
