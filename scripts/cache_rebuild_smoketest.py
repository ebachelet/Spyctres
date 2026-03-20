import os
import tempfile
import numpy as np

from Spyctres.phoenix import PhoenixLibrary
from Spyctres.fitting import fit_phoenix_full_spectrum
from Spyctres.io import SpectrumSegment
import warnings
# pysynphot is legacy and emits a pkg_resources deprecation warning.
# Suppress it in smoke-test scripts to keep output readable.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pysynphot.*",
)

def make_synthetic_segment(phoenix_dir):
    # Truth close to your existing fitting smoke test
    truth = dict(teff=5050.0, feh=-0.25, logg=4.25, rv=12.3)

    wave = np.linspace(5000.0, 5300.0, 2000)

    lib_truth = PhoenixLibrary(phoenix_dir, verbose=False)
    lib_truth.build_interpolator(
        observed_wave=wave,
        teff_grid=np.array([4800.0, 5000.0, 5200.0, 5400.0]),
        feh_grid=np.array([-0.5, 0.0]),
        logg_grid=np.array([4.0, 4.5]),
        cache_path=None,
        allow_missing=False,
    )

    flux = np.asarray(lib_truth.evaluate(truth["teff"], truth["feh"], truth["logg"]), dtype=float)
    err = np.full_like(flux, 0.02 * np.median(np.abs(flux)))
    rng = np.random.default_rng(12345)
    noisy_flux = flux + rng.normal(0.0, err, size=flux.size)

    seg = SpectrumSegment(
        wave=wave,
        flux=noisy_flux,
        err=err,
        mask=np.isfinite(noisy_flux) & np.isfinite(err) & (err > 0),
        meta={"instrument": "synthetic"},
        wave_medium="vacuum",
        wave_frame="rest",
        name="synthetic_cache_rebuild_test",
    )
    return seg


def run_fit(seg, phoenix_dir, cache_path, teff_grid, label):
    print("\n=== {0} ===".format(label))
    lib = PhoenixLibrary(phoenix_dir, verbose=True)

    result = fit_phoenix_full_spectrum(
        segments=[seg],
        phoenix_lib=lib,
        p0=(5050.0, -0.25, 4.25, 12.0),
        teff_grid=np.asarray(teff_grid, dtype=float),
        feh_grid=np.array([-0.5, 0.0]),
        logg_grid=np.array([4.0, 4.5]),
        cache_path=cache_path,
        rv_init="grid",
        mdeg=2,
        verbose=False,
    )
    
    print(
        "best:",
        {
            "teff": result["teff"],
            "feh": result["feh"],
            "logg": result["logg"],
            "rv": result["rv_kms"],
        },
    )
    print("chi2_red:", result["chi2_red"])


def main():
    phoenix_dir = os.environ["SPYCTRES_PHOENIX_DIR"]
    cache_path = os.path.join(tempfile.gettempdir(), "spyctres_cache_rebuild_test.npz")

    if os.path.exists(cache_path):
        os.remove(cache_path)

    seg = make_synthetic_segment(phoenix_dir)

    # First run writes cache
    run_fit(
        seg,
        phoenix_dir,
        cache_path,
        teff_grid=[4800.0, 5000.0, 5200.0, 5400.0],
        label="FIRST RUN",
    )

    # Second run uses same wavelength grid but a different Teff subgrid.
    # This should trigger "Cache mismatch, rebuilding" and then succeed.
    run_fit(
        seg,
        phoenix_dir,
        cache_path,
        teff_grid=[4700.0, 4900.0, 5100.0, 5300.0],
        label="SECOND RUN WITH DIFFERENT TEFF GRID",
    )


if __name__ == "__main__":
    main()
