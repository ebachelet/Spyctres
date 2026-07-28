"""Read-only setup diagnostics for Spyctres installations.

The checker is deliberately lightweight: it verifies imports, optional legacy
dependencies, PHOENIX discovery, and optional example-spectrum ingestion without
building PHOENIX caches or running fits.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import resources as importlib_resources
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import warnings


REQUIRED_MODULES = (
    "numpy",
    "scipy",
    "matplotlib",
    "astropy",
)

OPTIONAL_MODULES = (
    "pysynphot",
    "synphot",
    "stsynphot",
)

PACKAGE_DATA_FILES = (
    "LBL_A10_s0_w050_R0300000_T.fits",
    "Reader_Corliss_Lines.fits",
    "G_GAIA_responses.dat",
    "SLOAN_SDSS.gprime_filter.dat",
)

SOURCE_EXAMPLE_FILES = (
    Path("examples")
    / "data"
    / "gaia_benchmark"
    / "HIP79672_HARPS_1_R42KNorm.txt.gz",
    Path("examples")
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
)


if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "spyctres_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)


def _status(ok, label, detail=""):
    prefix = "OK" if ok else "FAIL"
    text = "[{0}] {1}".format(prefix, label)
    if detail:
        text += " — {0}".format(detail)
    print(text, flush=True)


def _warn(label, detail=""):
    text = "[WARN] {0}".format(label)
    if detail:
        text += " — {0}".format(detail)
    print(text, flush=True)


def _module_version(module):
    return str(getattr(module, "__version__", "unknown"))


def _import_for_check(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module(name)


def check_python():
    ok = sys.version_info >= (3, 12)
    _status(
        ok,
        "Python version",
        "{0}.{1}.{2}".format(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
    )
    return ok


def check_imports():
    ok = True
    for name in REQUIRED_MODULES:
        try:
            module = _import_for_check(name)
        except ImportError as exc:
            _status(False, "Required dependency {0}".format(name), str(exc))
            ok = False
            continue
        _status(True, "Required dependency {0}".format(name), _module_version(module))

    for name in OPTIONAL_MODULES:
        try:
            module = _import_for_check(name)
        except ImportError:
            _warn(
                "Optional dependency {0}".format(name),
                "missing; only legacy/synthetic-photometry workflows need it",
            )
            continue
        _status(True, "Optional dependency {0}".format(name), _module_version(module))

    try:
        import Spyctres
    except ImportError as exc:
        _status(False, "Spyctres import", str(exc))
        return False
    _status(True, "Spyctres import", getattr(Spyctres, "__file__", "unknown"))
    return ok


def check_cli_entrypoint():
    """Warn if the optional console script has not been installed yet."""
    executable = shutil.which("spyctres")
    if executable is not None:
        _status(True, "Spyctres CLI entry point", executable)
        return True

    _warn(
        "Spyctres CLI entry point",
        (
            "not found on PATH; run 'pip install -e .' again after updating "
            "the checkout, or use 'python -m Spyctres.cli ...'"
        ),
    )
    return True


def check_package_data():
    """Check package resources that should survive wheel/sdist installation."""
    try:
        data_root = importlib_resources.files("Spyctres.data")
    except (ModuleNotFoundError, AttributeError, TypeError) as exc:
        _status(False, "Bundled package data", str(exc))
        return False

    missing = [name for name in PACKAGE_DATA_FILES if not (data_root / name).is_file()]
    if missing:
        _status(False, "Bundled package data", "missing: {0}".format(", ".join(missing)))
        return False
    _status(True, "Bundled package data", "{0} checked".format(len(PACKAGE_DATA_FILES)))

    try:
        import Spyctres
    except ImportError:
        return True
    source_root = Path(getattr(Spyctres, "__file__", "")).resolve().parents[1]
    present = [str(path) for path in SOURCE_EXAMPLE_FILES if (source_root / path).is_file()]
    if present:
        _status(True, "Source-checkout examples", "{0} checked".format(len(present)))
    else:
        _warn(
            "Source-checkout examples",
            (
                "not found next to the installed package; this is normal for a "
                "wheel install. Clone the repository to run the numbered examples."
            ),
        )
    return True


def check_phoenix(args):
    if args.skip_phoenix:
        _warn("PHOENIX checks skipped", "--skip-phoenix was supplied")
        return True

    from Spyctres.config import default_config_path, resolve_phoenix_dir
    from Spyctres.phoenix import PhoenixLibrary

    try:
        phoenix_dir = resolve_phoenix_dir(args.phoenix_dir, require_exists=True)
    except FileNotFoundError as exc:
        _status(False, "PHOENIX directory", str(exc))
        return False

    if phoenix_dir is None:
        detail = (
            "pass --phoenix-dir, set SPYCTRES_PHOENIX_DIR, or configure {0}".format(
                default_config_path()
            )
        )
        if args.require_phoenix:
            _status(False, "PHOENIX directory not configured", detail)
            return False
        _warn("PHOENIX directory not configured", detail)
        return True

    _status(True, "PHOENIX directory", phoenix_dir)
    wave_path = Path(phoenix_dir) / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    if not wave_path.is_file():
        _status(False, "PHOENIX wavelength file", str(wave_path))
        return False
    _status(True, "PHOENIX wavelength file", str(wave_path))

    try:
        library = PhoenixLibrary(phoenix_dir, verbose=False)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _status(False, "PHOENIX library initialization", str(exc))
        return False
    _status(
        True,
        "PHOENIX wavelength grid",
        "{0} samples, {1:.3f}–{2:.3f} A".format(
            len(library.phoenix_wave),
            float(library.phoenix_wave[0]),
            float(library.phoenix_wave[-1]),
        ),
    )

    if args.skip_phoenix_scan:
        _warn("PHOENIX template scan skipped", "--skip-phoenix-scan was supplied")
        return True

    try:
        teff, feh, logg = library.available_axes()
    except RuntimeError as exc:
        _status(False, "PHOENIX template discovery", str(exc))
        return False
    _status(
        True,
        "PHOENIX template discovery",
        "Teff={0} values, [Fe/H]={1} values, logg={2} values".format(
            len(teff),
            len(feh),
            len(logg),
        ),
    )
    return True


def check_spectrum(args):
    if args.spectrum is None:
        _warn("Example spectrum ingestion skipped", "pass --spectrum and --reader")
        return True
    reader = getattr(args, "reader", None)
    instrument = getattr(args, "instrument", None)
    if reader is not None and instrument is not None:
        _status(False, "Example spectrum ingestion", "pass --reader or --instrument, not both")
        return False
    if instrument is not None:
        warnings.warn(
            "--instrument is deprecated for setup checks; use --reader.",
            DeprecationWarning,
            stacklevel=2,
        )
        reader = instrument
    if reader is None:
        _status(False, "Example spectrum ingestion", "--reader is required with --spectrum")
        return False

    from Spyctres.io import SpectrumCollection, SpectrumSegment, read_spectrum

    path = Path(args.spectrum).expanduser()
    if not path.is_file():
        _status(False, "Example spectrum file", str(path))
        return False

    try:
        spectrum = read_spectrum(str(path), reader=reader)
    except (OSError, ValueError, TypeError) as exc:
        _status(False, "Example spectrum ingestion", str(exc))
        return False

    if isinstance(spectrum, SpectrumSegment):
        segments = [spectrum]
        kind = "SpectrumSegment"
    elif isinstance(spectrum, SpectrumCollection):
        segments = list(spectrum.segments)
        kind = "SpectrumCollection"
    else:
        _status(False, "Example spectrum ingestion", type(spectrum).__name__)
        return False

    n_pix = sum(len(segment.wave) for segment in segments)
    wave_min = min(float(segment.wave.min()) for segment in segments)
    wave_max = max(float(segment.wave.max()) for segment in segments)
    _status(
        True,
        "Example spectrum ingestion",
        "{0}, {1} segment(s), {2} pixel(s), {3:.3f}–{4:.3f} A".format(
            kind,
            len(segments),
            int(n_pix),
            wave_min,
            wave_max,
        ),
    )
    return True


def build_parser(prog=None, description=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description
        or "Setup diagnostic for a local Spyctres + PHOENIX environment.",
        epilog=(
            "Examples:\n"
            "  spyctres doctor --skip-phoenix\n"
            "  spyctres doctor "
            "--spectrum examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--reader xshooter_merge1d\n\n"
            "Minimal advice: use --skip-phoenix for an I/O-only environment check, "
            "or configure PHOENIX with SPYCTRES_PHOENIX_DIR / ~/.config/spyctres/config.toml."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--phoenix-dir", default=None, help="Explicit PHOENIX root.")
    parser.add_argument(
        "--require-phoenix",
        action="store_true",
        help="Fail if no PHOENIX directory is configured.",
    )
    parser.add_argument(
        "--skip-phoenix",
        action="store_true",
        help="Skip all PHOENIX path/library checks.",
    )
    parser.add_argument(
        "--skip-phoenix-scan",
        action="store_true",
        help="Initialize PHOENIX but skip the template-file scan.",
    )
    parser.add_argument("--spectrum", default=None, help="Optional example spectrum file.")
    parser.add_argument("--reader", default=None, help="Reader alias for --spectrum.")
    parser.add_argument(
        "--instrument",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print tracebacks for unexpected checker failures.",
    )
    return parser


def run_checks(args):
    checks = []
    try:
        checks.append(check_python())
        checks.append(check_imports())
        checks.append(check_cli_entrypoint())
        checks.append(check_package_data())
        checks.append(check_phoenix(args))
        checks.append(check_spectrum(args))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        _status(False, "Unexpected setup-check failure", str(exc))
        if getattr(args, "debug", False):
            traceback.print_exc()
        checks.append(False)
    return all(bool(value) for value in checks)


def main(argv=None, *, prog=None, description=None):
    args = build_parser(prog=prog, description=description).parse_args(argv)
    ok = run_checks(args)
    print("", flush=True)
    if ok:
        print("Spyctres setup check passed.", flush=True)
        return 0
    print("Spyctres setup check failed. Fix the FAIL lines above and rerun.", flush=True)
    return 1
