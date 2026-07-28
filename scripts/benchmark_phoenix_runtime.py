#!/usr/bin/env python
"""Developer/regression runtime benchmark for the public PHOENIX workflow.

Default mode is intentionally cheap: it reads a bundled spectrum and builds the
reviewed setup, then writes a path-sanitized benchmark JSON.  Add ``--run-fit``
only when PHOENIX is configured and you want to measure an actual quicklook fit.
This script is for profiling decisions; it is not a validation of stellar
parameters.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc


_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Spyctres as sp
from Spyctres._serialization import atomic_write_json, atomic_write_csv_rows


EXAMPLE_18SCO = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "data"
    / "gaia_benchmark"
    / "HIP79672_HARPS_1_R42KNorm.txt.gz"
)


def _safe_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _timed(label, func):
    tracemalloc.start()
    start = time.perf_counter()
    out = func()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, {
        "label": label,
        "seconds": float(elapsed),
        "peak_memory_mb": float(peak / (1024.0 * 1024.0)),
    }


def _progress(event):
    message = event.get("message", str(event)) if isinstance(event, dict) else str(event)
    print(message, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Developer/regression runtime benchmark for Spyctres read/setup/"
            "optional PHOENIX quicklook fitting. Use this before adding "
            "acceleration layers."
        ),
        epilog=(
            "Cheap benchmark, no PHOENIX required:\n"
            "  python scripts/benchmark_phoenix_runtime.py "
            "--output-json /tmp/spyctres_runtime.json\n\n"
            "Opt-in quicklook fit benchmark:\n"
            "  python scripts/benchmark_phoenix_runtime.py --run-fit "
            "--output-json /tmp/spyctres_runtime_fit.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectrum", nargs="?", default=str(EXAMPLE_18SCO))
    parser.add_argument("--reader", default="gbs_v3_ascii")
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--defaults-mode", default="quicklook")
    parser.add_argument("--run-fit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-json", default="/tmp/spyctres_runtime_benchmark.json")
    parser.add_argument("--output-csv", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.reader is not None and args.instrument is not None:
        raise ValueError("Pass --reader or --instrument, not both.")
    if args.instrument is not None:
        args.reader = args.instrument
    if int(args.repeat) < 1:
        raise ValueError("--repeat must be >= 1.")

    records = []
    last_result = None
    for index in range(int(args.repeat)):
        print("Benchmark repeat {0}/{1}".format(index + 1, args.repeat), flush=True)
        spec, read_record = _timed(
            "read_spectrum",
            lambda: sp.read_spectrum(args.spectrum, reader=args.reader, warn_unknown=False),
        )
        setup, setup_record = _timed(
            "suggest_fit_setup",
            lambda: sp.suggest_fit_setup(spec, mode=args.defaults_mode),
        )
        iteration = {
            "repeat": int(index + 1),
            "read_spectrum": read_record,
            "suggest_fit_setup": setup_record,
            "setup_hash": setup.setup_hash,
        }
        if args.run_fit:
            result, fit_record = _timed(
                "fit_stellar_spectrum",
                lambda: sp.fit_stellar_spectrum(
                    spec,
                    model="phoenix",
                    setup=setup,
                    phoenix_dir=args.phoenix_dir,
                    progress_callback=_progress,
                ),
            )
            iteration["fit_stellar_spectrum"] = fit_record
            iteration["fit_summary"] = {
                key: result.summary.get(key)
                for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
            }
            last_result = result
        records.append(iteration)

    payload = {
        "schema_name": "spyctres.runtime_benchmark",
        "schema_status": "experimental",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_scope": "public_read_setup_optional_fit",
        "spectrum": _safe_path(args.spectrum),
        "reader": str(args.reader),
        "run_fit": bool(args.run_fit),
        "defaults_mode": str(args.defaults_mode),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "spyctres_version": getattr(sp, "__version__", _version("Spyctres")),
            "numpy_version": _version("numpy"),
            "scipy_version": _version("scipy"),
            "astropy_version": _version("astropy"),
        },
        "cache": {
            "phoenix_dir_configured": bool(args.phoenix_dir or os.environ.get("SPYCTRES_PHOENIX_DIR")),
            "note": (
                "Cold/warm PHOENIX cache interpretation requires comparing "
                "explicit --run-fit repeats under the same local PHOENIX setup."
            ),
        },
        "records": records,
    }
    if last_result is not None:
        payload["last_quality_flags"] = list(last_result.quality_flags)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, payload)
    print("Wrote benchmark JSON: {0}".format(args.output_json), flush=True)

    if args.output_csv:
        rows = []
        for item in records:
            row = {"repeat": item["repeat"], "setup_hash": item["setup_hash"]}
            for key in ("read_spectrum", "suggest_fit_setup", "fit_stellar_spectrum"):
                if key in item:
                    row["{0}_seconds".format(key)] = item[key]["seconds"]
                    row["{0}_peak_memory_mb".format(key)] = item[key]["peak_memory_mb"]
            rows.append(row)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "repeat",
            "setup_hash",
            "read_spectrum_seconds",
            "read_spectrum_peak_memory_mb",
            "suggest_fit_setup_seconds",
            "suggest_fit_setup_peak_memory_mb",
            "fit_stellar_spectrum_seconds",
            "fit_stellar_spectrum_peak_memory_mb",
        ]
        atomic_write_csv_rows(args.output_csv, fieldnames, rows)
        print("Wrote benchmark CSV: {0}".format(args.output_csv), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
