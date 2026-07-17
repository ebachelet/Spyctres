"""Small read-only command-line helpers for Spyctres.

This CLI is intentionally limited to discovery and ingestion inspection.  It
does not expose a general fitting command; the fitting API remains Python-first
until defaults, runtime budgets, and reporting are stable enough for a public
batch interface.
"""

from __future__ import annotations

import argparse
import json
from importlib import metadata as importlib_metadata
import sys

import numpy as np

from .io import (
    SpectrumCollection,
    SpectrumSegment,
    get_instrument_info,
    list_instruments,
    read_spectrum,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _package_version():
    try:
        return importlib_metadata.version("Spyctres")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _segment_summary(segment):
    wave = np.asarray(segment.wave, dtype=float)
    mask = np.asarray(segment.mask, dtype=bool)
    finite_wave = wave[np.isfinite(wave)]
    return {
        "name": segment.name,
        "n_pixels": int(wave.size),
        "n_valid": int(np.count_nonzero(mask)),
        "wave_min_A": float(np.nanmin(finite_wave)) if finite_wave.size else None,
        "wave_max_A": float(np.nanmax(finite_wave)) if finite_wave.size else None,
        "wave_medium": segment.wave_medium,
        "wave_frame": segment.wave_frame,
        "observer_frame": segment.observer_frame,
        "stellar_rest_status": segment.stellar_rest_status,
        "stellar_rv_applied_kms": segment.stellar_rv_applied_kms,
        "resolution": (
            None
            if segment.resolution is None
            else segment.resolution.to_metadata()
        ),
        "meta_keys": sorted(str(key) for key in segment.meta),
    }


def _spectrum_summary(spectrum, path, instrument):
    if isinstance(spectrum, SpectrumSegment):
        segments = [spectrum]
        kind = "SpectrumSegment"
    elif isinstance(spectrum, SpectrumCollection):
        segments = list(spectrum.segments)
        kind = "SpectrumCollection"
    else:
        raise TypeError("Unsupported reader output: {0}".format(type(spectrum).__name__))

    return {
        "path": str(path),
        "instrument": str(instrument),
        "kind": kind,
        "n_segments": int(len(segments)),
        "segments": [_segment_summary(segment) for segment in segments],
    }


def cmd_instruments(args):
    names = list_instruments(include_aliases=args.aliases)
    if args.json:
        _print_json({"instruments": names, "include_aliases": bool(args.aliases)})
        return 0

    label = "Accepted aliases" if args.aliases else "Supported instruments"
    print(label + ":")
    for name in names:
        print("  " + name)
    return 0


def cmd_instrument_info(args):
    info = get_instrument_info(args.instrument).to_metadata()
    if args.json:
        _print_json(info)
        return 0

    print("{canonical_name}".format(**info))
    print("  aliases: {0}".format(", ".join(info["aliases"])))
    for key in (
        "expected_file_type",
        "wavelength_location",
        "wavelength_unit",
        "default_wave_medium",
        "default_observer_frame",
        "default_stellar_rest_status",
        "resolving_power",
        "notes",
    ):
        value = info.get(key)
        if value is not None:
            print("  {0}: {1}".format(key, value))
    return 0


def cmd_inspect_spectrum(args):
    spectrum = read_spectrum(
        args.path,
        instrument=args.instrument,
        warn_unknown=not args.no_warn_unknown,
    )
    payload = _spectrum_summary(spectrum, args.path, args.instrument)
    if args.json:
        _print_json(payload)
        return 0

    print(
        "{kind}: {n_segments} segment(s) from {path}".format(**payload)
    )
    for index, segment in enumerate(payload["segments"], start=1):
        print(
            "  {0}. {name}: {n_pixels} pixels, {n_valid} valid, "
            "{wave_min_A:.3f}-{wave_max_A:.3f} A, "
            "medium={wave_medium}, observer={observer_frame}, "
            "stellar_rest={stellar_rest_status}".format(
                index,
                **segment,
            )
        )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="spyctres",
        description="Read-only Spyctres discovery and spectrum-inspection helpers.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Spyctres {0}".format(_package_version()),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    instruments = subparsers.add_parser(
        "instruments",
        help="List registered spectrum readers.",
        description="List registered Spyctres spectrum readers.",
    )
    instruments.add_argument(
        "--aliases",
        action="store_true",
        help="List all accepted aliases instead of canonical names.",
    )
    instruments.add_argument("--json", action="store_true", help="Emit JSON.")
    instruments.set_defaults(func=cmd_instruments)

    instrument_info = subparsers.add_parser(
        "instrument-info",
        help="Show metadata for one registered reader.",
        description="Show read-only metadata for one registered spectrum reader.",
    )
    instrument_info.add_argument("instrument", help="Canonical name or alias.")
    instrument_info.add_argument("--json", action="store_true", help="Emit JSON.")
    instrument_info.set_defaults(func=cmd_instrument_info)

    inspect = subparsers.add_parser(
        "inspect-spectrum",
        help="Read a spectrum and summarize the common-format container.",
        description=(
            "Read a spectrum with a registered reader and summarize the "
            "resulting SpectrumSegment/SpectrumCollection. This command does "
            "not fit, normalize, resample, or write outputs."
        ),
    )
    inspect.add_argument("path", help="Spectrum file to inspect.")
    inspect.add_argument(
        "--instrument",
        required=True,
        help="Registered reader name or alias.",
    )
    inspect.add_argument(
        "--no-warn-unknown",
        action="store_true",
        help="Suppress warnings about unknown wavelength/frame semantics.",
    )
    inspect.add_argument("--json", action="store_true", help="Emit JSON.")
    inspect.set_defaults(func=cmd_inspect_spectrum)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, TypeError, ValueError) as exc:
        print("spyctres: error: {0}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
