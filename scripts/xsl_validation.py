"""Batch PHOENIX validation against locally downloaded XSL DR3 spectra.

Example
-------
python scripts/xsl_validation.py examples/xsl_validation_manifest.csv \
  --wave-medium air --output /tmp/xsl_validation_results.json
"""

import argparse
import csv
import json
import os

import numpy as np

from Spyctres import fit_phoenix_spectrum
from Spyctres.io import SpectrumCollection, read_spectrum


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate PHOENIX classifications against XSL DR3 stars.",
        epilog=(
            "Example:\n  python scripts/xsl_validation.py "
            "examples/xsl_validation_manifest.csv --wave-medium air "
            "--output /tmp/xsl_validation_results.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", help="CSV containing path and reference parameters.")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--wave-medium", required=True, choices=["air", "vacuum"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--teff-default", type=float, default=5750.0)
    parser.add_argument("--feh-default", type=float, default=0.0)
    parser.add_argument("--logg-default", type=float, default=4.0)
    parser.add_argument("--mdeg", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=200)
    return parser


def _optional_float(row, key, default):
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _resolve_manifest_path(manifest, value):
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(manifest)), value))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    with open(args.manifest, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("XSL validation manifest contains no spectra.")
    if "path" not in rows[0]:
        raise ValueError("XSL validation manifest requires a path column.")

    output = []
    for index, row in enumerate(rows):
        path = _resolve_manifest_path(args.manifest, row["path"])
        teff_ref = _optional_float(row, "teff_ref", args.teff_default)
        feh_ref = _optional_float(row, "feh_ref", args.feh_default)
        logg_ref = _optional_float(row, "logg_ref", args.logg_default)
        record = {
            "path": path,
            "xsl_id": row.get("xsl_id", ""),
            "spectral_type": row.get("spectral_type", ""),
            "reference": {"teff": teff_ref, "feh": feh_ref, "logg": logg_ref},
        }
        if teff_ref > 12000.0:
            record.update(
                status="unsupported_physics",
                message="Reference Teff exceeds the PHOENIX ACES 12000 K boundary.",
            )
            output.append(record)
            continue

        spectrum = read_spectrum(
            path,
            instrument="xsl_dr3",
            wave_medium=args.wave_medium,
            warn_unknown=False,
        )
        if not isinstance(spectrum, SpectrumCollection):
            raise TypeError("XSL DR3 reader must return a SpectrumCollection.")
        cache_path = None
        if args.cache_dir:
            identifier = row.get("xsl_id") or "row_{0:04d}".format(index)
            cache_path = os.path.join(args.cache_dir, identifier + ".npz")
        result = fit_phoenix_spectrum(
            spectrum,
            phoenix_dir=args.phoenix_dir,
            warn_unknown=False,
            p0=(teff_ref, feh_ref, logg_ref, 0.0),
            exclude_regions=[(5450.0, 5900.0), (9940.0, 11500.0)],
            mdeg=args.mdeg,
            cache_path=cache_path,
            max_nfev=args.max_nfev,
        )
        record.update(
            status="ok" if result["success"] else "fit_failed",
            fit={
                "teff": result["teff"],
                "feh": result["feh"],
                "logg": result["logg"],
                "rv_kms": result["rv_kms"],
                "chi2_red": result["chi2_red"],
            },
            delta={
                "teff": result["teff"] - teff_ref,
                "feh": result["feh"] - feh_ref,
                "logg": result["logg"] - logg_ref,
            },
        )
        output.append(record)

    payload = {
        "schema_version": 1,
        "wave_medium_assumption": args.wave_medium,
        "xsl_dichroic_regions_excluded_A": [[5450.0, 5900.0], [9940.0, 11500.0]],
        "results": output,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    finite_teff = [item["delta"]["teff"] for item in output if "delta" in item]
    if finite_teff:
        print("Processed:", len(finite_teff))
        print("Median delta Teff [K]:", float(np.median(finite_teff)))
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
