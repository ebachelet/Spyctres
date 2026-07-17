#!/usr/bin/env python
"""Summarize Spyctres batch quickscan/refine timing checkpoints.

This is a read-only helper for answering practical throughput questions after
running a small pilot batch with ``examples/batch_quickscan_then_refine.py``.
It does not fit spectra; it only reads the JSON checkpoint produced by that
example and reports median/mean runtime plus a simple projection for larger
batches.

Example
-------
python scripts/throughput_summary.py /tmp/spyctres_batch_refined.json --project 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def _finite_seconds(records, key):
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value >= 0.0:
            values.append(value)
    return values


def _stats(values):
    if not values:
        return {
            "count": 0,
            "median_seconds": None,
            "mean_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": int(len(values)),
        "median_seconds": float(statistics.median(values)),
        "mean_seconds": float(statistics.fmean(values)),
        "min_seconds": float(min(values)),
        "max_seconds": float(max(values)),
    }


def load_records(paths):
    records = []
    for path in paths:
        path = Path(path).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("results", []):
            item = dict(record)
            item["_source_json"] = str(path)
            records.append(item)
    return records


def summarize_records(records, project_n=100):
    status_counts = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    total_values = _finite_seconds(records, "total_seconds")
    quick_values = _finite_seconds(records, "quick_seconds")
    refine_values = _finite_seconds(records, "refine_seconds")
    projected = {}
    if total_values and project_n is not None:
        median = statistics.median(total_values)
        mean = statistics.fmean(total_values)
        projected = {
            "n_spectra": int(project_n),
            "median_based_seconds": float(median * int(project_n)),
            "mean_based_seconds": float(mean * int(project_n)),
            "median_based_hours": float(median * int(project_n) / 3600.0),
            "mean_based_hours": float(mean * int(project_n) / 3600.0),
        }

    return {
        "n_records": int(len(records)),
        "status_counts": dict(sorted(status_counts.items())),
        "total_seconds": _stats(total_values),
        "quick_seconds": _stats(quick_values),
        "refine_seconds": _stats(refine_values),
        "projection": projected,
    }


def _format_seconds(value):
    if value is None:
        return "n/a"
    value = float(value)
    if value < 120:
        return "{0:.2f} s".format(value)
    return "{0:.2f} min".format(value / 60.0)


def print_summary(summary):
    print("Spyctres throughput summary")
    print("  records: {0}".format(summary["n_records"]))
    print("  statuses:")
    for status, count in summary["status_counts"].items():
        print("    {0}: {1}".format(status, count))

    for key, label in (
        ("quick_seconds", "quick scan"),
        ("refine_seconds", "focused refine"),
        ("total_seconds", "total per spectrum"),
    ):
        stats = summary[key]
        print(
            "  {0}: n={1}, median={2}, mean={3}, min={4}, max={5}".format(
                label,
                stats["count"],
                _format_seconds(stats["median_seconds"]),
                _format_seconds(stats["mean_seconds"]),
                _format_seconds(stats["min_seconds"]),
                _format_seconds(stats["max_seconds"]),
            )
        )

    projection = summary.get("projection") or {}
    if projection:
        print(
            "  projected {0} spectra: median-based={1} ({2:.2f} h), "
            "mean-based={3} ({4:.2f} h)".format(
                projection["n_spectra"],
                _format_seconds(projection["median_based_seconds"]),
                projection["median_based_hours"],
                _format_seconds(projection["mean_based_seconds"]),
                projection["mean_based_hours"],
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read a Spyctres batch JSON checkpoint and summarize quickscan/"
            "refine timing. This does not run fits."
        )
    )
    parser.add_argument("json_files", nargs="+", help="Batch JSON checkpoint(s).")
    parser.add_argument(
        "--project",
        type=int,
        default=100,
        help="Project total runtime for this many spectra. Default: 100.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.project is not None and args.project <= 0:
        parser.error("--project must be positive.")
    try:
        records = load_records(args.json_files)
        summary = summarize_records(records, project_n=args.project)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print("throughput_summary: error: {0}".format(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
