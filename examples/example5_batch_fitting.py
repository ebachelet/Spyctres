#!/usr/bin/env python
"""Example 5: batch quickscan then focused refinement.

This is the collaborator-facing pattern for tens to hundreds of spectra:
load PHOENIX once, run a cheap quick scan for each target, and refine only when
readiness and quality gates permit it.  This wrapper keeps the numbered example
path concise and delegates the operational details to
``batch_quickscan_then_refine.py``.

What this demonstrates
----------------------
How to avoid a full blind PHOENIX search for every spectrum by checkpointing a
cheap scan first and running focused refinement only when quality gates permit.

What this does not prove
------------------------
A batch result is not automatically analysis-ready.  The saved JSON/CSV
should be treated as a triage product until flagged spectra, assumptions,
residual plots, and representative real-spectrum validation have been reviewed.

Example PHOENIX-free dry run:

  python examples/example5_batch_fitting.py --dry-run

Example quicklook batch:

  python examples/example5_batch_fitting.py \
    --quicklook \
    --output-json /tmp/spyctres_example5_batch_quick.json \
    --summary-csv /tmp/spyctres_example5_batch_quick.csv \
    --plot-dir /tmp/spyctres_example5_plots \
    --max-plots 2 \
    --resume

Example gated refinement:

  python examples/example5_batch_fitting.py \
    --output-json /tmp/spyctres_example5_batch_refined.json \
    --summary-csv /tmp/spyctres_example5_batch_refined.csv \
    --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_DIR.parent
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import batch_quickscan_then_refine
import Spyctres as sp


EXAMPLE_SPECTRA = (
    sp.example_data_path("gaia_benchmark/HIP79672_HARPS_1_R42KNorm.txt.gz"),
    sp.example_data_path("gaia_benchmark/HIP37279_HARPS_1_R42KNorm.txt.gz"),
    # A deliberately harder metal-poor benchmark keeps the batch example honest:
    # ordinary and stress cases should be triaged separately.
    sp.example_data_path("gaia_benchmark/HIP76976_HARPS_1_R42KNorm.txt.gz"),
)

EXAMPLE_SPECTRUM_ROLES = {
    "HIP79672_HARPS_1_R42KNorm.txt.gz": "ordinary benchmark: solar-analog dwarf",
    "HIP37279_HARPS_1_R42KNorm.txt.gz": "ordinary benchmark: F subgiant",
    "HIP76976_HARPS_1_R42KNorm.txt.gz": "stress benchmark: very metal-poor subgiant",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 5: batch workflow wrapper for quickscan-then-refine. Uses "
            "three bundled Gaia benchmark spectra if no files/manifest are given."
        ),
        epilog=(
            "Dry run, no PHOENIX library needed:\n"
            "  python examples/example5_batch_fitting.py --dry-run\n\n"
            "Quicklook PHOENIX pass:\n"
            "  python examples/example5_batch_fitting.py --quicklook "
            "--output-json /tmp/spyctres_example5_batch_quick.json "
            "--summary-csv /tmp/spyctres_example5_batch_quick.csv "
            "--plot-dir /tmp/spyctres_example5_plots --max-plots 2 --resume\n\n"
            "Gated refinement:\n"
            "  python examples/example5_batch_fitting.py "
            "--output-json /tmp/spyctres_example5_batch_refined.json "
            "--summary-csv /tmp/spyctres_example5_batch_refined.csv --resume\n\n"
            "Next:\n"
            "  python scripts/throughput_summary.py /tmp/spyctres_example5_batch_refined.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectra", nargs="*")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--reader", default="gbs_v3_ascii")
    parser.add_argument("--instrument", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument("--quicklook", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read the input spectra and print reviewed setup/readiness "
            "summaries without loading PHOENIX or fitting."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plot-dir",
        default=None,
        help=(
            "Optional directory for a small number of representative fit plots. "
            "Use this to inspect sample successful/flagged fits without "
            "plotting an entire large batch."
        ),
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=3,
        help="Maximum representative plots to write with --plot-dir. Default: 3.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_example5_batch.json",
    )
    parser.add_argument("--summary-csv", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instrument is not None:
        if args.reader != "gbs_v3_ascii":
            raise ValueError("Pass --reader or --instrument, not both.")
        args.reader = args.instrument
    spectra = list(args.spectra) if args.spectra else [str(path) for path in EXAMPLE_SPECTRA]
    if args.dry_run:
        if args.manifest:
            raise ValueError(
                "--dry-run in this beginner wrapper expects positional spectra. "
                "For manifest batches, use the full operational "
                "examples/batch_quickscan_then_refine.py command."
            )
        print("Dry-run batch plan: no PHOENIX library will be loaded.", flush=True)
        for index, path in enumerate(spectra, start=1):
            display_path = str(path)
            try:
                display_path = str(Path(path).resolve().relative_to(_REPO_ROOT))
            except ValueError:
                pass
            print(
                "\nTarget {0}/{1}: {2}".format(index, len(spectra), display_path),
                flush=True,
            )
            role = EXAMPLE_SPECTRUM_ROLES.get(Path(path).name)
            if role:
                print("  bundled role: {0}".format(role), flush=True)
            spec = sp.read_spectrum(path, reader=args.reader)
            setup = sp.suggest_fit_setup(
                spec,
                mode="quicklook" if args.quicklook else "standard",
                assumed_resolution=args.resolution_R,
            )
            print(setup.summary_text(include_hash=False), flush=True)
            audit = sp.audit_spectrum_for_fit(
                spec,
                regions=setup.regions,
                assumed_resolution=args.resolution_R,
                intent="quicklook_classification",
            )
            print(
                "  audit: fit_ready={0}, fitted_pixels={1}, flags={2}".format(
                    audit.get("fit_ready"),
                    audit.get("n_fit_candidate"),
                    ", ".join(audit.get("interpretation_flags") or []) or "none",
                ),
                flush=True,
            )
        print(
            "\nNext: remove --dry-run and use --quicklook for a checkpointed "
            "first PHOENIX pass.",
            flush=True,
        )
        return 0
    delegated = []
    if args.manifest:
        delegated.extend(["--manifest", str(args.manifest)])
    else:
        delegated.extend(spectra)
    delegated.extend(["--reader", str(args.reader)])
    if args.phoenix_dir:
        delegated.extend(["--phoenix-dir", str(args.phoenix_dir)])
    if args.resolution_R is not None:
        delegated.extend(["--R", str(args.resolution_R)])
    if args.quicklook:
        delegated.append("--quicklook")
    if args.resume:
        delegated.append("--resume")
    if args.force:
        delegated.append("--force")
    if args.plot_dir:
        delegated.extend(["--plot-dir", str(args.plot_dir)])
        delegated.extend(["--max-plots", str(args.max_plots)])
    delegated.extend(["--output-json", str(args.output_json)])
    if args.summary_csv:
        delegated.extend(["--summary-csv", str(args.summary_csv)])
    exit_code = batch_quickscan_then_refine.main(delegated)
    if exit_code == 0:
        print(
            "\nScope note: Example 5 is a throughput/triage workflow. Inspect "
            "the checkpointed quality flags and representative referee plots "
            "before treating a batch as science-ready.",
            flush=True,
        )
        print(
            "Next: run scripts/throughput_summary.py on the saved JSON to "
            "estimate runtime for larger batches.",
            flush=True,
        )
        if args.plot_dir:
            print(
                "Representative plots, if written, are in: {0}".format(
                    args.plot_dir
                ),
                flush=True,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
