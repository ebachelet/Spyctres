#!/usr/bin/env python
"""Example 5: batch quickscan then focused refinement.

This is the collaborator-facing pattern for tens to hundreds of spectra:
load PHOENIX once, run a cheap quick scan for each target, and refine only when
readiness and quality gates permit it.  This wrapper keeps the numbered example
path concise and delegates the operational details to
``batch_quickscan_then_refine.py``.

Example quicklook batch:

  python examples/example5_batch_fitting.py \
    --quicklook \
    --output-json /tmp/spyctres_example5_batch_quick.json \
    --summary-csv /tmp/spyctres_example5_batch_quick.csv \
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


EXAMPLE_UVB = (
    _EXAMPLE_DIR / "data" / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 5: batch workflow wrapper for quickscan-then-refine. Uses "
            "the bundled X-SHOOTER UVB spectrum if no files/manifest are given."
        ),
        epilog=(
            "Quicklook:\n"
            "  python examples/example5_batch_fitting.py --quicklook "
            "--output-json /tmp/spyctres_example5_batch_quick.json "
            "--summary-csv /tmp/spyctres_example5_batch_quick.csv --resume\n\n"
            "Gated refinement:\n"
            "  python examples/example5_batch_fitting.py "
            "--output-json /tmp/spyctres_example5_batch_refined.json "
            "--summary-csv /tmp/spyctres_example5_batch_refined.csv --resume"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("spectra", nargs="*")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--instrument", default="xshooter")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument("--R", type=float, default=None, dest="resolution_R")
    parser.add_argument("--quicklook", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_example5_batch.json",
    )
    parser.add_argument("--summary-csv", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spectra = list(args.spectra) if args.spectra else [str(EXAMPLE_UVB)]
    delegated = []
    if args.manifest:
        delegated.extend(["--manifest", str(args.manifest)])
    else:
        delegated.extend(spectra)
    delegated.extend(["--instrument", str(args.instrument)])
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
    delegated.extend(["--output-json", str(args.output_json)])
    if args.summary_csv:
        delegated.extend(["--summary-csv", str(args.summary_csv)])
    return batch_quickscan_then_refine.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
