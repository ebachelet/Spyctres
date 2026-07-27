#!/usr/bin/env python
"""Example 4: publication-oriented scaffold wrapper.

The publication workflow is intentionally more conservative than the beginner
quickstart.  This wrapper keeps the numbered learning path tidy while delegating
the detailed X-SHOOTER UVB audit, Balmer-core mask sensitivity, systematic-plan,
and optional baseline fit to ``publication_quality_xshooter_uvb.py``.

What this demonstrates
----------------------
How Spyctres separates a reviewed, audit-first publication scaffold from the
shorter quicklook examples.

What this does not prove
------------------------
An audit-only pass is not a completed publication analysis.  Even an opt-in
baseline fit remains exploratory until metadata, masks, LSF assumptions,
systematic variants, and external validation are reviewed.

Example audit-only run:

  python examples/example4_publication_quality_fitting.py \
    --output-json /tmp/spyctres_example4_publication.json

Example opt-in baseline fit:

  python examples/example4_publication_quality_fitting.py \
    --run-baseline-fit \
    --output-json /tmp/spyctres_example4_publication_fit.json \
    --output-report-json /tmp/spyctres_example4_publication_report.json \
    --output-plot /tmp/spyctres_example4_publication_fit.png
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

import publication_quality_xshooter_uvb


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Example 4: publication-oriented X-SHOOTER UVB scaffold. Default "
            "is audit-only; expensive fits remain opt-in."
        ),
        epilog=(
            "Audit only:\n"
            "  python examples/example4_publication_quality_fitting.py "
            "--output-json /tmp/spyctres_example4_publication.json\n\n"
            "Opt-in baseline fit:\n"
            "  python examples/example4_publication_quality_fitting.py "
            "--run-baseline-fit "
            "--output-json /tmp/spyctres_example4_publication_fit.json "
            "--output-report-json /tmp/spyctres_example4_publication_report.json "
            "--output-plot /tmp/spyctres_example4_publication_fit.png\n\n"
            "Next:\n"
            "  python examples/example5_batch_fitting.py --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=None,
        help="Optional spectrum path; omitted means use the bundled UVB example.",
    )
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--run-baseline-fit",
        action="store_true",
        help="Delegate an expensive baseline PHOENIX fit to the scaffold.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_example4_publication.json",
    )
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-plot", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    delegated = ["--output-json", str(args.output_json)]
    if args.spectrum:
        delegated.append(str(args.spectrum))
    if args.instrument:
        delegated.extend(["--instrument", str(args.instrument)])
    if args.phoenix_dir:
        delegated.extend(["--phoenix-dir", str(args.phoenix_dir)])
    if args.run_baseline_fit:
        delegated.append("--run-baseline-fit")
    if args.output_report_json:
        delegated.extend(["--output-report-json", str(args.output_report_json)])
    if args.output_plot:
        delegated.extend(["--output-plot", str(args.output_plot)])
    exit_code = publication_quality_xshooter_uvb.main(delegated)
    if exit_code == 0:
        print(
            "\nScope note: Example 4 is a conservative scaffold. Its job is to "
            "make assumptions, blockers, and follow-up checks explicit before "
            "a result is promoted beyond exploratory use.",
            flush=True,
        )
        print(
            "Next: run examples/example5_batch_fitting.py --help to see the "
            "quickscan/refine pattern for many spectra.",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
