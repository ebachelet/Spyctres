"""Render saved XSL validation JSON payloads into classification plots.

Example
-------
python scripts/xsl_validation_plots.py /tmp/xsl_validation_results.json \
  --output-dir /tmp/xsl_validation_plots \
  --output-pdf /tmp/xsl_validation_plots.pdf
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

from Spyctres import ensure_matplotlib_config_dir

ensure_matplotlib_config_dir()

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Spyctres import plot_xsl_validation_payload


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Render validation_plot payloads saved by scripts/xsl_validation.py "
            "as per-target PNGs and/or a multi-page PDF."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/xsl_validation_plots.py "
            "examples/data/xsl_figure1_validation_coarse_results.json "
            "--output-dir /tmp/xsl_plots --output-pdf /tmp/xsl_plots.pdf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results_json", help="JSON output from scripts/xsl_validation.py.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for per-target image files. Defaults to a sibling "
            "<results_json_stem>_plots directory."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Optional multi-page PDF containing all rendered target plots.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("auto", "global", "per_segment", "none"),
        default="auto",
        help=(
            "Display normalization. 'auto' uses the payload default, normally "
            "global for XSL full-spectrum displays."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Per-target image format used with --output-dir.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--xsl-id",
        action="append",
        default=None,
        help="Render only this XSL ID; repeat to select multiple targets.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help=(
            "Render only rows with this status. Repeat to include several "
            "statuses. Defaults to ok."
        ),
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional cap on the number of rendered targets.",
    )
    return parser


def load_validation_results(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("Expected a validation JSON object with a 'results' list.")
    if not isinstance(payload["results"], list):
        raise ValueError("Validation JSON field 'results' must be a list.")
    return payload


def _safe_filename(value, fallback="target"):
    value = str(value or "").strip()
    if not value:
        value = fallback
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or fallback


def _plot_title(row):
    fit = row.get("fit") or {}
    reference = row.get("reference") or {}
    parts = [
        str(row.get("xsl_id", "XSL target")),
        str(row.get("spectral_type", "")).strip(),
        "[{0}]".format(row.get("validation_role", "unknown")),
    ]
    if fit:
        parts.append(
            "fit: Teff={0:g} K logg={1:g} [Fe/H]={2:g}".format(
                _finite_float(fit.get("teff"), 0.0),
                _finite_float(fit.get("logg"), 0.0),
                _finite_float(fit.get("feh"), 0.0),
            )
        )
    if reference:
        parts.append(
            "ref: Teff={0:g} K logg={1:g} [Fe/H]={2:g}".format(
                _finite_float(reference.get("teff"), 0.0),
                _finite_float(reference.get("logg"), 0.0),
                _finite_float(reference.get("feh"), 0.0),
            )
        )
    return "  |  ".join(part for part in parts if part)


def _finite_float(value, default):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def iter_validation_plot_rows(
    payload,
    *,
    xsl_ids=None,
    statuses=("ok",),
    max_targets=None,
):
    """Yield result rows that contain saved validation-plot payloads."""
    xsl_filter = None
    if xsl_ids:
        xsl_filter = {str(value).strip().upper() for value in xsl_ids}
    status_filter = None if statuses is None else {str(value) for value in statuses}
    count = 0
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        if (
            xsl_filter is not None
            and str(row.get("xsl_id", "")).upper() not in xsl_filter
        ):
            continue
        if (
            status_filter is not None
            and str(row.get("status", "")) not in status_filter
        ):
            continue
        plot_data = row.get("validation_plot")
        if not isinstance(plot_data, dict):
            continue
        yield row
        count += 1
        if max_targets is not None and count >= int(max_targets):
            break


def render_validation_plots(
    payload,
    *,
    output_dir=None,
    output_pdf=None,
    scale_mode="auto",
    image_format="png",
    dpi=160,
    xsl_ids=None,
    statuses=("ok",),
    max_targets=None,
):
    """Render saved XSL validation plot payloads.

    Returns a list of generated per-target image paths.  The optional PDF path
    is not included in that list because it contains all rendered targets.
    """
    rows = list(
        iter_validation_plot_rows(
            payload,
            xsl_ids=xsl_ids,
            statuses=statuses,
            max_targets=max_targets,
        )
    )
    if not rows:
        raise ValueError("No matching rows contain validation_plot payloads.")

    image_paths = []
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    pdf = PdfPages(output_pdf) if output_pdf is not None else None
    try:
        for row in rows:
            mode = None if scale_mode == "auto" else scale_mode
            fig, _axes = plot_xsl_validation_payload(
                row["validation_plot"],
                scale_mode=mode,
                title=_plot_title(row),
            )
            if output_path is not None:
                stem = "_".join(
                    item
                    for item in (
                        _safe_filename(row.get("xsl_id"), fallback="target"),
                        _safe_filename(row.get("spectral_type"), fallback=""),
                    )
                    if item
                )
                image_path = output_path / "{0}.{1}".format(stem, image_format)
                fig.savefig(image_path, dpi=dpi)
                image_paths.append(str(image_path))
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return image_paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_targets is not None and args.max_targets < 1:
        raise ValueError("--max-targets must be >= 1.")
    results_path = Path(args.results_json)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(results_path.with_name(results_path.stem + "_plots"))

    payload = load_validation_results(results_path)
    images = render_validation_plots(
        payload,
        output_dir=output_dir,
        output_pdf=args.output_pdf,
        scale_mode=args.scale_mode,
        image_format=args.format,
        dpi=args.dpi,
        xsl_ids=args.xsl_id,
        statuses=args.status or ("ok",),
        max_targets=args.max_targets,
    )
    print("Rendered {0} XSL validation plot(s).".format(len(images)), flush=True)
    print("Image directory: {0}".format(os.path.abspath(output_dir)), flush=True)
    if args.output_pdf:
        print("PDF: {0}".format(os.path.abspath(args.output_pdf)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
