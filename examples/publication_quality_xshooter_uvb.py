"""Example 7: publication-oriented X-SHOOTER UVB workflow scaffold.

This script is deliberately stricter than ``simple_phoenix_fit.py``. It is not
the one-command path for new users; it is a reproducible scaffold for expert
work where fitted stellar parameters should eventually be defensible in a paper
or technical report.

The default run is audit-only. It reads the bundled X-SHOOTER UVB spectrum,
constructs explicit Balmer-window segments with documented masks, runs both the
ordinary fit-readiness audit and the stricter publication-readiness audit, and
writes a JSON checkpoint. The expensive PHOENIX baseline fit is opt-in via
``--run-baseline-fit``.

Example audit-only run
----------------------
python examples/publication_quality_xshooter_uvb.py \
  --output-json /tmp/spyctres_publication_xshooter_uvb.json

Example baseline fit after PHOENIX is configured
------------------------------------------------
python examples/publication_quality_xshooter_uvb.py \
  --run-baseline-fit \
  --output-json /tmp/spyctres_publication_xshooter_uvb_fit.json \
  --output-plot /tmp/spyctres_publication_xshooter_uvb_fit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Let the example run directly from a source checkout, even if Spyctres has not
# been installed into the active Python environment yet.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "Spyctres").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Spyctres import (
    build_diagnostic_window_combinations,
    fit_stellar_spectrum,
    prepare_phoenix_fit_kwargs,
    publication_readiness_audit,
    select_diagnostic_windows,
)
from Spyctres.io import SpectrumCollection, SpectrumSegment, read_spectrum
from Spyctres.plotting import plot_fit_referee
from Spyctres.preprocessing import (
    archive_exclusion_masks_for_segment,
    audit_spectrum_for_fit,
)
from Spyctres.recipes import prepare_xshooter_balmer_case


EXAMPLE_UVB = (
    Path(__file__).resolve().parent
    / "data"
    / "TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits"
)


METAL_RV_WINDOWS = (
    {
        "label": "Ca II H/K sanity window",
        "region_A": [3900.0, 3995.0],
        "purpose": (
            "Independent wavelength/RV and line-depth sanity check; can be "
            "affected by interstellar Ca II, H epsilon, and local artifacts."
        ),
        "used_in_baseline_fit": False,
    },
    {
        "label": "Mg I b / blue-metal sanity window",
        "region_A": [5150.0, 5205.0],
        "purpose": (
            "Potential metal-line/RV consistency check for cooler spectra. "
            "For hot spectra it may carry little classification leverage."
        ),
        "used_in_baseline_fit": False,
    },
)


PUBLICATION_FOLLOWUP_STEPS = (
    "Inspect formal uncertainties, bad-pixel/product masks, and wavelength "
    "medium/frame provenance before interpreting fitted parameters.",
    "Compare joint Balmer-window and per-line Balmer fits; use leave-one-line-out "
    "checks to identify line-specific model or mask failures.",
    "Run systematic variants for continuum mode, Balmer-core mask width, fit "
    "windows, resolution assumptions, and residual/outlier policy.",
    "Run target-matched synthetic injection/recovery before claiming calibrated "
    "uncertainties for this data quality and spectral type.",
    "Only after these checks pass should an optional posterior/profile scan be "
    "used for publication uncertainty tables.",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Publication-oriented X-SHOOTER UVB scaffold. Default is audit-only; "
            "use --run-baseline-fit to run the expensive PHOENIX baseline fit."
        ),
        epilog=(
            "Examples:\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--output-json /tmp/spyctres_publication_xshooter_uvb.json\n\n"
            "  python examples/publication_quality_xshooter_uvb.py "
            "--run-baseline-fit "
            "--output-json /tmp/spyctres_publication_xshooter_uvb_fit.json "
            "--output-plot /tmp/spyctres_publication_xshooter_uvb_fit.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "spectrum",
        nargs="?",
        default=str(EXAMPLE_UVB),
        help=(
            "X-SHOOTER UVB-like reduced 1D spectrum. Defaults to the bundled "
            "UVB example in examples/data/."
        ),
    )
    parser.add_argument(
        "--instrument",
        default="xshooter",
        help="Registered Spyctres reader name. Default: xshooter.",
    )
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--output-json",
        default="/tmp/spyctres_publication_xshooter_uvb.json",
        help="Atomic JSON checkpoint/output path.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional baseline-fit referee plot path. Requires --run-baseline-fit.",
    )
    parser.add_argument(
        "--output-comparison-csv",
        default=None,
        help=(
            "Optional compact CSV table for the Balmer-core mask sensitivity "
            "grid. If --run-core-mask-fit-grid is set, fitted parameters are "
            "included."
        ),
    )
    parser.add_argument(
        "--output-comparison-plot",
        default=None,
        help=(
            "Optional compact PNG plot for the Balmer-core mask sensitivity "
            "grid. Does not run extra fits by itself."
        ),
    )
    parser.add_argument(
        "--output-diagnostic-window-csv",
        default=None,
        help=(
            "Optional CSV table of generic diagnostic windows available in "
            "the loaded spectrum."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the output JSON already exists, print its status and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when --resume finds an existing output JSON.",
    )
    parser.add_argument(
        "--run-baseline-fit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a baseline native-grid PHOENIX fit on the Balmer-window "
            "segments. Default: False, so the scaffold can run without PHOENIX."
        ),
    )
    parser.add_argument(
        "--balmer-window-mode",
        choices=("current", "notebook"),
        default="notebook",
        help="Balmer-window preset from Spyctres.recipes.",
    )
    parser.add_argument("--window-pad", type=float, default=5.0)
    parser.add_argument(
        "--norm-mode",
        choices=("poly", "sideband"),
        default="poly",
        help=(
            "Preparation normalization mode for the Balmer segments. 'poly' "
            "leaves continuum handling to the PHOENIX fit; 'sideband' applies "
            "explicit local sideband normalization first."
        ),
    )
    parser.add_argument("--sideband-width", type=float, default=10.0)
    parser.add_argument("--sideband-order", type=int, default=1)
    parser.add_argument(
        "--balmer-core-mask",
        type=float,
        default=10.0,
        help=(
            "Half-width in Angstrom for the opt-in Balmer-core exclusion mask. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--core-mask-grid",
        default="0,4,6,8,10,12",
        help=(
            "Comma-separated Balmer-core half-widths in Angstrom to audit as "
            "a sensitivity grid. Use an empty string to disable. Width 0 means "
            "no Balmer-core mask."
        ),
    )
    parser.add_argument(
        "--min-core-mask-retained-fraction",
        type=float,
        default=0.70,
        help=(
            "Diagnostic threshold for flagging overly large Balmer-core masks. "
            "A mask-grid point retaining less than this fraction of the fitted "
            "pixels available in the no-core-mask reference is flagged as "
            "excessive. This is provenance only; it does not alter the fit "
            "objective. Default: 0.70."
        ),
    )
    parser.add_argument(
        "--run-core-mask-fit-grid",
        action="store_true",
        help=(
            "Also run a PHOENIX fit for every --core-mask-grid value. This can "
            "be expensive and is off by default; the default grid records "
            "readiness/pixel-count sensitivity only."
        ),
    )
    parser.add_argument(
        "--archive-mask-policy",
        choices=("apply", "warn", "ignore"),
        default="apply",
        help=(
            "How to handle recognized archive/product bad-region catalogs. "
            "'apply' adds them as named exclusion masks; 'warn' leaves them "
            "unmasked but visible to readiness audits; 'ignore' records an "
            "explicit expert override."
        ),
    )
    parser.add_argument(
        "--R",
        type=float,
        default=None,
        dest="resolution_R",
        help=(
            "Optional user-supplied constant resolving power. Publication "
            "readiness treats this as assumed unless --allow-assumed-resolution "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--allow-assumed-resolution",
        action="store_true",
        help=(
            "Allow a user-supplied --R to pass the publication gate. Use only "
            "after separately validating that approximation."
        ),
    )
    parser.add_argument("--min-fit-pixels", type=int, default=200)
    parser.add_argument(
        "--max-rejected-inside-fit-window-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--baseline-defaults-mode",
        choices=("quicklook", "standard", "diagnostic"),
        default="standard",
    )
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--rv-grid-n", type=int, default=41)
    parser.add_argument("--multistart", type=int, default=3)
    parser.add_argument("--mdeg", type=int, default=2)
    parser.add_argument("--teff", type=float, default=None)
    parser.add_argument("--feh", type=float, default=None)
    parser.add_argument("--logg", type=float, default=None)
    parser.add_argument("--rv", type=float, default=None)
    parser.add_argument("--teff-min", type=float, default=None)
    parser.add_argument("--teff-max", type=float, default=None)
    parser.add_argument("--feh-min", type=float, default=None)
    parser.add_argument("--feh-max", type=float, default=None)
    parser.add_argument("--logg-min", type=float, default=None)
    parser.add_argument("--logg-max", type=float, default=None)
    parser.add_argument("--rv-min", type=float, default=None)
    parser.add_argument("--rv-max", type=float, default=None)
    return parser


def _json_native(value):
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_native(payload)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _read_existing(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_core_mask_grid(value):
    value = "" if value is None else str(value).strip()
    if not value:
        return []
    widths = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        width = float(item)
        if not np.isfinite(width) or width < 0.0:
            raise ValueError("--core-mask-grid values must be finite and >= 0.")
        widths.append(float(width))
    return list(dict.fromkeys(widths))


def _single_segment(spectrum):
    if isinstance(spectrum, SpectrumSegment):
        return spectrum
    if isinstance(spectrum, SpectrumCollection) and len(spectrum) == 1:
        return spectrum[0]
    raise ValueError(
        "This scaffold expects one UVB segment. Use the multi-arm notebook or "
        "build an explicit expert workflow for multi-segment products."
    )


def _resolution_assumption(args):
    if args.resolution_R is None:
        return None
    return {
        "quantity": "R",
        "value": float(args.resolution_R),
        "source": "user_override",
        "resolution_source": "user_override",
        "assumed_resolution_R": float(args.resolution_R),
        "assumption_warning": (
            "user-supplied constant resolution for publication scaffold; "
            "validate before using for final parameters"
        ),
    }


def _archive_masks_by_segment(segments, policy):
    if policy != "apply":
        return ()
    masks = []
    seen = set()
    for segment in segments:
        for mask in archive_exclusion_masks_for_segment(segment):
            if mask.name in seen:
                continue
            masks.append(mask)
            seen.add(mask.name)
    return tuple(masks)


def _prepare_balmer_collection(args, segment, *, core_mask_halfwidth):
    case = prepare_xshooter_balmer_case(
        segment,
        window_mode=args.balmer_window_mode,
        window_pad=args.window_pad,
        norm_mode=args.norm_mode,
        sideband_width=args.sideband_width,
        sideband_order=args.sideband_order,
        core_mask=core_mask_halfwidth if core_mask_halfwidth > 0 else None,
    )
    collection = SpectrumCollection(
        case.fit_segments,
        meta={
            "workflow": "publication_quality_xshooter_uvb_scaffold",
            "source_spectrum": str(args.spectrum),
            "balmer_case": case.provenance,
        },
        name="xshooter_uvb_publication_balmer_windows",
    )
    archive_masks = _archive_masks_by_segment(
        collection.segments,
        args.archive_mask_policy,
    )
    exclude_masks = tuple(case.exclude_masks) + tuple(archive_masks)
    return case, collection, exclude_masks


def _fit_window_summary(segments):
    rows = []
    for segment in segments:
        wave = np.asarray(segment.wave, dtype=float)
        good = np.isfinite(wave)
        rows.append(
            {
                "name": segment.name,
                "n_pixels": int(wave.size),
                "wmin_A": float(np.nanmin(wave[good])) if np.any(good) else None,
                "wmax_A": float(np.nanmax(wave[good])) if np.any(good) else None,
                "wave_medium": segment.wave_medium,
                "observer_frame": segment.observer_frame,
                "stellar_rest_status": segment.stellar_rest_status,
            }
        )
    return rows


def _diagnostic_window_payload(segment):
    selection = select_diagnostic_windows(segment, max_windows=12)
    combinations = build_diagnostic_window_combinations(
        selection,
        max_windows=8,
        max_single_windows=6,
    )
    return {
        "selection": selection,
        "recommended_combinations": combinations,
        "note": (
            "These generic windows are selected from the uploaded spectral "
            "range before any expensive PHOENIX window-combination fits. They "
            "include optical, Paschen/Brackett, Ca, TiO, and CO diagnostics "
            "when the wavelength coverage supports them."
        ),
    }


def _base_payload(args, spectrum_path, segment, case, collection, exclude_masks):
    generic_windows = _diagnostic_window_payload(segment)
    return {
        "schema_version": 1,
        "workflow": "publication_quality_xshooter_uvb_scaffold",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "audit_started",
        "input": {
            "spectrum": str(spectrum_path),
            "instrument": str(args.instrument),
        },
        "analysis_design": {
            "phase": "scaffold",
            "default_claim_status": "exploratory_until_publication_checks_pass",
            "balmer_windows": case.provenance["balmer_windows"],
            "metal_rv_windows": list(METAL_RV_WINDOWS),
            "generic_diagnostic_windows": generic_windows,
            "core_mask_grid_A": _parse_core_mask_grid(args.core_mask_grid),
            "core_mask_information_policy": {
                "method": "explicit_sensitivity_grid_with_information_penalty",
                "minimum_retained_fraction": float(
                    args.min_core_mask_retained_fraction
                ),
                "penalty_scope": "diagnostic_model_selection_not_chi2_objective",
                "preferred_width_rule": (
                    "Treat width=0 as a no-core-mask reference, then prefer "
                    "the smallest nonzero width that is not flagged excessive. "
                    "Do not select a larger mask merely because it lowers "
                    "in-window chi-square after discarding useful wing pixels."
                ),
                "literature_basis": [
                    "cappellari2023_ppxf_optimization",
                    "garcia_perez2016_aspcap",
                    "mashonkina2008_balmer_core_exclusion",
                ],
            },
            "core_mask_evaluation_note": (
                "Balmer-core masking is a sensitivity axis, not a hard-coded "
                "truth. Audit the grid first; run fit-grid variants only when "
                "the extra PHOENIX cost is justified."
            ),
            "followup_steps_before_publication_claims": list(PUBLICATION_FOLLOWUP_STEPS),
        },
        "balmer_case": case.provenance,
        "fit_segments": _fit_window_summary(collection.segments),
        "exclude_masks": {
            "names": [mask.name for mask in exclude_masks],
            "metadata": {mask.name: dict(mask.metadata) for mask in exclude_masks},
            "archive_mask_policy": args.archive_mask_policy,
        },
        "ordinary_readiness": None,
        "publication_readiness": None,
        "core_mask_sensitivity": None,
        "core_mask_sensitivity_recommendation": None,
        "baseline_fit": None,
    }


def _write_diagnostic_window_csv(path, diagnostic_payload):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = diagnostic_payload["selection"]["selected"]
    columns = [
        "id",
        "label",
        "region_A",
        "roles",
        "score",
        "n_usable_pixels",
        "usable_fraction",
        "feature_contrast",
        "risk_tags",
        "spectral_type_hint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "region_A": "{0:.2f}-{1:.2f}".format(*row["region_A"]),
                    "roles": ";".join(row.get("roles", ())),
                    "score": "{0:.5g}".format(float(row.get("score", 0.0))),
                    "n_usable_pixels": int(row.get("n_usable_pixels", 0)),
                    "usable_fraction": "{0:.5g}".format(
                        float(row.get("usable_fraction", 0.0))
                    ),
                    "feature_contrast": "{0:.5g}".format(
                        float(row.get("feature_contrast", 0.0))
                    ),
                    "risk_tags": ";".join(row.get("risk_tags", ())),
                    "spectral_type_hint": row.get("spectral_type_hint", ""),
                }
            )


def _core_mask_fit_value(record, key):
    fit = record.get("fit") or {}
    value = fit.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _write_core_mask_comparison_csv(path, records):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "core_mask_halfwidth_A",
        "n_fit_candidate",
        "information_retention_fraction",
        "additional_information_loss_fraction",
        "core_mask_information_penalty",
        "excessive_core_mask",
        "recommended_core_mask",
        "rejected_inside_fit_window_fraction",
        "publication_ready",
        "publication_blockers",
        "teff",
        "logg",
        "feh",
        "rv_kms",
        "chi2_red",
        "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            fit = record.get("fit") or {}
            writer.writerow(
                {
                    "core_mask_halfwidth_A": record["core_mask_halfwidth_A"],
                    "n_fit_candidate": record["n_fit_candidate"],
                    "information_retention_fraction": record.get(
                        "information_retention_fraction"
                    ),
                    "additional_information_loss_fraction": record.get(
                        "additional_information_loss_fraction"
                    ),
                    "core_mask_information_penalty": record.get(
                        "core_mask_information_penalty"
                    ),
                    "excessive_core_mask": record.get("excessive_core_mask"),
                    "recommended_core_mask": record.get("recommended_core_mask"),
                    "rejected_inside_fit_window_fraction": record[
                        "rejected_inside_fit_window_fraction"
                    ],
                    "publication_ready": record["publication_ready"],
                    "publication_blockers": ";".join(record["publication_blockers"]),
                    "teff": _core_mask_fit_value(record, "teff"),
                    "logg": _core_mask_fit_value(record, "logg"),
                    "feh": _core_mask_fit_value(record, "feh"),
                    "rv_kms": _core_mask_fit_value(record, "rv_kms"),
                    "chi2_red": _core_mask_fit_value(record, "chi2_red"),
                    "quality_flags": ";".join(fit.get("quality_flags", ())),
                }
            )


def _write_core_mask_comparison_plot(path, records):
    if path is None:
        return
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    widths = np.asarray([item["core_mask_halfwidth_A"] for item in records], dtype=float)
    nfit = np.asarray([item["n_fit_candidate"] for item in records], dtype=float)
    rejected = np.asarray(
        [item["rejected_inside_fit_window_fraction"] for item in records],
        dtype=float,
    )
    teff = np.asarray([_core_mask_fit_value(item, "teff") for item in records], dtype=float)
    logg = np.asarray([_core_mask_fit_value(item, "logg") for item in records], dtype=float)
    chi2 = np.asarray(
        [_core_mask_fit_value(item, "chi2_red") for item in records],
        dtype=float,
    )
    retained = np.asarray(
        [item.get("information_retention_fraction") for item in records],
        dtype=float,
    )
    penalty = np.asarray(
        [item.get("core_mask_information_penalty") for item in records],
        dtype=float,
    )
    recommended_width = _recommended_core_mask_width(records)
    has_fit = np.any(np.isfinite(teff)) or np.any(np.isfinite(chi2))
    nrows = 4 if has_fit else 3
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, 3.0 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    axes[0].plot(widths, nfit, marker="o")
    axes[0].set_ylabel("fitted pixels")
    axes[0].grid(alpha=0.25)
    axes[1].plot(widths, rejected, marker="o", color="tab:orange")
    axes[1].set_ylabel("rejected inside")
    axes[1].grid(alpha=0.25)
    axes[2].plot(widths, retained, marker="o", color="tab:green", label="retained")
    axes[2].plot(widths, penalty, marker="s", color="tab:red", label="penalty")
    axes[2].set_ylabel("information")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)
    if recommended_width is not None:
        for ax in axes:
            ax.axvline(recommended_width, color="tab:green", ls=":", alpha=0.45)
    if has_fit:
        if np.any(np.isfinite(teff)):
            axes[3].plot(widths, teff, marker="o", label="Teff")
            axes[3].set_ylabel("Teff [K]")
            ax_r = axes[3].twinx()
            if np.any(np.isfinite(logg)):
                ax_r.plot(widths, logg, marker="s", color="tab:green", label="logg")
                ax_r.set_ylabel("logg")
        if np.any(np.isfinite(chi2)):
            ax_chi = axes[3].twinx()
            ax_chi.spines.right.set_position(("outward", 54))
            ax_chi.plot(widths, chi2, marker="^", color="tab:red", label="χ²ν")
            ax_chi.set_ylabel("χ²ν")
        axes[3].grid(alpha=0.25)
    axes[-1].set_xlabel("Balmer-core mask half-width [Å]")
    fig.suptitle(
        "Balmer-core mask sensitivity "
        "(penalty is diagnostic; it is not added to χ²)"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _recommended_core_mask_width(records):
    for record in records:
        if record.get("recommended_core_mask"):
            return float(record["core_mask_halfwidth_A"])
    return None


def _run_readiness(args, collection, exclude_masks):
    resolution = _resolution_assumption(args)
    ordinary = audit_spectrum_for_fit(
        collection,
        exclude_masks=exclude_masks,
        intended_use="publication_scaffold_baseline",
        assumed_resolution=resolution,
    )
    publication = publication_readiness_audit(
        collection,
        exclude_masks=exclude_masks,
        intended_use="publication_quality_stellar_parameters",
        assumed_resolution=resolution,
        min_fit_pixels=args.min_fit_pixels,
        max_rejected_inside_fit_window_fraction=(
            args.max_rejected_inside_fit_window_fraction
        ),
        allow_assumed_resolution=args.allow_assumed_resolution,
    )
    return ordinary, publication


def _fit_kwargs_from_args(args, collection, exclude_masks):
    extra = {
        "exclude_masks": tuple(exclude_masks),
        "forward_model": "native_interp",
        "rv_init": "grid",
        "rv_grid_n": int(args.rv_grid_n),
        "multistart": int(args.multistart),
        "mdeg": int(args.mdeg),
        "max_nfev": int(args.max_nfev),
    }
    return prepare_phoenix_fit_kwargs(
        collection,
        auto_defaults=True,
        defaults_mode=args.baseline_defaults_mode,
        science_case="publication_scaffold",
        resolution_R=args.resolution_R,
        p0_overrides=(args.teff, args.feh, args.logg, args.rv),
        lower_bound_overrides=(args.teff_min, args.feh_min, args.logg_min, args.rv_min),
        upper_bound_overrides=(args.teff_max, args.feh_max, args.logg_max, args.rv_max),
        extra_kwargs=extra,
    )


def _run_baseline_fit(args, collection, exclude_masks, *, output_plot=None):
    fit_kwargs, suggestion = _fit_kwargs_from_args(args, collection, exclude_masks)
    print("Running baseline native-grid PHOENIX fit...", flush=True)
    result = fit_stellar_spectrum(
        collection,
        model="phoenix",
        phoenix_dir=args.phoenix_dir,
        auto_defaults=False,
        science_case="publication_scaffold",
        progress_callback=lambda event: print(event, flush=True),
        **fit_kwargs,
    )
    if suggestion is not None:
        result.summary["fit_default_suggestion"] = suggestion.to_dict()
    plot_paths = {}
    if output_plot:
        print("Writing baseline referee plot...", flush=True)
        fig, _axes = plot_fit_referee(
            result,
            segment=collection,
            savepath=output_plot,
            layout="stacked",
            xlim_mode="fit",
            flux_ylim_mode="visible",
            show_raw_model=False,
            figsize_per_segment=(15.0, 5.4),
        )
        plot_paths.update(getattr(fig, "spyctres_generated_files", {}) or {})
    return result.to_dict(
        include_arrays=False,
        plot_paths=plot_paths or None,
        relative_to=Path(args.output_json).parent,
        include_local_paths=False,
    )


def _summarize_core_mask_variant(width, ordinary, publication, fit_payload=None):
    return {
        "core_mask_halfwidth_A": float(width),
        "mask_enabled": bool(width > 0.0),
        "n_inside_fit_window": int(ordinary["n_inside_fit_window"]),
        "n_fit_candidate": int(ordinary["n_fit_candidate"]),
        "outside_fit_window_fraction": ordinary["outside_fit_window_fraction"],
        "rejected_inside_fit_window_fraction": (
            ordinary["rejected_inside_fit_window_fraction"]
        ),
        "interpretation_flags": list(ordinary["interpretation_flags"]),
        "publication_ready": bool(publication["publication_ready"]),
        "publication_blockers": list(publication["blockers"]),
        "segment_fit_pixels": [
            {
                "name": item["name"],
                "n_inside_fit_window": item["n_inside_fit_window"],
                "n_fit_candidate": item["n_fit_candidate"],
                "rejected_inside_fit_window_fraction": (
                    item["rejected_inside_fit_window_fraction"]
                ),
            }
            for item in ordinary["segments"]
        ],
        "fit": fit_payload,
    }


def _annotate_core_mask_information(records, *, min_retained_fraction):
    # Literature basis: Balmer-wing analyses commonly exclude only the
    # model-sensitive core while preserving the wings that carry Teff/logg
    # leverage; see Mashonkina et al. 2008,
    # https://doi.org/10.1051/0004-6361:20078060.
    if not records:
        return {
            "status": "no_core_mask_grid",
            "recommended_core_mask_halfwidth_A": None,
        }
    min_retained_fraction = float(min_retained_fraction)
    if not np.isfinite(min_retained_fraction) or not (0.0 < min_retained_fraction <= 1.0):
        raise ValueError("--min-core-mask-retained-fraction must be in (0, 1].")
    reference = next(
        (record for record in records if float(record["core_mask_halfwidth_A"]) == 0.0),
        None,
    )
    if reference is None:
        reference = max(records, key=lambda item: int(item["n_fit_candidate"]))
    reference_nfit = max(int(reference["n_fit_candidate"]), 1)
    reference_rejected = float(reference["rejected_inside_fit_window_fraction"])
    for record in records:
        nfit = int(record["n_fit_candidate"])
        retained = float(np.clip(nfit / reference_nfit, 0.0, 1.0))
        added_loss = float(max(0.0, 1.0 - retained))
        added_rejected = float(
            max(
                0.0,
                float(record["rejected_inside_fit_window_fraction"]) - reference_rejected,
            )
        )
        penalty = float(max(added_loss, added_rejected))
        record["core_mask_reference_halfwidth_A"] = float(
            reference["core_mask_halfwidth_A"]
        )
        record["reference_n_fit_candidate"] = int(reference_nfit)
        record["information_retention_fraction"] = retained
        record["additional_information_loss_fraction"] = added_loss
        record["additional_rejected_inside_fit_window_fraction"] = added_rejected
        record["core_mask_information_penalty"] = penalty
        record["excessive_core_mask"] = bool(retained < min_retained_fraction)
        record["core_mask_penalty_note"] = (
            "Diagnostic only: this penalty is not added to the optimizer's "
            "spectral chi-square. It helps compare mask-width variants so "
            "large masks do not look artificially attractive just because they "
            "discard difficult Balmer-wing pixels."
        )
        record["recommended_core_mask"] = False

    nonzero_eligible = [
        record
        for record in records
        if float(record["core_mask_halfwidth_A"]) > 0.0
        and not record["excessive_core_mask"]
        and int(record["n_fit_candidate"]) > 0
    ]
    if nonzero_eligible:
        recommended = min(
            nonzero_eligible,
            key=lambda item: (
                int(len(item.get("publication_blockers", ()))),
                float(item["core_mask_information_penalty"]),
                float(item["core_mask_halfwidth_A"]),
            ),
        )
        recommended["recommended_core_mask"] = True
        width = float(recommended["core_mask_halfwidth_A"])
        reason = (
            "smallest/best-retained nonzero core mask that is not flagged "
            "as excessive under the information-retention threshold"
        )
    else:
        width = None
        reason = "no nonzero core-mask width passed the information-retention threshold"
    return {
        "status": "evaluated",
        "reference_core_mask_halfwidth_A": float(reference["core_mask_halfwidth_A"]),
        "reference_n_fit_candidate": int(reference_nfit),
        "minimum_retained_fraction": float(min_retained_fraction),
        "recommended_core_mask_halfwidth_A": width,
        "recommendation_reason": reason,
        "selection_warning": (
            "This recommendation is for mask-width triage only. Final choice "
            "still requires visual residual checks, line-by-line stability, "
            "LSF validation, and synthetic recovery tests."
        ),
    }


def _run_core_mask_sensitivity(args, segment):
    widths = _parse_core_mask_grid(args.core_mask_grid)
    if not widths:
        return {
            "records": [],
            "recommendation": {
                "status": "no_core_mask_grid",
                "recommended_core_mask_halfwidth_A": None,
            },
        }
    print(
        "Auditing Balmer-core mask sensitivity for widths: {0}".format(
            ", ".join("{0:g}".format(width) for width in widths)
        ),
        flush=True,
    )
    records = []
    for width in widths:
        _case_i, collection_i, exclude_masks_i = _prepare_balmer_collection(
            args,
            segment,
            core_mask_halfwidth=width,
        )
        ordinary_i, publication_i = _run_readiness(
            args,
            collection_i,
            exclude_masks_i,
        )
        fit_payload = None
        if args.run_core_mask_fit_grid:
            print(
                "Running core-mask fit variant: halfwidth={0:g} A".format(width),
                flush=True,
            )
            fit_payload = _run_baseline_fit(
                args,
                collection_i,
                exclude_masks_i,
                output_plot=None,
            )
        records.append(
            _summarize_core_mask_variant(
                width,
                ordinary_i,
                publication_i,
                fit_payload=fit_payload,
            )
        )
    recommendation = _annotate_core_mask_information(
        records,
        min_retained_fraction=args.min_core_mask_retained_fraction,
    )
    print(
        "Core-mask recommendation: {0} A ({1})".format(
            recommendation["recommended_core_mask_halfwidth_A"],
            recommendation["recommendation_reason"],
        ),
        flush=True,
    )
    return {
        "records": records,
        "recommendation": recommendation,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_path = Path(args.output_json)
    if args.resume and output_path.exists() and not args.force:
        existing = _read_existing(output_path)
        print(
            "Existing checkpoint: status={0}, publication_ready={1}".format(
                existing.get("status"),
                existing.get("publication_readiness", {}).get("publication_ready"),
            ),
            flush=True,
        )
        return 0

    print("Reading X-SHOOTER UVB spectrum...", flush=True)
    spectrum = read_spectrum(args.spectrum, instrument=args.instrument)
    segment = _single_segment(spectrum)

    print("Preparing explicit Balmer-window scaffold...", flush=True)
    case, collection, exclude_masks = _prepare_balmer_collection(
        args,
        segment,
        core_mask_halfwidth=float(args.balmer_core_mask),
    )
    payload = _base_payload(args, args.spectrum, segment, case, collection, exclude_masks)
    _write_diagnostic_window_csv(
        args.output_diagnostic_window_csv,
        payload["analysis_design"]["generic_diagnostic_windows"],
    )
    _atomic_write_json(output_path, payload)
    print("Wrote scaffold checkpoint: {0}".format(output_path), flush=True)

    print("Running ordinary and publication-readiness audits...", flush=True)
    ordinary, publication = _run_readiness(args, collection, exclude_masks)
    payload["ordinary_readiness"] = ordinary
    payload["publication_readiness"] = publication
    payload["status"] = (
        "publication_gate_passed_needs_systematics"
        if publication["publication_ready"]
        else "exploratory_publication_gate_blocked"
    )
    _atomic_write_json(output_path, payload)
    print(
        "Publication readiness: ready={0}, blockers={1}".format(
            publication["publication_ready"],
            ", ".join(publication["blockers"]) or "none",
        ),
        flush=True,
    )

    core_mask_result = _run_core_mask_sensitivity(args, segment)
    payload["core_mask_sensitivity"] = core_mask_result["records"]
    payload["core_mask_sensitivity_recommendation"] = core_mask_result[
        "recommendation"
    ]
    _write_core_mask_comparison_csv(
        args.output_comparison_csv,
        payload["core_mask_sensitivity"],
    )
    _write_core_mask_comparison_plot(
        args.output_comparison_plot,
        payload["core_mask_sensitivity"],
    )
    _atomic_write_json(output_path, payload)

    if args.run_baseline_fit:
        payload["baseline_fit"] = _run_baseline_fit(
            args,
            collection,
            exclude_masks,
            output_plot=args.output_plot,
        )
        payload["status"] = (
            "baseline_fit_completed_needs_publication_systematics"
            if payload["baseline_fit"].get("success")
            else "baseline_fit_failed"
        )
        _atomic_write_json(output_path, payload)
        summary = {
            key: payload["baseline_fit"].get(key)
            for key in ("success", "teff", "feh", "logg", "rv_kms", "chi2_red")
        }
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(
            "Audit-only run complete. Add --run-baseline-fit after PHOENIX "
            "configuration is ready.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
