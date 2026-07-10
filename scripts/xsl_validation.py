"""Batch PHOENIX validation against locally downloaded XSL DR3 spectra.

Example
-------
python scripts/xsl_validation.py examples/xsl_validation_manifest.csv \
  --output /tmp/xsl_validation_results.json
"""

import argparse
import csv
import json
import os
import tempfile

import numpy as np

from Spyctres import fit_phoenix_spectrum
from Spyctres.io import SpectrumCollection, read_spectrum


ROLE_BUDGETS = {
    "standard": {
        "coarse_init": True,
        "multistart": 4,
        "max_nfev": 100,
        "coarse_decimate": 20,
        "statistics_group": "ordinary_recovery",
    },
    "diagnostic": {
        "coarse_init": True,
        "multistart": 2,
        "max_nfev": 50,
        "coarse_decimate": 30,
        "statistics_group": "diagnostic_only",
    },
    "unsupported": {
        "coarse_init": False,
        "multistart": 1,
        "max_nfev": 0,
        "coarse_decimate": 30,
        "statistics_group": "diagnostic_only",
    },
}

ROLE_BUDGET_KEYS = {
    "standard": "standard",
    "hot_boundary": "diagnostic",
    "unsupported_hot": "unsupported",
    "cool_stress": "diagnostic",
    "peculiar_stress": "diagnostic",
    "carbon_star": "diagnostic",
    "emission_line": "diagnostic",
    "variable_or_uncertain": "diagnostic",
    # Backward-compatible aliases for older local checkpoints/manifests.
    "unsupported_control": "unsupported",
    "model_stress_test": "diagnostic",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate PHOENIX classifications against XSL DR3 stars.",
        epilog=(
            "Example:\n  python scripts/xsl_validation.py "
            "examples/xsl_validation_manifest.csv "
            "--output /tmp/xsl_validation_results.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", help="CSV containing path and reference parameters.")
    parser.add_argument("--phoenix-dir", default=None)
    parser.add_argument(
        "--wave-medium",
        default="air",
        choices=["air", "vacuum"],
        help="XSL DR3 uses air wavelengths; override only for a transformed file.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--teff-default", type=float, default=5750.0)
    parser.add_argument("--feh-default", type=float, default=0.0)
    parser.add_argument("--logg-default", type=float, default=4.0)
    parser.add_argument("--mdeg", type=int, default=2)
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Override the manifest-role evaluation budget for every selected target.",
    )
    parser.add_argument("--wave-min", type=float, default=4000.0)
    parser.add_argument("--wave-max", type=float, default=9000.0)
    parser.add_argument(
        "--neutral-initialization",
        action="store_true",
        help=(
            "Use neutral p0 values instead of the literature values. This tests "
            "local convergence only; it is not a broad-grid blind classifier."
        ),
    )
    parser.add_argument(
        "--coarse-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override role-aware coarse initialization for every selected target.",
    )
    parser.add_argument(
        "--multistart",
        type=int,
        default=None,
        help="Override the role-aware number of shared-interpolator local starts.",
    )
    parser.add_argument("--coarse-decimate", type=int, default=None)
    parser.add_argument(
        "--plot-points-per-segment",
        type=int,
        default=4000,
        help=(
            "Maximum observed/model samples saved per segment for validation "
            "plots; use 0 to omit plot samples."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed targets in an existing checkpoint and retain all results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected targets even when --resume finds completed records.",
    )
    parser.add_argument(
        "--xsl-id",
        action="append",
        default=None,
        help="Process only this XSL ID; repeat to select multiple IDs.",
    )
    return parser


def _optional_float(row, key, default=None):
    value = str(row.get(key, "")).strip()
    return float(value) if value else default


def _reference_float(row, key):
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError("Manifest row {0!r} requires {1}.".format(row.get("xsl_id"), key))
    return float(value)


def _window_collection(collection, wave_min, wave_max):
    segments = []
    for segment in collection.segments:
        keep = (segment.wave >= wave_min) & (segment.wave <= wave_max)
        if np.any(keep):
            segments.append(segment.subset(keep, name=segment.name))
    if not segments:
        raise ValueError(
            "No XSL samples fall inside {0:g}-{1:g} Angstrom.".format(
                wave_min, wave_max
            )
        )
    return SpectrumCollection(
        segments,
        meta=dict(collection.meta),
        name=collection.name,
    )


def _resolve_manifest_path(manifest, value):
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(manifest)), value))


def _role_budget(row, args):
    role = str(row.get("validation_role", "standard")).strip().lower()
    base = ROLE_BUDGETS[ROLE_BUDGET_KEYS.get(role, "diagnostic")]
    budget = dict(base)
    if args.coarse_init is not None:
        budget["coarse_init"] = bool(args.coarse_init)
    if args.multistart is not None:
        budget["multistart"] = int(args.multistart)
    if args.max_nfev is not None:
        budget["max_nfev"] = int(args.max_nfev)
    if args.coarse_decimate is not None:
        budget["coarse_decimate"] = int(args.coarse_decimate)
    if budget["multistart"] < 1:
        raise ValueError("multistart budgets must be >= 1.")
    if budget["max_nfev"] < 0:
        raise ValueError("max_nfev budgets must be >= 0.")
    if budget["coarse_decimate"] < 1:
        raise ValueError("coarse_decimate budgets must be >= 1.")
    return role, budget


def _validate_manifest_rows(rows):
    """Require stable, unique target identifiers for resumable checkpoints."""
    seen = set()
    for index, row in enumerate(rows):
        xsl_id = str(row.get("xsl_id", "")).strip()
        if not xsl_id:
            raise ValueError(
                "Manifest row {0} requires a non-empty xsl_id for resumable "
                "validation.".format(index + 1)
            )
        key = xsl_id.upper()
        if key in seen:
            raise ValueError("Duplicate xsl_id in manifest: {0}".format(xsl_id))
        seen.add(key)


def _run_configuration(args):
    """Return checkpoint-critical settings that make resume scientifically safe."""
    return {
        "wave_medium": args.wave_medium,
        "fit_wave_range_A": [float(args.wave_min), float(args.wave_max)],
        "mdeg": int(args.mdeg),
        "neutral_initialization": bool(args.neutral_initialization),
        "coarse_init_override": args.coarse_init,
        "multistart_override": args.multistart,
        "max_nfev_override": args.max_nfev,
        "coarse_decimate_override": args.coarse_decimate,
    }


def _validate_resume_configuration(previous, current):
    previous_config = previous.get("run_configuration")
    if previous_config is None:
        return
    mismatches = [
        key
        for key, value in current.items()
        if previous_config.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Existing checkpoint was created with different validation settings "
            "({0}); use a different --output path.".format(
                ", ".join(sorted(mismatches))
            )
        )


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
    raise TypeError("Value is not JSON serializable: {0}".format(type(value).__name__))


def _ordinary_recovery_statistics(results):
    ordinary = [
        item
        for item in results
        if item.get("statistics_group") == "ordinary_recovery"
        and item.get("status") == "ok"
        and "delta" in item
    ]
    statistics = {"count": len(ordinary)}
    for key in ("teff", "logg", "feh"):
        values = np.asarray([item["delta"][key] for item in ordinary], dtype=float)
        statistics[key] = {
            "median_delta": None if values.size == 0 else float(np.median(values)),
            "median_absolute_delta": (
                None if values.size == 0 else float(np.median(np.abs(values)))
            ),
        }
    return statistics


def _validation_plot_payload(collection, fit_result, max_points_per_segment):
    """Return bounded observed/model samples for reproducible validation plots."""
    max_points_per_segment = int(max_points_per_segment)
    if max_points_per_segment < 0:
        raise ValueError("plot_points_per_segment must be >= 0.")
    if max_points_per_segment == 0:
        return None

    models = tuple(getattr(fit_result, "models", ()))
    used_masks = tuple(getattr(fit_result, "used_masks", ()))
    if len(models) != len(collection.segments):
        return None

    segments = []
    for index, (segment, model) in enumerate(zip(collection.segments, models)):
        wave = np.asarray(segment.wave, dtype=float)
        flux = np.asarray(segment.flux, dtype=float)
        model = np.asarray(model, dtype=float)
        if model.shape != wave.shape:
            raise ValueError("Reconstructed XSL model shape does not match its segment.")
        valid = np.flatnonzero(np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model))
        if valid.size == 0:
            continue
        if valid.size > max_points_per_segment:
            positions = np.unique(
                np.rint(
                    np.linspace(0, valid.size - 1, max_points_per_segment)
                ).astype(int)
            )
            selected = valid[positions]
        else:
            selected = valid
        if index < len(used_masks):
            used = np.asarray(used_masks[index], dtype=bool)
            if used.shape != wave.shape:
                raise ValueError("Reconstructed XSL used-mask shape does not match its segment.")
        else:
            used = np.asarray(segment.mask, dtype=bool)
        segments.append(
            {
                "name": segment.name,
                "arm": segment.meta.get("arm"),
                "wave_medium": segment.wave_medium,
                "observer_frame": segment.observer_frame,
                "stellar_rest_status": segment.stellar_rest_status,
                "lsf_sigma_kms": segment.meta.get("xsl_effective_lsf_sigma_kms"),
                "wave_A": wave[selected],
                "observed_flux": flux[selected],
                "model_flux": model[selected],
                "used": used[selected],
                "original_points": int(wave.size),
                "saved_points": int(selected.size),
            }
        )
    if not segments:
        return None
    return {
        "sampling": "uniform_index_from_finite_observed_model_pairs",
        "max_points_per_segment": max_points_per_segment,
        "display_defaults": {
            "scale_mode": "global",
            "available_scale_modes": ["global", "per_segment", "none"],
            "arm_scaling_applied_by_spyctres": False,
            "rv_correction_applied_by_spyctres": False,
            "notes": [
                (
                    "Use global scaling for XSL full-spectrum displays; "
                    "per-segment scaling is diagnostic only."
                ),
                (
                    "Segments are saved separately to preserve UVB/VIS/NIR "
                    "resolution metadata, not because Spyctres has realigned arms."
                ),
            ],
        },
        "segments": segments,
    }


def _atomic_write_json(path, payload):
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    native_payload = _json_native(payload)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{0}.".format(os.path.basename(path)), suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(native_payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.plot_points_per_segment < 0:
        raise ValueError("plot_points_per_segment must be >= 0.")
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    with open(args.manifest, "r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError("XSL validation manifest contains no spectra.")
    if "path" not in manifest_rows[0]:
        raise ValueError("XSL validation manifest requires a path column.")
    _validate_manifest_rows(manifest_rows)
    run_configuration = _run_configuration(args)
    manifest_order = [row.get("xsl_id", "") for row in manifest_rows]
    manifest_by_id = {row.get("xsl_id", ""): row for row in manifest_rows}
    rows = list(manifest_rows)
    if args.xsl_id:
        selected_ids = {value.strip().upper() for value in args.xsl_id}
        rows = [row for row in rows if row.get("xsl_id", "").strip().upper() in selected_ids]
        if not rows:
            raise ValueError("No manifest rows matched --xsl-id.")

    records_by_id = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        _validate_resume_configuration(previous, run_configuration)
        for item in previous.get("results", []):
            item = dict(item)
            xsl_id = item.get("xsl_id", "")
            row = manifest_by_id.get(xsl_id, {})
            role, budget = _role_budget(row or item, args)
            item["validation_role"] = role
            if "statistics_group" not in item:
                item["statistics_group"] = budget["statistics_group"]
            item.setdefault("validation_budget", dict(budget))
            initialization = item.get("initialization")
            if isinstance(initialization, dict):
                coarse = initialization.get("coarse")
                if isinstance(coarse, dict):
                    coarse.setdefault(
                        "candidates_complete",
                        len(coarse.get("candidates", []))
                        == coarse.get("candidates_evaluated", -1),
                    )
                solutions = initialization.get("local_solutions", [])
                tested_zero = any(
                    len(solution.get("start", [])) == 4
                    and np.isclose(
                        float(solution["start"][3]), 0.0, rtol=0.0, atol=1e-12
                    )
                    for solution in solutions
                )
                initialization.setdefault(
                    "stellar_rest_zero_rv_start_tested", bool(tested_zero)
                )
                initialization.setdefault(
                    "multistart_requested", initialization.get("multistart", 1)
                )
            records_by_id[xsl_id] = item

    def checkpoint():
        ordered = [
            records_by_id[xsl_id]
            for xsl_id in manifest_order
            if xsl_id in records_by_id
        ]
        known = set(manifest_order)
        ordered.extend(
            item for xsl_id, item in records_by_id.items() if xsl_id not in known
        )
        excluded_roles = sorted(
            {
                item.get("validation_role", "unknown")
                for item in ordered
                if item.get("statistics_group") != "ordinary_recovery"
            }
        )
        payload = {
            "schema_version": 2,
            "run_configuration": dict(run_configuration),
            "wave_medium_assumption": args.wave_medium,
            "fit_wave_range_A": [args.wave_min, args.wave_max],
            "budget_mode": "manifest_role_aware",
            "role_budget_defaults": ROLE_BUDGETS,
            "xsl_dichroic_regions_excluded_A": [
                [5450.0, 5900.0],
                [9940.0, 11500.0],
            ],
            "ordinary_recovery_statistics": _ordinary_recovery_statistics(ordered),
            "statistics_excluded_roles": excluded_roles,
            "results": ordered,
        }
        _atomic_write_json(args.output, payload)
        return payload

    completed_statuses = {"ok", "fit_failed", "unsupported_physics", "error"}
    processed_ids = []
    for index, row in enumerate(rows):
        xsl_id = row.get("xsl_id", "")
        if (
            args.resume
            and not args.force
            and xsl_id in records_by_id
            and records_by_id[xsl_id].get("status") in completed_statuses
        ):
            print("Skipping completed:", xsl_id)
            continue

        path = _resolve_manifest_path(args.manifest, row["path"])
        teff_ref = _reference_float(row, "teff_ref")
        feh_ref = _reference_float(row, "feh_ref")
        logg_ref = _reference_float(row, "logg_ref")
        role, budget = _role_budget(row, args)
        record = {
            "path": row["path"],
            "xsl_id": xsl_id,
            "star_name": row.get("star_name", ""),
            "spectral_type": row.get("spectral_type", ""),
            "validation_role": role,
            "statistics_group": budget["statistics_group"],
            "validation_budget": dict(budget),
            "reference_id": row.get("reference_id", ""),
            "notes": row.get("notes", ""),
            "reference": {
                "teff": teff_ref,
                "teff_err": _optional_float(row, "teff_err"),
                "feh": feh_ref,
                "feh_err": _optional_float(row, "feh_err"),
                "logg": logg_ref,
                "logg_err": _optional_float(row, "logg_err"),
            },
        }
        if teff_ref > 12000.0:
            record.update(
                status="unsupported_physics",
                message="Reference Teff exceeds the PHOENIX ACES 12000 K boundary.",
            )
            records_by_id[xsl_id] = record
            processed_ids.append(xsl_id)
            checkpoint()
            continue
        if int(budget["max_nfev"]) == 0:
            record.update(
                status="unsupported_physics",
                message=(
                    "Validation role is configured with max_nfev=0; target was "
                    "recorded without running an optimizer."
                ),
            )
            records_by_id[xsl_id] = record
            processed_ids.append(xsl_id)
            checkpoint()
            continue

        try:
            spectrum = read_spectrum(
                path,
                instrument="xsl_dr3",
                wave_medium=args.wave_medium,
                warn_unknown=False,
            )
            if not isinstance(spectrum, SpectrumCollection):
                raise TypeError("XSL DR3 reader must return a SpectrumCollection.")
            spectrum = _window_collection(spectrum, args.wave_min, args.wave_max)
            record["spectrum_provenance"] = {
                "instrument": spectrum.meta.get("instrument"),
                "xsl_release": spectrum.meta.get("xsl_release"),
                "xsl_combined_arms": spectrum.meta.get("xsl_combined_arms"),
                "xsl_flux_column": spectrum.meta.get("xsl_flux_column"),
                "xsl_log10_sampled": spectrum.meta.get("xsl_log10_sampled"),
                "xsl_header_provenance": spectrum.meta.get(
                    "xsl_header_provenance", {}
                ),
                "xsl_header_provenance_policy": spectrum.meta.get(
                    "xsl_header_provenance_policy"
                ),
                "xsl_arm_scaling_applied_by_spyctres": spectrum.meta.get(
                    "xsl_arm_scaling_applied_by_spyctres", False
                ),
                "xsl_rv_correction_applied_by_spyctres": spectrum.meta.get(
                    "xsl_rv_correction_applied_by_spyctres", False
                ),
            }
            cache_path = None
            if args.cache_dir:
                identifier = xsl_id or "row_{0:04d}".format(index)
                cache_path = os.path.join(args.cache_dir, identifier + ".npz")
            p0 = (
                (args.teff_default, args.feh_default, args.logg_default, 0.0)
                if args.neutral_initialization
                else (teff_ref, feh_ref, logg_ref, 0.0)
            )
            result = fit_phoenix_spectrum(
                spectrum,
                phoenix_dir=args.phoenix_dir,
                warn_unknown=False,
                p0=p0,
                exclude_regions=[(5450.0, 5900.0), (9940.0, 11500.0)],
                mdeg=args.mdeg,
                cache_path=cache_path,
                max_nfev=budget["max_nfev"],
                physical_init="coarse" if budget["coarse_init"] else None,
                coarse_decimate=budget["coarse_decimate"],
                multistart=budget["multistart"],
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            record.update(status="error", message=str(exc))
            records_by_id[xsl_id] = record
            processed_ids.append(xsl_id)
            checkpoint()
            continue
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
            initialization={
                "physical": result.get("physical_initialization"),
                "coarse": result.get("coarse_initialization"),
                "multistart": result.get("multistart", 1),
                "multistart_requested": result.get("multistart_requested", 1),
                "stellar_rest_zero_rv_start_tested": result.get(
                    "stellar_rest_zero_rv_start_tested", False
                ),
                "local_solutions": result.get("multistart_diagnostics", []),
            },
            quality_report=(
                result.quality_report()
                if hasattr(result, "quality_report")
                else result.get("quality_report")
            ),
        )
        plot_payload = _validation_plot_payload(
            spectrum, result, args.plot_points_per_segment
        )
        if plot_payload is not None:
            record["validation_plot"] = plot_payload
        records_by_id[xsl_id] = record
        processed_ids.append(xsl_id)
        checkpoint()

    payload = checkpoint()
    print("Processed targets:", len(processed_ids))
    statistics = payload["ordinary_recovery_statistics"]
    if statistics["count"]:
        print("Ordinary standard targets:", statistics["count"])
        print("Median delta Teff [K]:", statistics["teff"]["median_delta"])
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
