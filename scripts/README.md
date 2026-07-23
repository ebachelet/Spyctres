# Spyctres scripts

This directory contains development utilities, diagnostics, and validation
runners. These scripts are useful while developing Spyctres, but they are not
the polished public tutorial layer. For user-facing worked examples, start with
the files under `examples/`.

## Public examples

Public examples should live in `examples/`, not here. They are intended to be
read by new users and should have clear example calls, plots, and explanatory
text. If a workflow in `scripts/` becomes stable and educational enough for
normal users, promote a cleaned-up version to `examples/`.

## Setup and ingestion diagnostics

These scripts check whether a local installation, configuration, or input file
looks sensible before a fit is attempted.

- `check_spyctres_setup.py` checks the local Python environment, optional
  dependencies, PHOENIX path configuration, and available data/cache paths.
- `io_smoketest.py` reads reduced 1D spectra through the Spyctres I/O layer and
  prints normalized `SpectrumSegment` metadata. Use it to diagnose reader,
  wavelength-medium, frame, resolution, and mask metadata issues.
- `throughput_summary.py` reads a batch quickscan/refine JSON checkpoint and
  reports median/mean quick, refine, and total runtime, with a simple projection
  for larger batches. It does not run fits.

## Developer/regression smoke tests

These are compact engineering tests for developers. They are intentionally
shorter and more opinionated than full scientific validation runs. A successful
smoke test means that a workflow still executes and returns internally
consistent outputs; it is not by itself a stellar-classification validation.

- `phoenix_smoketest.py` checks PHOENIX template loading, small-grid
  interpolation, and cache round-tripping.
- `cache_rebuild_smoketest.py` checks cache invalidation/rebuild behaviour.
- `fitting_smoketest.py` fits a synthetic PHOENIX-generated spectrum and checks
  parameter recovery under controlled assumptions.
- `pepsi_fit_smoketest.py`, `xshooter_fit_smoketest.py`,
  `floyds_fit_smoketest.py`, and `gemini_fit_smoketest.py` exercise
  instrument-specific quicklook/regression workflows.
- `xshooter_notebook_scan_smoketest.py` is a notebook-faithful regression
  reference for the X-SHOOTER Balmer-window scan; it should not drift away from
  the maintained recipes without a deliberate reason.

Telluric policy for smoke tests: `--use-telluric-mask` means the
high-resolution transmission-threshold telluric model. Broad catalog telluric
regions are warning/provenance regions or explicit coarse fallbacks; they are
not the preferred fit mask when the transmission model is available.

## Validation runners

Validation runners compare Spyctres outputs against external reference data or
literature expectations. They are slower, more data-dependent, and more
scientific than smoke tests.

- `xsl_validation.py` runs batch validation against locally downloaded XSL DR3
  spectra using the manifest roles and validation budgets.
- `xsl_validation_plots.py` renders saved XSL validation JSON payloads into
  classification plots for inspection and review. It can also write compact
  Markdown/CSV/JSON/PNG reference-recovery summaries from the same JSON without
  rerunning PHOENIX. These summaries use publication-scaffold stability
  language and reviewer questions while keeping standard recovery targets
  separate from diagnostic/stress/unsupported targets.
- `external_spectra_validation.py` audits user-supplied SDSS/SEGUE and
  UVES-POP spectra through the common reader and fit-readiness layers. It writes
  resumable JSON/CSV summaries and optional audit plots. The default plot style
  is generic: raw flux, plotting-only robust local normalization, mask status,
  metadata warning regions, near-zero blocks, and suggested diagnostic windows.
  Archive/product regions appear only when a reader supplies that metadata. The
  script deliberately does not repair calibration, run PHOENIX fits, or
  introduce fitting presets for those archives.
- `diagnostic_window_audit.py` runs the PHOENIX-free diagnostic-window selector
  over a manifest, such as the bundled XSL Figure 1 validation manifest, and
  writes role-aware JSON/CSV summaries plus an optional top-window heatmap. Use
  it before changing diagnostic-window scores so standard targets and
  stress/peculiar targets remain separated.

Validation outputs can be large and should usually remain local unless they are
small, curated example assets that are intentionally added under `examples/`.
