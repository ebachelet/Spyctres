# Spyctres 0.5.0a1 external-testing checkpoint

This document records the current Spyctres 0.5.0a1 reviewable alpha checkpoint
before larger structural refactoring or SED/microlensing-source work begins.

This is a collaborator-testing checkpoint, not a stable release. It is meant to
let Etienne and a small number of scientific reviewers exercise the current
public workflow and comment on usability, scientific assumptions, diagnostics,
and missing pieces.

## Purpose

The checkpoint preserves the present PHOENIX-backed classification and
diagnostic behavior as the control state for later refactoring. The checkpoint
does not intentionally change optimizer algorithms, fitting defaults, PHOENIX
interpolation, RV conventions, LSF behavior, masking, continuum treatment,
segment weighting, wavelength/frame semantics, or result schemas.

## Current public workflow

```python
import Spyctres as sp

spec = sp.read_spectrum("my_spectrum.fits", reader="xshooter_merge1d")
setup = sp.suggest_fit_setup(spec)
result = sp.fit_stellar_spectrum(spec, model="phoenix", setup=setup)
sp.plot_fit_referee(result)
```

Use `reader=` for new code. The older `instrument=` keyword is still accepted
as a deprecated compatibility alias where appropriate.

## What currently works

The verified development branch currently supports:

- reading supported reduced spectra into `SpectrumSegment` or
  `SpectrumCollection`;
- explicit wavelength medium, observer-frame, stellar-rest, uncertainty, valid
  mask, and resolution metadata;
- diagnostic window selection and known-line lookup;
- warning and exclusion mask provenance;
- setup suggestions and readiness audits;
- local PHOENIX HiRes interpolation from a user-supplied PHOENIX directory;
- single- and multi-segment PHOENIX fitting;
- radial-velocity fitting with the standard astronomical sign convention;
- constant Gaussian instrumental broadening in log wavelength;
- per-segment multiplicative continuum treatment;
- result diagnostics, quality flags, reconstruction, plotting, and JSON/report
  serialization;
- batch quickscan/refine and validation workflows;
- setup diagnosis through `spyctres doctor`.

PHOENIX fitting requires an external local PHOENIX HiRes installation. It is
not bundled with Spyctres.

## Recommended first tests for reviewers

Start with the core examples before trying the advanced validation workflows.

```bash
spyctres doctor --skip-phoenix
python examples/example1_quickstart.py --no-show
python examples/example2_lines_windows_and_masks.py --no-show
python examples/example3_improving_a_phoenix_fit.py --no-show
```

After configuring PHOENIX:

```bash
export SPYCTRES_PHOENIX_DIR=/path/to/PHOENIXv2
spyctres doctor --require-phoenix --skip-phoenix-scan
python scripts/phoenix_smoketest.py
python scripts/fitting_smoketest.py
python examples/example1_quickstart.py --run-fit --no-show
```

Then continue with:

```bash
python examples/example4_reviewed_balmer_analysis.py --no-show
python examples/example4b_balmer_stability_checks.py --no-show
python examples/example5_batch_fitting.py --dry-run
```

Examples 6-8 are advanced or validation-oriented:

- Example 6: multi-arm X-SHOOTER classification;
- Example 7: XSL DR3 reference-star validation discipline;
- Example 8: PEPSI legacy line-window regression.

## What remains exploratory

The following are not yet validated as general precision-inference workflows:

- empirical accuracy of `Teff`, `[Fe/H]`, and `logg` across all spectral types;
- Gaia21ccu parameter inference beyond reviewed exploratory diagnostics;
- population-level XSL validation;
- PEPSI legacy line-window behavior outside its regression role;
- local Jacobian covariance for imperfect real spectra;
- correlated spectral noise and model systematics;
- rotational/macroturbulent broadening where not explicitly modelled;
- detailed abundance variations outside the current atmosphere grid;
- complex, empirical, or wavelength-dependent LSF fitting;
- physical-flux SED fitting;
- extinction, angular-radius, microlensing magnification, and source/blend
  decomposition in the modern PHOENIX workflow.

## Known alpha architectural limitations

The current alpha still has internal issues planned for later structural work:

- broad root exports retained for compatibility;
- legacy and modern code paths remain coupled;
- some large scientific modules still need later splitting;
- optional legacy SED dependencies are still in the main dependency set;
- the package keeps a `setuptools<81` runtime constraint for `pysynphot`;
- active wavelength-dependent LSF support is intentionally deferred.

These limitations should be visible to reviewers, but this checkpoint is meant
to review current behavior before changing architecture.

## Validation table

Validation environment:

- Python: 3.12.13
- NumPy: 1.26.4
- SciPy: 1.16.3
- Astropy: 7.2.1
- Spyctres: 0.5.0a1

Checkpoint validation performed on 2026-08-08.

| Check | Status | Notes |
| --- | --- | --- |
| Python compilation | PASS | `python -m compileall -q Spyctres scripts examples` |
| `git diff --check` | PASS | whitespace/artifact sanity |
| Full Python 3.12 tests | PASS | `447 passed, 15 warnings in 41.56 s` |
| Wheel build | PASS | `python -m build SOURCE --outdir /tmp/spyctres_build_check_050a1`; built `spyctres-0.5.0a1-py3-none-any.whl` |
| sdist build | PASS | same build produced `spyctres-0.5.0a1.tar.gz` |
| Clean wheel import | PASS | temporary venv import printed `0.5.0a1` |
| CLI help from wheel | PASS | `spyctres --help` |
| `spyctres doctor --skip-phoenix` | PASS | installed-wheel check passed; source-checkout check also passed with a PATH warning in the sandbox shell |
| PHOENIX discovery | PASS | `spyctres doctor --require-phoenix --skip-phoenix-scan` found the configured PHOENIX directory and wavelength file |
| PHOENIX smoke test | PASS | `python scripts/phoenix_smoketest.py` |
| Synthetic fitting smoke test | PASS | `python scripts/fitting_smoketest.py` |
| Multi-segment fit | PASS WITH CAVEAT | maintained tracked regression `tests/test_fitting_initialization.py::test_multisegment_weighted_chi2_and_dof_accounting` passed; no tracked `scripts/multisegment_fit_smoketest.py` is included in this checkpoint |
| Example 1 | PASS | `python examples/example1_quickstart.py --no-show`; fitted path also passed with `--run-fit` and wrote JSON/plot to `/tmp` |
| Example 2 | PASS | `python examples/example2_lines_windows_and_masks.py --no-show` |
| Example 3 | PASS | `python examples/example3_improving_a_phoenix_fit.py --no-show` |
| Example 4A | PASS | `python examples/example4_reviewed_balmer_analysis.py --no-show` |
| Example 4B | PASS | `python examples/example4b_balmer_stability_checks.py --no-show` |
| Example 5 | PASS | `python examples/example5_batch_fitting.py --dry-run` |
| Example 6 | PASS | advanced multi-arm, no-fit path |
| Example 7 | PASS | XSL validation lightweight path |
| Example 8 | PASS | PEPSI legacy line-window preparation path |

Two build-environment notes:

- `python -m build` was run from `/tmp` with the source path because an ignored
  local `build/` directory in the source checkout shadows the PyPA `build`
  module when invoked directly from the repository root.
- The isolated build emitted a setuptools deprecation warning about license
  classifiers. This is a packaging-cleanup item for later; it did not prevent
  wheel/sdist creation.

## Feedback requested

Please comment specifically on:

- whether installation and `spyctres doctor` are understandable;
- whether the public workflow and function names are clear;
- whether Examples 1-5 form a sensible beginner path;
- whether advanced/validation examples are clearly distinguished;
- wavelength medium, observer/rest-frame, barycentric, and RV sign conventions;
- instrumental resolving-power and LSF treatment;
- mask polarity and telluric/non-stellar-feature handling;
- per-segment continuum treatment and segment weighting;
- whether readiness audits and quality flags are too strict, too permissive, or
  about right;
- whether plots and serialized outputs contain the right information;
- runtime, stalls, progress feedback, memory issues, and confusing errors;
- which legacy Spyctres behavior must remain before beta testing.

## After this checkpoint

After review, development should proceed in small milestones:

1. safe imports and installation hygiene;
2. acyclic scientific foundation;
3. fitting/result internals;
4. presentation, legacy, data, and documentation cleanup;
5. modern physical-flux and microlensing-source SED inference.

Do not start those structural changes until this checkpoint has been reviewed.
