# Spyctres development plan

This file tracks the project-level implementation order for the modular
PHOENIX-backed Spyctres workflow. It is intentionally separate from reviewer
handoffs or private notes; it records only durable development priorities that
belong in the repository.

## Current direction

Spyctres is being developed as a lightweight, user-facing spectroscopy package:
read a reduced one-dimensional spectrum, convert it into a common
`SpectrumSegment`/`SpectrumCollection` representation, make conservative
first-pass assumptions explicit, run a PHOENIX-backed classification fit, and
return structured diagnostics that say whether the result is exploratory or
ready for deeper analysis.

The public path should remain simple:

```python
from Spyctres import read_spectrum, suggest_fit_setup, fit_stellar_spectrum, plot_fit_referee

spec = read_spectrum("my_spectrum.fits", instrument="xshooter")
setup = suggest_fit_setup(spec)  # inspect windows, assumptions, readiness, and actions
setup.summary()
result = fit_stellar_spectrum(spec, model="phoenix", setup=setup)
plot_fit_referee(result)
```

Expert and publication-oriented workflows should build on the same components,
but they must not silently relax metadata, LSF, mask, or uncertainty
requirements.

The reviewed setup object is now passable to the fit itself:

```python
setup = suggest_fit_setup(spec)
result = fit_stellar_spectrum(spec, model="phoenix", setup=setup)
```

The fit result embeds the exact reviewed setup, including a setup hash and
readiness interpretation. The remaining work is to promote this into the
numbered examples and, later, the versioned report schema.

## Priority order

1. **Parsimonious alpha user journey.**
   Finish the out-of-the-box path before adding major new fitting features. The
   immediate acceptance target is that a collaborator can follow a small,
   numbered set of examples and understand the package order from filenames
   alone. The initial paired script/notebook scaffold is now in place:

   - `example1_quickstart.py` / `example1_quickstart.ipynb`
   - `example2_lines_windows_and_masks.py` /
     `example2_lines_windows_and_masks.ipynb`
   - `example3_improving_a_phoenix_fit.py` /
     `example3_improving_a_phoenix_fit.ipynb`
   - `example4_publication_quality_fitting.py` /
     `example4_publication_quality_fitting.ipynb`
   - `example5_batch_fitting.py` / `example5_batch_fitting.ipynb`

   Remaining work: enrich the notebook prose after user testing, keep heavy
   PHOENIX runs opt-in where possible, and gradually move validation/stress
   material out of the numbered beginner path into clearly labelled areas such
   as `examples/validation/` or `examples/advanced/`. User-facing examples
   should continue to use bundled spectra under `examples/data/` unless
   explicitly labelled as external-user validation.

2. **Reviewed setup as a first-class object.**
   Initial implementation is in place: `suggest_fit_setup()` returns a
   `FitSetup` mapping-compatible object with `summary()`, `summary_text()`,
   `to_dict()`, `to_json()`, and a stable setup hash, and
   `fit_stellar_spectrum(..., setup=setup)` embeds the exact reviewed setup in
   the result. Remaining work: use this object throughout the numbered
   examples, migrate any ad-hoc setup summaries to `setup.summary()`, and fold
   the setup hash into the later versioned report schema.

3. **Intent-aware readiness and preprocessing discipline.**
   Continue hardening the common spectrum boundary, mask polarity, native
   archive/product masks, formal-error handling, wavelength-medium/frame
   metadata, and explicit resolution/LSF provenance. The next readiness change
   is to make the audit task-dependent rather than assigning one global
   severity to every flag. Planned intents include:

   - `inspect`
   - `quicklook_classification`
   - `atmospheric_parameters`
   - `radial_velocity`
   - `publication`

   The audit should report `ready_for_intent`, `blockers_for_intent`,
   `warnings_for_intent`, and `actions_for_intent`. The same missing metadata
   may be allowed for visual inspection, allowed with caveats for quicklook
   classification, and blocked for physical RV or publication-quality
   inference. Existing `blocker`/`review` labels remain useful display labels,
   but should not be the only policy layer.

4. **Curated public facade and documentation source of truth.**
   Keep the top-level namespace useful but not sprawling. The intended one
   import facade should cover ordinary user actions:

   - read and plot a spectrum;
   - audit readiness;
   - select and plot diagnostic windows;
   - build a simple mask;
   - fit and plot one line;
   - run a first-pass PHOENIX fit;
   - plot and compare fit results.

   Review whether helper functions such as `readiness_flag_actions()` should
   remain top-level or become properties of structured audit/setup objects once
   those exist. Also clarify whether `classify_spectrum()` should remain a
   direct alias, return a distinct first-pass classification object, or be
   deprecated to avoid confusing atmospheric-parameter fits with formal MK
   classification.

   Public call help should ultimately be generated from real signatures,
   numpydoc-style docstrings, and registered examples rather than maintained as
   a parallel metadata layer.

5. **Versioned serialized reports and provenance.**
   Add a versioned, web/Django-friendly serialized report schema before
   external clients depend on ad-hoc JSON fields. The schema should record at
   least: schema version, Spyctres version, git commit if available, model
   backend, model-grid manifest/hash, input checksum when allowed, fit setup
   hash, wavelength assumptions, mask policy, resolution source, quality flags,
   readiness intent, and path-sanitization policy. PHOENIX composition
   provenance should be explicit: model family/version, selected abundance
   pattern, alpha policy, microturbulence policy if known, solar abundance
   scale, and whether `[Fe/H]` is an iron abundance, global metallicity, or a
   grid label.

6. **Real-spectrum validation before further feature growth.**
   Expand and maintain validation on real spectra before adding heavier
   modelling features. Use two tracks:

   - XSL/X-SHOOTER behavior and stability: ingestion, window selection, branch
     proposal, multi-arm behavior, residual morphology, and stress-case
     recognition.
   - Benchmark accuracy: Gaia FGK Benchmark Stars and other independently
     constrained standards for Teff/logg/[Fe/H]/RV where appropriate.

   Stress/peculiar targets should remain separated from ordinary recovery
   statistics. XSL recovery thresholds should be class-dependent and expressed
   where possible as normalized residuals with broad absolute floors, not as
   one universal pass/fail rule.

7. **Quickscan/refine calibration and throughput.**
   Continue the quality-gated quickscan/refine approach for batches of tens to
   hundreds of spectra, but measure more than runtime. Track false narrowing:
   the fraction of validation stars whose accepted broad solution lies inside
   the automatically proposed refinement bounds. Report this by spectral
   family, luminosity class, metallicity, S/N, resolution, instrument, and
   ordinary/stress role. Boundary hits, fallback errors, archive-mask overlap,
   structured residuals, or window inconsistency should widen refinement bounds
   or skip refinement rather than confidently narrowing.

8. **Publication-oriented X-SHOOTER UVB workflow.**
   Add a separate expert workflow, not a replacement for
   `examples/simple_phoenix_fit.py`, for cases where the user wants defensible
   stellar parameters rather than a first-pass classification. The first target
   is an X-SHOOTER UVB example using explicit Balmer and metal/RV windows,
   formal uncertainties, documented masks, constrained resolution assumptions,
   blind coarse likelihood mapping, independent multistart local fits, per-line
   and joint Balmer diagnostics, profile-likelihood checks, systematic variants,
   checkpoint/resume support, and provenance/uncertainty tables.

   This workflow is allowed to be slower and more verbose than the public
   examples. It should report "publication-oriented" only after metadata,
   LSF, masking, recovery, residual, and alternative-model checks pass. Until
   then it should label results as exploratory.

   Initial scaffold: `examples/publication_quality_xshooter_uvb.py` and
   `examples/publication_quality_xshooter_uvb.ipynb` provide an audit-first
   workflow with explicit Balmer segments, documented masks, publication
   readiness checks, a Balmer-core mask sensitivity audit grid, atomic JSON
   checkpoints, generic diagnostic-window provenance, optional compact
   mask-sensitivity CSV/PNG summaries, an explicit core-mask information-loss
   penalty/recommendation, a bounded fit-level systematic-variant plan for
   continuum degree, preparation normalization, Balmer-core mask, resolution
   assumption, and Balmer window-set sensitivity, an opt-in baseline PHOENIX
   fit, cheap per-line Balmer observed-profile diagnostics, baseline-fit
   per-line model-residual diagnostics, opt-in execution of a bounded subset of
   fit-level systematic variants with per-variant JSON checkpoints, opt-in
   same-model synthetic injection/recovery trials with per-trial checkpoints,
   compact Markdown/CSV/PNG publication-summary artifacts generated from the
   saved checkpoint without launching extra fits, conservative
   calibration-interpretation labels for readiness, Balmer-core mask
   sensitivity, window-set sensitivity, and same-model recovery, and a separate
   bounded diagnostic-window comparison scaffold. The comparison
   scaffold plans trusted-baseline, role-balanced, single-window,
   leave-one-out, and leave-one-family-out checks without ranking by raw
   chi-square alone; expensive fits stay opt-in. It now includes held-out
   residual proxies and same-window residual summaries when reconstructed
   model arrays are available, so users can compare parameter stability and
   residual behaviour without treating raw in-fit chi-square as a winner
   selector. The publication summary now includes bounded suggested next
   commands for the current checkpoint state, writing fresh follow-up JSON
   products instead of mutating reviewed checkpoints. The remaining work is to
   interpret/calibrate the selected
   systematic variants and same-model recovery behaviour on real spectra, add
   external reference-star recovery validation, profile scans, and final
   uncertainty tables. The first diagnostic-window catalog
   expansion now includes the planned early-type He/Mg/Si family, CH G-band,
   cool-dwarf CaH/K I/FeH/VO family, and K-band Na I/Ca I family with explicit
   model-support and risk policies. The next step is to calibrate window
   rankings, masks, and fit-comparison behaviour on real spectra rather than
   expanding the catalog mechanically.

9. **LSF, hot-regime, and model-support safeguards.**
   Keep constant Gaussian LSF broadening as the production path for now. Define
   but do not yet activate the SDSS wavelength-dependent LSF acceptance tests:
   exact `wdisp` interpretation, unit conversion, constant-LSF equivalence,
   synthetic line-width recovery, flux conservation, edge padding, invalid
   resolution entries, RV recovery, atmospheric-parameter deltas, and runtime
   behavior. Enrich resolution provenance with distinctions such as
   `reported_nominal`, `adopted_for_fit`, `measured_from_calibration`,
   `wavelength_dependent`, `fitted_nuisance`, and `unknown`.

   Keep the PHOENIX `Teff > 12000 K` guard. Also tag hot-star and peculiar-star
   diagnostic branches with model-support status (`supported`, `uncertain`,
   `stress_only`, `unsupported`) so such windows can identify likely regime
   mismatch without implying ordinary PHOENIX refinement is valid.

10. **Legacy, import, and packaging hygiene.**
    Before beta, produce a migration table for legacy functions: current
    replacement, status, deprecation release, removal target, and optional
    dependency requirements. The modern core should not require importing
    `pysynphot`/`stsynphot` for ordinary use if those remain legacy-only.

    Revisit import-time Matplotlib setup. The current helper is useful for CI,
    restricted servers, and Django, but ordinary `import Spyctres` should avoid
    surprising global Matplotlib/cache/backend side effects. Prefer explicit
    setup in scripts/server entry points and documented deployment guidance.

    Categorize the current test-suite warnings before beta: expected
    third-party deprecations should be narrowly filtered or documented;
    numerical/resource warnings should be fixed; unexpected warning growth
    should fail CI. Continue package-data, wheel/sdist, and console-entry-point
    checks.

11. **Later beta features.**
   Defer heavier additions until the above pieces are stable: optional
   wavelength-dependent LSF fitting, compression/MOPED-style acceleration,
   posterior samplers, alternative atmosphere backends, a GUI, and publication
   SED/arm-scaling modes.

12. **Final pre-beta audit.**
   Before treating Spyctres as stable enough for broader collaborator use,
   run a whole-package audit: public API compatibility, source-distribution and
   package-data checks, Django/server-side plotting compatibility with
   non-interactive Matplotlib backends, Python 3.12 and 3.13 testing, example
   clarity and ordering, notebook reproducibility, warning baseline, legacy API
   compatibility, and a final pass over README/setup instructions.

## New publication-readiness gate

The stricter publication workflow uses `publication_readiness_audit()` as a
guardrail around the ordinary `audit_spectrum_for_fit()` result. A spectrum may
be good enough for quicklook classification while still failing the publication
gate because it has assumed rather than validated resolution, unknown wavelength
metadata, missing formal errors, artifact flags, unapplied archive bad regions,
or too few usable pixels.

This distinction is deliberate: ingestion and quick classification should be
forgiving, while publication-quality parameter claims should be conservative.

The next iteration should fold this into the intent-aware readiness model:
publication remains the strictest intent, while inspect and quicklook
classification may allow some missing metadata only when the output explicitly
labels which interpretations are invalid.

## Acceptance criteria before wider collaborator use

Spyctres should not be considered ready for broad collaborator use merely
because the code imports and the unit tests pass. The practical alpha-readiness
target is:

- a user can start with `import Spyctres as sp`;
- the five numbered example pairs run in order and introduce one concept at a
  time;
- setup inspection and the actual fit use the same reviewed setup object;
- readiness is task/intent-aware and explains which interpretations are valid;
- quicklook, atmospheric-parameter, RV, and publication claims are clearly
  separated;
- wavelength medium, observer frame, stellar-rest status, mask policy,
  formal-error policy, PHOENIX model-family/composition assumptions, and
  resolution/LSF source are recorded in serialized output;
- ordinary XSL behavior validation and Gaia benchmark-star accuracy validation
  have been reviewed separately;
- the bundled Gaia FGK Benchmark Stars subset remains small, cited, checksummed,
  and used as validation data rather than hidden fit priors;
- SDSS/UVES-POP external products can be ingested and audited without bundling
  those data files in the repository;
- batch quickscan/refine has measured runtime, skip-rate, and false-narrowing
  behavior on representative spectra;
- legacy APIs have a migration table and optional-dependency policy;
- import-time side effects, warnings, wheel/sdist packaging, console entry
  points, and Django/headless plotting behavior have been audited.

## Referee feedback questions currently open

- What should the exact intent-readiness matrix allow or block for quicklook
  classification versus atmospheric parameters, physical RV, and publication?
- Should `classify_spectrum()` remain as a documented first-pass PHOENIX-label
  helper, become a distinct classification object, or be deprecated as an alias?
- Should the current custom public-help registry be retained for GUI use if it
  is generated from signatures/docstrings, or should it be removed in favor of
  standard Python help plus tutorials?
- Should `readiness_flag_actions()` stay top-level during alpha, or move behind
  structured audit/setup objects once those exist?
- What PHOENIX composition policy should be adopted for metal-poor or
  alpha-enhanced regimes when the current fit is only three-dimensional?
- Which exact pass/fail tolerances should the first Gaia FGK Benchmark Stars
  accuracy gate use, given that the bundled files are normalized R=42,000
  HARPS products and Spyctres still has known PHOENIX/LSF/abundance limits?
