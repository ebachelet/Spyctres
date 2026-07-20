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
from Spyctres import read_spectrum, fit_stellar_spectrum, plot_fit_referee

spec = read_spectrum("my_spectrum.fits", instrument="xshooter")
result = fit_stellar_spectrum(spec, model="phoenix")
plot_fit_referee(result)
```

Expert and publication-oriented workflows should build on the same components,
but they must not silently relax metadata, LSF, mask, or uncertainty
requirements.

## Priority order

1. **Alpha first-run and batch usability.**
   Keep the setup checker, read-only CLI, public examples, batch quick-scan
   workflow, and throughput reporting working with bundled spectra. These are
   the pieces a collaborator should be able to run out of the box.

2. **Fit-readiness and preprocessing discipline.**
   Continue hardening the common spectrum boundary, mask polarity, native
   archive/product masks, formal-error handling, wavelength-medium/frame
   metadata, and explicit resolution/LSF provenance. The ordinary readiness
   audit should decide whether a quick classification fit is safe to attempt.

3. **Real-spectrum validation.**
   Expand and maintain validation on real spectra, especially X-SHOOTER/XSL
   reference spectra and bundled examples. Stress/peculiar targets should remain
   separated from ordinary recovery statistics.

4. **Publication-oriented X-SHOOTER UVB workflow.**
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
   checkpoints, and an opt-in baseline PHOENIX fit. The remaining work is to
   add the fit-level systematic variants, per-line checks, injection/recovery,
   profile scans, and final uncertainty tables.

5. **Public API and reporting polish.**
   Keep refining structured result objects, quality reports, plotting, and
   simple examples so the common path remains easy to teach and maintain.

6. **Later beta features.**
   Defer heavier additions until the above pieces are stable: optional
   wavelength-dependent LSF fitting, compression/MOPED-style acceleration,
   posterior samplers, alternative atmosphere backends, a GUI, and publication
   SED/arm-scaling modes.

## New publication-readiness gate

The stricter publication workflow uses `publication_readiness_audit()` as a
guardrail around the ordinary `audit_spectrum_for_fit()` result. A spectrum may
be good enough for quicklook classification while still failing the publication
gate because it has assumed rather than validated resolution, unknown wavelength
metadata, missing formal errors, artifact flags, unapplied archive bad regions,
or too few usable pixels.

This distinction is deliberate: ingestion and quick classification should be
forgiving, while publication-quality parameter claims should be conservative.
