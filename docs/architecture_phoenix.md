Spyctres.py: public API and backward compatibility
io.py: SpectrumSegment, SpectrumCollection, readers
preprocessing.py: masks, wavelength corrections, optional normalization
waveutils.py: wavelength/RV utilities
phoenix.py: PHOENIX backend
phoenix_forward.py: forward model construction
fitting.py: generic fitting machinery
recipes.py: science workflows
plotting.py: plotting helpers
scripts/: smoke tests
examples/: user-facing notebooks

## Mask/preprocessing review staging

The mask/preprocessing layer is intended to be strict at the scientific
boundary and compact at the user boundary.

Implemented safeguards:

- `SpectrumSegment.mask` uses `True == valid/use`; `invalid_mask` and
  `from_invalid_mask()` are provided for masked-array/specutils-style
  interop.
- `compose_fit_mask()` builds masks once before nonlinear residual
  evaluation, records mask polarity, numeric threshold direction, and
  nonfinite-mask-output policy, and rejects nonfinite numeric mask outputs by
  default.
- `MaskResult` keeps reason-resolved masks plus overlap-aware summary counts.
- PHOENIX fit diagnostics include mask-derived quality flags such as
  `mask_fraction_high`, `segment_mask_fraction_high`,
  `explicit_exclusion_dominates`, `too_few_fit_pixels`, and
  `nonfinite_mask_output`.
- Named exclusion masks should use `ExclusionMaskSpec` or `exclusion_mask()`
  when readability/provenance matters; duplicate names are rejected rather
  than silently renamed.

Staged future refinements:

- Keep the readable dict-of-boolean-arrays representation as canonical for
  now. Add optional `to_bitmask()` / `from_bitmask()` helpers later only if
  FITS DQ-array round-tripping or compact serialization becomes important.
- Treat `exclude_masks=` as a backward-compatible low-level spelling. Prefer
  documenting the single `exclude_mask=` entry point accepting a callable, a
  named spec, or a list of named specs.
- Add mask dilation only as an explicit future operation with provenance, not
  as hidden preprocessing.

## GUI roadmap

Design the GUI now, but build it after the public API and real-spectrum
validation are steadier.

Tier 1 should be a Jupyter-native optional extra that wraps the same public
API used by scripts and notebooks. It should show the spectrum, model,
residuals, shaded mask/exclusion regions, compact mask summaries, quality
flags, and visible fit progress. The PHOENIX fitting progress-callback API is
the bridge for this.

Tier 2 should remain deferred until the intended audience is clear. If users
are mostly notebook-comfortable collaborators, Tier 1 may be enough. If the
goal is non-Python users uploading spectra, consider a small local web app or
jdaviz/Specviz integration, with PHOENIX libraries cached and fits run off the
interface thread/process.
