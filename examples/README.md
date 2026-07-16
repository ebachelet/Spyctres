# Spyctres PHOENIX examples

This directory contains user-facing examples and validation notebooks for
PHOENIX-based spectral fitting in Spyctres.

All runnable commands in this document use spectra bundled under
`examples/data/`. If you want to use SDSS, UVES-POP, or another external
archive product, first download/copy that file yourself and replace the input
path explicitly; those archive files are not treated as reproducible packaged
examples unless they are present in `examples/data/`.

## Recommended order

Start with the public examples, then move into advanced and validation
workflows:

1. `simple_phoenix_fit.py` — shortest command-line path using the public API.
2. `batch_quickscan_then_refine.py` — batch-oriented quick scan followed by
   focused refinement; useful when fitting many spectra.
3. `high_resolution_sideband_normalization.py` — local line-window sideband
   normalization for UVES/PEPSI-like high-resolution diagnostics.
4. `full_spectrum_classification.ipynb` — first worked notebook, UVB only.
5. `xshooter_multiarm_classification.ipynb` — advanced multi-arm X-SHOOTER
   diagnostic workflow.
6. `xsl_figure1_validation.ipynb` — real-library validation against XSL DR3;
   useful after the basic workflow is familiar.
7. `pepsi_legacy_linefit_validation.ipynb` — developer validation for the
   PEPSI legacy line-window path, not the generic public classification path.

The examples are ordered by how much Spyctres-specific context they assume.
The first two are the best place for a new user to start.

## Example 1: command-line public API quickstart

For a shorter non-notebook workflow, use `simple_phoenix_fit.py`. It reads any
registered instrument format, invokes the public `fit_stellar_spectrum()`
workflow, prints the main fitted parameters plus a human-readable fit-quality
report, and can save compact JSON plus a diagnostic referee plot:

```bash
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_result.json \
  --output-plot /tmp/spyctres_fit.png
```

This is a real full-spectrum fit over the usable pixels, but it is deliberately
a minimal demonstration rather than a precision line-profile analysis. It shows
canonical ingestion, the native-grid PHOENIX forward model, structured output,
and interactive plotting. It does not yet fit rotational broadening,
macroturbulence, non-solar abundance patterns, or wavelength-dependent details
of an instrument's LSF. Consequently, individual line widths may visibly
disagree even when the broad atmospheric classification is informative. Use the
printed quality report to check flags, masked fraction, dropped segments, and
per-segment fit-pixel counts before interpreting the fitted parameters. Use the
instrument-specific recipes and local line diagnostics when assessing whether
line-width residuals arise from the LSF, stellar broadening, masks, or model
physics. The script uses `prepare_phoenix_fit_kwargs()` by default to choose a
conservative first-pass window, bounds, RV scan, and coarse initialization from
the loaded spectrum metadata. Expert users can still override those choices
with flags such as `--wmin`, `--wmax`, `--teff`, `--teff-min`, or
`--no-auto-defaults`.

For SDSS spectra supplied by a user, the reader deliberately leaves
`resolution=None` by default. It preserves `wdisp` LSF provenance when
available, but the PHOENIX fitter still uses only constant Gaussian broadening
unless the user supplies one explicitly. Saved metadata records this as
`lsf_source="sdss_wdisp_not_applied"`, and the readiness audit warns when the
fit proceeds with an explicit constant-R quicklook assumption instead. For a
quick visual classification pass on your own SDSS file, `--R 2000` is a
reasonable explicit approximation to try, but it is not precision SDSS LSF
modelling and fitted line widths should not be interpreted as calibrated
instrumental-broadening measurements. No SDSS spectra are currently bundled in
`examples/data/`, so the reproducible examples below use the packaged
X-SHOOTER, FLOYDS, PEPSI, and XSL files.

## Example 2: batch quick scan, then focused refinement

Use `batch_quickscan_then_refine.py` when you have many spectra and do not want
to explore a broad PHOENIX parameter box from scratch for every file. The
example defaults to the bundled X-SHOOTER UVB spectrum because it is fast,
line-rich, and avoids the extra arm/telluric choices involved in multi-arm
X-SHOOTER fitting:

```bash
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_batch_xshooter_uvb.json \
  --summary-csv /tmp/spyctres_batch_xshooter_uvb.csv \
  --resume
```

For a multi-file batch using bundled spectra, pass several packaged files:

```bash
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_VIS_TELL_CORR.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_batch.json \
  --summary-csv /tmp/spyctres_batch.csv \
  --resume
```

The JSON file is the authoritative product: it preserves the quick-scan
result, the focused refinement, quality flags, timing, and the local bounds
used for each spectrum. The optional CSV is a compact convenience table for
sorting many spectra by Teff, χ², or quality flags.

Minimal reproduction sequence, assuming PHOENIX is already configured:

```bash
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --quicklook \
  --output-json /tmp/spyctres_batch_quick.json \
  --summary-csv /tmp/spyctres_batch_quick.csv \
  --resume

python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_batch_refined.json \
  --summary-csv /tmp/spyctres_batch_refined.csv \
  --resume

python -m json.tool /tmp/spyctres_batch_refined.json
```

For products without validated LSF metadata, keep resolution explicit rather
than changing reader defaults. For example, user-supplied SDSS quicklook runs
should use an explicit approximate `--R 2000` only when that approximation is
scientifically acceptable for the inspection being done.

For mixed batches of bundled spectra, prefer a CSV manifest so each target can
carry its own reader and optional quicklook resolution assumption:

```csv
target_id,path,instrument,R
xshooter_uvb,examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits,xshooter,
floyds_blue,examples/data/Gaia21ccu_2024_11_23_FLOYDS.csv,floyds,500
```

Save that CSV, for example as `examples/my_batch_manifest.csv`, then run:

```bash
python examples/batch_quickscan_then_refine.py \
  --manifest examples/my_batch_manifest.csv \
  --output-json /tmp/spyctres_manifest_batch.json \
  --summary-csv /tmp/spyctres_manifest_batch.csv \
  --resume
```

Add `--refine-quality-policy skip-risky` when you want the example to stop
after the quicklook pass for targets whose pre-fit readiness audit flags
missing frame/resolution assumptions, obvious artifact signatures, no fitted
pixels, or undersampled LSF. The default is still `always`, which preserves the
original throughput demonstration.

The script loads the PHOENIX library once, then for each spectrum:

1. reads the spectrum through the normal Spyctres reader;
2. runs a spectrum-readiness audit over the suggested fit window;
3. runs a cheap quicklook fit;
4. builds a narrower Teff, [Fe/H], logg, and RV box around that result;
5. optionally skips or runs a focused second fit depending on
   `--refine-quality-policy`;
6. writes an atomic JSON checkpoint after the target finishes.

This is a throughput example, not a replacement for final scientific review.
Inspect the saved quality flags and residual diagnostics before interpreting a
large batch. Use `--quick-only` if you only want the first-pass scan, and tune
`--teff-margin`, `--logg-margin`, `--rv-margin`, `--quick-max-nfev`, and
`--refine-max-nfev` for the science case. Multi-arm X-SHOOTER fitting should be
introduced after the UVB workflow is understood, because the VIS/NIR arms bring
additional telluric, arm-scaling, and wavelength-coverage choices.

For high-resolution line-window work, do not rely on the broad full-spectrum
continuum alone. Use the sideband/local normalization helpers in
`Spyctres.recipes` for UVES-like or PEPSI-like line diagnostics, and keep the
PHOENIX full-spectrum multiplicative continuum for low/medium-resolution
classification or broad-window fitting. These are complementary workflows, not
competing defaults.

## Example 3: high-resolution local sideband normalization

Use `high_resolution_sideband_normalization.py` when you want to inspect a
UVES-like, PEPSI-like, or other high-resolution line window with a local
continuum defined from sidebands. This is a preprocessing/diagnostic example,
not a PHOENIX atmospheric-parameter fit:

```bash
python examples/high_resolution_sideband_normalization.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --line-label Hbeta \
  --line-center 4861.33 \
  --wmin 4830 --wmax 4895 \
  --sideband-left -55 -30 \
  --sideband-right 35 60
```

The script reads a single segment, windows it locally, stores explicit
sideband provenance, normalizes with `normalize_segment_sidebands()`, and plots
the original and normalized line window. Use this pattern before local
equivalent-width or line-profile diagnostics where broad full-spectrum
polynomials would be the wrong mental model.

## Example 4: first worked notebook

The `full_spectrum_classification.ipynb` notebook is meant to be a clean first
example of the generic PHOENIX fitting workflow. It is not intended to be the
final precision analysis for this spectrum, and it is not the full
benchmark-validation path used for development testing.

## Example 5: advanced X-SHOOTER multi-arm notebook

The `xshooter_multiarm_classification.ipynb` notebook shows how to fit selected
windows from UVB, VIS, and NIR together as a `SpectrumCollection`. It is useful
after the UVB-only notebook because it introduces arm-balanced weights,
sideband-normalized UVB windows, per-segment resolution metadata, and
arm-by-arm residual interpretation. It runs several PHOENIX fits and may take
several minutes on a normal workstation; progress messages are printed during
cache construction, RV scanning, and local optimization so users can tell it is
still working.

## Example 6: XSL real-spectrum validation

The Figure 1 validation sample from Verro et al. (2022) is listed in
`xsl_validation_manifest.csv`, with the official DR3 FITS products stored in
`data/`. Relative paths are resolved from the manifest. Run the batch analysis
with:

```bash
python scripts/xsl_validation.py examples/xsl_validation_manifest.csv \
  --output /tmp/xsl_validation_results.json --resume
```

Render the saved observed-versus-model classification panels without opening a
notebook:

```bash
python scripts/xsl_validation_plots.py /tmp/xsl_validation_results.json \
  --output-dir /tmp/xsl_validation_plots \
  --output-pdf /tmp/xsl_validation_plots.pdf
```

The plot renderer uses the saved `validation_plot` payloads and defaults to
one global display scale per star. Use `--scale-mode per_segment` only for
line-shape debugging, because that mode independently median-normalizes each
arm and should not be read as an arm-to-arm flux comparison.

The DR3 paper identifies the stellar-rest-frame wavelengths as air wavelengths,
which is therefore the reader and runner default. The runner uses 4000--9000 A
by default, excludes the published dichroic-contamination region, and records
fitted-minus-reference atmospheric parameters. A sparse physical-grid scan
selects the local PHOENIX interpolation region without using the literature
parameters as the optimizer start. Standard targets use four local starts;
model-stress and peculiar targets use two bounded starts and are excluded from
ordinary recovery statistics. Every target is written atomically to the JSON
checkpoint, so `--resume` safely skips completed stars after an interruption.
The JSON retains every coarse candidate and local solution. Stars above 12000 K
are reported as unsupported instead of being extrapolated with PHOENIX.

Open `xsl_figure1_validation.ipynb` for the worked validation example. It keeps
ordinary accuracy targets separate from three deliberate boundary tests: the
unsupported O star, a carbon star whose C/O chemistry is absent from the
current PHOENIX fit, and a very cool low-gravity supergiant. This distinction is
important: a predictable model-physics mismatch is evidence about the model's
domain, not automatically evidence that the numerical fitter is broken.

`simple_phoenix_fit.py` opens an interactive Matplotlib fit figure by default,
so users can zoom, pan, and inspect residuals. Add `--no-show` for automated or
headless runs; `--output-plot` can be used independently to save the figure. If
`--output-json` and `--output-plot` are both provided, the JSON records the plot
path relative to the JSON file and strips local cache/template paths by default.
The default plot focuses on the wavelength span actually used in the fit and
draws PHOENIX/model residuals only on fitted pixels; this avoids making masked
or out-of-window data look as if they influenced the solution. Use
`--plot-xlim all` when you deliberately want a full-segment mask/debug view.
By default, the script also opens a companion zoomed-line diagnostic figure.
`--line-groups auto` selects Balmer, Ca II, and Mg II 4481 lines that overlap
fitted pixels, and adds He I only when the fitted temperature is hot enough for
that diagnostic to be meaningful. Ca II H is labelled as the Ca II H + Hε blend.
The number of panels is determined from the lines found in the fitted wavelength
range; overlapping hot-line windows are merged into one panel. Disable this with
`--no-line-diagnostics`, or customize it with `--line-groups` and
`--line-window-half-width`. If `--output-plot` is set, the line diagnostics are
saved automatically as a `*_lines` companion image unless `--output-line-plot`
is specified.
The overview plot also annotates known non-stellar absorption features when they
overlap the fitted wavelength range. At present this includes the broad DIB
4428 diffuse interstellar band and the DIB 4882 region that can contaminate the
red wing of H-beta in hot-star fits. PHOENIX is not expected to reproduce DIB
absorption. Such regions are shown and flagged by default but not excluded from
the fit; use `--mask-dibs` or `--nonstellar-feature-policy mask_known` to
explicitly mask them, `--nonstellar-feature-policy ignore` to record but not
flag them, or `--no-show-dibs` to hide the visual annotation.
The script also checks curated residual windows, starting with the H-beta red
wing. These checks are diagnostic only: they add quality-report context when a
known line-profile region has coherent residuals, but they do not mask or refit
those pixels automatically. Treat them as prompts to inspect continuum
placement, LSF/rotation assumptions, DIB contamination, intrinsic/composite
Balmer structure, and model-domain limitations. A DIB-window residual in this
example is a candidate explanation to test, not a settled identification.
The shared known-feature catalog also contains common topocentric telluric
bands, but the simple UVB example only annotates the DIB subset.
These broad telluric catalog regions are for warning, annotation, and coarse
provenance. They are not a replacement for Spyctres' legacy high-resolution
telluric transmission template. When actual telluric masking is requested,
prefer the transmission-threshold helper:

```python
from Spyctres import telluric_transmission_exclusion_mask

telluric_mask = telluric_transmission_exclusion_mask(threshold=0.90)
result = fit_stellar_spectrum(
    spectrum,
    exclude_masks=[telluric_mask],
    # other PHOENIX settings...
)
```

The helper records `method="transmission_threshold"` provenance and warns in
mask metadata if the spectrum is not known to be on a raw topocentric wavelength
grid. Broad catalog telluric masks remain available through
`known_feature_masks()` / `nonstellar_feature_masks()`, but should be treated as
explicit coarse fallbacks.
For controlled mask experiments, run the example once with the default policy
and once with `--mask-dibs`, then compare the two saved JSON products with
`Spyctres.compare_fit_results()`. The comparison helper reports parameter,
chi-square, quality-flag, known-feature, and residual-window changes; it does
not decide which fit is scientifically correct.

### Controlled DIB-mask sensitivity check

Use this as a diagnostic sensitivity test when the quality report or plots flag
DIB 4428, DIB 4882, or a related residual window. The first run records the
catalog overlaps but keeps the stellar fit unchanged:

```bash
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_default.json \
  --output-plot /tmp/spyctres_default.png \
  --no-show
```

The second run applies the named DIB masks explicitly:

```bash
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --mask-dibs \
  --output-json /tmp/spyctres_mask_dibs.json \
  --output-plot /tmp/spyctres_mask_dibs.png \
  --no-show
```

Then compare the structured outputs:

```python
import json
from Spyctres import compare_fit_results

with open("/tmp/spyctres_default.json", "r", encoding="utf-8") as handle:
    default = json.load(handle)
with open("/tmp/spyctres_mask_dibs.json", "r", encoding="utf-8") as handle:
    masked = json.load(handle)

comparison = compare_fit_results(
    default,
    masked,
    labels=("default", "mask_dibs"),
    thresholds={"teff": 250.0, "logg": 0.25, "chi2_red": 1.0},
)
print(json.dumps(comparison, indent=2))
```

Interpret this as a robustness check. A large parameter or chi-square change
means the affected wavelength region matters for the current setup; it does not
by itself prove that the DIB identification is correct or that masking is the
final scientific choice. Inspect the residuals, other Balmer lines, continuum
placement, and LSF assumptions before drawing that conclusion.

## Example 7: PEPSI legacy line-window validation

The `pepsi_legacy_linefit_validation.ipynb` notebook is a developer validation
example for the PEPSI legacy line-window workflow. Use it to understand and
test that specialized path; do not treat it as the public first-contact
classification workflow.

## What this notebook demonstrates

The example notebook shows how to:

1. resolve the local PHOENIX template path from environment or config
2. read a reduced 1D spectrum with `Spyctres.io.read_spectrum`
3. inspect the returned `SpectrumSegment` metadata
4. ask Spyctres for auditable first-pass PHOENIX defaults
5. define Balmer-window fitting segments
6. exclude line-core pixels with a simple mask
7. run a PHOENIX fit for `(Teff, [Fe/H], logg, RV)`
8. reconstruct and plot the fitted model
9. interpret the result as a first model-based spectral classification

The fitter returns physical parameters rather than a formal MK spectral class label. In practice, the fitted parameters can be used as the basis for parameter-based classification.

## Reference input data

This example uses:

`examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits`

This is a reduced merged 1D X-SHOOTER UVB FITS product. Spyctres reconstructs the wavelength array from the FITS WCS information and reads the associated flux, uncertainty, mask, and metadata through the X-SHOOTER reader in `Spyctres/io.py`.

## PHOENIX path resolution

The local PHOENIX directory is resolved in this precedence order:

1. command-line value
2. environment variable `SPYCTRES_PHOENIX_DIR`
3. config file `~/.config/spyctres/config.toml`

Example config file:

```toml
[paths]
phoenix_dir = "/path/to/PHOENIXv2"
```

If no PHOENIX directory is found, the notebook will stop with an error and ask you to define one of the settings above.

## What level of result to expect

This notebook is designed as a first-pass full-spectrum classification example.

The fitted values should be treated as an initial model-based estimate, not as a final high-precision stellar analysis. Real spectra often require iteration over modelling choices such as:

- wavelength windows
- line-core masking
- instrumental resolving power
- wavelength medium and velocity conventions
- continuum treatment
- PHOENIX subgrid selection

In other words, the notebook shows the workflow cleanly, while leaving room for later refinement.

## Running the example

A typical workflow is:
1. work from a local editable Spyctres checkout
2. set `SPYCTRES_PHOENIX_DIR` or define `phoenix_dir` in the user config
3. launch Jupyter from the repository
4. open `examples/full_spectrum_classification.ipynb`
5. run the notebook from top to bottom

## Adapting the example to your own spectrum

To use your own reduced 1D spectrum, replace the input path in the notebook and, if necessary, choose the appropriate instrument reader in `Spyctres.io.read_spectrum`.

When adapting the example, you should check:

- wavelength coverage
- wavelength medium, for example air or vacuum
- wavelength frame, for example barycentric-corrected or not
- resolving power or effective line broadening
- whether the Balmer-window choice is still appropriate
- whether the default line-core mask is sensible for your science case
- whether the suggested first-pass fit bounds should be narrowed, widened, or
  replaced by expert values

## Advanced workflows

This notebook intentionally stays close to the generic fitting path.
Spyctres also includes a higher-level workflow layer in `Spyctres.recipes`
That module contains more specialized helpers for tasks such as:

- Balmer-window definitions
- Balmer-line metadata attachment
- sideband-based normalization
- line-core exclusion masks
- model-building and plotting helpers

Those tools can be useful references when you want a more instrument-specific or more tightly controlled workflow, but they are not required for this first example.

## Validation reference

The development validation reference for the Gaia21ccu X-SHOOTER UVB case remains the notebook-faithful smoke test:
```python scripts/xshooter_fit_smoketest.py \
  --preset xshooter_uvb_notebook \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits
```

That path is useful for regression testing and benchmark comparison. The notebook in this directory is the simpler user-facing worked example.

## Adding support for another instrument

Spyctres uses a generic internal spectrum container: `SpectrumSegment`
A new instrument reader should return a `SpectrumSegment` with:
- `wave`
- `flux`
- optional 1-sigma `err`
- boolean `mask` where `True` means valid/use
- `wave_medium`
- independent `observer_frame` and `stellar_rest_status`
- an optional `ResolutionDescriptor` for constant or wavelength-dependent LSF

`read_spectrum()` canonicalizes reader output to Angstrom and records ingestion
provenance. It does not normalize, resample, or merge the uploaded spectrum.

To add a new instrument:
- add a new reader function in `Spyctres/io.py`
- make that function return a `SpectrumSegment`
- register the reader under one or more aliases in the read_spectrum registry

Instrument-specific I/O belongs in `Spyctres/io.py`, while the fitter itself operates on generic `SpectrumSegment` objects.
