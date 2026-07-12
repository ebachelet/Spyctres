# Spyctres PHOENIX full-spectrum example

This directory contains a worked notebook example of PHOENIX-based full-spectrum fitting in Spyctres, using a reduced X-SHOOTER UVB spectrum of Gaia21ccu as the reference dataset.

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

## XSL real-spectrum validation

The Figure 1 validation sample from Verro et al. (2022) is listed in
`xsl_validation_manifest.csv`, with the official DR3 FITS products stored in
`data/`. Relative paths are resolved from the manifest. Run the batch analysis
with:

```bash
python scripts/xsl_validation.py examples/xsl_validation_manifest.csv \
  --output /tmp/xsl_validation_results.json --resume
```

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
4428 diffuse interstellar band, which PHOENIX is not expected to reproduce.
Such regions are shown and flagged by default but not excluded from the fit; use
`--mask-dibs` to explicitly mask them, or `--no-show-dibs` to hide the visual
annotation.
The script also checks curated residual windows, starting with the H-beta red
wing. These checks are diagnostic only: they add quality-report context when a
known line-profile region has coherent residuals, but they do not mask or refit
those pixels automatically. Treat them as prompts to inspect continuum
placement, LSF/rotation assumptions, and model-domain limitations.

The notebook is meant to be a clean first example of the generic PHOENIX fitting workflow. It is not intended to be the final precision analysis for this spectrum, and it is not the full benchmark-validation path used for development testing.

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
-PHOENIX subgrid selection

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
