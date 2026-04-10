# Spyctres PHOENIX full-spectrum example

This directory contains a worked notebook example of PHOENIX-based full-spectrum fitting in Spyctres, using a reduced X-SHOOTER UVB spectrum of Gaia21ccu as the reference dataset.

The notebook is meant to be a clean first example of the generic PHOENIX fitting workflow. It is not intended to be the final precision analysis for this spectrum, and it is not the full benchmark-validation path used for development testing.

## What this notebook demonstrates

The example notebook shows how to:

1. resolve the local PHOENIX template path from environment or config
2. read a reduced 1D spectrum with `Spyctres.io.read_spectrum`
3. inspect the returned `SpectrumSegment` metadata
4. define Balmer-window fitting segments
5. exclude line-core pixels with a simple mask
6. run a PHOENIX fit for `(Teff, [Fe/H], logg, RV)`
7. reconstruct and plot the fitted model
8. interpret the result as a first model-based spectral classification

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
- optional `err`
- default boolean `mask`
- metadata such as `wave_medium`, `wave_frame`, and `resolution_R` where available

To add a new instrument:
- add a new reader function in `Spyctres/io.py`
- make that function return a `SpectrumSegment`
- register the reader under one or more aliases in the read_spectrum registry

Instrument-specific I/O belongs in `Spyctres/io.py`, while the fitter itself operates on generic `SpectrumSegment` objects.
