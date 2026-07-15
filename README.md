# Spyctres

Spyctres is a Python package for stellar spectral fitting and spectral typing from reduced spectra. It can compare a measured spectrum with publicly available spectral-template libraries to find the closest match.

Developers: Etienne Bachelet and Yiannis Tsapras

Spyctres is still under active development. It includes core fitting utilities, instrument I/O helpers, plotting tools, and example workflows. Recent additions include PHOENIX template-based fitting and a clearer separation between generic fitting code, workflow recipes, examples, and smoke tests.

## Features

- spectral fitting utilities in `Spyctres/Spyctres.py`
- generic spectrum containers and reader dispatch in `Spyctres/io.py`
- PHOENIX template support in `Spyctres/phoenix.py`
- PHOENIX forward modelling in `Spyctres/phoenix_forward.py`
- fitting helpers in `Spyctres/fitting.py`
- workflow recipes in `Spyctres/recipes.py`
- plotting helpers in `Spyctres/plotting.py`

## Installation

Spyctres is currently intended for local editable installs during development. Creating and activating a virtual environment first is recommended.

Spyctres requires Python 3.12 or later.

```bash
git clone https://github.com/ebachelet/Spyctres.git
cd Spyctres
pip install -e .
```

Some legacy workflows use `pysynphot` and its successor package `stsynphot`:

```bash
pip install pysynphot stsynphot
```

Those workflows also require the stellar template libraries linked from the [pysynphot installation documentation](https://pysynphot.readthedocs.io/en/latest/index.html#pysynphot-installation-setup). After downloading and unpacking them, set `PYSYN_CDBS` to their local root directory:

```bash
export PYSYN_CDBS=/path/to/cdbs
```

PHOENIX workflows require additional scientific Python dependencies and a local PHOENIX template directory.

The PHOENIX templates may be downloaded from the Goettingen Spectral Library:

- PHOENIX archive: `https://phoenix.astro.physik.uni-goettingen.de/`
- PHOENIX v2 HiResFITS directory: `https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/`
- PHOENIX v2 wavelength file: `https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits`

The wavelength file must be placed in the root directory of the PHOENIX v2 models.

## PHOENIX template path and config file

The local PHOENIX path is resolved in this order:

1. explicit command-line value
2. environment variable `SPYCTRES_PHOENIX_DIR`
3. config file `~/.config/spyctres/config.toml`

The config file is the most convenient place for stable local paths that should
apply to notebooks, examples, and scripts. Create it like this:

```bash
mkdir -p ~/.config/spyctres
$EDITOR ~/.config/spyctres/config.toml
```

Minimal config:

```toml
[paths]
phoenix_dir = "/path/to/PHOENIXv2"
```

If you use a nonstandard XDG config root, Spyctres follows
`$XDG_CONFIG_HOME/spyctres/config.toml` instead. Use command-line
`--phoenix-dir` for one-off experiments, `SPYCTRES_PHOENIX_DIR` for temporary
shell sessions, and the config file for everyday use. Check what Spyctres sees
with:

```bash
python scripts/check_spyctres_setup.py --require-phoenix --skip-phoenix-scan
```

## Quick start

First check that the local environment and optional PHOENIX configuration are
visible to Spyctres:

```bash
python scripts/check_spyctres_setup.py \
  --spectrum examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter
```

If PHOENIX is not configured yet, the checker reports the missing path as a
warning unless `--require-phoenix` is supplied. Once `SPYCTRES_PHOENIX_DIR` or
`~/.config/spyctres/config.toml` points to the local PHOENIX root, run the
short command-line example:

```bash
python examples/simple_phoenix_fit.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter
```

This example reads the spectrum, suggests conservative first-pass fit defaults
from the loaded wavelength coverage and metadata, runs the native-grid PHOENIX
fit, prints a compact result and quality report, and opens a wide
observed/model/residual diagnostic plot. It is a first-contact classification
example, not a precision line-width or abundance analysis. Expert users can
override the suggested values with flags such as `--wmin`, `--wmax`, `--teff`,
`--teff-min`, or `--no-auto-defaults`.

Recommended example order:

1. `examples/simple_phoenix_fit.py` for the shortest command-line public-API
   path.
2. `examples/batch_quickscan_then_refine.py` for fitting many spectra with a
   cheap quick scan followed by focused local refinement.
3. `examples/full_spectrum_classification.ipynb` for the first worked
   PHOENIX classification notebook.
4. `examples/xshooter_multiarm_classification.ipynb` for advanced multi-arm
   fitting diagnostics.
5. `examples/xsl_figure1_validation.ipynb` for real-library XSL validation.
6. `examples/pepsi_legacy_linefit_validation.ipynb` for the PEPSI legacy
   line-window validation path.

Other useful entry points include `quick_example.py` for the legacy fitting
workflow and smoke tests under `scripts/`. See `examples/README.md` for data
paths, PHOENIX configuration, caveats, and the same recommended order with more
detail.

For batches, start with the X-SHOOTER UVB throughput example:

```bash
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output /tmp/spyctres_batch_xshooter_uvb.json \
  --resume
```

Replace the single example file with a list or shell-expanded directory of
spectra when processing many observations. The script loads PHOENIX once,
runs a cheap quicklook fit, narrows the local Teff/[Fe/H]/logg/RV bounds, runs
a focused refinement, and checkpoints after each spectrum.

Minimal copy/paste sequence, assuming PHOENIX is already configured:

```bash
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --quick-only \
  --output /tmp/spyctres_batch_quick.json \
  --resume

python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output /tmp/spyctres_batch_refined.json \
  --resume

python -m json.tool /tmp/spyctres_batch_refined.json
```

To open the PHOENIX example notebook:

```bash
jupyter lab examples/full_spectrum_classification.ipynb
```

## Supported readers

Current reader coverage includes:

- X-SHOOTER 1D products
- PEPSI `.dxt.nor`
- FLOYDS ASCII/CSV exports
- Gemini/GMOS ASCII exports

Readers return a generic `SpectrumSegment` object so that fitting code can remain instrument-agnostic.

PEPSI wavelength semantics are release-specific and are not inferred from the
`.dxt.nor` suffix. Use `product_profile="pets_stellar_rest"` for documented
NASA Exoplanet Archive PETS products (air wavelengths supplied in microns and
already shifted to the stellar rest frame), or
`product_profile="cds_aanda_671_a7"` for that CDS release's Angstrom,
Solar-System-barycentric products. The default `product_profile="generic"`
preserves the historical assumption that Arg is numerically in Angstrom and
leaves its medium and frame unknown. An explicit
`--use-ssbvel` correction is rejected for profiles whose wavelengths are
already barycentric or stellar-rest corrected.

All `read_spectrum()` results pass through a versioned common-format boundary.
Wavelengths are represented in Angstrom, uncertainties as 1-sigma standard
deviations, and masks use `True` to mean a valid/usable pixel. Observer-motion
frame and stellar-rest correction status are tracked independently. Instrumental
resolution is represented explicitly as constant or wavelength-dependent
`R`, Gaussian FWHM, or Gaussian sigma. Ingestion sorts but never resamples,
normalizes, coadds, or merges overlapping orders; use a `SpectrumCollection`
for separate arms or orders.

Scientific references supporting implemented algorithms are maintained in
`references.json`, together with their affected code paths and validation
notes.

## Public fitting API

For the shortest out-of-the-box PHOENIX workflow, pass a spectrum file plus the
reader name:

```python
from Spyctres import fit_stellar_spectrum

result = fit_stellar_spectrum(
    "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
    instrument="xshooter",
    phoenix_dir="/path/to/PHOENIXv2",
)
print(result["teff"], result["rv_kms"])
print(result.quality_report_text())
```

`classify_spectrum()` is an alias for the same workflow. It reads the spectrum
when given a path, asks `prepare_phoenix_fit_kwargs()` for conservative
first-pass PHOENIX defaults, runs the native-grid PHOENIX fit, reconstructs the
best-fit model, and returns a structured `PhoenixFitResult`. Expert users can
override any fitting keyword directly, for example `regions`, `p0`, `bounds`,
`rv_grid_n`, or `mdeg`.

The lower-level PHOENIX API accepts a `SpectrumSegment`, `SpectrumCollection`,
or any input supported by the canonical ingestion layer:

```python
from Spyctres import fit_phoenix_spectrum, suggest_phoenix_fit_defaults

defaults = suggest_phoenix_fit_defaults(spectrum, mode="quicklook")
result = fit_phoenix_spectrum(
    spectrum,
    phoenix_dir="/path/to/PHOENIXv2",
    **defaults.fit_kwargs,
)
print(result["teff"], result["rv_kms"])
print(result.quality_report_text())
result.to_json()
```

`PhoenixFitResult` retains dictionary-style access while also carrying model
arrays, actual fit masks, continuum coefficients, parameter covariance, and
auditable velocity/cache provenance. The compact `quality_report` and
`quality_report_text()` summaries surface the main warnings, mask fraction,
dropped segments, and per-segment fit-pixel counts without requiring users to
inspect the full nested diagnostics block. `describe_quality_flags()` provides
short explanations for individual warning strings, and `quality_report`
includes descriptions for the flags present in that result. Existing low-level
fitting functions keep returning dictionaries for backward compatibility.
The scientific default forward model is `native_interp`, which keeps the model
on a dense PHOENIX wavelength grid until after RV shifting and LSF convolution.
The older `interp_observed` path remains available only as an explicit
legacy/fast compatibility option.
`suggest_phoenix_fit_defaults()` chooses a conservative first-pass wavelength
window, parameter bounds, coarse grid, and RV scan budget from spectrum coverage
and metadata. It returns provenance, reasons, and warnings; it does not hide
air/vacuum, frame, or stellar-type assumptions, and every suggested keyword can
be overridden by expert users. The provenance also includes an
`interpretation` block for examples and future GUIs, naming the intended
quicklook/standard/diagnostic use, how to interpret `rv_kms`, any metadata-risk
flags, and the recommended next step before treating a result as scientific.
Broad telluric catalog regions are used for warning/provenance only by default.
They are intentionally coarser than Spyctres' legacy high-resolution telluric
transmission template. When you explicitly want to mask telluric absorption,
prefer the provenance-aware wrapper:

```python
from Spyctres import telluric_transmission_exclusion_mask

telluric_mask = telluric_transmission_exclusion_mask(threshold=0.90)
```

This returns an `ExclusionMaskSpec` compatible with `exclude_masks=` and records
`method="transmission_threshold"` in mask provenance. Broad catalog telluric
masks are still available as explicit coarse fallbacks through
`known_feature_masks()` / `nonstellar_feature_masks()`.
Long-running fits accept a `progress_callback`; callbacks now receive a
`FitProgressEvent` with fields such as `phase`, `message`, `fraction`, and
`elapsed_s`, while `str(event)` remains the printable status message.
For exploratory fits with outliers that are not yet fully masked, the PHOENIX
fitters expose SciPy's robust least-squares losses through `loss` and
`loss_f_scale`; the default `loss="linear"` preserves ordinary least squares.
Because residuals are already normalized by their 1-sigma uncertainties,
`loss_f_scale=1.0` is the natural starting point. Prefer `loss="soft_l1"` or
`loss="huber"` for exploratory outlier-robust fits; `cauchy` and `arctan` are
more aggressive and can make optimization less well behaved.
For spectra whose formal uncertainties are unrealistically small, an optional
`error_floor_fraction` adds a per-segment fractional uncertainty floor in
quadrature and records the applied floor in the fit diagnostics. This floor is
a median-flux proxy for the local continuum level; use it cautiously for
heavily line-blanketed spectra where the median may sit below the continuum.
When robust loss or an error floor is active, results explicitly record the
optimizer cost, raw nominal-error chi-square, effective-error chi-square, and
quality flags so these diagnostic modes are not mistaken for ordinary Gaussian
least-squares fits.

## Local line diagnostics

Local Gaussian measurements provide quick RV, width, equivalent-width, and
residual checks without replacing the physical PHOENIX fit:

```python
from Spyctres import LineSpec, fit_line, plot_line_fit

line = LineSpec("Halpha", 6562.80, kind="absorption", wave_medium="air")
diagnostic = fit_line(segment, line)
fig, axes = plot_line_fit(diagnostic)
```

Results report laboratory and segment wavelength media, observed line width,
instrumental FWHM when available, uncertainty estimates, and quality flags.
Positive equivalent width denotes absorption; emission-line area is reported
as positive `line_flux` in flux-times-Angstrom units.

Air/vacuum conversion conventions are explicit. `ciddor1996` remains the
PHOENIX workflow default, while `vald3` preserves the historical Spyctres line-
list conversion. Both leave wavelengths at or below 2000 Angstrom unchanged
and record the selected method when converting a `SpectrumSegment`.

```python
from Spyctres import convert_wavelength_medium, convert_segment_wavelength_medium

wave_vac = convert_wavelength_medium(
    wave_air, from_medium="air", to_medium="vacuum", method="ciddor1996"
)
segment_vac = convert_segment_wavelength_medium(segment_air, "vacuum")
```

## Project structure

Spyctres is organized around four layers:

- generic fitting core
- workflow recipes
- user-facing examples
- developer smoke tests

Notable files:

- `Spyctres/recipes.py`
- `examples/full_spectrum_classification.ipynb`
- `scripts/xshooter_fit_smoketest.py`

## Current limitations

PHOENIX support should still be treated as alpha.

In particular:

- the example notebook is a first-pass classification workflow, not a final precision analysis
- some workflows still require user judgment for wavelength windows, masking, resolving power, and continuum treatment
- instrument-specific metadata quality varies across input formats
- packaging and documentation are still minimal
