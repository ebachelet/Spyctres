# Spyctres

Spyctres is a Python package for stellar spectral fitting and spectral typing from reduced spectra. It can compare a measured spectrum with publicly available spectral-template libraries to find the closest match.

Developer: Etienne Bachelet

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

## PHOENIX template path

The local PHOENIX path is resolved in this order:

1. explicit command-line value
2. environment variable `SPYCTRES_PHOENIX_DIR`
3. config file `~/.config/spyctres/config.toml`

Example config:

```toml
[paths]
phoenix_dir = "/path/to/PHOENIXv2"
```

## Quick start

Useful entry points in the repository include:

- `quick_example.py` for the legacy fitting workflow
- `examples/full_spectrum_classification.ipynb` for PHOENIX classification
- smoke tests under `scripts/`

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

## Public PHOENIX fitting API

The high-level API accepts a `SpectrumSegment`, `SpectrumCollection`, or any
input supported by the canonical ingestion layer:

```python
from Spyctres import fit_phoenix_spectrum

result = fit_phoenix_spectrum(
    spectrum,
    phoenix_dir="/path/to/PHOENIXv2",
    p0=(5750.0, 0.0, 4.5, 0.0),
    forward_model="native_interp",
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
For exploratory fits with outliers that are not yet fully masked, the PHOENIX
fitters expose SciPy's robust least-squares losses through `loss` and
`loss_f_scale`; the default `loss="linear"` preserves ordinary least squares.
For spectra whose formal uncertainties are unrealistically small, an optional
`error_floor_fraction` adds a per-segment fractional uncertainty floor in
quadrature and records the applied floor in the fit diagnostics.

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
