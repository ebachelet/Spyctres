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
