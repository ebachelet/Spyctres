# Spyctres

Spyctres is a Python package for stellar spectral fitting and spectral-typing workflows.

The original package provides line-based fitting utilities and supporting tools for reduced stellar spectra. This branch extends Spyctres with an alpha PHOENIX-based fitting path, a workflow layer for instrument-specific recipes, expanded spectrum I/O, and a worked notebook example.

## Current scope

Spyctres currently includes:

- core spectral fitting utilities in `Spyctres/Spyctres.py`
- generic spectrum containers and reader dispatch in `Spyctres/io.py`
- local PHOENIX HiRes template support in `Spyctres/phoenix.py`
- a native-grid PHOENIX forward model in `Spyctres/phoenix_forward.py`
- generic PHOENIX fitting in `Spyctres/fitting.py`
- workflow-level helpers in `Spyctres/recipes.py`
- plotting helpers in `Spyctres/plotting.py`

## Alpha PHOENIX support

This branch adds an alpha PHOENIX workflow for full-spectrum fitting of reduced 1D spectra.

The main PHOENIX-related components are:

- `Spyctres/phoenix.py`  
  local PHOENIX backend, grid handling, and caching

- `Spyctres/phoenix_forward.py`  
  native-grid wavelength-space forward modelling

- `Spyctres/fitting.py`  
  generic PHOENIX fitting entry points

- `Spyctres/recipes.py`  
  higher-level workflow helpers, including Balmer-window and sideband-based recipes

The current validated benchmark path is the X-SHOOTER UVB Balmer-wing workflow using the native-grid PHOENIX forward model.

## Installation

Spyctres is currently intended for local editable installs during development.

A typical setup is:

```bash
git clone https://github.com/ebachelet/Spyctres.git
cd Spyctres
pip install -e .
```

You will also need the scientific Python stack required by the package and, for PHOENIX-based workflows, a local PHOENIX template directory.

## PHOENIX template path

The local PHOENIX template path is resolved in this precedence order:

1. explicit command-line value
2. environment variable SPYCTRES_PHOENIX_DIR
3. config file `~/.config/spyctres/config.toml`

Example config file:
```TOML
[paths]
phoenix_dir = "/path/to/PHOENIXv2"
```

## Quick start

The repository currently includes:

- `quick_example.py`
legacy package example
- `examples/full_spectrum_classification.ipynb`
worked notebook example for PHOENIX-based full-spectrum classification of a reduced X-SHOOTER UVB spectrum
- smoke tests under `scripts/`
including X-SHOOTER, PEPSI, FLOYDS, Gemini/GMOS, I/O, and PHOENIX-related checks

A useful starting point for the PHOENIX alpha path is the examples notebook:
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

## Examples and workflow layers

Spyctres now has a clearer separation between:

- generic fitting core
- workflow-specific recipe helpers
- user-facing examples
- developer smoke tests

The higher-level workflow layer lives in: `spyctres/recipes.py`

The user-facing PHOENIX example lives in: `examples/full_spectrum_classification.ipynb`

## Current limitations

This is still an alpha implementation of PHOENIX support.

In particular:
- the notebook example is a first-pass classification workflow, not a final precision analysis
- some workflows still require user judgment for wavelength windows, masking, resolving power, and continuum treatment
- instrument-specific metadata quality varies across input formats
- packaging and documentation are still minimal

## Development status

This README is intended to provide a clearer project entry point and to make alpha PHOENIX support reviewable upstream.

For the PHOENIX alpha work, the main goals are:

- local PHOENIX template support
- generic cross-instrument fitting structure
- a clean workflow layer above the fitting core
- user-facing worked examples

