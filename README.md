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

For a fresh Conda-based development environment:

```bash
conda create -n spyctres-dev-py312 python=3.12
conda activate spyctres-dev-py312
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

Use this checklist for a first local run from a source checkout.

1. Activate the Python 3.12 environment and install Spyctres in editable mode:

   ```bash
   conda activate spyctres-dev-py312
   pip install -e .
   ```

2. Configure the local PHOENIX path, preferably in
   `~/.config/spyctres/config.toml`:

   ```toml
   [paths]
   phoenix_dir = "/path/to/PHOENIXv2"
   ```

3. Check that the environment, package import, PHOENIX path, and bundled
   example spectrum are visible:

   ```bash
   python scripts/check_spyctres_setup.py \
     --spectrum examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
     --instrument xshooter
   ```

   If PHOENIX is not configured yet, the checker reports the missing path as a
   warning unless `--require-phoenix` is supplied. For a stricter setup check,
   use:

   ```bash
   python scripts/check_spyctres_setup.py --require-phoenix --skip-phoenix-scan
   ```

4. Run the shortest command-line fitting example:

   ```bash
   python examples/simple_phoenix_fit.py \
     examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
     --instrument xshooter
   ```

   This example reads the spectrum, suggests conservative first-pass fit
   defaults from the loaded wavelength coverage and metadata, runs the
   native-grid PHOENIX fit, prints a compact result and quality report, and
   opens a wide observed/model/residual diagnostic plot. It is a first-contact
   classification example, not a precision line-width or abundance analysis.
   It also prints a pre-fit spectrum-readiness audit: missing wavelength-frame
   metadata, missing uncertainty or resolution assumptions, obvious artifact
   signatures, and the number of pixels actually entering the chosen fit
   window. This audit does not repair the spectrum; it tells you whether the
   result should be treated as a normal first-pass fit or as quicklook triage.
   Expert users can override the suggested values with flags such as `--wmin`,
   `--wmax`, `--teff`, `--teff-min`, or `--no-auto-defaults`.

5. For many spectra, first run a cheap quicklook batch to identify sensible
   local parameter ranges:

   ```bash
   python examples/batch_quickscan_then_refine.py \
     examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
     --instrument xshooter \
     --quicklook \
     --output-json /tmp/spyctres_batch_quick.json \
     --summary-csv /tmp/spyctres_batch_quick.csv \
     --resume
   ```

6. Then rerun with focused refinement. The script reuses the quick-pass result
   to build local Teff/[Fe/H]/logg/RV bounds, rather than blindly searching the
   broad classification box for every spectrum:

   ```bash
   python examples/batch_quickscan_then_refine.py \
     examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
     --instrument xshooter \
     --output-json /tmp/spyctres_batch_refined.json \
     --summary-csv /tmp/spyctres_batch_refined.csv \
     --resume
   ```

   For heterogeneous folders, use a CSV manifest instead of relying on one
   global instrument/resolution assumption. This example uses only spectra
   bundled under `examples/data/`; replace the paths only when you intentionally
   supply your own external spectra:

   ```csv
   target_id,path,instrument,R
   xshooter_uvb,examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits,xshooter,
   floyds_blue,examples/data/Gaia21ccu_2024_11_23_FLOYDS.csv,floyds,500
   ```

   Save that CSV, for example as `examples/my_batch_manifest.csv`, then run:

   ```bash
   python examples/batch_quickscan_then_refine.py \
     --manifest examples/my_batch_manifest.csv \
     --output-json /tmp/spyctres_batch_manifest.json \
     --summary-csv /tmp/spyctres_batch_manifest.csv \
     --resume
   ```

   The batch example defaults to `--archive-mask-policy apply` and
   `--refine-quality-policy skip-risky`: recognized archive/product bad regions
   are applied as named masks with provenance, and the expensive focused-refine
   stage is skipped when readiness or quick-result quality flags still require
   human review. Use `--archive-mask-policy warn` to leave archive regions
   fitted but flagged, `--archive-mask-policy ignore` only as an explicit expert
   override, and `--refine-quality-policy always` only for developer-style
   stress testing.

Recommended example order:

1. `examples/simple_phoenix_fit.py` for the shortest command-line public-API
   path.
2. `examples/batch_quickscan_then_refine.py` for fitting many spectra with a
   cheap quick scan followed by focused local refinement.
3. `examples/high_resolution_sideband_normalization.py` for local
   sideband-normalized line-window diagnostics.
4. `examples/full_spectrum_classification.ipynb` for the first worked
   PHOENIX classification notebook.
5. `examples/xshooter_multiarm_classification.ipynb` for advanced multi-arm
   fitting diagnostics.
6. `examples/xsl_figure1_validation.ipynb` for real-library XSL validation.
7. `examples/publication_quality_xshooter_uvb.py` and
   `examples/publication_quality_xshooter_uvb.ipynb` for the expert
   publication-oriented X-SHOOTER UVB scaffold.
8. `examples/pepsi_legacy_linefit_validation.ipynb` for the PEPSI legacy
   line-window validation path.

`quick_example.py` is retained only as a compatibility pointer to these
maintained workflows; smoke tests live under `scripts/`. See
`examples/README.md` for data paths, PHOENIX configuration, caveats, and the
same recommended order with more detail.

The maintained project roadmap is in
[`docs/development_plan.md`](docs/development_plan.md). In particular,
publication-oriented parameter fitting is tracked as a separate expert workflow:
quick classification remains lightweight, while publication-quality use must
pass stricter metadata, uncertainty, mask, LSF, residual, and recovery checks.

For batches, start with the X-SHOOTER UVB throughput example:

```bash
python examples/batch_quickscan_then_refine.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_batch_xshooter_uvb.json \
  --summary-csv /tmp/spyctres_batch_xshooter_uvb.csv \
  --resume
```

Replace the single example file with a list or shell-expanded directory of
spectra when processing many observations. The script loads PHOENIX once,
runs a cheap quicklook fit, narrows the local Teff/[Fe/H]/logg/RV bounds, runs
a focused refinement, and checkpoints after each spectrum. The optional CSV is
only a compact table for sorting/filtering; the JSON keeps the full auditable
per-spectrum provenance and quality reports.

Minimal copy/paste sequence, assuming PHOENIX is already configured:

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

After a small pilot batch, summarize the timing before launching hundreds of
spectra:

```bash
python scripts/throughput_summary.py /tmp/spyctres_batch_refined.json --project 100
```

The summary reports quick-scan, focused-refine, and total per-spectrum timing,
then projects an approximate wall time for the requested number of spectra.

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
- UVES-POP ASCII spectra
- SDSS/SEGUE `spec-PLATE-MJD-FIBER` FITS spectra

Readers return a generic `SpectrumSegment` object so that fitting code can remain instrument-agnostic.
You can inspect the registered reader assumptions from Python:

```python
from Spyctres import get_instrument_info, list_instruments

print(list_instruments())
print(get_instrument_info("xshooter").to_metadata())
```

This is a discoverability layer only: it documents what each reader accepts and
records, but it does not silently apply wavelength-frame corrections or invent a
precision LSF.

The installed package also provides a deliberately read-only CLI for discovery
and ingestion checks:

```bash
spyctres instruments
spyctres instrument-info xshooter
spyctres inspect-spectrum examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter
```

If `spyctres` is not found after pulling a newer checkout, rerun
`pip install -e .` inside the active environment so the console script is
refreshed. The same read-only commands can always be run as
`python -m Spyctres.cli ...` from a source checkout.

There is intentionally no general `spyctres fit` command yet. Fitting remains
Python-first until the public batch defaults and reporting are stable enough
for a command-line fitting interface.

Inside Python, public entry points also carry lightweight call help. If a user
calls one without enough information, Spyctres reports the minimal call pattern
and a one-line fix. The same records are available directly:

```python
from Spyctres import (
    describe_public_function,
    format_public_function_help,
    list_public_functions,
)

print(list_public_functions())
print(format_public_function_help("fit_stellar_spectrum"))
help_record = describe_public_function("read_spectrum")
```

The structured `describe_public_function()` output is intended for notebooks
and a future GUI; the formatted helper is meant for terminal use.

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

Reader support means "Spyctres can ingest the product into a common
`SpectrumSegment`/`SpectrumCollection` shape." It does not always mean the file
is immediately ready for a precision PHOENIX fit. Use:

```python
from Spyctres import audit_spectrum_for_fit, read_spectrum

spec = read_spectrum("my_spectrum.fits", instrument="sdss")
audit = audit_spectrum_for_fit(
    spec,
    fit_windows=[(3800.0, 5200.0)],
    assumed_resolution={"quantity": "R", "value": 2000, "source": "quicklook"},
)
```

The audit records whether wavelength medium, observer frame, stellar-rest
status, uncertainties, resolution, sampling, and obvious artifact signatures are
adequate for the intended fit. Native data-quality masks supplied by a reader
are preferred. For products without such flags, Spyctres also provides an
explicit same-grid fallback artifact mask, but it is opt-in and recorded as a
quicklook/product assumption rather than applied silently.

For stricter work, use `publication_readiness_audit()` around the same spectrum
and masks. It deliberately treats some quicklook assumptions as blockers, for
example assumed-but-unvalidated resolution, missing formal errors, unknown
wavelength/frame metadata, unapplied archive bad-region overlap, SDSS tabulated
LSF provenance that is not yet applied by the fitter, or too few usable pixels.
This is a guardrail for expert workflows, not a replacement for real validation
on benchmark spectra.

UVES-POP and SDSS are currently best treated as ingestion plus quicklook
classification inputs unless the user supplies or validates the missing fit
assumptions. In particular, SDSS spectra are read as vacuum/heliocentric with
`resolution=None`; `--R 2000` is only an explicit quicklook approximation, not
precision SDSS LSF modelling. PHOENIX fitting examples recommend
`--sdss-mask-policy stellar_strict`, while the generic reader default remains
`and_mask_conservative`. Programmatic workflows can use
`sdss_quicklook_resolution_assumption()` to package that approximation as
provenance. If a standard SDSS `wdisp` column is present, Spyctres preserves it
with `lsf_source="sdss_wdisp_not_applied"` and can attach an opt-in tabulated
`sigma_kms` descriptor via `read_sdss_spec(..., attach_wdisp_resolution=True)`.
The current PHOENIX fitter still requires constant LSF broadening, so the
readiness audit warns when SDSS tabulated LSF is present but the likelihood is
using a constant-R assumption. UVES-POP spectra carry a nominal `R=80000`
descriptor with cautionary metadata; wavelength medium and frame remain
unknown unless supplied by the user or external provenance.
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
That provenance now also includes advisory diagnostic-window candidates selected
from the loaded wavelength coverage. These are broad windows around features
such as Balmer, Paschen, Brackett, He/Mg/Si hot-star checks, CH G-band,
Ca I/Ca II, Mg/Na/K alkali and metal lines, TiO/VO/CaH/FeH molecular bands,
and K-band Na I/Ca I/CO; they help decide which quick follow-up checks are
sensible for hot, intermediate, or cool stars without launching an expensive
blind all-combinations fit. The catalog is
defined in canonical vacuum Angstrom, stellar-rest-frame coordinates; selection
converts those broad windows to each segment's declared wavelength medium and
records the operational window, RV padding, score components, risk policies,
and contiguous-coverage diagnostics in provenance.
When you want to compare the influence of different feature families, use the
bounded diagnostic-window comparison scaffold. It defaults to a dry run that
writes a JSON/CSV/PNG plan without loading PHOENIX; actual fits require the
explicit `--run-fits` flag:

```bash
python examples/diagnostic_window_comparison.py \
  examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits \
  --instrument xshooter \
  --output-json /tmp/spyctres_windows.json \
  --output-csv /tmp/spyctres_windows.csv \
  --output-plot /tmp/spyctres_windows.png
```

The comparison policy is deliberately conservative: it runs only a small set of
trusted baseline, role-balanced, single-window, and leave-one-out/family-out
checks, records held-out windows as provenance, excludes stress-only windows by
default, and warns that raw χ² should not be used as the sole model-selection
criterion. When comparison fits are run with reconstructed model arrays, the
runner also scores selected-but-held-out windows on valid pixels that were not
used by that fit, and it scores every completed fit over the same union of
planned diagnostic windows. The common-window residual summary is often the
cleaner cross-comparison view because each fit is judged on the same feature
set, while the held-out residuals remain a stricter generalization check. Both
are proxy diagnostics, not calibrated publication likelihoods or automatic
model-selection rules.
For publication-oriented Balmer-window work, Spyctres treats line-core masking
as an explicit sensitivity choice rather than a hidden fit parameter. The
publication scaffold records the retained-pixel fraction and an
information-loss penalty for each tested core-mask width. This penalty is not
added to the spectral χ²; it exists to discourage choosing an overly wide mask
just because discarded Balmer-wing pixels made the apparent fit easier.
The scaffold also writes a bounded systematic-variant plan covering continuum
degree, preparation normalization, Balmer-core mask width, resolution
assumptions, and single-line/leave-one-line-out Balmer window sets. These
variants are reported for review but are not run by default, keeping the public
workflow lightweight and avoiding hidden fit grids.
It also records cheap observed-profile diagnostics for each Balmer line:
sideband coverage, mask fractions, line-depth and equivalent-width proxies,
wing asymmetry, and DIB overlaps. These identify lines needing review before
running expensive variants; they are not calibrated line measurements. When an
opt-in baseline PHOENIX fit is run, the scaffold adds per-line model-residual
diagnostics that keep fitted-pixel residuals separate from masked/core overplot
checks.
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
