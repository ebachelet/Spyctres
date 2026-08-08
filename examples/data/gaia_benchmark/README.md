# Gaia FGK Benchmark Stars example subset

This directory contains a deliberately small subset of the public Gaia FGK
Benchmark Stars Library for Spyctres validation examples. The files are the
R=42,000, continuum-normalized HARPS text products from the 2025/GBSv3 web
library, kept as `.txt.gz` so the repository does not grow unnecessarily.

Use the dedicated reader:

```bash
python scripts/io_smoketest.py \
  --reader gbs_v3_ascii \
  examples/data/gaia_benchmark/HIP79672_HARPS_1_R42KNorm.txt.gz \
  --no-show \
  --plot-dir /tmp/spyctres_io_plots
```

Calibration/provenance summary:

- wavelength column: `waveobs`, in nm, converted by Spyctres to Angstrom;
- flux: normalized flux from the benchmark-library product;
- uncertainty: third column `err`, treated as 1-sigma flux uncertainty;
- wavelength range: 480-680 nm;
- resolution: common degraded product, R=42,000;
- frame: source library describes spectra as RV-corrected to laboratory
  wavelengths; Spyctres records `stellar_rest_status="corrected"`;
- wavelength medium: `air` in the Spyctres GBSv3 R42KNorm reader profile. The
  bundled web table gives `waveobs` in nm but does not state the air/vacuum
  convention explicitly; Spyctres adopts air for this profile because strong
  optical lines in the bundled examples align with air rest wavelengths and the
  air/vacuum offset otherwise appears as a spurious ~80 km/s RV;
- masks: no separate mask files are bundled here; Spyctres uses finite
  wavelength/flux and positive finite error values, then relies on normal
  diagnostic/audit masks for fitting.

These spectra are meant for external validation of parameter recovery. The
reference parameters in `manifest.json` are comparison labels, not hidden fit
priors.

For a failure diagnosis, the validation runner can keep the same
reference-independent parameter bounds while changing only the fitted window
set. For example, Arcturus/cool-giant checks can compare the default branch
windows with a broader optical metal-forest diagnostic:

```bash
python scripts/gaia_benchmark_validation.py \
  --run-fits \
  --target Arcturus \
  --window-set broad_metal_forest \
  --line-plot-dir /tmp/spyctres_gbs_line_plots \
  --output-json /tmp/spyctres_gbs_arcturus_metal.json \
  --output-csv /tmp/spyctres_gbs_arcturus_metal.csv \
  --fit-plot-dir /tmp/spyctres_gbs_arcturus_plots \
  --force
```

Add `--line-plot-reference-model` to overlay a same-grid diagnostic model at
the manifest Teff/logg/[Fe/H]. This is a visualization aid only: the reference
parameters are not fit priors, and the plotting machinery is the generic
`Spyctres.plot_model_line_windows()` helper.

An optional `--error-floor-fraction` can be used for diagnostic sensitivity
checks when formal errors are much smaller than PHOENIX/model/continuum
systematics. Non-zero error floors should be reported separately from ordinary
no-floor recovery statistics; they are not a hidden default.

Please cite the Gaia FGK Benchmark Stars Library papers listed in
`references.json` if these files are used in validation outputs or derived
figures.
