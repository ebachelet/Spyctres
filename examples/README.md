# Spyctres examples

This directory contains the maintained, numbered Spyctres learning path.  Each
numbered example has a Python script and a matching no-output notebook with the
same stem.  The scripts are convenient for command-line checks; the notebooks
are the preferred teaching format.

All maintained examples use curated spectra or validation payloads bundled
under `examples/data/`.  User-supplied external SDSS, UVES-POP, PEPSI, XSL, or
local spectra can still be read by Spyctres, but they should be passed
explicitly by the user and are not assumed by the reproducible examples.

## Recommended order

| Step | Files | Main idea |
| --- | --- | --- |
| 1 | `example1_quickstart.py` / `example1_quickstart.ipynb` | Clean first success: read a benchmark spectrum, inspect it, review a setup, and optionally run one PHOENIX fit. |
| 2 | `example2_lines_windows_and_masks.py` / `example2_lines_windows_and_masks.ipynb` | Diagnostic windows, known-line lookup, mask/warning provenance, and local line-fit caveats. |
| 3 | `example3_improving_a_phoenix_fit.py` / `example3_improving_a_phoenix_fit.ipynb` | Compare quicklook and stronger PHOENIX setup choices without pretending the strongest-looking fit is automatically final. |
| 4A | `example4_reviewed_balmer_analysis.py` / `example4_reviewed_balmer_analysis.ipynb` | Reviewed-analysis preparation for the complex Gaia21ccu X-SHOOTER UVB Balmer case. |
| 4B | `example4b_balmer_stability_checks.py` / `example4b_balmer_stability_checks.ipynb` | Bounded follow-up checks: continuum degree, Balmer-core mask width, line selection, residual triage, and resolution/LSF sensitivity. |
| 5 | `example5_batch_fitting.py` / `example5_batch_fitting.ipynb` | Batch quickscan-then-refine workflow for tens to hundreds of spectra, with representative inspection plots. |
| 6 | `example6_multiarm_classification.py` / `example6_multiarm_classification.ipynb` | Optional multi-arm X-SHOOTER classification using UVB, VIS, and NIR arms as separate segments. |
| 7 | `example7_xsl_reference_validation.py` / `example7_xsl_reference_validation.ipynb` | XSL DR3 reference-star validation discipline, including ordinary/stress/unsupported target separation. |
| 8 | `example8_pepsi_legacy_linefit_validation.py` / `example8_pepsi_legacy_linefit_validation.ipynb` | PEPSI legacy line-window compatibility validation, including the optional compact data/model diagnostic grid. |

The core beginner path is Examples 1-5.  Examples 6-8 are advanced or
validation-oriented: useful, but not mandatory before ordinary Spyctres use.

## Quick no-PHOENIX tour

These commands exercise ingestion, plotting setup, masks, and reviewed plans
without requiring a configured PHOENIX template library:

```bash
python examples/example1_quickstart.py --no-show
python examples/example2_lines_windows_and_masks.py --no-show
python examples/example3_improving_a_phoenix_fit.py --no-show
python examples/example4_reviewed_balmer_analysis.py --no-show
python examples/example4b_balmer_stability_checks.py --no-show
python examples/example5_batch_fitting.py --dry-run
python examples/example6_multiarm_classification.py --no-show
python examples/example7_xsl_reference_validation.py --no-show
python examples/example8_pepsi_legacy_linefit_validation.py --no-show
```

## First PHOENIX fits

After PHOENIX is configured, opt in to fits explicitly:

```bash
python examples/example1_quickstart.py \
  --run-fit \
  --output-json /tmp/spyctres_example1_fit.json \
  --output-plot /tmp/spyctres_example1_fit.png \
  --no-show

python examples/example3_improving_a_phoenix_fit.py \
  --run-fits \
  --output-json /tmp/spyctres_example3.json \
  --output-plot /tmp/spyctres_example3_standard.png \
  --no-show

python examples/example4_reviewed_balmer_analysis.py \
  --run-level fit \
  --allow-exploratory-fit \
  --override-reason "tutorial residual review; not final analysis" \
  --output-json /tmp/spyctres_example4a.json \
  --output-plot /tmp/spyctres_example4a.png \
  --no-show
```

For batch work:

```bash
python examples/example5_batch_fitting.py \
  --quicklook \
  --output-json /tmp/spyctres_example5_batch_quick.json \
  --summary-csv /tmp/spyctres_example5_batch_quick.csv \
  --plot-dir /tmp/spyctres_example5_plots \
  --max-plots 2 \
  --resume

python scripts/throughput_summary.py /tmp/spyctres_example5_batch_quick.json --project 100
```

## The public mental model

The examples use the public one-import path:

```python
import Spyctres as sp

spec = sp.read_spectrum("my_spectrum.fits", reader="xshooter_merge1d")
setup = sp.suggest_fit_setup(spec)
result = sp.fit_stellar_spectrum(spec, model="phoenix", setup=setup)
sp.plot_fit_referee(result)
```

Use `reader=` rather than `instrument=`.  A reader describes a data product and
its metadata conventions, not merely the telescope or spectrograph name.

To see available readers:

```python
import Spyctres as sp

print(sp.list_readers())
print(sp.get_reader_info("xshooter_merge1d").to_metadata())
```

If Spyctres does not yet have a reader for your file, build or request a small
reader that returns a `SpectrumSegment` or `SpectrumCollection` with explicit:

- wavelength in Angstrom;
- `wave_medium`;
- observer-motion frame;
- stellar-rest status;
- 1-sigma uncertainty if available;
- `valid_mask=True` for usable pixels;
- resolution/LSF metadata if known.

## Example roles and caveats

Example 1 uses a clean bundled Gaia benchmark spectrum so the first user
experience is not dominated by artifacts.  Examples 4A/4B deliberately use the
more difficult Gaia21ccu X-SHOOTER UVB spectrum because it teaches how Spyctres
separates an exploratory fit from a result that has enough review to interpret.

XSL DR3 spectra in Example 7 were observed with X-SHOOTER, but they are not the
same product as generic reduced X-SHOOTER arms.  They are merged, library
spectra with their own documented wavelength, rest-frame, arm-scaling, and
effective-resolution provenance.  Spyctres therefore uses the product-specific
`xsl_dr3` reader and does not apply hidden arm rescaling or another RV
correction.

PEPSI products in Example 8 need similar caution.  The `.dxt.nor` suffix alone
does not uniquely define the wavelength frame for every public or local
delivery.  The example keeps the working wavelength hypothesis explicit and
delegates the full legacy optimizer to `scripts/pepsi_fit_smoketest.py` so that
PEPSI regression logic is maintained in one place.

## Supporting files

`examples/batch_quickscan_then_refine.py` remains as an operational helper used
by Example 5.  It is not a separate beginner tutorial; it contains the full
batch implementation that the numbered wrapper delegates to.

Developer smoke tests and validation runners live under `scripts/`.  See
`scripts/README.md` for the distinction between public examples,
setup/ingestion diagnostics, developer smoke tests, and validation runners.
