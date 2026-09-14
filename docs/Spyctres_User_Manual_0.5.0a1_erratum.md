# Spyctres 0.5.0a1 user-manual erratum

## PHOENIX model-library layout

Date: 14 September 2026

This erratum supersedes the directory trees shown in sections 2.5 and B.1 of
the Spyctres 0.5.0a1 user manual.

Configure `phoenix_dir` as the `HiResFITS` library root containing the native
wavelength file. In the standard Göttingen layout, the stellar templates are
one level deeper:

```text
HiResFITS/
├── WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
└── PHOENIX-ACES-AGSS-COND-2011/
    ├── Z-1.0/
    ├── Z-0.5/
    ├── Z-0.0/
    └── Z+0.5/
```

For example:

```toml
[paths]
phoenix_dir = "/path/to/HiResFITS"
```

Spyctres prefers this standard nested template tree. For backward
compatibility, it also accepts a flat installation whose ordinary `Z-*`
directories are directly under `HiResFITS`. If both trees are usable, they are
not merged: the nested tree wins deterministically.

Directories such as `Z-0.0.Alpha=+0.40` are alpha-enhanced grids. They are
excluded from the current ordinary `(Teff, [Fe/H], logg)` interpolator because
Spyctres does not yet expose `[alpha/Fe]` as an interpolation dimension.

Verify the complete installation, including template discovery, with:

```bash
spyctres doctor --require-phoenix
```

The `--skip-phoenix-scan` option deliberately checks only the configured root
and wavelength grid; it does not verify that templates can be discovered.
