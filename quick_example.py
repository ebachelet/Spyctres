"""Compatibility pointer for Spyctres examples.

The historical ``quick_example.py`` depended on private local spectra and
optional packages that are not part of the lightweight Spyctres install.  The
maintained first-run workflows now live under ``examples/`` and use only data
bundled with the repository.
"""


def main():
    commands = [
        (
            "Check the installed package and optional PHOENIX setup",
            "spyctres doctor --skip-phoenix",
        ),
        (
            "Clean first-run example, no PHOENIX fit required",
            "python examples/example1_quickstart.py --no-show",
        ),
        (
            "First PHOENIX-backed fit, after configuring PHOENIX",
            "python examples/example1_quickstart.py "
            "--run-fit "
            "--output-json /tmp/spyctres_example1_fit.json "
            "--output-plot /tmp/spyctres_example1_fit.png "
            "--no-show",
        ),
        (
            "Batch quick scan followed by focused refinement",
            "python examples/example5_batch_fitting.py "
            "--quicklook "
            "--output-json /tmp/spyctres_batch_quick.json "
            "--summary-csv /tmp/spyctres_batch_quick.csv "
            "--plot-dir /tmp/spyctres_batch_plots "
            "--max-plots 2 "
            "--resume",
        ),
        (
            "Numbered example guide",
            "python -m pip show Spyctres",
        ),
    ]

    print("Spyctres quick examples now live in examples/.\n")
    print("Modern public API:")
    print("  import Spyctres as sp")
    print('  spec = sp.read_spectrum("my_spectrum.fits", reader="xshooter_merge1d")')
    print("  setup = sp.suggest_fit_setup(spec)")
    print('  result = sp.fit_stellar_spectrum(spec, model="phoenix", setup=setup)')
    print("  sp.plot_fit_referee(result)")
    print()
    for label, command in commands:
        print(label + ":")
        print("  " + command)
        print()
    print(
        "These commands use spectra under examples/data/. For external SDSS, "
        "UVES-POP, or user spectra, copy/download the file yourself and pass "
        "the path explicitly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
