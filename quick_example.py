"""Compatibility pointer for Spyctres examples.

The historical ``quick_example.py`` depended on private local spectra and
optional packages that are not part of the lightweight Spyctres install.  The
maintained first-run workflows now live under ``examples/`` and use only data
bundled with the repository.
"""


def main():
    commands = [
        (
            "Single-spectrum public API quickstart",
            "python examples/simple_phoenix_fit.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter",
        ),
        (
            "Batch quick scan followed by focused refinement",
            "python examples/batch_quickscan_then_refine.py "
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits "
            "--instrument xshooter --resume",
        ),
        (
            "Numbered example guide",
            "python -m pip show Spyctres && less examples/README.md",
        ),
    ]

    print("Spyctres quick examples now live in examples/.\n")
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
