import subprocess
import sys
from pathlib import Path


def test_setup_checker_runs_without_phoenix_scan():
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/check_spyctres_setup.py",
        "--skip-phoenix",
        "--spectrum",
        "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
        "--instrument",
        "xshooter",
    ]

    completed = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Spyctres setup check passed." in completed.stdout
    assert "Spyctres CLI entry point" in completed.stdout
    assert "Example spectrum ingestion" in completed.stdout
