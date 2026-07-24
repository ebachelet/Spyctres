import subprocess
import sys
from pathlib import Path

import pytest

from Spyctres.cli import build_parser


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_command(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize(
    "command, expected",
    [
        (["-m", "Spyctres.cli", "--help"], "Read-only Spyctres discovery"),
        (["-m", "Spyctres.cli", "inspect-spectrum", "--help"], "Minimal call:"),
        (["scripts/check_spyctres_setup.py", "--help"], "Setup diagnostic"),
        (["scripts/io_smoketest.py", "--help"], "Setup/ingestion diagnostic"),
        (["examples/simple_phoenix_fit.py", "--help"], "Minimal fit_stellar_spectrum"),
        (
            ["examples/example1_quickstart.py", "--help"],
            "Example 1 quickstart",
        ),
        (
            ["examples/example2_lines_windows_and_masks.py", "--help"],
            "Example 2",
        ),
        (
            ["examples/example3_improving_a_phoenix_fit.py", "--help"],
            "Example 3",
        ),
        (
            ["examples/example4_publication_quality_fitting.py", "--help"],
            "Example 4",
        ),
        (
            ["examples/example5_batch_fitting.py", "--help"],
            "Example 5",
        ),
        (["examples/branch_quickscan.py", "--help"], "Branch quickscan"),
        (
            ["examples/diagnostic_window_comparison.py", "--help"],
            "Diagnostic-window comparison",
        ),
        (
            ["examples/high_resolution_sideband_normalization.py", "--help"],
            "sideband normalization",
        ),
        (
            ["examples/batch_quickscan_then_refine.py", "--help"],
            "batch workflow",
        ),
        (
            ["examples/publication_quality_xshooter_uvb.py", "--help"],
            "Publication-oriented",
        ),
    ],
)
def test_user_facing_commands_print_help_without_running_fits(command, expected):
    completed = _run_command(*command)

    assert completed.returncode == 0
    assert expected in completed.stdout
    assert "Loading PHOENIX" not in completed.stdout
    assert "Running baseline" not in completed.stdout


def test_read_only_cli_rejects_abbreviated_options():
    parser = build_parser()

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["instruments", "--j"])

    assert caught.value.code == 2


def test_simple_example_rejects_abbreviated_long_options():
    completed = _run_command(
        "examples/simple_phoenix_fit.py",
        "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
        "--instrument",
        "xshooter",
        "--plot-la",
        "stacked",
    )

    assert completed.returncode == 2
    assert "unrecognized arguments: --plot-la stacked" in completed.stderr
