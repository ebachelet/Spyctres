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
        (
            ["scripts/benchmark_phoenix_runtime.py", "--help"],
            "runtime benchmark",
        ),
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
            ["examples/example4_reviewed_balmer_analysis.py", "--help"],
            "Example 4A",
        ),
        (
            ["examples/example4b_balmer_stability_checks.py", "--help"],
            "Example 4B",
        ),
        (
            ["examples/example5_batch_fitting.py", "--help"],
            "Example 5",
        ),
        (
            ["examples/example6_multiarm_classification.py", "--help"],
            "Example 6",
        ),
        (
            ["examples/example7_xsl_reference_validation.py", "--help"],
            "Example 7",
        ),
        (
            ["examples/example8_pepsi_legacy_linefit_validation.py", "--help"],
            "Example 8",
        ),
        (
            ["scripts/gaia_benchmark_validation.py", "--help"],
            "Gaia FGK Benchmark Stars validation",
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


def test_numbered_examples_reject_abbreviated_long_options():
    completed = _run_command(
        "examples/example1_quickstart.py",
        "--plot-di",
        "/tmp/spyctres_plots",
    )

    assert completed.returncode == 2
    assert "unrecognized arguments: --plot-di" in completed.stderr
