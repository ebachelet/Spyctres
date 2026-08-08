import json
import subprocess
import sys
from pathlib import Path


def test_runtime_benchmark_default_runs_without_phoenix(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out_json = tmp_path / "runtime" / "benchmark.json"
    out_csv = tmp_path / "runtime" / "benchmark.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_phoenix_runtime.py",
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert payload["schema_name"] == "spyctres.runtime_benchmark"
    assert payload["schema_status"] == "experimental"
    assert payload["run_fit"] is False
    assert payload["reader"] == "gbs_v3_ascii"
    assert not payload["spectrum"].startswith("/")
    assert payload["records"]
    assert "read_spectrum" in payload["records"][0]
    assert "suggest_fit_setup" in payload["records"][0]
    assert "fit_stellar_spectrum" not in payload["records"][0]
    assert out_csv.exists()
    assert "read_spectrum_seconds" in out_csv.read_text(encoding="utf-8")
