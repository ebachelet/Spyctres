import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "throughput_summary.py"
    spec = importlib.util.spec_from_file_location("throughput_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_throughput_summary_projects_batch_runtime():
    module = _load_module()
    records = [
        {
            "status": "ok",
            "quick_seconds": 2.0,
            "refine_seconds": 8.0,
            "total_seconds": 12.0,
        },
        {
            "status": "refine_skipped",
            "quick_seconds": 4.0,
            "total_seconds": 6.0,
        },
    ]

    summary = module.summarize_records(records, project_n=100)

    assert summary["n_records"] == 2
    assert summary["status_counts"] == {"ok": 1, "refine_skipped": 1}
    assert summary["quick_seconds"]["median_seconds"] == pytest.approx(3.0)
    assert summary["refine_seconds"]["count"] == 1
    assert summary["total_seconds"]["mean_seconds"] == pytest.approx(9.0)
    assert summary["projection"]["median_based_seconds"] == pytest.approx(900.0)


def test_throughput_summary_cli_reads_checkpoint(tmp_path, capsys):
    module = _load_module()
    checkpoint = tmp_path / "batch.json"
    checkpoint.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "status": "quick_ok",
                        "quick_seconds": 1.5,
                        "total_seconds": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module.main([str(checkpoint), "--project", "10", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 1
    assert payload["status_counts"] == {"quick_ok": 1}
    assert payload["projection"]["median_based_seconds"] == pytest.approx(20.0)
