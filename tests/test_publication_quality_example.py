import json
import subprocess
import sys
from pathlib import Path


def test_publication_quality_xshooter_uvb_audit_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "publication_scaffold.json"
    cmd = [
        sys.executable,
        "examples/publication_quality_xshooter_uvb.py",
        "--output-json",
        str(output_json),
        "--force",
    ]

    completed = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output_json.read_text())
    assert payload["workflow"] == "publication_quality_xshooter_uvb_scaffold"
    assert payload["baseline_fit"] is None
    assert payload["ordinary_readiness"]["n_fit_candidate"] > 0
    assert "publication_readiness" in payload
    assert "balmer_windows" in payload["analysis_design"]
    assert payload["analysis_design"]["metal_rv_windows"]
    assert payload["analysis_design"]["core_mask_grid_A"] == [0.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    sensitivity = payload["core_mask_sensitivity"]
    assert [item["core_mask_halfwidth_A"] for item in sensitivity] == [
        0.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
    ]
    assert sensitivity[0]["n_fit_candidate"] > sensitivity[-1]["n_fit_candidate"]
    assert all(item["fit"] is None for item in sensitivity)
