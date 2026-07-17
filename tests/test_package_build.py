import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


RUNTIME_DATA_FILES = (
    "Bressan_Isochrones.dat",
    "G_GAIA_responses.dat",
    "H_2MASS_responses.dat",
    "J_2MASS_responses.dat",
    "K_2MASS_responses.dat",
    "LBL_A10_s0_w050_R0300000_T.fits",
    "Reader_Corliss_Lines.fits",
    "Roman_Filters.dat",
    "SLOAN_SDSS.gprime_filter.dat",
    "SLOAN_SDSS.iprime_filter.dat",
    "SLOAN_SDSS.rprime_filter.dat",
    "SLOAN_SDSS.uprime_filter.dat",
    "SLOAN_SDSS.zprime_filter.dat",
)


def _run(cmd, cwd, **kwargs):
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        **kwargs,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _copy_source_tree(root, destination):
    ignore = shutil.ignore_patterns(
        ".git",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        "build",
        "dist",
        ".ipynb_checkpoints",
        "reviewer_*.md",
        "spyctres_*review*.md",
        "spyctres_referee_*.md",
        "spyctres_phase_a_review_submission_*.md",
    )
    shutil.copytree(root, destination, ignore=ignore)


def test_clean_wheel_and_sdist_install_preserves_runtime_data_and_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "src"
    dist_dir = tmp_path / "dist"
    target = tmp_path / "target"
    _copy_source_tree(root, src)
    dist_dir.mkdir()

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist_dir),
        ],
        cwd=src,
    )
    _run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)],
        cwd=src,
    )

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    with tarfile.open(sdists[0], "r:gz") as handle:
        members = set(handle.getnames())
    assert any(member.endswith("references.json") for member in members)
    for filename in RUNTIME_DATA_FILES:
        assert any(
            member.endswith("Spyctres/data/{0}".format(filename))
            for member in members
        )

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheels[0]),
        ],
        cwd=tmp_path,
    )

    check_code = """
from importlib.resources import files
required = {required!r}
base = files("Spyctres.data")
missing = [name for name in required if not base.joinpath(name).is_file()]
if missing:
    raise SystemExit("missing package data: " + ", ".join(missing))
""".format(required=RUNTIME_DATA_FILES)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    _run([sys.executable, "-c", check_code], cwd=tmp_path, env=env)

    entry_points = list(target.glob("*.dist-info/entry_points.txt"))
    assert len(entry_points) == 1
    assert "spyctres = Spyctres.cli:main" in entry_points[0].read_text(
        encoding="utf-8"
    )

    script = target / "bin" / "spyctres"
    assert script.is_file()
    completed = _run(
        [str(script), "instruments", "--json"],
        cwd=tmp_path,
        env=env,
    )
    payload = json.loads(completed.stdout)
    assert "xshooter" in payload["instruments"]
