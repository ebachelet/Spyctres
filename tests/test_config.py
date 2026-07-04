from pathlib import Path

import pytest

from Spyctres.config import resolve_phoenix_dir


def test_resolve_phoenix_dir_precedence_and_normalization(tmp_path, monkeypatch):
    cli = tmp_path / "cli"
    env = tmp_path / "env"
    cfg = tmp_path / "cfg"
    for path in (cli, env, cfg):
        path.mkdir()
    monkeypatch.setenv("SPYCTRES_PHOENIX_DIR", str(env))

    assert resolve_phoenix_dir(cli, {"paths": {"phoenix_dir": str(cfg)}}) == str(
        cli.resolve()
    )
    assert resolve_phoenix_dir(None, {"paths": {"phoenix_dir": str(cfg)}}) == str(
        env.resolve()
    )


def test_resolve_phoenix_dir_rejects_missing_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("SPYCTRES_PHOENIX_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_phoenix_dir(tmp_path / "missing")
