from pathlib import Path

from Spyctres.matplotlib_setup import ensure_matplotlib_config_dir


def test_matplotlib_config_helper_sets_writable_cache(monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    path = ensure_matplotlib_config_dir()

    assert Path(path).name == "spyctres_matplotlib_cache"
    assert Path(path).is_dir()


def test_matplotlib_config_helper_preserves_user_value(monkeypatch, tmp_path):
    custom = tmp_path / "custom_mpl"
    monkeypatch.setenv("MPLCONFIGDIR", str(custom))

    assert ensure_matplotlib_config_dir() == str(custom)
