from __future__ import annotations

import os
from pathlib import Path
import tomllib


def get_xdg_config_home() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        p = Path(xdg).expanduser()
        if p.is_absolute():
            return p
    return Path.home() / ".config"


def default_config_path() -> Path:
    return get_xdg_config_home() / "spyctres" / "config.toml"


def load_user_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path).expanduser() if path is not None else default_config_path()
    if not cfg_path.is_file():
        return {}
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def get_config_value(config: dict, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_setting(cli_value, env_var_name: str | None = None, config_value=None, default=None):
    if cli_value is not None:
        return cli_value
    if env_var_name:
        env_val = os.environ.get(env_var_name, None)
        if env_val not in (None, ""):
            return env_val
    if config_value is not None:
        return config_value
    return default
