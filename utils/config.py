"""
utils/config.py — Load and expose config.yaml as a Python dict.
"""
import yaml
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config file and return as a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
