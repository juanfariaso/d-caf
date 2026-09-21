"""Configuration helpers"""
import yaml

def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML config as a plain dict.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}
