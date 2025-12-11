from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping/object at the top level.")
    return data


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Two-stage argument parsing:
        1. Extract --config path (if any).
        2. Use YAML content to override parser defaults, then parse the remaining args.
    Command-line values still take precedence over YAML defaults.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    config_args, remaining_argv = config_parser.parse_known_args()

    config_path = Path(config_args.config).expanduser().resolve() if config_args.config else None
    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_dict = _load_yaml_config(config_path)
        parser.set_defaults(**config_dict)

    args = parser.parse_args(remaining_argv)
    args.config = config_path
    return args
