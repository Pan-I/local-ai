from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config() -> dict:
    config = load_yaml(CONFIG_DIR / "config.example.yaml")

    local_path = CONFIG_DIR / "local.yaml"

    if local_path.exists():
        local_config = load_yaml(local_path)
        config = deep_merge(config, local_config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result