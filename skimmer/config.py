import re
from typing import Optional, Mapping

import yaml


CONFIG_OVERRIDE_PATTERN = re.compile(r'([^=\s]+)\s*=\s*(.*)')

def build_config_dict(config_file: Optional[str], overrides: dict|list[str]) -> dict:
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    if isinstance(overrides, Mapping):
        config.update(overrides)
    else:
        for override in overrides:
            match = CONFIG_OVERRIDE_PATTERN.match(override)
            if not match:
                raise Exception(f"Invalid config override string: '{override}'")
            key, value = match.groups()
            config[key] = yaml.safe_load(value)

    return config
