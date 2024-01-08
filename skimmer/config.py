import re
from typing import Optional, Mapping

import yaml


CONFIG_OVERRIDE_PATTERN = re.compile(r'([^=\s]+)\s*=\s*(.*)')

def build_config_dict(config_file: Optional[str], overrides: dict|list[str]) -> dict:
    """
        Load config yaml file, and apply parameter value "overrides" (which can be specified to
        arguments of command line tools such as rouge_eval.py).

        :param config_file: path to config yaml file
        :param overrides: list of config parameter overrides. This can be a dict, or a list of
            strings in the form "key=yaml-value" (where yaml-value is a valid YAML expression).
    """

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
