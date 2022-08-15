from os import PathLike
import json
from pathlib import Path
from typing import Dict


default_fields = [
    "env",
    "model",
    "optimizer",
    "algo",
    "runner",
]
def add_default_config_fields(config: Dict) -> Dict:
    for field in default_fields:
        if field not in config:
            config[field] = {}
    
    return config


def log_config(config: Dict, path: PathLike) -> None:
    """
    TODO: this might rather belong in the logging module
    TODO: ensure that folder does not already exist to avoid overwriting
    TODO: pathlib.Path cannot be logged because it doesn't have a __name__
    TODO: add run name to config to be logged
    TODO: add build function (+ module) to config to be logged
    """

    # define a pattern to look for, when loading objects from the params.json again
    regex_pattern = "^__callable__(.+)__from__(.+)$"
    # strip start of line and end of line from the string and replace the capture group with curly braces, so we can use it as placeholders for "{}".format
    lambda_expression = regex_pattern[1:-1].replace("(.+)", "{}")
    config["object_resolve_pattern"] = regex_pattern

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "config.json", "w") as f:
        # If the object is not a native python type, pass it through the lambda_expression that saves a string with name and module of the object
        json.dump(
            config,
            f,
            default=lambda o: lambda_expression.format(o.__name__, o.__module__),
            indent=4,
        )
