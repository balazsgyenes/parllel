import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict


def init_log_folder(path: PathLike) -> None:
    """This is the only method responsible for creating a new folder.
    If the folder already exists, it raises FileExistsError so as not to
    overwrite previous data.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)


def log_config(config: Dict, path: PathLike) -> None:

    # define a pattern to look for, when loading objects from the params.json again
    regex_pattern = "^__callable__(.+)__from__(.+)$"
    config["object_resolve_pattern"] = regex_pattern

    # convert the regex pattern to a string format specifier by stripping ^ and
    # $ at beginning and end and converting (.+) -> {}.
    format_specifier = regex_pattern[1:-1].replace("(.+)", "{}")

    def encode_non_basic_types(obj: Any) -> str:
        """Encodes any non-basic type (str, int, float, bool, None) as a str.
        """
        if isinstance(obj, Path):
            return str(obj)
        return format_specifier.format(obj.__name__, obj.__module__)

    with open(path / "config.json", "w") as f:
        json.dump(
            config,
            f,
            default=encode_non_basic_types,
            indent=4,
        )
