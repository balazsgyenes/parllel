import importlib
import json
from os import PathLike
from pathlib import Path
import re
from typing import Any, Dict


CONFIG_FILENAME = "config.json"
MODEL_FILENAME = "model.pt"
PATTERN_KEY = "object_resolve_pattern"


def init_log_folder(path: PathLike) -> None:
    """This is the only method responsible for creating a new folder.
    If the folder already exists, it raises FileExistsError so as not to
    overwrite previous data.
    """
    if path is None:
        print("WARNING: No log_dir was specified, so nothing from this run "
            "will be saved.")
        return
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)


def log_config(config: Dict, path: PathLike) -> None:

    if path is None:
        return

    # define a pattern to look for, when loading objects from the params.json again
    regex_pattern = "^__callable__(.+)__from__(.+)$"
    config[PATTERN_KEY] = regex_pattern

    # convert the regex pattern to a string format specifier by stripping ^ and
    # $ at beginning and end and converting (.+) -> {}.
    format_specifier = regex_pattern[1:-1].replace("(.+)", "{}")

    def encode_non_basic_types(obj: Any) -> str:
        """Encodes any non-basic type (str, int, float, bool, None) as a str.
        """
        if isinstance(obj, Path):
            return str(obj)
        return format_specifier.format(obj.__name__, obj.__module__)

    with open(path, "w") as f:
        json.dump(
            config,
            f,
            default=encode_non_basic_types,
            indent=4,
        )


def load_config(path: PathLike, skip_resolving: bool = False) -> Dict:
    path = Path(path)

    if path.is_dir():
        path = path / CONFIG_FILENAME

    with open(path, "r") as config_file:
        config: Dict = json.load(config_file)

    if not skip_resolving:
        pattern = config.pop(PATTERN_KEY)
        config = resolve_non_basic_types(config, pattern)

    return config


def resolve_non_basic_types(input: Dict, pattern: str) -> Dict:
    """Resolves dict values of the form specified by pattern (e.g. __callable__<name>__from__<module>)

    Args:
        input: Dictionary containing object names in the form of the specified pattern (e.g. input[key] =  '__callable__<name>__from__<module>')
        pattern: The regex pattern to look for in strings of the dict. (e.g. '^__callable__(.+)__from__(.+)$'

    Returns:
        A resolved version of the dictionary in the form of input[key] = name

    Examples:
        >>> resolved = resolve_non_basic_types({'nonlinearity': '__callable__Tanh__from__torch.nn', "inner_dict": {'nonlinearity': '__callable__Tanh__from__torch.nn'}}, '^__callable__(.+)__from__(.+)$')
        >>> func = resolved['nonlinearity']
        >>> func()

    """
    for key, val in input.items():
        if isinstance(val, str):
            match = re.search(pattern, val)
            if match is not None:
                object_name = match.group(1)
                module_name = match.group(2)

                if object_name in globals():
                    # object is already defined in the global scope, so
                    # importing it would overwrite this
                    callable_object = globals()[object_name]
                else:
                    if module_name == "__main__":
                        # object was created in main module and cannot be
                        # imported
                        input[key] = None
                        continue
                    else:
                        # try to import object from the module where it is
                        # defined
                        module = importlib.import_module(module_name)
                        callable_object = getattr(module, object_name)
                input[key] = callable_object

        # recurse into nested dicts
        elif isinstance(val, dict):
            input[key] = resolve_non_basic_types(val, pattern)

    return input
