from typing import Callable, Dict


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


def add_metadata(config: Dict, build_func: Callable) -> Dict:
    # TODO: where is the canonical place to define log_dir?
    config["meta"]["log_dir"] = config["log_dir"]

    # TODO: maybe do this automagically using the call stack?
    config["meta"]["build_func"] = build_func

    # TODO: save git commit hash

    return config


def merge_dicts(a: Dict, b: Dict, /) -> Dict:
    for key in b:
        if isinstance(a.get(key, None), dict) and isinstance(b[key], dict):
            a[key] = merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]
    return a
