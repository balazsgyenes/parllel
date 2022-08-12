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
