from typing import Dict


def merge_dicts(a: Dict, b: Dict, /) -> Dict:
    for key in b:
        if isinstance(a.get(key, None), dict) and isinstance(b[key], dict):
            a[key] = merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]
    return a
