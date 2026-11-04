"""Write JSON without exposing partially written results."""

import json
from pathlib import Path


def atomic_write_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)
