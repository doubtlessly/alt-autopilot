from __future__ import annotations
import json, os
from datetime import datetime, timezone
from typing import Any, Dict, List

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
