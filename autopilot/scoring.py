from __future__ import annotations

def confidence(structure: str, expansion_ok: bool, trigger_ok: bool, atr_ok: bool) -> int:
    s = 0
    s += 30 if structure == "4h-uptrend" else 20 if structure in ("range-high-reclaim","flat-accept-rs") else 0
    s += 25 if expansion_ok else 0
    s += 25 if trigger_ok else 0
    s += 20 if atr_ok else 0
    return max(0, min(100, int(s)))
