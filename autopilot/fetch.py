from __future__ import annotations
import time
import pandas as pd
from .log import get_logger
log = get_logger()

def to_df(ohlcv: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    if df.empty: return df
    if df["t"].max() > 10**12:
        df["t"] = (df["t"] // 1000).astype(int)
    for c in ["o","h","l","c","v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def fetch_ohlcv_safe(ex, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    for i in range(2):
        try:
            arr = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            return to_df(arr)
        except Exception as e:
            time.sleep(1.2 * (i+1))
    return to_df([])
