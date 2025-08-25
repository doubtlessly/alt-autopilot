from __future__ import annotations
import numpy as np
import pandas as pd

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),
                    (df["h"]-pc).abs(),
                    (df["l"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def obv_proxy(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["c"].diff().fillna(0)) * df["v"]).cumsum()

def prior_range_high_1h(df_1h: pd.DataFrame, min_look=36, max_look=60) -> float:
    look = min(max_look, max(min_look, len(df_1h)-2))
    return float(df_1h["h"].iloc[-(look+1):-1].max())

def slope_up(series: pd.Series, look: int = 5, tol: float = 0.999) -> bool:
    if len(series) < look + 1: return False
    return series.iloc[-1] >= series.iloc[-look] * tol

def pct_return(series: pd.Series, look: int) -> float:
    if len(series) <= look: return 0.0
    return float(series.iloc[-1] / series.iloc[-look] - 1.0)
