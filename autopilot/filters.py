from __future__ import annotations
import pandas as pd
from .ta import ema, atr, obv_proxy, prior_range_high_1h, slope_up, pct_return
from . import config as C

class TAFeatures:
    def __init__(self, df4: pd.DataFrame, df1: pd.DataFrame, df15: pd.DataFrame, df1d: pd.DataFrame, df_btc4: pd.DataFrame):
        # 4h
        self.df4 = df4.copy()
        self.df4["ema20"] = ema(self.df4["c"], 20)
        self.df4["ema50"] = ema(self.df4["c"], 50)
        # 1h
        self.df1 = df1.copy()
        self.df1["ema20"] = ema(self.df1["c"], 20)
        self.df1["ema50"] = ema(self.df1["c"], 50)
        self.df1["atr14"] = atr(self.df1, 14)
        self.df1["atr_pct"] = self.df1["atr14"] / self.df1["c"] * 100
        self.df1["obv"] = obv_proxy(self.df1)
        # 15m
        self.df15 = df15.copy()
        # daily
        self.df1d = df1d.copy()
        self.df1d["ema20"] = ema(self.df1d["c"], 20)
        self.df1d["ema50"] = ema(self.df1d["c"], 50)
        # BTC 4h for RS
        self.df_btc4 = df_btc4.copy()
        self.prh = prior_range_high_1h(self.df1)

    # Gates
    def atr_ok(self) -> bool:
        x = float(self.df1["atr_pct"].iloc[-1])
        return C.ATR_BAND[0] <= x <= C.ATR_BAND[1]

    def structure_ok(self) -> tuple[bool, str]:
        ema_up_4h = (self.df4["ema20"].iloc[-1] > self.df4["ema50"].iloc[-1]) and slope_up(self.df4["ema20"])
        reclaim_ok = self.df1["c"].iloc[-1] > self.prh and self.df1["c"].iloc[-2] <= self.prh
        ema20_flat_up = slope_up(self.df4["ema20"], look=5, tol=0.999)
        above_ema20_1h = self.df1["c"].iloc[-1] > self.df1["ema20"].iloc[-1]
        # RS vs BTC 4h
        sym_ret = pct_return(self.df4["c"], C.RS_LOOKBACK_4H)
        btc_ret = pct_return(self.df_btc4["c"], C.RS_LOOKBACK_4H)
        rs_ok = (sym_ret - btc_ret) >= C.RS_EDGE
        ok = ema_up_4h or reclaim_ok or (ema20_flat_up and above_ema20_1h and rs_ok)
        which = "4h-uptrend" if ema_up_4h else ("range-high-reclaim" if reclaim_ok else "flat-accept-rs" if ok else "none")
        return ok, which

    def expansion_ok(self) -> bool:
        obv_up = self.df1["obv"].iloc[-1] > self.df1["obv"].iloc[-5]
        above_ema20 = self.df1["c"].iloc[-1] > self.df1["ema20"].iloc[-1]
        return (self.df1["c"].iloc[-1] >= max(self.prh * 0.998, self.df1["ema20"].iloc[-1])) and (obv_up or above_ema20)

    def trigger_ok(self) -> tuple[bool, str]:
        last_close = float(self.df15["c"].iloc[-1])
        last_low   = float(self.df15["l"].iloc[-1])
        single_above = last_close > self.prh
        retest_hold  = (last_low <= self.prh) and (last_close > self.prh)
        ok = single_above or retest_hold
        t = "breakout" if single_above else ("retest-hold" if retest_hold else "n/a")
        return ok, t

    def invalidation(self) -> float:
        return float(self.df15["l"].iloc[-8:].min())
