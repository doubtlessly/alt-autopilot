from __future__ import annotations
import time
import ccxt
from .log import get_logger
log = get_logger()

def init_exchange(exchange_id: str) -> ccxt.Exchange:
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 20000})
    ex.load_markets()
    return ex

def list_spot_usdt(ex: ccxt.Exchange, quote: str = "USDT") -> list[str]:
    return [s for s in ex.symbols if s.endswith(f"/{quote}") and ex.markets.get(s, {}).get("spot")]

def fetch_tickers_safe(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception as e:
        log.warning(f"fetch_tickers failed, falling back per-symbol: {e}")
        tickers = {}
        for s in ex.symbols:
            try:
                tickers[s] = ex.fetch_ticker(s)
                time.sleep(ex.rateLimit/1000.0)
            except Exception:
                continue
        return tickers

def quote_volume_usd(t) -> float:
    if not t:
        return 0.0
    if "quoteVolume" in t and t["quoteVolume"]:
        try: return float(t["quoteVolume"])
        except: pass
    info = t.get("info", {}) if isinstance(t, dict) else {}
    for k in ("volValue","quoteVolume","volValue24h","volValue24"):
        v = info.get(k)
        if v:
            try: return float(v)
            except: continue
    return 0.0
