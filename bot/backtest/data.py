from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..state import Bar

log = logging.getLogger(__name__)

# Default to Alpaca (can be overridden)
_BACKTEST_SOURCE = "alpaca"
_ADAPTER = None  # registered AlpacaAdapter (optional)

def register_backtest_adapter(adapter) -> None:
    global _ADAPTER
    _ADAPTER = adapter
    log.info("Backtest adapter registered.")

def set_backtest_source(source: str) -> None:
    """Set the backtest data source: 'alpaca', 'yahoo', or 'csv'."""
    global _BACKTEST_SOURCE
    _BACKTEST_SOURCE = source.lower()
    log.info(f"Backtest source set to {_BACKTEST_SOURCE}")

def _csv_path(symbol: str, tf: str) -> Path:
    return Path("data") / f"{symbol.upper()}_{tf}.csv"

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for want in ["open","high","low","close","volume"]:
        if want in cols_lower and cols_lower[want] != want.title():
            rename[cols_lower[want]] = want.title()
    df = df.rename(columns=rename)
    if "timestamp" in cols_lower:
        ts_col = cols_lower["timestamp"]
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        mask = ts.notna()
        df = df.loc[mask].copy()
        df.index = ts[mask]
        df.drop(columns=[ts_col], inplace=True, errors="ignore")
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = df.index.tz_convert("UTC")
        mask = df.index.notna()
        df = df.loc[mask].copy()
    else:
        return df.iloc[0:0]
    for col in ["Open","High","Low","Close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open","High","Low","Close","Volume"]]

def _bars_from_df(df: pd.DataFrame) -> List[Bar]:
    if df.empty:
        return []
    out: List[Bar] = []
    for ts, row in df.iterrows():
        if pd.isna(ts):
            continue
        out.append(Bar(
            timestamp=ts.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"])
        ))
    return out

def _read_csv(symbol: str, tf: str) -> List[Bar]:
    p = _csv_path(symbol, tf)
    if not p.exists():
        log.warning("CSV not found for %s at %s", symbol, p)
        return []
    try:
        df = pd.read_csv(p)
        df = _normalize_df(df)
        bars = _bars_from_df(df)
        log.info("Loaded %d bars for %s (CSV)", len(bars), symbol)
        return bars
    except Exception as e:
        log.warning("Failed to read CSV for %s: %s", symbol, e)
        return []

def _chunked_iter(start: datetime, end: datetime, days: int):
    cur = start
    step = pd.Timedelta(days=days)
    while cur < end:
        nx = min(end, cur + step)
        yield cur, nx
        cur = nx

def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}T"
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return df.resample(rule, label="right", closed="right").agg(agg).dropna()

def _load_from_yahoo(symbol: str, tf: str, start: datetime, end: datetime) -> List[Bar]:
    try:
        import yfinance as yf
    except Exception as e:
        log.error("yfinance is not installed: %s", e)
        return []
    span_days = max(1, int((end - start).total_seconds() // 86400))
    # Map to Yahoo-supported intervals
    if tf == "1m":
        base = "1m" if span_days <= 30 else "5m"
    elif tf == "3m":
        base = "1m" if span_days <= 30 else "5m"
    else:  # "5m"
        base = "5m"
    
    # Yahoo limitation warning
    if base == "1m" and span_days > 7:
        log.warning("Yahoo only provides 1m data for last 7 days. Requesting %d days.", span_days)
    
    chunk_days = 7 if base == "1m" else 60  # Yahoo limits
    frames = []
    for s,e in _chunked_iter(start, end, chunk_days):
        try:
            df = yf.download(symbol, start=s, end=e, interval=base, progress=False, 
                           prepost=False, auto_adjust=False, threads=True)
            if not df.empty:
                df = _normalize_df(df)
                frames.append(df)
        except Exception as ex:
            log.warning("Yahoo fetch failed for %s %s %s-%s: %s", symbol, base, s, e, ex)
    if not frames:
        log.info("Loaded 0 bars for %s (Yahoo)", symbol)
        return []
    df_all = pd.concat(frames).sort_index().drop_duplicates()
    # Resample if the user asked for 3m but we pulled 1m
    if tf == "3m" and base == "1m":
        df_all = _resample(df_all, 3)
    bars = _bars_from_df(df_all)
    log.info("Loaded %d bars for %s (Yahoo)", len(bars), symbol)
    return bars

def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def load_bars(symbol: str, tf: str, start: datetime, end: datetime, adapter=None) -> List[Bar]:
    """
    Load historical bars for backtesting.
    Priority: 1) Alpaca (if adapter available), 2) Yahoo, 3) CSV
    """
    now = datetime.now(timezone.utc)
    end_eff = _coerce_utc(end)
    start_eff = _coerce_utc(start)
    cutoff = now - timedelta(minutes=20)
    if end_eff > cutoff:
        end_eff = cutoff
    if end_eff <= start_eff:
        log.warning("Effective backtest window is empty after clamp (%s to %s).", start_eff, end_eff)
        return []

    # Use registered adapter if available
    if adapter is None and _ADAPTER is not None:
        adapter = _ADAPTER

    # Try data sources based on configured preference
    
    # Try data sources based on configured preference
    if _BACKTEST_SOURCE == "polygon":
        # For now, Polygon integration is pending - fall back to Yahoo
        log.warning("Polygon support coming soon. Using Yahoo Finance for now.")
        bars = _load_from_yahoo(symbol, tf, start_eff, end_eff)
        if bars:
            return bars
    
    if _BACKTEST_SOURCE == "alpaca":
        if adapter is not None:
            try:
                log.info("Loading bars for %s from Alpaca (IEX feed)...", symbol)
                bars = adapter.historical_bars(symbol, tf, start_eff, end_eff)
                if bars:
                    log.info("Loaded %d bars for %s (Alpaca)", len(bars), symbol)
                    return bars
                else:
                    log.warning("Alpaca returned 0 bars for %s", symbol)
            except Exception as e:
                log.warning("Alpaca data load failed for %s: %s", symbol, e)
        else:
            log.warning("Alpaca source requested but no adapter registered. Falling back to Yahoo.")
        
        # Fallback to Yahoo if Alpaca fails
        bars = _load_from_yahoo(symbol, tf, start_eff, end_eff)
        if bars:
            return bars
    
    elif _BACKTEST_SOURCE == "yahoo":
        bars = _load_from_yahoo(symbol, tf, start_eff, end_eff)
        if bars:
            return bars
    
    elif _BACKTEST_SOURCE == "csv":
        csv_bars = _read_csv(symbol, tf)
        if csv_bars:
            return csv_bars
    
    # Final fallback chain: try everything
    log.info("Trying all data sources for %s...", symbol)
    
    # Try adapter first
    if adapter is not None:
        try:
            bars = adapter.historical_bars(symbol, tf, start_eff, end_eff)
            if bars:
                return bars
        except Exception as e:
            log.warning("Adapter fallback failed for %s: %s", symbol, e)
    
    # Try Yahoo
    bars = _load_from_yahoo(symbol, tf, start_eff, end_eff)
    if bars:
        return bars
    
    # Try CSV as last resort
    csv_bars = _read_csv(symbol, tf)
    if csv_bars:
        return csv_bars

    log.warning("No data sources succeeded for %s", symbol)
    return []
