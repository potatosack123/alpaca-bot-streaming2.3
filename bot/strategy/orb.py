# bot/strategy/orb.py
from __future__ import annotations
from typing import Optional, Dict, Tuple
from datetime import time

from ..state import SessionState, Signal, SignalType, Bar
from .base import StrategyBase

class ORB(StrategyBase):
    name = "ORB"
    default_timeframe = "1m"
    supported_timeframes = {"1m"}  # compute 5m range from 1m bars
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.ranges: Dict[str, Tuple[float,float,bool]] = {}  # sym -> (lo, hi, locked)

    def on_start(self, session_state: SessionState) -> None:
        self.ranges.clear()

    def _mins_from_open(self, bar: Bar) -> Optional[int]:
        if not bar.timestamp: return None
        t = bar.timestamp.astimezone().time()
        if t < time(9,30) or t > time(16,0): return None
        return (bar.timestamp.astimezone().hour*60 + bar.timestamp.astimezone().minute) - (9*60+30)

    def on_bar(self, symbol: str, bar: Bar, state: SessionState) -> Optional[Signal]:
        mins = self._mins_from_open(bar)
        if mins is None:  # ignore pre/post
            return None

        lo, hi, locked = self.ranges.get(symbol, (bar.low, bar.high, False))

        # Build the opening range during the first window
        if not locked:
            lo = min(lo, bar.low); hi = max(hi, bar.high)
            if mins + 1 >= self.window_minutes:
                locked = True
            self.ranges[symbol] = (lo, hi, locked)
            return None

        # After range is locked: trade the breakout
        if bar.high >= hi:
            return Signal(SignalType.BUY)   # optional: custom SL = lo, TP = hi + (hi-lo)
        # (Short side is optional in small-caps; skip by default)
        return None
