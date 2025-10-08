from __future__ import annotations
from typing import Optional, Dict, Deque
from collections import deque
from ..state import SessionState, Signal, SignalType, Bar
from .base import StrategyBase

class BaselineSMA(StrategyBase):
    name = "BaselineSMA"
    def __init__(self, window: int = 20):
        self.window = window
        self.buffers: Dict[str, Deque[float]] = {}
    def on_start(self, session_state: SessionState) -> None:
        self.buffers.clear()
    def on_bar(self, symbol: str, bar: Bar, state: SessionState) -> Optional[Signal]:
        buf = self.buffers.setdefault(symbol, deque(maxlen=self.window))
        buf.append(bar.close)
        if len(buf) < self.window:
            return None
        sma = sum(buf) / len(buf)
        prev_close = list(buf)[-2] if len(buf) >= 2 else bar.close
        crossed_up = prev_close <= sma and bar.close > sma
        crossed_down = prev_close >= sma and bar.close < sma
        if crossed_up:
            return Signal(SignalType.BUY)
        if crossed_down:
            return Signal(SignalType.SELL)
        return None
