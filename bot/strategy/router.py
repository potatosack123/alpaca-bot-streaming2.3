# bot/strategy/router.py
from __future__ import annotations
from typing import Optional, Dict
from datetime import time
from ..state import SessionState, Signal, SignalType, Bar
from .gap_and_go import GapAndGo
from .orb import ORB

class Router:
    """Small, deterministic router for Account-Builder phase."""
    def __init__(self,
                 gag: Optional[GapAndGo] = None,
                 orb: Optional[ORB] = None):
        self.gag = gag or GapAndGo()
        self.orb = orb or ORB(window_minutes=5)
        self.active: Dict[str, str] = {}  # symbol -> 'GAG'|'ORB'|'NONE'

    @staticmethod
    def _is_open(bar: Bar) -> bool:
        if not bar.timestamp: return False
        t = bar.timestamp.astimezone().time()
        return time(9,30) <= t <= time(16,0)

    def on_start(self, state: SessionState) -> None:
        self.gag.on_start(state); self.orb.on_start(state)
        self.active.clear()

    def on_bar(self, symbol: str, bar: Bar, state: SessionState) -> Optional[Signal]:
        # If not open yet, let GAG learn premarket high
        if not self._is_open(bar):
            self.gag.on_bar(symbol, bar, state)
            self.active[symbol] = "NONE"
            return None

        # First 1–2 minutes → try Gap-and-Go
        local = bar.timestamp.astimezone()
        first_minutes = (local.hour == 9) and (local.minute in (30,31))
        if first_minutes and self.active.get(symbol) in (None, "NONE", "GAG"):
            sig = self.gag.on_bar(symbol, bar, state)
            self.active[symbol] = "GAG"
            if sig: return sig

        # Else default to ORB once range is locked
        sig = self.orb.on_bar(symbol, bar, state)
        self.active[symbol] = "ORB"
        return sig

    # Helper for data layer: enforce 1-minute feed for both strategies
    @property
    def required_timeframe(self) -> str:
        return "1m"
