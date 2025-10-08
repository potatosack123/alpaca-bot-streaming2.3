from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from ..state import SessionState, Signal

class StrategyBase(ABC):
    name: str = "Base"
    def on_start(self, session_state: SessionState) -> None: ...
    @abstractmethod
    def on_bar(self, symbol: str, bar, state: SessionState) -> Optional[Signal]: ...
    def on_stop(self, session_state: SessionState) -> None: ...
