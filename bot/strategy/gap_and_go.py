# bot/strategy/gap_and_go.py - WITH SHORT SELLING SUPPORT
from __future__ import annotations
from typing import Optional, Dict, Literal
from datetime import time, datetime
from collections import deque
import pytz
import logging

from ..state import SessionState, Signal, SignalType, Bar
from .base import StrategyBase

log = logging.getLogger(__name__)

class GapAndGo(StrategyBase):
    name = "GapAndGo"
    default_timeframe = "1m"
    supported_timeframes = {"1m"}

    def __init__(self,
                 min_gap_pct: float = 3.0,
                 max_price: float = 500.0,
                 min_volume: int = 50000,
                 max_float_m: float = 100.0,
                 confirm_bars: int = 1,                    # Changed from 0 to 1
                 trade_cutoff_minute: int = 5,
                 max_hold_minutes: int = 60,               # Increased from 30
                 gap_fill_exit_pct: float = 0.96,          # Changed from 0.98 (4% room)
                 exit_time_hour: int = 15,                 # Changed from 10 to 3pm
                 exit_time_minute: int = 45,               # Changed from 30 to 3:45pm
                 trade_direction: Literal["long_only", "short_only", "both"] = "both",  # NEW
                 volume_surge_factor: float = 1.5,         # NEW: Require volume surge
                 debug: bool = True
    ):
        self.min_gap_pct = min_gap_pct
        self.max_price = max_price
        self.min_volume = min_volume
        self.max_float_m = max_float_m
        self.confirm_bars = confirm_bars
        self.trade_cutoff_minute = trade_cutoff_minute
        self.max_hold_minutes = max_hold_minutes
        self.gap_fill_exit_pct = gap_fill_exit_pct
        self.exit_time_hour = exit_time_hour
        self.exit_time_minute = exit_time_minute
        self.trade_direction = trade_direction
        self.volume_surge_factor = volume_surge_factor
        self.debug = debug
        
        # State tracking
        self.premarket_high: Dict[str, float] = {}
        self.previous_close: Dict[str, float] = {}
        self.first_break_done: Dict[str, bool] = {}
        self.buffer: Dict[str, deque] = {}
        self.in_position: Dict[str, bool] = {}
        self.position_direction: Dict[str, str] = {}  # NEW: Track 'long' or 'short'
        self.entry_time: Dict[str, datetime] = {}
        self.entry_price: Dict[str, float] = {}
        self.reference_price: Dict[str, float] = {}   # NEW: Store reference for exit logic
        self.cumulative_volume: Dict[str, int] = {}
        self.avg_volume: Dict[str, float] = {}        # NEW: Track average volume
        self.volume_samples: Dict[str, list] = {}     # NEW: For volume calculation
        self.current_date: Dict[str, str] = {}
        self.debug_logged: Dict[str, bool] = {}
        
        self.east = pytz.timezone("America/New_York")

    def on_start(self, session_state: SessionState) -> None:
        self.premarket_high.clear()
        self.previous_close.clear()
        self.first_break_done.clear()
        self.buffer.clear()
        self.in_position.clear()
        self.position_direction.clear()
        self.entry_time.clear()
        self.entry_price.clear()
        self.reference_price.clear()
        self.cumulative_volume.clear()
        self.avg_volume.clear()
        self.volume_samples.clear()
        self.current_date.clear()
        self.debug_logged.clear()
        
        if self.debug:
            log.info(f"GapAndGo initialized: min_gap={self.min_gap_pct}%, "
                    f"max_price=${self.max_price}, min_volume={self.min_volume}, "
                    f"direction={self.trade_direction}, volume_surge={self.volume_surge_factor}x")

    def _get_eastern_time(self, bar: Bar) -> datetime:
        """Convert bar timestamp to Eastern time"""
        if not bar.timestamp:
            return None
        if bar.timestamp.tzinfo is None:
            utc_time = bar.timestamp.replace(tzinfo=pytz.UTC)
        else:
            utc_time = bar.timestamp
        return utc_time.astimezone(self.east)

    def _is_premarket(self, eastern_time: datetime) -> bool:
        """Check if time is premarket (4am-9:30am ET)"""
        t = eastern_time.time()
        return time(4, 0) <= t < time(9, 30)

    def _is_market_hours(self, eastern_time: datetime) -> bool:
        """Check if time is during market hours (9:30am-4pm ET)"""
        t = eastern_time.time()
        return time(9, 30) <= t <= time(16, 0)

    def _should_exit_time(self, eastern_time: datetime) -> bool:
        """Check if we should exit based on time (after exit time)"""
        t = eastern_time.time()
        exit_t = time(self.exit_time_hour, self.exit_time_minute)
        return t >= exit_t

    def _reset_daily_state(self, symbol: str, date_str: str):
        """Reset state for a new trading day"""
        if self.current_date.get(symbol) != date_str:
            self.current_date[symbol] = date_str
            self.first_break_done[symbol] = False
            self.buffer[symbol] = deque(maxlen=max(1, self.confirm_bars))
            self.in_position[symbol] = False
            self.position_direction.pop(symbol, None)
            self.entry_time.pop(symbol, None)
            self.entry_price.pop(symbol, None)
            self.reference_price.pop(symbol, None)
            self.cumulative_volume[symbol] = 0
            self.debug_logged[symbol] = False

    def _update_avg_volume(self, symbol: str, bar: Bar, eastern_time: datetime):
        """Track average volume (excluding first 5 minutes of day)"""
        minutes_since_open = (eastern_time.hour - 9) * 60 + (eastern_time.minute - 30)
        
        # Only track after first 5 minutes to get realistic average
        if minutes_since_open > 5:
            if symbol not in self.volume_samples:
                self.volume_samples[symbol] = []
            
            self.volume_samples[symbol].append(bar.volume)
            
            # Keep last 100 bars for rolling average
            if len(self.volume_samples[symbol]) > 100:
                self.volume_samples[symbol].pop(0)
            
            self.avg_volume[symbol] = sum(self.volume_samples[symbol]) / len(self.volume_samples[symbol])

    def on_bar(self, symbol: str, bar: Bar, state: SessionState) -> Optional[Signal]:
        """
        Gap-and-go strategy with long and short capability
        """
        eastern_time = self._get_eastern_time(bar)
        if not eastern_time:
            return None
        
        date_str = eastern_time.strftime("%Y-%m-%d")
        self._reset_daily_state(symbol, date_str)
        
        # Track cumulative volume
        self.cumulative_volume[symbol] = self.cumulative_volume.get(symbol, 0) + bar.volume
        
        # === PREMARKET: Track the high ===
        if self._is_premarket(eastern_time):
            ph = self.premarket_high.get(symbol, float("-inf"))
            new_high = max(ph, bar.high)
            if new_high != ph:
                self.premarket_high[symbol] = new_high
                if self.debug and not self.debug_logged.get(symbol, False):
                    log.info(f"[{symbol}] Premarket high updated: ${new_high:.2f}")
            return None
        
        # === AFTER HOURS: Track close for next day ===
        if not self._is_market_hours(eastern_time):
            self.previous_close[symbol] = bar.close
            return None
        
        # === MARKET HOURS ===
        
        # Update average volume tracking
        self._update_avg_volume(symbol, bar, eastern_time)
        
        # Get reference high (premarket high or previous close)
        ref_high = self.premarket_high.get(symbol)
        if ref_high is None or ref_high == float("-inf"):
            ref_high = self.previous_close.get(symbol)
            if ref_high is None:
                ref_high = bar.open
                self.premarket_high[symbol] = ref_high
        
        # === POSITION MANAGEMENT (if we're already in) ===
        if self.in_position.get(symbol, False):
            direction = self.position_direction.get(symbol, "long")
            entry_px = self.entry_price.get(symbol, ref_high)
            ref_px = self.reference_price.get(symbol, ref_high)
            entry_tm = self.entry_time.get(symbol)
            
            # Exit 1: Time-based exit (max hold time)
            if entry_tm:
                hold_minutes = (eastern_time - entry_tm).total_seconds() / 60
                if hold_minutes >= self.max_hold_minutes:
                    if self.debug:
                        log.info(f"[{symbol}] EXIT ({direction}): Max hold time ({hold_minutes:.1f}min) @ ${bar.close:.2f}")
                    self.in_position[symbol] = False
                    self.position_direction.pop(symbol, None)
                    self.entry_time.pop(symbol, None)
                    self.entry_price.pop(symbol, None)
                    self.reference_price.pop(symbol, None)
                    return Signal(SignalType.SELL if direction == "long" else SignalType.BUY)
            
            # Exit 2: Gap fill logic (different for long vs short)
            if direction == "long":
                # For longs: exit if price drops below threshold
                gap_fill_threshold = entry_px * self.gap_fill_exit_pct
                if bar.close < gap_fill_threshold:
                    if self.debug:
                        log.info(f"[{symbol}] EXIT (long): Gap fill (${bar.close:.2f} < ${gap_fill_threshold:.2f})")
                    self.in_position[symbol] = False
                    self.position_direction.pop(symbol, None)
                    self.entry_time.pop(symbol, None)
                    self.entry_price.pop(symbol, None)
                    self.reference_price.pop(symbol, None)
                    return Signal(SignalType.SELL)
            
            else:  # short
                # For shorts: exit if price rises back above threshold
                gap_fill_threshold = entry_px * (2 - self.gap_fill_exit_pct)  # Inverse logic
                if bar.close > gap_fill_threshold:
                    if self.debug:
                        log.info(f"[{symbol}] EXIT (short): Gap fill (${bar.close:.2f} > ${gap_fill_threshold:.2f})")
                    self.in_position[symbol] = False
                    self.position_direction.pop(symbol, None)
                    self.entry_time.pop(symbol, None)
                    self.entry_price.pop(symbol, None)
                    self.reference_price.pop(symbol, None)
                    return Signal(SignalType.BUY)  # Cover short
            
            # Exit 3: Scheduled exit time
            if self._should_exit_time(eastern_time):
                if self.debug:
                    log.info(f"[{symbol}] EXIT ({direction}): Scheduled time {self.exit_time_hour}:{self.exit_time_minute:02d}")
                self.in_position[symbol] = False
                self.position_direction.pop(symbol, None)
                self.entry_time.pop(symbol, None)
                self.entry_price.pop(symbol, None)
                self.reference_price.pop(symbol, None)
                return Signal(SignalType.SELL if direction == "long" else SignalType.BUY)
            
            return None
        
        # === ENTRY LOGIC ===
        
        # Only trade in first N minutes after open
        minutes_since_open = (eastern_time.hour - 9) * 60 + (eastern_time.minute - 30)
        
        # Only log debug info once per day during entry window
        if self.debug and not self.debug_logged.get(symbol, False) and minutes_since_open == 0:
            gap_pct = ((bar.close - ref_high) / ref_high) * 100 if ref_high > 0 else 0
            log.info(f"[{symbol}] Market open - Price: ${bar.close:.2f}, Ref: ${ref_high:.2f}, "
                    f"Gap: {gap_pct:.2f}%, Volume: {self.cumulative_volume.get(symbol, 0):,}")
            self.debug_logged[symbol] = True
        
        if minutes_since_open > self.trade_cutoff_minute:
            return None
        
        # Skip if already entered today
        if self.first_break_done.get(symbol, False):
            return None
        
        # Filter 1: Price must be under maximum
        if bar.close > self.max_price:
            return None
        
        # Filter 2: Minimum volume requirement
        if self.cumulative_volume.get(symbol, 0) < self.min_volume:
            return None
        
        # Filter 3: Volume surge requirement (if we have average data)
        avg_vol = self.avg_volume.get(symbol, 0)
        if avg_vol > 0:
            if bar.volume < (avg_vol * self.volume_surge_factor):
                return None
        
        # Filter 4: Calculate gap percentage
        if ref_high <= 0:
            return None
            
        gap_pct = ((bar.close - ref_high) / ref_high) * 100
        abs_gap = abs(gap_pct)
        
        # Determine if this is a gap-up or gap-down
        is_gap_up = gap_pct >= self.min_gap_pct
        is_gap_down = gap_pct <= -self.min_gap_pct
        
        # Check if we should trade this direction
        should_trade_long = is_gap_up and self.trade_direction in ["long_only", "both"]
        should_trade_short = is_gap_down and self.trade_direction in ["short_only", "both"]
        
        if not (should_trade_long or should_trade_short):
            return None
        
        # Determine direction
        direction = "long" if should_trade_long else "short"
        
        # Check for breakout with optional confirmation bars
        buf = self.buffer.setdefault(symbol, deque(maxlen=max(1, self.confirm_bars)))
        buf.append(bar.close)
        
        # Breakout condition depends on direction
        if direction == "long":
            # Gap up: high breaks above reference AND close above reference
            broke = bar.high >= ref_high and bar.close >= ref_high
            if self.confirm_bars > 0:
                broke = broke and len(buf) == self.confirm_bars and all(x >= ref_high for x in buf)
        else:
            # Gap down: low breaks below reference AND close below reference
            broke = bar.low <= ref_high and bar.close <= ref_high
            if self.confirm_bars > 0:
                broke = broke and len(buf) == self.confirm_bars and all(x <= ref_high for x in buf)
        
        if broke:
            if self.debug:
                log.info(f"[{symbol}] ENTRY ({direction.upper()}): Gap {gap_pct:.2f}% @ ${bar.close:.2f} "
                        f"(Ref: ${ref_high:.2f}, Vol: {self.cumulative_volume.get(symbol, 0):,})")
            
            self.first_break_done[symbol] = True
            self.in_position[symbol] = True
            self.position_direction[symbol] = direction
            self.entry_time[symbol] = eastern_time
            self.entry_price[symbol] = bar.close
            self.reference_price[symbol] = ref_high
            
            return Signal(SignalType.BUY if direction == "long" else SignalType.SELL)
        
        return None

    def on_stop(self, session_state: SessionState) -> None:
        """Cleanup when strategy stops"""
        if self.debug:
            log.info("GapAndGo stopped")
