from __future__ import annotations
import threading
import logging
import time as _time
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytz
import queue

from .state import AppSettings, SessionState, RunMode, SignalType, ForceMode
from .strategy import STRATEGIES, load_external_strategies
from .strategy.base import StrategyBase
from .broker.alpaca_adapter import AlpacaAdapter
from .backtest.engine import run_backtest
from .backtest.data import load_bars, set_backtest_source
from .realtime.stream import RealtimeManager

# --- helpers for slot time windows ---
from datetime import time as dtime
import pytz
from .state import StrategySlot

_east = pytz.timezone("America/New_York")

def _parse_hhmm(hhmm: str) -> dtime:
    try:
        hh, mm = hhmm.split(":")
        return dtime(hour=int(hh), minute=int(mm))
    except Exception:
        return dtime(9,30)

def _in_window_east(bar_ts_utc: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    t = bar_ts_utc.astimezone(_east).time()
    a = _parse_hhmm(start_hhmm); b = _parse_hhmm(end_hhmm)
    if a <= b:
        return a <= t <= b
    return (t >= a) or (t <= b)
# -------------------------------------

log = logging.getLogger(__name__)
trades_log = logging.getLogger("trades")

class Controller:
    def __init__(self, settings: AppSettings):
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.recent_trades: List[Dict[str, Any]] = []
        self.settings = settings
        self.state = SessionState()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.clear()
        self._adapter: Optional[AlpacaAdapter] = None
        self._live_confirmed = False
        load_external_strategies(self.settings.extra_strategy_paths)

    def _log_trade_entry(self, symbol: str, side: str, qty: int, price: float, 
                         sl: float, tp: float, strategy_name: str, 
                         slot_info: dict = None, settings: dict = None):
        """Comprehensive trade entry logging - COMPACT format"""
        log_parts = [
            f"ENTRY {side.upper()} {symbol}",
            f"x{qty}@${price:.2f}",
            f"SL${sl:.2f}",
            f"TP${tp:.2f}",
        ]
        
        # Strategy info
        if slot_info:
            log_parts.append(f"{slot_info.get('name', strategy_name)}[P{slot_info.get('priority', 'N/A')}]")
            log_parts.append(f"{slot_info.get('timeframe', 'N/A')}")
        else:
            log_parts.append(f"{strategy_name}")
        
        # Risk parameters (compact)
        if settings:
            log_parts.append(f"R{settings.get('risk_percent', 'N/A')}%")
        
        trades_log.info(" | ".join(log_parts))

    def _log_trade_exit(self, symbol: str, side: str, qty: int, entry_price: float,
                        exit_price: float, reason: str, strategy_name: str = None,
                        slot_info: dict = None):
        """Comprehensive trade exit logging - COMPACT format"""
        if side == "BUY":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = ((exit_price / entry_price) - 1) * 100
        else:
            pnl = (entry_price - exit_price) * qty
            pnl_pct = ((entry_price / exit_price) - 1) * 100
        
        log_parts = [
            f"EXIT {reason} {symbol}",
            f"x{qty}",
            f"${entry_price:.2f}→${exit_price:.2f}",
            f"P&L ${pnl:+.2f}({pnl_pct:+.2f}%)",
        ]
        
        # Strategy info (compact)
        if slot_info:
            log_parts.append(f"{slot_info.get('name', strategy_name or 'N/A')}[P{slot_info.get('priority', 'N/A')}]")
        elif strategy_name:
            log_parts.append(f"{strategy_name}")
        
        trades_log.info(" | ".join(log_parts))

    def connect(self, api_key: str, api_secret: str) -> str:
        self._adapter = AlpacaAdapter(api_key, api_secret, force_mode=self.settings.force_mode.value if hasattr(self.settings.force_mode, "value") else str(self.settings.force_mode))
        mode = self._adapter.connect()
        self.state.connection_mode = mode
        log.info("Connection mode: %s", mode.upper())
        return mode

    def start(self, run_mode: RunMode, run_folder: Optional[Path]=None, live_confirmed: bool=False) -> None:
        if self._worker and self._worker.is_alive():
            log.warning("Worker already running.")
            return
        
        self.positions.clear()
        self.recent_trades.clear()
        
        self.state.run_mode = run_mode
        self.state.started = True
        self.state.paused = False
        self.state.should_stop = False
        self.state.flatten_on_stop = self.settings.flatten_on_stop
        self.state.run_folder = run_folder
        self._stop_event.clear()
        self._pause_event.clear()
        self._live_confirmed = live_confirmed

        self._worker = threading.Thread(target=self._run_loop, name=f"{run_mode}-worker", daemon=True)
        self._worker.start()

    def pause(self) -> None:
        self.state.paused = True
        self._pause_event.set()
        log.info("Paused: new entries halted; managing exits continues.")

    def resume(self) -> None:
        self.state.paused = False
        self._pause_event.clear()
        log.info("Resumed: new entries allowed.")

    def stop(self, flatten: bool=False) -> None:
        if flatten or self.settings.flatten_on_stop:
            log.info("Flatten & Stop requested.")
            if self._adapter and self.state.run_mode == RunMode.LIVE:
                try:
                    self._adapter.flatten_all()
                except Exception as e:
                    log.exception("Flatten failed: %s", e)
        self.state.should_stop = True
        self._stop_event.set()

    def _run_loop(self) -> None:
        try:
            if self.state.run_mode == RunMode.LIVE:
                self._run_live()
            else:
                self._run_backtest()
        except Exception as e:
            log.exception("Worker crashed: %s", e)
        finally:
            self.state.started = False
            self.state.paused = False
            self.state.should_stop = False
            self.positions.clear()
            log.info("Run ended.")

    def _wait_for_market_open(self) -> bool:
        if not self._adapter:
            return False
        log.info("Market appears closed. Waiting for next open...")
        while not self._stop_event.is_set():
            is_open, next_open, _ = self._adapter.get_clock_info()
            if is_open:
                log.info("Market opened. Starting live loop.")
                return True
            sleep_s = 30
            if next_open:
                now = datetime.now(timezone.utc)
                delta = (next_open - now).total_seconds()
                if delta <= 60:
                    sleep_s = 5
                elif delta <= 300:
                    sleep_s = 15
            _time.sleep(sleep_s)
        return False

    def _run_live(self) -> None:
        if not self._adapter:
            log.error("No connection. Provide API keys and connect first.")
            return

        if not self._adapter.is_market_open_now():
            if not self._wait_for_market_open():
                return

        if self.state.connection_mode == "live" and not self._live_confirmed:
            log.warning("Live mode requires user confirmation. Aborting start.")
            return

        symbols = [s.strip().upper() for s in self.settings.symbols.split(",") if s.strip()]
        tf = self.settings.timeframe
        # global defaults (fractions)
        risk_pct = self.settings.risk_percent / 100.0
        sl_pct = self.settings.stop_loss_percent / 100.0
        tp_pct = self.settings.take_profit_percent / 100.0
        lunch_skip = self.settings.lunch_skip
        east = pytz.timezone("America/New_York")

        # --------- Multi-slot init (priority order) ----------
        slots: List[StrategySlot] = getattr(self.settings, "strategy_slots", []) or []
        enabled_slots = [s for s in slots if s.enabled and s.name in STRATEGIES]
        enabled_slots.sort(key=lambda s: s.priority)  # 1 = highest

        slot_strats: List[tuple[StrategySlot, StrategyBase]] = []
        for s in enabled_slots:
            try:
                strat_cls = STRATEGIES[s.name]
                slot_strats.append((s, strat_cls()))
            except Exception as e:
                log.warning("Slot strategy init failed (%s): %s", s.name, e)
        multi_mode = len(slot_strats) > 0
        # -----------------------------------------------------

        # single-strategy fallback
        strategy: Optional[StrategyBase] = None
        if not multi_mode:
            strat_cls = STRATEGIES.get(self.settings.selected_strategy)
            if not strat_cls:
                log.error("Strategy not found: %s", self.settings.selected_strategy)
                return
            strategy = strat_cls()
            strategy.on_start(self.state)
        else:
            for s, strat in slot_strats:
                try:
                    strat.on_start(self.state)
                except Exception as e:
                    log.debug("on_start %s failed: %s", s.name, e)

        positions: Dict[str, Dict[str, Any]] = {}

        md_queue: "queue.Queue" = queue.Queue()
        tu_queue: "queue.Queue" = queue.Queue()
        stream = None
        try:
            stream = RealtimeManager(
                api_key=self._adapter.api_key,
                api_secret=self._adapter.api_secret,
                symbols=symbols,
                paper=(self.state.connection_mode != "live"),
                on_bar_queue=md_queue,
                on_trade_update_queue=tu_queue,
            )
            stream.start()
            log.info("Realtime streams started (data & trading; IEX feed).")
        except Exception as e:
            log.warning("Streaming unavailable (%s). Falling back to REST polling.", e)
            stream = None

        log.info("Live loop started for %s | TF=%s | mode=%s", symbols, tf, self.state.connection_mode)

        last_pl_update_ts = 0.0
        while not self._stop_event.is_set():
            paused = self._pause_event.is_set()

            try:
                while True:
                    evt = tu_queue.get_nowait()
                    if evt[0] == "trade_update":
                        try:
                            self.state.realized_pnl = self._adapter.get_today_pnl()
                        except Exception as e:
                            log.debug("today PnL refresh failed: %s", e)
            except queue.Empty:
                pass

            try:
                evt = md_queue.get(timeout=1.0) if stream else None
            except queue.Empty:
                evt = None

            now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)

            if evt:
                etype, sym, bar = evt
                if etype == "bar" and sym in symbols:
                    o = float(bar["o"]); h = float(bar["h"]); l = float(bar["l"]); c = float(bar["c"])
                    # bar timestamp normalization
                    bar_ts = now_utc
                    try:
                        tval = bar.get("t")
                        if tval is not None:
                            if isinstance(tval, (int, float)):
                                if tval > 1e12:
                                    bar_ts = datetime.fromtimestamp(tval / 1e9, tz=timezone.utc)
                                elif tval > 1e10:
                                    bar_ts = datetime.fromtimestamp(tval / 1e3, tz=timezone.utc)
                                else:
                                    bar_ts = datetime.fromtimestamp(tval, tz=timezone.utc)
                            elif isinstance(tval, str):
                                iso = tval.replace("Z", "+00:00")
                                dtp = datetime.fromisoformat(iso)
                                bar_ts = dtp if dtp.tzinfo else dtp.replace(tzinfo=timezone.utc)
                    except Exception:
                        bar_ts = now_utc

                    pos = positions.get(sym)
                    if pos:
                        if sym in self.positions:
                            self.positions[sym]["current_price"] = c
                            entry_px = self.positions[sym]["entry_price"]
                            qty = self.positions[sym]["qty"]
                            pnl = (c - entry_px) * qty if self.positions[sym]["side"] == "BUY" else (entry_px - c) * qty
                            pnl_pct = ((c / entry_px) - 1) * 100 if self.positions[sym]["side"] == "BUY" else ((entry_px / c) - 1) * 100
                            self.positions[sym]["pnl"] = pnl
                            self.positions[sym]["pnl_pct"] = pnl_pct
                        
                        side = pos["side"]; sl = pos["sl"]; tp = pos["tp"]
                        exit_now = False; exit_px = None; exit_reason = ""
                        if side == "BUY":
                            if l <= sl: exit_now = True; exit_px = sl; exit_reason = "Stop Loss"
                            elif h >= tp: exit_now = True; exit_px = tp; exit_reason = "Take Profit"
                        else:
                            if h >= sl: exit_now = True; exit_px = sl; exit_reason = "Stop Loss"
                            elif l <= tp: exit_now = True; exit_px = tp; exit_reason = "Take Profit"
                        
                        if exit_now and exit_px is not None:
                            try:
                                qty = pos["qty"]
                                side_close = "sell" if side == "BUY" else "buy"
                                self._adapter.submit_market_order(sym, qty, side_close)
                                
                                # Enhanced exit logging
                                slot_info = None
                                if "slot" in self.positions.get(sym, {}):
                                    slot_info = {
                                        "name": self.positions[sym].get("slot"),
                                        "priority": self.positions[sym].get("priority"),
                                    }
                                self._log_trade_exit(
                                    sym, side, qty, pos["entry_price"], exit_px,
                                    exit_reason.upper().replace(" ", "_"),
                                    None, slot_info
                                )
                                
                                positions.pop(sym, None)
                                self.recent_trades.insert(0, {
                                    "time": bar_ts.strftime("%H:%M:%S"),
                                    "action": "CLOSE",
                                    "symbol": sym,
                                    "price": f"${exit_px:.2f}",
                                    "qty": qty,
                                    "reason": exit_reason
                                })
                                if len(self.recent_trades) > 15:
                                    self.recent_trades.pop()
                                self.positions.pop(sym, None)
                            except Exception as e:
                                log.exception("Exit order failed: %s", e)

                    allow_entry = True
                    # (global lunch check stays for single-strategy mode; per-slot handled below for multi-mode)
                    if not multi_mode and lunch_skip:
                        ts_east = bar_ts.astimezone(east)
                        if ts_east.hour == 12:
                            allow_entry = False

                    if allow_entry and not paused and sym not in positions:
                        bar_obj = type("BarObj", (), dict(timestamp=bar_ts, open=o, high=h, low=l, close=c, volume=0))

                        if multi_mode:
                            chosen_signal = None
                            chosen_slot: Optional[StrategySlot] = None
                            for s, strat in slot_strats:  # priority order
                                if not _in_window_east(bar_ts, s.start_hhmm, s.end_hhmm):
                                    continue
                                try:
                                    sig = strat.on_bar(sym, bar_obj, self.state)
                                except Exception as e:
                                    log.debug("on_bar %s error: %s", s.name, e); sig = None
                                if sig and sig.type in (SignalType.BUY, SignalType.SELL):
                                    chosen_signal = sig
                                    chosen_slot = s
                                    break

                            if chosen_signal and chosen_slot:
                                # per-slot lunch skip check
                                eff_lunch = chosen_slot.lunch_skip if chosen_slot.lunch_skip is not None else False
                                if eff_lunch:
                                    ts_east = bar_ts.astimezone(east)
                                    if ts_east.hour == 12:
                                        chosen_signal = None

                            if chosen_signal and chosen_slot:
                                # Use slot's own parameters (no more "use_global")
                                eff_risk = chosen_slot.risk_percent if chosen_slot.risk_percent is not None else 1.0
                                eff_sl = chosen_slot.sl_percent if chosen_slot.sl_percent is not None else 1.0
                                eff_tp = chosen_slot.tp_percent if chosen_slot.tp_percent is not None else 2.0

                                risk_fraction = float(eff_risk) / 100.0
                                sl_fraction = float(eff_sl) / 100.0
                                tp_fraction = float(eff_tp) / 100.0

                                if sl_fraction > 0:
                                    try:
                                        equity = self._adapter.get_account_equity()
                                        qty = max(1, int((risk_fraction * equity) / (c * sl_fraction)))
                                        side = "buy" if chosen_signal.type == SignalType.BUY else "sell"
                                        self._adapter.submit_market_order(sym, qty, side)
                                        if side == "buy":
                                            sl = c * (1 - sl_fraction); tp = c * (1 + tp_fraction)
                                        else:
                                            sl = c * (1 + sl_fraction); tp = c * (1 - tp_fraction)
                                        positions[sym] = dict(side=side.upper(), qty=qty, entry_price=c, sl=sl, tp=tp,
                                                             slot_name=chosen_slot.name, slot_priority=chosen_slot.priority)
                                        self.positions[sym] = {
                                            "symbol": sym, "side": side.upper(),
                                            "entry_time": bar_ts.strftime("%H:%M:%S"),
                                            "entry_price": c, "current_price": c, "qty": qty,
                                            "pnl": 0.0, "pnl_pct": 0.0, "stop_loss": sl, "take_profit": tp,
                                            "slot": chosen_slot.name, "priority": chosen_slot.priority
                                        }
                                        self.recent_trades.insert(0, {
                                            "time": bar_ts.strftime("%H:%M:%S"), "action": "OPEN",
                                            "symbol": sym, "price": f"${c:.2f}", "qty": qty,
                                            "reason": f"{chosen_slot.name}[P{chosen_slot.priority}]"
                                        })
                                        if len(self.recent_trades) > 15: self.recent_trades.pop()
                                        
                                        # Enhanced logging
                                        slot_info = {
                                            "name": chosen_slot.name,
                                            "priority": chosen_slot.priority,
                                            "timeframe": chosen_slot.timeframe,
                                            "start_hhmm": chosen_slot.start_hhmm,
                                            "end_hhmm": chosen_slot.end_hhmm,
                                            "lunch_skip": chosen_slot.lunch_skip if chosen_slot.lunch_skip is not None else False,
                                        }
                                        settings_info = {
                                            "risk_percent": eff_risk,
                                            "sl_percent": eff_sl,
                                            "tp_percent": eff_tp,
                                        }
                                        self._log_trade_entry(sym, side, qty, c, sl, tp, 
                                                             chosen_slot.name, slot_info, settings_info)
                                        
                                    except Exception as e:
                                        log.exception("Entry order failed: %s", e)
                        else:
                            sig = strategy.on_bar(sym, bar_obj, self.state)
                            if sig and sig.type in (SignalType.BUY, SignalType.SELL):
                                if sl_pct > 0:
                                    try:
                                        equity = self._adapter.get_account_equity()
                                        qty = max(1, int((risk_pct * equity) / (c * sl_pct)))
                                        side = "buy" if sig.type == SignalType.BUY else "sell"
                                        self._adapter.submit_market_order(sym, qty, side)
                                        if side == "buy":
                                            sl = c * (1 - sl_pct); tp = c * (1 + tp_pct)
                                        else:
                                            sl = c * (1 + sl_pct); tp = c * (1 - tp_pct)
                                        positions[sym] = dict(side=side.upper(), qty=qty, entry_price=c, sl=sl, tp=tp)
                                        self.positions[sym] = {
                                            "symbol": sym,
                                            "side": side.upper(),
                                            "entry_time": bar_ts.strftime("%H:%M:%S"),
                                            "entry_price": c,
                                            "current_price": c,
                                            "qty": qty,
                                            "pnl": 0.0,
                                            "pnl_pct": 0.0,
                                            "stop_loss": sl,
                                            "take_profit": tp
                                        }
                                        self.recent_trades.insert(0, {
                                            "time": bar_ts.strftime("%H:%M:%S"),
                                            "action": "OPEN",
                                            "symbol": sym,
                                            "price": f"${c:.2f}",
                                            "qty": qty,
                                            "reason": "Strategy Signal"
                                        })
                                        if len(self.recent_trades) > 15:
                                            self.recent_trades.pop()
                                        
                                        # Enhanced entry logging
                                        settings_info = {
                                            "risk_percent": self.settings.risk_percent,
                                            "sl_percent": self.settings.stop_loss_percent,
                                            "tp_percent": self.settings.take_profit_percent,
                                        }
                                        self._log_trade_entry(sym, side, qty, c, sl, tp,
                                                             self.settings.selected_strategy, None, settings_info)
                                    except Exception as e:
                                        log.exception("Entry order failed: %s", e)

            now_ts = _time.time()
            if now_ts - last_pl_update_ts >= 2.0:
                try:
                    self.state.realized_pnl = self._adapter.get_today_pnl()
                except Exception:
                    pass
                try:
                    self.state.unrealized_pnl = self._adapter.get_unrealized_pl_sum()
                except Exception:
                    pass
                self.state.last_pl_update = datetime.utcnow()
                last_pl_update_ts = now_ts

        try:
            if stream:
                stream.stop()
        except Exception:
            pass

        if not multi_mode and strategy:
            strategy.on_stop(self.state)
        else:
            for _, strat in slot_strats:
                try: strat.on_stop(self.state)
                except Exception: pass

    def _run_backtest(self) -> None:
        symbols = [s.strip().upper() for s in self.settings.symbols.split(",") if s.strip()]
        tf = self.settings.timeframe
        strat_cls = STRATEGIES.get(self.settings.selected_strategy)
        if not strat_cls:
            log.error("Strategy not found: %s", self.settings.selected_strategy)
            return
        strategy: StrategyBase = strat_cls()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("backtests") / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        self.state.run_folder = run_dir

        backtest_logger = logging.getLogger("backtest")
        fh = logging.FileHandler(run_dir / "backtest.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        backtest_logger.addHandler(fh)
        backtest_logger.setLevel(logging.INFO)

        adapter_for_data: Optional[AlpacaAdapter] = None
        try:
            from .config_store import load_credentials
            k, s = load_credentials()
            if k and s:
                adapter_for_data = AlpacaAdapter(k, s, force_mode=self.settings.force_mode.value if hasattr(self.settings.force_mode, "value") else str(self.settings.force_mode))
                adapter_for_data.connect()
        except Exception as e:
            log.warning("Backtest: could not init Alpaca data adapter: %s", e)

        # UPDATED: Use date range from settings instead of calculating from years
        end = self.settings.backtest_end_date
        start = self.settings.backtest_start_date
        
        # Ensure dates are timezone-aware
        if end and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if start and start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        
        # Fallback to defaults if not set
        if not end:
            end = datetime.now(timezone.utc)
        if not start:
            start = end - timedelta(days=730)  # 2 years default
        
        # Log the date range
        log.info(f"Backtest date range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({(end-start).days} days)")
        
        # Validate date range
        if start >= end:
            log.error("Invalid date range: start date must be before end date")
            return
        
        # Warn if range exceeds Polygon free tier limit (2 years)
        days_diff = (end - start).days
        if days_diff > 730 and self.settings.backtest_source.value == "polygon":
            log.warning("Date range exceeds 730 days (%d days). Polygon free tier is limited to 2 years of data.", days_diff)

        try:
            set_backtest_source(self.settings.backtest_source.value if hasattr(self.settings.backtest_source, 'value') else str(self.settings.backtest_source))
        except Exception:
            set_backtest_source('alpaca')

        bars_cache = {}
        def loader(sym: str):
            key = (sym, tf, start, end)
            if key not in bars_cache:
                bars_cache[key] = load_bars(sym, tf, start, end, adapter_for_data)
            return bars_cache[key]

        try:
            stats = run_backtest(symbols, tf, strategy, vars(self.settings), loader, run_dir)
            self.state.stats = stats
            self.state.realized_pnl = stats.get("total_pnl", 0.0)
            self.state.unrealized_pnl = 0.0
            self.state.last_pl_update = datetime.utcnow()
        finally:
            backtest_logger.removeHandler(fh)
            fh.close()
