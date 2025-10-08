from __future__ import annotations
import logging
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from .strategy import STRATEGIES
from .state import StrategySlot
import tkinter as tk
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from tkcalendar import DateEntry
from datetime import datetime, timedelta

import customtkinter as ctk

from .state import AppSettings, SessionState, RunMode, ForceMode, BacktestSource
from .config_store import load_settings, save_settings
from .controller import Controller

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class UITextHandler(logging.Handler):
    def __init__(self, text_widget: ctk.CTkTextbox):
        super().__init__()
        self.text_widget = text_widget
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            self.text_widget.after(0, self._append, msg + "\n")
        except Exception:
            pass
    def _append(self, text: str) -> None:
        try:
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", text)
            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")
        except Exception:
            pass

def _try_call(controller, names: List[str], *args, **kwargs):
    for n in names:
        fn = getattr(controller, n, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                return fn()
    raise AttributeError(f"Controller has none of: {', '.join(names)}")

def _find_latest_backtest_folder() -> Optional[Path]:
    bdir = Path("backtests")
    if not bdir.exists(): return None
    runs = sorted([p for p in bdir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def _load_equity_csv(run_folder: Path):
    import pandas as pd
    f = run_folder / "equity.csv"
    if not f.exists(): return [], []
    df = pd.read_csv(f)
    tcol = None
    for c in ("timestamp","Timestamp","datetime","time","date","DateTime","Date"):
        if c in df.columns: tcol=c; break
    if tcol is None: return [], []
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    m = ts.notna(); ts = ts[m]
    ecol = "equity" if "equity" in df.columns else ("Equity" if "Equity" in df.columns else None)
    if ecol is None: return [], []
    ys = df.loc[m, ecol].astype(float)
    return [t.to_pydatetime() for t in ts], list(ys.values)

def _plot_series(canvas: FigureCanvasTkAgg, x, y, title: str, ylabel: str):
    fig: Figure = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    if x and y:
        ax.plot(x, y)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel(ylabel)
    fig.tight_layout(); canvas.draw()

# ---------- NEW: tiny helper to build/read strategy slots ----------
def _collect_slots_from_ui(slot_var_list) -> List[StrategySlot]:
    slots: List[StrategySlot] = []
    for v in slot_var_list:
        try:
            enabled = bool(v["enabled"].get())
            name = v["name"].get().strip()
            prio = int(v["prio"].get())
            start = v["start"].get().strip()
            end = v["end"].get().strip()
            use_global = bool(v["use_global"].get())
            r = float(v["risk"].get() or 0.0)
            s = float(v["sl"].get() or 0.0)
            t = float(v["tp"].get() or 0.0)
            slots.append(StrategySlot(
                enabled=enabled, name=name, priority=prio,
                start_hhmm=start, end_hhmm=end,
                timeframe=v.get("tf").get() if "tf" in v else "1m",         # NEW
                lunch_skip=bool(v.get("lunch").get()) if "lunch" in v else None,  # NEW (None means use global)
                use_global=use_global,
                risk_percent=None if use_global else r,
                sl_percent=None if use_global else s,
                tp_percent=None if use_global else t
            ))
        except Exception:
            # ignore malformed rows
            pass
    return slots

def _serialize_slots(slots: List[StrategySlot]) -> List[dict]:
    out = []
    for s in slots:
        out.append(dict(
            enabled=s.enabled, name=s.name, priority=s.priority,
            start_hhmm=s.start_hhmm, end_hhmm=s.end_hhmm,
            timeframe=getattr(s, "timeframe", "1m"),              # NEW
            lunch_skip=getattr(s, "lunch_skip", None),            # NEW
            use_global=s.use_global,
            risk_percent=s.risk_percent, sl_percent=s.sl_percent, tp_percent=s.tp_percent
        ))
    return out
# ------------------------------------------------------------------

def run_ui() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    raw = load_settings()
    # load raw saved slots (if any)
    raw_slots = raw.get("strategy_slots", []) or []
    slots_loaded: List[StrategySlot] = []
    try:
        for d in raw_slots:
            slots_loaded.append(StrategySlot(
            enabled=bool(d.get("enabled", False)),
            name=str(d.get("name","BaselineSMA")),
            priority=int(d.get("priority", 1)),
            start_hhmm=str(d.get("start_hhmm","09:30")),
            end_hhmm=str(d.get("end_hhmm","16:00")),
            timeframe=str(d.get("timeframe","1m")),                 # NEW
            lunch_skip=d.get("lunch_skip", None),                   # NEW (None = use global)
            use_global=bool(d.get("use_global", True)),
            risk_percent=d.get("risk_percent", None),
            sl_percent=d.get("sl_percent", None),
            tp_percent=d.get("tp_percent", None),
        ))
    except Exception:
        slots_loaded = []

    try:
        settings = AppSettings(
            symbols=str(raw.get("symbols", "AAPL,MSFT")),
            timeframe=str(raw.get("timeframe", "1m")),
            lunch_skip=bool(raw.get("lunch_skip", True)),
            risk_percent=float(raw.get("risk_percent", 1.0)),
            stop_loss_percent=float(raw.get("stop_loss_percent", 1.0)),
            take_profit_percent=float(raw.get("take_profit_percent", 2.0)),
            selected_strategy=str(raw.get("selected_strategy", "BaselineSMA")),
            flatten_on_stop=bool(raw.get("flatten_on_stop", False)),
            force_mode=ForceMode(raw.get("force_mode","auto")) if isinstance(raw.get("force_mode","auto"), str) else raw.get("force_mode", ForceMode.AUTO),
            extra_strategy_paths=list(raw.get("extra_strategy_paths", [])),
            backtest_start_date=raw.get("backtest_start_date"),
            backtest_end_date=raw.get("backtest_end_date"),
            backtest_source=BacktestSource(raw.get("backtest_source","alpaca")) if isinstance(raw.get("backtest_source","alpaca"), str) else raw.get("backtest_source", BacktestSource.ALPACA),
            data_feed=str(raw.get("data_feed", "iex")),
            strategy_slots=slots_loaded,  # NEW
        )
    except Exception:
        settings = AppSettings()

    state = SessionState(run_mode=RunMode.BACKTEST, flatten_on_stop=settings.flatten_on_stop)
    controller = Controller(settings)

    root = ctk.CTk()
    root.title("Alpaca Stock Bot - Professional Edition")
    root.geometry("1400x900")
    
    # Status bar at top
    status_bar = ctk.CTkFrame(root, height=50)
    status_bar.pack(fill="x", padx=10, pady=(10,0))
    
    conn_status_label = ctk.CTkLabel(status_bar, text="CONNECTION:", font=("Arial", 11, "bold"))
    conn_status_label.pack(side="left", padx=(10,5))
    conn_status_value = ctk.CTkLabel(status_bar, text="Disconnected", font=("Arial", 11))
    conn_status_value.pack(side="left", padx=(0,20))
    
    market_status_label = ctk.CTkLabel(status_bar, text="MARKET:", font=("Arial", 11, "bold"))
    market_status_label.pack(side="left", padx=(0,5))
    market_status_value = ctk.CTkLabel(status_bar, text="Unknown", font=("Arial", 11))
    market_status_value.pack(side="left", padx=(0,20))
    
    bot_status_label = ctk.CTkLabel(status_bar, text="BOT:", font=("Arial", 11, "bold"))
    bot_status_label.pack(side="left", padx=(0,5))
    bot_status_value = ctk.CTkLabel(status_bar, text="Stopped", font=("Arial", 11))
    bot_status_value.pack(side="left")
    
    def update_status_bar():
        cm = getattr(controller.state, "connection_mode", None)
        if cm:
            conn_status_value.configure(text=cm.upper(), text_color="#00FF00")
        else:
            conn_status_value.configure(text="Disconnected", text_color="#888888")
        if hasattr(controller, '_adapter') and controller._adapter:
            try:
                is_open = controller._adapter.is_market_open_now()
                if is_open:
                    market_status_value.configure(text="OPEN", text_color="#00FF00")
                else:
                    market_status_value.configure(text="CLOSED", text_color="#FF0000")
            except Exception:
                market_status_value.configure(text="Unknown", text_color="#888888")
        if controller.state.started:
            if controller.state.paused:
                bot_status_value.configure(text="Paused", text_color="#FFAA00")
            else:
                bot_status_value.configure(text="Running", text_color="#00FF00")
        else:
            bot_status_value.configure(text="Stopped", text_color="#888888")
        root.after(3000, update_status_bar)
    root.after(1000, update_status_bar)
    
    tabs = ctk.CTkTabview(root, width=1360, height=800)
    tabs.pack(fill="both", expand=True, padx=10, pady=10)
    tab_conn = tabs.add("Connection")
    tab_trading = tabs.add("Trading")
    tab_backtest = tabs.add("Backtest")
    tab_charts = tabs.add("Charts")
    tab_logs = tabs.add("Logs")
    tab_settings = tabs.add("Settings")

    # ========== TRADING TAB ==========
    header1 = ctk.CTkLabel(tab_trading, text="Trading Controls", font=("Arial", 16, "bold"))
    header1.pack(anchor="w", padx=10, pady=(15,5))

    ctl = ctk.CTkFrame(tab_trading)
    ctl.pack(fill="x", padx=10, pady=(0,10))

    mode_var = tk.StringVar(value="Backtest")
    ctk.CTkOptionMenu(ctl, variable=mode_var, values=["Backtest","Live"]).pack(side="left", padx=(10,8), pady=10)

    activity_log_data = []

    def on_start():
        _save_current_settings()
        state.run_mode = RunMode.BACKTEST if mode_var.get().startswith("Back") else RunMode.LIVE
        controller.settings = settings
        if state.run_mode == RunMode.BACKTEST:
            _try_call(controller, ["start_backtest","start_backtesting","start"], RunMode.BACKTEST)
        else:
            _try_call(controller, ["start_live","start_trading","start"], RunMode.LIVE)
        live_buf["start_equity"] = None
        live_buf["x"].clear()
        live_buf["y"].clear()
        activity_log_data.clear()

    def on_pause():
        if not controller.state.started:
            return
        if not messagebox.askyesno("Confirm Pause", 
            "Pause will:\n• Stop NEW entries\n• Continue managing open positions\n• Keep stop-loss/take-profit active\n\nContinue?"):
            return
        _try_call(controller, ["pause","toggle_pause","set_pause"])

    def on_stop():
        controller.stop(flatten=False)

    def on_flatten_stop():
        if state.run_mode == RunMode.LIVE:
            if not messagebox.askyesno("Confirm Flatten", "Flatten & Stop will market-close all open positions. Continue?"): return
        controller.stop(flatten=True)

    start_btn = ctk.CTkButton(ctl, text="Start", command=on_start, fg_color="#00AA00", hover_color="#008800")
    start_btn.pack(side="left", padx=6, pady=10)

    pause_btn = ctk.CTkButton(ctl, text="Pause", command=on_pause, fg_color="#CCAA00", hover_color="#AA8800")
    pause_btn.pack(side="left", padx=6, pady=10)
    pause_btn.configure(state="disabled")

    stop_btn = ctk.CTkButton(ctl, text="Stop", command=on_stop, fg_color="#666666", hover_color="#555555")
    stop_btn.pack(side="left", padx=6, pady=10)
    stop_btn.configure(state="disabled")

    flatten_stop_btn = ctk.CTkButton(ctl, text="⚠️ Flatten & Stop", command=on_flatten_stop, fg_color="#FF0000", hover_color="#CC0000")
    flatten_stop_btn.pack(side="left", padx=6, pady=10)
    flatten_stop_btn.configure(state="disabled")

    def update_button_states():
        if controller.state.started:
            start_btn.configure(state="disabled")
            pause_btn.configure(state="normal")
            stop_btn.configure(state="normal")
            flatten_stop_btn.configure(state="normal")
        else:
            start_btn.configure(state="normal")
            pause_btn.configure(state="disabled")
            stop_btn.configure(state="disabled")
            flatten_stop_btn.configure(state="disabled")
        root.after(500, update_button_states)
    root.after(500, update_button_states)

    # Simplified Settings (Symbols only)
    header2 = ctk.CTkLabel(tab_trading, text="Trading Settings", font=("Arial", 16, "bold"))
    header2.pack(anchor="w", padx=10, pady=(15,5))

    form = ctk.CTkFrame(tab_trading)
    form.pack(fill="x", padx=10, pady=(0,10))

    ctk.CTkLabel(form, text="Symbols (comma-separated)", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    symbols_var = tk.StringVar(value=settings.symbols)
    ctk.CTkEntry(form, textvariable=symbols_var, width=600).grid(row=0, column=1, padx=10, pady=10, sticky="w")

    # Strategy Playbook (Self-Contained, No "Use Global")
    playbook = ctk.CTkFrame(tab_trading)
    playbook.pack(fill="x", padx=10, pady=(10,10))
    ctk.CTkLabel(
        playbook,
        text="Strategy Playbook (All settings per strategy)",
        font=("Arial", 14, "bold")
    ).grid(row=0, column=0, columnspan=12, padx=10, pady=(10,6), sticky="w")

    # Header row (removed "Use Global" column)
    hdrs = ["On", "Strategy", "TF", "Prio", "Start", "End", "Skip", "Risk%", "SL%", "TP%", ""]
    for i, h in enumerate(hdrs):
        ctk.CTkLabel(playbook, text=h, font=("Arial", 10, "bold")).grid(row=1, column=i, padx=6, pady=(2,4), sticky="w")

    strategy_names = sorted(STRATEGIES.keys()) or ["BaselineSMA"]
    root._slot_vars = []

    def _place_add_button():
        """Place/relocate the Add button under the last row."""
        add_row = 2 + len(root._slot_vars)
        add_btn.grid(row=add_row, column=0, padx=6, pady=(6,4), sticky="w", columnspan=2)

    def _add_slot_row(preset: Optional[StrategySlot] = None):
        """Append one strategy slot row."""
        r = 2 + len(root._slot_vars)
        s = preset or StrategySlot()

        # Default values if using global (since we removed "use_global")
        default_risk = s.risk_percent if s.risk_percent is not None else settings.risk_percent
        default_sl = s.sl_percent if s.sl_percent is not None else settings.stop_loss_percent
        default_tp = s.tp_percent if s.tp_percent is not None else settings.take_profit_percent

        # tkinter variables for this row
        v_enabled = tk.BooleanVar(value=s.enabled)
        v_name = tk.StringVar(value=s.name if s.name in STRATEGIES else strategy_names[0])
        v_tf = tk.StringVar(value=(s.timeframe if s.timeframe in ("1m","3m","5m") else "1m"))
        v_prio = tk.StringVar(value=str(s.priority))
        v_start = tk.StringVar(value=s.start_hhmm)
        v_end = tk.StringVar(value=s.end_hhmm)
        v_skip = tk.BooleanVar(value=(s.lunch_skip if s.lunch_skip is not None else False))
        v_risk = tk.StringVar(value=f"{default_risk:.2f}")
        v_sl = tk.StringVar(value=f"{default_sl:.2f}")
        v_tp = tk.StringVar(value=f"{default_tp:.2f}")

        # widgets for the row (removed "Use Global" checkbox)
        ctk.CTkCheckBox(playbook, text="", variable=v_enabled).grid(row=r, column=0, padx=6, pady=2, sticky="w")
        ctk.CTkOptionMenu(playbook, values=strategy_names, variable=v_name, width=140).grid(row=r, column=1, padx=6, pady=2, sticky="w")
        ctk.CTkOptionMenu(playbook, values=["1m","3m","5m"], variable=v_tf, width=70).grid(row=r, column=2, padx=6, pady=2, sticky="w")
        ctk.CTkEntry(playbook, textvariable=v_prio, width=40).grid(row=r, column=3, padx=6, pady=2, sticky="w")
        ctk.CTkEntry(playbook, textvariable=v_start, width=60).grid(row=r, column=4, padx=4, pady=2, sticky="w")
        ctk.CTkEntry(playbook, textvariable=v_end, width=60).grid(row=r, column=5, padx=4, pady=2, sticky="w")
        ctk.CTkCheckBox(playbook, text="", variable=v_skip).grid(row=r, column=6, padx=6, pady=2, sticky="w")
        ctk.CTkEntry(playbook, textvariable=v_risk, width=60).grid(row=r, column=7, padx=4, pady=2, sticky="w")
        ctk.CTkEntry(playbook, textvariable=v_sl, width=60).grid(row=r, column=8, padx=4, pady=2, sticky="w")
        ctk.CTkEntry(playbook, textvariable=v_tp, width=60).grid(row=r, column=9, padx=4, pady=2, sticky="w")

        idx = len(root._slot_vars)

        def _remove_row():
            for c in playbook.grid_slaves(row=r):
                c.destroy()
            try:
                root._slot_vars.pop(idx)
            except Exception:
                pass
            _place_add_button()

        rm_btn = ctk.CTkButton(playbook, text="−", width=28, command=_remove_row)
        rm_btn.grid(row=r, column=10, padx=6, pady=2, sticky="w")

        # save this row's vars (removed use_global)
        root._slot_vars.append(dict(
            enabled=v_enabled, name=v_name, tf=v_tf,
            prio=v_prio, start=v_start, end=v_end, lunch=v_skip,
            risk=v_risk, sl=v_sl, tp=v_tp
        ))

    # "Add Slot +" button
    add_btn = ctk.CTkButton(playbook, text="Add Slot +", width=100,
                            command=lambda: (_add_slot_row(), _place_add_button()))

    # seed rows from saved settings (or two default rows)
    saved_slots = settings.strategy_slots or []
    if saved_slots:
        for s in saved_slots:
            _add_slot_row(s)
    else:
        # Add 2 default slots to start
        _add_slot_row(StrategySlot(enabled=False, name="GapAndGo", priority=1, timeframe="1m",
                                   risk_percent=2.0, sl_percent=0.5, tp_percent=1.0))
        _add_slot_row(StrategySlot(enabled=False, name="ORB", priority=2, timeframe="5m",
                                   risk_percent=1.5, sl_percent=1.0, tp_percent=2.0))

    _place_add_button()

    # Helper to collect slots (updated to remove use_global)
    def _collect_slots_from_ui(slot_var_list) -> List[StrategySlot]:
        slots: List[StrategySlot] = []
        for v in slot_var_list:
            try:
                enabled = bool(v["enabled"].get())
                name = v["name"].get().strip()
                prio = int(v["prio"].get())
                start = v["start"].get().strip()
                end = v["end"].get().strip()
                r = float(v["risk"].get() or 0.0)
                s = float(v["sl"].get() or 0.0)
                t = float(v["tp"].get() or 0.0)
                slots.append(StrategySlot(
                    enabled=enabled, name=name, priority=prio,
                    start_hhmm=start, end_hhmm=end,
                    timeframe=v.get("tf").get() if "tf" in v else "1m",
                    lunch_skip=bool(v.get("lunch").get()) if "lunch" in v else False,
                    use_global=False,  # Always False now
                    risk_percent=r,
                    sl_percent=s,
                    tp_percent=t
                ))
            except Exception:
                pass
        return slots

    def _serialize_slots(slots: List[StrategySlot]) -> List[dict]:
        out = []
        for s in slots:
            out.append(dict(
                enabled=s.enabled, name=s.name, priority=s.priority,
                start_hhmm=s.start_hhmm, end_hhmm=s.end_hhmm,
                timeframe=getattr(s, "timeframe", "1m"),
                lunch_skip=getattr(s, "lunch_skip", False),
                use_global=False,  # Always False
                risk_percent=s.risk_percent, 
                sl_percent=s.sl_percent, 
                tp_percent=s.tp_percent
            ))
        return out

    # Save settings function (updated)
    def _save_current_settings():
        settings.symbols = symbols_var.get().strip()
        # Use first enabled slot's settings as defaults, or keep existing
        slots = _collect_slots_from_ui(root._slot_vars)
        enabled_slots = [s for s in slots if s.enabled]
        if enabled_slots:
            first = enabled_slots[0]
            settings.timeframe = first.timeframe
            settings.risk_percent = first.risk_percent
            settings.stop_loss_percent = first.sl_percent
            settings.take_profit_percent = first.tp_percent
            settings.lunch_skip = first.lunch_skip
            settings.selected_strategy = first.name
        
        settings.force_mode = ForceMode(force_var.get())
        # NEW: Save date range from backtest tab
        if 'backtest_widgets' in globals() and backtest_widgets:
            try:
                settings.backtest_start_date = backtest_widgets['start_date_picker'].get_date()
                settings.backtest_end_date = backtest_widgets['end_date_picker'].get_date()
            except Exception:
                pass
        try: settings.backtest_source = BacktestSource(backtest_widgets['bsrc_var'].get()) if 'backtest_widgets' in globals() else BacktestSource.ALPACA
        except Exception: settings.backtest_source = BacktestSource.ALPACA
        settings.extra_strategy_paths = [p for p in extra_paths_var.get().split(";") if p.strip()]
        settings.strategy_slots = slots

        d = dict(
            symbols=settings.symbols, timeframe=settings.timeframe, lunch_skip=settings.lunch_skip,
            risk_percent=settings.risk_percent, stop_loss_percent=settings.stop_loss_percent,
            take_profit_percent=settings.take_profit_percent,
            selected_strategy=settings.selected_strategy,
            force_mode=settings.force_mode.value if hasattr(settings.force_mode,"value") else str(settings.force_mode),
            extra_strategy_paths=settings.extra_strategy_paths,
            backtest_start_date=settings.backtest_start_date,
            backtest_end_date=settings.backtest_end_date,
            backtest_source=settings.backtest_source.value if hasattr(settings.backtest_source,"value") else str(settings.backtest_source),
            data_feed=settings.data_feed,
            strategy_slots=_serialize_slots(slots)
        )
        save_settings(d)

    def update_settings_lock():
        if controller.state.started:
            for widget in form.winfo_children():
                if isinstance(widget, (ctk.CTkEntry, ctk.CTkOptionMenu, ctk.CTkCheckBox)):
                    widget.configure(state="disabled")
            for widget in playbook.winfo_children():
                if isinstance(widget, (ctk.CTkEntry, ctk.CTkOptionMenu, ctk.CTkCheckBox, ctk.CTkButton)):
                    widget.configure(state="disabled")
        else:
            for widget in form.winfo_children():
                if isinstance(widget, (ctk.CTkEntry, ctk.CTkOptionMenu, ctk.CTkCheckBox)):
                    widget.configure(state="normal")
            for widget in playbook.winfo_children():
                if isinstance(widget, (ctk.CTkEntry, ctk.CTkOptionMenu, ctk.CTkCheckBox, ctk.CTkButton)):
                    widget.configure(state="normal")
        root.after(500, update_settings_lock)
    root.after(500, update_settings_lock)

    # Performance (P&L) - unchanged
    header3 = ctk.CTkLabel(tab_trading, text="Performance", font=("Arial", 16, "bold"))
    header3.pack(anchor="w", padx=10, pady=(15,5))

    pl_container = ctk.CTkFrame(tab_trading)
    pl_container.pack(fill="x", padx=10, pady=(0,10))

    realized_card = ctk.CTkFrame(pl_container, width=300, height=100)
    realized_card.pack(side="left", padx=10, pady=10, fill="both", expand=True)
    realized_card.pack_propagate(False)

    ctk.CTkLabel(realized_card, text="REALIZED P&L", font=("Arial", 12, "bold")).pack(pady=(15,5))
    rpnl_var = tk.StringVar(value="$0.00")
    rpnl_label = ctk.CTkLabel(realized_card, textvariable=rpnl_var, font=("Arial", 32, "bold"))
    rpnl_label.pack()

    unrealized_card = ctk.CTkFrame(pl_container, width=300, height=100)
    unrealized_card.pack(side="left", padx=10, pady=10, fill="both", expand=True)
    unrealized_card.pack_propagate(False)

    ctk.CTkLabel(unrealized_card, text="UNREALIZED P&L", font=("Arial", 12, "bold")).pack(pady=(15,5))
    upnl_var = tk.StringVar(value="$0.00")
    upnl_label = ctk.CTkLabel(unrealized_card, textvariable=upnl_var, font=("Arial", 32, "bold"))
    upnl_label.pack()

    def _poll_pl():
        try:
            # Only show P&L if in LIVE mode, not backtest
            if controller.state.run_mode == RunMode.LIVE and controller.state.started:
                rp = getattr(controller.state, "realized_pnl", 0.0)
                up = getattr(controller.state, "unrealized_pnl", 0.0)
            else:
                rp = 0.0
                up = 0.0
                
            if isinstance(rp, (int, float)):
                rpnl_var.set(f"${rp:.2f}")
                if rp > 0:
                    rpnl_label.configure(text_color="#00FF00")
                elif rp < 0:
                    rpnl_label.configure(text_color="#FF0000")
                else:
                    rpnl_label.configure(text_color="#FFFFFF")
            if isinstance(up, (int, float)):
                upnl_var.set(f"${up:.2f}")
                if up > 0:
                    upnl_label.configure(text_color="#00FF00")
                elif up < 0:
                    upnl_label.configure(text_color="#FF0000")
                else:
                    upnl_label.configure(text_color="#FFFFFF")
        finally:
            root.after(3000, _poll_pl)
    root.after(3000, _poll_pl)

    # Open Positions Table - unchanged
    header4 = ctk.CTkLabel(tab_trading, text="Open Positions", font=("Arial", 16, "bold"))
    header4.pack(anchor="w", padx=10, pady=(15,5))

    positions_frame = ctk.CTkFrame(tab_trading, height=150)
    positions_frame.pack(fill="both", padx=10, pady=(0,10), expand=False)

    pos_tree = ttk.Treeview(positions_frame, columns=("Symbol","Side","Entry Time","Entry Price","Current Price","Qty","P&L","P&L %","Stop Loss","Take Profit"), show="headings", height=5)
    pos_tree.pack(fill="both", expand=True, padx=5, pady=5)

    for col in pos_tree["columns"]:
        pos_tree.heading(col, text=col)
        pos_tree.column(col, width=100)

    # Activity Log - unchanged
    header5 = ctk.CTkLabel(tab_trading, text="Recent Activity (Last 15 trades)", font=("Arial", 16, "bold"))
    header5.pack(anchor="w", padx=10, pady=(15,5))

    activity_frame = ctk.CTkFrame(tab_trading, height=150)
    activity_frame.pack(fill="both", padx=10, pady=(0,10), expand=False)

    activity_tree = ttk.Treeview(activity_frame, columns=("Time","Action","Symbol","Price","Qty","Reason"), show="headings", height=5)
    activity_tree.pack(fill="both", expand=True, padx=5, pady=5)

    for col in activity_tree["columns"]:
        activity_tree.heading(col, text=col)
        activity_tree.column(col, width=120)

    def _poll_positions():
        for item in pos_tree.get_children():
            pos_tree.delete(item)
        if hasattr(controller, 'positions') and controller.positions:
            for symbol, pos_data in controller.positions.items():
                try:
                    entry_time = pos_data.get("entry_time", "")
                    entry_price = pos_data.get("entry_price", 0.0)
                    current_price = pos_data.get("current_price", 0.0)
                    qty = pos_data.get("qty", 0)
                    pnl = pos_data.get("pnl", 0.0)
                    pnl_pct = pos_data.get("pnl_pct", 0.0)
                    sl = pos_data.get("stop_loss", 0.0)
                    tp = pos_data.get("take_profit", 0.0)
                    side = pos_data.get("side", "")
                    pos_tree.insert("", "end", values=(
                        symbol, side, entry_time,
                        f"${entry_price:.2f}", f"${current_price:.2f}",
                        qty, f"${pnl:.2f}", f"{pnl_pct:.2f}%", f"${sl:.2f}", f"${tp:.2f}"
                    ))
                except Exception as e:
                    logging.debug(f"Failed to display position {symbol}: {e}")
        root.after(2000, _poll_positions)

    def _poll_activity():
        for item in activity_tree.get_children():
            activity_tree.delete(item)
        if hasattr(controller, 'recent_trades') and controller.recent_trades:
            for trade in controller.recent_trades:
                try:
                    activity_tree.insert("", "end", values=(
                        trade.get("time", ""),
                        trade.get("action", ""),
                        trade.get("symbol", ""),
                        trade.get("price", ""),
                        trade.get("qty", ""),
                        trade.get("reason", "")
                    ))
                except Exception as e:
                    logging.debug(f"Failed to display trade: {e}")
        root.after(2000, _poll_activity)

    root.after(2000, _poll_positions)
    root.after(2000, _poll_activity)

    # ========== CONNECTION TAB ==========
    conn_header = ctk.CTkLabel(tab_conn, text="API Credentials", font=("Arial", 16, "bold"))
    conn_header.pack(anchor="w", padx=10, pady=(15,5))
        
    conn = ctk.CTkFrame(tab_conn)
    conn.pack(fill="x", padx=10, pady=(0,20))
        
    ctk.CTkLabel(conn, text="API Key").grid(row=0, column=0, padx=10, pady=10, sticky="e")
    from .config_store import load_credentials
    k, s = load_credentials()
    api_key_var = tk.StringVar(value=k or "")
    ctk.CTkEntry(conn, textvariable=api_key_var, width=420).grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
    ctk.CTkLabel(conn, text="API Secret").grid(row=1, column=0, padx=10, pady=10, sticky="e")
    api_secret_var = tk.StringVar(value=s or "")
    ctk.CTkEntry(conn, textvariable=api_secret_var, show="*", width=420).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
    ctk.CTkLabel(conn, text="Connection mode").grid(row=2, column=0, padx=10, pady=10, sticky="e")
    conn_mode_var = tk.StringVar(value=str(getattr(controller.state, "connection_mode", "(disconnected)")))
    ctk.CTkEntry(conn, textvariable=conn_mode_var, state="disabled", width=180).grid(row=2, column=1, padx=10, pady=10, sticky="w")

    def on_save_credentials():
        from .config_store import save_credentials, verify_credentials
        d = load_settings()
        d["api_key"] = api_key_var.get()
        d["api_secret"] = api_secret_var.get()
        save_settings(d)
        save_credentials(api_key_var.get(), api_secret_var.get())
        if verify_credentials():
            messagebox.showinfo("Saved", "Credentials saved and verified in keyring.")
        else:
            messagebox.showerror("Error", "Credentials not saved properly")

    ctk.CTkButton(conn, text="Save credentials", command=on_save_credentials).grid(row=3, column=1, padx=10, pady=10, sticky="w")

    def on_connect():
        try:
            _try_call(controller, ["connect","ensure_connected","init_connection","reconnect"], api_key_var.get(), api_secret_var.get())
        except Exception as e:
            messagebox.showerror("Connect failed", str(e)); return
        cm = getattr(controller.state, "connection_mode", None)
        conn_mode_var.set(str(cm) if cm else "(unknown)")
        messagebox.showinfo("Connected", f"Connection mode: {conn_mode_var.get()}")
        
    ctk.CTkButton(conn, text="Connect", command=on_connect).grid(row=3, column=1, padx=10, pady=10, sticky="e")

    def _poll_conn_mode():
        cm = getattr(controller.state, "connection_mode", None)
        conn_mode_var.set(str(cm) if cm else "(disconnected)")
        root.after(3000, _poll_conn_mode)
    root.after(3000, _poll_conn_mode)
        
    # Account Info Section
    acct_header = ctk.CTkLabel(tab_conn, text="Account Information (Live)", font=("Arial", 16, "bold"))
    acct_header.pack(anchor="w", padx=10, pady=(15,5))
        
    acct_info_frame = ctk.CTkFrame(tab_conn)
    acct_info_frame.pack(fill="x", padx=10, pady=(0,10))
        
    equity_label = ctk.CTkLabel(acct_info_frame, text="Equity:", font=("Arial", 12, "bold"))
    equity_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
    equity_value = ctk.CTkLabel(acct_info_frame, text="$0.00", font=("Arial", 12))
    equity_value.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
    buying_power_label = ctk.CTkLabel(acct_info_frame, text="Buying Power:", font=("Arial", 12, "bold"))
    buying_power_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
    buying_power_value = ctk.CTkLabel(acct_info_frame, text="$0.00", font=("Arial", 12))
    buying_power_value.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
    day_trades_label = ctk.CTkLabel(acct_info_frame, text="Day Trades:", font=("Arial", 12, "bold"))
    day_trades_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
    day_trades_value = ctk.CTkLabel(acct_info_frame, text="0", font=("Arial", 12))
    day_trades_value.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
    def _poll_account_info():
        if hasattr(controller, '_adapter') and controller._adapter and controller.state.connection_mode:
            try:
                account = controller._adapter._trading_client.get_account()
                eq = float(getattr(account, "equity", 0.0))
                bp = float(getattr(account, "buying_power", 0.0))
                dt = int(getattr(account, "daytrade_count", 0))
                equity_value.configure(text=f"${eq:,.2f}")
                buying_power_value.configure(text=f"${bp:,.2f}")
                day_trades_value.configure(text=str(dt))
            except Exception as e:
                logging.debug(f"Failed to fetch account info: {e}")
        root.after(5000, _poll_account_info)
    root.after(5000, _poll_account_info)

    # ========== BACKTEST TAB ==========
    def setup_backtest_tab(tab_backtest, settings, controller):
        """Setup the backtest configuration tab with date range selectors"""
        
        bt_header = ctk.CTkLabel(tab_backtest, text="Backtest Configuration", font=("Arial", 16, "bold"))
        bt_header.pack(anchor="w", padx=10, pady=(15,5))
        
        bt_config = ctk.CTkFrame(tab_backtest)
        bt_config.pack(fill="x", padx=10, pady=(0,10))
        
        # Date Range Section
        date_frame = ctk.CTkFrame(bt_config)
        date_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=(10,5), sticky="ew")
        
        ctk.CTkLabel(date_frame, text="📅 Backtest Date Range", 
                     font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(10,5))
        
        # Start Date
        start_frame = ctk.CTkFrame(date_frame)
        start_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(start_frame, text="Start Date:", font=("Arial", 11, "bold")).pack(side="left", padx=(5,10))
        
        # Set default dates
        default_end = datetime.now()
        default_start = default_end - timedelta(days=730)  # 2 years back
        
        if settings.backtest_start_date:
            default_start = settings.backtest_start_date
        if settings.backtest_end_date:
            default_end = settings.backtest_end_date
        
        # DateEntry widgets with calendar popup
        start_date_picker = DateEntry(
            start_frame,
            width=15,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
            year=default_start.year,
            month=default_start.month,
            day=default_start.day,
            selectbackground='darkblue',
            selectforeground='white',
            normalbackground='white',
            normalforeground='black',
            weekendbackground='lightgray',
            weekendforeground='black',
            othermonthbackground='white',
            othermonthforeground='gray',
            othermonthwebackground='white',
            othermonthweforeground='gray',
            font=("Arial", 10)
        )
        start_date_picker.pack(side="left", padx=5)
        
        # Quick preset buttons for start date
        preset_frame_start = ctk.CTkFrame(start_frame)
        preset_frame_start.pack(side="left", padx=(20,5))
        
        def set_start_3m():
            new_date = datetime.now() - timedelta(days=90)
            start_date_picker.set_date(new_date)
        
        def set_start_6m():
            new_date = datetime.now() - timedelta(days=180)
            start_date_picker.set_date(new_date)
        
        def set_start_1y():
            new_date = datetime.now() - timedelta(days=365)
            start_date_picker.set_date(new_date)
        
        def set_start_2y():
            new_date = datetime.now() - timedelta(days=730)
            start_date_picker.set_date(new_date)
        
        ctk.CTkButton(preset_frame_start, text="3M", width=45, height=24, 
                      command=set_start_3m).pack(side="left", padx=2)
        ctk.CTkButton(preset_frame_start, text="6M", width=45, height=24, 
                      command=set_start_6m).pack(side="left", padx=2)
        ctk.CTkButton(preset_frame_start, text="1Y", width=45, height=24, 
                      command=set_start_1y).pack(side="left", padx=2)
        ctk.CTkButton(preset_frame_start, text="2Y", width=45, height=24, 
                      command=set_start_2y).pack(side="left", padx=2)
        
        # End Date
        end_frame = ctk.CTkFrame(date_frame)
        end_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(end_frame, text="End Date:", font=("Arial", 11, "bold")).pack(side="left", padx=(5,10))
        
        end_date_picker = DateEntry(
            end_frame,
            width=15,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
            year=default_end.year,
            month=default_end.month,
            day=default_end.day,
            selectbackground='darkblue',
            selectforeground='white',
            normalbackground='white',
            normalforeground='black',
            weekendbackground='lightgray',
            weekendforeground='black',
            othermonthbackground='white',
            othermonthforeground='gray',
            othermonthwebackground='white',
            othermonthweforeground='gray',
            font=("Arial", 10)
        )
        end_date_picker.pack(side="left", padx=5)
        
        # Quick preset buttons for end date
        preset_frame_end = ctk.CTkFrame(end_frame)
        preset_frame_end.pack(side="left", padx=(20,5))
        
        def set_end_today():
            end_date_picker.set_date(datetime.now())
        
        def set_end_yesterday():
            end_date_picker.set_date(datetime.now() - timedelta(days=1))
        
        def set_end_week_ago():
            end_date_picker.set_date(datetime.now() - timedelta(days=7))
        
        ctk.CTkButton(preset_frame_end, text="Today", width=60, height=24, 
                      command=set_end_today).pack(side="left", padx=2)
        ctk.CTkButton(preset_frame_end, text="Yesterday", width=70, height=24, 
                      command=set_end_yesterday).pack(side="left", padx=2)
        ctk.CTkButton(preset_frame_end, text="1 Week Ago", width=80, height=24, 
                      command=set_end_week_ago).pack(side="left", padx=2)
        
        # Date range info
        info_label = ctk.CTkLabel(date_frame, text="", font=("Arial", 10), text_color="#888888")
        info_label.pack(anchor="w", padx=10, pady=(5,10))
        
        def update_date_info():
            """Update the date range information label"""
            try:
                start = start_date_picker.get_date()
                end = end_date_picker.get_date()
                delta = end - start
                days = delta.days
                
                if days < 0:
                    info_label.configure(text="⚠️ Warning: End date is before start date!", 
                                        text_color="#ff4444")
                elif days > 730:  # More than 2 years
                    info_label.configure(text=f"ℹ️ Date range: {days} days ({days/365:.1f} years) - Note: Polygon free tier limited to 2 years", 
                                        text_color="#ffaa00")
                else:
                    info_label.configure(text=f"ℹ️ Date range: {days} days ({days/365:.1f} years)", 
                                        text_color="#888888")
            except Exception as e:
                info_label.configure(text="", text_color="#888888")
        
        # Update info when dates change
        start_date_picker.bind("<<DateEntrySelected>>", lambda e: update_date_info())
        end_date_picker.bind("<<DateEntrySelected>>", lambda e: update_date_info())
        update_date_info()  # Initial update
        
        # Data Source Section
        ctk.CTkLabel(bt_config, text="Data source:", font=("Arial", 11, "bold")).grid(
            row=1, column=0, padx=10, pady=(15,10), sticky="e")
        bsrc_var = tk.StringVar(value=(settings.backtest_source.value 
                                       if hasattr(settings.backtest_source,"value") 
                                       else str(settings.backtest_source)))
        
        # Updated to include Polygon
        data_source_menu = ctk.CTkOptionMenu(bt_config, variable=bsrc_var, 
                                             values=["alpaca", "yahoo", "polygon", "csv"],
                                             width=150)
        data_source_menu.grid(row=1, column=1, padx=10, pady=(15,10), sticky="w")
        
        # Data source info
        source_info = {
            "alpaca": "Real-time & historical data via Alpaca API (requires API key)",
            "yahoo": "Free historical data from Yahoo Finance (no API key needed)",
            "polygon": "High-quality data via Polygon.io (free tier: 2 years, 5 calls/min)",
            "csv": "Load from local CSV files in data/ folder"
        }
        
        source_info_label = ctk.CTkLabel(bt_config, text="", font=("Arial", 9), 
                                         text_color="#888888", wraplength=500)
        source_info_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(0,10), sticky="w")
        
        def update_source_info(*args):
            source = bsrc_var.get()
            info_text = source_info.get(source, "")
            source_info_label.configure(text=f"ℹ️ {info_text}")
        
        bsrc_var.trace('w', update_source_info)
        update_source_info()  # Initial update
        
        # Polygon API Key Section (shown only when Polygon is selected)
        polygon_frame = ctk.CTkFrame(bt_config)
        
        ctk.CTkLabel(polygon_frame, text="Polygon API Key:", 
                     font=("Arial", 11, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        polygon_key_var = tk.StringVar(value=getattr(settings, 'polygon_api_key', ''))
        polygon_key_entry = ctk.CTkEntry(polygon_frame, textvariable=polygon_key_var, 
                                         width=400, show="*")
        polygon_key_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        ctk.CTkLabel(polygon_frame, text="Get your free API key at polygon.io", 
                     font=("Arial", 9), text_color="#888888").grid(
            row=1, column=0, columnspan=2, padx=10, pady=(0,10), sticky="w")
        
        def toggle_polygon_frame(*args):
            """Show/hide Polygon API key input based on selected source"""
            if bsrc_var.get() == "polygon":
                polygon_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=(5,10), sticky="ew")
            else:
                polygon_frame.grid_forget()
        
        bsrc_var.trace('w', toggle_polygon_frame)
        toggle_polygon_frame()  # Initial state
        
        # Backtest Results Section
        bt_results_header = ctk.CTkLabel(tab_backtest, text="Latest Backtest Results", 
                                         font=("Arial", 16, "bold"))
        bt_results_header.pack(anchor="w", padx=10, pady=(15,5))
        
        bt_results_frame = ctk.CTkFrame(tab_backtest)
        bt_results_frame.pack(fill="both", padx=10, pady=(0,10), expand=True)
        
        bt_results_text = ctk.CTkTextbox(bt_results_frame, height=400, width=1320)
        bt_results_text.pack(fill="both", expand=True, padx=10, pady=10)
        bt_results_text.insert("1.0", "No backtest results yet. Run a backtest to see results here.")
        bt_results_text.configure(state="disabled")

        def update_backtest_results():
            """Update the backtest results display"""
            run = _find_latest_backtest_folder()
            if run and (run / "trades.csv").exists():
                import pandas as pd
                try:
                    trades_df = pd.read_csv(run / "trades.csv")
                    stats_text = f"Backtest Run: {run.name}\n"
                    stats_text += "=" * 60 + "\n\n"
                    stats_text += f"Total Trades: {len(trades_df)}\n"
                    if len(trades_df) > 0:
                        winners = len(trades_df[trades_df['pnl'] > 0])
                        losers = len(trades_df[trades_df['pnl'] < 0])
                        win_rate = (winners / len(trades_df)) * 100
                        total_pnl = trades_df['pnl'].sum()
                        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
                        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losers > 0 else 0
                        
                        stats_text += f"Winners: {winners} | Losers: {losers}\n"
                        stats_text += f"Win Rate: {win_rate:.2f}%\n"
                        stats_text += f"Total P&L: ${total_pnl:.2f}\n"
                        stats_text += f"Average Win: ${avg_win:.2f}\n"
                        stats_text += f"Average Loss: ${avg_loss:.2f}\n"
                    else:
                        stats_text += "No trades executed.\n"
                        
                    bt_results_text.configure(state="normal")
                    bt_results_text.delete("1.0", "end")
                    bt_results_text.insert("1.0", stats_text)
                    bt_results_text.configure(state="disabled")
                except Exception as e:
                    logging.error(f"Failed to load backtest results: {e}")
            
            # Check again in 5 seconds if backtest is running
            if controller.state.started and controller.state.run_mode == RunMode.BACKTEST:
                bt_results_text.after(5000, update_backtest_results)
            elif not controller.state.started:
                # Check every 10 seconds when idle for completed backtests
                bt_results_text.after(10000, update_backtest_results)
        
        # Start the update loop
        update_backtest_results()
        
        
        # Return the variables and widgets that need to be accessed elsewhere
        return {
            'start_date_picker': start_date_picker,
            'end_date_picker': end_date_picker,
            'bsrc_var': bsrc_var,
            'polygon_key_var': polygon_key_var,
            'bt_results_text': bt_results_text
        }

    # CALL THE FUNCTION TO CREATE THE BACKTEST TAB UI
    backtest_widgets = setup_backtest_tab(tab_backtest, settings, controller)

    # ========== CHARTS TAB ==========
    chart_ctl = ctk.CTkFrame(tab_charts)
    chart_ctl.pack(fill="x", padx=10, pady=(10,6))

    view_var = tk.StringVar(value="Backtest Equity")
    ctk.CTkOptionMenu(chart_ctl, variable=view_var, values=["Live Equity","Backtest Equity"], 
                        width=180).pack(side="left", padx=(0,8))

    auto_var = tk.BooleanVar(value=False)  # Disabled by default to prevent freezing
    ctk.CTkCheckBox(chart_ctl, text="Auto-refresh", variable=auto_var).pack(side="left", padx=6)

    interval_var = tk.StringVar(value="10")  # Increased default interval
    ctk.CTkLabel(chart_ctl, text="Interval (s)").pack(side="left", padx=(10,2))
    ctk.CTkEntry(chart_ctl, textvariable=interval_var, width=60).pack(side="left", padx=(0,10))

    def on_refresh():
        """Manual refresh with lock check"""
        if _refresh_lock["active"]:
            logging.info("Refresh already in progress, skipping...")
            return
            
        if view_var.get().startswith("Backtest"):
            _refresh_backtest_chart()
        else:
            _refresh_live_chart()

    refresh_btn = ctk.CTkButton(chart_ctl, text="🔄 Refresh", command=on_refresh, width=100)
    refresh_btn.pack(side="left", padx=6)

    stats_btn = ctk.CTkButton(chart_ctl, text="📊 Stats", width=100, 
                                command=lambda: messagebox.showinfo("Stats", "Detailed stats panel"))
    stats_btn.pack(side="left", padx=6)

    # Main container for chart and stats
    main_container = ctk.CTkFrame(tab_charts)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)

    # Left: Chart area
    chart_frame = ctk.CTkFrame(main_container)
    chart_frame.pack(side="left", fill="both", expand=True, padx=(0,5))

    # Right: Stats panel
    stats_frame = ctk.CTkFrame(main_container, width=280)
    stats_frame.pack(side="right", fill="y", padx=(5,0))
    stats_frame.pack_propagate(False)

    # Stats panel header
    stats_header = ctk.CTkLabel(stats_frame, text="📈 Performance Metrics", 
                                font=("Arial", 14, "bold"))
    stats_header.pack(pady=(15,10), padx=10)

    # Stats display area
    stats_display = ctk.CTkFrame(stats_frame)
    stats_display.pack(fill="both", expand=True, padx=10, pady=(0,10))

    # Stats variables
    stat_vars = {
        "starting_equity": tk.StringVar(value="$100,000.00"),
        "current_equity": tk.StringVar(value="$100,000.00"),
        "total_return": tk.StringVar(value="$0.00"),
        "total_return_pct": tk.StringVar(value="0.00%"),
        "peak_equity": tk.StringVar(value="$100,000.00"),
        "max_drawdown": tk.StringVar(value="0.00%"),
        "total_trades": tk.StringVar(value="0"),
        "win_rate": tk.StringVar(value="0.00%"),
        "profit_factor": tk.StringVar(value="0.00"),
        "sharpe_ratio": tk.StringVar(value="0.00"),
        "avg_win": tk.StringVar(value="$0.00"),
        "avg_loss": tk.StringVar(value="$0.00"),
    }

    # Create stat labels
    def create_stat_row(parent, label_text, var, row):
        label = ctk.CTkLabel(parent, text=label_text, font=("Arial", 11), 
                        anchor="w")
        label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        value = ctk.CTkLabel(parent, textvariable=var, font=("Arial", 11, "bold"), 
                            anchor="e")
        value.grid(row=row, column=1, sticky="e", padx=10, pady=5)
        return value

    # Section: Equity
    ctk.CTkLabel(stats_display, text="💰 Equity", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10,5))
    create_stat_row(stats_display, "Starting:", stat_vars["starting_equity"], 1)
    create_stat_row(stats_display, "Current:", stat_vars["current_equity"], 2)
    create_stat_row(stats_display, "P&L:", stat_vars["total_return"], 3)
    pct_label = create_stat_row(stats_display, "Return:", stat_vars["total_return_pct"], 4)

    # Section: Risk
    ctk.CTkLabel(stats_display, text="⚠️ Risk", font=("Arial", 12, "bold")).grid(
        row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(15,5))
    create_stat_row(stats_display, "Peak Equity:", stat_vars["peak_equity"], 6)
    dd_label = create_stat_row(stats_display, "Max DD:", stat_vars["max_drawdown"], 7)

    # Section: Trading
    ctk.CTkLabel(stats_display, text="📊 Trading", font=("Arial", 12, "bold")).grid(
        row=8, column=0, columnspan=2, sticky="w", padx=10, pady=(15,5))
    create_stat_row(stats_display, "Total Trades:", stat_vars["total_trades"], 9)
    wr_label = create_stat_row(stats_display, "Win Rate:", stat_vars["win_rate"], 10)
    create_stat_row(stats_display, "Profit Factor:", stat_vars["profit_factor"], 11)
    create_stat_row(stats_display, "Sharpe Ratio:", stat_vars["sharpe_ratio"], 12)
    create_stat_row(stats_display, "Avg Win:", stat_vars["avg_win"], 13)
    create_stat_row(stats_display, "Avg Loss:", stat_vars["avg_loss"], 14)

    # Matplotlib figure with dual charts
    fig = Figure(figsize=(10, 8), facecolor='#1a1a1a')
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Helper: Calculate drawdown
    def calculate_drawdown(equity_values):
        """Calculate drawdown series from equity curve"""
        if not equity_values or len(equity_values) == 0:
            return []
            
        equity_arr = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = ((equity_arr - running_max) / running_max) * 100
        return drawdown.tolist()

    # Helper: Load trades for markers
    def load_trade_markers(run_folder):
        """Load trade entry/exit points from trades.csv"""
        import pandas as pd
        trades_file = run_folder / "trades.csv"
        if not trades_file.exists():
            return [], []
            
        try:
            df = pd.read_csv(trades_file)
            entries = []
            exits = []
                
            for _, row in df.iterrows():
                entry_time = pd.to_datetime(row['entry_time'], utc=True)
                exit_time = pd.to_datetime(row['exit_time'], utc=True)
                pnl = float(row['pnl'])
                    
                entries.append(entry_time.to_pydatetime())
                exits.append((exit_time.to_pydatetime(), pnl))
                
            return entries, exits
        except Exception as e:
            logging.debug(f"Failed to load trade markers: {e}")
            return [], []

    # Helper: Calculate Sharpe ratio
    def calculate_sharpe(returns_series, risk_free_rate=0.0):
        """Calculate Sharpe ratio from returns"""
        if len(returns_series) < 2:
            return 0.0
            
        returns = np.array(returns_series)
        excess_returns = returns - risk_free_rate
            
        if np.std(returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized

    # Enhanced plotting function with better performance
    def plot_professional_chart(x_data, y_data, trades_data=None, title="Equity Curve"):
        """Plot dual chart with equity and drawdown"""
        try:
            # Clear figure efficiently
            fig.clear()
                
            # Set dark theme colors
            bg_color = '#1a1a1a'
            grid_color = '#333333'
            text_color = '#e0e0e0'
            equity_color = '#00d4ff'
            positive_color = '#00ff88'
            negative_color = '#ff4444'
                
            if not x_data or not y_data or len(x_data) == 0 or len(y_data) == 0:
                # Empty state
                ax = fig.add_subplot(111, facecolor=bg_color)
                ax.text(0.5, 0.5, "No data available", 
                        ha="center", va="center", fontsize=14, color=text_color)
                ax.set_facecolor(bg_color)
                fig.patch.set_facecolor(bg_color)
                canvas.draw_idle()  # Use draw_idle instead of draw
                return
                
            # Limit data points for performance (downsample if needed)
            max_points = 2000
            if len(x_data) > max_points:
                step = len(x_data) // max_points
                x_data = x_data[::step]
                y_data = y_data[::step]
                
             # Create subplots: Equity (top, 70%) and Drawdown (bottom, 30%)
            gs = fig.add_gridspec(2, 1, height_ratios=[7, 3], hspace=0.15)
            ax1 = fig.add_subplot(gs[0], facecolor=bg_color)
            ax2 = fig.add_subplot(gs[1], facecolor=bg_color, sharex=ax1)
                
            # === Top Chart: Equity Curve ===
            starting_equity = y_data[0] if y_data else 100000
            current_equity = y_data[-1] if y_data else starting_equity
            line_color = positive_color if current_equity >= starting_equity else negative_color
                
            # Plot equity line with glow effect
            ax1.plot(x_data, y_data, color=line_color, linewidth=2.5, alpha=0.9, zorder=3)
            ax1.plot(x_data, y_data, color=line_color, linewidth=6, alpha=0.2, zorder=2)
                
            # Fill under curve
            fill_color = positive_color if current_equity >= starting_equity else negative_color
            ax1.fill_between(x_data, y_data, starting_equity, 
                                color=fill_color, alpha=0.15, zorder=1)
                
            # Horizontal line at starting equity
            ax1.axhline(y=starting_equity, color=text_color, linestyle='--', 
                        linewidth=1, alpha=0.4, label=f'Start: ${starting_equity:,.0f}')
                
            # Plot trade markers if available (limit to 200 most recent)
            if trades_data:
                entries, exits = trades_data
                # Limit markers for performance
                if len(entries) > 200:
                    entries = entries[-200:]
                if len(exits) > 200:
                    exits = exits[-200:]
                    
                if entries:
                    entry_values = []
                    for entry_time in entries:
                        try:
                            idx = min(range(len(x_data)), 
                                        key=lambda i: abs((x_data[i] - entry_time).total_seconds()))
                            entry_values.append(y_data[idx])
                        except:
                            continue
                        
                    if entry_values:
                        ax1.scatter(entries[:len(entry_values)], entry_values, marker='^', 
                                    color=positive_color, s=80, alpha=0.7, zorder=4, 
                                    edgecolors='white', linewidths=0.5)
                    
                if exits:
                    exit_times = [e[0] for e in exits]
                    exit_pnls = [e[1] for e in exits]
                    exit_values = []
                    exit_colors = []
                        
                    for exit_time, pnl in zip(exit_times, exit_pnls):
                        try:
                            idx = min(range(len(x_data)), 
                                        key=lambda i: abs((x_data[i] - exit_time).total_seconds()))
                            exit_values.append(y_data[idx])
                            exit_colors.append(positive_color if pnl > 0 else negative_color)
                        except:
                            continue
                        
                    if exit_values:
                        ax1.scatter(exit_times[:len(exit_values)], exit_values, marker='v', 
                                    c=exit_colors, s=80, alpha=0.7, zorder=4, 
                                    edgecolors='white', linewidths=0.5)
                
            # Styling
            ax1.set_title(title, fontsize=16, fontweight='bold', color=text_color, pad=20)
            ax1.set_ylabel('Equity ($)', fontsize=12, color=text_color, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.3, color=grid_color)
            ax1.tick_params(colors=text_color, labelsize=10)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color(grid_color)
            ax1.spines['bottom'].set_color(grid_color)
                
            # Format y-axis
            from matplotlib.ticker import FuncFormatter
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
                
            # === Bottom Chart: Drawdown ===
            drawdown = calculate_drawdown(y_data)
                
            if drawdown:
                # Plot drawdown area
                ax2.fill_between(x_data, drawdown, 0, color=negative_color, alpha=0.5, zorder=1)
                ax2.plot(x_data, drawdown, color=negative_color, linewidth=2, alpha=0.8, zorder=2)
                    
                # Zero line
                ax2.axhline(y=0, color=text_color, linestyle='-', linewidth=1, alpha=0.3)
                    
                # Styling
                ax2.set_ylabel('Drawdown (%)', fontsize=11, color=text_color, fontweight='bold')
                ax2.set_xlabel('Time', fontsize=11, color=text_color, fontweight='bold')
                ax2.grid(True, linestyle=':', alpha=0.3, color=grid_color)
                ax2.tick_params(colors=text_color, labelsize=9)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_color(grid_color)
                ax2.spines['bottom'].set_color(grid_color)
                    
                # Format y-axis
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))
                
            # Rotate x-axis labels
            import matplotlib.pyplot as plt
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
            fig.patch.set_facecolor(bg_color)
            fig.tight_layout()
            canvas.draw_idle()  # Use draw_idle for better performance
                
        except Exception as e:
            logging.error(f"Plot failed: {e}")
            # Show error in chart
            try:
                fig.clear()
                ax = fig.add_subplot(111, facecolor='#1a1a1a')
                ax.text(0.5, 0.5, f"Chart Error: {str(e)}", 
                        ha="center", va="center", fontsize=12, color='#ff4444')
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#1a1a1a')
                canvas.draw_idle()
            except:
                pass

    # Update stats panel with error handling
    def update_stats_panel(equity_data, trades_df=None):
        """Update the stats panel with calculated metrics"""
        try:
            if not equity_data or len(equity_data) < 2:
                return
                
            starting = equity_data[0]
            current = equity_data[-1]
            peak = max(equity_data)
            total_return_val = current - starting
            total_return_pct_val = ((current / starting) - 1) * 100
                
            # Update equity stats
            stat_vars["starting_equity"].set(f"${starting:,.2f}")
            stat_vars["current_equity"].set(f"${current:,.2f}")
            stat_vars["total_return"].set(f"${total_return_val:,.2f}")
            stat_vars["total_return_pct"].set(f"{total_return_pct_val:+.2f}%")
            stat_vars["peak_equity"].set(f"${peak:,.2f}")
                
            # Color code return
            if total_return_val > 0:
                pct_label.configure(text_color="#00ff88")
            elif total_return_val < 0:
                pct_label.configure(text_color="#ff4444")
            else:
                pct_label.configure(text_color="#e0e0e0")
                
            # Calculate max drawdown
            drawdown = calculate_drawdown(equity_data)
            max_dd = min(drawdown) if drawdown else 0.0
            stat_vars["max_drawdown"].set(f"{max_dd:.2f}%")
            dd_label.configure(text_color="#ff4444" if max_dd < -10 else "#ffaa00" if max_dd < -5 else "#e0e0e0")
                
            # Calculate Sharpe ratio
            if len(equity_data) > 1:
                returns = np.diff(equity_data) / equity_data[:-1]
                sharpe = calculate_sharpe(returns)
                stat_vars["sharpe_ratio"].set(f"{sharpe:.2f}")
                
            # Update trading stats from trades_df
            if trades_df is not None and not trades_df.empty:
                total_trades = len(trades_df)
                winners = len(trades_df[trades_df['pnl'] > 0])
                win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
                    
                stat_vars["total_trades"].set(str(total_trades))
                stat_vars["win_rate"].set(f"{win_rate:.2f}%")
                    
                # Color code win rate
                if win_rate >= 60:
                    wr_label.configure(text_color="#00ff88")
                elif win_rate >= 45:
                    wr_label.configure(text_color="#ffaa00")
                else:
                    wr_label.configure(text_color="#ff4444")
                    
                # Profit factor
                gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
                gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
                stat_vars["profit_factor"].set(f"{profit_factor:.2f}")
                    
                # Average win/loss
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
                avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winners) > 0 else 0
                stat_vars["avg_win"].set(f"${avg_win:.2f}")
                stat_vars["avg_loss"].set(f"${avg_loss:.2f}")
        except Exception as e:
            logging.error(f"Stats panel update failed: {e}")

    # Refresh lock to prevent overlapping updates
    _refresh_lock = {"active": False}

    # Auto-refresh timer with throttling
    def _auto_tick():
        try:
            secs = int(interval_var.get() or "10")
        except Exception:
            secs = 10
        
        if auto_var.get() and not _refresh_lock["active"]:
            try:
                on_refresh()
            except Exception as e:
                logging.error(f"Chart refresh failed: {e}")
        
        root.after(max(2000, secs*1000), _auto_tick)

    root.after(5000, _auto_tick)

     # Live equity buffer
    live_buf = {"x": [], "y": [], "start_equity": None}

    # Refresh lock to prevent overlapping updates
    _refresh_lock = {"active": False}

    # Refresh functions with locking
    def _refresh_live_chart():
        """Refresh live equity chart"""
        if _refresh_lock["active"]:
            logging.debug("Skipping refresh - already in progress")
            return
        
        _refresh_lock["active"] = True
        try:
            eq = None
            rp = getattr(controller.state, "realized_pnl", None)
            up = getattr(controller.state, "unrealized_pnl", None)
            
            if isinstance(rp, (int,float)) and isinstance(up, (int,float)):
                if live_buf["start_equity"] is None:
                    try:
                        live_buf["start_equity"] = float(controller._adapter.get_account_equity()) if controller._adapter else 100000.0
                    except Exception:
                        live_buf["start_equity"] = 100000.0
                
                if live_buf["start_equity"] is not None:
                    eq = live_buf["start_equity"] + float(rp) + float(up)
            
            now = datetime.now(timezone.utc)
            if eq is not None:
                live_buf["x"].append(now)
                live_buf["y"].append(eq)
            
            # Keep last 1000 points to avoid memory issues
            if len(live_buf["x"]) > 1000:
                live_buf["x"] = live_buf["x"][-1000:]
                live_buf["y"] = live_buf["y"][-1000:]
            
            plot_professional_chart(live_buf["x"], live_buf["y"], 
                                   title=f"Live Equity - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            update_stats_panel(live_buf["y"])
        except Exception as e:
            logging.error(f"Live chart refresh failed: {e}")
        finally:
            _refresh_lock["active"] = False

    def _refresh_backtest_chart():
        """Refresh backtest equity chart"""
        if _refresh_lock["active"]:
            logging.debug("Skipping refresh - already in progress")
            return
        
        _refresh_lock["active"] = True
        try:
            import pandas as pd
            
            run = _find_latest_backtest_folder()
            if not run:
                plot_professional_chart([], [], title="Backtest Equity (No Data)")
                return
            
            # Load equity data
            x, y = _load_equity_csv(run)
            
            # Load trade markers
            trades_data = load_trade_markers(run)
            
            # Load trades CSV for stats
            trades_df = None
            trades_file = run / "trades.csv"
            if trades_file.exists():
                try:
                    trades_df = pd.read_csv(trades_file)
                except Exception:
                    pass
            
            plot_professional_chart(x, y, trades_data=trades_data, 
                                   title=f"Backtest Equity - {run.name}")
            update_stats_panel(y, trades_df)
        except Exception as e:
            logging.error(f"Backtest chart refresh failed: {e}")
        finally:
            _refresh_lock["active"] = False

    # ========== LOGS TAB ==========
    log_box = ctk.CTkTextbox(tab_logs, height=700, width=1320)
    log_box.pack(fill="both", expand=True, padx=10, pady=10)
    log_box.configure(state="disabled")
    ui_handler = UITextHandler(log_box)
    logging.getLogger().addHandler(ui_handler)
    logging.getLogger().setLevel(logging.INFO)

    # ========== SETTINGS TAB ==========
    settings_header = ctk.CTkLabel(tab_settings, text="Advanced Settings", font=("Arial", 16, "bold"))
    settings_header.pack(anchor="w", padx=10, pady=(15,5))
        
    adv = ctk.CTkFrame(tab_settings)
    adv.pack(fill="x", padx=10, pady=(0,20))
        
    ctk.CTkLabel(adv, text="Force mode").grid(row=0, column=0, padx=10, pady=10, sticky="e")
    force_var = tk.StringVar(value=settings.force_mode.value if hasattr(settings.force_mode,"value") else str(settings.force_mode))
    ctk.CTkOptionMenu(adv, variable=force_var, values=["auto","paper","live"]).grid(row=0, column=1, padx=10, pady=10, sticky="w")

    ctk.CTkLabel(adv, text="Extra strategy paths (semicolon-separated)").grid(row=2, column=0, padx=10, pady=10, sticky="e")
    extra_paths_var = tk.StringVar(value=";".join(settings.extra_strategy_paths))
    ctk.CTkEntry(adv, textvariable=extra_paths_var, width=520).grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
    ctk.CTkButton(tab_settings, text="Save Settings", command=_save_current_settings, fg_color="#0066CC", hover_color="#004488").pack(padx=10, pady=(10,10), anchor="w")

    root.mainloop()
