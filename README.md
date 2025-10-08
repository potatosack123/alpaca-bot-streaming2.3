# Alpaca Stock Bot — Streaming Live + Backtest (Strategy-Ready)

A modular desktop trading bot with **dark-mode UI**, **live trading** (paper/live), and **backtesting**. Uses **Alpaca** SDKs. Safety-first controls (Pause/Stop/Flatten), persistent settings, rotating logs, and plugin strategies.

MIT licensed — free to use.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## UI Tabs
- **Trading** — Start/Pause/Stop/Flatten; Mode (Live/Backtest); Symbols; TF (1m/3m/5m); Lunch Skip; Risk %; SL/TP %; Strategy; P/L.
- **Connection** — API Key/Secret; Save; read-only connection mode.
- **Logs** — live app/trade/backtest logs.
- **Charts** — shows saved path for backtest equity curve.
- **Settings** — Force mode: Auto/Paper/Live; Flatten on Stop; Extra strategy paths.

## Run Controls
- **Pause** — stop new entries; manage exits.
- **Stop** — graceful end; leaves positions.
- **Flatten & Stop** — market-close all positions then stop (asks for confirmation in Live).
- **Flatten on Stop** — (OFF by default) Stop behaves like Flatten & Stop.

## Safety & Behavior
- **Force mode**: `"auto"|"paper"|"live"`; **auto** tries paper, then live.
- **Live confirmation**: one-time per session if connected to **live**.
- **Market hours guard**: waits for `next_open` using Alpaca Clock.
- **Lunch skip**: blocks entries 12:00–13:00 ET.
- **Orders**: market only (v1); SL/TP managed by the bot.

## Backtesting
- Data: prefers Alpaca historical (if keys) else CSV `data/<SYMBOL>_<TF>.csv`:
  ```csv
  timestamp,open,high,low,close,volume
  2024-05-01 09:30:00,100,101,99.5,100.7,12000
  ```
- Outputs per run in `backtests/<timestamp>/`:  
  `backtest.log`, `equity.csv`, `trades.csv`, `chart.png`.

## Streaming (Recommended for Live)
- **Market data** via `StockDataStream` (IEX default; SIP if your plan allows).
- **Trading updates** via `TradingStream` for fills/cancels/rejects.
- Controller falls back to REST polling if streaming is unavailable.
- **P/L accuracy (live)**: unrealized P/L is summed from Alpaca position fields; daily realized P/L = `account.equity - account.last_equity`.

## Strategies
- Base interface: `on_start(state)`, `on_bar(symbol, bar, state) -> Signal|None`, `on_stop(state)`.
- Built-in: `BaselineSMA` (simple SMA cross).
- External packs: drop folders under `strategies/` and list them in Settings → Extra paths.

## Logging
- Rotating files under `logs/`: `app.log`, `trades.log`, `backtest.log` (global) plus **per-run** `backtests/<ts>/backtest.log`.
- Format: `timestamp | level | module | message`. Logs stream into UI.

## Notes
- Choose feed (IEX vs SIP) per your plan. Streaming reduces REST rate usage and latency.
- The bot is a template: extend with streaming quotes, order idempotency, portfolio history, etc.


**Data feed:** Choose `IEX` (free) or `SIP` (paid) in Settings → Data feed. Backtests and streaming will use this feed.