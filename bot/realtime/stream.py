from __future__ import annotations
import logging
import threading
from typing import Iterable, Optional

log = logging.getLogger(__name__)

try:
    from alpaca.data.live import StockDataStream
    HAS_DATA_STREAM = True
except Exception:
    HAS_DATA_STREAM = False

try:
    from alpaca.data.enums import DataFeed
except Exception:
    DataFeed = None  # type: ignore

try:
    from alpaca.trading.stream import TradingStream
    HAS_TRADING_STREAM = True
except Exception:
    HAS_TRADING_STREAM = False

class RealtimeManager:
    def __init__(self, api_key: str, api_secret: str, symbols: Iterable[str], paper: bool = True, on_bar_queue=None, on_trade_update_queue=None) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = list(symbols)
        self.paper = paper
        self.on_bar_queue = on_bar_queue
        self.on_trade_update_queue = on_trade_update_queue
        self._stop = threading.Event()
        self._t_data: Optional[threading.Thread] = None
        self._t_trading: Optional[threading.Thread] = None

    def start(self) -> None:
        if HAS_DATA_STREAM:
            self._t_data = threading.Thread(target=self._run_data, name="alpaca-data-stream", daemon=True)
            self._t_data.start()
        else:
            log.warning("StockDataStream unavailable; fallback to REST polling.")

        if HAS_TRADING_STREAM:
            self._t_trading = threading.Thread(target=self._run_trading, name="alpaca-trading-stream", daemon=True)
            self._t_trading.start()
        else:
            log.warning("TradingStream unavailable; trade updates won't stream.")

    def stop(self) -> None:
        self._stop.set()

    def _run_data(self) -> None:
        try:
            if DataFeed is not None:
                wss = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX)
            else:
                wss = StockDataStream(self.api_key, self.api_secret)

            async def on_bar(bar):
                try:
                    if self.on_bar_queue is not None:
                        self.on_bar_queue.put(("bar", str(bar.symbol), {
                            "t": getattr(bar, "timestamp", getattr(bar, "time", None)),
                            "o": float(getattr(bar, "open", 0.0)),
                            "h": float(getattr(bar, "high", 0.0)),
                            "l": float(getattr(bar, "low", 0.0)),
                            "c": float(getattr(bar, "close", 0.0)),
                            "v": int(getattr(bar, "volume", 0)),
                        }))
                except Exception as e:
                    log.exception("on_bar handler failed: %s", e)

            wss.subscribe_bars(on_bar, *self.symbols)

            while not self._stop.is_set():
                try:
                    wss.run()
                except Exception as e:
                    log.warning("Data stream error: %s (reconnecting)", e)
                self._stop.wait(3.0)
                if self._stop.is_set():
                    break
        except Exception as e:
            log.exception("Data stream crashed: %s", e)

    def _run_trading(self) -> None:
        try:
            ts = TradingStream(self.api_key, self.api_secret, paper=self.paper)

            async def on_update(data):
                try:
                    if self.on_trade_update_queue is not None:
                        ev = getattr(data, "event", None)
                        order = getattr(data, "order", None)
                        symbol = getattr(order, "symbol", None) if order else None
                        qty = getattr(order, "filled_qty", None) or getattr(order, "qty", None)
                        px = getattr(order, "filled_avg_price", None) or getattr(order, "limit_price", None) or getattr(order, "average_price", None)
                        self.on_trade_update_queue.put(("trade_update", {
                            "event": str(ev) if ev else None,
                            "symbol": str(symbol) if symbol else None,
                            "qty": float(qty) if qty is not None else None,
                            "price": float(px) if px is not None else None,
                        }))
                except Exception as e:
                    log.exception("trade update handler failed: %s", e)

            ts.subscribe_trade_updates(on_update)

            while not self._stop.is_set():
                try:
                    ts.run()
                except Exception as e:
                    log.warning("Trading stream error: %s (reconnecting)", e)
                self._stop.wait(3.0)
                if self._stop.is_set():
                    break
        except Exception as e:
            log.exception("Trading stream crashed: %s", e)
