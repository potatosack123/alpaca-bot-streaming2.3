from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from ..state import Bar

log = logging.getLogger(__name__)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest, StockLatestBarRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_PY = True
except Exception:
    ALPACA_PY = False

try:
    import alpaca_trade_api as tradeapi
    TRADE_API = True
except Exception:
    TRADE_API = False


def _to_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class AlpacaAdapter:
    """Adapter forcing IEX for market data; robust to missing timestamps and pagination."""

    def __init__(self, api_key: str, api_secret: str, force_mode: str = "auto", data_feed: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.force_mode = force_mode
        self.data_feed = "iex"  # lower case for alpaca-py 0.42.x
        self.connection_mode: Optional[str] = None
        self._trading_client = None
        self._data_client = None

    def connect(self, context: Optional[str] = None, quiet: bool = False) -> str:
        def _log_connected(mode: str) -> None:
            msg = f"Connected to Alpaca {mode.upper()} endpoint."
            (log.debug if quiet else log.info)(msg)
            log.info("Market data feed: IEX (forced)")
        if self.force_mode == "paper":
            self._connect_paper(); self.connection_mode = "paper"; _log_connected("paper")
        elif self.force_mode == "live":
            self._connect_live(); self.connection_mode = "live"; _log_connected("live")
        else:
            try:
                self._connect_paper(); self.connection_mode = "paper"; _log_connected("paper")
            except Exception as e:
                log.warning("Paper connection failed; trying live. %s", e)
                self._connect_live(); self.connection_mode = "live"; _log_connected("live")
        # Register this adapter with backtest data loader so engine doesn't have to pass it explicitly.
        try:
            from ..backtest import data as btdata
            btdata.register_backtest_adapter(self)
        except Exception:
            pass
        return self.connection_mode or "paper"

    def _connect_paper(self) -> None:
        if ALPACA_PY:
            self._trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
            self._data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
            return
        if TRADE_API:
            self._trading_client = tradeapi.REST(self.api_key, self.api_secret, base_url="https://paper-api.alpaca.markets")
            self._data_client = self._trading_client
            return
        raise RuntimeError("Alpaca SDK not installed.")

    def _connect_live(self) -> None:
        if ALPACA_PY:
            self._trading_client = TradingClient(self.api_key, self.api_secret, paper=False)
            self._data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
            return
        if TRADE_API:
            self._trading_client = tradeapi.REST(self.api_key, self.api_secret, base_url="https://api.alpaca.markets")
            self._data_client = self._trading_client
            return
        raise RuntimeError("Alpaca SDK not installed.")

    # clock
    def is_market_open_now(self) -> bool:
        try:
            c = self._trading_client.get_clock()
            return bool(getattr(c, "is_open", False))
        except Exception:
            return False

    def get_clock_info(self):
        """Returns (is_open, next_open_time, next_close_time)"""
        try:
            clock = self._trading_client.get_clock()
            is_open = bool(getattr(clock, "is_open", False))
            next_open = getattr(clock, "next_open", None)
            next_close = getattr(clock, "next_close", None)
            return is_open, next_open, next_close
        except Exception as e:
            log.warning("get_clock_info failed: %s", e)
            return False, None, None

    # account
    def get_account_equity(self) -> float:
        a = self._trading_client.get_account()
        return float(getattr(a, "equity", 0.0))

    def flatten_all(self) -> None:
        if ALPACA_PY:
            try:
                poss = self._trading_client.get_all_positions()
                for p in poss:
                    qtyf = float(getattr(p, "qty", 0))
                    if qtyf == 0: continue
                    side = OrderSide.SELL if qtyf > 0 else OrderSide.BUY
                    req = MarketOrderRequest(symbol=getattr(p, "symbol", ""), qty=abs(int(qtyf)), side=side, time_in_force=TimeInForce.DAY)
                    self._trading_client.submit_order(order_data=req)
            except Exception as e:
                log.warning("flatten_all failed: %s", e)
        elif TRADE_API:
            try:
                poss = self._trading_client.list_positions()
                for p in poss:
                    qty = abs(int(float(getattr(p, "qty", 0))))
                    if qty <= 0: continue
                    side = "sell" if float(getattr(p, "qty", 0)) > 0 else "buy"
                    self._trading_client.submit_order(symbol=getattr(p, "symbol", ""), qty=qty, side=side, type="market", time_in_force="day")
            except Exception as e:
                log.warning("flatten_all failed: %s", e)

    def submit_market_order(self, symbol: str, qty: int, side: str) -> None:
        if ALPACA_PY:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side.lower()=="buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
            self._trading_client.submit_order(order_data=req)
        elif TRADE_API:
            self._trading_client.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")

    # data helpers
    def _extract_bars(self, resp, symbol: str):
        # Try multiple shapes
        if resp is None:
            return []
        if isinstance(resp, dict):
            seq = resp.get(symbol) or resp.get("bars") or []
            return list(seq)
        for attr in ("bars","data","items"):
            seq = getattr(resp, attr, None)
            if isinstance(seq, list):
                return seq
        try:
            seq = resp[symbol]  # type: ignore[index]
            if isinstance(seq, list):
                return seq
        except Exception:
            pass
        maybe = getattr(resp, symbol, None)
        if isinstance(maybe, list):
            return maybe
        return []

    def latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        if ALPACA_PY:
            try:
                r = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
                resp = self._data_client.get_stock_latest_trade(r)
                t = resp[symbol].trade if isinstance(resp, dict) else getattr(resp, "trade", None)
                if t is None:
                    return None
                ts = getattr(t, "timestamp", None)
                if isinstance(ts, datetime):
                    pass
                elif ts is not None:
                    try: ts = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                    except Exception: ts = None
                return {"t": _to_utc(ts), "p": float(getattr(t, "price", 0.0))}
            except Exception:
                return None
        if TRADE_API:
            try:
                t = self._data_client.get_last_trade(symbol)
                ts = getattr(t, "timestamp", None)
                if isinstance(ts, datetime):
                    pass
                elif ts is not None:
                    try: ts = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                    except Exception: ts = None
                return {"t": _to_utc(ts), "p": float(getattr(t, "price", 0.0))}
            except Exception:
                return None
        return None

    def latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        if ALPACA_PY:
            try:
                r = StockLatestBarRequest(symbol_or_symbols=symbol, feed="iex")
                resp = self._data_client.get_stock_latest_bar(r)
                b = resp[symbol] if isinstance(resp, dict) else resp
                ts = getattr(b, "timestamp", None)
                if isinstance(ts, datetime):
                    pass
                elif ts is not None:
                    try: ts = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                    except Exception: ts = None
                return {"t": _to_utc(ts), "o": float(getattr(b, "open", 0.0)), "h": float(getattr(b, "high", 0.0)), "l": float(getattr(b, "low", 0.0)), "c": float(getattr(b, "close", 0.0)), "v": int(getattr(b, "volume", 0))}
            except Exception:
                return None
        if TRADE_API:
            try:
                b = self._data_client.get_barset(symbol, "minute", 1)[symbol][0]
                ts = getattr(b, "t", None)
                if isinstance(ts, datetime):
                    pass
                elif ts is not None:
                    try: ts = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                    except Exception: ts = None
                return {"t": _to_utc(ts), "o": float(getattr(b, "o", 0.0)), "h": float(getattr(b, "h", 0.0)), "l": float(getattr(b, "l", 0.0)), "c": float(getattr(b, "c", 0.0)), "v": int(getattr(b, "v", 0))}
            except Exception:
                return None
        return None

    def historical_bars(self, symbol: str, tf: str, start: datetime, end: datetime) -> List[Bar]:
        bars: List[Bar] = []
        if ALPACA_PY:
            try:
                unit = TimeFrameUnit.Minute
                mult = 1 if tf == "1m" else (3 if tf == "3m" else 5)
                page_token = None
                
                while True:
                    # Build request kwargs, only include page_token if not None
                    req_kwargs = {
                        "symbol_or_symbols": symbol,
                        "timeframe": TimeFrame(mult, unit),
                        "start": start,
                        "end": end,
                        "feed": "iex"
                    }
                    if page_token is not None:
                        req_kwargs["page_token"] = page_token
                    
                    req = StockBarsRequest(**req_kwargs)
                    resp = self._data_client.get_stock_bars(req)
                    
                    # Extract bars from response
                    seq = []
                    if hasattr(resp, 'data') and isinstance(resp.data, dict):
                        seq = resp.data.get(symbol, [])
                    elif isinstance(resp, dict):
                        seq = resp.get(symbol, [])
                    
                    got = 0
                    for b in seq:
                        ts = getattr(b, "timestamp", None)
                        if isinstance(ts, datetime):
                            pass
                        elif ts is not None:
                            try: 
                                ts = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                            except Exception: 
                                ts = None
                        if ts is None:
                            continue
                        bars.append(Bar(
                            timestamp=_to_utc(ts), 
                            open=float(getattr(b, "open", 0.0)), 
                            high=float(getattr(b, "high", 0.0)), 
                            low=float(getattr(b, "low", 0.0)), 
                            close=float(getattr(b, "close", 0.0)), 
                            volume=int(getattr(b, "volume", 0))
                        ))
                        got += 1
                    
                    # Check for next page
                    page_token = getattr(resp, "next_page_token", None)
                    if page_token is None:
                        page_token = getattr(resp, "page_token", None)
                    if page_token is None:
                        page_token = getattr(resp, "next_token", None)
                    
                    if not page_token or got == 0:
                        break
                        
            except Exception as e:
                log.warning("historical_bars failed for %s: %s", symbol, e)
                
        elif TRADE_API:
            try:
                data = self._data_client.get_barset(symbol, "minute", limit=1000)[symbol]
                for b in data:
                    ts = getattr(b, "t", None)
                    if isinstance(ts, datetime):
                        pass
                    elif ts is not None:
                        try: 
                            ts = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                        except Exception: 
                            ts = None
                    if ts is None:
                        continue
                    bars.append(Bar(
                        timestamp=_to_utc(ts), 
                        open=float(getattr(b, "o", 0.0)), 
                        high=float(getattr(b, "h", 0.0)), 
                        low=float(getattr(b, "l", 0.0)), 
                        close=float(getattr(b, "c", 0.0)), 
                        volume=int(getattr(b, "v", 0))
                    ))
            except Exception as e:
                log.warning("historical_bars failed for %s: %s", symbol, e)
                
        return bars
