from __future__ import annotations
import logging
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

import pytz
import pandas as pd

from ..state import Bar, SessionState, SignalType, RunMode

log = logging.getLogger(__name__)

@dataclass
class Position:
    """Active position tracker"""
    symbol: str
    entry_time: datetime
    entry_price: float
    shares: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass  
class Trade:
    """Completed trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'

def run_backtest(
    symbols: List[str], 
    tf: str, 
    strategy, 
    settings: Dict[str, Any],
    loader: Callable[[str], List[Bar]], 
    run_dir: Path
) -> Dict[str, Any]:
    """
    Run backtest with optimized logging.
    """
    trades: List[Trade] = []
    equity_records = []
    east = pytz.timezone("US/Eastern")
    
    # Counters for progress logging
    trade_counter = 0
    bar_counter = 0
    
    # Initialize account
    starting_cash = 100000.0
    cash = starting_cash
    positions: Dict[str, Position] = {}
    
    # Create session state for strategy
    session_state = SessionState(
        run_mode=RunMode.BACKTEST,
        started=True,
        paused=False,
        should_stop=False
    )
    
    # Strategy initialization
    try:
        strategy.on_start(session_state)
        log.info("Strategy initialized: %s", type(strategy).__name__)
    except Exception as e:
        log.warning("Strategy on_start failed: %s", e)
    
    # Extract settings
    risk_pct = settings.get("risk_percent", 1.0) / 100.0
    sl_pct = settings.get("stop_loss_percent", 1.0) / 100.0
    tp_pct = settings.get("take_profit_percent", 2.0) / 100.0
    
    log.info("Backtest settings: risk=%.2f%%, SL=%.2f%%, TP=%.2f%%", 
             risk_pct*100, sl_pct*100, tp_pct*100)

    for sym in symbols:
        bars = loader(sym)
        bars = [b for b in bars if getattr(b, "timestamp", None) is not None]
        if not bars:
            log.info("No bars returned for %s; skipping.", sym)
            continue
        
        log.info("Processing %d bars for %s", len(bars), sym)

        for bar in bars:
            bar_counter += 1
            
            # Progress indicator every 1000 bars
            if bar_counter % 1000 == 0:
                log.info("Progress: %d bars processed, %d trades, equity=$%.2f", 
                        bar_counter, trade_counter, cash + sum(p.shares * bar.close for p in positions.values()))
            
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_east = ts.astimezone(east)
            
            # Check existing position for stop-loss / take-profit
            if sym in positions:
                pos = positions[sym]
                hit_sl = pos.stop_loss and bar.low <= pos.stop_loss
                hit_tp = pos.take_profit and bar.high >= pos.take_profit
                
                if hit_sl:
                    exit_price = pos.stop_loss
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    pnl_pct = ((exit_price / pos.entry_price) - 1) * 100
                    cash += exit_price * pos.shares
                    
                    hold_time = ts - pos.entry_time
                    hold_minutes = hold_time.total_seconds() / 60
                    
                    trades.append(Trade(
                        symbol=sym,
                        entry_time=pos.entry_time,
                        exit_time=ts,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        shares=pos.shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="stop_loss"
                    ))
                    
                    trade_counter += 1
                    # Only log every 10th trade or significant losses
                    if trade_counter % 10 == 0 or pnl < -100:
                        log.debug("Trade #%d: STOP_LOSS %s | pnl=$%.2f (%+.2f%%) | hold=%.1fmin", 
                                 trade_counter, sym, pnl, pnl_pct, hold_minutes)
                    
                    del positions[sym]
                    
                elif hit_tp:
                    exit_price = pos.take_profit
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    pnl_pct = ((exit_price / pos.entry_price) - 1) * 100
                    cash += exit_price * pos.shares
                    
                    hold_time = ts - pos.entry_time
                    hold_minutes = hold_time.total_seconds() / 60
                    
                    trades.append(Trade(
                        symbol=sym,
                        entry_time=pos.entry_time,
                        exit_time=ts,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        shares=pos.shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="take_profit"
                    ))
                    
                    trade_counter += 1
                    # Only log every 10th trade or significant wins
                    if trade_counter % 10 == 0 or pnl > 100:
                        log.debug("Trade #%d: TAKE_PROFIT %s | pnl=$%.2f (%+.2f%%) | hold=%.1fmin", 
                                 trade_counter, sym, pnl, pnl_pct, hold_minutes)
                    
                    del positions[sym]
            
            # Get strategy signal
            try:
                signal = strategy.on_bar(sym, bar, session_state)
            except Exception as e:
                log.warning("Strategy error on %s at %s: %s", sym, ts, e)
                signal = None
            
            # Process signal
            if signal and signal.type == SignalType.BUY:
                if sym not in positions:
                    current_equity = cash + sum(
                        p.shares * bar.close for p in positions.values()
                    )
                    position_value = current_equity * risk_pct
                    shares = int(position_value / bar.close)
                    
                    if shares > 0 and (shares * bar.close) <= cash:
                        cash -= shares * bar.close
                        
                        stop_loss = bar.close * (1 - (signal.sl_pct or sl_pct))
                        take_profit = bar.close * (1 + (signal.tp_pct or tp_pct))
                        
                        positions[sym] = Position(
                            symbol=sym,
                            entry_time=ts,
                            entry_price=bar.close,
                            shares=shares,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        
                        # Minimal entry logging - only log occasionally or large positions
                        if len(positions) % 5 == 1 or shares * bar.close > current_equity * 0.05:
                            log.debug("ENTRY: BUY %s | qty=%d | price=$%.2f | positions=%d", 
                                     sym, shares, bar.close, len(positions))
            
            elif signal and signal.type == SignalType.SELL:
                if sym in positions:
                    pos = positions[sym]
                    exit_price = bar.close
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    pnl_pct = ((exit_price / pos.entry_price) - 1) * 100
                    cash += exit_price * pos.shares
                    
                    hold_time = ts - pos.entry_time
                    hold_minutes = hold_time.total_seconds() / 60
                    
                    trades.append(Trade(
                        symbol=sym,
                        entry_time=pos.entry_time,
                        exit_time=ts,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        shares=pos.shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="signal"
                    ))
                    
                    trade_counter += 1
                    if trade_counter % 10 == 0:
                        log.debug("Trade #%d: SIGNAL_EXIT %s | pnl=$%.2f (%+.2f%%)", 
                                 trade_counter, sym, pnl, pnl_pct)
                    
                    del positions[sym]
            
            # Calculate current equity (do this less frequently)
            if bar_counter % 100 == 0:
                position_value = sum(p.shares * bar.close for p in positions.values())
                current_equity = cash + position_value
                
                equity_records.append({
                    "timestamp": ts,
                    "equity": current_equity,
                    "cash": cash,
                    "positions_value": position_value
                })
    
    # Close any remaining positions at last price
    for sym, pos in list(positions.items()):
        exit_price = pos.entry_price
        pnl = 0.0
        cash += exit_price * pos.shares
        
        trades.append(Trade(
            symbol=sym,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=0.0,
            exit_reason="end_of_backtest"
        ))
        log.info("Closed remaining position in %s", sym)
    
    # Strategy cleanup
    try:
        strategy.on_stop(session_state)
    except Exception as e:
        log.warning("Strategy on_stop failed: %s", e)
    
    # Save artifacts with enhanced data
    eq_df = pd.DataFrame(equity_records)
    if not eq_df.empty:
        eq_df.to_csv(run_dir / "equity.csv", index=False)
        log.info("Saved equity curve to %s", run_dir / "equity.csv")
    
    # Enhanced trades CSV with all strategy metadata
    trades_df = pd.DataFrame([{
        "symbol": t.symbol,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "shares": t.shares,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "exit_reason": t.exit_reason,
        "strategy": type(strategy).__name__,
        "risk_pct": settings.get("risk_percent", 0.0),
        "sl_pct": settings.get("stop_loss_percent", 0.0),
        "tp_pct": settings.get("take_profit_percent", 0.0),
        "timeframe": tf,
        "hold_time_minutes": (t.exit_time - t.entry_time).total_seconds() / 60
    } for t in trades])
    
    trades_df.to_csv(run_dir / "trades.csv", index=False)
    log.info("Saved %d trades to %s", len(trades), run_dir / "trades.csv")
    
    # Calculate statistics
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    final_equity = cash
    total_return = ((final_equity - starting_cash) / starting_cash) * 100
    
    stats = {
        "trades": len(trades),
        "winners": len(winning_trades),
        "losers": len(losing_trades),
        "win_rate": 0.0 if not trades else (len(winning_trades) / len(trades)) * 100,
        "total_pnl": sum(t.pnl for t in trades),
        "avg_win": sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        "avg_loss": sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        "largest_win": max((t.pnl for t in trades), default=0),
        "largest_loss": min((t.pnl for t in trades), default=0),
        "starting_equity": starting_cash,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "profit_factor": abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0
    }
    
    log.info("=" * 60)
    log.info("BACKTEST RESULTS")
    log.info("=" * 60)
    log.info("Total Trades: %d (W: %d, L: %d)", stats["trades"], stats["winners"], stats["losers"])
    log.info("Win Rate: %.2f%%", stats["win_rate"])
    log.info("Total P&L: $%.2f (%.2f%%)", stats["total_pnl"], stats["total_return_pct"])
    log.info("Avg Win: $%.2f | Avg Loss: $%.2f", stats["avg_win"], stats["avg_loss"])
    log.info("Profit Factor: %.2f", stats["profit_factor"])
    log.info("=" * 60)
    
    # Compact summary with key strategy info
    log.info("")
    log.info("Strategy: %s | Timeframe: %s | Risk: %.2f%% | SL: %.2f%% | TP: %.2f%%",
             type(strategy).__name__, tf,
             settings.get("risk_percent", 0.0),
             settings.get("stop_loss_percent", 0.0),
             settings.get("take_profit_percent", 0.0))
    
    # Calculate average hold time
    if trades:
        avg_hold = sum((t.exit_time - t.entry_time).total_seconds() / 60 for t in trades) / len(trades)
        log.info("Average hold time: %.1f minutes", avg_hold)
    
    # Show top 5 best and worst trades for quick insight
    if trades:
        sorted_trades = sorted(trades, key=lambda t: t.pnl, reverse=True)
        log.info("")
        log.info("Top 5 Winners:")
        for i, t in enumerate(sorted_trades[:5], 1):
            log.info("  %d. %s: $%.2f (%+.2f%%) | %s to %s", 
                    i, t.symbol, t.pnl, t.pnl_pct,
                    t.entry_time.strftime("%m/%d %H:%M"),
                    t.exit_time.strftime("%m/%d %H:%M"))
        
        log.info("")
        log.info("Top 5 Losers:")
        for i, t in enumerate(sorted_trades[-5:][::-1], 1):
            log.info("  %d. %s: $%.2f (%+.2f%%) | %s to %s",
                    i, t.symbol, t.pnl, t.pnl_pct,
                    t.entry_time.strftime("%m/%d %H:%M"),
                    t.exit_time.strftime("%m/%d %H:%M"))
    
    log.info("=" * 60)
    
    return stats
