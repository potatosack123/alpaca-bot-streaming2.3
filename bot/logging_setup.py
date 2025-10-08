from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

class UILogHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.queue.put(msg)
        except Exception:
            pass

def _build_file_handler(path: Path, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(path, maxBytes=1024*1024, backupCount=5, encoding="utf-8")
    handler.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    return handler

def setup_logging(ui_queue=None) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    app_file = _build_file_handler(LOG_DIR / "app.log", logging.INFO)
    trades_file = _build_file_handler(LOG_DIR / "trades.log", logging.INFO)
    backtest_file = _build_file_handler(LOG_DIR / "backtest.log", logging.INFO)
    root.addHandler(app_file)
    logging.getLogger("trades").addHandler(trades_file)
    logging.getLogger("backtest").addHandler(backtest_file)

    if ui_queue is not None:
        ui_handler = UILogHandler(ui_queue)
        ui_handler.setLevel(logging.INFO)
        ui_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        root.addHandler(ui_handler)
