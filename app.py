#!/usr/bin/env python3
"""
Entry point to launch the Alpaca Stock Bot UI.
"""
from bot.logging_setup import setup_logging
from bot.ui import run_ui
from bot.config_store import ensure_runtime_folders

def main() -> None:
    ensure_runtime_folders()
    setup_logging()
    run_ui()

if __name__ == "__main__":
    main()
