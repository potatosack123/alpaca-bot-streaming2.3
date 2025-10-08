from __future__ import annotations
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

try:
    import keyring  # type: ignore
except Exception:
    keyring = None

log = logging.getLogger(__name__)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = CONFIG_DIR / "settings.json"
SECRETS_FILE = CONFIG_DIR / "_secrets.json"  # obfuscated fallback (not secure)
SERVICE_NAME = "alpaca_bot"

# Default to 2 years back from today
_default_end = datetime.now()
_default_start = _default_end - timedelta(days=730)

DEFAULT_SETTINGS = {
    "symbols": "AAPL,MSFT",
    "timeframe": "1m",
    "lunch_skip": True,
    "risk_percent": 1.0,
    "stop_loss_percent": 1.0,
    "take_profit_percent": 2.0,
    "selected_strategy": "BaselineSMA",
    "flatten_on_stop": False,
    "force_mode": "auto",
    "extra_strategy_paths": [],
    "backtest_start_date": _default_start.isoformat(),  # ISO format for JSON
    "backtest_end_date": _default_end.isoformat(),
    "backtest_source": "alpaca",
    "polygon_api_key": "",  # New field for Polygon API key
}

def ensure_runtime_folders() -> None:
    for p in [Path("logs"), Path("backtests"), Path("data")]:
        p.mkdir(parents=True, exist_ok=True)

def load_settings() -> Dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            
            # Convert date strings back to datetime objects
            if "backtest_start_date" in settings and isinstance(settings["backtest_start_date"], str):
                try:
                    settings["backtest_start_date"] = datetime.fromisoformat(settings["backtest_start_date"])
                except Exception:
                    settings["backtest_start_date"] = _default_start
            
            if "backtest_end_date" in settings and isinstance(settings["backtest_end_date"], str):
                try:
                    settings["backtest_end_date"] = datetime.fromisoformat(settings["backtest_end_date"])
                except Exception:
                    settings["backtest_end_date"] = _default_end
            
            # Migrate old backtest_years to date range if present
            if "backtest_years" in settings and "backtest_start_date" not in settings:
                years = int(settings.get("backtest_years", 2))
                settings["backtest_end_date"] = datetime.now()
                settings["backtest_start_date"] = settings["backtest_end_date"] - timedelta(days=365 * years)
            
            return settings
        except Exception as e:
            log.exception("Failed reading settings.json: %s", e)
    
    save_settings(DEFAULT_SETTINGS)
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict[str, Any]) -> None:
    # Convert datetime objects to ISO strings for JSON serialization
    settings_to_save = settings.copy()
    
    if "backtest_start_date" in settings_to_save:
        if isinstance(settings_to_save["backtest_start_date"], datetime):
            settings_to_save["backtest_start_date"] = settings_to_save["backtest_start_date"].isoformat()
    
    if "backtest_end_date" in settings_to_save:
        if isinstance(settings_to_save["backtest_end_date"], datetime):
            settings_to_save["backtest_end_date"] = settings_to_save["backtest_end_date"].isoformat()
    
    SETTINGS_FILE.write_text(json.dumps(settings_to_save, indent=2), encoding="utf-8")

def save_credentials(api_key: str, api_secret: str) -> None:
    if keyring:
        keyring.set_password(SERVICE_NAME, "ALPACA_API_KEY", api_key)
        keyring.set_password(SERVICE_NAME, "ALPACA_API_SECRET", api_secret)
        log.info("Saved Alpaca credentials to OS keyring.")
        return
    payload = {"k": _obf(api_key), "s": _obf(api_secret)}
    SECRETS_FILE.write_text(json.dumps(payload), encoding="utf-8")
    log.warning("Keyring unavailable. Saved credentials to %s (obfuscated, NOT secure).", SECRETS_FILE)

def save_polygon_key(api_key: str) -> None:
    """Save Polygon API key to keyring or fallback"""
    if keyring:
        keyring.set_password(SERVICE_NAME, "POLYGON_API_KEY", api_key)
        log.info("Saved Polygon API key to OS keyring.")
    else:
        # Add to secrets file
        try:
            if SECRETS_FILE.exists():
                payload = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            else:
                payload = {}
            payload["polygon"] = _obf(api_key)
            SECRETS_FILE.write_text(json.dumps(payload), encoding="utf-8")
            log.warning("Keyring unavailable. Saved Polygon key to %s (obfuscated, NOT secure).", SECRETS_FILE)
        except Exception as e:
            log.error("Failed to save Polygon key: %s", e)

def load_polygon_key() -> Optional[str]:
    """Load Polygon API key from keyring or fallback"""
    if keyring:
        try:
            key = keyring.get_password(SERVICE_NAME, "POLYGON_API_KEY")
            if key:
                return key
        except Exception as e:
            log.warning("Keyring error loading Polygon key: %s", e)
    
    # Try fallback file
    if SECRETS_FILE.exists():
        try:
            data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if "polygon" in data:
                return _deobf(data["polygon"])
        except Exception as e:
            log.exception("Failed reading Polygon key from fallback: %s", e)
    
    return None

def _obf(s: str) -> str:
    return ''.join(chr(ord(c) ^ 0x39) for c in s)

def _deobf(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return ''.join(chr(ord(c) ^ 0x39) for c in s)

def load_credentials() -> Tuple[Optional[str], Optional[str]]:
    if keyring:
        try:
            k = keyring.get_password(SERVICE_NAME, "ALPACA_API_KEY")
            s = keyring.get_password(SERVICE_NAME, "ALPACA_API_SECRET")
            log.info(f"Loaded from keyring - Key starts with: {k[:10] if k else 'None'}")
            return k, s
        except Exception as e:
            log.warning("Keyring error: %s", e)
    if SECRETS_FILE.exists():
        try:
            data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            k = _deobf(data.get("k"))
            s = _deobf(data.get("s"))
            log.info(f"Loaded from secrets file - Key starts with: {k[:10] if k else 'None'}")
            return k, s
        except Exception as e:
            log.exception("Failed reading fallback secrets: %s", e)
    log.warning("No credentials found in keyring or secrets file")
    return None, None

def verify_credentials() -> bool:
    """Check if credentials are saved and valid"""
    k, s = load_credentials()
    if not k or not s:
        log.error("No credentials found")
        return False
    log.info(f"Credentials found - Key starts with: {k[:10]}")
    return True
