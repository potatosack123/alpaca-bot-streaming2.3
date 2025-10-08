from __future__ import annotations
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, Type, List

# Core base + built-in baseline strategy
from .base import StrategyBase
from .baseline import BaselineSMA

# ---- Strategy registry (define ONCE) ----
STRATEGIES: Dict[str, Type[StrategyBase]] = {
    BaselineSMA.name: BaselineSMA
}
# -----------------------------------------

# Optional: register extra built-ins if available
try:
    from .gap_and_go import GapAndGo
    STRATEGIES[GapAndGo.name] = GapAndGo
except Exception:
    pass

try:
    from .orb import ORB
    STRATEGIES[ORB.name] = ORB
except Exception:
    pass


def load_external_strategies(extra_paths: List[str]) -> Dict[str, Type[StrategyBase]]:
    """
    Dynamically load additional Strategy classes from user-specified paths.
    Any discovered Strategy subclasses are ADDED to STRATEGIES (no reassignment).
    """
    for p in extra_paths:
        path = Path(p)
        if not path.exists():
            continue
        if str(path) not in sys.path:
            sys.path.append(str(path))
        for py in path.glob("**/*.py"):
            try:
                spec = importlib.util.spec_from_file_location(py.stem, py)
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                for _, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, StrategyBase) and obj is not StrategyBase:
                        STRATEGIES[obj.name] = obj
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed loading strategy from %s: %s", py, e
                )
    return STRATEGIES
