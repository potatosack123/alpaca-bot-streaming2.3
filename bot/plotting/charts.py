from __future__ import annotations
from typing import List, Tuple
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg  # non-GUI canvas

def save_equity_curve_png(points: List[Tuple[object, float]], out_path: Path) -> None:
    """Save equity curve to PNG safely from any thread (no GUI backend required)."""
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig = Figure(figsize=(8, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)  # bind Agg canvas (headless)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)  # uses the Agg canvas; safe in worker threads
