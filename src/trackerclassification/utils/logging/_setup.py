from __future__ import annotations
import logging
import sys

from ._color import ColorFormatter


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with colored, nicely formatted output.
    Call this once at program startup (e.g. in main.py).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)