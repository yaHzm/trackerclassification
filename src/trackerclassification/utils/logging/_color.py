from __future__ import annotations
import logging
import sys

# ANSI color codes
RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",    # cyan
    logging.INFO: "\033[32m",     # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",    # red
    logging.CRITICAL: "\033[41m", # red background
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level_color = COLORS.get(record.levelno, "")
        levelname = f"{level_color}{record.levelname}{RESET}"

        # You can adjust this format string as you like
        fmt = f"[%(asctime)s] [%(name)s] [{levelname}] %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
        return formatter.format(record)