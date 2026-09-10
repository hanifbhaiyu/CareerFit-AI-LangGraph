"""Logging setup shared by the API and the ingestion script."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a readable single-line format."""
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. uvicorn reload)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Third-party libraries are noisy at INFO; keep our own logs readable.
    for noisy in ("httpx", "httpcore", "urllib3", "google", "grpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
