from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO, console: Optional[Console] = None, log_file: Optional[str] = None) -> None:
    if console is None:
        console = Console()
    handler = RichHandler(console=console, show_path=False, rich_tracebacks=True)
    handlers = [handler]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handlers.append(file_handler)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
