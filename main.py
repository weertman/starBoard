from __future__ import annotations

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import uuid

from PySide6.QtWidgets import QApplication

# Allow running from project root without installing as a package
this_file = Path(__file__).resolve()
project_root = this_file.parent
sys.path.insert(0, str(project_root))  # so 'src' is importable

from src.ui.main_window import MainWindow
from src.data.archive_paths import archive_root, logs_root
from src.utils.interaction_logger import get_interaction_logger


class _SessionFilter(logging.Filter):
    """Inject a stable session id into every record."""
    def __init__(self, sid: str):
        super().__init__()
        self.sid = sid

    def filter(self, record: logging.LogRecord) -> bool:
        record.sid = self.sid
        return True


def _setup_logging() -> str:
    """Set up application logging and return the session ID."""
    log_dir = archive_root()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "starboard.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s | sid=%(sid)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    level_name = os.getenv("STARBOARD_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    fh = RotatingFileHandler(str(log_path), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root_logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root_logger.addHandler(ch)

    # Stable per-process session id (can override for integration tests)
    sid = os.getenv("STARBOARD_SESSION_ID") or (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    sess_filter = _SessionFilter(sid)
    fh.addFilter(sess_filter)
    ch.addFilter(sess_filter)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.info("Logging initialized. Log file=%s level=%s", log_path, level_name)
    
    return sid


def main():
    session_id = _setup_logging()
    
    # Initialize interaction logger for user analytics
    interaction_logger = get_interaction_logger()
    interaction_logger.initialize(session_id, logs_root())
    
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
