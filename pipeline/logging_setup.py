"""
Centralized logging for operational events (job start/finish/fail,
rollback actions, server-level errors) -- separate from the per-stage
print() timing/debug output in orchestrator.py, which stays as-is since
it's meant for interactive terminal reading during a demo.

Writes to both the console and a rotating file at jobs/pipeline.log so
there's a durable record to check after the process has exited --
"where do I look when something went wrong" for the deployment runbook.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from config import JOBS_DIR

LOG_PATH = os.path.join(JOBS_DIR, "pipeline.log")

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured

    root = logging.getLogger("pipeline")

    if not _configured:
        root.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
        )

        file_handler = RotatingFileHandler(
            LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)

        root.addHandler(file_handler)
        root.addHandler(console_handler)
        root.propagate = False

        _configured = True

    return logging.getLogger(f"pipeline.{name}")
