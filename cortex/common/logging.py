"""
Structured logging configuration for Cortex components.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any


class CortexJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    formatter = (
        CortexJSONFormatter()
        if json_format
        else logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if extra_fields:

        class ExtraFieldsFilter(logging.Filter):
            def filter(self, record):
                if not hasattr(record, "extra"):
                    record.extra = {}
                record.extra.update(extra_fields)
                return True

        root_logger.addFilter(ExtraFieldsFilter())


def get_logger(name: str, extra: dict[str, Any] | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if extra:

        class ExtraContextFilter(logging.Filter):
            def __init__(self, extra_fields):
                super().__init__()
                self.extra_fields = extra_fields

            def filter(self, record):
                if not hasattr(record, "extra"):
                    record.extra = {}
                record.extra.update(self.extra_fields)
                return True

        for f in logger.filters:
            if hasattr(f, "extra_fields") and f.extra_fields == extra:
                logger.removeFilter(f)
        logger.addFilter(ExtraContextFilter(extra))
    return logger
